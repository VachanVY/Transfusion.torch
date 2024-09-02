import math
import time
import random
import matplotlib.pyplot as plt
import os
import warnings
import dataclasses as dc
import typing as tp
warnings.filterwarnings("ignore") # W0901 12:56:55.922000 133054240231424 torch/fx/experimental/symbolic_shapes.py:4449] [0/1] xindex is not in var_ranges, defaulting to unknown range.

import torch
from torch import Tensor, nn
from torch.nn import functional as F
import torch.utils.data
from torchvision import datasets, transforms

from torch_src.configs import CelebA_config as config
from torch_src.diffusion_utils import DiffusionUtils
from torch_src.transfusion import Transfussion
from torch_src.llama2c import Transformer as LLaMA

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class CelebADataset(torch.utils.data.IterableDataset):
    def __init__(
        self, *,
        transforms:transforms.Compose, 
        split:str,
        config:config,
    ):
        super().__init__()
        self.transforms = transforms
        self.split = split

        self.MAX_ATTRS = len(config.ATTRS)
        self.IGNORE_TOK = config.IGNORE_TOKEN
        self.EOS = config.EOS.item()
        self.text_maxlen = config.text_maxlen
        self.dict_attrs = {attr: i for i, attr in enumerate(config.ATTRS)}

        self.ATTRS = config.ATTRS
        self.tokenizer = lambda x: config.tokenize(x, dict_attrs=self.dict_attrs)

    def process_labels(self, nhot_tensor:Tensor):
        attr_lst = [i for i, j in zip(self.ATTRS, nhot_tensor) if j == 1]
        if nhot_tensor[20] != 1: # male attr index = 20
            attr_lst.append("Female")
            if "Male" in attr_lst and "Female" in attr_lst: assert False, "male and female, 💀💀💀"
        random.shuffle(attr_lst)
        tokens = torch.tensor(self.tokenizer(attr_lst) + [self.EOS])
        seq = F.pad(tokens, pad=(0, self.text_maxlen-len(tokens)-1), mode="constant", value=self.IGNORE_TOK).long()
        return seq, seq==self.IGNORE_TOK # mask True values

    def __iter__(self):
        while True:
            ds = datasets.CelebA(
                root='data', split=self.split, target_type='attr',
                download=True, transform=self.transforms
            )
            for x, y in ds:
                seq, mask = self.process_labels(y)
                yield x, seq, mask


class DataLoader:
    transforms = transforms.Compose([
        # torchvision.transforms.Resize((config.H, config.W)),
        transforms.CenterCrop((config.H, config.W)),
        transforms.ToTensor(), # [0, 255] -> [0.0, 1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # [0.0, 1.0] -> [-1.0, 1.0]
    ])
    
    def iter_batches(self, batch_size:int, split:str="train"):
        ds = CelebADataset(transforms=self.transforms, split=split, config=config)
        # TODO: when num_workers > 0, returns duplicates, use worker id to avoid duplicates
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, drop_last=True,
            num_workers=0, prefetch_factor=None, shuffle=False
        )

        for x, y, mask in dl:
            x:Tensor = x.to(config.device)
            y:Tensor = y.to(config.device)
            idx_from:int = mask.int().argmax(dim=1).min().item()
            yield x, y, idx_from


class CosineDecayWithWarmup:
    def __init__(
        self,
        warmup_steps:int,
        max_learning_rate:float,
        decay_steps:int,
        min_learning_rate:float
    ):
        self.warmup_steps = warmup_steps
        self.max_learning_rate = max_learning_rate
        self.decay_steps = decay_steps
        self.min_learning_rate = min_learning_rate

    def __call__(self, step):
        # linear warmup for warmup_steps steps
        if step < self.warmup_steps:
            return self.max_learning_rate * step / self.warmup_steps
        # if it > decay_steps, return min learning rate
        if step > self.decay_steps:
            return self.min_learning_rate
        # in between, use cosine decay down to min learning rate
        decay_ratio = (step - self.warmup_steps) / (self.decay_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_learning_rate + coeff * (self.max_learning_rate - self.min_learning_rate)


class TransfusionTrainer:
    def __init__(self, model:nn.Module, config:config):
        self.model = model
        self.autocast = torch.autocast(
            device_type=config.device.type,
            dtype={"bfloat16": torch.bfloat16,
                    "float32" : torch.float32}[config.dtype_type]
        )

        self.BOI = config.BOI.clone().to(config.device) # (,)
        self.IGNORE_TOKEN = config.IGNORE_TOKEN.clone().to(config.device) # (,)
        self.EOI = config.EOI.clone().to(config.device) # (,)

        self.diff_utils = DiffusionUtils(linear_schedule=True, config=config)

        self.balancing_coeff = config.balancing_coeff

    def _compute_loss(
        self,
        txt_logits:Tensor,  # (B, T, vocab_size)
        true_tokens:Tensor, # (B, T)
        true_noise:Tensor,  # (B, C, H, W)
        pred_noise:Tensor,  # (B, C, H, W)
        ignore_idx:int
    ):
        lm_loss = F.cross_entropy(txt_logits.movedim(-1, -2), true_tokens, ignore_index=ignore_idx)
        diff_loss = F.mse_loss(pred_noise, true_noise)
        return lm_loss + self.balancing_coeff*diff_loss, lm_loss.item(), diff_loss.item()
    
    def make_instances(self, text:Tensor, min_idx:int, batch_size:int): # (B, T)
        if random.random() < 0.8:
            sep_idx = random.randint(1, min_idx)
        else:
            sep_idx = 0
        seq_before = torch.cat([
            text[:, :sep_idx],
            self.BOI.repeat(batch_size).unsqueeze(1),
            self.IGNORE_TOKEN.repeat(batch_size).unsqueeze(1)
        ], dim=1) # (B, sep_idx + 2)
        seq_after = torch.cat([
            self.EOI.repeat(batch_size).unsqueeze(1),
            text[:, sep_idx:],
        ], dim=1) # (B, 1 + (T - sep_idx))
        
        (toks_before, tar_toks_bef), (toks_after, tar_toks_af) = (seq_before[:, :-1].clone(), seq_before[:, 1:].clone()), (seq_after[:, :-1].clone(), seq_after[:, 1:].clone()) # (B, Tbf=sep_idx+1), (B, Taf=T-sep_idx)
        return (toks_before, tar_toks_bef), (toks_after, tar_toks_af)
    
    def _ignore_tokens(self, tar_toks_bef:Tensor, tar_toks_af:Tensor, IGNORE_IDX:int):
        tar_toks_bef[:, -1] = IGNORE_IDX
        tar_toks_af[tar_toks_af == self.IGNORE_TOKEN] = IGNORE_IDX
        return tar_toks_bef, tar_toks_af

                    #  (B, H, W, C)   (B, T)       (B,)
    def __call__(self, images:Tensor, text:Tensor, idx_from:int):
        B = images.size(0)

        timesteps = torch.randint(0, config.num_timesteps, (B,), device=config.device)
        noisy_images, true_noise = self.diff_utils.noisy_it(images, timesteps)

        IGNORE_IDX = -1
        (toks_before, tar_toks_bef), (toks_after, tar_toks_af) = self.make_instances(text, idx_from, B)
        tar_toks_bef, tar_toks_af = self._ignore_tokens(tar_toks_bef, tar_toks_af, IGNORE_IDX)

        with self.autocast:
            txt_bef_logits, pred_noise, txt_aft_logits = self.model(toks_before, noisy_images, toks_after, timesteps)
            loss, lm_loss, diff_loss = self._compute_loss(
                torch.cat([txt_bef_logits, txt_aft_logits], dim=1), # (B, Taf + Tbf, vocab_size)
                torch.cat([tar_toks_bef, tar_toks_af], dim=1),      # (B, Taf + Tbf)
                true_noise, pred_noise,                             # (B, C, H, W), (B, C, H, W)
                ignore_idx=IGNORE_IDX
            )
        return loss, lm_loss, diff_loss
    
    
# TODO: Add code for eval, something like accuracy, etc.
@torch.no_grad()
def evaluate(gen_with_image:bool):
    print(f"\n\nGenerating {'with' if gen_with_image else 'without'} image\n\n")
    model.eval()
    img, text, idx_from = next(val_iterator) # (B=1, H, W, C), (B=1, T)
    (toks_before_, tar_toks_bef), (toks_after_, tar_toks_af) = transfusion_trainer.make_instances(text, idx_from, batch_size=1)
    toks_before, image, toks_after = model.generate(
        toks_before=toks_before_,
        image=img if gen_with_image else None,
        diff_utils=transfusion_trainer.diff_utils
    )
    return toks_before, image.permute(0, 2, 3, 1).add(1.0).div(2.0).clip(0, 1), toks_after # (B=1, Tbf), (B=1, H, W, C), (B=1, Taf)


def train(losses:list[float]=[]):
    global start_iter
    num_grad_accumalation_steps = config.num_grad_accumalation_steps
    X_batch, y_batch, idx_from = next(train_iterator)
    t0 = time.time()
    for step in range(start_iter, config.num_steps):
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if (step % config.eval_freq == 0 and step > start_iter) or step == config.num_steps-1:
            toks_before, gen_image, toks_after = evaluate(gen_with_image=random.choice([True, False]))
            text_before = config.detokenize(toks_before.squeeze().cpu().tolist()) if toks_before.size(1) > 0 else ""
            text_after = config.detokenize(toks_after.squeeze().cpu().tolist()) if toks_after.size(1) > 0 else ""
            plt.imshow(gen_image.squeeze().cpu().numpy())
            plt.title(text_before)
            plt.xlabel(text_after)
            plt.savefig(f"images/gen_{step}.png")
            plt.show(block=False)
            plt.pause(interval=5)
            plt.close()

            checkpoint = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "model_args": model_args,
                "step": step,
                "losses": losses
            }

            print(f"Saving checkpoint to {config.ckpt_dir} ...", end=" ==> ")
            torch.save(checkpoint, os.path.join(config.ckpt_dir, "ckpt.pt"))
            # fear of losing the checkpoint
            torch.save(checkpoint, os.path.join(config.ckpt_dir, f"ckpt_{step}.pt"))
            print("Done.")

        if step == config.num_steps//4:
            num_grad_accumalation_steps *= 2
            print("Changing num_grad_accumalation_steps to", num_grad_accumalation_steps)


        optimizer.zero_grad(set_to_none=True)
        
        for _ in range(num_grad_accumalation_steps):
            loss, lm_loss, diff_loss = transfusion_trainer(images=X_batch, text=y_batch, idx_from=idx_from)
            X_batch, y_batch, idx_from = next(train_iterator) ### TO DEBUG ### comment this line to overfit on single batch and check if loss reaches the minimum (~0.0)
            loss.backward()
        if config.clipnorm is not None:
            norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.clipnorm, error_if_nonfinite=True)
        optimizer.step()

        t1 = time.time()
        dt = t1-t0; t0 = t1
        lossf = loss.detach().cpu().item() * config.num_grad_accumalation_steps
        if step % config.log_interval == 0:
            print(
                f"| Step: {step} || Loss: {lossf:.4f} || lm_loss: {lm_loss:.4f} || diff_loss: {diff_loss:.4f} |"
                f"| LR: {lr:e} || dt: {dt*1000:.2f}ms |", end="")
            print(f"| Norm: {norm:.4f} |" if config.clipnorm is not None else "")
        losses.append(lossf)
    return losses


if __name__ == "__main__":
    os.makedirs(config.ckpt_dir, exist_ok=True)
    os.makedirs("images/", exist_ok=True)
    
    losses, start_iter = [], 0
    COMPILE = True # sppeeeddd...
    if "scratch" in config.init_from:
        model_config = config
        model_args = dc.asdict(config())
        model = Transfussion(
            model=LLaMA(model_config),
            config=model_config
        ).to(config.device)
        if COMPILE:
            model.compile()

        best_val_loss = 1e9
        checkpoint = None
    elif "resume" in config.init_from:
        print("Resuming training using checkpoint...")
        def get_model_state(state_dict:dict[str, Tensor]):
            unwanted_prefix = "_orig_mod." # this prefix gets added when a loaded Model was compiled
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
            return state_dict
        
        ckpt_path = os.path.join(config.ckpt_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=config.device)

        model_args = checkpoint["model_args"]
        model_config = config(**model_args)
        start_iter = checkpoint["step"]

        model = Transfussion(
            model=LLaMA(model_config),
            config=model_config
        ).to(config.device)
        if COMPILE:
            model.compile()
        
        model.load_state_dict(checkpoint["model_state"])

        losses = checkpoint["losses"]

    optimizer = model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.max_lr,
        betas=(config.beta1, config.beta2),
        device_type=config.device.type
    )
    if ("resume" in config.init_from) and ("optimizer" in checkpoint):
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None # free memory

    get_lr = CosineDecayWithWarmup(
        warmup_steps=config.warmup_steps,
        max_learning_rate=config.max_lr,
        decay_steps=config.decay_steps,
        min_learning_rate=config.min_lr
    ) if not config.no_decay else lambda _: config.max_lr

    train_iterator = iter(DataLoader().iter_batches(config.batch_size, split="train"))
    val_iterator = iter(DataLoader().iter_batches(batch_size=1, split="valid"))

    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, "Million")
    transfusion_trainer = TransfusionTrainer(model, config)
    losses = train(losses=losses)
    
    plt.plot(losses)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss Vs Steps")
    plt.grid(True)
    plt.savefig("images/loss_vs_steps.png")
    plt.show(block=False)
    plt.pause(interval=10)
    plt.close()
