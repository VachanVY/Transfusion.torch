from dataclasses import dataclass
from typing import Optional, Literal, Any

import torch


@dataclass
class MNIST_config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_type = "bfloat16" if torch.cuda.is_available() else "float32"
    # device = torch.device("cpu") # uncomment for debugging

    # Diffusion Args
    var_range:tuple[float, float] = (1e-4, 2e-2)
    num_timesteps:int = 400

    # Vit Args
    patch_size:int = 2
    H:int = 28
    W:int = 28
    in_channels:int = 1
    out_channels:int = in_channels
    N:int = H*W//patch_size**2
    assert N*patch_size**2 == H*W

    # transformer Args
    d_model:int = 348
    num_heads:int = 6
    assert d_model % 2 == 0
    assert d_model % num_heads == 0
    num_layers:int = 7
    num_classes:int = 10
    dropout_rate:float = 0.0
    text_maxlen:int = 6
    maxlen:int = 2*N + text_maxlen

    # Training Args
    batch_size:int = 64
    num_steps:int = 15_000
    decay_steps:int = num_steps
    warmup_steps:int = 100
    max_lr:float = 3e-4
    min_lr:float = 0.0*max_lr
    no_decay:bool = True
    beta1:float = 0.9
    beta2:float = 0.99 # 0.95 in paper # for smaller datasets a bit higher is better # was 0.975 changed to 0.99
    clipnorm:float = 1e0
    weight_decay:float = 0.0 # 1e0 in paper
    
    patience:int = 10
    num_grad_accumalation_steps:int = 1
    checkpoint_dir:str = "checkpoints"
    return_best_train_states:bool = True
    log_interval:int = 25
    eval_freq:int = 400

    # Transfusion Args
    balancing_coeff:float = 5.0

    BOI:Optional[torch.Tensor] = torch.tensor(num_classes, dtype=torch.long) # 10
    IGNORE_TOKEN:Optional[torch.Tensor] = torch.tensor(num_classes+1, dtype=torch.long) # 11
    EOI:Optional[torch.Tensor] = torch.tensor(num_classes+2, dtype=torch.long) # 12
    EOS:Optional[torch.Tensor] = torch.tensor(num_classes+3, dtype=torch.long) # 13 

    lm_output_units:int = num_classes + int(BOI is not None) + int(IGNORE_TOKEN is not None) + int(EOI is not None) + int(EOS is not None)


"""@dataclass
class CelebA_config:
    '''???? Million Parameters || ???ms per step on 4090'''  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_type = "bfloat16" if torch.cuda.is_available() else "float32"

    # Diffusion Args
    var_range:tuple[int, int] = (1e-4, 2e-2)
    num_timesteps:int = 1000

    # Vit Args
    patch_size:int = 4 # maybe change to 2
    H:int = 128
    W:int = 128
    in_channels:int = 3
    out_channels:int = in_channels
    N:int = H*W//patch_size**2
    assert N*patch_size**2 == H*W

    # transformer Args
    d_model:int = 768
    num_heads:int = 12
    assert d_model % 2 == 0
    assert d_model % num_heads == 0
    num_layers:int = 12
    dropout_rate:float = 0.0
    text_maxlen:int = 21+4 # 4 for BOI, EOI, EOS and IGNORE_TOKEN
    maxlen:int = N + text_maxlen

    # Training Args
    batch_size:int = 58 # perfectly fits on a single 4090
    num_steps:int = 100_000
    decay_steps:int = num_steps
    warmup_steps:int = 100
    ## in paper they use 3e-4, but for pure diffusion model, got loss spikes for lr more than 1e-4
    ## TODO: Tune it
    max_lr:float = 1e-4
    min_lr:float = 0.1*max_lr
    no_decay:bool = False
    beta1:float = 0.9
    beta2:float = 0.95
    clipnorm:Optional[float] = 1e0
    weight_decay:float = 1e-1
    # 60*(1024+25)*1 = 62940
    
    patience:int = 10
    num_grad_accumalation_steps:int = 1
    ckpt_dir:str = "checkpoints/celeba_85M"
    return_best_train_states:bool = True
    log_interval:int = 1
    eval_freq:int =  2000
    init_from:Literal["scratch", "resume"] = "resume" # "resume" or "scratch"
    assert init_from in ["resume", "scratch"]

    # Transfusion Args
    balancing_coeff:float = 5.0
    num_classes:int = 41
    BOI:Optional[torch.Tensor] = torch.tensor(num_classes, dtype=torch.long) # 41
    IGNORE_TOKEN:Optional[torch.Tensor] = torch.tensor(num_classes+1, dtype=torch.long) # 42
    EOI:Optional[torch.Tensor] = torch.tensor(num_classes+2, dtype=torch.long) # 44
    EOS:Optional[torch.Tensor] = torch.tensor(num_classes+3, dtype=torch.long) # 43

    lm_output_units:int = (
        num_classes + 
        int(BOI is not None) + 
        int(IGNORE_TOKEN is not None) + 
        int(EOI is not None) +
        int(EOS is not None)
    )

    ATTRS = [
            '5_o_Clock_Shadow',            'Arched_Eyebrows',           'Attractive',            'Bags_Under_Eyes',            'Bald',            'Bangs',            'Big_Lips',
            'Big_Nose',            'Black_Hair',            'Blond_Hair',            'Blurry',            'Brown_Hair',            'Bushy_Eyebrows',            'Chubby',
            'Double_Chin',            'Eyeglasses',            'Goatee',            'Gray_Hair',            'Heavy_Makeup',            'High_Cheekbones',            'Male',
            'Mouth_Slightly_Open',            'Mustache',            'Narrow_Eyes',            'No_Beard',            'Oval_Face',            'Pale_Skin',            'Pointy_Nose',
            'Receding_Hairline',            'Rosy_Cheeks',            'Sideburns',            'Smiling',            'Straight_Hair',            'Wavy_Hair',            'Wearing_Earrings',
            'Wearing_Hat',            'Wearing_Lipstick',            'Wearing_Necklace',            'Wearing_Necktie',            'Young',  "Female", 
            " ".join(([""]*1)), " ".join(([""]*2)), " ".join(([""]*3)), " ".join(([""]*4)), " ".join(([""]*5))
        ]
    ATTRS_DICT = {attr: i for i, attr in enumerate(ATTRS)}
    ATTRS_DICT_INV = {i: attr for i, attr in enumerate(ATTRS)}

    def tokenize(text:list[str], dict_attrs:dict[str, int]=ATTRS_DICT):
        return [dict_attrs[attr] for attr in text] # (t,)
    def detokenize(text:list[int]|int, dict_attrs:dict[int, str]=ATTRS_DICT_INV):
        if not isinstance(text, list):
            text = [text]
        return " ".join([dict_attrs[attr] for attr in text])
    

def loss_vs_lr():
    import random
    from transfusion import Transfussion
    from llama2c import Transformer
    from ..train import TransfusionTrainer

    model = Transfussion(
        model=Transformer(CelebA_config),
        config=CelebA_config
    )

    trainer = TransfusionTrainer(model, CelebA_config)
    opt = model.configure_optimizers(
            weight_decay=CelebA_config.weight_decay,
            learning_rate=CelebA_config.max_lr,
            betas=(CelebA_config.beta1, CelebA_config.beta2),
            device_type=CelebA_config.device.type
        )

    def get_loss(lr):
        for param_group in opt.param_groups:
            param_group["lr"] = lr
        image = torch.randn(8, 3, 128, 128).to(CelebA_config.device)
        text = torch.randint(0, CelebA_config.num_classes, (8, CelebA_config.text_maxlen)).to(CelebA_config.device)
        loss = trainer(image, text, random.randint(3, 10))
        loss.backward()
        opt.step()
        return loss.detach().cpu().item()
    
    lrs = (10**torch.linspace(-6, -2, 100)).tolist()
    lrs = [lr for lr in lrs for _ in range(2)]

    losses = [get_loss(lr) for lr in lrs]
    return losses, lrs




if __name__ == "__main__":
    import matplotlib.pyplot as plt

    losses, lrs = loss_vs_lr()
    plt.figure(figsize=(15,5))
    plt.xlabel("Log base10 Learning Rate: Do 10^(x) to get actual learning rate")
    plt.ylabel("Loss")
    plt.ylim(0.0, 1.5)
    plt.xticks([-6, -5, -4]+torch.arange(-3.5, -2.0, 0.1).tolist())
    plt.plot(torch.log10(torch.tensor(lrs)), losses)
    plt.savefig("images/loss_vs_lr.png")
    plt.show()"""