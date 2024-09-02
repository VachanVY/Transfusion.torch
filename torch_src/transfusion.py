import typing as tp
import math

import torch
from torch import Tensor, nn, _assert

from torch_src.diffusion_utils import DiffusionUtils


class TextOps(nn.Module):
    def __init__(self, config:tp.Any):
        super().__init__()
        self.label_embed = nn.Embedding(
            num_embeddings=config.lm_output_units,
            embedding_dim=config.d_model,
        )
        nn.init.normal_(self.label_embed.weight, mean=0.0, std=0.02)
    
    def forward(self, labels:Tensor):
        return self.label_embed(labels)
    

class TimeEmbedding(nn.Module):
    def __init__(self, config:tp.Any, dim:int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, config.d_model),
            nn.SELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.half_dim = dim // 2

    def _init_weights(self, module:nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def sinusoidal_embeddings(self, x:Tensor):
        """https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py"""
        embeddings = torch.exp(-math.log(10000) * torch.arange(0, self.half_dim, device=x.device) / self.half_dim)
        embeddings = x[:, None] * embeddings[None]
        # in link implementation, concat(cos, sin, -1) is done
        embeddings = torch.concatenate([torch.cos(embeddings), torch.sin(embeddings)], -1)
        return embeddings

    def forward(self, x:Tensor): # (B,)
        x = self.sinusoidal_embeddings(x) # (B,) => (B, dim)
        x = self.mlp(x) # (B, d_model)
        return x # (B, d_model)
    

class PatchOps(nn.Module):
    def __init__(self, config:tp.Any):
        super().__init__()
        self.p = config.patch_size
        self.patch_proj = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.d_model,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=True
        )
        w = self.patch_proj.weight.data
        nn.init.xavier_uniform_(
            w.view([w.shape[0], -1]) # init like linear layer
        )
        self.H, self.W = config.H, config.W
        self.c = config.out_channels
        self.time_emb = TimeEmbedding(config, dim=config.d_model//2)

    def patchify(self, images:Tensor):
        B, C, H, W = images.shape
        patches = self.patch_proj(images) # (B, dim, H//P, W//P)
        patches = patches.flatten(2).transpose(1, 2)
        return patches # (B, N = (H*W)/P**2, embed_dim or dim)
    
    def unpacthify(self, x:Tensor): # (B, N = H*W//P**2, D = (P**2)*C)
        h, w = self.H//self.p, self.W//self.p # int(x.shape[1]**0.5) # h = H//P
        x = x.reshape(-1, h, w, self.p, self.p, self.c) # (B, C, H//P, W//P, P, P)
        x = torch.einsum('bhwpqc->bchpwq', x) # (B, C, H//P, P, W//P, P)
        x = x.reshape(-1, self.c, h*self.p, w*self.p) # (B, C, H, W)
        return x
    
    def forward(self, images:Tensor, timesteps:Tensor): # (B, C, H, W), (B,)
        # We add an embedding of the timestep t to every patch vector before the linear layer)
        return self.patchify(images) + self.time_emb(timesteps)[:, None, :] # (B, N, D) + (B, 1, D)


class Transfussion(nn.Module):
    def __init__(self, model:nn.Module, config:tp.Any):
        super().__init__()
        self.model = model
        self.embeddings = TextOps(config)
        self.patch_ops = PatchOps(config)
        self.emb_linear = nn.Linear(config.d_model, config.lm_output_units)
        self.patch_linear = nn.Linear(config.d_model, (config.patch_size**2)*config.out_channels)

        self.register_buffer("BOI", config.BOI)
        self.register_buffer("IGNORE_TOKEN", config.IGNORE_TOKEN)
        self.register_buffer("EOI", config.EOI)
        self.register_buffer("EOS", config.EOS)
        self.register_buffer("causal_mask", torch.tril(torch.ones(1, 1, config.maxlen, config.maxlen), diagonal=0).bool())

        self.txt_maxlens = config.text_maxlen

    def _compute_mask(self, from_:int, num_image_tokens:int):
        mask = self.causal_mask.clone()
        mask[:, :, from_:from_+num_image_tokens, from_:from_+num_image_tokens] = True # TODO: check when num_image_tokens = 0
        return torch.where(mask, 0., -torch.inf) # True will be not be masked, False will be masked

    def forward(
        self,
        toks_before:Tensor,               # (B, Tbf)
        images:tp.Optional[Tensor],       # (B, C, H, W)
        toks_after:tp.Optional[Tensor],   # (B, Taf)
        timesteps:tp.Optional[Tensor],    # (B,)
    ):
        _assert(timesteps is None if images is None else True, "images and timesteps should be provided together")
        
        (B, Tbf) = toks_before.size(); N = 0; Taf = 0

        model_in:Tensor = self.embeddings(toks_before) # (B, Tbf, D)
        if images is not None:
            img_patch:Tensor = self.patch_ops(images, timesteps) # (B, N, D) # We add an embedding of the timestep t to every patch vector before the linear layer)
            N:int = img_patch.size(1)
            model_in = torch.cat((model_in, img_patch), dim=1) # (B, T = Tbf + N, D)
            if toks_after is not None:
                Taf = toks_after.size(1)
                model_in = torch.cat((model_in, self.embeddings(toks_after)), dim=1) # (B, T = Tbf + N + Taf, D)
        model_out:Tensor = self.model(model_in, self._compute_mask(Tbf, N)) # (B, T = Tbf + N + Taf, D)

        emb_before, pred_noise_patch, emb_aft = model_out.split([Tbf, N, Taf], dim=1) # (B, Tbf, D), (B, N, D), (B, Taf, D)
        pred_noise_patch = self.patch_linear(pred_noise_patch) # (B, N, (P**2)*C)

        lin_before, lin_after = self.emb_linear(
            torch.cat([emb_before, emb_aft], dim=1)
        ).split([Tbf, Taf], dim=1) # (B, Tbf, vocab_size), (B, Taf, vocab_size)

        pred_noise = self.patch_ops.unpacthify(pred_noise_patch) if images is not None else None # (B, C, H, W)
        return lin_before, pred_noise, lin_after # (B, Tbf, vocab_size), (B, C, H, W), (B, Taf, vocab_size)
    
    @torch.no_grad()
    def generate(self, toks_before:Tensor, image:tp.Optional[Tensor], diff_utils:DiffusionUtils) -> tuple[Tensor, Tensor, Tensor]: # (B=1, Tbf), (B=1, C, H, W)
        _assert(self.training == False, "Should be in eval mode for generation")

        # LM mode
        if toks_before[:, -1].squeeze() != self.BOI:
            _assert(image is None, "image should be None when generating text")
            for _ in range(self.txt_maxlens - toks_before.size(1)):
                logits, _, _ = self(toks_before, None, None, None) # (B=1, Tbf, vocab_size)
                nxt_tok = logits[:, -1:].argmax(dim=-1) # (B=1, 1)
                toks_before = torch.cat((toks_before, nxt_tok), dim=1) # (B=1, Tbf+1)
                if (nxt_tok.squeeze() == self.BOI): break
            del logits, nxt_tok
            
        # Diffusion mode
        if image is None:
            image = diff_utils.generate(
                model=self,
                toks_before=toks_before
            )

        toks_after, timesteps = self.EOI.view(1, 1), torch.tensor([0]).to(toks_before.device)
        # LM mode
        for _ in range((self.txt_maxlens-2)-toks_before.size(1)):
            _, _, logits_af = self(toks_before, image, toks_after, timesteps) # (B=1, Tbf, vocab_size), (B=1, C, H, W), (B=1, Taf, vocab_size), (B=1,)
            nxt_tok = logits_af[:, -1:].argmax(dim=-1) # (B=1, 1)
            if nxt_tok.squeeze() == self.EOS: break
            toks_after = torch.cat((toks_after, nxt_tok), dim=1) # (B=1, Taf+1)
        return toks_before, image, toks_after
        
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        import inspect
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_config = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_config)
        return optimizer
