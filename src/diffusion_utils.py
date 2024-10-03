from tqdm import trange
import typing as tp

import torch
from torch import Tensor, nn


class DiffusionUtils:
    def __init__(self, linear_schedule:bool, config:tp.Any):
        self.num_timesteps = config.num_timesteps # (nT,)
        self.beta_range = list(config.var_range)
        self.device = config.device
        self.get_vars(linear_schedule=linear_schedule)

        self.H, self.W = config.H, config.W
        self.in_channels = config.in_channels

    def get_vars(self, linear_schedule:bool):
        if linear_schedule:
            self.beta = torch.linspace(start=self.beta_range[0], end=self.beta_range[1], steps=self.num_timesteps, device=self.device) # (nT,)
            self.alpha = (1-self.beta) # (nT,)
            self.alpha_bar = torch.concatenate(
                (torch.tensor([1.], device=self.device), self.alpha.cumprod(axis=0)),
                axis=0
            ) # (nT,)
        else:
            s, max_beta = 0.008, 0.999
            t = torch.arange(self.num_timesteps + 1, device=self.device)
            f = torch.cos((t / self.num_timesteps + s) / (1 + s) * torch.pi / 2) ** 2
            self.alpha = torch.clip(f[1:] / f[:-1], 1 - max_beta, 1)
            self.alpha = torch.cat((torch.tensor([1.], device=self.device), self.alpha))
            self.beta = 1 - self.alpha
            self.alpha_bar = self.alpha.cumprod(axis=0)

    def noisy_it(self, X:Tensor, t:Tensor): # (B, H, W, C), (B,)
        noise = torch.normal(mean=0.0, std=1.0, size=X.shape, device=self.device) # (B,)

        alpha_bar_t = self.alpha_bar[t][:, None, None, None] # (B, 1, 1, 1) <= (B,) <= (nT,)
        return torch.sqrt(alpha_bar_t)*X + torch.sqrt(1 - alpha_bar_t) * noise, noise # noisy_image, noise
    
    def one_step_ddpm(self, xt:Tensor, pred_noise:Tensor, t:Tensor):
        alpha_t, alpha_bar_t = self.alpha[t, None, None, None], self.alpha_bar[t, None, None, None]
        xt_minus_1 = (
            (1/torch.sqrt(alpha_t))
            *
            (xt - (1-alpha_t)*pred_noise/torch.sqrt(1-alpha_bar_t)
            ) + torch.sqrt(self.beta[t])*torch.normal(mean=0.0, std=1.0, size=xt.shape, device=self.device)
        )
        return xt_minus_1
    
    def one_step_ddim(self, xt:Tensor, pred_noise:Tensor, t:Tensor) -> Tensor:
        raise NotImplementedError("TODO")
    
    @torch.no_grad()
    def generate(
        self, *,
        model:nn.Module,
        modality_tokens:list[Tensor],   # [(Tbf,), (N, D), (Taf,)] OR any other format
        modality_strings:list[str],     # ['text', 'image', 'text']
        use_ddim:bool=False,            # False for now until implemented
        autocast:torch.autocast
    ):
        """
        model args:\n
            `modality_tokens` : `list[Tensor]`, shape [(B, Tbf), (B, N, D), (B, Taf)]
            `modality_strings`: `list[str]`,    shape ['text', 'image', 'text']

        """
        patch_ops = model.patch_ops
        sample_func = self.one_step_ddim if use_ddim else self.one_step_ddpm

        noised_image = torch.normal(mean=0.0, std=1.0, size=(self.in_channels, self.H, self.W), device=self.device) # (B, C, H, W)
        modality_strings.append("image"); modality_tokens += [()] # add empty slot for image modality
        for i in trange(0, self.num_timesteps-1):
            timesteps = torch.tensor([self.num_timesteps - i - 1], device=self.device) # (B,)
            modality_tokens[-1] = (patch_ops.patchify(noised_image), timesteps) # (B, N, D)
            # model takes "noisy image" (in the form of modality_tokens[-1]) and returns the noise which should be removed from the "noisy image"
            with autocast:
                modality_token_emb, _ = model.forward_unbatched(
                    modality_tokens=modality_tokens,
                    modality_strings=modality_strings
                )
            pred_noise = patch_ops.unpatchify(
                modality_token_emb[-1] # noise to be removed from the noisy image
            ) # (N = H*W//P**2, D = (P**2)*C) => (C, H, W)
            
            noised_image = sample_func(noised_image[None], pred_noise[None], timesteps)[0] # (C, H, W)
        
        modality_tokens[-1] = (patch_ops.patchify(noised_image), torch.tensor(0, device=self.device).view((1,))) # generated image: (N, D), timesteps: 0
        return modality_tokens, modality_strings
