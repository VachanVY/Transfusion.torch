from tqdm import trange
import typing as tp

import torch
from torch import Tensor, nn


class DiffusionUtils:
    def __init__(self, linear_schedule:bool, config:tp.Any):
        self.num_timesteps = config.num_timesteps # (nT,)
        self.beta_range = config.var_range
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
        raise NotImplementedError("...")
    
    @torch.no_grad()
    def generate(
        self, *,
        model:nn.Module, # x:Tensor, # (B, C = 3 or 1, H, W) t:Tensor, # (B,) y:tp.Optional[Tensor]=None, # (B,) key:tp.Optional[Tensor]=None
        toks_before:Tensor,
        use_ddim:bool=False, # False for now until implemented
    ):
        sample_func = self.one_step_ddim if use_ddim else self.one_step_ddpm

        x = torch.normal(mean=0.0, std=1.0, size=(1, self.in_channels, self.H, self.W), device=self.device)
        for i in trange(0, self.num_timesteps-1):
            t = torch.tensor([self.num_timesteps - i - 1]*1, device=self.device) # (B,)
            _, noise, _ = model(toks_before, x, None, t)
            x = sample_func(x, noise, t)
        return x
