from __future__ import annotations

import math

import torch


def cosine_beta_schedule(timesteps: int, offset: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alpha_bar = torch.cos(
        ((x / timesteps) + offset) / (1 + offset) * math.pi * 0.5
    ).square()
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(0.0001, 0.999).float()


class GaussianDiffusion:
    def __init__(self, timesteps: int, device: torch.device) -> None:
        self.timesteps = timesteps
        self.betas = cosine_beta_schedule(timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    @staticmethod
    def _extract(values: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return values.gather(0, timesteps).view(-1, 1, 1, 1)

    def add_noise(
        self,
        clean_images: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self._extract(self.sqrt_alpha_bars, timesteps) * clean_images
            + self._extract(self.sqrt_one_minus_alpha_bars, timesteps) * noise
        )
