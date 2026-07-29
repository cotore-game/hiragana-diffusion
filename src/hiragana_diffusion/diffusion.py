from __future__ import annotations

import math

import torch
from torch import nn


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

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        initial_noise: torch.Tensor,
        characters: torch.Tensor,
        styles: torch.Tensor,
        sampling_steps: int,
    ) -> torch.Tensor:
        if not 1 <= sampling_steps <= self.timesteps:
            raise ValueError(
                f"sampling_steps must be between 1 and {self.timesteps}"
            )

        image = initial_noise
        schedule = torch.linspace(
            self.timesteps - 1,
            0,
            sampling_steps,
            device=image.device,
        ).round().long()

        for index, timestep in enumerate(schedule):
            timesteps = torch.full(
                (image.shape[0],),
                int(timestep.item()),
                device=image.device,
                dtype=torch.long,
            )
            predicted_noise = model(image, timesteps, characters, styles)
            alpha_bar = self.alpha_bars[timestep]
            predicted_clean = (
                image - torch.sqrt(1.0 - alpha_bar) * predicted_noise
            ) / torch.sqrt(alpha_bar)
            predicted_clean = predicted_clean.clamp(-1.0, 1.0)

            if index + 1 == len(schedule):
                image = predicted_clean
                continue

            previous_alpha_bar = self.alpha_bars[schedule[index + 1]]
            image = (
                torch.sqrt(previous_alpha_bar) * predicted_clean
                + torch.sqrt(1.0 - previous_alpha_bar) * predicted_noise
            )

        return image
