from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def timestep_embedding(timesteps: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(half, device=timesteps.device) / half
    )
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    if dimension % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def group_norm(channels: int) -> nn.GroupNorm:
    groups = min(32, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualBlock(nn.Module):
    def __init__(
        self, input_channels: int, output_channels: int, condition_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(
            group_norm(input_channels), nn.SiLU(), nn.Conv2d(input_channels, output_channels, 3, padding=1)
        )
        self.condition = nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, output_channels))
        self.output = nn.Sequential(
            group_norm(output_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
        )
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv2d(input_channels, output_channels, 1)
        )

    def forward(self, image: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = self.input(image)
        hidden = hidden + self.condition(condition)[:, :, None, None]
        return self.skip(image) + self.output(hidden)


class DownBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, condition_dim: int, dropout: float) -> None:
        super().__init__()
        self.residual = ResidualBlock(input_channels, output_channels, condition_dim, dropout)
        self.downsample = nn.Conv2d(output_channels, output_channels, 4, stride=2, padding=1)

    def forward(self, image: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.residual(image, condition)
        return self.downsample(skip), skip


class UpBlock(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int, condition_dim: int, dropout: float) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(input_channels, output_channels, 4, stride=2, padding=1)
        self.residual = ResidualBlock(output_channels + skip_channels, output_channels, condition_dim, dropout)

    def forward(self, image: torch.Tensor, skip: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        image = self.upsample(image)
        return self.residual(torch.cat((image, skip), dim=1), condition)


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        character_count: int,
        font_count: int,
        base_channels: int = 64,
        channel_multipliers: tuple[int, ...] = (1, 2, 4),
        condition_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        channels = [base_channels * multiplier for multiplier in channel_multipliers]
        self.condition_dim = condition_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(condition_dim, condition_dim * 4),
            nn.SiLU(),
            nn.Linear(condition_dim * 4, condition_dim),
        )
        self.character_embedding = nn.Embedding(character_count, condition_dim)
        self.font_embedding = nn.Embedding(font_count, condition_dim)
        self.input = nn.Conv2d(1, channels[0], 3, padding=1)

        self.down_blocks = nn.ModuleList()
        current = channels[0]
        for channel in channels[:-1]:
            self.down_blocks.append(DownBlock(current, channel, condition_dim, dropout))
            current = channel

        self.middle = nn.ModuleList(
            [
                ResidualBlock(current, channels[-1], condition_dim, dropout),
                ResidualBlock(channels[-1], channels[-1], condition_dim, dropout),
            ]
        )
        current = channels[-1]
        self.up_blocks = nn.ModuleList()
        for skip_channel in reversed(channels[:-1]):
            self.up_blocks.append(
                UpBlock(current, skip_channel, skip_channel, condition_dim, dropout)
            )
            current = skip_channel

        self.output = nn.Sequential(
            group_norm(current), nn.SiLU(), nn.Conv2d(current, 1, 3, padding=1)
        )

    def forward(
        self,
        image: torch.Tensor,
        timesteps: torch.Tensor,
        characters: torch.Tensor,
        font_ids: torch.Tensor,
    ) -> torch.Tensor:
        condition = (
            self.time_mlp(timestep_embedding(timesteps, self.condition_dim))
            + self.character_embedding(characters)
            + self.font_embedding(font_ids)
        )
        hidden = self.input(image)
        skips: list[torch.Tensor] = []
        for block in self.down_blocks:
            hidden, skip = block(hidden, condition)
            skips.append(skip)
        for block in self.middle:
            hidden = block(hidden, condition)
        for block in self.up_blocks:
            hidden = block(hidden, skips.pop(), condition)
        return self.output(hidden)
