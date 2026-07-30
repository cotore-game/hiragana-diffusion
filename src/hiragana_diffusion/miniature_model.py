from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MiniatureResidualBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        condition_dim: int,
    ) -> None:
        super().__init__()
        self.input = nn.Conv2d(
            input_channels, output_channels, kernel_size=3, padding=1
        )
        self.condition = nn.Linear(condition_dim, output_channels)
        self.output = nn.Conv2d(
            output_channels, output_channels, kernel_size=3, padding=1
        )
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv2d(input_channels, output_channels, kernel_size=1)
        )

    def forward(
        self, image: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.input(image)
        hidden = hidden + self.condition(condition)[:, :, None, None]
        hidden = F.relu(hidden)
        return F.relu(self.skip(image) + self.output(hidden))


class MiniatureDownBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        condition_dim: int,
    ) -> None:
        super().__init__()
        self.residual = MiniatureResidualBlock(
            input_channels, output_channels, condition_dim
        )

    def forward(
        self, image: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.residual(image, condition)
        return F.avg_pool2d(skip, kernel_size=2), skip


class MiniatureUpBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        skip_channels: int,
        output_channels: int,
        condition_dim: int,
    ) -> None:
        super().__init__()
        self.residual = MiniatureResidualBlock(
            input_channels + skip_channels,
            output_channels,
            condition_dim,
        )

    def forward(
        self,
        image: torch.Tensor,
        skip: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        image = F.interpolate(image, size=skip.shape[-2:], mode="nearest")
        return self.residual(torch.cat((image, skip), dim=1), condition)


class MiniatureConditionalUNet(nn.Module):
    def __init__(
        self,
        character_count: int,
        font_count: int,
        timesteps: int = 1000,
        base_channels: int = 8,
        channel_multipliers: tuple[int, ...] = (1, 2, 3),
        condition_dim: int = 32,
    ) -> None:
        super().__init__()
        if len(channel_multipliers) < 2:
            raise ValueError("channel_multipliers must contain at least two values")

        channels = [
            base_channels * multiplier for multiplier in channel_multipliers
        ]
        self.time_embedding = nn.Embedding(timesteps, condition_dim)
        self.character_embedding = nn.Embedding(character_count, condition_dim)
        self.font_embedding = nn.Embedding(font_count, condition_dim)
        self.condition = nn.Sequential(
            nn.Linear(condition_dim, condition_dim),
            nn.ReLU(),
        )
        self.input = nn.Conv2d(1, channels[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        current = channels[0]
        for channel in channels[:-1]:
            self.down_blocks.append(
                MiniatureDownBlock(current, channel, condition_dim)
            )
            current = channel

        self.middle = nn.ModuleList(
            (
                MiniatureResidualBlock(
                    current, channels[-1], condition_dim
                ),
                MiniatureResidualBlock(
                    channels[-1], channels[-1], condition_dim
                ),
            )
        )
        current = channels[-1]

        self.up_blocks = nn.ModuleList()
        for skip_channels in reversed(channels[:-1]):
            self.up_blocks.append(
                MiniatureUpBlock(
                    current, skip_channels, skip_channels, condition_dim
                )
            )
            current = skip_channels

        self.output = nn.Conv2d(current, 1, kernel_size=3, padding=1)

    def forward(
        self,
        image: torch.Tensor,
        timesteps: torch.Tensor,
        characters: torch.Tensor,
        font_ids: torch.Tensor,
    ) -> torch.Tensor:
        condition = self.condition(
            self.time_embedding(timesteps)
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
