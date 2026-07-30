from __future__ import annotations

from typing import Any

from torch import nn

from hiragana_diffusion.model import ConditionalUNet
from hiragana_diffusion.miniature_model import MiniatureConditionalUNet


def build_model(model_arguments: dict[str, Any]) -> nn.Module:
    arguments = dict(model_arguments)
    architecture = str(arguments.pop("architecture", "standard"))

    if architecture == "standard":
        return ConditionalUNet(**arguments)
    if architecture == "miniature":
        return MiniatureConditionalUNet(**arguments)

    raise ValueError(f"unknown model architecture: {architecture!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
