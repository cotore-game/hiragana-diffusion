from __future__ import annotations

import unittest

import torch
from torch import nn

from hiragana_diffusion.model_factory import build_model, count_parameters
from hiragana_diffusion.miniature_model import MiniatureConditionalUNet


MODEL_ARGUMENTS = {
    "architecture": "miniature",
    "character_count": 46,
    "font_count": 9,
    "timesteps": 1000,
    "base_channels": 8,
    "channel_multipliers": (1, 2, 3),
    "condition_dim": 32,
}


class MiniatureModelTests(unittest.TestCase):
    def test_model_stays_below_miniature_list_limit(self) -> None:
        model = build_model(MODEL_ARGUMENTS)
        self.assertLess(count_parameters(model), 200_000)

    def test_model_uses_miniature_friendly_layers(self) -> None:
        model = build_model(MODEL_ARGUMENTS)
        forbidden = (nn.GroupNorm, nn.SiLU, nn.ConvTranspose2d)
        self.assertFalse(
            any(isinstance(module, forbidden) for module in model.modules())
        )

    def test_forward_and_backward_preserve_64_pixel_shape(self) -> None:
        model = MiniatureConditionalUNet(
            character_count=46,
            font_count=9,
            timesteps=1000,
            base_channels=8,
            channel_multipliers=(1, 2, 3),
            condition_dim=32,
        )
        output = model(
            torch.randn(2, 1, 64, 64),
            torch.tensor([0, 999]),
            torch.tensor([0, 45]),
            torch.tensor([0, 8]),
        )

        self.assertEqual(output.shape, (2, 1, 64, 64))
        output.square().mean().backward()
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.parameters())
        )

    def test_factory_keeps_standard_checkpoint_compatibility(self) -> None:
        model = build_model(
            {
                "character_count": 46,
                "font_count": 9,
                "base_channels": 8,
                "channel_multipliers": (1, 2),
                "condition_dim": 32,
                "dropout": 0.0,
            }
        )
        self.assertNotIsInstance(model, MiniatureConditionalUNet)


if __name__ == "__main__":
    unittest.main()
