from __future__ import annotations

import unittest

import torch

from hiragana_diffusion.diffusion import GaussianDiffusion, cosine_beta_schedule
from hiragana_diffusion.model import ConditionalUNet


class TrainingTests(unittest.TestCase):
    def test_cosine_schedule_is_valid(self) -> None:
        betas = cosine_beta_schedule(1000)
        self.assertEqual(betas.shape, (1000,))
        self.assertTrue(torch.all(betas > 0))
        self.assertTrue(torch.all(betas < 1))

    def test_noise_formula_preserves_shape(self) -> None:
        diffusion = GaussianDiffusion(100, torch.device("cpu"))
        images = torch.randn(4, 1, 64, 64)
        timesteps = torch.tensor([0, 1, 50, 99])
        noisy = diffusion.add_noise(images, timesteps, torch.randn_like(images))
        self.assertEqual(noisy.shape, images.shape)

    def test_unet_preserves_image_shape(self) -> None:
        model = ConditionalUNet(
            character_count=46,
            style_count=3,
            base_channels=16,
            channel_multipliers=(1, 2, 4),
            condition_dim=64,
            dropout=0.0,
        )
        output = model(
            torch.randn(2, 1, 64, 64),
            torch.tensor([0, 999]),
            torch.tensor([0, 45]),
            torch.tensor([0, 2]),
        )
        self.assertEqual(output.shape, (2, 1, 64, 64))
        output.square().mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
