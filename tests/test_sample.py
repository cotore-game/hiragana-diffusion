from __future__ import annotations

import unittest

import torch

from scripts.sample import load_sampling_bundle


class SampleTests(unittest.TestCase):
    def test_loads_inference_only_archive(self) -> None:
        payload = {
            "format_version": 1,
            "state_dict": {
                "style_embedding.weight": torch.tensor([[1.0]])
            },
            "model_arguments": {"character_count": 46, "style_count": 2},
            "condition_names": ("gothic", "mincho"),
            "diffusion_timesteps": 1000,
            "weights": "model",
        }

        model_arguments, state_dict, conditions, timesteps, weights = (
            load_sampling_bundle(payload, use_ema=False)
        )

        self.assertEqual(model_arguments["character_count"], 46)
        self.assertEqual(model_arguments["font_count"], 2)
        self.assertNotIn("style_count", model_arguments)
        self.assertIn("font_embedding.weight", state_dict)
        self.assertEqual(conditions, ("gothic", "mincho"))
        self.assertEqual(timesteps, 1000)
        self.assertEqual(weights, "model")

    def test_rejects_ema_request_for_single_weight_archive(self) -> None:
        payload = {
            "format_version": 1,
            "state_dict": {},
            "model_arguments": {},
            "condition_names": (),
            "diffusion_timesteps": 1000,
            "weights": "model",
        }

        with self.assertRaisesRegex(ValueError, "one weight set"):
            load_sampling_bundle(payload, use_ema=True)
