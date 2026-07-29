from __future__ import annotations

import unittest

import torch

from scripts.export_model import build_inference_payload


class ExportModelTests(unittest.TestCase):
    def test_payload_excludes_training_state(self) -> None:
        checkpoint = {
            "model": {"weight": torch.tensor([1.0])},
            "ema_model": {"weight": torch.tensor([2.0])},
            "optimizer": {"unnecessary": True},
            "scaler": {"unnecessary": True},
            "model_arguments": {"character_count": 46},
            "styles": ("gothic", "mincho"),
            "epoch": 100,
            "global_step": 7200,
            "config": {
                "dataset": "data/datasets/takao-baseline-64",
                "seed": 20260729,
                "timesteps": 1000,
            },
        }

        payload = build_inference_payload(checkpoint, use_ema=False)

        self.assertEqual(payload["weights"], "model")
        self.assertIs(payload["state_dict"], checkpoint["model"])
        self.assertNotIn("optimizer", payload)
        self.assertNotIn("scaler", payload)
        self.assertEqual(payload["training"]["global_step"], 7200)
