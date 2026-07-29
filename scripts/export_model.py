#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export inference-only weights from a training checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPOSITORY_ROOT / "outputs" / "takao-baseline-64" / "latest.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "artifacts"
            / "takao-baseline-64"
            / "model"
            / "inference.pt"
        ),
    )
    parser.add_argument(
        "--ema-model",
        action="store_true",
        help="Export EMA weights instead of directly trained weights.",
    )
    return parser.parse_args()


def build_inference_payload(
    checkpoint: dict[str, object],
    use_ema: bool,
) -> dict[str, object]:
    state_key = "ema_model" if use_ema else "model"
    training_config = dict(checkpoint["config"])
    condition_names = checkpoint.get("font_ids", checkpoint.get("styles"))
    if condition_names is None:
        raise ValueError("checkpoint contains no font_ids")

    return {
        "format_version": 1,
        "state_dict": checkpoint[state_key],
        "model_arguments": checkpoint["model_arguments"],
        "condition_names": tuple(condition_names),
        "diffusion_timesteps": int(training_config["timesteps"]),
        "image_size": 64,
        "weights": state_key,
        "training": {
            "epoch": int(checkpoint["epoch"]),
            "global_step": int(checkpoint["global_step"]),
            "dataset": str(training_config["dataset"]),
            "seed": int(training_config["seed"]),
        },
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    arguments = parse_arguments()
    checkpoint = torch.load(
        arguments.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    payload = build_inference_payload(checkpoint, arguments.ema_model)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, arguments.output)
    print(f"Exported model: {arguments.output}")
    print(f"weights={payload['weights']} size={arguments.output.stat().st_size}")
    print(f"sha256={sha256(arguments.output)}")


if __name__ == "__main__":
    main()
