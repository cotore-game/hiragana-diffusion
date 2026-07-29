#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from hiragana_diffusion.dataset_generation import generate_dataset, load_config


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a reproducible grayscale PNG hiragana dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "dataset.example.json",
        help="Path to the dataset JSON configuration.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "datasets",
        help="Directory in which the named dataset directory is created.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config_path = arguments.config.resolve()
    config = load_config(config_path)
    dataset_root = generate_dataset(
        config=config,
        output_root=arguments.output_root.resolve(),
        config_source=config_path,
    )
    print(f"Generated dataset: {dataset_root}")


if __name__ == "__main__":
    main()
