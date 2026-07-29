#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from hiragana_diffusion.dataset_generation import HIRAGANA
from hiragana_diffusion.diffusion import GaussianDiffusion
from hiragana_diffusion.model import ConditionalUNet


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate conditional hiragana samples from a checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPOSITORY_ROOT / "outputs" / "takao-baseline-64" / "latest.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "outputs" / "takao-baseline-64" / "samples",
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument(
        "--ema-model",
        action="store_true",
        help="Use the EMA model instead of the directly trained model.",
    )
    return parser.parse_args()


def load_label_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        Path("/usr/share/fonts/truetype/takao-gothic/TakaoGothic.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def load_sampling_bundle(
    payload: dict[str, object],
    use_ema: bool,
) -> tuple[dict[str, object], dict[str, torch.Tensor], tuple[str, ...], int, str]:
    if int(payload.get("format_version", 0)) == 1:
        if use_ema:
            raise ValueError(
                "inference-only archives contain one weight set; "
                "do not specify --ema-model"
            )
        return (
            dict(payload["model_arguments"]),
            dict(payload["state_dict"]),
            tuple(payload["condition_names"]),
            int(payload["diffusion_timesteps"]),
            str(payload["weights"]),
        )

    state_key = "ema_model" if use_ema else "model"
    return (
        dict(payload["model_arguments"]),
        dict(payload[state_key]),
        tuple(payload["styles"]),
        int(dict(payload["config"])["timesteps"]),
        state_key,
    )


def to_images(tensor: torch.Tensor) -> list[Image.Image]:
    arrays = (
        ((tensor.clamp(-1.0, 1.0) + 1.0) * 127.5)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    return [Image.fromarray(array[0], mode="L") for array in arrays]


def save_outputs(
    images: list[Image.Image],
    styles: tuple[str, ...],
    output: Path,
    seed: int,
    steps: int,
) -> Path:
    character_count = len(HIRAGANA)
    image_size = images[0].width
    label_width = 88
    header_height = 28
    grid = Image.new(
        "L",
        (
            label_width + character_count * image_size,
            header_height + len(styles) * image_size,
        ),
        color=255,
    )
    draw = ImageDraw.Draw(grid)
    font = load_label_font(15)

    for character_index, character in enumerate(HIRAGANA):
        x = label_width + character_index * image_size + image_size // 2
        draw.text((x, 4), character, font=font, fill=0, anchor="mt")

    run_directory = output / f"seed-{seed}-steps-{steps}"
    for style_index, style in enumerate(styles):
        y = header_height + style_index * image_size
        draw.text((4, y + image_size // 2), style, font=font, fill=0, anchor="lm")
        style_directory = run_directory / style
        style_directory.mkdir(parents=True, exist_ok=True)

        for character_index, character in enumerate(HIRAGANA):
            image = images[style_index * character_count + character_index]
            x = label_width + character_index * image_size
            grid.paste(image, (x, y))
            image.save(
                style_directory
                / f"{character_index:02d}_U+{ord(character):04X}.png",
                format="PNG",
                optimize=True,
            )

    run_directory.mkdir(parents=True, exist_ok=True)
    grid_path = run_directory / "grid.png"
    grid.save(grid_path, format="PNG", optimize=True)
    return grid_path


def main() -> None:
    arguments = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(
        arguments.checkpoint,
        map_location=device,
        weights_only=False,
    )
    (
        model_arguments,
        state_dict,
        styles,
        diffusion_timesteps,
        weight_name,
    ) = load_sampling_bundle(payload, arguments.ema_model)
    model = ConditionalUNet(**model_arguments).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    character_count = int(model_arguments["character_count"])
    if character_count != len(HIRAGANA):
        raise ValueError(
            f"checkpoint expects {character_count} characters, "
            f"but the generator defines {len(HIRAGANA)}"
        )

    generator = torch.Generator(device=device).manual_seed(arguments.seed)
    base_noise = torch.randn(
        character_count,
        1,
        arguments.image_size,
        arguments.image_size,
        generator=generator,
        device=device,
    )
    initial_noise = base_noise.repeat(len(styles), 1, 1, 1)
    characters = torch.arange(character_count, device=device).repeat(len(styles))
    style_indices = torch.arange(len(styles), device=device).repeat_interleave(
        character_count
    )

    diffusion = GaussianDiffusion(diffusion_timesteps, device)
    print(
        f"device={device} model={weight_name} samples={len(initial_noise)} "
        f"steps={arguments.steps}"
    )
    generated = diffusion.ddim_sample(
        model=model,
        initial_noise=initial_noise,
        characters=characters,
        styles=style_indices,
        sampling_steps=arguments.steps,
    )
    grid_path = save_outputs(
        images=to_images(generated),
        styles=styles,
        output=arguments.output,
        seed=arguments.seed,
        steps=arguments.steps,
    )
    print(f"Saved samples: {grid_path}")


if __name__ == "__main__":
    main()
