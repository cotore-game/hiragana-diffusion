#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from hiragana_diffusion.diffusion import GaussianDiffusion
from hiragana_diffusion.model import ConditionalUNet
from hiragana_diffusion.training_data import HiraganaDataset


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a conditional hiragana DDPM.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "train.example.json",
    )
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
        ema_parameter.lerp_(parameter, 1.0 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


def main() -> None:
    arguments = parse_arguments()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = HiraganaDataset(REPOSITORY_ROOT / config["dataset"])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(config["num_workers"]) > 0,
        generator=generator,
    )
    model_arguments = {
        "character_count": dataset.character_count,
        "style_count": len(dataset.styles),
        "base_channels": int(config["base_channels"]),
        "channel_multipliers": tuple(config["channel_multipliers"]),
        "condition_dim": int(config["condition_dim"]),
        "dropout": float(config["dropout"]),
    }
    model = ConditionalUNet(**model_arguments).to(device)
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    diffusion = GaussianDiffusion(int(config["timesteps"]), device)
    use_amp = bool(config["mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    output = REPOSITORY_ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    global_step = 0

    if arguments.resume:
        checkpoint = torch.load(arguments.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        ema_model.load_state_dict(checkpoint["ema_model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"device={device} samples={len(dataset)} parameters={parameter_count:,}")
    print(f"styles={dataset.styles}")

    for epoch in range(start_epoch, int(config["epochs"])):
        model.train()
        total_loss = 0.0
        for images, characters, styles in loader:
            images = images.to(device, non_blocking=True)
            characters = characters.to(device, non_blocking=True)
            styles = styles.to(device, non_blocking=True)
            timesteps = torch.randint(
                0, diffusion.timesteps, (images.shape[0],), device=device
            )
            noise = torch.randn_like(images)
            noisy_images = diffusion.add_noise(images, timesteps, noise)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                predicted_noise = model(
                    noisy_images, timesteps, characters, styles
                )
                loss = nn.functional.mse_loss(predicted_noise, noise)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                model.parameters(), float(config["gradient_clip_norm"])
            )
            scaler.step(optimizer)
            scaler.update()
            update_ema(ema_model, model, float(config["ema_decay"]))
            total_loss += loss.item() * images.shape[0]
            global_step += 1

        average_loss = total_loss / len(dataset)
        print(f"epoch={epoch + 1}/{config['epochs']} loss={average_loss:.6f}")

        should_save = (
            (epoch + 1) % int(config["save_every_epochs"]) == 0
            or epoch + 1 == int(config["epochs"])
        )
        if should_save:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "model_arguments": model_arguments,
                "styles": dataset.styles,
                "config": config,
            }
            torch.save(checkpoint, output / f"checkpoint-{epoch + 1:04d}.pt")
            torch.save(checkpoint, output / "latest.pt")


if __name__ == "__main__":
    main()
