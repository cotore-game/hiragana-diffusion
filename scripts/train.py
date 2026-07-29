#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
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


def format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def print_progress(message: str, complete: bool = False) -> None:
    if sys.stdout.isatty():
        end = "\n" if complete else ""
        print(f"\r\033[2K{message}", end=end, flush=True)
    else:
        print(message, flush=True)


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
        "font_count": len(dataset.font_ids),
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

    epoch_count = int(config["epochs"])
    batches_per_epoch = len(loader)
    total_steps = epoch_count * batches_per_epoch
    initial_global_step = global_step
    log_every_steps = int(config.get("log_every_steps", 10))
    if log_every_steps <= 0:
        raise ValueError("log_every_steps must be positive")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"device={device} samples={len(dataset)} parameters={parameter_count:,}",
        flush=True,
    )
    print(f"font_ids={dataset.font_ids}", flush=True)
    print(
        f"epochs={start_epoch + 1}-{epoch_count} "
        f"batches_per_epoch={batches_per_epoch} "
        f"steps={global_step}/{total_steps}",
        flush=True,
    )

    training_started_at = time.monotonic()
    for epoch in range(start_epoch, epoch_count):
        model.train()
        total_loss = 0.0
        epoch_started_at = time.monotonic()
        for batch_index, (images, characters, font_ids) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            characters = characters.to(device, non_blocking=True)
            font_ids = font_ids.to(device, non_blocking=True)
            timesteps = torch.randint(
                0, diffusion.timesteps, (images.shape[0],), device=device
            )
            noise = torch.randn_like(images)
            noisy_images = diffusion.add_noise(images, timesteps, noise)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                predicted_noise = model(
                    noisy_images, timesteps, characters, font_ids
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

            should_log = (
                batch_index % log_every_steps == 0
                or batch_index == batches_per_epoch
            )
            if should_log:
                elapsed = time.monotonic() - training_started_at
                completed_this_run = global_step - initial_global_step
                seconds_per_step = elapsed / completed_this_run
                remaining_epoch_steps = batches_per_epoch - batch_index
                remaining_total_steps = total_steps - global_step
                progress_percent = 100.0 * global_step / total_steps
                progress_message = (
                    f"epoch={epoch + 1}/{epoch_count} "
                    f"batch={batch_index}/{batches_per_epoch} "
                    f"step={global_step}/{total_steps} "
                    f"progress={progress_percent:5.1f}% "
                    f"loss={loss.item():.6f} "
                    f"step_time={seconds_per_step:.3f}s "
                    f"epoch_eta={format_duration(remaining_epoch_steps * seconds_per_step)} "
                    f"total_eta={format_duration(remaining_total_steps * seconds_per_step)}"
                )
                print_progress(progress_message)

        average_loss = total_loss / len(dataset)
        epoch_elapsed = time.monotonic() - epoch_started_at
        print_progress(
            f"epoch_complete={epoch + 1}/{epoch_count} "
            f"loss={average_loss:.6f} "
            f"elapsed={format_duration(epoch_elapsed)}",
            complete=True,
        )

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
                "font_ids": dataset.font_ids,
                "config": config,
            }
            if bool(config.get("keep_numbered_checkpoints", False)):
                torch.save(checkpoint, output / f"checkpoint-{epoch + 1:04d}.pt")
            torch.save(checkpoint, output / "latest.pt")


if __name__ == "__main__":
    main()
