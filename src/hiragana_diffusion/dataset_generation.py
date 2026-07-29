from __future__ import annotations

import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


HIRAGANA = (
    "あいうえお"
    "かきくけこ"
    "さしすせそ"
    "たちつてと"
    "なにぬねの"
    "はひふへほ"
    "まみむめも"
    "やゆよ"
    "らりるれろ"
    "わをん"
)


@dataclass(frozen=True)
class Range:
    minimum: float
    maximum: float

    @classmethod
    def from_json(cls, value: object, name: str) -> Range:
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"{name} must be a two-element array")

        minimum = float(value[0])
        maximum = float(value[1])
        if minimum > maximum:
            raise ValueError(f"{name} minimum must not exceed maximum")
        return cls(minimum, maximum)

    def sample(self, random_generator: random.Random) -> float:
        return random_generator.uniform(self.minimum, self.maximum)


@dataclass(frozen=True)
class AugmentationConfig:
    translate_x: Range
    translate_y: Range
    scale: Range
    rotation_degrees: Range


@dataclass(frozen=True)
class FontConfig:
    id: str
    path: Path
    exclude_characters: frozenset[str]


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    image_size: int
    font_size: int
    samples_per_character: int
    seed: int
    supersampling: int
    augmentation: AugmentationConfig
    fonts: tuple[FontConfig, ...]


def _require_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def load_config(path: Path) -> DatasetConfig:
    raw = _require_mapping(json.loads(path.read_text(encoding="utf-8")), "config")
    augmentation_raw = _require_mapping(raw.get("augmentation"), "augmentation")
    fonts_raw = raw.get("fonts")

    if not isinstance(fonts_raw, list) or not fonts_raw:
        raise ValueError("fonts must be a non-empty array")

    fonts: list[FontConfig] = []
    font_ids: set[str] = set()
    for index, font_value in enumerate(fonts_raw):
        font_raw = _require_mapping(font_value, f"fonts[{index}]")
        font_id = str(font_raw["id"])
        if not font_id or font_id in font_ids:
            raise ValueError(f"font id must be non-empty and unique: {font_id!r}")
        font_ids.add(font_id)

        excluded_raw = font_raw.get("exclude_characters", [])
        if isinstance(excluded_raw, str):
            excluded = frozenset(excluded_raw)
        elif isinstance(excluded_raw, list) and all(
            isinstance(character, str) and len(character) == 1
            for character in excluded_raw
        ):
            excluded = frozenset(excluded_raw)
        else:
            raise ValueError(
                f"fonts[{index}].exclude_characters must be a string "
                "or an array of one-character strings"
            )
        unknown = excluded.difference(HIRAGANA)
        if unknown:
            raise ValueError(
                f"fonts[{index}].exclude_characters contains unknown characters: "
                f"{''.join(sorted(unknown))}"
            )

        fonts.append(
            FontConfig(
                id=font_id,
                path=Path(str(font_raw["path"])).expanduser(),
                exclude_characters=excluded,
            )
        )

    config = DatasetConfig(
        name=str(raw["name"]),
        image_size=int(raw["image_size"]),
        font_size=int(raw["font_size"]),
        samples_per_character=int(raw["samples_per_character"]),
        seed=int(raw["seed"]),
        supersampling=int(raw.get("supersampling", 4)),
        augmentation=AugmentationConfig(
            translate_x=Range.from_json(
                augmentation_raw.get("translate_x"), "augmentation.translate_x"
            ),
            translate_y=Range.from_json(
                augmentation_raw.get("translate_y"), "augmentation.translate_y"
            ),
            scale=Range.from_json(
                augmentation_raw.get("scale"), "augmentation.scale"
            ),
            rotation_degrees=Range.from_json(
                augmentation_raw.get("rotation_degrees"),
                "augmentation.rotation_degrees",
            ),
        ),
        fonts=tuple(fonts),
    )
    _validate_config(config)
    return config


def _validate_config(config: DatasetConfig) -> None:
    if not config.name or Path(config.name).name != config.name:
        raise ValueError("name must be a single non-empty path component")
    if config.image_size <= 0:
        raise ValueError("image_size must be positive")
    if config.font_size <= 0:
        raise ValueError("font_size must be positive")
    if config.samples_per_character <= 0:
        raise ValueError("samples_per_character must be positive")
    if config.supersampling <= 0:
        raise ValueError("supersampling must be positive")
    if config.augmentation.scale.minimum <= 0:
        raise ValueError("augmentation.scale values must be positive")

    for font in config.fonts:
        if not font.path.is_file():
            raise FileNotFoundError(f"font does not exist: {font.path}")


def _render_glyph_mask(
    character: str,
    font: ImageFont.FreeTypeFont,
    scale: float,
    rotation_degrees: float,
) -> Image.Image:
    probe = Image.new("L", (1, 1))
    probe_draw = ImageDraw.Draw(probe)
    left, top, right, bottom = probe_draw.textbbox((0, 0), character, font=font)
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        raise ValueError(f"font produced an empty glyph for {character!r}")

    mask = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(mask)
    draw.text((-left, -top), character, font=font, fill=255)

    scaled_size = (
        max(1, round(mask.width * scale)),
        max(1, round(mask.height * scale)),
    )
    mask = mask.resize(scaled_size, resample=Image.Resampling.LANCZOS)
    return mask.rotate(
        rotation_degrees,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=0,
    )


def render_image(
    character: str,
    font: ImageFont.FreeTypeFont,
    image_size: int,
    supersampling: int,
    translate_x: float,
    translate_y: float,
    scale: float,
    rotation_degrees: float,
) -> Image.Image:
    canvas_size = image_size * supersampling
    mask = _render_glyph_mask(character, font, scale, rotation_degrees)

    x = round((canvas_size - mask.width) / 2 + translate_x * supersampling)
    y = round((canvas_size - mask.height) / 2 + translate_y * supersampling)
    canvas = Image.new("L", (canvas_size, canvas_size), color=0)
    canvas.paste(mask, (x, y))

    canvas = canvas.resize(
        (image_size, image_size),
        resample=Image.Resampling.LANCZOS,
    )
    return ImageOps.invert(canvas)


def generate_dataset(
    config: DatasetConfig,
    output_root: Path,
    config_source: Path,
) -> Path:
    dataset_root = output_root / config.name
    if dataset_root.exists() and any(dataset_root.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {dataset_root}. "
            "Choose another dataset name or move the existing dataset."
        )

    dataset_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_source, dataset_root / "config.json")
    manifest_path = dataset_root / "manifest.csv"
    random_generator = random.Random(config.seed)

    fieldnames = [
        "relative_path",
        "character",
        "codepoint",
        "character_index",
        "font_id",
        "sample_index",
        "translate_x",
        "translate_y",
        "scale",
        "rotation_degrees",
    ]

    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()

        for font_config in config.fonts:
            font = ImageFont.truetype(
                str(font_config.path),
                config.font_size * config.supersampling,
            )
            for character_index, character in enumerate(HIRAGANA):
                if character in font_config.exclude_characters:
                    continue

                character_directory = (
                    dataset_root
                    / font_config.id
                    / f"{character_index:02d}_U+{ord(character):04X}"
                )
                character_directory.mkdir(parents=True, exist_ok=True)

                for sample_index in range(config.samples_per_character):
                    if sample_index == 0:
                        translate_x = 0.0
                        translate_y = 0.0
                        scale = 1.0
                        rotation_degrees = 0.0
                    else:
                        translate_x = config.augmentation.translate_x.sample(
                            random_generator
                        )
                        translate_y = config.augmentation.translate_y.sample(
                            random_generator
                        )
                        scale = config.augmentation.scale.sample(random_generator)
                        rotation_degrees = (
                            config.augmentation.rotation_degrees.sample(
                                random_generator
                            )
                        )

                    image = render_image(
                        character=character,
                        font=font,
                        image_size=config.image_size,
                        supersampling=config.supersampling,
                        translate_x=translate_x,
                        translate_y=translate_y,
                        scale=scale,
                        rotation_degrees=rotation_degrees,
                    )
                    image_path = character_directory / f"{sample_index:04d}.png"
                    image.save(image_path, format="PNG", optimize=True)

                    writer.writerow(
                        {
                            "relative_path": image_path.relative_to(
                                dataset_root
                            ).as_posix(),
                            "character": character,
                            "codepoint": f"U+{ord(character):04X}",
                            "character_index": character_index,
                            "font_id": font_config.id,
                            "sample_index": sample_index,
                            "translate_x": f"{translate_x:.6f}",
                            "translate_y": f"{translate_y:.6f}",
                            "scale": f"{scale:.6f}",
                            "rotation_degrees": f"{rotation_degrees:.6f}",
                        }
                    )

    return dataset_root
