from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class HiraganaDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, root: Path) -> None:
        self.root = root
        manifest = root / "manifest.csv"
        if not manifest.is_file():
            raise FileNotFoundError(f"manifest not found: {manifest}")

        with manifest.open(encoding="utf-8", newline="") as file:
            self.rows = list(csv.DictReader(file))
        if not self.rows:
            raise ValueError(f"manifest contains no samples: {manifest}")

        self.styles = tuple(sorted({row["style"] for row in self.rows}))
        self.style_to_index = {
            style: index for index, style in enumerate(self.styles)
        }
        self.character_count = max(int(row["character_index"]) for row in self.rows) + 1

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        image_path = self.root / row["relative_path"]
        with Image.open(image_path) as image:
            grayscale = image.convert("L")
            array = np.asarray(grayscale, dtype=np.float32).copy()

        image_tensor = torch.from_numpy(array).unsqueeze(0) / 127.5 - 1.0
        character = torch.tensor(int(row["character_index"]), dtype=torch.long)
        style = torch.tensor(self.style_to_index[row["style"]], dtype=torch.long)
        return image_tensor, character, style
