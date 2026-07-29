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

        self.font_ids = tuple(sorted({row["font_id"] for row in self.rows}))
        self.font_id_to_index = {
            font_id: index for index, font_id in enumerate(self.font_ids)
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
        font_id = torch.tensor(
            self.font_id_to_index[row["font_id"]], dtype=torch.long
        )
        return image_tensor, character, font_id
