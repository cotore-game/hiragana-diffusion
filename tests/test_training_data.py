from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from hiragana_diffusion.training_data import HiraganaDataset


class TrainingDataTests(unittest.TestCase):
    def test_uses_individual_font_ids_as_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            rows = []
            for font_id in ("font-a", "font-b"):
                relative_path = Path(font_id) / "sample.png"
                image_path = root / relative_path
                image_path.parent.mkdir(parents=True)
                Image.new("L", (8, 8), color=255).save(image_path)
                rows.append(
                    {
                        "relative_path": relative_path.as_posix(),
                        "character_index": "0",
                        "font_id": font_id,
                    }
                )

            with (root / "manifest.csv").open(
                "w", encoding="utf-8", newline=""
            ) as manifest:
                writer = csv.DictWriter(
                    manifest,
                    fieldnames=("relative_path", "character_index", "font_id"),
                )
                writer.writeheader()
                writer.writerows(rows)

            dataset = HiraganaDataset(root)

            self.assertEqual(dataset.font_ids, ("font-a", "font-b"))
            self.assertEqual(dataset[0][2].item(), 0)
            self.assertEqual(dataset[1][2].item(), 1)


if __name__ == "__main__":
    unittest.main()
