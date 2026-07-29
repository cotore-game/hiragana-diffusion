from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import ImageFont

from hiragana_diffusion.dataset_generation import HIRAGANA, render_image


TAKAO_GOTHIC = Path(
    "/usr/share/fonts/truetype/takao-gothic/TakaoGothic.ttf"
)


class DatasetGenerationTests(unittest.TestCase):
    def test_modern_hiragana_list_contains_46_characters(self) -> None:
        self.assertEqual(len(HIRAGANA), 46)
        self.assertEqual(len(set(HIRAGANA)), 46)

    @unittest.skipUnless(TAKAO_GOTHIC.is_file(), "Takao Gothic is unavailable")
    def test_rendered_image_is_grayscale_png_compatible(self) -> None:
        font = ImageFont.truetype(str(TAKAO_GOTHIC), 48 * 4)
        image = render_image(
            character="あ",
            font=font,
            image_size=64,
            supersampling=4,
            translate_x=0.0,
            translate_y=0.0,
            scale=1.0,
            rotation_degrees=0.0,
        )

        self.assertEqual(image.mode, "L")
        self.assertEqual(image.size, (64, 64))
        self.assertLess(image.getextrema()[0], 255)
        self.assertEqual(image.getextrema()[1], 255)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "sample.png"
            image.save(output_path)
            self.assertEqual(output_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
