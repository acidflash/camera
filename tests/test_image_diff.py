"""Enhetstester för den mekaniska pixeldiffen (app/image_diff.py).

Den här funktionen är den "mekaniska spärren" i flödet – den garanterar att
t.ex. en bil som stått parkerad hela tiden aldrig utlöser en notis. Därför är
det extra viktigt att dess beteende är förutsägbart.
"""

import io
import unittest

import numpy as np
from PIL import Image

from app.image_diff import diff_ratio


def _png_gray(rows: list[list[int]]) -> bytes:
    """Bygger en gråskale-PNG från en radlista med pixelvärden."""
    img = Image.new("L", (len(rows[0]), len(rows)))
    for y, row in enumerate(rows):
        for x, value in enumerate(row):
            img.putpixel((x, y), value)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _solid(rows: int, cols: int, value: int) -> bytes:
    return _png_gray([[value] * cols for _ in range(rows)])


class DiffRatioTest(unittest.TestCase):
    def test_identical_images_are_zero(self) -> None:
        a = _solid(10, 10, 128)
        self.assertEqual(diff_ratio(a, a), 0.0)

    def test_completely_different_images_are_high(self) -> None:
        a = _solid(10, 10, 10)
        b = _solid(10, 10, 240)
        self.assertGreater(diff_ratio(a, b), 0.9)

    def test_single_changed_pixel_is_small_ratio(self) -> None:
        # Stor bild (320x240) med exakt en markant ändrad pixel -> liten, men
        # inte noll, skillnad.
        rows = [[100] * 320 for _ in range(240)]
        rows[120][160] = 200  # delta 100 > pixel_threshold=30
        a = _solid(240, 320, 100)
        ratio = diff_ratio(a, _png_gray(rows))
        self.assertGreater(ratio, 0.0)
        self.assertLess(ratio, 0.1)

    def test_below_threshold_change_counts_as_zero(self) -> None:
        # delta 20 < pixel_threshold=30 -> ingen pixel räknas som ändrad.
        rows = [[100] * 20 for _ in range(20)]
        rows[10][10] = 120
        a = _solid(20, 20, 100)
        self.assertEqual(diff_ratio(a, _png_gray(rows)), 0.0)

    def test_mask_ignores_change_outside_masked_area(self) -> None:
        # Ändringen sker i vänstra halvan (t.ex. "vägen"); masken bevakar
        # bara högra halvan (t.ex. "grusplanen") -> ska räknas som 0.
        rows = [[100] * 20 for _ in range(20)]
        for y in range(20):
            rows[y][2] = 240  # stor ändring, men i omaskerad kolumn
        a = _solid(20, 20, 100)
        mask = np.zeros((20, 20), dtype=bool)
        mask[:, 10:] = True  # bara högra halvan bevakas
        self.assertEqual(diff_ratio(a, _png_gray(rows), resize=(20, 20), mask=mask), 0.0)

    def test_mask_counts_change_inside_masked_area(self) -> None:
        rows = [[100] * 20 for _ in range(20)]
        for y in range(20):
            rows[y][15] = 240  # ändring i bevakad kolumn
        a = _solid(20, 20, 100)
        mask = np.zeros((20, 20), dtype=bool)
        mask[:, 10:] = True
        ratio = diff_ratio(a, _png_gray(rows), resize=(20, 20), mask=mask)
        self.assertGreater(ratio, 0.0)
        # Nämnaren är maskens storlek (200 pixlar), inte hela bilden (400).
        self.assertAlmostEqual(ratio, 20 / 200)

    def test_empty_mask_is_zero(self) -> None:
        a = _solid(10, 10, 100)
        b = _solid(10, 10, 240)
        mask = np.zeros((10, 10), dtype=bool)
        self.assertEqual(diff_ratio(a, b, resize=(10, 10), mask=mask), 0.0)


if __name__ == "__main__":
    unittest.main()
