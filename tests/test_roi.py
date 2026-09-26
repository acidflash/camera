"""Enhetstester för ROI-maskning (app/roi.py) - polygonen anger det
IGNORERADE området (t.ex. vägen), resten av bilden (t.ex. grusplanen)
bevakas som vanligt."""

import io
import unittest

import numpy as np
from PIL import Image

from app.roi import (
    build_bool_mask,
    mask_image_inside_polygon,
    parse_roi_masks,
)


class ParseRoiMasksTest(unittest.TestCase):
    def test_empty_string_gives_no_masks(self) -> None:
        self.assertEqual(parse_roi_masks(""), {})
        self.assertEqual(parse_roi_masks("   "), {})

    def test_parses_named_polygon(self) -> None:
        raw = '{"Carport": [[0.0,0.4],[1.0,0.4],[1.0,1.0],[0.0,1.0]]}'
        masks = parse_roi_masks(raw)
        self.assertEqual(
            masks["Carport"], [(0.0, 0.4), (1.0, 0.4), (1.0, 1.0), (0.0, 1.0)]
        )

    def test_invalid_json_gives_no_masks(self) -> None:
        self.assertEqual(parse_roi_masks("{not json"), {})

    def test_non_dict_json_gives_no_masks(self) -> None:
        self.assertEqual(parse_roi_masks("[1,2,3]"), {})

    def test_too_few_points_is_skipped(self) -> None:
        raw = '{"Carport": [[0.0,0.0],[1.0,1.0]]}'
        self.assertEqual(parse_roi_masks(raw), {})


class BuildBoolMaskTest(unittest.TestCase):
    def test_right_half_polygon_masks_right_half(self) -> None:
        polygon = [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)]
        mask = build_bool_mask(polygon, width=10, height=10)
        self.assertEqual(mask.shape, (10, 10))
        self.assertFalse(mask[:, :5].any())
        self.assertTrue(mask[:, 5:].all())


class MaskImageInsidePolygonTest(unittest.TestCase):
    def test_pixels_inside_polygon_are_blackened(self) -> None:
        # Stor bild med marginal runt polygonkanten, så JPEG-kompressionens
        # blockartefakter vid själva gränsen inte stör kontrollen.
        img = Image.new("RGB", (40, 40), (255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        polygon = [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)]
        result_bytes = mask_image_inside_polygon(buf.getvalue(), polygon)
        with Image.open(io.BytesIO(result_bytes)) as result:
            arr = np.asarray(result.convert("RGB"))
        self.assertTrue((arr[:, :16] > 245).all())
        self.assertTrue((arr[:, 24:] < 10).all())


if __name__ == "__main__":
    unittest.main()
