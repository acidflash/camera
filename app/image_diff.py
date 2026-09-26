from __future__ import annotations

import io
import logging

import numpy as np
from PIL import Image

log = logging.getLogger("camera-analyzer.diff")


def diff_ratio(
    baseline: bytes,
    current: bytes,
    resize: tuple[int, int] = (320, 240),
    pixel_threshold: int = 30,
    mask: np.ndarray | None = None,
) -> float:
    """Andel pixlar (0.0-1.0) som skiljer sig markant mellan två bilder.

    Bilderna skalas ner och görs gråskaliga innan jämförelse, dels för
    prestanda, dels för att dämpa brus/kompressionsartefakter som annars
    ger falska skillnader mellan i övrigt identiska bilder.

    `mask`, om satt, är en boolean-array med formen (resize[1], resize[0])
    (höjd, bredd) - bara pixlar där masken är True (t.ex. grusplanen, inte
    vägen) räknas med i jämförelsen, både i täljare och nämnare.
    """
    a = _load_grayscale(baseline, resize)
    b = _load_grayscale(current, resize)
    delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
    changed = delta > pixel_threshold
    if mask is not None:
        changed = changed & mask
        total = int(np.count_nonzero(mask))
    else:
        total = delta.size
    if total == 0:
        return 0.0
    return np.count_nonzero(changed) / total


def _load_grayscale(image_bytes: bytes, resize: tuple[int, int]) -> np.ndarray:
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("L").resize(resize)
        return np.asarray(img)
