from __future__ import annotations

import io
import json
import logging

import numpy as np
from PIL import Image, ImageDraw

log = logging.getLogger("camera-analyzer.roi")

# En polygon som andelar (0.0-1.0) av bildens bredd/höjd, oberoende av
# faktisk upplösning. Punkterna definierar området som SKA IGNORERAS
# (t.ex. vägen bakom tomten) - resten av bilden (t.ex. grusplanen)
# bevakas som vanligt. Det är enklare att rita in en liten "störande" yta
# att maskera bort än att rita om hela det bevakade området för hand.
Polygon = list[tuple[float, float]]

ROI_MASK_PROMPT_NOTE = (
    "OBS: delar av bilden är medvetet svartmaskade (t.ex. vägen som INTE "
    "ska bevakas). De svarta partierna är inte en del av scenen - "
    "ignorera dem helt och kommentera dem aldrig."
)


def parse_roi_masks(raw: str) -> dict[str, Polygon]:
    """Läser ROI_MASKS env-variabeln: en JSON-dict från kameranamn (eller
    kamera-ID) till en polygon av (x,y)-punkter i andelar (0.0-1.0) av
    bildens bredd/höjd. Polygonen är området som ska IGNORERAS, t.ex.:

        {"Carport": [[0.42,0.0],[0.86,0.0],[0.86,0.14],[0.43,0.15]]}

    Tomt värde = ingen maskering (standard, bakåtkompatibelt beteende).
    """
    raw = raw.strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        log.error("Kunde inte tolka ROI_MASKS som JSON, ignorerar: %r", raw)
        return {}
    if not isinstance(data, dict):
        log.error("ROI_MASKS måste vara ett JSON-objekt (kameranamn -> polygon), ignorerar")
        return {}
    masks: dict[str, Polygon] = {}
    for name, points in data.items():
        try:
            polygon = [(float(x), float(y)) for x, y in points]
        except (TypeError, ValueError):
            log.error("Ogiltig polygon för %r i ROI_MASKS, hoppar över", name)
            continue
        if len(polygon) < 3:
            log.error("Polygon för %r i ROI_MASKS har färre än 3 punkter, hoppar över", name)
            continue
        masks[name] = polygon
    return masks


def _polygon_to_pixels(polygon: Polygon, width: int, height: int) -> list[tuple[float, float]]:
    return [(x * width, y * height) for x, y in polygon]


def build_bool_mask(polygon: Polygon, width: int, height: int) -> np.ndarray:
    """Bygger en boolean-mask med formen (height, width) - True innanför
    polygonen (det ignorerade området, t.ex. vägen)."""
    img = Image.new("L", (width, height), 0)
    ImageDraw.Draw(img).polygon(_polygon_to_pixels(polygon, width, height), fill=255)
    return np.asarray(img) > 0


def mask_image_inside_polygon(image_bytes: bytes, polygon: Polygon) -> bytes:
    """Svärtar allt innanför polygonen (t.ex. vägen) i bilden, så att
    Ollama-analysen inte "ser" det ignorerade området men resten av
    bilden (t.ex. grusplanen) syns oförändrad."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        mask_img = Image.new("L", img.size, 0)
        ImageDraw.Draw(mask_img).polygon(
            _polygon_to_pixels(polygon, *img.size), fill=255
        )
        result = Image.composite(Image.new("RGB", img.size, (0, 0, 0)), img, mask_img)
        buf = io.BytesIO()
        result.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
