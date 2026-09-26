#!/usr/bin/env python3
"""Kalibreringsverktyg för ROI_MASKS.

Tar en exempelbild (t.ex. en snapshot från Protects webbgränssnitt av
Carport-kameran) och en polygon, och sparar en förhandsvisning där allt
INNANFÖR polygonen är rödtonat - så du kan se om det ignorerade området
(t.ex. vägen) stämmer innan du sätter ROI_MASKS i .env.

Användning:
    python scripts/roi_preview.py carport.jpg \\
        --polygon "0.42,0.0 0.86,0.0 0.86,0.14 0.43,0.15" \\
        --out carport_preview.jpg

Punkterna är andelar (0.0-1.0) av bildens bredd/höjd, i medurs eller
moturs ordning, samma format som en post i ROI_MASKS-JSON:en. Justera
punkterna, kör om, tills det röda området täcker exakt det du vill
ignorera (t.ex. vägen) och resten av bilden (t.ex. grusplanen) förblir
klar/obemängd. Klistra sedan in samma punkter i ROI_MASKS i .env:

    ROI_MASKS={"Carport": [[0.42,0.0],[0.86,0.0],[0.86,0.14],[0.43,0.15]]}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.roi import Polygon  # noqa: E402


def parse_polygon(raw: str) -> Polygon:
    points: Polygon = []
    for pair in raw.split():
        x_str, y_str = pair.split(",")
        points.append((float(x_str), float(y_str)))
    if len(points) < 3:
        raise ValueError("Polygonen behöver minst 3 punkter")
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", help="Sökväg till en exempelbild från kameran")
    parser.add_argument(
        "--polygon",
        required=True,
        help='Punkter som "x,y x,y x,y ..." i andelar 0.0-1.0, t.ex. "0.42,0.0 0.86,0.0 0.86,0.14 0.43,0.15"',
    )
    parser.add_argument("--out", default="roi_preview.jpg", help="Var förhandsvisningen sparas")
    args = parser.parse_args()

    polygon = parse_polygon(args.polygon)

    with Image.open(args.image) as img:
        img = img.convert("RGB")
        width, height = img.size
        pixels = [(x * width, y * height) for x, y in polygon]

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        # Rödtonad, halvgenomskinlig overlay över allt INNANFÖR polygonen,
        # så det maskerade (ignorerade) området syns tydligt.
        mask = Image.new("L", img.size, 0)
        ImageDraw.Draw(mask).polygon(pixels, fill=255)
        draw.bitmap((0, 0), mask, fill=(255, 0, 0, 110))
        draw.line(pixels + [pixels[0]], fill=(255, 255, 0, 255), width=3)

        result = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        result.save(args.out, format="JPEG", quality=90)

    json_points = ",".join(f"[{x},{y}]" for x, y in polygon)
    print(f"Sparade förhandsvisning: {args.out}")
    print("Röd/genomskinlig yta = maskeras bort (ignoreras). Gul linje = polygonens kant.")
    print()
    print("Om det stämmer, klistra in i .env:")
    print(f'ROI_MASKS={{"<kameranamn>": [{json_points}]}}')


if __name__ == "__main__":
    main()
