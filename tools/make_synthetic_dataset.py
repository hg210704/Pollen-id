"""Create a small FAKE pollen dataset (drawn shapes) in the same layout as the real one.

Used only to test that the pipeline works end to end, never for real results.

    python tools/make_synthetic_dataset.py --out synthetic_data
    python scripts/prepare_data.py --root synthetic_data/paldat_images/LM \
        --metadata synthetic_data/paldat_images/metadata.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


def draw_grain(rng: random.Random, spikes: int, hue: tuple[int, int, int], size: int = 96) -> Image.Image:
    bg = (235, 235, 228)
    img = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(img)
    cx, cy = size / 2 + rng.uniform(-6, 6), size / 2 + rng.uniform(-6, 6)
    r = size * rng.uniform(0.26, 0.32)
    rot = rng.uniform(0, 2 * math.pi)
    for i in range(spikes):  # species signature: number of spikes
        a = rot + 2 * math.pi * i / spikes
        d.line([(cx + r * math.cos(a), cy + r * math.sin(a)),
                (cx + (r + size * 0.12) * math.cos(a), cy + (r + size * 0.12) * math.sin(a))],
               fill=tuple(max(0, c - 40) for c in hue), width=3)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=hue, outline=(60, 60, 60), width=2)
    return img.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.8)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic_data")
    ap.add_argument("--genera", type=int, default=3)
    ap.add_argument("--species-per-genus", type=int, default=3)
    ap.add_argument("--records", type=int, default=5, help="specimen records per species")
    ap.add_argument("--images-per-record", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    base = Path(a.out) / "paldat_images"
    names = ["Aaa", "Bbb", "Ccc", "Ddd", "Eee", "Fff"]
    rows = []
    k = 0
    for gi in range(a.genera):
        genus = names[gi % len(names)] + ("" if gi < len(names) else str(gi))
        for si in range(a.species_per_genus):
            species = f"{genus}_sp{si + 1}"
            spikes = 4 + k
            hue = (rng.randint(120, 230), rng.randint(120, 230), rng.randint(60, 160))
            k += 1
            folder = base / "LM" / species
            folder.mkdir(parents=True, exist_ok=True)
            n = 1
            for rec in range(a.records):
                page = f"https://example.org/pub/{species}/{1000 + rec}"
                for _ in range(a.images_per_record):
                    fname = f"{species}_LM_{n}.jpg"
                    draw_grain(rng, spikes, hue).save(folder / fname, quality=92)
                    rows.append({"genus": genus, "species": species, "caption": "hydrated pollen",
                                 "image_url": f"https://example.org/img/{fname}", "page_url": page,
                                 "file": f"LM/{species}/{fname}"})
                    n += 1
    with open(base / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["genus", "species", "caption", "image_url", "page_url", "file"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} images for {k} species to {base}")


if __name__ == "__main__":
    main()
