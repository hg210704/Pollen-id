"""Synthetic training images, made ONLY from real TRAIN images.

Why: most species have about three photos. Synthetic variants (any rotation, stain colour, focus,
brightness, noise, zoom) teach the model to ignore those nuisances so it can recognise the pollen
shape itself, and they balance species that have very few images.

What it cannot do: invent a new specimen. A variant carries no more botanical information than the
photo it came from, so real extra images will always help more.

Safety rules built in:
  * variants are made only from images in the train split, and are added only to the train split;
  * validation and test images are never touched, so reported accuracy still measures real images;
  * if the splits are rebuilt later and a source image moves to val/test, its variants are dropped
    automatically when the training data is loaded (see data.add_synthetic_rows).
No PyTorch needed.
"""
from __future__ import annotations

import io
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

from .config import resolve_path

MANIFEST = "synthetic_manifest.csv"
STAIN_TINTS = [(1.00, 0.55, 0.45), (1.00, 0.80, 0.40), (0.60, 0.80, 1.00), (0.70, 1.00, 0.70), (0.90, 0.60, 1.00)]


def _corner_color(img: Image.Image) -> tuple[int, int, int]:
    a = np.asarray(img.convert("RGB"))
    corners = np.stack([a[0, 0], a[0, -1], a[-1, 0], a[-1, -1]]).astype(np.float32)
    return tuple(int(c) for c in corners.mean(0))


def make_variant(img: Image.Image, rng: random.Random) -> tuple[Image.Image, list[str]]:
    """One random, morphology-preserving variant of an RGB image (same size as the input)."""
    img = img.convert("RGB")
    w, h = img.size
    ops: list[str] = []

    # geometry: pollen has no 'up', so rotation and mirroring are safe
    img = img.rotate(rng.uniform(0, 360), resample=Image.BICUBIC, expand=True, fillcolor=_corner_color(img))
    ops.append("rotate")
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT); ops.append("flip")
    s = rng.uniform(0.72, 1.0)  # zoom: crop a random window, scale back up
    cw, ch = max(8, int(img.width * s)), max(8, int(img.height * s))
    x0, y0 = rng.randint(0, img.width - cw), rng.randint(0, img.height - ch)
    img = img.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.BICUBIC)
    ops.append(f"zoom{s:.2f}")

    # colour: slides are stained in many colours and lit differently
    arr = np.asarray(img, dtype=np.float32)
    arr = arr * np.array([rng.uniform(0.85, 1.15) for _ in range(3)], dtype=np.float32)
    if rng.random() < 0.35:
        tint = np.array(rng.choice(STAIN_TINTS), dtype=np.float32)
        alpha = rng.uniform(0.15, 0.40)
        arr = arr * ((1 - alpha) + alpha * tint * 1.2)
        ops.append("stain")
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    if rng.random() < 0.5:  # small hue rotation
        hsv = np.asarray(img.convert("HSV")).copy()
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) + int(rng.uniform(-14, 14))) % 256
        img = Image.fromarray(hsv, "HSV").convert("RGB"); ops.append("hue")
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.82, 1.18))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.82, 1.2))
    if rng.random() < 0.25:
        img = ImageEnhance.Color(img).enhance(rng.uniform(0.6, 1.4))

    # optics and sensor: focus blur, noise, compression
    if rng.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.3))); ops.append("blur")
    if rng.random() < 0.4:
        a = np.asarray(img, dtype=np.float32)
        img = Image.fromarray((a + np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(2, 7), a.shape))
                              .clip(0, 255).astype(np.uint8)); ops.append("noise")
    if rng.random() < 0.3:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=rng.randint(60, 92))
        buf.seek(0)
        img = Image.open(buf).convert("RGB"); ops.append("jpeg")
    return img, ops


def synthetic_dir(cfg: dict) -> Path:
    return resolve_path(cfg["paths"]["cache_dir"]) / "synthetic"


def build_synthetic(cfg: dict, clean: bool = False, seed: int = 42) -> dict:
    """Top every species up to data.synthetic_target_per_species TRAIN images with variants."""
    dcfg = cfg["data"]
    target = int(dcfg.get("synthetic_target_per_species", 12))
    max_per_source = int(dcfg.get("synthetic_max_per_source", 6))
    cache = resolve_path(cfg["paths"]["cache_dir"])
    splits_path = cache / "splits.csv"
    if not splits_path.exists():
        raise FileNotFoundError("Run  python scripts/prepare_data.py  first (it creates the train/val/test split).")
    out_root = synthetic_dir(cfg)
    if clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(splits_path, keep_default_na=False)
    train = df[df["split"] == "train"]
    old = pd.read_csv(cache / MANIFEST, dtype=str, keep_default_na=False) if (cache / MANIFEST).exists() \
        else pd.DataFrame(columns=["species", "path", "source", "ops"])
    old = old[old["path"].map(lambda p: Path(p).exists())]
    rows = old.to_dict("records")

    created = 0
    for species, sdf in train.groupby("species"):
        sources = sorted(sdf["path"])
        have = sum(1 for r in rows if r["species"] == species)
        need = min(max(0, target - len(sources)), max_per_source * len(sources)) - have
        if need <= 0:
            continue
        folder = out_root / species
        folder.mkdir(exist_ok=True)
        k = have
        for i in range(need):
            src = sources[(have + i) % len(sources)]  # round-robin so every source is used evenly
            rng = random.Random(f"{seed}-{species}-{k}")
            with Image.open(src) as im:
                variant, ops = make_variant(im, rng)
            while (folder / f"{species}_SYN_{k + 1}.jpg").exists():
                k += 1
            path = folder / f"{species}_SYN_{k + 1}.jpg"
            variant.save(path, quality=92)
            rows.append({"species": species, "path": str(path), "source": src, "ops": "+".join(ops)})
            k += 1
            created += 1
    manifest = pd.DataFrame(rows, columns=["species", "path", "source", "ops"])
    manifest.to_csv(cache / MANIFEST, index=False)
    summary = {"created": created, "total_synthetic": int(len(manifest)),
               "species_with_synthetic": int(manifest["species"].nunique()) if len(manifest) else 0,
               "target_per_species": target, "folder": str(out_root)}
    (cache / "synthetic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
