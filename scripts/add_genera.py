"""Add new genus/species folders to the dataset, then (optionally) train again.

    python scripts/add_genera.py --src D:/new/paldat_images            # copy + rebuild splits only
    python scripts/add_genera.py --src D:/new/paldat_images --train finetune
    python scripts/add_genera.py --src D:/new/paldat_images --train retrain

--src is either a paldat_images folder (containing LM/ and metadata.csv) or a folder that directly
holds <Genus_species>/ sub-folders. Existing train/val/test assignments are KEPT, so images that were
in the test set never move into training.

  finetune : start from the active model, new classes get new head rows (much faster)
  retrain  : train from ImageNet weights on everything (best final accuracy)
"""
import _bootstrap  # noqa: F401
import argparse
import json
import logging
import shutil
from pathlib import Path

import pandas as pd

from pollen.config import load_config, resolve_path
from pollen.data import build_dataset
from pollen.registry import active_run_id
from pollen.train import train

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--src", required=True)
ap.add_argument("--train", choices=["none", "finetune", "retrain"], default="none")
a = ap.parse_args()
logging.basicConfig(level=logging.INFO)

cfg = load_config()
root = resolve_path(cfg["data"]["root"])
src = Path(a.src)
src_meta = src / "metadata.csv" if (src / "metadata.csv").exists() else src.parent / "metadata.csv"
if (src / "LM").is_dir():
    src = src / "LM"
if not src.is_dir():
    raise SystemExit(f"Source folder not found: {src}")

added_species, added_images = set(), 0
for sp_dir in sorted(p for p in src.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))):
    dest = root / sp_dir.name
    dest.mkdir(parents=True, exist_ok=True)
    for f in sp_dir.iterdir():
        if f.is_file() and not (dest / f.name).exists():
            shutil.copy2(f, dest / f.name)
            added_images += 1
            added_species.add(sp_dir.name)
print(f"Copied {added_images} new images across {len(added_species)} species.")

meta_dst = resolve_path(cfg["data"]["metadata_csv"])
if src_meta.exists():
    new = pd.read_csv(src_meta, dtype=str)
    old = pd.read_csv(meta_dst, dtype=str) if meta_dst.exists() else new.iloc[0:0]
    merged = pd.concat([old, new]).drop_duplicates(subset=["file"], keep="first")
    merged.to_csv(meta_dst, index=False)

summary = build_dataset(cfg, keep_existing_splits=True)
print(json.dumps({k: summary[k] for k in ("n_images", "n_species", "n_genera", "images_per_split")}, indent=2))

if a.train == "finetune":
    prev = active_run_id(resolve_path(cfg["paths"]["runs_dir"]))
    if prev is None:
        raise SystemExit("No active model to fine-tune from. Use --train retrain.")
    cfg["train"].update(epochs_frozen=2, epochs_finetune=15, lr_backbone=5e-5)
    train(cfg, init_from=prev, note=f"fine-tuned from {prev} after adding {len(added_species)} species")
elif a.train == "retrain":
    train(cfg, note=f"retrained after adding {len(added_species)} species")
