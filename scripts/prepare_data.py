"""Validate images, remove corrupt/duplicates from training, and build leakage-free splits.

    python scripts/prepare_data.py
    python scripts/prepare_data.py --root D:/data/paldat_images/LM --metadata D:/data/paldat_images/metadata.csv
    python scripts/prepare_data.py --reset-splits      # forget earlier train/val/test assignment
    python scripts/prepare_data.py --move-rejected     # also move bad files into <root>/_rejected
"""
import _bootstrap  # noqa: F401
import argparse
import json
import logging

from pollen.config import load_config
from pollen.data import build_dataset

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--root", default=None, help="folder with <Genus_species>/ sub-folders (default: config data.root)")
ap.add_argument("--metadata", default=None, help="metadata.csv from the collection script")
ap.add_argument("--reset-splits", action="store_true")
ap.add_argument("--move-rejected", action="store_true")
a = ap.parse_args()
logging.basicConfig(level=logging.INFO)
cfg = load_config(overrides={"data.metadata_csv": a.metadata} if a.metadata else None)
summary = build_dataset(cfg, a.root, keep_existing_splits=not a.reset_splits, reject_to_folder=a.move_rejected)
print(json.dumps(summary, indent=2))
print("\nSplits written to", cfg["paths"]["cache_dir"], "- next: python scripts/train.py")
