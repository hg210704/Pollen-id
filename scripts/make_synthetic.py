r"""Create synthetic training images (variants of TRAIN photos only) and list them in a manifest.

    python scripts\make_synthetic.py             # top species up to data.synthetic_target_per_species
    python scripts\make_synthetic.py --clean     # delete old synthetic images and make them again
    python scripts\make_synthetic.py --target 20 # override the target for this run

Run it after  python scripts\prepare_data.py  and before training. Training picks the images up
automatically (data.use_synthetic: true in config.yaml). Validation and test images are never touched.
"""
import _bootstrap  # noqa: F401
import argparse
import json

from pollen.config import load_config
from pollen.synthetic import build_synthetic

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--target", type=int, default=None, help="training images per species to aim for")
ap.add_argument("--max-per-source", type=int, default=None, help="max variants of one photo")
ap.add_argument("--clean", action="store_true", help="delete existing synthetic images first")
a = ap.parse_args()

ov = {}
if a.target is not None:
    ov["data.synthetic_target_per_species"] = a.target
if a.max_per_source is not None:
    ov["data.synthetic_max_per_source"] = a.max_per_source
summary = build_synthetic(load_config(overrides=ov or None), clean=a.clean)
print(json.dumps(summary, indent=2))
print("\nNext: python -m pollen.train")
