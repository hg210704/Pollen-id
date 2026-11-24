"""Identify one or more images with the ACTIVE model.

    python scripts/predict.py path/to/image.jpg [more.jpg ...] [--gradcam out_folder]
"""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

from PIL import Image

from pollen.predict import Predictor

ap = argparse.ArgumentParser()
ap.add_argument("images", nargs="+")
ap.add_argument("--gradcam", default=None, help="folder to save attention-map overlays")
ap.add_argument("--tta", type=int, default=4, help="test-time augmentation views: 1, 4 or 8")
a = ap.parse_args()

predictor = Predictor.from_active()
print(f"Model: {predictor.run_id}\n")
for path in a.images:
    r = predictor.predict(Image.open(path), tta_views=a.tta, gradcam=bool(a.gradcam))
    flag = "  (UNCERTAIN - treat as a lead only)" if r["uncertain"] else ""
    print(f"{path}\n  genus:   {r['genus']}  ({r['genus_confidence']:.1%})\n"
          f"  species: {r['species']}  ({r['species_confidence']:.1%}){flag}")
    for t in r["top5"]:
        print(f"    {t['probability']:6.1%}  {t['species']}")
    if a.gradcam:
        import base64
        out = Path(a.gradcam); out.mkdir(exist_ok=True)
        (out / f"{Path(path).stem}_gradcam.png").write_bytes(base64.b64decode(r["gradcam"].split(",")[1]))
