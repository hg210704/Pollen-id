import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from make_synthetic_dataset import draw_grain  # noqa: E402,F401
import make_synthetic_dataset as synth  # noqa: E402

from pollen.config import load_config  # noqa: E402


def build_synthetic(base: Path, **kw):
    """Write a small fake dataset and return the paths."""
    import subprocess
    args = [sys.executable, str(ROOT / "tools" / "make_synthetic_dataset.py"), "--out", str(base)]
    for k, v in kw.items():
        args += [f"--{k.replace('_', '-')}", str(v)]
    subprocess.run(args, check=True, capture_output=True)
    return base / "paldat_images" / "LM", base / "paldat_images" / "metadata.csv"


@pytest.fixture()
def synth_cfg(tmp_path):
    root, meta = build_synthetic(tmp_path / "data")
    cfg = load_config(overrides={
        "paths.cache_dir": str(tmp_path / "cache"), "paths.runs_dir": str(tmp_path / "runs"),
        "data.root": str(root), "data.metadata_csv": str(meta), "data.image_size": 64, "data.num_workers": 0,
        "model.backbone": "resnet18", "model.pretrained": False,
        "train.epochs_frozen": 1, "train.epochs_finetune": 4, "train.batch_size": 8, "train.amp": False,
        "train.lr_head": 3e-3, "train.lr_backbone": 1e-3, "eval.tta_views": 1, "eval.gradcam_samples": 2,
    })
    return cfg
