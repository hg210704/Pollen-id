"""Data pipeline tests (no PyTorch needed)."""
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

from pollen.data import build_dataset, check_no_leakage, genus_of, pretty


def test_names():
    assert genus_of("Acer_campestre") == "Acer"
    assert pretty("Acer_campestre") == "Acer campestre"


def test_split_has_no_leakage_and_every_species_is_evaluable(synth_cfg):
    build_dataset(synth_cfg)
    df = pd.read_csv(Path(synth_cfg["paths"]["cache_dir"]) / "splits.csv", keep_default_na=False)
    assert check_no_leakage(df) == []
    assert (df.groupby("record")["split"].nunique() == 1).all()      # a specimen is never split
    per = df.groupby("species")["split"].agg(set)
    assert all({"train", "val", "test"} <= s for s in per)             # 5 records each -> all three splits


def test_identical_images_in_different_records_stay_together(synth_cfg):
    root = Path(synth_cfg["data"]["root"])
    sp = sorted(root.iterdir())[0]
    files = sorted(sp.glob("*.jpg"))
    shutil.copy(files[0], sp / "copy_of_first.jpg")  # same pixels, no metadata -> own record, but same image
    # exact duplicates are removed outright
    summary = build_dataset(synth_cfg)
    assert summary["validation"]["exact_duplicates_removed"] >= 1
    df = pd.read_csv(Path(synth_cfg["paths"]["cache_dir"]) / "splits.csv", keep_default_na=False)
    assert "copy_of_first.jpg" not in set(df["path"].map(lambda p: Path(p).name))


def test_corrupt_and_tiny_images_are_rejected(synth_cfg):
    root = Path(synth_cfg["data"]["root"])
    sp = sorted(root.iterdir())[0]
    (sp / "broken.jpg").write_bytes(b"not an image")
    Image.new("RGB", (8, 8)).save(sp / "tiny.png")
    summary = build_dataset(synth_cfg)
    assert summary["validation"]["corrupt"] == 1
    assert summary["validation"]["too_small"] == 1
    df = pd.read_csv(Path(synth_cfg["paths"]["cache_dir"]) / "splits.csv", keep_default_na=False)
    names = set(df["path"].map(lambda p: Path(p).name))
    assert "broken.jpg" not in names and "tiny.png" not in names


def test_existing_assignments_survive_new_data(synth_cfg):
    """Adding a new genus must not move any image between train/val/test."""
    build_dataset(synth_cfg)
    cache = Path(synth_cfg["paths"]["cache_dir"])
    before = pd.read_csv(cache / "splits.csv", keep_default_na=False).set_index("record")["split"].to_dict()
    root = Path(synth_cfg["data"]["root"])
    src = sorted(root.iterdir())[0]
    new = root / "Zzz_new"
    new.mkdir()
    for i, f in enumerate(sorted(src.glob("*.jpg"))[:6]):
        img = Image.open(f).rotate(37 * (i + 1)).transpose(Image.FLIP_LEFT_RIGHT)  # different pixels
        img.save(new / f"Zzz_new_LM_{i + 1}.jpg")
    build_dataset(synth_cfg)
    after = pd.read_csv(cache / "splits.csv", keep_default_na=False)
    assert "Zzz_new" in set(after["species"])
    for rec, split in before.items():
        assert after.set_index("record")["split"].to_dict()[rec] == split


def test_label_map_links_species_to_genus(synth_cfg):
    import json
    build_dataset(synth_cfg)
    lm = json.loads((Path(synth_cfg["paths"]["cache_dir"]) / "label_map.json").read_text())
    for sp, g in zip(lm["species"], lm["genus_of_species"]):
        assert lm["genera"][g] == genus_of(sp)
