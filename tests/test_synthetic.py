"""Synthetic-data tests (no PyTorch needed). The key promise: val/test stay real, and nothing leaks."""
from pathlib import Path

import pandas as pd
from PIL import Image

from pollen.data import build_dataset, load_splits, real_rows
from pollen.synthetic import build_synthetic, make_variant
import random


def _prepare(cfg, target=14):
    cfg["data"]["use_synthetic"] = True
    cfg["data"]["synthetic_target_per_species"] = target
    build_dataset(cfg)
    return build_synthetic(cfg)


def test_variants_are_valid_and_different(synth_cfg):
    build_dataset(synth_cfg)
    src = next(Path(synth_cfg["data"]["root"]).rglob("*.jpg"))
    im = Image.open(src).convert("RGB")
    v1, ops = make_variant(im, random.Random(1))
    v2, _ = make_variant(im, random.Random(2))
    assert v1.size == im.size and v1.mode == "RGB" and "rotate" in ops
    assert v1.tobytes() != im.tobytes() and v1.tobytes() != v2.tobytes()


def test_synthetic_only_enters_training_and_comes_from_train_photos(synth_cfg):
    summary = _prepare(synth_cfg)
    assert summary["created"] > 0
    df, _ = load_splits(synth_cfg)
    syn = df[df["synthetic"]]
    assert len(syn) == summary["total_synthetic"] and set(syn["split"]) == {"train"}
    assert not df[df["split"].isin(["val", "test"])]["synthetic"].any()          # val/test stay real
    real = real_rows(df)
    train_paths = set(real.loc[real["split"] == "train", "path"])
    manifest = pd.read_csv(Path(synth_cfg["paths"]["cache_dir"]) / "synthetic_manifest.csv", dtype=str)
    assert set(manifest["source"]) <= train_paths                                  # every source is a train photo


def test_every_species_is_topped_up_and_rerun_is_idempotent(synth_cfg):
    _prepare(synth_cfg, target=14)
    df, _ = load_splits(synth_cfg)
    per = df[df["split"] == "train"].groupby("species").size()
    assert (per >= 14).all()
    again = build_synthetic(synth_cfg)
    assert again["created"] == 0


def test_rebuilding_splits_cannot_leak_a_test_photo_into_training(synth_cfg):
    """If a source photo moves to the test split later, its synthetic variants must disappear."""
    _prepare(synth_cfg)
    cache = Path(synth_cfg["paths"]["cache_dir"])
    manifest = pd.read_csv(cache / "synthetic_manifest.csv", dtype=str)
    victim = manifest["source"].iloc[0]
    splits = pd.read_csv(cache / "splits.csv", keep_default_na=False)
    splits.loc[splits["path"] == victim, "split"] = "test"
    splits.to_csv(cache / "splits.csv", index=False)
    df, _ = load_splits(synth_cfg)
    kept_sources = set(manifest.loc[manifest["path"].isin(df[df["synthetic"]]["path"]), "source"])
    assert victim not in kept_sources
    assert len(df[df["synthetic"]]) < len(manifest)


def test_clean_regenerates(synth_cfg):
    _prepare(synth_cfg)
    out = build_synthetic(synth_cfg, clean=True)
    assert out["created"] == out["total_synthetic"] > 0
