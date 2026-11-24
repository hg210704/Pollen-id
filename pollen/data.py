"""Dataset handling: scan folders, validate images, split WITHOUT leakage (no torch needed).

Folder layout (written by the collection script):
    <root>/<Genus_species>/<Genus_species>_LM_<n>.jpg
    metadata.csv with columns: genus, species, caption, image_url, page_url, file

Leakage rule: several photos come from the same specimen record (same ``page_url``) and
look almost identical. All images of one record - plus any records that share an
identical image - are kept in the SAME split (train, val or test).
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

from .config import resolve_path

log = logging.getLogger("pollen.data")
ImageFile.LOAD_TRUNCATED_IMAGES = False

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- naming
def genus_of(species_key: str) -> str:
    """'Acer_campestre' -> 'Acer'."""
    return species_key.split("_")[0]


def pretty(species_key: str) -> str:
    """'Acer_campestre' -> 'Acer campestre'."""
    return species_key.replace("_", " ")


# --------------------------------------------------------------------------- scanning
def _load_metadata(metadata_csv: Path | None) -> dict[str, tuple[str, str]]:
    """Map '<species>/<filename>' -> (page_url, caption)."""
    if metadata_csv is None or not Path(metadata_csv).exists():
        return {}
    meta = pd.read_csv(metadata_csv, dtype=str).fillna("")
    out: dict[str, tuple[str, str]] = {}
    for _, row in meta.iterrows():
        parts = row.get("file", "").replace("\\", "/").split("/")
        if len(parts) >= 2:
            out[f"{parts[-2]}/{parts[-1]}"] = (row.get("page_url", ""), row.get("caption", ""))
    return out


def scan_dataset(root: Path, metadata_csv: Path | None = None) -> pd.DataFrame:
    """List every image with species, genus, specimen record and caption."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {root}\n"
            "Put your images in <root>/<Genus_species>/ or change data.root in config.yaml.")
    rows = []
    for sp_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))):
        for f in sorted(sp_dir.iterdir()):
            if f.suffix.lower() in IMG_EXTS:
                rows.append({"path": str(f), "species": sp_dir.name, "genus": genus_of(sp_dir.name),
                             "key": f"{sp_dir.name}/{f.name}"})
    if not rows:
        raise FileNotFoundError(f"No images found under {root}")
    df = pd.DataFrame(rows)
    meta = _load_metadata(metadata_csv)
    page = df["key"].map(lambda k: meta.get(k, ("", ""))[0])
    df["caption"] = df["key"].map(lambda k: meta.get(k, ("", ""))[1])
    df["page_url"] = page
    # Images without a known record are treated as their own record (flagged below).
    df["record"] = [f"{sp}|{url}" if url else f"{sp}|norecord:{k}"
                    for sp, url, k in zip(df.species, page, df.key)]
    df["has_record"] = page != ""
    return df


# --------------------------------------------------------------------------- validation
def _ahash(gray: Image.Image, side: int = 16) -> str:
    """256-bit average hash; identical for (near-)identical images."""
    small = np.asarray(gray.resize((side, side), Image.BILINEAR), dtype=np.float32)
    bits = (small > small.mean()).flatten()
    return hashlib.md5(np.packbits(bits).tobytes()).hexdigest()


def validate_images(df: pd.DataFrame, min_side: int = 32) -> tuple[pd.DataFrame, dict]:
    """Drop unreadable, tiny and exact-duplicate images. Returns (clean_df, report)."""
    corrupt, tiny, md5s, ahashes, widths = [], [], [], [], []
    for p in df["path"]:
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                w, h = im.size
                ah = _ahash(im.convert("L"))
            md5 = hashlib.md5(Path(p).read_bytes()).hexdigest()
        except Exception as e:  # corrupt / not an image
            corrupt.append({"path": p, "error": str(e)[:200]})
            md5s.append(None); ahashes.append(None); widths.append(0)
            continue
        if min(w, h) < min_side:
            tiny.append(p)
        md5s.append(md5); ahashes.append(ah); widths.append(min(w, h))
    df = df.copy()
    df["md5"], df["ahash"], df["min_side"] = md5s, ahashes, widths

    bad = df["md5"].isna() | (df["min_side"] < min_side)
    dup_mask = df[~bad].duplicated(subset=["species", "md5"], keep="first")
    duplicates = df[~bad][dup_mask]["path"].tolist()
    clean = df[~bad].drop(index=df[~bad][dup_mask].index).reset_index(drop=True)
    report = {
        "images_found": int(len(df)),
        "images_kept": int(len(clean)),
        "corrupt": corrupt,
        "too_small": tiny,
        "exact_duplicates_removed": duplicates,
        "images_without_record": int((~clean["has_record"]).sum()),
    }
    return clean, report


def move_rejected(report: dict, root: Path) -> None:
    """Optional: move rejected files to <root>/_rejected/ (never deletes anything)."""
    dest = Path(root) / "_rejected"
    dest.mkdir(exist_ok=True)
    rejected = [c["path"] for c in report["corrupt"]] + report["too_small"] + report["exact_duplicates_removed"]
    for p in rejected:
        src = Path(p)
        if src.exists():
            shutil.move(str(src), str(dest / f"{src.parent.name}__{src.name}"))


# --------------------------------------------------------------------------- grouping
def assign_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Group = specimen record, merged with any records that contain an identical image."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    first: dict[tuple[str, str], str] = {}
    for rec, sp, ah in zip(df["record"], df["species"], df["ahash"]):
        find(rec)
        k = (sp, ah)
        if k in first:
            union(rec, first[k])
        else:
            first[k] = rec
    df = df.copy()
    df["group"] = df["record"].map(find)
    return df


# --------------------------------------------------------------------------- splitting
def make_splits(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42,
                prior: dict[str, str] | None = None, image_level_single: bool = False) -> pd.DataFrame:
    """Assign every group to train/val/test per species (stratified by species).

    * ``prior`` maps record -> split from an earlier run, so existing test images stay in
      the test set when new data is added. Only NEW groups get assigned.
    * Species with 1 group  -> all train (cannot be evaluated).
    * Species with 2 groups -> train + test.
    * Species with >=3      -> at least one group each in val and test.
    """
    prior = prior or {}
    order = ("test", "val", "train")
    group_split: dict[str, str] = {}
    for species, sdf in df.groupby("species"):
        groups = sorted(sdf["group"].unique())
        existing: dict[str, str] = {}
        for g in groups:
            prev = [prior[r] for r in sdf.loc[sdf["group"] == g, "record"] if r in prior]
            if prev:  # conservative: if any member was held out, keep the group held out
                existing[g] = "test" if "test" in prev else "val" if "val" in prev else "train"
        new = [g for g in groups if g not in existing]
        random.Random(f"{seed}-{species}").shuffle(new)

        total = len(groups)
        n_test = 0 if total < 2 else max(1, round(test_frac * total))
        n_val = 0 if total < 3 else max(1, round(val_frac * total))
        while total - n_val - n_test < 1:
            if n_val > 0:
                n_val -= 1
            else:
                n_test -= 1
        target = {"test": n_test, "val": n_val, "train": total - n_val - n_test}
        counts = Counter(existing.values())
        for g in new:
            deficits = {s: target[s] - counts[s] for s in order}
            best = max(order, key=lambda s: (deficits[s], -order.index(s)))
            existing[g] = best
            counts[best] += 1
        group_split.update(existing)

    df = df.copy()
    df["split"] = df["group"].map(group_split)
    df["split_note"] = ""
    if image_level_single:  # optimistic, clearly flagged
        rng = np.random.default_rng(seed)
        for species, sdf in df.groupby("species"):
            if sdf["group"].nunique() == 1 and len(sdf) >= 3 and not prior:
                idx = rng.permutation(sdf.index.to_numpy())
                n_t = max(1, round(test_frac * len(idx)))
                n_v = max(1, round(val_frac * len(idx))) if len(idx) >= 5 else 0
                df.loc[idx[:n_t], "split"] = "test"
                df.loc[idx[n_t:n_t + n_v], "split"] = "val"
                df.loc[idx[n_t + n_v:], "split"] = "train"
                df.loc[idx, "split_note"] = "image-level split (leaky, optimistic)"
    return df


def check_no_leakage(df: pd.DataFrame) -> list[str]:
    """Return a list of groups that appear in more than one split (should be empty)."""
    leaks = df[df["split_note"] == ""].groupby("group")["split"].nunique()
    return leaks[leaks > 1].index.tolist()


# --------------------------------------------------------------------------- labels + orchestration
def build_label_map(df: pd.DataFrame) -> dict:
    species = sorted(df["species"].unique())
    genera = sorted({genus_of(s) for s in species})
    g_idx = {g: i for i, g in enumerate(genera)}
    return {"species": species, "genera": genera,
            "genus_of_species": [g_idx[genus_of(s)] for s in species]}


def build_dataset(cfg: dict, root: str | Path | None = None, keep_existing_splits: bool = True,
                  reject_to_folder: bool = False) -> dict:
    """Scan -> validate -> group -> split -> write splits.csv, label_map.json, summary."""
    dcfg = cfg["data"]
    root = resolve_path(root or dcfg["root"])
    meta_csv = resolve_path(dcfg["metadata_csv"]) if dcfg.get("metadata_csv") else None
    cache = resolve_path(cfg["paths"]["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)

    df = scan_dataset(root, meta_csv)
    df, report = validate_images(df, dcfg.get("min_image_side", 32))
    if reject_to_folder:
        move_rejected(report, root)
    df = assign_groups(df)

    prior = None
    splits_path = cache / "splits.csv"
    if keep_existing_splits and splits_path.exists():
        old = pd.read_csv(splits_path, dtype=str)
        prior = dict(zip(old["record"], old["split"]))
    df = make_splits(df, dcfg["val_fraction"], dcfg["test_fraction"], dcfg["seed"], prior,
                     dcfg.get("image_level_split_for_single_record", False))
    leaks = check_no_leakage(df)
    if leaks:
        raise RuntimeError(f"Leakage detected in {len(leaks)} groups - this is a bug.")

    label_map = build_label_map(df)
    df.drop(columns=["md5", "min_side"]).to_csv(splits_path, index=False)
    (cache / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    per_sp = df.groupby("species").agg(images=("path", "size"), records=("group", "nunique"))
    summary = {
        "root": str(root),
        "n_images": int(len(df)),
        "n_species": len(label_map["species"]),
        "n_genera": len(label_map["genera"]),
        "images_per_split": {s: int((df["split"] == s).sum()) for s in SPLITS},
        "validation": {k: (v if not isinstance(v, list) else len(v)) for k, v in report.items()},
        "species_with_few_images(<5)": per_sp[per_sp["images"] < 5].index.tolist(),
        "train_only_species(1 record, cannot be evaluated)": per_sp[per_sp["records"] < 2].index.tolist(),
        "no_validation_species(<3 records)": per_sp[per_sp["records"] < 3].index.tolist(),
        "median_images_per_species": float(per_sp["images"].median()),
    }
    (cache / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (cache / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def add_synthetic_rows(df: pd.DataFrame, cache: Path) -> pd.DataFrame:
    """Append synthetic images to the TRAIN split only.

    A synthetic image is used only if its source photo is (still) in the train split, so rebuilding
    the splits can never leak a test image's look-alikes into training."""
    manifest = cache / "synthetic_manifest.csv"
    df = df.copy()
    df["synthetic"] = False
    if not manifest.exists():
        return df
    m = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    train_sources = set(df.loc[df["split"] == "train", "path"])
    m = m[m["source"].isin(train_sources) & m["species"].isin(set(df["species"]))]
    m = m[m["path"].map(lambda p: Path(p).exists())]
    if m.empty:
        return df
    rec = m["species"] + "|synthetic:" + m["source"]
    extra = pd.DataFrame({
        "path": m["path"].values, "species": m["species"].values,
        "genus": m["species"].map(genus_of).values, "key": (m["species"] + "/" + m["path"].map(lambda p: Path(p).name)).values,
        "caption": "synthetic", "page_url": "", "record": rec.values, "has_record": True, "ahash": "",
        "group": rec.values, "split": "train", "split_note": "synthetic", "synthetic": True})
    return pd.concat([df, extra], ignore_index=True)


def real_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that are real photos (drops synthetic ones; works on a CSV that stored 'True'/'False')."""
    if "synthetic" not in df.columns:
        return df
    return df[~df["synthetic"].astype(str).str.lower().eq("true")]


def load_splits(cfg: dict) -> tuple[pd.DataFrame, dict]:
    cache = resolve_path(cfg["paths"]["cache_dir"])
    if not (cache / "splits.csv").exists():
        build_dataset(cfg)
    df = pd.read_csv(cache / "splits.csv", keep_default_na=False)
    label_map = json.loads((cache / "label_map.json").read_text(encoding="utf-8"))
    if cfg["data"].get("use_synthetic", False):
        df = add_synthetic_rows(df, cache)
    return df, label_map
