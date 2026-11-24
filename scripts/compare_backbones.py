"""Train the candidate backbones with identical settings, pick the best on VALIDATION results.

    python scripts/compare_backbones.py                    # full training for each (slow)
    python scripts/compare_backbones.py --quick            # 10 fine-tune epochs each
    python scripts/compare_backbones.py --backbones convnext_tiny tf_efficientnetv2_s

The winner is chosen by validation score (mean of genus and species top-1), never by test results,
and is set as the active model. A short write-up is saved to runs/backbone_comparison.md.
"""
import _bootstrap  # noqa: F401
import argparse
import json
import logging

from pollen.config import load_config, resolve_path
from pollen.registry import set_active
from pollen.train import train

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--backbones", nargs="+", default=["convnext_tiny", "tf_efficientnetv2_s", "deit_small_patch16_224"])
ap.add_argument("--quick", action="store_true")
a = ap.parse_args()
logging.basicConfig(level=logging.INFO)

rows = []
for name in a.backbones:
    cfg = load_config(overrides={"model.backbone": name, **({"train.epochs_finetune": 10} if a.quick else {})})
    try:
        run_dir = train(cfg, note=f"backbone comparison: {name}")
    except Exception as e:  # keep going if one backbone fails (e.g. out of memory)
        rows.append({"backbone": name, "error": str(e)})
        continue
    m = json.loads((run_dir / "metrics.json").read_text())
    v = m["val"] or {}
    score = ((v.get("genus_top1") or 0) + (v.get("species_top1") or 0)) / 2
    rows.append({"backbone": name, "run": run_dir.name, "val_score": score,
                 "val_genus": v.get("genus_top1"), "val_species": v.get("species_top1"),
                 "test_species": (m["test"] or {}).get("species_top1")})

ok = [r for r in rows if "run" in r]
runs_dir = resolve_path(load_config()["paths"]["runs_dir"])
lines = ["# Backbone comparison", "", "| backbone | val genus | val species | val score |", "|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['backbone']} | " + (f"{r['val_genus']:.3f} | {r['val_species']:.3f} | {r['val_score']:.3f} |"
                                           if "run" in r else f"failed: {r['error'][:60]} | | |"))
if ok:
    best = max(ok, key=lambda r: r["val_score"])
    set_active(runs_dir, best["run"])
    lines += ["", f"**Chosen: {best['backbone']}** - highest mean of validation genus and species accuracy "
                  f"({best['val_score']:.3f}). Chosen on validation data only; test numbers stay untouched."]
(runs_dir / "backbone_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
