"""Honest evaluation: top-1/top-5, per-class P/R/F1, confusion, calibration, abstention."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

from .config import load_config, resolve_path
from .data import pretty, real_rows
from .datasets import PollenDataset, eval_transform
from .model import create_model, decode

log = logging.getLogger("pollen.evaluate")


# --------------------------------------------------------------------------- inference helpers
def tta_views(n: int) -> list[tuple[int, bool]]:
    """(rotation in 90-degree steps, flip). Pollen has no 'up', so rotations are safe."""
    allv = [(k, f) for f in (False, True) for k in range(4)]
    return [(0, False)] if n <= 1 else (allv[:4] if n <= 4 else allv)


def tta_forward(model, x: torch.Tensor, n_views: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Average logits over test-time-augmentation views."""
    g_sum = s_sum = 0
    views = tta_views(n_views)
    for k, flip in views:
        xv = torch.rot90(x, k, dims=(2, 3))
        if flip:
            xv = torch.flip(xv, dims=(3,))
        g, s = model(xv)
        g_sum, s_sum = g_sum + g, s_sum + s
    return g_sum / len(views), s_sum / len(views)


@torch.no_grad()
def collect_logits(model, loader: DataLoader, device: torch.device, n_views: int = 1):
    model.eval()
    lg, ls, ys, yg = [], [], [], []
    for x, y_s, y_g in loader:
        g, s = tta_forward(model, x.to(device), n_views)
        lg.append(g.float().cpu()); ls.append(s.float().cpu())
        ys.append(y_s); yg.append(y_g)
    return torch.cat(lg), torch.cat(ls), torch.cat(ys), torch.cat(yg)


# --------------------------------------------------------------------------- calibration
def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Single temperature T minimising NLL on the validation set."""
    if len(y) == 0:
        return 1.0
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=60)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().clamp(0.5, 5.0))


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def tune_threshold(conf: np.ndarray, correct: np.ndarray, target_acc: float, min_cov: float,
                   default: float) -> tuple[float, list[dict]]:
    """Smallest threshold whose ANSWERED predictions reach target_acc while still answering
    at least min_cov of the images; otherwise the default."""
    curve, chosen = [], None
    for t in np.linspace(0, 0.99, 100):
        acc_mask = conf >= t
        cov = float(acc_mask.mean()) if len(conf) else 0.0
        acc = float(correct[acc_mask].mean()) if acc_mask.any() else float("nan")
        curve.append({"threshold": round(float(t), 3), "coverage": cov, "accuracy": acc})
        if chosen is None and acc_mask.any() and acc >= target_acc and cov >= min_cov:
            chosen = float(t)
    return (chosen if chosen is not None else default), curve[::5]


# --------------------------------------------------------------------------- metrics
def compute_metrics(lg: torch.Tensor, ls: torch.Tensor, y_g: torch.Tensor, y_s: torch.Tensor,
                    label_map: dict, temperature_s: float = 1.0, threshold: float | None = None,
                    train_counts: dict[str, int] | None = None) -> dict:
    """Everything the dashboard and report need for one split."""
    n = len(y_s)
    if n == 0:
        return {"n_images": 0}
    species, genera = label_map["species"], label_map["genera"]
    gos = torch.tensor(label_map["genus_of_species"])
    g_dec, s_dec = decode(lg, ls, gos)
    s_raw = ls.argmax(1)

    top5 = ls.topk(min(5, ls.shape[1]), dim=1).indices
    probs = F.softmax(ls / temperature_s, dim=1)
    conf = probs[torch.arange(n), s_dec].numpy()
    correct = (s_dec == y_s).numpy()

    out: dict = {
        "n_images": int(n),
        "genus_top1": float((g_dec == y_g).float().mean()),
        "species_top1": float(correct.mean()),
        "species_top1_unconstrained": float((s_raw == y_s).float().mean()),
        "species_top5": float((top5 == y_s[:, None]).any(1).float().mean()),
        "ece_species": expected_calibration_error(conf, correct),
    }

    # per-class precision / recall / F1 (species present in this split)
    labels = sorted(set(y_s.tolist()) | set(s_dec.tolist()))
    p, r, f, sup = precision_recall_fscore_support(y_s.numpy(), s_dec.numpy(), labels=labels,
                                                   zero_division=0)
    per_class = [{"species": species[c], "name": pretty(species[c]), "genus": genera[label_map["genus_of_species"][c]],
                  "precision": float(pp), "recall": float(rr), "f1": float(ff), "support": int(ss),
                  "n_train": (train_counts or {}).get(species[c], 0)}
                 for c, pp, rr, ff, ss in zip(labels, p, r, f, sup) if ss > 0]
    out["macro_f1_species"] = float(np.mean([c["f1"] for c in per_class])) if per_class else float("nan")
    out["per_class"] = sorted(per_class, key=lambda c: (c["recall"], c["f1"]))  # weakest first

    # confusion: genus (full matrix) + species (sparse: only non-zero cells)
    ng = len(genera)
    gm = np.zeros((ng, ng), dtype=int)
    np.add.at(gm, (y_g.numpy(), g_dec.numpy()), 1)
    out["confusion_genus"] = {"labels": genera, "matrix": gm.tolist()}
    cells: dict[tuple[int, int], int] = {}
    for t, pr in zip(y_s.tolist(), s_dec.tolist()):
        cells[(t, pr)] = cells.get((t, pr), 0) + 1
    out["confusion_species"] = [
        {"true": species[t], "pred": species[pr], "count": c,
         "true_genus": genera[label_map["genus_of_species"][t]],
         "pred_genus": genera[label_map["genus_of_species"][pr]]}
        for (t, pr), c in sorted(cells.items(), key=lambda kv: -kv[1])]
    errors = [c for c in out["confusion_species"] if c["true"] != c["pred"]]
    out["most_confused_pairs"] = errors[:25]
    n_err = sum(c["count"] for c in errors)
    same_genus = sum(c["count"] for c in errors if c["true_genus"] == c["pred_genus"])
    out["errors_within_correct_genus_share"] = (same_genus / n_err) if n_err else None

    if threshold is not None:
        ans = conf >= threshold
        out["at_threshold"] = {
            "threshold": threshold,
            "coverage": float(ans.mean()),
            "accuracy_on_answered": float(correct[ans].mean()) if ans.any() else None,
            "abstained": int((~ans).sum()),
        }
    return out


def headline(m: dict) -> dict:
    keys = ("n_images", "genus_top1", "species_top1", "species_top5", "macro_f1_species", "ece_species")
    return {k: m.get(k) for k in keys}


# --------------------------------------------------------------------------- run-level evaluation
def load_run_model(run_dir: Path, device: torch.device):
    ckpt = torch.load(Path(run_dir) / "model.pt", map_location="cpu", weights_only=True)
    model = create_model(ckpt["model_cfg"], ckpt["label_map"], ckpt["image_size"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def make_loader(df: pd.DataFrame, label_map: dict, size: int, cfg: dict, batch: int | None = None) -> DataLoader:
    ds = PollenDataset(df, label_map, eval_transform(size))
    return DataLoader(ds, batch_size=batch or cfg["train"]["batch_size"], shuffle=False,
                      num_workers=cfg["data"].get("num_workers", 0))


def evaluate_run(run_dir: str | Path, cfg: dict | None = None, gradcam_samples: int | None = None) -> dict:
    """Evaluate a finished run on val + test, fit temperature, tune threshold, write artifacts."""
    run_dir = Path(run_dir)
    cfg = cfg or json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_run_model(run_dir, device)
    label_map, size = ckpt["label_map"], ckpt["image_size"]
    splits = real_rows(pd.read_csv(run_dir / "splits.csv", keep_default_na=False))  # synthetic images never count
    n_views = cfg["eval"].get("tta_views", 1)
    train_counts = splits[splits["split"] == "train"]["species"].value_counts().to_dict()

    # -- validation: temperature + abstention threshold
    val_df = splits[splits["split"] == "val"]
    temp_s, temp_g, threshold, curve = 1.0, 1.0, cfg["eval"]["default_threshold"], []
    results: dict = {}
    if len(val_df):
        lg, ls, y_s, y_g = collect_logits(model, make_loader(val_df, label_map, size, cfg), device, n_views)
        temp_s, temp_g = fit_temperature(ls, y_s), fit_temperature(lg, y_g)
        _, s_dec = decode(lg, ls, torch.tensor(label_map["genus_of_species"]))
        conf = F.softmax(ls / temp_s, 1)[torch.arange(len(y_s)), s_dec].numpy()
        threshold, curve = tune_threshold(conf, (s_dec == y_s).numpy(), cfg["eval"]["selective_accuracy"],
                                          cfg["eval"]["min_coverage"], cfg["eval"]["default_threshold"])
        results["val"] = compute_metrics(lg, ls, y_g, y_s, label_map, temp_s, threshold, train_counts)

    # -- test: the number that counts
    test_df = splits[splits["split"] == "test"]
    if len(test_df):
        lg, ls, y_s, y_g = collect_logits(model, make_loader(test_df, label_map, size, cfg), device, n_views)
        results["test"] = compute_metrics(lg, ls, y_g, y_s, label_map, temp_s, threshold, train_counts)
        if gradcam_samples is None:
            gradcam_samples = cfg["eval"].get("gradcam_samples", 0)
        if gradcam_samples:
            save_gradcam_samples(model, test_df, label_map, size, device, run_dir / "gradcam",
                                 gradcam_samples, lg, ls, y_s)

    t = results.get("test", {})
    tgt_g, tgt_s = cfg["eval"]["target_genus_acc"], cfg["eval"]["target_species_acc"]
    metrics = {
        "temperature_species": temp_s, "temperature_genus": temp_g, "threshold": threshold,
        "selective_curve": curve, "tta_views": n_views,
        "val": headline(results.get("val", {})), "test": headline(t),
        "targets": {"genus_top1": tgt_g, "species_top1": tgt_s},
        "target_met": bool(t and t["genus_top1"] >= tgt_g and t["species_top1"] >= tgt_s),
        "at_threshold_test": t.get("at_threshold"),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "evaluation.json").write_text(json.dumps(results), encoding="utf-8")
    (run_dir / "report.md").write_text(build_report(metrics, results, splits, label_map, train_counts),
                                       encoding="utf-8")
    return metrics


def build_report(metrics: dict, results: dict, splits: pd.DataFrame, label_map: dict,
                 train_counts: dict[str, int]) -> str:
    """Plain-language report. States clearly whether the target was met and, if not, why."""
    t = results.get("test")
    lines = ["# Run report", ""]
    if not t:
        lines += ["**No test set.** No species has at least 2 specimen records, so nothing could be held out. "
                  "Collect more records per species (each record = one PalDat page) before judging accuracy.", ""]
        return "\n".join(lines)
    tg, ts = metrics["targets"]["genus_top1"], metrics["targets"]["species_top1"]
    lines += [
        f"- Test images: {t['n_images']}",
        f"- Genus top-1: **{t['genus_top1']:.1%}** (target {tg:.0%})",
        f"- Species top-1: **{t['species_top1']:.1%}** (target {ts:.0%}), top-5: {t['species_top5']:.1%}",
        f"- Macro F1 (species): {t['macro_f1_species']:.3f}, calibration error (ECE): {t['ece_species']:.3f}",
    ]
    at = t.get("at_threshold")
    if at and at["accuracy_on_answered"] is not None:
        lines.append(f"- With the abstention threshold ({at['threshold']:.2f}) the model answers {at['coverage']:.0%} "
                     f"of images with {at['accuracy_on_answered']:.1%} accuracy; the rest are marked uncertain.")
    lines.append("")
    lines.append("## Target met" if metrics["target_met"] else "## Target NOT met - why")
    if not metrics["target_met"]:
        per_sp = splits.groupby("species").agg(images=("path", "size"), records=("group", "nunique"))
        few_rec = per_sp[per_sp["records"] < 3]
        weak = [c for c in t["per_class"] if c["recall"] < 0.5][:10]
        lines += [
            f"- Species with fewer than 3 specimen records (no validation / weak evidence): "
            f"{len(few_rec)} of {len(per_sp)}.",
            f"- Median images per species: {per_sp['images'].median():.0f}; median specimen records: "
            f"{per_sp['records'].median():.0f}.",
        ]
        share = t.get("errors_within_correct_genus_share")
        if share is not None:
            lines.append(f"- {share:.0%} of species errors stay inside the correct genus (look-alike congeners).")
        if weak:
            lines.append("- Weakest species (recall < 50%): " +
                         ", ".join(f"{c['name']} ({c['recall']:.0%}, {c['n_train']} train imgs)" for c in weak))
        lines += ["", "What usually helps, in order: more specimen records per weak species, adding SEM or "
                       "additional LM views, higher image size, a stronger backbone. Report these numbers as they are."]
    return "\n".join(lines) + "\n"


def save_gradcam_samples(model, test_df: pd.DataFrame, label_map: dict, size: int, device, out_dir: Path,
                         n: int, lg: torch.Tensor, ls: torch.Tensor, y_s: torch.Tensor) -> None:
    """Save Grad-CAM overlays for the first n MISCLASSIFIED test images (most informative)."""
    from PIL import Image

    from .datasets import ResizePad
    from .gradcam import GradCAM, overlay

    out_dir.mkdir(exist_ok=True)
    _, s_dec = decode(lg, ls, torch.tensor(label_map["genus_of_species"]))
    wrong = (s_dec != y_s).nonzero().flatten().tolist()[:n]
    cam, tf = GradCAM(model), eval_transform(size)
    for i in wrong:
        img = ResizePad(size)(Image.open(test_df.iloc[i]["path"]).convert("RGB"))
        x = tf(Image.open(test_df.iloc[i]["path"]).convert("RGB")).unsqueeze(0).to(device)
        heat, _ = cam(x, int(s_dec[i]))
        name = f"{i}_true-{label_map['species'][int(y_s[i])]}_pred-{label_map['species'][int(s_dec[i])]}.png"
        overlay(img, heat).save(out_dir / name)
    cam.close()


def main() -> None:
    import argparse

    from .registry import latest_run_id

    ap = argparse.ArgumentParser(description="Re-evaluate a finished run on its val/test splits.")
    ap.add_argument("--run", default="latest", help="run id or 'latest'")
    ap.add_argument("--gradcam", type=int, default=None, help="number of Grad-CAM samples to save")
    a = ap.parse_args()
    cfg = load_config()
    runs_dir = resolve_path(cfg["paths"]["runs_dir"])
    run_id = latest_run_id(runs_dir) if a.run == "latest" else a.run
    metrics = evaluate_run(runs_dir / run_id, gradcam_samples=a.gradcam)
    print(json.dumps({k: metrics[k] for k in ("val", "test", "target_met", "threshold")}, indent=2))
    print(f"Report: {runs_dir / run_id / 'report.md'}")


if __name__ == "__main__":
    main()
