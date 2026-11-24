"""Training: stage 1 trains only the heads, stage 2 fine-tunes everything (cosine schedule).

Writes to runs/<run_id>/: config.json, splits.csv, label_map.json, history.json (per epoch),
status.json (live, polled by the dashboard), model.pt (best weights), metrics.json, report.md.
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import apply_overrides, load_config, parse_overrides, resolve_path
from .data import load_splits
from .datasets import PollenDataset, eval_transform, train_transform
from .evaluate import evaluate_run
from .model import HierarchicalNet, create_model, decode
from .registry import new_run_id, run_path, set_active, write_json_atomic

log = logging.getLogger("pollen.train")


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- mixing
def mix_batch(x, y_s, y_g, mixup_alpha: float, cutmix_alpha: float):
    """Optional MixUp / CutMix. Returns x, (ys_a, ys_b), (yg_a, yg_b), lam."""
    if (mixup_alpha <= 0 and cutmix_alpha <= 0) or random.random() < 0.5:
        return x, (y_s, y_s), (y_g, y_g), 1.0
    perm = torch.randperm(x.size(0), device=x.device)
    use_cut = cutmix_alpha > 0 and (mixup_alpha <= 0 or random.random() < 0.5)
    alpha = cutmix_alpha if use_cut else mixup_alpha
    lam = float(np.random.beta(alpha, alpha))
    if use_cut:
        h, w = x.shape[2:]
        cut = math.sqrt(1 - lam)
        ch, cw = int(h * cut), int(w * cut)
        cy, cx = random.randint(0, h - 1), random.randint(0, w - 1)
        y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, h)
        x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, w)
        x = x.clone()
        x[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
        lam = 1 - (y2 - y1) * (x2 - x1) / (h * w)
    else:
        x = lam * x + (1 - lam) * x[perm]
    return x, (y_s, y_s[perm]), (y_g, y_g[perm]), lam


# --------------------------------------------------------------------------- one epoch
def run_epoch(model, loader, device, crit_s, crit_g, tcfg, optimizer=None, scaler=None, scheduler=None,
              frozen_backbone: bool = False) -> dict:
    training = optimizer is not None
    model.train(training)
    if training and frozen_backbone:
        model.backbone.eval()  # keep BatchNorm statistics fixed while only the heads learn
    total_loss = n = c_s = c_g = 0
    use_amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    gw = tcfg.get("genus_loss_weight", 0.5)
    with torch.set_grad_enabled(training):
        for x, y_s, y_g in loader:
            x, y_s, y_g = x.to(device, non_blocking=True), y_s.to(device), y_g.to(device)
            if training:
                x, (sa, sb), (ga, gb), lam = mix_batch(x, y_s, y_g, tcfg.get("mixup_alpha", 0.0),
                                                       tcfg.get("cutmix_alpha", 0.0))
            with torch.autocast(device_type=device.type, enabled=use_amp):
                lg, ls = model(x)
                if training:
                    loss = lam * (crit_s(ls, sa) + gw * crit_g(lg, ga)) + \
                           (1 - lam) * (crit_s(ls, sb) + gw * crit_g(lg, gb))
                else:
                    loss = crit_s(ls, y_s) + gw * crit_g(lg, y_g)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if tcfg.get("grad_clip"):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
                scaler.step(optimizer); scaler.update()
                if scheduler is not None:
                    scheduler.step()
            g_dec, s_dec = decode(lg.detach().float(), ls.detach().float(), model.genus_of_species)
            b = x.size(0)
            total_loss += float(loss.detach()) * b; n += b
            c_s += int((s_dec == y_s).sum()); c_g += int((g_dec == y_g).sum())
    return {"loss": total_loss / max(n, 1), "species_acc": c_s / max(n, 1), "genus_acc": c_g / max(n, 1)}


# --------------------------------------------------------------------------- fine-tune support
def expand_from_checkpoint(model: HierarchicalNet, ckpt_path: Path, new_map: dict) -> int:
    """Initialise a model with MORE genera/species from an older checkpoint: backbone copied,
    head rows copied for every class that existed before. Returns number of classes reused."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    old_map, sd = ckpt["label_map"], ckpt["state_dict"]
    model.backbone.load_state_dict({k[9:]: v for k, v in sd.items() if k.startswith("backbone.")})
    reused = 0
    for head, key in (("species_head", "species"), ("genus_head", "genera")):
        old_idx = {n: i for i, n in enumerate(old_map[key])}
        w, b = sd[f"{head}.weight"], sd[f"{head}.bias"]
        with torch.no_grad():
            for new_i, name in enumerate(new_map[key]):
                if name in old_idx:
                    getattr(model, head).weight[new_i] = w[old_idx[name]]
                    getattr(model, head).bias[new_i] = b[old_idx[name]]
                    reused += key == "species"
    return reused


# --------------------------------------------------------------------------- main training routine
def train(cfg: dict, run_id: str | None = None, init_from: str | None = None, note: str = "") -> Path:
    tcfg, dcfg = cfg["train"], cfg["data"]
    runs_dir = resolve_path(cfg["paths"]["runs_dir"])
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id or new_run_id(cfg["model"]["backbone"])
    run_dir = run_path(runs_dir, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {**cfg, "note": note, "init_from": init_from}
    write_json_atomic(run_dir / "config.json", cfg)

    total_epochs = tcfg["epochs_frozen"] + tcfg["epochs_finetune"]
    status = {"run_id": run_id, "state": "running", "stage": "preparing", "epoch": 0,
              "total_epochs": total_epochs, "started_at": time.time(), "pid": os.getpid(),
              "elapsed_seconds": 0, "eta_seconds": None, "best_score": None, "lr": None, "error": None}
    write_json_atomic(run_dir / "status.json", status)
    history: list[dict] = []
    try:
        set_seed(tcfg["seed"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        df, label_map = load_splits(cfg)
        df.to_csv(run_dir / "splits.csv", index=False)
        write_json_atomic(run_dir / "label_map.json", label_map)
        size = dcfg["image_size"]
        tr_df, va_df = df[df["split"] == "train"], df[df["split"] == "val"]
        has_val = len(va_df) > 0
        if not has_val:
            log.warning("No validation images (no species has >=3 specimen records). "
                        "Early stopping will watch the training loss instead.")

        # data loaders
        sp_idx = {s: i for i, s in enumerate(label_map["species"])}
        counts = tr_df["species"].map(sp_idx).value_counts().reindex(range(len(label_map["species"])), fill_value=0)
        train_ds = PollenDataset(tr_df, label_map, train_transform(size))
        sampler = None
        if tcfg.get("balanced_sampler", True):
            w = 1.0 / counts.clip(lower=1).to_numpy()
            sampler = WeightedRandomSampler(torch.tensor(w[train_ds.species_idx], dtype=torch.double),
                                            num_samples=len(train_ds), replacement=True)
        nw = dcfg.get("num_workers", 0)
        train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], sampler=sampler, shuffle=sampler is None,
                                  num_workers=nw, drop_last=len(train_ds) > tcfg["batch_size"],
                                  pin_memory=device.type == "cuda", persistent_workers=nw > 0)
        val_loader = (DataLoader(PollenDataset(va_df, label_map, eval_transform(size)), batch_size=tcfg["batch_size"],
                                 shuffle=False, num_workers=nw) if has_val else None)

        # model + losses
        model = create_model(cfg["model"], label_map, size).to(device)
        if init_from:
            prev = run_path(runs_dir, init_from) / "model.pt"
            reused = expand_from_checkpoint(model, prev, label_map)
            log.info("Initialised from %s (%d species reused)", init_from, reused)
        weight = None
        if tcfg.get("class_weights") and not tcfg.get("balanced_sampler"):
            cw = (counts.sum() / (len(counts) * counts.clip(lower=1))).to_numpy()
            weight = torch.tensor(cw, dtype=torch.float32, device=device)
        crit_s = nn.CrossEntropyLoss(weight=weight, label_smoothing=tcfg["label_smoothing"])
        crit_g = nn.CrossEntropyLoss(label_smoothing=tcfg["label_smoothing"])
        scaler = torch.amp.GradScaler("cuda", enabled=bool(tcfg.get("amp", True)) and device.type == "cuda")

        best, best_epoch, bad, epoch = float("-inf"), 0, 0, 0
        metric = tcfg.get("early_stopping_metric", "val_loss")      # 'val_loss' or 'val_accuracy'
        min_delta = float(tcfg.get("early_stopping_min_delta", 1e-4))

        def make_score(tr: dict, va: dict | None) -> float:
            """Higher is better. Used to choose the best epoch AND to decide when to stop."""
            if va is None:
                return -tr["loss"]                                  # no validation images: watch training loss
            if metric == "val_loss":
                return -va["loss"]
            return (va["genus_acc"] + va["species_acc"]) / 2

        t0 = time.time()

        def save_checkpoint() -> None:
            torch.save({"state_dict": model.state_dict(), "label_map": label_map, "image_size": size,
                        "model_cfg": {k: cfg["model"][k] for k in ("backbone", "drop_rate")} | {"pretrained": False}},
                       run_dir / "model.pt")

        def log_epoch(stage: str, tr: dict, va: dict | None, lr: float) -> float:
            nonlocal epoch
            epoch += 1
            score = make_score(tr, va)
            rec = {"epoch": epoch, "stage": stage, "lr": lr,
                   "train_loss": tr["loss"], "train_species_acc": tr["species_acc"], "train_genus_acc": tr["genus_acc"],
                   "val_loss": va["loss"] if va else None, "val_species_acc": va["species_acc"] if va else None,
                   "val_genus_acc": va["genus_acc"] if va else None, "score": score}
            history.append(rec)
            write_json_atomic(run_dir / "history.json", history)
            elapsed = time.time() - t0
            status.update(stage=stage, epoch=epoch, elapsed_seconds=elapsed, lr=lr,
                          eta_seconds=elapsed / epoch * (total_epochs - epoch),
                          best_score=max(best, score))
            write_json_atomic(run_dir / "status.json", status)
            log.info("epoch %d/%d [%s] train_loss=%.3f val_species=%s", epoch, total_epochs, stage, tr["loss"],
                     f"{va['species_acc']:.3f}" if va else "n/a")
            return score

        # ---- stage 1: heads only
        for p in model.backbone.parameters():
            p.requires_grad = False
        opt = torch.optim.AdamW(model.heads(), lr=tcfg["lr_head"], weight_decay=tcfg["weight_decay"])
        for _ in range(tcfg["epochs_frozen"]):
            tr = run_epoch(model, train_loader, device, crit_s, crit_g, tcfg, opt, scaler, frozen_backbone=True)
            va = run_epoch(model, val_loader, device, crit_s, crit_g, tcfg) if has_val else None
            score = log_epoch("heads", tr, va, tcfg["lr_head"])
            if score > best:
                best, best_epoch = score, epoch
                save_checkpoint()

        # ---- stage 2: whole network, warm-up then cosine decay
        for p in model.backbone.parameters():
            p.requires_grad = True
        opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": tcfg["lr_backbone"]},
                                 {"params": model.heads(), "lr": tcfg["lr_head"]}], weight_decay=tcfg["weight_decay"])
        steps_per_epoch = max(len(train_loader), 1)
        total_steps = max(tcfg["epochs_finetune"] * steps_per_epoch, 1)
        warm = tcfg.get("warmup_epochs", 1) * steps_per_epoch

        def lr_lambda(step: int) -> float:
            if step < warm:
                return (step + 1) / max(warm, 1)
            prog = (step - warm) / max(total_steps - warm, 1)
            return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        for _ in range(tcfg["epochs_finetune"]):
            tr = run_epoch(model, train_loader, device, crit_s, crit_g, tcfg, opt, scaler, sched)
            va = run_epoch(model, val_loader, device, crit_s, crit_g, tcfg) if has_val else None
            score = log_epoch("finetune", tr, va, opt.param_groups[0]["lr"])
            if score > best + min_delta:
                best, best_epoch, bad = score, epoch, 0
                save_checkpoint()
            else:
                bad += 1
                if bad >= tcfg["early_stopping_patience"]:
                    log.info("Early stopping at epoch %d: no improvement in %s for %d epochs; keeping the weights from epoch %d", epoch, metric, bad, best_epoch)
                    break

        status.update(stage="evaluating")
        write_json_atomic(run_dir / "status.json", status)
        metrics = evaluate_run(run_dir, cfg)
        status.update(state="finished", stage="done", best_epoch=best_epoch,
                      elapsed_seconds=time.time() - t0, eta_seconds=0)
        write_json_atomic(run_dir / "status.json", status)
        if not (runs_dir / "active.json").exists():
            set_active(runs_dir, run_id)
        log.info("Finished %s: test genus %.3f, species %.3f", run_id,
                 (metrics["test"] or {}).get("genus_top1") or float("nan"),
                 (metrics["test"] or {}).get("species_top1") or float("nan"))
    except Exception as e:
        status.update(state="failed", error=f"{type(e).__name__}: {e}")
        write_json_atomic(run_dir / "status.json", status)
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    return run_dir


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Train the pollen genus/species model.")
    ap.add_argument("--config", default=None, help="path to a config YAML (default: config.yaml)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--backbone", default=None, help="shortcut for --set model.backbone=...")
    ap.add_argument("--init-from", default=None, help="run id to start from (fine-tune after adding classes)")
    ap.add_argument("--note", default="", help="free-text note shown on the dashboard")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                    help="override any config value, e.g. train.epochs_finetune=20 (repeatable)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ov = parse_overrides(a.overrides)
    if a.backbone:
        ov["model.backbone"] = a.backbone
    cfg = load_config(a.config, ov)
    train(cfg, a.run_id, a.init_from, a.note)


if __name__ == "__main__":
    main()
