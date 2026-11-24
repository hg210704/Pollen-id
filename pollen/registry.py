"""Runs live in runs/<run_id>/. This module lists them, tracks the ACTIVE model and diffs runs."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from .config import flatten

RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,120}$")


def new_run_id(backbone: str) -> str:
    return f"run_{time.strftime('%Y%m%d_%H%M%S')}_{re.sub(r'[^A-Za-z0-9_]', '', backbone)}"


def run_path(runs_dir: Path, run_id: str) -> Path:
    """Validated path (blocks '../' tricks)."""
    if not RUN_ID_RE.match(run_id) or run_id in (".", ".."):
        raise ValueError(f"Invalid run id: {run_id!r}")
    return Path(runs_dir) / run_id


def _read(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json_atomic(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def _run_dirs(runs_dir: Path) -> list[Path]:
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    return sorted((p for p in runs_dir.iterdir() if p.is_dir() and RUN_ID_RE.match(p.name)), reverse=True)


def active_run_id(runs_dir: Path) -> str | None:
    """Explicitly activated run, else the newest finished run that has a model."""
    chosen = _read(Path(runs_dir) / "active.json", {}).get("run_id")
    if chosen and (Path(runs_dir) / chosen / "model.pt").exists():
        return chosen
    for p in _run_dirs(runs_dir):
        if (p / "model.pt").exists() and _read(p / "status.json", {}).get("state") == "finished":
            return p.name
    return None


def set_active(runs_dir: Path, run_id: str) -> None:
    if not (run_path(runs_dir, run_id) / "model.pt").exists():
        raise FileNotFoundError(f"Run {run_id} has no trained model")
    write_json_atomic(Path(runs_dir) / "active.json", {"run_id": run_id, "set_at": time.time()})


def latest_run_id(runs_dir: Path) -> str:
    dirs = [p for p in _run_dirs(runs_dir) if (p / "model.pt").exists()]
    if not dirs:
        raise FileNotFoundError("No trained runs yet. Run: python scripts/train.py")
    return dirs[0].name


def summarize(run_dir: Path, active: str | None = None) -> dict:
    cfg = _read(run_dir / "config.json", {})
    status = _read(run_dir / "status.json", {})
    metrics = _read(run_dir / "metrics.json", {})
    test, val = metrics.get("test") or {}, metrics.get("val") or {}
    return {
        "id": run_dir.name,
        "state": status.get("state", "unknown"),
        "created": status.get("started_at"),
        "backbone": cfg.get("model", {}).get("backbone"),
        "image_size": cfg.get("data", {}).get("image_size"),
        "epochs": (cfg.get("train", {}).get("epochs_frozen", 0) + cfg.get("train", {}).get("epochs_finetune", 0)),
        "test_genus_top1": test.get("genus_top1"),
        "test_species_top1": test.get("species_top1"),
        "test_species_top5": test.get("species_top5"),
        "test_macro_f1": test.get("macro_f1_species"),
        "val_genus_top1": val.get("genus_top1"),
        "val_species_top1": val.get("species_top1"),
        "target_met": metrics.get("target_met"),
        "has_model": (run_dir / "model.pt").exists(),
        "active": run_dir.name == active,
        "note": cfg.get("note", ""),
    }


def list_runs(runs_dir: Path) -> list[dict]:
    active = active_run_id(runs_dir)
    return [summarize(p, active) for p in _run_dirs(runs_dir)]


def get_run(runs_dir: Path, run_id: str) -> dict:
    d = run_path(runs_dir, run_id)
    if not d.is_dir():
        raise FileNotFoundError(f"Run {run_id} not found")
    report = d / "report.md"
    return {
        "summary": summarize(d, active_run_id(runs_dir)),
        "status": _read(d / "status.json", {}),
        "config": _read(d / "config.json", {}),
        "history": _read(d / "history.json", []),
        "metrics": _read(d / "metrics.json", {}),
        "evaluation": _read(d / "evaluation.json", {}),
        "report": report.read_text(encoding="utf-8") if report.exists() else "",
    }


def diff_runs(runs_dir: Path, a: str, b: str) -> dict:
    """What changed between run A and run B: config differences + metric deltas."""
    ca = flatten(_read(run_path(runs_dir, a) / "config.json", {}))
    cb = flatten(_read(run_path(runs_dir, b) / "config.json", {}))
    changes = [{"key": k, "a": ca.get(k), "b": cb.get(k)}
               for k in sorted(set(ca) | set(cb)) if ca.get(k) != cb.get(k) and k != "note"]
    ma = _read(run_path(runs_dir, a) / "metrics.json", {})
    mb = _read(run_path(runs_dir, b) / "metrics.json", {})
    deltas = []
    for split in ("test", "val"):
        for metric in ("genus_top1", "species_top1", "species_top5", "macro_f1_species", "ece_species"):
            va, vb = (ma.get(split) or {}).get(metric), (mb.get(split) or {}).get(metric)
            deltas.append({"metric": f"{split}.{metric}", "a": va, "b": vb,
                           "delta": (vb - va) if isinstance(va, (int, float)) and isinstance(vb, (int, float)) else None})
    return {"a": a, "b": b, "config_changes": changes, "metric_delta": deltas}
