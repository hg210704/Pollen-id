"""Configuration loading with command-line overrides ("train.epochs_finetune=5")."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.yaml"


def resolve_path(p: str | Path) -> Path:
    """Relative paths are relative to the project folder; absolute paths are kept."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def parse_overrides(items: Iterable[str]) -> dict[str, Any]:
    """['train.epochs_finetune=5', 'model.pretrained=false'] -> {'train.epochs_finetune': 5, ...}"""
    out: dict[str, Any] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Override '{item}' must look like section.key=value")
        out[key.strip()] = yaml.safe_load(value)  # understands ints, floats, true/false, text
    return out


def apply_overrides(cfg: dict, overrides: dict[str, Any]) -> dict:
    cfg = copy.deepcopy(cfg)
    for dotted, value in overrides.items():
        node = cfg
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return cfg


def load_config(path: str | Path | None = None,
                overrides: dict[str, Any] | Iterable[str] | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        if not isinstance(overrides, dict):
            overrides = parse_overrides(overrides)
        cfg = apply_overrides(cfg, overrides)
    return cfg


def flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out
