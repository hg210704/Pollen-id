"""Export a run's model for use outside this project (TorchScript always, ONNX if installed)."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import load_config, resolve_path
from .evaluate import load_run_model
from .registry import active_run_id, run_path


def export_run(run_dir: Path, fmt: str = "torchscript") -> Path:
    model, ckpt = load_run_model(run_dir, torch.device("cpu"))
    size = ckpt["image_size"]
    example = torch.randn(1, 3, size, size)
    if fmt == "torchscript":
        out = run_dir / "model.torchscript.pt"
        torch.jit.trace(model, example, check_trace=False).save(str(out))
        return out
    if fmt == "onnx":
        out = run_dir / "model.onnx"
        kwargs = dict(input_names=["image"], output_names=["genus_logits", "species_logits"],
                      dynamic_axes={"image": {0: "batch"}, "genus_logits": {0: "batch"},
                                    "species_logits": {0: "batch"}}, opset_version=17)
        try:
            torch.onnx.export(model, example, str(out), dynamo=False, **kwargs)
        except TypeError:  # older torch has no 'dynamo' argument
            torch.onnx.export(model, example, str(out), **kwargs)
        except ModuleNotFoundError as e:
            raise SystemExit(f"ONNX export needs extra packages: pip install onnx onnxscript  ({e})")
        return out
    raise ValueError("format must be 'torchscript' or 'onnx'")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a trained run.")
    ap.add_argument("--run", default="active", help="run id or 'active'")
    ap.add_argument("--format", choices=["torchscript", "onnx"], default="torchscript")
    a = ap.parse_args()
    runs_dir = resolve_path(load_config()["paths"]["runs_dir"])
    rid = active_run_id(runs_dir) if a.run == "active" else a.run
    if rid is None:
        raise SystemExit("No trained model yet.")
    print("Saved:", export_run(run_path(runs_dir, rid), a.format))


if __name__ == "__main__":
    main()
