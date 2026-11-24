"""Inference: image in -> genus, species, top-5, confidence, 'uncertain' flag, optional Grad-CAM."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from .config import load_config, resolve_path
from .data import pretty
from .datasets import ResizePad, eval_transform
from .evaluate import load_run_model, tta_forward
from .gradcam import GradCAM, overlay
from .model import decode
from .registry import active_run_id


def _png_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class Predictor:
    def __init__(self, run_dir: str | Path, device: str | None = None):
        self.run_dir = Path(run_dir)
        self.run_id = self.run_dir.name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, ckpt = load_run_model(self.run_dir, self.device)
        self.label_map, self.size = ckpt["label_map"], ckpt["image_size"]
        metrics_path = self.run_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        self.temp_s = metrics.get("temperature_species", 1.0)
        self.temp_g = metrics.get("temperature_genus", 1.0)
        self.threshold = metrics.get("threshold", 0.5)
        self.tf = eval_transform(self.size)
        self.genus_of_species = torch.tensor(self.label_map["genus_of_species"])

    @classmethod
    def from_active(cls) -> "Predictor":
        runs_dir = resolve_path(load_config()["paths"]["runs_dir"])
        rid = active_run_id(runs_dir)
        if rid is None:
            raise FileNotFoundError("No trained model yet. Train one with: python scripts/train.py")
        return cls(runs_dir / rid)

    def predict(self, image: Image.Image, tta_views: int = 4, gradcam: bool = False, topk: int = 5) -> dict:
        image = ImageOps.exif_transpose(image).convert("RGB")
        x = self.tf(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            lg, ls = tta_forward(self.model, x, tta_views)
        lg, ls = lg.cpu(), ls.cpu()
        _, s_dec = decode(lg, ls, self.genus_of_species)
        p_species = F.softmax(ls / self.temp_s, dim=1)[0]
        p_genus = F.softmax(lg / self.temp_g, dim=1)[0]
        s = int(s_dec[0])
        g = int(self.genus_of_species[s])
        conf = float(p_species[s])
        species, genera = self.label_map["species"], self.label_map["genera"]
        top = p_species.topk(min(topk, len(species)))
        result = {
            "model": self.run_id,
            "genus": genera[g],
            "species": pretty(species[s]),
            "species_key": species[s],
            "genus_confidence": float(p_genus[g]),
            "species_confidence": conf,
            "uncertain": conf < self.threshold,
            "threshold": self.threshold,
            "top5": [{"species": pretty(species[int(i)]), "species_key": species[int(i)],
                      "genus": genera[int(self.genus_of_species[int(i)])], "probability": float(p)}
                     for p, i in zip(top.values, top.indices)],
        }
        if species[s].startswith("Notpollen_"):  # optional negative class (see README)
            result["uncertain"] = True
            result["message"] = "The model thinks this is not a pollen micrograph."
        if result["uncertain"]:
            result.setdefault("message", "Low confidence: this may be a species the model was not trained on. "
                                         "Treat the best guess as a lead, not an identification.")
        view = ResizePad(self.size)(image)
        result["input_view"] = _png_data_url(view)
        if gradcam:
            cam = GradCAM(self.model)
            try:
                heat, _ = cam(x, s)
            finally:
                cam.close()
            result["gradcam"] = _png_data_url(overlay(view, heat))
        return result
