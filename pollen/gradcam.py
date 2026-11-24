"""Grad-CAM: shows WHICH part of the pollen grain drove the species prediction."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def _target_layer(backbone: nn.Module) -> nn.Module:
    """Last spatial feature layer: ConvNeXt stages, EfficientNet conv_head, ViT last block."""
    if hasattr(backbone, "stages"):
        return backbone.stages[-1]
    if hasattr(backbone, "conv_head"):
        return backbone.conv_head
    if hasattr(backbone, "blocks"):
        return backbone.blocks[-1]
    convs = [m for m in backbone.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError("No layer found for Grad-CAM on this backbone")
    return convs[-1]


class GradCAM:
    def __init__(self, model: nn.Module):
        self.model = model
        self._acts: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None
        self._handle = _target_layer(model.backbone).register_forward_hook(self._on_forward)

    def _on_forward(self, _module, _inputs, output):
        self._acts = output
        if output.requires_grad:
            output.register_hook(lambda g: setattr(self, "_grads", g))

    def close(self) -> None:
        self._handle.remove()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, int]:
        """x: (1,3,H,W). Returns (cam in [0,1] as HxW array, class index explained)."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            _, species_logits = self.model(x)
            if class_idx is None:
                class_idx = int(species_logits.argmax(1))
            species_logits[0, class_idx].backward()
        acts, grads = self._acts[0].detach(), self._grads[0].detach()
        if acts.dim() == 2:  # transformer tokens (N, C) -> (C, side, side)
            n_prefix = getattr(self.model.backbone, "num_prefix_tokens", 1)
            acts, grads = acts[n_prefix:], grads[n_prefix:]
            side = int(math.sqrt(acts.shape[0]))
            acts = acts[: side * side].T.reshape(-1, side, side)
            grads = grads[: side * side].T.reshape(-1, side, side)
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * acts).sum(0)).cpu().numpy()
        cam = cam / (cam.max() + 1e-8)
        return cam, class_idx


def overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a jet-coloured heatmap over an RGB image of any size."""
    from matplotlib import colormaps

    heat = Image.fromarray((cam * 255).astype(np.uint8)).resize(image.size, Image.BICUBIC)
    rgba = colormaps["jet"](np.asarray(heat) / 255.0)[..., :3]
    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    out = (1 - alpha) * base + alpha * rgba
    return Image.fromarray((out.clip(0, 1) * 255).astype(np.uint8))
