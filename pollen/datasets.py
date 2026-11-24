"""Torch side of the data pipeline: augmentation transforms and the Dataset class."""
from __future__ import annotations

import random
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def _corner_color(img: Image.Image) -> tuple[int, int, int]:
    a = np.asarray(img)
    corners = np.stack([a[0, 0], a[0, -1], a[-1, 0], a[-1, -1]]).astype(np.float32)
    return tuple(int(c) for c in corners.mean(0))


class ResizePad:
    """Resize so the longest side == size (aspect ratio kept), pad to a square with the
    background colour taken from the image corners."""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        s = self.size / max(w, h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        img = img.resize((nw, nh), Image.BICUBIC)
        canvas = Image.new("RGB", (self.size, self.size), _corner_color(img))
        canvas.paste(img, ((self.size - nw) // 2, (self.size - nh) // 2))
        return canvas


class RandomRotateFill:
    """Rotate by any angle (pollen has no 'up'), filling corners with the background colour."""

    def __call__(self, img: Image.Image) -> Image.Image:
        return img.rotate(random.uniform(0, 360), resample=Image.BICUBIC, expand=True,
                          fillcolor=_corner_color(img))


class StainJitter:
    """Random per-channel gain and an occasional strong stain tint: slides come in many colours."""
    TINTS = [(1.00, 0.55, 0.45), (1.00, 0.80, 0.40), (0.60, 0.80, 1.00), (0.70, 1.00, 0.70), (0.90, 0.60, 1.00)]

    def __init__(self, p: float = 0.6, gain: float = 0.18, tint_p: float = 0.25):
        self.p, self.gain, self.tint_p = p, gain, tint_p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        arr = np.asarray(img, dtype=np.float32)
        arr = arr * np.array([random.uniform(1 - self.gain, 1 + self.gain) for _ in range(3)], dtype=np.float32)
        if random.random() < self.tint_p:
            alpha = random.uniform(0.15, 0.40)
            arr = arr * ((1 - alpha) + alpha * np.array(random.choice(self.TINTS), dtype=np.float32) * 1.2)
        return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


class AddNoise:
    def __init__(self, p: float = 0.3, std: float = 0.02):
        self.p, self.std = p, std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std if random.random() < self.p else x


def train_transform(size: int) -> Callable:
    """Augmentations that keep pollen morphology intact (no elastic/perspective warps)."""
    return T.Compose([
        RandomRotateFill(),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        ResizePad(size),
        T.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        StainJitter(),
        T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.03),
        T.RandomGrayscale(p=0.08),
        T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.2))], p=0.25),
        T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD), AddNoise(),
        T.RandomErasing(p=0.2, scale=(0.02, 0.10), value=0),  # small occlusions (debris, bubbles)
    ])


def eval_transform(size: int) -> Callable:
    return T.Compose([ResizePad(size), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class PollenDataset(Dataset):
    """Returns (image_tensor, species_index, genus_index)."""

    def __init__(self, df: pd.DataFrame, label_map: dict, transform: Callable):
        sp_idx = {s: i for i, s in enumerate(label_map["species"])}
        self.paths = df["path"].tolist()
        self.species_idx = [sp_idx[s] for s in df["species"]]
        gos = label_map["genus_of_species"]
        self.genus_idx = [gos[i] for i in self.species_idx]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
        return self.transform(img), self.species_idx[i], self.genus_idx[i]
