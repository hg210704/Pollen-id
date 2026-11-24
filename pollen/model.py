"""Shared backbone with two heads (genus, species) and genus-consistent decoding."""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

_FIXED_SIZE_FAMILIES = ("vit", "deit", "beit")  # transformers need the input size at build time


class HierarchicalNet(nn.Module):
    def __init__(self, backbone: str, n_genus: int, n_species: int, genus_of_species: list[int],
                 pretrained: bool = True, drop_rate: float = 0.2, image_size: int = 384):
        super().__init__()
        kwargs = {}
        if any(k in backbone for k in _FIXED_SIZE_FAMILIES):
            kwargs["img_size"] = image_size
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0,
                                          drop_rate=drop_rate, **kwargs)
        dim = self.backbone.num_features
        self.genus_head = nn.Linear(dim, n_genus)
        self.species_head = nn.Linear(dim, n_species)
        self.register_buffer("genus_of_species", torch.tensor(genus_of_species, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        return self.genus_head(feats), self.species_head(feats)

    def heads(self) -> list[nn.Parameter]:
        return list(self.genus_head.parameters()) + list(self.species_head.parameters())


def decode(genus_logits: torch.Tensor, species_logits: torch.Tensor,
           genus_of_species: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Consistent prediction: pick the genus, then the best species INSIDE that genus."""
    g = genus_logits.argmax(1)
    outside = genus_of_species.unsqueeze(0) != g.unsqueeze(1)
    s = species_logits.masked_fill(outside, float("-inf")).argmax(1)
    return g, s


def create_model(model_cfg: dict, label_map: dict, image_size: int,
                 pretrained: bool | None = None) -> HierarchicalNet:
    return HierarchicalNet(
        backbone=model_cfg["backbone"],
        n_genus=len(label_map["genera"]),
        n_species=len(label_map["species"]),
        genus_of_species=label_map["genus_of_species"],
        pretrained=model_cfg.get("pretrained", True) if pretrained is None else pretrained,
        drop_rate=model_cfg.get("drop_rate", 0.2),
        image_size=image_size,
    )
