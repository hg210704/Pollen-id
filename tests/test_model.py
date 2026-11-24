"""Model tests (skipped automatically when PyTorch is not installed)."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from pollen.model import create_model, decode  # noqa: E402


def test_decode_is_genus_consistent():
    # 2 genera, 4 species: species 0,1 in genus 0; species 2,3 in genus 1
    gos = torch.tensor([0, 0, 1, 1])
    genus_logits = torch.tensor([[5.0, 0.0]])             # genus 0 wins
    species_logits = torch.tensor([[0.0, 1.0, 9.0, 2.0]])  # raw best is species 2 (genus 1)
    g, s = decode(genus_logits, species_logits, gos)
    assert int(g) == 0 and int(s) == 1                    # best species INSIDE genus 0


def test_forward_shapes():
    lm = {"species": ["A_x", "A_y", "B_z"], "genera": ["A", "B"], "genus_of_species": [0, 0, 1]}
    model = create_model({"backbone": "resnet18", "drop_rate": 0.0, "pretrained": False}, lm, 64)
    g, s = model(torch.randn(2, 3, 64, 64))
    assert g.shape == (2, 2) and s.shape == (2, 3)
