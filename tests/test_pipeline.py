"""End-to-end smoke test on a synthetic dataset: prepare -> train -> evaluate -> predict -> API.
Skipped automatically when PyTorch is not installed. Takes a minute or two on CPU."""
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("fastapi")

from PIL import Image  # noqa: E402

from pollen.data import build_dataset  # noqa: E402
from pollen.predict import Predictor  # noqa: E402
from pollen.train import train  # noqa: E402


@pytest.fixture()
def trained(synth_cfg):
    build_dataset(synth_cfg)
    return synth_cfg, train(synth_cfg, run_id="run_test_resnet18")


def test_training_writes_everything_the_dashboard_needs(trained):
    cfg, run_dir = trained
    for name in ("config.json", "status.json", "history.json", "metrics.json", "evaluation.json",
                 "report.md", "model.pt", "label_map.json", "splits.csv"):
        assert (run_dir / name).exists(), name
    assert json.loads((run_dir / "status.json").read_text())["state"] == "finished"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    # synthetic species differ clearly, so even a tiny run must beat chance (1/9 species, 1/3 genus) by far
    assert metrics["test"]["genus_top1"] > 0.6
    assert metrics["test"]["species_top1"] > 0.4
    assert 0.0 <= metrics["threshold"] <= 1.0


def test_predictor_returns_consistent_answer(trained):
    cfg, run_dir = trained
    p = Predictor(run_dir, device="cpu")
    img = Image.open(next(Path(cfg["data"]["root"]).rglob("*.jpg")))
    r = p.predict(img, tta_views=1, gradcam=True)
    assert r["species_key"].split("_")[0] == r["genus"]       # species belongs to the predicted genus
    assert len(r["top5"]) == 5 and abs(sum(t["probability"] for t in r["top5"])) <= 1.0001
    assert r["gradcam"].startswith("data:image/png;base64,")


def test_api_predict_runs_and_activate(trained, monkeypatch):
    from fastapi.testclient import TestClient

    import api.main as m
    cfg, run_dir = trained
    monkeypatch.setattr(m, "_cfg", lambda: cfg)
    client = TestClient(m.app)
    assert client.get("/health").json()["active_model"] == run_dir.name
    assert client.get("/runs").json()[0]["state"] == "finished"
    assert client.get("/dataset/stats").json()["totals"]["species"] == 9
    img = next(Path(cfg["data"]["root"]).rglob("*.jpg"))
    with open(img, "rb") as f:
        r = client.post("/predict?gradcam=true", files={"file": ("x.jpg", f, "image/jpeg")})
    assert r.status_code == 200 and r.json()["genus"]
    bad = client.post("/predict", files={"file": ("x.txt", b"hello", "text/plain")})
    assert bad.status_code == 400
    assert client.get("/runs/..%2F..%2Fetc").status_code == 404           # no path tricks
    assert client.post("/models/%s/activate" % run_dir.name).json()["active"] == run_dir.name
