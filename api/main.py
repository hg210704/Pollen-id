"""FastAPI backend + static frontend.

    uvicorn api.main:app --port 8000        ->  http://localhost:8000
"""
from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from pollen.config import ROOT, load_config, resolve_path
from pollen.data import pretty
from pollen.predict import Predictor
from pollen.registry import (active_run_id, diff_runs, get_run, list_runs, new_run_id, run_path, set_active,
                             write_json_atomic)

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
ALLOWED_OVERRIDE_PREFIXES = ("train.", "model.", "data.", "eval.")
SAFE_VALUE = re.compile(r"^[\w\-./]+$")

app = FastAPI(title="Pollen ID", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://localhost:5173"],
                   allow_methods=["*"], allow_headers=["*"])

_lock = threading.Lock()
_predictor: Predictor | None = None
_train_proc: subprocess.Popen | None = None


def _cfg() -> dict:
    return load_config()


def _runs_dir() -> Path:
    return resolve_path(_cfg()["paths"]["runs_dir"])


def _get_predictor() -> Predictor:
    global _predictor
    rid = active_run_id(_runs_dir())
    if rid is None:
        raise HTTPException(409, "No trained model yet. Train one with `python scripts/train.py` "
                                 "or start a run on the Training page.")
    with _lock:
        if _predictor is None or _predictor.run_id != rid:
            _predictor = Predictor(_runs_dir() / rid)
        return _predictor


# --------------------------------------------------------------------------- prediction
@app.post("/predict")
def predict(file: UploadFile = File(...), gradcam: bool = Query(False), tta: int = Query(4, ge=1, le=8)):
    """Upload one pollen image -> genus, species, top-5, confidence, 'uncertain' flag, optional Grad-CAM."""
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image is larger than 15 MB. Resize it and try again.")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "That file is not a readable image. Use JPG, PNG, TIFF or WebP.")
    predictor = _get_predictor()
    with _lock:  # one model, one request at a time (keeps Grad-CAM hooks safe)
        return predictor.predict(image, tta_views=tta, gradcam=gradcam)


# --------------------------------------------------------------------------- runs / models
@app.get("/runs")
def runs() -> list[dict]:
    return list_runs(_runs_dir())


@app.get("/runs/{run_id}")
def run_detail(run_id: str) -> dict:
    try:
        return get_run(_runs_dir(), run_id)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(404, str(e))


@app.get("/runs/{run_id}/diff/{other_id}")
def run_diff(run_id: str, other_id: str) -> dict:
    try:
        return diff_runs(_runs_dir(), run_id, other_id)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(404, str(e))


@app.get("/models")
def models() -> list[dict]:
    return [r for r in list_runs(_runs_dir()) if r["has_model"]]


@app.post("/models/{run_id}/activate")
def activate(run_id: str) -> dict:
    global _predictor
    try:
        set_active(_runs_dir(), run_id)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(404, str(e))
    with _lock:
        _predictor = None
    return {"active": run_id}


# --------------------------------------------------------------------------- training control
class TrainRequest(BaseModel):
    backbone: str | None = Field(None, description="e.g. convnext_tiny")
    init_from: str | None = Field(None, description="run id to fine-tune from")
    note: str = ""
    overrides: dict[str, Any] = Field(default_factory=dict, description="e.g. {'train.epochs_finetune': 20}")


def _training_running() -> bool:
    if _train_proc is not None and _train_proc.poll() is None:
        return True
    for r in list_runs(_runs_dir()):
        if r["state"] in ("running", "queued"):
            pid = (get_run(_runs_dir(), r["id"])["status"] or {}).get("pid")
            if pid and _pid_alive(int(pid)):
                return True
    return False


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


@app.post("/train")
def start_training(req: TrainRequest) -> dict:
    """Start a training run in a separate process; poll GET /runs/{id} for progress."""
    global _train_proc
    overrides = dict(req.overrides)
    if req.backbone:
        overrides["model.backbone"] = req.backbone
    for k, v in overrides.items():
        if not k.startswith(ALLOWED_OVERRIDE_PREFIXES) or not re.match(r"^[\w.]+$", k):
            raise HTTPException(400, f"Setting '{k}' cannot be changed here.")
        if not isinstance(v, (bool, int, float, str)) or (isinstance(v, str) and not SAFE_VALUE.match(v)):
            raise HTTPException(400, f"Value for '{k}' must be a number, true/false or a plain word.")
    with _lock:
        if _training_running():
            raise HTTPException(409, "A training run is already in progress. Wait for it to finish.")
        cfg = load_config(overrides=overrides)
        run_id = new_run_id(cfg["model"]["backbone"])
        run_dir = run_path(_runs_dir(), run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(run_dir / "status.json", {"run_id": run_id, "state": "queued", "stage": "starting",
                                                    "epoch": 0, "started_at": time.time()})
        cmd = [sys.executable, "-m", "pollen.train", "--run-id", run_id, "--note", req.note[:200]]
        if req.init_from:
            run_path(_runs_dir(), req.init_from)  # validates the id
            cmd += ["--init-from", req.init_from]
        for k, v in overrides.items():
            cmd += ["--set", f"{k}={str(v).lower() if isinstance(v, bool) else v}"]
        log = open(run_dir / "train.log", "w", encoding="utf-8")
        _train_proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
    return {"run_id": run_id, "state": "queued"}


@app.get("/train/status")
def train_status() -> dict:
    """Most recent run's live status (for the dashboard)."""
    rs = list_runs(_runs_dir())
    if not rs:
        return {"state": "none"}
    return get_run(_runs_dir(), rs[0]["id"])["status"] | {"run_id": rs[0]["id"]}


# --------------------------------------------------------------------------- dataset
def _splits() -> pd.DataFrame:
    path = resolve_path(_cfg()["paths"]["cache_dir"]) / "splits.csv"
    if not path.exists():
        raise HTTPException(409, "Dataset not prepared yet. Run `python scripts/prepare_data.py`.")
    return pd.read_csv(path, keep_default_na=False)


@app.get("/dataset/stats")
def dataset_stats() -> dict:
    df = _splits()
    sp = df.groupby(["genus", "species"]).agg(
        images=("path", "size"), records=("group", "nunique"),
        train=("split", lambda s: int((s == "train").sum())),
        val=("split", lambda s: int((s == "val").sum())),
        test=("split", lambda s: int((s == "test").sum()))).reset_index()
    sp["name"] = sp["species"].map(pretty)
    sp["low_data"] = (sp["images"] < 5) | (sp["records"] < 3)
    genus = df.groupby("genus").agg(images=("path", "size"), species=("species", "nunique")).reset_index()
    return {
        "totals": {"images": int(len(df)), "species": int(sp.shape[0]), "genera": int(genus.shape[0]),
                   "low_data_species": int(sp["low_data"].sum()),
                   "split_images": {s: int((df["split"] == s).sum()) for s in ("train", "val", "test")}},
        "genera": genus.sort_values("images", ascending=False).to_dict("records"),
        "species": sp.sort_values("images").to_dict("records"),
    }


@app.get("/dataset/gallery/{species}")
def dataset_gallery(species: str, limit: int = Query(12, ge=1, le=48)) -> list[dict]:
    df = _splits()
    sub = df[df["species"] == species].head(limit)
    if sub.empty:
        raise HTTPException(404, "Unknown species")
    return [{"file": Path(r.path).name, "caption": r.caption, "page_url": r.page_url, "split": r.split,
             "url": f"/dataset/image/{species}/{Path(r.path).name}"} for r in sub.itertuples()]


@app.get("/dataset/image/{species}/{filename}")
def dataset_image(species: str, filename: str):
    """Serves only files that are listed in splits.csv (no arbitrary file access)."""
    df = _splits()
    hit = df[(df["species"] == species) & (df["path"].map(lambda p: Path(p).name) == filename)]
    if hit.empty or not Path(hit.iloc[0]["path"]).exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(hit.iloc[0]["path"])


@app.get("/health")
def health() -> dict:
    return {"ok": True, "active_model": active_run_id(_runs_dir())}


# --------------------------------------------------------------------------- frontend (must be last)
WEB_DIR = ROOT / "web"
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
