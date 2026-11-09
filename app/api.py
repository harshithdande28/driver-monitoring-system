"""REST API for the driver monitoring models.

Endpoints
    GET  /health       liveness check + which models are loaded
    GET  /model-info   model name, classes, calibration, test-set metrics
    POST /predict      upload one image -> driver activity + face measurements

The API is stateless (one image per request). Time-based signals such as
PERCLOS, yawn counts and the driver-state machine need a video stream, so they
live in the live monitor (scripts/driver_monitor.py), not here.

Run locally (repo root, .venv active):
    uvicorn app.api:app --reload
    -> interactive docs at http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import io
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

REPO = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO / "models"
MAX_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


# ------------------------------------------------------------------ schemas
class ClassProb(BaseModel):
    code: str
    name: str
    probability: float


class Activity(BaseModel):
    code: str
    name: str
    confidence: float
    p_distracted: float
    is_distracted: bool
    top3: list[ClassProb]
    probabilities: dict[str, float]


class Face(BaseModel):
    found: bool
    eye_aspect_ratio: float | None = None
    mouth_aspect_ratio: float | None = None
    yaw_deg: float | None = None
    pitch_deg: float | None = None
    roll_deg: float | None = None


class PredictResponse(BaseModel):
    activity: Activity | None
    face: Face | None
    image_size: tuple[int, int]
    latency_ms: float


# --------------------------------------------------------------------- app
def create_app(classifier: Any = None, face_tracker: Any = None,
               distraction_threshold: float = 0.70) -> FastAPI:
    """Build the app. Pass objects explicitly (tests) or leave None to load from models/."""
    state: dict[str, Any] = {"classifier": classifier, "face": face_tracker}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # load models once at startup (skipped when objects were passed in)
        if state["classifier"] is None:
            state["classifier"] = _load_classifier()
        if state["face"] is None:
            state["face"] = _load_face_tracker()
        yield

    app = FastAPI(title="Driver Monitoring API",
                  description="Distracted-driver classification (10 classes, calibrated) "
                              "and face measurements for drowsiness/gaze.",
                  version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok",
                "classifier_loaded": state["classifier"] is not None,
                "face_tracker_loaded": state["face"] is not None}

    @app.get("/model-info")
    def model_info() -> dict[str, Any]:
        clf = state["classifier"]
        if clf is None:
            raise HTTPException(503, "Classifier not loaded — put the ONNX files in models/.")
        meta = clf.meta
        return {k: meta.get(k) for k in ("model", "deploy_file", "temperature", "image_size",
                                         "class_names", "test_metrics")} | \
               {"distraction_threshold": distraction_threshold}

    @app.post("/predict", response_model=PredictResponse)
    async def predict(file: UploadFile = File(..., description="JPEG/PNG photo of the driver")):
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(415, f"Unsupported file type {file.content_type}. "
                                     f"Use one of {sorted(ALLOWED_TYPES)}.")
        data = await file.read()
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "Image larger than 10 MB.")
        try:
            rgb = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        except (UnidentifiedImageError, OSError):
            raise HTTPException(400, "Could not decode the image.")

        if state["classifier"] is None and state["face"] is None:
            raise HTTPException(503, "No models loaded.")
        t0 = time.perf_counter()

        activity = None
        clf = state["classifier"]
        if clf is not None:
            pred = clf.predict(rgb)
            order = np.argsort(pred.probs)[::-1]
            activity = Activity(
                code=pred.class_code, name=pred.class_name,
                confidence=round(pred.confidence, 4),
                p_distracted=round(pred.p_distracted, 4),
                is_distracted=pred.p_distracted >= distraction_threshold,
                top3=[ClassProb(code=clf.class_codes[i], name=clf.class_names[clf.class_codes[i]],
                                probability=round(float(pred.probs[i]), 4)) for i in order[:3]],
                probabilities={c: round(float(p), 4) for c, p in zip(clf.class_codes, pred.probs)},
            )

        face = None
        if state["face"] is not None:
            _, m = state["face"].process(rgb)
            face = Face(found=False) if m is None else Face(
                found=True, eye_aspect_ratio=round(m.ear, 4), mouth_aspect_ratio=round(m.mar, 4),
                yaw_deg=round(m.yaw, 1), pitch_deg=round(m.pitch, 1), roll_deg=round(m.roll, 1))

        return PredictResponse(activity=activity, face=face,
                               image_size=(rgb.shape[1], rgb.shape[0]),
                               latency_ms=round((time.perf_counter() - t0) * 1000, 1))

    return app


def _load_classifier():
    import json

    from src.inference.onnx_classifier import DistractionClassifier

    meta = MODELS_DIR / "model_meta.json"
    if not meta.exists():
        print("⚠ models/model_meta.json not found — /predict will return face data only.")
        return None
    deploy = json.loads(meta.read_text()).get("deploy_file", "distraction_fp32.onnx")
    for name in (deploy, "distraction_fp32.onnx", "distraction_int8.onnx"):
        if (MODELS_DIR / name).exists():
            print("Loaded classifier:", name)
            return DistractionClassifier(MODELS_DIR / name)
    print("⚠ No ONNX model found in models/.")
    return None


def _load_face_tracker():
    try:
        from src.drowsiness.landmarks import FaceTracker
        return FaceTracker(video=False)
    except Exception as e:  # noqa: BLE001
        print(f"⚠ Face tracker unavailable: {e}")
        return None


app = create_app()
