"""API tests with fake models (no ONNX files or camera needed).

Requires the API packages:  pip install -r requirements-api.txt
"""

import io

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from app.api import create_app  # noqa: E402
from src.drowsiness.metrics import FaceMeasurements  # noqa: E402
from src.inference.onnx_classifier import Prediction  # noqa: E402

CODES = [f"c{i}" for i in range(10)]


class FakeClassifier:
    class_codes = CODES
    class_names = {c: f"class {c}" for c in CODES}
    meta = {"model": "fake", "deploy_file": "fake.onnx", "temperature": 1.0, "image_size": 224,
            "class_names": class_names, "test_metrics": []}

    def predict(self, rgb):
        assert rgb.ndim == 3 and rgb.shape[2] == 3
        p = np.full(10, 0.02, dtype=np.float32)
        p[1] = 0.82                                    # "texting - right"
        return Prediction(probs=p, class_code="c1", class_name="class c1",
                          confidence=0.82, p_distracted=0.98)


class FakeFace:
    def process(self, rgb, timestamp_ms=0):
        return np.zeros((478, 2)), FaceMeasurements(ear=0.3, mar=0.05, yaw=-2.0, pitch=5.0, roll=1.0)


@pytest.fixture
def client():
    with TestClient(create_app(FakeClassifier(), FakeFace())) as c:
        yield c


def _png(w=64, h=48):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (120, 80, 40)).save(buf, format="PNG")
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["classifier_loaded"]


def test_model_info(client):
    assert client.get("/model-info").json()["model"] == "fake"


def test_predict(client):
    r = client.post("/predict", files={"file": ("x.png", _png(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["activity"]["code"] == "c1"
    assert body["activity"]["is_distracted"] is True
    assert len(body["activity"]["top3"]) == 3
    assert abs(sum(body["activity"]["probabilities"].values()) - 1.0) < 1e-3
    assert body["face"]["found"] and body["image_size"] == [64, 48]


def test_rejects_wrong_type(client):
    r = client.post("/predict", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_rejects_corrupt_image(client):
    r = client.post("/predict", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400
