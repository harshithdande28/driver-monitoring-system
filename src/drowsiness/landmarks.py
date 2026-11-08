"""MediaPipe Face Landmarker wrapper + camera helper (laptop side)."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import numpy as np

from src.drowsiness.metrics import FaceMeasurements, measure_face

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "models" / "face_landmarker.task"


def ensure_face_model(path: Path = MODEL_PATH) -> Path:
    """Download the Face Landmarker model once (~4 MB)."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading face landmark model to {path} ...")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


def open_camera(index: int = 0):
    """Open a camera; DirectShow is faster and more reliable on Windows."""
    import cv2

    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
    return cv2.VideoCapture(index)


class FaceTracker:
    """Runs MediaPipe on video frames and returns landmarks + measurements."""

    def __init__(self, model_path: Path | None = None):
        import mediapipe as mp

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path or ensure_face_model())),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,   # gives head pose directly
        )
        self._mp = mp
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, frame_rgb: np.ndarray, timestamp_ms: int
                ) -> tuple[np.ndarray | None, FaceMeasurements | None]:
        """frame_rgb: HxWx3 RGB uint8. Returns (landmarks in pixels, measurements) or (None, None)."""
        timestamp_ms = max(timestamp_ms, self._last_ts + 1)   # MediaPipe needs increasing timestamps
        self._last_ts = timestamp_ms
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.face_landmarks:
            return None, None
        h, w = frame_rgb.shape[:2]
        pts = np.array([(lm.x * w, lm.y * h) for lm in result.face_landmarks[0]], dtype=np.float32)
        transform = (np.array(result.facial_transformation_matrixes[0])
                     if result.facial_transformation_matrixes else np.eye(4))
        return pts, measure_face(pts, transform)

    def close(self) -> None:
        self._landmarker.close()
