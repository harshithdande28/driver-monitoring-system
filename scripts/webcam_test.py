"""Webcam + MediaPipe sanity check.

Opens the webcam, runs MediaPipe Face Landmarker on every frame, and draws
the face landmarks plus an FPS counter. If this works, your camera, Python
environment and MediaPipe install are all good.

Usage:
    python scripts/webcam_test.py            # default camera (index 0)
    python scripts/webcam_test.py --camera 1 # try another camera (e.g. phone webcam)

Press "q" (with the video window focused) to quit.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


def ensure_model(path: Path = MODEL_PATH) -> Path:
    """Download the Face Landmarker model once (~4 MB) if it is missing."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading face landmark model to {path} ...")
        urllib.request.urlretrieve(MODEL_URL, path)
        print("Download complete.")
    return path


def open_camera(index: int) -> cv2.VideoCapture:
    """Open a camera. On Windows, DirectShow opens faster and more reliably."""
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
    return cv2.VideoCapture(index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Webcam + MediaPipe face landmark test")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    args = parser.parse_args()

    model_path = ensure_model()

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
    )

    cap = open_camera(args.camera)
    if not cap.isOpened():
        sys.exit(
            f"Could not open camera {args.camera}. Check that no other app (Zoom, Teams) "
            "is using it and that Windows camera access is allowed for desktop apps."
        )

    print("Camera opened. Press 'q' in the video window to quit.")
    prev_time = time.perf_counter()
    start = time.perf_counter()

    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("Failed to read a frame from the camera.")
                break

            # OpenCV gives BGR; MediaPipe (and our models later) expect RGB.
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int((time.perf_counter() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            h, w = frame_bgr.shape[:2]
            face_found = bool(result.face_landmarks)
            if face_found:
                for lm in result.face_landmarks[0]:
                    cv2.circle(frame_bgr, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 0), -1)

            now = time.perf_counter()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            status = "Face detected" if face_found else "No face"
            color = (0, 200, 0) if face_found else (0, 0, 255)
            cv2.putText(frame_bgr, f"{status} | {fps:.1f} FPS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("Webcam test - press q to quit", frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
