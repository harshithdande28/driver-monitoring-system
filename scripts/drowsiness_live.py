"""Live drowsiness + gaze monitor on your webcam (laptop).

Usage (from the repo root, with .venv active):
    python scripts/drowsiness_live.py              # default camera
    python scripts/drowsiness_live.py --camera 1   # e.g. phone used as a webcam

First 15 s: sit normally, eyes open, looking straight ahead (calibration).
Keys (video window focused):  q = quit   c = recalibrate   m = toggle face mesh dots
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.drowsiness.detector import (DrowsinessConfig, DrowsinessDetector,  # noqa: E402
                                     DrowsinessState)
from src.drowsiness.landmarks import FaceTracker, open_camera  # noqa: E402
from src.drowsiness.metrics import LEFT_EYE, MOUTH_CORNERS, MOUTH_VERTICAL, RIGHT_EYE  # noqa: E402

PANEL_W = 320
GREEN, ORANGE, RED, WHITE, GRAY, YELLOW = ((60, 200, 60), (0, 140, 255), (40, 40, 230),
                                           (240, 240, 240), (150, 150, 150), (0, 220, 255))
LEVEL_COLORS = {"alert": GREEN, "drowsy": ORANGE, "critical": RED}


def _text(img, txt, org, color=WHITE, scale=0.55, thick=1):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_face(frame: np.ndarray, pts: np.ndarray, state: DrowsinessState, all_dots: bool) -> None:
    if all_dots:
        for x, y in pts.astype(int):
            cv2.circle(frame, (x, y), 1, GRAY, -1)
    eye_color = RED if state.eyes_closed else GREEN
    for idx in (RIGHT_EYE, LEFT_EYE):
        cv2.polylines(frame, [pts[list(idx)].astype(np.int32)], True, eye_color, 1, cv2.LINE_AA)
    mouth_color = ORANGE if state.yawning else GREEN
    for a, b in MOUTH_VERTICAL:
        cv2.line(frame, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), mouth_color, 1)
    cv2.line(frame, tuple(pts[MOUTH_CORNERS[0]].astype(int)),
             tuple(pts[MOUTH_CORNERS[1]].astype(int)), mouth_color, 1)


def draw_panel(frame: np.ndarray, state: DrowsinessState, fps: float,
               cfg: DrowsinessConfig, baseline_ear: float | None) -> np.ndarray:
    """Return frame with a status panel on the right."""
    h = frame.shape[0]
    panel = np.full((h, PANEL_W, 3), 30, dtype=np.uint8)
    y = 35

    if state.phase == "calibrating":
        _text(panel, "CALIBRATING", (15, y), YELLOW, 0.8, 2); y += 35
        _text(panel, "Sit normally, eyes open,", (15, y)); y += 22
        _text(panel, "look straight ahead.", (15, y)); y += 30
        cv2.rectangle(panel, (15, y), (PANEL_W - 15, y + 18), GRAY, 1)
        cv2.rectangle(panel, (15, y), (15 + int((PANEL_W - 30) * state.calibration_progress), y + 18),
                      YELLOW, -1)
        y += 45
        if not state.face_found:
            _text(panel, "No face detected!", (15, y), RED, 0.6, 2)
    else:
        # overall status (drowsiness level, plus eyes-off-road)
        if state.level == "critical":
            title, color = "CRITICAL", RED
        elif state.level == "drowsy":
            title, color = "DROWSY", ORANGE
        elif state.looking_away:
            title, color = "EYES OFF ROAD", YELLOW
        else:
            title, color = "ATTENTIVE", GREEN
        cv2.rectangle(panel, (10, 10), (PANEL_W - 10, 55), color, -1)
        _text(panel, title, (20, 42), (20, 20, 20), 0.85, 2)
        y = 85

        def row(label, value, warn=False):
            nonlocal y
            _text(panel, label, (15, y), GRAY)
            _text(panel, value, (150, y), RED if warn else WHITE)
            y += 24

        row("Face", "yes" if state.face_found else "NOT FOUND", not state.face_found)
        if state.ear is not None and baseline_ear:
            row("Eye openness", f"{state.ear / baseline_ear:.0%}", state.eyes_closed)
        row("Eyes closed", f"{state.eyes_closed_seconds:.1f} s", state.eyes_closed_seconds > 0.5)
        row("PERCLOS", "warming up" if state.perclos is None else f"{state.perclos:.0%}",
            state.perclos is not None and state.perclos >= cfg.perclos_drowsy)
        row("Mouth (MAR)", "-" if state.mar is None else f"{state.mar:.2f}", state.yawning)
        row("Yawns", f"{state.yawns_in_window} / {int(cfg.yawn_window_seconds // 60)} min",
            state.yawns_in_window >= cfg.yawns_drowsy)
        row("Head nods", f"{state.nods_in_window} / min", state.nods_in_window >= cfg.nods_drowsy)
        if state.yaw_offset is not None:
            row("Head yaw", f"{state.yaw_offset:+.0f} deg",
                abs(state.yaw_offset) > cfg.look_away_yaw_deg)
            row("Head pitch", f"{state.pitch_offset:+.0f} deg",
                abs(state.pitch_offset) > cfg.look_away_pitch_deg)
        row("Off road", f"{state.away_seconds:.1f} s", state.looking_away)

        y += 8
        for reason in state.reasons[:4]:
            _text(panel, "! " + reason, (15, y), ORANGE, 0.5); y += 22

    _text(panel, f"{fps:4.1f} FPS   q quit  c recal", (15, h - 15), GRAY, 0.45)
    return np.hstack([frame, panel])


def main() -> None:
    parser = argparse.ArgumentParser(description="Live drowsiness monitor")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    cfg = DrowsinessConfig.from_dict(load_config()["drowsiness"])
    detector = DrowsinessDetector(cfg)
    tracker = FaceTracker()
    cap = open_camera(args.camera)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {args.camera}. Close other apps using it and check "
                 "Windows camera permissions for desktop apps.")

    start = time.perf_counter()
    prev = start
    fps = 0.0
    show_mesh = False
    print("Calibrating for", cfg.calibration_seconds, "s — sit normally and look ahead.")
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("Camera stopped sending frames.")
                break
            now = time.perf_counter()
            t = now - start
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)   # MediaPipe expects RGB
            pts, meas = tracker.process(frame_rgb, int(t * 1000))
            state = detector.update(t, meas)

            if pts is not None:
                draw_face(frame_bgr, pts, state, show_mesh)
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6)) if fps else 1.0 / max(now - prev, 1e-6)
            prev = now
            baseline_ear = detector.baseline.ear if detector.baseline else None
            cv2.imshow("Driver monitor - drowsiness", draw_panel(frame_bgr, state, fps, cfg, baseline_ear))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                detector.reset()
                print("Recalibrating...")
            if key == ord("m"):
                show_mesh = not show_mesh
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()
        if detector.baseline:
            b = detector.baseline
            print(f"Your baseline: EAR {b.ear:.3f}, MAR {b.mar:.3f}, "
                  f"yaw {b.yaw:+.1f} deg, pitch {b.pitch:+.1f} deg")


if __name__ == "__main__":
    main()
