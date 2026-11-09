"""Full live driver monitor: distraction + drowsiness + alerts + trip summary.

Usage (repo root, .venv active):
    python scripts/driver_monitor.py                 # default camera
    python scripts/driver_monitor.py --camera 1      # phone as webcam (side angle)
    python scripts/driver_monitor.py --no-classifier # drowsiness/gaze only
    python scripts/driver_monitor.py --mute          # start muted

Needs models/distraction_int8.onnx (or _fp32) + models/model_meta.json from
notebooks/03_export_onnx.ipynb.

Keys (video window focused):  q quit   c recalibrate   s sound on/off   m face mesh
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.config import load_config  # noqa: E402
from src.drowsiness.detector import DrowsinessConfig, DrowsinessDetector  # noqa: E402
from src.drowsiness.landmarks import FaceTracker, open_camera  # noqa: E402
from src.monitor.alerts import AlertConfig, AlertPolicy, AudioPlayer  # noqa: E402
from src.monitor.fusion import DriverState, DriverStateEngine, FusionConfig  # noqa: E402
from src.monitor.trip_log import TripLog  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
from drowsiness_live import draw_face  # noqa: E402

PANEL_W = 340
STATE_COLORS = {"ATTENTIVE": (60, 200, 60), "DISTRACTED": (0, 220, 255),
                "DROWSY": (0, 140, 255), "CRITICAL": (40, 40, 230)}
WHITE, GRAY, RED, YELLOW = (240, 240, 240), (150, 150, 150), (40, 40, 230), (0, 220, 255)


def _text(img, txt, org, color=WHITE, scale=0.55, thick=1):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def find_model(cfg_mon: dict) -> Path | None:
    for candidate in (REPO / cfg_mon["model_file"], REPO / "models" / "distraction_int8.onnx",
                      REPO / "models" / "distraction_fp32.onnx"):
        if candidate.exists() and (candidate.parent / "model_meta.json").exists():
            return candidate
    return None


def render(frame: np.ndarray, s: DriverState, fps: float, muted: bool,
           classifier_on: bool, baseline_ear: float | None) -> np.ndarray:
    d = s.drowsiness
    h, w = frame.shape[:2]
    color = STATE_COLORS[s.state]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 8)                 # state border
    if s.activity_name:
        label = f"{s.activity_name}  {s.activity_confidence:.0%}"
        cv2.rectangle(frame, (8, 8), (18 + 11 * len(label), 40), (20, 20, 20), -1)
        _text(frame, label, (14, 31), WHITE, 0.6, 1)

    panel = np.full((h, PANEL_W, 3), 30, dtype=np.uint8)
    if d.phase == "calibrating":
        _text(panel, "CALIBRATING", (15, 40), YELLOW, 0.8, 2)
        _text(panel, "Sit normally, eyes open,", (15, 75))
        _text(panel, "look straight ahead.", (15, 97))
        cv2.rectangle(panel, (15, 115), (PANEL_W - 15, 133), GRAY, 1)
        cv2.rectangle(panel, (15, 115), (15 + int((PANEL_W - 30) * d.calibration_progress), 133), YELLOW, -1)
        if not d.face_found:
            _text(panel, "No face detected!", (15, 165), RED, 0.6, 2)
    else:
        cv2.rectangle(panel, (10, 10), (PANEL_W - 10, 58), color, -1)
        _text(panel, s.state, (20, 44), (20, 20, 20), 0.95, 2)
        y = 88

        def row(label, value, warn=False):
            nonlocal y
            _text(panel, label, (15, y), GRAY)
            _text(panel, value, (155, y), RED if warn else WHITE)
            y += 23

        _text(panel, "DISTRACTION", (15, y), YELLOW, 0.5); y += 22
        if classifier_on:
            row("Activity", (s.activity_code or "-") + (f" {s.activity_confidence:.0%}" if s.activity_code else ""),
                s.activity_distracted)
            row("P(distracted)", f"{s.p_distracted:.0%}", s.p_distracted >= 0.7)
        else:
            row("Classifier", "off")
        row("Off road", f"{d.away_seconds:.1f} s", s.gaze_distracted)
        y += 6
        _text(panel, "DROWSINESS", (15, y), YELLOW, 0.5); y += 22
        if d.ear is not None and baseline_ear:
            row("Eye openness", f"{d.ear / baseline_ear:.0%}", d.eyes_closed)
        row("Eyes closed", f"{d.eyes_closed_seconds:.1f} s", d.eyes_closed_seconds > 0.5)
        row("PERCLOS", "warming up" if d.perclos is None else f"{d.perclos:.0%}",
            d.perclos is not None and d.perclos >= 0.15)
        row("Yawns", str(d.yawns_in_window), d.yawning)
        row("Head nods", str(d.nods_in_window))
        y += 6
        for reason in s.reasons[:4]:
            _text(panel, "! " + reason[:38], (15, y), (0, 165, 255), 0.47); y += 21

    _text(panel, f"{fps:4.1f} FPS   sound {'OFF' if muted else 'on'}", (15, h - 32), GRAY, 0.45)
    _text(panel, "q quit  c recal  s sound  m mesh", (15, h - 12), GRAY, 0.45)
    return np.hstack([frame, panel])


def main() -> None:
    ap = argparse.ArgumentParser(description="Live driver monitor")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-classifier", action="store_true")
    ap.add_argument("--mute", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    mon = cfg["monitor"]

    classifier = None
    if not args.no_classifier:
        model_path = find_model(mon)
        if model_path is None:
            print("⚠ No ONNX model found in models/ — running drowsiness/gaze only. "
                  "Run notebooks/03_export_onnx.ipynb and copy the files into models/.")
        else:
            from src.inference.onnx_classifier import DistractionClassifier
            classifier = DistractionClassifier(model_path)
            print(f"Distraction model: {model_path.name}")

    class_codes = classifier.class_codes if classifier else [f"c{i}" for i in range(10)]
    class_names = classifier.class_names if classifier else {c: c for c in class_codes}
    detector = DrowsinessDetector(DrowsinessConfig.from_dict(cfg["drowsiness"]))
    engine = DriverStateEngine(FusionConfig.from_dict(mon), class_codes, class_names)
    policy = AlertPolicy(AlertConfig.from_dict(mon))
    player = AudioPlayer(REPO / "models" / "sounds")
    player.set_muted(args.mute)
    trip = TripLog()

    tracker = FaceTracker()
    cap = open_camera(args.camera)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {args.camera}.")

    start = prev = time.perf_counter()
    fps, frame_i, show_mesh = 0.0, 0, False
    every = max(1, int(mon["classify_every_n_frames"]))
    print("Calibrating — sit normally and look ahead.")
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("Camera stopped sending frames.")
                break
            now = time.perf_counter()
            t = now - start
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)   # models expect RGB

            pts, meas = tracker.process(frame_rgb, int(t * 1000))
            drowsy = detector.update(t, meas)
            probs = classifier.predict(frame_rgb).probs if classifier and frame_i % every == 0 else None
            state = engine.update(t, probs, drowsy)

            actions = []
            if drowsy.phase == "monitoring":
                actions = policy.update(t, state.state)
                player.handle(actions)
                trip.update(t, state, actions)

            if pts is not None:
                draw_face(frame_bgr, pts, drowsy, show_mesh)
            inst = 1.0 / max(now - prev, 1e-6)
            fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
            prev = now
            frame_i += 1
            baseline_ear = detector.baseline.ear if detector.baseline else None
            cv2.imshow("Driver monitor", render(frame_bgr, state, fps, player.muted,
                                                classifier is not None, baseline_ear))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                detector.reset(); engine.reset()
                player.handle(policy.update(t, "ATTENTIVE"))
                print("Recalibrating...")
            if key == ord("s"):
                player.set_muted(not player.muted, alarm_active=policy.alarm_on)
            if key == ord("m"):
                show_mesh = not show_mesh
    finally:
        player.close()
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()
        if trip.events:
            print(trip.format_summary())
            print("Saved:", trip.save(REPO / mon["trips_dir"]))


if __name__ == "__main__":
    main()
