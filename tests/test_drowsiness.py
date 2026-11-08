"""Tests for the drowsiness metrics and detector (no camera or MediaPipe needed)."""

import numpy as np
import pytest

from src.drowsiness.detector import Baseline, DrowsinessConfig, DrowsinessDetector
from src.drowsiness.metrics import (LEFT_EYE, RIGHT_EYE, FaceMeasurements, euler_from_matrix,
                                    eye_aspect_ratio)

FPS = 20
DT = 1.0 / FPS
OPEN = FaceMeasurements(ear=0.30, mar=0.05, yaw=0.0, pitch=0.0, roll=0.0)
CLOSED = FaceMeasurements(ear=0.10, mar=0.05, yaw=0.0, pitch=0.0, roll=0.0)
YAWN = FaceMeasurements(ear=0.28, mar=0.80, yaw=0.0, pitch=0.0, roll=0.0)
TURNED = FaceMeasurements(ear=0.30, mar=0.05, yaw=40.0, pitch=0.0, roll=0.0)
NODDED = FaceMeasurements(ear=0.30, mar=0.05, yaw=0.0, pitch=25.0, roll=0.0)


def _run(det, frames, t0=0.0):
    """frames: list of (measurement, seconds). Returns (last state, end time)."""
    t, state = t0, None
    for m, secs in frames:
        for _ in range(int(round(secs * FPS))):
            state = det.update(t, m)
            t += DT
    return state, t


@pytest.fixture
def det():
    d = DrowsinessDetector(DrowsinessConfig())
    d.set_baseline(Baseline(ear=0.30, mar=0.05, yaw=0.0, pitch=0.0))
    return d


# ---------------------------------------------------------------- metrics
def test_ear_open_vs_closed():
    pts = np.zeros((478, 2))
    for idx in (RIGHT_EYE, LEFT_EYE):
        p1, p2, p3, p4, p5, p6 = idx
        pts[p1], pts[p4] = (0, 0), (30, 0)          # eye 30 px wide
        pts[p2], pts[p3] = (10, -5), (20, -5)       # 10 px tall when open
        pts[p6], pts[p5] = (10, 5), (20, 5)
    open_ear = eye_aspect_ratio(pts, RIGHT_EYE)
    for idx in (RIGHT_EYE,):
        p1, p2, p3, p4, p5, p6 = idx
        pts[p2], pts[p3], pts[p6], pts[p5] = (10, -1), (20, -1), (10, 1), (20, 1)
    closed_ear = eye_aspect_ratio(pts, RIGHT_EYE)
    assert open_ear == pytest.approx(10 / 30)
    assert closed_ear < open_ear / 3


def test_euler_identity_and_yaw():
    assert euler_from_matrix(np.eye(4)) == pytest.approx((0.0, 0.0, 0.0))
    a = np.radians(30)
    rot_y = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    yaw, pitch, roll = euler_from_matrix(rot_y)
    assert abs(yaw) == pytest.approx(30, abs=1e-6) and abs(pitch) < 1e-6


# --------------------------------------------------------------- detector
def test_calibration_sets_baseline():
    d = DrowsinessDetector(DrowsinessConfig(calibration_seconds=5))
    state, _ = _run(d, [(OPEN, 3)])
    assert state.phase == "calibrating" and 0.5 < state.calibration_progress < 1
    state, _ = _run(d, [(OPEN, 2.5)], t0=3.0)
    assert d.calibrated and d.baseline.ear == pytest.approx(0.30)


def test_blink_is_not_microsleep(det):
    state, t = _run(det, [(OPEN, 2), (CLOSED, 0.3)])
    assert state.eyes_closed and not state.microsleep and state.level == "alert"


def test_microsleep_is_critical(det):
    state, _ = _run(det, [(OPEN, 2), (CLOSED, 2.0)])
    assert state.microsleep and state.level == "critical"


def test_perclos_drowsy(det):
    # 15 s with eyes closed 0.5 s of every 2 s (25%) — long enough for PERCLOS, no microsleep
    frames = [(OPEN, 1.5), (CLOSED, 0.5)] * 8
    state, _ = _run(det, frames)
    assert state.perclos is not None and state.perclos >= 0.2
    assert state.level == "drowsy" and not state.microsleep


def test_talking_is_not_a_yawn(det):
    state, _ = _run(det, [(OPEN, 1), (YAWN, 0.4), (OPEN, 1)])
    assert state.yawns_in_window == 0


def test_three_yawns_drowsy(det):
    state, _ = _run(det, [(OPEN, 2), (YAWN, 2), (OPEN, 5)] * 3)
    assert state.yawns_in_window == 3 and state.level == "drowsy"


def test_looking_away_needs_two_seconds(det):
    state, _ = _run(det, [(OPEN, 1), (TURNED, 1.0)])
    assert not state.looking_away
    state, _ = _run(det, [(TURNED, 1.5)], t0=2.0)
    assert state.looking_away


def test_closed_eyes_ignored_while_turned(det):
    turned_closed = FaceMeasurements(ear=0.10, mar=0.05, yaw=40.0, pitch=0.0, roll=0.0)
    state, _ = _run(det, [(OPEN, 1), (turned_closed, 3)])
    assert not state.microsleep and state.looking_away


def test_face_lost_counts_as_looking_away(det):
    state, _ = _run(det, [(OPEN, 1), (None, 2.5)])
    assert state.looking_away and not state.face_found


def test_head_nods(det):
    state, _ = _run(det, [(OPEN, 2), (NODDED, 0.5), (OPEN, 2), (NODDED, 0.5), (OPEN, 2)])
    assert state.nods_in_window == 2 and state.level == "drowsy"
