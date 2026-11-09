"""Tests for the driver state engine, alert policy and trip log."""

import numpy as np
import pytest

from src.drowsiness.detector import DrowsinessState
from src.monitor.alerts import AlertConfig, AlertPolicy, make_sounds
from src.monitor.fusion import DriverStateEngine, FusionConfig
from src.monitor.trip_log import TripLog

CODES = [f"c{i}" for i in range(10)]
NAMES = {c: f"class {c}" for c in CODES}
FPS = 10
DT = 1.0 / FPS


def probs_for(idx, conf=0.9):
    p = np.full(10, (1 - conf) / 9)
    p[idx] = conf
    return p


def awake(**kw):
    return DrowsinessState(phase="monitoring", face_found=True, **kw)


@pytest.fixture
def engine():
    return DriverStateEngine(FusionConfig(), CODES, NAMES)


def run(engine, seconds, probs, drowsy, t0=0.0):
    t, s = t0, None
    for _ in range(int(seconds * FPS)):
        s = engine.update(t, probs, drowsy)
        t += DT
    return s, t


# ---------------------------------------------------------------- fusion
def test_safe_driving_is_attentive(engine):
    s, _ = run(engine, 3, probs_for(0), awake())
    assert s.state == "ATTENTIVE" and s.activity_code == "c0"


def test_brief_phone_glance_does_not_trigger(engine):
    s, t = run(engine, 2, probs_for(0), awake())
    s, _ = run(engine, 1.0, probs_for(1), awake(), t0=t)
    assert s.state == "ATTENTIVE"


def test_sustained_texting_is_distracted(engine):
    s, t = run(engine, 1, probs_for(0), awake())
    s, _ = run(engine, 4, probs_for(1), awake(), t0=t)
    assert s.state == "DISTRACTED" and s.activity_code == "c1"


def test_low_confidence_does_not_trigger(engine):
    unsure = np.full(10, 0.06)
    unsure[0] = 0.46                       # P(distracted) = 0.54 < 0.70
    s, _ = run(engine, 5, unsure, awake())
    assert s.state == "ATTENTIVE"


def test_eyes_off_road_is_distracted(engine):
    s, _ = run(engine, 2, probs_for(0), awake(looking_away=True, away_seconds=2.5))
    assert s.state == "DISTRACTED" and s.gaze_distracted


def test_drowsy_and_critical(engine):
    s, _ = run(engine, 1, probs_for(0), awake(level="drowsy", reasons=["PERCLOS 20%"]))
    assert s.state == "DROWSY"
    s, _ = run(engine, 1, probs_for(0), awake(level="critical", microsleep=True))
    assert s.state == "CRITICAL"


def test_distracted_and_drowsy_is_critical(engine):
    s, _ = run(engine, 4, probs_for(1), awake(level="drowsy"))
    assert s.state == "CRITICAL"


def test_classifier_skipped_frames_still_work(engine):
    t, s = 0.0, None
    for i in range(40):                     # classifier only every 2nd frame
        s = engine.update(t, probs_for(6) if i % 2 == 0 else None, awake())
        t += DT
    assert s.state == "DISTRACTED" and s.activity_code == "c6"


# ---------------------------------------------------------------- alerts
def test_alert_escalation():
    p = AlertPolicy(AlertConfig(distracted_repeat_seconds=4, drowsy_beep_seconds=5,
                                drowsy_speech_seconds=30))
    assert p.update(0.0, "ATTENTIVE") == []
    assert p.update(1.0, "DISTRACTED") == ["chime"]
    assert p.update(2.0, "DISTRACTED") == []              # cooldown
    assert p.update(5.0, "DISTRACTED") == ["chime"]       # repeats
    a = p.update(6.0, "DROWSY")
    assert "beep" in a and any(x.startswith("speak:") for x in a)
    assert p.update(7.0, "CRITICAL") == ["alarm_start"]
    assert p.update(8.0, "CRITICAL") == []                # alarm keeps looping
    assert p.update(9.0, "ATTENTIVE") == ["alarm_stop"]   # stops immediately


def test_speech_cooldown():
    p = AlertPolicy(AlertConfig(drowsy_speech_seconds=30))
    first = p.update(0.0, "DROWSY")
    p.update(1.0, "ATTENTIVE")
    again = p.update(2.0, "DROWSY")
    assert any(x.startswith("speak:") for x in first)
    assert not any(x.startswith("speak:") for x in again)   # not twice within 30 s
    assert "beep" in again


def test_sound_files_created(tmp_path):
    sounds = make_sounds(tmp_path)
    assert all(p.exists() and p.stat().st_size > 1000 for p in sounds.values())


# --------------------------------------------------------------- trip log
def test_trip_summary(engine, tmp_path):
    log = TripLog()
    t = 0.0
    for probs, drowsy, secs in [(probs_for(0), awake(), 5),
                                (probs_for(1), awake(), 5),
                                (probs_for(0), awake(), 5)]:
        for _ in range(int(secs * FPS)):
            log.update(t, engine.update(t, probs, drowsy))
            t += DT
    s = log.summary()
    assert s["distraction_episodes"] == 1
    assert 2.0 < s["time_in_state_seconds"]["DISTRACTED"] < 4.5
    assert s["duration_seconds"] == pytest.approx(15, abs=0.2)
    assert log.save(tmp_path).exists()
    assert "TRIP SUMMARY" in log.format_summary()
