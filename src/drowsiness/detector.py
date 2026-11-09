"""Stateful drowsiness + gaze detector with per-driver calibration.

Feed it one FaceMeasurements per frame (or None when no face is found) with a
timestamp in seconds. All rules are time-based, so they behave the same at
10 FPS or 30 FPS.

Phases
  1. calibrating — the first N seconds: records YOUR normal open-eye EAR,
     resting mouth opening and neutral head angle (medians, so blinks don't matter).
  2. monitoring  — thresholds are relative to that baseline.

Signals
  eyes_closed   EAR < eye_closed_ratio * baseline EAR (ignored while looking away,
                because EAR is unreliable when the head is turned, and during
                yawns, because the eyes naturally squint)
  microsleep    eyes closed continuously >= microsleep_seconds        -> CRITICAL
  PERCLOS       % of time in SLOW eye closures (>= 0.4 s, so normal blinks
                don't count) over a rolling window                     -> DROWSY
  yawns         mouth open beyond threshold for >= yawn_min_seconds    -> DROWSY (count)
  nods          short pitch excursions (head drops and comes back)     -> DROWSY (count)
  looking_away  head turned / face lost for >= look_away_seconds       -> DISTRACTED
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.drowsiness.metrics import FaceMeasurements


@dataclass
class DrowsinessConfig:
    calibration_seconds: float = 15.0
    eye_closed_ratio: float = 0.70
    microsleep_seconds: float = 1.5
    perclos_window_seconds: float = 60.0
    perclos_min_seconds: float = 10.0
    perclos_drowsy: float = 0.15
    perclos_min_closure_seconds: float = 0.4
    yawn_mar_min: float = 0.50
    yawn_mar_margin: float = 0.35
    yawn_min_seconds: float = 1.0
    yawn_window_seconds: float = 180.0
    yawns_drowsy: int = 3
    look_away_yaw_deg: float = 25.0
    look_away_pitch_deg: float = 20.0
    look_away_seconds: float = 2.0
    nod_pitch_deg: float = 15.0
    nod_min_seconds: float = 0.2
    nod_max_seconds: float = 1.5
    nod_window_seconds: float = 60.0
    nods_drowsy: int = 2

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DrowsinessConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class Baseline:
    ear: float
    mar: float
    yaw: float
    pitch: float


@dataclass
class DrowsinessState:
    phase: str = "calibrating"            # calibrating | monitoring
    calibration_progress: float = 0.0     # 0..1
    face_found: bool = False
    ear: float | None = None
    mar: float | None = None
    yaw_offset: float | None = None       # degrees from your neutral pose
    pitch_offset: float | None = None
    eyes_closed: bool = False
    eyes_closed_seconds: float = 0.0
    perclos: float | None = None          # None until enough data
    yawning: bool = False
    yawns_in_window: int = 0
    nods_in_window: int = 0
    looking_away: bool = False            # sustained >= look_away_seconds
    away_seconds: float = 0.0
    microsleep: bool = False
    level: str = "alert"                  # alert | drowsy | critical  (drowsiness only)
    reasons: list[str] = field(default_factory=list)


class _Episode:
    """Tracks how long a boolean condition has been continuously true."""

    def __init__(self):
        self.start: float | None = None

    def update(self, active: bool, t: float) -> tuple[float, float | None]:
        """Returns (current duration, duration of an episode that just ENDED or None)."""
        if active:
            if self.start is None:
                self.start = t
            return t - self.start, None
        ended = None if self.start is None else t - self.start
        self.start = None
        return 0.0, ended


class DrowsinessDetector:
    def __init__(self, config: DrowsinessConfig | None = None):
        self.cfg = config or DrowsinessConfig()
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        self.baseline: Baseline | None = None
        self._calib: list[FaceMeasurements] = []
        self._calib_start: float | None = None
        self._closed_hist: deque[list] = deque()   # [timestamp, counts_as_closed]
        self._yawn_times: deque[float] = deque()
        self._nod_times: deque[float] = deque()
        self._closed = _Episode()
        self._yawn = _Episode()
        self._away = _Episode()
        self._nod = _Episode()
        self._yawn_counted = False

    def set_baseline(self, baseline: Baseline) -> None:
        """Skip calibration (e.g. reuse a saved baseline)."""
        self.baseline = baseline

    @property
    def calibrated(self) -> bool:
        return self.baseline is not None

    def _calibrate(self, t: float, m: FaceMeasurements | None) -> DrowsinessState:
        if self._calib_start is None:
            self._calib_start = t
        if m is not None:
            self._calib.append(m)
        elapsed = t - self._calib_start
        progress = min(1.0, elapsed / self.cfg.calibration_seconds)
        if progress >= 1.0:
            if len(self._calib) < 10:      # face barely seen: keep calibrating
                self._calib_start = t
                self._calib.clear()
                progress = 0.0
            else:
                arr = np.array([[s.ear, s.mar, s.yaw, s.pitch] for s in self._calib])
                # 75th percentile EAR = typical OPEN eye (blinks pull the median down slightly)
                self.baseline = Baseline(ear=float(np.percentile(arr[:, 0], 75)),
                                         mar=float(np.median(arr[:, 1])),
                                         yaw=float(np.median(arr[:, 2])),
                                         pitch=float(np.median(arr[:, 3])))
                return DrowsinessState(phase="monitoring", calibration_progress=1.0,
                                       face_found=m is not None)
        return DrowsinessState(phase="calibrating", calibration_progress=progress,
                               face_found=m is not None,
                               ear=m.ear if m else None, mar=m.mar if m else None)

    # ----------------------------------------------------------------- update
    def update(self, t: float, m: FaceMeasurements | None) -> DrowsinessState:
        """t: timestamp in seconds; m: this frame's measurements or None (no face)."""
        if not self.calibrated:
            return self._calibrate(t, m)

        c, b = self.cfg, self.baseline
        s = DrowsinessState(phase="monitoring", calibration_progress=1.0, face_found=m is not None)

        # --- head pose / looking away (face lost also counts: head turned too far) ---
        head_away = m is None
        if m is not None:
            s.ear, s.mar = m.ear, m.mar
            s.yaw_offset, s.pitch_offset = m.yaw - b.yaw, m.pitch - b.pitch
            head_away = (abs(s.yaw_offset) > c.look_away_yaw_deg
                         or abs(s.pitch_offset) > c.look_away_pitch_deg)
        s.away_seconds, _ = self._away.update(head_away, t)
        s.looking_away = s.away_seconds >= c.look_away_seconds

        # --- mouth wide open (yawn candidate) ---
        yawn_thr = max(c.yawn_mar_min, b.mar + c.yawn_mar_margin)
        mouth_open = m is not None and m.mar > yawn_thr

        # --- eyes: only trusted when the face is visible, roughly frontal, and not
        #     mid-yawn (eyes naturally squint during a yawn) ---
        eyes_valid = m is not None and not head_away and not mouth_open
        s.eyes_closed = eyes_valid and m.ear < c.eye_closed_ratio * b.ear
        s.eyes_closed_seconds, _ = self._closed.update(s.eyes_closed, t)
        s.microsleep = s.eyes_closed_seconds >= c.microsleep_seconds

        # PERCLOS counts only SLOW closures (>= perclos_min_closure_seconds); ordinary
        # blinks (~0.1–0.4 s) are ignored. Once a closure qualifies, its earlier frames
        # are back-filled as closed so the full closure duration counts.
        if eyes_valid:
            qualifies = s.eyes_closed and s.eyes_closed_seconds >= c.perclos_min_closure_seconds
            self._closed_hist.append([t, qualifies])
            if qualifies and self._closed.start is not None:
                for entry in reversed(self._closed_hist):
                    if entry[0] < self._closed.start:
                        break
                    entry[1] = True
        while self._closed_hist and t - self._closed_hist[0][0] > c.perclos_window_seconds:
            self._closed_hist.popleft()
        if self._closed_hist and t - self._closed_hist[0][0] >= c.perclos_min_seconds:
            s.perclos = float(np.mean([closed for _, closed in self._closed_hist]))

        # --- yawns: counted once per episode, when it has lasted long enough ---
        yawn_dur, _ = self._yawn.update(mouth_open, t)
        s.yawning = yawn_dur >= c.yawn_min_seconds
        if s.yawning and not self._yawn_counted:
            self._yawn_times.append(t)
            self._yawn_counted = True
        if not mouth_open:
            self._yawn_counted = False
        while self._yawn_times and t - self._yawn_times[0] > c.yawn_window_seconds:
            self._yawn_times.popleft()
        s.yawns_in_window = len(self._yawn_times)

        # --- nods: pitch leaves neutral briefly and comes back ---
        pitched = m is not None and abs(m.pitch - b.pitch) > c.nod_pitch_deg
        _, nod_ended = self._nod.update(pitched, t)
        if nod_ended is not None and c.nod_min_seconds <= nod_ended <= c.nod_max_seconds:
            self._nod_times.append(t)
        while self._nod_times and t - self._nod_times[0] > c.nod_window_seconds:
            self._nod_times.popleft()
        s.nods_in_window = len(self._nod_times)

        # --- overall drowsiness level ---
        if s.microsleep:
            s.reasons.append(f"eyes closed {s.eyes_closed_seconds:.1f}s (microsleep)")
        if s.perclos is not None and s.perclos >= c.perclos_drowsy:
            s.reasons.append(f"PERCLOS {s.perclos:.0%}")
        if s.yawns_in_window >= c.yawns_drowsy:
            s.reasons.append(f"{s.yawns_in_window} yawns")
        if s.nods_in_window >= c.nods_drowsy:
            s.reasons.append(f"{s.nods_in_window} head nods")
        if s.microsleep:
            s.level = "critical"
        elif s.reasons:
            s.level = "drowsy"
        if s.looking_away:
            s.reasons.append(f"eyes off road {s.away_seconds:.1f}s")
        return s
