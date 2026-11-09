"""Driver state engine: fuses the distraction classifier with the drowsiness module.

    ATTENTIVE  🟢  nothing wrong
    DISTRACTED 🟡  sustained distracting activity (phone, drinking, ...) OR eyes off road
    DROWSY     🟠  PERCLOS / yawns / nods
    CRITICAL   🔴  microsleep, OR distracted and drowsy at the same time

Classifier probabilities are averaged over a short time window, and a driver
only counts as distracted after the smoothed P(distracted) stays above a
threshold for a few seconds — so single-frame mistakes and quick glances never
trigger an alert.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.drowsiness.detector import DrowsinessState

STATES = ("ATTENTIVE", "DISTRACTED", "DROWSY", "CRITICAL")


@dataclass
class FusionConfig:
    smoothing_seconds: float = 1.0
    distraction_threshold: float = 0.70
    distraction_seconds: float = 2.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FusionConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DriverState:
    state: str = "ATTENTIVE"
    activity_code: str | None = None       # smoothed top class, e.g. "c1"
    activity_name: str | None = None
    activity_confidence: float = 0.0
    p_distracted: float = 0.0              # smoothed
    distracted_seconds: float = 0.0        # how long p_distracted has been above threshold
    activity_distracted: bool = False      # classifier says distracted (sustained)
    gaze_distracted: bool = False          # head turned / face lost (sustained)
    drowsiness: DrowsinessState | None = None
    reasons: list[str] = field(default_factory=list)


class DriverStateEngine:
    def __init__(self, config: FusionConfig, class_codes: list[str], class_names: dict[str, str]):
        self.cfg = config
        self.class_codes = class_codes
        self.class_names = class_names
        self._probs: deque[tuple[float, np.ndarray]] = deque()
        self._above_since: float | None = None

    def reset(self) -> None:
        self._probs.clear()
        self._above_since = None

    def update(self, t: float, probs: np.ndarray | None, drowsy: DrowsinessState) -> DriverState:
        """probs: calibrated class probabilities for this frame, or None if not classified."""
        c = self.cfg
        if probs is not None:
            self._probs.append((t, np.asarray(probs, dtype=np.float32)))
        while self._probs and t - self._probs[0][0] > c.smoothing_seconds:
            self._probs.popleft()

        s = DriverState(drowsiness=drowsy)

        # --- activity (classifier), smoothed over time ---
        if self._probs:
            mean = np.mean([p for _, p in self._probs], axis=0)
            idx = int(np.argmax(mean))
            s.activity_code = self.class_codes[idx]
            s.activity_name = self.class_names[s.activity_code]
            s.activity_confidence = float(mean[idx])
            s.p_distracted = float(1.0 - mean[0])
        above = s.p_distracted >= c.distraction_threshold
        if above:
            self._above_since = t if self._above_since is None else self._above_since
            s.distracted_seconds = t - self._above_since
        else:
            self._above_since = None
        s.activity_distracted = above and s.distracted_seconds >= c.distraction_seconds
        s.gaze_distracted = drowsy.phase == "monitoring" and drowsy.looking_away

        if s.activity_distracted:
            s.reasons.append(f"{s.activity_name} ({s.activity_confidence:.0%}) for "
                             f"{s.distracted_seconds:.1f}s")
        if s.gaze_distracted:
            s.reasons.append(f"eyes off road {drowsy.away_seconds:.1f}s")

        # --- combine ---
        distracted = s.activity_distracted or s.gaze_distracted
        if drowsy.phase != "monitoring":
            s.state = "DISTRACTED" if s.activity_distracted else "ATTENTIVE"
        elif drowsy.level == "critical" or (distracted and drowsy.level == "drowsy"):
            s.state = "CRITICAL"
        elif drowsy.level == "drowsy":
            s.state = "DROWSY"
        elif distracted:
            s.state = "DISTRACTED"
        else:
            s.state = "ATTENTIVE"
        # drowsiness reasons (looking-away is already listed above)
        s.reasons += [r for r in drowsy.reasons if not r.startswith("eyes off road")]
        return s
