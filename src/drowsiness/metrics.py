"""Per-frame face measurements from MediaPipe Face Mesh landmarks.

EAR (eye aspect ratio)   — eye height / eye width. Drops toward 0 when the eye closes.
MAR (mouth aspect ratio) — inner-lip opening / mouth width. Rises sharply during a yawn.
Head pose                — yaw (left/right), pitch (up/down), roll, in degrees, from the
                           4x4 facial transformation matrix MediaPipe estimates.

Landmark indices refer to MediaPipe's 468/478-point face mesh. "Right eye" is the
subject's right eye (left side of a non-mirrored image).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# p1 (outer corner), p2, p3 (upper lid), p4 (inner corner), p5, p6 (lower lid)
RIGHT_EYE = (33, 160, 158, 133, 153, 144)
LEFT_EYE = (362, 385, 387, 263, 373, 380)
# Three vertical inner-lip pairs and the two mouth corners
MOUTH_VERTICAL = ((81, 178), (13, 14), (311, 402))
MOUTH_CORNERS = (78, 308)


@dataclass
class FaceMeasurements:
    ear: float         # mean of both eyes
    mar: float
    yaw: float         # degrees
    pitch: float       # degrees
    roll: float        # degrees


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(points: np.ndarray, idx: tuple[int, ...]) -> float:
    """EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|)  (Soukupová & Čech, 2016)."""
    p1, p2, p3, p4, p5, p6 = (points[i, :2] for i in idx)
    width = _dist(p1, p4)
    if width < 1e-6:
        return 0.0
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * width)


def mouth_aspect_ratio(points: np.ndarray) -> float:
    """Mean inner-lip opening divided by mouth width."""
    width = _dist(points[MOUTH_CORNERS[0], :2], points[MOUTH_CORNERS[1], :2])
    if width < 1e-6:
        return 0.0
    heights = [_dist(points[a, :2], points[b, :2]) for a, b in MOUTH_VERTICAL]
    return float(np.mean(heights) / width)


def euler_from_matrix(matrix: np.ndarray) -> tuple[float, float, float]:
    """(yaw, pitch, roll) in degrees from a 3x3 rotation or 4x4 transform matrix.

    Only *changes* relative to the driver's calibrated baseline are used for
    decisions, so the absolute sign convention does not matter.
    """
    r = np.asarray(matrix, dtype=float)[:3, :3]
    sy = np.hypot(r[0, 0], r[1, 0])
    pitch = np.degrees(np.arctan2(r[2, 1], r[2, 2]))
    yaw = np.degrees(np.arctan2(-r[2, 0], sy))
    roll = np.degrees(np.arctan2(r[1, 0], r[0, 0]))
    return float(yaw), float(pitch), float(roll)


def measure_face(points_px: np.ndarray, transform: np.ndarray) -> FaceMeasurements:
    """points_px: (N, 2 or 3) landmarks in PIXEL coordinates; transform: 4x4 matrix."""
    ear = (eye_aspect_ratio(points_px, RIGHT_EYE) + eye_aspect_ratio(points_px, LEFT_EYE)) / 2
    yaw, pitch, roll = euler_from_matrix(transform)
    return FaceMeasurements(ear=ear, mar=mouth_aspect_ratio(points_px),
                            yaw=yaw, pitch=pitch, roll=roll)
