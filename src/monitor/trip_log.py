"""Trip log: every state change and event with a timestamp, plus an end-of-trip summary."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.monitor.fusion import STATES, DriverState


@dataclass
class TripEvent:
    t: float                 # seconds since the trip started
    kind: str                # state_change | yawn | microsleep | alert
    detail: str


@dataclass
class TripLog:
    started_at: datetime = field(default_factory=datetime.now)
    events: list[TripEvent] = field(default_factory=list)
    time_in_state: dict[str, float] = field(default_factory=lambda: {s: 0.0 for s in STATES})
    _state: str | None = None
    _last_t: float | None = None
    _yawning: bool = False
    _microsleep: bool = False
    _start_t: float | None = None

    def update(self, t: float, s: DriverState, alert_actions: list[str] | None = None) -> None:
        """Call once per frame while monitoring (after calibration)."""
        if self._start_t is None:
            self._start_t = t
        rel = t - self._start_t
        if self._last_t is not None and self._state is not None:
            self.time_in_state[self._state] += t - self._last_t
        self._last_t = t

        if s.state != self._state:
            reason = "; ".join(s.reasons) if s.reasons else ""
            self.events.append(TripEvent(rel, "state_change",
                                         f"{self._state or 'START'} -> {s.state}"
                                         + (f" ({reason})" if reason else "")))
            self._state = s.state

        d = s.drowsiness
        if d is not None:
            if d.yawning and not self._yawning:
                self.events.append(TripEvent(rel, "yawn", ""))
            if d.microsleep and not self._microsleep:
                self.events.append(TripEvent(rel, "microsleep", ""))
            self._yawning, self._microsleep = d.yawning, d.microsleep

        for a in alert_actions or []:
            if a != "alarm_stop":
                self.events.append(TripEvent(rel, "alert", a))

    # --------------------------------------------------------------- summary
    def summary(self) -> dict:
        total = sum(self.time_in_state.values())
        count = lambda kind, detail=None: sum(  # noqa: E731
            1 for e in self.events if e.kind == kind and (detail is None or detail in e.detail))
        return {
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "duration_seconds": round(total, 1),
            "time_in_state_seconds": {k: round(v, 1) for k, v in self.time_in_state.items()},
            "percent_attentive": round(100 * self.time_in_state["ATTENTIVE"] / total, 1) if total else None,
            "distraction_episodes": count("state_change", "-> DISTRACTED"),
            "drowsy_episodes": count("state_change", "-> DROWSY"),
            "critical_episodes": count("state_change", "-> CRITICAL"),
            "yawns": count("yawn"),
            "microsleeps": count("microsleep"),
            "alerts": count("alert"),
        }

    def format_summary(self) -> str:
        s = self.summary()
        t = s["time_in_state_seconds"]
        lines = [
            "=" * 46, " TRIP SUMMARY", "=" * 46,
            f" Duration            {s['duration_seconds']:.0f} s",
            f" Attentive           {s['percent_attentive']}% of the time",
            f" Distracted          {t['DISTRACTED']:.0f} s  ({s['distraction_episodes']} episodes)",
            f" Drowsy              {t['DROWSY']:.0f} s  ({s['drowsy_episodes']} episodes)",
            f" Critical            {t['CRITICAL']:.0f} s  ({s['critical_episodes']} episodes)",
            f" Yawns               {s['yawns']}",
            f" Microsleeps         {s['microsleeps']}",
            f" Alerts played       {s['alerts']}",
            "=" * 46,
        ]
        return "\n".join(lines)

    def save(self, folder: str | Path) -> Path:
        """Write events CSV + summary JSON to folder/trip_<timestamp>_*.  Returns the JSON path."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        stem = "trip_" + self.started_at.strftime("%Y%m%d_%H%M%S")
        with open(folder / f"{stem}_events.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_seconds", "kind", "detail"])
            for e in self.events:
                w.writerow([f"{e.t:.2f}", e.kind, e.detail])
        out = folder / f"{stem}_summary.json"
        out.write_text(json.dumps(self.summary(), indent=2))
        return out
