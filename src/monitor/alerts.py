"""Escalating alerts.

AlertPolicy decides WHAT to play and WHEN (pure logic, unit-tested).
AudioPlayer actually plays it (Windows sounds + offline text-to-speech).

    ATTENTIVE   silent (any alarm stops immediately)
    DISTRACTED  soft chime on entry, repeated every few seconds
    DROWSY      beep on entry and repeated + spoken warning (with a cooldown)
    CRITICAL    continuous alarm until the driver is no longer critical
"""

from __future__ import annotations

import queue
import sys
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class AlertConfig:
    distracted_repeat_seconds: float = 4.0
    drowsy_beep_seconds: float = 5.0
    drowsy_speech_seconds: float = 30.0
    drowsy_message: str = "You seem tired. Please take a break."

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AlertConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class AlertPolicy:
    """Turns a stream of driver states into sound actions:
    "chime", "beep", "speak:<text>", "alarm_start", "alarm_stop"."""

    def __init__(self, config: AlertConfig):
        self.cfg = config
        self.prev_state = "ATTENTIVE"
        self.alarm_on = False
        self._last: dict[str, float] = {}

    def _due(self, key: str, t: float, every: float) -> bool:
        last = self._last.get(key)
        if last is None or t - last >= every:
            self._last[key] = t
            return True
        return False

    def update(self, t: float, state: str) -> list[str]:
        c, actions = self.cfg, []
        entered = state != self.prev_state

        if state == "CRITICAL":
            if not self.alarm_on:
                actions.append("alarm_start")
                self.alarm_on = True
        elif self.alarm_on:
            actions.append("alarm_stop")
            self.alarm_on = False

        if entered:   # restart repeat timers so a new episode alerts immediately
            for key in ("chime", "beep"):
                self._last.pop(key, None)

        if state == "DISTRACTED" and self._due("chime", t, c.distracted_repeat_seconds):
            actions.append("chime")
        elif state == "DROWSY":
            if self._due("beep", t, c.drowsy_beep_seconds):
                actions.append("beep")
            if self._due("speech", t, c.drowsy_speech_seconds):
                actions.append("speak:" + c.drowsy_message)

        self.prev_state = state
        return actions


# ------------------------------------------------------------------ audio
def _write_tone(path: Path, freqs: list[float], seconds: float, volume: float = 0.5,
                rate: int = 22050) -> None:
    """Write a short WAV made of consecutive tones (with fade in/out to avoid clicks)."""
    chunks = []
    for f in freqs:
        n = int(rate * seconds / len(freqs))
        tt = np.arange(n) / rate
        tone = np.sin(2 * np.pi * f * tt)
        fade = np.minimum(1, np.minimum(np.arange(n), np.arange(n)[::-1]) / (0.01 * rate))
        chunks.append(tone * fade)
    data = (np.concatenate(chunks) * volume * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data.tobytes())


def make_sounds(folder: Path) -> dict[str, Path]:
    sounds = {
        "chime": (folder / "chime.wav", [880, 1320], 0.5, 0.35),
        "beep": (folder / "beep.wav", [1000, 0, 1000], 0.6, 0.6),
        "alarm": (folder / "alarm.wav", [1400, 900] * 2, 1.0, 0.9),
    }
    for path, freqs, secs, vol in sounds.values():
        if not path.exists():
            _write_tone(path, freqs, secs, vol)
    return {k: v[0] for k, v in sounds.items()}


class AudioPlayer:
    """Plays alert sounds without blocking the video loop.

    Windows: winsound (built in) for tones, pyttsx3 (offline voices) for speech.
    Other OSes: prints the alert to the console instead.
    """

    def __init__(self, sounds_dir: Path, enable_speech: bool = True):
        self.sounds = make_sounds(sounds_dir)
        self.muted = False
        self._win = sys.platform.startswith("win")
        self._speech_q: queue.Queue[str | None] = queue.Queue()
        self._speech_thread = None
        if enable_speech:
            self._speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
            self._speech_thread.start()

    def _speech_worker(self) -> None:
        try:
            import pyttsx3
            engine = pyttsx3.init()      # must be created in the thread that uses it
            engine.setProperty("rate", 165)
        except Exception as e:  # noqa: BLE001
            print(f"[alerts] Text-to-speech unavailable ({e}); spoken warnings disabled.")
            return
        while True:
            text = self._speech_q.get()
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:  # noqa: BLE001
                print(f"[alerts] speech failed: {e}")

    def _play(self, name: str, loop: bool = False) -> None:
        if self._win:
            import winsound
            flags = winsound.SND_FILENAME | winsound.SND_ASYNC
            if loop:
                flags |= winsound.SND_LOOP
            winsound.PlaySound(str(self.sounds[name]), flags)
        else:
            print(f"[alert sound] {name}{' (looping)' if loop else ''}")

    def _stop(self) -> None:
        if self._win:
            import winsound
            winsound.PlaySound(None, 0)

    def handle(self, actions: list[str]) -> None:
        for a in actions:
            if a == "alarm_stop":
                self._stop()
            elif self.muted:
                continue
            elif a == "alarm_start":
                self._play("alarm", loop=True)
            elif a in ("chime", "beep"):
                self._play(a)
            elif a.startswith("speak:"):
                self._speech_q.put(a[len("speak:"):])

    def set_muted(self, muted: bool, alarm_active: bool = False) -> None:
        """Mute/unmute. When unmuting during CRITICAL, the alarm resumes."""
        self.muted = muted
        if muted:
            self._stop()
        elif alarm_active:
            self._play("alarm", loop=True)

    def close(self) -> None:
        self._stop()
        self._speech_q.put(None)
