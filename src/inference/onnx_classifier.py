"""Distraction classifier running on ONNX Runtime (CPU, no PyTorch needed)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.inference.preprocess import preprocess_rgb


@dataclass
class Prediction:
    probs: np.ndarray          # calibrated probabilities, shape (10,)
    class_code: str            # e.g. "c1"
    class_name: str            # e.g. "texting - right"
    confidence: float
    p_distracted: float        # 1 - P(safe driving)


class DistractionClassifier:
    """Loads `<model>.onnx` plus `model_meta.json` written by notebooks/03_export_onnx."""

    def __init__(self, model_path: str | Path, meta_path: str | Path | None = None,
                 threads: int | None = None):
        import onnxruntime as ort

        model_path = Path(model_path)
        meta_path = Path(meta_path) if meta_path else model_path.parent / "model_meta.json"
        self.meta = json.loads(meta_path.read_text())
        self.class_codes: list[str] = self.meta["class_codes"]
        self.class_names: dict[str, str] = self.meta["class_names"]
        self.image_size: int = self.meta["image_size"]

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, frame_rgb: np.ndarray) -> Prediction:
        """frame_rgb: HxWx3 uint8 RGB. The model outputs calibrated probabilities."""
        probs = self.session.run(None, {self.input_name: preprocess_rgb(frame_rgb, self.image_size)})[0][0]
        idx = int(np.argmax(probs))
        code = self.class_codes[idx]
        return Prediction(probs=probs, class_code=code, class_name=self.class_names[code],
                          confidence=float(probs[idx]), p_distracted=float(1.0 - probs[0]))
