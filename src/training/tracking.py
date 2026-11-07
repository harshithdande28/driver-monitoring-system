"""Thin MLflow wrapper that never crashes training.

If MLflow has a problem (version change, Drive hiccup), a warning is printed
and training continues; history.csv on Drive is always written regardless.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any


class Tracker:
    def __init__(self, tracking_dir: str | Path, experiment: str, run_name: str,
                 run_id: str | None = None):
        self.run_id: str | None = None
        self._mlflow = None
        try:
            import mlflow
            Path(tracking_dir).mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(Path(tracking_dir).resolve().as_uri())
            mlflow.set_experiment(experiment)
            run = mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run(run_name=run_name)
            self.run_id = run.info.run_id
            self._mlflow = mlflow
        except Exception as e:  # noqa: BLE001 — tracking must never stop training
            warnings.warn(f"MLflow disabled for this run: {e}")

    def log_params(self, params: dict[str, Any]) -> None:
        self._safe(lambda: self._mlflow.log_params(params))

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self._safe(lambda: self._mlflow.log_metrics(metrics, step=step))

    def end(self) -> None:
        self._safe(lambda: self._mlflow.end_run())

    def _safe(self, fn) -> None:
        if self._mlflow is None:
            return
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"MLflow call failed: {e}")
