"""Load the project configuration (configs/default.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read a YAML config file into a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def class_names(config: dict[str, Any]) -> dict[str, str]:
    """Map class codes (c0..c9) to human-readable names, in order."""
    return dict(sorted(config["classes"].items(), key=lambda kv: int(kv[0][1:])))
