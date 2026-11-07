"""Checkpoint save/load. Writes are atomic so a Colab disconnect mid-save
can never leave a corrupted checkpoint on Drive."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    return torch.load(path, map_location=device, weights_only=False)
