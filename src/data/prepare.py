"""Make the dataset available on Colab's fast local disk."""

from __future__ import annotations

import subprocess
from pathlib import Path


def extract_dataset(zip_path: str | Path, local_dir: str | Path) -> Path:
    """Unzip the State Farm archive to local_dir unless it's already there."""
    zip_path, local_dir = Path(zip_path), Path(local_dir)
    if (local_dir / "driver_imgs_list.csv").exists():
        print("Dataset already extracted at", local_dir)
        return local_dir
    if not zip_path.exists():
        raise FileNotFoundError(
            f"{zip_path} not found. Run notebooks/00_setup_data.ipynb first.")
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path.name} (1–2 minutes)...")
    subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(local_dir)], check=True)
    print("Done:", local_dir)
    return local_dir
