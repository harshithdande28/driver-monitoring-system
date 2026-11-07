"""PyTorch Dataset and DataLoaders for the State Farm images."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.transforms import build_transforms


class DriverDataset(Dataset):
    """Images listed in a split DataFrame (columns: path, classname)."""

    def __init__(self, df: pd.DataFrame, root: str | Path, transform, class_codes: list[str]):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(class_codes)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image = Image.open(self.root / row["path"]).convert("RGB")
        return self.transform(image), self.class_to_idx[row["classname"]]


def subsample(df: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Keep `fraction` of each (split, class) group — used for the smoke test."""
    if fraction >= 1.0:
        return df
    keep: list = []
    for _, group in df.groupby(["split", "classname"]):
        n = max(1, int(len(group) * fraction))
        keep.extend(group.sample(n, random_state=seed).index)
    return df.loc[keep]


def build_dataloaders(split_df: pd.DataFrame, root: str | Path, class_codes: list[str],
                      image_size: int = 224, batch_size: int = 64, num_workers: int = 2,
                      fraction: float = 1.0, seed: int = 42) -> dict[str, DataLoader]:
    """Return {'train', 'val', 'test'} DataLoaders built from the driver-wise split."""
    df = subsample(split_df, fraction, seed)
    loaders = {}
    for split in ("train", "val", "test"):
        part = df[df["split"] == split]
        is_train = split == "train"
        dataset = DriverDataset(part, root, build_transforms(is_train, image_size), class_codes)
        generator = torch.Generator().manual_seed(seed)
        loaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers,
            pin_memory=torch.cuda.is_available(), drop_last=False,
            persistent_workers=num_workers > 0, generator=generator,
        )
    return loaders
