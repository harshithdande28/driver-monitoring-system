"""Exploratory data analysis plots for the State Farm dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def plot_class_distribution(df: pd.DataFrame, class_names: dict[str, str]) -> plt.Figure:
    """Bar chart of images per class."""
    counts = df["classname"].value_counts().reindex(list(class_names))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([f"{c}\n{class_names[c]}" for c in counts.index], counts.values, color="#4C72B0")
    ax.set_ylabel("Images")
    ax.set_title("Images per class")
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    return fig


def plot_images_per_driver(df: pd.DataFrame) -> plt.Figure:
    """Bar chart of images per driver, colored by split if a split column exists."""
    counts = df.groupby("subject")["img"].count().sort_values(ascending=False)
    colors = "#4C72B0"
    if "split" in df.columns:
        palette = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
        split_of = df.groupby("subject")["split"].first()
        colors = [palette[split_of[s]] for s in counts.index]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_ylabel("Images")
    ax.set_title("Images per driver" + (" (blue=train, orange=val, green=test)"
                                        if "split" in df.columns else ""))
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    fig.tight_layout()
    return fig


def plot_sample_grid(df: pd.DataFrame, data_dir: str | Path,
                     class_names: dict[str, str], per_class: int = 3,
                     seed: int = 42) -> plt.Figure:
    """Grid of random sample images: one row per class."""
    data_dir = Path(data_dir)
    classes = list(class_names)
    fig, axes = plt.subplots(len(classes), per_class,
                             figsize=(per_class * 3, len(classes) * 2.3))
    for row, cls in enumerate(classes):
        sample = df[df["classname"] == cls].sample(per_class, random_state=seed)
        for col, (_, rec) in enumerate(sample.iterrows()):
            ax = axes[row][col]
            ax.imshow(Image.open(data_dir / rec["path"]))
            ax.set_axis_off()
            if col == 0:
                ax.set_title(f"{cls}: {class_names[cls]}", fontsize=9, loc="left")
    fig.tight_layout()
    return fig
