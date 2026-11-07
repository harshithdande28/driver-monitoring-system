"""Driver-wise train/val/test split for the State Farm dataset.

Why split by driver?
    Consecutive frames of the same driver are almost identical. A random,
    image-level split puts near-duplicate frames in both train and test, so the
    model can score ~99% by memorising people instead of learning behaviours.
    Holding out entire drivers measures what we actually care about: how well
    the model works on people it has never seen.

How the split is chosen:
    Many random driver assignments are tried (deterministically, from a seed).
    Each is scored on (a) how closely every split's class distribution matches
    the overall distribution and (b) how close each split's share of images is
    to its target fraction. The best-scoring assignment is kept.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def load_driver_list(data_dir: str | Path) -> pd.DataFrame:
    """Read driver_imgs_list.csv and add a relative image path column.

    Columns: subject (driver id), classname (c0..c9), img (file name), path.
    """
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "driver_imgs_list.csv")
    expected = {"subject", "classname", "img"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"driver_imgs_list.csv is missing columns: {missing}")
    df["path"] = "imgs/train/" + df["classname"] + "/" + df["img"]
    return df


def _split_score(df: pd.DataFrame, assignment: dict[str, str],
                 targets: dict[str, float], overall: pd.Series) -> float:
    """Lower is better: class-distribution mismatch + image-share mismatch."""
    split_col = df["subject"].map(assignment)
    score = 0.0
    n_total = len(df)
    for split in SPLITS:
        part = df.loc[split_col == split, "classname"]
        dist = part.value_counts(normalize=True).reindex(overall.index, fill_value=0.0)
        score += float(np.abs(dist - overall).sum())             # class balance
        score += abs(len(part) / n_total - targets[split]) * 2.0  # size balance
    return score


def make_driver_split(df: pd.DataFrame, val_fraction: float = 0.15,
                      test_fraction: float = 0.15, seed: int = 42,
                      trials: int = 2000) -> pd.DataFrame:
    """Return a copy of df with a 'split' column; no driver appears in two splits."""
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be in (0, 1) and sum to < 1")

    drivers = np.array(sorted(df["subject"].unique()))
    n = len(drivers)
    n_val = max(1, round(n * val_fraction))
    n_test = max(1, round(n * test_fraction))
    if n_val + n_test >= n:
        raise ValueError(f"Not enough drivers ({n}) for the requested split")

    targets = {"train": 1 - val_fraction - test_fraction,
               "val": val_fraction, "test": test_fraction}
    overall = df["classname"].value_counts(normalize=True).sort_index()
    rng = np.random.default_rng(seed)

    best_assignment: dict[str, str] | None = None
    best_score = float("inf")
    for _ in range(trials):
        perm = rng.permutation(drivers)
        assignment = {d: "test" for d in perm[:n_test]}
        assignment.update({d: "val" for d in perm[n_test:n_test + n_val]})
        assignment.update({d: "train" for d in perm[n_test + n_val:]})
        score = _split_score(df, assignment, targets, overall)
        if score < best_score:
            best_score, best_assignment = score, assignment

    out = df.copy()
    out["split"] = out["subject"].map(best_assignment)
    check_no_leakage(out)
    return out


def check_no_leakage(df: pd.DataFrame) -> None:
    """Raise if any driver appears in more than one split."""
    splits_per_driver = df.groupby("subject")["split"].nunique()
    leaked = splits_per_driver[splits_per_driver > 1]
    if not leaked.empty:
        raise AssertionError(f"Drivers in multiple splits: {list(leaked.index)}")


def summarize_split(df: pd.DataFrame) -> pd.DataFrame:
    """Drivers, images and image share per split."""
    summary = df.groupby("split").agg(drivers=("subject", "nunique"), images=("img", "count"))
    summary["image_share"] = (summary["images"] / summary["images"].sum()).round(3)
    return summary.reindex(list(SPLITS))


def class_counts_by_split(df: pd.DataFrame) -> pd.DataFrame:
    """Image counts per class (rows) and split (columns)."""
    return (df.pivot_table(index="classname", columns="split", values="img",
                           aggfunc="count", fill_value=0)
              .reindex(columns=list(SPLITS)))
