"""Evaluation metrics: 10-class, binary (safe vs distracted), calibration error.

All functions take NumPy arrays so they work on logits from any framework.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_recall_fscore_support, roc_auc_score)

SAFE_IDX = 0  # c0 = safe driving


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE: average gap between confidence and accuracy, weighted by bin size."""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def multiclass_metrics(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    preds = probs.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "log_loss": log_loss(labels, np.clip(probs, 1e-7, 1), labels=list(range(probs.shape[1]))),
        "ece": expected_calibration_error(probs, labels),
    }


def binary_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Safe (c0) vs distracted (c1–c9). 'Positive' = distracted.

    P(distracted) = 1 - P(safe). Recall = share of real distraction caught;
    false-alarm rate = share of safe frames wrongly flagged.
    """
    p_distracted = 1.0 - probs[:, SAFE_IDX]
    y_true = (labels != SAFE_IDX).astype(int)
    y_pred = (p_distracted >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "bin_accuracy": (tp + tn) / len(y_true),
        "bin_precision": tp / max(tp + fp, 1),
        "bin_recall": tp / max(tp + fn, 1),
        "bin_f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "false_alarm_rate": fp / max(fp + tn, 1),
        "bin_roc_auc": roc_auc_score(y_true, p_distracted),
    }


def per_class_report(probs: np.ndarray, labels: np.ndarray,
                     class_names: dict[str, str]) -> pd.DataFrame:
    preds = probs.argmax(axis=1)
    p, r, f, s = precision_recall_fscore_support(labels, preds, labels=list(range(len(class_names))),
                                                 zero_division=0)
    return pd.DataFrame({"class": list(class_names), "name": list(class_names.values()),
                         "precision": p, "recall": r, "f1": f, "support": s}).round(3)


def plot_confusion_matrix(probs: np.ndarray, labels: np.ndarray,
                          class_names: dict[str, str], title: str = "") -> plt.Figure:
    """Row-normalized confusion matrix (each row = true class, sums to 1)."""
    preds = probs.argmax(axis=1)
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))), normalize="true")
    ticks = [f"{c} {n}" for c, n in class_names.items()]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] >= 0.01:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(ticks)), ticks, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(ticks)), ticks, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or "Confusion matrix (row-normalized)")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    return fig


def plot_reliability(probs_before: np.ndarray, probs_after: np.ndarray, labels: np.ndarray,
                     n_bins: int = 15, title: str = "") -> plt.Figure:
    """Reliability diagram: confidence vs actual accuracy, before/after calibration."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    edges = np.linspace(0, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    for probs, name in ((probs_before, "before"), (probs_after, "after")):
        conf = probs.max(axis=1)
        correct = probs.argmax(axis=1) == labels
        accs = [correct[(conf > lo) & (conf <= hi)].mean() if ((conf > lo) & (conf <= hi)).any()
                else np.nan for lo, hi in zip(edges[:-1], edges[1:])]
        ece = expected_calibration_error(probs, labels, n_bins)
        ax.plot(centers, accs, marker="o", label=f"{name} (ECE {ece:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", c="gray", label="perfect")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title or "Reliability diagram")
    ax.legend()
    fig.tight_layout()
    return fig
