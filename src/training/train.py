"""Two-phase training loop with checkpoint/resume, AMP and early stopping.

Phase 1 ("head"):     backbone frozen, only the new classifier head trains.
Phase 2 ("finetune"): everything trains with a lower learning rate.

Every epoch writes `last.pt` (full state for resuming) and, when validation
loss improves, `best.pt` (weights for evaluation/export) to Google Drive.
Re-running training with the same run directory resumes automatically.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from src.models.factory import (build_model, count_parameters, freeze_bn_if_frozen,
                                set_backbone_trainable)
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.tracking import Tracker


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    model_name: str
    num_classes: int = 10
    pretrained: bool = True
    head_epochs: int = 2
    finetune_epochs: int = 10
    lr_head: float = 1e-3
    lr_finetune: float = 3e-4
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    early_stopping_patience: int = 3
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def total_epochs(self) -> int:
        return self.head_epochs + self.finetune_epochs

    def phase(self, epoch: int) -> str:
        return "head" if epoch < self.head_epochs else "finetune"


def _make_optimizer(model: nn.Module, cfg: TrainConfig, phase: str):
    trainable = [p for p in model.parameters() if p.requires_grad]
    lr = cfg.lr_head if phase == "head" else cfg.lr_finetune
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=cfg.weight_decay)
    n_epochs = cfg.head_epochs if phase == "head" else cfg.finetune_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, n_epochs))
    return optimizer, scheduler


def _configure_phase(model: nn.Module, cfg: TrainConfig, phase: str) -> None:
    has_backbone = hasattr(model, "features")
    if has_backbone:
        set_backbone_trainable(model, trainable=(phase == "finetune"))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, Any]:
    """Plain (un-smoothed) cross-entropy, accuracy, macro-F1, plus raw outputs."""
    model.eval()
    logits_all, labels_all = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            logits = model(images)
        logits_all.append(logits.float().cpu())
        labels_all.append(labels)
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    preds = logits.argmax(1)
    return {
        "loss": F.cross_entropy(logits, labels).item(),
        "acc": (preds == labels).float().mean().item(),
        "macro_f1": f1_score(labels.numpy(), preds.numpy(), average="macro"),
        "logits": logits,
        "labels": labels,
    }


def train_model(cfg: TrainConfig, loaders: dict[str, DataLoader], run_dir: str | Path,
                tracking_dir: str | Path | None = None, device: str | None = None) -> pd.DataFrame:
    """Train (or resume) a model. Returns the per-epoch history."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    torch.backends.cudnn.benchmark = True

    model = build_model(cfg.model_name, cfg.num_classes, cfg.pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    # ---- resume if a checkpoint exists ----
    state = load_checkpoint(run_dir / "last.pt", device)
    start_epoch, best_val_loss, bad_epochs, history, run_id = 0, float("inf"), 0, [], None
    if state is not None:
        model.load_state_dict(state["model"])
        start_epoch = state["epoch"] + 1
        best_val_loss = state["best_val_loss"]
        bad_epochs = state["bad_epochs"]
        history = state["history"]
        run_id = state.get("mlflow_run_id")
        print(f"Resuming {cfg.model_name} from epoch {start_epoch}/{cfg.total_epochs}")
        if state.get("finished"):
            print("Training already finished for this run.")
            return pd.DataFrame(history)

    tracker = Tracker(tracking_dir, "distraction", run_dir.name, run_id) if tracking_dir else None
    if tracker and state is None:
        tracker.log_params({k: v for k, v in vars(cfg).items() if k != "extra"} | cfg.extra)

    current_phase, optimizer, scheduler = None, None, None
    finished = False
    for epoch in range(start_epoch, cfg.total_epochs):
        phase = cfg.phase(epoch)
        if phase != current_phase:
            _configure_phase(model, cfg, phase)
            optimizer, scheduler = _make_optimizer(model, cfg, phase)
            if state is not None and state.get("phase") == phase and epoch == start_epoch:
                optimizer.load_state_dict(state["optimizer"])
                scheduler.load_state_dict(state["scheduler"])
                scaler.load_state_dict(state["scaler"])
            current_phase = phase
            total, trainable = count_parameters(model)
            print(f"\n=== Phase '{phase}': {trainable:,} / {total:,} parameters trainable ===")

        # ---- train one epoch ----
        model.train()
        if hasattr(model, "features"):
            freeze_bn_if_frozen(model)
        t0 = time.time()
        run_loss, run_correct, run_n = 0.0, 0, 0
        for images, labels in loaders["train"]:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item() * labels.size(0)
            run_correct += (logits.argmax(1) == labels).sum().item()
            run_n += labels.size(0)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # ---- validate ----
        val = evaluate(model, loaders["val"], device)
        row = {
            "epoch": epoch + 1, "phase": phase, "lr": lr,
            "train_loss": run_loss / run_n, "train_acc": run_correct / run_n,
            "val_loss": val["loss"], "val_acc": val["acc"], "val_macro_f1": val["macro_f1"],
            "epoch_time_s": round(time.time() - t0, 1),
        }
        history.append(row)
        print(f"Epoch {epoch + 1:2d}/{cfg.total_epochs} [{phase}] "
              f"train_loss {row['train_loss']:.3f} acc {row['train_acc']:.3f} | "
              f"val_loss {row['val_loss']:.3f} acc {row['val_acc']:.3f} "
              f"F1 {row['val_macro_f1']:.3f} | {row['epoch_time_s']}s")
        if tracker:
            tracker.log_metrics({k: v for k, v in row.items() if isinstance(v, float)}, step=epoch + 1)

        # ---- best model + early stopping (early stopping only in fine-tuning) ----
        if val["loss"] < best_val_loss:
            best_val_loss, bad_epochs = val["loss"], 0
            save_checkpoint({"model": model.state_dict(), "model_name": cfg.model_name,
                             "epoch": epoch + 1, "val": {k: row[k] for k in
                             ("val_loss", "val_acc", "val_macro_f1")}}, run_dir / "best.pt")
            print(f"  ✓ new best model saved (val_loss {best_val_loss:.4f})")
        elif phase == "finetune":
            bad_epochs += 1

        finished = (epoch + 1 == cfg.total_epochs) or (
            phase == "finetune" and bad_epochs >= cfg.early_stopping_patience)
        save_checkpoint({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "phase": phase, "best_val_loss": best_val_loss,
            "bad_epochs": bad_epochs, "history": history,
            "mlflow_run_id": tracker.run_id if tracker else None, "finished": finished,
        }, run_dir / "last.pt")
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        if finished and epoch + 1 < cfg.total_epochs:
            print(f"Early stopping: no val_loss improvement for {bad_epochs} epochs.")
        if finished:
            break

    if tracker:
        tracker.end()
    print(f"\nBest val_loss: {best_val_loss:.4f} — weights at {run_dir / 'best.pt'}")
    return pd.DataFrame(history)


def plot_history(history: pd.DataFrame, title: str = ""):
    """Loss and accuracy curves; the dashed line marks the start of fine-tuning."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ("loss", "acc")):
        ax.plot(history["epoch"], history[f"train_{metric}"], marker="o", label="train")
        ax.plot(history["epoch"], history[f"val_{metric}"], marker="o", label="val (unseen drivers)")
        ft = history.loc[history["phase"] == "finetune", "epoch"]
        if len(ft) and (history["phase"] == "head").any():
            ax.axvline(ft.iloc[0] - 0.5, ls="--", c="gray")
        ax.set_xlabel("epoch")
        ax.set_title(f"{title} {metric}")
        ax.legend()
    fig.tight_layout()
    return fig
