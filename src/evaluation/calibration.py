"""Temperature scaling (Guo et al., 2017).

A single scalar T divides the logits: softmax(logits / T). T is fitted on the
VALIDATION set by minimising negative log-likelihood, then applied unchanged to
the test set and to live predictions. It never changes which class is
predicted, only how confident the model is — which matters because the alert
logic thresholds on confidence.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
    logits_t = torch.as_tensor(logits, dtype=torch.float32)
    labels_t = torch.as_tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)  # optimise log(T) so T stays positive
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits_t / log_t.exp(), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Return calibrated probabilities."""
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
