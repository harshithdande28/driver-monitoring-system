"""Grad-CAM: which parts of the image drove the prediction.

Used to check that the model looks at the hands, phone, cup and head — not at
the background, seat or car interior (which would mean it learned shortcuts).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD


def target_layer(model: nn.Module) -> nn.Module:
    """Last convolutional block — the highest-level spatial features."""
    return model.features[-1]


class GradCAM:
    def __init__(self, model: nn.Module, layer: nn.Module | None = None):
        self.model = model.eval()
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        layer = layer if layer is not None else target_layer(model)
        self._handle = layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output
        output.register_hook(self._save_grad)

    def _save_grad(self, grad):
        self.gradients = grad

    def remove(self) -> None:
        self._handle.remove()

    def __call__(self, image: torch.Tensor, class_idx: int | None = None
                 ) -> tuple[np.ndarray, int, np.ndarray]:
        """image: normalized tensor (3, H, W). Returns (heatmap HxW in [0,1], class, probs)."""
        x = image.unsqueeze(0).to(next(self.model.parameters()).device)
        with torch.enable_grad():
            self.model.zero_grad(set_to_none=True)
            logits = self.model(x)
            probs = logits.softmax(1)[0].detach().cpu().numpy()
            if class_idx is None:
                class_idx = int(logits.argmax(1).item())
            logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[1:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx, probs


def denormalize(image: torch.Tensor) -> np.ndarray:
    """Normalized tensor (3,H,W) -> RGB image (H,W,3) in [0,1]."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (image.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = plt.get_cmap("jet")(cam)[..., :3]
    return np.clip((1 - alpha) * rgb + alpha * heat, 0, 1)


def plot_gradcam_grid(model: nn.Module, dataset, indices: list[int],
                      class_names: dict[str, str], ncols: int = 4) -> plt.Figure:
    """Grid of Grad-CAM overlays for dataset[indices]; title = true vs predicted."""
    codes = list(class_names)
    cam_fn = GradCAM(model)
    nrows = int(np.ceil(len(indices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.6, nrows * 3.2))
    axes = np.atleast_1d(axes).ravel()
    try:
        for ax, idx in zip(axes, indices):
            image, label = dataset[idx]
            cam, pred, probs = cam_fn(image)
            ax.imshow(overlay(denormalize(image), cam))
            ok = pred == label
            ax.set_title(f"true {codes[label]} {class_names[codes[label]][:16]}\n"
                         f"pred {codes[pred]} ({probs[pred]:.0%})",
                         fontsize=8, color="green" if ok else "red")
        for ax in axes:
            ax.set_axis_off()
    finally:
        cam_fn.remove()
    fig.tight_layout()
    return fig
