"""Build models by name and control which parts are trainable."""

from __future__ import annotations

from torch import nn
from torchvision import models as tvm

from src.models.baseline import BaselineCNN

AVAILABLE = ("baseline", "efficientnet_b0", "mobilenet_v3_large")


def build_model(name: str, num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """Create a model with a fresh `num_classes` output layer."""
    if name == "baseline":
        return BaselineCNN(num_classes)

    if name == "efficientnet_b0":
        weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if name == "mobilenet_v3_large":
        weights = tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        model = tvm.mobilenet_v3_large(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model

    raise ValueError(f"Unknown model '{name}'. Choose from {AVAILABLE}")


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze/unfreeze the feature extractor; the classifier head always trains.

    Frozen BatchNorm layers are also put in eval mode by `freeze_bn_if_frozen`
    during training so their running statistics stay at the pretrained values.
    """
    for p in model.features.parameters():
        p.requires_grad = trainable


def freeze_bn_if_frozen(model: nn.Module) -> None:
    """Keep BatchNorm layers of a frozen backbone in eval mode."""
    for m in model.features.modules():
        if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
