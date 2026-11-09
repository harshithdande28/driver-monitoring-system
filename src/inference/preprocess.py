"""Image preprocessing for deployment — NumPy + Pillow only (no PyTorch).

Reproduces the evaluation transform used in training
(torchvision Resize((224, 224)) -> ToTensor -> Normalize(ImageNet)) exactly:
torchvision's Resize on a PIL image uses Pillow's bilinear resize, so the
model sees the same pixels on the laptop as it did in Colab. The export
notebook checks this numerically.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_rgb(image_rgb: np.ndarray | Image.Image, image_size: int = 224) -> np.ndarray:
    """RGB image (HxWx3 uint8 array or PIL) -> float32 array (1, 3, size, size).

    IMPORTANT: OpenCV frames are BGR — convert with cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) first.
    """
    pil = image_rgb if isinstance(image_rgb, Image.Image) else Image.fromarray(image_rgb)
    pil = pil.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    x = np.asarray(pil, dtype=np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.transpose(2, 0, 1)[None].astype(np.float32)
