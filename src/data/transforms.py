"""Image transforms.

The EVAL transform is the single source of truth for preprocessing: the
evaluation notebook, ONNX export and the live demo all reuse it, so the model
sees exactly the same kind of input in training and in deployment.

No horizontal flip anywhere: flipping turns "texting - right" (c1) into
"texting - left" (c3) and would corrupt the labels.
"""

from __future__ import annotations

from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(train: bool, image_size: int = 224) -> T.Compose:
    """Return the training (augmented) or evaluation transform pipeline."""
    normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if not train:
        return T.Compose([T.Resize((image_size, image_size)), T.ToTensor(), normalize])

    return T.Compose([
        # Small crops/shifts: camera position varies slightly between cars.
        T.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(1.1, 1.5)),
        T.RandomRotation(10),
        T.RandomPerspective(distortion_scale=0.15, p=0.3),
        # Lighting changes between drivers, times of day and cameras.
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03),
        # Occasional grayscale: a step toward robustness on IR/night cameras.
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        normalize,
        # Random occlusion so the model can't rely on one small region.
        T.RandomErasing(p=0.25, scale=(0.02, 0.12)),
    ])
