"""Inference utilities for pretrained MobileNetV3 Small."""
from __future__ import annotations

from functools import lru_cache
from PIL import Image, ImageOps


def prepare_image(image: Image.Image) -> Image.Image:
    """Apply EXIF rotation and convert images to RGB."""
    return ImageOps.exif_transpose(image).convert("RGB")


def top_predictions(logits, categories: list[str], k: int = 5) -> list[dict]:
    """Return sorted ImageNet class probabilities from unnormalized logits."""
    import torch

    if k < 1:
        raise ValueError("k must be at least 1")
    probabilities = torch.softmax(logits.squeeze(0), dim=0)
    values, indices = probabilities.topk(min(k, len(categories)))
    return [
        {"label": categories[int(i)], "confidence": float(p)}
        for p, i in zip(values.tolist(), indices.tolist())
    ]


@lru_cache(maxsize=1)
def load_classifier():
    """Load official TorchVision pretrained ImageNet weights (downloads once)."""
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    model.eval()
    return model, weights.transforms(), weights.meta["categories"]


def classify(image: Image.Image, k: int = 5) -> list[dict]:
    import torch

    model, transforms, categories = load_classifier()
    tensor = transforms(prepare_image(image)).unsqueeze(0)
    with torch.inference_mode():
        logits = model(tensor)
    return top_predictions(logits, categories, k=k)
