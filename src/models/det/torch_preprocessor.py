# src/models/det/torch_preprocessor.py: convert standard corners into torchvision detection targets

import torch

from src.models.base.base_preprocessor import BasePreprocessor

NUM_CORNER_CLASSES = 4


class TorchDetPreprocessor(BasePreprocessor):
    """Turns each of the 4 corners into a fixed-size pseudo-box target for torchvision detection models."""

    def __init__(self, image_size=224, box_size=0.1, label_offset=1):
        self.image_size = image_size
        self.box_pixels = box_size * image_size
        self.label_offset = label_offset

    def __call__(self, corners):
        half = self.box_pixels / 2
        labels = torch.arange(NUM_CORNER_CLASSES, device=corners.device) + self.label_offset
        targets = []
        for sample in corners:
            cx = sample[:, 0] * self.image_size
            cy = sample[:, 1] * self.image_size
            boxes = torch.stack([cx - half, cy - half, cx + half, cy + half], dim=1)
            targets.append({"boxes": boxes, "labels": labels})
        return targets
