# src/models/det/torch_postprocessor.py: convert torchvision detection predictions into standard corners

import torch

from src.models.base.base_postprocessor import BasePostprocessor

NUM_CORNER_CLASSES = 4


class TorchDetPostprocessor(BasePostprocessor):
    """Selects the highest-scoring box per corner class and decodes its center to (N,4,2)."""

    def __init__(self, image_size=224, label_offset=1):
        self.image_size = image_size
        self.label_offset = label_offset

    def __call__(self, raw_output):
        n = len(raw_output)
        corners = torch.full((n, NUM_CORNER_CLASSES, 2), 0.5)
        for i, pred in enumerate(raw_output):
            boxes, labels, scores = pred["boxes"], pred["labels"], pred["scores"]
            for c in range(NUM_CORNER_CLASSES):
                mask = labels == (c + self.label_offset)
                if not mask.any():
                    continue
                box = boxes[mask][scores[mask].argmax()]
                corners[i, c, 0] = (box[0] + box[2]) / 2 / self.image_size
                corners[i, c, 1] = (box[1] + box[3]) / 2 / self.image_size
        return corners
