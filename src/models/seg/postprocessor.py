# src/models/seg/postprocessor.py: convert raw seg mask logits into standard corners

import numpy as np
import torch

from src.models.base.base_postprocessor import BasePostprocessor
from src.utils.geometry import mask_to_corners


class SegPostprocessor(BasePostprocessor):
    """Thresholds mask logits and extracts (N, 4, 2) corners via extreme points on the mask."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, raw_output):
        probs = torch.sigmoid(raw_output)
        masks = (probs > self.threshold).squeeze(1).detach().cpu().numpy()
        corners = np.stack([mask_to_corners(mask) for mask in masks])
        return torch.from_numpy(corners)
