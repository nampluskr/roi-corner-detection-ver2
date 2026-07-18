# src/losses/bce_loss.py: binary cross-entropy loss for mask logits

import torch.nn as nn

from src.losses.base_loss import BaseLoss


class BCELoss(BaseLoss):
    """Binary cross-entropy on raw mask logits against a binary mask target."""

    def __init__(self, weight=1.0):
        super().__init__(weight=weight)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, raw_output, target):
        return self.criterion(raw_output, target)
