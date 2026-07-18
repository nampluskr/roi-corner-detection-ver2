# src/losses/dice_loss.py: soft Dice loss for mask logits

import torch

from src.losses.base_loss import BaseLoss


class DiceLoss(BaseLoss):
    """Soft Dice loss between sigmoid mask probabilities and a binary mask target."""

    def __init__(self, smooth=1.0, weight=1.0):
        super().__init__(weight=weight)
        self.smooth = smooth

    def forward(self, raw_output, target):
        probs = torch.sigmoid(raw_output).reshape(raw_output.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
        intersection = (probs * target).sum(dim=1)
        union = probs.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return (1.0 - dice).mean()
