# src/losses/focal_loss.py: sigmoid focal loss for the sparse per-cell corner classification map

import torch
import torch.nn.functional as F

from src.losses.base_loss import BaseLoss


class FocalLoss(BaseLoss):
    """RetinaNet-style sigmoid focal loss between a per-class classification map and a binary target."""

    def __init__(self, alpha=0.25, gamma=2.0, weight=1.0):
        super().__init__(weight=weight)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, raw_output, target):
        logits = raw_output["cls"]
        cls_target = target["cls"]
        prob = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, cls_target, reduction="none")
        p_t = prob * cls_target + (1.0 - prob) * (1.0 - cls_target)
        alpha_t = self.alpha * cls_target + (1.0 - self.alpha) * (1.0 - cls_target)
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * ce
        return loss.mean()
