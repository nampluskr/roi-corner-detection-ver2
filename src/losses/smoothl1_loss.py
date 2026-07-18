# src/losses/smoothl1_loss.py: masked smooth L1 loss for the per-cell box/point regression map

import torch

from src.losses.base_loss import BaseLoss


class SmoothL1Loss(BaseLoss):
    """Smooth L1 loss on box/point regression, masked to positive cells and sigmoid-bounded offset channels."""

    def __init__(self, beta=1.0, weight=1.0):
        super().__init__(weight=weight)
        self.beta = beta

    def forward(self, raw_output, target):
        pred = raw_output["box"].clone()
        pred[:, 0:2] = torch.sigmoid(pred[:, 0:2])
        box_target = target["box"]
        pos_mask = target["pos_mask"]

        diff = (pred - box_target).abs() * pos_mask
        loss = torch.where(diff < self.beta, 0.5 * diff.pow(2) / self.beta, diff - 0.5 * self.beta)
        denom = pos_mask.sum().clamp(min=1.0) * pred.shape[1]
        return loss.sum() / denom
