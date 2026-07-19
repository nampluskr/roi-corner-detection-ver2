# src/losses/heatmap_mse_loss.py: mean squared error loss for sigmoid corner heatmaps

import torch
import torch.nn as nn

from src.losses.base_loss import BaseLoss


class HeatmapMSELoss(BaseLoss):
    """Mean squared error between sigmoid heatmap logits and Gaussian heatmap targets."""

    def __init__(self, weight=1.0):
        super().__init__(weight=weight)
        self.criterion = nn.MSELoss()

    def forward(self, raw_output, target):
        return self.criterion(torch.sigmoid(raw_output), target)
