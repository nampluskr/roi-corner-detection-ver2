# src/models/heads/heatmap_head.py: predicts four corner heatmap logits from decoded features

import torch.nn as nn


class HeatmapHead(nn.Module):
    """Projects a decoded spatial feature to four corner heatmap logits."""

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 4, kernel_size=1)

    def forward(self, decoded_feature):
        return self.conv(decoded_feature)
