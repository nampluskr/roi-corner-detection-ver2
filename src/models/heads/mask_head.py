# src/models/heads/mask_head.py: predicts binary mask logits from a decoded spatial feature

import torch.nn as nn


class MaskHead(nn.Module):
    """Projects a decoded spatial feature to single-channel binary mask logits."""

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, decoded_feature):
        return self.conv(decoded_feature)
