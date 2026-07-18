# src/models/heads/coordinate_head.py: predicts flattened corner coordinates from CNN features

import torch.nn as nn


class CoordGapHead(nn.Module):
    """Dropout followed by a linear projection from a global feature to 8 raw corner values."""

    def __init__(self, in_channels, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, 8)

    def forward(self, global_feature):
        return self.fc(self.dropout(global_feature))


class CoordSpatialHead(nn.Module):
    """Strided convolutions and pooling followed by a linear projection to 8 raw corner values."""

    def __init__(self, in_channels, dropout=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * 4 * 4, 8),
        )

    def forward(self, spatial_feature):
        return self.layers(spatial_feature)
