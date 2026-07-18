# src/models/heads/detection_head.py: predicts per-cell corner classification and box/point regression maps

import torch.nn as nn

from src.models.blocks.conv_block import ConvBlock

NUM_CORNER_CLASSES = 4
BOX_CHANNELS = {"box": 4, "point": 2}


class DetectionHead(nn.Module):
    """Splits a shared trunk into a per-class classification map and a class-agnostic box/point regression map."""

    def __init__(self, in_channels, hidden_channels=256, head="box"):
        super().__init__()
        if head not in BOX_CHANNELS:
            raise ValueError("Unknown det head: %s. Supported: %s"
                              % (head, ", ".join(BOX_CHANNELS)))
        self.head = head
        self.trunk = ConvBlock(in_channels, hidden_channels, kernel_size=3, stride=1)
        self.cls_conv = nn.Conv2d(hidden_channels, NUM_CORNER_CLASSES, kernel_size=1)
        self.box_conv = nn.Conv2d(hidden_channels, BOX_CHANNELS[head], kernel_size=1)

    def forward(self, feature):
        x = self.trunk(feature)
        return {"cls": self.cls_conv(x), "box": self.box_conv(x)}
