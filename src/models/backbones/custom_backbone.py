# src/models/backbones/custom_backbone.py: project baseline encoder with no pretrained weights

import torch.nn as nn

from src.models.backbones.base_backbone import BaseBackbone
from src.models.blocks.conv_block import ConvBlock

DEFAULT_STAGE_CHANNELS = (64, 128, 256, 512)


class CustomBackbone(BaseBackbone):
    """Stem plus four downsampling ConvBlock stages, reaching output stride 16."""

    def __init__(self, in_channels=3, stage_channels=DEFAULT_STAGE_CHANNELS):
        super().__init__()
        self.stem = ConvBlock(in_channels, stage_channels[0], kernel_size=3, stride=2)
        self.stage1 = ConvBlock(stage_channels[0], stage_channels[0], kernel_size=3, stride=1)
        self.stage2 = ConvBlock(stage_channels[0], stage_channels[1], kernel_size=3, stride=2)
        self.stage3 = ConvBlock(stage_channels[1], stage_channels[2], kernel_size=3, stride=2)
        self.stage4 = ConvBlock(stage_channels[2], stage_channels[3], kernel_size=3, stride=2)
        self.out_channels = stage_channels[3]
        self.out_stride = 16

    def forward(self, images):
        x = self.stem(images)
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        return {"final": s4, "stages": [s1, s2, s3, s4]}
