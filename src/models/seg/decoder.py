# src/models/seg/decoder.py: U-Net decoder fusing encoder stages with additive skip connections

import torch.nn as nn

from src.models.blocks.conv_block import ConvBlock
from src.models.blocks.deconv_block import DeconvBlock


class SegDecoder(nn.Module):
    """Upsamples the deepest encoder stage and adds shallower stages back in, low to high resolution."""

    def __init__(self, stage_channels, upsample="interpolate_conv"):
        super().__init__()
        channels = list(reversed(stage_channels))
        self.up_blocks = nn.ModuleList()
        self.fuse_blocks = nn.ModuleList()
        for in_channels, out_channels in zip(channels[:-1], channels[1:]):
            self.up_blocks.append(DeconvBlock(in_channels, out_channels, mode=upsample))
            self.fuse_blocks.append(ConvBlock(out_channels, out_channels, kernel_size=3, stride=1))
        self.out_channels = channels[-1]

    def forward(self, stages):
        x = stages[-1]
        skips = list(reversed(stages[:-1]))
        for up_block, fuse_block, skip in zip(self.up_blocks, self.fuse_blocks, skips):
            x = up_block(x)
            if x.shape[-2:] != skip.shape[-2:]:
                raise ValueError("decoder feature shape %s does not match skip feature shape %s"
                                 % (tuple(x.shape[-2:]), tuple(skip.shape[-2:])))
            x = fuse_block(x + skip)
        return x
