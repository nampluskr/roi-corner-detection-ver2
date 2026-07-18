# src/models/necks/multi_scale_neck.py: stage channel projection and top-down fusion stopping at grid_stride

import torch.nn as nn

from src.models.blocks.conv_block import ConvBlock
from src.models.blocks.deconv_block import DeconvBlock


class MultiScaleNeck(nn.Module):
    """Projects stages to a common channel width and fuses top-down, stopping at grid_stride resolution."""

    def __init__(self, stage_channels, stage_strides, grid_stride=16, out_channels=256, upsample="interpolate_conv"):
        super().__init__()
        if grid_stride not in stage_strides:
            raise ValueError("grid_stride %d not in stage_strides %s" % (grid_stride, stage_strides))
        self.grid_index = stage_strides.index(grid_stride)
        used_channels = stage_channels[self.grid_index:]

        self.laterals = nn.ModuleList([
            ConvBlock(c, out_channels, kernel_size=1, stride=1) for c in used_channels
        ])
        self.up_blocks = nn.ModuleList()
        self.fuse_blocks = nn.ModuleList()
        for _ in range(len(used_channels) - 1):
            self.up_blocks.append(DeconvBlock(out_channels, out_channels, mode=upsample))
            self.fuse_blocks.append(ConvBlock(out_channels, out_channels, kernel_size=3, stride=1))
        self.out_channels = out_channels

    def forward(self, stages):
        used = stages[self.grid_index:]
        laterals = [lateral(feat) for lateral, feat in zip(self.laterals, used)]
        x = laterals[-1]
        skips = list(reversed(laterals[:-1]))
        for up_block, fuse_block, skip in zip(self.up_blocks, self.fuse_blocks, skips):
            x = up_block(x)
            if x.shape[-2:] != skip.shape[-2:]:
                raise ValueError("neck feature shape %s does not match skip feature shape %s"
                                  % (tuple(x.shape[-2:]), tuple(skip.shape[-2:])))
            x = fuse_block(x + skip)
        return x
