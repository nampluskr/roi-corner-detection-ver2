# src/models/blocks/deconv_block.py: upsampling block for dense decoders

import torch.nn as nn
import torch.nn.functional as F

from src.models.blocks.conv_block import ConvBlock

DECONV_MODES = ("interpolate_conv", "transposed_conv")


class DeconvBlock(nn.Module):
    """Doubles spatial resolution via interpolation plus Conv2d, or via a transposed convolution."""

    def __init__(self, in_channels, out_channels, mode="interpolate_conv", scale_factor=2):
        super().__init__()
        if mode not in DECONV_MODES:
            raise ValueError("Unknown DeconvBlock mode: %s. Supported: %s"
                             % (mode, ", ".join(DECONV_MODES)))
        self.mode = mode
        self.scale_factor = scale_factor
        if mode == "interpolate_conv":
            self.conv = ConvBlock(in_channels, out_channels, kernel_size=3, stride=1)
        else:
            self.deconv = nn.ConvTranspose2d(in_channels, out_channels,
                                             kernel_size=scale_factor, stride=scale_factor)
            self.norm = nn.BatchNorm2d(out_channels)
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.mode == "interpolate_conv":
            x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
            return self.conv(x)
        return self.act(self.norm(self.deconv(x)))
