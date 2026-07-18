# src/models/blocks/conv_block.py: shared Conv2d + normalization + activation block

import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d followed by batch normalization and activation, shared by encoders, decoders and necks."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, activation=nn.ReLU):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = activation(inplace=True) if activation is not None else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
