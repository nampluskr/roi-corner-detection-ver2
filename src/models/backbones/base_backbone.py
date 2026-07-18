# src/models/backbones/base_backbone.py: base class for image encoders

import torch.nn as nn


class BaseBackbone(nn.Module):
    """Base class for an image encoder producing a native final feature and stage features."""

    def forward(self, images):
        raise NotImplementedError
