# src/models/adapters/base_adapter.py: base class converting native backbone features to a FeatureBundle

import torch.nn as nn


class BaseBackboneAdapter(nn.Module):
    """Base class converting a backbone's native feature dict into a FeatureBundle."""

    def forward(self, native_features):
        raise NotImplementedError
