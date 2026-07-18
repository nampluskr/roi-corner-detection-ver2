# src/models/adapters/cnn_adapter.py: adapts CNN backbone features into a FeatureBundle

import torch.nn as nn

from src.models.adapters.base_adapter import BaseBackboneAdapter
from src.models.features import FeatureBundle


class CNNBackboneAdapter(BaseBackboneAdapter):
    """Pools the final CNN feature map into FeatureBundle.global_feature and passes stages through."""

    def __init__(self, keep_spatial=True, keep_stages=True):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.keep_spatial = keep_spatial
        self.keep_stages = keep_stages

    def forward(self, native_features):
        final = native_features["final"]
        global_feature = self.pool(final).flatten(1)
        spatial_feature = final if self.keep_spatial else None
        stages = native_features.get("stages") if self.keep_stages else None
        return FeatureBundle(global_feature=global_feature, spatial_feature=spatial_feature, stages=stages)
