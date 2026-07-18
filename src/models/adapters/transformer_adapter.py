# src/models/adapters/transformer_adapter.py: adapts ViT token features into a FeatureBundle

from src.models.adapters.base_adapter import BaseBackboneAdapter
from src.models.features import FeatureBundle


class TransformerBackboneAdapter(BaseBackboneAdapter):
    """Reshapes ViT cls/token features into FeatureBundle.global_feature and spatial_feature."""

    def __init__(self, keep_spatial=True, keep_global=True):
        super().__init__()
        self.keep_spatial = keep_spatial
        self.keep_global = keep_global

    def forward(self, native_features):
        global_feature = native_features["cls"] if self.keep_global else None
        spatial_feature = None
        if self.keep_spatial:
            tokens = native_features["tokens"]
            grid_h, grid_w = native_features["grid_size"]
            n, l, c = tokens.shape
            spatial_feature = tokens.transpose(1, 2).reshape(n, c, grid_h, grid_w)
        return FeatureBundle(global_feature=global_feature, spatial_feature=spatial_feature, stages=None)
