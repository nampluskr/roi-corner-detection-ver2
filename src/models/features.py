# src/models/features.py: FeatureBundle, FeatureSpec and FeatureExtractor composition primitives

import torch.nn as nn


class FeatureBundle:
    """Container for a backbone adapter's global, spatial and per-stage features."""

    def __init__(self, global_feature=None, spatial_feature=None, stages=None):
        self.global_feature = global_feature
        self.spatial_feature = spatial_feature
        self.stages = stages


class FeatureSpec:
    """Describes the channel, stride and capability metadata produced by a backbone/adapter pair."""

    def __init__(self, backbone_name, adapter_name, global_channels=None,
                 spatial_channels=None, spatial_stride=None, stage_channels=None, stage_strides=None):
        self.backbone_name = backbone_name
        self.adapter_name = adapter_name
        self.global_channels = global_channels
        self.spatial_channels = spatial_channels
        self.spatial_stride = spatial_stride
        self.stage_channels = stage_channels
        self.stage_strides = stage_strides

    @property
    def has_global(self):
        return self.global_channels is not None

    @property
    def has_spatial(self):
        return self.spatial_channels is not None

    @property
    def has_stages(self):
        return self.stage_channels is not None

    def require(self, capability):
        if not getattr(self, "has_%s" % capability):
            raise ValueError("FeatureSpec for backbone '%s' + adapter '%s' does not provide '%s'"
                             % (self.backbone_name, self.adapter_name, capability))


class FeatureExtractor(nn.Module):
    """Composes a backbone and adapter to turn images into a FeatureBundle under a fixed FeatureSpec."""

    def __init__(self, backbone, adapter, spec):
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter
        self.spec = spec

    def forward(self, images):
        native_features = self.backbone(images)
        return self.adapter(native_features)
