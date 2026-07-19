# src/models/reg/model.py: custom backbone and pretrained backbone coordinate regression models

from src.models.base.base_model import BaseModel
from src.models.backbones.custom_backbone import CustomBackbone
from src.models.backbones.torch_backbone import SUPPORTED_BACKBONES, VIT_BACKBONES, TorchBackbone
from src.models.backbones.timm_backbone import SUPPORTED_TIMM_BACKBONES, TIMM_VIT_BACKBONES, TimmBackbone
from src.models.adapters.cnn_adapter import CNNBackboneAdapter
from src.models.adapters.transformer_adapter import TransformerBackboneAdapter
from src.models.features import FeatureExtractor, FeatureSpec
from src.models.heads.coordinate_head import CoordGapHead, CoordSpatialHead


def _build_extractor_and_head(encoder, backbone_name, is_vit, head, dropout):
    adapter_name = "vit" if is_vit else "cnn"
    if head == "gap":
        if is_vit:
            adapter = TransformerBackboneAdapter(keep_spatial=False, keep_global=True)
        else:
            adapter = CNNBackboneAdapter(keep_spatial=False, keep_stages=False)
        spec = FeatureSpec(backbone_name, adapter_name, global_channels=encoder.out_channels)
        coordinate_head = CoordGapHead(spec.global_channels, dropout=dropout)
    elif head == "spatial":
        if is_vit:
            adapter = TransformerBackboneAdapter(keep_spatial=True, keep_global=False)
        else:
            adapter = CNNBackboneAdapter(keep_spatial=True, keep_stages=False)
        spec = FeatureSpec(backbone_name, adapter_name,
                           global_channels=encoder.out_channels,
                           spatial_channels=encoder.out_channels)
        coordinate_head = CoordSpatialHead(spec.spatial_channels, dropout=dropout)
    else:
        raise ValueError("Unknown reg head: %s. Supported: gap, spatial" % head)
    return FeatureExtractor(encoder, adapter, spec), coordinate_head


class CustomRegModel(BaseModel):
    """CustomBackbone plus a matching adapter feeding a coordinate head for direct corner regression."""

    def __init__(self, in_channels=3, dropout=0.2, head="gap"):
        super().__init__()
        head = head or "gap"
        encoder = CustomBackbone(in_channels=in_channels)
        self.head_name = head
        self.extractor, self.head = _build_extractor_and_head(
            encoder, "custom", False, head, dropout)

    def forward(self, images):
        bundle = self.extractor(images)
        if self.head_name == "gap":
            return self.head(bundle.global_feature)
        return self.head(bundle.spatial_feature)


class TorchRegModel(BaseModel):
    """Pretrained torchvision or timm backbone plus a matching adapter feeding a coordinate head."""

    def __init__(self, backbone, dropout=0.2, head="gap"):
        super().__init__()
        head = head or "gap"
        if backbone in SUPPORTED_BACKBONES:
            encoder = TorchBackbone(backbone)
        elif backbone in SUPPORTED_TIMM_BACKBONES:
            encoder = TimmBackbone(backbone)
        else:
            supported = SUPPORTED_BACKBONES + SUPPORTED_TIMM_BACKBONES
            raise ValueError("Unknown reg backbone: %s. Supported: %s"
                             % (backbone, ", ".join(supported)))

        is_vit = backbone in VIT_BACKBONES or backbone in TIMM_VIT_BACKBONES
        self.head_name = head
        self.extractor, self.head = _build_extractor_and_head(
            encoder, backbone, is_vit, head, dropout)

    def forward(self, images):
        bundle = self.extractor(images)
        if self.head_name == "gap":
            return self.head(bundle.global_feature)
        return self.head(bundle.spatial_feature)
