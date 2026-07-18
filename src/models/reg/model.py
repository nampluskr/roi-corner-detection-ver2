# src/models/reg/model.py: selectable backbone + adapter + coordinate head for corner regression

from src.models.base.base_model import BaseModel
from src.models.backbones.custom_backbone import CustomBackbone
from src.models.backbones.torch_backbone import SUPPORTED_BACKBONES, VIT_BACKBONES, TorchBackbone
from src.models.backbones.timm_backbone import SUPPORTED_TIMM_BACKBONES, TIMM_VIT_BACKBONES, TimmBackbone
from src.models.adapters.cnn_adapter import CNNBackboneAdapter
from src.models.adapters.transformer_adapter import TransformerBackboneAdapter
from src.models.features import FeatureExtractor, FeatureSpec
from src.models.heads.coordinate_head import CoordGapHead, CoordSpatialHead


class RegModel(BaseModel):
    """Selectable backbone plus a matching adapter feeding a coordinate head for direct corner regression."""

    def __init__(self, in_channels=3, dropout=0.2, backbone="custom", head="coord_gap"):
        super().__init__()
        backbone = backbone or "custom"
        head = head or "coord_gap"
        if backbone == "custom":
            encoder = CustomBackbone(in_channels=in_channels)
            backbone_name = "custom"
        elif backbone in SUPPORTED_BACKBONES:
            encoder = TorchBackbone(backbone)
            backbone_name = backbone
        elif backbone in SUPPORTED_TIMM_BACKBONES:
            encoder = TimmBackbone(backbone)
            backbone_name = backbone
        else:
            supported = ("custom",) + SUPPORTED_BACKBONES + SUPPORTED_TIMM_BACKBONES
            raise ValueError("Unknown reg backbone: %s. Supported: %s"
                             % (backbone, ", ".join(supported)))

        is_vit = backbone in VIT_BACKBONES or backbone in TIMM_VIT_BACKBONES
        adapter_name = "vit" if is_vit else "cnn"
        if head == "coord_gap":
            if is_vit:
                adapter = TransformerBackboneAdapter(keep_spatial=False, keep_global=True)
            else:
                adapter = CNNBackboneAdapter(keep_spatial=False, keep_stages=False)
            spec = FeatureSpec(backbone_name, adapter_name, global_channels=encoder.out_channels)
            coordinate_head = CoordGapHead(spec.global_channels, dropout=dropout)
        elif head == "coord_spatial":
            if is_vit:
                adapter = TransformerBackboneAdapter(keep_spatial=True, keep_global=False)
            else:
                adapter = CNNBackboneAdapter(keep_spatial=True, keep_stages=False)
            spec = FeatureSpec(backbone_name, adapter_name,
                               global_channels=encoder.out_channels,
                               spatial_channels=encoder.out_channels)
            coordinate_head = CoordSpatialHead(spec.spatial_channels, dropout=dropout)
        else:
            raise ValueError("Unknown reg head: %s. Supported: coord_gap, coord_spatial" % head)

        self.head_name = head
        self.extractor = FeatureExtractor(encoder, adapter, spec)
        self.head = coordinate_head

    def forward(self, images):
        bundle = self.extractor(images)
        if self.head_name == "coord_gap":
            return self.head(bundle.global_feature)
        return self.head(bundle.spatial_feature)
