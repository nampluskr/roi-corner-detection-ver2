# src/models/heatmap/model.py: stage-based composable heatmap model for corner localization

from src.models.base.base_model import BaseModel
from src.models.backbones.custom_backbone import CustomBackbone
from src.models.backbones.torch_backbone import (
    EFFICIENTNET_BACKBONES,
    RESNET_BACKBONES,
    SWIN_BACKBONES,
    VGG_BACKBONES,
    TorchBackbone,
)
from src.models.backbones.timm_backbone import TIMM_CNN_BACKBONES, TimmBackbone
from src.models.adapters.cnn_adapter import CNNBackboneAdapter
from src.models.features import FeatureExtractor, FeatureSpec
from src.models.heads.heatmap_head import HeatmapHead
from src.models.seg.decoder import SegDecoder

TORCH_HEATMAP_BACKBONES = RESNET_BACKBONES + EFFICIENTNET_BACKBONES + SWIN_BACKBONES + VGG_BACKBONES
SUPPORTED_HEATMAP_BACKBONES = ("custom",) + TORCH_HEATMAP_BACKBONES + TIMM_CNN_BACKBONES


class HeatmapModel(BaseModel):
    """Stage-returning backbone plus a U-Net additive-skip decoder feeding a four-corner heatmap head."""

    def __init__(self, in_channels=3, backbone="custom", upsample="interpolate_conv"):
        super().__init__()
        backbone = backbone or "custom"
        if backbone == "custom":
            encoder = CustomBackbone(in_channels=in_channels)
        elif backbone in TORCH_HEATMAP_BACKBONES:
            encoder = TorchBackbone(backbone)
        elif backbone in TIMM_CNN_BACKBONES:
            encoder = TimmBackbone(backbone)
        else:
            raise ValueError("Unknown heatmap backbone: %s. Supported: %s"
                             % (backbone, ", ".join(SUPPORTED_HEATMAP_BACKBONES)))

        adapter = CNNBackboneAdapter(keep_spatial=False, keep_stages=True)
        spec = FeatureSpec(backbone, "cnn",
                           stage_channels=encoder.stage_channels,
                           stage_strides=encoder.stage_strides)
        spec.require("stages")

        self.extractor = FeatureExtractor(encoder, adapter, spec)
        self.decoder = SegDecoder(spec.stage_channels, upsample=upsample)
        self.head = HeatmapHead(self.decoder.out_channels)
        self.heatmap_stride = spec.stage_strides[0]

    def forward(self, images):
        bundle = self.extractor(images)
        decoded = self.decoder(bundle.stages)
        return self.head(decoded)
