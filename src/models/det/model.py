# src/models/det/model.py: selectable custom detection model and torchvision whole-model detection model

import os
import torch
import torchvision.models.detection as detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.ssd import SSDClassificationHead

from src.models.base.base_model import BaseModel
from src.models.backbones.custom_backbone import CustomBackbone
from src.models.backbones.torch_backbone import EFFICIENTNET_BACKBONES, RESNET_BACKBONES
from src.models.backbones.torch_backbone import SWIN_BACKBONES, VGG_BACKBONES, VIT_BACKBONES, TorchBackbone
from src.models.backbones.timm_backbone import TIMM_CNN_BACKBONES, TIMM_VIT_BACKBONES, TimmBackbone
from src.models.adapters.cnn_adapter import CNNBackboneAdapter
from src.models.features import FeatureExtractor, FeatureSpec
from src.models.necks.multi_scale_neck import MultiScaleNeck
from src.models.heads.detection_head import NUM_CORNER_CLASSES, DetectionHead

TORCH_DET_BACKBONES = RESNET_BACKBONES + EFFICIENTNET_BACKBONES + SWIN_BACKBONES + VGG_BACKBONES
SUPPORTED_DET_BACKBONES = ("custom",) + TORCH_DET_BACKBONES + TIMM_CNN_BACKBONES

TORCHDET_WEIGHTS = {
    "fasterrcnn_resnet50_fpn": "/mnt/d/backbones/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
    "retinanet_resnet50_fpn": "/mnt/d/backbones/retinanet_resnet50_fpn_coco-eeacb38b.pth",
    "ssd300_vgg16": "/mnt/d/backbones/ssd300_vgg16_coco-b556d3b4.pth",
}
TORCHDET_BUILDERS = {
    "fasterrcnn_resnet50_fpn": detection.fasterrcnn_resnet50_fpn,
    "retinanet_resnet50_fpn": detection.retinanet_resnet50_fpn,
    "ssd300_vgg16": detection.ssd300_vgg16,
}
TORCHDET_LABEL_OFFSET = {
    "fasterrcnn_resnet50_fpn": 1,
    "retinanet_resnet50_fpn": 0,
    "ssd300_vgg16": 1,
}
SUPPORTED_TORCHDET_MODELS = tuple(TORCHDET_WEIGHTS.keys())


class DetModel(BaseModel):
    """Stage-returning backbone plus a multi-scale neck feeding a per-cell detection head."""

    def __init__(self, in_channels=3, backbone="custom", neck_channels=256, grid_stride=16,
                 head="box", upsample="interpolate_conv"):
        super().__init__()
        backbone = backbone or "custom"
        if backbone == "custom":
            encoder = CustomBackbone(in_channels=in_channels)
        elif backbone in VIT_BACKBONES or backbone in TIMM_VIT_BACKBONES:
            raise ValueError("det backbone %s has no stages capability (ViT/DINOv2 family). Supported: %s"
                              % (backbone, ", ".join(SUPPORTED_DET_BACKBONES)))
        elif backbone in TORCH_DET_BACKBONES:
            encoder = TorchBackbone(backbone)
        elif backbone in TIMM_CNN_BACKBONES:
            encoder = TimmBackbone(backbone)
        else:
            raise ValueError("Unknown det backbone: %s. Supported: %s"
                              % (backbone, ", ".join(SUPPORTED_DET_BACKBONES)))

        adapter = CNNBackboneAdapter(keep_spatial=False, keep_stages=True)
        spec = FeatureSpec(backbone, "cnn",
                            stage_channels=encoder.stage_channels, stage_strides=encoder.stage_strides)
        spec.require("stages")

        self.extractor = FeatureExtractor(encoder, adapter, spec)
        self.neck = MultiScaleNeck(spec.stage_channels, spec.stage_strides,
                                    grid_stride=grid_stride, out_channels=neck_channels, upsample=upsample)
        self.head = DetectionHead(self.neck.out_channels, head=head)
        self.grid_stride = grid_stride

    def forward(self, images):
        bundle = self.extractor(images)
        feature = self.neck(bundle.stages)
        return self.head(feature)


class TorchDetModel(BaseModel):
    """Torchvision whole detection model adapted to 4 corner classes via classifier replacement."""

    def __init__(self, model="fasterrcnn_resnet50_fpn", pretrained=True):
        super().__init__()
        model = model or "fasterrcnn_resnet50_fpn"
        if model not in TORCHDET_BUILDERS:
            raise ValueError("Unknown torchdet model: %s. Supported: %s"
                             % (model, ", ".join(SUPPORTED_TORCHDET_MODELS)))

        self.model_name = model
        self.label_offset = TORCHDET_LABEL_OFFSET[model]
        num_classes = NUM_CORNER_CLASSES + self.label_offset
        self.net = self.build_model(model, num_classes, pretrained=pretrained)

    def build_model(self, model, num_classes, pretrained=True):
        builder = TORCHDET_BUILDERS[model]
        kwargs = {"weights": None, "weights_backbone": None}
        if model != "ssd300_vgg16":
            kwargs["min_size"] = 224
            kwargs["max_size"] = 224

        if pretrained:
            net = builder(num_classes=91, **kwargs)
            self.load_local_weights(net, TORCHDET_WEIGHTS[model])
            self.replace_classifier(net, num_classes)
            return net

        return builder(num_classes=num_classes, **kwargs)

    def load_local_weights(self, net, path):
        if not os.path.exists(path):
            raise FileNotFoundError("Local torchdet weight not found: %s" % path)
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        net.load_state_dict(state_dict, strict=True)

    def replace_classifier(self, net, num_classes):
        if self.model_name == "fasterrcnn_resnet50_fpn":
            in_features = net.roi_heads.box_predictor.cls_score.in_features
            net.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            return

        if self.model_name == "retinanet_resnet50_fpn":
            num_anchors = net.head.classification_head.num_anchors
            in_channels = net.backbone.out_channels
            net.head.classification_head = RetinaNetClassificationHead(
                in_channels, num_anchors, num_classes)
            return

        head = net.head.classification_head
        in_channels = [m.in_channels for m in head.module_list]
        num_anchors = [m.out_channels // 91 for m in head.module_list]
        net.head.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)

    def forward(self, images, targets=None):
        if self.training and targets is not None:
            return self.net(images, targets)
        return self.net(images)
