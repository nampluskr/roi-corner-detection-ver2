# src/models/backbones/torchvision_backbone.py: torchvision CNN backbone wrappers

import os
import torch
import torchvision.models as models

from src.models.backbones.base_backbone import BaseBackbone

BACKBONE_WEIGHTS = {
    "resnet18": "/mnt/d/backbones/resnet18-f37072fd.pth",
    "resnet34": "/mnt/d/backbones/resnet34-b627a593.pth",
    "resnet50": "/mnt/d/backbones/resnet50-0676ba61.pth",
    "efficientnet_b0": "/mnt/d/backbones/efficientnet_b0_rwightman-7f5810bc.pth",
    "vgg16": "/mnt/d/backbones/vgg16-397923af.pth",
    "vgg16_bn": "/mnt/d/backbones/vgg16_bn-6c64b313.pth",
    "vgg19": "/mnt/d/backbones/vgg19-dcbb9e9d.pth",
    "vgg19_bn": "/mnt/d/backbones/vgg19_bn-c79401a0.pth",
}

BACKBONE_BUILDERS = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "efficientnet_b0": models.efficientnet_b0,
    "vgg16": models.vgg16,
    "vgg16_bn": models.vgg16_bn,
    "vgg19": models.vgg19,
    "vgg19_bn": models.vgg19_bn,
}

SUPPORTED_BACKBONES = tuple(BACKBONE_BUILDERS.keys())
RESNET_BACKBONES = ("resnet18", "resnet34", "resnet50")
EFFICIENTNET_BACKBONES = ("efficientnet_b0",)
VGG_BACKBONES = ("vgg16", "vgg16_bn", "vgg19", "vgg19_bn")


class TorchBackbone(BaseBackbone):
    """Torchvision CNN backbone returning final and per-stage feature maps."""

    def __init__(self, backbone="resnet50", pretrained=True):
        super().__init__()
        if backbone not in BACKBONE_BUILDERS:
            raise ValueError("Unknown torch backbone: %s. Supported: %s"
                             % (backbone, ", ".join(SUPPORTED_BACKBONES)))

        net = BACKBONE_BUILDERS[backbone](weights=None)
        if pretrained:
            self.load_local_weights(net, BACKBONE_WEIGHTS[backbone])

        self.backbone_name = backbone
        if backbone in RESNET_BACKBONES:
            self.family = "resnet"
            self.conv1 = net.conv1
            self.bn1 = net.bn1
            self.relu = net.relu
            self.maxpool = net.maxpool
            self.layer1 = net.layer1
            self.layer2 = net.layer2
            self.layer3 = net.layer3
            self.layer4 = net.layer4
            self.out_channels = net.fc.in_features
            self.stage_channels = self.resnet_stage_channels(backbone)
            self.stage_strides = (4, 8, 16, 32)
        elif backbone in EFFICIENTNET_BACKBONES:
            self.family = "features"
            self.features = net.features
            self.out_channels = net.classifier[-1].in_features
            self.stage_channels = (self.out_channels,)
            self.stage_strides = (32,)
        else:
            self.family = "features"
            self.features = net.features
            self.out_channels = 512
            self.stage_channels = (self.out_channels,)
            self.stage_strides = (32,)
        self.out_stride = 32

    def load_local_weights(self, net, path):
        if not os.path.exists(path):
            raise FileNotFoundError("Local torchvision weight not found: %s" % path)
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        net.load_state_dict(state_dict, strict=True)

    def resnet_stage_channels(self, backbone):
        if backbone in ("resnet18", "resnet34"):
            return (64, 128, 256, 512)
        return (256, 512, 1024, 2048)

    def forward(self, images):
        if self.family == "features":
            final = self.features(images)
            return {"final": final, "stages": [final]}

        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        s1 = self.layer1(x)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        return {"final": s4, "stages": [s1, s2, s3, s4]}


TorchvisionBackbone = TorchBackbone
