# src/models/backbones/torch_backbone.py: torchvision CNN backbone wrappers

import os
import torch
import torch.nn as nn
import torchvision.models as models

from src.models.backbones.base_backbone import BaseBackbone

BACKBONE_WEIGHTS = {
    "resnet18": "/mnt/d/backbones/resnet18-f37072fd.pth",
    "resnet34": "/mnt/d/backbones/resnet34-b627a593.pth",
    "resnet50": "/mnt/d/backbones/resnet50-0676ba61.pth",
    "efficientnet_b0": "/mnt/d/backbones/efficientnet_b0_rwightman-7f5810bc.pth",
    "vgg16": "/mnt/d/backbones/vgg16-397923af.pth",
    "vgg19": "/mnt/d/backbones/vgg19-dcbb9e9d.pth",
    "vit_b_16": "/mnt/d/backbones/vit_b_16-c867db91.pth",
    "swin_t": "/mnt/d/backbones/swin_t-704ceda3.pth",
}

BACKBONE_BUILDERS = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "efficientnet_b0": models.efficientnet_b0,
    "vgg16": models.vgg16,
    "vgg19": models.vgg19,
    "vit_b_16": models.vit_b_16,
    "swin_t": models.swin_t,
}

SUPPORTED_BACKBONES = tuple(BACKBONE_BUILDERS.keys())
RESNET_BACKBONES = ("resnet18", "resnet34", "resnet50")
EFFICIENTNET_BACKBONES = ("efficientnet_b0",)
VGG_BACKBONES = ("vgg16", "vgg19")
VIT_BACKBONES = ("vit_b_16",)
SWIN_BACKBONES = ("swin_t",)


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
            self.family = "efficientnet"
            self.features = net.features
            self.out_channels = net.classifier[-1].in_features
            self.stage_indices = (1, 2, 3, 5, 8)
            self.stage_channels = (16, 24, 40, 112, 1280)
            self.stage_strides = (2, 4, 8, 16, 32)
        elif backbone in SWIN_BACKBONES:
            self.family = "swin"
            self.stem = net.features
            self.norm = net.norm
            self.out_channels = net.head.in_features
            self.stage_indices = (1, 3, 5, 7)
            self.stage_channels = (96, 192, 384, 768)
            self.stage_strides = (4, 8, 16, 32)
        elif backbone in VIT_BACKBONES:
            self.family = "vit"
            self.conv_proj = net.conv_proj
            self.class_token = net.class_token
            self.encoder = net.encoder
            self.out_channels = net.hidden_dim
            self.patch_size = net.patch_size
        elif backbone in VGG_BACKBONES:
            self.family = "vgg"
            self.features = net.features
            self.out_channels = 512
            self.stage_channels = (64, 128, 256, 512, 512)
            self.stage_strides = (2, 4, 8, 16, 32)
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
        if self.family == "efficientnet":
            x = images
            stages = []
            for i, layer in enumerate(self.features):
                x = layer(x)
                if i in self.stage_indices:
                    stages.append(x)
            return {"final": stages[-1], "stages": stages}

        if self.family == "swin":
            x = images
            stages = []
            for i, layer in enumerate(self.stem):
                x = layer(x)
                if i in self.stage_indices:
                    stage = x
                    if i == self.stage_indices[-1]:
                        stage = self.norm(stage)
                    stages.append(stage.permute(0, 3, 1, 2).contiguous())
            return {"final": stages[-1], "stages": stages}

        if self.family == "vit":
            n = images.shape[0]
            patches = self.conv_proj(images).flatten(2).transpose(1, 2)
            cls = self.class_token.expand(n, -1, -1)
            tokens = self.encoder(torch.cat([cls, patches], dim=1))
            grid_h = images.shape[2] // self.patch_size
            grid_w = images.shape[3] // self.patch_size
            return {"cls": tokens[:, 0], "tokens": tokens[:, 1:], "grid_size": (grid_h, grid_w)}

        if self.family == "vgg":
            x = images
            stages = []
            for layer in self.features:
                x = layer(x)
                if isinstance(layer, nn.MaxPool2d):
                    stages.append(x)
            return {"final": stages[-1], "stages": stages}

        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        s1 = self.layer1(x)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        return {"final": s4, "stages": [s1, s2, s3, s4]}
