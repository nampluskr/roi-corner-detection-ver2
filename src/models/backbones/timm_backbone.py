# src/models/backbones/timm_backbone.py: timm CNN/transformer backbone wrappers

import os
import timm
from safetensors.torch import load_file

from src.models.backbones.base_backbone import BaseBackbone

TIMM_BACKBONE_WEIGHTS = {
    "wide_resnet50_2.tv_in1k": "/mnt/d/backbones/wide_resnet50_2.tv_in1k/model.safetensors",
    "deit_base_distilled_patch16_224.fb_in1k": "/mnt/d/backbones/deit_base_distilled_patch16_224.fb_in1k/model.safetensors",
    "cait_s24_224.fb_dist_in1k": "/mnt/d/backbones/cait_s24_224.fb_dist_in1k/model.safetensors",
}

TIMM_CNN_BACKBONES = ("wide_resnet50_2.tv_in1k",)
TIMM_VIT_PREFIX_TOKENS = {
    "deit_base_distilled_patch16_224.fb_in1k": 2,
    "cait_s24_224.fb_dist_in1k": 1,
}
TIMM_VIT_BACKBONES = tuple(TIMM_VIT_PREFIX_TOKENS.keys())
SUPPORTED_TIMM_BACKBONES = TIMM_CNN_BACKBONES + TIMM_VIT_BACKBONES


class TimmBackbone(BaseBackbone):
    """timm model wrapper returning the same native CNN/ViT feature contract as TorchBackbone."""

    def __init__(self, backbone="wide_resnet50_2.tv_in1k", pretrained=True):
        super().__init__()
        if backbone not in SUPPORTED_TIMM_BACKBONES:
            raise ValueError("Unknown timm backbone: %s. Supported: %s"
                             % (backbone, ", ".join(SUPPORTED_TIMM_BACKBONES)))

        net = timm.create_model(backbone, pretrained=False)
        if pretrained:
            self.load_local_weights(net, TIMM_BACKBONE_WEIGHTS[backbone])
        net.reset_classifier(0)

        self.backbone_name = backbone
        self.net = net
        self.out_channels = net.num_features
        if backbone in TIMM_CNN_BACKBONES:
            self.family = "cnn"
            feature_info = net.feature_info[1:]
            self.stage_channels = tuple(info["num_chs"] for info in feature_info)
            self.stage_strides = tuple(info["reduction"] for info in feature_info)
        else:
            self.family = "vit"
            self.patch_size = net.patch_embed.patch_size[0]
            self.prefix_tokens = TIMM_VIT_PREFIX_TOKENS[backbone]
        self.out_stride = 32

    def load_local_weights(self, net, path):
        if not os.path.exists(path):
            raise FileNotFoundError("Local timm weight not found: %s" % path)
        state_dict = load_file(path)
        net.load_state_dict(state_dict, strict=True)

    def forward(self, images):
        if self.family == "cnn":
            stages = list(self.net.forward_intermediates(images, intermediates_only=True))[1:]
            return {"final": stages[-1], "stages": stages}

        tokens = self.net.forward_features(images)
        grid_h = images.shape[2] // self.patch_size
        grid_w = images.shape[3] // self.patch_size
        return {"cls": tokens[:, 0], "tokens": tokens[:, self.prefix_tokens:], "grid_size": (grid_h, grid_w)}
