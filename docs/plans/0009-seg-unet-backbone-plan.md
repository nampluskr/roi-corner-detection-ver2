# seg U-Net skip backbone 비교 harness

다음 표는 이 plan의 상태와 적용 범위를 정리한다.

| 항목 | 값 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-18 |
| 적용 범위 | `docs/architecture/model-assembly.md`, `docs/references/backbones.md`, `experiments/configs.py`, `src/core/factory.py`, `src/losses/bce_loss.py`(신규), `src/losses/dice_loss.py`(신규), `src/models/backbones/custom_backbone.py`, `src/models/backbones/timm_backbone.py`, `src/models/backbones/torch_backbone.py`, `src/models/blocks/deconv_block.py`(신규), `src/models/heads/mask_head.py`(신규), `src/models/seg/__init__.py`(신규), `src/models/seg/decoder.py`(신규), `src/models/seg/model.py`(신규), `src/models/seg/postprocessor.py`(신규), `src/models/seg/preprocessor.py`(신규), `src/models/seg/wrapper.py`(신규) |
| 관련 문서 | [docs/architecture/model-assembly.md](../architecture/model-assembly.md), [docs/references/backbones.md](../references/backbones.md), [docs/plans/0003-reg-backbone-experiments-plan.md](0003-reg-backbone-experiments-plan.md), [docs/plans/0008-reg-timm-backbone-plan.md](0008-reg-timm-backbone-plan.md) |

## 1. 목적과 배경

canonical 문서 4.3, 5, 5.2, 6.1, 6.2절은 `CustomSegModel`을 `backbone + adapter + SegDecoder +
MaskHead` 조립으로, U-Net additive skip decoder를 `decoder: {name: unet, upsample:
interpolate_conv, skip_connection: add}` base config와 5.5절의 "기본 후보"로 이미 정의하고 있다
([model-assembly.md:383-392](../architecture/model-assembly.md#L383-L392),
[model-assembly.md:450-457](../architecture/model-assembly.md#L450-L457)). 그러나 `src/` 안에는
`seg` 관련 코드가 전혀 없다(전체 검색으로 확인).

backbone별 `forward()` 코드를 직접 읽고 실행해 확인한 결과, 현재는 `custom`, `resnet18`,
`resnet34`, `resnet50`만 진짜 다단계 `stages` 리스트를 반환하고, `vgg16`, `vgg19`,
`wide_resnet50_2.tv_in1k`는 architecture상 다단계 추출이 가능한데도 wrapper 코드가 마지막 feature
하나만 `stages=[final]`로 반환한다. 이번 plan은 두 가지를 함께 수행한다.

1. `vgg16`, `vgg19`, `wide_resnet50_2.tv_in1k`의 wrapper 코드를 확장해 진짜 다단계 `stages`를
   반환하도록 만든다.
2. `custom`, `resnet18`, `resnet34`, `resnet50`과 함께 이 세 backbone도 U-Net additive skip
   decoder 기반 `seg` 모델로 연결해, 초기 7개 backbone 비교 harness를 완성한다.
3. 후속 반영으로 `efficientnet_b0`, `swin_t`의 stage 추출을 추가해 총 9개 backbone 비교 harness로
   확장한다.

직접 검증한 backbone별 `stages` 반환 현황과 이번 plan 적용 후의 목표는 다음과 같다.

| backbone | 현재 `stages` 반환 | 이번 plan 적용 후 |
|---|---|---|
| `custom` | 다단계(4개 원소) | 변경 없음 |
| `resnet18`, `resnet34`, `resnet50` | 다단계(4개 원소) | 변경 없음 |
| `vgg16`, `vgg19` | 단일 원소 | 다단계(5개 원소)로 확장 |
| `wide_resnet50_2.tv_in1k`(timm) | 단일 원소 | 다단계(4개 원소)로 확장 |
| `efficientnet_b0` | 단일 원소 | 후속 반영으로 다단계(5개 원소)로 확장 |
| `swin_t` | 단일 원소 | 후속 반영으로 다단계(4개 원소)로 확장 |
| `vit_b_16`, timm ViT 계열 | `stages` 없음 | 변경 없음(구조상 대상 아님) |

`vgg16`/`vgg19`는 torchvision `features` Sequential 안에서 각 `MaxPool2d` 직후 output을 모으면
5단계(stride 2, 4, 8, 16, 32, 채널 64, 128, 256, 512, 512)를 얻을 수 있음을 `torch`로 직접 shape를
찍어 확인했다. `wide_resnet50_2.tv_in1k`는 timm의 `forward_intermediates(images,
intermediates_only=True)`가 stem 이후 5단계(stride 2, 4, 8, 16, 32)를 반환하며, `net.feature_info`
로 채널과 stride를 코드로 조회할 수 있음을 timm 1.0.22에서 직접 확인했다. 후속 검증에서는
`efficientnet_b0`가 torchvision `features`에서 stride 2, 4, 8, 16, 32의 stage를 반환하고,
`swin_t`가 torchvision `features`에서 stride 4, 8, 16, 32의 stage를 반환함을 확인했다.

기본값으로 `unet` + `skip_connection=add`를 쓰는 것은 canonical 12.3절의 아직 미확정인
"segmentation 기본 decoder를 plain과 U-Net additive skip 중에서 선택한다"는 결정과 충돌하지 않는다
([model-assembly.md:750-758](../architecture/model-assembly.md#L750-L758)). 12.3절이 정하지 않은
것은 measured benchmark 이후의 "최종 default"이고, 이번 plan은 backbone 비교 harness를 만들기 위해
구체적인 decoder 하나가 필요할 뿐이다. `plain` 대 `unet+add` 비교 자체(11.4절 ablation 2단계)는 이
plan의 범위가 아니다.

canonical 6.1/6.2절의 backbone family 호환표에는 현재 `VGG`가 없다(`CustomBackbone`, `ResNet`,
`MobileNet/EfficientNet`, `ViT/DINOv2`, `Swin`만 있음). `wide_resnet50_2.tv_in1k`는 architecture상
ResNet 계열이므로 기존 `ResNet` row에 이미 포함된 것으로 본다. `CLAUDE.md` 6절 규칙에 따라 요구사항이
바뀌면 코드보다 canonical 문서를 먼저 수정해야 하므로, 이번 plan은 6.1/6.2절에 `VGG` row를 추가하는
작업을 3.2절에 포함한다.

## 2. 범위

이번 plan에 포함하는 항목은 다음과 같다.

- `docs/architecture/model-assembly.md` 6.1, 6.2절 backbone family 호환표에 `VGG` row를
  추가한다.
- `src/models/backbones/torch_backbone.py`에 `VGG_BACKBONES`, `EFFICIENTNET_BACKBONES`,
  `SWIN_BACKBONES` 전용 다단계 `stages` 추출 forward 로직을 추가하고, 더 이상 도달하지 않는
  catch-all `else` 분기를 제거한다.
- `src/models/backbones/timm_backbone.py`의 `family="cnn"` 분기를 `net.feature_info`와
  `forward_intermediates`를 사용해 다단계 `stages`를 반환하도록 확장한다.
- `docs/references/backbones.md`의 `resnet34`, `resnet50`, `efficientnet_b0`, `swin_t`,
  `vgg16`, `vgg19`, `wide_resnet50_2.tv_in1k` row "적용 방법"/"적용 방법과 제약" 열에 `seg`
  U-Net additive skip backbone 연결 문구를 추가한다(`resnet18`은 이미 "seg 기준 backbone" 문구가
  있어 내용만 확인한다).
- `experiments/configs.py`에 `custom`, `resnet18`, `resnet34`, `resnet50`, `efficientnet_b0`,
  `swin_t`, `vgg16`, `vgg19`, `wide_resnet50_2.tv_in1k` 각각의 `head="mask"` seg config 9개를
  추가한다.
- `src/core/factory.py::get_wrapper`에 `seg` 분기를 추가해 `SegWrapper`를 연결한다.
- `src/losses/bce_loss.py`, `src/losses/dice_loss.py`를 신규 작성한다. 둘 다 `BaseLoss`를 상속하고
  `forward(raw_output, target)` contract를 따른다.
- `src/models/backbones/custom_backbone.py`에 `self.stage_channels`, `self.stage_strides` 속성을
  추가한다(`TorchBackbone`은 이미 있으나 `CustomBackbone`에는 빠져 있다).
- `src/models/blocks/deconv_block.py`를 신규 작성한다. `DeconvBlock`은 `interpolate_conv`(기본)와
  `transposed_conv` 두 upsample mode를 모두 구현해 canonical 3.3절 contract를 그대로 채운다
  ([model-assembly.md:243-257](../architecture/model-assembly.md#L243-L257)).
- `src/models/heads/mask_head.py`를 신규 작성한다. `MaskHead`는 decoded feature를 1x1 `Conv2d`로
  단일 채널 mask logit으로 projection만 한다.
- `src/models/seg/decoder.py`를 신규 작성한다. `SegDecoder`는 U-Net additive skip만 구현하며,
  `bundle.stages`를 낮은 해상도에서 높은 해상도 순으로 `DeconvBlock` + element-wise add +
  `ConvBlock`로 fusion한다. stage 개수가 4개(ResNet 계열)든 5개(VGG, wide_resnet50_2)든 동일하게
  동작한다. decoder feature와 skip feature의 공간 크기가 다르면 silent crop 없이 `ValueError`를
  발생시킨다(3.3절 요구사항).
- `src/models/seg/model.py`를 신규 작성한다. `SegModel`은 backbone을 `("custom", "resnet18",
  "resnet34", "resnet50", "efficientnet_b0", "swin_t", "vgg16", "vgg19",
  "wide_resnet50_2.tv_in1k")`로 제한하고,
  `FeatureSpec`에 `stage_channels`/`stage_strides`를 채운 뒤 `spec.require("stages")`를 호출한다.
  `CNNBackboneAdapter(keep_spatial=False, keep_stages=True)` -> `SegDecoder` -> `MaskHead` 순으로
  조립하고 `self.mask_stride`를 노출한다.
- `src/models/seg/postprocessor.py`를 신규 작성한다. `SegPostprocessor`는 sigmoid와 threshold
  적용 후 현재 미사용 상태인 `src/utils/geometry.py::mask_to_corners`를 sample별로 호출해
  `(N, 4, 2)` corner를 만든다.
- `src/models/seg/preprocessor.py`를 신규 작성한다. `SegPreprocessor`는 `(N, 4, 2)` 정규화 corner를
  `PIL.ImageDraw.polygon`으로 `(N, 1, mask_size, mask_size)` binary mask로 rasterize한다. Pillow는
  torchvision의 기존 transitive dependency이므로 신규 의존성을 추가하지 않는다.
- `src/models/seg/wrapper.py`를 신규 작성한다. `SegWrapper`는 `RegWrapper`와 동일한 2단 `AdamW`
  learning rate와 `ReduceLROnPlateau`, `losses={"bce": BCELoss(), "dice": DiceLoss()}`,
  `metrics={"iou": PolygonIoU()}` 구성을 따른다.
- 빈 `src/models/seg/__init__.py`를 추가한다.

이번 plan에서 제외하는 항목은 다음과 같다.

- `plain`, `unet` + `concat`, `fpn` decoder variant. canonical 11.4절 ablation 2, 3, 4단계에서
  다루는 별도 비교 대상이며, 이번 plan은 U-Net additive skip 하나만 구현한다.
- `transposed_conv`를 실제 기본값이나 `experiments/configs.py` config로 사용하는 것.
  `DeconvBlock`에는 3.3절 contract대로 두 mode를 모두 구현하지만 이번 plan의 smoke test 밖에서는
  사용하지 않는다(11.4절 ablation 4단계 대상).
- `plain` 대 `unet+add` 최종 기본값 결정(12.3절). 이번 plan은 harness를 만들기 위해 decoder 하나를
  고정할 뿐이고, 실제 기본값 결정은 measured benchmark 결과가 필요하다.
- decoder 출력을 encoder 최저 stage 해상도보다 더 확대해 입력 원본 해상도까지 복원하는 것. mask는
  `mask_stride` 해상도로 유지한다(3.1절 결정표 참고). 더 정밀한 subpixel 정밀도(F1.4절 F5 제약)는
  이후 refinement 카테고리 작업으로 남긴다.
- `scripts/config.py`의 전역 `DEFAULTS["head"]="coord_gap"`이나 CLI 연결 로직 변경. `SegWrapper`는
  `get_wrapper_kwargs`가 항상 `--head`를 전달하는 것과 호환되도록 `head` kwarg를 받되 사용하지
  않는다(seg에는 head architecture가 `mask` 하나뿐이다). 검증 명령에서는 `--head mask`를 명시적으로
  전달한다. 이는 이번 plan이 만든 결함이 아니라 알려진 CLI 편의성 gap이다.

## 3. 구현 계획

### 3.1. backbone별 mask 해상도 결정표

`image_size=224` 기준으로 decoder는 각 backbone의 최저 stage 해상도(`stage_strides[0]`)까지만
복원한다. `SegPreprocessor`와 `SegModel.mask_stride`는 이 표를 따른다.

| backbone | `stage_strides` | `mask_stride` | mask 해상도(224 입력) |
|---|---|---:|---|
| `custom` | `(2, 4, 8, 16)` | 2 | 112x112 |
| `efficientnet_b0` | `(2, 4, 8, 16, 32)` | 2 | 112x112 |
| `vgg16`, `vgg19` | `(2, 4, 8, 16, 32)` | 2 | 112x112 |
| `resnet18`, `resnet34`, `resnet50` | `(4, 8, 16, 32)` | 4 | 56x56 |
| `swin_t` | `(4, 8, 16, 32)` | 4 | 56x56 |
| `wide_resnet50_2.tv_in1k` | `(4, 8, 16, 32)` | 4 | 56x56 |

### 3.2. `docs/architecture/model-assembly.md` 6.1, 6.2절 갱신

6.1절 표의 `ResNet` row 다음에 `VGG` row를 추가한다.

```text
| VGG | `global`, `spatial`, `stages` | coordinate와 multi-stage dense decoder |
```

6.2절 표의 `ResNet` row 다음에 `VGG` row를 추가한다.

```text
| VGG | 지원 | 지원 | 지원 | 지원 |
```

`wide_resnet50_2.tv_in1k`는 architecture상 ResNet 계열이므로 기존 `ResNet` row로 이미 커버된다고
보고 별도 row를 추가하지 않는다.

### 3.3. `src/models/backbones/torch_backbone.py` 확장(VGG 다단계 stage)

`import torch` 다음 줄에 `import torch.nn as nn`을 추가한다. `__init__`의 `VIT_BACKBONES` 분기
다음, 기존 catch-all `else` 분기를 아래 `elif backbone in VGG_BACKBONES:` 분기로 교체한다.
`VGG_BACKBONES`가 `SUPPORTED_BACKBONES`의 나머지 항목을 모두 명시적으로 덮게 되므로 catch-all
`else`는 더 이상 도달하지 않아 제거한다.

```python
        elif backbone in VGG_BACKBONES:
            self.family = "vgg"
            self.features = net.features
            self.out_channels = 512
            self.stage_channels = (64, 128, 256, 512, 512)
            self.stage_strides = (2, 4, 8, 16, 32)
        self.out_stride = 32
```

`forward`에는 `family == "swin"` 분기 다음에 아래 분기를 추가한다. `net.features` 안의 각
`MaxPool2d` 직후 output을 모으면 stride 2, 4, 8, 16, 32의 5단계가 나온다(직접 shape 확인 완료).

```python
        if self.family == "vgg":
            x = images
            stages = []
            for layer in self.features:
                x = layer(x)
                if isinstance(layer, nn.MaxPool2d):
                    stages.append(x)
            return {"final": stages[-1], "stages": stages}
```

### 3.4. `src/models/backbones/timm_backbone.py` 확장(wide_resnet50_2 다단계 stage)

`__init__`의 `family == "cnn"` 분기를 아래로 교체한다. `net.feature_info`는 timm이 model 생성
시점부터 제공하는 stage 메타데이터 리스트이며, index 0은 stem 직후(`stride=2`) feature다. 이
project는 `TorchBackbone`의 ResNet family와 동일하게 stem을 제외한 4단계(`stride 4, 8, 16, 32`)를
사용하므로 index 0을 버린다.

```python
        if backbone in TIMM_CNN_BACKBONES:
            self.family = "cnn"
            feature_info = net.feature_info[1:]
            self.stage_channels = tuple(info["num_chs"] for info in feature_info)
            self.stage_strides = tuple(info["reduction"] for info in feature_info)
```

`forward`의 `family == "cnn"` 분기를 아래로 교체한다. `forward_intermediates(images,
intermediates_only=True)`는 timm 1.0.22에서 `wide_resnet50_2.tv_in1k`에 대해 stride 2, 4, 8, 16,
32의 5개 feature map을 반환함을 직접 확인했다.

```python
        if self.family == "cnn":
            stages = list(self.net.forward_intermediates(images, intermediates_only=True))[1:]
            return {"final": stages[-1], "stages": stages}
```

### 3.5. `src/models/blocks/deconv_block.py`(신규)

```python
# src/models/blocks/deconv_block.py: upsampling block for dense decoders

import torch.nn as nn
import torch.nn.functional as F

from src.models.blocks.conv_block import ConvBlock

DECONV_MODES = ("interpolate_conv", "transposed_conv")


class DeconvBlock(nn.Module):
    """Doubles spatial resolution via interpolation plus Conv2d, or via a transposed convolution."""

    def __init__(self, in_channels, out_channels, mode="interpolate_conv", scale_factor=2):
        super().__init__()
        if mode not in DECONV_MODES:
            raise ValueError("Unknown DeconvBlock mode: %s. Supported: %s"
                             % (mode, ", ".join(DECONV_MODES)))
        self.mode = mode
        self.scale_factor = scale_factor
        if mode == "interpolate_conv":
            self.conv = ConvBlock(in_channels, out_channels, kernel_size=3, stride=1)
        else:
            self.deconv = nn.ConvTranspose2d(in_channels, out_channels,
                                             kernel_size=scale_factor, stride=scale_factor)
            self.norm = nn.BatchNorm2d(out_channels)
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.mode == "interpolate_conv":
            x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
            return self.conv(x)
        return self.act(self.norm(self.deconv(x)))
```

### 3.6. `src/models/heads/mask_head.py`(신규)

```python
# src/models/heads/mask_head.py: predicts binary mask logits from a decoded spatial feature

import torch.nn as nn


class MaskHead(nn.Module):
    """Projects a decoded spatial feature to single-channel binary mask logits."""

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, decoded_feature):
        return self.conv(decoded_feature)
```

### 3.7. `src/models/seg/decoder.py`(신규)

```python
# src/models/seg/decoder.py: U-Net decoder fusing encoder stages with additive skip connections

import torch.nn as nn

from src.models.blocks.conv_block import ConvBlock
from src.models.blocks.deconv_block import DeconvBlock


class SegDecoder(nn.Module):
    """Upsamples the deepest encoder stage and adds shallower stages back in, low to high resolution."""

    def __init__(self, stage_channels, upsample="interpolate_conv"):
        super().__init__()
        channels = list(reversed(stage_channels))
        self.up_blocks = nn.ModuleList()
        self.fuse_blocks = nn.ModuleList()
        for in_channels, out_channels in zip(channels[:-1], channels[1:]):
            self.up_blocks.append(DeconvBlock(in_channels, out_channels, mode=upsample))
            self.fuse_blocks.append(ConvBlock(out_channels, out_channels, kernel_size=3, stride=1))
        self.out_channels = channels[-1]

    def forward(self, stages):
        x = stages[-1]
        skips = list(reversed(stages[:-1]))
        for up_block, fuse_block, skip in zip(self.up_blocks, self.fuse_blocks, skips):
            x = up_block(x)
            if x.shape[-2:] != skip.shape[-2:]:
                raise ValueError("decoder feature shape %s does not match skip feature shape %s"
                                 % (tuple(x.shape[-2:]), tuple(skip.shape[-2:])))
            x = fuse_block(x + skip)
        return x
```

### 3.8. `src/models/seg/model.py`(신규)

```python
# src/models/seg/model.py: selectable stage-returning backbone + U-Net decoder + mask head

from src.models.base.base_model import BaseModel
from src.models.backbones.custom_backbone import CustomBackbone
from src.models.backbones.torch_backbone import RESNET_BACKBONES, VGG_BACKBONES, TorchBackbone
from src.models.backbones.timm_backbone import TIMM_CNN_BACKBONES, TimmBackbone
from src.models.adapters.cnn_adapter import CNNBackboneAdapter
from src.models.features import FeatureExtractor, FeatureSpec
from src.models.seg.decoder import SegDecoder
from src.models.heads.mask_head import MaskHead

SUPPORTED_SEG_BACKBONES = ("custom",) + RESNET_BACKBONES + VGG_BACKBONES + TIMM_CNN_BACKBONES


class SegModel(BaseModel):
    """Stage-returning backbone plus a U-Net additive-skip decoder feeding a binary mask head."""

    def __init__(self, in_channels=3, backbone="custom", upsample="interpolate_conv"):
        super().__init__()
        backbone = backbone or "custom"
        if backbone == "custom":
            encoder = CustomBackbone(in_channels=in_channels)
        elif backbone in RESNET_BACKBONES or backbone in VGG_BACKBONES:
            encoder = TorchBackbone(backbone)
        elif backbone in TIMM_CNN_BACKBONES:
            encoder = TimmBackbone(backbone)
        else:
            raise ValueError("Unknown seg backbone: %s. Supported: %s"
                             % (backbone, ", ".join(SUPPORTED_SEG_BACKBONES)))

        adapter = CNNBackboneAdapter(keep_spatial=False, keep_stages=True)
        spec = FeatureSpec(backbone, "cnn",
                           stage_channels=encoder.stage_channels, stage_strides=encoder.stage_strides)
        spec.require("stages")

        self.extractor = FeatureExtractor(encoder, adapter, spec)
        self.decoder = SegDecoder(spec.stage_channels, upsample=upsample)
        self.head = MaskHead(self.decoder.out_channels)
        self.mask_stride = spec.stage_strides[0]

    def forward(self, images):
        bundle = self.extractor(images)
        decoded = self.decoder(bundle.stages)
        return self.head(decoded)
```

`src/models/backbones/custom_backbone.py`의 `__init__`에는 다음 두 줄을 `self.out_stride = 16` 다음에
추가한다.

```python
self.stage_channels = stage_channels
self.stage_strides = (2, 4, 8, 16)
```

### 3.9. `src/models/seg/preprocessor.py`(신규)

```python
# src/models/seg/preprocessor.py: rasterize standard corners into a seg mask target

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.models.base.base_preprocessor import BasePreprocessor


class SegPreprocessor(BasePreprocessor):
    """Rasterizes (N, 4, 2) normalized corners into a (N, 1, mask_size, mask_size) binary mask."""

    def __init__(self, mask_size):
        self.mask_size = mask_size

    def __call__(self, corners):
        device = corners.device
        corners = corners.detach().cpu().numpy()
        masks = np.zeros((corners.shape[0], 1, self.mask_size, self.mask_size), dtype=np.float32)
        for i, quad in enumerate(corners):
            points = [(float(x) * self.mask_size, float(y) * self.mask_size) for x, y in quad]
            image = Image.new("L", (self.mask_size, self.mask_size), 0)
            ImageDraw.Draw(image).polygon(points, outline=1, fill=1)
            masks[i, 0] = np.array(image, dtype=np.float32)
        return torch.from_numpy(masks).to(device)
```

### 3.10. `src/models/seg/postprocessor.py`(신규)

```python
# src/models/seg/postprocessor.py: convert raw seg mask logits into standard corners

import numpy as np
import torch

from src.models.base.base_postprocessor import BasePostprocessor
from src.utils.geometry import mask_to_corners


class SegPostprocessor(BasePostprocessor):
    """Thresholds mask logits and extracts (N, 4, 2) corners via extreme points on the mask."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, raw_output):
        probs = torch.sigmoid(raw_output)
        masks = (probs > self.threshold).squeeze(1).detach().cpu().numpy()
        corners = np.stack([mask_to_corners(mask) for mask in masks])
        return torch.from_numpy(corners)
```

### 3.11. `src/losses/bce_loss.py`(신규), `src/losses/dice_loss.py`(신규)

```python
# src/losses/bce_loss.py: binary cross-entropy loss for mask logits

import torch.nn as nn

from src.losses.base_loss import BaseLoss


class BCELoss(BaseLoss):
    """Binary cross-entropy on raw mask logits against a binary mask target."""

    def __init__(self, weight=1.0):
        super().__init__(weight=weight)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, raw_output, target):
        return self.criterion(raw_output, target)
```

```python
# src/losses/dice_loss.py: soft Dice loss for mask logits

import torch

from src.losses.base_loss import BaseLoss


class DiceLoss(BaseLoss):
    """Soft Dice loss between sigmoid mask probabilities and a binary mask target."""

    def __init__(self, smooth=1.0, weight=1.0):
        super().__init__(weight=weight)
        self.smooth = smooth

    def forward(self, raw_output, target):
        probs = torch.sigmoid(raw_output).reshape(raw_output.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
        intersection = (probs * target).sum(dim=1)
        union = probs.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return (1.0 - dice).mean()
```

두 loss 모두 `BaseWrapper.compute_losses`가 전달하는 같은 `(raw_output, target)` 쌍을 받으며,
`BaseWrapper.train_step`이 `weight` 속성으로 자동 합산하므로 `SegWrapper`에서 별도 합산 로직이
필요하지 않다.

### 3.12. `src/models/seg/wrapper.py`(신규)

```python
# src/models/seg/wrapper.py: composes SegModel/SegPreprocessor/SegPostprocessor and BCE+Dice loss

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.base.base_wrapper import BaseWrapper
from src.models.seg.model import SegModel
from src.models.seg.preprocessor import SegPreprocessor
from src.models.seg.postprocessor import SegPostprocessor
from src.losses.bce_loss import BCELoss
from src.losses.dice_loss import DiceLoss
from src.metrics.polygon_iou import PolygonIoU


class SegWrapper(BaseWrapper):
    """Wraps SegModel training/evaluation/inference behind the shared Trainer/Evaluator/Predictor interface."""

    def __init__(self, in_channels=3, backbone="custom", head="mask", image_size=224,
                 optimizer=None, scheduler=None, preprocessor=None, postprocessor=None,
                 losses=None, metrics=None, device=None):
        # head kwarg accepted for CLI compatibility with get_wrapper_kwargs; seg has one head type
        model = SegModel(in_channels=in_channels, backbone=backbone)
        preprocessor = preprocessor or SegPreprocessor(image_size // model.mask_stride)
        postprocessor = postprocessor or SegPostprocessor()
        super().__init__(model, preprocessor, postprocessor, optimizer=optimizer,
                         scheduler=scheduler, losses=losses, metrics=metrics, device=device)
        backbone_ids = {id(p) for p in self.model.extractor.parameters()}
        head_params = [p for p in self.model.parameters() if id(p) not in backbone_ids]
        self.set_optimizer(self.optimizer or AdamW([
            {"params": self.model.extractor.parameters(), "lr": 1e-5},
            {"params": head_params, "lr": 1e-4},
        ]))
        self.set_scheduler(self.scheduler or ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2,
            threshold=1e-4, threshold_mode="abs", min_lr=1e-7))
        self.set_losses(self.losses or {"bce": BCELoss(), "dice": DiceLoss()})
        self.set_metrics(self.metrics or {"iou": PolygonIoU()})
```

### 3.13. `src/core/factory.py` 연결

`get_wrapper`에 다음 분기를 `reg` 분기 다음, `raise NotImplementedError` 앞에 추가한다.

```python
if method == "seg":
    from src.models.seg.wrapper import SegWrapper
    return SegWrapper(device=device, **kwargs)
```

### 3.14. `experiments/configs.py` 확장

```python
{"method": "seg", "batch_size": 4, "max_epochs": 5, "backbone": "custom", "head": "mask"},
{"method": "seg", "batch_size": 4, "max_epochs": 5, "backbone": "resnet18", "head": "mask"},
{"method": "seg", "batch_size": 4, "max_epochs": 5, "backbone": "resnet34", "head": "mask"},
{"method": "seg", "batch_size": 4, "max_epochs": 5, "backbone": "resnet50", "head": "mask"},
{"method": "seg", "batch_size": 4, "max_epochs": 5, "backbone": "efficientnet_b0", "head": "mask"},
{"method": "seg", "batch_size": 4, "max_epochs": 5, "backbone": "swin_t", "head": "mask"},
{"method": "seg", "batch_size": 4, "max_epochs": 5, "backbone": "vgg16", "head": "mask"},
{"method": "seg", "batch_size": 4, "max_epochs": 5, "backbone": "vgg19", "head": "mask"},
{"method": "seg", "batch_size": 4, "max_epochs": 5, "backbone": "wide_resnet50_2.tv_in1k", "head": "mask"},
```

plan 0008과 동일하게, `CONFIGS`에 위 9개 항목을 새로 추가하는 작업이며 기존 reg config의 로컬 주석
처리 상태는 건드리지 않는다.

### 3.15. `docs/references/backbones.md` 갱신

2절 표(`적용 방법` 열)의 `resnet34`, `resnet50`, `efficientnet_b0`, `swin_t` row에 `seg` U-Net
additive skip backbone으로 연결 완료되었다는 문구를 추가한다. `resnet18` row는 이미 "`reg`,
`heatmap`, `seg` 기준 backbone" 문구가 있으므로 내용만 확인한다.

3.1절 표(`적용 방법과 제약` 열)의 `vgg16`, `vgg19`, `wide_resnet50_2.tv_in1k` row에 `seg` U-Net
additive skip backbone 연결 완료 문구를 추가한다. `vgg16`/`vgg19`는 기존 "legacy feature baseline,
F6에 불리" 문구를 유지하면서 seg 연결 사실을 덧붙이고, `wide_resnet50_2.tv_in1k`는 기존 "`reg` timm
CNN backbone으로 연결 완료, `TimmBackbone` `family="cnn"`" 문구 뒤에 다단계 `stages` 지원과 `seg`
연결 사실을 덧붙인다.

## 4. 완료 기준

이 plan은 다음 조건을 만족하면 `Done`으로 본다.

- `docs/architecture/model-assembly.md` 6.1, 6.2절에 `VGG` row가 추가된다.
- `TorchBackbone("vgg16")`, `TorchBackbone("vgg19")`의 `forward()`가 stride 2, 4, 8, 16, 32에
  해당하는 5개 stage를 반환하고, `stage_channels`가 `(64, 128, 256, 512, 512)`와 일치한다.
- `TorchBackbone("efficientnet_b0")`의 `forward()`가 stride 2, 4, 8, 16, 32에 해당하는 5개 stage를
  반환하고, `stage_channels`가 `(16, 24, 40, 112, 1280)`과 일치한다.
- `TorchBackbone("swin_t")`의 `forward()`가 stride 4, 8, 16, 32에 해당하는 4개 stage를 반환하고,
  `stage_channels`가 `(96, 192, 384, 768)`과 일치한다.
- `TimmBackbone("wide_resnet50_2.tv_in1k")`의 `forward()`가 stride 4, 8, 16, 32에 해당하는 4개
  stage를 반환하고, `stage_channels`가 `(256, 512, 1024, 2048)`과 일치한다.
- `DeconvBlock`이 `interpolate_conv`, `transposed_conv` 두 mode 모두에서 입력 대비 2배 해상도의
  출력을 반환한다.
- `SegDecoder`가 4단계와 5단계 dummy feature list 모두를 3.1절 결정표와 일치하는 채널/해상도로
  fusion하고, skip shape를 의도적으로 어긋나게 하면 `ValueError`를 발생시킨다.
- `SegModel(backbone=<9개 지원 backbone 중 하나>)`가 3.1절 결정표와 일치하는 `(B, 1, Hm, Wm)` raw
  mask logit을 반환하고, `SegModel(backbone="vit_b_16")` 등 목록 밖 backbone은 `ValueError`를
  발생시킨다.
- `SegWrapper(backbone=<9개 중 하나>, device="cpu")`가 2-sample smoke `train_step`/`eval_step`을
  shape 오류 없이 완료하고, loss/metric 결과에 `bce`, `dice`, `iou`가 보고된다.
- `src/core/factory.py::get_wrapper("seg", backbone=..., head="mask")`가 `SegWrapper` 인스턴스를
  반환한다.
- `experiments/configs.py`에 seg config 9개가 추가되고, 기존 reg config와 로컬 주석 처리 상태는
  그대로 유지된다.
- `docs/references/backbones.md`의 `resnet34`, `resnet50`, `efficientnet_b0`, `swin_t`, `vgg16`, `vgg19`,
  `wide_resnet50_2.tv_in1k` row에 seg 연결 내용이 반영된다.
- `docs/plans/0009-seg-unet-backbone-plan.md` 상태가 `Approved`에서 `Done`으로 갱신된다.

## 5. 검증

구현 후 다음 순서로 검증한다.

```bash
conda activate pytorch_env
python -c "import torch; from src.models.backbones.torch_backbone import TorchBackbone; \
for b in ['efficientnet_b0', 'swin_t', 'vgg16', 'vgg19']: \
    m = TorchBackbone(b); out = m(torch.zeros(1,3,224,224)); \
    print(b, m.stage_channels, [s.shape for s in out['stages']])"
python -c "import torch; from src.models.backbones.timm_backbone import TimmBackbone; \
m = TimmBackbone('wide_resnet50_2.tv_in1k'); out = m(torch.zeros(1,3,224,224)); \
print(m.stage_channels, m.stage_strides, [s.shape for s in out['stages']])"
python -c "import torch; from src.models.blocks.deconv_block import DeconvBlock; \
for mode in ['interpolate_conv', 'transposed_conv']: \
    b = DeconvBlock(64, 32, mode=mode); print(mode, b(torch.zeros(1,64,7,7)).shape)"
python -c "import torch; from src.models.seg.decoder import SegDecoder; \
d = SegDecoder((64,128,256,512)); \
stages = [torch.zeros(1,64,56,56), torch.zeros(1,128,28,28), torch.zeros(1,256,14,14), torch.zeros(1,512,7,7)]; \
print(d(stages).shape)"
python -c "import torch; from src.models.seg.model import SegModel; \
for b in ['custom', 'resnet18', 'resnet34', 'resnet50', 'efficientnet_b0', 'swin_t', 'vgg16', 'vgg19', 'wide_resnet50_2.tv_in1k']: \
    m = SegModel(backbone=b); print(b, m(torch.zeros(2,3,224,224)).shape, m.mask_stride)"
python -c "from src.models.seg.model import SegModel; \
try: SegModel(backbone='vit_b_16')\nexcept ValueError as e: print('OK:', e)"
python -c "from experiments.configs import CONFIGS; print(len(CONFIGS)); \
print([c['backbone'] for c in CONFIGS if c['method'] == 'seg'])"
python scripts/train.py --method seg --backbone custom --head mask --device cpu \
  --train_size 2 --valid_size 2 --batch_size 1 --max_epochs 1 --patience 1 --num_workers 0 \
  --output_dir /tmp/seg_custom_smoke
python scripts/train.py --method seg --backbone resnet18 --head mask --device cpu \
  --train_size 2 --valid_size 2 --batch_size 1 --max_epochs 1 --patience 1 --num_workers 0 \
  --output_dir /tmp/seg_resnet18_smoke
python scripts/train.py --method seg --backbone resnet34 --head mask --device cpu \
  --train_size 2 --valid_size 2 --batch_size 1 --max_epochs 1 --patience 1 --num_workers 0 \
  --output_dir /tmp/seg_resnet34_smoke
python scripts/train.py --method seg --backbone resnet50 --head mask --device cpu \
  --train_size 2 --valid_size 2 --batch_size 1 --max_epochs 1 --patience 1 --num_workers 0 \
  --output_dir /tmp/seg_resnet50_smoke
python scripts/train.py --method seg --backbone efficientnet_b0 --head mask --device cpu \
  --train_size 2 --valid_size 2 --batch_size 1 --max_epochs 1 --patience 1 --num_workers 0 \
  --output_dir /tmp/seg_efficientnet_b0_smoke
python scripts/train.py --method seg --backbone swin_t --head mask --device cpu \
  --train_size 2 --valid_size 2 --batch_size 1 --max_epochs 1 --patience 1 --num_workers 0 \
  --output_dir /tmp/seg_swin_t_smoke
python scripts/train.py --method seg --backbone vgg16 --head mask --device cpu \
  --train_size 2 --valid_size 2 --batch_size 1 --max_epochs 1 --patience 1 --num_workers 0 \
  --output_dir /tmp/seg_vgg16_smoke
python scripts/train.py --method seg --backbone vgg19 --head mask --device cpu \
  --train_size 2 --valid_size 2 --batch_size 1 --max_epochs 1 --patience 1 --num_workers 0 \
  --output_dir /tmp/seg_vgg19_smoke
python scripts/train.py --method seg --backbone wide_resnet50_2.tv_in1k --head mask --device cpu \
  --train_size 2 --valid_size 2 --batch_size 1 --max_epochs 1 --patience 1 --num_workers 0 \
  --output_dir /tmp/seg_wide_resnet50_smoke
```

검증 결과에서는 `TorchBackbone`/`TimmBackbone`의 확장된 다단계 stage shape, `DeconvBlock`/
`SegDecoder`/`SegModel`의 shape, 지원하지 않는 backbone에서의 `ValueError`,
`experiments/configs.py`의 seg config 개수와 backbone 목록, 9개 backbone 각각의 smoke train 성공
여부를 확인한다. 검증 후 `/tmp/seg_*_smoke` 산출물은 삭제한다.
