# det CustomDetModel 조립

다음 표는 이 plan의 상태와 적용 범위를 정리한다.

| 항목 | 값 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-18 |
| 적용 범위 | `docs/architecture/model-assembly.md`, `experiments/configs.py`, `src/core/factory.py`, `src/losses/focal_loss.py`(신규), `src/losses/smoothl1_loss.py`(신규), `src/models/det/__init__.py`(신규), `src/models/det/model.py`(신규), `src/models/det/postprocessor.py`(신규), `src/models/det/preprocessor.py`(신규), `src/models/det/wrapper.py`(신규), `src/models/heads/detection_head.py`(신규), `src/models/necks/__init__.py`(신규), `src/models/necks/multi_scale_neck.py`(신규) |
| 관련 문서 | [docs/architecture/model-assembly.md](../architecture/model-assembly.md), [docs/references/backbones.md](../references/backbones.md), [docs/plans/0009-seg-unet-backbone-plan.md](0009-seg-unet-backbone-plan.md), [docs/plans/0010-torchseg-model-plan.md](0010-torchseg-model-plan.md) |

## 1. 목적과 배경

canonical 문서 4.4절은 `CustomDetModel`을 `CustomBackbone/pretrained backbone + backbone adapter +
multi-scale neck + DetectionHead` 조립으로, multi-scale neck의 책임을 "stage channel projection과
multi-scale feature 생성", detection head의 책임을 "corner class, box 또는 point raw output 생성"으로
이미 정의하고 있다
([model-assembly.md:346-357](../architecture/model-assembly.md#L346-L357)). 그러나 `src/`에는 `reg`,
`seg`만 구현되어 있고 `det`는 전혀 없다(전체 검색으로 확인).

이번 plan은 `det`의 세 번째이자 마지막 methodology를 Category A(custom composable) 범위로 한정해
구현한다. raw output 표현은 grid cell마다 corner class별 objectness(classification)를 예측하고,
같은 cell의 center offset을 회귀하는 구조를 공통 기반으로 삼는다. canonical 2.2절이 "boxes or
points"를 같은 표현 축의 대안으로 병기하므로, box width/height 회귀 채널의 유무를 `representation`
파라미터(`"box"` 또는 `"point"`)로 선택 가능하게 만든다. `DetPostprocessor`는 confidence가 가장 높은
cell의 `(dx, dy)` offset만으로 좌표를 decode하고 `(dw, dh)`는 전혀 사용하지 않으므로, 두 표현의
차이는 `DetectionHead`의 box 출력 채널 수와 `DetPreprocessor`가 `dw, dh` target을 채우는지 여부뿐이며
`DetPostprocessor`와 `SmoothL1Loss`는 두 표현에서 코드 변경 없이 동일하게 동작한다. corner는 실제
크기를 가진 object가 아니라 두 변의 교차점이므로, box 표현을 쓰더라도 width/height는 학습을 위한
인위적 placeholder 값으로 취급하고 이후 ablation 대상으로 남긴다. 외부 whole-model detector(Faster
R-CNN, RetinaNet, YOLO, DETR 등, Category C)는 이번 plan의 범위 밖이며, `seg`가 0009
(`CustomSegModel`)와 0010(`TorchSegModel`)으로 나뉜 것과 동일하게 별도 후속 plan으로 미룬다.

canonical 6.2절 backbone family 호환표는 `stages` capability가 필요한 decoder/neck이 capability
없는 backbone에서 "생성 단계에 실패해야 한다"고 명시하고, custom detection 열에서 ViT/DINOv2를
"초기 제외"로 표시한다
([model-assembly.md:281-294](../architecture/model-assembly.md#L281-L294)). `TransformerBackboneAdapter`는
항상 `stages=None`을 반환하므로(`src/models/adapters/transformer_adapter.py` 직접 확인), det 조립은
ViT 계열 backbone에서 즉시 `ValueError`를 발생시켜야 한다.

canonical 2.5절 complexity 기준에 따르면 실패 가능한 postprocess를 포함하는 det는 "높음" 등급이다
([model-assembly.md:189-200](../architecture/model-assembly.md#L189-L200)). 이번 plan의
`DetPostprocessor`는 `BaseWrapper.compute_metrics`가 기대하는 대로 항상 단일 `(N, 4, 2)` tensor를
best-effort로 반환하는 `RegPostprocessor`/`SegPostprocessor`와 동일한 단순화된 계약을 따른다(실패
flag 없음). 두 corner가 같은 grid cell에 몰리는 등 기하학적으로 퇴화된 입력에서도 예외 없이 최선의
좌표를 반환한다는 뜻이며, "높음" 등급은 이 best-effort decode가 언제든 기하학적으로 무의미한 좌표를
반환할 수 있다는 의미로 기록한다(실패 flag 계약 변경은 범위 밖).

## 2. 범위

이번 plan에 포함하는 항목은 다음과 같다.

- grid 기반 표현의 구체적 설계를 확정한다: `grid_stride` 기본값, `representation`(`box`/`point`)
  선택 방식, class-agnostic box regression 채널 구성, box 표현에서의 `box_size` placeholder 기본값.
- `src/models/necks/multi_scale_neck.py`를 신규 작성한다. `MultiScaleNeck`은 `ConvBlock`으로 stage별
  채널을 공통 채널로 projection하고, `bundle.stages`를 top-down으로 fusion하되 `grid_stride`에 해당하는
  stage에서 fusion을 멈춘다.
- `src/models/heads/detection_head.py`를 신규 작성한다. `DetectionHead`는 neck feature에서
  classification map과 box regression map(표현에 따라 2채널 또는 4채널)을 생성한다.
- `src/losses/focal_loss.py`, `src/losses/smoothl1_loss.py`를 신규 작성한다. 둘 다 `BaseLoss`를
  상속한다.
- `src/models/det/model.py`, `preprocessor.py`, `postprocessor.py`, `wrapper.py`, `__init__.py`를
  신규 작성한다.
- `src/core/factory.py::get_wrapper`에 `det` 분기를 추가한다.
- `experiments/configs.py`에 det backbone 비교 config(box 표현 9개, point 표현 대표 1개)를
  추가한다(seg와 동일한 CNN-only backbone 목록, ViT 제외).
- `docs/architecture/model-assembly.md` 4.4절에 표현 확정 방식(`representation` 파라미터),
  `grid_stride`/`box_size` 파라미터화 내용을 보강한다.

이번 plan에서 제외하는 항목은 다음과 같다.

- 외부 whole-model detector(Faster R-CNN, RetinaNet, SSD, YOLOv8n, DETR-R50 등, Category C)는
  제외한다. `docs/references/backbones.md` 3.3절에 이미 이들 weight와 "적용 방법과 제약"이 자리
  표시되어 있으며, `seg`가 0009/0010으로 나뉜 전례를 따라 별도 후속 plan(가칭
  `00xx-torchdet-model-plan.md`)에서 다룬다. 이번 plan은 그 파일을 수정하지 않는다.
- `representation="point"`와 `"box"` 사이의 정량적 성능 비교(ablation 실험 실행과 결과 보고)는
  하지 않는다. 이번 plan은 두 표현을 코드로 모두 지원하고 최소 smoke 검증만 수행하며, 실제 비교
  실험은 후속 작업으로 남긴다.
- `docs/references/backbones.md`의 개별 backbone row 수정은 하지 않는다. `seg`와 동일하게 `stages`를
  반환하는 9개 backbone(`custom`, `resnet18/34/50`, `efficientnet_b0`, `swin_t`, `vgg16/19`,
  `wide_resnet50_2.tv_in1k`)을 그대로 재사용하며, 이미 `stages` 확장이 0009에서 완료되어 있으므로
  backbone 코드 자체는 수정하지 않는다.
- `BasePostprocessor`/`BaseWrapper`의 실패 flag 계약 확장은 하지 않는다. `det`도 `reg`/`seg`와 같은
  단순화된 best-effort 단일 tensor 반환 계약을 따른다.

## 3. 구현 계획

### 3.1. grid/target 설계 결정

**`grid_stride` 기본값: 16.** det 대상 9개 backbone의 `stage_strides`를 직접 확인한 결과는 다음과
같다.

| backbone | `stage_channels` | `stage_strides` | stride 16 위치 | 비고 |
| --- | --- | --- | --- | --- |
| `custom` | `(64, 128, 256, 512)` | `(2, 4, 8, 16)` | 마지막(index 3) | 이 stage가 이미 최심층이므로 neck에서 fusion 없음 |
| `resnet18`, `resnet34`, `resnet50` | `(64..512)`/`(256..2048)` | `(4, 8, 16, 32)` | index 2 | stride 32 stage 1개를 fusion |
| `efficientnet_b0` | `(16, 24, 40, 112, 1280)` | `(2, 4, 8, 16, 32)` | index 3 | stride 32 stage 1개를 fusion |
| `vgg16`, `vgg19` | `(64, 128, 256, 512, 512)` | `(2, 4, 8, 16, 32)` | index 3 | stride 32 stage 1개를 fusion |
| `swin_t` | `(96, 192, 384, 768)` | `(4, 8, 16, 32)` | index 2 | stride 32 stage 1개를 fusion |
| `wide_resnet50_2.tv_in1k` | `(256, 512, 1024, 2048)` | `(4, 8, 16, 32)` | index 2 | stride 32 stage 1개를 fusion |
| `vit_b_16`, timm ViT 2종 | 없음(`stages=None`) | 없음 | - | det 생성 시 `ValueError` |

9개 backbone 전부가 stride 16 stage를 직접 보유하므로, `grid_stride=16`은 어떤 backbone에서도
"이미 있는 stage를 그대로 쓰거나 최대 1개의 더 깊은 stage만 fusion하면 되는" 가장 균일한 선택이다.
`image_size=224` 기준 grid 크기는 `14x14=196` cell이다. `grid_stride=8`(28x28=784 cell)은 subpixel
정밀도는 높아지지만 양성 cell이 4개뿐인 구조에서 class imbalance가 4배 악화되고, `grid_stride=32`
(7x7=49 cell)는 offset 회귀로 보정할 여지가 줄어 인접 코너가 같은 cell로 충돌할 위험이 커진다. 16은
두 극단 사이의 절충점이며, `grid_stride`는 `MultiScaleNeck`/`DetModel`/`DetPreprocessor`/
`DetPostprocessor` 생성자 인자로 노출해 이후 ablation을 막지 않는다.

**표현 선택: `representation="box"|"point"` 파라미터, classification은 4채널로 공통.**
`DetectionHead`는 두 raw map을 생성한다.

- classification map `(B, 4, Gh, Gw)`: channel index가 corner class(0=TL, 1=TR, 2=BR, 3=BL)를
  결정론적으로 나타낸다. 표현과 무관하게 항상 4채널이다.
- box regression map `(B, C, Gh, Gw)`: `representation="box"`면 `C=4`(`dx, dy, dw, dh`),
  `representation="point"`면 `C=2`(`dx, dy`만). 둘 다 corner class에 무관하게 **공유**한다.

class별로 별도 box map(`(B, 4*C, Gh, Gw)`)을 두는 대신 공유 map을 선택한 이유는, 4개 코너가 둥근
사각형 패널의 4개 서로 다른 사분면에 위치해 동일 grid cell에서 두 class의 target이 동시에 존재할
가능성이 사실상 없기 때문이다(`is_invalid_corners`가 이미 이런 퇴화 케이스를 최소거리 기준으로
표시한다). 공유 map은 head 파라미터 수를 줄이면서도 이 도메인 제약 하에서는 정보 손실이 없다.

`DetPostprocessor`는 `raw_output["box"][:, 0:2]`만 읽어 offset을 decode하므로 `C=2`든 `C=4`든
코드 변경이 없다. `SmoothL1Loss`도 `pred`/`target`의 실제 channel 수에 대해 elementwise로 동작하므로
표현에 무관하게 그대로 재사용한다. 즉 `representation`은 `DetectionHead`의 `box_conv` out_channels와
`DetPreprocessor`가 `dw, dh` target을 채우는지 여부에만 영향을 준다.

**`box_size` 기본값: 0.1(정규화 좌표계, `representation="box"`에서만 사용).** grid cell 한 변의
길이가 `1/14 ≈ 0.071`이므로, 0.1은 같은 자릿수이면서 0에 너무 가깝지 않은(회귀 target이 붕괴하지
않는) 값이다. 이 값은 `DetPostprocessor`의 decode 경로에서 전혀 사용되지 않는다 — canonical 4.4절이
detection postprocessor 책임을 "confidence selection, center decode와 ordering"으로만 한정하므로,
box width/height는 decode에 기여하지 않는 순수 training-time 보조 target이다. `DetPreprocessor`/
`DetWrapper` 생성자 인자로 노출해 이후 ablation을 막지 않는다.

**target/raw output shape 요약**

| 구성요소 | shape (`representation="box"`) | shape (`representation="point"`) | 비고 |
| --- | --- | --- | --- |
| neck 출력 feature | `(B, neck_channels, 14, 14)` | 동일 | `neck_channels` 기본값 256 |
| `DetectionHead` classification 출력 | `(B, 4, 14, 14)` | 동일 | sigmoid 전 raw logit |
| `DetectionHead` box 출력 | `(B, 4, 14, 14)` | `(B, 2, 14, 14)` | box: `(dx,dy,dw,dh)`, point: `(dx,dy)` |
| `DetPreprocessor` target | `dict(cls=(N,4,14,14), box=(N,4,14,14), pos_mask=(N,1,14,14))` | `dict(cls=..., box=(N,2,14,14), pos_mask=...)` | `DetTarget(dict)` subclass |
| `DetPostprocessor` 출력 | `(N, 4, 2)` | `(N, 4, 2)` | channel index가 곧 corner 순서(TL,TR,BR,BL) |

**corner ordering: `order_corners` 호출하지 않는다.** `seg`의 `mask_to_corners`는 미분화된 mask에서
기하학적 극값으로 순서를 사후 추론해야 하므로 `order_corners`가 필요하지만, det는 classification map의
channel index 자체가 학습으로 강제된 corner identity다(channel 0은 오직 TL만 예측하도록 훈련됨).
`order_corners`를 decode 후에 다시 적용하면, 학습이 덜 된 초기 단계에서 채널이 서로 가까운 좌표를
예측할 때 기하학적 재정렬이 오히려 올바른 채널 라벨을 뒤섞을 수 있다. 따라서 기본 decode 경로에서는
`order_corners`를 호출하지 않는다. `order_corners`는 별도 디버깅/sanity-check 용도로 남겨두되 이번
plan의 `DetPostprocessor` 구현에는 포함하지 않는다.

### 3.2. `src/models/necks/multi_scale_neck.py`(신규)

`MultiScaleNeck`은 `ConvBlock`(1x1 lateral projection)으로 각 stage를 공통 채널로 맞추고,
`DeconvBlock` + element-wise add + `ConvBlock`으로 `SegDecoder`와 동일한 top-down fusion 패턴을
적용하되, `stage_strides`에서 `grid_stride`에 해당하는 index부터 시작해 그보다 얕은(고해상도) stage는
사용하지 않는다.

```python
# src/models/necks/multi_scale_neck.py: stage channel projection and top-down fusion stopping at grid_stride

import torch.nn as nn

from src.models.blocks.conv_block import ConvBlock
from src.models.blocks.deconv_block import DeconvBlock


class MultiScaleNeck(nn.Module):
    """Projects stages to a common channel width and fuses top-down, stopping at grid_stride resolution."""

    def __init__(self, stage_channels, stage_strides, grid_stride=16, out_channels=256, upsample="interpolate_conv"):
        super().__init__()
        if grid_stride not in stage_strides:
            raise ValueError("grid_stride %d not in stage_strides %s" % (grid_stride, stage_strides))
        self.grid_index = stage_strides.index(grid_stride)
        used_channels = stage_channels[self.grid_index:]

        self.laterals = nn.ModuleList([
            ConvBlock(c, out_channels, kernel_size=1, stride=1) for c in used_channels
        ])
        self.up_blocks = nn.ModuleList()
        self.fuse_blocks = nn.ModuleList()
        for _ in range(len(used_channels) - 1):
            self.up_blocks.append(DeconvBlock(out_channels, out_channels, mode=upsample))
            self.fuse_blocks.append(ConvBlock(out_channels, out_channels, kernel_size=3, stride=1))
        self.out_channels = out_channels

    def forward(self, stages):
        used = stages[self.grid_index:]
        laterals = [lateral(feat) for lateral, feat in zip(self.laterals, used)]
        x = laterals[-1]
        skips = list(reversed(laterals[:-1]))
        for up_block, fuse_block, skip in zip(self.up_blocks, self.fuse_blocks, skips):
            x = up_block(x)
            if x.shape[-2:] != skip.shape[-2:]:
                raise ValueError("neck feature shape %s does not match skip feature shape %s"
                                  % (tuple(x.shape[-2:]), tuple(skip.shape[-2:])))
            x = fuse_block(x + skip)
        return x
```

`custom` backbone처럼 `grid_index`가 마지막 index면 `up_blocks`/`fuse_blocks`가 빈 `ModuleList`가
되어 `DeconvBlock`을 전혀 사용하지 않는다. 나머지 8개 backbone은 정확히 1개의 `DeconvBlock`을
사용한다. 이는 canonical 4.5절의 "`DeconvBlock`: neck에 따라 조건부"를 구체적인 메커니즘으로
설명한다([model-assembly.md:359-374](../architecture/model-assembly.md#L359-L374)).

### 3.3. `src/models/heads/detection_head.py`(신규)

`MaskHead` 선례를 따라 최종 channel projection만 담당하되, projection 전에 3x3 `ConvBlock` trunk를
하나 둔다(class/box 두 갈래가 같은 neck feature를 공유하므로 얕은 trunk로 분리 전 표현력을 보강).
`representation` 인자로 box 출력 채널 수를 2 또는 4로 결정한다.

```python
# src/models/heads/detection_head.py: predicts per-cell corner classification and box/point regression maps

import torch.nn as nn

from src.models.blocks.conv_block import ConvBlock

NUM_CORNER_CLASSES = 4
BOX_CHANNELS = {"box": 4, "point": 2}  # dx, dy[, dw, dh], shared across corner classes


class DetectionHead(nn.Module):
    """Splits a shared trunk into a per-class classification map and a class-agnostic box/point regression map."""

    def __init__(self, in_channels, hidden_channels=256, representation="box"):
        super().__init__()
        if representation not in BOX_CHANNELS:
            raise ValueError("Unknown det representation: %s. Supported: %s"
                              % (representation, ", ".join(BOX_CHANNELS)))
        self.representation = representation
        self.trunk = ConvBlock(in_channels, hidden_channels, kernel_size=3, stride=1)
        self.cls_conv = nn.Conv2d(hidden_channels, NUM_CORNER_CLASSES, kernel_size=1)
        self.box_conv = nn.Conv2d(hidden_channels, BOX_CHANNELS[representation], kernel_size=1)

    def forward(self, feature):
        x = self.trunk(feature)
        return {"cls": self.cls_conv(x), "box": self.box_conv(x)}
```

`cls`/`box` 모두 raw logit이며 sigmoid는 head 밖(loss, postprocessor)에서 적용한다
(`MaskHead` 계약과 동일).

### 3.4. `src/losses/focal_loss.py`(신규)

RetinaNet 스타일 sigmoid focal loss로, classification map 전체(대부분 음성)에 대해 계산한다.
`raw_output`/`target`이 모두 dict이므로 `"cls"` key만 읽는다. `representation`과 무관하게 동일하다.

```python
# src/losses/focal_loss.py: sigmoid focal loss for the sparse per-cell corner classification map

import torch
import torch.nn.functional as F

from src.losses.base_loss import BaseLoss


class FocalLoss(BaseLoss):
    """RetinaNet-style sigmoid focal loss between a per-class classification map and a binary target."""

    def __init__(self, alpha=0.25, gamma=2.0, weight=1.0):
        super().__init__(weight=weight)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, raw_output, target):
        logits = raw_output["cls"]
        cls_target = target["cls"]
        prob = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, cls_target, reduction="none")
        p_t = prob * cls_target + (1.0 - prob) * (1.0 - cls_target)
        alpha_t = self.alpha * cls_target + (1.0 - self.alpha) * (1.0 - cls_target)
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * ce
        return loss.mean()
```

### 3.5. `src/losses/smoothl1_loss.py`(신규)

box regression은 `pos_mask`(양성 cell)에서만 계산한다. `dx, dy` 채널은 sigmoid로 cell 내부 offset으로
제한한 뒤 target과 비교하고(postprocessor의 decode 경로와 train/inference를 일치시키기 위함),
`representation="box"`의 `dw, dh`는 placeholder 상수 target에 대해 그대로 회귀한다. 채널 수는
`raw_output["box"]`/`target["box"]`의 실제 shape을 그대로 따르므로 `representation`에 대한 분기가
코드에 없다.

```python
# src/losses/smoothl1_loss.py: masked smooth L1 loss for the per-cell box/point regression map

import torch

from src.losses.base_loss import BaseLoss


class SmoothL1Loss(BaseLoss):
    """Smooth L1 loss on box/point regression, masked to positive cells and sigmoid-bounded offset channels."""

    def __init__(self, beta=1.0, weight=1.0):
        super().__init__(weight=weight)
        self.beta = beta

    def forward(self, raw_output, target):
        pred = raw_output["box"].clone()
        pred[:, 0:2] = torch.sigmoid(pred[:, 0:2])
        box_target = target["box"]
        pos_mask = target["pos_mask"]

        diff = (pred - box_target).abs() * pos_mask
        loss = torch.where(diff < self.beta, 0.5 * diff.pow(2) / self.beta, diff - 0.5 * self.beta)
        denom = pos_mask.sum().clamp(min=1.0) * pred.shape[1]
        return loss.sum() / denom
```

### 3.6. `src/models/det/model.py`(신규)

`RegModel`/`SegModel`과 동일한 backbone dispatch 패턴을 따르되, ViT 계열은 dispatch 전에 즉시
`ValueError`를 발생시킨다(요구사항: "생성 단계에 실패해야 한다"). `representation`을 생성자 인자로
받아 `DetectionHead`에 그대로 전달한다.

```python
# src/models/det/model.py: selectable stage-returning backbone plus a multi-scale neck and detection head

from src.models.base.base_model import BaseModel
from src.models.backbones.custom_backbone import CustomBackbone
from src.models.backbones.torch_backbone import EFFICIENTNET_BACKBONES, RESNET_BACKBONES
from src.models.backbones.torch_backbone import SWIN_BACKBONES, VGG_BACKBONES, VIT_BACKBONES, TorchBackbone
from src.models.backbones.timm_backbone import TIMM_CNN_BACKBONES, TIMM_VIT_BACKBONES, TimmBackbone
from src.models.adapters.cnn_adapter import CNNBackboneAdapter
from src.models.features import FeatureExtractor, FeatureSpec
from src.models.necks.multi_scale_neck import MultiScaleNeck
from src.models.heads.detection_head import DetectionHead

TORCH_DET_BACKBONES = RESNET_BACKBONES + EFFICIENTNET_BACKBONES + SWIN_BACKBONES + VGG_BACKBONES
SUPPORTED_DET_BACKBONES = ("custom",) + TORCH_DET_BACKBONES + TIMM_CNN_BACKBONES


class DetModel(BaseModel):
    """Stage-returning backbone plus a multi-scale neck feeding a per-cell detection head."""

    def __init__(self, in_channels=3, backbone="custom", neck_channels=256, grid_stride=16,
                 representation="box", upsample="interpolate_conv"):
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
        self.head = DetectionHead(self.neck.out_channels, representation=representation)
        self.grid_stride = grid_stride
        self.representation = representation

    def forward(self, images):
        bundle = self.extractor(images)
        feature = self.neck(bundle.stages)
        return self.head(feature)
```

`spec.require("stages")`는 dispatch 단계의 명시적 `ValueError`에 더해진 방어선이다(향후 backbone
추가 시 실수로 `stages`가 빠진 backbone이 조용히 통과하는 것을 막는다).

### 3.7. `src/models/det/preprocessor.py`(신규)

`(N, 4, 2)` corners를 grid target dict로 변환한다. `representation="point"`면 `box_target`의 채널
수를 2로 줄이고 `dw, dh`를 채우지 않는다. `DetTarget`은 `dict`를 상속해 `"cls"`/`"box"`/`"pos_mask"`
key 접근은 그대로 유지하면서, `BaseLoss.__call__`이 호출하는 `len(target)`이 dict key 개수(3)가
아니라 실제 batch size를 반환하도록 `__len__`만 override한다(`BaseLoss`/`BaseWrapper`는 변경하지
않는, det 쪽에서만 닫힌 수정).

```python
# src/models/det/preprocessor.py: convert standard corners into per-cell classification and box/point targets

import torch

from src.models.base.base_preprocessor import BasePreprocessor

BOX_CHANNELS = {"box": 4, "point": 2}


class DetTarget(dict):
    """Dict-based target whose __len__ reports batch size instead of key count for BaseLoss weighting."""

    def __len__(self):
        return self["cls"].shape[0]


class DetPreprocessor(BasePreprocessor):
    """Assigns each of the 4 corners to one grid cell, producing (N,4,Gh,Gw) cls and (N,C,Gh,Gw) box targets."""

    def __init__(self, grid_stride=16, image_size=224, representation="box", box_size=0.1):
        if representation not in BOX_CHANNELS:
            raise ValueError("Unknown det representation: %s. Supported: %s"
                              % (representation, ", ".join(BOX_CHANNELS)))
        self.grid_h = image_size // grid_stride
        self.grid_w = image_size // grid_stride
        self.representation = representation
        self.box_size = box_size

    def __call__(self, corners):
        device = corners.device
        n = corners.shape[0]
        channels = BOX_CHANNELS[self.representation]
        cls_target = torch.zeros(n, 4, self.grid_h, self.grid_w, device=device)
        box_target = torch.zeros(n, channels, self.grid_h, self.grid_w, device=device)
        idx = torch.arange(n, device=device)

        for c in range(4):
            x = corners[:, c, 0].clamp(0.0, 1.0 - 1e-6)
            y = corners[:, c, 1].clamp(0.0, 1.0 - 1e-6)
            gx = (x * self.grid_w).long()
            gy = (y * self.grid_h).long()
            dx = x * self.grid_w - gx.float()
            dy = y * self.grid_h - gy.float()

            cls_target[idx, c, gy, gx] = 1.0
            box_target[idx, 0, gy, gx] = dx
            box_target[idx, 1, gy, gx] = dy
            if self.representation == "box":
                box_target[idx, 2, gy, gx] = self.box_size
                box_target[idx, 3, gy, gx] = self.box_size

        pos_mask = cls_target.amax(dim=1, keepdim=True)
        return DetTarget(cls=cls_target, box=box_target, pos_mask=pos_mask)
```

두 corner가 같은 cell에 배정되면(퇴화 입력) 나중에 처리된 class가 해당 cell의 box target을
덮어쓴다. `is_invalid_corners`로 걸러지는 극단적 퇴화 케이스 외에는 실무상 발생하지 않는다.

### 3.8. `src/models/det/postprocessor.py`(신규)

`raw_output` dict에서 class별 confidence 최대 cell을 선택하고, 그 cell의 offset을 sigmoid로 decode해
`(N, 4, 2)`를 만든다. `pos_mask`나 box width/height는 사용하지 않으므로 `representation`에 대한
분기가 없다.

```python
# src/models/det/postprocessor.py: convert raw classification/box maps into standard corners

import torch

from src.models.base.base_postprocessor import BasePostprocessor


class DetPostprocessor(BasePostprocessor):
    """Selects the highest-confidence cell per corner class and decodes its center offset to (N,4,2)."""

    def __init__(self, grid_stride=16, image_size=224):
        self.grid_h = image_size // grid_stride
        self.grid_w = image_size // grid_stride

    def __call__(self, raw_output):
        cls_logits = raw_output["cls"]
        box_raw = raw_output["box"]
        n = cls_logits.shape[0]
        device = cls_logits.device

        cls_prob = torch.sigmoid(cls_logits).reshape(n, 4, -1)
        best = cls_prob.argmax(dim=-1)
        gy = best // self.grid_w
        gx = best % self.grid_w
        offset = torch.sigmoid(box_raw[:, 0:2])

        idx = torch.arange(n, device=device)
        corners = torch.zeros(n, 4, 2, device=device)
        for c in range(4):
            dx = offset[idx, 0, gy[:, c], gx[:, c]]
            dy = offset[idx, 1, gy[:, c], gx[:, c]]
            corners[:, c, 0] = (gx[:, c].float() + dx) / self.grid_w
            corners[:, c, 1] = (gy[:, c].float() + dy) / self.grid_h
        return corners
```

### 3.9. `src/models/det/wrapper.py`(신규)

`RegWrapper`와 동일한 differential-LR 패턴을 그대로 따른다. `representation`을 생성자 인자로 받아
`DetModel`/`DetPreprocessor`에 전달한다.

```python
# src/models/det/wrapper.py: composes DetModel/DetPreprocessor/DetPostprocessor and Focal+SmoothL1 loss

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.base.base_wrapper import BaseWrapper
from src.models.det.model import DetModel
from src.models.det.preprocessor import DetPreprocessor
from src.models.det.postprocessor import DetPostprocessor
from src.losses.focal_loss import FocalLoss
from src.losses.smoothl1_loss import SmoothL1Loss
from src.metrics.polygon_iou import PolygonIoU


class DetWrapper(BaseWrapper):
    """Wraps DetModel training/evaluation/inference behind the shared Trainer/Evaluator/Predictor interface."""

    def __init__(self, in_channels=3, backbone="custom", head="detection", neck_channels=256,
                 grid_stride=16, representation="box", box_size=0.1, image_size=224,
                 optimizer=None, scheduler=None, preprocessor=None, postprocessor=None,
                 losses=None, metrics=None, device=None):
        # head kwarg accepted for CLI compatibility with get_wrapper_kwargs; det has one head type
        model = DetModel(in_channels=in_channels, backbone=backbone, neck_channels=neck_channels,
                          grid_stride=grid_stride, representation=representation)
        preprocessor = preprocessor or DetPreprocessor(
            grid_stride=model.grid_stride, image_size=image_size,
            representation=representation, box_size=box_size)
        postprocessor = postprocessor or DetPostprocessor(
            grid_stride=model.grid_stride, image_size=image_size)
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
        self.set_losses(self.losses or {"cls": FocalLoss(), "box": SmoothL1Loss()})
        self.set_metrics(self.metrics or {"iou": PolygonIoU()})
```

`PolygonIoU`는 `(B,4,2)` 계약을 그대로 소비하므로 수정 없이 재사용한다. `Trainer`의
`DEFAULT_MONITOR = "iou"` early-stopping과도 별도 배선 없이 맞물린다.

### 3.10. `src/models/det/__init__.py`, `src/models/necks/__init__.py`(신규)

프로젝트 내 모든 `__init__.py`가 빈 파일이므로(직접 확인), 두 파일도 빈 파일로 둔다.

### 3.11. `src/core/factory.py`(수정)

`get_wrapper`에 `det` 분기를 추가한다. 기존 `reg`/`seg` 분기와 동일한 lazy import 패턴을 따른다.

```python
    if method == "det":
        from src.models.det.wrapper import DetWrapper
        return DetWrapper(device=device, **kwargs)
```

### 3.12. `experiments/configs.py`(수정)

`seg`와 동일한 9개 CNN-only backbone 목록으로 `representation="box"`(기본값) det config를
추가하고, `representation` 옵션이 실제로 CLI에서 선택 가능함을 보여주는 `custom` backbone
point 표현 config 1개를 추가한다(ViT 계열 제외, 주석 처리 상태 유지).

```python
    # {"method": "det", "batch_size": 4, "max_epochs": 5, "backbone": "custom", "head": "detection"},
    # {"method": "det", "batch_size": 4, "max_epochs": 5, "backbone": "resnet18", "head": "detection"},
    # {"method": "det", "batch_size": 4, "max_epochs": 5, "backbone": "resnet34", "head": "detection"},
    # {"method": "det", "batch_size": 4, "max_epochs": 5, "backbone": "resnet50", "head": "detection"},
    # {"method": "det", "batch_size": 4, "max_epochs": 5, "backbone": "efficientnet_b0", "head": "detection"},
    # {"method": "det", "batch_size": 4, "max_epochs": 5, "backbone": "swin_t", "head": "detection"},
    # {"method": "det", "batch_size": 4, "max_epochs": 5, "backbone": "vgg16", "head": "detection"},
    # {"method": "det", "batch_size": 4, "max_epochs": 5, "backbone": "vgg19", "head": "detection"},
    # {"method": "det", "batch_size": 4, "max_epochs": 5, "backbone": "wide_resnet50_2.tv_in1k", "head": "detection"},
    # {"method": "det", "batch_size": 4, "max_epochs": 5, "backbone": "custom", "head": "detection", "representation": "point"},
```

`experiments/run.py`, `scripts/config.py`는 method-agnostic passthrough이므로 수정하지 않는다.
`scripts/config.py::get_wrapper_kwargs`가 `representation` key를 forward하지 않으므로, det
config에서 `representation`을 실제로 CLI까지 전달하려면 `PASS_KEYS`(`experiments/run.py`)와
`get_wrapper_kwargs`(`scripts/config.py`)에 `representation`을 추가해야 한다. 이 두 파일은
method-agnostic이지만 `representation` 항목 자체는 det 전용 키이므로, `scripts/config.py::DEFAULTS`에
`representation="box"`를 추가하고 `parse_args`가 `--representation` CLI 인자를 받도록 한다(다른
method는 이 값을 무시한다).

### 3.13. `docs/architecture/model-assembly.md` 4.4절 보강(수정)

4.4절은 조합 구조와 component 책임만 정의하고 raw output 표현(point vs box)이나 grid 해상도는 아직
결정하지 않은 상태다. `CLAUDE.md` 6절 규칙("요구사항이 바뀌면 코드보다 canonical 문서를 먼저
수정한다")에 따라, 이번 plan에서 확정하는 구체적 설계 결정을 4.4절 표 뒤, 4.5절 앞에 문단으로
추가한다. 나머지 절 번호는 변경되지 않는다.

```markdown
`CustomDetModel`의 raw output 표현은 `representation` 파라미터로 `box`와 `point` 중 선택한다.
두 표현 모두 grid cell마다 corner class별 objectness(classification)와 center offset을
예측하며, `box`는 여기에 box width/height 회귀를 추가한다. box width/height는 corner가 실제
크기를 가진 object가 아니라 line-intersection point라는 점을 반영해 학습을 위한 인위적
placeholder 값(기본값 0.1, 정규화 좌표계 기준)으로 취급하며, 이후 ablation 대상으로 남긴다.
detection postprocessor는 confidence가 가장 높은 cell의 center offset만으로 corner를 decode하므로
box width/height는 두 표현 모두에서 최종 corner 좌표에 영향을 주지 않는다. grid 해상도는
`grid_stride` 파라미터로 노출하며 기본값은 16이다(6.2절 지원 대상 9개 backbone이 모두 stride 16
stage를 직접 보유한다). box regression은 corner class에 무관한 채널을 공유한다.
```

6.1/6.2절 backbone family 호환표는 이미 custom detection 열을 갖고 있고 이번 plan이 그 판정을
바꾸지 않으므로 수정하지 않는다.

## 4. 완료 기준

이 plan은 다음 조건을 만족하면 `Done`으로 본다.

- `MultiScaleNeck`이 `stage_strides`에 `grid_stride`가 없으면 `ValueError`를 발생시키고, 있으면
  `(B, neck_channels, 14, 14)` feature를 반환한다. `custom` backbone은 `up_blocks`/`fuse_blocks`가
  비어 `DeconvBlock`을 사용하지 않고, 나머지 8개 backbone은 정확히 1개의 `DeconvBlock`을 사용한다.
- `DetectionHead(256, representation="box")`가 `(B, 4, 14, 14)` classification map과
  `(B, 4, 14, 14)` box map을, `representation="point"`는 `(B, 2, 14, 14)` box map을 raw logit으로
  반환한다. 알 수 없는 `representation` 값은 `ValueError`를 발생시킨다.
- `DetModel(backbone=<9개 지원 backbone 중 하나>)`가 `forward()`에서 `{"cls":..., "box":...}` dict를
  반환하고, `DetModel(backbone="vit_b_16")`, `DetModel(backbone="deit_base_distilled_patch16_224.fb_in1k")`
  등 ViT 계열은 즉시 `ValueError`를 발생시킨다.
- `DetPreprocessor(grid_stride=16, representation="box")(corners)`가 `box` 채널 4개인 `DetTarget`
  dict를, `representation="point"`는 채널 2개인 dict를 반환한다. `len(target)`이 dict key 개수(3)가
  아니라 실제 batch size와 같다.
- `DetPostprocessor(grid_stride=16)(raw_output)`이 `representation`과 무관하게 `(N, 4, 2)` tensor를
  반환하고, 알려진 corner 위치로부터 만든 완전 확신(one-hot) classification map을 넣으면 원래
  좌표를 grid 해상도(`1/14`) 오차 이내로 복원한다.
- `FocalLoss`, `SmoothL1Loss`가 `BaseLoss` contract(`reset`/`update`/`compute`/`__call__`)를 그대로
  따르고, `DetWrapper(backbone=<9개 중 하나>, representation=<"box"|"point">, device="cpu")`가
  2-sample smoke `train_step`/`eval_step`을 shape 오류 없이 완료하며 loss/metric 결과에 `cls`,
  `box`, `iou`가 보고된다.
- `src/core/factory.py::get_wrapper("det", backbone=..., head="detection")`가 `DetWrapper` 인스턴스를
  반환한다.
- `experiments/configs.py`에 det config 10개(box 표현 9개 + point 표현 1개)가 추가되고, 기존
  reg/seg config와 로컬 주석 처리 상태는 그대로 유지된다. `scripts/config.py`, `experiments/run.py`가
  `representation` 인자를 CLI까지 전달한다.
- `docs/architecture/model-assembly.md` 4.4절에 `representation` 파라미터, `grid_stride=16`,
  `box_size=0.1` placeholder 내용이 반영되고, 6.1/6.2절은 변경되지 않는다.
- `docs/plans/0011-det-custom-model-plan.md` 상태가 `Approved`에서 `Done`으로 갱신된다.

## 5. 검증

구현 후 다음 순서로 검증한다.

```bash
conda activate pytorch_env

python -c "import torch; from src.models.necks.multi_scale_neck import MultiScaleNeck; \
n = MultiScaleNeck((64,128,256,512), (2,4,8,16), grid_stride=16); \
stages = [torch.zeros(1,64,112,112), torch.zeros(1,128,56,56), torch.zeros(1,256,28,28), torch.zeros(1,512,14,14)]; \
print(n(stages).shape, len(n.up_blocks))"

python -c "import torch; from src.models.necks.multi_scale_neck import MultiScaleNeck; \
n = MultiScaleNeck((256,512,1024,2048), (4,8,16,32), grid_stride=16); \
stages = [torch.zeros(1,256,56,56), torch.zeros(1,512,28,28), torch.zeros(1,1024,14,14), torch.zeros(1,2048,7,7)]; \
print(n(stages).shape, len(n.up_blocks))"

python -c "import torch; from src.models.heads.detection_head import DetectionHead; \
h_box = DetectionHead(256, representation='box'); out_box = h_box(torch.zeros(2,256,14,14)); \
h_point = DetectionHead(256, representation='point'); out_point = h_point(torch.zeros(2,256,14,14)); \
print(out_box['cls'].shape, out_box['box'].shape, out_point['box'].shape)"

python -c "import torch; from src.models.det.model import DetModel; \
for b in ['custom', 'resnet18', 'resnet34', 'resnet50', 'efficientnet_b0', 'swin_t', 'vgg16', 'vgg19', 'wide_resnet50_2.tv_in1k']: \
    m = DetModel(backbone=b); out = m(torch.zeros(2,3,224,224)); \
    print(b, out['cls'].shape, out['box'].shape)"

python - <<'PY'
from src.models.det.model import DetModel
for bad in ("vit_b_16", "deit_base_distilled_patch16_224.fb_in1k"):
    try:
        DetModel(backbone=bad)
        print("FAIL: no error for", bad)
    except ValueError as e:
        print("OK:", bad, e)
PY

python -c "import torch; from src.models.det.preprocessor import DetPreprocessor; \
corners = torch.rand(3,4,2)*0.8; \
t_box = DetPreprocessor(grid_stride=16, representation='box')(corners); \
t_point = DetPreprocessor(grid_stride=16, representation='point')(corners); \
print(t_box['cls'].shape, t_box['box'].shape, t_box['pos_mask'].shape, len(t_box)); \
print(t_point['box'].shape)"

python -c "import torch; from src.models.det.postprocessor import DetPostprocessor; \
raw = {'cls': torch.randn(2,4,14,14), 'box': torch.randn(2,4,14,14)}; \
post = DetPostprocessor(grid_stride=16); print(post(raw).shape)"

python -c "from src.losses.focal_loss import FocalLoss; from src.losses.smoothl1_loss import SmoothL1Loss; \
import torch; raw = {'cls': torch.randn(2,4,14,14), 'box': torch.randn(2,4,14,14)}; \
target = {'cls': torch.zeros(2,4,14,14), 'box': torch.zeros(2,4,14,14), 'pos_mask': torch.zeros(2,1,14,14)}; \
print(FocalLoss()(raw, target).item(), SmoothL1Loss()(raw, target).item())"

python -c "from experiments.configs import CONFIGS; print(len(CONFIGS)); \
print([c['backbone'] for c in CONFIGS if c['method'] == 'det'])"

python scripts/train.py --method det --backbone custom --head detection --device cpu \
  --train_size 2 --valid_size 2 --batch_size 2 --max_epochs 1 --patience 1 --num_workers 0 \
  --output_dir /tmp/det_smoke

python scripts/train.py --method det --backbone custom --head detection --representation point \
  --device cpu --train_size 2 --valid_size 2 --batch_size 2 --max_epochs 1 --patience 1 \
  --num_workers 0 --output_dir /tmp/det_point_smoke
```

검증 결과에서는 `MultiScaleNeck`의 backbone별 fusion 단계 수와 출력 shape, `DetectionHead`/
`DetModel`의 `(cls, box)` shape(표현별 box 채널 수 포함), ViT 계열 backbone에서의 `ValueError`,
`DetPreprocessor`/`DetPostprocessor`의 shape와 `len(target)` 정합성, `FocalLoss`/`SmoothL1Loss`의
스칼라 loss 값, `experiments/configs.py` det config 개수와 backbone/representation 목록, box와
point 두 표현 모두에서 CPU smoke train의 loss/metric(`cls`, `box`, `iou`) 보고 여부를 확인한다.
검증 후 `/tmp/det_smoke`, `/tmp/det_point_smoke` 산출물은 삭제한다.
