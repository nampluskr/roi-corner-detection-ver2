---
tags: [roi-corner-detection, model, architecture, composition]
created: 2026-07-13
updated: 2026-07-13
---

# 모델 구성 및 조립 설계

이 문서는 PMD OLED fringe 영상에서 네 가상 corner를 검출하는 model을 공통 component로
구성하는 방법을 정의한다. 방법론의 의미와 비교 축은 `docs/methods-codex.md`, 사용 가능한
pretrained weight는 `docs/backbones.md`를 기준으로 한다.

핵심 설계는 학습 model을 backbone, backbone adapter, prediction head로 조립하고, loss와
postprocess를 model 밖에서 독립적으로 관리하는 것이다. 주요 method code는 `reg`, `seg`,
`det`로 정리하고, dense geometric prediction, refinement, rule-based pipeline은 별도 구현
그룹으로 둔다.

## 1. 문서 목적과 범위

이 설계의 목적은 다음과 같다.

- backbone과 head를 문자열 설정으로 교체한다.
- 서로 다른 CNN과 Transformer의 feature를 공통 interface로 변환한다.
- backbone 비교에서 head, target, loss와 postprocess를 고정한다.
- head 비교에서 같은 backbone과 feature contract를 사용한다.
- external whole model과 refinement, rule-based pipeline의 차이를 명시한다.
- 모든 방법이 최종적으로 정규화된 네 corner와 성공 여부를 반환하게 한다.

이 문서는 model component와 조립 규칙을 정의한다. optimizer, scheduler, dataset split과
experiment output 경로는 공통 training 및 experiment 문서에서 정의한다.

## 2. 공통 모델 계약

모든 학습 model은 입력 image를 받아 method별 raw output을 반환한다. loss와 postprocessor는
raw output을 직접 사용하고, evaluator에는 표준 corner만 전달한다.

```text
images
-> BaseModel
-> raw_output
-> BaseLoss and BasePostprocessor
-> corners, success, failure_reason
```

### 2.1. 입력과 raw output

공통 입력과 최종 출력은 다음 계약을 따른다.

| 항목 | 계약 |
|---|---|
| image input | `(B, 3, H, W)`, 기본 `H = W = 224` |
| corner target | `(B, 4, 2)`, `[0, 1]`, `TL`, `TR`, `BR`, `BL` |
| coordinate raw output | `(B, 8)` logits 또는 bounded offset logits |
| heatmap raw output | `(B, 4, Hh, Wh)` |
| mask raw output | `(B, 1, Hm, Wm)` logits |
| line raw output | line center, displacement와 confidence map의 구조화된 dictionary |
| final output | `(B, 4, 2)` corners와 표본별 success, failure reason |

`BaseModel.forward(images)`는 corner ordering, sigmoid decode, contour fitting이나 line intersection을
수행하지 않는다. 이 연산은 method별 postprocessor가 담당한다.

### 2.2. FeatureBundle과 FeatureSpec

backbone의 native output shape은 서로 다르므로 모든 `FeatureExtractor`는 공통
`FeatureBundle`을 반환한다.

```text
FeatureBundle
├── global: (B, D) or null
├── spatial: (B, C, Hf, Wf) or null
└── stages: ordered feature maps or null
```

각 field의 의미는 다음과 같다.

| field | 의미 | 주요 consumer |
|---|---|---|
| `global` | image 전체를 집계한 feature vector | coordinate GAP head |
| `spatial` | 위치 정보를 보존한 마지막 feature map | coordinate spatial, heatmap, mask head |
| `stages` | 해상도가 다른 stage feature map | decoder, FPN, segmentation head |

`FeatureSpec`은 batch를 제외한 feature shape와 capability를 model 생성 시 제공한다. 최소 metadata는
`global_dim`, `spatial_channels`, `spatial_stride`, `stage_channels`, `stage_strides`다. head factory는
이 metadata로 layer 크기를 결정하고 지원하지 않는 조합을 model 생성 전에 거부한다.

### 2.3. BaseModel과 BaseWrapper의 책임

상위 component의 책임은 다음과 같이 분리한다.

| component | 책임 |
|---|---|
| `BaseModel` | image에서 raw output을 계산한다. |
| `FeatureExtractor` | backbone과 adapter를 조립해 `FeatureBundle`을 반환한다. |
| `PredictionHead` | 공통 feature를 method별 raw output으로 변환한다. |
| `BasePreprocessor` | corner label을 coordinate, heatmap, mask, box target으로 변환한다. |
| `BaseLoss` | raw output과 method target으로 loss를 계산한다. |
| `BasePostprocessor` | raw output을 표준 corner와 success로 변환한다. |
| `BaseWrapper` | model, preprocessor, loss, postprocessor와 refinement를 조립한다. |

## 3. Feature Extraction

`FeatureExtractor`는 `backbone` 문자열로 encoder와 전용 adapter를 선택한다. 가중치 경로는
backbone 이름과 분리하며, local file, package cache 또는 명시적 download directory를 사용할 수
있다.

```text
FeatureExtractor
├── backbone module
├── backbone adapter
└── FeatureSpec
```

### 3.1. Custom CNN backbone

`backbone="custom"`은 pretrained weight를 사용하지 않는 기본 baseline이다. GAN discriminator,
U-Net 또는 autoencoder encoder에서 사용하는 반복 Conv block을 참고하되 특정 task의 출력 layer는
포함하지 않는다.

기본 구조는 다음과 같다.

```text
input
-> stem convolution
-> stage 1 convolution blocks
-> stage 2 downsampling blocks
-> stage 3 downsampling blocks
-> stage 4 downsampling blocks
-> spatial feature map
```

기본 Conv block은 `3 x 3 Conv2d`, normalization, activation과 선택적 stride-2 downsampling으로
구성한다. 첫 baseline은 네 stage와 출력 stride 16을 사용하고, activation과 channel width는 config로
관리한다. `custom + coord_gap`을 pretrained prior가 없는 프로젝트 기준 model로 사용한다.

### 3.2. torchvision CNN과 Transformer backbone

초기 backbone registry는 다음 항목을 지원한다.

| backbone | native feature | 기본 adapter output |
|---|---|---|
| `resnet18`, `resnet50` | CNN stage maps | `global`, `spatial`, `stages` |
| `mobilenet_v3_small` | CNN feature map | `global`, `spatial`, `stages` |
| `efficientnet_b0` | CNN feature map | `global`, `spatial`, `stages` |
| `vit_b_16` | class token과 patch tokens | `global`, token-grid `spatial` |
| `swin_t` | hierarchical window features | `global`, `spatial`, `stages` |

추가 weight가 존재하더라도 첫 구현에서는 registry를 위 목록으로 제한한다. ResNet-34,
MobileNetV2, MobileNetV3-Large와 대형 CNN은 baseline이 안정된 뒤 같은 adapter family에 등록한다.

### 3.3. DINOv2와 timm backbone

foundation registry의 초기 항목은 `dinov2_vits14`와 `dinov2_vitb14`다. DINOv2는 patch token에서
class token과 register token을 구분한 뒤 `global`과 2D token-grid `spatial`을 만든다. 기본
foundation 실험은 backbone을 동결하고 prediction head만 학습한다.

DeiT, CaiT, DINOv2 register-token 변형과 ViT-L은 조건부 backbone이다. 이들은 architecture,
입력 해상도, token layout과 model size가 달라 전용 adapter metadata와 별도 experiment가 필요하다.

### 3.4. Backbone adapter와 공통 feature interface

adapter는 native output을 의미적으로 같은 feature field로 변환하되 모든 feature의 숫자 shape를
강제로 같게 만들지는 않는다. channel과 spatial size가 다르면 head constructor가 `FeatureSpec`을
받아 필요한 projection을 생성한다.

adapter별 동작은 다음과 같다.

| adapter | 변환 규칙 |
|---|---|
| CNN adapter | 마지막 feature map을 `spatial`, adaptive average pooling 결과를 `global`로 반환한다. |
| ViT adapter | class token 또는 pooled patch token을 `global`, patch token grid를 `spatial`로 반환한다. |
| Swin adapter | 마지막 stage를 `spatial`, 각 stage를 `stages`, pooled last stage를 `global`로 반환한다. |
| DINOv2 adapter | class와 register token을 제외한 patch token을 grid로 복원하고 pooled representation을 만든다. |

입력 해상도와 patch size가 맞지 않아 token 수를 2D grid로 복원할 수 없거나, 요청한 feature field가
없으면 adapter는 조용히 fallback하지 않고 명시적 오류를 반환한다.

### 3.5. Backbone registry와 pretrained weight

backbone registry entry는 최소한 다음 정보를 가진다.

```yaml
backbone:
  name: resnet18
  source: torchvision
  adapter: cnn
  pretrained: true
  weights: /mnt/d/backbones/resnet18-f37072fd.pth
  freeze: false
```

`name`은 architecture 식별자이며 파일명이나 절대 경로가 아니다. 다른 PC에서는 `weights`만
변경하고 experiment identifier의 backbone 이름을 유지한다. weight URL, size와 SHA-256은
`docs/backbones.md`를 따른다.

## 4. Prediction Head

head는 필요한 feature field를 선언하고 raw output만 반환한다. target 생성, loss와 corner decode는
head의 책임이 아니다.

### 4.1. Coordinate GAP head

`head="coord_gap"`은 `global` feature에 MLP 또는 linear projection을 적용해 8개 coordinate logits를
출력한다.

```text
global feature
-> optional dropout
-> linear projection
-> eight logits
```

`custom + coord_gap`을 첫 `reg` baseline으로 사용한다. 모든 backbone이 안정적으로 `global`을
제공할 수 있어 가장 넓은 backbone 비교에 적합하다.

### 4.2. Coordinate spatial head

`head="coord_spatial"`은 `spatial` feature map의 위치 정보를 일부 보존하면서 8개 logits를 출력한다.
구현은 channel projection, 제한된 spatial pooling과 MLP로 구성한다. 입력 해상도 변화에 대응하도록
adaptive pooling을 사용하고 고정 `Hf x Wf` flatten에 의존하지 않는다.

### 4.3. Heatmap head

`head="heatmap"`은 네 corner별 heatmap을 출력한다. CNN 또는 Swin의 `spatial`과 `stages`를 우선
사용하며 decoder가 target heatmap 해상도로 upsampling한다. ViT와 DINOv2는 single-scale token
grid에 별도 decoder가 필요하므로 초기 compatibility 목록에서 제외한다.

### 4.4. Mask head

`head="mask"`는 binary panel mask logits를 출력한다. rounded OLED의 실제 mask를 감독하고,
postprocessor는 원호와 holder 접촉부를 제외한 네 직선 변을 fitting해 가상 corner를 계산한다.

torchvision의 FCN, DeepLabV3와 LR-ASPP 전체 model은 generic mask head 조합으로 재구성하지 않는다.
이들은 동일 mask raw-output contract를 제공하는 external whole-model adapter로 연결한다.

### 4.5. Line-map head

`head="line_map"`은 boundary center, displacement와 confidence 같은 dense representation을 출력한다.
내부 fringe가 많은 false line을 생성하므로 raw M-LSD output을 그대로 panel boundary로 사용하지
않는다. postprocessor는 modulation mask의 boundary band와 기하 조건을 함께 사용한다.

## 5. 주요 방법 그룹

프로젝트의 주된 method code는 `reg`, `seg`, `det`다. 기존의 `direct`, `homography`,
`vit_direct`, `foundation`, `torchseg`, `torchdet`, `yolo`, `detr_box`, `detr_point`는 독립 최상위
method가 아니라 세 그룹의 target, model 또는 training variant로 기록한다.

### 5.1. `reg`: Feature-based coordinate prediction

`reg`는 backbone feature에서 coordinate 또는 bounded homography offset을 회귀한다.

| variant 축 | 값 |
|---|---|
| backbone | `custom`, CNN, ViT, Swin, DINOv2 |
| head | `coord_gap`, `coord_spatial` |
| target | `corners`, `homography_offsets` |
| loss | Wing, SmoothL1 |
| postprocess | sigmoid reshape, canonical offset decode |
| refinement | none, local STN, GCN |

기존 이름은 다음처럼 흡수한다.

| 기존 code | `reg`에서의 표현 |
|---|---|
| `direct` | `target=corners` |
| `homography` | `target=homography_offsets` |
| `vit_direct` | `backbone=vit_b_16` 또는 `swin_t` |
| `foundation` | `backbone=dinov2_*`, `freeze=true` |

### 5.2. `seg`: Segmentation

`seg`는 binary panel mask를 예측한 뒤 mask boundary의 four-side line fitting으로 가상 corner를
복원한다.

| variant 축 | 값 |
|---|---|
| model usage | composable backbone과 mask head, torchvision whole segmentation model |
| target | binary panel mask |
| loss | BCE와 Dice |
| postprocess | four-side line fitting, 조건부 contour ablation |
| refinement | none, local STN |

기존 `torchseg`는 `seg`의 `usage=whole_model` variant다. 기존 `hybrid`는 새로운 top-level method로
두지 않고 learned mask와 rule-based four-side fitting을 결합한 `seg` postprocess variant로 둔다.

### 5.3. `det`: Detection

`det`는 corner를 네 class의 작은 box 또는 query point로 검출한다.

| variant | 구현 |
|---|---|
| custom grid | generic backbone과 custom detection head |
| torchvision | Faster R-CNN 또는 RetinaNet whole-model adapter |
| YOLO | Ultralytics whole-model adapter |
| DETR box | pretrained DETR whole-model adapter와 4-class box head |
| DETR point | DETR backbone과 transformer를 초기화하고 point query head를 새로 학습 |

Detection model은 training 호출 규약과 internal loss가 다르므로 generic coordinate model에 head만
교체하는 방식으로 흡수하지 않는다. 모든 variant는 postprocess 이후 같은 corner contract를
지킨다.

## 6. 별도 구현 그룹

주요 세 method와 입출력 또는 실행 방식이 다른 기능은 다음 그룹으로 관리한다.

### 6.1. Dense geometric prediction

`heatmap`과 `line`은 segmentation과 분리한다. 두 방법 모두 dense map을 출력하지만 mask가 아닌
corner probability 또는 boundary geometry를 직접 표현한다.

| method | model output | postprocess |
|---|---|---|
| `heatmap` | four corner heatmaps | soft-argmax |
| `line` | center, displacement와 confidence maps | boundary grouping과 intersection |

### 6.2. Refinement

`gcn`과 `local_stn`은 image만 받는 base method가 아니라 image와 initial corners를 함께 받는다.
따라서 `PredictionHead` variant가 아니라 `RefinementModel`로 구현한다.

```text
image, initial corners
-> refinement model
-> corner offsets
-> refined corners
```

`local_stn`은 가상 corner 주변에 실제 sharp intersection이 없을 수 있으므로 두 직선 변을 관찰할
수 있는 충분한 patch 범위를 사용한다. `cornerSubPix`는 기본 refinement에 포함하지 않는다.

### 6.3. Rule-based

`classical_contour`와 `classical_line`은 학습 가능한 backbone, adapter와 head가 없다. 이들은
`RuleBasedPipeline`으로 구현하고 같은 final corner contract와 metric bank를 사용한다.

| method | 핵심 pipeline |
|---|---|
| `classical_contour` | fringe modulation mask, boundary samples, four-side fitting |
| `classical_line` | fringe suppression, boundary-focused lines, orientation grouping, intersections |

### 6.4. Hybrid pipeline

hybrid는 별도 model architecture가 아니라 learned output과 rule-based postprocess의 조합이다.
초기 hybrid는 `seg` mask에 four-side line fitting을 적용한다. 같은 mask checkpoint에 여러
postprocessor를 적용해 model 성능과 geometry postprocess 성능을 분리해 평가한다.

## 7. Model Factory와 Compatibility

model factory는 문자열 설정을 registry entry로 해석하고, backbone adapter와 head의 capability를
검증한 뒤 model을 조립한다.

### 7.1. `backbone`과 `head` 문자열 registry

초기 공개 문자열은 다음 값으로 제한한다.

```text
backbone:
  custom
  resnet18
  resnet50
  mobilenet_v3_small
  efficientnet_b0
  vit_b_16
  swin_t
  dinov2_vits14
  dinov2_vitb14

head:
  coord_gap
  coord_spatial
  heatmap
  mask
  line_map
```

알 수 없는 문자열, duplicate alias나 지원하지 않는 weight 형식은 생성 단계에서 명시적 오류로
처리한다. architecture 이름과 weight 파일명을 하나의 문자열에 결합하지 않는다.

### 7.2. 허용 backbone-head 조합

초기 compatibility는 다음과 같다.

| backbone family | `coord_gap` | `coord_spatial` | `heatmap` | `mask` | `line_map` |
|---|---:|---:|---:|---:|---:|
| custom CNN | 지원 | 지원 | 지원 | 지원 | 지원 |
| ResNet | 지원 | 지원 | 지원 | 지원 | 지원 |
| MobileNet/EfficientNet | 지원 | 지원 | 지원 | 지원 | 조건부 |
| ViT/DINOv2 | 지원 | 지원 | 초기 제외 | 초기 제외 | 초기 제외 |
| Swin | 지원 | 지원 | 지원 | 지원 | 조건부 |

가능한 model 수는 단순한 전체 backbone 수와 전체 head 수의 곱이 아니라 compatibility table에서
허용한 조합 수다. 지원 조합에는 shape가 다른 경우에도 `FeatureSpec` 기반 projection과 decoder를
factory가 구성한다.

### 7.3. External whole-model adapter

segmentation, detection과 DETR whole model은 generic `FeatureExtractor + PredictionHead`로 강제
분해하지 않는다. `ExternalWholeModelAdapter`가 package별 training, evaluation 호출 차이와 native
output을 흡수하고 `BaseWrapper`에는 공통 raw-output contract를 제공한다.

## 8. 설정과 실험 조립

method, model, loss, postprocess와 refinement는 서로 다른 config section으로 기록한다.

### 8.1. `reg` baseline: `custom + coord_gap`

첫 baseline 설정은 다음과 같다.

```yaml
method:
  code: reg
  target: corners

model:
  architecture: composable
  backbone: custom
  head: coord_gap
  pretrained: false

loss:
  name: wing

postprocess:
  name: sigmoid_reshape

refinement:
  name: none
```

### 8.2. Backbone 비교 설정

backbone 비교에서는 `method=reg`, `target=corners`, `head=coord_gap`, loss, postprocess, input size,
optimizer와 data split을 고정한다. pretrained backbone과 from-scratch custom CNN의 차이는 별도
metadata와 결과 column으로 기록한다.

### 8.3. Head 및 postprocess 비교 설정

head 비교에서는 backbone, initialization, target과 training condition을 고정한다. postprocess
비교에서는 같은 raw-output checkpoint를 사용한다. `seg`의 four-side fitting과 다른 mask
postprocess는 같은 mask logits 또는 저장된 prediction을 입력으로 사용한다.

## 9. 구현 순서와 검증 기준

구현 순서는 다음과 같다.

1. `FeatureBundle`, `FeatureSpec`, backbone adapter와 head capability contract를 정의한다.
2. `backbone=custom`, `head=coord_gap`인 `reg` baseline을 구현한다.
3. ResNet-18 adapter를 추가하고 custom CNN과 같은 head로 forward 및 training을 비교한다.
4. `coord_spatial`, `heatmap`, `mask` head와 decoder를 순서대로 추가한다.
5. torchvision CNN, ViT, Swin, DINOv2 adapter를 compatibility table에 따라 추가한다.
6. `seg` whole-model adapter와 rule-based four-side fitting을 연결한다.
7. detection whole-model adapter, refinement와 classical pipeline을 별도 그룹으로 추가한다.

component 검증은 다음 조건을 만족해야 한다.

- 모든 registered backbone은 선언한 `FeatureSpec`과 실제 feature shape가 일치한다.
- 모든 허용 backbone-head 조합은 batch size 1과 2에서 forward가 성공한다.
- 지원하지 않는 조합과 알 수 없는 문자열은 생성 시 명시적 오류를 반환한다.
- coordinate head는 `(B, 8)`, heatmap과 mask head는 선언한 dense shape를 반환한다.
- postprocessor 이후 corner shape, 범위와 순서는 공통 출력 계약을 만족한다.
- frozen backbone은 gradient를 생성하지 않고 head는 gradient를 생성한다.
- local weight가 없거나 checksum이 다르면 무시하지 않고 실패 원인을 반환한다.
- 같은 seed와 설정의 model initialization과 synthetic batch forward는 재현 가능해야 한다.
