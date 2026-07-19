---
tags: [roi-corner-detection, model, architecture, composition, ablation, ssot]
status: canonical
created: 2026-07-15
updated: 2026-07-19
---

# 모델 재조립 카테고리 및 비교 설계

이 문서는 PMD OLED fringe 영상의 네 가상 corner 검출에 대한 프로젝트 전체 설계의 단일 기준
문서다. 공통 계약, 도메인 제약, data stage, method registry, model 조립, 평가와 ablation 설계를
함께 정의한다. Pretrained weight의 파일 목록과 출처는
[backbones.md](../references/backbones.md)에서 사실 정보로만 관리한다.

## 1. 문서 governance와 공통 계약

### 1.1. Canonical status

이 문서는 architecture, method registry, experiment comparison의 유일한 SSOT다. 구현, config와
새 문서는 이 문서의 용어와 계약을 따라야 한다. 이전 방법론 및 model 설계 문서는 historical
reference이며 이 문서와 충돌할 때는 이 문서가 우선한다.

문서 상태는 다음과 같이 구분한다.

| 문서 | 상태 | 역할 |
|---|---|---|
| `docs/architecture/model-assembly.md` | canonical | 프로젝트 전체 설계와 비교 기준 |
| `docs/references/backbones.md` | reference | weight 출처, checksum과 파일 정보 |
| `docs/deprecated/*.md` | deprecated | 이전 후보안과 historical mapping |

### 1.2. 범위와 용어

이 문서는 같은 component를 공유하는 model을 조립 카테고리로 묶고 변경한 component가 성능에
미치는 영향을 분리해 비교한다. Python class 구현은 이 문서의 범위에 포함하지 않지만 public
registry와 component contract는 이 문서에서 확정한다.

실험 항목은 다음 수준으로 구분한다.

| 수준 | 의미 | 예시 |
|---|---|---|
| method | corner를 예측하는 핵심 표현과 원리 | `reg`, `seg`, `det`, `heatmap`, `line` |
| model | component의 architecture 조합 | `CustomBackbone + plain decoder + mask head` |
| variant | loss, postprocess, skip, upsampling 또는 freeze 설정 | `skip=add`, `upsample=interpolate_conv` |

같은 method code가 하나의 조립 카테고리에만 속한다고 가정하지 않는다. 예를 들어 `seg`는
`CustomBackbone + SegDecoder + MaskHead`로 조립하거나 torchvision whole segmentation model로
구성할 수 있다. 두 model은 같은 target과 최종 corner 계약을 사용하지만 재조립 가능한 경계가
다르다.

### 1.3. 공통 입출력과 평가 계약

모든 method는 내부 표현과 관계없이 다음 계약을 지킨다.

| 항목 | 계약 |
|---|---|
| image input | `(B, 3, H, W)`, 기본 `H = W = 224`, ImageNet normalization |
| corner target | `(B, 4, 2)`, `[0, 1]`, `TL`, `TR`, `BR`, `BL` 순서 |
| final output | `(B, 4, 2)` corners와 표본별 success, failure reason |
| corner ordering | method 경계에서 한 번만 정규화 |
| CSV | `image_dir,image_name,x1,y1,x2,y2,x3,y3,x4,y4` |
| experiment output | `outputs/<dataset>/<method>/<model>/<exp_name>/` |

모든 raw output은 method별 postprocessor를 거쳐 표준 corner로 변환한다. 실패 가능한
postprocess는 성공 여부와 failure reason을 반환하고 evaluator는 실패 표본을 평균에서 조용히
제외하지 않는다.

공통 metric bank는 다음과 같다.

| metric | 평가 관점 | 좋은 방향 |
|---|---|---|
| Polygon IoU | 사각형 영역 일치도 | 클수록 좋음 |
| MCD | 평균 corner 좌표 오차 | 작을수록 좋음 |
| MaxCD | 가장 큰 단일 corner 오차 | 작을수록 좋음 |
| Reprojection Error | homography 기반 복원 오차 | 작을수록 좋음 |
| PCK@0.02, PCK@0.05 | 거리 임계값 안의 corner 비율 | 클수록 좋음 |
| SR | 유효한 네 corner를 반환한 비율 | 클수록 좋음 |
| CPU/GPU latency | preprocess, inference, postprocess를 포함한 비용 | 작을수록 좋음 |
| Model size | 저장과 배포 비용 | 작을수록 좋음 |

현재 공통 evaluator는 normalized corner 기준으로 `iou`, `mcd`, `maxcd`, `pck_002`, `pck_005`, `sr`을
계산해 `metrics.json`으로 저장한다. `Reprojection Error`, latency와 model size는 비교 결과 해석 전에
별도 구현한다. 공통 predictor는 같은 test split의 target과 final corner prediction을
`predictions.csv`로 저장하며, 현 dataset batch contract에는 image path가 없으므로 split-local `index`를
식별자로 사용한다.

평가와 prediction 산출물은 experiment output 경로에 다음 파일로 추가된다.

```text
outputs/<dataset>/<method>/<model>/<exp_name>/
-> history.json
-> metrics.json
-> model.pth
-> predictions.csv
```

### 1.4. 도메인 제약과 data stage

대상은 rounded OLED panel의 단일 방향 fringe 영상이며 label corner는 실제 sharp pixel corner가
아닌 네 직선 변의 연장 교점이다. 따라서 panel 내부 fringe는 영역 단서일 수 있지만 raw line
detector에는 false boundary를 만들 수 있고 `cornerSubPix`는 기본 refinement가 아니다.

공통 도메인 제약은 다음과 같다.

| 코드 | 제약 |
|---|---|
| F1 | 대상은 axis-aligned box가 아닌 convex quadrilateral이다. |
| F2 | reference quadrilateral 면적은 target-domain profile에서 약 30-46%다. 이 값은 test-time area rejection 조건이 아니다. |
| F3 | 네 corner는 image 경계 안에 있다. |
| F4 | measured data는 적고 synthetic data는 많다. |
| F5 | phase restoration을 위해 subpixel precision이 중요하다. |
| F6 | CPU latency와 model size의 배포 제약이 있다. |
| F7 | illumination, glare, vignette 변화가 존재한다. |
| F8 | panel occlusion은 없고 네 corner가 관측된다. |
| F9 | 모든 method는 공통 입출력과 평가 계약을 준수한다. |

학습 data는 다음 세 논리 stage로 관리한다.

| stage | 목적 |
|---|---|
| `public` | 공개 corner dataset에서 일반 corner 표현을 학습한다. |
| `synthetic` | fringe pattern과 광학 변동으로 target domain에 적응한다. |
| `measured` | 소량의 PMD data로 fine-tuning과 최종 평가를 수행한다. |

## 2. 재조립 관점의 분류 기준

### 2.1. 1차 축: 모델 조립 형태

1차 분류는 model 내부 component를 프로젝트 factory에서 어느 범위까지 교체할 수 있는지에 따른다.

| category | 조립 형태 | 프로젝트가 교체하는 범위 |
|---|---|---|
| A | composable custom model | backbone, adapter, decoder, neck와 head |
| B | pretrained backbone composable model | pretrained backbone, adapter, decoder와 head |
| C | external whole model | package model 외부의 adapter와 postprocessor |
| D | iterative refinement | base prediction 뒤의 refinement model |
| E | learned geometry hybrid | learned raw output 뒤의 rule-based geometry |
| F | rule-based pipeline | image processing과 geometry parameter |

Category A와 B는 같은 `FeatureBundle` 계약을 사용한다. Category C는 package 내부 encoder,
decoder와 head의 결합을 유지한다. Category D와 E는 base model 뒤에 조립되며 Category F에는
학습 가능한 backbone과 head가 없다.

### 2.2. 2차 축: 출력 표현

2차 분류는 model이 생성하는 raw output과 이를 corner로 변환하는 방식에 따른다.

| output representation | raw output | 대표 method | 기본 postprocess |
|---|---|---|---|
| coordinate | `(B, 8)` logits 또는 offsets | `reg` | sigmoid reshape 또는 offset decode |
| heatmap | `(B, 4, Hh, Wh)` | `heatmap` | soft-argmax |
| mask | `(B, 1, Hm, Wm)` logits | `seg` | four-side line fitting |
| line map | structured dense maps | `line` | grouping과 intersection |
| boxes or points | model별 detection output | `det` | selection, center decode와 ordering |
| corner offsets | `(B, 4, 2)` offsets | refinement | base corner와 offset 결합 |

조립 형태와 출력 표현은 독립적인 축이다. mask output은 composable custom model과 external whole
model에서 모두 생성할 수 있고, coordinate output은 custom backbone과 pretrained backbone에서
같은 head로 생성할 수 있다.

### 2.3. Model source와 usage

Model source와 재사용 범위는 method와 별도 축으로 기록한다.

| source | 의미 |
|---|---|
| `torchvision` | `torchvision.models` backbone 또는 whole model을 사용한다. |
| `external` | timm, Ultralytics 또는 external repository model을 사용한다. |
| `custom` | 전체 architecture를 project에서 직접 구현한다. |
| `none` | 학습 가능한 model이 없는 rule-based pipeline이다. |

| usage | 의미 |
|---|---|
| `backbone_only` | backbone만 재사용하고 task component를 직접 구현한다. |
| `whole_model` | segmentation 또는 detection model 전체를 재사용한다. |
| `adapter` | pretrained model을 동결하거나 부분 동결하고 작은 adapter 또는 head를 학습한다. |
| `from_scratch` | 전체 network를 처음부터 학습한다. |

### 2.4. Current method registry와 historical mapping

현재 registry는 다음 method를 사용한다.

| current method | 핵심 표현 | 기본 조립 또는 postprocess |
|---|---|---|
| `reg` | coordinate 또는 homography offset | coordinate head와 sigmoid 또는 offset decode |
| `seg` | binary panel mask | dense decoder, mask head와 four-side fitting |
| `det` | corner box 또는 point | custom head 또는 whole-model adapter |
| `heatmap` | four corner heatmaps | dense decoder와 soft-argmax |
| `line` | boundary geometry maps | grouping과 intersection |
| refinement | initial corner offsets | `local_stn` 또는 `gcn` |
| rule-based | contour 또는 line candidates | classical geometry pipeline |

이전 문서의 이름은 historical reference에서만 다음처럼 해석한다.

| previous name | current registry 표현 |
|---|---|
| `direct` | `reg`, `target=corners` |
| `homography` | `reg`, `target=homography_offsets` |
| `vit_direct` | `reg`, ViT 또는 Swin backbone variant |
| `foundation` | `reg`, DINOv2 backbone과 `freeze=true` variant |
| `torchseg` | `seg`, `usage=whole_model` variant |
| `torchdet`, `yolo`, `detr_box` | `det`, external whole-model variant |
| `gcn`, `local_stn` | refinement variant |
| `classical_contour`, `classical_line` | rule-based pipeline variant |

### 2.5. Complexity 기록 원칙

구현 복잡도는 architecture, training, dependency와 postprocess를 분리해 기록한다. External model은
설치 가능 여부가 아니라 trainer, checkpoint와 evaluator adapter 비용까지 포함해 평가한다.

| 등급 | 기준 |
|---|---|
| 낮음 | 단일 output, 단일 loss, 결정적 postprocess, 외부 의존이 거의 없다. |
| 중간 | dense target, decoder, 복합 loss 또는 geometry postprocess가 필요하다. |
| 높음 | 반복 refinement, 실패 가능한 postprocess 또는 외부 weight가 필요하다. |
| 매우 높음 | external repository 통합 또는 native interface와 큰 차이가 있다. |

## 3. 공통 layer block과 feature extraction

### 3.1. 입력부터 후처리까지의 공통 흐름

Composable model의 공통 흐름은 다음과 같다.

```text
raw image
-> image preprocessing
-> images: (B, 3, 224, 224)
-> backbone
-> native backbone features
-> backbone adapter
-> FeatureBundle
-> optional decoder or neck
-> prediction head
-> raw output
-> postprocessor
-> corners, success, failure_reason
```

학습에서는 image 흐름과 별도로 `BasePreprocessor`가 corner label을 method target으로 변환한다.
`BaseLoss`는 head의 raw output과 method target을 직접 사용한다.

### 3.2. ConvBlock 계약

`ConvBlock`은 custom encoder, decoder와 neck에서 공유하는 기본 convolution block이다. 특정
task의 output layer는 포함하지 않는다.

| 항목 | 계약 |
|---|---|
| input | `(B, Cin, H, W)` |
| operation | `Conv2d`, normalization, activation |
| kernel | 기본 `3 x 3` |
| stride | feature 유지에는 1, downsampling에는 2 |
| output | `(B, Cout, Ho, Wo)` |
| configurable | channel width, stride, normalization과 activation |

Downsampling은 별도 pooling을 암묵적으로 추가하지 않고 block의 stride 설정에 명시한다. 같은
`ConvBlock` 정의를 `CustomBackbone` stage와 custom detection neck에 재사용하되 각 consumer가
channel과 stride를 config로 지정한다.

### 3.3. DeconvBlock 계약

`DeconvBlock`은 decoder feature의 해상도를 복원하는 공통 block이다. 이름은 decoder block의
역할을 나타내며 `ConvTranspose2d`만을 의미하지 않는다.

지원하는 mode는 다음과 같다.

| mode | operation | 기본 여부 | 비교 목적 |
|---|---|---:|---|
| `interpolate_conv` | interpolation, `Conv2d`, normalization, activation | 기본 | 안정적인 upsampling baseline |
| `transposed_conv` | `ConvTranspose2d`, normalization, activation | 조건부 | 학습 가능한 upsampling과 artifact 비교 |

`interpolate_conv`의 기본 scale factor는 2다. decoder는 각 stage의 목표 해상도와 channel을
명시하고, 입력 크기 때문에 skip feature와 shape가 맞지 않으면 silent crop이나 fallback을 하지
않고 오류를 반환한다.

### 3.4. CustomBackbone 구조

`CustomBackbone`은 pretrained weight를 사용하지 않는 project baseline encoder다. config registry
값은 기존 계약과 같이 `custom`을 유지한다.

기본 구조는 다음과 같다.

```text
images
-> stem ConvBlock
-> encoder stage 1
-> encoder stage 2 with downsampling
-> encoder stage 3 with downsampling
-> encoder stage 4 with downsampling
-> native final feature and stage features
```

첫 baseline은 네 encoder stage와 output stride 16을 사용한다. 각 stage는 하나 이상의
`ConvBlock`으로 구성하고 channel width, block 반복 수, normalization과 activation은 config에
기록한다. `CustomBackbone`은 coordinate, segmentation과 custom detection model에서 공유하며
task-specific decoder와 head를 포함하지 않는다.

### 3.5. Backbone, adapter와 FeatureExtractor의 경계

`CustomBackbone`과 `FeatureExtractor`는 동의어가 아니다. component 경계는 다음과 같다.

| component | 입력 | 출력 | 책임 |
|---|---|---|---|
| `CustomBackbone` | image tensor | native final feature와 stage features | custom encoder 계산 |
| backbone adapter | native features | `global`, `spatial`, `stages` | 공통 의미와 layout으로 변환 |
| `FeatureSpec` | model 생성 metadata | channel, stride와 capability | 조합 가능 여부 검증 |
| `FeatureExtractor` | image tensor | `FeatureBundle` | backbone, adapter와 spec 조립 |

Adapter는 모든 backbone의 channel과 spatial size를 강제로 같게 만들지 않는다. decoder, neck와
head factory는 `FeatureSpec`을 읽고 필요한 projection을 생성한다.

## 4. Category A: Composable custom model

### 4.1. 공통 조립 구조

Category A는 `CustomBackbone`의 feature를 공통 adapter로 변환하고 task component를 조립한다.

```text
CustomBackbone
-> backbone adapter
-> FeatureBundle
-> task-specific decoder or neck
-> PredictionHead
-> raw output
```

Category A의 주요 조합은 다음과 같다.

| 설계 명칭 | 조합 |
|---|---|
| `CustomRegModel` | `CustomBackbone + adapter + gap/spatial head` |
| `CustomSegModel` | `CustomBackbone + adapter + SegDecoder + MaskHead` |
| `CustomHeatmapModel` | `CustomBackbone + adapter + SegDecoder + HeatmapHead` |
| `CustomDetModel` | `CustomBackbone + adapter + multi-scale neck + DetectionHead` |

### 4.2. CustomRegModel 조합

`CustomRegModel`은 decoder 없이 `FeatureBundle`의 `global` 또는 `spatial` field에서 coordinate를
예측한다.

| variant | feature | head 구성 | raw output |
|---|---|---|---|
| `gap` | `global` | dropout과 linear projection | `(B, 8)` |
| `spatial` | `spatial` | projection, adaptive spatial pooling과 MLP | `(B, 8)` |

두 variant는 같은 `CustomBackbone`, corner target, loss와 postprocess를 사용한다. 비교에서는
global aggregation과 spatial information 유지의 차이를 검증한다.

### 4.3. CustomSegModel 조합

`CustomSegModel`은 `FeatureBundle` 뒤에 독립 `SegDecoder`와 `MaskHead`를 조립한다.

```text
FeatureBundle.spatial and optional stages
-> SegDecoder
-> decoded spatial feature
-> MaskHead
-> mask logits: (B, 1, Hm, Wm)
```

`MaskHead`는 최종 channel projection만 담당한다. 해상도 복원과 skip fusion은 `SegDecoder`의
책임이며 target 생성, BCE와 Dice loss, threshold와 geometry fitting은 model 밖에서 관리한다.

### 4.4. CustomHeatmapModel 조합

`CustomHeatmapModel`은 `FeatureBundle.stages`를 사용해 dense decoder와 corner별 heatmap head를
조립한다. 첫 구현은 `seg`의 U-Net additive skip decoder를 재사용하고 최종 projection만
`HeatmapHead`로 교체한다.

```text
FeatureBundle.stages
-> SegDecoder
-> decoded spatial feature
-> HeatmapHead
-> heatmap logits: (B, 4, Hh, Wh)
```

`HeatmapHead`는 네 corner channel로 projection만 담당한다. target 생성은 normalized corner를
corner별 Gaussian heatmap으로 rasterize하며, loss는 sigmoid heatmap과 target 사이의 MSE를 사용한다.
postprocessor는 soft-argmax로 `(B, 4, 2)` corner를 복원한다. Token-only ViT 계열은 token-to-spatial
adapter가 필요하므로 첫 heatmap 구현에서는 제외한다.

### 4.5. CustomDetModel 조합

`CustomDetModel`은 `FeatureBundle.stages`를 사용해 multi-scale neck과 detection head를 조립한다.

| component | 책임 |
|---|---|
| multi-scale neck | stage channel projection과 multi-scale feature 생성 |
| detection head | corner class, box 또는 point raw output 생성 |
| detection postprocessor | confidence selection, center decode와 corner ordering |

Custom detection은 external YOLO, torchvision detector나 DETR whole model과 구분한다. 외부 model의
internal loss와 호출 규약은 Category C의 adapter가 처리한다.

`CustomDetModel`의 raw output 표현은 `head` 파라미터로 `box`(기본값)와 `point` 중 선택한다. 다른
method의 `head`가 network branch/module 선택인 것과 달리, det는 `DetectionHead` 하나만 사용하므로
`head`가 그 안의 raw output 채널 구성을 선택하는 역할을 겸한다.

| variant | box_conv 채널 | raw output |
|---|---|---|
| `box`(기본값) | 4 | corner class별 center offset과 box width/height |
| `point` | 2 | corner class별 center offset만 |

두 variant 모두 grid cell마다 corner class별 objectness(classification)와 center offset을
예측하며, `box`는 여기에 box width/height 회귀를 추가한다. box width/height는 corner가 실제
크기를 가진 object가 아니라 line-intersection point라는 점을 반영해 학습을 위한 인위적
placeholder 값(기본값 0.1, 정규화 좌표계 기준)으로 취급하며, 이후 ablation 대상으로 남긴다.
detection postprocessor는 confidence가 가장 높은 cell의 center offset만으로 corner를 decode하므로
box width/height는 두 variant 모두에서 최종 corner 좌표에 영향을 주지 않는다. grid 해상도는
`grid_stride` 파라미터로 노출하며 기본값은 16이다(6.2절 지원 대상 9개 backbone이 모두 stride 16
stage를 직접 보유한다). box regression은 corner class에 무관한 채널을 공유한다.

향후 Category C(외부 whole-model detector, YOLO/torchvision detector/DETR)를 추가하더라도 해당
adapter는 항상 box 좌표만 출력하므로 `head`는 `box`로 고정된다. `seg`의 `TorchSegModel`이
`head="mask"`로 고정되는 것과 같은 패턴이며, `head`에 Category C 전용 옵션이 늘어나지 않는다.

### 4.6. 공통 block 재사용 관계

Custom model별 block 재사용 관계는 다음과 같다.

| component | custom `reg` | custom `seg` | custom `heatmap` | custom `det` |
|---|---:|---:|---:|---:|
| `ConvBlock` | 사용 | 사용 | 사용 | 사용 |
| `DeconvBlock` | 미사용 | 사용 | 사용 | neck에 따라 조건부 |
| `CustomBackbone` | 사용 | 사용 | 사용 | 사용 |
| backbone adapter | 사용 | 사용 | 사용 | 사용 |
| dense decoder | 미사용 | 사용 | 사용 | 미사용 |
| multi-scale neck | 미사용 | 미사용 | 미사용 | 사용 |
| task head | coordinate | mask | heatmap | detection |

같은 `CustomBackbone` checkpoint를 여러 method에 그대로 재사용한다는 의미는 아니다. target과
gradient가 다르므로 공정 비교에서는 같은 initialization을 사용하고 method별로 별도 학습한다.

## 5. CustomSegModel과 decoder variant

`CustomSegModel`은 U-Net architecture로 고정하지 않는다. U-Net은 교체 가능한 `SegDecoder`
variant이며 skip connection은 별도의 실험 속성으로 기록한다.

기본 config 표현은 다음과 같다.

```yaml
model:
  architecture: composable
  backbone: custom
  decoder:
    name: unet
    upsample: interpolate_conv
    skip_connection: add
  head: mask
```

### 5.1. Plain decoder

`decoder.name=plain`은 마지막 `spatial` feature만 사용한다. encoder의 `stages`를 skip feature로
사용하지 않는다.

```text
FeatureBundle.spatial
-> repeated DeconvBlocks
-> decoded feature
-> MaskHead
```

Plain decoder는 가장 작은 dense baseline이며 multi-scale encoder feature의 효과를 측정하는 기준이
된다. `skip_connection`은 `none`만 허용한다.

### 5.2. U-Net additive skip decoder

`decoder.name=unet`, `skip_connection=add`는 decoder feature와 같은 해상도의 encoder stage를
channel projection한 뒤 element-wise addition으로 결합한다.

```text
decoder feature
-> DeconvBlock
-> add projected encoder stage
-> ConvBlock
```

Additive skip은 fusion 뒤 channel 수를 증가시키지 않는다. skip connection의 기본 후보로 사용하고
plain decoder와 첫 기준 비교를 수행한다.

### 5.3. U-Net concatenation skip decoder

`decoder.name=unet`, `skip_connection=concat`은 decoder feature와 projected encoder stage를 channel
dimension으로 결합한 뒤 `ConvBlock`으로 projection한다.

```text
decoder feature
-> DeconvBlock
-> concatenate projected encoder stage
-> ConvBlock with channel projection
```

Concatenation은 더 많은 feature를 보존하지만 parameter, memory와 latency가 증가할 수 있다. 첫
baseline이 아니라 additive skip 이후의 ablation으로 평가한다.

### 5.4. FPN decoder

`decoder.name=fpn`은 여러 encoder stage에 lateral projection을 적용하고 top-down feature와
결합한다. segmentation과 custom detection에서 multi-scale 설계 원칙을 공유할 수 있지만 각
task의 neck 또는 decoder instance와 head는 분리한다.

FPN은 plain과 U-Net baseline이 안정된 뒤 조건부 variant로 평가한다. ViT와 DINOv2처럼 native
multi-stage feature가 없는 backbone에는 별도 intermediate feature 계약 없이 적용하지 않는다.

### 5.5. Skip connection ablation

허용 decoder 조합은 다음과 같다.

| decoder | skip 설정 | 입력 feature | 초기 상태 |
|---|---|---|---|
| `plain` | `none` | 마지막 `spatial` | 기준 지원 |
| `unet` | `add` | `spatial`, `stages` | 기본 후보 |
| `unet` | `concat` | `spatial`, `stages` | 추가 ablation |
| `fpn` | lateral connection | 여러 `stages` | 조건부 |

Skip connection 효과를 비교할 때 고정하는 조건은 다음과 같다.

- 같은 `CustomBackbone` 구조와 initialization을 사용한다.
- decoder stage 수, 목표 출력 해상도와 `MaskHead`를 고정한다.
- mask target, BCE와 Dice loss, optimizer, data split과 seed를 고정한다.
- 같은 postprocessor와 threshold를 사용한다.
- parameter 수, FLOPs, CPU/GPU latency와 peak memory를 함께 기록한다.

첫 비교는 `plain + interpolate_conv`와 `unet + add + interpolate_conv`다. `unet + concat`, FPN과
`transposed_conv`는 각각 독립 ablation으로 수행한다. mask Dice와 BCE뿐 아니라 Polygon IoU,
MCD, MaxCD, PCK와 SR을 함께 보고해 mask 품질과 최종 geometry 품질을 분리한다.

## 6. Category B: Pretrained backbone composable model

### 6.1. CNN과 Transformer adapter 조합

Category B는 pretrained backbone을 사용하지만 project의 adapter, decoder와 head를 조립한다.
`reg` method에서는 `TorchRegModel`(`TorchBackbone` 또는 `TimmBackbone` + adapter + gap/spatial
head)이 이 조합을 담당한다.

| backbone family | adapter output | 주요 조합 |
|---|---|---|
| ResNet | `global`, `spatial`, `stages` | coordinate, dense decoder와 custom detection |
| VGG | `global`, `spatial`, `stages` | coordinate와 multi-stage dense decoder |
| MobileNet/EfficientNet | `global`, `spatial`, 제한된 `stages` | coordinate와 lightweight dense model, EfficientNet-B0 U-Net decoder |
| ViT/DINOv2 | `global`, token-grid `spatial` | coordinate와 조건부 single-scale decoder |
| Swin | `global`, `spatial`, `stages` | coordinate와 multi-stage dense decoder |

Adapter는 native feature의 의미와 layout을 통일하며 channel projection과 task output은 decoder,
neck 또는 head가 담당한다.

### 6.2. Backbone-head compatibility

초기 compatibility는 다음과 같다.

| backbone family | coordinate | plain dense | U-Net/FPN dense | custom detection |
|---|---:|---:|---:|---:|
| `CustomBackbone` | 지원 | 지원 | 지원 | 지원 |
| ResNet | 지원 | 지원 | 지원 | 지원 |
| VGG | 지원 | 지원 | 지원 | 지원 |
| MobileNet/EfficientNet | 지원 | 지원 | 조건부 지원 | 조건부 |
| ViT/DINOv2 | 지원 | 조건부 | 초기 제외 | 초기 제외 |
| Swin | 지원 | 지원 | 지원 | 조건부 |

`stages`가 필요한 decoder나 neck은 해당 capability가 없는 backbone에서 생성 단계에 실패해야 한다.
Factory는 single-scale feature를 multi-stage feature처럼 조용히 복제하지 않는다.

### 6.3. CustomBackbone과 pretrained backbone 비교

Backbone 비교에서는 method, head, decoder, target, loss, postprocess, input size, optimizer와 data
split을 고정한다. Pretrained 여부, pretrained dataset, parameter 수와 latency는 결과 metadata에
기록한다.

`CustomBackbone`은 pretrained prior가 없는 project baseline이고 pretrained CNN과 Transformer는
data efficiency 가설을 검증하는 variant다. 서로 다른 pretrained 상태의 결과를 architecture 효과로만
해석하지 않는다.

### 6.4. 2단계 학습(freeze-then-unfreeze) 정책

Category A(custom backbone composable)와 Category B/C(pretrained backbone composable 또는 external
whole model)는 학습 stage 구성이 다르다. 판정 기준은 pretrained backbone이 project head/decoder/neck과
분리된 형태로 존재하는지 여부이며, 구체적인 판정 방식은 method와 model 구조마다 다르다.

| model 종류 | 학습 stage | optimizer 구성 |
|---|---|---|
| Category A composable (`CustomRegModel`, `SegModel(custom)`, `HeatmapModel(custom)`, `DetModel(custom)`) | 단일 stage | 단일 optimizer, 전체 parameter `lr=1e-4` (`DetModel`은 처음부터 backbone/head 2-group이지만 backbone이 무작위 초기화라 freeze 대상이 아님) |
| Category B composable (`TorchRegModel`, `SegModel`과 `HeatmapModel`의 pretrained backbone variant) | 2단계(`warmup_epochs`) | 1단계: non-backbone parameter만 `lr=1e-4`. 2단계: backbone `lr=1e-5`, non-backbone `lr=1e-4` |
| Category C external whole model (`TorchDetModel`, `YoloDetModel`, `DetrDetModel`) | 2단계(`warmup_epochs`) | 1단계: non-backbone parameter만 `lr=1e-4`(DETR은 classifier 포함). 2단계: backbone `lr=1e-5`, 나머지 `lr=1e-4` |

Category B/C는 pretrained backbone의 feature를 project head가 처음부터 크게 바꾸지 않도록, 첫
`warmup_epochs` epoch 동안 backbone을 freeze(`requires_grad=False`)하고 head만 학습한 뒤, 이후
epoch부터 backbone을 unfreeze해서 전체를 differential learning rate로 학습한다. Phase 전환 시
optimizer는 mutate가 아니라 새로 생성한다. Adam류 optimizer의 momentum이 이전 phase의 gradient에
오염되지 않고, param group 구성 자체가 phase마다 다르기 때문이다. `warmup_epochs=0`이면 Category B/C도
생성 시점부터 2단계의 최종 optimizer 구성으로 단일 phase 학습한다.

Backbone 식별 방식은 model 구조에 따라 다르다.

| model | backbone 식별 방식 |
|---|---|
| `TorchRegModel`, `SegModel`/`HeatmapModel`(pretrained backbone) | `self.model.extractor` 서브모듈 |
| `TorchDetModel` | `self.model.net.backbone` 서브모듈(torchvision 표준 구조) |
| `YoloDetModel` | `self.model.net.model[:-1]`(마지막 레이어 `Detect`를 제외한 `nn.ModuleList` 슬라이스) |
| `DetrDetModel` | `named_parameters()` 이름이 `net.model.backbone`으로 시작하는 parameter(이름 매칭) |

`TorchRegModel`/`SegModel`처럼 단일 `nn.Module` 서브모듈로 backbone을 식별할 수 있는 경우
`BaseWrapper.get_backbone_module()`을 오버라이드해 공통 `set_backbone_trainable()`을 그대로 쓴다.
`YoloDetModel`(레이어 리스트)과 `DetrDetModel`(이름 매칭)처럼 단일 서브모듈로 식별할 수 없는 경우
해당 wrapper가 `set_backbone_trainable()` 자체를 오버라이드한다.

이 정책은 wrapper(`BaseWrapper`의 `on_fit_start`/`on_epoch_start` hook)와 `Trainer`의 epoch loop에
구현되며 model(`nn.Module`) 자체에는 반영되지 않는다. `TorchSegModel`(torchvision segmentation
whole-model)은 backbone이 FCN/DeepLabV3/LRASPP마다 다른 방식으로 내부에 포함되어 있어 이번 범위에서
제외하며, 필요 시 별도 조사와 plan으로 다룬다.

## 7. Category C: External whole model

### 7.1. Whole-model adapter의 경계

External whole model은 encoder, decoder, neck와 head를 generic component로 강제 분해하지 않는다.
`ExternalWholeModelAdapter`가 package별 호출 규약과 native output을 공통 raw-output contract로
변환한다.

```text
images and optional native targets
-> external whole model
-> package-native output or internal loss
-> ExternalWholeModelAdapter
-> common raw-output contract
```

### 7.2. Segmentation, detection과 DETR 조합

External whole-model 대상은 다음과 같다.

| family | 예시 | 프로젝트가 교체하는 범위 |
|---|---|---|
| segmentation | FCN, DeepLabV3, LR-ASPP | output adapter와 mask postprocessor |
| detection | Faster R-CNN, RetinaNet, YOLO | class mapping과 corner postprocessor |
| set prediction | DETR box, 조건부 DETR point | query selection과 corner ordering |

`TorchSegModel`은 torchvision segmentation whole model을 Category C `seg`, `usage=whole_model`
variant로 감싼다. `SegModel`이 stage-returning backbone과 project `SegDecoder`를 조립하는 Category B
model인 반면, `TorchSegModel`은 FCN, DeepLabV3, LR-ASPP 내부 encoder, decoder와 segmentation head의
결합을 유지하고 native output의 `"out"` tensor만 `(B, 1, Hm, Wm)` mask logits contract로 변환한다.
COCO pretrained classifier는 panel class와 직접 대응하지 않으므로 local checkpoint를 load한 뒤 binary
mask classifier로 교체하고 project mask target으로 fine-tuning한다.

`TorchDetModel`은 torchvision detection whole model(Faster R-CNN ResNet-50-FPN, RetinaNet
ResNet-50-FPN, SSD300 VGG16)을 Category C `det`, `usage=whole_model` variant로 감싼다.
`TorchSegModel`과 달리 native output이 `DetModel`의 grid 기반 `{"cls", "box"}` dense map이 아니라
image당 가변 개수의 `{"boxes", "labels", "scores"}` 목록이므로, 기존 `DetPreprocessor`/
`DetPostprocessor`/`FocalLoss`/`SmoothL1Loss`를 재사용하지 않고, 같은 `preprocessor.py`/
`postprocessor.py` 파일 안에 별도 class `TorchDetPreprocessor`/`TorchDetPostprocessor`를 둔다.
COCO pretrained classifier는 마지막 classifier layer를 4개 corner class(+ package가 background
class를 예약하면 1개 추가)로 교체해 project corner target으로 fine-tuning한다. Faster R-CNN과
SSD300은 label 0을 background로 예약하므로 corner class `c`를 label `c + 1`로, RetinaNet은
background class가 없으므로 corner class `c`를 label `c` 그대로 매핑한다(package별 `label_offset`).

Whole model의 internal loss를 사용하는 경우 `BaseWrapper` adapter가 loss dictionary를 공통 trainer에
연결한다. Evaluator에는 package-native output을 직접 전달하지 않는다. torchvision detection whole
model은 `train()` mode에서만 `(images, targets)`를 함께 받아 native loss dict를 반환하고 `eval()`
mode에서는 항상 예측 목록만 반환하므로, `TorchDetWrapper`는 `BaseWrapper.train_step`/`eval_step`을
override해 이 비대칭을 흡수한다. validation loop에서는 native loss를 얻을 수 없으므로 valid loss
column은 0으로 남고, 조기 종료는 공통 `PolygonIoU` metric만 사용한다.

`YoloDetModel`은 Ultralytics YOLOv8-Nano(`ultralytics.nn.tasks.DetectionModel`)를 Category C `det`,
`usage=whole_model` variant로 감싼다. `TorchDetModel`과 달리 backbone, neck과 detection head가
하나의 anchor-free single-stage 구조로 결합되어 있고, native output도 torchvision의 가변 개수
`{"boxes", "labels", "scores"}` 목록이 아니라 `train()` mode에서 raw dict
`{"boxes", "scores", "feats"}`(anchor별 undecoded box regression과 class logit)를, `eval()` mode에서
`(decoded_tensor, raw dict)` 2-tuple을 반환하므로, 같은 `preprocessor.py`/`postprocessor.py` 파일
안에 별도 class `YoloDetPreprocessor`/`YoloDetPostprocessor`를 둔다. COCO pretrained classifier는
`Detect` head의 per-scale classification branch(`cv3`) 마지막 `Conv2d`만 4개 corner class로 교체하고,
class-agnostic box regression branch(`cv2`)는 그대로 재사용해 project corner target으로
fine-tuning한다. corner는 실제 넓이를 가진 object가 아니므로 `TorchDetPreprocessor`처럼 고정 크기
normalized pseudo-box(`box_size`)로 변환해 Ultralytics native loss(`v8DetectionLoss`, box/cls/dfl
3-항 합) 입력을 구성한다.

Ultralytics native loss는 `DetectionModel.loss(batch, preds=...)`로 노출되며 `train()`/`eval()` 양쪽
raw dict를 그대로 받아들이므로, `YoloDetWrapper`는 `TorchDetWrapper`와 달리 validation loop에서도
native loss(box/cls/dfl)를 채운다. box 좌표 decode와 NMS는 eval-mode `decoded_tensor`에 대해서만
수행하며, `YoloDetPostprocessor`가 class별 최고 score box의 중심점을 공통 `(N,4,2)` corners contract로
변환한다.

`DetrDetModel`은 Hugging Face `transformers.DetrForObjectDetection`을 Category C `det`,
`usage=whole_model` variant로 감싼다. 프로젝트는 `/mnt/d/backbones/facebook-detr-resnet-50` local
snapshot을 `local_files_only=True`로 로드하고, COCO classifier를 4-class corner classifier로
교체한다. Hugging Face DETR가 제공하는 Hungarian matching 기반 native loss를 `DetrDetWrapper`에서
train과 validation 양쪽에 연결한다. 같은 `preprocessor.py`/`postprocessor.py` 파일 안의 별도 class인
`DetrDetPreprocessor`는 corner를 고정 크기 pseudo-box label로 변환하고, `DetrDetPostprocessor`는
no-object class를 제외한 corner class별 최고 score query를 선택하고, 해당 query의 normalized box
center를 공통 `(N,4,2)` corner contract로 변환한다. Fine-tuning은 pretrained DETR 본체의 box
output이 불안정해지지 않도록 backbone, transformer와 새 classifier에 서로 다른 learning rate를
적용하고 gradient clipping을 사용한다.

### 7.3. 교체 가능한 요소와 제한 사항

External whole model에서도 preprocess, final postprocess와 refinement는 비교할 수 있다. 반면 package
내부 encoder나 decoder만 교체하는 실험은 별도 integration 없이 composable model 비교에 포함하지
않는다.

Whole model의 성능 비교에는 weight 출처, pretrained task, dependency version, license, model size와
end-to-end latency를 기록한다.

## 8. Category D: Iterative refinement

### 8.1. Base prediction과 refinement 조합

Refinement는 image만 입력받는 base method가 아니라 image와 initial corners를 함께 사용한다.

```text
base model
-> initial corners
image and initial corners
-> RefinementModel
-> corner offsets
-> refined corners
```

Base prediction과 refinement prediction을 별도 checkpoint와 결과 column으로 기록해 base 성능과
추가 이득을 분리한다.

### 8.2. Local STN과 GCN

Refinement variant의 차이는 다음과 같다.

| refinement | 입력 정보 | 출력 | 주요 비용 |
|---|---|---|---|
| `local_stn` | corner 주변 local image feature | local offsets | ROI sampling과 local encoder |
| `gcn` | initial polygon과 image feature | iterative offsets | graph 연산과 반복 step |

`local_stn` patch는 가상 corner 주변의 두 직선 변을 포함할 만큼 충분히 커야 한다. Sharp physical
corner가 존재한다고 가정하지 않는다.

### 8.3. Base method compatibility

Refinement compatibility는 다음과 같다.

| base output | `local_stn` | `gcn` |
|---|---:|---:|
| coordinate regression | 지원 | 지원 |
| heatmap corners | 지원 | 지원 |
| segmentation fitting corners | 지원 | 지원 |
| detection corners | 지원 | 지원 |
| 실패한 postprocess | 미적용 | 미적용 |

같은 stored base prediction에 refinement를 적용하고 base model을 다시 학습하지 않는다. Joint
training은 별도 experiment로 분리한다.

## 9. Category E: Learned geometry hybrid

### 9.1. Learned output과 geometry postprocess

Hybrid는 별도 backbone architecture가 아니라 learned raw output과 rule-based geometry의 조합이다.

```text
learned dense model
-> mask or line raw output
-> rule-based geometry postprocessor
-> corners, success, failure_reason
```

Model accuracy와 geometry postprocess accuracy를 분리하기 위해 raw prediction을 저장하고 같은 raw
output에 여러 postprocessor를 적용한다.

### 9.2. Segmentation postprocess variant

초기 segmentation postprocess 비교는 다음과 같다.

| postprocess | 핵심 계산 | 예상 실패 원인 |
|---|---|---|
| four-side fitting | 네 직선 변 fitting과 intersection | boundary sample 부족, 잘못된 side grouping |
| contour approximation | contour polygon 근사 | rounded corner와 holder 접촉부 |
| 조건부 line refinement | mask boundary band 내부 line fitting | fringe 또는 glare에 의한 false line |

Threshold와 geometry parameter는 validation set에서 확정하고 test set에서 변경하지 않는다.

### 9.3. 동일 checkpoint 기반 비교

Postprocess 비교는 같은 mask logits 또는 저장된 probability map을 입력으로 사용한다. 각 variant는
성공 표본의 정확도만 보고하지 않고 전체 SR, 실패 원인 분포와 end-to-end latency를 함께 보고한다.

## 10. Category F: Rule-based pipeline

### 10.1. Classical contour pipeline

Classical contour pipeline은 학습 model 없이 modulation mask와 boundary geometry에서 corner를
계산한다.

```text
image
-> fringe-aware preprocessing
-> modulation or panel mask
-> boundary sampling
-> four-side fitting
-> corners
```

### 10.2. Classical line pipeline

Classical line pipeline은 fringe suppression 뒤에 panel boundary line을 선택한다.

```text
image
-> fringe suppression
-> boundary-focused line candidates
-> orientation grouping
-> line intersections
-> corners
```

### 10.3. 고정 parameter와 실패 처리

Classical parameter는 validation set에서 확정하고 test set에서 표본별로 변경하지 않는다. 실패
가능 조건은 명시적 reason code로 반환한다.

| failure | 예시 reason |
|---|---|
| 충분한 boundary 또는 line 후보가 없음 | `insufficient_candidates` |
| 네 side group을 구성할 수 없음 | `side_grouping_failed` |
| intersection이 image 밖에 있음 | `out_of_bounds` |
| polygon이 convex하지 않음 | `non_convex_polygon` |

## 11. 카테고리 간 compatibility와 성능 비교

### 11.1. 블록 포함 및 공유 관계

카테고리별 block 포함 관계는 다음과 같다.

| category | backbone | adapter | decoder or neck | head | geometry postprocess | refinement |
|---|---:|---:|---:|---:|---:|---:|
| A composable custom | custom | 사용 | 선택 | 사용 | method별 | 선택 |
| B pretrained composable | pretrained | 사용 | 선택 | 사용 | method별 | 선택 |
| C external whole model | model 내부 | external adapter | model 내부 | model 내부 | 사용 | 선택 |
| D iterative refinement | base model 사용 | base model 사용 | refinement 내부 | offset head | offset decode | 자체 category |
| E learned hybrid | base model 사용 | base model 사용 | base model 사용 | base model 사용 | 핵심 component | 선택 |
| F rule-based | 미사용 | 미사용 | 미사용 | 미사용 | 전체 pipeline | 조건부 |

### 11.2. 지원 조합과 금지 조합

Factory와 experiment config는 다음 조합 규칙을 적용한다.

| 요청 조합 | 처리 |
|---|---|
| `gap`과 `global` capability | 허용 |
| `spatial`과 `spatial` capability | 허용 |
| U-Net/FPN과 `stages` capability | 허용 |
| U-Net/FPN과 `stages=null` | 생성 오류 |
| `plain` decoder와 `skip_connection=add/concat` | 생성 오류 |
| coordinate head와 dense decoder | 생성 오류 |
| external whole model과 generic internal head 교체 | 생성 오류 |
| 실패한 base corners와 refinement | 표본별 미적용 |

지원하지 않는 조합은 silent fallback하지 않는다. 오류에는 요청한 component, 필요한 capability와
실제 `FeatureSpec`을 포함한다.

### 11.3. 카테고리별 성능 가설

다음 내용은 benchmark 결과가 아니라 검증할 가설이다.

| 조합 | 정확도와 강건성 가설 | 배포 비용 가설 | 주요 위험 |
|---|---|---|---|
| custom coordinate | dominant panel에서 강한 기준선 | 작고 빠름 | local boundary 정보 손실 |
| heatmap | corner confidence와 위치 표현에 유리 | decoder 비용 발생 | heatmap 해상도 한계 |
| plain segmentation | mask 기반 geometry 분리에 유리 | 중간 | 경계 세부 정보 손실 |
| U-Net segmentation | 고해상도 boundary 복원 가능 | memory와 latency 증가 | fringe texture 전달 |
| custom detection | corner별 confidence 제공 | neck과 head 비용 | small box와 ordering 문제 |
| external whole model | pretrained prior 활용 | 대체로 큼 | 원래 task와 interface 불일치 |
| refinement | systematic corner bias 감소 가능 | 추가 latency | 잘못된 initial corner 의존 |
| classical | 빠르고 설명 가능 | 가장 작음 | illumination과 threshold 민감도 |

### 11.4. 단계별 ablation matrix

비교 실험은 다음 순서로 수행한다.

| 단계 | 고정 요소 | 변경 요소 | 목적 |
|---|---|---|---|
| 1 | `CustomBackbone`, target, loss와 postprocess | `gap`, `spatial` | global과 spatial head 비교 |
| 2 | `CustomBackbone`, mask head와 training | plain, U-Net add | skip connection 기본 효과 |
| 3 | U-Net stage와 training | add, concat | skip fusion 방식 비교 |
| 4 | U-Net add와 training | `interpolate_conv`, `transposed_conv` | upsampling 방식 비교 |
| 5 | head와 training | custom, ResNet, ViT, Swin backbone | backbone prior와 architecture 비교 |
| 6 | raw mask checkpoint | geometry postprocessor | model과 postprocess 효과 분리 |
| 7 | stored base corners | refinement 없음, local STN, GCN | refinement 순수 이득 비교 |
| 8 | dataset과 final metric | composable, external, classical | category 간 end-to-end 비교 |

모든 단계는 Polygon IoU, MCD, MaxCD, Reprojection Error, PCK@0.02, PCK@0.05, SR, CPU/GPU
latency와 model size를 보고한다. Dense model에는 raw representation metric을 추가하되 최종 선택은
공통 corner metric과 SR을 기준으로 한다.

## 12. 모델 선택 기준과 열린 결정

### 12.1. 정확도와 subpixel precision 중심 선택

정확도 중심 선택에서는 MCD 평균만 사용하지 않는다. MaxCD, Reprojection Error, PCK@0.02와 SR을
함께 보고하고 measured data에서 일관된 개선이 있는지 확인한다. Refinement는 base model보다 모든
주요 corner metric에서 개선되면서 SR을 낮추지 않을 때 채택한다.

### 12.2. CPU latency와 model size 중심 선택

배포 후보는 preprocess, model inference와 postprocess를 포함한 end-to-end CPU latency로 비교한다.
동일 정확도 범위에서는 parameter 수, serialized model size, peak memory와 failure handling이 작은
조합을 우선한다.

### 12.3. 구현 전 확정할 항목

다음 항목은 baseline 결과를 확인한 뒤 확정한다.

- `CustomBackbone`의 stage channel width와 block 반복 수를 확정한다.
- segmentation 기본 decoder를 plain과 U-Net additive skip 중에서 선택한다.
- U-Net concatenation과 FPN을 정식 registry에 포함할지 결정한다.
- Custom detection neck과 point 또는 box head를 확정한다.
- refinement의 joint training을 별도 연구 실험으로 진행할지 결정한다.

기본 가정은 `DeconvBlock.mode=interpolate_conv`이며 additive skip U-Net은 segmentation 기본 후보일
뿐 확정 default가 아니다. 실제 default는 plain과 U-Net additive skip의 measured benchmark 이후에
선택한다.
