---
tags: [roi-corner-detection, methodology, classification]
status: deprecated
superseded_by: ../architecture/model-assembly.md
created: 2026-07-12
updated: 2026-07-15
---

# 방법론 분류 체계

이 문서는 historical 후보 catalog다. 현재 프로젝트 설계의 기준은
[model-assembly.md](../architecture/model-assembly.md)이며, 이 문서의 method 분류와 우선순위는
새 구현 또는 실험의 근거로 사용하지 않는다.

이 문서는 평면 사각형 객체의 네 코너를 검출하는 방법론을 핵심 아이디어와 구현 복잡도에
따라 분류하고, 각 방법론의 model source, backbone, loss, postprocess, refinement를 독립된
축으로 정의한다. `roi-corner-detection-ver1`의 방법론을 재구성하고 DETR, ViT/Swin 직접
회귀, STN 기반 local refinement, 순수 classical CV를 후보로 포함한다.

## 1. 분류 목적

ver1의 방법론 이름에는 서로 다른 성격의 정보가 같은 수준에 섞여 있다. `direct`와
`heatmap`은 출력 표현을 나타내고, `torchseg`는 구현 출처를 나타내며, `foundation`은
사전학습과 학습 전략을 나타낸다. 이런 목록은 구현 현황을 나열하기에는 편하지만 다음 비교를
어렵게 한다.

- 같은 출력 표현에서 backbone만 교체한 실험
- 같은 model에서 loss만 교체한 실험
- 같은 raw output에 서로 다른 postprocess를 적용한 실험
- 같은 base method에 local refinement를 추가한 실험
- 직접 구현과 library whole model 재사용의 비용 비교

따라서 독립 방법론은 코너를 어떤 중간 표현과 추론 원리로 구하는지에 따라 정의하고, model
출처와 backbone, loss, postprocess, refinement는 별도 실험 축으로 관리한다.

## 2. 공통 문제 정의

모든 방법론은 서로 다른 내부 표현을 사용하더라도 동일한 입력과 최종 출력을 공유한다.

### 2.1. 입력과 출력

공통 입출력 계약은 다음과 같다.

- 입력은 이미지 텐서 `(B, 3, H, W)`이다.
- 학습 시 정답은 정규화된 코너 텐서 `(B, 4, 2)`이다.
- 코너 순서는 `TL`, `TR`, `BR`, `BL`이다.
- 최종 출력은 정규화 범위 `[0, 1]`의 `(B, 4, 2)`이다.
- 후처리 실패가 가능한 방법론은 예측 코너와 함께 성공 여부를 반환한다.

### 2.2. 공통 평가

모든 방법론은 같은 최종 코너 형식으로 변환된 이후 다음 지표로 평가한다.

| 지표 | 평가 관점 | 좋은 방향 |
|---|---|---|
| Polygon IoU | 사각형 영역 일치도 | 클수록 좋음 |
| MCD | 평균 코너 좌표 오차 | 작을수록 좋음 |
| MaxCD | 가장 큰 단일 코너 오차 | 작을수록 좋음 |
| Reprojection Error | 호모그래피 기반 복원 오차 | 작을수록 좋음 |
| PCK | 임계 거리 안의 코너 비율 | 클수록 좋음 |
| SR | 후처리를 포함한 검출 성공률 | 클수록 좋음 |
| CPU/GPU latency | end-to-end 실행 비용 | 작을수록 좋음 |
| Model size | 저장 및 배포 비용 | 작을수록 좋음 |

정확도 지표는 추론 성공 표본만의 평균과 SR을 함께 보고해야 한다. 실패 표본을 제외한 평균만
제시하면 contour, detection, line intersection처럼 후처리 실패가 존재하는 방법론이 실제보다
좋게 평가될 수 있다.

### 2.3. PMD OLED fringe 도메인

ver1의 `data/synthetic/preview_03`과 PMD 참고 이미지를 검토한 결과, 목표 입력은 흰 문서가
아니라 stage 위에 놓인 rounded OLED panel의 단일 방향 fringe 영상이다. panel 내부에는 강한
주기 fringe, 위치별 contrast 변화, blur와 noise가 있고, 외부에는 stage texture, 조명 변화,
holder와 screw가 존재한다. 따라서 내부 fringe는 panel 영역의 단서이지만 raw line detector에
입력하면 다수의 false boundary를 생성할 수 있다.

`preview_03`의 reference quad 면적은 이미지의 약 30-46%이므로, panel이 항상 이미지의 50% 이상을
차지한다고 가정하는 데이터 검증이나 method parameter는 이 preview와 함께 사용하지 않는다. 실측
ROI 면적을 확인한 뒤 공통 도메인 제약과 합성 profile을 같은 기준으로 확정한다.

레이블의 네 점은 둥근 OLED 외곽의 실제 sharp pixel corner가 아니다. 각 직선 변을 원호 너머로
연장한 교점이 `TL`, `TR`, `BR`, `BL`이다. 이 가상 corner는 panel mask 바깥에 놓일 수 있으므로
`cornerSubPix`처럼 실제 국소 intensity corner를 전제하는 refinement는 기본 비교 대상에서
제외한다. mask 기반 방법은 contour를 네 점 polygon으로 근사하는 대신 직선 변 구간을 robust하게
fitting하고 인접 변의 교점으로 corner를 복원해야 한다.

DocTr와 DocScanner는 문서 외곽, 텍스트 및 종이의 비선형 변형을 전제로 한 document geometry
모델이다. 평면 OLED의 fringe 영역과 가상 corner를 검출하는 현재 문제에는 사전학습 도메인과
출력 표현이 맞지 않고 dense deformation supervision도 없다. 두 모델은 이 catalog와 구현
우선순위에서 제외한다.

## 3. 다축 분류 체계

하나의 실험 방법론은 다음 축의 조합으로 정의한다.

```text
Method specification
├── family
├── model
│   ├── source
│   ├── architecture
│   ├── backbone
│   ├── head
│   └── output_type
├── loss
├── postprocess
└── refinement
```

### 3.1. Method family

`family`는 코너를 추론하는 핵심 아이디어를 나타낸다.

| family | 핵심 아이디어 |
|---|---|
| `regression` | 이미지 전체 특징에서 좌표나 기하 parameter를 직접 회귀한다. |
| `dense_prediction` | heatmap, mask, line map과 같은 공간 표현을 먼저 예측한다. |
| `detection` | 코너를 box, class 또는 query로 표현해 검출한다. |
| `iterative_refinement` | 초기 코너를 반복적으로 보정한다. |
| `classical_cv` | 학습 모델 없이 영상 처리와 기하 연산으로 코너를 계산한다. |

### 3.2. Model source

`source`는 model 또는 backbone 구현의 출처를 나타낸다. `torch.models`가 아니라
`torchvision.models`가 정확한 package 경로다.

| source | 정의 | 예시 |
|---|---|---|
| `torchvision` | torchvision의 backbone 또는 whole model을 사용한다. | ResNet, MobileNetV3, DeepLabV3, RetinaNet |
| `external` | torchvision 이외 library 또는 외부 repository를 사용한다. | timm DINOv2, Ultralytics YOLO, DETR |
| `custom` | backbone을 포함한 전체 network를 프로젝트에서 직접 구현한다. | small CNN, custom line network |
| `none` | 학습 가능한 model이 없다. | contour 기반 classical CV |

`source=torchvision`과 `architecture=custom`은 동시에 성립할 수 있다. 예를 들어 ResNet
backbone 위에 직접 작성한 coordinate head를 연결한 모델은 source가 `torchvision`이고
architecture는 `custom_regressor`다.

### 3.3. Model usage

같은 source 안에서도 재사용 범위가 다르므로 다음 값을 별도로 기록한다.

| usage | 정의 |
|---|---|
| `backbone_only` | 특징 추출기만 재사용하고 head는 직접 구현한다. |
| `whole_model` | detection 또는 segmentation model 전체를 재사용한다. |
| `adapter` | 사전학습 model을 동결하거나 부분 동결하고 작은 adapter/head만 학습한다. |
| `from_scratch` | 전체 architecture와 parameter를 직접 구성하고 처음부터 학습한다. |

### 3.4. Loss와 postprocess

loss와 postprocess는 method 이름 안에 암묵적으로 넣지 않고 명시적인 실험 속성으로 둔다.

```yaml
method:
  family: dense_prediction
  model:
    source: torchvision
    usage: backbone_only
    architecture: custom_heatmap
    backbone: resnet18
    output_type: heatmap
  loss:
    name: mse
  postprocess:
    name: soft_argmax
  refinement:
    name: none
```

## 4. 구현 복잡도 기준

복잡도는 코드 길이 하나로 판단하지 않고 architecture, training, dependency, postprocess의
네 요소를 함께 평가한다.

| 등급 | 판정 기준 |
|---|---|
| 낮음 | 단일 출력과 단일 loss를 사용하며 후처리가 결정적이고 외부 의존이 거의 없다. |
| 중간 | dense target 생성, decoder, 복합 loss 또는 기하 후처리 중 하나가 필요하다. |
| 높음 | 다단계 구조, 반복 정제, 복합 loss, 실패 가능한 후처리 또는 외부 weight가 필요하다. |
| 매우 높음 | 외부 repository 통합, label 변환, custom operator 또는 원래 task와의 큰 interface 차이가 있다. |

전체 복잡도 외에도 다음 값을 개별적으로 기록하는 것이 좋다.

```yaml
complexity:
  architecture: medium
  training: high
  dependency: low
  postprocess: medium
  overall: high
```

## 5. Group A: 전역 좌표 및 기하 회귀

이 그룹은 이미지 전체의 특징에서 네 코너 좌표 또는 이에 대응하는 기하 parameter를 한 번에
예측한다. 출력과 후처리가 단순해 baseline 및 backbone ablation에 적합하다.

### 5.1. Direct coordinate regression

`direct`는 backbone 특징을 집계한 뒤 8개 좌표를 직접 회귀한다.

```text
image
-> backbone
-> GAP or spatial head
-> eight coordinate logits
-> sigmoid
-> four corners
```

권장 사양은 다음과 같다.

| 항목 | 정의 |
|---|---|
| code | `direct` |
| output | 8 coordinate logits |
| target | normalized corner coordinates |
| loss | Wing 또는 SmoothL1 |
| postprocess | sigmoid + reshape |
| refinement | none 또는 local STN |
| complexity | 낮음 |

GAP head는 가장 단순하지만 공간 위치 정보가 압축될 수 있다. spatial head는 feature map의
배치를 일부 보존하므로 좌표 회귀에 더 적합할 수 있지만 parameter와 입력 해상도 의존성이
증가한다. 가상 corner를 직접 target으로 사용하므로 rounded panel의 실제 원호 끝점을 별도로
검출할 필요가 없다.

### 5.2. Homography offset regression

`homography`는 정준 사각형의 네 꼭짓점에서 실제 코너까지의 bounded offset을 회귀한다.

```text
image
-> backbone and spatial head
-> eight offset logits
-> alpha * tanh
-> canonical corners + offsets
-> four corners
```

권장 사양은 다음과 같다.

| 항목 | 정의 |
|---|---|
| code | `homography` |
| output | 8 bounded offsets |
| target | offsets from canonical corners |
| loss | SmoothL1 |
| postprocess | tanh scaling + canonical corner addition |
| complexity | 낮음 |

정준 사각형 가정은 학습 범위를 제한해 수렴을 도울 수 있지만, 실제 코너 분포가 정준 위치에서
크게 벗어나면 offset 범위가 새로운 제약이 된다.

### 5.3. ViT/Swin 기반 직접 회귀

`vit_direct`는 direct regression의 CNN backbone을 ViT 또는 Swin Transformer로 교체한
모델 변형이다. 최종 출력, loss, postprocess가 `direct`와 같으므로 독립 family가 아니라
backbone experiment로 관리한다.

```text
image
-> ViT patch tokens or Swin feature maps
-> token/spatial aggregation
-> coordinate head
-> eight coordinate logits
-> sigmoid
-> four corners
```

두 backbone의 특성은 다음과 같다.

| backbone | 공간 표현 | 장점 | 주의점 |
|---|---|---|---|
| ViT | 고정 patch token | 전역 관계 모델링 | patch 크기와 데이터 요구량이 큼 |
| Swin | shifted local window와 계층 feature | CNN과 유사한 multi-scale 구조 | 구현 설정과 stage 선택이 복잡함 |

권장 실험은 ResNet18, ViT-B/16, Swin-T가 같은 coordinate head, loss, 입력 해상도를 사용하도록
맞추는 것이다. parameter 수와 pretrained dataset 차이가 결과에 영향을 주므로 정확도뿐 아니라
model size와 latency도 함께 비교한다.

### 5.4. Foundation adapter

`foundation`은 DINOv2와 같은 foundation backbone을 동결하고 작은 spatial head 또는 adapter만
학습한다.

```text
image
-> frozen DINOv2 patch encoder
-> patch grid reshape
-> lightweight spatial head
-> coordinates
```

소량 measured data에서 과적합을 줄일 수 있지만 추론 시 backbone 전체가 필요하므로 CPU 배포
비용이 크다. 학습 가능 parameter가 작다는 사실과 추론 model이 가볍다는 사실을 구분해야 한다.

## 6. Group B: Dense prediction

이 그룹은 코너 좌표 대신 공간적인 중간 표현을 예측한다. 좌표 정보가 feature map에 유지되지만
target 생성과 postprocess가 추가된다.

### 6.1. Heatmap keypoint detection

`heatmap`은 코너마다 하나의 probability map을 생성한다.

```text
image
-> backbone
-> upsampling decoder
-> four heatmaps
-> soft-argmax
-> four corners
```

기본 loss는 Gaussian target에 대한 MSE다. 양성 영역이 매우 작으면 focal heatmap loss도 비교할
수 있다. argmax는 pixel 격자에 묶이지만 soft-argmax는 subpixel 좌표를 직접 생성할 수 있다.
가상 corner heatmap의 중심은 panel mask 바깥에 놓일 수 있으므로 mask와 keypoint target을 같은
표현으로 취급하지 않는다.

### 6.2. Custom segmentation corner

`seg`는 fringe modulation이 나타나는 panel 내부를 binary mask로 예측하고, mask boundary의
직선 변을 fitting해 가상 corner를 복원한다.

```text
image
-> backbone and segmentation decoder
-> binary mask logits
-> threshold
-> boundary samples
-> four-side robust line fitting
-> adjacent-line intersections
-> four corners
```

기본 loss는 BCE와 Dice의 조합이다. fitting은 rounded corner arc와 holder 접촉부를 제외한 직선
구간을 사용한다. postprocess는 contour 부재, side 부족, 퇴화 사각형과 같은 실패 모드를
명시적으로 처리해야 한다.

### 6.3. Torchvision whole segmentation model

`torchseg`는 DeepLabV3, FCN, LR-ASPP 같은 torchvision model 전체를 재사용한다. `seg`와 target,
loss, postprocess가 같으므로 독립적인 추론 아이디어가 아니라 `seg`의 model variant로 볼 수
있다. 다만 직접 구현 decoder와 whole model 재사용의 비용을 비교하기 위해 catalog code는
보존할 수 있다.

### 6.4. Learned line map

`line`은 panel boundary 전용 dense line representation을 예측한 뒤 네 변을 그룹화하고 교점을
계산한다. 내부 fringe는 긴 선분을 다수 포함하므로 일반 M-LSD 출력만으로 panel boundary를
선택하는 방식은 기본 방법으로 사용하지 않는다.

```text
image
-> line detection network
-> center and displacement maps
-> line segment decoding
-> boundary line grouping
-> intersections
-> four corners
```

점보다 긴 변 전체를 학습 신호로 사용할 수 있다는 장점이 있다. 반면 line decoding, 방향
분류, boundary 선택, 평행선과 교점의 수치 안정성까지 관리해야 하므로 구현 복잡도는 높다.

## 7. Group C: 객체 검출 및 set prediction

이 그룹은 코너를 작은 객체, class가 있는 box 또는 transformer query로 표현한다.

### 7.1. Custom grid detector

`det`는 feature grid에서 objectness, box offset, 크기, corner class를 예측한다.

| 항목 | 정의 |
|---|---|
| output | grid별 objectness, box, corner class |
| loss | Focal + SmoothL1 + CrossEntropy |
| postprocess | class별 top-1 box center |
| complexity | 높음 |

단일 사각형이고 네 코너가 모두 보인다는 제약을 이용하면 일반 검출기보다 후처리를 단순화할
수 있다. 하지만 코너 box 크기는 실제 object가 아니라 학습을 위해 만든 parameter이므로 box
크기 ablation이 필요하다.

### 7.2. Torchvision detector

`torchdet`는 Faster R-CNN 또는 RetinaNet 전체를 재사용한다. model 내부 loss 규약이 일반
wrapper와 다르므로 training step adapter가 필요하다. 추론에서는 corner class별 최고 score
box의 중심을 선택한다.

### 7.3. Ultralytics YOLO detector

`yolo`는 코너를 네 class의 작은 object로 변환해 single-stage detector로 학습한다. loss는
Ultralytics 구현의 box regression, Distribution Focal Loss, classification loss를 사용하고,
postprocess는 NMS 이후 class별 box center를 선택한다.

YOLO는 설치와 실행이 비교적 쉽지만 dataset 변환, 외부 training interface, checkpoint 형식이
공통 pipeline과 다르다. 공정 비교를 위해 image split, augmentation, metric 입력을 공통 규약에
맞춰야 한다.

### 7.4. DETR corner detection

`detr`은 transformer query가 코너 집합을 직접 예측한다. DETR package 설치와 pretrained
weight 다운로드를 별도로 관리한다.

#### Box DETR

첫 번째 구현은 공개 pretrained weight를 재사용하기 쉬운 box 방식이다.

```text
image
-> CNN backbone
-> transformer encoder and decoder
-> object queries
-> corner classes and boxes
-> query selection
-> box centers
-> four corners
```

기본 loss는 classification, L1 box regression, GIoU의 조합이다. DETR은 set prediction을
사용하므로 NMS가 필요 없다. corner label은 `TL`, `TR`, `BR`, `BL`의 네 class로 정의한다.

#### Point DETR

두 번째 구현은 query가 box 대신 2D point를 직접 예측한다.

```text
image
-> backbone
-> transformer
-> four corner queries
-> four normalized points
```

고정 query가 고정 corner ID를 담당하면 Hungarian matching을 생략할 수 있다. 반대로 query
순서를 고정하지 않으면 bipartite matching으로 예측과 정답을 대응시켜야 한다. Point DETR은
box 크기라는 인위적인 parameter를 제거하지만 pretrained detection head를 그대로 활용하기
어렵다.

DETR의 권장 metadata는 다음과 같다.

```yaml
dependencies:
  package: external_detr
  weights:
    download: separate
    source: null
    checksum: null
    license: null
    cache_path: null
```

구현 순서는 pretrained Box DETR adapter, Box DETR fine-tuning, Point DETR 순으로 둔다.

## 8. Group D: 반복 및 local refinement

이 그룹은 coarse prediction을 초기값으로 받고 이미지 또는 graph 특징을 사용해 코너를
정제한다.

### 8.1. Polygon GCN

`gcn`은 네 코너를 graph node로 보고 인접 corner와 image feature를 이용해 좌표를 반복적으로
보정한다.

```text
image and initial corners
-> local feature sampling
-> graph convolution
-> coordinate offsets
-> repeated refinement
-> final corners
```

각 iteration에 deep supervision을 적용할 수 있다. 네 점만 사용하는 graph는 매우 작으므로
GCN의 장점이 제한될 수 있으며, 초기값 품질에 대한 민감도 분석이 필요하다.

### 8.2. STN 기반 local zoom-in refinement

네 코너 주변을 각각 확대하는 구조는 단일 global STN보다 `grid_sample` 또는 ROIAlign 기반
local refinement로 정의하는 편이 정확하다.

```text
image and coarse corners
-> four differentiable sampling grids
-> four zoomed corner patches
-> shared local refinement network
-> four coordinate offsets
-> refined corners
```

학습 loss는 coarse prediction과 refined prediction을 함께 감독한다.

```text
L_total = L_coarse + lambda_refine * L_refined
```

필요하면 crop 중심이 coarse corner에서 과도하게 벗어나지 않도록 consistency loss를 추가한다.
가상 corner 주변에는 실제 sharp edge intersection이 없을 수 있다. 따라서 refinement network는
원호와 holder를 포함할 수 있는 충분한 범위에서 두 직선 변의 방향과 위치를 함께 관찰해야 하며,
작은 patch만으로 `cornerSubPix`를 대체한다고 가정하지 않는다.

권장 설정 항목은 다음과 같다.

```yaml
refinement:
  name: local_stn
  patch_size: 64
  zoom_scale: 4.0
  shared_head: true
  offset_loss: wing
```

STN refinement는 `direct`, `heatmap`, `seg`, `det`, `detr` 뒤에 공통으로 연결할 수 있다.
따라서 독립 base method로만 구현하지 않고 `refinement=none/local_stn`이라는 교차 실험 축으로
관리한다.

## 9. Group E: 순수 classical CV

이 그룹은 deep learning을 완전히 배제하고 image processing과 projective geometry만으로 코너를
계산한다. backbone과 loss가 없으며 모든 parameter는 image processing 설정값이다.

### 9.1. Contour pipeline

`classical_contour`는 fringe modulation 또는 local contrast를 이용해 panel 영역을 분리한 뒤,
boundary의 직선 변을 fitting한다.

```text
image
-> illumination normalization
-> fringe modulation or local contrast map
-> panel mask and boundary samples
-> robust four-side line fitting
-> adjacent-line intersections
-> corner ordering
-> four corners
```

후보 mask는 크기만으로 선택하지 않고 convexity, image area ratio, fringe modulation support,
edge length와 angle을 함께 검사한다. rounded contour를 네 점으로 직접 근사하지 않으며, 원호와
holder 접촉부를 제외한 side sample만 사용한다.

### 9.2. Line pipeline

`classical_line`은 panel의 네 경계선을 찾고 교점으로 코너를 계산한다.

```text
image
-> fringe suppression or modulation mask
-> boundary-focused edge map
-> Hough or robust line candidates
-> orientation clustering
-> four boundary line selection
-> line intersections
-> four corners
```

원본 fringe에 Canny, LSD 또는 Hough를 바로 적용하면 내부 stripe가 다수의 긴 false line을
만든다. line 후보는 panel mask의 boundary band에 제한하고, low-pass 또는 modulation 기반
전처리로 fringe를 억제한다. line 선택은 수평과 수직이라는 image axis 가정보다 두 개의 dominant
orientation group을 찾는 방식이 perspective와 rotation에 더 강하다.

### 9.3. Classical CV의 역할

순수 CV는 다음 세 역할을 가진다.

- 학습 비용과 model size가 0인 배포 baseline
- synthetic 또는 measured data의 영상 난이도 진단 도구
- hybrid method의 postprocess 구성 요소에 대한 ablation baseline

구현 자체는 중간 수준이지만 glare, fringe, vignette, 낮은 contrast에 대한 안정화 복잡도는
높다. dataset별 threshold를 따로 조정하면 공정 비교가 깨지므로 validation set에서 확정한
parameter를 test set에 그대로 적용한다.

## 10. 방법론 마스터 표

독립 base method와 중요한 model variant를 함께 정리하면 다음과 같다.

| group | code | 핵심 표현 | model source | 기본 loss | 기본 postprocess | 복잡도 |
|---|---|---|---|---|---|---|
| Regression | `direct` | coordinates | torchvision/custom | Wing | sigmoid + reshape | 낮음 |
| Regression | `homography` | bounded offsets | torchvision/custom | SmoothL1 | canonical + tanh offset | 낮음 |
| Regression | `vit_direct` | coordinates | torchvision/external | Wing | sigmoid + reshape | 중간 |
| Regression | `foundation` | coordinates | external | Wing | sigmoid + reshape | 높음 |
| Dense | `heatmap` | four heatmaps | torchvision/custom | MSE | soft-argmax | 중간 |
| Dense | `seg` | binary mask | torchvision/custom | BCE + Dice | four-side line fitting | 중간 |
| Dense | `torchseg` | binary mask | torchvision whole model | BCE + Dice | four-side line fitting | 중간 |
| Dense | `line` | line maps | external/custom | Focal + SmoothL1 | grouping + intersection | 높음 |
| Detection | `det` | grid boxes | torchvision/custom | Focal + SmoothL1 + CE | class top-1 | 높음 |
| Detection | `torchdet` | boxes | torchvision whole model | internal | class top-1 | 중간 |
| Detection | `yolo` | boxes | external | internal | NMS + box center | 중간 |
| Detection | `detr_box` | query boxes | external | CE + L1 + GIoU | query selection | 높음 |
| Detection | `detr_point` | query points | external/custom | CE + point L1 | query ordering/matching | 매우 높음 |
| Refinement | `gcn` | iterative offsets | torchvision/custom | SmoothL1 | final iteration | 높음 |
| Refinement | `local_stn` | local offsets | custom | Wing/SmoothL1 | coarse + offset | 높음 |
| Geometry | `hybrid` | learned mask | torchvision/custom | BCE + Dice | boundary line fitting | 높음 |
| Classical | `classical_contour` | modulation mask | none | 없음 | four-side line fitting | 중간 |
| Classical | `classical_line` | boundary line candidates | none | 없음 | grouping + intersection | 중간 |

`vit_direct`, `torchseg`, `torchdet`, `local_stn`은 catalog에서 추적할 수 있지만 엄밀한 비교
설계에서는 각각 backbone variant, whole-model variant, refinement axis로 취급한다.

## 11. Loss 분류

loss는 raw output과 target representation에 의해 결정된다.

| loss family | 적용 method | 역할 |
|---|---|---|
| Wing | direct, vit_direct, foundation, local_stn | 작은 좌표 오차를 강조한다. |
| SmoothL1 | homography, gcn, box/point regression | outlier에 강한 연속값 회귀를 제공한다. |
| MSE | heatmap | Gaussian heatmap 전체를 회귀한다. |
| BCE + Dice | seg, torchseg, hybrid | pixel classification과 영역 overlap을 함께 최적화한다. |
| Focal | det objectness, line center map | 희소 positive와 class imbalance를 완화한다. |
| CrossEntropy | corner class detection | 네 corner ID를 분류한다. |
| GIoU | DETR box | 예측 box와 target box의 기하 겹침을 최적화한다. |
| Model internal | torchdet, yolo | 외부 model의 원래 loss 조합을 유지한다. |

같은 model에서 loss를 비교할 때는 target, optimizer, scheduler, augmentation을 고정한다. 서로
다른 출력 표현의 loss 값을 직접 비교하지 않고 최종 corner metric으로 비교한다.

## 12. Postprocess 분류

postprocess는 raw output을 표준 corner `(B, 4, 2)`로 변환한다.

| postprocess family | 입력 | 적용 method | 실패 가능성 |
|---|---|---|---|
| sigmoid reshape | coordinate logits | direct, vit_direct, foundation | 낮음 |
| canonical offset | bounded offsets | homography | 낮음 |
| soft-argmax | heatmaps | heatmap | 낮음 |
| four-side line fitting | binary mask | seg, torchseg, hybrid, classical_contour | 높음 |
| box center selection | detection output | det, torchdet, yolo, detr_box | 중간 |
| query point decode | point queries | detr_point | 중간 |
| line intersection | line maps/candidates | line, classical_line | 높음 |
| iterative offset | coarse corners | gcn, local_stn | 낮음 |

실패 가능한 postprocess는 실패 원인을 다음과 같이 구조화해 저장한다.

```text
no_candidate
insufficient_corners
duplicate_corner
degenerate_polygon
out_of_bounds
numerical_failure
```

## 13. 실험 식별자와 설정

실험 이름은 구현 출처, backbone, 핵심 출력, loss, postprocess, refinement를 조합한다.

```text
torchvision-resnet18__coordinates__wing__sigmoid__none
torchvision-resnet18__heatmap__mse__soft-argmax__local-stn
external-dinov2-vits14__coordinates__wing__sigmoid__none
external-detr-r50__query-boxes__detr-set__query-center__none
none-classical__mask__none__four-side-fit__none
```

설정 파일에는 다음 정보를 명시한다.

```yaml
method:
  code: heatmap
  family: dense_prediction

model:
  source: torchvision
  usage: backbone_only
  architecture: custom_heatmap
  backbone: resnet18
  pretrained: true
  output_type: heatmap

loss:
  name: mse

postprocess:
  name: soft_argmax

refinement:
  name: none

dependency:
  package: null
  weights: null
```

## 14. 공정 비교 원칙

방법론 비교에서 변경하는 축 이외의 조건을 고정한다.

### 14.1. Backbone 비교

backbone 비교에서는 output representation, head, loss, postprocess를 고정한다. ResNet, ViT,
Swin의 parameter 수와 pretrained dataset이 다르면 이를 함께 보고한다.

### 14.2. Loss 비교

loss 비교에서는 model initialization, target, data split, optimizer를 고정한다. loss별 권장
learning rate가 다르더라도 공통 설정 비교와 method별 tuning 결과를 구분한다.

### 14.3. Postprocess 비교

postprocess 비교에서는 동일한 raw output checkpoint를 사용한다. threshold와 geometric
parameter는 validation set에서 결정하고 test set에서는 변경하지 않는다.

### 14.4. Refinement 비교

refinement 비교에서는 같은 base prediction을 입력으로 사용한다. `none`, `local_stn`, 필요 시
`gcn`을 비교하고 추가 latency와 parameter를 함께 측정한다. 가상 corner에는 실제 sharp pixel
corner가 없으므로 `cornerSubPix`를 기본 refinement 후보로 포함하지 않는다.

### 14.5. External model 비교

외부 model은 package version, weight 출처, license, checkpoint hash, 원본 pretrained dataset을
기록한다. 외부 training pipeline을 사용할 때도 최종 evaluation은 공통 evaluator에서 수행한다.

## 15. 구현 우선순위

구현 난이도와 비교 가치를 고려한 권장 순서는 다음과 같다.

1. fringe modulation mask와 four-side line fitting을 사용하는 `classical_contour`,
   `classical_line` baseline을 확정한다.
2. `direct`, `homography`, `heatmap`, `seg`, `hybrid`를 공통 pipeline baseline으로 구현한다.
3. ResNet, ViT, Swin을 `direct`의 backbone 축에서 비교한다.
4. `local_stn` refinement를 동일 base prediction에 적용하고, 가상 corner에서의 실효성을
   검증한다.
5. `torchdet`, `yolo`를 box detection baseline으로 연결한다.
6. pretrained Box DETR adapter와 fine-tuning을 구현한다.
7. Point DETR을 별도 연구 방법론으로 구현한다.

순수 CV와 단순 regression을 먼저 구현하면 dataset 및 metric 오류를 빠르게 발견할 수 있다.
DETR은 dependency와 interface 위험이 크므로 공통 pipeline이 안정된 뒤 도입한다.

## 16. 최종 분류 원칙

방법론 catalog를 유지할 때 다음 원칙을 적용한다.

1. 출력 표현이나 추론 원리가 바뀌면 독립 method로 등록한다.
2. backbone만 바뀌면 같은 method의 model variant로 등록한다.
3. loss만 바뀌면 loss experiment로 등록한다.
4. raw output 이후 알고리즘만 바뀌면 postprocess experiment로 등록한다.
5. 여러 base method 뒤에 공통으로 붙을 수 있으면 refinement로 등록한다.
6. 외부 model의 원래 task가 다르면 adapter와 whole pipeline을 구분한다.
7. 학습 없는 classical CV도 같은 최종 출력과 metric 계약을 지키면 정식 baseline으로 등록한다.

이 원칙에 따르면 ViT/Swin 직접 회귀는 `direct`의 backbone variant이고, local STN은 공통
refinement다. DETR과 순수 classical CV는 독립된 핵심 아이디어를 가지므로 별도 method family에
포함한다. DocTr와 DocScanner는 document geometry prior, dense deformation 출력 및 현재 PMD OLED
fringe 도메인의 불일치 때문에 catalog에서 제외한다.
