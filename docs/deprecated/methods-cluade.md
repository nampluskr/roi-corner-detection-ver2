---
status: deprecated
superseded_by: ../architecture/model-assembly.md
---

# 방법론 분류 체계

이 문서는 historical 후보안이다. 현재 프로젝트 설계의 기준은
[model-assembly.md](../architecture/model-assembly.md)이며, 이 문서의 분류, 제약과 method 이름은
새 구현 또는 실험의 근거로 사용하지 않는다.

이 문서는 평면 사각형 코너 검출 방법론의 공통 계약, 제약조건, 분류 체계를 제안했던 이전
후보안이다. 입력 이미지와 4개 코너 좌표를 받아 표준 코너 4개를 출력하고 공통 메트릭으로
평가한다는 계약을 중심으로 제약조건과 방법론 목록을 재도출했다. ver1과 ver2의 코드는
"이렇게 구현할 수 있다"는 구현 예시로만 인용하며, 규약의 근거로 삼지 않는다.

## 1. 공통 계약

모든 방법론이 공유하는 세 가지 계약이다. 이 계약이 방법론 분류의 기준점이다.

### 1.1 입출력 계약

입력과 출력은 방법론과 무관하게 고정된다.

- 입력: 이미지 텐서 `(B, 3, H, W)`, 기본 `H = W = 224`, ImageNet 평균/표준편차로 정규화한다.
  변환 순서는 `Resize -> augmentation -> ToTensor -> Normalize`다. 학습 시에는 정답 코너 4개가
  학습 신호로 함께 주어진다.
- 출력: 표준 코너 텐서 `(N, 4, 2)`, 정규화 [0, 1], `TL $\to$ TR $\to$ BR $\to$ BL` 시계방향
  순서다. 경계에서 한 번 정준화한다.
- 평가: `predict_step`은 표준 코너 `(N, 4, 2)`를 반환하고, 모든 방법론에 대해 동일한 공통
  메트릭 뱅크로 평가한다. 공통 메트릭은 IoU(다각형 넓이), MCD(평균 코너 거리), MaxCD(최대
  코너 거리), Reprojection Error(호모그래피, 서브픽셀 정밀도 필수 메트릭), PCK@0.02,
  PCK@0.05, SR(성공률, NaN이 아닌 유효 예측 비율)이다.
- CSV 정규형: `image_dir,image_name,x1,y1,x2,y2,x3,y3,x4,y4`, 정규화 [0, 1], 소수점 6자리,
  `TL $\to$ TR $\to$ BR $\to$ BL`이다.

SR은 후처리가 실패할 수 있는 방법론(seg, det, line, hybrid 등)을 공정하게 비교하기 위한
장치다. 구현 예시로는 ver1 `src/utils/geometry.py`의 `order_corners`와
`src/core/evaluator.py`의 공통 메트릭 구성을 참조한다.

### 1.2 모델 독립성 계약

각 방법론은 독립적으로 작성되며 다른 방법론에 의존하지 않는다. 방법론은 다음 네 파트로
구성되고 각각 공통 상위 모듈을 상속한다.

| 파트 | 상위 모듈 | 책임 |
|---|---|---|
| model | `BaseModel(nn.Module)` | `forward(images) -> raw_output` |
| preprocessor | `BasePreprocessor` | `__call__(corners) -> method target` |
| postprocessor | `BasePostprocessor` | `__call__(raw_output) -> (N, 4, 2)` |
| losses | `BaseLoss` | `forward(raw_output, target) -> loss` |

이 네 파트를 묶는 조립기는 공통 `BaseWrapper`이며 `train_step`, `eval_step`, `predict_step`을
제공한다. 계약의 핵심은 postprocessor와 `predict_step`만 표준 입출력 경계에 닿고, 방법론
고유 로직은 preprocessor와 postprocessor 사이에 위치한다는 것이다. 구현 예시로는
`src/models/<method>/`의 4개 파일과 `get_wrapper` dispatch 키를 참조한다.

### 1.3 데이터 계약

데이터는 세 단계로 구성되며 모두 CSV 정규형(1.1)으로 수렴한 뒤 공통 데이터셋으로 로드된다.

- public: smartdoc + midv2020을 `gt_corners.csv`로 변환한다. `order_corners`,
  `is_invalid_corners`, `mask_to_corners`로 순서와 유효성을 정리한다.
- synthetic: 합성 생성기가 LabelMe JSON을 생성한다. 생성 스크립트는 `src/data/` 밖에 둔다.
- measured: LabelMe 형식 라벨로 주어진다.

synthetic과 measured가 모두 LabelMe 형식이므로, 단일 LabelMe 파서
`parse_labelme(json_dir, image_dir) -> CSV`가 두 단계를 CSV 정규형으로 흡수하는 통합
경로다. 이 파서는 아직 구현되지 않은 항목(TODO)이다.

## 2. 제약조건

PMD fringe 패턴 패널 검사 도메인과 공통 계약에서 제약을 재도출한다. F1-F8은 계승하되 F4를
재범위화하고, 공통 계약에서 F9를 새로 도출한다.

| 제약 | 내용 |
|---|---|
| F1 | 임의의 볼록 사각형(축 정렬 OBB 아님). 표준 `(N, 4, 2)` 자유 코너 출력의 근거. |
| F2 | 단일 지배 패널, 이미지의 50% 이상 차지. |
| F3 | 4개 코너는 항상 이미지 경계 내부. 정규화 [0, 1] 좌표와 경계 밖 외삽 금지의 근거. |
| F4 | 실측 소량, 합성 다수. 방법론 제외 규칙이 아니라 데이터 전략 제약이다. 3단계 학습(공개 $\to$ 합성 $\to$ 실측)이 완화책이다. |
| F5 | 서브픽셀 정밀도 필수. Reprojection Error를 공통 메트릭에 상시 포함한다. |
| F6 | CPU 배포 지연/모델 크기 예산. 무거운 transformer/foundation 방법론의 실제 판별 기준. |
| F7 | 조명/글레어/비네팅 변동 강건성. |
| F8 | 패널 가림 없음. 4개 코너가 모두 관측되어 결정적 코너 순서화가 가능하다는 근거. |
| F9 | 공통 계약 준수. 모든 방법론은 표준 `(N, 4, 2)` [0, 1] `TL $\to$ TR $\to$ BR $\to$ BL` 출력을 내고 공통 wrapper와 메트릭 뱅크로 학습/평가 가능해야 한다. 네이티브 출력을 정준화할 수 없거나 학습이 불가능한 방법론은 postprocessor에서 적응시키거나 학습 없는 연구 행으로 강등한다. |

F9는 "독립 모델 + 공통 입출력 + 3단계 데이터"라는 틀에서 직접 도출되며, classical과 line의
처리를 판단하는 기준이다.

### 2.1 F4 재해석과 detr

ver1은 detr(set prediction)을 F4 사유로 제외했다. ver2는 F4를 데이터 전략 제약으로
재범위화하므로 detr을 조건부로 편입한다. 합성 데이터 사전학습과 foundation 초기화로 F4를
완화하고, F4/F6 근거가 measured 벤치마크에서 성립하면 배포형 방법론으로 인정한다. 성립하지
않으면 detr은 합성 전용 연구 행으로 축소한다.

## 3. 카테고리

공통 계약이 입출력으로 정의되고 출력 표현이 preprocessor, postprocessor, loss를 함께
결정하므로, 1차 축을 출력 표현으로 삼는다. 이는 계약 경계(postprocessor)와 1:1로 정렬된다.
backbone 출처는 한 방법론 안에서 model마다 달라지는 model 단위 속성이므로 2차로 강등한다.

### 3.1 1차 축: 출력 표현 (후처리 계열)

| 계열 | 의미 |
|---|---|
| `coord-regression` | 8좌표 직접 회귀 |
| `heatmap` | 코너별 heatmap, argmax |
| `mask-contour` | 분할 마스크, findContours |
| `box-detection` | 코너를 객체로 검출, 박스 중심 |
| `line-intersection` | 에지/직선 검출, 교점 계산 |
| `iterative-refine` | 초기 추정 + 반복 정제 |
| `set-prediction` | 쿼리 기반 집합 예측, Hungarian |

### 3.2 2차 축: model family

| family | 의미 |
|---|---|
| `torchvision-custom-head` | torchvision backbone + 직접 head/decoder |
| `torchvision-whole-model` | torchvision seg/det 모델 통째 재사용 |
| `external-pretrained` | 외부 사전학습 인코더/모델(DINOv2, M-LSD, DocTr 등) |
| `custom-arch` | 아키텍처 전체를 직접 설계 |
| `no-model` | 학습 가능한 모델 없음 |

3차 태그로 손실 계열, 학습 가능 여부, 외부 의존, 데이터 단계 적합을 붙인다.

## 4. 목록 재구성

### 4.1 통합 판정

이전에 나열한 18개 항목을 공통 계약 기준으로 재판정한다. 같은 타깃/손실/후처리 규약을
유지한 채 backbone만 다른 항목은 독립 method가 아니라 부모 method의 model로 흡수한다.

| 이전 항목 | 판정 | 근거 |
|---|---|---|
| `direct` | 유지 | coord-regression 기준 방법론 |
| `doc` | `direct`의 model로 흡수 | 같은 wing/coord 규약, backbone(문서 사전학습)만 다름 |
| `vit` | `direct`의 model로 흡수 | 같은 coord 규약, supervised ViT backbone만 다름 |
| `doctr` | `direct`의 model로 흡수 | 같은 coord 규약, DocTr/DocScanner 인코더만 다름 |
| `foundation` | 유지 | backbone 동결이라는 학습 레짐 차이, F6 관련성 |
| `homography` | 유지 | 정준 좌표 + bounded offset 타깃/후처리 상이 |
| `heatmap` | 유지 | heatmap 타깃/argmax 후처리 상이 |
| `seg` | 유지 | mask-contour 기준 방법론 |
| `torchseg` | `seg`의 model로 흡수 | 같은 bce-dice/contour 규약, 모델 통째 재사용만 다름 |
| `det` | 유지 | grid box-detection 타깃/후처리 상이 |
| `torchdet` | `det`의 model로 흡수 | 같은 box-detection 규약, torchvision 검출기 재사용 |
| `yolo` | `det`의 model로 흡수 | 같은 box-detection 규약, 외부 검출기 재사용 |
| `gcn` | 유지 | iterative-refine 기준 방법론 |
| `stn` | 유지(조건부) | 초기 추정 + 정제이나 이미지 crop 정제로 gcn과 규약 차이. 병합 여부 열린 결정 |
| `line` | 유지 | M-LSD line-intersection 상이 |
| `hybrid` | 유지 | DL 마스크 + 고전 CV 교점, 후처리 상이 |
| `classical` | `hybrid`의 no-train model로 흡수 | 학습 마스크 없는 hybrid 후처리. F9상 학습 불가 |
| `detr` | 유지(조건부) | set-prediction 규약 상이. F4/F6로 배포형 여부 결정 |

### 4.2 재구성된 방법론 목록

11개 method(+detr 조건부)로 통합한다. 최상위 이름은 바꾸지 않고, 흡수된 항목은 부모
method의 model id로 보존한다.

| 순서 | code | 출력 표현 | models |
|---|---|---|---|
| 1 | `direct` | coord-regression | resnet_gap, resnet_spatial, vit_finetune, doc_pretrained, doctr |
| 2 | `homography` | coord-regression(offset) | resnet_spatial |
| 3 | `heatmap` | heatmap | resnet_deconv |
| 4 | `foundation` | coord-regression(동결) | dinov2_linear |
| 5 | `seg` | mask-contour | unet_head, torchvision_whole |
| 6 | `det` | box-detection | grid_head, torchdet, yolo |
| 7 | `gcn` | iterative-refine | resnet_gcn |
| 8 | `stn` | iterative-refine | localization_refine |
| 9 | `line` | line-intersection | mlsd |
| 10 | `hybrid` | line-intersection | unet_head, no_train |
| 11 | `detr` | set-prediction | detr_r50 |

### 4.3 재명명

최상위 method 이름은 불변이며, 재분류만 일어난다. 흡수 항목은 다음 model id로 보존한다.

- `doc` $\to$ `direct/doc_pretrained`
- `vit` $\to$ `direct/vit_finetune`
- `doctr` $\to$ `direct/doctr`
- `torchseg` $\to$ `seg/torchvision_whole`
- `torchdet` $\to$ `det/torchdet`
- `yolo` $\to$ `det/yolo`
- `classical` $\to$ `hybrid/no_train`

## 5. 재분류 마스터 표

재구성된 11개 method의 마스터 표다.

| code | 출력 표현 | model family | 손실 계열 | 후처리 계열 | 학습 | 외부 의존 | 데이터 단계 |
|---|---|---|---|---|---|---|---|
| `direct` | coord-regression | tv-custom-head / external-pretrained / custom-arch | wing | sigmoid + reshape | 예 | DINOv2, DocTr(일부 model) | 3단계 전체 |
| `homography` | coord-regression | tv-custom-head | smoothl1 | 정준 좌표 + tanh offset | 예 | 없음 | 3단계 전체 |
| `heatmap` | heatmap | tv-custom-head | mse | soft-argmax | 예 | 없음 | 3단계 전체 |
| `foundation` | coord-regression | external-pretrained | wing | sigmoid + reshape | 예(head) | DINOv2, timm | synthetic, measured |
| `seg` | mask-contour | tv-custom-head / tv-whole-model | bce-dice | findContours + approxPolyDP | 예 | OpenCV, torchvision seg | 3단계 전체 |
| `det` | box-detection | tv-custom-head / tv-whole-model / external | det-multi / det-internal | 박스 중심 decode | 예 | torchvision det, ultralytics | 3단계 전체 |
| `gcn` | iterative-refine | tv-custom-head | smoothl1 | 최종 GCN 반복 출력 | 예 | 없음 | 3단계 전체 |
| `stn` | iterative-refine | custom-arch | smoothl1 | crop-refine + 서브픽셀 offset | 예 | 없음 | 3단계 전체 |
| `line` | line-intersection | external-pretrained | line-multi | M-LSD 세그먼트 교점 | 예 | M-LSD 가중치 | 3단계 전체 |
| `hybrid` | line-intersection | tv-custom-head / no-model | bce-dice / 없음 | Canny + Hough + cornerSubPix | 예 / 아니오(no_train) | OpenCV | 3단계 전체 |
| `detr` | set-prediction | external-pretrained | detr-set | Hungarian query decode | 예 | 검출기 가중치 | synthetic 우선(조건부) |

## 6. 구현과 후처리

구체적인 backbone/head 조립은 구현 세부이며, ver1 `docs/models/01..13`과
`src/models/<method>/`가 예시 실현이다.

### 6.1 후처리 표

| 출력 표현 계열 | 후처리 단계 | 공유 method | 실패 가능 |
|---|---|---|---|
| coord-regression | sigmoid + reshape | direct, foundation | 아니오 |
| coord-regression(offset) | 정준 좌표 + tanh offset | homography | 아니오 |
| heatmap | soft-argmax | heatmap | 아니오 |
| mask-contour | findContours + approxPolyDP | seg | 예 |
| box-detection | 박스 중심 decode | det | 예 |
| line-intersection | Canny/Hough/M-LSD 교점 + cornerSubPix | line, hybrid | 예 |
| iterative-refine | 최종 반복 출력 / crop-refine | gcn, stn | 아니오 |
| set-prediction | Hungarian query decode | detr | 예 |

### 6.2 손실 표

| 손실 계열 | method | 비고 |
|---|---|---|
| wing | direct, foundation | apply_sigmoid |
| smoothl1 | homography, gcn, stn | homography는 offset, gcn은 deep supervision |
| mse | heatmap | heatmap 타깃 |
| bce-dice | seg, hybrid | 마스크 타깃 |
| det-multi | det | focal obj + smoothl1 box + ce class |
| det-internal | det(torchdet, yolo model) | 검출기 내부 손실 |
| line-multi | line | focal center + masked smoothl1 displacement |
| detr-set | detr | set prediction 손실 |

## 7. 열린 결정

- `stn`과 `gcn`을 iterative-refine 안에서 병합할지, 별도 method로 유지할지.
- `foundation`(동결)과 `direct/vit_finetune`(파인튜닝)의 경계를 model 레지스트리에서 어떻게
  명시할지.
- `detr`의 F4/F6 완화 전제가 measured 벤치마크에서 성립하는지 검증.
- `hybrid/no_train`(구 classical)을 no-train model로 둘지, 별도 baseline 행으로 둘지.
- LabelMe 파서 `src/data/labelme.py` 구현.
