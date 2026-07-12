# 프로젝트 에이전트 지침

이 파일은 `roi-corner-detection-ver2` 프로젝트에서 문서와 코드를 작성하는 에이전트가
따라야 하는 프로젝트 규칙과 설계 계약을 정의한다. `CLAUDE.md`와 `AGENTS.md`는 같은 내용을
유지하며 한 파일을 변경하면 다른 파일에도 동일하게 반영한다.

## 1. 프로젝트 개요

이 프로젝트는 평면 사각형 객체의 네 코너 좌표를 검출하는 여러 방법론을 하나의 공통 데이터,
학습, 평가 pipeline에서 구현하고 비교한다. `roi-corner-detection-ver1`을 리팩토링하며 다음
문제를 해결하는 것이 목적이다.

- dataset, data source, method, model, backbone 용어를 분리한다.
- 방법론의 핵심 아이디어와 구현 출처를 같은 분류 수준에 섞지 않는다.
- model, loss, postprocess, refinement를 독립적인 실험 축으로 관리한다.
- 모든 방법론을 같은 최종 corner 형식과 metric으로 평가한다.
- 실험 산출물 경로와 이름을 일관되게 관리한다.

## 2. 설계 문서

`docs/`의 방법론 문서는 현재 설계안을 비교하기 위한 문서다. 확정되지 않은 항목은 구현 전에
사용자와 결정한다.

| 문서 | 역할 | 상태 |
|---|---|---|
| `docs/methods-cluade.md` | 출력 표현 중심의 11개 method 통합안 | 후보안 |
| `docs/methods-codex.md` | 핵심 아이디어 중심의 상세 catalog와 다축 분류안 | 후보안 |

두 문서에서 공통으로 채택하는 원칙은 다음과 같다.

1. 최종 출력은 정규화된 네 코너로 통일한다.
2. backbone 출처와 method의 핵심 아이디어를 분리한다.
3. loss와 postprocess를 명시적인 실험 속성으로 관리한다.
4. library whole model 재사용과 custom head 구현을 구분한다.
5. 학습 없는 classical CV도 공통 출력과 평가 계약을 지키면 baseline으로 인정한다.
6. 외부 model은 dependency, weight, license, 원래 task와의 interface 차이를 기록한다.

방법론의 최종 개수와 흡수 관계는 아직 확정하지 않는다. 특히 DocTr/DocScanner whole pipeline,
STN refinement, classical CV를 독립 method로 둘지 variant로 둘지는 열린 결정이다.

## 3. 공통 입출력 계약

모든 방법론은 내부 표현과 관계없이 다음 계약을 지킨다.

### 3.1. 입력

입력 규약은 다음과 같다.

- image tensor shape은 `(B, 3, H, W)`다.
- 기본 입력 크기는 `H = W = 224`다.
- 기본 normalization은 ImageNet mean과 standard deviation을 사용한다.
- 학습 시 정답 corner tensor `(B, 4, 2)`를 함께 제공한다.
- augmentation은 image와 corner에 동일한 기하 변환을 적용한다.

### 3.2. 출력

출력 규약은 다음과 같다.

- 최종 corner tensor shape은 `(B, 4, 2)`다.
- 좌표 범위는 정규화된 `[0, 1]`이다.
- corner 순서는 `TL`, `TR`, `BR`, `BL`이다.
- corner 순서 정규화는 method 경계에서 한 번만 수행한다.
- 실패 가능한 postprocess는 표본별 성공 여부와 실패 원인을 함께 반환한다.

### 3.3. CSV

공통 CSV 형식은 다음과 같다.

```text
image_dir,image_name,x1,y1,x2,y2,x3,y3,x4,y4
```

좌표는 `[0, 1]` 범위, 소수점 6자리, `TL`, `TR`, `BR`, `BL` 순서로 저장한다.

## 4. 공통 평가 계약

모든 method의 raw output은 postprocess 이후 표준 corner로 변환하고 같은 metric bank에서
평가한다.

| metric | 평가 관점 | 좋은 방향 |
|---|---|---|
| Polygon IoU | 사각형 영역 일치도 | 클수록 좋음 |
| MCD | 평균 corner 좌표 오차 | 작을수록 좋음 |
| MaxCD | 가장 큰 단일 corner 오차 | 작을수록 좋음 |
| Reprojection Error | homography 기반 복원 오차 | 작을수록 좋음 |
| PCK@0.02 | 엄격한 거리 기준 성공률 | 클수록 좋음 |
| PCK@0.05 | 완화된 거리 기준 성공률 | 클수록 좋음 |
| SR | 유효한 네 corner를 반환한 비율 | 클수록 좋음 |
| CPU/GPU latency | end-to-end 추론 시간 | 작을수록 좋음 |
| Model size | 저장 및 배포 비용 | 작을수록 좋음 |

정확도 metric은 SR과 함께 보고한다. postprocess 실패 표본을 제외한 평균만 보고하지 않는다.
latency는 preprocess, model inference, postprocess를 모두 포함하고 warm-up 이후 측정한다.

## 5. 도메인 제약

방법론 설계와 평가에 적용하는 제약은 다음과 같다.

| 코드 | 제약 |
|---|---|
| F1 | 대상은 axis-aligned box가 아닌 임의의 convex quadrilateral이다. |
| F2 | image에는 하나의 지배적인 panel이 있으며 image 면적의 50% 이상을 차지한다. |
| F3 | 네 corner는 항상 image 경계 안에 있다. |
| F4 | measured data는 적고 synthetic data는 많다. |
| F5 | phase restoration을 위해 subpixel precision이 중요하다. |
| F6 | CPU latency와 model size에 배포 제약이 있다. |
| F7 | illumination, glare, vignette 변화가 존재한다. |
| F8 | panel occlusion은 없으며 네 corner가 모두 관측된다. |
| F9 | 모든 method는 공통 입출력과 평가 계약을 준수해야 한다. |

F4는 method 제외 규칙이 아니라 data strategy 제약이다. data 부족을 이유로 transformer나
detection method를 사전에 제외하지 않고 synthetic pretraining과 pretrained weight로 완화한
뒤 measured benchmark에서 판단한다.

## 6. 데이터 단계

학습 데이터는 다음 세 단계로 구분한다.

1. `public`: SmartDoc, MIDV-2020 등 공개 dataset으로 일반적인 corner 표현을 학습한다.
2. `synthetic`: fringe pattern과 광학 변동을 포함한 합성 dataset으로 target domain에 적응한다.
3. `measured`: 소량의 실측 PMD dataset으로 최종 fine-tuning과 평가를 수행한다.

각 data source의 native label은 공통 CSV 형식으로 변환한 뒤 같은 dataset class에서 로드한다.
synthetic과 measured의 LabelMe annotation은 하나의 공통 parser로 처리한다.

## 7. 방법론 분류 축

method specification은 다음 구조를 따른다.

```text
Method specification
├── family
├── model
│   ├── source
│   ├── usage
│   ├── architecture
│   ├── backbone
│   ├── head
│   └── output_type
├── loss
├── postprocess
└── refinement
```

### 7.1. 핵심 아이디어 family

핵심 아이디어에 따른 family는 다음과 같다.

| family | 의미 |
|---|---|
| `regression` | image 전체 특징에서 coordinate 또는 geometry parameter를 직접 회귀한다. |
| `dense_prediction` | heatmap, mask, line map을 먼저 예측한다. |
| `detection` | corner를 box, class 또는 transformer query로 검출한다. |
| `document_geometry` | deformation field 또는 rectification으로 corner를 복원한다. |
| `iterative_refinement` | 초기 corner를 graph 또는 local image feature로 반복 보정한다. |
| `classical_cv` | 학습 model 없이 image processing과 geometry로 corner를 계산한다. |

### 7.2. Model source

model source는 다음 값으로 분류한다.

| source | 의미 |
|---|---|
| `torchvision` | `torchvision.models`의 backbone 또는 whole model을 사용한다. |
| `external` | timm, Ultralytics 또는 외부 repository의 model을 사용한다. |
| `custom` | 전체 architecture를 프로젝트에서 직접 구현한다. |
| `none` | 학습 가능한 model이 없는 classical CV다. |

정확한 package 이름은 `torchvision.models`다. `torch.models`라는 표현을 사용하지 않는다.

### 7.3. Model usage

같은 source라도 재사용 범위를 다음과 같이 구분한다.

| usage | 의미 |
|---|---|
| `backbone_only` | backbone만 재사용하고 task head는 직접 구현한다. |
| `whole_model` | segmentation 또는 detection model 전체를 재사용한다. |
| `adapter` | pretrained model을 동결하거나 부분 동결하고 작은 adapter/head를 학습한다. |
| `from_scratch` | 전체 network를 직접 구성해 처음부터 학습한다. |

## 8. 방법론 후보

현재 검토하는 method와 model variant는 다음과 같다.

| group | code | 핵심 표현 | 기본 loss | 기본 postprocess |
|---|---|---|---|---|
| Regression | `direct` | coordinates | Wing 또는 SmoothL1 | sigmoid + reshape |
| Regression | `homography` | bounded offsets | SmoothL1 | canonical + tanh offset |
| Regression | `vit_direct` | coordinates | Wing | sigmoid + reshape |
| Regression | `foundation` | coordinates | Wing | sigmoid + reshape |
| Dense | `heatmap` | corner heatmaps | MSE | soft-argmax |
| Dense | `seg` | binary mask | BCE + Dice | contour approximation |
| Dense | `torchseg` | binary mask | BCE + Dice | contour approximation |
| Dense | `line` | line maps | Focal + SmoothL1 | grouping + intersection |
| Detection | `det` | grid boxes | Focal + SmoothL1 + CE | class top-1 |
| Detection | `torchdet` | boxes | model internal | class top-1 |
| Detection | `yolo` | boxes | model internal | NMS + box center |
| Detection | `detr_box` | query boxes | CE + L1 + GIoU | query selection |
| Detection | `detr_point` | query points | CE + point L1 | query ordering 또는 matching |
| Document | `doctr` | deformation 또는 rectification | model-specific | boundary recovery |
| Document | `docscanner` | recurrent deformation | model-specific | boundary recovery |
| Refinement | `gcn` | iterative offsets | SmoothL1 | final iteration |
| Refinement | `local_stn` | local offsets | Wing 또는 SmoothL1 | coarse + offset |
| Geometry | `hybrid` | learned mask | BCE + Dice | Canny + Hough + cornerSubPix |
| Classical | `classical_contour` | contour | 없음 | polygon approximation |
| Classical | `classical_line` | line candidates | 없음 | grouping + intersection |

`vit_direct`, `torchseg`, `torchdet`, `local_stn`은 catalog에서 추적할 수 있지만 각각 backbone,
whole-model, refinement variant로도 해석할 수 있다. 최종 registry 구조는 구현 전에 확정한다.

## 9. 주요 방법론 결정

신규 후보를 구현할 때 다음 기준을 적용한다.

### 9.1. ViT와 Swin

ViT/Swin 직접 회귀는 `direct`와 같은 coordinate target, loss, postprocess를 유지하고 backbone만
교체한다. ResNet, ViT, Swin 비교에서는 head와 학습 조건을 고정하고 parameter 수, pretrained
dataset, latency를 함께 보고한다.

### 9.2. DETR

DETR은 pretrained Box DETR을 먼저 adapter로 연결한 뒤 Point DETR을 검토한다. external package와
weight는 source, version, checksum, license, cache path를 기록하고 별도로 다운로드한다. Box
DETR은 classification, L1, GIoU loss를 사용한다. Point DETR은 고정 corner query 또는 Hungarian
matching 중 하나를 명시적으로 선택한다.

### 9.3. DocTr와 DocScanner

ver1의 `doc`은 실제 DocTr/DocScanner가 아닌 pretrained ResNet coordinate regression이다.
이를 실제 document geometry model과 혼동하지 않는다. DocTr/DocScanner는 먼저 pretrained
inference adapter로 domain 적합성을 확인한 뒤 fine-tuning을 검토한다. dense deformation
supervision이 없는 상태에서 재학습을 전제로 하지 않는다.

### 9.4. STN local refinement

corner 주변을 개별 확대하는 구조는 single global STN보다 `grid_sample` 또는 ROIAlign 기반
local refinement로 정의한다. `direct`, `heatmap`, `seg`, `det`, `detr` 뒤에 공통으로 적용할
수 있으므로 `refinement=none/local_stn` 축으로 비교한다.

### 9.5. Classical CV

순수 classical CV는 contour pipeline과 line pipeline을 별도 baseline으로 평가한다. validation
set에서 확정한 threshold와 geometry parameter를 test set에서 변경하지 않는다. 학습 parameter가
없어도 공통 output, SR, accuracy, latency 계약을 그대로 따른다.

## 10. 구현 복잡도

복잡도는 architecture, training, dependency, postprocess를 각각 판단한다.

| 등급 | 기준 |
|---|---|
| 낮음 | 단일 output, 단일 loss, 결정적 postprocess, 외부 의존이 거의 없다. |
| 중간 | dense target, decoder, 복합 loss 또는 geometry postprocess가 필요하다. |
| 높음 | 다단계 구조, 반복 refinement, 실패 가능한 postprocess 또는 외부 weight가 필요하다. |
| 매우 높음 | 외부 repository 통합, label 변환 또는 원래 task와 큰 interface 차이가 있다. |

외부 model을 단순히 설치할 수 있다는 이유로 구현 복잡도를 낮게 평가하지 않는다. 공통 trainer,
checkpoint, evaluator에 연결하는 adapter 비용을 포함한다.

## 11. 모델 모듈 계약

각 독립 method는 다음 네 역할을 분리한다.

| 파트 | 상위 모듈 | 책임 |
|---|---|---|
| model | `BaseModel(nn.Module)` | `forward(images) -> raw_output` |
| preprocessor | `BasePreprocessor` | `__call__(corners) -> method_target` |
| postprocessor | `BasePostprocessor` | `__call__(raw_output) -> corners, success` |
| loss | `BaseLoss` | `forward(raw_output, target) -> loss` |

`BaseWrapper`는 model, preprocessor, postprocessor, loss를 조립하고 공통 `train_step`,
`eval_step`, `predict_step`을 제공한다. 외부 whole model처럼 호출 규약이 다른 경우 adapter에서
차이를 흡수하고 evaluator에는 표준 corner만 전달한다.

## 12. 폴더 및 산출물 규칙

목표 폴더 구조는 다음과 같다.

```text
roi-corner-detection-ver2/
├── data/
│   ├── measured/
│   ├── public/
│   └── synthetic/
├── docs/
├── experiments/
│   ├── measured/
│   ├── public/
│   └── synthetic/
├── notebooks/
│   ├── measured/
│   ├── public/
│   └── synthetic/
├── outputs/
│   ├── measured/
│   ├── public/
│   └── synthetic/
├── scripts/
├── src/
│   ├── core/
│   ├── data/
│   ├── losses/
│   ├── metrics/
│   ├── models/
│   └── utils/
├── AGENTS.md
├── CLAUDE.md
├── PLAN.md
└── README.md
```

실험 산출물은 다음 경로 규칙을 사용한다.

```text
outputs/<dataset>/<method>/<model>/<exp_name>/
```

`dataset`은 `public`, `synthetic`, `measured`의 논리 단계이며 data source 이름과 구분한다.
`method`는 핵심 예측 원리, `model`은 architecture와 backbone 조합, `exp_name`은 loss,
postprocess, refinement, 학습 설정의 차이를 식별한다.

## 13. 공정 비교 원칙

실험에서는 변경하는 축 이외의 조건을 고정한다.

- backbone 비교에서는 output, head, loss, postprocess를 고정한다.
- loss 비교에서는 model initialization, target, data split, optimizer를 고정한다.
- postprocess 비교에서는 같은 raw output checkpoint를 사용한다.
- refinement 비교에서는 같은 base prediction을 입력으로 사용한다.
- external model 비교에서는 weight 출처와 pretrained dataset을 기록한다.
- test set을 보고 threshold나 geometry parameter를 조정하지 않는다.
- 공통 설정 비교와 method별 tuning 결과를 별도 표로 보고한다.

## 14. 구현 우선순위

권장 구현 순서는 다음과 같다.

1. `classical_contour`, `classical_line` baseline을 구현한다.
2. `direct`, `homography`, `heatmap`, `seg` 공통 pipeline을 구현한다.
3. ResNet, ViT, Swin backbone을 같은 direct head에서 비교한다.
4. `cornerSubPix`, `local_stn` refinement를 같은 base prediction에 적용한다.
5. `torchdet`, `yolo` detection adapter를 구현한다.
6. pretrained Box DETR adapter와 fine-tuning을 구현한다.
7. Point DETR을 별도 연구 방법론으로 검토한다.
8. DocTr와 DocScanner pretrained inference adapter를 평가한다.
9. dense supervision과 성능 이득이 확인되면 document model fine-tuning을 진행한다.

## 15. 문서 작성 규칙

모든 Markdown 문서는 다음 규칙을 따른다.

- 본문은 서술체를 사용한다.
- em dash, 유니코드 화살표, 이모지를 사용하지 않는다.
- Markdown 본문의 화살표는 `$	o$`를 사용한다.
- fenced code block과 inline code 안에서는 ASCII `->`를 사용한다.
- 폴더 구조 tree는 `├ ─ │ └` 문자를 사용한다.
- header level을 건너뛰지 않고 H4 아래 level은 사용하지 않는다.
- 수평 구분선은 사용하지 않는다. YAML frontmatter의 `---`는 예외다.
- table과 list 앞에는 내용을 소개하는 문장을 둔다.
- 폴더와 파일 목록은 폴더를 알파벳순으로 먼저 나열하고 파일을 알파벳순으로 나열한다.
- Jupyter notebook cell의 `source` 배열 마지막 원소는 줄바꿈으로 끝나지 않는다.

## 16. 코드 작성 규칙

모든 Python 코드는 다음 규칙을 따른다.

- 식별자, 주석, docstring, 문자열에 한국어를 사용하지 않는다.
- 세로 정렬을 위한 불필요한 공백을 넣지 않는다.
- 경로 처리는 `pathlib.Path` 대신 `os.path`를 사용한다.
- type hint를 사용하지 않는다.
- 모든 파일의 첫 줄은 `# path/from/project/root.py: one-line description` 형식으로 작성한다.
- 첫 줄 header 다음에 빈 줄 하나를 두고 import를 작성한다.
- class와 top-level function은 한 줄 docstring을 작성한다.
- method에는 docstring을 작성하지 않는다.
- 주석은 필요한 경우에만 최소한으로 작성한다.
- `src/` 아래 모든 폴더에는 빈 `__init__.py`를 둔다.
- `src/` 내부 import는 `src.xxx` 형식의 absolute import를 사용한다.
- `scripts/`, `experiments/`에서는 project root를 `sys.path`에 추가한 뒤 `src.xxx`로 import한다.

## 17. 동기화 규칙

`CLAUDE.md`와 `AGENTS.md`는 프로젝트 지침의 동기화 사본이다. 한 파일의 내용이 변경되면 같은
작업에서 다른 파일을 동일한 내용으로 갱신하고 두 파일의 byte-level 일치 여부를 검증한다.
