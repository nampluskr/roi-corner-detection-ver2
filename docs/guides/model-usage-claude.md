---
tags: [roi-corner-detection, usage, comparison, cli, ver1-reuse, guide]
status: guide
created: 2026-07-16
updated: 2026-07-16
---

# 모델 성능 비교 사용 안내

이 문서는 client 수준에서 여러 model의 성능을 비교하는 사용 안내다. 전체 설계 기준은
[model-assembly.md](../architecture/model-assembly.md)이며 이 문서는 그 계약과 metric을
재정의하지 않고 참조만 한다. 폴더와 파일 구조는 [src-layout-claude.md](src-layout-claude.md)를
따른다. 사용 방식은 ver1 프로젝트의 CLI(`scripts/train.py`, `scripts/evaluate.py`,
`scripts/predict.py`, `experiments/run.py`)를 그대로 쓴다.

## 1. 문서 목적과 전제

이 절은 이 문서를 읽는 사람과 이 문서가 다루는 범위를 밝힌다.

### 1.1. 대상 독자와 canonical 문서와의 관계

대상 독자는 이미 정의된 model들을 실행하고 그 성능을 공통 기준으로 비교하려는 사용자다. model
내부 architecture나 새 component를 설계하는 사람은 먼저 canonical 문서를 읽는다. 이 문서의 모든
용어(method, model, variant, metric, category)는 canonical 문서에서 정의한 그대로를 쓴다.

### 1.2. ver1 CLI 방식과 실험 산출물 경로 규약

실행은 ver1의 CLI를 그대로 쓴다. `scripts/train.py`로 학습하고, `scripts/evaluate.py`로 test
성능을 산출하고, `scripts/predict.py`로 예측 CSV를 만든다. 여러 조합을 한 번에 돌릴 때는
`experiments/run.py`가 `experiments/configs.py`의 목록을 순회하며 각 config를 CLI args로 펼쳐
실행한다. 산출물 경로는 `outputs/<dataset>/<method>/<model>/<exp_name>/`이며 여기서 `dataset`은
`public`, `synthetic`, `measured` stage 중 하나다.

## 2. 준비물과 config 구조

이 절은 비교를 시작하기 전에 정하는 config 구조를 설명한다.

### 2.1. dataset stage 선택과 data split 고정

먼저 dataset stage를 고른다. stage는 학습 목적에 따라 다음 세 가지다.

| stage | 목적 |
|---|---|
| `public` | 공개 corner dataset에서 일반 corner 표현을 학습한다. |
| `synthetic` | fringe pattern과 광학 변동으로 target domain에 적응한다. |
| `measured` | 소량의 PMD data로 fine-tuning과 최종 평가를 수행한다. |

공정 비교를 위해 같은 seed와 split ratio를 쓴다. `scripts/config.py`의 `seed`를 고정하면 train,
valid, test split이 비교 대상 사이에서 동일하게 유지된다.

### 2.2. scripts/config.py DEFAULT와 args override

`scripts/config.py`의 `DEFAULTS` dict가 method, image size, batch size, epoch 수 등의 기본값을
담는다. CLI args는 이 기본값을 override한다. 예를 들어 method와 backbone만 바꾸어 실행하는
명령은 다음과 같다.

```bash
python scripts/train.py --method reg --backbone resnet50 --batch_size 4 --max_epochs 50 --save
```

`--method`가 registry code(`reg`, `seg`, `det`, `heatmap`, `line`, `refinement`, `classical`)를
받고, backbone을 명시하지 않으면 `DEFAULT_BACKBONES`의 method별 기본값을 쓴다.

### 2.3. experiments/configs.py로 비교 조합 정의

여러 조합을 한 번에 비교할 때는 `experiments/configs.py`의 `CONFIGS`에 config dict를 나열한다.
같은 축만 바꾼 dict를 나열하면 그 축의 효과를 분리해 비교할 수 있다. 예를 들어 backbone만 바꾼
비교는 다음과 같이 정의한다.

```python
CONFIGS = [
    {"method": "reg", "backbone": "custom", "max_epochs": 50},
    {"method": "reg", "backbone": "resnet50", "max_epochs": 50},
]
```

## 3. 단일 model 실행 시나리오

이 절은 하나의 model을 학습하고 평가하고 예측하는 기본 흐름을 설명한다.

### 3.1. train.py로 학습하고 checkpoint 저장

`scripts/train.py`는 train과 valid loader를 만들고 `get_wrapper`로 wrapper를 조립한 뒤 `Trainer`로
학습한다. `--save`를 주면 checkpoint와 학습 history를 output 폴더에 저장한다. `--patience`가 0보다
크면 early stopping을 쓴다.

### 3.2. evaluate.py로 metric 산출과 저장 경로

`scripts/evaluate.py`는 저장된 checkpoint를 불러와 test loader에서 공통 metric bank를 산출한다.
결과는 output 폴더에 저장된다. metric bank는 Polygon IoU, MCD, MaxCD, Reprojection Error,
PCK@0.02, PCK@0.05, SR, CPU와 GPU latency, model size다.

### 3.3. predict.py로 pred_corners.csv 생성

`scripts/predict.py`는 test 이미지에 대해 예측한 corner를 `pred_corners.csv`로 쓴다. CSV 컬럼은
`image_dir,image_name,x1,y1,x2,y2,x3,y3,x4,y4`다. 실패 가능한 postprocess는 성공 여부와 failure
reason을 함께 남긴다.

## 4. 두 model 성능 비교 시나리오

이 절은 두 model을 같은 조건에서 비교하는 방법을 설명한다.

### 4.1. 같은 축만 바꾼 config 쌍 준비

비교의 핵심은 한 번에 하나의 축만 바꾸는 것이다. 두 config 사이에서 backbone 또는 decoder 또는
head 중 하나만 다르게 두고 나머지 target, loss, postprocess, input size, optimizer, data split,
seed는 고정한다. 그래야 성능 차이를 바꾼 축의 효과로 해석할 수 있다.

### 4.2. run.py로 batch 실행과 산출물 수집

`experiments/run.py`가 `CONFIGS`의 각 config에 대해 train, evaluate, predict를 순서대로 subprocess로
실행한다. 실행 명령은 다음과 같다.

```bash
python experiments/run.py --mode all
```

각 config의 산출물은 `outputs/<dataset>/<method>/<model>/<exp_name>/`에 모인다.

### 4.3. 공통 metric bank로 결과 비교

두 실행의 evaluate 결과를 같은 metric bank로 비교한다. 정확도 metric만 보지 않고 SR과 latency와
model size를 함께 본다. mask나 heatmap 같은 dense model은 raw representation metric도 참고하되
최종 판단은 공통 corner metric과 SR로 한다.

## 5. 카테고리별 비교 워크플로우

이 절은 canonical 문서의 ablation matrix(section 11.4)를 따라 축별 비교 방법을 설명한다. 각
비교는 그 축만 바꾸고 나머지를 고정한다.

### 5.1. head 비교 (coord_gap 대 coord_spatial)

`reg` method에서 backbone과 target과 loss와 postprocess를 고정하고 head만 `coord_gap`과
`coord_spatial`로 바꾼다. global aggregation과 spatial 정보 유지의 차이를 본다.

### 5.2. decoder와 skip 비교 (plain 대 U-Net add)

`seg` method에서 backbone과 mask head와 training을 고정하고 decoder를 `plain`과 `unet` additive
skip으로 바꾼다. skip connection의 기본 효과를 본다. 이후 `unet` concat과 FPN을 각각 독립
ablation으로 수행한다.

### 5.3. backbone 비교 (custom 대 pretrained)

method와 head와 decoder와 target과 loss와 postprocess를 고정하고 backbone만 `custom`과 pretrained
CNN 또는 Transformer로 바꾼다. pretrained 여부, pretrained dataset, parameter 수, latency는 결과
metadata에 기록하고 backbone architecture 효과로만 해석하지 않는다.

### 5.4. postprocess 비교 (동일 checkpoint, geometry 교체)

같은 mask checkpoint 또는 저장된 probability map을 입력으로 두고 geometry postprocessor만 바꾼다.
four-side fitting과 contour approximation과 line refinement를 비교한다. model 정확도와 postprocess
정확도를 분리하기 위해 성공 표본의 정확도만 보지 않고 전체 SR과 실패 원인 분포와 latency를 함께
본다.

### 5.5. refinement 비교 (base 대 local STN 대 GCN)

같은 저장된 base corner에 refinement를 적용한다. refinement 없음과 `local_stn`과 `gcn`을
비교하고 base model을 다시 학습하지 않는다. refinement는 base보다 모든 주요 corner metric에서
개선되면서 SR을 낮추지 않을 때만 채택한다.

### 5.6. category 간 end-to-end 비교

dataset과 최종 metric을 고정하고 composable model과 external whole model과 classical pipeline을
end-to-end로 비교한다. external whole model은 weight 출처, pretrained task, dependency version,
license, model size, end-to-end latency를 함께 기록한다.

## 6. 결과 해석 기준

이 절은 비교 결과로 model을 고르는 기준을 설명한다.

### 6.1. 정확도와 subpixel precision 우선 선택

정확도 중심 선택에서는 MCD 평균만 쓰지 않는다. MaxCD, Reprojection Error, PCK@0.02, SR을 함께
보고 measured data에서 일관된 개선이 있는지 확인한다.

### 6.2. CPU latency와 model size 우선 선택

배포 후보는 preprocess와 model inference와 postprocess를 포함한 end-to-end CPU latency로 비교한다.
동일 정확도 범위에서는 parameter 수, serialized model size, peak memory, failure handling이 작은
조합을 우선한다.

### 6.3. metric 하나만으로 결론 내리지 않기

하나의 metric만 보고 결론 내리지 않는다. 정확도 metric과 SR과 비용 metric을 함께 보고, dense
model은 raw representation metric도 참고하되 최종 판단은 공통 corner metric과 SR로 한다.

## 7. 흔한 실수와 금지 조합

이 절은 비교에서 자주 발생하는 실수와 factory가 막는 조합을 설명한다.

### 7.1. 지원하지 않는 component 조합과 생성 오류

지원하지 않는 조합은 silent fallback하지 않고 생성 단계에서 오류를 낸다. 예를 들어 `stages`
capability가 없는 backbone에 U-Net이나 FPN을 요청하거나, `plain` decoder에 skip connection을
요청하거나, coordinate head에 dense decoder를 붙이면 오류가 발생한다. 오류에는 요청한 component,
필요한 capability, 실제 `FeatureSpec`이 포함된다.

### 7.2. test set에서 parameter 변경 금지

threshold와 geometry parameter는 validation set에서 확정하고 test set에서 표본별로 바꾸지 않는다.
classical parameter도 마찬가지로 validation set에서 확정한다.

### 7.3. 실패 표본을 평균에서 제외하지 않기

실패 가능한 postprocess는 성공 여부와 failure reason을 반환한다. evaluator는 실패 표본을 평균에서
조용히 제외하지 않고 전체 SR과 실패 원인 분포를 함께 보고한다.
