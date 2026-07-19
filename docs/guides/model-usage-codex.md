---
tags: [roi-corner-detection, model-usage, experiment, comparison, proposal, codex]
status: proposal
created: 2026-07-16
updated: 2026-07-16
---

# 모델 구성 및 성능 비교 사용 가이드

## 1. 문서 목적과 SSOT 관계

이 문서는 client가 동일한 data, training과 evaluation pipeline에서 model component를 변경하고 성능을
비교하는 사용 시나리오를 제안한다. Canonical method, model, variant와 metric 계약은
[모델 재조립 카테고리 및 비교 설계](../architecture/model-assembly.md)를 따른다. 목표 source 배치는
[src 폴더 구조 및 모듈 배치 설계](src-layout-codex.md)를 참고한다.

이 문서의 상태는 `proposal`이다. 아래 command와 Python API는 구현 목표이며 해당 파일이 구현되기
전에는 실행 가능한 interface로 간주하지 않는다.

## 2. Config에서 output까지의 실행 흐름

CLI, batch experiment와 Python API는 같은 factory와 wrapper lifecycle을 사용한다.

```text
DEFAULTS
-> optional YAML config
-> CLI or experiment overrides
-> resolved config
-> get_dataloader
-> get_wrapper
-> method wrapper
-> model component factory
-> Trainer, Evaluator or Predictor
-> outputs/<dataset>/<method>/<model>/<exp_name>/
```

Wrapper는 model, preprocessor, postprocessor, losses, metrics, optimizer, scheduler와 device를 소유한다.
Client는 component를 직접 연결하지 않고 config를 `get_wrapper`에 전달한다. Python 사용자는 dependency
injection이 필요할 때 wrapper 생성자에 loss나 optimizer instance를 직접 전달할 수 있다.

## 3. Config 작성

Config는 data, method, model, training, runtime과 output 설정을 구분한다. Nested dictionary가 기준
표현이며 dotted override는 leaf 값을 변경한다.

### 3.1. DEFAULTS

`scripts/config.py`의 `DEFAULTS`는 단일 실행과 batch experiment가 공유한다.

| 영역 | 대표 key | 목적 |
|---|---|---|
| data | `dataset`, `csv_path`, `image_size`, split sample 수 | data stage와 loader 입력 |
| method | `method` | `reg`, `seg`, `det`, `heatmap`, `line` 선택 |
| model | `architecture`, `backbone`, `adapter`, `decoder`, `neck`, `head` | model component 조립 |
| training | `batch_size`, `max_epochs`, `patience`, loss, optimizer, scheduler | 학습 조건 |
| runtime | `seed`, `device`, `num_workers` | 실행 환경과 재현성 |
| output | `exp_name`, `output_dir`, `checkpoint`, `save` | 산출물과 checkpoint |

사용자가 제공하지 않은 값만 `DEFAULTS`에서 채운다. Resolved config는 output 폴더에 저장해 checkpoint를
생성한 조립 조건을 추적한다.

### 3.2. Component-level config

Composable segmentation model의 예시는 다음과 같다.

```yaml
dataset: measured
method: seg
model:
  architecture: composable
  backbone:
    name: custom
    pretrained: false
  adapter:
    name: cnn
  decoder:
    name: unet
    upsample: interpolate_conv
    skip_connection: add
  head:
    name: mask
training:
  batch_size: 4
  max_epochs: 50
  patience: 5
  optimizer:
    name: adamw
  scheduler:
    name: reduce_on_plateau
runtime:
  seed: 42
  device: cuda
```

Wrapper가 method별 기본 loss와 optimizer parameter group을 결정한다. Config에 loss 또는 learning rate를
지정하면 wrapper의 해당 default만 override한다.

### 3.3. YAML과 dotted override

단일 YAML에서 비교할 leaf만 command line으로 변경할 수 있다.

```bash
python scripts/train.py \
  --config configs/seg_custom.yaml \
  --set model.decoder.name=plain \
  --set model.decoder.skip_connection=none \
  --save
```

Config 우선순위는 다음과 같다.

1. `scripts/config.py`의 `DEFAULTS`
2. `--config`로 지정한 YAML
3. 반복 가능한 `--set key=value`
4. `--device`, `--checkpoint`, `--output_dir` 같은 명시적 runtime argument

Unknown dotted key, type 변환 실패와 지원하지 않는 component 조합은 실행 전에 오류로 처리한다.

### 3.4. Output 경로와 실험 이름

기본 output 경로는 다음과 같다.

```text
outputs/<dataset>/<method>/<model>/<exp_name>/
```

`model`은 architecture, backbone과 주요 task component를 식별한다. `exp_name`은 loss, postprocess,
refinement, seed와 training 차이를 식별한다. 사용자가 지정하지 않으면 `get_experiment`가 결정적인
이름을 생성한다.

예상 산출물은 다음과 같다.

| 파일 | 생성 단계 | 내용 |
|---|---|---|
| `resolved_config.yaml` | 모든 mode | 병합이 끝난 실제 config |
| `model.pth` | train | model `state_dict` |
| `history.json` | train | epoch별 train과 validation 결과 |
| `eval_result.json` | evaluate | 공통 metric 결과 |
| `pred_corners.csv` | predict | 표준 corner 예측 |
| `run.log` | 모든 mode | 실행 log |

## 4. 단일 모델 실행

단일 실행은 `scripts/` entry point를 사용한다. 모든 script는 같은 config, data factory와 wrapper
factory를 공유한다.

### 4.1. Train

YAML config를 사용한 예시는 다음과 같다.

```bash
python scripts/train.py --config configs/reg_custom.yaml --save
```

간단한 ver1-style argument도 지원한다.

```bash
python scripts/train.py \
  --method reg \
  --backbone resnet50 \
  --batch_size 4 \
  --max_epochs 50 \
  --device cuda \
  --save
```

`scripts/train.py`는 train과 validation dataloader, wrapper와 `Trainer`를 생성한다. `patience`가 0보다
크면 `fit_early_stop`을 사용하고 그렇지 않으면 `fit`을 사용한다.

### 4.2. Evaluate

학습에 사용한 config와 checkpoint를 지정한다.

```bash
python scripts/evaluate.py \
  --config outputs/measured/reg/custom_gap/baseline/resolved_config.yaml \
  --checkpoint outputs/measured/reg/custom_gap/baseline/model.pth \
  --save
```

Evaluator는 wrapper의 training metric을 공통 metric bank로 교체한다. 결과에는 accuracy metric, SR과
failure reason 분포를 포함한다.

### 4.3. Predict

Label이 없는 image CSV에도 같은 checkpoint를 사용할 수 있다.

```bash
python scripts/predict.py \
  --config outputs/measured/reg/custom_gap/baseline/resolved_config.yaml \
  --checkpoint outputs/measured/reg/custom_gap/baseline/model.pth \
  --set data.has_corners=false \
  --save
```

`Predictor`는 dataset 순서를 유지하고 다음 column을 저장한다.

```text
image_dir,image_name,x1,y1,x2,y2,x3,y3,x4,y4
```

### 4.4. Checkpoint 재사용

Checkpoint는 model `state_dict`만 포함한다. 같은 model config로 wrapper를 생성한 뒤
`load_model(wrapper.model, checkpoint)`를 호출한다. Backbone, decoder, head 또는 output channel이
다르면 load를 중단하고 config mismatch를 보고한다.

Optimizer와 scheduler state를 이용한 training resume은 초기 범위에 포함하지 않는다. Model structure가
같으면 기존 checkpoint를 새로운 loss나 postprocessor로 평가할 수 있다.

## 5. 다중 실험 실행

Batch experiment는 ver1과 같이 `experiments/configs.py`의 dictionary 목록을 `experiments/run.py`가
script subprocess로 실행한다.

### 5.1. experiments/configs.py

Skip connection 비교 config의 예시는 다음과 같다.

```python
CONFIGS = [
    {
        "exp_name": "seg_plain",
        "dataset": "measured",
        "method": "seg",
        "model": {
            "architecture": "composable",
            "backbone": {"name": "custom", "pretrained": False},
            "decoder": {
                "name": "plain",
                "upsample": "interpolate_conv",
                "skip_connection": "none",
            },
            "head": {"name": "mask"},
        },
        "runtime": {"seed": 42},
    },
    {
        "exp_name": "seg_unet_add",
        "dataset": "measured",
        "method": "seg",
        "model": {
            "architecture": "composable",
            "backbone": {"name": "custom", "pretrained": False},
            "decoder": {
                "name": "unet",
                "upsample": "interpolate_conv",
                "skip_connection": "add",
            },
            "head": {"name": "mask"},
        },
        "runtime": {"seed": 42},
    },
]
```

두 config에서는 decoder와 skip 설정만 다르게 하고 data split, initialization, head, loss, optimizer와
postprocessor를 같게 유지한다.

### 5.2. experiments/run.py

전체 lifecycle 또는 특정 mode를 실행하는 예시는 다음과 같다.

```bash
python experiments/run.py --mode all
python experiments/run.py --mode train
python experiments/run.py --mode evaluate
python experiments/run.py --mode predict
```

`run.py`는 nested config를 dotted CLI argument로 변환하고 `scripts/<mode>.py`를 subprocess로 호출한다.
한 experiment가 실패해도 나머지를 계속 실행하고 마지막에 성공과 실패 목록을 출력한다.

### 5.3. Benchmark 실행

학습된 config를 공통 test set에서 비교한다.

```bash
python experiments/benchmark.py --device cuda
```

Benchmark row에는 config identity, accuracy, success, parameter, size, memory와 latency를 기록한다.
Model-only latency와 preprocess 및 postprocess를 포함한 end-to-end latency는 별도 column으로 구분한다.

## 6. 모델 비교 시나리오

각 비교에서는 한 번에 하나의 실험 축만 변경한다. 공통 설정 비교와 model별 tuning 결과는 서로 다른
table로 보고한다.

### 6.1. Coordinate head 비교

첫 regression 비교는 다음과 같다.

| 고정 요소 | 변경 요소 | 확인 항목 |
|---|---|---|
| CustomBackbone, initialization, target, Wing loss, postprocess와 data split | `gap`, `spatial` | spatial 정보가 corner precision에 미치는 영향 |

Parameter 수와 latency도 기록해 spatial head의 정확도 이득과 비용을 분리한다.

### 6.2. Segmentation skip connection 비교

첫 segmentation 비교는 다음과 같다.

| 기준 model | 비교 model | 고정 요소 |
|---|---|---|
| plain, skip none | U-Net, skip add | CustomBackbone, decoder stage, upsampling, mask head, loss, postprocess와 seed |

Mask BCE와 Dice뿐 아니라 Polygon IoU, MCD, MaxCD, PCK와 SR을 보고한다. Encoder stage의 fringe
texture가 geometry fitting에 미치는 영향도 failure reason으로 확인한다.

### 6.3. Upsampling 방식 비교

같은 U-Net additive decoder에서 `interpolate_conv`와 `transposed_conv`만 변경한다. Boundary precision,
artifact, parameter 수, latency와 peak memory를 비교한다.

### 6.4. Backbone 비교

같은 method와 head에서 backbone만 변경한다.

| 비교 후보 | 고정 요소 | 함께 기록할 metadata |
|---|---|---|
| CustomBackbone, ResNet, ViT, Swin | output, head, target, loss, postprocess와 training | pretrained 여부와 dataset, parameter, latency와 memory |

Pretrained model과 from-scratch model의 차이를 architecture 효과만으로 해석하지 않는다.

### 6.5. reg, seg와 det 비교

Method 간 비교에서는 같은 data split과 final metric bank를 사용하지만 native target과 loss는 각 wrapper의
계약을 따른다.

| method | raw output | 최종 변환 |
|---|---|---|
| `reg` | coordinate logits 또는 offsets | sigmoid reshape 또는 offset decode |
| `seg` | binary mask logits | threshold와 four-side fitting |
| `det` | boxes 또는 points와 confidence | selection, center decode와 ordering |

공통 corner metric, SR과 end-to-end 비용으로 비교한다. Native loss 값은 method 간 순위에 사용하지 않는다.

### 6.6. Postprocessor 비교

동일 mask checkpoint의 raw logits 또는 probability map을 저장한 뒤 four-side fitting, contour
approximation과 조건부 line refinement를 적용한다. Model inference를 반복하지 않고 postprocessor의
순수 효과를 측정한다.

### 6.7. Refinement 비교

같은 base prediction을 저장하고 refinement 없음, local STN과 GCN을 비교한다. 실패한 base corner에는
refinement를 적용하지 않으며 base와 refined metric, 추가 latency를 함께 기록한다.

## 7. Python API 사용

Notebook이나 분석 script에서도 CLI와 같은 factory와 wrapper를 사용한다.

```python
from src.core.evaluator import Evaluator
from src.core.factory import get_dataloader, get_wrapper
from src.core.predictor import Predictor
from src.core.trainer import Trainer

wrapper = get_wrapper(
    "seg",
    device="cuda",
    model={
        "architecture": "composable",
        "backbone": {"name": "custom", "pretrained": False},
        "decoder": {
            "name": "unet",
            "upsample": "interpolate_conv",
            "skip_connection": "add",
        },
        "head": {"name": "mask"},
    },
)

train_loader = get_dataloader("train", csv_path, batch_size=4, seed=42)
valid_loader = get_dataloader("valid", csv_path, batch_size=4, seed=42)
test_loader = get_dataloader("test", csv_path, batch_size=4, seed=42)

trainer = Trainer(wrapper, output_dir=output_dir)
history = trainer.fit_early_stop(train_loader, valid_loader, max_epochs=50, patience=5)

evaluator = Evaluator(wrapper, output_dir=output_dir)
metrics = evaluator.evaluate(test_loader)

predictor = Predictor(wrapper, output_dir=output_dir)
predictions = predictor.predict(test_loader)
```

외부에서 model, optimizer 또는 metric을 직접 주입할 때도 wrapper 생성자를 사용한다. Core runner를
상속하거나 복제하지 않는다.

## 8. Metric과 결과 해석

공통 benchmark metric은 다음과 같다.

| metric | 해석 | 좋은 방향 |
|---|---|---|
| Polygon IoU | quad 영역 일치도 | 큼 |
| MCD | 평균 corner 거리 | 작음 |
| MaxCD | 가장 큰 단일 corner 거리 | 작음 |
| Reprojection Error | homography 복원 오차 | 작음 |
| PCK@0.02, PCK@0.05 | threshold 안의 corner 비율 | 큼 |
| SR | 유효한 네 corner 반환 비율 | 큼 |
| CPU/GPU latency | end-to-end 실행 비용 | 작음 |
| Model size | 저장과 배포 비용 | 작음 |

Postprocess 실패 sample을 제외한 accuracy만 단독으로 보고하지 않는다. Accuracy, SR, failure reason
count와 전체 sample 수를 함께 기록한다.

## 9. 재현성과 공정 비교

실험 시작 전 고정하거나 기록할 항목은 다음과 같다.

- Dataset stage, source CSV와 split index를 기록한다.
- Seed, input size, augmentation, batch size와 sample 수를 고정한다.
- 변경 축 이외의 model component와 initialization을 고정한다.
- Optimizer, scheduler, learning rate, epoch와 early stopping 조건을 고정한다.
- Validation set에서 선택한 threshold를 test set에서 변경하지 않는다.
- External weight source, version, checksum, pretrained dataset과 license를 기록한다.
- Latency device, batch size, warm-up과 iteration 수를 기록한다.
- Resolved config와 checkpoint를 같은 output 폴더에 보존한다.

여러 seed에서는 seed별 결과와 평균 및 분산을 모두 보존한다.

## 10. 오류 처리와 점검표

대표 오류와 처리 방법은 다음과 같다.

| 증상 | 확인 항목 | 처리 |
|---|---|---|
| checkpoint key mismatch | backbone, decoder, head와 output channel | 학습 시 resolved config로 wrapper 재생성 |
| U-Net 또는 FPN 생성 실패 | backbone의 `stages` capability | multi-stage 지원 backbone 선택 |
| coordinate head 생성 실패 | `global` 또는 `spatial` capability | adapter와 head 요구 사항 확인 |
| prediction에 `NaN` 존재 | postprocess failure reason | SR과 reason distribution 확인 |
| output 충돌 | dataset, model과 `exp_name` | 고유한 experiment identity 지정 |
| 결과가 재현되지 않음 | seed, split, augmentation과 weight | resolved config diff 확인 |
| external model 학습 실패 | native target와 train mode | 전용 external wrapper 확인 |

최종 비교표 작성 전 점검 항목은 다음과 같다.

- 모든 experiment가 같은 test split을 사용한다.
- Checkpoint와 resolved config가 일치한다.
- 공통 metric bank와 corner ordering을 사용한다.
- 실패 sample 수와 failure reason이 저장되어 있다.
- Model-only 비용과 end-to-end 비용을 구분한다.
- 비교 축 이외의 변경 사항이 config diff에 없다.
