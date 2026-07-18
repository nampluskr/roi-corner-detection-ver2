# reg backbone과 head matrix 비교 실험

다음 표는 이 plan의 상태와 적용 범위를 정리한다.

| 항목 | 값 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-18 |
| 적용 범위 | `src/models/heads/coordinate_head.py`, `src/models/reg/`, `scripts/config.py`, `experiments/configs.py`, `experiments/run.py` |
| 관련 문서 | [docs/architecture/model-assembly.md](../architecture/model-assembly.md), [docs/plans/0002-reg-minimal-training-plan.md](0002-reg-minimal-training-plan.md), [docs/plans/0003-reg-backbone-experiments-plan.md](0003-reg-backbone-experiments-plan.md) |

## 1. 목적과 배경

현재 프로젝트의 `reg` method는 새로 구현된 `CustomBackbone`과 ver1의 torchvision ResNet-50 경로를
함께 지원하지만, coordinate head는 `coord_gap` 하나만 사용한다. `docs/architecture/model-assembly.md`
Section 4.2는 `coord_gap`과 `coord_spatial`을 `CustomRegModel`의 head variant로 정의하며,
Section 11.4의 첫 단계도 두 head의 비교를 요구한다.

ver1의 `src/models/direct/model.py`는 torchvision backbone을 default로 사용했고, `head_type="gap"`과
`head_type="spatial"`을 선택할 수 있었다. 이번 작업은 ver1의 spatial head 아이디어를 참고하되,
ver2의 `FeatureExtractor`, `CNNBackboneAdapter`, `CoordGapHead` component 경계를 유지하면서
`CoordSpatialHead`를 추가한다. 결과적으로 `custom`, `resnet50` 두 backbone과 `coord_gap`,
`coord_spatial` 두 head를 조합한 총 4개 조건을 같은 train/valid 흐름으로 평가한다.

## 2. 범위

이번 plan에 포함하는 항목은 다음과 같다.

- `CoordSpatialHead`를 추가해 final spatial feature map에서 `(B, 8)` raw coordinate를 예측한다.
- `CustomRegModel`과 `RegWrapper`에 `head` 또는 `head_type` 선택 인자를 추가한다.
- `coord_gap`은 기존 `FeatureBundle.global_feature`를 사용하고, `coord_spatial`은
  `FeatureBundle.spatial_feature`를 사용한다.
- `scripts/config.py`가 `--head`를 받고 wrapper kwargs, model name, experiment name에 반영한다.
- `experiments/configs.py`가 `2 backbones * 2 heads`의 총 4개 config를 나열한다.
- `experiments/run.py`의 `PASS_KEYS`에 `head`를 추가해 ver1 방식 subprocess 실행을 유지한다.

이번 plan에서 제외하는 항목은 다음과 같다.

- segmentation, heatmap, detection head 구현
- ResNet-18, ResNet-34, MobileNet 등 추가 backbone 확장
- head별 learning rate 또는 optimizer 정책 분리
- test split 평가 스크립트와 latency benchmark
- canonical 문서 변경

Canonical 문서에는 이미 `coord_spatial` head variant가 정의되어 있으므로 이번 plan에서는
architecture SSOT를 변경하지 않는다.

## 3. 구현 계획

### 3.1. Coordinate head 확장

`src/models/heads/coordinate_head.py`에 `CoordSpatialHead`를 추가한다. 기존 `CoordGapHead`는 변경하지
않는다.

`CoordSpatialHead`의 구조는 ver1 `DirectModel(head_type="spatial")`을 참고해 다음처럼 고정한다.

```text
spatial_feature
-> Conv2d(in_channels, 128, kernel_size=3, stride=2, padding=1)
-> ReLU(inplace=True)
-> Conv2d(128, 64, kernel_size=3, stride=2, padding=1)
-> ReLU(inplace=True)
-> AdaptiveAvgPool2d(4)
-> Flatten
-> Dropout
-> Linear(64 * 4 * 4, 8)
```

출력은 `coord_gap`과 동일하게 `(B, 8)` raw coordinate logits다. Loss와 postprocessor는 기존
`WingLoss(apply_sigmoid=True)`와 `RegPostprocessor`를 그대로 사용한다.

### 3.2. `reg` model과 wrapper 확장

`CustomRegModel`에 `head="coord_gap"` 인자를 추가한다. 지원 head는 `coord_gap`과 `coord_spatial`만
허용한다.

조립 규칙은 다음과 같다.

| head | adapter 설정 | FeatureSpec | head input |
| --- | --- | --- | --- |
| `coord_gap` | `keep_spatial=False`, `keep_stages=False` | `global_channels=encoder.out_channels` | `bundle.global_feature` |
| `coord_spatial` | `keep_spatial=True`, `keep_stages=False` | `global_channels=encoder.out_channels`, `spatial_channels=encoder.out_channels` | `bundle.spatial_feature` |

`forward()`는 head 값에 따라 global 또는 spatial feature를 선택한다. 지원하지 않는 head 값은 silent
fallback 없이 `ValueError`를 발생시키고, 오류 메시지에는 지원 목록 `coord_gap, coord_spatial`을
포함한다.

`RegWrapper`는 `head="coord_gap"` 인자를 받아 `CustomRegModel`에 전달한다. 기존 optimizer는
`self.model.extractor.parameters()`와 나머지 head parameter group을 분리하는 정책을 유지한다.

### 3.3. CLI config와 output 경로

`scripts/config.py`에 `head="coord_gap"` 기본값과 `--head` CLI 인자를 추가한다. `get_wrapper_kwargs()`는
`backbone`과 함께 `head`도 wrapper에 전달한다.

`get_model_name(cfg)`는 backbone과 head를 함께 사용해 다음 값을 반환한다.

| backbone | head | model name |
| --- | --- | --- |
| `custom` | `coord_gap` | `custom_coord_gap` |
| `custom` | `coord_spatial` | `custom_coord_spatial` |
| `resnet50` | `coord_gap` | `resnet50_coord_gap` |
| `resnet50` | `coord_spatial` | `resnet50_coord_spatial` |

`get_experiment(cfg)`는 head까지 포함해 다음 형식으로 확장한다.

```text
reg_bs<batch_size>_ep<max_epochs>_<backbone>_<head>
```

기존 `coord_gap` output directory는 새 experiment name에 head suffix가 붙으므로 `0003` 결과와
구분된다. `outputs/<dataset>/<method>/<model>/<exp_name>/` 경로 규칙은 유지한다.

### 3.4. Experiments config와 runner

`experiments/configs.py`는 ver1과 같은 module-level `CONFIGS` list-of-dicts 방식을 유지하고, 다음
네 config를 나열한다.

```python
CONFIGS = [
    {"method": "reg", "batch_size": 4, "max_epochs": 5, "backbone": "custom", "head": "coord_gap"},
    {"method": "reg", "batch_size": 4, "max_epochs": 5, "backbone": "custom", "head": "coord_spatial"},
    {"method": "reg", "batch_size": 4, "max_epochs": 5, "backbone": "resnet50", "head": "coord_gap"},
    {"method": "reg", "batch_size": 4, "max_epochs": 5, "backbone": "resnet50", "head": "coord_spatial"},
]
```

`experiments/run.py`는 subprocess 실행 방식을 유지하고 `PASS_KEYS`에 `head`만 추가한다.

```python
PASS_KEYS = [
    "backbone", "head", "device", "batch_size", "max_epochs", "num_workers",
    "train_size", "valid_size", "test_size", "checkpoint", "output_dir",
]
```

기본 실행 명령은 다음과 같다.

```bash
conda activate pytorch_env
python experiments/run.py
```

## 4. 완료 기준

이 plan은 다음 조건을 만족하면 `Done`으로 본다.

- `docs/plans/0004-reg-head-backbone-matrix-plan.md` 상태가 `Approved`에서 `Done`으로 갱신되어 있다.
- `CoordGapHead`와 `CoordSpatialHead`가 모두 dummy input에서 `(B, 8)`을 반환한다.
- `RegWrapper(backbone="custom", head="coord_gap")`가 `(B, 8)` raw output을 반환한다.
- `RegWrapper(backbone="custom", head="coord_spatial")`가 `(B, 8)` raw output을 반환한다.
- `RegWrapper(backbone="resnet50", head="coord_gap")`가 `(B, 8)` raw output을 반환한다.
- `RegWrapper(backbone="resnet50", head="coord_spatial")`가 `(B, 8)` raw output을 반환한다.
- `python scripts/train.py ...` smoke 학습이 네 조건 모두에서 성공한다.
- `experiments/configs.py`가 4개 config를 포함하고, `experiments/run.py`가 `head`를 CLI로 전달한다.
- 각 실행 결과가 `outputs/public/reg/<model>/<exp_name>/` 아래에서 네 model directory로 분리된다.

## 5. 검증

구현 후 다음 순서로 검증한다.

```bash
conda activate pytorch_env
python -c "import torch; from src.models.heads.coordinate_head import CoordGapHead, CoordSpatialHead; print(CoordGapHead(16)(torch.zeros(2, 16)).shape); print(CoordSpatialHead(16)(torch.zeros(2, 16, 14, 14)).shape)"
python -c "import torch; from src.models.reg.wrapper import RegWrapper; configs=[('custom','coord_gap'),('custom','coord_spatial'),('resnet50','coord_gap'),('resnet50','coord_spatial')]; [print(b, h, RegWrapper(backbone=b, head=h, device='cpu').model(torch.zeros(2, 3, 224, 224)).shape) for b, h in configs]"
python scripts/train.py --method reg --backbone custom --head coord_gap --device cpu --train_size 2 --valid_size 2 --batch_size 1 --max_epochs 1 --patience 1 --num_workers 0 --output_dir /tmp/reg_custom_gap_smoke
python scripts/train.py --method reg --backbone custom --head coord_spatial --device cpu --train_size 2 --valid_size 2 --batch_size 1 --max_epochs 1 --patience 1 --num_workers 0 --output_dir /tmp/reg_custom_spatial_smoke
python scripts/train.py --method reg --backbone resnet50 --head coord_gap --device cpu --train_size 2 --valid_size 2 --batch_size 1 --max_epochs 1 --patience 1 --num_workers 0 --output_dir /tmp/reg_resnet50_gap_smoke
python scripts/train.py --method reg --backbone resnet50 --head coord_spatial --device cpu --train_size 2 --valid_size 2 --batch_size 1 --max_epochs 1 --patience 1 --num_workers 0 --output_dir /tmp/reg_resnet50_spatial_smoke
python -c "from experiments.configs import CONFIGS; from scripts.config import get_output_dir; print(len(CONFIGS)); print([get_output_dir(c) for c in CONFIGS])"
```

검증 결과에서는 네 조건의 raw output shape, smoke train 성공 여부, output directory 분리, 기존
`coord_gap` 동작 유지 여부를 확인한다.
