# reg backbone 비교 실험 실행

다음 표는 이 plan의 상태와 적용 범위를 정리한다.

| 항목 | 값 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-18 |
| 적용 범위 | `src/models/backbones/`, `src/models/reg/`, `scripts/config.py`, `scripts/train.py`, `experiments/configs.py`, `experiments/run.py` |
| 관련 문서 | [docs/architecture/model-assembly.md](../architecture/model-assembly.md), [docs/references/backbones.md](../references/backbones.md), [docs/plans/0002-reg-minimal-training-plan.md](0002-reg-minimal-training-plan.md) |

## 1. 목적과 배경

현재 `reg` method는 `CustomBackbone + CNNBackboneAdapter + CoordGapHead` 조합으로 최소 학습이
가능하다. 다음 비교 단계에서는 같은 coordinate head, target, loss, postprocess와 data split을
유지한 채 backbone만 `custom`과 `torchvision.models.resnet50`으로 바꾸어 validation 결과를 비교한다.

`docs/architecture/model-assembly.md`는 Category B에서 pretrained backbone composable model을
정의하고, `docs/references/backbones.md`는 로컬 `resnet50-0676ba61.pth` ImageNet-1K weight를
권장 backbone으로 기록한다. ver1의 `src/models/direct/model.py`는 torchvision ResNet backbone을
default로 사용했고, `experiments/configs.py`와 `experiments/run.py`는 여러 backbone config를
subprocess로 순차 실행했다. 이번 작업은 이 ver1 구현을 참고하되, ver2의 `FeatureExtractor`,
`CNNBackboneAdapter`, `CoordGapHead` component 경계를 유지하는 방식으로 ResNet-50 선택과 두 backbone
실험 묶음 실행 기능을 추가한다.

## 2. 범위

이번 plan에 포함하는 항목은 다음과 같다.

- `TorchvisionBackbone`을 추가해 `resnet50` backbone을 지원한다.
- `RegWrapper`와 `CustomRegModel`을 backbone 선택 가능 구조로 확장한다.
- `scripts/train.py --method reg --backbone custom`과 `--backbone resnet50`을 모두 지원한다.
- `scripts/config.py`의 output 경로를 `outputs/<dataset>/<method>/<model>/<exp_name>/` 규칙에 맞춘다.
- ver1과 같은 `CONFIGS` list-of-dicts 방식으로 `experiments/configs.py`에 custom backbone과
  ResNet-50 backbone 실험 config 두 개를 정의한다.
- ver1과 같은 subprocess runner 방식으로 `experiments/run.py`를 추가해 한 번의 명령으로 두 config를
  순차 학습한다.

이번 plan에서 제외하는 항목은 다음과 같다.

- test split 전용 평가 스크립트 추가
- ResNet-18, MobileNet, EfficientNet, ViT, Swin 등 다른 backbone 지원
- parallel multi-process 학습 실행
- 새로운 dataset 생성 또는 CSV 구조 변경
- metric bank 확장과 latency benchmark

## 3. 구현 계획

### 3.1. ResNet-50 backbone 조립

`src/models/backbones/torchvision_backbone.py`를 추가한다. 파일은 project Python 규칙에 맞춰 첫 줄
header, 한 줄 class docstring, `os.path` 기반 경로 처리를 사용한다.

구현은 ver1 `src/models/direct/model.py`의 다음 결정을 참고한다.

- `torchvision.models`에서 builder를 가져오고 `weights=None`으로 network를 만든다.
- 로컬 `/mnt/d/backbones/resnet50-0676ba61.pth` 파일을 `torch.load(..., map_location="cpu",
  weights_only=True)`로 읽는다.
- 로컬 state dict를 `load_state_dict`로 적용하고 네트워크 다운로드는 사용하지 않는다.
- ResNet-50의 `fc.in_features`가 `2048`이라는 channel 정보를 head 입력 channel로 사용한다.

다만 ver1 코드를 그대로 복사하지 않고 다음처럼 ver2 구조에 맞게 분리한다.

- ver1의 `DirectModel`처럼 `net.fc = nn.Identity()`를 붙인 whole ResNet을 직접 head에 연결하지 않는다.
- ResNet stage forward를 `TorchvisionBackbone` 책임으로 분리한다.
- GAP 처리는 기존 `CNNBackboneAdapter`가 담당한다.
- coordinate projection은 기존 `CoordGapHead`가 담당한다.
- wrapper optimizer parameter group은 `self.model.extractor.parameters()`와 head parameter 기준을 유지한다.

지원 범위는 다음처럼 고정한다.

| 항목 | 결정 |
| --- | --- |
| backbone id | `resnet50` |
| source | `torchvision.models.resnet50` |
| weight | `/mnt/d/backbones/resnet50-0676ba61.pth` |
| pretrained 기본값 | `True` |
| final feature | `layer4` output |
| stages | `[layer1, layer2, layer3, layer4]` |
| global channel | `2048` |
| output stride | `32` |

Forward 흐름은 다음과 같다.

```text
images
-> conv1
-> bn1
-> relu
-> maxpool
-> layer1
-> layer2
-> layer3
-> layer4
-> {"final": layer4, "stages": [layer1, layer2, layer3, layer4]}
```

Weight loading은 로컬 파일이 있으면 ver1과 같이 `torch.load(..., map_location="cpu",
weights_only=True)`로 state dict를 읽어 `strict=True`로 적용한다. 로컬 파일이 없으면 네트워크
다운로드를 시도하지 않고 명시적 `FileNotFoundError`를 발생시킨다.

### 3.2. `reg` model과 wrapper 확장

`src/models/reg/model.py`는 기존 custom 전용 구현을 보존하면서 backbone 선택을 받도록 확장한다.
public class 이름은 기존 import 안정성을 위해 `CustomRegModel`을 유지하되, `backbone="custom"`과
`backbone="resnet50"`을 인자로 받게 한다.

조립 규칙은 다음과 같다.

| 요청 backbone | 조립 |
| --- | --- |
| `custom` 또는 `None` | `CustomBackbone + CNNBackboneAdapter + CoordGapHead` |
| `resnet50` | `TorchvisionBackbone("resnet50") + CNNBackboneAdapter + CoordGapHead` |

지원하지 않는 backbone 값은 silent fallback 없이 `ValueError`를 발생시킨다. 오류 메시지에는 요청한
backbone과 지원 목록 `custom, resnet50`을 포함한다.

`src/models/reg/wrapper.py`는 `backbone="custom"` 인자를 추가하고 `CustomRegModel`에 전달한다. 기존
optimizer의 두 parameter group 구조는 유지한다.

### 3.3. CLI config와 output 경로

`scripts/config.py`를 다음 정책으로 확장한다.

| 설정 | 기본값 |
| --- | --- |
| `dataset` | `public` |
| `method` | `reg` |
| `backbone` | `custom` |
| `model` | backbone과 head에서 자동 생성 |

`get_model_name(cfg)`를 추가하고 다음 값을 반환하게 한다.

| backbone | model name |
| --- | --- |
| `custom` | `custom_coord_gap` |
| `resnet50` | `resnet50_coord_gap` |

`get_output_dir(cfg, base="outputs")`는 다음 경로를 반환한다.

```text
outputs/<dataset>/<method>/<model>/<exp_name>/
```

`get_experiment(cfg)`는 기존 비교가 쉬운 이름을 유지하되 backbone을 포함한다.

```text
reg_bs<batch_size>_ep<max_epochs>_<backbone>
```

### 3.4. Experiments runner

`experiments/` 디렉터리를 추가하고 빈 `__init__.py`를 둔다. ver1의
`experiments/configs.py`처럼 module-level `CONFIGS` list에 실험 dict를 직접 나열한다. 이번 plan의
초기 config는 다음 두 개로 고정한다.

```python
CONFIGS = [
    {"method": "reg", "batch_size": 4, "max_epochs": 5, "backbone": "custom"},
    {"method": "reg", "batch_size": 4, "max_epochs": 5, "backbone": "resnet50"},
]
```

두 config는 `scripts/config.py`의 기본 `csv_path`, `seed`, `image_size`, `patience`, `train_size`,
`valid_size`와 `num_workers`를 그대로 사용한다. 이후 실험 규모를 키울 때는 ver1과 같이 이 dict에
필요한 key를 추가한다.

`experiments/run.py`는 ver1과 같은 subprocess 실행 방식을 사용한다. project root를 `sys.path`에
추가하고, `CONFIGS`를 읽은 뒤 각 config dict를 CLI argument로 변환해 `scripts/train.py`를 호출한다.
`scripts/train.py` 내부 helper를 새로 분리하지 않는다.

Runner 구현 정책은 다음과 같다.

```python
PASS_KEYS = [
    "backbone", "device", "batch_size", "max_epochs", "num_workers",
    "train_size", "valid_size", "test_size", "checkpoint", "output_dir",
]
```

`get_cli_args(cfg)`는 ver1처럼 `["--method", cfg["method"], "--save"]`로 시작하고, `PASS_KEYS`에
있는 key만 `--key value` 형태로 추가한다. 따라서 `experiments/configs.py`의 dict가 실험별 override의
단일 source가 된다.

Runner 동작은 다음처럼 고정한다.

- 기본 실행은 두 config를 순차 학습한다.
- `--mode`는 ver1 구조를 따르되 이번 plan에서는 `train`만 지원한다.
- 각 config의 output directory가 이미 있어도 덮어쓰기 검사는 하지 않고, 기존 `history.json`은 같은
  경로에서 갱신될 수 있다.
- 한 config가 실패해도 나머지 config를 계속 실행하고, 마지막에 성공과 실패 요약을 출력한다.
- 실패가 하나라도 있으면 process exit code는 `1`로 반환한다.

기본 실행 명령은 다음과 같다.

```bash
conda activate pytorch_env
python experiments/run.py
```

빠른 smoke 검증은 개별 `scripts/train.py` 명령으로 수행하고, `experiments/run.py`는 실제
`CONFIGS` 두 항목을 순차 실행하는지 확인한다.

## 4. 완료 기준

이 plan은 다음 조건을 만족하면 `Done`으로 본다.

- `docs/plans/0003-reg-backbone-experiments-plan.md` 상태가 `Approved`에서 `Done`으로 갱신되어 있다.
- `TorchvisionBackbone("resnet50")` 생성과 dummy forward가 성공한다.
- `RegWrapper(backbone="custom")`와 `RegWrapper(backbone="resnet50")`가 모두 `(B, 8)` raw output을 반환한다.
- `python scripts/train.py --method reg --backbone custom ...` smoke 학습이 성공한다.
- `python scripts/train.py --method reg --backbone resnet50 ...` smoke 학습이 성공한다.
- ver1 방식의 `CONFIGS` list와 subprocess runner가 custom과 ResNet-50 두 config를 순차 실행한다.
- 각 실행 결과가 `outputs/public/reg/custom_coord_gap/<exp_name>/`와
  `outputs/public/reg/resnet50_coord_gap/<exp_name>/` 아래에 저장된다.

## 5. 검증

구현 후 다음 순서로 검증한다.

```bash
conda activate pytorch_env
python -c "from src.models.backbones.torchvision_backbone import TorchvisionBackbone; m = TorchvisionBackbone('resnet50'); print(m.out_channels)"
python -c "import torch; from src.models.reg.wrapper import RegWrapper; w = RegWrapper(backbone='resnet50', device='cpu'); print(w.model(torch.zeros(2, 3, 224, 224)).shape)"
python scripts/train.py --method reg --backbone custom --device cpu --train_size 2 --valid_size 2 --batch_size 1 --max_epochs 1 --patience 1 --num_workers 0 --output_dir /tmp/reg_custom_smoke
python scripts/train.py --method reg --backbone resnet50 --device cpu --train_size 2 --valid_size 2 --batch_size 1 --max_epochs 1 --patience 1 --num_workers 0 --output_dir /tmp/reg_resnet50_smoke
python experiments/run.py
```

검증 결과에서는 ResNet-50이 로컬 weight를 사용했는지, 두 backbone output directory가 분리되었는지,
기존 custom backbone 기본 실행이 유지되는지를 확인한다.
