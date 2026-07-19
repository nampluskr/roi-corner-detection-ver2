# 학습 warmup_epochs 정책 추가

다음 표는 이 plan의 상태와 적용 범위를 정리한다.

| 항목 | 값 |
|---|---|
| 상태 | Superseded |
| 작성일 | 2026-07-19 |
| 적용 범위 | `docs/architecture/model-assembly.md`, `docs/guides/model-usage-codex.md`, `docs/guides/model-usage-claude.md`, `experiments/configs.py`, `experiments/run.py`, `scripts/config.py`, `src/core/trainer.py`, `src/models/base/base_wrapper.py`, `src/models/reg/wrapper.py`, `src/models/seg/wrapper.py`, `src/models/det/wrapper.py` |
| 관련 문서 | [docs/architecture/model-assembly.md](../architecture/model-assembly.md), [docs/plans/0003-reg-backbone-experiments-plan.md](0003-reg-backbone-experiments-plan.md), [docs/plans/0009-seg-unet-backbone-plan.md](0009-seg-unet-backbone-plan.md), [docs/plans/0010-torchseg-model-plan.md](0010-torchseg-model-plan.md), [docs/plans/0011-det-custom-model-plan.md](0011-det-custom-model-plan.md) |

이 plan은 [docs/plans/0020-reg-two-stage-training-plan.md](0020-reg-two-stage-training-plan.md)로
대체됐다. 아래 본문은 작성 시점의 초안을 historical record로 보존한다.

## 1. 목적과 배경

현재 `reg`, `seg`, `det` wrapper는 composable model에서 `self.model.extractor.parameters()`를
backbone parameter group으로 분리하고, extractor를 제외한 나머지 parameter를 head 또는 non-backbone
parameter group으로 묶는다. 기본 optimizer는 backbone group에 `lr=1e-5`, non-backbone group에
`lr=1e-4`를 적용한다.

이 구조는 pretrained backbone을 fine-tuning할 때 새로 초기화된 head, decoder 또는 neck의 큰 gradient가
pretrained representation을 초반부터 과하게 흔드는 것을 줄이기 위한 differential learning rate 정책에
해당한다. 그러나 현재 코드에는 학습 초반에 backbone을 잠시 고정하고 non-backbone module만 먼저 학습한
뒤 backbone fine-tuning을 시작하는 `warmup_epochs` 정책이 없다.

이번 plan은 `warmup_epochs`를 training policy로 추가한다. `warmup_epochs=1`이면 epoch 1에서는
backbone extractor를 freeze하고 non-backbone parameter만 학습하며, epoch 2부터 extractor를 unfreeze해
기존 differential learning rate로 fine-tuning한다. `backbone="custom"`은 pretrained prior를 보호할
대상이 아니므로 warmup을 적용하지 않고, 기본 optimizer도 단일 learning rate 정책을 우선 후보로 둔다.

## 2. 범위

이번 plan에 포함하는 항목은 다음과 같다.

- canonical 문서에 training policy 축으로 `warmup_epochs`, `optimizer_policy`, `backbone_lr`,
  `non_backbone_lr`를 명시한다.
- `warmup_epochs`의 의미를 epoch 단위 2단계 학습으로 정의한다. `epoch <= warmup_epochs`에서는
  composable model의 `model.extractor` parameter를 freeze하고, 이후 epoch에서는 unfreeze한다.
- `warmup_epochs` 기본값은 `0`으로 두어 기존 실험 동작을 보존한다.
- pretrained composable backbone에서는 사용자가 지정한 `warmup_epochs`를 적용한다. 초기 권장 실험값은
  `warmup_epochs=1`이다.
- `backbone="custom"`에서는 `warmup_epochs`를 적용하지 않는다. 사용자가 `warmup_epochs > 0`을 지정해도
  wrapper는 이를 무시하거나 `0`으로 정규화하고, 로그 또는 history metadata에 실제 적용값을 남긴다.
- `backbone="custom"` composable model의 기본 optimizer는 전체 trainable parameter에 단일
  `lr=1e-4`를 적용하는 정책으로 정리한다.
- pretrained composable model의 기본 optimizer는 기존처럼 extractor group `lr=1e-5`, non-backbone group
  `lr=1e-4`를 유지한다.
- `TorchSegModel` 같은 external whole-model variant는 `model.extractor` 계약이 없으므로 이번
  `warmup_epochs` 적용 대상에서 제외한다. 이 경우 기본 optimizer는 기존처럼 전체 parameter 단일
  `lr=1e-4`를 유지한다.
- CLI와 batch experiment config에서 `warmup_epochs`를 지정할 수 있게 한다.
- training log 또는 saved history에서 warmup 적용 여부와 optimizer policy를 확인할 수 있게 한다.

이번 plan에서 제외하는 항목은 다음과 같다.

- learning rate scheduler warmup, cosine schedule, one-cycle schedule, linear ramp 같은 LR schedule
  warmup은 제외한다. 이번 plan의 warmup은 backbone freeze와 unfreeze만 의미한다.
- layer-wise learning rate decay, 일부 stage만 unfreeze하는 gradual unfreezing, BatchNorm만 별도 처리하는
  정책은 제외한다.
- `warmup_epochs` 값을 자동 탐색하는 ablation 실행과 결과 분석은 제외한다. 이번 plan은 설정과 실행
  경로를 추가하고 smoke 검증까지만 수행한다.
- external detector나 external segmentation whole model 내부 backbone을 분리해 freeze하는 기능은
  제외한다. 이들은 각 whole-model adapter별 구조 분석이 필요하므로 후속 plan에서 다룬다.

## 3. 구현 계획

### 3.1. training policy 계약

`warmup_epochs`는 non-negative integer로 정의한다. `0`이면 warmup을 사용하지 않고 모든 trainable
parameter가 epoch 1부터 학습된다. `1`이면 첫 epoch에서만 backbone extractor가 freeze되고, epoch 2
시작 전에 unfreeze된다.

optimizer policy는 다음 표의 기준을 따른다.

| model 종류 | 조건 | `warmup_epochs` 적용 | 기본 optimizer policy |
|---|---|---|---|
| custom composable | `isinstance(self.model, CustomRegModel)`(reg 기준, seg/det는 대응하는 custom model class) | 적용하지 않음 | `single_lr`, 전체 parameter `lr=1e-4` |
| pretrained composable | custom model class가 아니고 `model.extractor` 있음(`TorchRegModel` 포함) | 적용함 | `differential_lr`, extractor `lr=1e-5`, non-backbone `lr=1e-4` |
| external whole model | `model.extractor` 없음 | 적용하지 않음 | `single_lr`, 전체 parameter `lr=1e-4` |

reg는 `RegModel`이 `CustomRegModel`과 `TorchRegModel`로 나뉘어 있으므로([docs/plans/0019-reg-model-split-plan.md](0019-reg-model-split-plan.md))
판정 기준이 `backbone` 문자열 비교가 아니라 class 비교다. seg와 det는 아직 문자열/`hasattr` 기준을
유지하며, 각 method가 custom/pretrained model class를 분리하는 시점에 동일한 방식으로 옮겨간다.

`head`라는 이름은 method에 따라 실제 범위가 달라질 수 있으므로, 구현과 문서에서는 extractor를 제외한
parameter group을 `non_backbone`으로 부른다. `reg`에서는 coordinate head에 가깝고, `seg`에서는 decoder와
mask head, `det`에서는 neck과 detection head가 포함된다.

### 3.2. `BaseWrapper` 확장

`src/models/base/base_wrapper.py`는 warmup의 공통 상태와 hook을 관리한다. 초기 필드는 다음과 같다.

```python
self.warmup_epochs = warmup_epochs
self.applied_warmup_epochs = 0
self.optimizer_policy = None
```

`BaseWrapper`에는 다음 helper를 추가한다.

- `get_backbone_module()`은 기본적으로 `self.model.extractor`를 반환하고, 없으면 `None`을 반환한다.
- `set_backbone_trainable(trainable)`은 backbone module의 parameter `requires_grad`를 일괄 변경한다.
- `uses_backbone_warmup()`은 `applied_warmup_epochs > 0`이고 backbone module이 있을 때만 `True`를
  반환한다.
- `on_fit_start(max_epochs)`는 warmup을 사용할 경우 학습 시작 전에 backbone을 freeze한다.
- `on_epoch_start(epoch)`는 `epoch == applied_warmup_epochs + 1` 시점에 backbone을 unfreeze한다.

`requires_grad=False`인 parameter가 optimizer group에 남아 있어도 gradient가 생성되지 않으므로 optimizer를
재생성하지 않는다. 이렇게 하면 scheduler와 optimizer state를 유지하면서 phase 전환만 수행할 수 있다.

### 3.3. `Trainer` hook 추가

`src/core/trainer.py`의 `fit()`과 `fit_early_stop()`은 epoch loop 안에서 train epoch를 시작하기 전에
`self.wrapper.on_epoch_start(epoch)`를 호출한다. 호출 위치는 다음 순서를 따른다.

```text
on_fit_start(max_epochs)
for epoch in 1..max_epochs:
    on_epoch_start(epoch)
    train(...)
    evaluate(...)
    on_epoch_end(valid_score)
```

이 순서는 `warmup_epochs=1`일 때 epoch 1 train 전에 freeze가 유지되고, epoch 2 train 전에 unfreeze가
적용되도록 보장한다.

### 3.4. method wrapper별 optimizer 정책

`RegWrapper`, `SegWrapper`, `DetWrapper`는 `warmup_epochs=0` kwarg를 받는다. wrapper는 model 생성 후
model class와 `extractor` 존재 여부에 따라 실제 적용 policy를 정한다.

`RegWrapper`는 `CustomRegModel`과 `TorchRegModel`을 모두 다루므로 `isinstance(self.model, CustomRegModel)`
여부로 기본 optimizer를 나눈다 - `CustomRegModel`이면 warmup을 적용하지 않고 단일 learning rate를
쓰고, `TorchRegModel`이면 warmup을 적용하고 differential learning rate를 쓴다. `DetWrapper`는 아직
`DetModel` 하나만 다루므로 `backbone == "custom"` 여부로 기본 optimizer를 나눈다. `SegWrapper`는
`SegModel`과 `TorchSegModel`을 모두 다루므로 `hasattr(self.model, "extractor")`와 `backbone == "custom"`을
함께 확인한다.

pretrained composable model의 optimizer group 구성은 기존 계약을 유지한다.

```python
backbone_ids = {id(p) for p in self.model.extractor.parameters()}
non_backbone_params = [p for p in self.model.parameters() if id(p) not in backbone_ids]
AdamW([
    {"params": self.model.extractor.parameters(), "lr": 1e-5},
    {"params": non_backbone_params, "lr": 1e-4},
])
```

custom composable model과 external whole model의 optimizer는 단일 group을 사용한다.

```python
AdamW(self.model.parameters(), lr=1e-4)
```

### 3.5. CLI와 experiment config

`scripts/config.py`는 `DEFAULTS`에 `warmup_epochs=0`을 추가하고, `parse_args()`에
`--warmup_epochs` integer argument를 추가한다. `get_wrapper_kwargs()`는 값이 `None`이 아니면
`warmup_epochs`를 wrapper로 전달한다.

`experiments/run.py::PASS_KEYS`에는 `warmup_epochs`를 추가한다. 이후 batch config는 pretrained
backbone 실험에만 다음처럼 값을 명시한다.

```python
{"method": "reg", "batch_size": 4, "max_epochs": 5, "backbone": "resnet50", "head": "coord_spatial", "warmup_epochs": 1}
```

`experiments/configs.py`의 기존 후보 목록을 일괄 변경하지는 않는다. warmup ablation이 필요할 때 동일
backbone config를 하나 복제하고 `warmup_epochs=1`만 추가해 비교 축을 분리한다.

## 4. 완료 기준

이번 plan은 다음 조건을 만족하면 `Done`으로 볼 수 있다.

- canonical 문서가 `warmup_epochs`를 backbone freeze 기반 training policy로 설명한다.
- CLI에서 `--warmup_epochs 1`을 받을 수 있고 wrapper까지 전달된다.
- `Trainer.fit()`과 `Trainer.fit_early_stop()` 모두 epoch 시작 hook을 호출한다.
- pretrained composable model은 `warmup_epochs=1`일 때 epoch 1에서 extractor parameter가
  `requires_grad=False`, epoch 2에서 `requires_grad=True`가 된다.
- `backbone="custom"`은 사용자가 `warmup_epochs=1`을 지정해도 실제 적용값이 `0`이고, 전체 parameter가
  단일 learning rate optimizer로 학습된다.
- `TorchSegModel` 같은 `extractor` 없는 whole model은 warmup을 적용하지 않고 단일 learning rate
  optimizer를 유지한다.
- 기존 `warmup_epochs=0` 실행은 기존 학습 동작과 호환된다.
- optimizer policy와 실제 적용 warmup epoch 수가 log 또는 history metadata에서 확인 가능하다.

## 5. 검증

구현 후 검증은 conda 환경 `pytorch_env`에서 수행한다.

```bash
conda activate pytorch_env
python -c "from src.models.reg.wrapper import RegWrapper; w = RegWrapper(backbone='custom', warmup_epochs=1, device='cpu'); print(w.applied_warmup_epochs, len(w.optimizer.param_groups))"
python -c "from src.models.reg.wrapper import RegWrapper; w = RegWrapper(backbone='resnet18', warmup_epochs=1, device='cpu'); w.on_fit_start(2); print(w.applied_warmup_epochs, next(w.model.extractor.parameters()).requires_grad); w.on_epoch_start(2); print(next(w.model.extractor.parameters()).requires_grad)"
python -c "from src.models.seg.wrapper import SegWrapper; w = SegWrapper(model='unet', backbone='custom', warmup_epochs=1, device='cpu'); print(w.applied_warmup_epochs, len(w.optimizer.param_groups))"
python -c "from src.models.det.wrapper import DetWrapper; w = DetWrapper(backbone='resnet18', warmup_epochs=1, device='cpu'); w.on_fit_start(2); print(w.applied_warmup_epochs, next(w.model.extractor.parameters()).requires_grad); w.on_epoch_start(2); print(next(w.model.extractor.parameters()).requires_grad)"
python scripts/train.py --method reg --backbone custom --head coord_gap --device cpu --max_epochs 1 --train_size 2 --valid_size 2 --num_workers 0 --warmup_epochs 1 --output_dir /tmp/reg_custom_warmup_smoke
python scripts/train.py --method reg --backbone resnet18 --head coord_gap --device cpu --max_epochs 2 --train_size 2 --valid_size 2 --num_workers 0 --warmup_epochs 1 --output_dir /tmp/reg_resnet18_warmup_smoke
```

검증 결과에서는 custom backbone의 실제 warmup 적용값이 `0`인지, pretrained backbone의 freeze와 unfreeze
전환이 epoch 경계에서 일어나는지, 기존 smoke train이 정상 완료되는지 확인한다.
