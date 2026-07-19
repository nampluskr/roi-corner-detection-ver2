# reg 2단계 학습(freeze-then-unfreeze) 정책 도입

다음 표는 이 plan의 상태와 적용 범위를 정리한다.

| 항목 | 값 |
|---|---|
| 상태 | Done |
| 작성일 | 2026-07-19 |
| 적용 범위 | `docs/architecture/model-assembly.md`, `docs/plans/0015-training-warmup-plan.md`, `src/models/base/base_wrapper.py`, `src/core/trainer.py`, `src/models/reg/wrapper.py`, `scripts/config.py` |
| 관련 문서 | [docs/architecture/model-assembly.md](../architecture/model-assembly.md), [docs/plans/0015-training-warmup-plan.md](0015-training-warmup-plan.md), [docs/plans/0019-reg-model-split-plan.md](0019-reg-model-split-plan.md) |

## 1. 목적과 배경

`docs/plans/0015-training-warmup-plan.md`(상태 Draft, 미구현)는 backbone freeze/unfreeze 기반
`warmup_epochs` 정책을 제안했다. 하나의 optimizer가 처음부터 backbone/non-backbone 두 param group을
갖고, freeze 구간에는 backbone group의 `requires_grad`만 `False`로 두는 방식이었다. `docs/plans/0019-reg-model-split-plan.md`(Done)에서 reg가 `CustomRegModel`/`TorchRegModel`로 분리되면서 0015의
판정 기준도 `isinstance(self.model, CustomRegModel)`로 갱신됐지만, freeze/unfreeze 메커니즘 자체는
실제로 구현된 적이 없다.

이번 plan은 0015를 재검토해 2단계 학습으로 다시 설계한다. 핵심 재설계는 다음과 같다.

- `CustomRegModel`은 warmup 없이 단일 optimizer, 단일 stage로 학습한다.
- backbone과 head가 분리된 model(`TorchRegModel` 포함, Category B pretrained composable)은 2단계로
  학습한다. 1단계는 backbone을 freeze하고 non-backbone parameter만 학습하고, 2단계는 전체를
  unfreeze해서 differential learning rate로 학습한다.
- 이 staging 메커니즘은 model(`nn.Module`)이 아니라 wrapper와 `Trainer`에 구현한다.

phase 전환 시 optimizer는 mutate가 아니라 recreate한다. 1단계는 non-backbone parameter만 담은 단일
learning rate optimizer, 2단계는 backbone/non-backbone differential learning rate optimizer를 각각 새로
생성한다. Adam류 optimizer의 momentum이 이전 phase의 gradient(freeze 해제 직후 아직 맞지 않는 backbone
gradient)에 오염되지 않고, param group 구성 자체가 phase마다 다르다는 요구사항과도 자연스럽게 맞는다.
전환 시점은 0015와 동일하게 고정 epoch 수(`warmup_epochs`, epoch 1 freeze, epoch 2부터 unfreeze)로
지정한다.

적용 범위는 이번 plan에서 reg로 좁힌다. seg는 아직 model.py가 custom/pretrained class로 분리되지 않았고,
det는 `TorchDetModel` 외에 `YoloDetModel`/`DetrDetModel` 같은 외부 whole-model이 섞여 있어 판정 조건이
reg보다 복잡하다. reg에서 `Trainer`/`BaseWrapper` 공통 hook을 먼저 검증한 뒤 후속 plan에서 seg/det로
확장한다.

0015는 이 plan으로 대체된다. 0015 파일은 지우지 않고 상태와 backward note만 갱신한다.

## 2. 범위

이번 plan에 포함하는 항목은 다음과 같다.

- `docs/plans/0015-training-warmup-plan.md`의 상태를 `Draft`에서 `Superseded`로 갱신하고 문서
  최상단에 이 plan이 0020으로 대체됐다는 backward note를 추가한다. 본문(1~5절)은 historical record로
  보존하고 다시 쓰지 않는다.
- `docs/architecture/model-assembly.md`의 training policy 서술(0015가 추가하려 했던 자리)에 2단계
  학습 정책을 반영한다. `CustomRegModel`은 단일 stage/단일 optimizer, pretrained composable
  (`TorchRegModel` 포함)은 freeze-then-unfreeze 2단계 정책이라는 것을 canonical하게 명시한다.
- `src/models/base/base_wrapper.py`에 2단계 학습에 필요한 공통 hook을 추가한다.
  - `__init__`에 `warmup_epochs=0` kwarg와 `self.applied_warmup_epochs` 필드를 추가한다.
  - `get_backbone_module()`, `set_backbone_trainable(trainable)` helper를 추가한다.
  - `build_optimizer(phase)`, `build_scheduler(optimizer)`를 `NotImplementedError` 기본 정의로 두고
    subclass가 override한다.
  - `on_fit_start(max_epochs)`는 `applied_warmup_epochs > 0`이면 backbone을 freeze하고 phase 1
    optimizer/scheduler를 생성한다.
  - `on_epoch_start(epoch)`를 새로 추가한다. `epoch == applied_warmup_epochs + 1`이면 backbone을
    unfreeze하고 phase 2 optimizer/scheduler를 재생성한다.
- `src/core/trainer.py`의 `fit()`과 `fit_early_stop()` epoch loop에서 매 epoch train 직전에
  `self.wrapper.on_epoch_start(epoch)`를 호출한다(`on_fit_start`/`on_epoch_end`는 이미 호출되고 있음).
- `src/models/reg/wrapper.py`의 `RegWrapper`가 `isinstance(self.model, CustomRegModel)` 여부로 2단계
  학습 적용 대상인지 판정한다. `CustomRegModel`이면 `warmup_epochs`를 무시(0으로 정규화)하고 기존과
  동일하게 단일 optimizer(`AdamW(self.model.parameters(), lr=1e-4)`)를 사용한다. `TorchRegModel`이면
  `warmup_epochs` kwarg(기본값 1)를 받아 `build_optimizer(phase)`를 구현한다.
  - phase 1: `AdamW(non_backbone_params, lr=1e-4)`, backbone 전체 `requires_grad=False`.
  - phase 2: `AdamW([{"params": extractor.parameters(), "lr": 1e-5}, {"params": non_backbone_params, "lr": 1e-4}])`,
    backbone 전체 `requires_grad=True`(기존 `RegWrapper`의 optimizer 구성과 동일).
  - `build_scheduler(optimizer)`는 기존 `ReduceLROnPlateau(..., mode="max", factor=0.5, patience=2,
    threshold=1e-4, threshold_mode="abs", min_lr=1e-7)` 구성을 재사용한다.
- `scripts/config.py`의 `DEFAULTS`에 `warmup_epochs=1`을 추가하고 `--warmup_epochs` CLI 인자,
  `get_wrapper_kwargs()` 전달을 추가한다.

이번 plan에서 제외하는 항목은 다음과 같다(후속 plan에서 다룸):

- seg/det로의 2단계 학습 확장. seg의 custom/pretrained model class 분리, det의 external whole-model
  판정 로직 정리가 선행되어야 한다.
- `experiments/configs.py`의 기존 `REG_CONFIGS` 일괄 재작성. 기존 config는 `warmup_epochs` 미지정 시
  기본값 `1`로 동작하므로 수정 없이 계속 유효하다.
- learning rate scheduler warmup(cosine, one-cycle, linear ramp), layer-wise learning rate decay,
  gradual unfreezing 같은 다른 warmup 변형.
- `warmup_epochs` 값을 자동 탐색하는 ablation 실행과 결과 분석.

## 3. 구현 계획

### 3.1. `src/models/base/base_wrapper.py`

`__init__`에 `warmup_epochs=0`을 추가하고 `self.warmup_epochs`, `self.applied_warmup_epochs = 0`을
초기화한다. `applied_warmup_epochs`는 실제로 적용되는 값으로, subclass가 model 종류에 따라 생성자에서
결정해 재대입한다(`CustomRegModel`이면 0, 그 외면 `warmup_epochs` 그대로).

```python
def get_backbone_module(self):
    return getattr(self.model, "extractor", None)

def set_backbone_trainable(self, trainable):
    backbone = self.get_backbone_module()
    if backbone is None:
        return
    for p in backbone.parameters():
        p.requires_grad = trainable

def build_optimizer(self, phase):
    raise NotImplementedError

def build_scheduler(self, optimizer):
    raise NotImplementedError

def on_fit_start(self, max_epochs):
    if self.applied_warmup_epochs <= 0:
        return
    self.set_backbone_trainable(False)
    self.set_optimizer(self.build_optimizer(phase=1))
    self.set_scheduler(self.build_scheduler(self.optimizer))

def on_epoch_start(self, epoch):
    if self.applied_warmup_epochs <= 0:
        return
    if epoch == self.applied_warmup_epochs + 1:
        self.set_backbone_trainable(True)
        self.set_optimizer(self.build_optimizer(phase=2))
        self.set_scheduler(self.build_scheduler(self.optimizer))
```

### 3.2. `src/core/trainer.py`

`fit()`과 `fit_early_stop()`의 epoch loop 맨 앞에 `self.wrapper.on_epoch_start(epoch)` 호출을
추가한다.

```text
on_fit_start(max_epochs)
for epoch in 1..max_epochs:
    on_epoch_start(epoch)
    train(...)
    evaluate(...)
    on_epoch_end(valid_score)
```

### 3.3. `src/models/reg/wrapper.py`

`RegWrapper.__init__`은 model 생성 직후 `isinstance(model, CustomRegModel)` 여부로
`applied_warmup_epochs`를 결정하고, `BaseWrapper.__init__`에 넘긴다. `optimizer`/`scheduler`가
명시적으로 주어지지 않으면 `applied_warmup_epochs > 0` 여부에 따라 phase 1 또는 phase 2 optimizer를
생성자 시점에 만든다(2단계 대상이면 phase 1로 시작, 아니면 바로 phase 2 상당의 단일 optimizer).

```python
def build_optimizer(self, phase):
    if isinstance(self.model, CustomRegModel):
        return AdamW(self.model.parameters(), lr=1e-4)
    non_backbone_params = self._non_backbone_params()
    if phase == 1:
        return AdamW(non_backbone_params, lr=1e-4)
    return AdamW([
        {"params": self.model.extractor.parameters(), "lr": 1e-5},
        {"params": non_backbone_params, "lr": 1e-4},
    ])

def build_scheduler(self, optimizer):
    return ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2,
                             threshold=1e-4, threshold_mode="abs", min_lr=1e-7)

def _non_backbone_params(self):
    backbone_ids = {id(p) for p in self.model.extractor.parameters()}
    return [p for p in self.model.parameters() if id(p) not in backbone_ids]
```

### 3.4. `scripts/config.py`

`DEFAULTS`에 `warmup_epochs=1`을 추가하고, `parse_args()`에 `--warmup_epochs` integer argument를
추가한다. `get_wrapper_kwargs()`는 값이 설정되어 있으면 `warmup_epochs`를 wrapper kwargs에 포함한다.

### 3.5. 문서 갱신

`docs/architecture/model-assembly.md`의 training policy 서술 자리에 `CustomRegModel`(단일 stage)과
pretrained composable(`TorchRegModel` 포함, 2단계) 두 행을 명시한다. `docs/plans/0015-training-warmup-plan.md` 최상단에는 "이 plan은 [docs/plans/0020-reg-two-stage-training-plan.md](0020-reg-two-stage-training-plan.md)로 대체됐다"는 한 문장짜리 backward note를 추가하고 헤더 표의 상태를
`Superseded`로 바꾼다.

## 4. 완료 기준

- `docs/plans/0015-training-warmup-plan.md` 상태가 `Superseded`이고 0020을 가리키는 backward note가
  있다.
- `docs/architecture/model-assembly.md`에 `CustomRegModel` 단일 stage, pretrained composable 2단계
  학습 정책이 반영되어 있다.
- `RegWrapper(backbone='custom')`은 `warmup_epochs`를 몇으로 지정하든 `applied_warmup_epochs == 0`이고
  optimizer가 단일 param group이다.
- `RegWrapper(backbone='resnet18', warmup_epochs=1)`은 `on_fit_start(max_epochs)` 호출 후 backbone
  `requires_grad == False`이고 optimizer가 non-backbone만 담은 단일 learning rate group이다.
  `on_epoch_start(2)` 호출 후 backbone `requires_grad == True`이고 optimizer가 새로 생성된 differential
  learning rate 2-group이다(이전 optimizer 객체와 다른 새 인스턴스).
- `Trainer.fit()`과 `fit_early_stop()` 모두 매 epoch 시작 시 `on_epoch_start(epoch)`를 호출한다.
- `warmup_epochs=0`으로 지정하면 pretrained composable model도 기존과 동일하게 단일 phase,
  생성자 시점의 differential learning rate optimizer 그대로 학습한다(회귀 없음).
- `docs/plans/0020-reg-two-stage-training-plan.md`가 승인 후 `Done`으로 갱신된다.

## 5. 검증

구현 후 검증은 conda 환경 `pytorch_env`에서 수행한다.

```bash
conda activate pytorch_env
python -c "
from src.models.reg.wrapper import RegWrapper

w = RegWrapper(backbone='custom', warmup_epochs=1, device='cpu')
print('custom applied_warmup_epochs:', w.applied_warmup_epochs, 'groups:', len(w.optimizer.param_groups))

w2 = RegWrapper(backbone='resnet18', warmup_epochs=1, device='cpu')
w2.on_fit_start(2)
opt_phase1 = w2.optimizer
print('phase1 requires_grad:', next(w2.model.extractor.parameters()).requires_grad, 'groups:', len(opt_phase1.param_groups))
w2.on_epoch_start(2)
print('phase2 requires_grad:', next(w2.model.extractor.parameters()).requires_grad, 'groups:', len(w2.optimizer.param_groups))
print('optimizer recreated:', w2.optimizer is not opt_phase1)
"
python scripts/train.py --method reg --backbone custom --head coord_gap --device cpu --max_epochs 1 --train_size 2 --valid_size 2 --num_workers 0 --output_dir /tmp/reg_custom_2stage_smoke
python scripts/train.py --method reg --backbone resnet18 --head coord_gap --device cpu --max_epochs 2 --train_size 2 --valid_size 2 --num_workers 0 --warmup_epochs 1 --output_dir /tmp/reg_resnet18_2stage_smoke
```

두 smoke train 모두 정상 종료하는지, `resnet18` 실행이 epoch 1/2 경계에서 optimizer가 재생성되는지
로그의 `lr=` 표기로 확인한다.
