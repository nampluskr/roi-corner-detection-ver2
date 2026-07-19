---
title: seg/det 2단계(freeze-then-unfreeze) 학습 확장
---

| 항목 | 내용 |
|---|---|
| 상태 | Done |
| 작성일 | 2026-07-19 |
| 적용 범위 | `docs/architecture/model-assembly.md`, `src/models/seg/wrapper.py`, `src/models/det/wrapper.py`, `scripts/config.py`, `experiments/configs.py` |
| 관련 문서 | [0020-reg-two-stage-training-plan.md](0020-reg-two-stage-training-plan.md) |

## 목적과 배경

0020(Done)에서 `reg`에 2단계(freeze-then-unfreeze) 학습 정책을 구현했다. `BaseWrapper`에 공통 hook
(`get_backbone_module`, `set_backbone_trainable`, `build_optimizer(phase)`/`build_scheduler(optimizer)`
추상 메서드, `on_fit_start`/`on_epoch_start`)을 추가하고 `Trainer.fit()`/`fit_early_stop()`이 매 epoch
`on_epoch_start(epoch)`를 호출하도록 배선했다. 이번 plan은 이 정책을 `seg`/`det`로 확장한다.

`seg`는 `reg`와 구조가 거의 동일하다. `SegModel`(Category A, custom backbone, `self.extractor` 보유)과
`TorchSegModel`(Category B, torchvision whole-model, `self.extractor` 없음)로 이미 나뉘어 있고,
`SegWrapper.set_default_optimizer()`가 `hasattr(self.model, "extractor")`로 분기해 custom은 단일
optimizer, `TorchSegModel`은 처음부터 differential-lr 2-group optimizer를 구성한다. reg와 동일한
`isinstance`/`hasattr` 판정과 `build_optimizer(phase)` 오버라이드를 그대로 적용할 수 있다.

`det`는 이질적이다. `DetModel`(custom backbone, `DetWrapper`)은 `self.extractor`를 가지며 처음부터
backbone/head differential-lr 2-group optimizer를 쓴다. reg의 `CustomRegModel`과 반대로 이미 2-group이지만,
freeze 대상이 되는 pretrained backbone이 아니라 처음부터 학습하는 custom backbone이므로 2단계 학습 정책
적용 대상이 아니다(reg의 `CustomRegModel`과 동일하게 단일 stage 취급, warmup 무시).

pretrained whole-model 3종(`TorchDetModel`/`TorchDetWrapper`, `YoloDetModel`/`YoloDetWrapper`,
`DetrDetModel`/`DetrDetWrapper`)에 우선 적용한다. 이들은 `self.extractor` 같은 분리된 backbone 속성이
없고 각 wrapper가 이미 서로 다른 `train_step`/`eval_step`을 오버라이드하는 커스텀 학습 루프를 가진다.
조사 결과 backbone 식별 방식이 모델마다 다르다.

`TorchDetModel.net`은 torchvision.models.detection 표준 구조라 `net.backbone`(`BackboneWithFPN` 인스턴스)이
명확한 서브모듈이다. `YoloDetModel.net`(Ultralytics `DetectionModel`)은 backbone/neck/head가 분리된
서브모듈이 아니라 `net.model`이라는 23개 레이어의 단일 `nn.ModuleList`이고, 마지막 레이어(`net.model[-1]`,
`Detect`)만 head다. `net.model[:-1]`을 backbone으로 취급한다. `DetrDetWrapper`는 이미 이름 매칭
(`name.startswith("net.model.backbone")`)으로 backbone/classifier/other 3-group을 구성하는
`build_optimizer()`를 갖고 있어 이 로직을 재사용한다.

이 차이 때문에 `BaseWrapper.get_backbone_module()`(단일 `self.model.extractor` 속성 가정)은 det pretrained
3종에 그대로 쓸 수 없다. 각 det wrapper가 자신의 구조에 맞게 `get_backbone_module()`(또는
`set_backbone_trainable()` 자체)을 오버라이드한다.

## 범위

포함하는 항목은 다음과 같다.

- `docs/architecture/model-assembly.md`의 6.4절(2단계 학습 정책)을 seg/det까지 포함하도록 갱신한다.
  판정 기준을 Category B pretrained composable/whole-model 전반으로 일반화하고, det의 backbone 식별
  방식이 모델마다 다르다는 점(`net.backbone` 서브모듈, `net.model[:-1]` 레이어 슬라이스, 이름 매칭)을
  명시한다. `DetModel`(custom)과 `TorchSegModel`은 구조적으로 2단계 대상이 아님을 명시한다.
- `src/models/seg/wrapper.py`: `SegWrapper`에 reg와 동일한 패턴을 적용한다. `isinstance(net, SegModel)
  and backbone == "custom"`이면 `applied_warmup_epochs = 0`, 아니면 `warmup_epochs`를 그대로 적용한다.
  `build_optimizer(phase)`/`build_scheduler(optimizer)` 오버라이드와 `warmup_epochs` 생성자 kwarg(기본값
  1)를 추가한다.
- `src/models/det/wrapper.py`: `DetWrapper`(custom `DetModel`)는 `warmup_epochs=None` 무시용 kwarg만
  추가하고 기존 동작은 그대로 유지한다. `TorchDetWrapper`, `YoloDetWrapper`, `DetrDetWrapper` 각각에
  `get_backbone_module()`(또는 `set_backbone_trainable()` 오버라이드)과 `build_optimizer(phase)`/
  `build_scheduler(optimizer)`를 추가하고 `warmup_epochs` 생성자 kwarg(기본값 1)를 받는다.
- `scripts/config.py`: `get_wrapper_kwargs()`의 `warmup_epochs` scoping을 `method == "reg"`에서
  `method in ("reg", "seg", "det")`로 확장한다.
- `experiments/configs.py`의 `SEG_CONFIGS`/`DET_CONFIGS` pretrained 항목에 예시로
  `"warmup_epochs": 1`을 추가한다.

제외하고 후속 plan에서 다루는 항목은 다음과 같다.

- `DetModel`(custom det)에 대한 2단계 학습. 처음부터 backbone이 무작위 초기화이므로 freeze 대상이 아니다.
- `TorchSegModel`에 대한 2단계 학습. torchvision whole-model이라 backbone/head가 분리된 서브모듈로
  노출되지 않는다. `TorchSegModel.net`은 FCN/DeepLabV3/LRASPP 등 모델마다 backbone 노출 방식이 달라
  향후 필요 시 별도 조사와 plan으로 다룬다.
- learning rate scheduler warmup, layer-wise LR decay 등 다른 warmup 변형.
- ultralytics 버전 업그레이드 시 `net.model` 구조가 바뀔 가능성에 대한 방어적 코드.

## 완료 기준

다음 항목이 모두 충족되면 이 plan을 Done으로 본다.

- `SegWrapper(backbone="custom")`는 `warmup_epochs`를 몇으로 지정하든 `applied_warmup_epochs == 0`이고
  단일 optimizer(1 group)다.
- `SegWrapper(backbone="resnet18", warmup_epochs=1)`은 reg의 resnet18 검증과 동일한 phase 전환 동작을
  보인다.
- `SegWrapper(model="fcn_resnet50")`(`TorchSegModel`)는 기존과 동일하게 단일 optimizer로 동작하고
  회귀가 없다.
- `DetWrapper`는 기존 생성자 시그니처(`warmup_epochs=None` 추가 제외)와 동작을 그대로 유지한다.
- `TorchDetWrapper(warmup_epochs=1)`은 `net.backbone` freeze/unfreeze와 optimizer 재생성이 정상 동작한다.
- `YoloDetWrapper(warmup_epochs=1)`은 `net.model[:-1]` 레이어들의 `requires_grad`가 일괄 토글되고
  optimizer가 재생성된다.
- `DetrDetWrapper(warmup_epochs=1)`은 이름 매칭 기반 backbone freeze/unfreeze와 3-group optimizer
  재생성이 정상 동작한다.
- `scripts/train.py --method seg`/`--method det`가 `--warmup_epochs`를 받아도 `TypeError` 없이 동작한다.
- `docs/architecture/model-assembly.md` 6.4절이 seg/det 확장 내용을 반영한다.
- `experiments/configs.py`의 `SEG_CONFIGS`/`DET_CONFIGS` pretrained 항목 하나 이상에 예시로
  `"warmup_epochs": 1`이 추가되어 있다.

## 검증

다음 명령으로 직접 검증한다.

```bash
conda activate pytorch_env
python -c "
from src.models.seg.wrapper import SegWrapper

w = SegWrapper(backbone='custom', warmup_epochs=1, device='cpu')
print('seg custom applied_warmup_epochs:', w.applied_warmup_epochs, 'groups:', len(w.optimizer.param_groups))

w2 = SegWrapper(backbone='resnet18', warmup_epochs=1, device='cpu')
w2.on_fit_start(2)
opt1 = w2.optimizer
print('seg resnet18 phase1:', next(w2.model.extractor.parameters()).requires_grad, len(opt1.param_groups))
w2.on_epoch_start(2)
print('seg resnet18 phase2:', next(w2.model.extractor.parameters()).requires_grad, len(w2.optimizer.param_groups), w2.optimizer is not opt1)
"
python -c "
from src.models.det.wrapper import TorchDetWrapper, YoloDetWrapper, DetrDetWrapper

for cls, kwargs in [(TorchDetWrapper, dict(model='fasterrcnn_resnet50_fpn')),
                     (YoloDetWrapper, dict(model='yolov8n')),
                     (DetrDetWrapper, dict(model='detr_resnet50'))]:
    w = cls(warmup_epochs=1, device='cpu', **kwargs)
    w.on_fit_start(2)
    opt1 = w.optimizer
    w.on_epoch_start(2)
    print(cls.__name__, 'phase transition ok:', w.optimizer is not opt1)
"
python scripts/train.py --method seg --backbone custom --device cpu --max_epochs 1 --train_size 2 --valid_size 2 --batch_size 2 --num_workers 0 --output_dir /tmp/seg_custom_2stage_smoke
python scripts/train.py --method seg --backbone resnet18 --device cpu --max_epochs 2 --train_size 2 --valid_size 2 --batch_size 2 --num_workers 0 --warmup_epochs 1 --output_dir /tmp/seg_resnet18_2stage_smoke
python scripts/train.py --method det --backbone custom --device cpu --max_epochs 1 --train_size 2 --valid_size 2 --batch_size 2 --num_workers 0 --output_dir /tmp/det_custom_smoke
python scripts/train.py --method det --model fasterrcnn_resnet50_fpn --device cpu --max_epochs 2 --train_size 2 --valid_size 2 --batch_size 2 --num_workers 0 --warmup_epochs 1 --output_dir /tmp/det_torchdet_2stage_smoke
```

각 smoke train이 정상 종료하는지, 직접 Python 검증에서 phase 전환(`requires_grad` 토글, optimizer
재생성) 결과가 기대대로 출력되는지 확인한다. 로컬 가중치 파일이 있으면 `YoloDetWrapper`/`DetrDetWrapper`도
동일한 방식으로 smoke train을 추가 검증한다.
