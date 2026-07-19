# RegModel을 CustomRegModel/TorchRegModel로 분리

다음 표는 이 plan의 상태와 적용 범위를 정리한다.

| 항목 | 값 |
|---|---|
| 상태 | Done |
| 작성일 | 2026-07-19 |
| 적용 범위 | `src/models/reg/model.py`, `src/models/reg/wrapper.py`, `docs/architecture/model-assembly.md`, `docs/plans/0006-reg-model-rename-plan.md`, `docs/plans/0015-training-warmup-plan.md` |
| 관련 문서 | [docs/architecture/model-assembly.md](../architecture/model-assembly.md), [docs/plans/0006-reg-model-rename-plan.md](0006-reg-model-rename-plan.md), [docs/plans/0015-training-warmup-plan.md](0015-training-warmup-plan.md) |

## 1. 목적과 배경

`RegModel`은 2026-07-18 [docs/plans/0006-reg-model-rename-plan.md](0006-reg-model-rename-plan.md)에서
`CustomRegModel`이라는 이름을 `RegModel`로 바꾸며 통합됐다. 근거는 canonical 문서
(`docs/architecture/model-assembly.md` 2.1절, 4절, 6절)가 Category A(custom backbone)와
Category B(pretrained backbone)는 같은 `FeatureBundle` 계약을 공유하므로 하나의 class로 조립한다고
정의했기 때문이다. `backbone="custom"`과 `backbone="resnet50"`은 통합 이후 같은 `__init__` 분기,
같은 adapter/head 조립 로직을 공유하는 하나의 class였다.

이번 plan은 이 통합을 다시 나눈다. `RegModel`을 `CustomRegModel`(custom backbone 전용)과
`TorchRegModel`(torchvision/timm backbone 전용)으로 분리해 0006 이전 상태로 되돌리고, backbone
선택 축을 class 경계로 승격한다.

이 결정은 [docs/plans/0015-training-warmup-plan.md](0015-training-warmup-plan.md)(상태 Draft, 아직
미구현)의 warmup 정책과 직접 연결된다. 0015는 `warmup_epochs` 정책을 `backbone == "custom"` 문자열
비교로 결정하도록 설계돼 있었다. 이번 분리의 목적 자체가 이 정책 경계를 class 경계로 승격하는 것이다.
`CustomRegModel`은 warmup 없이 단일 learning rate로 학습하고, `TorchRegModel`을 포함해 다른
backbone이나 외부 모델을 쓰는 model은 warmup 정책 적용 대상이다. 즉 `isinstance(self.model,
CustomRegModel)` 여부가 0015가 실제로 구현될 때 쓸 판정 기준이 된다. 0015가 아직 Draft이므로 이번
plan에서 0015 본문(3.1절 표, 3.4절 서술)을 이 기준으로 갱신해 구현 순서 충돌을 막는다.

`det`(`DetModel`/`TorchDetModel`/`YoloDetModel`/`DetrDetModel`)와 `seg`(`SegModel`/`TorchSegModel`,
`SegWrapper.build_model` 분기)에 이미 custom vs 외부 backbone/whole-model 분리 패턴이 있으므로,
이번 분리는 seg의 패턴(같은 조립 축, 분리는 `model.py`의 class 두 개 + `wrapper.py`의 `build_model`
분기)을 따른다. det처럼 완전히 다른 학습 루프는 필요 없다 - `CustomRegModel`과 `TorchRegModel`은
여전히 같은 `FeatureExtractor` -> head 흐름과 같은 `RegPreprocessor`/`RegPostprocessor`/`WingLoss`/
`PolygonIoU`를 공유하는 `BaseWrapper` 그대로 쓴다.

## 2. 범위

이번 plan에 포함하는 항목은 다음과 같다.

- `src/models/reg/model.py`: `RegModel` 하나를 `CustomRegModel`(backbone 고정 `custom`,
  `CustomBackbone`만 사용)과 `TorchRegModel`(backbone은 `SUPPORTED_BACKBONES` 또는
  `SUPPORTED_TIMM_BACKBONES` 중 하나, `TorchBackbone`/`TimmBackbone` 사용)으로 분리한다. 두 class
  모두 `head="coord_gap"/"coord_spatial"` 분기, adapter 선택(`is_vit` 여부에 따른
  `CNNBackboneAdapter`/`TransformerBackboneAdapter`), `FeatureExtractor` 조립은 동일하게 유지하되
  공통 helper(`_build_extractor_and_head`)로 중복을 제거한다. `forward()`는 mixin 없이 각 class가
  독립적으로 정의한다(det/seg 기존 model.py들의 방식과 일치).
- `src/models/reg/wrapper.py`: 모듈 수준 `build_model(in_channels, dropout, backbone, head)` 함수를
  추가해 `backbone`이 `None` 또는 `"custom"`이면 `CustomRegModel`, 그 외에는 `TorchRegModel`을
  생성한다. 미지원 backbone 문자열은 `TorchRegModel.__init__`에서 여전히 `ValueError`. `RegWrapper`
  class 자체는 이름과 외부 인터페이스(`get_wrapper`에서 부르는 방식)를 바꾸지 않는다.
- `docs/architecture/model-assembly.md`: 2.1절의 "Category A와 B는 같은 FeatureBundle 계약을
  사용한다"는 서술은 유지하고(계약은 실제로 공유됨), 4.1절 조립 표의 `RegModel` 행을 `CustomRegModel`
  행으로 되돌리고, 4.2절 제목과 본문을 `CustomRegModel` 기준으로 되돌린다. 6.1절에 `TorchRegModel`을
  Category B의 reg 조합으로 명시적으로 추가한다.
- `docs/plans/0006-reg-model-rename-plan.md`: 완료된 plan 본문은 historical record로 보존하고, 이번
  0019 plan을 가리키는 backward note를 추가한다.
- `docs/plans/0015-training-warmup-plan.md`(Draft): 3.1절 정책 표의 custom composable 행 조건을
  `isinstance(self.model, CustomRegModel)`(reg 기준, seg/det는 대응하는 custom model class가 생기면
  동일한 방식으로 옮겨간다는 설명 포함)으로 갱신한다. 3.4절 서술도 `RegWrapper`가
  `isinstance(self.model, CustomRegModel)`로 optimizer policy를 나눈다고 다시 쓴다. 정책 표의
  3분류(custom composable, pretrained composable, external whole model) 체계 자체는 바꾸지 않는다.

이번 plan에서 제외하는 항목은 다음과 같다(후속 plan에서 다룸):

- `experiments/configs.py`의 기존 `REG_CONFIGS` 항목 재작성. `backbone=` 문자열 kwarg는 그대로
  동작하므로 기존 config는 수정 없이 계속 유효하다.
- `scripts/config.py`의 `model=` kwarg 도입. `RegWrapper`는 여전히 `backbone=` 문자열만으로 분기하고
  det/seg처럼 별도 `model=` CLI 인자를 받지 않는다.
- 0015 warmup 기능 자체의 구현(freeze/unfreeze, `Trainer` hook). 이번 plan은 0015 문서의 서술만 새
  class 이름에 맞춰 갱신하고 실제 warmup 코드는 손대지 않는다.
- `RegPreprocessor`/`RegPostprocessor`/`WingLoss`/`PolygonIoU`/`RegWrapper`의 이름 변경. 이번 plan은
  `model.py` 안의 model class 분리와 그 파급 문서만 다룬다.

## 3. 완료 기준

- `src/models/reg/model.py`에 `RegModel` class가 없고 `CustomRegModel`, `TorchRegModel`이 있다.
- `from src.models.reg.model import CustomRegModel, TorchRegModel`과
  `from src.models.reg.wrapper import RegWrapper`가 정상 import된다.
- `RegWrapper(backbone="custom")`은 `CustomRegModel` 인스턴스를, `RegWrapper(backbone="resnet18")`은
  `TorchRegModel` 인스턴스를 생성한다.
- 기존 `head="coord_gap"`/`"coord_spatial"`, `dropout` 동작은 두 class 모두 이전과 동일한 output shape
  `(B, 8)`을 낸다.
- `docs/architecture/model-assembly.md`에 `RegModel` 문자열이 남아 있지 않고, `CustomRegModel`,
  `TorchRegModel`이 4.1/4.2/6절에 반영되어 있다.
- `docs/plans/0006-reg-model-rename-plan.md`에 0019를 가리키는 backward note가 있다.
- `docs/plans/0015-training-warmup-plan.md`(Draft)의 3.1절 정책 표와 3.4절 서술이
  `isinstance(self.model, CustomRegModel)` 기준으로 갱신되어 있다.
- `docs/plans/0019-reg-model-split-plan.md`가 작성되고 승인 후 `Done`으로 갱신된다.

## 4. 검증

구현 후 검증은 conda 환경 `pytorch_env`에서 수행한다.

```bash
conda activate pytorch_env
python -c "
from src.models.reg.model import CustomRegModel, TorchRegModel
from src.models.reg.wrapper import RegWrapper
import torch

m1 = RegWrapper(backbone='custom')
m2 = RegWrapper(backbone='resnet18')
assert type(m1.model).__name__ == 'CustomRegModel'
assert type(m2.model).__name__ == 'TorchRegModel'
x = torch.randn(2, 3, 224, 224)
print(m1.model(x).shape, m2.model(x).shape)
"
grep -rn "RegModel\b" docs/architecture/model-assembly.md src/models/reg/model.py src/models/reg/wrapper.py | grep -v "CustomRegModel\|TorchRegModel"
```

두 번째 명령은 결과가 없어야 한다(순수 `RegModel` 잔존 문자열 없음 확인). 기존 `experiments/run.py`로
`backbone=custom`과 `backbone=resnet18` 두 configs를 짧은 epoch로 smoke-training해 학습 루프가 계속
동작하는지 확인한다.
