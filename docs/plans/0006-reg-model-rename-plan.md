# CustomRegModel에서 RegModel로 이름 변경

다음 표는 이 plan의 상태와 적용 범위를 정리한다.

| 항목 | 값 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-18 |
| 적용 범위 | `docs/architecture/model-assembly.md`, `src/models/reg/model.py`, `src/models/reg/wrapper.py`, `docs/plans/0003-reg-backbone-experiments-plan.md` |
| 관련 문서 | [docs/architecture/model-assembly.md](../architecture/model-assembly.md), [docs/plans/0003-reg-backbone-experiments-plan.md](0003-reg-backbone-experiments-plan.md), [docs/plans/0005-torch-backbone-expansion-plan.md](0005-torch-backbone-expansion-plan.md) |

## 1. 목적과 배경

`0003-reg-backbone-experiments-plan.md`는 backbone 선택 기능을 처음 추가할 때 "기존 import
안정성을 위해" public class 이름 `CustomRegModel`을 유지하기로 결정했다. 당시 지원 범위는 `custom`과
`resnet50` 두 가지였다.

`0005-torch-backbone-expansion-plan.md` 이후 지원 backbone은 `custom`, `resnet18`, `resnet34`,
`resnet50`, `efficientnet_b0`, `vgg16`, `vgg19`로 늘어났다. canonical 문서는 Category A(custom
backbone)와 Category B(pretrained backbone)가 같은 `FeatureBundle` 계약과 조립 구조를 공유한다고
정의하는데, 현재 구현은 이 둘을 이미 하나의 class로 통합해 다루고 있다. `CustomRegModel`이라는
이름은 이제 실제 지원 범위보다 좁게 표현되므로 `RegModel`로 바꾼다.

## 2. 범위

이번 plan에 포함하는 항목은 다음과 같다.

- `src/models/reg/model.py`의 public class 이름을 `CustomRegModel`에서 `RegModel`로 변경한다.
- `src/models/reg/wrapper.py`의 import, docstring, 내부 사용처를 `RegModel` 기준으로 갱신한다.
- `docs/architecture/model-assembly.md` 4.2절 제목, 조립 표, 본문에서 `CustomRegModel`을
  `RegModel`로 갱신한다.
- `docs/plans/0003-reg-backbone-experiments-plan.md`의 이름 유지 결정 문단에 이 plan을 가리키는
  backward note를 추가한다.

이번 plan에서 제외하는 항목은 다음과 같다.

- `src/models/reg/model.py` 파일명 변경. class 이름만 바꾸고 module 경로는 유지한다.
- `docs/plans/0002`, `0004`, `0005` 문서의 본문 재작성. 완료된 plan은 작성 시점의 결정을 그대로
  보존하는 historical record이므로 수정하지 않는다.
- backbone 목록, head 목록, 조립 구조, 학습 동작의 변경. 이번 plan은 이름 변경만 다룬다.

## 3. 구현 계획

### 3.1. `src/models/reg/model.py`

`class CustomRegModel(BaseModel):`를 `class RegModel(BaseModel):`로 변경하고 class docstring의
표현도 backbone 선택지를 포함하도록 유지한다. `CustomBackbone`, `TorchBackbone` 등 다른 식별자는
바꾸지 않는다.

### 3.2. `src/models/reg/wrapper.py`

파일 첫 줄 header, class docstring, `from src.models.reg.model import RegModel` import문과
`RegModel(...)` 생성 호출을 갱신한다. `RegWrapper` class 이름은 이번 범위에 포함하지 않는다.

### 3.3. `docs/architecture/model-assembly.md`

4.2절 제목 "CustomRegModel 조합"을 "RegModel 조합"으로 바꾸고, 4.1절 조립 표의
`CustomRegModel` 행과 4.2절 본문의 `CustomRegModel` 표기를 `RegModel`로 바꾼다. 조합 내용
자체(backbone, adapter, head 구성)는 변경하지 않는다.

### 3.4. `docs/plans/0003-reg-backbone-experiments-plan.md`

"public class 이름은 기존 import 안정성을 위해 `CustomRegModel`을 유지하되" 문단 바로 뒤에, 이
결정이 `0006-reg-model-rename-plan.md`에서 대체되었다는 한 문장짜리 backward note를 추가한다.
문단 자체의 원문은 지우지 않는다.

## 4. 완료 기준

이 plan은 다음 조건을 만족하면 `Done`으로 본다.

- `docs/plans/0006-reg-model-rename-plan.md` 상태가 `Approved`에서 `Done`으로 갱신되어 있다.
- `src/models/reg/model.py`와 `src/models/reg/wrapper.py`에 `CustomRegModel` 문자열이 없다.
- `docs/architecture/model-assembly.md`에 `CustomRegModel` 문자열이 없다.
- `docs/plans/0003-reg-backbone-experiments-plan.md`에 이 plan을 가리키는 backward note가 있다.
- `from src.models.reg.model import RegModel`과 `from src.models.reg.wrapper import RegWrapper`가
  정상 import된다.

## 5. 검증

문서 생성과 식별자 rename만 포함하므로 별도 test suite 실행은 없다. 다음 명령으로 import와 잔존
문자열을 확인한다.

```bash
conda activate pytorch_env
python -c "from src.models.reg.model import RegModel; from src.models.reg.wrapper import RegWrapper; print(RegModel, RegWrapper)"
grep -rn "CustomRegModel" docs/architecture/model-assembly.md src/models/reg/model.py src/models/reg/wrapper.py
```

두 번째 명령은 결과가 없어야 한다.
