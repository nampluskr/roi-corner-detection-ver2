# det head 파라미터로 representation 통합

다음 표는 이 plan의 상태와 적용 범위를 정리한다.

| 항목 | 값 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-19 |
| 적용 범위 | `docs/architecture/model-assembly.md`, `experiments/configs.py`, `experiments/run.py`, `scripts/config.py`, `src/models/det/model.py`, `src/models/det/preprocessor.py`, `src/models/det/wrapper.py`, `src/models/heads/detection_head.py` |
| 관련 문서 | [docs/architecture/model-assembly.md](../architecture/model-assembly.md), [docs/plans/0011-det-custom-model-plan.md](0011-det-custom-model-plan.md) |

## 1. 목적과 배경

`0011-det-custom-model-plan.md` 구현에서 `DetWrapper`는 다른 method와의 CLI 일관성을 위해
`head="detection"` 인자를 받았으나 실제로는 아무 동작도 하지 않았고(det에는 `DetectionHead`
하나뿐), box/point 표현 선택은 별도의 `representation` 인자가 담당했다. 이 결과 `head`와
`representation`이라는 두 인자가 동시에 존재하면서 `head`는 항상 상수, `representation`은 실제
선택지라는 비대칭이 생겼다.

`seg`의 `TorchSegModel`(`docs/plans/0010-torchseg-model-plan.md`)을 포함해 다른 method에서
`head`는 backbone/model 선택과 무관하게 항상 실제로 쓰이는 값이며, det도 향후 외부 whole-model
detector(Faster R-CNN/YOLO/DETR, Category C)를 추가하더라도 그쪽은 항상 box 좌표만 출력하므로
`head`에 새 옵션이 늘어나지 않는다. 즉 det의 `head`는 실질적으로 대체로 상수였던 자리였고,
`representation`이 실제 분기 정보였다. 이 plan은 이 둘을 하나로 합쳐 `head="box"`(기본값) 또는
`head="point"`로 표현하고 `representation` 파라미터를 완전히 삭제한다.

`get_experiment`(`scripts/config.py`)는 이미 `head` 값을 `exp_name`에 포함하므로, 이 변경만으로
box/point 실험의 `exp_name`과 `output_dir`이 자동으로 구분된다(`representation`을 별도로
`exp_name`에 반영하는 추가 작업이 불필요해진다).

## 2. 범위

포함 항목은 다음과 같다.

- `src/models/heads/detection_head.py`의 `representation` 생성자 인자를 `head`로 rename한다.
  `BOX_CHANNELS` lookup과 에러 메시지도 동일하게 맞춘다.
- `src/models/det/model.py`의 `DetModel` 생성자 인자를 `representation="box"`에서 `head="box"`로
  바꾸고 `DetectionHead(..., head=head)`로 전달한다. 미사용 상태였던 `self.representation` 속성은
  제거한다(다른 곳에서 읽지 않음, `RegModel`의 `self.head_name`과 달리 `forward`에서 분기가
  필요하지 않으므로 `self.head_name`도 추가하지 않는다).
- `src/models/det/preprocessor.py`의 `DetPreprocessor` 생성자 인자를 `head="box"`로 바꾸고
  내부 `self.representation`을 `self.head`로 rename한다.
- `src/models/det/wrapper.py`의 `DetWrapper` 생성자에서 `head="detection"`과
  `representation="box"` 두 인자를 `head="box"` 하나로 합친다. `DetModel`/`DetPreprocessor`
  생성 시 `head=head`를 전달한다. `DetPostprocessor`는 원래도 표현과 무관했으므로 변경하지 않는다.
- `scripts/config.py`: `DEFAULTS`에서 `representation="box"` 항목을 삭제한다(`head`는 이미
  `DEFAULTS`에 있으며 method마다 값을 다르게 해석하는 기존 관례를 그대로 따른다. 예를 들어 reg는
  `head="coord_gap"`, seg는 config에서 `head="mask"`를 명시적으로 지정하는 것처럼 det도 config에서
  `head="box"`/`head="point"`를 명시한다). `get_wrapper_kwargs`와 `parse_args`에서
  `representation` 관련 코드를 삭제한다.
- `experiments/run.py`: `PASS_KEYS`에서 `"representation"`을 삭제한다(`"head"`는 이미 있음).
- `experiments/configs.py`: det config 9개 항목의 `"head": "detection"`을 `"head": "box"`로
  바꾸고, 마지막 point 예시 항목은 `"head": "detection", "representation": "point"` 두 key를
  `"head": "point"` 하나로 합친다.
- `docs/architecture/model-assembly.md` 4.4절의 `representation` 관련 서술을 `head`로 고치고,
  4.2절의 `RegModel` variant 표와 형식을 맞춰 head 값과 raw output을 표로 정리한다. 12.3절의
  "point 또는 box representation을 확정한다" 문구도 `head`로 용어를 통일한다.

제외 항목(후속 plan에서 수행)은 다음과 같다.

- Category C(외부 whole-model detector) 통합은 여전히 범위 밖이다.
- `docs/plans/0011-det-custom-model-plan.md`은 완료된 이력 문서이므로 본문을 고치지 않고 그대로
  보존한다. 이 plan이 0011의 `representation` 설계를 대체한다는 사실은 위 "관련 문서"와 canonical
  문서 갱신으로 추적한다.

## 3. 완료 기준

다음을 만족하면 이 plan을 `Done`으로 본다.

- 코드베이스 전체에서 `representation`이라는 식별자가 더 이상 존재하지 않는다(문서의 일반적인
  "output representation" 서술은 제외, 이는 canonical 2.2절의 분류 축 이름이며 이 plan이 다루는
  det의 생성자 인자와 무관하다).
- `DetWrapper(backbone=<9개 중 하나>, head=<"box"|"point">, device="cpu")`가 정상 동작하고,
  `DetectionHead`/`DetPreprocessor`가 `head` 값에 따라 올바른 채널 수를 생성한다.
- `scripts/train.py --method det --backbone custom --head box`와
  `--head point` 두 CPU smoke train이 모두 성공한다.
- `experiments/configs.py`의 det 항목이 `head` 값만으로 box/point를 구분하고, `get_experiment`가
  이 둘에 대해 서로 다른 `exp_name`을 생성한다.

## 4. 검증

새 unit test framework는 추가하지 않는다. `conda activate pytorch_env` 뒤 다음을 확인한다.

- `DetectionHead`/`DetPreprocessor`를 `head="box"`와 `head="point"`로 각각 생성해 raw output과
  target의 채널 수를 확인한다.
- 알 수 없는 `head` 값을 넘겼을 때 `DetectionHead`와 `DetPreprocessor`가 각각 `ValueError`를
  발생시키는지 확인한다.
- `python scripts/train.py --method det --backbone custom --head box ...`와
  `--head point ...` 두 CPU smoke train을 1 epoch, 소규모 sample로 실행해 성공을 확인한다.
- `python -c` 한 줄로 `experiments/configs.py`의 det 항목에서 `get_experiment`가 서로 다른
  `exp_name`을 반환하는지 확인한다.
