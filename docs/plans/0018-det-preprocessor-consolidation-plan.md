# det preprocessor/postprocessor 파일 통합

다음 표는 이 plan의 상태와 적용 범위를 정리한다.

| 항목 | 값 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-19 |
| 적용 범위 | `docs/architecture/model-assembly.md`, `src/models/det/preprocessor.py`, `src/models/det/postprocessor.py`, `src/models/det/wrapper.py`, 삭제 대상: `src/models/det/torch_preprocessor.py`, `src/models/det/torch_postprocessor.py`, `src/models/det/yolo_preprocessor.py`, `src/models/det/yolo_postprocessor.py`, `src/models/det/detr_preprocessor.py`, `src/models/det/detr_postprocessor.py` |
| 관련 문서 | [docs/architecture/model-assembly.md](../architecture/model-assembly.md), [docs/plans/0013-det-torchdet-model-plan.md](0013-det-torchdet-model-plan.md), [docs/plans/0014-det-yolo-model-plan.md](0014-det-yolo-model-plan.md), [docs/plans/0017-det-hf-detr-model-plan.md](0017-det-hf-detr-model-plan.md) |

## 1. 목적과 배경

`det` method는 variant를 추가할 때마다 `<variant>_preprocessor.py`/`<variant>_postprocessor.py` 파일
쌍을 새로 만들어 왔다. 그 결과 `src/models/det/`에는 `preprocessor.py`/`postprocessor.py`(custom
`DetModel`용), `torch_preprocessor.py`/`torch_postprocessor.py`(`TorchDetModel`용),
`yolo_preprocessor.py`/`yolo_postprocessor.py`(`YoloDetModel`용),
`detr_preprocessor.py`/`detr_postprocessor.py`(`DetrDetModel`용) 6개 파일이 쌓여 있다.

각 파일은 단일 class로 20에서 35줄 수준이라 파일 분리로 얻는 이점(파일 크기 관리, 병행 편집 충돌
회피)이 크지 않다. 반면 같은 `det` method 안에서 `model.py`와 `wrapper.py`는 4개 variant class를
한 파일에 모아 두는 반대 컨벤션을 쓰고 있어, 같은 method 내부에서 파일 분리 기준이 일관되지 않는다.
`wrapper.py`도 6개 preprocessor/postprocessor 파일을 개별 import해야 해서 import block이
불필요하게 길다.

이 plan은 `det`의 preprocessor와 postprocessor를 각각 `preprocessor.py`, `postprocessor.py` 하나로
통합해 `model.py`/`wrapper.py`와 같은 파일 배치 컨벤션으로 맞춘다.

## 2. 범위

포함 항목은 다음과 같다.

- `src/models/det/preprocessor.py`에 `DetPreprocessor`(기존), `TorchDetPreprocessor`,
  `YoloDetPreprocessor`, `DetrDetPreprocessor`를 한 파일로 모은다.
- `src/models/det/postprocessor.py`에 `DetPostprocessor`(기존), `TorchDetPostprocessor`,
  `YoloDetPostprocessor`, `DetrDetPostprocessor`를 한 파일로 모은다.
- 통합 시 각 class와 그 docstring, 로직은 그대로 유지한다. 중복 정의된 `NUM_CORNER_CLASSES = 4`는
  파일당 한 번만 선언해 재사용한다.
- `src/models/det/torch_preprocessor.py`, `torch_postprocessor.py`, `yolo_preprocessor.py`,
  `yolo_postprocessor.py`, `detr_preprocessor.py`, `detr_postprocessor.py` 6개 파일을 삭제한다.
- `src/models/det/wrapper.py`의 import block을 `from src.models.det.preprocessor import (...)`,
  `from src.models.det.postprocessor import (...)` 형태로 정리한다.
- `docs/architecture/model-assembly.md` 7.2절의 `TorchDetPreprocessor`/`TorchDetPostprocessor` 관련
  서술에서, 별도 파일이 아니라 `preprocessor.py`/`postprocessor.py` 내 별도 class로 존재한다는 점을
  반영한다.

제외 항목은 다음과 같다.

- preprocessor/postprocessor의 내부 로직, 인터페이스, 파라미터 변경.
- `model.py`, `wrapper.py`의 class 구조나 학습 동작 변경.
- `reg`, `seg` method의 파일 구조 변경(이미 단일 파일이므로 대상 아님).

## 3. 구현 계획

1. `preprocessor.py`에 `TorchDetPreprocessor`, `YoloDetPreprocessor`, `DetrDetPreprocessor`
   class 정의를 이식하고, `torch_preprocessor.py`, `yolo_preprocessor.py`, `detr_preprocessor.py`를
   삭제한다. `postprocessor.py`도 동일하게 처리한다.
2. `wrapper.py`의 import를 갱신한다.
3. `docs/architecture/model-assembly.md` 7.2절 서술을 갱신한다.
4. `python -c "from src.models.det.wrapper import DetWrapper, TorchDetWrapper, YoloDetWrapper, DetrDetWrapper"`
   로 import 성공을 확인한다.

## 4. 완료 기준

다음 조건을 모두 만족하면 이 plan을 `Done`으로 본다.

- `src/models/det/`에 `torch_preprocessor.py`, `torch_postprocessor.py`, `yolo_preprocessor.py`,
  `yolo_postprocessor.py`, `detr_preprocessor.py`, `detr_postprocessor.py`가 존재하지 않는다.
- `preprocessor.py`와 `postprocessor.py` 각각에 4개 variant class가 모두 정의되어 있다.
- `src/core/factory.py::get_wrapper("det", ...)`이 `model=None`, `fasterrcnn_resnet50_fpn`,
  `yolov8n`, `detr_resnet50` 4개 경로 모두 회귀 없이 동작한다.
- `docs/architecture/model-assembly.md` 7.2절이 새 파일 배치를 반영한다.
- 이 문서의 상태가 `Draft`에서 `Approved`를 거쳐 `Done`으로 갱신된다.

## 5. 검증

구현 후 다음을 확인한다.

```bash
conda activate pytorch_env
python -c "from src.models.det.wrapper import DetWrapper, TorchDetWrapper, YoloDetWrapper, DetrDetWrapper; print('OK')"
python -c "from src.core.factory import get_wrapper; \
print(type(get_wrapper('det', device='cpu')).__name__); \
print(type(get_wrapper('det', model='fasterrcnn_resnet50_fpn', device='cpu')).__name__); \
print(type(get_wrapper('det', model='yolov8n', device='cpu')).__name__); \
print(type(get_wrapper('det', model='detr_resnet50', device='cpu')).__name__)"
```

4개 wrapper class 이름이 각각 `DetWrapper`, `TorchDetWrapper`, `YoloDetWrapper`, `DetrDetWrapper`로
출력되는지 확인한다.

## 6. 검증 결과

2026-07-19 구현 완료 후 다음을 확인했다.

```bash
conda activate pytorch_env
python -c "from src.models.det.wrapper import DetWrapper, TorchDetWrapper, YoloDetWrapper, DetrDetWrapper; print('OK')"
python -c "from src.core.factory import get_wrapper; \
print(type(get_wrapper('det', device='cpu')).__name__); \
print(type(get_wrapper('det', model='fasterrcnn_resnet50_fpn', device='cpu')).__name__); \
print(type(get_wrapper('det', model='yolov8n', device='cpu')).__name__); \
print(type(get_wrapper('det', model='detr_resnet50', device='cpu')).__name__)"
```

`DetWrapper`, `TorchDetWrapper`, `YoloDetWrapper`, `DetrDetWrapper` 4개 wrapper class가 모두 기대한
이름으로 반환됐고 import 오류나 회귀가 없었다. `src/models/det/`의 `torch_preprocessor.py`,
`torch_postprocessor.py`, `yolo_preprocessor.py`, `yolo_postprocessor.py`, `detr_preprocessor.py`,
`detr_postprocessor.py` 6개 파일을 삭제하고, 각 class를 `preprocessor.py`/`postprocessor.py`로
이식했다. `docs/architecture/model-assembly.md` 7.2절의 서술도 새 파일 배치를 반영하도록 갱신했다.
