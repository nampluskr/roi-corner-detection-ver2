# YoloDetModel whole detection model 추가

다음 표는 이 plan의 상태와 적용 범위를 정리한다.

| 항목 | 값 |
| --- | --- |
| 상태 | Draft |
| 작성일 | 2026-07-19 |
| 적용 범위 | `docs/architecture/model-assembly.md`, `docs/references/backbones.md`, `experiments/configs.py`, `src/core/factory.py`, `src/models/det/model.py`, `src/models/det/wrapper.py`, `src/models/det/yolo_preprocessor.py`(신규), `src/models/det/yolo_postprocessor.py`(신규), `pytorch_env`(신규 dependency: `ultralytics`) |
| 관련 문서 | [docs/architecture/model-assembly.md](../architecture/model-assembly.md), [docs/references/backbones.md](../references/backbones.md), [docs/plans/0011-det-custom-model-plan.md](0011-det-custom-model-plan.md), [docs/plans/0013-det-torchdet-model-plan.md](0013-det-torchdet-model-plan.md) |

## 1. 목적과 배경

현재 `det`는 `DetModel`(Category A/B composable)과 `TorchDetModel`(Category C, torchvision whole
model 3종)이 구현되어 있다. canonical 문서 2.4절의 historical mapping table은 `torchdet`, `yolo`,
`detr_box`를 모두 `det` method의 external whole-model variant로 이미 분류해 두었고,
`docs/references/backbones.md` 3.3절에는 `yolov8n.pt`(Ultralytics YOLOv8-Nano, COCO detection)가
"`det` YOLO corner box fine-tuning" 용도로 조건부 등록되어 있다. `docs/plans/0013-det-torchdet-model-plan.md`는
YOLO를 native API와 weight 포맷이 torchvision detection과 다르다는 이유로 명시적으로 범위에서
제외하고 후속 plan으로 미뤘다. 이번 plan이 그 후속 plan이며, YOLO(Ultralytics YOLOv8-Nano) 하나를
`det`의 새 Category C whole-model class로 추가한다.

`TorchDetModel`과 마찬가지로 YOLO도 `backbone`이 아니라 `model` 인자로 선택하는 architecture family
선택자를 사용한다. 이는 `seg`/`det`에서 이미 확인된 원칙, 즉 `backbone`은 project가 조립하는
composable model(Category A/B)의 encoder 교체에, `model`은 project가 분해할 수 없는 whole
model(Category C)의 family 선택에 쓰인다는 원칙을 그대로 따른다.

YOLOv8과 torchvision detection의 차이는 `TorchDetModel`을 확장하는 방식으로 흡수할 수 없을 만큼
크다. 로컬 `torchvision 0.20.1+cu121`은 이미 설치되어 있어 0013에서 직접 API를 확인할 수 있었지만,
`ultralytics` 패키지는 현재 `pytorch_env`에 설치되어 있지 않다(`python -c "import ultralytics"`가
`ModuleNotFoundError`). 공개된 Ultralytics YOLOv8 구조를 기준으로 예상되는 차이는 다음과 같다.

- `yolov8n.pt` checkpoint는 `torch.load`로 열면 `ultralytics.nn.tasks.DetectionModel`(pickled
  `nn.Module`)과 class 이름(COCO 80종), stride 등 부가 정보를 담은 dict다. torchvision처럼
  `state_dict`만 담긴 파일이 아니므로 로딩 방식 자체가 다르다.
- 학습 loss는 `model.forward(images, targets)`가 직접 반환하지 않는다. Ultralytics는
  `ultralytics.utils.loss.v8DetectionLoss` 같은 별도 loss class에 raw feature map 예측과, 배치
  전체를 한 번에 담는 `{"batch_idx", "cls", "bboxes", "img"}` 형식의 target을 넘겨 loss를 계산한다.
  이는 `TorchDetModel`이 쓰는 image당 `{"boxes", "labels"}` list 형식과도 다르다.
  `TorchDetPreprocessor`/`TorchDetPostprocessor`를 재사용할 수 없다.
- Ultralytics는 자체 `ultralytics.YOLO` 고수준 wrapper와 `Trainer`/`DataLoader`/CLI를 제공한다. 이
  project의 `BaseWrapper`/`Trainer`/`Evaluator`/`Predictor` 계약과 중복되므로, 이번 plan은 고수준
  `ultralytics.YOLO`를 쓰지 않고 `ultralytics.nn.tasks.DetectionModel`과 `v8DetectionLoss` 같은
  하위 구성요소만 가져와 `TorchDetModel`이 torchvision `net`을 감싼 것과 동일한 방식으로 감싼다.

이 문서에 적힌 Ultralytics 구체 API(모듈 경로, 함수 시그니처, 반환 형식)는 공개 문서와 코드 구조를
기준으로 한 설계 가정이며, `torchvision`처럼 로컬에서 직접 확인한 값이 아니다. 3.0절에서 이를
구현 전에 검증하는 단계를 별도로 둔다.

## 2. 범위

포함 항목은 다음과 같다.

- `pytorch_env`에 `ultralytics` 패키지를 설치한다(신규 dependency 추가, 사용자 승인 필요).
- `src/models/det/model.py`에 `YoloDetModel` class와 `YOLODET_WEIGHTS`,
  `SUPPORTED_YOLODET_MODELS` catalog를 추가한다. 대상은 `yolov8n` 하나다.
- `src/models/det/yolo_preprocessor.py`(신규)에 `YoloDetPreprocessor`를 추가한다. `(N,4,2)` 정규화
  corners를 Ultralytics 학습 batch 형식(`batch_idx`, `cls`, `bboxes` normalized xywh)으로 변환한다.
- `src/models/det/yolo_postprocessor.py`(신규)에 `YoloDetPostprocessor`를 추가한다. eval-mode raw
  예측(post-NMS 또는 post-decode box 목록)을 공통 `(N,4,2)` corners contract로 decode한다.
- `src/models/det/wrapper.py`에 `YoloDetWrapper` class를 추가한다. `train_step`/`eval_step`을
  override해 `v8DetectionLoss` 기반 native loss 계산과 학습 호출을 구현한다.
- `src/core/factory.py::get_wrapper`의 `det` 분기가 `model` kwarg 값에 따라 `DetWrapper`,
  `TorchDetWrapper`, `YoloDetWrapper` 중 하나를 선택하도록 확장한다.
- `docs/architecture/model-assembly.md` 7.2절에 `YoloDetModel` 설명을 `TorchDetModel` 단락과 같은
  형식으로 추가한다.
- `docs/references/backbones.md`의 `yolov8n.pt` row를 `YoloDetModel` 연결 계획과 classifier(head)
  교체 방식으로 갱신한다.
- `experiments/configs.py`에 `yolov8n` whole-model `det` config 1개를 추가한다.

제외 항목(후속 plan에서 수행)은 다음과 같다.

- DETR(Meta) 통합. `docs/references/backbones.md`의 `detr-r50-e632da11.pth`는 원본
  facebookresearch/detr repo의 checkpoint이며 torchvision이나 (현재 `pytorch_env`에 설치되지 않은)
  HuggingFace `transformers`의 `DetrForObjectDetection`과 state dict key가 그대로 맞지 않는다.
  Hungarian bipartite matching 기반 set prediction loss는 anchor 기반 YOLO/torchvision detection과도
  전혀 다른 학습 paradigm이라 별도 조사와 plan이 필요하다. canonical 7.2절 "set prediction" row에
  해당하며 계속 범위 밖으로 남긴다.
- `yolov8n` 외 다른 YOLOv8 크기(s/m/l/x)나 다른 YOLO 버전(v9-v11 등). `docs/references/backbones.md`에
  로컬로 검증된 파일이 `yolov8n.pt` 하나뿐이므로 이번 plan은 이 하나로 한정한다.
- `representation="point"`에 대응하는 YOLO variant. YOLOv8 detection head는 항상 box 좌표만 native
  출력하므로 `head`는 `TorchDetModel`과 동일하게 CLI 호환을 위해서만 받고 고정값으로 취급한다.
- Ultralytics 고수준 `ultralytics.YOLO`/`Trainer`/자체 augmentation pipeline 채택. 이 project의
  `Dataloader`/`Trainer`/`BaseWrapper` 계약을 그대로 쓰고, Ultralytics 쪽에서는 model과 loss
  구성요소만 가져온다.
- validation(eval) loop에서 native loss를 반드시 재현하는 것. 3.0절 검증 결과 `v8DetectionLoss`가
  `eval()` mode에서도 raw feature map만 있으면 호출 가능한 것으로 확인되면 valid loss를 채우고,
  그렇지 않으면 0013과 동일하게 valid loss column을 0으로 남기고 `PolygonIoU` metric만으로 조기
  종료한다. 최종 결정은 검증 결과에 따라 4절 완료 기준에서 확정한다.

## 3. 구현 계획

### 3.0. 사전 조사와 환경 준비

구현 code를 작성하기 전에 다음을 `pytorch_env`에서 직접 확인하고, 실제 결과가 아래 3.1-3.6절의
가정과 다르면 이 plan을 갱신한 뒤 구현을 진행한다.

```bash
conda activate pytorch_env
pip install ultralytics
python -c "import ultralytics; print(ultralytics.__version__)"
python -c "
import torch
ckpt = torch.load('/mnt/d/backbones/yolov8n.pt', map_location='cpu', weights_only=False)
print(type(ckpt), list(ckpt.keys()))
model = ckpt['model']
print(type(model), model.nc, model.names)
"
```

확인 항목은 다음과 같다.

- checkpoint dict의 정확한 key 구성과 `model` 객체의 class, `nc`(class 개수), `names`, `stride`.
- `model.train()` 상태에서 raw output의 정확한 형식(scale별 feature map list인지, tensor인지)과
  `v8DetectionLoss(model)(preds, batch)` 호출에 필요한 `batch` dict의 정확한 key와 shape.
- `model.eval()` 상태에서 `model(images)` 반환 형식과, box 좌표를 얻기 위해
  `ultralytics.utils.ops.non_max_suppression`(또는 동등 함수)을 별도로 호출해야 하는지 여부.
- classifier(head)를 80-class COCO에서 4-class corner로 교체하는 정확한 방법(`Detect` module의
  `nc`/`cv3`를 직접 교체하는지, `model.yaml`을 `nc=4`로 다시 빌드한 뒤 backbone/neck weight만
  `strict=False`로 load하는지).

### 3.1. family catalog와 `model` 선택 계약

`det`의 `model` 값은 `TorchDetModel`과 동일하게 architecture family 선택자로 쓴다.

| `model` 값 | class | 비고 |
|---|---|---|
| `None` | `DetModel` | 기존 custom/backbone 조합 grid 기반 model |
| `fasterrcnn_resnet50_fpn`, `retinanet_resnet50_fpn`, `ssd300_vgg16` | `TorchDetModel` | 0013에서 추가됨 |
| `yolov8n` | `YoloDetModel` | 이번 plan에서 추가, COCO 80종 head를 corner 4종 head로 교체 |

```python
YOLODET_WEIGHTS = {
    "yolov8n": "/mnt/d/backbones/yolov8n.pt",
}
SUPPORTED_YOLODET_MODELS = tuple(YOLODET_WEIGHTS.keys())
```

### 3.2. `src/models/det/model.py`에 `YoloDetModel` 추가

`YoloDetModel`은 `BaseModel`을 상속하고 다음 책임만 가진다.

- 지원 model name을 검증한다.
- `YOLODET_WEIGHTS[model]` checkpoint에서 `ultralytics.nn.tasks.DetectionModel`을 불러오고, 3.0절
  검증 결과에 맞는 방식으로 classifier(head)를 `NUM_CORNER_CLASSES=4`로 교체한다.
- `forward(images, targets=None)`에서 `self.training and targets is not None`이면 native loss
  계산에 필요한 raw feature map을, 아니면 eval-mode 예측을 그대로 반환한다. 정확한 반환 형식은
  3.0절 검증 결과에 따라 확정한다. 이 함수는 `BaseModel.forward(images)` 단일 인자 계약과 다르므로
  `TorchDetModel`처럼 `YoloDetWrapper`만 이 model을 사용한다.

### 3.3. `src/models/det/yolo_preprocessor.py`(신규)

`(N,4,2)` 정규화 corners를 Ultralytics 학습 batch 형식으로 변환한다. corner는 실제 넓이를 가진
object가 아니므로 `TorchDetPreprocessor`처럼 고정 pseudo-box 크기(`box_size`, normalized xywh 기준)를
사용해 anchor/target 대응을 만든다. 정확한 key 이름과 shape는 3.0절 검증 결과로 확정한다.

### 3.4. `src/models/det/yolo_postprocessor.py`(신규)

eval-mode raw 예측(post-NMS box 목록 또는 post-decode tensor)을 공통 `(N,4,2)` corners contract로
decode한다. `TorchDetPostprocessor`와 동일하게 class별 최고 score box의 중심점을 사용하고, 예측이
없는 class는 image 중심(0.5, 0.5)로 fallback한다.

### 3.5. `src/models/det/wrapper.py`에 `YoloDetWrapper` 추가

`TorchDetWrapper`와 동일한 구조로 `train_step`/`eval_step`/`predict_step`을 override한다.

- `train_step`: `YoloDetPreprocessor`로 batch target을 만들고, `YoloDetModel`의 raw output과 함께
  `v8DetectionLoss`(또는 동등 구성요소)를 호출해 loss dict를 얻은 뒤 backward한다. loss key 이름은
  Ultralytics native loss 항목(box/cls/dfl 등)을 그대로 쓴다.
  `self.losses`는 `TorchDetWrapper`처럼 빈 dict에서 시작해 native loss key로 동적으로 채운다.
  `head`/`backbone` kwarg는 CLI 호환을 위해서만 받고 사용하지 않는다.
- `eval_step`/`predict_step`: `YoloDetModel`의 eval-mode 예측을 `YoloDetPostprocessor`로 decode하고
  `PolygonIoU` metric을 계산한다.
- optimizer는 `TorchDetWrapper`와 동일하게 전체 parameter에 단일 `AdamW` group을 적용한다.

### 3.6. `src/core/factory.py` dispatch

```python
if method == "det":
    from src.models.det.model import SUPPORTED_TORCHDET_MODELS, SUPPORTED_YOLODET_MODELS
    if kwargs.get("model") in SUPPORTED_TORCHDET_MODELS:
        from src.models.det.wrapper import TorchDetWrapper
        return TorchDetWrapper(device=device, **kwargs)
    if kwargs.get("model") in SUPPORTED_YOLODET_MODELS:
        from src.models.det.wrapper import YoloDetWrapper
        return YoloDetWrapper(device=device, **kwargs)
    from src.models.det.wrapper import DetWrapper
    return DetWrapper(device=device, **kwargs)
```

### 3.7. config와 canonical 문서

`experiments/configs.py`에 다음 후보를 추가한다.

```python
{"method": "det", "model": "yolov8n", "batch_size": 4, "max_epochs": 5, "backbone": "", "head": "box"},
```

`docs/architecture/model-assembly.md` 7.2절에는 `TorchDetModel` 단락과 같은 형식으로
`YoloDetModel`의 조립 경계와 classifier 교체 방식을 기록한다. `docs/references/backbones.md`의
`yolov8n.pt` row는 "적용 방법과 제약" 열에 `YoloDetModel` 연결 계획과 head 교체 방식을 반영한다.

## 4. 완료 기준

이 plan은 다음 조건을 만족하면 `Done`으로 본다.

- 3.0절 사전 조사가 완료되고, 실제 Ultralytics API가 이 문서의 가정과 다른 경우 해당 절이 실제
  확인된 내용으로 갱신된다.
- `docs/architecture/model-assembly.md` 7.2절에 `YoloDetModel`의 조립 경계가 명시된다.
- `docs/references/backbones.md`의 `yolov8n.pt` row에 `YoloDetModel` 연결 계획이 반영된다.
- `YoloDetModel(model="yolov8n")`이 local checkpoint를 불러오고 4-class corner classifier(head)
  교체까지 완료한다. `YoloDetModel(model="unknown")`은 지원 목록을 포함한 `ValueError`를 발생시킨다.
- `YoloDetWrapper(model="yolov8n", device="cpu")`가 2-sample smoke `train_step`과 `eval_step`을
  shape/타입 오류 없이 완료하고, `train_step`이 반환하는 loss 이름이 Ultralytics native loss key와
  일치한다.
- `src/core/factory.py::get_wrapper("det", model="yolov8n")`이 `YoloDetWrapper`를 반환하고, 기존
  `model=fasterrcnn_resnet50_fpn`/`model=None` 경로는 회귀 없이 그대로 동작한다.
- `experiments/configs.py`에 `yolov8n` whole-model det config가 추가된다.
- 이 문서의 상태가 `Approved`에서 `Done`으로 갱신된다.

## 5. 검증

구현 후 다음 순서로 검증한다. 정확한 명령은 3.0절 검증 결과에 따라 조정될 수 있다.

```bash
conda activate pytorch_env
python -c "import torch; from src.models.det.model import YoloDetModel; \
net = YoloDetModel(model='yolov8n'); net.eval(); \
out = net([torch.rand(3,224,224)]); print(type(out))"
python -c "from src.models.det.model import YoloDetModel; \
try: YoloDetModel(model='unknown')
except ValueError as e: print('OK:', e)"
python -c "from src.core.factory import get_wrapper; \
w = get_wrapper('det', model='yolov8n', device='cpu'); \
print(type(w).__name__)"
python scripts/train.py --method det --model yolov8n --backbone '' --head box --device cpu \
  --train_size 2 --valid_size 2 --batch_size 2 --max_epochs 1 --patience 1 --num_workers 0 \
  --output_dir /tmp/det_yolov8n_smoke
```

검증 결과에서는 `YoloDetModel`의 train/eval output 형태, `ValueError`, factory dispatch, smoke
train 성공 여부, train loss 이름이 Ultralytics native loss key와 일치하는지, `iou` metric이 유효한
값을 갖는지 확인한다. 검증 후 `/tmp/det_yolov8n_smoke` 산출물은 삭제한다.
