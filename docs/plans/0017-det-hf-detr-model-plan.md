# Hugging Face DetrDetModel whole detection model 추가

다음 표는 이 plan의 상태와 적용 범위를 정리한다.

| 항목 | 값 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-19 |
| 적용 범위 | `docs/architecture/model-assembly.md`, `docs/references/backbones.md`, `experiments/configs.py`, `src/core/factory.py`, `src/models/det/model.py`, `src/models/det/wrapper.py`, `src/models/det/detr_preprocessor.py`(신규), `src/models/det/detr_postprocessor.py`(신규), `pytorch_env`(신규 dependency: `transformers`), `/mnt/d/backbones/facebook-detr-resnet-50/` |
| 관련 문서 | [docs/architecture/model-assembly.md](../architecture/model-assembly.md), [docs/references/backbones.md](../references/backbones.md), [docs/plans/0013-det-torchdet-model-plan.md](0013-det-torchdet-model-plan.md), [docs/plans/0014-det-yolo-model-plan.md](0014-det-yolo-model-plan.md), [docs/plans/0016-det-detr-model-plan.md](0016-det-detr-model-plan.md) |

## 1. 목적과 배경

`det` method는 현재 `DetModel`(Category A/B composable), `TorchDetModel`(Category C, torchvision
whole model 3종), `YoloDetModel`(Category C, Ultralytics YOLOv8-Nano)을 갖는다. canonical 문서
7.2절은 DETR를 set prediction family로 분류하고, `det`의 external whole-model variant로 흡수할 수
있다고 설명한다. 0016 plan은 Meta 원본 `detr-r50-e632da11.pth` checkpoint를 strict load하기 위해
facebookresearch/detr 코드 이식 또는 Hugging Face key remapping을 조사하는 방향이었으나, 이 접근은
구현 부담과 유지보수 위험이 크므로 폐기한다.

이번 plan은 Hugging Face `transformers.DetrForObjectDetection`이 제공하는 구현된 DETR architecture와
native Hungarian loss를 사용한다. 사용자는 `facebook/detr-resnet-50` snapshot을
`/mnt/d/backbones/facebook-detr-resnet-50/`에 직접 다운로드했다. 이 project는 해당 local snapshot을
`local_files_only=True`로 로드하고, COCO classifier를 corner 4-class classifier로 교체해
fine-tuning한다. 기존 `/mnt/d/backbones/detr-r50-e632da11.pth`는 보존하지만 이번 `DetrDetModel`
구현에서는 사용하지 않는다.

현재 확인된 local snapshot 파일은 다음과 같다.

| 파일 | 크기 | SHA-256 |
| --- | --- | --- |
| `facebook-detr-resnet-50/config.json` | 4,592 B | `e7bcf3992363f27717a863f14b193140ad2e41d4338ee012730e58a92cae17e6` |
| `facebook-detr-resnet-50/model.safetensors` | 166,587,896 B | `830f5e2eeaada8c8c8281779dcc8ab12833972eb8514ed0a35be6c1d4420ad81` |
| `facebook-detr-resnet-50/preprocessor_config.json` | 290 B | `0673fea2a6d3cf92cdbab3c7426c0ecdf8a4729a2a4d5199033dcd66a2b8759b` |

## 2. 범위

포함 항목은 다음과 같다.

- `pytorch_env`에 `transformers`를 설치한다. 기존 `torch 2.5.1+cu121`와 `torchvision 0.20.1+cu121`
  CUDA build를 보존하기 위해 `transformers[torch]`는 사용하지 않는다.
- `src/models/det/model.py`에 `DETRDET_MODEL_DIR`, `SUPPORTED_DETRDET_MODELS` catalog와
  `DetrDetModel(BaseModel)` class를 추가한다. 지원 model id는 `detr_resnet50` 하나다.
- `DetrDetModel`은 `transformers.DetrForObjectDetection.from_pretrained`를 local snapshot path로
  호출하고, 4개 corner class용 `id2label`/`label2id`, `ignore_mismatched_sizes=True`,
  `local_files_only=True`를 사용한다.
- `src/models/det/detr_preprocessor.py`에 `DetrDetPreprocessor`를 추가한다. `(N,4,2)` 정규화 corners를
  Hugging Face DETR labels 형식인 image별 `{"class_labels": LongTensor(4), "boxes": FloatTensor(4,4)}`
  로 변환한다.
- `src/models/det/detr_postprocessor.py`에 `DetrDetPostprocessor`를 추가한다. Hugging Face output의
  `logits`와 `pred_boxes`를 공통 `(N,4,2)` corners contract로 decode한다.
- `src/models/det/wrapper.py`에 `DetrDetWrapper(BaseWrapper)`를 추가한다. `train_step`, `eval_step`,
  `predict_step`을 override해 Hugging Face native loss와 project metric logging을 연결한다.
- `src/core/factory.py::get_wrapper`의 `det` 분기에 `SUPPORTED_DETRDET_MODELS` 체크를 추가한다.
- `docs/architecture/model-assembly.md` 7.2절에 Hugging Face `DetrDetModel` 조립 경계를 추가한다.
- `docs/references/backbones.md`의 detection 후보 표에 local Hugging Face DETR snapshot row를 추가하고,
  기존 Meta DETR checkpoint row는 이번 구현에서 사용하지 않는다고 갱신한다.
- `experiments/configs.py`에 `detr_resnet50` whole-model `det` config 1개를 주석 처리 상태로 추가한다.

제외 항목은 다음과 같다.

- 기존 `/mnt/d/backbones/detr-r50-e632da11.pth`를 Hugging Face key로 remapping하거나 strict load하는 작업.
- facebookresearch/detr 원본 repository 코드를 `src/models/det/detr/` 아래 vendoring하는 작업.
- `pytorch_model.bin` 사용. 이번 plan은 `model.safetensors`를 canonical Hugging Face weight로 사용한다.
- Deformable DETR, Conditional DETR, DETR point head, query 수 또는 threshold 튜닝 ablation.
- production 성능 보장. 이번 plan의 완료 기준은 구현 smoke와 trainer 연결 성공이다.

## 3. 구현 계획

### 3.1. 환경과 local snapshot 검증

구현 전에 다음을 확인한다.

```bash
conda activate pytorch_env
pip install transformers
python -c "import torch, torchvision, transformers; print(torch.__version__, torchvision.__version__, transformers.__version__)"
python -c "from transformers import DetrForObjectDetection; DetrForObjectDetection.from_pretrained('/mnt/d/backbones/facebook-detr-resnet-50', id2label={0:'corner0',1:'corner1',2:'corner2',3:'corner3'}, label2id={'corner0':0,'corner1':1,'corner2':2,'corner3':3}, ignore_mismatched_sizes=True, local_files_only=True)"
```

설치 후 `torch`와 `torchvision` version이 각각 `2.5.1+cu121`, `0.20.1+cu121`로 유지되는지 확인한다.
`transformers[torch]`를 사용하지 않았는데도 torch package가 바뀌었다면 설치를 중단하고 환경을
복구한다.

### 3.2. `DetrDetModel` 추가

`src/models/det/model.py`에 다음 catalog를 추가한다.

```python
DETRDET_MODEL_DIR = {
    "detr_resnet50": "/mnt/d/backbones/facebook-detr-resnet-50",
}
SUPPORTED_DETRDET_MODELS = tuple(DETRDET_MODEL_DIR.keys())
```

`DetrDetModel`은 다음 책임만 가진다.

- 지원 model name을 검증한다.
- local snapshot directory가 없으면 `FileNotFoundError`를 발생시킨다.
- `DetrForObjectDetection.from_pretrained(path, id2label=..., label2id=..., ignore_mismatched_sizes=True,
  local_files_only=True)`로 model을 만든다.
- `forward(images, labels=None)`는 Hugging Face model을 `pixel_values=images`, `labels=labels`로 호출하고
  output object를 그대로 반환한다.

### 3.3. 전처리와 후처리

`DetrDetPreprocessor`는 `box_size=0.1`을 기본값으로 사용한다. 각 sample의 네 corner를 normalized
`cxcywh` pseudo-box로 만들고, `class_labels`는 `0, 1, 2, 3`을 사용한다. Hugging Face DETR는 no-object
class를 내부적으로 추가하므로 target label에는 no-object를 넣지 않는다.

`DetrDetPostprocessor`는 output의 `logits`에 softmax를 적용하고 마지막 no-object class를 제외한다.
corner class `0..3` 각각에 대해 가장 높은 score query 하나를 고른 뒤, 같은 query의
`pred_boxes[..., 0:2]` 중심점을 반환한다. 첫 구현에서는 confidence threshold fallback을 두지 않는다.

### 3.4. `DetrDetWrapper` 추가

`DetrDetWrapper`는 `TorchDetWrapper`와 `YoloDetWrapper`의 Category C signature를 따른다. `backbone`과
`head` kwarg는 CLI 호환을 위해 받지만 사용하지 않는다.

학습과 검증 동작은 다음과 같다.

- `train_step`: images와 targets를 device로 옮기고, preprocessor로 labels를 만든 뒤
  `self.model(images, labels=labels)`를 호출한다. `outputs.loss`로 backward하고, `outputs.loss_dict`의
  항목을 `BaseLoss` accumulator에 기록한다.
- `eval_step`: train과 같은 labels를 넘겨 valid loss를 기록한다. 같은 output을 postprocessor로
  decode하고 `PolygonIoU` metric을 갱신한다.
- `predict_step`: labels 없이 forward하고 postprocessor 결과를 numpy로 반환한다.
- optimizer는 DETR fine-tuning 안정성을 위해 backbone `lr=1e-6`, pretrained transformer와 box head
  `lr=1e-5`, 새 classifier `lr=1e-4`의 `AdamW(weight_decay=1e-4)` parameter group을 적용한다.
  backward 뒤에는 `clip_grad_norm_(max_norm=0.1)`을 적용하고, scheduler는 기존 Category C wrapper와
  동일하게 `ReduceLROnPlateau(mode="max")`를 사용한다.

### 3.5. factory, config와 문서 갱신

`src/core/factory.py`의 `det` 분기는 `SUPPORTED_TORCHDET_MODELS`, `SUPPORTED_YOLODET_MODELS`,
`SUPPORTED_DETRDET_MODELS` 순서로 whole-model adapter를 확인한다. 기존 `model=None` 경로는 마지막에
`DetWrapper`로 유지한다.

`experiments/configs.py`에는 다음 후보를 주석 처리 상태로 추가한다.

```python
# {"method": "det", "model": "detr_resnet50", "batch_size": 4, "max_epochs": 5, "backbone": "", "head": "box"},
```

canonical architecture 문서와 weight catalog에는 Hugging Face local snapshot 방식, classifier 교체,
query selection decode 방식을 기록한다.

## 4. 완료 기준

다음 조건을 모두 만족하면 이 plan을 `Done`으로 본다.

- `transformers` 설치 후 `torch 2.5.1+cu121`와 `torchvision 0.20.1+cu121`가 유지된다.
- `/mnt/d/backbones/facebook-detr-resnet-50`에서 `local_files_only=True` model load가 성공한다.
- `docs/architecture/model-assembly.md` 7.2절에 Hugging Face `DetrDetModel` 조립 경계가 명시된다.
- `docs/references/backbones.md`의 Hugging Face DETR snapshot row와 기존 Meta DETR row가 갱신된다.
- `DetrDetModel(model="detr_resnet50")`이 dummy forward에서 `logits`와 `pred_boxes`를 반환한다.
- `DetrDetModel(model="unknown")`은 지원 목록을 포함한 `ValueError`를 발생시킨다.
- `DetrDetWrapper(model="detr_resnet50", device="cpu")`가 2-sample smoke `train_step`과 `eval_step`을
  shape와 type 오류 없이 완료하고, valid result에 loss와 `iou`가 포함된다.
- `src/core/factory.py::get_wrapper("det", model="detr_resnet50")`이 `DetrDetWrapper`를 반환하고, 기존
  `model=None`, `model=fasterrcnn_resnet50_fpn`, `model=yolov8n` 경로가 회귀 없이 동작한다.
- `experiments/configs.py`에 `detr_resnet50` config가 추가된다.
- 이 문서의 상태가 `Draft`에서 `Approved`를 거쳐 `Done`으로 갱신된다.

## 5. 검증

구현 후 다음 순서로 검증한다.

```bash
conda activate pytorch_env
python -c "import torch, torchvision, transformers; print(torch.__version__, torchvision.__version__, transformers.__version__)"
python -c "from src.models.det.model import DetrDetModel; import torch; net = DetrDetModel(model='detr_resnet50'); out = net(torch.rand(2,3,224,224)); print(out.logits.shape, out.pred_boxes.shape)"
python -c "from src.models.det.model import DetrDetModel; \
try: DetrDetModel(model='unknown')
except ValueError as e: print('OK:', e)"
python -c "from src.core.factory import get_wrapper; w = get_wrapper('det', model='detr_resnet50', device='cpu'); print(type(w).__name__)"
python scripts/train.py --method det --model detr_resnet50 --backbone '' --head box --device cpu \
  --train_size 2 --valid_size 2 --batch_size 2 --max_epochs 1 --patience 1 --num_workers 0 \
  --output_dir /tmp/det_detr_resnet50_smoke
```

검증 결과에서는 local Hugging Face load, output shape, `ValueError`, factory dispatch, smoke train 성공
여부, train/valid loss 이름, valid `iou` 기록 여부를 확인한다. 검증 후 `/tmp/det_detr_resnet50_smoke`
산출물은 삭제한다.

2026-07-19 검증 결과 `transformers 5.14.1` 설치 후 `torch 2.5.1+cu121`와
`torchvision 0.20.1+cu121`가 유지됐다. `DetrDetModel` dummy forward는 `logits=(2,100,5)`와
`pred_boxes=(2,100,4)`를 반환했고, `DetrDetWrapper` dummy train/eval과
`scripts/train.py --method det --model detr_resnet50` 2-sample smoke가 성공했다. Factory dispatch는
`model=None`, `fasterrcnn_resnet50_fpn`, `yolov8n`, `detr_resnet50` 모두 기대 wrapper를 반환했다.

같은 날 production config에서 1 epoch 이후 predicted box가 `NaN`이 되어 Hugging Face matcher의 GIoU
계산이 실패하는 문제가 확인됐다. `DetrDetWrapper`는 이를 막기 위해 DETR 전용 differential learning
rate와 gradient clipping을 적용하도록 갱신했다.
`train_size=16`, `valid_size=8`, `max_epochs=5`, CPU smoke는 NaN 없이 완료됐다.
