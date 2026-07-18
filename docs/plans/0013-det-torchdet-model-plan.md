# TorchDetModel whole detection model 3종 추가

다음 표는 이 plan의 상태와 적용 범위를 정리한다.

| 항목 | 값 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-19 |
| 적용 범위 | `docs/architecture/model-assembly.md`, `docs/references/backbones.md`, `experiments/configs.py`, `src/core/factory.py`, `src/models/det/model.py`, `src/models/det/wrapper.py`, `src/models/det/torch_preprocessor.py`(신규), `src/models/det/torch_postprocessor.py`(신규) |
| 관련 문서 | [docs/architecture/model-assembly.md](../architecture/model-assembly.md), [docs/references/backbones.md](../references/backbones.md), [docs/plans/0010-torchseg-model-plan.md](0010-torchseg-model-plan.md), [docs/plans/0011-det-custom-model-plan.md](0011-det-custom-model-plan.md), [docs/plans/0012-det-head-box-point-rename-plan.md](0012-det-head-box-point-rename-plan.md) |

## 1. 목적과 배경

현재 `det`는 `DetModel`(Category A/B composable, grid 기반 anchor-free head) 하나만 구현되어 있다.
canonical 문서 7절은 Category C external whole model을 segmentation, detection, set prediction
세 family로 분류하며, `docs/references/backbones.md`에는 이미 torchvision detection whole model
COCO weight 세 개(`fasterrcnn_resnet50_fpn`, `retinanet_resnet50_fpn`, `ssd300_vgg16`)가 등록되어
있다. 이번 plan은 이 세 모델을 `TorchDetModel`로 감싸 `det` method에 Category C variant를 추가한다.
YOLO(Ultralytics)와 DETR(Meta, set prediction)은 native API와 output 구조가 torchvision detection
계열과도 다르므로 이번 plan에서 제외하고 후속 plan으로 분리한다.

`seg`의 `TorchSegModel`(`docs/plans/0010-torchseg-model-plan.md`)은 native output(`"out"` tensor)이
기존 `SegPreprocessor`/`BCELoss`/`DiceLoss`/`SegPostprocessor`와 그대로 호환되어 `SegWrapper` 하나로
`SegModel`/`TorchSegModel`을 함께 dispatch할 수 있었다. 반면 torchvision detection whole model은
로컬 `torchvision 0.20.1+cu121`에서 직접 확인한 결과 다음과 같이 `det`의 기존 계약과 근본적으로
다르다.

- `model.forward(images, targets)`는 `model.training`일 때만 (images, targets)를 함께 받아 native
  loss dict를 반환한다. `model.eval()`에서는 targets를 넘겨도 항상 image당 `{"boxes", "labels",
  "scores"}` 가변 길이 list를 반환하며, native loss는 얻을 수 없다.
- native output은 `DetModel`의 grid 기반 `{"cls", "box"}` dense map이 아니라 image별 가변 개수의
  box 목록이므로, 기존 `DetPreprocessor`/`DetPostprocessor`/`FocalLoss`/`SmoothL1Loss`를 재사용할 수
  없다.
- `BaseWrapper.train_step`은 `self.model(images)` 한 인자 호출만 지원하므로 native
  `(images, targets)` 학습 호출과 맞지 않는다.

이 차이 때문에 `seg`처럼 기존 `DetWrapper` 안에 분기를 추가하는 대신, **`TorchDetModel`은
`DetModel`과 같은 파일에 두되(구조 조립 대상이라는 점은 동일), wrapper는 `TorchDetWrapper`라는
별도 class로 분리**한다. `train_step`/`eval_step`을 override해 canonical 7.2절이 말하는 "whole
model의 internal loss를 사용하는 경우 BaseWrapper adapter가 loss dictionary를 공통 trainer에
연결한다"를 구현한다.

## 2. 범위

포함 항목은 다음과 같다.

- `src/models/det/model.py`에 `TorchDetModel` class와 `TORCHDET_BUILDERS`, `TORCHDET_WEIGHTS`,
  `TORCHDET_LABEL_OFFSET` catalog를 추가한다. 대상은 `fasterrcnn_resnet50_fpn`,
  `retinanet_resnet50_fpn`, `ssd300_vgg16` 세 개다.
- `src/models/det/torch_preprocessor.py`(신규)에 `TorchDetPreprocessor`를 추가한다. `(N,4,2)`
  정규화 corners를 torchvision detection target 형식(`list[{"boxes": (4,4) xyxy pixel Tensor,
  "labels": (4,) Tensor}]`)으로 변환한다.
- `src/models/det/torch_postprocessor.py`(신규)에 `TorchDetPostprocessor`를 추가한다. eval-mode
  native output(`list[{"boxes", "labels", "scores"}]`)을 공통 `(N,4,2)` corners contract로 decode한다.
- `src/models/det/wrapper.py`에 `TorchDetWrapper` class를 추가한다. `train_step`/`eval_step`을
  override해 native `(images, targets)` 학습 호출과 native loss dict 기록을 구현한다.
- `src/core/factory.py::get_wrapper`의 `det` 분기가 `model` kwarg 값에 따라 `DetWrapper`
  (custom/backbone 조합) 또는 `TorchDetWrapper`(torchvision whole model)를 선택하도록 확장한다.
- `docs/architecture/model-assembly.md` 7.2절에 `TorchDetModel` 설명을 추가하고, `SegModel`/
  `TorchSegModel` 조립 경계 서술과 형식을 맞춘다.
- `docs/references/backbones.md`의 Faster R-CNN, RetinaNet, SSD300 weight row에 `TorchDetModel`
  연결 계획과 classifier 교체 방식을 반영한다.
- `experiments/configs.py`에 torchdet whole-model `det` config 3개를 추가한다.

제외 항목(후속 plan에서 수행)은 다음과 같다.

- YOLO(Ultralytics) 통합. native API, 학습 loop, weight 포맷이 torchvision detection과 달라 별도
  adapter가 필요하다.
- DETR(Meta) 통합. set prediction(Hungarian matching, 고정 query slot) 계열이라 anchor 기반
  torchvision detection과도 다른 adapter가 필요하다. canonical 7.2절의 "set prediction" row에 해당한다.
- `representation="point"`에 대응하는 torchvision whole-model variant. torchvision detection
  model은 항상 box 좌표만 native 출력하므로 `head`는 `TorchSegModel`의 `head="mask"`처럼 CLI 호환을
  위해서만 받고 고정값으로 취급한다(자세한 배경은 [[0012-det-head-box-point-rename-plan]] 참조).
  torchvision detection이 아닌 순수 point/keypoint whole model 통합은 범위 밖이다.
- validation(eval) loop에서 native loss dict를 재현하는 것. torchvision detection model은
  `eval()` mode에서 loss를 반환하지 않으므로, 이번 plan은 valid loss column을 0으로 남기고
  조기 종료는 `PolygonIoU` metric(`monitor="iou"`)만 사용한다.
- `TorchDetPostprocessor`의 클래스별 예측이 없는 경우(post-NMS 0개)에 대한 정교한 fallback 전략
  비교. 이번 plan은 image 중심점(0.5, 0.5) 단일 fallback만 사용하고 개선은 별도 plan으로 남긴다.

## 3. 구현 계획

### 3.1. family catalog와 `model` 선택 계약

`det`의 `model` 값은 `seg`와 동일하게 architecture family 선택자로 사용한다.

| `model` 값 | class | 비고 |
|---|---|---|
| `None` | `DetModel` | 기존 custom/backbone 조합 grid 기반 model |
| `fasterrcnn_resnet50_fpn` | `TorchDetModel` | background(0) + corner(1-4), `num_classes=5` |
| `retinanet_resnet50_fpn` | `TorchDetModel` | background 없음, corner(0-3), `num_classes=4` |
| `ssd300_vgg16` | `TorchDetModel` | background(0) + corner(1-4), `num_classes=5`, 입력 내부에서 항상 300x300으로 resize(고정, 교체 불가) |

```python
TORCHDET_WEIGHTS = {
    "fasterrcnn_resnet50_fpn": "/mnt/d/backbones/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
    "retinanet_resnet50_fpn": "/mnt/d/backbones/retinanet_resnet50_fpn_coco-eeacb38b.pth",
    "ssd300_vgg16": "/mnt/d/backbones/ssd300_vgg16_coco-b556d3b4.pth",
}
TORCHDET_LABEL_OFFSET = {
    "fasterrcnn_resnet50_fpn": 1,
    "retinanet_resnet50_fpn": 0,
    "ssd300_vgg16": 1,
}
SUPPORTED_TORCHDET_MODELS = tuple(TORCHDET_WEIGHTS.keys())
```

`label_offset`은 corner class index(0-3)와 torchvision model이 쓰는 label id 사이의 변환 상수다.
Faster R-CNN/SSD는 label 0을 background로 예약하므로 corner class `c`는 label `c + 1`, RetinaNet은
background class가 없으므로 corner class `c`는 label `c` 그대로 사용한다(로컬 torchvision에서
`RetinaNetClassificationHead(..., num_classes=4)`로 직접 검증).

### 3.2. `src/models/det/model.py`에 `TorchDetModel` 추가

`TorchDetModel`은 `BaseModel`을 상속하고 다음 책임만 가진다.

- 지원 model name을 검증하고 `TORCHDET_LABEL_OFFSET[model]`을 `self.label_offset`으로 노출한다.
- torchvision detection builder를 `min_size=224, max_size=224`로 호출한다(SSD는 이 kwarg를
  지원하지 않고 항상 300x300으로 강제 resize하므로 전달하지 않는다).
- local COCO checkpoint를 91-class model로 strict load한 뒤 마지막 classifier를 4 또는 5-class로
  교체한다(family별 교체 코드는 private helper로 분리).
- `forward(images, targets=None)`에서 `self.training and targets is not None`이면
  `self.net(images, targets)`(native loss dict), 아니면 `self.net(images)`(native prediction list)를
  그대로 반환한다. 이 함수는 `BaseModel.forward(images)` 단일 인자 계약과 다르므로
  `TorchDetWrapper`만 이 model을 사용한다.

```python
def build_classifier(model_name, net, num_classes):
    if model_name == "fasterrcnn_resnet50_fpn":
        in_features = net.roi_heads.box_predictor.cls_score.in_features
        net.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    elif model_name == "retinanet_resnet50_fpn":
        num_anchors = net.head.classification_head.num_anchors
        in_channels = net.backbone.out_channels
        net.head.classification_head = RetinaNetClassificationHead(in_channels, num_anchors, num_classes)
    elif model_name == "ssd300_vgg16":
        head = net.head.classification_head
        in_channels = [m.in_channels for m in head.module_list]
        num_anchors = [m.out_channels // 91 for m in head.module_list]
        net.head.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)
    return net
```

`num_classes`는 family catalog에서 `label_offset == 0`이면 4, 아니면 5로 계산한다
(`num_classes = NUM_CORNER_CLASSES + label_offset`).

### 3.3. `src/models/det/torch_preprocessor.py`(신규)

```python
class TorchDetPreprocessor(BasePreprocessor):
    def __init__(self, image_size=224, box_size=0.1, label_offset=1):
        self.image_size = image_size
        self.box_pixels = box_size * image_size
        self.label_offset = label_offset

    def __call__(self, corners):
        targets = []
        for sample in corners:
            cx = sample[:, 0] * self.image_size
            cy = sample[:, 1] * self.image_size
            half = self.box_pixels / 2
            boxes = torch.stack([cx - half, cy - half, cx + half, cy + half], dim=1)
            labels = torch.arange(4, device=corners.device) + self.label_offset
            targets.append({"boxes": boxes, "labels": labels})
        return targets
```

`box_size` 기본값은 `DetPreprocessor`와 동일한 0.1(정규화 좌표계)을 pixel로 환산한 placeholder다.
corner는 실제 넓이를 가진 object가 아니므로 이 값 자체는 학습 신호가 아니라 anchor matching을 위한
고정 크기다.

### 3.4. `src/models/det/torch_postprocessor.py`(신규)

```python
class TorchDetPostprocessor(BasePostprocessor):
    def __init__(self, image_size=224, label_offset=1):
        self.image_size = image_size
        self.label_offset = label_offset

    def __call__(self, raw_output):
        n = len(raw_output)
        corners = torch.full((n, 4, 2), 0.5)
        for i, pred in enumerate(raw_output):
            boxes, labels, scores = pred["boxes"], pred["labels"], pred["scores"]
            for c in range(4):
                mask = labels == (c + self.label_offset)
                if not mask.any():
                    continue
                best = scores[mask].argmax()
                box = boxes[mask][best]
                corners[i, c, 0] = (box[0] + box[2]) / 2 / self.image_size
                corners[i, c, 1] = (box[1] + box[3]) / 2 / self.image_size
        return corners
```

`raw_output`이 `DetModel`처럼 dict가 아니라 list이므로 `DetPostprocessor`와 완전히 다른 구현이
필요하다. 예측이 없는 class는 image 중심(0.5, 0.5)로 fallback한다.

### 3.5. `src/models/det/wrapper.py`에 `TorchDetWrapper` 추가

```python
class TorchDetWrapper(BaseWrapper):
    """Wraps TorchDetModel with native torchvision detection train/eval semantics."""

    def __init__(self, backbone=None, head="box", model=None, box_size=0.1, image_size=224,
                 optimizer=None, scheduler=None, preprocessor=None, postprocessor=None,
                 metrics=None, device=None):
        net = TorchDetModel(model=model)
        preprocessor = preprocessor or TorchDetPreprocessor(
            image_size=image_size, box_size=box_size, label_offset=net.label_offset)
        postprocessor = postprocessor or TorchDetPostprocessor(
            image_size=image_size, label_offset=net.label_offset)
        super().__init__(net, preprocessor, postprocessor, optimizer=optimizer,
                          scheduler=scheduler, losses=None, metrics=metrics, device=device)
        self.set_optimizer(self.optimizer or AdamW(self.model.parameters(), lr=1e-4))
        self.set_scheduler(self.scheduler or ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2,
            threshold=1e-4, threshold_mode="abs", min_lr=1e-7))
        self.set_metrics(self.metrics or {"iou": PolygonIoU()})

    def train_step(self, images, targets):
        self.model.train()
        images = images.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        native_targets = self.preprocessor(targets)

        self.optimizer.zero_grad()
        loss_dict = self.model(list(images), native_targets)
        loss = sum(loss_dict.values())
        loss.backward()
        self.optimizer.step()

        for name, value in loss_dict.items():
            self.losses.setdefault(name, BaseLoss()).update(value.item(), len(images))
        return {**self.get_loss_results(), **self.get_metric_results()}

    @torch.no_grad()
    def eval_step(self, images, targets):
        self.model.eval()
        images = images.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        raw_output = self.model(list(images))
        self.compute_metrics(raw_output, targets)
        return {**self.get_loss_results(), **self.get_metric_results()}

    @torch.no_grad()
    def predict_step(self, images):
        self.model.eval()
        raw_output = self.model(list(images.to(self.device, non_blocking=True)))
        preds = self.postprocessor(raw_output)
        return preds.cpu().numpy()
```

`head`는 `SegWrapper`의 `TorchSegModel` 경로와 동일하게 CLI 호환을 위해서만 받고 사용하지 않는다.
`self.losses`는 `set_losses(None)`으로 빈 dict에서 시작해 `train_step`에서 native loss dict의 key로
동적으로 채워진다(family마다 key 이름이 다르므로 고정 schema를 두지 않는다). `compute_losses`는
`TorchDetWrapper`에서 사용하지 않으므로 override하지 않고 단순히 호출되지 않게 둔다.

optimizer는 `SegWrapper`의 `TorchSegModel` 경로처럼 전체 parameter에 단일 `AdamW` group을
적용한다(backbone과 head를 구분하는 differential LR은 이번 plan에서 도입하지 않는다).

### 3.6. `src/core/factory.py` dispatch

```python
if method == "det":
    from src.models.det.model import SUPPORTED_TORCHDET_MODELS
    if kwargs.get("model") in SUPPORTED_TORCHDET_MODELS:
        from src.models.det.wrapper import TorchDetWrapper
        return TorchDetWrapper(device=device, **kwargs)
    from src.models.det.wrapper import DetWrapper
    return DetWrapper(device=device, **kwargs)
```

`DetWrapper`는 `model` kwarg를 받지 않으므로, `get_wrapper_kwargs`가 `model`을 넘길 때
`DetWrapper.__init__`이 `TypeError`를 내지 않도록 `DetWrapper`에도 무시 가능한 `model=None` kwarg를
추가한다(전달만 받고 사용하지 않음, `SegWrapper`가 `head` kwarg를 그렇게 다루는 것과 동일한 패턴).

### 3.7. config와 canonical 문서

`experiments/configs.py`에 다음 후보를 추가한다.

```python
{"method": "det", "model": "fasterrcnn_resnet50_fpn", "batch_size": 4, "max_epochs": 5, "backbone": "", "head": "box"},
{"method": "det", "model": "retinanet_resnet50_fpn", "batch_size": 4, "max_epochs": 5, "backbone": "", "head": "box"},
{"method": "det", "model": "ssd300_vgg16", "batch_size": 4, "max_epochs": 5, "backbone": "", "head": "box"},
```

`docs/architecture/model-assembly.md` 7.2절에는 `TorchSegModel` 단락과 같은 형식으로
`TorchDetModel`의 조립 경계, family별 label offset과 classifier 교체 방식, valid loss가 항상 0으로
남는 known limitation을 기록한다. `docs/references/backbones.md`의 세 weight row는 "프로젝트가
사용 계획" 열에 `TorchDetModel` 연결과 classifier 교체 방식을 반영한다.

## 4. 완료 기준

이 plan은 다음 조건을 만족하면 `Done`으로 본다.

- `docs/architecture/model-assembly.md` 7.2절에 `TorchDetModel`의 조립 경계와 family별 label
  offset이 명시된다.
- `docs/references/backbones.md`의 Faster R-CNN, RetinaNet, SSD300 weight row에 `TorchDetModel`
  연결 계획이 반영된다.
- `TorchDetModel(model=<3개 중 하나>)`가 `pretrained` local checkpoint를 strict load하고 classifier
  교체까지 완료한다. `TorchDetModel(model="unknown")`은 지원 목록을 포함한 `ValueError`를
  발생시킨다.
- `TorchDetModel.forward`가 train mode에서 native loss dict를, eval mode에서 image당
  `{"boxes","labels","scores"}` list를 반환한다.
- `TorchDetPreprocessor`/`TorchDetPostprocessor`가 각 family의 `label_offset`에 맞는 target/decode를
  생성한다.
- `TorchDetWrapper(model=<3개 중 하나>, device="cpu")`가 2-sample smoke `train_step`과 `eval_step`을
  shape/타입 오류 없이 완료하고, `train_step`이 반환하는 loss 이름이 family의 native loss key와
  일치한다.
- `src/core/factory.py::get_wrapper("det", model=<torchdet model name>)`이 `TorchDetWrapper`를,
  `model=None`이면 기존 `DetWrapper`를 반환한다.
- `experiments/configs.py`에 torchdet whole-model det config 3개가 추가된다.
- `docs/plans/0013-det-torchdet-model-plan.md` 상태가 `Approved`에서 `Done`으로 갱신된다.

## 5. 검증

구현 후 다음 순서로 검증한다.

```bash
conda activate pytorch_env
python -c "import torch; from src.models.det.model import TorchDetModel, SUPPORTED_TORCHDET_MODELS; \
for m in SUPPORTED_TORCHDET_MODELS: \
    net = TorchDetModel(model=m); net.eval(); \
    out = net([torch.rand(3,224,224)]); \
    print(m, out[0].keys(), net.label_offset)"
python -c "from src.models.det.model import TorchDetModel; \
try: TorchDetModel(model='unknown')\nexcept ValueError as e: print('OK:', e)"
python -c "from src.core.factory import get_wrapper; \
w = get_wrapper('det', model='fasterrcnn_resnet50_fpn', device='cpu'); \
print(type(w).__name__)"
python -c "from src.core.factory import get_wrapper; \
w = get_wrapper('det', backbone='custom', head='box', device='cpu'); \
print(type(w).__name__)"
python scripts/train.py --method det --model fasterrcnn_resnet50_fpn --backbone '' --head box --device cpu \
  --train_size 2 --valid_size 2 --batch_size 2 --max_epochs 1 --patience 1 --num_workers 0 \
  --output_dir /tmp/det_fasterrcnn_resnet50_fpn_smoke
python scripts/train.py --method det --model retinanet_resnet50_fpn --backbone '' --head box --device cpu \
  --train_size 2 --valid_size 2 --batch_size 2 --max_epochs 1 --patience 1 --num_workers 0 \
  --output_dir /tmp/det_retinanet_resnet50_fpn_smoke
python scripts/train.py --method det --model ssd300_vgg16 --backbone '' --head box --device cpu \
  --train_size 2 --valid_size 2 --batch_size 2 --max_epochs 1 --patience 1 --num_workers 0 \
  --output_dir /tmp/det_ssd300_vgg16_smoke
```

검증 결과에서는 `TorchDetModel`의 train/eval output 형태, `ValueError`, factory dispatch, 3개
whole-model variant의 smoke train 성공 여부, train loss 이름이 family별 native loss key와
일치하는지, valid loss column이 0으로 남고 `iou` metric만 유효한 값을 갖는지 확인한다. 검증 후
`/tmp/det_*_smoke` 산출물은 삭제한다.
