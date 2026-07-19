# YoloDetModel whole detection model 추가

다음 표는 이 plan의 상태와 적용 범위를 정리한다.

| 항목 | 값 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-19 |
| 적용 범위 | `docs/architecture/model-assembly.md`, `docs/references/backbones.md`, `experiments/configs.py`, `src/core/factory.py`, `src/models/det/model.py`, `src/models/det/wrapper.py`, `src/models/det/yolo_preprocessor.py`(신규), `src/models/det/yolo_postprocessor.py`(신규), `pytorch_env`(신규 dependency: `ultralytics --no-deps` 및 하위 패키지, 기존 `torch 2.5.1+cu121`/`torchvision 0.20.1+cu121` CUDA 빌드 보존) |
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
- (해소됨) validation(eval) loop의 native loss 재현 여부. 3.0절 검증 결과 `model.eval()` 상태의
  raw dict(`out[1]`)가 train-mode와 동일한 key 구성이라 `model.loss(batch, preds=out[1])`을 그대로
  재사용할 수 있음을 확인했다. 3.5절에 반영되어 valid loss도 채운다.

## 3. 구현 계획

### 3.0. 사전 조사와 환경 준비

2026-07-19 확인 결과 `pytorch_env`에는 `ultralytics`가 설치되어 있지 않고, `torch 2.5.1+cu121`,
`torchvision 0.20.1+cu121`이 CUDA 지원 빌드로 설치되어 있다. `/mnt/d/backbones/yolov8n.pt`(6.2M)는
로컬에 존재하며 SHA-256(`f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`)이
`docs/references/backbones.md`의 등록값과 일치한다. upstream 원본은
`https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt`(release tag `v8.3.0`)다.

`pip install ultralytics`를 그대로 실행하면 pip가 `ultralytics`의 요구 버전에 맞춰 `torch`/
`torchvision`을 재설치하려 시도해 기존 CUDA 12.1 빌드가 CPU-only 빌드로 덮어써질 위험이 있다.
설치는 `--no-deps`로 `ultralytics` 본체만 받고, 학습에 필요한 핵심 하위 의존성만 개별 설치한다.

```bash
conda activate pytorch_env
pip install ultralytics --no-deps
pip install "ultralytics-thop>=2.0.0" pandas seaborn tqdm psutil py-cpuinfo
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
python -c "import ultralytics; print(ultralytics.__version__)"
```

설치 직후 위 `torch`/`torchvision` 버전과 `torch.cuda.is_available()`이 설치 전 값(`2.5.1+cu121`,
`0.20.1+cu121`, `True`)과 같은지 반드시 재확인한다. 값이 바뀌었다면 `ultralytics`가 의존성을
재설치한 것이므로 `pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url
https://download.pytorch.org/whl/cu121`로 원복한 뒤 원인을 다시 조사한다.

가중치는 이미 `/mnt/d/backbones/yolov8n.pt`에 있으므로 이번 plan에서는 별도로 다운로드하지 않는다.
다른 PC에서 재현할 때는 다음으로 받고 SHA-256을 비교한다.

```bash
curl -L --fail --silent --show-error -o yolov8n.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
sha256sum yolov8n.pt
# f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36 와 일치해야 한다.
```

환경 준비 후 `pytorch_env`에서 `ultralytics 8.4.101`을 설치하고 실제 API를 직접 확인했다. 확인
결과는 다음과 같으며, 아래 3.1-3.6절은 이 결과를 반영해 갱신되었다.

**checkpoint 구조.** `torch.load(path, map_location="cpu", weights_only=False)`는 `"model"` key에
`ultralytics.nn.tasks.DetectionModel`(pickled `nn.Module`, 원본 `nc=80`)을 담은 dict를 반환한다.
`weights_only=True`는 pickled 객체 특성상 쓸 수 없다. 로컬 신뢰 파일이므로
`weights_only=False`를 사용한다.

**classifier(head) 교체.** `model.model[-1]`이 `ultralytics.nn.modules.head.Detect` module이다.
class 개수는 `Detect.cv3`(classification branch, scale별 `nn.Sequential`)의 마지막 층인
`nn.Conv2d(in_ch, nc, 1)`에서만 결정되고, `Detect.cv2`(box regression branch, DFL
`reg_max=16` 기준 `4*reg_max` 채널)는 class 개수와 무관해 교체하지 않는다. 교체 절차는 다음과
같다.

```python
import torch.nn as nn

detect = model.model[-1]
for seq in detect.cv3:
    in_ch = seq[-1].in_channels
    seq[-1] = nn.Conv2d(in_ch, NUM_CORNER_CLASSES, kernel_size=1)
detect.nc = NUM_CORNER_CLASSES
detect.no = NUM_CORNER_CLASSES + detect.reg_max * 4
model.nc = NUM_CORNER_CLASSES
model.names = {i: "corner%d" % i for i in range(NUM_CORNER_CLASSES)}
model.yaml["nc"] = NUM_CORNER_CLASSES
```

**loss 준비.** `v8DetectionLoss`는 `DetectionModel.init_criterion()`이 첫 `.loss()` 호출 시점에
lazy하게 생성해 `model.criterion`에 캐싱하므로, 이 project는 `v8DetectionLoss`를 직접 import하지
않고 `model.loss(batch, preds=...)`만 호출한다. `v8DetectionLoss.__init__`은 `model.args`를
`self.hyp`로 읽어 `.box`(기본 7.5)/`.cls`(기본 0.5)/`.dfl`(기본 1.5) gain을 얻는데, checkpoint에서
바로 꺼낸 `model`은 이 속성이 없다. 첫 `.loss()` 호출 전에 다음을 설정해야 한다.

```python
from ultralytics.cfg import get_cfg

model.args = get_cfg()
```

**train-mode forward.** `model.train()` 상태의 `model(images)`는 3-scale feature map list가 아니라
dict를 직접 반환한다.

```python
out = model(images)
# out["boxes"]:  (B, 4*reg_max, num_anchors) = (B, 64, A)  raw box regression, undecoded
# out["scores"]: (B, nc, num_anchors) = (B, 4, A)           raw class logits
# out["feats"]:  [3개 raw scale feature map]
```

**loss와 backward.** `DetectionModel.loss(self, batch, preds=None)`는 `preds`가 없으면 내부에서
`self.forward(batch["img"])`를 호출하므로, 이미 계산한 `out`을 `preds=out`으로 재사용할 수 있다.
`batch` dict의 key/shape는 다음과 같다.

```python
batch = {
    "img": images,               # (B, 3, H, W)
    "batch_idx": Tensor(n,),     # 각 corner가 속한 sample index (0 ~ B-1)
    "cls": Tensor(n,),           # 각 corner의 class id (0 ~ NUM_CORNER_CLASSES-1), float
    "bboxes": Tensor(n, 4),      # normalized center-xywh pseudo-box (cx, cy, w, h)
}
loss, loss_detach = model.loss(batch, preds=out)
# loss: shape (3,) = [box, cls, dfl], 이미 batch_size로 scale됨
loss.sum().backward()  # 새 head weight까지 gradient가 흐르는 것을 확인
```

**eval-mode forward와 NMS decode.** `model.eval()` 상태의 `model(images)`는 2-tuple을 반환한다.

```python
model.eval()
with torch.no_grad():
    out = model(images)
decoded = out[0]  # (B, 4+nc, num_anchors), NMS 입력 가능한 decoded tensor
# out[1]은 train-mode와 동일한 raw dict
```

box 좌표를 얻으려면 별도 NMS 호출이 필요하다. 이 project가 가정한 `ultralytics.utils.ops.non_max_suppression`은
설치된 8.4.101 버전에는 없고, 실제로는 `ultralytics.utils.nms.non_max_suppression`으로 이동했다.

```python
from ultralytics.utils.nms import non_max_suppression

results = non_max_suppression(decoded, conf_thres=0.001, iou_thres=0.5, max_det=10, nc=4)
# results: 길이 B인 list, 각 원소 shape (max_det, 6) = [x1, y1, x2, y2, score, class_id]
# 좌표는 xyxy pixel 단위(입력 image 크기 기준)
```

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
- `YOLODET_WEIGHTS[model]` checkpoint에서 `ultralytics.nn.tasks.DetectionModel`을 `weights_only=False`로
  불러오고, 3.0절에서 확인한 `Detect.cv3` 마지막 층 교체 절차로 classifier(head)를
  `NUM_CORNER_CLASSES=4`로 바꾼다. 생성 시점에 `model.args = get_cfg()`를 설정해 첫 `.loss()` 호출을
  준비해 둔다.
- `forward(images)`는 내부 `ultralytics.nn.tasks.DetectionModel.forward`를 그대로 호출해 반환한다.
  `self.net.training`이 `True`면 dict(`{"boxes", "scores", "feats"}`, raw/undecoded)를, `False`면
  2-tuple(`(decoded (B,4+nc,A), 위와 같은 dict)`)을 그대로 돌려준다. `BaseModel.forward(images)`
  단일 인자 계약과 호환되므로 `targets` 인자는 받지 않는다. loss 계산은 `self.net.loss(batch,
  preds=out)`로 `YoloDetWrapper`가 별도로 수행한다.

### 3.3. `src/models/det/yolo_preprocessor.py`(신규)

`(N,4,2)` 정규화 corners를 Ultralytics 학습 batch dict로 변환한다. corner는 실제 넓이를 가진
object가 아니므로 `TorchDetPreprocessor`처럼 고정 pseudo-box 크기(`box_size`, normalized 기준)를
사용해 각 corner를 중심으로 하는 작은 box를 만든다. 출력 dict는 3.0절에서 확인한 `v8DetectionLoss`
입력 계약을 그대로 따른다.

```python
{
    "img": images,             # (B, 3, H, W), forward 호출에 재사용
    "batch_idx": Tensor(n,),   # sample 내 corner를 flatten한 뒤 원래 sample index
    "cls": Tensor(n,),         # corner 순서(0~3)를 그대로 class id로 사용, float
    "bboxes": Tensor(n, 4),    # normalized center-xywh, w=h=box_size 고정
}
```

`img`는 preprocessor가 아니라 `YoloDetWrapper.train_step`이 채운다(preprocessor는 corners만 받는
`BasePreprocessor.__call__(corners)` 계약을 유지한다).

### 3.4. `src/models/det/yolo_postprocessor.py`(신규)

`YoloDetModel.forward`의 eval-mode 2-tuple 중 `decoded`(`out[0]`)를 입력으로 받아
`ultralytics.utils.nms.non_max_suppression(decoded, conf_thres=0.001, iou_thres=0.5, max_det=?,
nc=4)`를 호출하고, 그 결과(길이 B list, 각 원소 `(max_det, 6)` = `[x1,y1,x2,y2,score,class_id]`
xyxy pixel)를 공통 `(N,4,2)` corners contract로 decode한다. `TorchDetPostprocessor`와 동일하게
class별 최고 score box의 중심점을 사용하고, 예측이 없는 class는 image 중심(0.5, 0.5)로 fallback한다.
pixel 좌표를 `image_size`로 나눠 정규화한다.

### 3.5. `src/models/det/wrapper.py`에 `YoloDetWrapper` 추가

`TorchDetWrapper`와 동일한 구조로 `train_step`/`eval_step`/`predict_step`을 override한다.

- `train_step`: `YoloDetPreprocessor`로 `batch_idx`/`cls`/`bboxes`를 만들고 `img`를 채운 뒤
  `self.model.net(images)`(train-mode forward)의 결과를 `preds`로 `self.model.net.loss(batch,
  preds=out)`에 넘긴다. 반환되는 `loss`는 shape `(3,)` = `[box, cls, dfl]`이며, `loss_names =
  self.model.net.criterion.loss_names`(구현 시점에 실제 속성 이름을 재확인)로 각 원소에 이름을
  붙여 `self.losses`(빈 dict에서 시작, `TorchDetWrapper`와 동일 패턴)를 동적으로 채운다.
  `loss.sum().backward()` 후 `optimizer.step()`을 호출한다. `head`/`backbone` kwarg는 CLI 호환을
  위해서만 받고 사용하지 않는다.
- `eval_step`: `self.model.net`을 eval mode로 전환해 `(decoded, raw_dict)`를 얻고, `raw_dict`를
  `preds=`로 `self.model.net.loss(batch, preds=raw_dict)`에 넘겨 valid loss도 채운다(3.0절에서
  eval-mode raw dict가 train-mode와 동일한 key 구성임을 확인했으므로 `model.loss`를 그대로 재사용할
  수 있다). `decoded`는 `YoloDetPostprocessor`로 corners를 decode해 `PolygonIoU` metric을 계산한다.
- `predict_step`: `eval_step`과 동일한 eval-mode forward 후 `YoloDetPostprocessor`만 적용한다.
- optimizer는 `TorchDetWrapper`와 동일하게 전체 parameter에 단일 `AdamW` group을 적용한다.

### 3.5.1. gradient freeze bug 수정과 `box_size` 조정

구현 완료 후 실제 production config(`{"method": "det", "model": "yolov8n", "batch_size": 4,
"max_epochs": 5, "backbone": "", "head": "box"}`)로 학습하면 train iou가 모든 epoch에서 `0.000`에
머무르고 `cls` loss가 수십에서 수백 단위로 폭증하며 실질 성능이 거의 나오지 않는 문제가 발견됐다.
비교 기준인 `TorchDetModel`(`fasterrcnn_resnet50_fpn`)은 동일 조건에서 valid iou 0.9 이상을
안정적으로 달성해, 문제가 데이터나 학습 루프가 아니라 `YoloDetModel` 쪽에 있음을 시사했다.

원인은 Ultralytics가 배포한 `yolov8n.pt` checkpoint의 모든 parameter가 `requires_grad=False`로
저장돼 있다는 점이었다(추론 전용 배포를 가정한 값). `YoloDetModel.build_model`이 `ckpt["model"]`을
그대로 쓰면서 classifier(head)만 새로 만든 `nn.Conv2d`로 교체했으므로 새 head는 학습됐지만, backbone과
neck을 포함한 나머지 전체 network는 gradient가 전혀 흐르지 않는 상태로 5 epoch 내내 고정돼 있었다.
`cls` loss 폭증과 iou=0 정체는 이 고정된 backbone/neck이 corner 좌표를 전혀 학습하지 못한
직접적인 결과다.

수정은 `net = ckpt["model"].float()` 직후, classifier 교체 이전에 `net.requires_grad_(True)`를
추가해 전체 network를 명시적으로 학습 가능 상태로 되돌리는 것이다. 이 project는 `TorchDetModel`과
동일하게 whole-model을 fine-tuning하는 것을 전제로 하므로, Ultralytics의 배포 전용 기본값을
project 전제에 맞게 덮어써야 한다.

```python
def build_model(self, path):
    if not os.path.exists(path):
        raise FileNotFoundError("Local yolodet weight not found: %s" % path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = ckpt["model"].float()
    # Ultralytics saves inference checkpoints with requires_grad=False on every
    # parameter (deploy-only assumption); this project fine-tunes the whole net.
    net.requires_grad_(True)
    self.replace_classifier(net, NUM_CORNER_CLASSES)
    net.args = get_cfg()
    return net
```

같은 조사 과정에서 `YoloDetWrapper`의 `box_size` 기본값을 `0.1`에서 `0.3`으로 올렸다. corner를
중심으로 한 pseudo-box가 지나치게 작으면 `TaskAlignedAssigner`가 positive anchor로 배정하는 개수가
줄어 `target_scores_sum` 정규화가 불안정해지기 때문이다. 이 조정은 freeze bug 수정의 보조
조치이며 root cause는 아니다.

수정 후 실제 production config로 재학습한 결과 valid iou가 epoch 1의 0.612에서 epoch 5의 0.9439까지
안정적으로 상승했고, `TorchDetModel` 벤치마크(iou > 0.9)와 부합하는 성능을 확인했다. 다만 train
split의 iou는 모든 epoch에서 `0.000`으로 표시되는데, 이는 결함이 아니라
`YoloDetWrapper.train_step`이 학습 속도를 위해 loss 계산만 수행하고 `update_metrics()`를 호출하지
않는 설계 때문이다(`eval_step`만 postprocess와 metric 갱신을 수행한다). 이 동작은 흔한 YOLO 계열
학습 루프 관례와 일치하며 별도 수정 대상이 아니다.

동일 조사에서 `--train_size 40 --valid_size 20`처럼 매우 적은 샘플로 40 epoch를 도는 smoke test를
수행했을 때는 valid iou가 최고 0.065(epoch 25)에 그치고 `cls` loss가 42-45 부근에서 정체돼
수렴하지 못하는 현상이 관찰됐다. 동일한 소량 샘플로 `fasterrcnn_resnet50_fpn`을 10 epoch 학습하면
iou 0.77까지 정상적으로 수렴하므로, 이 정체는 `requires_grad_(True)` 수정과 무관하게 YOLO의
dense-anchor `TaskAlignedAssigner`가 극소량 샘플에서 positive anchor 다양성 부족으로 불안정해지는
현상으로 판단된다. 실제 production 규모(전체 데이터셋)에서는 재현되지 않으므로 이 plan의 범위
밖으로 남긴다.

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
- `YoloDetModel.build_model`이 checkpoint 로딩 직후 `net.requires_grad_(True)`를 호출해 전체
  network가 학습 가능 상태로 fine-tuning된다(3.5.1절).
- production config로 재학습한 valid iou가 `TorchDetModel` 벤치마크(0.9 이상)에 부합한다(3.5.1절).
- 이 문서의 상태가 `Approved`에서 `Done`으로 갱신된다.

## 5. 검증

구현 후 다음 순서로 검증한다.

```bash
conda activate pytorch_env
python -c "import torch; from src.models.det.model import YoloDetModel; \
net = YoloDetModel(model='yolov8n'); net.eval(); \
out = net(torch.rand(2,3,224,224)); print(type(out), out[0].shape)"
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

검증 결과에서는 `YoloDetModel`의 train/eval output 형태(`BaseModel.forward(images)` 단일 인자
계약, eval-mode 2-tuple), `ValueError`, factory dispatch, smoke train 성공 여부, train/eval loss
이름이 `[box, cls, dfl]` 3개인지, `iou` metric이 유효한 값을 갖는지 확인한다. 검증 후
`/tmp/det_yolov8n_smoke` 산출물은 삭제한다.
