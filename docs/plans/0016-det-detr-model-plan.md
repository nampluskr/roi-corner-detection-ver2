# DetrDetModel whole detection model 추가

다음 표는 이 plan의 상태와 적용 범위를 정리한다.

| 항목 | 값 |
| --- | --- |
| 상태 | Deprecated |
| 작성일 | 2026-07-19 |
| 적용 범위 | `docs/architecture/model-assembly.md`, `docs/references/backbones.md`, `experiments/configs.py`, `src/core/factory.py`, `src/models/det/model.py`, `src/models/det/wrapper.py`, `src/models/det/detr_preprocessor.py`(신규), `src/models/det/detr_postprocessor.py`(신규), `src/models/det/detr/`(신규 vendored 서브패키지, 또는 `pytorch_env`(신규 dependency: `transformers`) 중 3.0절 investigation 결과로 확정) |
| 관련 문서 | [docs/architecture/model-assembly.md](../architecture/model-assembly.md), [docs/references/backbones.md](../references/backbones.md), [docs/plans/0011-det-custom-model-plan.md](0011-det-custom-model-plan.md), [docs/plans/0013-det-torchdet-model-plan.md](0013-det-torchdet-model-plan.md), [docs/plans/0014-det-yolo-model-plan.md](0014-det-yolo-model-plan.md) |

이 plan은 폐기한다. Meta 원본 checkpoint strict-load, vendoring, remapping 접근을 폐기하고,
[0017-det-hf-detr-model-plan.md](0017-det-hf-detr-model-plan.md)에서 Hugging Face pretrained DETR
adapter 방식으로 대체한다.

## 1. 목적과 배경

`det` method는 현재 `DetModel`(Category A/B composable), `TorchDetModel`(Category C, torchvision
whole model 3종, plan 0013), `YoloDetModel`(Category C, Ultralytics YOLOv8-Nano, plan 0014) 세
variant를 갖는다. canonical 문서 `model-assembly.md` 7.2절은 Category C external whole-model을
segmentation, detection, set prediction 세 family로 분류하고, `docs/references/backbones.md`
3.3절에는 이미 `detr-r50-e632da11.pth`(DETR ResNet-50, COCO detection, Meta 원본
facebookresearch/detr repo checkpoint)가 조건부 등록되어 있다. plan 0014는 DETR 통합을 명시적으로
범위에서 제외하며 canonical 7.2절 set prediction row에 해당하며 계속 범위 밖으로 남긴다고 기록했다.
이번 plan은 그 후속으로, DETR을 `det`의 네 번째 Category C whole-model variant로 추가한다.

핵심 조사 결과는 다음과 같다(`pytorch_env`에서 직접 확인).

- `transformers` 패키지는 `pytorch_env`에 설치되어 있지 않다(`ModuleNotFoundError`).
- `detr-r50-e632da11.pth`는 top-level이 `{"model": <458개 key state_dict>}` 구조이며, 키 이름이
  `backbone.0.body.*`(torchvision 스타일 ResNet-50 stem/layer1-4), `transformer.encoder.layers.N.*`,
  `transformer.decoder.layers.N.*`(각 6층, `self_attn`/`multihead_attn`/`linear1`/`linear2`/
  `norm1-3` 하위 이름), `transformer.decoder.norm.*`, `class_embed.weight`(92, 256)(91 COCO class +
  background), `bbox_embed.layers.{0,1,2}.*`(3-layer MLP box head), `query_embed.weight`(100, 256)
  (100 object query), `input_proj.weight/bias`(2048->256 1x1 conv)로 구성된다. 이는 원본
  facebookresearch/detr repo의 state_dict 포맷 그대로이며 HuggingFace
  `transformers.DetrForObjectDetection`의 키 이름(`model.backbone.conv_encoder.model.*`,
  `class_labels_classifier.*`, `model.query_position_embeddings.weight` 등)과 다르다.
- `torchvision` 0.20.1+cu121, `torch` 2.5.1+cu121, `ultralytics` 8.4.101, `timm` 1.0.22가 설치되어
  있다. `transformers`는 설치되어 있지 않다.

이 사실이 이번 plan의 핵심 설계 결정을 정한다. checkpoint와 strict하게 호환되는 로더를 만들려면
facebookresearch/detr 원본 아키텍처 코드를 최소 형태로 이 프로젝트 안에 이식(vendoring)하거나,
`transformers`를 신규 설치하고 key remapping을 적용해야 한다. 정확한 방식은 구현 착수 전
investigation(3.0절) 단계에서 두 방식을 실제로 프로토타이핑하고 비교해 확정한다. 이는 plan 0014가
Ultralytics API 세부사항을 investigation 단계에서 실제로 확인하고 가정을 수정했던 것과 동일한
패턴이다. Hungarian matching 기반 target 구성은 기존 `TorchDetPreprocessor`/`YoloDetPreprocessor`와
동일한 고정 크기 pseudo-box 패턴을 재사용한다.

## 2. 범위

포함 항목은 다음과 같다.

- 3.0 investigation: 코드 벤더링과 `transformers` 설치 두 방식을 실제로 검증하고 최종 방식을
  확정한다.
- 확정된 방식에 따라 DETR forward가 가능한 model 코드를 확보하고 `detr-r50-e632da11.pth`를 strict
  load한다.
- `src/models/det/model.py`에 `DETRDET_WEIGHTS`, `SUPPORTED_DETRDET_MODELS` catalog와
  `DetrDetModel(BaseModel)` class를 추가한다. `class_embed`(92-class COCO+background)를 corner
  5-class(4+background) classifier로 교체한다.
- `src/models/det/detr_preprocessor.py`(신규)에 `DetrDetPreprocessor`를 추가한다. `(N,4,2)` 정규화
  corners를 DETR/Hungarian matching이 기대하는 target 형식(`list[{"labels": Tensor(4,), "boxes":
  Tensor(4,4) normalized cxcywh}]`, 고정 크기 pseudo-box)으로 변환한다.
- `src/models/det/detr_postprocessor.py`(신규)에 `DetrDetPostprocessor`를 추가한다. `{"pred_logits":
  (B,100,5), "pred_boxes": (B,100,4)}` 출력에서 클래스별 최고 confidence query를 선택해 공통
  `(N,4,2)` corners contract로 decode한다.
- `src/models/det/wrapper.py`에 `DetrDetWrapper(BaseWrapper)`를 추가한다. `train_step`/`eval_step`/
  `predict_step`을 override해 Hungarian matcher와 `SetCriterion`(class CE + box L1 + GIoU 3항
  weighted sum) 기반 native loss 계산을 구현한다.
- `src/core/factory.py::get_wrapper`의 `det` 분기에 `SUPPORTED_DETRDET_MODELS` 체크를 추가한다.
- `docs/architecture/model-assembly.md` 7.2절에 `TorchDetModel`/`YoloDetModel` 단락과 같은 형식으로
  `DetrDetModel` 조립 경계 설명을 추가한다.
- `docs/references/backbones.md`의 `detr-r50-e632da11.pth` row의 적용 방법과 제약 열을 이번 plan
  링크와 구체적인 classifier 교체 및 query selection 방식으로 갱신한다.
- `experiments/configs.py`에 `detr_resnet50` whole-model `det` config 1개를 주석 처리 상태로
  추가한다.

제외 항목(후속 plan에서 수행)은 다음과 같다.

- `representation="point"` DETR variant. DETR box head는 항상 box 좌표만 native 출력하므로 `head`는
  `TorchDetModel`/`YoloDetModel`과 동일하게 CLI 호환을 위해서만 받고 고정값으로 취급한다.
- Deformable DETR, Conditional DETR 등 DETR 변형 아키텍처. 로컬에 검증된 checkpoint가
  `detr-r50-e632da11.pth` 하나뿐이므로 이번 plan은 이 하나로 한정한다.
- Hungarian matching 비용 함수(class cost, L1 cost, GIoU cost) 가중치 튜닝이나 auxiliary decoder
  loss(`aux_loss`, 6개 decoder layer 각각의 중간 loss) 채택 여부에 대한 상세 ablation. 최소 동작
  가능한 기본값(원본 DETR 논문 기본 가중치: cls 1, bbox 5, giou 2)만 적용하고 세부 튜닝은 범위 밖으로
  남긴다.
- validation(eval) loop에서 Hungarian matching 기반 native loss를 재현할지 여부는 investigation
  결과에 따라 결정한다. matching 자체가 무작위성 없는 결정적 알고리즘이므로 `TorchDetWrapper`처럼
  valid loss를 0으로 두지 않고 `YoloDetWrapper`처럼 채울 수 있는지 3.0절에서 확인 후 반영한다.

## 3. 구현 계획

### 3.0. 사전 조사

구현 착수 전 다음을 실제로 검증하고 그 결과로 이후 절들을 갱신한다.

1. 아키텍처 코드 확보 방식을 결정한다. 두 후보를 프로토타이핑한다.
   - (A) facebookresearch/detr 핵심 모듈(`backbone.py`, `transformer.py`(순수 `nn.Transformer`
     기반, 원본과 동일 sublayer 이름), `position_encoding.py`, `matcher.py`(`HungarianMatcher`),
     `SetCriterion` 등가물, `detr.py`(`DETR` class 조립))을 `src/models/det/detr/` 아래 최소 형태로
     이식하고 `detr-r50-e632da11.pth`의 `ckpt["model"]`을 `strict=True`로 load해 성공하는지
     확인한다.
   - (B) `pytorch_env`에 `transformers`를 설치하고 `DetrForObjectDetection`을 생성한 뒤, 원본
     458개 key를 HF 키 이름으로 remap하는 함수를 작성해 `load_state_dict(strict=True)`가
     성공하는지 확인한다.
   - 두 방식 모두 성공하면 신규 pip dependency가 없고 프로젝트의 코드/라이선스 정책과 일관된 (A)를
     기본으로 채택한다. (A)가 strict load에 실패하면 (B)로 전환하고 그 사유를 이 절에 기록한다.
2. forward 출력 shape를 확인한다. 선택된 구현으로 `model(pixel_values)` 또는 동등 호출이
   `{"pred_logits": (B,100,92), "pred_boxes": (B,100,4)}`(cxcywh, normalized)를 반환하는지
   확인한다.
3. loss API를 확인한다. `HungarianMatcher`/`SetCriterion`(또는 HF `DetrForObjectDetection`이 내부적
   으로 계산하는 loss_dict)의 정확한 호출 시그니처, target 형식(`{"labels": LongTensor(n,), "boxes":
   FloatTensor(n,4)}` per image), 반환되는 개별 loss 항목 이름(`loss_ce`, `loss_bbox`, `loss_giou`
   등)을 확인한다.
4. requires_grad 상태를 확인한다. plan 0014의 3.5.1 사례(Ultralytics 배포 checkpoint가
   `requires_grad=False`로 저장됨)와 동일한 위험이 있는지 `detr-r50-e632da11.pth` load 직후
   `next(net.parameters()).requires_grad`로 확인하고, `False`면 `net.requires_grad_(True)`를
   추가한다.
5. train/eval 출력 대칭성을 확인한다. torchvision detection(비대칭, `TorchDetWrapper`가 override로
   흡수)과 달리 DETR forward는 train/eval 모드 구분 없이 항상 동일한 `{"pred_logits",
   "pred_boxes"}` 구조를 반환하는지 확인한다. 대칭이면 `eval_step`에서도 native loss(Hungarian
   matching 기반)를 재현할 수 있으므로 2절 제외 항목의 마지막 항목을 이에 맞게 갱신한다.

### 3.1. family catalog와 model 선택 계약

```python
DETRDET_WEIGHTS = {
    "detr_resnet50": "/mnt/d/backbones/detr-r50-e632da11.pth",
}
SUPPORTED_DETRDET_MODELS = tuple(DETRDET_WEIGHTS.keys())
```

다음 표는 `model` 값에 따른 class 선택 계약을 정리한다.

| model 값 | class | 비고 |
| --- | --- | --- |
| `None` | `DetModel` | 기존 custom/backbone 조합 grid 기반 model |
| `fasterrcnn_resnet50_fpn`, `retinanet_resnet50_fpn`, `ssd300_vgg16` | `TorchDetModel` | plan 0013 |
| `yolov8n` | `YoloDetModel` | plan 0014 |
| `detr_resnet50` | `DetrDetModel` | 이번 plan, COCO 92-class(91+background) head를 corner 5-class(4+background) head로 교체 |

### 3.2. `src/models/det/model.py`에 `DetrDetModel` 추가

`BaseModel`을 상속하고 다음 책임만 가진다.

- 지원 model name을 검증하고 `DETRDET_WEIGHTS[model]` checkpoint를 `torch.load(...,
  weights_only=True)`로 strict load한다. 순수 state_dict이므로 Ultralytics의 pickled 객체와 달리
  `weights_only=True` 사용이 가능하다.
- `class_embed`(`nn.Linear(256, 92)`)를 `nn.Linear(256, NUM_CORNER_CLASSES + 1)`(5-class,
  background class 유지)로 교체한다. `bbox_embed`(3-layer MLP)는 class 개수와 무관하므로 그대로
  재사용한다.
- `forward(images)`는 `BaseModel.forward(images)` 단일 인자 계약을 따르고 `{"pred_logits",
  "pred_boxes"}` dict를 그대로 반환한다. 3.0절에서 train/eval 비대칭이 확인되면 이 절을 갱신한다.

### 3.3. `src/models/det/detr_preprocessor.py`(신규)

corner는 실제 넓이를 가진 object가 아니므로 `TorchDetPreprocessor`/`YoloDetPreprocessor`와 동일하게
고정 크기 normalized pseudo-box(`box_size`)를 사용한다.

```python
class DetrDetPreprocessor(BasePreprocessor):
    def __init__(self, box_size=0.1):
        self.box_size = box_size

    def __call__(self, corners):
        targets = []
        for sample in corners:
            n = sample.shape[0]
            wh = torch.full((n, 2), self.box_size, device=corners.device)
            boxes = torch.cat([sample, wh], dim=1)
            labels = torch.arange(n, device=corners.device)
            targets.append({"labels": labels, "boxes": boxes})
        return targets
```

정확한 target key 이름과 좌표계(cxcywh vs xyxy, normalized vs pixel)는 3.0절에서 확정된 loss API에
맞춰 조정한다.

### 3.4. `src/models/det/detr_postprocessor.py`(신규)

`{"pred_logits": (B,100,5), "pred_boxes": (B,100,4)}`에서 각 corner class(0-3)에 대해 softmax
confidence가 가장 높은 query 하나를 선택하고, 그 query의 `pred_boxes`(cxcywh normalized) 중심점을
`(N,4,2)` corners contract로 decode한다. `TorchDetPostprocessor`/`YoloDetPostprocessor`와 동일하게
예측 신뢰도가 임계값 이하이거나 해당 class를 아무 query도 담당하지 않으면 image 중심(0.5, 0.5)로
fallback한다.

### 3.5. `src/models/det/wrapper.py`에 `DetrDetWrapper` 추가

`TorchDetWrapper`/`YoloDetWrapper`와 동일한 생성자 signature 패턴(`backbone=None, head="box",
model=None, box_size=..., image_size=224, optimizer=None, scheduler=None, preprocessor=None,
postprocessor=None, metrics=None, device=None`)을 따른다.

- `train_step`: `DetrDetPreprocessor`로 target을 만들고 `self.model(images)`로 forward한 뒤,
  `HungarianMatcher`로 예측-정답 query를 매칭하고 `SetCriterion`(class CE + box L1 + GIoU, 원본
  기본 가중치 cls=1, bbox=5, giou=2)으로 loss를 계산한다. 개별 loss 항목을 `self.losses`(빈 dict에서
  시작, `TorchDetWrapper`/`YoloDetWrapper`와 동일 패턴)에 동적으로 채운다.
- `eval_step`/`predict_step`: 3.0절에서 확인된 train/eval 대칭성에 따라 동일한 forward 경로를 쓰고,
  `DetrDetPostprocessor`로 corners를 decode해 `PolygonIoU` metric을 계산한다.
- optimizer는 `TorchDetWrapper`/`YoloDetWrapper`와 동일하게 전체 parameter에 단일 `AdamW` group을
  적용한다. differential LR은 도입하지 않는다.

### 3.6. `src/core/factory.py` dispatch

```python
if method == "det":
    from src.models.det.model import (
        SUPPORTED_TORCHDET_MODELS, SUPPORTED_YOLODET_MODELS, SUPPORTED_DETRDET_MODELS,
    )
    if kwargs.get("model") in SUPPORTED_TORCHDET_MODELS:
        from src.models.det.wrapper import TorchDetWrapper
        return TorchDetWrapper(device=device, **kwargs)
    if kwargs.get("model") in SUPPORTED_YOLODET_MODELS:
        from src.models.det.wrapper import YoloDetWrapper
        return YoloDetWrapper(device=device, **kwargs)
    if kwargs.get("model") in SUPPORTED_DETRDET_MODELS:
        from src.models.det.wrapper import DetrDetWrapper
        return DetrDetWrapper(device=device, **kwargs)
    from src.models.det.wrapper import DetWrapper
    return DetWrapper(device=device, **kwargs)
```

### 3.7. config와 canonical 문서 갱신

`experiments/configs.py`에 다음 후보를 주석 처리 상태로 추가한다. 기존 파일의 대부분 det entry가
주석 처리되어 있고 `yolov8n`만 활성화된 관례를 따른다.

```python
# {"method": "det", "model": "detr_resnet50", "batch_size": 4, "max_epochs": 5, "backbone": "", "head": "box"},
```

`docs/architecture/model-assembly.md` 7.2절에는 `TorchDetModel`/`YoloDetModel` 단락과 같은 형식으로
`DetrDetModel`의 조립 경계, checkpoint 이식 방식(vendored 코드 또는 key remapping, 3.0절 결정에
따름), classifier 교체와 query selection 방식을 기록한다. `docs/references/backbones.md`의
`detr-r50-e632da11.pth` row의 적용 방법과 제약 열을 `DetrDetModel` 연결 계획으로 갱신하고 이 plan
문서로 링크한다.

## 4. 완료 기준

다음 항목이 모두 충족되면 이 plan을 `Done`으로 볼 수 있다.

- 3.0절 investigation이 완료되고 아키텍처 코드 확보 방식이 실제 검증 결과에 따라 확정된다(vendoring
  또는 key remapping).
- `docs/architecture/model-assembly.md` 7.2절에 `DetrDetModel` 조립 경계가 명시된다.
- `docs/references/backbones.md`의 `detr-r50-e632da11.pth` row가 갱신된다.
- `DetrDetModel(model="detr_resnet50")`이 checkpoint를 load하고 classifier 교체까지 완료한다.
  `DetrDetModel(model="unknown")`은 지원 목록을 포함한 `ValueError`를 발생시킨다.
- `DetrDetWrapper(model="detr_resnet50", device="cpu")`가 2-sample smoke `train_step`/`eval_step`을
  shape/타입 오류 없이 완료하고, loss 이름이 native `SetCriterion`(또는 동등물) key와 일치한다.
- `src/core/factory.py::get_wrapper("det", model="detr_resnet50")`이 `DetrDetWrapper`를 반환하고,
  기존 `model=yolov8n`/`model=fasterrcnn_resnet50_fpn`/`model=None` 경로는 회귀 없이 그대로
  동작한다.
- `experiments/configs.py`에 `detr_resnet50` config가 추가된다.
- 상태가 `Draft`에서 `Approved`를 거쳐 `Done`으로 갱신된다.

## 5. 검증

구현 후 plan 0013/0014와 동일한 순서로 `python -c` smoke check와 `scripts/train.py --method det
--model detr_resnet50 --train_size 2 --valid_size 2 --max_epochs 1 --device cpu`를 실행해
shape/loss/factory dispatch를 확인하고, `/tmp/det_detr_resnet50_smoke` 산출물은 검증 후 삭제한다.
실행은 `conda activate pytorch_env` 활성화 후 수행한다.
