# TorchSegModel whole segmentation model 추가

다음 표는 이 plan의 상태와 적용 범위를 정리한다.

| 항목 | 값 |
|---|---|
| 상태 | Done |
| 작성일 | 2026-07-18 |
| 적용 범위 | `docs/architecture/model-assembly.md`, `docs/references/backbones.md`, `experiments/configs.py`, `experiments/run.py`, `scripts/config.py`, `src/models/seg/model.py`, `src/models/seg/wrapper.py` |
| 관련 문서 | [docs/architecture/model-assembly.md](../architecture/model-assembly.md), [docs/references/backbones.md](../references/backbones.md), [docs/plans/0009-seg-unet-backbone-plan.md](0009-seg-unet-backbone-plan.md) |

## 1. 목적과 배경

현재 `seg` 구현은 `SegModel` 하나가 stage-returning backbone, `CNNBackboneAdapter`, `SegDecoder`,
`MaskHead`를 조립하는 Category B composable model이다. 이 구조는 U-Net additive skip decoder와
backbone별 stage 비교에 적합하지만, torchvision의 FCN, DeepLabV3, LR-ASPP처럼 encoder, decoder와
segmentation head가 package 내부에서 이미 결합된 whole segmentation model과는 조립 경계가 다르다.

canonical 문서 2.1절과 7절은 Category C를 external whole model로 분리하고, 이전 `torchseg` 이름을
`seg`, `usage=whole_model` variant로 해석한다. 따라서 torchvision segmentation model을 현재
`SegModel`의 조건 분기로 합치면 `FeatureSpec`, `stage_channels`, `SegDecoder`, `MaskHead` 계약이
없는 모델까지 같은 class 안에 섞이게 된다. 이번 plan은 `SegModel`은 composable U-Net 계열로 유지하고,
같은 `src/models/seg/model.py` 안에 `TorchSegModel` class를 별도로 추가해 Category C whole-model
variant를 명확하게 표현한다.

로컬 `torchvision 0.20.1+cu121`에서 직접 확인한 결과, `fcn_resnet50`, `deeplabv3_resnet50`,
`deeplabv3_mobilenet_v3_large`, `lraspp_mobilenet_v3_large`는 `num_classes=1` 생성 시
`model(images)["out"]`으로 `(B, 1, H, W)` mask logits를 반환한다. 기존 `SegPreprocessor`,
`BCELoss`, `DiceLoss`, `SegPostprocessor`는 이 raw output contract를 그대로 사용할 수 있으므로
wrapper와 training loop는 최소 변경으로 재사용한다.

## 2. 범위

이번 plan에 포함하는 항목은 다음과 같다.

- `docs/architecture/model-assembly.md`의 Category C segmentation 설명에 `TorchSegModel`을 추가하고,
  `SegModel`과 `TorchSegModel`의 조립 경계를 분리해 기록한다.
- `docs/references/backbones.md`의 FCN, DeepLabV3, LR-ASPP COCO segmentation weight row에
  `TorchSegModel` 연결 계획과 binary mask head 교체 방식을 반영한다.
- `src/models/seg/model.py`에 `TorchSegModel` class와 `SUPPORTED_TORCHSEG_MODELS`,
  `TORCHSEG_WEIGHTS` catalog를 추가한다.
- `TorchSegModel`은 torchvision segmentation model의 native `OrderedDict` output에서 `"out"` logits만
  반환하고 `self.mask_stride = 1`을 노출한다.
- COCO pretrained whole-model weight는 21-class torchvision model을 만든 뒤 local checkpoint를 strict
  load하고, 마지막 classifier layer를 1-channel binary mask head로 교체하는 방식으로 사용한다.
  LR-ASPP는 auxiliary classifier가 없으므로 `aux_loss` 없이 strict load한다.
- `pretrained=False`에서는 `num_classes=1`, `weights=None`, `weights_backbone=None`으로 바로 생성해
  smoke test와 구조 검증을 빠르게 수행할 수 있게 한다.
- `src/models/seg/wrapper.py`는 `model` kwarg를 받아 `model in ("unet", None)`이면 기존 `SegModel`,
  `model in SUPPORTED_TORCHSEG_MODELS`이면 `TorchSegModel`을 선택한다.
- `SegWrapper` optimizer parameter group은 기존 `SegModel`의 `extractor` 존재 여부에 의존하지 않게
  보강한다. `TorchSegModel`은 전체 model parameter에 단일 learning rate를 적용한다.
- `scripts/config.py::get_wrapper_kwargs`가 `model` 값을 wrapper로 전달하도록 확장한다.
- `experiments/run.py::PASS_KEYS`에 `model`을 추가해 batch config에서 `TorchSegModel` variant를
  실행할 수 있게 한다.
- `experiments/configs.py`에 torchseg whole-model config를 추가한다. config는 `method="seg"`,
  `model=<torchseg model name>`, `backbone=""`, `head="mask"` 형태를 기본 후보로 사용한다.

이번 plan에서 제외하는 항목은 다음과 같다.

- torchvision segmentation model의 internal auxiliary loss를 trainer loss로 함께 사용하는 것. 이번
  plan은 공통 `BCELoss`와 `DiceLoss`만 사용한다.
- COCO 21-class semantic output을 panel class로 직접 mapping하는 것. panel class가 COCO에 없으므로
  마지막 classifier는 binary mask head로 교체하고 project target으로 fine-tuning한다.
- non-torchvision external segmentation repository를 추가하는 것. 이들은 별도 adapter와 dependency
  검토가 필요하므로 후속 Category C plan으로 남긴다.
- `SegPostprocessor`의 four-side fitting 개선, threshold ablation, contour approximation 비교.
  whole-model variant도 기존 postprocess를 사용하고, postprocess 비교는 별도 plan에서 수행한다.

## 3. 구현 계획

### 3.1. model 선택 계약

`seg` method의 `model` 값은 segmentation architecture family 선택자로 사용한다. 초기 계약은 다음과
같다.

| `model` 값 | class | 의미 |
|---|---|---|
| `None`, `unet` | `SegModel` | stage-returning backbone과 project U-Net decoder 조립 |
| `fcn_resnet50` | `TorchSegModel` | torchvision FCN ResNet-50 whole segmentation model |
| `deeplabv3_resnet50` | `TorchSegModel` | torchvision DeepLabV3 ResNet-50 whole segmentation model |
| `deeplabv3_mobilenet_v3_large` | `TorchSegModel` | torchvision DeepLabV3 MobileNetV3-Large whole segmentation model |
| `lraspp_mobilenet_v3_large` | `TorchSegModel` | torchvision LR-ASPP MobileNetV3-Large whole segmentation model |

`backbone`은 기존 `SegModel`에서 backbone 선택자로 유지한다. `TorchSegModel` config에서는 output path와
experiment name을 명확하게 하기 위해 `model`을 실제 architecture 이름으로 채우고, `backbone`은 빈
문자열 또는 생략을 허용한다.

### 3.2. `src/models/seg/model.py` 확장

`TorchSegModel`은 `BaseModel`을 상속하고 다음 책임만 가진다.

- 지원 model name을 검증한다.
- torchvision segmentation builder를 호출한다.
- local COCO checkpoint를 load한다.
- 마지막 classifier를 1-channel binary mask head로 교체한다.
- `forward(images)`에서 native output의 `"out"` tensor만 반환한다.

초기 catalog는 다음과 같다.

```python
TORCHSEG_WEIGHTS = {
    "fcn_resnet50": "/mnt/d/backbones/fcn_resnet50_coco-1167a1af.pth",
    "deeplabv3_resnet50": "/mnt/d/backbones/deeplabv3_resnet50_coco-cd0a2569.pth",
    "deeplabv3_mobilenet_v3_large": "/mnt/d/backbones/deeplabv3_mobilenet_v3_large-fc3c493d.pth",
    "lraspp_mobilenet_v3_large": "/mnt/d/backbones/lraspp_mobilenet_v3_large-d234d4ea.pth",
}
```

`fcn_resnet50`, `deeplabv3_resnet50`, `deeplabv3_mobilenet_v3_large`는 `aux_loss=True`로 21-class
model을 만들고 classifier와 aux classifier의 마지막 `Conv2d`를 교체한다.
`lraspp_mobilenet_v3_large`는 auxiliary classifier가 없으므로 `aux_loss` 없이 21-class model을 만들고
`LRASPPHead`의 `low_classifier`와 `high_classifier`를 1-channel `Conv2d`로 교체한다. 교체 함수는
private helper로 두어 class 본문을 짧게 유지한다.

### 3.3. `src/models/seg/wrapper.py` 확장

`SegWrapper.__init__`에 `model=None` kwarg를 추가한다. 선택 규칙은 다음과 같다.

```python
if model in (None, "unet"):
    net = SegModel(in_channels=in_channels, backbone=backbone)
elif model in SUPPORTED_TORCHSEG_MODELS:
    net = TorchSegModel(model=model)
else:
    raise ValueError(...)
```

`SegPreprocessor(image_size // net.mask_stride)`와 `SegPostprocessor`는 그대로 사용한다.
`TorchSegModel.mask_stride`는 1이므로 mask target은 입력 resize 해상도와 같은 크기를 가진다.

optimizer는 model 종류에 따라 나눈다. `SegModel`은 기존처럼 extractor와 head parameter group을
분리하고, `TorchSegModel`은 전체 parameter에 단일 `AdamW` group을 적용한다.

### 3.4. CLI와 experiment config 연결

`scripts/config.py::get_wrapper_kwargs`는 `args.model`이 비어 있지 않으면 `kwargs["model"]`을 추가한다.
`experiments/run.py::PASS_KEYS`에는 `"model"`을 추가한다.

`experiments/configs.py`에는 다음 후보를 추가한다.

```python
{"method": "seg", "model": "fcn_resnet50", "batch_size": 4, "max_epochs": 5, "backbone": "", "head": "mask"},
{"method": "seg", "model": "deeplabv3_resnet50", "batch_size": 4, "max_epochs": 5, "backbone": "", "head": "mask"},
{"method": "seg", "model": "deeplabv3_mobilenet_v3_large", "batch_size": 4, "max_epochs": 5, "backbone": "", "head": "mask"},
{"method": "seg", "model": "lraspp_mobilenet_v3_large", "batch_size": 4, "max_epochs": 5, "backbone": "", "head": "mask"},
```

`get_model_name`은 이미 `model` override를 우선 사용하므로 output path는
`outputs/<dataset>/seg/<model>/<exp_name>/` 규칙을 만족한다. `get_experiment`는 기존 형식을 유지하되,
필요하면 후속 plan에서 `model`을 포함하도록 개선한다.

## 4. 완료 기준

이 plan은 다음 조건을 만족하면 `Done`으로 본다.

- `docs/architecture/model-assembly.md`에 `SegModel`과 `TorchSegModel`의 Category B, Category C 경계가
  명시된다.
- `docs/references/backbones.md`의 FCN, DeepLabV3, LR-ASPP weight row에 `TorchSegModel` 사용 계획이
  반영된다.
- `TorchSegModel(model=<4개 지원 model 중 하나>, pretrained=False)`가 `(B, 1, H, W)` raw mask logits를
  반환하고 `mask_stride == 1`을 가진다.
- `TorchSegModel(pretrained=True)`가 local COCO checkpoint를 load한 뒤 binary classifier 교체까지
  완료한다.
- `TorchSegModel(model="unknown")`은 지원 목록을 포함한 `ValueError`를 발생시킨다.
- `SegWrapper(model="unet", backbone=<기존 지원 backbone>)`는 기존 `SegModel` 경로를 유지한다.
- `SegWrapper(model=<torchseg model name>, backbone="", device="cpu")`는 2-sample smoke
  `train_step`과 `eval_step`을 shape 오류 없이 완료한다.
- `scripts/config.py::get_wrapper_kwargs`와 `experiments/run.py::PASS_KEYS`가 `model` 값을 전달한다.
- `experiments/configs.py`에 torchseg whole-model seg config 4개가 추가된다.
- `docs/plans/0010-torchseg-model-plan.md` 상태가 `Approved`에서 `Done`으로 갱신된다.

## 5. 검증

구현 후 다음 순서로 검증한다.

```bash
conda activate pytorch_env
python -c "import torch; from src.models.seg.model import TorchSegModel; \
for m in ['fcn_resnet50', 'deeplabv3_resnet50', 'deeplabv3_mobilenet_v3_large', 'lraspp_mobilenet_v3_large']: \
    net = TorchSegModel(model=m, pretrained=False); out = net(torch.zeros(1,3,224,224)); \
    print(m, out.shape, net.mask_stride)"
python -c "import torch; from src.models.seg.model import TorchSegModel; \
for m in ['fcn_resnet50', 'deeplabv3_resnet50', 'deeplabv3_mobilenet_v3_large', 'lraspp_mobilenet_v3_large']: \
    net = TorchSegModel(model=m, pretrained=True); out = net(torch.zeros(1,3,224,224)); \
    print(m, out.shape)"
python -c "from src.models.seg.model import TorchSegModel; \
try: TorchSegModel(model='unknown')\nexcept ValueError as e: print('OK:', e)"
python -c "from scripts.config import parse_args, get_wrapper_kwargs; \
args = parse_args(); args.model = 'fcn_resnet50'; print(get_wrapper_kwargs(args))"
python -c "from experiments.configs import CONFIGS; \
print([c.get('model') for c in CONFIGS if c['method'] == 'seg' and c.get('model')])"
python scripts/train.py --method seg --model fcn_resnet50 --backbone '' --head mask --device cpu \
  --train_size 2 --valid_size 2 --batch_size 2 --max_epochs 1 --patience 1 --num_workers 0 \
  --output_dir /tmp/seg_fcn_resnet50_smoke
python scripts/train.py --method seg --model deeplabv3_resnet50 --backbone '' --head mask --device cpu \
  --train_size 2 --valid_size 2 --batch_size 2 --max_epochs 1 --patience 1 --num_workers 0 \
  --output_dir /tmp/seg_deeplabv3_resnet50_smoke
python scripts/train.py --method seg --model deeplabv3_mobilenet_v3_large --backbone '' --head mask --device cpu \
  --train_size 2 --valid_size 2 --batch_size 2 --max_epochs 1 --patience 1 --num_workers 0 \
  --output_dir /tmp/seg_deeplabv3_mobilenet_v3_large_smoke
python scripts/train.py --method seg --model lraspp_mobilenet_v3_large --backbone '' --head mask --device cpu \
  --train_size 2 --valid_size 2 --batch_size 2 --max_epochs 1 --patience 1 --num_workers 0 \
  --output_dir /tmp/seg_lraspp_mobilenet_v3_large_smoke
```

검증 결과에서는 `TorchSegModel`의 output shape, local pretrained load와 binary classifier 교체,
지원하지 않는 model에서의 `ValueError`, CLI `model` 전달, config 목록, 4개 whole-model variant의
smoke train 성공 여부를 확인한다. DeepLab 계열의 ASPP BatchNorm 경로가 train mode에서 batch size 1을
허용하지 않으므로 smoke train은 batch size 2로 수행한다. 검증 후 `/tmp/seg_*_smoke` 산출물은 삭제한다.
