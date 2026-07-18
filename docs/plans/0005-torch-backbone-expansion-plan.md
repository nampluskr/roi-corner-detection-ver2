# torchvision backbone 확장과 TorchBackbone rename

다음 표는 이 plan의 상태와 적용 범위를 정리한다.

| 항목 | 값 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-18 |
| 적용 범위 | `src/models/backbones/torchvision_backbone.py`, `src/models/reg/model.py`, `scripts/config.py`, `experiments/configs.py` |
| 관련 문서 | [docs/architecture/model-assembly.md](../architecture/model-assembly.md), [docs/references/backbones.md](../references/backbones.md), [docs/plans/0003-reg-backbone-experiments-plan.md](0003-reg-backbone-experiments-plan.md), [docs/plans/0004-reg-head-backbone-matrix-plan.md](0004-reg-head-backbone-matrix-plan.md) |

## 1. 목적과 배경

현재 `reg` method는 project custom backbone과 torchvision ResNet-50만 선택할 수 있다. 사용자는
`torchvision.models`의 ResNet-18, ResNet-34, ResNet-50, EfficientNet, VGG 계열을 `backbone` 인자로
선택 가능하게 확장하고, class 이름을 `TorchvisionBackbone`에서 `TorchBackbone`으로 바꾸기를 원한다.
이번 평가 config에서는 head를 `coord_spatial`로 고정하고 backbone 차이만 비교한다.

`docs/references/backbones.md`에는 이번 범위에 필요한 로컬 ImageNet weight가 이미 기록되어 있다.
로컬 환경의 `torchvision 0.20.1+cu121`에서 `efficientnet_b0`, `vgg16`, `vgg16_bn`, `vgg19`,
`vgg19_bn`의 state dict가 각 builder와 호환됨을 import와 `load_state_dict(strict=False)`로
확인했다. 이번 작업은 canonical architecture를 바꾸지 않고 Category B의 torchvision backbone
선택지를 늘리는 구현 확장이다.

## 2. 범위

이번 plan에 포함하는 backbone id는 다음과 같다.

| backbone id | torchvision builder | local weight |
| --- | --- | --- |
| `resnet18` | `torchvision.models.resnet18` | `/mnt/d/backbones/resnet18-f37072fd.pth` |
| `resnet34` | `torchvision.models.resnet34` | `/mnt/d/backbones/resnet34-b627a593.pth` |
| `resnet50` | `torchvision.models.resnet50` | `/mnt/d/backbones/resnet50-0676ba61.pth` |
| `efficientnet_b0` | `torchvision.models.efficientnet_b0` | `/mnt/d/backbones/efficientnet_b0_rwightman-7f5810bc.pth` |
| `vgg16` | `torchvision.models.vgg16` | `/mnt/d/backbones/vgg16-397923af.pth` |
| `vgg16_bn` | `torchvision.models.vgg16_bn` | `/mnt/d/backbones/vgg16_bn-6c64b313.pth` |
| `vgg19` | `torchvision.models.vgg19` | `/mnt/d/backbones/vgg19-dcbb9e9d.pth` |
| `vgg19_bn` | `torchvision.models.vgg19_bn` | `/mnt/d/backbones/vgg19_bn-c79401a0.pth` |

이번 plan에서 제외하는 항목은 다음과 같다.

- `efficientnet_b5`, `efficientnet_b0_ra`처럼 upstream 또는 builder 호환성이 명확하지 않은 weight
- MobileNet, Wide ResNet, ViT, Swin, DINOv2 확장
- segmentation 또는 detection whole model 연결
- backbone별 custom adapter, decoder, freeze policy, optimizer policy 분리
- `src/models/backbones/torchvision_backbone.py` 파일명 변경

파일명은 import churn을 줄이기 위해 유지하고, class 이름만 `TorchBackbone`으로 변경한다.

## 3. 구현 계획

### 3.1. `TorchBackbone` rename과 compatibility

`src/models/backbones/torchvision_backbone.py`의 public class를 `TorchBackbone`으로 변경한다. 기존
`TorchvisionBackbone` 이름은 이번 작업에서 제거하지 않고 alias로 남긴다.

```python
TorchvisionBackbone = TorchBackbone
```

이 alias는 기존 import를 깨지 않기 위한 compatibility shim이며, 새 코드에서는 `TorchBackbone`만
import한다.

### 3.2. Backbone registry 확장

`BACKBONE_WEIGHTS`와 `BACKBONE_BUILDERS`를 8개 id로 확장한다. `SUPPORTED_BACKBONES` tuple을 추가해
오류 메시지와 config 검증에서 같은 목록을 사용한다.

Weight loading은 기존 정책을 유지한다.

- `weights=None`으로 torchvision model을 생성한다.
- 로컬 weight가 없으면 네트워크 다운로드를 시도하지 않고 `FileNotFoundError`를 발생시킨다.
- `torch.load(path, map_location="cpu", weights_only=True)`로 읽는다.
- `load_state_dict(state_dict, strict=True)`를 기본으로 사용한다.

### 3.3. Backbone family별 feature extraction

`TorchBackbone`은 backbone family별로 final spatial feature를 반환한다.

| family | feature extraction | out channel source | stage output |
| --- | --- | --- | --- |
| ResNet | `conv1/bn1/relu/maxpool/layer1..layer4` | `net.fc.in_features` | `[layer1, layer2, layer3, layer4]` |
| EfficientNet | `net.features(images)` | `net.classifier[-1].in_features` | `[final]` |
| VGG | `net.features(images)` | `net.classifier[0].in_features // 7 // 7` 또는 known channel | `[final]` |

이번 `reg` method는 `coord_gap`과 `coord_spatial`만 사용하므로 `final` feature가 핵심이다. EfficientNet과
VGG의 `stages`는 full multi-scale 계약이 아니므로 `[final]`만 반환하고, `stage_channels`와
`stage_strides`는 full decoder용으로 사용하지 않는다. `CNNBackboneAdapter(keep_stages=False)`를
사용하는 현재 `reg` 경로에서는 이 제한이 문제되지 않는다.

VGG final channel은 `vgg16`, `vgg16_bn`, `vgg19`, `vgg19_bn` 모두 `512`로 고정한다. EfficientNet-B0
final channel은 `1280`으로 둔다.

### 3.4. `reg` model과 config 연결

`src/models/reg/model.py`는 `custom`이 아니면 `TorchBackbone(backbone)`으로 생성한다. 지원 목록은
`custom`과 `SUPPORTED_BACKBONES`를 합친 값이다. unsupported backbone 오류 메시지에는 전체 목록을
포함한다.

`scripts/config.py`는 `get_model_name(cfg)`를 다음 규칙으로 일반화한다.

```text
<backbone>_<head>
```

예를 들어 `efficientnet_b0 + coord_spatial`은 `efficientnet_b0_coord_spatial`이다. `custom`도 같은
규칙을 사용해 기존 `custom_coord_gap`, `custom_coord_spatial` 값을 유지한다.

`get_experiment(cfg)`는 현재 형식 `reg_bs<batch_size>_ep<max_epochs>_<backbone>_<head>`를 유지한다.
`get_wrapper_kwargs()`와 `parse_args()`는 이미 `backbone`과 `head`를 전달하므로 변경하지 않는다.

### 3.5. Experiments config 확장

`experiments/configs.py`는 backbone 차이만 비교하도록 모든 config의 `head`를 `coord_spatial`로
고정한다. `coord_gap`은 이번 evaluation batch에서 제외한다. Full matrix는 backbone 9개와 head 2개의
18개 조건이 될 수 있지만, 이번 plan의 CONFIGS는 head ablation이 아니라 backbone ablation이다.

기본 CONFIGS는 다음 9개 조건으로 고정한다.

- `custom`: `coord_spatial`
- `resnet18`: `coord_spatial`
- `resnet34`: `coord_spatial`
- `resnet50`: `coord_spatial`
- `efficientnet_b0`: `coord_spatial`
- `vgg16`: `coord_spatial`
- `vgg16_bn`: `coord_spatial`
- `vgg19`: `coord_spatial`
- `vgg19_bn`: `coord_spatial`

`coord_gap` 비교가 필요하면 후속 ablation에서 별도 plan으로 `experiments/configs.py`에 dict를
추가한다. `run.py`는 이미 `head`를 전달하므로 변경하지 않는다.

## 4. 완료 기준

이 plan은 다음 조건을 만족하면 `Done`으로 본다.

- `docs/plans/0005-torch-backbone-expansion-plan.md` 상태가 `Approved`에서 `Done`으로 갱신되어 있다.
- `TorchBackbone` class가 존재하고 새 코드가 이 이름을 import한다.
- `TorchvisionBackbone` alias가 유지되어 기존 import가 깨지지 않는다.
- 8개 torchvision backbone이 모두 로컬 weight로 생성되고 dummy forward에서 `final` feature를 반환한다.
- `RegWrapper(backbone=<id>, head="coord_spatial")`가 8개 torchvision backbone 모두에서 `(B, 8)` raw output을 반환한다.
- `scripts/config.py`의 output directory가 `<backbone>_<head>` model segment로 분리된다.
- `experiments/configs.py`가 계획한 9개 조건을 포함하고 모든 config의 `head`가 `coord_spatial`이다.

## 5. 검증

구현 후 다음 순서로 검증한다.

```bash
conda activate pytorch_env
python -c "from src.models.backbones.torchvision_backbone import TorchBackbone, TorchvisionBackbone; print(TorchBackbone); print(TorchvisionBackbone)"
python -c "import torch; from src.models.backbones.torchvision_backbone import TorchBackbone; names=['resnet18','resnet34','resnet50','efficientnet_b0','vgg16','vgg16_bn','vgg19','vgg19_bn']; [print(n, TorchBackbone(n)(torch.zeros(1,3,224,224))['final'].shape) for n in names]"
python -c "import torch; from src.models.reg.wrapper import RegWrapper; names=['resnet18','resnet34','resnet50','efficientnet_b0','vgg16','vgg16_bn','vgg19','vgg19_bn']; [print(n, RegWrapper(backbone=n, head='coord_spatial', device='cpu').model(torch.zeros(1,3,224,224)).shape) for n in names]"
python -c "from experiments.configs import CONFIGS; from scripts.config import get_output_dir; print(len(CONFIGS)); print(all(c.get('head') == 'coord_spatial' for c in CONFIGS)); print([get_output_dir(c) for c in CONFIGS])"
python scripts/train.py --method reg --backbone efficientnet_b0 --head coord_spatial --device cpu --train_size 2 --valid_size 2 --batch_size 1 --max_epochs 1 --patience 1 --num_workers 0 --output_dir /tmp/reg_efficientnet_b0_spatial_smoke
python scripts/train.py --method reg --backbone vgg16 --head coord_spatial --device cpu --train_size 2 --valid_size 2 --batch_size 1 --max_epochs 1 --patience 1 --num_workers 0 --output_dir /tmp/reg_vgg16_spatial_smoke
```

검증 결과에서는 새 class 이름, 기존 alias 호환성, 모든 local weight load 성공, `reg` wrapper output
shape, output directory 분리, CONFIGS의 `coord_spatial` 고정 여부, EfficientNet과 VGG 대표 smoke
train 성공 여부를 확인한다.
