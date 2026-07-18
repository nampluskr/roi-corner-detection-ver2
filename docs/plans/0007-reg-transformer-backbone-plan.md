# reg ViT/Swin backbone 지원과 backbone ablation 확장

다음 표는 이 plan의 상태와 적용 범위를 정리한다.

| 항목 | 값 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-18 |
| 적용 범위 | `src/models/backbones/torch_backbone.py`, `src/models/adapters/transformer_adapter.py`(신규), `src/models/reg/model.py`, `experiments/configs.py` |
| 관련 문서 | [docs/architecture/model-assembly.md](../architecture/model-assembly.md), [docs/references/backbones.md](../references/backbones.md), [docs/plans/0005-torch-backbone-expansion-plan.md](0005-torch-backbone-expansion-plan.md) |

## 1. 목적과 배경

현재 `reg` method는 `custom` backbone과 CNN 계열 torchvision backbone(`resnet18/34/50`,
`efficientnet_b0`, `vgg16/19`)만 지원한다. canonical 문서 11.4절 5단계는 backbone 비교 대상으로
"custom, ResNet, ViT, Swin backbone"을 명시하고([model-assembly.md:727](../architecture/model-assembly.md#L727)),
6.1절은 ViT/DINOv2 adapter 출력을 `global`과 token-grid `spatial`로, Swin adapter 출력을 CNN과
같은 `global`, `spatial`, `stages`로 정의한다([model-assembly.md:477-482](../architecture/model-assembly.md#L477-L482)).
이번 plan은 이 중 ViT와 Swin을 `reg`에 연결해 5단계 ablation의 backbone 축을 넓힌다.

`docs/references/backbones.md`에는 이 범위에 필요한 `vit_b_16-c867db91.pth`, `swin_t-704ceda3.pth`
로컬 weight가 이미 권장 목록에 있고([backbones.md 2절](../references/backbones.md)), 두 weight
모두 로컬 `torchvision 0.20.1+cu121`의 `models.vit_b_16`/`models.swin_t`에 `strict=True`로 load
가능함을 확인했다.

이번 작업은 canonical architecture를 바꾸지 않는다. 6.1절이 이미 ViT/Swin의 adapter 출력 형태를
정의해두었으므로, 이 plan은 그 정의를 코드로 채우는 구현 확장이며 `docs/architecture/model-assembly.md`
수정은 포함하지 않는다.

## 2. 범위

이번 plan에 포함하는 backbone은 다음 두 가지다.

| backbone id | torchvision builder | local weight | native 출력 |
| --- | --- | --- | --- |
| `vit_b_16` | `torchvision.models.vit_b_16` | `/mnt/d/backbones/vit_b_16-c867db91.pth` | patch token sequence, CLS token 분리 |
| `swin_t` | `torchvision.models.swin_t` | `/mnt/d/backbones/swin_t-704ceda3.pth` | `(N, H, W, C)` spatial map, permute 필요 |

`head`는 `coord_spatial`로 고정한다. `experiments/configs.py`에는 두 backbone 모두
`coord_spatial` config만 추가하고 `coord_gap` config는 추가하지 않는다.

이번 plan에서 제외하는 항목은 다음과 같다.

- DINOv2 계열(`dinov2_vits14`, `dinov2_vitb14` 등). DINOv2는 torchvision builder가 아니라 별도
  loading 경로가 필요하고, canonical 문서 2.4절은 이를 `freeze=true` variant로 별도 분류한다
  ([model-assembly.md:183](../architecture/model-assembly.md#L183)). Freeze 정책(어떤 parameter를
  고정하고 어떤 optimizer group을 쓸지)은 이번 plan에서 결정하지 않는다.
- ViT/Swin의 다른 크기 변형(`vit_l_16`, `swin_s`, `swin_b` 등)과 DeiT, CaiT 같은 timm 계열.
- `coord_gap` head를 ViT/Swin에 연결하는 실험 config 추가. head 선택 자체는 코드로는 계속
  지원하되, 이번 backbone ablation batch는 `coord_spatial`만 다룬다.
- multi-scale decoder, dense mask head 등 `stages` capability를 사용하는 조합. ViT는 6.2절
  compatibility 표에서 이 조합이 초기 제외로 분류되어 있다([model-assembly.md:496](../architecture/model-assembly.md#L496)).

## 3. 구현 계획

### 3.1. `TorchBackbone`에 ViT/Swin family 추가

`BACKBONE_WEIGHTS`와 `BACKBONE_BUILDERS`에 `vit_b_16`, `swin_t` 항목을 추가한다. 두 family 상수
`VIT_BACKBONES = ("vit_b_16",)`, `SWIN_BACKBONES = ("swin_t",)`를 `RESNET_BACKBONES` 등과 같은
위치에 정의한다.

Swin은 native feature가 이미 CNN과 비슷한 spatial map이므로 `TorchBackbone.__init__`과
`forward`에 `family == "swin"` 분기를 추가해 기존 CNN 계열과 같은 dict 계약을 반환한다.

```python
elif backbone in SWIN_BACKBONES:
    self.family = "swin"
    self.stem = net.features
    self.norm = net.norm
    self.out_channels = net.head.in_features
    self.stage_channels = (self.out_channels,)
    self.stage_strides = (32,)
```

```python
elif self.family == "swin":
    final = self.norm(self.stem(images)).permute(0, 3, 1, 2).contiguous()
    return {"final": final, "stages": [final]}
```

이 경로는 `CNNBackboneAdapter`를 그대로 재사용한다. `torch.load` weight 규칙과 `strict=True`
정책은 기존과 동일하게 유지한다.

ViT는 CNN 계열과 output 형태가 달라 같은 dict key를 쓰지 않는다. `family == "vit"` 분기는 CLS
token과 patch token을 분리해 반환한다.

```python
elif backbone in VIT_BACKBONES:
    self.family = "vit"
    self.conv_proj = net.conv_proj
    self.class_token = net.class_token
    self.encoder = net.encoder
    self.out_channels = net.hidden_dim
    self.patch_size = net.patch_size
```

```python
elif self.family == "vit":
    n = images.shape[0]
    patches = self.conv_proj(images).flatten(2).transpose(1, 2)
    cls = self.class_token.expand(n, -1, -1)
    tokens = self.encoder(torch.cat([cls, patches], dim=1))
    grid_h = images.shape[2] // self.patch_size
    grid_w = images.shape[3] // self.patch_size
    return {"cls": tokens[:, 0], "tokens": tokens[:, 1:], "grid_size": (grid_h, grid_w)}
```

`SUPPORTED_BACKBONES`는 `BACKBONE_BUILDERS.keys()`에서 자동 계산되므로 별도 목록 갱신이 필요
없다.

### 3.2. `TransformerBackboneAdapter` 신규 작성

`src/models/adapters/transformer_adapter.py`에 `BaseBackboneAdapter`를 상속하는
`TransformerBackboneAdapter`를 추가한다. `native_features`의 `tokens (N, L, C)`를
`grid_size`로 reshape하고 permute해 `(N, C, H, W)` spatial feature를 만들고, `cls`를
`global_feature`로 사용한다.

```python
class TransformerBackboneAdapter(BaseBackboneAdapter):
    """Reshapes ViT patch tokens into a token-grid spatial map and exposes the CLS token as global feature."""

    def __init__(self, keep_spatial=True, keep_global=True):
        super().__init__()
        self.keep_spatial = keep_spatial
        self.keep_global = keep_global

    def forward(self, native_features):
        global_feature = native_features["cls"] if self.keep_global else None
        spatial_feature = None
        if self.keep_spatial:
            tokens = native_features["tokens"]
            grid_h, grid_w = native_features["grid_size"]
            n, l, c = tokens.shape
            spatial_feature = tokens.transpose(1, 2).reshape(n, c, grid_h, grid_w)
        return FeatureBundle(global_feature=global_feature, spatial_feature=spatial_feature, stages=None)
```

`stages`는 항상 `None`이다. ViT는 이번 plan에서 `stages` capability를 제공하지 않는다.

### 3.3. `RegModel`의 backbone-family별 adapter 선택

`src/models/reg/model.py`는 `backbone in VIT_BACKBONES`인 경우 `TransformerBackboneAdapter`를,
그 외(`custom`, ResNet, EfficientNet, VGG, Swin)는 기존 `CNNBackboneAdapter`를 사용하도록
분기한다. `FeatureSpec`의 `global_channels`/`spatial_channels`는 `encoder.out_channels`를
그대로 사용한다(ViT-B/16은 768).

```python
elif backbone in SUPPORTED_BACKBONES:
    encoder = TorchBackbone(backbone)
    backbone_name = backbone
    ...
if backbone in VIT_BACKBONES:
    adapter_cls = TransformerBackboneAdapter
else:
    adapter_cls = CNNBackboneAdapter
```

`coord_gap`/`coord_spatial` head 생성 로직 자체는 바꾸지 않는다. `CoordSpatialHead`는 임의의
`(N, C, H, W)` 입력을 받으므로 ViT의 `14x14x768`, Swin의 `7x7x768` spatial feature 모두 추가
수정 없이 동작한다.

### 3.4. `experiments/configs.py` 확장

기존 6개 config에 다음 2개를 추가한다.

```python
{"method": "reg", "batch_size": 4, "max_epochs": 5, "backbone": "vit_b_16", "head": "coord_spatial"},
{"method": "reg", "batch_size": 4, "max_epochs": 5, "backbone": "swin_t", "head": "coord_spatial"},
```

`scripts/config.py`의 `get_model_name`/`get_experiment`는 이미 `<backbone>_<head>` 규칙을
쓰므로 별도 수정이 필요 없다.

## 4. 완료 기준

이 plan은 다음 조건을 만족하면 `Done`으로 본다.

- `TorchBackbone("vit_b_16")`과 `TorchBackbone("swin_t")`가 로컬 weight로 생성되고 dummy input
  forward가 성공한다.
- `TransformerBackboneAdapter`가 ViT native feature를 받아 `global_feature`가 `(B, 768)`,
  `spatial_feature`가 `(B, 768, 14, 14)`인 `FeatureBundle`을 반환한다.
- `RegModel(backbone="vit_b_16", head="coord_spatial")`과
  `RegModel(backbone="swin_t", head="coord_spatial")`이 `(B, 8)` raw output을 반환한다.
- `RegWrapper(backbone="vit_b_16", head="coord_spatial")`와
  `RegWrapper(backbone="swin_t", head="coord_spatial")`로 2 sample smoke train이 성공한다.
- `experiments/configs.py`에 `vit_b_16`, `swin_t` config 2개가 `head="coord_spatial"`로
  추가되어 총 8개 config가 된다.
- `docs/plans/0007-reg-transformer-backbone-plan.md` 상태가 `Approved`에서 `Done`으로 갱신된다.

## 5. 검증

구현 후 다음 순서로 검증한다.

```bash
conda activate pytorch_env
python -c "import torch; from src.models.backbones.torch_backbone import TorchBackbone; \
b = TorchBackbone('vit_b_16'); out = b(torch.zeros(1,3,224,224)); \
print(out['cls'].shape, out['tokens'].shape, out['grid_size'])"
python -c "import torch; from src.models.backbones.torch_backbone import TorchBackbone; \
b = TorchBackbone('swin_t'); print(b(torch.zeros(1,3,224,224))['final'].shape)"
python -c "import torch; from src.models.reg.wrapper import RegWrapper; \
for n in ['vit_b_16', 'swin_t']: \
    m = RegWrapper(backbone=n, head='coord_spatial', device='cpu').model; \
    print(n, m(torch.zeros(2,3,224,224)).shape)"
python -c "from experiments.configs import CONFIGS; print(len(CONFIGS)); \
print([c['backbone'] for c in CONFIGS])"
python scripts/train.py --method reg --backbone vit_b_16 --head coord_spatial --device cpu \
  --train_size 2 --valid_size 2 --batch_size 1 --max_epochs 1 --patience 1 --num_workers 0 \
  --output_dir /tmp/reg_vit_b_16_spatial_smoke
python scripts/train.py --method reg --backbone swin_t --head coord_spatial --device cpu \
  --train_size 2 --valid_size 2 --batch_size 1 --max_epochs 1 --patience 1 --num_workers 0 \
  --output_dir /tmp/reg_swin_t_spatial_smoke
```

검증 결과에서는 native feature shape, adapter 출력 shape, `RegModel`/`RegWrapper` output shape,
`experiments/configs.py` config 개수와 backbone 목록, ViT/Swin 각각의 smoke train 성공 여부를
확인한다.
