# reg timm CNN/Transformer backbone 지원

다음 표는 이 plan의 상태와 적용 범위를 정리한다.

| 항목 | 값 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-18 |
| 적용 범위 | `src/models/backbones/timm_backbone.py`(신규), `src/models/reg/model.py`, `experiments/configs.py`, `docs/references/backbones.md` |
| 관련 문서 | [docs/architecture/model-assembly.md](../architecture/model-assembly.md), [docs/references/backbones.md](../references/backbones.md), [docs/plans/0007-reg-transformer-backbone-plan.md](0007-reg-transformer-backbone-plan.md) |

## 1. 목적과 배경

`docs/plans/0007-reg-transformer-backbone-plan.md`에서 `reg`는 `custom`, ResNet, EfficientNet,
VGG(torchvision source)와 ViT/Swin(torchvision source)까지 지원하게 되었다. canonical 문서
6.1절 backbone family 표는 `ResNet`, `MobileNet/EfficientNet`, `ViT/DINOv2`, `Swin` family를
architecture 단위로 정의하며, source(torchvision/external/custom)는 2.3절에서 별도 축으로
분리되어 있다([model-assembly.md:149-151](../architecture/model-assembly.md#L149-L151)). 즉
timm(`source=external`) backbone은 이미 정의된 `ResNet`, `ViT` family row에 해당하는
architecture를 다른 source에서 가져오는 것이므로, 이번 plan은 canonical 문서를 수정하지 않고
6.1/6.2절 정의를 그대로 채우는 구현 확장이다.

`docs/references/backbones.md` 3.1절(조건부 가중치)에 로컬 weight가 이미 있고, `pytorch_env`에서
직접 검증한 결과 다음 3개 weight는 architecture 원본 classifier까지 포함해 `strict=True`로 그대로
load된다.

| backbone id (timm 모델명) | family | 로컬 weight | 검증된 `forward_features` shape (224 입력) |
|---|---|---|---|
| `wide_resnet50_2.tv_in1k` | CNN | `/mnt/d/backbones/wide_resnet50_2.tv_in1k/model.safetensors` | `(N, 2048, 7, 7)` |
| `deit_base_distilled_patch16_224.fb_in1k` | ViT (2 prefix token: cls+dist) | `/mnt/d/backbones/deit_base_distilled_patch16_224.fb_in1k/model.safetensors` | `(N, 198, 768)` |
| `cait_s24_224.fb_dist_in1k` | ViT (1 prefix token: cls) | `/mnt/d/backbones/cait_s24_224.fb_dist_in1k/model.safetensors` | `(N, 197, 384)` |

세 architecture 모두 `timm.create_model(name, pretrained=False)`(기본 classifier 포함)로 만든
뒤 `strict=True`로 local weight를 load하고, 이어서 timm의 범용 API `net.reset_classifier(0)`으로
classifier head를 제거해도 `forward_features()` 출력 shape는 바뀌지 않음을 확인했다. 이 덕분에
`TorchBackbone`처럼 architecture별로 submodule을 직접 재조립할 필요가 없고, CNN family는
`{"final", "stages"}` dict를, ViT family는 `{"cls", "tokens", "grid_size"}` dict를 그대로
반환하면 기존 `CNNBackboneAdapter`/`TransformerBackboneAdapter`를 코드 변경 없이 재사용할 수
있다(신규 adapter 파일 불필요, plan 0007과의 차이점).

## 2. 범위

포함하는 항목은 다음과 같다.

- `src/models/backbones/timm_backbone.py` 신규 작성. `TimmBackbone` 클래스 하나로 CNN family와
  ViT family를 모두 처리한다(architecture별 submodule 분해 없이 `net.forward_features()` 재사용).
- `src/models/reg/model.py`에 `TimmBackbone`을 backbone 후보로 연결하고, ViT family 여부에 따라
  기존 `CNNBackboneAdapter`/`TransformerBackboneAdapter` 선택 분기를 timm 이름까지 확장한다.
- `experiments/configs.py`에 위 표 3개 backbone의 `head="coord_spatial"` config를 추가한다.
  (plan 0007과 동일하게 `coord_gap` config는 추가하지 않는다. head 선택 자체는 코드로 계속
  지원한다.)
- `docs/references/backbones.md` 3.1절 해당 3개 row의 "적용 방법과 제약" 열 문구를 `reg` 연결
  완료 상태로 갱신한다(표 위치와 나머지 row는 유지).

이번 plan에서 제외하는 항목은 다음과 같다.

- `deit_base_distilled_patch16_384`(해상도 384 ablation), `cait_m48_448`(대형 model, 카탈로그에
  "F6에 부적합"으로 명시).
- distillation head(`head_dist`) 자체를 학습에 사용하는 것. DeiT의 dist token은 patch token
  spatial과 cls global feature 계산에서 완전히 제외한다(prefix token으로만 건너뛴다).
- `stages` capability(multi-scale decoder, U-Net/FPN dense head 등). timm CNN family도
  `forward_features()`가 마지막 stage만 주므로 `stages=[final]` 단일 원소로 제한한다(기존
  EfficientNet/VGG와 동일한 제약).
- DINOv2, ViT-L 등 다른 조건부 foundation backbone(별도 plan 대상, 이미 0007에서도 제외됨).

## 3. 구현 계획

### 3.1. `src/models/backbones/timm_backbone.py` (신규)

```python
# src/models/backbones/timm_backbone.py: timm CNN/transformer backbone wrappers

import os
import timm
from safetensors.torch import load_file

from src.models.backbones.base_backbone import BaseBackbone

TIMM_BACKBONE_WEIGHTS = {
    "wide_resnet50_2.tv_in1k": "/mnt/d/backbones/wide_resnet50_2.tv_in1k/model.safetensors",
    "deit_base_distilled_patch16_224.fb_in1k": "/mnt/d/backbones/deit_base_distilled_patch16_224.fb_in1k/model.safetensors",
    "cait_s24_224.fb_dist_in1k": "/mnt/d/backbones/cait_s24_224.fb_dist_in1k/model.safetensors",
}

TIMM_CNN_BACKBONES = ("wide_resnet50_2.tv_in1k",)
TIMM_VIT_PREFIX_TOKENS = {
    "deit_base_distilled_patch16_224.fb_in1k": 2,
    "cait_s24_224.fb_dist_in1k": 1,
}
TIMM_VIT_BACKBONES = tuple(TIMM_VIT_PREFIX_TOKENS.keys())
SUPPORTED_TIMM_BACKBONES = TIMM_CNN_BACKBONES + TIMM_VIT_BACKBONES


class TimmBackbone(BaseBackbone):
    """timm model wrapper returning the same native CNN/ViT feature contract as TorchBackbone."""

    def __init__(self, backbone="wide_resnet50_2.tv_in1k", pretrained=True):
        super().__init__()
        if backbone not in SUPPORTED_TIMM_BACKBONES:
            raise ValueError("Unknown timm backbone: %s. Supported: %s"
                             % (backbone, ", ".join(SUPPORTED_TIMM_BACKBONES)))

        net = timm.create_model(backbone, pretrained=False)
        if pretrained:
            self.load_local_weights(net, TIMM_BACKBONE_WEIGHTS[backbone])
        net.reset_classifier(0)

        self.backbone_name = backbone
        self.net = net
        self.out_channels = net.num_features
        if backbone in TIMM_CNN_BACKBONES:
            self.family = "cnn"
            self.stage_channels = (self.out_channels,)
            self.stage_strides = (32,)
        else:
            self.family = "vit"
            self.patch_size = net.patch_embed.patch_size[0]
            self.prefix_tokens = TIMM_VIT_PREFIX_TOKENS[backbone]
        self.out_stride = 32

    def load_local_weights(self, net, path):
        if not os.path.exists(path):
            raise FileNotFoundError("Local timm weight not found: %s" % path)
        state_dict = load_file(path)
        net.load_state_dict(state_dict, strict=True)

    def forward(self, images):
        if self.family == "cnn":
            final = self.net.forward_features(images)
            return {"final": final, "stages": [final]}

        tokens = self.net.forward_features(images)
        grid_h = images.shape[2] // self.patch_size
        grid_w = images.shape[3] // self.patch_size
        return {"cls": tokens[:, 0], "tokens": tokens[:, self.prefix_tokens:], "grid_size": (grid_h, grid_w)}
```

weight를 `net`의 기본(classifier 포함) 구조로 먼저 `strict=True` load하는 이유는 로컬
safetensors가 원본 classifier 포함 checkpoint이기 때문이다(사전 검증 완료). `reset_classifier(0)`은
이후에 호출해 classifier parameter를 optimizer 대상에서 제외한다.

### 3.2. `src/models/reg/model.py` 연결

`SUPPORTED_TIMM_BACKBONES`, `TIMM_VIT_BACKBONES`, `TimmBackbone`을 import한다. backbone 선택
분기에 `elif backbone in SUPPORTED_TIMM_BACKBONES: encoder = TimmBackbone(backbone)`를 추가하고,
`ValueError`의 `supported` 목록에도 포함한다. adapter 선택의 `is_vit` 판정을
`backbone in VIT_BACKBONES or backbone in TIMM_VIT_BACKBONES`로 확장한다. `coord_gap`/`coord_spatial`
head 생성 로직과 `FeatureSpec` 계산은 plan 0007과 동일하게 `encoder.out_channels`를 그대로 쓰므로
추가 변경이 없다.

### 3.3. `experiments/configs.py` 확장

```python
{"method": "reg", "batch_size": 4, "max_epochs": 5, "backbone": "wide_resnet50_2.tv_in1k", "head": "coord_spatial"},
{"method": "reg", "batch_size": 4, "max_epochs": 5, "backbone": "deit_base_distilled_patch16_224.fb_in1k", "head": "coord_spatial"},
{"method": "reg", "batch_size": 4, "max_epochs": 5, "backbone": "cait_s24_224.fb_dist_in1k", "head": "coord_spatial"},
```

`scripts/config.py`의 `get_model_name`/`get_experiment`는 backbone 문자열을 그대로 이름에
사용하므로 추가 수정이 없다(디렉터리/experiment 이름에 timm 모델명의 `.`가 그대로 들어간다).

### 3.4. `docs/references/backbones.md` 갱신

3.1절의 `wide_resnet50_2.tv_in1k`, `deit_base_distilled_patch16_224.fb_in1k`,
`cait_s24_224.fb_dist_in1k` 세 row의 "적용 방법과 제약" 열 문구를 `reg backbone으로 연결 완료`
취지로 갱신한다. 표 구조와 나머지 row는 유지한다.

## 4. 완료 기준

이 plan은 다음 조건을 만족하면 `Done`으로 본다.

- `TimmBackbone("wide_resnet50_2.tv_in1k")`가 local weight로 생성되고 `forward`가
  `{"final": (B, 2048, 7, 7), "stages": [...]}`를 반환한다.
- `TimmBackbone("deit_base_distilled_patch16_224.fb_in1k")`와 `TimmBackbone("cait_s24_224.fb_dist_in1k")`가
  각각 `cls (B, 768)`/`tokens (B, 196, 768)`/`grid_size (14, 14)`, `cls (B, 384)`/`tokens (B, 196, 384)`/
  `grid_size (14, 14)`를 반환한다.
- `RegModel(backbone=<위 3개>, head="coord_spatial")`이 모두 `(B, 8)` raw output을 반환한다.
- `RegWrapper(backbone=<위 3개>, head="coord_spatial")`로 2 sample CPU smoke train이 각각 성공한다.
- `experiments/configs.py`에 3개 config가 추가된다.
- `docs/references/backbones.md` 3.1절 해당 3개 row 설명이 갱신된다.
- `docs/plans/0008-reg-timm-backbone-plan.md` 상태가 `Approved`에서 `Done`으로 갱신된다.

## 5. 검증

구현 후 다음 순서로 검증한다.

```bash
conda activate pytorch_env
python -c "import torch; from src.models.backbones.timm_backbone import TimmBackbone; \
b = TimmBackbone('wide_resnet50_2.tv_in1k'); print(b(torch.zeros(1,3,224,224))['final'].shape)"
python -c "import torch; from src.models.backbones.timm_backbone import TimmBackbone; \
for n in ['deit_base_distilled_patch16_224.fb_in1k', 'cait_s24_224.fb_dist_in1k']: \
    b = TimmBackbone(n); out = b(torch.zeros(1,3,224,224)); \
    print(n, out['cls'].shape, out['tokens'].shape, out['grid_size'])"
python -c "import torch; from src.models.reg.wrapper import RegWrapper; \
for n in ['wide_resnet50_2.tv_in1k', 'deit_base_distilled_patch16_224.fb_in1k', 'cait_s24_224.fb_dist_in1k']: \
    m = RegWrapper(backbone=n, head='coord_spatial', device='cpu').model; \
    print(n, m(torch.zeros(2,3,224,224)).shape)"
python -c "from experiments.configs import CONFIGS; print(len(CONFIGS)); \
print([c['backbone'] for c in CONFIGS])"
python scripts/train.py --method reg --backbone wide_resnet50_2.tv_in1k --head coord_spatial --device cpu \
  --train_size 2 --valid_size 2 --batch_size 1 --max_epochs 1 --patience 1 --num_workers 0 \
  --output_dir /tmp/reg_wide_resnet50_2_timm_smoke
python scripts/train.py --method reg --backbone deit_base_distilled_patch16_224.fb_in1k --head coord_spatial --device cpu \
  --train_size 2 --valid_size 2 --batch_size 1 --max_epochs 1 --patience 1 --num_workers 0 \
  --output_dir /tmp/reg_deit_base_distilled_smoke
python scripts/train.py --method reg --backbone cait_s24_224.fb_dist_in1k --head coord_spatial --device cpu \
  --train_size 2 --valid_size 2 --batch_size 1 --max_epochs 1 --patience 1 --num_workers 0 \
  --output_dir /tmp/reg_cait_s24_224_smoke
```

검증 결과에서는 native feature shape, `RegModel`/`RegWrapper` output shape, `experiments/configs.py`
config 개수와 backbone 목록, 세 backbone 각각의 smoke train 성공 여부를 확인한다. 검증 후
`/tmp/reg_*_smoke` 산출물은 삭제한다.

참고: `experiments/configs.py`는 현재 사용자가 로컬에서 8개 중 6개 config를 주석 처리한 상태다.
이번 plan은 정식 `CONFIGS` 목록에 3개 항목을 새로 추가하는 작업이며, 기존 주석 처리 상태는
건드리지 않는다(주석 처리된 6개 줄은 그대로 유지한다).
