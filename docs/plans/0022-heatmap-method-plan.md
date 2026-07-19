# heatmap method 구현 계획

다음 표는 이 plan의 상태와 적용 범위를 정리한다.

| 항목 | 값 |
|---|---|
| 상태 | Done |
| 작성일 | 2026-07-19 |
| 적용 범위 | `docs/architecture/model-assembly.md`, `src/losses/heatmap_mse_loss.py`, `src/models/heads/heatmap_head.py`, `src/models/heatmap/`, `src/core/factory.py`, `scripts/config.py`, `experiments/configs.py` |
| 관련 문서 | [docs/architecture/model-assembly.md](../architecture/model-assembly.md), [0001-src-implementation-plan.md](0001-src-implementation-plan.md), [0009-seg-unet-backbone-plan.md](0009-seg-unet-backbone-plan.md) |

## 1. 목적과 배경

`heatmap`은 canonical method registry에서 `reg`와 분리된 method이며, raw output도 `(B, 4, Hh, Wh)`
heatmap으로 정의되어 있다. 따라서 `reg`의 head variant가 아니라 독립 method package로 구현한다.

첫 구현은 이미 검증된 `seg`의 stage 기반 dense model 경로를 재사용한다. `custom`, torchvision
CNN/Swin, timm CNN backbone을 지원하고, token-only ViT 계열은 별도 token-to-spatial adapter 설계가
필요하므로 이번 범위에서 제외한다.

## 2. 범위

이번 plan에 포함하는 항목은 다음과 같다.

- canonical 문서에 `CustomHeatmapModel` 조합, 지원 backbone, target, loss와 soft-argmax
  postprocess를 명시한다.
- `src/models/heads/heatmap_head.py`를 추가해 decoded feature를 4-channel corner heatmap logits로
  projection한다.
- `src/models/heatmap/` package를 추가하고 `model.py`, `preprocessor.py`, `postprocessor.py`,
  `wrapper.py`, `__init__.py`를 구현한다.
- backbone 범위는 `custom`, `resnet18`, `resnet34`, `resnet50`, `efficientnet_b0`, `swin_t`,
  `vgg16`, `vgg19`, `wide_resnet50_2.tv_in1k`로 고정한다.
- decoder는 v1에서 기존 U-Net additive skip 구조를 사용하며 output stride는 backbone의 최저 stage
  stride를 따른다.
- `HeatmapPreprocessor`는 `(N, 4, 2)` normalized corners를 `(N, 4, Hh, Wh)` Gaussian target으로
  변환한다.
- `HeatmapPostprocessor`는 raw logits에 soft-argmax를 적용해 `(N, 4, 2)` normalized corners를
  반환한다.
- `src/losses/heatmap_mse_loss.py`를 추가하고 sigmoid heatmap과 Gaussian target 사이의 MSE를
  사용한다.
- `src/core/factory.py`, `scripts/config.py`, `experiments/configs.py`에 `method="heatmap"` 연결을
  추가한다.

이번 plan에서 제외하는 항목은 다음과 같다.

- `vit_b_16`, timm DeiT, timm CaiT 같은 token-only backbone 연결.
- external whole heatmap model 통합.
- argmax, local peak refinement, DARK/UDP 같은 정밀 후처리 비교.
- heatmap auxiliary branch를 붙인 `reg` hybrid 실험.

## 3. 구현 결정

heatmap model의 기본 흐름은 다음과 같이 고정한다.

```text
Backbone
-> CNNBackboneAdapter(keep_stages=True)
-> dense decoder
-> HeatmapHead
-> raw heatmap logits
-> soft-argmax
-> corners
```

target과 postprocess 기본값은 다음과 같다.

| 항목 | 기본값 |
|---|---|
| heatmap channels | 4, corner 순서 `TL`, `TR`, `BR`, `BL` |
| heatmap size | decoder output shape |
| Gaussian sigma | 2.0 heatmap pixels |
| target peak | 각 corner channel별 max 1.0으로 정규화 |
| loss | `HeatmapMSELoss`, `MSE(sigmoid(logits), target)` |
| soft-argmax beta | 10.0 |
| coordinate decode | grid cell center expectation, `[0, 1]` normalized |

public interface 변경은 다음과 같다.

- CLI에서 `python scripts/train.py --method heatmap --backbone custom --head heatmap`을 지원한다.
- `get_wrapper_kwargs()`는 `method in ("reg", "seg", "det", "heatmap")`에 대해 `warmup_epochs`를
  전달한다.
- output path는 기존 규칙대로 `outputs/<dataset>/heatmap/<backbone>_heatmap/<exp_name>/`를 사용한다.
- `experiments/configs.py`에는 heatmap backbone config template을 추가하되, 현재 active run queue는
  바꾸지 않는다.

## 4. 완료 기준

이 plan은 다음 조건을 만족하면 `Done`으로 볼 수 있다.

- `HeatmapWrapper(backbone="custom", device="cpu")`가 생성된다.
- 지원 backbone 전체가 dummy input `(2, 3, 224, 224)`에 대해 raw output `(2, 4, Hh, Wh)`를 반환한다.
- preprocessor target shape와 model output shape가 모든 지원 backbone에서 일치한다.
- postprocessor가 모든 지원 backbone output을 `(2, 4, 2)` normalized corners로 변환한다.
- `src/core/factory.py::get_wrapper("heatmap")`가 `HeatmapWrapper`를 반환한다.
- smoke training이 custom backbone과 pretrained backbone 각 1개에서 성공한다.
- plan 문서 상태가 `Draft`에서 `Done`으로 갱신된다.

## 5. 검증

검증은 conda 환경 `pytorch_env`에서 수행한다.

```bash
conda activate pytorch_env
python -c "import torch; from src.models.heatmap.wrapper import HeatmapWrapper; w = HeatmapWrapper(backbone='custom', device='cpu'); x = torch.randn(2, 3, 224, 224); y = w.model(x); print(tuple(y.shape)); print(tuple(w.postprocessor(y).shape))"
python -c "import torch; from src.models.heatmap.model import SUPPORTED_HEATMAP_BACKBONES, HeatmapModel; [print(name, tuple(HeatmapModel(backbone=name)(torch.randn(1, 3, 224, 224)).shape)) for name in SUPPORTED_HEATMAP_BACKBONES]"
python scripts/train.py --method heatmap --backbone custom --head heatmap --train_size 2 --valid_size 2 --max_epochs 1 --batch_size 1 --num_workers 0 --output_dir /tmp/heatmap_custom_smoke
python scripts/train.py --method heatmap --backbone resnet18 --head heatmap --train_size 2 --valid_size 2 --max_epochs 1 --batch_size 1 --num_workers 0 --output_dir /tmp/heatmap_resnet18_smoke
```

## 6. 가정

이번 plan은 다음 기본값을 승인된 가정으로 둔다.

- heatmap v1은 stage-capable backbone만 지원한다.
- `head="heatmap"`은 CLI와 output naming 호환을 위한 값이며 heatmap v1의 유일한 head다.
- soft-argmax는 첫 구현의 기본 postprocess이고, peak refinement는 후속 ablation으로 분리한다.
- decoder 구조 비교는 heatmap v1의 목표가 아니며, 기존 U-Net additive skip 구조를 baseline으로
  사용한다.
