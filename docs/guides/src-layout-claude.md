---
tags: [roi-corner-detection, src, layout, structure, ver1-reuse, guide]
status: guide
created: 2026-07-16
updated: 2026-07-16
---

# src 폴더 구조 안내

이 문서는 ver1 프로젝트(`260701_roi-corner-detection-ver1`)의 실행 skeleton 위에서 ver2 `src/`
폴더와 파일 구조를 정리한 안내다. 전체 설계 기준은
[model-assembly.md](../architecture/model-assembly.md)이며 이 문서는 그 SSOT 구성요소를 실제
파일과 class에 대응시킨다. 이 문서는 구조 안내이므로 SSOT의 계약, registry 항목, metric,
category를 재정의하지 않고 참조만 한다.

## 1. 설계 방침

ver2는 백지에서 새 구조를 만들지 않고 ver1 skeleton을 그대로 옮긴 뒤 ver2 SSOT가 새로 도입한
조립 primitive만 추가한다. ver1에서 유지하는 사용 방식은 다음과 같다.

- wrapper 방식으로 model, loss, metrics, optimizer, scheduler, device를 조합한다.
- `scripts/`의 `train.py`, `evaluate.py`, `predict.py`, `config.py`를 같은 방식으로 사용하고
  `config.py`의 DEFAULT를 args로 override한다.
- `experiments/run.py`와 `configs.py` batch 실행을 그대로 사용한다.
- ver1의 `data`를 그대로 사용한다.
- `core/`의 `trainer.py`, `evaluator.py`, `predictor.py`, `factory.py`를 그대로 사용한다.

ver1과 다른 두 지점은 다음과 같이 반영한다. 첫째, method 이름은 SSOT registry code(`reg`, `seg`,
`det`, `heatmap`, `line`, `refinement`, `classical`)를 쓰고 ver1의 `direct`, `homography`, `doc`,
`foundation`은 `reg` package의 variant로 흡수한다. 둘째, ver1 model이 method별로 backbone과
decoder를 hardcode하던 부분을 ver2에서는 공통 primitive(`blocks`, `backbones`, `features`,
`decoders`, `heads`)에서 조립한다.

## 2. 전체 폴더 구조

전체 `src/` 구조는 다음과 같다. 주석의 `+`는 ver2에서 새로 추가하는 항목이고 나머지는 ver1에서
재사용하거나 최소 수정하는 항목이다. 폴더를 알파벳순으로 먼저, 파일을 알파벳순으로 나열한다.

```text
src/
├ backbones/            # + ver2 composition primitive
│  ├ __init__.py
│  ├ custom_backbone.py
│  ├ pretrained_backbone.py
│  └ adapter.py
├ blocks/               # + ver2 composition primitive
│  ├ __init__.py
│  ├ conv_block.py
│  └ deconv_block.py
├ core/                 # ver1 reuse
│  ├ __init__.py
│  ├ evaluator.py
│  ├ factory.py
│  ├ predictor.py
│  └ trainer.py
├ data/                 # ver1 reuse
│  ├ __init__.py
│  ├ dataloader.py
│  ├ dataset.py
│  ├ images.py
│  ├ midv2020.py
│  ├ smartdoc.py
│  └ transforms.py
├ decoders/             # + ver2 composition primitive
│  ├ __init__.py
│  ├ plain.py
│  ├ unet.py
│  ├ fpn.py
│  └ neck.py
├ features/             # + ver2 composition primitive
│  ├ __init__.py
│  ├ feature_bundle.py
│  ├ feature_extractor.py
│  └ feature_spec.py
├ heads/                # + ver2 shared head
│  ├ __init__.py
│  ├ coord_head.py
│  ├ mask_head.py
│  ├ heatmap_head.py
│  ├ line_head.py
│  └ detection_head.py
├ losses/               # ver1 reuse, ver2 add
│  ├ __init__.py
│  ├ base_loss.py
│  ├ bce_loss.py
│  ├ cross_entropy_loss.py
│  ├ dice_loss.py
│  ├ focal_loss.py
│  ├ mse_loss.py
│  ├ smooth_l1_loss.py
│  └ wing_loss.py
├ metrics/              # ver1 reuse, ver2 add
│  ├ __init__.py
│  ├ base_metric.py
│  ├ max_cd.py
│  ├ mcd.py
│  ├ pck.py
│  ├ polygon_iou.py
│  ├ reprojection_error.py
│  └ success_rate.py
├ models/
│  ├ __init__.py
│  ├ base/             # ver1 reuse
│  │  ├ __init__.py
│  │  ├ base_model.py
│  │  ├ base_postprocessor.py
│  │  ├ base_preprocessor.py
│  │  └ base_wrapper.py
│  ├ classical/        # + ver2 rule-based pipeline, SSOT category F
│  ├ det/              # ver1 det/torchdet merge, SSOT det
│  ├ heatmap/          # ver1 heatmap reuse, SSOT heatmap
│  ├ hybrid/           # ver1 hybrid reuse, SSOT category E
│  ├ line/             # ver1 line reuse, SSOT line
│  ├ refinement/       # ver1 gcn + ver2 local_stn, SSOT category D
│  ├ reg/              # ver1 direct/homography/doc/foundation merge, SSOT reg
│  └ seg/              # ver1 seg/torchseg merge, SSOT seg
├ utils/                # ver1 reuse
│  ├ __init__.py
│  ├ geometry.py
│  ├ homography.py
│  ├ io.py
│  ├ measure.py
│  └ plot.py
└ __init__.py
```

각 method package(`classical/`, `det/`, `heatmap/`, `hybrid/`, `line/`, `refinement/`, `reg/`,
`seg/`)는 ver1과 같이 `__init__.py`, `model.py`, `preprocessor.py`, `postprocessor.py`,
`wrapper.py`를 담는다. 위 tree에서는 반복을 피하기 위해 package 폴더만 표시했다. 학습이 없는
`classical/`은 loss와 optimizer가 없으므로 `preprocessor.py`를 두지 않을 수 있다.

## 3. ver1에서 그대로 재사용하는 항목

다음 항목은 ver1에서 변경 없이 옮긴다. wrapper 계약과 data 흐름이 ver2에서도 동일하기 때문이다.

| 대상 | 재사용 이유 |
|---|---|
| `core/trainer.py`, `core/evaluator.py`, `core/predictor.py` | wrapper의 `train_step`, `eval_step`, `predict_step` 계약이 ver2에서도 동일하다. |
| `core/factory.py`의 `get_dataloader`, `get_dataset`, `get_samples`, `get_transform`, `get_logger` | data 흐름과 logging이 ver2에서도 동일하다. |
| `data/` 전체 | ver1 data를 그대로 사용한다. |
| `utils/` 전체 | geometry, homography, io, measure, plot 유틸을 재사용한다. |
| `models/base/` 전체 | wrapper 계약의 기반이며 변경하지 않는다. |
| `losses/`, `metrics/` 기존 파일 | ver2 metric bank가 ver1 metric 파일과 일치한다. |

## 4. ver1에서 최소 수정하는 항목

다음 항목은 ver1 코드를 유지하되 ver2 요구에 맞춰 작게 수정한다.

| 대상 | 수정 내용 | SSOT 근거 |
|---|---|---|
| `core/factory.py`의 `get_wrapper` | dispatch key를 SSOT registry code(`reg`, `seg`, `det`, `heatmap`, `line`, `refinement`, `classical`)로 교체한다. | section 2.4, 8, 10 |
| `scripts/config.py`의 `get_output_dir` | 경로에 `dataset`과 `model` segment를 추가해 `outputs/<dataset>/<method>/<model>/<exp_name>/`로 맞춘다. | section 1.3 |
| `scripts/config.py`의 `DEFAULTS`, `DEFAULT_BACKBONES` | dataset stage(`public`, `synthetic`, `measured`)와 새 backbone 기본값을 추가한다. | section 1.4 |
| `experiments/configs.py`의 `CONFIGS` | ablation matrix에 대응하는 조합 목록으로 교체한다. | section 11.4 |
| `losses/`, `metrics/` | 필요한 loss와 metric이 있으면 같은 base class 방식으로 파일을 추가한다. | section 1.3 |

## 5. ver2에서 새로 추가하는 조립 primitive

다음 폴더와 파일은 ver2 SSOT가 도입한 composable primitive이며 method model이 이 primitive에서
조립된다. class와 function은 다음과 같다.

| 파일 | class와 function | SSOT 근거 |
|---|---|---|
| `blocks/conv_block.py` | class `ConvBlock`, Conv2d와 norm과 activation, width와 stride 설정 | section 3.2 |
| `blocks/deconv_block.py` | class `DeconvBlock`, mode `interpolate_conv` 기본과 `transposed_conv` | section 3.3 |
| `backbones/custom_backbone.py` | class `CustomBackbone`, stem과 encoder stage 4개, stride 16 | section 3.4 |
| `backbones/pretrained_backbone.py` | class `PretrainedBackbone`, torchvision과 timm과 DINOv2 weight wrapping | section 6.1 |
| `backbones/adapter.py` | class `BackboneAdapter`, native feature를 `global`과 `spatial`과 `stages`로 변환 | section 3.5 |
| `features/feature_spec.py` | class `FeatureSpec`, channel과 stride와 capability flag | section 3.5 |
| `features/feature_bundle.py` | class `FeatureBundle`, `global`과 `spatial`과 `stages` field | section 3.5, 4.1 |
| `features/feature_extractor.py` | class `FeatureExtractor`, backbone과 adapter와 `FeatureSpec`을 `FeatureBundle`로 조립 | section 3.5 |
| `decoders/plain.py` | class `PlainDecoder`, spatial만 사용, skip `none` | section 5.1 |
| `decoders/unet.py` | class `UNetDecoder`, `skip_connection`은 `add`와 `concat` | section 5.2, 5.3 |
| `decoders/fpn.py` | class `FPNDecoder`, lateral와 top-down | section 5.4 |
| `decoders/neck.py` | class `MultiScaleNeck`, custom detection용 | section 4.4 |

공통 head는 `heads/` 폴더로 분리하며 method model이 import한다. class는 다음과 같다.

| 파일 | class | SSOT 근거 |
|---|---|---|
| `heads/coord_head.py` | class `CoordGapHead`, `CoordSpatialHead`, 출력 `(B, 8)` | section 4.2 |
| `heads/mask_head.py` | class `MaskHead`, 최종 channel projection | section 4.3 |
| `heads/heatmap_head.py` | class `HeatmapHead`, 출력 `(B, 4, Hh, Wh)` | section 2.2 |
| `heads/line_head.py` | class `LineHead`, structured dense map | section 2.2 |
| `heads/detection_head.py` | class `DetectionHead`, corner box 또는 point raw output | section 4.4 |

## 6. method package와 registry code 매핑

method package 이름은 SSOT registry code를 쓴다. ver1의 여러 method는 하나의 registry package로
통합하거나 package 안의 variant로 흡수한다. 각 package는 ver1과 같이 `__init__.py`, `model.py`,
`preprocessor.py`, `postprocessor.py`, `wrapper.py`를 담는다.

| ver2 package | 흡수하는 ver1 method | variant 축 | SSOT 근거 |
|---|---|---|---|
| `models/reg/` | `direct`, `homography`, `doc`, `foundation` | `target`(corners, homography_offsets), backbone(custom, ResNet, ViT, DINOv2 frozen) | section 2.4 |
| `models/seg/` | `seg`, `torchseg` | `usage`(composable, whole_model), decoder variant | section 5, 7 |
| `models/det/` | `det`, `torchdet` | `usage`(composable, whole_model) | section 4.4, 7 |
| `models/heatmap/` | `heatmap` | backbone, decoder | section 2.2 |
| `models/line/` | `line` | backbone | section 2.2 |
| `models/refinement/` | `gcn`과 새 `local_stn` | `refiner`(gcn, local_stn) | section 8.2 |
| `models/hybrid/` | `hybrid` | postprocess variant | section 9 |
| `models/classical/` | 새 rule-based | pipeline(contour, line) | section 10 |

새로 추가하는 package는 `models/refinement/`와 `models/classical/`이다. `models/refinement/`는
ver1 `gcn`을 흡수하고 `local_stn` refiner를 추가한다. `models/classical/`은 학습이 없으므로
`wrapper.py`가 optimizer와 scheduler와 loss 없이 inference 계약만 채운다.

## 7. primitive와 method model의 연결

ver1의 `direct` model은 torchvision backbone에 linear head를 직접 붙이고 `seg` model은 UNet을
hardcode한다. ver2에서는 composable method(SSOT category A와 B)의 `model.py`가 새 primitive를
조립하고 head는 `heads/`에서 import한다.

| ver2 method model | ver1 방식 | ver2 조립 방식 | SSOT 근거 |
|---|---|---|---|
| `reg/model.py` | ResNet과 linear head hardcode | `FeatureExtractor`와 adapter 뒤 `heads/coord_head.py`의 `CoordGapHead` 또는 `CoordSpatialHead` | section 4.2 |
| `seg/model.py` | UNet hardcode | `FeatureExtractor` 뒤 decoder variant(`plain`, `unet` add/concat, `fpn`)와 `heads/mask_head.py`의 `MaskHead`, config로 skip 선택 | section 5 |
| `det/model.py` | detection head hardcode | `FeatureExtractor.stages` 뒤 `MultiScaleNeck`과 `heads/detection_head.py`의 `DetectionHead` | section 4.4 |

method `model.py`는 head를 정의하지 않고 `heads/`에서 import해 조립한다. ver1 model이 노출하던
`backbone`과 `head` 속성은 유지해 wrapper의 layer별 learning rate 계약을 깨지 않는다.

## 8. 구현 전 확인할 항목

다음 항목은 구현 시작 전에 확정한다.

- `reg` package가 흡수하는 target과 backbone variant를 어떤 config field로 선택할지 확정한다.
- `seg`와 `det`의 `usage`(composable, whole_model)를 하나의 package 안에서 분기할지 확정한다.
- ver1 `losses/`, `metrics/`에서 ver2 metric bank에 없거나 추가로 필요한 항목이 있는지 확정한다.
