---
tags: [roi-corner-detection, source-layout, wrapper, proposal, codex]
status: proposal
created: 2026-07-16
updated: 2026-07-16
---

# src 폴더 구조 및 모듈 배치 설계

## 1. 문서 목적과 SSOT 관계

이 문서는 [모델 재조립 카테고리 및 비교 설계](../architecture/model-assembly.md)를 구현할 때 사용할
`src/` 하위 폴더, Python 파일, class와 function의 배치를 제안한다. Architecture, method registry,
공통 입출력과 평가 계약은 SSOT가 결정하며 이 문서는 해당 계약을 변경하지 않는다.

이 문서의 상태는 `proposal`이다. 실제 파일이 존재하지 않는 동안 아래 import path를 구현 완료된
API로 간주하지 않는다.

## 2. ver1 재사용 원칙

ver1의 wrapper 중심 실행 방식을 공통 lifecycle로 유지한다. Wrapper는 학습과 추론에 필요한 객체를
소유하고 core runner는 method 내부 구현을 알지 않은 채 wrapper의 step을 호출한다.

주요 객체의 책임은 다음과 같다.

| 구성요소 | 책임 | 제외 책임 |
|---|---|---|
| method wrapper | model, preprocessor, postprocessor, losses, metrics, optimizer, scheduler, device 소유 | epoch loop와 output 경로 결정 |
| model component factory | backbone, adapter, decoder, neck, head와 model 조립 | loss, metric과 optimizer 생성 |
| `src/core/factory.py` | transform, dataset, dataloader, wrapper와 logger 생성 | method별 tensor 계산 |
| `Trainer` | train, validation, early stopping과 scheduler event 전달 | raw output 해석 |
| `Evaluator` | 공통 metric bank를 사용한 평가 | method별 postprocess 구현 |
| `Predictor` | 표준 corner 수집과 CSV 저장 | model component 조립 |

`BaseWrapper`의 생성 계약은 ver1과 같은 형태를 유지한다.

```python
BaseWrapper(
    model,
    preprocessor,
    postprocessor,
    optimizer=None,
    scheduler=None,
    losses=None,
    metrics=None,
    device=None,
)
```

Composable wrapper는 shared model factory에서 model만 생성한다. Target, loss와 parameter group은 해당
method wrapper가 결정한다. Torchvision detector처럼 train과 inference 호출 규약이 다른 whole model은
전용 wrapper가 `train_step`, `eval_step`, `predict_step`을 override한다.

## 3. src 폴더 구조

`src/`의 목표 구조는 다음과 같다. Project root의 `data/`, `docs/`, `experiments/`, `notebooks/`,
`outputs/`, `scripts/` 구조는 이 tree에 포함하지 않는다. 이 문서 작업에서는 Python 폴더를 생성하지
않는다.

```text
src/
├── core/
│   ├── __init__.py
│   ├── evaluator.py
│   ├── factory.py
│   ├── predictor.py
│   └── trainer.py
├── data/
│   ├── __init__.py
│   ├── dataloader.py
│   ├── dataset.py
│   ├── images.py
│   ├── midv2020.py
│   ├── smartdoc.py
│   └── transforms.py
├── losses/
│   ├── __init__.py
│   ├── base_loss.py
│   ├── bce_loss.py
│   ├── cross_entropy_loss.py
│   ├── dice_loss.py
│   ├── focal_loss.py
│   ├── mse_loss.py
│   ├── smooth_l1_loss.py
│   └── wing_loss.py
├── metrics/
│   ├── __init__.py
│   ├── base_metric.py
│   ├── max_cd.py
│   ├── mcd.py
│   ├── pck.py
│   ├── polygon_iou.py
│   ├── reprojection_error.py
│   └── success_rate.py
├── models/
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base_adapter.py
│   │   ├── cnn_adapter.py
│   │   └── transformer_adapter.py
│   ├── backbones/
│   │   ├── __init__.py
│   │   ├── base_backbone.py
│   │   ├── custom_backbone.py
│   │   ├── timm_backbone.py
│   │   └── torchvision_backbone.py
│   ├── base/
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   ├── base_postprocessor.py
│   │   ├── base_preprocessor.py
│   │   ├── base_wrapper.py
│   │   └── prediction_result.py
│   ├── blocks/
│   │   ├── __init__.py
│   │   ├── conv_block.py
│   │   └── deconv_block.py
│   ├── decoders/
│   │   ├── __init__.py
│   │   ├── fpn_decoder.py
│   │   ├── plain_decoder.py
│   │   └── unet_decoder.py
│   ├── det/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── postprocessor.py
│   │   ├── preprocessor.py
│   │   └── wrapper.py
│   ├── external/
│   │   ├── __init__.py
│   │   ├── detection.py
│   │   ├── segmentation.py
│   │   └── wrapper.py
│   ├── heads/
│   │   ├── __init__.py
│   │   ├── coordinate_head.py
│   │   ├── detection_head.py
│   │   ├── heatmap_head.py
│   │   └── mask_head.py
│   ├── heatmap/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── postprocessor.py
│   │   ├── preprocessor.py
│   │   └── wrapper.py
│   ├── line/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── postprocessor.py
│   │   ├── preprocessor.py
│   │   └── wrapper.py
│   ├── necks/
│   │   ├── __init__.py
│   │   └── multi_scale_neck.py
│   ├── refinement/
│   │   ├── __init__.py
│   │   ├── gcn.py
│   │   ├── local_stn.py
│   │   └── wrapper.py
│   ├── reg/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── postprocessor.py
│   │   ├── preprocessor.py
│   │   └── wrapper.py
│   ├── rule_based/
│   │   ├── __init__.py
│   │   ├── contour.py
│   │   ├── line.py
│   │   └── wrapper.py
│   ├── seg/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── postprocessor.py
│   │   ├── preprocessor.py
│   │   └── wrapper.py
│   ├── __init__.py
│   ├── factory.py
│   └── features.py
├── utils/
│   ├── __init__.py
│   ├── geometry.py
│   ├── homography.py
│   ├── io.py
│   ├── measure.py
│   └── plot.py
└── __init__.py
```

`src/` 아래 모든 폴더에는 빈 `__init__.py`를 둔다. Project root 폴더와 산출물 경로는 SSOT와
[모델 구성 및 성능 비교 사용 가이드](model-usage-codex.md)에서 관리한다.

## 4. src/core

Core는 wrapper lifecycle만 사용하며 method별 model 구조에 의존하지 않는다.

| 파일 | class 또는 function | 역할 | ver1 기준 |
|---|---|---|---|
| `evaluator.py` | `DEFAULT_METRICS`, `Evaluator` | 공통 metric 설정, `evaluate`, 결과 저장 | 최소 수정 |
| `factory.py` | `get_transform`, `get_dataset`, `get_dataloader`, `get_samples` | data 객체 생성 | 재사용 |
| `factory.py` | `get_wrapper`, `get_logger` | current method dispatch와 logger 생성 | dispatch 수정 |
| `predictor.py` | `resolve_image_ids`, `Predictor` | 표준 corner 예측과 CSV 저장 | 최소 수정 |
| `trainer.py` | `is_improved`, `format_result`, `Trainer` | train, validation, early stopping과 history | 재사용 |

`Trainer`는 wrapper의 step, reset, result와 epoch event method만 사용한다. `Evaluator`와 `Predictor`는
SSOT의 failure 계약을 지원하도록 `PredictionResult`를 인식하되, corner tensor만 반환하는 ver1
postprocessor에는 `NaN` 기반 compatibility 경로를 제공한다.

## 5. src/data

Data 계층은 ver1 파일과 public interface를 같은 경로에 우선 그대로 이관한다.

| 파일 | class 또는 function | 역할 |
|---|---|---|
| `dataloader.py` | `Dataloader` | split별 shuffle, worker와 seed 설정 |
| `dataset.py` | `BaseDataset`, `Dataset`, `Subset` | CSV loading, split과 subset |
| `dataset.py` | `CornerDataset`, `ImageDataset` | labeled sample과 image-only sample loading |
| `images.py` | `create_data` | 일반 image source를 공통 CSV로 변환 |
| `midv2020.py` | `create_data` | MIDV-2020 label 변환 |
| `smartdoc.py` | `create_data` | SmartDoc label 변환 |
| `transforms.py` | `Compose`, `Resize` | joint transform과 resize |
| `transforms.py` | `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomRotation`, `RandomPerspective`, `RandomScale`, `RandomAffine` | image와 corner의 동기화된 geometry augmentation |
| `transforms.py` | `ColorJitter`, `GaussianBlur`, `GaussianNoise` | appearance augmentation |
| `transforms.py` | `ToTensor`, `Normalize`, `Denormalize`, `ToNumpy` | tensor와 normalization 변환 |

Parity 검증에서는 같은 CSV, split ratio와 seed로 ver1과 sample index, corner tensor와 image tensor
shape가 같은지 확인한다.

## 6. src/losses와 src/metrics

Loss 파일과 기본 consumer는 다음과 같다.

| 파일 | class | 기본 consumer |
|---|---|---|
| `base_loss.py` | `BaseLoss` | 모든 trainable method |
| `bce_loss.py` | `BCELoss` | segmentation |
| `cross_entropy_loss.py` | `CrossEntropyLoss` | detection class output |
| `dice_loss.py` | `DiceLoss` | segmentation |
| `focal_loss.py` | `FocalLoss` | heatmap과 detection confidence |
| `mse_loss.py` | `MSELoss` | heatmap |
| `smooth_l1_loss.py` | `SmoothL1Loss` | box, point와 offset |
| `wing_loss.py` | `WingLoss` | coordinate regression |

Metric 파일과 result key는 다음과 같다.

| 파일 | class | result key |
|---|---|---|
| `base_metric.py` | `BaseMetric` | stateful metric lifecycle |
| `max_cd.py` | `MaxCD` | `max_cd` |
| `mcd.py` | `MCD` | `mcd` |
| `pck.py` | `PCK` | `pck@0.02`, `pck@0.05` |
| `polygon_iou.py` | `PolygonIoU` | `iou` |
| `reprojection_error.py` | `ReprojectionError` | `reproj_error` |
| `success_rate.py` | `SuccessRate` | `sr` |

실패 sample은 accuracy 평균에서 제외할 수 있지만 전체 SR과 failure reason 분포를 함께 보고한다.

## 7. src/models 공통 component

공통 component는 model 내부 부품이다. Loss, metric, optimizer, scheduler와 postprocessor는 이 계층에
포함하지 않는다.

### 7.1. Base class와 wrapper

Base 파일 배치는 다음과 같다.

| 파일 | class | 계약 |
|---|---|---|
| `base/base_model.py` | `BaseModel` | `forward(images) -> raw_output` |
| `base/base_preprocessor.py` | `BasePreprocessor` | corner를 method target으로 변환 |
| `base/base_postprocessor.py` | `BasePostprocessor` | raw output을 표준 결과로 변환 |
| `base/base_wrapper.py` | `BaseWrapper` | train, eval과 predict lifecycle |
| `base/prediction_result.py` | `PredictionResult` | corners, success와 failure reason 운반 |

### 7.2. Block

Block 파일과 class는 다음과 같다.

| 파일 | class | 책임 |
|---|---|---|
| `blocks/conv_block.py` | `ConvBlock` | convolution, normalization, activation과 downsampling |
| `blocks/deconv_block.py` | `DeconvBlock` | `interpolate_conv` 또는 `transposed_conv` upsampling |

Skip projection과 add 또는 concat은 `DeconvBlock`이 아니라 decoder가 담당한다.

### 7.3. Backbone과 adapter

Backbone과 adapter 파일 배치는 다음과 같다.

| 파일 | class | 책임 |
|---|---|---|
| `backbones/base_backbone.py` | `BaseBackbone` | native feature interface |
| `backbones/custom_backbone.py` | `CustomBackbone` | final spatial feature와 encoder stages 생성 |
| `backbones/timm_backbone.py` | `TimmBackbone` | timm model feature 연결 |
| `backbones/torchvision_backbone.py` | `TorchvisionBackbone` | torchvision model feature 연결 |
| `adapters/base_adapter.py` | `BaseBackboneAdapter` | adapter interface |
| `adapters/cnn_adapter.py` | `CNNBackboneAdapter` | CNN native feature 변환 |
| `adapters/transformer_adapter.py` | `TransformerBackboneAdapter` | token과 hierarchical feature 변환 |

### 7.4. Feature contract

`models/features.py`에는 다음 class를 둔다.

| class | 책임 |
|---|---|
| `FeatureSpec` | global, spatial, stages capability와 channel 및 stride metadata |
| `FeatureBundle` | `global`, `spatial`, `stages` runtime feature 운반 |
| `FeatureExtractor` | backbone, adapter와 spec 조립 |

Consumer가 요구하는 capability가 없으면 factory 생성 단계에서 오류를 반환한다.

### 7.5. Decoder, neck와 head

Task component 파일과 class는 다음과 같다.

| 파일 | class | 책임 |
|---|---|---|
| `decoders/fpn_decoder.py` | `FPNSegDecoder` | multi-stage lateral feature 생성 |
| `decoders/plain_decoder.py` | `PlainSegDecoder` | final spatial feature 복원 |
| `decoders/unet_decoder.py` | `UNetSegDecoder` | add 또는 concat skip fusion |
| `necks/multi_scale_neck.py` | `MultiScaleNeck` | detection multi-scale feature 생성 |
| `heads/coordinate_head.py` | `CoordGapHead`, `CoordSpatialHead` | coordinate raw output 생성 |
| `heads/detection_head.py` | `DetectionHead` | confidence, class와 box 또는 point 출력 |
| `heads/heatmap_head.py` | `HeatmapHead` | four-corner heatmap logits 생성 |
| `heads/mask_head.py` | `MaskHead` | decoded feature를 mask logits로 projection |

`models/factory.py`에는 `get_backbone`, `get_adapter`, `get_decoder`, `get_neck`, `get_head`, `get_model`을
둔다. 이 factory는 optimizer나 training object를 생성하지 않는다.

## 8. Method별 model module

Trainable method는 기본적으로 `model.py`, `preprocessor.py`, `postprocessor.py`, `wrapper.py`를 가진다.

### 8.1. Regression

Regression 파일 배치는 다음과 같다.

| 파일 | class | 책임 |
|---|---|---|
| `reg/model.py` | `RegModel` | feature extractor와 coordinate head 조립 |
| `reg/preprocessor.py` | `RegPreprocessor` | corner 또는 homography offset target 생성 |
| `reg/postprocessor.py` | `RegPostprocessor` | sigmoid reshape 또는 offset decode |
| `reg/wrapper.py` | `RegWrapper` | loss, metric, optimizer와 scheduler 기본값 소유 |

ver1의 `direct`와 `homography`는 `reg`의 target과 postprocess variant로 매핑한다.

### 8.2. Segmentation

Segmentation 파일 배치는 다음과 같다.

| 파일 | class | 책임 |
|---|---|---|
| `seg/model.py` | `SegModel` | feature extractor, decoder와 mask head 조립 |
| `seg/preprocessor.py` | `SegPreprocessor` | polygon mask target 생성 |
| `seg/postprocessor.py` | `SegPostprocessor` | threshold와 four-side fitting |
| `seg/wrapper.py` | `SegWrapper` | BCE, Dice, metric과 optimizer 구성 |

Plain, U-Net add, U-Net concat과 FPN은 `SegModel`의 decoder config다.

### 8.3. Detection

Detection 파일 배치는 다음과 같다.

| 파일 | class | 책임 |
|---|---|---|
| `det/model.py` | `DetModel` | feature extractor, neck와 detection head 조립 |
| `det/preprocessor.py` | `DetPreprocessor` | box, point와 class target 생성 |
| `det/postprocessor.py` | `DetPostprocessor` | selection, center decode와 ordering |
| `det/wrapper.py` | `DetWrapper` | custom detection loss와 optimizer 구성 |

### 8.4. Heatmap과 line

Dense method의 class 배치는 다음과 같다.

| 폴더 | class | 계약 |
|---|---|---|
| `heatmap/` | `HeatmapModel`, `HeatmapPreprocessor`, `HeatmapPostprocessor`, `HeatmapWrapper` | heatmap과 soft-argmax |
| `line/` | `LineModel`, `LinePreprocessor`, `LinePostprocessor`, `LineWrapper` | line map과 intersection |

### 8.5. External whole model

External 파일 배치는 다음과 같다.

| 파일 | class | 책임 |
|---|---|---|
| `external/detection.py` | `ExternalDetModel` | detector, YOLO 또는 DETR native 호출 연결 |
| `external/segmentation.py` | `ExternalSegModel` | whole segmentation output 표준화 |
| `external/wrapper.py` | `ExternalSegWrapper`, `ExternalDetWrapper` | internal loss와 mode 차이 흡수 |

Package-native output은 evaluator나 predictor로 직접 전달하지 않는다.

### 8.6. Refinement와 rule-based

후단 방법론 파일 배치는 다음과 같다.

| 파일 | class | 책임 |
|---|---|---|
| `refinement/gcn.py` | `GCNRefinementModel` | graph iterative offset 예측 |
| `refinement/local_stn.py` | `LocalSTNRefinementModel` | local ROI offset 예측 |
| `refinement/wrapper.py` | `RefinementWrapper` | base corner와 refinement training 조립 |
| `rule_based/contour.py` | `ClassicalContourPipeline` | contour 기반 quad 계산 |
| `rule_based/line.py` | `ClassicalLinePipeline` | line grouping과 intersection |
| `rule_based/wrapper.py` | `RuleBasedWrapper` | evaluator와 predictor lifecycle 연결 |

Rule-based method를 `scripts/train.py`에서 요청하면 지원하지 않는 mode 오류를 반환한다.

## 9. scripts와 experiments

Client entry point는 ver1 구조를 유지한다.

| 파일 | function 또는 상수 | 책임 |
|---|---|---|
| `scripts/config.py` | `DEFAULTS`, `DEFAULT_BACKBONES` | 공통 기본값 |
| `scripts/config.py` | `parse_args`, `load_config`, `apply_overrides`, `resolve_config` | 설정 병합 |
| `scripts/config.py` | `get_experiment`, `get_model_name`, `get_output_dir`, `get_wrapper_kwargs` | 이름, 경로와 wrapper argument |
| `scripts/train.py` | `main` | dataloader, wrapper와 `Trainer` 실행 |
| `scripts/evaluate.py` | `main` | checkpoint와 `Evaluator` 실행 |
| `scripts/predict.py` | `main` | checkpoint와 `Predictor` 실행 |
| `experiments/configs.py` | `CONFIGS` | experiment dictionary 목록 |
| `experiments/run.py` | `MODES`, `get_cli_args`, `run`, `main` | script subprocess 반복 실행 |
| `experiments/benchmark.py` | `benchmark_config`, `main` | 공통 평가와 비용 측정 |

## 10. Class와 function 배치표

Client 요청에서 결과까지의 호출 관계는 다음과 같다.

| 단계 | public 호출 | 생성 또는 호출 대상 |
|---|---|---|
| config | `resolve_config()` | 병합된 config dictionary |
| data | `get_dataloader()` | dataset과 `Dataloader` |
| wrapper | `get_wrapper(method, device, **kwargs)` | method wrapper |
| model | `get_model(model_config)` | composable 또는 external model |
| feature | `get_backbone()`, `get_adapter()` | `FeatureExtractor` |
| task | `get_decoder()`, `get_neck()`, `get_head()` | task component |
| train | `Trainer.fit()` 또는 `fit_early_stop()` | wrapper train과 eval step |
| evaluate | `Evaluator.evaluate()` | metric dictionary |
| predict | `Predictor.predict()` | corner DataFrame |

## 11. 재사용, 수정과 신규 구현 구분

초기 이관 범위는 다음과 같다.

| 구분 | 대상 | 적용 원칙 |
|---|---|---|
| 재사용 | data, 기존 loss와 metric, `Trainer`, geometry utility | 같은 path와 public interface로 이관 |
| 최소 수정 | `BaseWrapper`, `Evaluator`, `Predictor`, checkpoint와 measurement utility | failure metadata와 end-to-end 측정 지원 |
| 구조 수정 | core factory, scripts config, experiments config와 benchmark | current registry와 output path 반영 |
| 신규 구현 | block, adapter, feature contract, decoder, head, neck와 model factory | SSOT 조립 계약 구현 |
| method 재구성 | `reg`, `seg`, `det`, `heatmap`, `line` | ver1 lifecycle과 current registry 결합 |

Checkpoint는 ver1처럼 model `state_dict`를 `.pth`로 저장한다. Optimizer와 scheduler resume state는 초기
범위에 포함하지 않는다.

## 12. 구현 순서와 검증 기준

권장 구현 순서는 다음과 같다.

1. ver1 data, loss, metric, core와 utility를 이관하고 parity를 검증한다.
2. Base wrapper와 `PredictionResult` 계약을 확정한다.
3. 공통 block, backbone, feature contract와 factory를 구현한다.
4. `reg`, `seg`, `heatmap`, `det` 순서로 wrapper를 연결한다.
5. External whole model, refinement와 rule-based wrapper를 연결한다.
6. scripts와 experiments를 통합하고 benchmark schema를 검증한다.

구현 완료 기준은 다음과 같다.

- 같은 seed에서 ver1 data split과 sample tensor가 일치한다.
- 모든 trainable wrapper가 model, loss, metric, optimizer, scheduler와 device를 소유한다.
- Core runner가 method 내부 구조 없이 wrapper lifecycle만 호출한다.
- Invalid component 조합이 silent fallback 없이 생성 오류를 반환한다.
- Checkpoint round trip 후 같은 입력에 같은 raw output을 생성한다.
- CLI와 Python API가 같은 resolved config와 output 경로를 사용한다.
- Accuracy, SR, failure reason과 end-to-end 비용을 같은 benchmark row에 기록한다.
