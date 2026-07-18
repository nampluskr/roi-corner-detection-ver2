# src/ 구현

| 항목 | 값 |
| --- | --- |
| 상태 | Draft |
| 작성일 | 2026-07-18 |
| 적용 범위 | `src/` 전체 신규 구현(core, data, losses, metrics, models, utils) |
| 관련 문서 | [docs/architecture/model-assembly.md](../architecture/model-assembly.md), [docs/guides/src-layout-codex.md](../guides/src-layout-codex.md), [docs/guides/src-layout-claude.md](../guides/src-layout-claude.md) |

## 1. 목적과 배경

`260712_roi-corner-detection-ver2`는 아직 `src/`가 존재하지 않는다. 프로젝트 전체 설계의 SSOT는
[docs/architecture/model-assembly.md](../architecture/model-assembly.md)이며, 이를 실제 구현에
대응시키는 두 개의 proposal 문서(`docs/guides/src-layout-codex.md`,
`docs/guides/src-layout-claude.md`)가 이미 존재한다. 두 문서는 레이아웃(component를 `models/`
하위에 nest할지, top-level에 형제로 둘지)과 세부 패키지 구성(특히 external whole model 흡수 방식,
rule-based 패키지 이름)에서 차이가 있다.

사용자는 codex안을 기반으로 재검토한 구조를 채택하기로 했고, 다음 네 가지를 확정했다.

- 레이아웃: codex안 채택. `backbones/adapters/blocks/decoders/heads/necks/features.py`를 모두
  `src/models/` 하위에 둔다. SSOT Section 3.5의 component 경계가 model-internal part로만
  언급되는 것과 일치한다.
- external whole model(ver1의 `torchseg`, `torchdet`) 흡수 방식: `models/external/`로 분리한
  독립 패키지. SSOT Section 2.1이 Category C를 별도 조립 카테고리로 명시하므로, `seg`/`det`
  패키지 내부에 `usage` 플래그로 분기하지 않는다.
- 규칙 기반(F 카테고리) 패키지명: `rule_based`.
- 구현 착수 순서: `reg`가 SSOT Section 11.4 ablation matrix의 1단계이자 decoder 없이
  FeatureExtractor + coordinate head만 필요해 복잡도가 가장 낮으므로 첫 method package로 삼는다.

ver1(`260701_roi-corner-detection-ver1`)은 wrapper 기반 lifecycle(`BaseWrapper.train_step` /
`eval_step` / `predict_step`, `core/trainer.py`, `core/factory.py`)이 이미 SSOT와 호환되는 형태로
구현되어 있음을 코드로 직접 확인했다(`base_wrapper.py`, `core/factory.py`, `direct/model.py`,
`scripts/config.py`). 이 계획은 ver1을 백지에서 다시 만들지 않고, ver1 lifecycle 위에 SSOT가 새로
요구하는 조립 primitive(FeatureBundle/FeatureSpec, 공통 backbone/adapter/decoder/head, category
경계에 맞는 method 패키지 재편)를 얹는 방식으로 진행한다.

## 2. 범위

포함:
- `src/` 전체 디렉터리 구조 신설(core, data, losses, metrics, models, utils)
- ver1에서 재사용 가능한 모듈 이관(data, losses, metrics, utils, core 대부분)
- `PredictionResult` 계약 신설과 관련 모듈(`base_wrapper`, `evaluator`, `predictor`) 갱신
- 조립 primitive 신규 구현(blocks, backbones, adapters, features.py, decoders, necks, heads,
  factory.py)
- method package 구현: `reg`, `seg`, `heatmap`, `det`, `line`, `external`, `refinement`,
  `rule_based`
- `scripts/config.py` 확장과 `scripts/train.py`/`evaluate.py`/`predict.py`,
  `experiments/configs.py`/`run.py`/`benchmark.py` adapt
- Phase 1/3/4 smoke test의 `tests/` 정식화

제외 (후속 plan에서 수행):
- timm 기반 backbone(`timm_backbone.py`) 도입
- `hybrid/` 성격의 실험(현재는 `seg`/`line`의 postprocessor variant로 흡수하며 별도 패키지를
  만들지 않음)
- 신규 dataset 추가나 `data/` 구조 변경
- 실제 학습 실행과 실험 결과 비교(별도 실행 plan에서 다룬다)

## 3. 최종 src/ 구조

```text
src/
├ core/                     # ver1 재사용, factory.get_wrapper만 수정
│  ├ __init__.py
│  ├ evaluator.py
│  ├ factory.py
│  ├ predictor.py
│  └ trainer.py
├ data/                     # ver1 그대로 재사용
│  ├ __init__.py
│  ├ dataloader.py
│  ├ dataset.py
│  ├ images.py
│  ├ midv2020.py
│  ├ smartdoc.py
│  └ transforms.py
├ losses/                   # ver1 재사용
│  ├ __init__.py
│  ├ base_loss.py
│  ├ bce_loss.py
│  ├ cross_entropy_loss.py
│  ├ dice_loss.py
│  ├ focal_loss.py
│  ├ mse_loss.py
│  ├ smooth_l1_loss.py
│  └ wing_loss.py
├ metrics/                  # ver1 재사용
│  ├ __init__.py
│  ├ base_metric.py
│  ├ max_cd.py
│  ├ mcd.py
│  ├ pck.py
│  ├ polygon_iou.py
│  ├ reprojection_error.py
│  └ success_rate.py
├ models/
│  ├ adapters/              # 신규: native feature -> FeatureBundle
│  │  ├ __init__.py
│  │  ├ base_adapter.py
│  │  ├ cnn_adapter.py
│  │  └ transformer_adapter.py
│  ├ backbones/             # 신규
│  │  ├ __init__.py
│  │  ├ base_backbone.py
│  │  ├ custom_backbone.py
│  │  └ torchvision_backbone.py
│  ├ base/                  # ver1 재사용 + PredictionResult 신규
│  │  ├ __init__.py
│  │  ├ base_model.py
│  │  ├ base_postprocessor.py
│  │  ├ base_preprocessor.py
│  │  ├ base_wrapper.py
│  │  └ prediction_result.py
│  ├ blocks/                # 신규
│  │  ├ __init__.py
│  │  ├ conv_block.py
│  │  └ deconv_block.py
│  ├ decoders/               # 신규
│  │  ├ __init__.py
│  │  ├ fpn_decoder.py
│  │  ├ plain_decoder.py
│  │  └ unet_decoder.py
│  ├ det/                   # ver1 det 재편
│  │  ├ __init__.py
│  │  ├ model.py
│  │  ├ postprocessor.py
│  │  ├ preprocessor.py
│  │  └ wrapper.py
│  ├ external/              # 신규: ver1 torchseg + torchdet 흡수, Category C
│  │  ├ __init__.py
│  │  ├ detection.py
│  │  ├ segmentation.py
│  │  └ wrapper.py
│  ├ heads/                 # 신규
│  │  ├ __init__.py
│  │  ├ coordinate_head.py
│  │  ├ detection_head.py
│  │  ├ heatmap_head.py
│  │  └ mask_head.py
│  ├ heatmap/                # ver1 재사용, primitive로 재조립
│  │  ├ __init__.py
│  │  ├ model.py
│  │  ├ postprocessor.py
│  │  ├ preprocessor.py
│  │  └ wrapper.py
│  ├ line/                  # ver1 재사용, primitive로 재조립
│  │  ├ __init__.py
│  │  ├ model.py
│  │  ├ postprocessor.py
│  │  ├ preprocessor.py
│  │  └ wrapper.py
│  ├ necks/                 # 신규
│  │  ├ __init__.py
│  │  └ multi_scale_neck.py
│  ├ refinement/             # ver1 gcn 흡수 + local_stn 신규, Category D
│  │  ├ __init__.py
│  │  ├ gcn.py
│  │  ├ local_stn.py
│  │  └ wrapper.py
│  ├ reg/                   # ver1 direct+homography+doc+foundation 흡수
│  │  ├ __init__.py
│  │  ├ model.py
│  │  ├ postprocessor.py
│  │  ├ preprocessor.py
│  │  └ wrapper.py
│  ├ rule_based/            # 신규, Category F
│  │  ├ __init__.py
│  │  ├ contour.py
│  │  ├ line.py
│  │  └ wrapper.py
│  ├ seg/                   # ver1 seg 재편, torchseg는 external/로 분리
│  │  ├ __init__.py
│  │  ├ model.py
│  │  ├ postprocessor.py
│  │  ├ preprocessor.py
│  │  └ wrapper.py
│  ├ __init__.py
│  ├ factory.py             # 신규: get_backbone/get_adapter/get_decoder/get_neck/get_head/get_model
│  └ features.py            # 신규: FeatureSpec, FeatureBundle, FeatureExtractor
├ utils/                    # ver1 그대로 재사용
│  ├ __init__.py
│  ├ geometry.py
│  ├ homography.py
│  ├ io.py
│  ├ measure.py
│  └ plot.py
└ __init__.py
```

`ver1`의 `hybrid/`는 SSOT Category E(learned geometry hybrid)에 대응하지만, `seg`/`line`의 raw
output에 다른 geometry postprocessor를 적용하는 실험 축(SSOT Section 9)이므로 별도 method
package가 아니라 `seg`/`line`의 postprocessor variant로 흡수한다. 별도 `hybrid/` 패키지는 만들지
않는다. `timm_backbone.py`는 timm 의존성 도입 시점까지 보류하고 이번 plan 범위에서는 만들지 않는다
(torchvision backbone으로 Category B를 먼저 검증).

## 4. 구현 순서

### Phase 0: scaffolding

모든 디렉터리와 빈 `__init__.py`만 생성한다. 로직 없음.

### Phase 1: ver1 이관 (거의 그대로)

재사용: `data/`, `losses/`, `metrics/`, `utils/`, `core/trainer.py`, `core/evaluator.py`,
`core/predictor.py`, `models/base/base_model.py`, `models/base/base_preprocessor.py`.

최소 수정: `models/base/base_wrapper.py`(Phase 2에서 PredictionResult 인식하도록 조정),
`models/base/base_postprocessor.py`(docstring만 갱신), `core/factory.py`의 `get_wrapper` dispatch
table을 SSOT registry code(`reg`, `seg`, `det`, `heatmap`, `line`, `external_seg`, `external_det`,
`refinement`, `rule_based`)로 재작성.

검증: 가짜 `nn.Identity` model을 `BaseWrapper`에 연결해 tiny synthetic CSV로 `Trainer.fit` 1
epoch 실행.

### Phase 2: PredictionResult와 contract 확정

`models/base/prediction_result.py`에 `PredictionResult`(fields: `corners`, `success`,
`failure_reason`, method `to_numpy()`)를 정의한다. `BaseWrapper.compute_metrics`/`predict_step`,
`core/evaluator.py`, `core/predictor.py`가 `PredictionResult`와 기존 bare-tensor 반환 모두를
처리하도록 갱신한다. 실패 표본은 accuracy metric에서 제외하되 SR과 failure reason 분포에는
포함한다(SSOT Section 1.3, 9.3).

이 단계는 어떤 method package보다 먼저 끝나야 한다. Phase 5의 모든 wrapper가 이 계약에 의존한다.

### Phase 3: composition primitive

순서: `blocks/conv_block.py` (`ConvBlock`) -> `blocks/deconv_block.py` (`DeconvBlock`,
`interpolate_conv` 기본) -> `backbones/base_backbone.py` (`BaseBackbone`) ->
`backbones/custom_backbone.py` (`CustomBackbone`, stage 4개 stride 16) ->
`backbones/torchvision_backbone.py` (`TorchvisionBackbone`) -> `adapters/base_adapter.py`
(`BaseBackboneAdapter`) -> `adapters/cnn_adapter.py` (`CNNBackboneAdapter`) ->
`adapters/transformer_adapter.py` (`TransformerBackboneAdapter`) -> `models/features.py`
(`FeatureSpec`, `FeatureBundle`, `FeatureExtractor`) -> `models/factory.py`의 `get_backbone`,
`get_adapter`만 먼저 구현.

핵심 계약(SSOT Section 3.5):

```python
class FeatureBundle:
    global_feature   # (B, C) or None
    spatial_feature  # (B, C, H, W) or None
    stages           # list[(B, Ci, Hi, Wi)] or None

class FeatureSpec:
    backbone_name, adapter_name
    global_channels, spatial_channels, spatial_stride
    stage_channels, stage_strides
    has_global, has_spatial, has_stages
    def require(self, capability): ...  # raises named error if missing
```

검증: `CustomBackbone`+`CNNBackboneAdapter`와 `TorchvisionBackbone("resnet18")`+`CNNBackboneAdapter`가
서로 다른 channel width에도 불구하고 같은 `FeatureBundle` field 계약을 만족하는지 assert.

### Phase 4: decoder, neck, head, model factory 완성

순서: `decoders/plain_decoder.py` (`PlainSegDecoder`) -> `decoders/unet_decoder.py`
(`UNetSegDecoder`, skip `add`/`concat`) -> `decoders/fpn_decoder.py` (`FPNSegDecoder`, 뒤에 구현,
조건부 candidate) -> `necks/multi_scale_neck.py` (`MultiScaleNeck`) -> `heads/coordinate_head.py`
(`CoordGapHead`, `CoordSpatialHead`) -> `heads/mask_head.py` (`MaskHead`) ->
`heads/heatmap_head.py` (`HeatmapHead`) -> `heads/detection_head.py` (`DetectionHead`) ->
`models/factory.py`의 `get_decoder`, `get_neck`, `get_head`, `get_model` 완성.

각 factory 함수는 `FeatureSpec`을 검증하고 지원하지 않는 조합에 요청 component, 필요 capability,
실제 `FeatureSpec`을 명시한 생성 시점 오류를 발생시킨다(SSOT Section 11.2, "no silent fallback").

검증: SSOT Section 11.2의 금지 조합 두 가지가 반드시 오류를 내는지 assert.
- `unet`/`fpn` decoder + `stages=None`인 backbone
- `plain` decoder + `skip_connection="add"`

이 테스트가 계획 전체에서 SSOT 준수를 가장 직접적으로 검증하는 항목이다.

### Phase 5: method package (reg 먼저)

구현 순서: `reg` -> `seg`(ver1 seg 흡수) -> `heatmap` -> `det` -> `line` -> `external`(ver1
torchseg + torchdet 흡수) -> `refinement`(ver1 gcn 흡수, `local_stn` 신규, base postprocess 실패
표본은 skip) -> `rule_based`(신규, `contour.py`/`line.py`, `wrapper.py`는 optimizer/scheduler
없이 no-op `train_step`).

`reg/model.py`는 ver1 `direct/model.py`(backbone+head 직결 구조)를 `FeatureExtractor` +
`heads/coordinate_head.py`로 교체한 형태다. `reg/wrapper.py`는 ver1 `direct/wrapper.py`,
`homography/wrapper.py`를 target variant(`corners`, `homography_offsets`)로 병합한다.

검증(reg 완료 시점): `RegWrapper.train_step`/`predict_step`이 Phase 1 스타일 smoke test를
통과하고 `core/factory.py`의 `get_wrapper("reg", ...)`가 정상 dispatch되는지 확인.

### Phase 6: scripts, experiments, config

`scripts/config.py`를 ver1에서 확장한다(교체하지 않음). 우선순위: `DEFAULTS` < YAML file <
`--set` override < 명시적 함수 kwargs.

- `DEFAULTS`에 `dataset`(`public`/`synthetic`/`measured`), `model` key 추가
- `DEFAULT_BACKBONES`를 SSOT registry code로 재키
- `get_output_dir(cfg, base="outputs")`를 `outputs/<dataset>/<method>/<model>/<exp_name>/`로 변경.
  이는 AGENTS.md/CLAUDE.md 자체의 hard requirement이므로 첫 커밋부터 정확해야 한다.
- `get_experiment(cfg)`에 decoder/skip/backbone 축을 반영해 ablation run을 구분 가능하게 확장
- 신규: `load_config(path)`, `apply_overrides(cfg, set_args)`, `resolve_config(args)`
- `parse_args()`에 `--config`, `--set` 플래그 추가

`scripts/train.py`/`evaluate.py`/`predict.py`, `experiments/configs.py`/`run.py`를 ver1에서
adapt한다. `experiments/benchmark.py`는 latency/model size metric 측정을 위해 신규 또는 adapt.

### Phase 7: 최소 테스트

Phase 1/3/4의 smoke test를 `tests/`로 정식화한다.

## 5. 완료 기준

- 위 최종 src/ 구조의 모든 디렉터리와 파일이 존재하고 각 `src/` 하위 폴더에 빈 `__init__.py`가
  있을 것
- Phase 1의 1 epoch smoke test(fake `nn.Identity` model + tiny synthetic CSV)가 통과할 것
- Phase 3의 `FeatureBundle`/`FeatureSpec` 계약 assert(서로 다른 backbone 간 필드 계약 일치)가
  통과할 것
- Phase 4의 SSOT Section 11.2 금지 조합 두 가지가 생성 시점 오류를 낼 것(가장 중요한 검증)
- Phase 5에서 `RegWrapper.train_step`/`predict_step`이 smoke test를 통과하고
  `core/factory.py`의 `get_wrapper("reg", ...)`가 정상 dispatch될 것
- `scripts/train.py`/`evaluate.py`/`predict.py`가 `outputs/<dataset>/<method>/<model>/<exp_name>/`
  경로 규칙을 정확히 따를 것
- 코드 작성 규칙(AGENTS.md Section 4: 한국어 미사용, `os.path` 사용, type hint 미사용, 파일 헤더
  형식, `src.xxx` absolute import 등)을 모든 신규 파일이 따를 것

## 6. 검증

각 phase는 다음 SSOT 요구사항에 직접 대응한다.

| phase | 검증 | SSOT 근거 |
|---|---|---|
| 1 | data/trainer parity, 1 epoch fit | Section 2 공통 계약 |
| 3 | FeatureBundle/FeatureSpec이 backbone마다 channel 강제 통일 없이 같은 필드 계약을 만족 | Section 3.5 |
| 4 | 금지 조합 두 가지가 생성 오류 반환 (가장 중요) | Section 11.2 |
| 5 | RegWrapper train/predict step + factory dispatch | Section 11.4 ablation 1단계 |
| 6 이후 | CLI train/evaluate/predict가 정확한 output 경로와 CSV header 생성 | Section 1.3, AGENTS.md 경로 규칙 |

Phase별 검증은 pytest 기반 smoke test 또는 스크립트 실행으로 수행하며, 각 phase 구현이 끝나는
시점에 해당 검증을 실행하고 결과를 기록한다.

## 참고 파일

- [docs/architecture/model-assembly.md](../architecture/model-assembly.md) (SSOT)
- [docs/guides/src-layout-codex.md](../guides/src-layout-codex.md) (채택한 레이아웃의 원안)
- ver1 `src/models/base/base_wrapper.py`, `src/core/factory.py`, `src/models/direct/model.py`,
  `scripts/config.py` (재사용 대상 실제 계약)
