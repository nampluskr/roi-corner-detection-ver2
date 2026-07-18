# reg method 최소 학습 파일 세트

| 항목 | 값 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-18 |
| 적용 범위 | `src/models/base/`, `src/models/backbones/`, `src/models/adapters/`, `src/models/blocks/`, `src/models/heads/`, `src/models/reg/`, `src/models/features.py`, `src/losses/`, `src/metrics/`, `src/utils/geometry.py`, `src/data/` |
| 관련 문서 | [docs/architecture/model-assembly.md](../architecture/model-assembly.md), [docs/plans/0001-src-implementation-plan.md](0001-src-implementation-plan.md) |

## 1. 목적과 배경

`260712_roi-corner-detection-ver2`에는 이미 [src/core/trainer.py](../../src/core/trainer.py)와
`src/core/factory.py`(`get_logger`)가 존재한다. 이 plan의 목적은 `Trainer`로 실제 학습을 돌릴 수
있는 최소 `reg` method 구현을 만드는 것이다.

`docs/plans/0001-src-implementation-plan.md`가 `src/` 전체(모든 method package, decoder, neck,
scripts, experiments)를 다루는 대형 계획이라 승인과 실행에 시간이 걸린다. 이 plan은 `0001`을
`Draft` 상태로 보류한 채, `reg`만 좁게 떼어내 먼저 학습 가능한 상태로 만든다.

SSOT([docs/architecture/model-assembly.md](../architecture/model-assembly.md)) Section 3.2,
3.4, 3.5, 4.1, 4.2에 따르면 `CustomRegModel`은 다음 조합이다.

```text
CustomBackbone -> backbone adapter -> FeatureBundle -> coord_gap head -> raw output (B, 8)
```

`coord_gap` variant는 `FeatureBundle.global_feature`에 dropout과 linear projection만 적용한다.
decoder나 neck을 쓰지 않아 SSOT Section 11.4 ablation matrix의 1단계이자 구현 복잡도가 가장
낮다.

ver1(`260701_roi-corner-detection-ver1`)의 `direct/model.py`는 torchvision resnet whole model을
그대로 쓰는 구조라 `CustomBackbone` 기반 계약과 다르므로 재사용하지 않는다. 반면 ver1의
`models/base/*`, `losses/wing_loss.py`, `metrics/polygon_iou.py`, `utils/geometry.py`,
`data/*`는 이미 SSOT와 호환되는 계약이라 그대로 이관한다.

## 2. 범위

포함:
- `src/models/base/`: `base_model.py`, `base_wrapper.py`, `base_preprocessor.py`,
  `base_postprocessor.py` (ver1 그대로 이관, 수정 없음)
- `src/models/blocks/conv_block.py`: `ConvBlock` 신규 작성 (SSOT 3.2절 계약)
- `src/models/backbones/`: `base_backbone.py`(`BaseBackbone`), `custom_backbone.py`
  (`CustomBackbone`, SSOT 3.4절 stem + 4-stage, stride 16)
- `src/models/adapters/`: `base_adapter.py`(`BaseBackboneAdapter`), `cnn_adapter.py`
  (`CNNBackboneAdapter`, native feature -> `FeatureBundle.global_feature`)
- `src/models/features.py`: `FeatureBundle`, `FeatureSpec`(reg에 필요한 `global` capability만),
  `FeatureExtractor`
- `src/models/heads/coordinate_head.py`: `CoordGapHead`만 (spatial variant는 제외)
- `src/models/reg/`: `model.py`(`CustomRegModel`), `preprocessor.py`(`RegPreprocessor`),
  `postprocessor.py`(`RegPostprocessor`), `wrapper.py`(`RegWrapper`)
- `src/losses/`: `base_loss.py`, `wing_loss.py` (ver1 그대로 이관)
- `src/metrics/`: `base_metric.py`, `polygon_iou.py` (ver1 그대로 이관)
- `src/utils/geometry.py` (ver1 그대로 이관, `polygon_area` 의존)
- `src/data/`: `dataset.py`, `dataloader.py`, `transforms.py` (ver1 그대로 이관)
- `src/` 및 각 신규 하위 폴더에 빈 `__init__.py`

제외 (후속 plan에서 수행):
- 학습/synthetic 데이터 준비와 실제 학습 실행
- `coord_spatial` head variant, `TorchvisionBackbone`, `TransformerBackboneAdapter`
- `seg`/`det`/`heatmap`/`line`/`external`/`refinement`/`rule_based` 등 다른 method package
- `decoders/`, `necks/`, `models/factory.py`의 `get_decoder`/`get_neck`/`get_head`/`get_model`
  (reg는 decoder/neck을 쓰지 않으므로 이번 범위에서 불필요)
- `PredictionResult` 계약과 `core/evaluator.py`/`core/predictor.py` 갱신 (`0001` Phase 2 범위)
- `scripts/`, `experiments/`
- `0001-src-implementation-plan.md`의 나머지 Phase 전체 (그대로 `Draft` 유지)

## 3. 완료 기준

- 위 "범위 - 포함" 목록의 모든 파일이 존재하고, 각 신규 `src/` 하위 폴더에 빈 `__init__.py`가
  있을 것
- `RegWrapper`가 `BaseWrapper`를 상속하고 `train_step`/`eval_step`/`predict_step`을
  `src/core/trainer.py`의 `Trainer.train`/`evaluate`/`fit`이 그대로 호출 가능한 인터페이스로
  제공할 것
- `CustomRegModel`이 `FeatureExtractor`(backbone + adapter + spec)와 `CoordGapHead`만으로
  구성되고 decoder/neck에 의존하지 않을 것
- 모든 신규 파일이 AGENTS.md/CLAUDE.md 코드 규칙(파일 헤더 형식, 한국어 미사용, `os.path` 사용,
  type hint 미사용, `src.xxx` absolute import, 클래스/top-level function 한 줄 docstring)을
  따를 것
- `PredictionResult` 계약을 도입하지 않고 ver1과 동일하게 bare tensor/numpy array 반환 형태를
  유지할 것

## 4. 검증

실제 학습 데이터(synthetic CSV 등)가 이번 plan 범위에 없으므로 자동 smoke test는 포함하지
않는다. 대신 다음을 수동으로 확인한다.

- `RegWrapper()`가 예외 없이 생성되는지 (torch가 설치된 환경에서 직접 import 및 생성 확인)
- `RegWrapper`의 `train_step`/`eval_step`/`predict_step` 시그니처가 `BaseWrapper`와 일치하고
  `Trainer.train`/`Trainer.evaluate`/`Trainer.fit`이 그대로 호출 가능한 구조인지 코드 리뷰로
  확인
- `CustomBackbone` -> `CNNBackboneAdapter` -> `FeatureExtractor`가 만드는 `FeatureBundle`의
  `global_feature`가 `CoordGapHead`의 입력 channel과 일치하는지 코드 리뷰로 확인

## 참고 파일

- [docs/architecture/model-assembly.md](../architecture/model-assembly.md) Section 3.2, 3.4,
  3.5, 4.1, 4.2 (SSOT)
- [docs/plans/0001-src-implementation-plan.md](0001-src-implementation-plan.md) (보류 중인 전체
  계획, 구조와 네이밍 일관성 참고용)
- ver1 `src/models/base/*.py`, `src/models/direct/*.py`, `src/losses/wing_loss.py`,
  `src/metrics/polygon_iou.py`, `src/utils/geometry.py`, `src/data/*.py` (재사용 대상 실제 계약)
- 이미 존재하는 `src/core/trainer.py`, `src/core/factory.py`
