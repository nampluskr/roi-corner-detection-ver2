# 공통 평가 metric과 prediction 저장 계획

다음 표는 이 plan의 상태와 적용 범위를 정리한다.

| 항목 | 값 |
|---|---|
| 상태 | Done |
| 작성일 | 2026-07-19 |
| 적용 범위 | `docs/architecture/model-assembly.md`, `src/metrics/`, `src/core/evaluator.py`, `src/core/predictor.py`, `scripts/evaluate.py`, `scripts/predict.py`, `experiments/run.py`, `experiments/configs.py` |
| 관련 문서 | [docs/architecture/model-assembly.md](../architecture/model-assembly.md), [0001-src-implementation-plan.md](0001-src-implementation-plan.md), [0022-heatmap-method-plan.md](0022-heatmap-method-plan.md) |

## 1. 목적과 배경

`reg`, `seg`, `det`, `heatmap` method가 모두 공통 corner contract `(N, 4, 2)`를 반환할 수 있게
되었으므로, 다음 단계는 method 간 성능을 같은 기준으로 비교하는 평가 기반을 만드는 것이다. 현재
학습 loop는 `history.json`과 `PolygonIoU` 중심이며, canonical metric bank에 정의된 MCD, MaxCD,
Reprojection Error, PCK, SR, latency와 model size는 아직 공통 실행 경로에 연결되어 있지 않다.

이번 plan은 새 method를 추가하지 않고, 모든 구현 method가 같은 test split과 같은 출력 파일 구조로
평가되도록 metric, evaluator, predictor와 batch runner를 정리한다.

## 2. 범위

이번 plan에 포함하는 항목은 다음과 같다.

- `src/metrics/`에 `CornerDistanceMetric`, `MeanCornerDistance`, `MaxCornerDistance`, `PCK`,
  `SuccessRate`를 추가한다.
- `ReprojectionError`는 homography 계산 안정성 검증이 필요하므로 interface만 문서화하고 구현은
  후속 plan으로 둔다.
- `src/core/evaluator.py`를 추가해 test loader에서 wrapper의 `eval_step` 또는 inference 경로로 공통
  metric dictionary를 산출하고 `metrics.json`을 저장한다.
- `src/core/predictor.py`를 추가해 image path, target corner, predicted corner와 success 여부를
  `predictions.csv`로 저장한다.
- `scripts/evaluate.py`를 추가해 checkpoint를 load하고 test split 평가를 수행한다.
- `scripts/predict.py`를 추가해 checkpoint를 load하고 prediction CSV를 생성한다.
- `experiments/run.py`에 `evaluate`, `predict` mode를 추가하고 기존 `CONFIGS` 기반 CLI argument 전달을
  재사용한다. checkpoint가 config에 없으면 해당 config의 기본 output path 아래 `model.pth`를 사용한다.
- `experiments/configs.py`에 method comparison config block을 추가하되, 기존 active run queue는
  변경하지 않는다.
- `docs/architecture/model-assembly.md`의 metric bank 상태를 현재 구현과 맞게 갱신한다.

이번 plan에서 제외하는 항목은 다음과 같다.

- 새 model method, 새 backbone, refinement 구현.
- raw tensor 저장. `reg` raw logits, `seg` mask logits, `det` native output, `heatmap` logits는 용량과
  method별 shape가 달라 후속 analysis plan에서 다룬다.
- postprocess failure reason의 완전한 구조화. 이번 plan에서는 NaN corner를 failure로 처리하고
  `failure_reason="invalid_prediction"`만 기록한다.
- Reprojection Error의 실제 수치 구현.
- benchmark 결과 해석 문서 작성. 이번 plan은 실행 기반을 만드는 데 한정한다.

## 3. 구현 결정

metric은 normalized coordinate 기준으로 계산한다. `targets`와 `preds`는 모두 `(N, 4, 2)`이며 corner
순서는 `TL`, `TR`, `BR`, `BL`로 이미 method boundary에서 정규화되었다고 본다.

공통 metric 기본 set은 다음과 같다.

| metric key | 계산 | 좋은 방향 |
|---|---|---|
| `iou` | 기존 `PolygonIoU` | 클수록 좋음 |
| `mcd` | sample별 네 corner Euclidean distance 평균의 dataset 평균 | 작을수록 좋음 |
| `maxcd` | sample별 네 corner Euclidean distance 최댓값의 dataset 평균 | 작을수록 좋음 |
| `pck_002` | corner distance가 `0.02` 이하인 corner 비율 | 클수록 좋음 |
| `pck_005` | corner distance가 `0.05` 이하인 corner 비율 | 클수록 좋음 |
| `sr` | NaN이 없는 prediction sample 비율 | 클수록 좋음 |

저장 파일은 다음 구조로 고정한다.

```text
outputs/<dataset>/<method>/<model>/<exp_name>/
-> history.json
-> model.pth
-> metrics.json
-> predictions.csv
```

`predictions.csv`는 다음 column 순서를 사용한다.

```text
index,success,failure_reason,
target_x1,target_y1,target_x2,target_y2,target_x3,target_y3,target_x4,target_y4,
pred_x1,pred_y1,pred_x2,pred_y2,pred_x3,pred_y3,pred_x4,pred_y4
```

image path는 현재 dataloader batch가 반환하지 않으므로 이번 plan의 prediction CSV에는 포함하지 않는다.
이미지 식별자는 split된 dataset 내 순번 `index`를 사용한다. image path 저장은 dataset batch contract
변경이 필요하므로 후속 plan으로 분리한다.

## 4. 구현 항목

구현은 다음 순서로 진행한다.

1. `src/metrics/corner_distance.py`에 MCD, MaxCD와 PCK metric class를 추가한다.
2. `src/metrics/success_rate.py`에 NaN 없는 prediction 비율 metric을 추가한다.
3. `src/core/evaluator.py`에 `DEFAULT_METRICS`, `Evaluator.evaluate()`와 `Evaluator.save()`를 추가한다.
4. `src/core/predictor.py`에 `Predictor.predict()`와 `Predictor.save()`를 추가한다.
5. `scripts/evaluate.py`와 `scripts/predict.py`에서 `parse_args()`, `get_output_dir()`,
   `get_wrapper_kwargs()`, `load_model()`을 재사용한다.
6. 모든 method wrapper의 기본 metric set을 `DEFAULT_METRICS`와 일치시키거나, trainer/evaluator가
   명시적으로 metric set을 주입하도록 정리한다.
7. `experiments/run.py`의 `MODES`를 `["train", "evaluate", "predict"]`로 확장한다.
8. `experiments/configs.py`에 `METHOD_COMPARISON_CONFIGS` template을 추가하고 `CONFIGS`에는 자동으로
   섞지 않는다.
9. canonical 문서의 metric bank 구현 상태와 저장 파일 규칙을 갱신한다.

## 5. 완료 기준

이 plan은 다음 조건을 만족하면 `Done`으로 볼 수 있다.

- 새 metric class가 NaN prediction을 명확히 처리하고, 정상 corner 입력에서 기대값을 반환한다.
- `Evaluator`가 `iou`, `mcd`, `maxcd`, `pck_002`, `pck_005`, `sr`을 포함한 dictionary를 반환한다.
- `scripts/evaluate.py --method reg --checkpoint <path>`가 `metrics.json`을 저장한다.
- `scripts/predict.py --method reg --checkpoint <path>`가 `predictions.csv`를 저장한다.
- `experiments/run.py --mode evaluate`와 `--mode predict`가 기존 config argument 전달 규칙으로 동작한다.
- `reg`, `seg`, `det`, `heatmap` wrapper가 evaluator와 predictor에서 같은 public interface로 실행된다.
- plan 문서 상태가 `Draft`에서 `Done`으로 갱신된다.

## 6. 검증

검증은 conda 환경 `pytorch_env`에서 수행한다.

```bash
conda activate pytorch_env
python -c "import numpy as np; from src.metrics.corner_distance import MeanCornerDistance, MaxCornerDistance, PCK; p=np.zeros((1,4,2), dtype=np.float32); t=np.zeros((1,4,2), dtype=np.float32); m=MeanCornerDistance(); m.update(p,t); print(m.compute()); x=MaxCornerDistance(); x.update(p,t); print(x.compute()); k=PCK(0.02); k.update(p,t); print(k.compute())"
python scripts/train.py --method reg --backbone custom --head coord_gap --train_size 2 --valid_size 2 --max_epochs 1 --batch_size 1 --num_workers 0 --save --output_dir /tmp/reg_eval_smoke
python scripts/evaluate.py --method reg --backbone custom --head coord_gap --test_size 2 --batch_size 1 --num_workers 0 --checkpoint /tmp/reg_eval_smoke/model.pth --output_dir /tmp/reg_eval_smoke
python scripts/predict.py --method reg --backbone custom --head coord_gap --test_size 2 --batch_size 1 --num_workers 0 --checkpoint /tmp/reg_eval_smoke/model.pth --output_dir /tmp/reg_eval_smoke
python -c "import json, csv; print(sorted(json.load(open('/tmp/reg_eval_smoke/metrics.json')).keys())); print(next(csv.reader(open('/tmp/reg_eval_smoke/predictions.csv'))))"
```

가능하면 같은 smoke를 `seg`, `det`, `heatmap`의 저장된 tiny checkpoint에도 반복한다. checkpoint가 없는
method는 `train_size=2`, `valid_size=2`, `max_epochs=1`로 먼저 생성한다.

## 7. 가정

이번 plan은 다음 기본값을 승인된 가정으로 둔다.

- metric은 모두 normalized coordinate 기준으로 계산한다.
- `sr`은 postprocessor가 NaN 없는 `(4, 2)` corner를 반환한 sample 비율로 정의한다.
- `BaseMetric.update()`가 NaN prediction을 제외하는 기존 동작은 distance metric과 IoU에 유지하고,
  `SuccessRate`만 NaN sample을 실패로 count한다.
- prediction CSV에는 이번 plan에서 image path를 포함하지 않는다.
- Reprojection Error와 latency/model size 자동 측정은 후속 plan으로 분리한다.
