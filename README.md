# roi-corner-detection-ver2

평면 사각형 객체의 4개 코너 좌표를 검출하는 11개 방법론을 하나의 공통 데이터, 학습,
평가 파이프라인에서 구현하고 비교하는 프로젝트다. `roi-corner-detection-ver1`을
리팩토링한 버전이며, dataset/data source/method/model/backbone 용어 분리와
`outputs/<dataset>/<method>/<model>/<exp_name>/` 산출물 경로 규칙을 도입한다.

## 1. 폴더 구조

```text
roi-corner-detection-ver2/
├ data/
│ ├ measured/
│ ├ public/
│ └ synthetic/
├ docs/
├ experiments/
│ ├ measured/
│ ├ public/
│ └ synthetic/
├ notebooks/
│ ├ measured/
│ ├ public/
│ └ synthetic/
├ outputs/
│ ├ measured/
│ ├ public/
│ └ synthetic/
├ scripts/
├ src/
│ ├ core/
│ ├ data/
│ ├ losses/
│ ├ metrics/
│ ├ models/
│ └ utils/
├ CLAUDE.md
├ PLAN.md
└ README.md
```

## 2. 방법론 (method)

평면 사각형 코너를 검출하는 11개 방법론을 출력 표현(후처리 계열)을 1차 축으로 분류한다.
공통 입출력 계약, 제약조건, 상세 분류 체계는 [docs/methods.md](docs/methods.md)를 참조한다.

| 출력 표현 | 방법론 |
|---|---|
| coord-regression | `direct`, `homography`, `foundation` |
| heatmap | `heatmap` |
| mask-contour | `seg` |
| box-detection | `det` |
| iterative-refine | `gcn`, `stn` |
| line-intersection | `line`, `hybrid` |
| set-prediction | `detr` (조건부) |

이전 18개 항목 중 `doc`/`vit`/`doctr`는 `direct`, `torchseg`는 `seg`, `torchdet`/`yolo`는
`det`, `classical`은 `hybrid`의 model로 흡수했다.

## 3. 핵심 제약

(추후 작성)

## 4. 데이터셋 단계 (dataset)

(추후 작성)

## 5. 평가 메트릭

(추후 작성)

## 6. CLI 인자와 실험 설정

(추후 작성)

## 7. 실험명 / checkpoint / 산출물 경로 규칙

(추후 작성)

## 8. Benchmark 결과 컬럼

(추후 작성)

## 9. src 모듈 시그니처

(추후 작성)

## 10. scripts 사용법

(추후 작성)

## 11. 참조 프로젝트

(추후 작성)
