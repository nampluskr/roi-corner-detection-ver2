@/home/nampl/.codex/RTK.md

# 프로젝트 에이전트 지침

## 1. Canonical project design

프로젝트 전체 설계의 단일 기준은
[docs/architecture/model-assembly.md](docs/architecture/model-assembly.md)다. Architecture, method
registry, 공통 입출력, data stage, metric, domain constraint와 experiment comparison을 변경할 때는
먼저 이 문서를 갱신한다.

문서의 상태와 역할은 다음과 같다.

| 문서 | 상태 | 역할 |
|---|---|---|
| `docs/architecture/model-assembly.md` | canonical | 프로젝트 전체 설계의 SSOT |
| `docs/references/backbones.md` | reference | weight 출처, checksum과 파일 정보 |
| `docs/deprecated/*.md` | deprecated | 이전 후보안과 historical reference |

Deprecated 문서는 새 구현, config 또는 experiment의 근거로 사용하지 않는다. 문서 구조의 상세는
`docs/README.md`를 따른다.

## 2. 작업 범위와 산출물

구현과 문서는 현재 workspace에 존재하는 범위에서 작업한다. 새 data, src, experiments 또는 outputs
folder는 사용자 요청 또는 canonical design의 구현 단계가 없으면 만들지 않는다.

실험 산출물 경로는 다음 규칙을 사용한다.

```text
outputs/<dataset>/<method>/<model>/<exp_name>/
```

`dataset`은 `public`, `synthetic`, `measured`의 논리 stage다. 자세한 method, model과 variant의
의미는 canonical design을 따른다.

## 3. 문서 작성 규칙

모든 Markdown 문서는 다음 규칙을 따른다.

- 본문은 서술체를 사용한다.
- em dash, 유니코드 화살표, 이모지를 사용하지 않는다.
- Markdown 본문의 화살표는 `$\to$`를 사용한다.
- fenced code block과 inline code 안에서는 ASCII `->`를 사용한다.
- 폴더 구조 tree는 `├ ─ │ └` 문자를 사용한다.
- header level을 건너뛰지 않고 H4 아래 level은 사용하지 않는다.
- 수평 구분선은 사용하지 않는다. YAML frontmatter의 `---`는 예외다.
- table과 list 앞에는 내용을 소개하는 문장을 둔다.
- 폴더와 파일 목록은 폴더를 알파벳순으로 먼저 나열하고 파일을 알파벳순으로 나열한다.
- Jupyter notebook cell의 `source` 배열 마지막 원소는 줄바꿈으로 끝나지 않는다.

## 4. 코드 작성 규칙

모든 Python 코드는 다음 규칙을 따른다.

- 식별자, 주석, docstring, 문자열에 한국어를 사용하지 않는다.
- 세로 정렬을 위한 불필요한 공백을 넣지 않는다.
- 경로 처리는 `pathlib.Path` 대신 `os.path`를 사용한다.
- type hint를 사용하지 않는다.
- 모든 파일의 첫 줄은 `# path/from/project/root.py: one-line description` 형식으로 작성한다.
- 첫 줄 header 다음에 빈 줄 하나를 두고 import를 작성한다.
- class와 top-level function은 한 줄 docstring을 작성한다.
- method에는 docstring을 작성하지 않는다.
- 주석은 필요한 경우에만 최소한으로 작성한다.
- `src/` 아래 모든 폴더에는 빈 `__init__.py`를 둔다.
- `src/` 내부 import는 `src.xxx` 형식의 absolute import를 사용한다.
- `scripts/`, `experiments/`에서는 project root를 `sys.path`에 추가한 뒤 `src.xxx`로 import한다.

## 5. 실행 환경

Python 실행과 검증은 conda 환경 `pytorch_env`를 사용한다. 코드 실행, `python -c` 검증,
스크립트 실행 전에 먼저 이 환경을 활성화한다.

```bash
conda activate pytorch_env
```

## 6. Plan 문서 규칙

Canonical 문서를 갱신할 정도의 실제 작업(구현, 구조 변경, 문서 체계 확장 등)은 실행 전에
`docs/plans/`의 계획 문서에 범위와 완료 기준을 기록하고 검토와 승인을 받는다. 요구사항이나 설계가
바뀌면 코드보다 canonical 문서를 먼저 수정한다.

계획 문서의 경로와 명명 규칙은 다음과 같다.

- 경로: `docs/plans/NNNN-topic-plan.md`
- 번호 `NNNN`은 4자리 0-padding이며 순증가한다. 번호를 재사용하거나 삭제하지 않는다.
- 상태는 `Draft`, `Approved`, `Done` 중 하나를 사용한다.
- 완료된 plan도 파일을 지우지 않고 이력으로 보존한다.

각 plan 문서는 다음 구성 요소를 갖춘다.

- 표준 헤더 표: 상태, 작성일, 적용 범위, 관련 문서
- 목적과 배경: 이 작업이 왜 필요한지
- 범위: 포함 항목과 제외 항목(후속 plan에서 수행)을 구분해서 기록
- 완료 기준: 무엇이 충족되면 이 plan을 `Done`으로 볼 수 있는지
- 검증: 빌드/테스트 방법, 또는 문서 생성만이면 그 사실과 확인 항목

작업 순서는 다음과 같다.

1. 작업 전에 관련 plan이 있는지 `docs/plans/`를 확인한다.
2. 없으면 새 `NNNN-topic-plan.md` 초안을 작성해 사용자에게 검토와 승인을 받는다.
3. 승인된 plan을 기준으로 canonical 문서를 먼저 갱신하고, 이후 코드를 구현한다.
4. 구현 결과에 맞게 관련 검증을 수행하고 plan의 상태를 갱신한다(`Draft` -> `Approved` -> `Done`).

## 7. 동기화 규칙

`CLAUDE.md`와 `AGENTS.md`는 같은 작업 지침의 동기화 사본이다. 한 파일의 내용이 변경되면 같은
작업에서 다른 파일을 동일한 내용으로 갱신하고 SHA-256으로 byte-level 일치를 검증한다.
