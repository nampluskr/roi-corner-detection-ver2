# 문서 안내

이 폴더는 프로젝트 설계, proposal, reference와 historical 문서를 분리해 관리한다. 새 설계 또는 구현
결정을 기록할 때는 canonical SSOT를 먼저 갱신한다.

현재 문서 구조는 다음과 같다.

```text
docs/
├── architecture/
│   └── model-assembly.md
├── deprecated/
│   ├── methods-cluade.md
│   ├── methods-codex.md
│   └── model-design.md
├── guides/
│   ├── model-usage-claude.md
│   ├── model-usage-codex.md
│   ├── src-layout-claude.md
│   └── src-layout-codex.md
├── references/
│   └── backbones.md
└── README.md
```

문서의 역할은 다음과 같다.

| 문서 | 상태 | 역할 |
|---|---|---|
| [architecture/model-assembly.md](architecture/model-assembly.md) | canonical | 프로젝트 전체 설계, method registry와 실험 비교의 SSOT |
| [guides/model-usage-claude.md](guides/model-usage-claude.md) | guide | ver1 CLI 기반 모델 성능 비교 안내 |
| [guides/model-usage-codex.md](guides/model-usage-codex.md) | proposal | 모델 구성, 실행과 성능 비교 사용 시나리오 |
| [guides/src-layout-claude.md](guides/src-layout-claude.md) | guide | ver1 skeleton 기반 `src/` 구조 안내 |
| [guides/src-layout-codex.md](guides/src-layout-codex.md) | proposal | ver1 기반 source 구조와 class 및 function 배치 제안 |
| [references/backbones.md](references/backbones.md) | reference | pretrained weight의 출처, checksum과 파일 정보 |
| [deprecated/](deprecated/) | deprecated | 이전 후보안과 historical terminology |

Guide와 proposal 문서는 canonical 설계를 구현하거나 사용하는 방안을 구체화하지만 SSOT보다 우선하지
않는다. Deprecated 문서는 기존 결정의 배경과 이름 변환을 보존할 뿐이다. 새 model, config, dataset
또는 experiment 결정은 [architecture/model-assembly.md](architecture/model-assembly.md)에 먼저
기록한다.
