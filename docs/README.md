# 문서 안내

이 폴더는 프로젝트 설계, reference와 historical 문서를 분리해 관리한다. 새 설계 또는 구현 결정을
기록할 때는 canonical SSOT를 먼저 갱신한다.

현재 문서 구조는 다음과 같다.

```text
docs/
├── architecture/
│   └── model-assembly.md
├── deprecated/
│   ├── methods-cluade.md
│   ├── methods-codex.md
│   └── model-design.md
├── references/
│   └── backbones.md
└── README.md
```

활성 문서의 역할은 다음과 같다.

| 문서 | 상태 | 역할 |
|---|---|---|
| [architecture/model-assembly.md](architecture/model-assembly.md) | canonical | 프로젝트 전체 설계, method registry와 실험 비교의 SSOT |
| [references/backbones.md](references/backbones.md) | reference | pretrained weight의 출처, checksum과 파일 정보 |
| [deprecated/](deprecated/) | deprecated | 이전 후보안과 historical terminology |

Deprecated 문서는 기존 결정의 배경과 이름 변환을 보존할 뿐이다. 새 model, config, dataset 또는
experiment 결정은 [architecture/model-assembly.md](architecture/model-assembly.md)에만 기록한다.
