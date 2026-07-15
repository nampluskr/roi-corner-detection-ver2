# roi-corner-detection-ver2

PMD OLED fringe 영상에서 평면 사각형 객체의 네 가상 corner를 검출하는 project다. 현재 workspace는
문서 설계 중심이며 implementation folder는 설계가 확정된 뒤 추가한다.

프로젝트 설계의 단일 기준은 [model-assembly.md](docs/architecture/model-assembly.md)다. 이 문서는
공통 입출력, 도메인 제약, data stage, method registry, model 조립과 평가 기준을 정의한다.

문서의 전체 안내와 상태는 [docs/README.md](docs/README.md)에서 확인할 수 있다. Pretrained weight의
파일 정보와 checksum은 [backbones.md](docs/references/backbones.md)에서 관리한다.

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
