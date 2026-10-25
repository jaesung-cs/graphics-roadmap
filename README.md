<div align="center">

# Graphics Roadmap

**Stable Diffusion 을 이해하기 위한 오픈소스 독학 커리큘럼**

[![Lessons](https://img.shields.io/badge/lessons-17-2dd4bf)](#커리큘럼)
[![Hours](https://img.shields.io/badge/hours-35-818cf8)](#커리큘럼)
[![Lessons license](https://img.shields.io/badge/lessons-CC%20BY--NC--SA%204.0-lightgrey)](lessons/LICENSE.md)
[![Site license](https://img.shields.io/badge/site-MIT-blue)](LICENSE)
[![Node](https://img.shields.io/badge/node-%E2%89%A520.19-339933?logo=node.js&logoColor=white)](https://nodejs.org/)

[예시 사이트](https://jaesung-cs.github.io/graphics-roadmap/) · [설치와 실행](#설치와-실행) · [커리큘럼](#커리큘럼) · [강의와 퀴즈](#강의와-퀴즈) · [기여](#기여) · [라이선스](#라이선스)

</div>

## 소개

글 한 줄로 이미지를 만드는 Stable Diffusion 을 바닥부터 이해하기 위한 독학 커리큘럼입니다. vector·미분·확률에서 출발해 신경망의
학습, CNN·Transformer·CLIP, VAE·DDPM·flow matching 을 거쳐 U-Net, latent diffusion, text conditioning, classifier-free guidance,
sampler 까지 Stable Diffusion 의 부품을 하나씩 읽어 나갑니다. 리포를 fork 해 지도를 띄우면 강의를 읽고 퀴즈를 풀며 혼자서 진도를
이어 갈 수 있습니다.

이 리포는 [Robotics Roadmap](https://github.com/woolimi/robotics-roadmap)(Woolim Park)을 fork 해 고친 것입니다. 무엇을 고쳤는지는
[라이선스](#라이선스)에 적었습니다.

## 특징

- **fork 해서 혼자 진행하는 자율형 학습**: 강의를 읽고 퀴즈를 풀면 결과가 내 fork 의 `progress/` 에 쌓이고, 다시 풀 때는 틀린 원리만
  다른 문항으로 묻습니다. 퀴즈가 없는 Stable Diffusion 강의는 다 읽은 뒤 "다 읽음" 으로 표시합니다.
- **기술의 의존 관계가 보이는 지도**: 강의가 선수 관계로 이어져 있어서, Stable Diffusion 의 text conditioning 을 이해하려면 CLIP 과
  latent diffusion 을, latent diffusion 을 이해하려면 VAE 와 U-Net 을 먼저 읽어야 한다는 것이 지도에서 바로 보입니다.
- **논문 기반 강의**: 강의마다 근거 논문을 밝히고 그 방법과 숫자를 따라갑니다.
- **숫자와 그림으로 설명**: 코드 없이 손으로 따라 할 숫자 예시와 그림으로 설명하고, 긴 유도는 접어 둔 "심화" 블록에 있습니다.
- **업데이트 알림**: 이 리포에 새 강의가 올라오면 지도 화면에서 버튼 하나로 내 fork 에 합칩니다.

## 설치와 실행

[Node.js](https://nodejs.org/) 20.19 이상(또는 22.12 이상)과 git 이 필요합니다. 퀴즈 기록을 내 GitHub 에 남길 수 있도록 fork 해서 씁니다.

1. 이 페이지 오른쪽 위의 **Fork** 를 눌러 내 계정으로 복사합니다
2. 내 fork 를 받아 지도를 띄웁니다

   ```bash
   git clone https://github.com/<내 계정>/graphics-roadmap.git
   cd graphics-roadmap/site
   npm install
   npm run dev
   ```

3. 브라우저에서 http://localhost:5173 을 엽니다

### 퀴즈 기록

퀴즈를 제출하거나 "다 읽음" 을 누르면 지도를 띄운 dev server 가 그 결과를 `progress/<강의>.json` 에 저장합니다. 이 파일은 내 컴퓨터에만
있으므로, 공부를 마칠 때 커밋해서 push 해 두면 다른 컴퓨터에서 clone 해도 진행 상태가 그대로 보입니다.

```bash
git add progress
git commit -m "Record quiz progress"
git push
```

### 강의 업데이트

강의는 계속 늘어나고 고쳐집니다. `npm run dev` 로 지도를 띄우면 dev server 가 원본 리포(jaesung-cs/graphics-roadmap)를 확인해서, 내
fork 에 없는 변경이 있으면 화면 위쪽에 업데이트할지 묻습니다. **업데이트** 를 누르면 원본의 변경을 내 fork 에 합치고, 퀴즈 기록은
그대로 남습니다. 합친 결과는 다음에 퀴즈 기록을 push 할 때 함께 올라갑니다.

원본은 `progress/` 를 건드리지 않으므로 기록만 커밋했다면 충돌하지 않습니다. `lessons/` 나 `site/` 를 직접 고쳤다면 원본의 변경과 겹쳐
자동으로 합치지 못할 수 있는데, 그때는 터미널에서 직접 합칩니다.

```bash
git fetch upstream
git merge upstream/main
```

## 커리큘럼

| 영역 | 다루는 것 |
| --- | --- |
| 수학 기초 | vector·행렬, 미분·gradient, 확률·Gaussian |
| Deep Learning | linear regression 에서 PyTorch 까지의 신경망 기초, CNN·ResNet·ViT·CLIP, tokenizer·Transformer·GPT-2 |
| Image Generation | VAE·DDPM·flow matching, 그리고 Stable Diffusion 의 U-Net, latent diffusion, text conditioning(cross-attention), classifier-free guidance, sampler |

원 하나가 강의 하나이고, 선은 먼저 읽어 두면 좋은 강의를 잇습니다. 가운데의 Vectors & Matrices 에서 시작해 수학 기초를 읽고, 아래의
Deep Learning 트리에서 신경망을 익힌 뒤 위의 Image Generation 트리를 따라 Stable Diffusion 까지 나아가면 됩니다. 선수 관계는 권장
순서일 뿐이라 어떤 강의든 바로 열 수 있습니다.

## 강의와 퀴즈

강의는 학습 목표, 본문, 정리 순서로 이루어집니다. 본문은 직접 따라 계산하는 표, 그림, 수식, 절마다 답을 접어 둔 확인 문제로 짜여
있고, 긴 유도는 "심화" 블록에 접어 두어 처음 읽을 때는 건너뛸 수 있습니다.

Deep Learning 트리와 생성 모델 기초 강의는 끝에 퀴즈가 있습니다. 강의의 원리 하나마다 한 문항을 묻고, 제출하면 틀린 문항의 해설이
보이고, 다시 풀 때는 틀린 원리만 같은 원리의 다른 문항으로 묻습니다. 틀린 문항이 남지 않으면 그 강의는 완료로 칠해지고 이어지는 강의가
밝아집니다. Stable Diffusion 강의에는 퀴즈가 없고, 끝까지 읽은 뒤 "다 읽음" 으로 표시하면 완료로 칠해집니다.

## 프로젝트 구조

```text
graphics-roadmap/
├── lessons/              # 강의
│   ├── branches.yaml     # 지도의 동심원·영역·branch 설정
│   └── <강의>/
│       ├── README.md     # 강의 본문 (frontmatter 에 선수 강의·시간·논문)
│       ├── quiz.yaml     # 퀴즈 (Stable Diffusion 강의에는 없음)
│       └── figures/      # SVG 그림
├── progress/             # 내 퀴즈·읽음 기록 (fork 에서만 커밋)
├── site/                 # 지도와 강의를 보여 주는 Vite + React 앱
├── docs/                 # 강의 문장 패턴(korean-prose.md)
├── CONTRIBUTING.md       # 강의·퀴즈·지도 규칙
└── CLAUDE.md             # Claude Code 로 강의를 쓸 때의 작업 순서와 규칙
```

## 기여

틀린 내용, 어색한 설명, 깨진 링크는 [issue](https://github.com/jaesung-cs/graphics-roadmap/issues) 로 알려 주시면 고치겠습니다.
사실 정정은 근거가 되는 논문·공식 문서·소스 코드 링크를 함께 남겨 주시면 가장 빠르게 반영됩니다.

강의를 고치거나 새로 쓰는 pull request 도 환영합니다. 강의 형식, 문장 규칙, 퀴즈 규칙, 지도 사이트의 구조는
[CONTRIBUTING.md](CONTRIBUTING.md) 에 있고, 문장을 다듬을 때 고치는 패턴은 [docs/korean-prose.md](docs/korean-prose.md) 에
있습니다. Claude Code 로 강의를 쓰면 리포의 [CLAUDE.md](CLAUDE.md) 를 자동으로 읽어 같은 작업 순서와 사실 확인 규칙을 따릅니다. PR 을
올리기 전에 `site/` 에서 `npm test` 로 형식 검사를 통과하는지 확인해 주세요.

## 라이선스

이 리포는 Woolim Park 의 [Robotics Roadmap](https://github.com/woolimi/robotics-roadmap) 을 바탕으로 만든 개작물입니다.

- 강의(`lessons/`)와 문서(`docs/`): [CC BY-NC-SA 4.0](lessons/LICENSE.md). Robotics Roadmap 의 강의를 고친 것과 새로 쓴 강의 모두
  같은 라이선스입니다. 출처를 밝히면 비영리 목적으로 자유롭게 쓰고 고칠 수 있고, 고친 자료도 같은 조건으로 공유해야 합니다
- 지도 사이트(`site/`): [MIT](LICENSE). 원본의 저작권 표시를 그대로 두었습니다

원본에서 고친 것:

- 로보틱스 강의(ROS 2, 좌표 변환·kinematics·제어, 상태 추정·SLAM·navigation·자율주행, 강화학습, imitation learning, VLA, world model)를
  지웠습니다
- 남긴 강의(수학 기초, 신경망 기초, Vision, Language, VAE·DDPM·flow matching)는 원본 그대로 두고, 지운 강의로 가는 링크만 풀었습니다.
  그래서 본문의 예시와 다음 강의 예고에 로봇 이야기가 남아 있습니다
- Stable Diffusion branch 의 강의를 새로 썼습니다
- 지도를 수학 기초와 두 트리(Deep Learning, Image Generation)로 다시 짰고, 퀴즈가 없는 강의를 "다 읽음" 으로 완료하는 기능을 사이트에
  더했습니다
- README·CONTRIBUTING·CLAUDE.md 를 이 커리큘럼에 맞게 고쳤고, 로보틱스 지도를 찍은 `docs/` 의 이미지를 지웠습니다
