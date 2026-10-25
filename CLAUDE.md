# Graphics Roadmap

Stable Diffusion 을 이해하기 위한 한국어 독학 커리큘럼이다. [Robotics Roadmap](https://github.com/woolimi/robotics-roadmap) 을 fork 해
로보틱스 강의를 지우고 Stable Diffusion branch 를 더했다. `lessons/<id>/` 에 강의가 있고, `site/` 의 Vite + React 앱이 강의
frontmatter 를 읽어 선수 관계 지도를 그린다. 공부하는 법은 [README.md](README.md) 에 있다.

## 먼저 읽을 문서

- [CONTRIBUTING.md](CONTRIBUTING.md): 강의 구성, frontmatter, 본문·퀴즈 규칙, 지도 사이트 구조의 정본
- [docs/korean-prose.md](docs/korean-prose.md): 문장 검수 때 고치는 한국어 패턴
- 기준 강의: [lessons/ddpm](lessons/ddpm/README.md), [lessons/vae](lessons/vae/README.md) 의 README·`quiz.yaml`·`figures/`
- [lessons/LICENSE.md](lessons/LICENSE.md): 원본에서 고친 내용 목록. 강의를 지우거나 원본 강의를 고치면 이 목록과 README 의 라이선스 절을 함께 고친다

## 강의 작성 순서

1. 강의의 자리를 확인한다. `lessons/branches.yaml` 에서 branch 를 찾고, `requires` 의 강의와 이 강의를 `requires` 에 둔 강의
   (`grep -l "requires:.*<id>" lessons/*/README.md`)의 제목과 숫자를 읽어 앞뒤에서 무엇을 가르치는지 본다. 앞 강의(vector·행렬,
   미분·gradient, 확률·Gaussian, CNN·ResNet, Transformer, CLIP, VAE, DDPM, flow matching)가 가르치는 내용은 다시 가르치지 않고 한두
   문장으로 짚는다
2. 근거 자료를 직접 연다. 논문 원문(arXiv), 공식 문서, 소스 코드(GitHub)에서 방법과 숫자를 확인한다. 라이브러리의 기본값(diffusers 의
   scheduler·pipeline 설정, Stable Diffusion checkpoint 의 config)은 소스 코드와 config 파일에서 확인한다
3. 본문을 쓴다. 서론 → 학습 목표 → 개념마다 한 절(직관 → 그림 → 숫자 예시 → 실제 모델·논문·도구 사례) → 정리 → 참고 자료 순서이고,
   일반 식·유도·긴 손계산은 "심화" 블록에 접는다. 심화 밖 본문은 15k ~ 22k 자, 전체는 20k ~ 28k 자 안팎이다
4. 그림을 그린다. SVG 로 `figures/` 에 두고 [DDPM 의 그림](lessons/ddpm/figures/network-io.svg) 과 같은 어두운 카드와
   색(배경 `#0b1220`, 글자 `#e2e8f0`, 강조 `#38bdf8`·`#34d399`·`#f59e0b`·`#f87171`·`#a78bfa`, 보조 `#94a3b8`)을 쓴다. 그림을 만드는
   script 는 리포에 넣지 않는다. 넣기 전에 PNG 로 렌더해(headless Chrome `--headless=new --screenshot`) 글자 겹침과 잘림을 눈으로 확인하고,
   그림 속 숫자를 본문 예시와 맞춘다
5. 퀴즈를 쓴다. Stable Diffusion branch(`stable-diffusion`)의 강의에는 퀴즈를 두지 않는다. 다른 branch 의 강의는
   [lessons/ddpm/quiz.yaml](lessons/ddpm/quiz.yaml) 과 같은 모양으로 topic 6 ~ 8개, topic 마다 variant 3개다
6. 형식을 검사한다: `cd site && npm test`
7. 문장을 따로 검수한다. 다 쓴 뒤 강의 파일과 [docs/korean-prose.md](docs/korean-prose.md) 만 놓고 문장만 고치는 단계를 한 번 더
   거친다(가능하면 새 subagent 로). 사실·숫자·표·frontmatter·`](#slug)` 로 링크된 제목은 건드리지 않는다

## 사실 확인

- 숫자·날짜·기본값·인용은 원문에서 확인한 것만 쓴다. 확인하지 못한 주장은 고치거나, 단정하지 않는 서술로 바꾸거나, 지운다
- 본문의 숫자마다 논문·문서·직접 계산 중 어디서 왔는지 밝힌다. 계산을 확인한 script 는 리포에 넣지 않는다
- 숫자뿐 아니라 논문이 밝힌 결과의 이유도 원문으로 확인한다
- 서술이 직관과 다르다는 지적을 받으면 원문을 찾아 "강의 서술 | 원문 근거 | 판정" 표로 대조한 뒤 고친다
- frontmatter `papers` 와 `## 참고 자료` 에는 직접 열어 본 자료만 싣는다
- 여러 강의에 걸쳐 나오는 사실은 다른 강의의 서술과 맞춘다. 원문으로 확인해 여러 강의에서 맞춘 예:
  - flow matching 강의는 Lipman 등을 따라 t = 0 을 noise 로 두어 정답 속도가 x₁ − ε 이고, Stable Diffusion 3 논문과 diffusers 는
    t = 1 을 noise 로 두어 ε − x₀ 이며 sampling 의 step 길이가 음수다([arXiv 2403.03206](https://arxiv.org/abs/2403.03206))
  - Stable Diffusion 이 쓰는 autoencoder 의 KL weight 는 LDM `kl-f8` 설정의 `kl_weight: 1.0e-06` 이다(vae, latent-diffusion)
  - Stable Diffusion v1.5 의 noise schedule 은 `scaled_linear`(0.00085 ~ 0.012, T = 1000)이고 √ᾱ₁₀₀₀ ≈ 0.068 이다(ddpm, latent-diffusion)

## 설명 방식

- 가정하는 지식은 중학교 산수다. 처음 나오는 용어·기호·약어는 등장한 자리에서 설명하고, "뒤에서 설명한다" 로 미루지 않는다. 학습 목표,
  표, 그림, 확인 문제, 퀴즈에서 처음 나와도 같다
- 숫자 예시를 먼저 보이고 기호는 그 뒤에 쓴다. 숫자에서 일반 식으로 넘어갈 때는 "이 예의 값 | 기호로 쓴 값" 대응 표를 먼저 두고
  식은 그 표의 마지막 줄로 적는다
- 비율("1 늘 때마다 3 는다")과 실제 변화량("2 늘었다")을 한 문장에 섞지 않는다
- 값이 바뀌는 과정을 그린 그림에는 변화 전, 변화 후, 변화량을 모두 적는다
- gradient 는 부호로 설명한다(+ 면 parameter 를 늘릴 때 loss 가 늘고, − 면 줄고, 0 이면 변화가 없다)
- 정리에는 bullet 과 다음 강의 예고만 둔다. 면접 문답 문단은 넣지 않는다
- 코드 블록을 넣지 않는다. technical term 은 영어 원어로 쓴다("pretraining", "dataset")

## 작업 범위

- 강의 하나를 고칠 때는 그 강의 폴더만 바꾼다. `lessons/branches.yaml` 이나 `site/` 를 바꾸면 `npm test` 로 지도 레이아웃 검사까지 통과시킨다
- 원본에서 가져온 강의의 로봇 예시는 모두 이미지 예시로 바꿨다. 새로 드는 예시도 사진·그림·그래픽에서 고른다
- `progress/` 는 fork 에서만 커밋하는 퀴즈 기록이다. 원본 리포에는 커밋하지 않는다
- 사이트의 튜닝값과 경로는 `site/src/config.ts` 의 `CONFIG` 한 곳에 두고, 환경변수를 새로 읽지 않는다
- 커밋 메시지는 영어로 쓴다
- 라이선스: 강의·`docs/` 는 CC BY-NC-SA 4.0(원본 출처 표시, 고친 내용 표시, 비영리, 같은 라이선스), `site/` 는 MIT(원본 저작권 표시 유지)다
