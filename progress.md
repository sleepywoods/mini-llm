# 진행 기록

## 현재 단계: 0. 환경 세팅 (0-4까지 완료, 다음: 0-5 나머지 패키지 설치)

## 확정된 결정
- 환경: RTX 4060 Laptop (8GB), Windows + VS Code
- 데이터: 영어 (사전학습은 TinyStories)
- 모델: GPT-2와 이름·구조가 같은 클래스를 직접 구현 → 학습 후 GPT2LMHeadModel로 가중치 변환 → HF 업로드
  - 구조(코드)는 연동 단계부터 HF GPT2LMHeadModel 사용, 가중치는 100% 직접 학습한 것
  - 졸업 프로젝트 예측기가 fx 추적하는 대상은 내 클래스가 아니라 HF GPT2LMHeadModel
- 목표: HF에 올린 모델을 졸업 프로젝트(GPU 인스턴스 추천 / 실행시간 예측)에서 검증용으로 사용
- 순서: 졸업 프로젝트 연동(7단계)을 LoRA·LLaDA보다 먼저 진행. LoRA·LLaDA는 이후 심화 과정
- 환경 분리: 졸업 프로젝트(예측, CPU torch)와 미니 LLM(학습·실측, CUDA torch)은 별도 venv
- 학습 정밀도: bf16 autocast (가중치는 fp32로 저장)
- 졸업 프로젝트는 fp16 추론 대상 → 4단계에서 fp16 변환 검증 필수
- 7단계에 "Docker로 감싸서 AWS에 배포 + latency/throughput 측정" 포함

## 환경 정보
- 미니 LLM: C:\dev\mini-llm\.venv
  - Python 3.11, torch 2.14.1+cu130 (CUDA 런타임 13.0)
- 졸업 프로젝트: CloudGPU\.venv (OneDrive 경로)
  - torch 2.14.0+cpu, transformers 4.45.1 (CPU판은 의도적 선택)
  - 백업: C:\dev\grad-project-venv-backup.txt
- 전역 Python에도 torch 2.14.0+cpu / transformers 4.45.1 설치되어 있음 (건드리지 않음)
- NVIDIA 드라이버 592.82 (지원 CUDA 최대 13.1 → cu132 대신 cu130 선택)
- 규칙: venv 안에서는 python / python -m pip만 사용. py -3.11은 venv 생성 시에만 (venv를 무시하고 전역 Python 실행함)

## 완료 로그
<!-- 단계마다 아래 형식으로 추가
### [단계명] - YYYY-MM-DD
- 설정: 모델 크기 / 데이터 / 주요 하이퍼파라미터
- 결과: 최종 train/val loss, 생성 샘플 예시
- 배운 점:
- 막혔던 점 / 해결:
-->
1단계 결과: GPT-2 토크나이저 사용 (vocab 50257, EOT 50256), tiktoken == HF ID 일치
배운 점: 30M 설정에서 wte가 파라미터의 64%이지만 조회 연산은 거의 0. 반면 weight tying된 lm_head는 가장 큰 행렬곱
2단계 결과: 30M 모델(6층, 6헤드, 384차원) 구현, 파라미터 30,044,544로 HF GPT2LMHeadModel과 일치, 초기 loss 10.97 (기대값 10.83)
배운 점: nn.Linear는 (출력, 입력), HF Conv1D는 (입력, 출력) → 4단계에서 c_attn, c_proj, c_fc, mlp.c_proj만 전치
막혔던 점: 노트북 셀 반복 실행으로 view 에러 → 재시작. 슬라이스한 targets에 view 사용 불가 → reshape
3단계 결과: 30M, 9,000스텝(2.95억 토큰), 약 1시간 20분 / train 1.545, val 1.547 / temp 0.8 + top-k 50에서 문법적인 이야기 생성
학습 설정: micro batch 16 × K 8, bf16, lr 6e-4 → 6e-5 (warmup 200, cosine), clip 1.0, wd 0.1 (2D만)
배운 점: bf16은 학습 전체로는 1.4배 (행렬곱만 4.5배). Windows sysmem fallback으로 VRAM 초과 시 OOM 대신 급격히 느려짐. top-k 없이 높은 temperature를 쓰면 학습 안 된 토큰이 섞여 무너짐
막혔던 점: 노트북 view 에러 → 커널 재시작. Windows os.replace PermissionError → 재시도 추가. exFAT 오류 → chkdsk

### 0. 환경 세팅 (진행 중) - 2026-10-04 ~ 2026-10-06
- 설정: venv 격리, torch 2.14.1+cu130, numpy 설치
- 결과 (check_gpu.py, matmul 4096×4096):
  - CPU fp32 0.55 / GPU fp32 6.65 / fp16 29.96 / bf16 30.20 TFLOPS
  - VRAM total 8.59GB, free 7.44GB → 실사용 기준 ~7GB
- 배운 점:
  - nvidia-smi의 CUDA Version = 드라이버가 지원하는 최대 버전. PyTorch wheel에 CUDA 런타임 포함 → Toolkit 별도 설치 불필요
  - CUDA 연산은 비동기 → 시간 측정 시 워밍업 + torch.cuda.synchronize() 필수
  - fp16/bf16은 텐서 코어로 fp32 대비 약 4.5배
  - 4096² 정사각 행렬 = 최선의 경우. 실제 레이어는 작아서 효율 낮음 → 루프라인만으로 부족, 보정 필요
- 막혔던 점 / 해결:
  - py -3.11로 확인 시 전역 패키지가 보여 졸업 프로젝트 환경을 착각 → venv 안에서 python으로 재확인
  - torch 2.14.0 CUDA판 없음 → 2.14.1 사용 (patch 차이라 영향 없음)
  - NumPy 미설치 경고 → numpy 설치
- 남은 작업: 0-5 패키지 설치 + requirements.txt, 0-6 폴더 구조 + git

## 질문 메모
- 3단계 측정: bf16 B=16 → 56.5k tok/s, 4.58GB. fp32(TF32) 대비 1.39배 (행렬곱 벤치마크 4.5배와 차이 = 메모리 한계 연산)
Windows sysmem fallback: VRAM 초과 시 OOM 대신 성능 급락 (B=48: 17배 느려짐) → 예측기/실측 시 주의
- 4단계: 최신 transformers로 저장한 모델이 transformers 4.45.1 + CPU torch(졸업 프로젝트 환경)에서 로드·fx 추적되는지 검증
- 4단계: fp32 vs fp16 logits 비교 (NaN/inf, 최대 오차, greedy 생성 일치) — bf16 학습 모델의 fp16 오버플로 확인
- 7단계: 모델 시리즈 크기 결정
  - 10M/30M은 오버헤드 지배 영역, 125M/350M은 연산·메모리 한계 영역 → 둘 다 포함해야 예측기 전체 검증 가능
  - 350M은 학습 부담 시 랜덤 가중치로 시간 측정만 하는 방안 검토
  - 소형 모델은 파라미터 대부분이 임베딩 → 파라미터 수와 연산량이 비례하지 않음
- 7단계: 4060 루프라인 값은 3~5회 측정 중앙값 사용, 행렬 크기별 효율 측정
- 장기: 졸업 프로젝트 venv를 OneDrive 밖으로 옮기는 것 검토 (동기화 부하)