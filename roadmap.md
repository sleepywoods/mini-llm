# 미니 LLM 구축 로드맵

## 공통 원칙
- 핵심 구조(어텐션, 학습 루프, 손실)는 PyTorch로 직접 구현한다.
- 직접 구현하는 GPT 클래스는 레이어 이름과 구조를 GPT-2(nanoGPT 방식)와 맞춘다.
  (transformer.wte / wpe / h.N.ln_1 / attn.c_attn / attn.c_proj / ln_2 / mlp.c_fc / mlp.c_proj / ln_f / lm_head)
  → 나중에 가중치를 HF `GPT2LMHeadModel`로 옮기기 쉽게 하기 위함
- 하드웨어: RTX 4060 Laptop (VRAM 8GB), Windows + VS Code. 필요 시 AWS GPU 인스턴스 보조 사용
- 데이터 언어: 영어

## 0. 환경 세팅
- Windows + VS Code (Python, Jupyter 확장), 가상환경(venv)
- PyTorch(CUDA) 설치 → `torch.cuda.is_available()`, `torch.cuda.get_device_name()` 확인
- 패키지: transformers, tokenizers, tiktoken, datasets, peft, accelerate, safetensors, huggingface_hub
- 실험 추적: tensorboard 또는 wandb
- Windows 주의: DataLoader num_workers>0이면 `if __name__ == "__main__":` 필요, torch.compile은 초반엔 사용 안 함
- 완료 기준: GPU에서 간단한 행렬곱 실행 확인

## 1. 토크나이저
- 이해: 문자 단위 → BPE(Byte Pair Encoding) → 바이트 레벨 BPE
- 실습: 미니 BPE 직접 구현(학습용) → GPT-2 토크나이저(tiktoken / HF) 가져와 사용
- 완료 기준: 인코딩/디코딩 왕복 확인, vocab size가 모델 크기(임베딩 파라미터)에 주는 영향 설명 가능

## 2. 트랜스포머 모델 (Decoder-only, GPT-2 구조)
- 구성: 토큰/위치 임베딩, causal self-attention(Q/K/V를 c_attn 하나로), multi-head, MLP(GELU), LayerNorm(pre-LN), residual, weight tying
- 실습: GPT-2와 이름·구조가 같은 클래스를 직접 구현
- 규모 예시: n_layer 6, n_head 6, n_embd 384, context 256 (약 30M 파라미터, 임베딩 포함)
- 완료 기준: 파라미터 수 손으로 계산, 각 텐서 shape 설명 가능, 랜덤 초기화 상태의 loss ≈ ln(vocab_size) 확인

## 3. 사전학습 (Pre-training)
- 데이터: TinyStories (영어)
- 학습 루프: next-token prediction, cross-entropy, AdamW
- 개선 항목(기본 영상 코드 대비):
  - train/val 분리, 일정 step마다 val loss 측정
  - bf16 autocast, gradient accumulation, gradient clipping
  - LR warmup + cosine decay
  - epoch 대신 step/토큰 수 기준 관리
  - 체크포인트: best + last만 저장, optimizer/step 상태 포함 → 학습 재개 가능
- 평가: train/val loss, perplexity, 주기적 샘플 생성(temperature, top-k)
- 완료 기준: val loss 수렴, 문법적으로 그럴듯한 이야기 생성

## 4. HF 형식 변환 & 업로드 (1차 배포)
- 개념: 학습 결과 = state_dict(이름 → 텐서). 같은 구조의 GPT2LMHeadModel에 이름을 맞춰 복사
- 변환 스크립트: 이름 매핑 + Conv1D 대응을 위한 전치(c_attn, c_proj, c_fc 등)
- 검증: 같은 입력에 대해 내 모델과 HF 모델의 logits가 일치(torch.allclose)하는지 확인
- 업로드: HF 가입 → write 토큰 → `hf auth login` → `push_to_hub` (모델 + 토크나이저, safetensors)
- 모델 카드: pipeline_tag: text-generation, 라이선스, 파라미터 수, 학습 데이터, 학습 설정
- 완료 기준: 다른 환경에서 `AutoModelForCausalLM.from_pretrained("아이디/모델")`로 로드 후 생성 성공

## 5. 파인튜닝 (SFT)
- instruction 데이터셋, chat template, 프롬프트 부분 loss 마스킹(-100)
- full fine-tuning으로 먼저 수행 (LoRA와 비교용)
- 완료 기준: 질문 형식 입력에 답변 형식으로 응답

## 6. RAG
- 문서 청킹 → 임베딩 모델 → 벡터 DB(FAISS 또는 Chroma) → 검색 → 프롬프트 주입
- 직접 만든 소형 모델은 생성 품질이 낮으므로 공개 소형 모델(0.5B~1.5B)로도 비교
- 평가: 검색 정확도(recall@k)와 답변 품질을 분리 측정
- 완료 기준: 내 문서 기반 질문에 근거 있는 답변

## 7. 졸업 프로젝트 연동
- 크기만 다르고 구조가 같은 모델 시리즈(예: 10M / 30M / 125M)를 HF에 업로드
- 졸업 프로젝트의 실행시간 예측기 검증용으로 사용
  - fx graph 추적이 정상 동작하는지 확인
  - 4060 / AWS 인스턴스 실측 추론 시간 vs 예측값 비교 (모델 크기별 스케일링 검증)
- 완료 기준: 졸업 프로젝트에서 내 HF 모델을 불러와 인스턴스 추천까지 동작

---
아래 8~9단계는 졸업 프로젝트 이후 진행하는 심화 과정

## 8. LoRA / QLoRA
- 원리: 가중치 업데이트를 저랭크 행렬(A·B)로 근사, rank/alpha 의미
- 실습: LoRA 레이어 직접 구현 → peft로 공개 모델 튜닝 → QLoRA(4bit, 8GB VRAM 기준 1B급)
- 비교: full fine-tuning 대비 메모리, 학습 파라미터 수, 성능
- 완료 기준: 어댑터 저장/병합(merge), rank별 차이 설명 가능

## 9. Large Language Diffusion Model (LLaDA 방식)
- 개념: autoregressive 대신 masked diffusion. 마스킹 비율 t를 무작위로 뽑아 토큰을 [MASK]로 가리고, 양방향 트랜스포머가 가려진 토큰을 예측
- 구현: causal mask 제거, [MASK] 토큰 추가, 마스킹 위치만 loss(1/t 가중), 전부 마스크에서 시작해 단계적으로 복원·재마스킹하는 샘플링
- 실습: 3단계 학습 코드를 재사용해 "마스킹 방식만 바꾸기", AR 모델과 생성 품질·속도 비교
- 배포: 비표준 구조이므로 커스텀 코드(trust_remote_code)로 HF 업로드
- 참고: LLaDA 논문(2025)과 함께 진행
- (선택) 졸업 프로젝트 예측기의 비표준 구조(양방향 어텐션, 반복 샘플링) 엣지 케이스로 활용
