"""변환된 HF 모델 폴더를 Hub에 업로드
사용: python src/upload_to_hub.py --repo_id 내아이디/tinystories-gpt2-30m
"""
import argparse
import os

import paths  # noqa: F401  HF_HOME(D) → 로그인 토큰 위치
from paths import DATA_ROOT

import torch
from huggingface_hub import HfApi

CARD = """---
language: en
license: mit
library_name: transformers
pipeline_tag: text-generation
datasets:
- roneneldan/TinyStories
---

# {repo_name}

GPT-2 구조(6 layers, 6 heads, 384 dim, context 256, 약 30M 파라미터)를 PyTorch로 직접 구현해
TinyStories로 처음부터 사전학습한 뒤, `GPT2LMHeadModel` 형식으로 변환한 학습용 모델입니다.

- 학습: {steps:,} steps, 약 {tokens}M tokens, bf16 mixed precision, RTX 4060 Laptop
- 학습 데이터: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (CDLA-Sharing-1.0)
- 검증 loss: {val_loss:.4f}
- 토크나이저: GPT-2 (vocab 50257)
- fp16 추론 검증: fp32 대비 greedy 생성 일치

## 사용법
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForCausalLM.from_pretrained("{repo_id}")
ids = tok("Once upon a time", return_tensors="pt")
out = model.generate(**ids, max_new_tokens=100, do_sample=True, temperature=0.8, top_k=50)
print(tok.decode(out[0], skip_special_tokens=True))
```

## 한계
어린이용 짧은 영어 이야기만 학습했습니다. 일반 지식, 다른 언어, 사실 정확성이 필요한 용도에는 맞지 않습니다.
"""

CARD_SFT = """---
language: en
license: mit
library_name: transformers
pipeline_tag: text-generation
base_model: skmbkang/tinystories-gpt2-30m
datasets:
- roneneldan/TinyStoriesInstruct
---

# {repo_name}

[skmbkang/tinystories-gpt2-30m](https://huggingface.co/skmbkang/tinystories-gpt2-30m)을
TinyStoriesInstruct로 SFT(지도 미세조정)한 모델입니다. 조건(단어, 특징, 줄거리)에 맞는 짧은 이야기를 씁니다.

- 학습: {steps:,} steps, lr 1e-4 → 1e-5, 답변(이야기) 토큰만 loss 계산, Summary 50% 무작위 제거
- 학습 데이터: [TinyStoriesInstruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct)
- 검증 loss (이야기 토큰): 1.548 → {val_loss:.3f}

## 평가 (valid 100개, Words 3개 조건, temperature 0.8, top-k 50)
| 모델 | 단어 포함률 | 3개 모두 포함 |
|---|---|---|
| 사전학습 모델 | 21.0% | 1.0% |
| 이 모델 | 65.3% | 26.0% |

## 프롬프트 형식 (반드시 이 형식으로)
조건 줄은 필요한 것만 넣으면 되고, 마지막 줄은 항상 `Story:` + 줄바꿈입니다.
Words: kite, brave, windy
Features: Dialogue
Summary: A girl flies a kite on a windy day.
Story:

## 사용법
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForCausalLM.from_pretrained("{repo_id}")
prompt = "Words: kite, brave, windy\\nStory:\\n"
ids = tok(prompt, return_tensors="pt")
out = model.generate(**ids, max_new_tokens=200, do_sample=True, temperature=0.8, top_k=50,
                     pad_token_id=50256)
print(tok.decode(out[0], skip_special_tokens=True))
```

## 한계
30M 소형 모델이라 이름 혼동 등 이야기 일관성이 떨어지고, 단어 대신 비슷한 말을 쓰기도 합니다.
프롬프트와 이야기를 합쳐 256토큰까지만 처리합니다.
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", required=True)            # 예: 내아이디/tinystories-gpt2-30m
    p.add_argument("--run_name", default="gpt30m")
    p.add_argument("--private", action="store_true")
    p.add_argument("--sft", action="store_true")          # SFT 모델 카드 사용
    args = p.parse_args()

    folder = os.path.join(DATA_ROOT, "hf", args.run_name)
    ckpt = torch.load(os.path.join(paths.CKPT_DIR, args.run_name, "best.pt"),
                      map_location="cpu", weights_only=False)
    ta = ckpt["train_args"]
    tokens = ta["micro_batch"] * ckpt["model_config"]["block_size"] * ta["grad_accum"] * (ckpt["step"] + 1)

    template = CARD_SFT if args.sft else CARD
    card = template.format(repo_name=args.repo_id.split("/")[-1], repo_id=args.repo_id,
                           steps=ckpt["step"] + 1, tokens=round(tokens / 1e6),
                           val_loss=ckpt["best_val"])
    with open(os.path.join(folder, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)

    api = HfApi()
    print("로그인 계정:", api.whoami()["name"])
    api.create_repo(args.repo_id, private=args.private, exist_ok=True)   # 저장소 생성 (있으면 그대로)
    api.upload_folder(folder_path=folder, repo_id=args.repo_id,
                      commit_message=f"Upload {args.run_name} (val loss {ckpt['best_val']:.4f})")
    print(f"업로드 완료: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()