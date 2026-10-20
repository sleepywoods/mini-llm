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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", required=True)            # 예: 내아이디/tinystories-gpt2-30m
    p.add_argument("--run_name", default="gpt30m")
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    folder = os.path.join(DATA_ROOT, "hf", args.run_name)
    ckpt = torch.load(os.path.join(paths.CKPT_DIR, args.run_name, "best.pt"),
                      map_location="cpu", weights_only=False)
    ta = ckpt["train_args"]
    tokens = ta["micro_batch"] * ckpt["model_config"]["block_size"] * ta["grad_accum"] * (ckpt["step"] + 1)

    card = CARD.format(repo_name=args.repo_id.split("/")[-1], repo_id=args.repo_id,
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