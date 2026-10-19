"""학습된 모델로 이야기 생성
사용: python src/sample.py --prompt "Once upon a time" --temperature 0.8 --top_k 50
"""
import argparse
import os

import paths  # noqa: F401
from paths import CKPT_DIR

import tiktoken
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig


@torch.no_grad()                                            # 생성은 학습이 아니므로 기울기 불필요
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, eot=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size:]           # (B, ≤256) 최근 256토큰만 사용
        logits, _ = model(idx_cond)                         # (B, T, V)
        logits = logits[:, -1, :] / temperature             # (B, V) 마지막 위치만 = 다음 토큰 예측
        if top_k is not None:
            kth = torch.topk(logits, top_k).values[:, [-1]]  # (B, 1) k번째로 큰 점수
            logits[logits < kth] = float("-inf")             # 나머지는 확률 0으로
        probs = F.softmax(logits, dim=-1)                   # (B, V)
        next_id = torch.multinomial(probs, num_samples=1)   # (B, 1) 확률에 따라 하나 뽑기
        idx = torch.cat([idx, next_id], dim=1)              # (B, T+1) 입력 뒤에 붙임
        if eot is not None and (next_id == eot).all():      # 이야기 끝이면 중단
            break
    return idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", default="gpt30m")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    device = "cuda"
    torch.manual_seed(args.seed)

    ckpt = torch.load(os.path.join(CKPT_DIR, args.run_name, "best.pt"),
                      map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_config"]))          # 저장된 설정으로 같은 구조 생성
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"step {ckpt['step']} 모델, val loss {ckpt['best_val']:.4f}\n")

    enc = tiktoken.get_encoding("gpt2")
    start = torch.tensor([enc.encode(args.prompt)], device=device)   # (1, T)

    for i in range(args.num_samples):
        out = generate(model, start, args.max_new_tokens,
                       args.temperature, args.top_k, eot=enc.eot_token)
        text = enc.decode([t for t in out[0].tolist() if t != enc.eot_token])
        print(f"===== 샘플 {i + 1} =====\n{text}\n")


if __name__ == "__main__":
    main()