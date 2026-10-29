"""SFT 평가: Words 조건을 얼마나 따르는가 (사전학습 모델 vs SFT 모델)
사용: python src/eval_sft.py --n 100
"""
import argparse
import os
import random
import re

import paths  # noqa: F401
from paths import CKPT_DIR

import tiktoken
import torch
from huggingface_hub import hf_hub_download

from model import GPT, GPTConfig
from prepare_sft import REPO, build_prompt, iter_chunks, parse_example
from sample import generate

BLOCK = 256


def load(run):
    ckpt = torch.load(os.path.join(CKPT_DIR, run, "best.pt"), map_location="cpu", weights_only=False)
    m = GPT(GPTConfig(**ckpt["model_config"]))
    m.load_state_dict(ckpt["model"])
    return m.to("cuda").eval()


def word_hits(text, words):
    t = text.lower()
    return [re.search(rf"\b{re.escape(w.lower())}", t) is not None for w in words]   # 단어 앞부분 일치


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--base", default="gpt30m")
    p.add_argument("--sft", default="sft30m")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    enc = tiktoken.get_encoding("gpt2")
    path = hf_hub_download(REPO, "TinyStories-Instruct-valid.txt", repo_type="dataset")
    pool = [e for e in (parse_example(c) for c in iter_chunks(path)) if "words" in e]
    cases = random.Random(args.seed).sample(pool, args.n)         # 학습에 안 쓴 valid에서 n개

    results = {}
    for name in (args.base, args.sft):
        model = load(name)
        torch.manual_seed(args.seed)                              # 두 모델에 같은 난수 조건
        hits, ended, lengths, samples = [], 0, [], []
        for i, ex in enumerate(cases):
            words = [w.strip() for w in ex["words"].split(",")]
            prompt = build_prompt({"words": ex["words"]})          # Words만 주는 요청
            ids = enc.encode(prompt)
            start = torch.tensor([ids], device="cuda")             # (1, P)
            out = generate(model, start, BLOCK - len(ids), temperature=0.8, top_k=50,
                           eot=enc.eot_token)[0, len(ids):].tolist()   # 생성된 부분만
            if enc.eot_token in out:
                ended += 1
                out = out[:out.index(enc.eot_token)]
            story = enc.decode(out)
            h = word_hits(story, words)
            hits.append(h)
            lengths.append(len(out))
            if i < 2:
                samples.append((prompt, story))
        results[name] = (hits, ended, lengths, samples)
        del model
        torch.cuda.empty_cache()

    print(f"\n평가 예시 {args.n}개 (조건: Words 3개만)\n")
    print(f"{'모델':10s} {'단어 포함률':>10s} {'3개 모두':>8s} {'EOT로 끝남':>10s} {'평균 길이':>8s}")
    for name, (hits, ended, lengths, _) in results.items():
        per_word = sum(sum(h) for h in hits) / sum(len(h) for h in hits)
        all3 = sum(all(h) for h in hits) / len(hits)
        print(f"{name:10s} {per_word:10.1%} {all3:8.1%} {ended / len(hits):10.1%} {sum(lengths) / len(lengths):8.0f}")

    for name, (_, _, _, samples) in results.items():
        print(f"\n========== {name} 샘플 ==========")
        for prompt, story in samples:
            print(prompt + story[:400] + ("..." if len(story) > 400 else "") + "\n" + "-" * 40)


if __name__ == "__main__":
    main()