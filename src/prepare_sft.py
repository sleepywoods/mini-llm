"""TinyStoriesInstruct → SFT 학습 데이터
결과 (DATA_DIR):
  sft_{split}_tokens.npy  (N, 257) uint16   예시마다 [프롬프트 + 이야기 + EOT + EOT 패딩]
  sft_{split}_plen.npy    (N,)     uint16   프롬프트 길이 P
  sft_{split}_len.npy     (N,)     uint16   패딩 전 전체 길이 L
사용: python src/prepare_sft.py
"""
import collections
import os
import random
import time

import paths  # noqa: F401  HF_HOME → D (huggingface_hub보다 먼저)
from paths import DATA_DIR

import numpy as np
import tiktoken
from huggingface_hub import hf_hub_download

REPO = "roneneldan/TinyStoriesInstruct"
BLOCK = 256
enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token

FIELDS = {"Words:": "words", "Features:": "features", "Summary:": "summary",
          "Random sentence:": "random_sentence", "Story:": "story"}
ORDER = ["words", "features", "summary", "random_sentence"]
LABEL = {"words": "Words", "features": "Features", "summary": "Summary",
         "random_sentence": "Random sentence"}


def parse_example(chunk):
    ex, story_lines = {}, []
    for line in chunk.split("\n"):
        line = line.strip()
        if not line:
            continue
        for prefix, key in FIELDS.items():
            if line.startswith(prefix):
                value = line[len(prefix):].strip()
                if key == "story":
                    if value:
                        story_lines.append(value)
                else:
                    ex[key] = value
                break
        else:
            story_lines.append(line)
    ex["story"] = "\n".join(story_lines)
    return ex


def build_prompt(ex):
    lines = [f"{LABEL[k]}: {ex[k]}" for k in ORDER if k in ex]
    return "\n".join(lines) + "\nStory:\n"


def drop_summary(ex, rng, p=0.5):
    if "summary" in ex and ("words" in ex or "features" in ex) and rng.random() < p:
        ex = {k: v for k, v in ex.items() if k != "summary"}
    return ex


def iter_chunks(path):
    """파일을 한 줄씩 읽으며 <|endoftext|> 줄 기준으로 예시 하나씩 내보냄 (메모리 절약)"""
    buf, first = [], True
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip() == "<|endoftext|>":
                if not first:                  # 파일 첫 조각은 앞부분이 잘려 있으므로 버림
                    yield "".join(buf)
                first, buf = False, []
            else:
                buf.append(line)


def process(split, fname, rng, batch=20_000):
    t0 = time.time()
    path = hf_hub_download(REPO, fname, repo_type="dataset")
    print(f"[{split}] {fname} ({os.path.getsize(path) / 1e9:.2f} GB)")

    rows, plens, lens = [], [], []
    stats = collections.Counter()
    pending = []

    def flush():
        prompts = [build_prompt(e) for e in pending]
        stories = [e["story"] for e in pending]
        p_ids_list = enc.encode_ordinary_batch(prompts, num_threads=8)
        s_ids_list = enc.encode_ordinary_batch(stories, num_threads=8)
        for e, p_ids, s_ids in zip(pending, p_ids_list, s_ids_list):
            tokens = p_ids + s_ids + [EOT]
            if len(tokens) > BLOCK + 1:            # 256 초과 → 버림
                stats["too_long"] += 1
                continue
            stats["kept"] += 1
            for k in ORDER:
                stats[k] += k in e
            row = np.full(BLOCK + 1, EOT, dtype=np.uint16)   # (257,) EOT로 패딩
            row[:len(tokens)] = tokens
            rows.append(row)
            plens.append(len(p_ids))
            lens.append(len(tokens))
        pending.clear()

    for chunk in iter_chunks(path):
        ex = parse_example(chunk)
        if not ex["story"]:
            stats["empty"] += 1
            continue
        pending.append(drop_summary(ex, rng))
        if len(pending) >= batch:
            flush()
            if stats["kept"] % 200_000 < batch:
                print(f"  {stats['kept']:,}개 처리 ({time.time() - t0:.0f}초)")
    if pending:
        flush()

    tokens = np.stack(rows)                                  # (N, 257)
    np.save(os.path.join(DATA_DIR, f"sft_{split}_tokens.npy"), tokens)
    np.save(os.path.join(DATA_DIR, f"sft_{split}_plen.npy"), np.array(plens, dtype=np.uint16))
    np.save(os.path.join(DATA_DIR, f"sft_{split}_len.npy"), np.array(lens, dtype=np.uint16))

    total = stats["kept"] + stats["too_long"]
    loss_tok = sum(l - p for l, p in zip(lens, plens))
    print(f"[{split}] 사용 {stats['kept']:,} / 256 초과로 버림 {stats['too_long']:,} "
          f"({stats['too_long'] / total:.1%}) / 빈 예시 {stats['empty']}")
    print(f"  조건 비율: " + ", ".join(f"{LABEL[k]} {stats[k] / stats['kept']:.0%}" for k in ORDER))
    print(f"  예시당 평균: 프롬프트 {np.mean(plens):.0f}토큰, loss 계산 {loss_tok / len(lens):.0f}토큰")
    print(f"  저장 {tokens.nbytes / 1e6:.0f} MB, {time.time() - t0:.0f}초")


if __name__ == "__main__":
    rng = random.Random(42)                # condition dropout 재현용
    process("valid", "TinyStories-Instruct-valid.txt", rng)
    process("train", "TinyStories-Instruct-train.txt", rng)