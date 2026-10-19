import os
import time

from paths import DATA_DIR          # ← 반드시 datasets보다 먼저 import

import numpy as np
import tiktoken
from datasets import load_dataset

OUT_DIR = DATA_DIR                  # ← 기존 OUT_DIR 줄을 이걸로 교체
enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token                     # 50256


def tokenize_split(split):
    t0 = time.time()
    ds = load_dataset("roneneldan/TinyStories", split=split)
    print(f"[{split}] 이야기 수: {len(ds):,}")

    chunks = []
    batch = 10_000
    for i in range(0, len(ds), batch):
        texts = ds[i:i + batch]["text"]                              # list[str]
        ids_list = enc.encode_ordinary_batch(texts, num_threads=8)   # list[list[int]]
        for ids in ids_list:
            ids.append(EOT)                                          # 이야기 끝 표시
        flat = np.fromiter((t for ids in ids_list for t in ids), dtype=np.uint16)
        chunks.append(flat)
        if (i // batch) % 20 == 0:
            print(f"  {i:,} / {len(ds):,}")

    arr = np.concatenate(chunks)                                     # (전체 토큰 수,)
    path = os.path.join(OUT_DIR, f"{split}.bin")
    arr.tofile(path)
    print(f"[{split}] 토큰 {len(arr):,}개 → {path} "
          f"({os.path.getsize(path) / 1e6:.1f} MB, {time.time() - t0:.0f}초)")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    # tokenize_split("validation")
    tokenize_split("train")    # validation이 잘 되면 주석 해제