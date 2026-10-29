import os

import numpy as np
import torch

from paths import DATA_DIR

_cache = {}


def _load(split):
    split = "valid" if split in ("val", "valid") else "train"
    if split not in _cache:
        tok = np.load(os.path.join(DATA_DIR, f"sft_{split}_tokens.npy"))                  # (N, 257)
        plen = np.load(os.path.join(DATA_DIR, f"sft_{split}_plen.npy")).astype(np.int64)  # (N,)
        length = np.load(os.path.join(DATA_DIR, f"sft_{split}_len.npy")).astype(np.int64) # (N,)
        _cache[split] = (tok, plen, length)
    return _cache[split]


def get_batch(split, batch_size, block_size, device):
    tok, plen, length = _load(split)
    assert block_size == tok.shape[1] - 1, "SFT 데이터는 block_size 256 기준으로 만들어짐"

    ix = np.random.randint(0, len(tok), size=batch_size)          # (B,) 랜덤 예시 번호
    chunk = torch.from_numpy(tok[ix].astype(np.int64))             # (B, 257)
    x = chunk[:, :-1].contiguous()                                 # (B, T) 입력
    y = chunk[:, 1:].clone()                                       # (B, T) 정답 (한 칸 밀림)

    pos = torch.arange(block_size)                                 # (T,)  [0, 1, ..., 255]
    P = torch.from_numpy(plen[ix])[:, None]                        # (B, 1)
    L = torch.from_numpy(length[ix])[:, None]                      # (B, 1)
    mask = (pos < P - 1) | (pos >= L - 1)                          # (B, T) True = loss에서 제외
    y[mask] = -100
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)