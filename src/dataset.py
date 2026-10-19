import os

import numpy as np
import torch

from paths import DATA_DIR

_cache = {}   # split별로 한 번만 로드해서 재사용


def _load(split):
    if split not in _cache:
        fname = "train.bin" if split == "train" else "validation.bin"
        # 파일 전체를 RAM에 한 번만 올림 (train 948MB)
        _cache[split] = np.fromfile(os.path.join(DATA_DIR, fname), dtype=np.uint16)
    return _cache[split]


def get_batch(split, batch_size, block_size, device):
    # 원래는 np.memmap 명령어를 사용해서 호출할 때마다 파일을 열어야했는데 _cache에 저장해서 계속 RAM에 적재시킴
    data = _load(split)                                                     # (N,)
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)  # (B,) 랜덤 시작 위치
    offsets = np.arange(block_size + 1)                                     # (T+1,) [0, 1, ..., 256]
    # x,y용 따로 구하지 말고 한번에 다 들고오기
    chunk = data[ix[:, None] + offsets]                                     # (B, T+1) 한 번에 잘라옴
    chunk = torch.from_numpy(chunk.astype(np.int64))                        # (B, T+1)

    x = chunk[:, :-1].contiguous()                                          # (B, T) 입력
    y = chunk[:, 1:].contiguous()                                           # (B, T) 정답 (한 칸 밀림)
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)