import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    block_size: int = 256     # 최대 context 길이
    vocab_size: int = 50257   # GPT-2 토크나이저
    n_layer: int = 6          # 블록 수
    n_head: int = 6           # 헤드 수
    n_embd: int = 384         # C


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.c_proj.RESIDUAL_SCALE = True     # 초기화 때 표준편차를 줄일 층 표시
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd

    def forward(self, x):
        B, T, C = x.size()
        hs = C // self.n_head
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)          # 각 (B, T, C)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)           # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)           # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)           # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)            # (B, T, C)
        return self.c_proj(y)                                       # (B, T, C)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.c_proj.RESIDUAL_SCALE = True

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))   # (B,T,C) → (B,T,4C) → (B,T,C)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # (B, T, C)
        x = x + self.mlp(self.ln_2(x))    # (B, T, C)
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=nn.LayerNorm(cfg.n_embd),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # weight tying: 입력 임베딩과 출력층이 같은 행렬을 공유
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)    # 모든 하위 층에 초기화 함수 적용

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "RESIDUAL_SCALE", False):
                std *= (2 * self.cfg.n_layer) ** -0.5   # 잔차로 더해지는 층은 더 작게
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.cfg.block_size, f"T={T} > block_size={self.cfg.block_size}"

        pos = torch.arange(T, device=idx.device)       # (T,)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)   # (B, T, C)
        for block in self.transformer.h:
            x = block(x)                               # (B, T, C)
        x = self.transformer.ln_f(x)                   # (B, T, C)
        logits = self.lm_head(x)                       # (B, T, V)

        loss = None
        if targets is not None:                        # 2-6에서 설명
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss