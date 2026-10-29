"""TinyStories 사전학습 스크립트

스모크 테스트: python src/train.py --run_name smoke --max_steps 60 --eval_interval 20 --eval_iters 10
재개 테스트:   python src/train.py --run_name smoke --max_steps 80 --eval_interval 20 --eval_iters 10 --resume
본 학습:       python src/train.py --run_name gpt30m
이어서 학습:   python src/train.py --run_name gpt30m --resume
"""
import argparse
import math
import os
import time
from dataclasses import asdict

import paths  # noqa: F401  HF_HOME 등 경로 설정 (가장 먼저)
from paths import CKPT_DIR, LOG_DIR

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import dataset
import sft_dataset
from model import GPT, GPTConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", default="gpt30m")       # 체크포인트/로그 폴더 이름
    p.add_argument("--resume", action="store_true")      # last.pt에서 이어서 학습
    p.add_argument("--max_steps", type=int, default=9000)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=50)  # 평가 시 볼 배치 수
    p.add_argument("--micro_batch", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=8)   # K
    p.add_argument("--max_lr", type=float, default=6e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data", choices=["pretrain", "sft"], default="pretrain")
    p.add_argument("--init_from", default=None)   # 이 run의 best.pt 가중치로 시작 (SFT용)
    return p.parse_args()

def batch_fn(args):
    return sft_dataset.get_batch if args.data == "sft" else dataset.get_batch

def get_lr(step, args):
    min_lr = args.max_lr / 10
    if step < args.warmup_steps:                           # warmup: 선형 증가
        return args.max_lr * (step + 1) / args.warmup_steps
    if step >= args.max_steps:
        return min_lr
    ratio = (step - args.warmup_steps) / (args.max_steps - args.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))        # cosine decay
    return min_lr + coeff * (args.max_lr - min_lr)


@torch.no_grad()                                           # 평가에는 기울기 불필요 → 메모리 절약
def estimate_loss(model, args, block_size, device):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(args.eval_iters)
        for i in range(args.eval_iters):
            x, y = batch_fn(args)(split, args.micro_batch, block_size, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def _safe_replace(tmp, path, retries=20, wait=0.5):
    """Windows에서 다른 프로세스(백신 등)가 파일을 잠깐 잡고 있으면 재시도"""
    for i in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == retries - 1:
                raise                     # 10초 동안 계속 실패하면 진짜 문제 → 에러로 알림
            time.sleep(wait)


def save_checkpoint(path, model, optimizer, step, best_val, args, cfg):
    ckpt = {
        "model": model.state_dict(),            # 가중치
        "optimizer": optimizer.state_dict(),    # m, v 등 옵티마이저 상태
        "step": step,                           # 마지막으로 완료한 스텝
        "best_val": best_val,
        "model_config": asdict(cfg),            # 4단계 변환 때 모델 구조 정보로 사용
        "train_args": vars(args),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(),
            "numpy": np.random.get_state(),
        },
    }
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    _safe_replace(tmp, path)      # ← os.replace(tmp, path) 대신


def main():
    args = parse_args()
    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)                  # get_batch의 랜덤 위치도 고정
    torch.set_float32_matmul_precision("high")

    # ---- 모델, 옵티마이저 ----
    cfg = GPTConfig()
    model = GPT(cfg).to(device)
    if args.init_from and not args.resume:
        init = torch.load(os.path.join(CKPT_DIR, args.init_from, "best.pt"),
                          map_location="cpu", weights_only=False)
        model.load_state_dict(init["model"])       # 가중치만 가져옴 (옵티마이저 상태는 새로)
        print(f"초기 가중치: {args.init_from} (step {init['step']}, val {init['best_val']:.4f})")
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.max_lr, betas=(0.9, 0.95), fused=True,
    )

    # ---- 체크포인트 경로, 재개 ----
    run_dir = os.path.join(CKPT_DIR, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    last_path = os.path.join(run_dir, "last.pt")
    best_path = os.path.join(run_dir, "best.pt")

    start_step, best_val = 0, float("inf")
    if args.resume:
        # 우리가 직접 만든 파일이라 weights_only=False로 전체 로드 (numpy 난수 상태 포함)
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        best_val = ckpt["best_val"]
        torch.set_rng_state(ckpt["rng"]["torch"])
        torch.cuda.set_rng_state(ckpt["rng"]["cuda"])
        np.random.set_state(ckpt["rng"]["numpy"])
        print(f"재개: step {start_step}부터, best_val {best_val:.4f}")

    writer = SummaryWriter(os.path.join(LOG_DIR, args.run_name),
                           purge_step=start_step if args.resume else None)
    tokens_per_step = args.micro_batch * cfg.block_size * args.grad_accum
    print(f"업데이트당 토큰 {tokens_per_step:,} / 총 {args.max_steps:,}스텝 "
          f"≈ {tokens_per_step * args.max_steps / 1e6:.0f}M 토큰")

    if args.init_from and start_step == 0:
        base = estimate_loss(model, args, cfg.block_size, device)
        print(f"[학습 전] train {base['train']:.4f} | val {base['val']:.4f}")

    last_done = start_step - 1                 # 마지막으로 완료한 스텝
    try:
        for step in range(start_step, args.max_steps):
            t0 = time.perf_counter()

            # 1. 학습률
            lr = get_lr(step, args)
            for group in optimizer.param_groups:
                group["lr"] = lr

            # 2. gradient accumulation
            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            for _ in range(args.grad_accum):
                x, y = batch_fn(args)("train", args.micro_batch, cfg.block_size, device)   # (B, T)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
                loss = loss / args.grad_accum
                loss_accum += loss.detach()
                loss.backward()

            # 3. clipping → 4. 업데이트
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            last_done = step

            torch.cuda.synchronize()
            dt = time.perf_counter() - t0

            # 로그
            if step % 10 == 0:
                eta_min = (args.max_steps - step - 1) * dt / 60
                print(f"step {step:5d} | loss {loss_accum.item():.4f} | lr {lr:.2e} | "
                      f"norm {norm.item():5.2f} | {dt * 1000:4.0f} ms | "
                      f"{tokens_per_step / dt:7,.0f} tok/s | 남은 시간 {eta_min:5.1f}분")
                writer.add_scalar("train/loss", loss_accum.item(), step)
                writer.add_scalar("train/lr", lr, step)
                writer.add_scalar("train/grad_norm", norm.item(), step)
                writer.add_scalar("perf/tokens_per_sec", tokens_per_step / dt, step)

            # 평가 + 체크포인트
            if (step + 1) % args.eval_interval == 0 or step == args.max_steps - 1:
                losses = estimate_loss(model, args, cfg.block_size, device)
                writer.add_scalar("eval/train_loss", losses["train"], step)
                writer.add_scalar("eval/val_loss", losses["val"], step)
                msg = f"[eval] step {step} | train {losses['train']:.4f} | val {losses['val']:.4f}"
                if losses["val"] < best_val:
                    best_val = losses["val"]
                    save_checkpoint(best_path, model, optimizer, step, best_val, args, cfg)
                    msg += " | best 저장"
                save_checkpoint(last_path, model, optimizer, step, best_val, args, cfg)
                print(msg)

    except KeyboardInterrupt:
        print(f"\n중단 요청 → step {last_done}까지 last.pt 저장 후 종료")
        if last_done >= 0:
            save_checkpoint(last_path, model, optimizer, last_done, best_val, args, cfg)

    writer.close()
    print(f"완료. best val loss {best_val:.4f} → {best_path}")


if __name__ == "__main__":
    main()