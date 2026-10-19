import os

# 무거운 파일을 둘 루트 (외장 드라이브)
DATA_ROOT = os.environ.get("MINILLM_DATA", r"D:\mini-llm-data")

# HF 캐시를 D로. datasets/transformers를 import하기 "전에" 설정되어야 적용됨
os.environ.setdefault("HF_HOME", os.path.join(DATA_ROOT, "hf-cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

DATA_DIR = os.path.join(DATA_ROOT, "data")          # train.bin, val.bin
CKPT_DIR = os.path.join(DATA_ROOT, "checkpoints")   # 체크포인트
LOG_DIR  = os.path.join(DATA_ROOT, "runs")          # tensorboard 로그

for d in (DATA_DIR, CKPT_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)