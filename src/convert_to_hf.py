"""best.pt → HF GPT2LMHeadModel 형식으로 변환·저장
사용: python src/convert_to_hf.py --run_name gpt30m
"""
import argparse
import os

import paths  # noqa: F401  HF_HOME을 D로 (transformers보다 먼저)
from paths import CKPT_DIR, DATA_ROOT

import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

# HF는 이 4종류를 Conv1D로 구현 → weight를 (입력, 출력)으로 저장 → 전치 필요
TRANSPOSED = ("attn.c_attn.weight", "attn.c_proj.weight",
              "mlp.c_fc.weight", "mlp.c_proj.weight")


def build_hf_config(mc):
    return GPT2Config(
        vocab_size=mc["vocab_size"],
        n_positions=mc["block_size"],
        n_embd=mc["n_embd"],
        n_layer=mc["n_layer"],
        n_head=mc["n_head"],
        activation_function="gelu_new",   # tanh 근사 GELU
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        bos_token_id=50256, eos_token_id=50256,
    )


def convert(run_name):
    ckpt = torch.load(os.path.join(CKPT_DIR, run_name, "best.pt"),
                      map_location="cpu", weights_only=False)
    mine = ckpt["model"]                                  # 우리 state_dict

    hf = GPT2LMHeadModel(build_hf_config(ckpt["model_config"]))
    hf_sd = hf.state_dict()

    new_sd = {}
    for k, v in mine.items():
        assert k in hf_sd, f"HF에 없는 이름: {k}"
        if k.endswith(TRANSPOSED):
            v = v.t()                                     # (출력, 입력) → (입력, 출력)
        assert v.shape == hf_sd[k].shape, f"{k}: {tuple(v.shape)} vs {tuple(hf_sd[k].shape)}"
        new_sd[k] = v

    missing = set(hf_sd) - set(new_sd)
    assert not missing, f"채우지 못한 HF 파라미터: {missing}"

    hf.load_state_dict(new_sd, strict=True)               # 이름·shape이 하나라도 다르면 에러
    print(f"변환 완료: step {ckpt['step']}, val loss {ckpt['best_val']:.4f}")
    return hf


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", default="gpt30m")
    args = p.parse_args()

    hf = convert(args.run_name)
    out_dir = os.path.join(DATA_ROOT, "hf", args.run_name)
    hf.save_pretrained(out_dir)                           # config.json + model.safetensors
    AutoTokenizer.from_pretrained("gpt2").save_pretrained(out_dir)   # 토크나이저도 같은 폴더에
    print("저장:", out_dir)
    for f in sorted(os.listdir(out_dir)):
        print(f"  {f:28s} {os.path.getsize(os.path.join(out_dir, f)) / 1e6:8.2f} MB")