import torch

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("bf16 supported:", torch.cuda.is_bf16_supported())   # Ada(4060)는 True
free, total = torch.cuda.mem_get_info()                     # 바이트 단위
print(f"VRAM free/total: {free/1e9:.2f} / {total/1e9:.2f} GB")

import time

def bench(device, n=4096, iters=10, dtype=torch.float32):
    a = torch.randn(n, n, device=device, dtype=dtype)   # (n, n)
    b = torch.randn(n, n, device=device, dtype=dtype)   # (n, n)
    for _ in range(3):              # 워밍업(warmup): 첫 실행의 초기화 비용 제외
        a @ b
    if device == "cuda":
        torch.cuda.synchronize()    # 워밍업 작업이 끝날 때까지 대기
    t0 = time.perf_counter()
    for _ in range(iters):
        c = a @ b                   # (n, n)
    if device == "cuda":
        torch.cuda.synchronize()    # 측정 구간의 GPU 작업 완료 대기
    dt = (time.perf_counter() - t0) / iters
    tflops = 2 * n**3 / dt / 1e12   # matmul FLOPs ≈ 2·n³ (곱셈 + 덧셈)
    print(f"{device:4s} {str(dtype):14s}: {dt*1000:8.2f} ms/iter, {tflops:6.2f} TFLOPS")

bench("cpu", iters=3)
bench("cuda", dtype=torch.float32)
bench("cuda", dtype=torch.float16)
bench("cuda", dtype=torch.bfloat16)