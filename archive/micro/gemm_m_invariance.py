"""Is torch/cuBLAS bf16 GEMM row-wise invariant to M (number of rows)?
For each Qwen3 projection shape (N, K), compute y = x[:M] @ W^T for M in a sweep
and compare the first rows bitwise against M=1 and against each other."""
import sys

import torch

torch.manual_seed(0)
dev = "cuda"
# (name, N_out, K_in) for Qwen3-0.6B / 1.7B / 8B projections
shapes = {
    "0.6B qkv": (4096, 1024), "0.6B o": (1024, 2048), "0.6B gateup": (6144, 1024), "0.6B down": (1024, 3072),
    "0.6B lm_head": (151936, 1024),
    "1.7B qkv": (4096, 2048), "1.7B down": (2048, 6144),
    "8B qkv": (6144, 4096), "8B gateup": (24576, 4096), "8B down": (4096, 12288),
}
Ms = [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 100, 128, 200, 256, 512, 1024, 2048]
print(torch.__version__, torch.cuda.get_device_name(0))
for name, (N, K) in shapes.items():
    W = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
    X = torch.randn(max(Ms), K, device=dev, dtype=torch.bfloat16)
    ref = (X[:1] @ W.T)[0]
    groups, sig = [], {}
    for M in Ms:
        y = (X[:M] @ W.T)[0]
        key = y.view(torch.int16).cpu().numpy().tobytes()
        sig.setdefault(key, []).append(M)
    classes = list(sig.values())
    print(f"{name:14s} N={N:6d} K={K:5d}: {len(classes)} distinct row-0 results across M; classes={classes}", flush=True)
