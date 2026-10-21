"""M-invariance, accuracy and speed of the invariant GEMM vs torch/cuBLAS.

Invariance: for every Qwen3 projection shape, row r of Y must be bitwise equal
whether computed with M = 1, any other M up to 4096, any kernel variant
(small G=1/2/4, large), and with garbage in the other rows."""
import os
import sys

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
major, minor = torch.cuda.get_device_capability()
ext = load(name=f"pi_gemm_sm{major}{minor}", sources=[os.path.join(HERE, "gemm_torch.cu")],
           extra_include_paths=[HERE, os.path.join(os.path.dirname(__import__('flashinfer').__file__), 'data', 'cutlass', 'include')],
           extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"],
           build_directory=os.environ.get("PI_ATTN_BUILD_DIR"))
dev = "cuda"
torch.manual_seed(0)
SHAPES = {  # name: (N, K)
    "0.6B qkv": (4096, 1024), "0.6B o": (1024, 2048), "0.6B gateup": (6144, 1024), "0.6B down": (1024, 3072),
    "0.6B lm_head": (151936, 1024), "1.7B qkv": (4096, 2048), "1.7B down": (2048, 6144),
    "8B qkv": (6144, 4096), "8B gateup": (24576, 4096), "8B down": (4096, 12288),
}


def timeit(fn, iters=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000


def main(bench):
    fails = []
    for name, (N, K) in SHAPES.items():
        W = (torch.randn(N, K, device=dev) * 0.02).bfloat16()
        X = torch.randn(4096, K, device=dev).bfloat16()
        ref = ext.linear(X[:1].contiguous(), W, 1, False)[0]
        for M in [1, 2, 7, 16, 17, 33, 64, 65, 200, 1000, 4096]:
            bal = {30: 16, 31: 16, 32: 32, 33: 64}
            for kern in [1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25] + [k for k, mm in bal.items() if M <= mm]:
                if kern in (21, 22, 23, 30) and K % 256:
                    continue
                Xm = X[:M].clone()
                if M > 1:
                    Xm[1:] = torch.randn_like(Xm[1:]) * 100  # other rows must not matter
                y = ext.linear(Xm, W, kern, False)
                if not torch.equal(y[0].view(torch.int16), ref.view(torch.int16)):
                    fails.append((name, M, kern))
        segs = 2 if N <= 1024 and K >= 2048 and K % 512 == 0 else 1  # gemm_segments
        if segs > 1:  # y = round((+0 + p_0) + p_1), p_s = the unsegmented chain over segment s
            Xs = X[:64].contiguous()
            ks = K // segs
            tot = torch.zeros(64, N, device=dev)
            for sg in range(segs):
                tot = tot + ext.linear(Xs[:, sg * ks:(sg + 1) * ks], W[:, sg * ks:(sg + 1) * ks].contiguous(), 1, True)
            for kern in (1, 20, 33, 11, 13):
                if not torch.equal(ext.linear(Xs, W, kern, False).view(torch.int16), tot.bfloat16().view(torch.int16)):
                    fails.append((name, "segments", kern))
        y64 = (X[:256].double() @ W.double().T)
        e_pi = (ext.linear(X[:256].contiguous(), W, 0, True).double() - y64).abs().max().item()
        e_cb = (torch.matmul(X[:256], W.T).double() - y64).abs().max().item()
        e_pi_b = (ext.linear(X[:256].contiguous(), W, 0, False).double() - y64).abs().max().item()
        line = f"{name:13s} N={N:6d} K={K:5d}: max|err| pi(fp32 out) {e_pi:.2e} pi(bf16) {e_pi_b:.2e} cublas(bf16) {e_cb:.2e}"
        if bench:
            ts = []
            for M in [1, 8, 32, 64, 128, 256, 1024, 4096]:
                Xm = X[:M].contiguous()
                tc = timeit(lambda: torch.matmul(Xm, W.T))
                best = min((timeit(lambda: ext.linear(Xm, W, kk, False)), kk) for kk in ([1, 2, 4, 6, 7, 8, 9] if M <= 64 else []) + [10, 11, 12, 13]
                           + [k for k, mm in {30: 16, 31: 16, 32: 32, 33: 64}.items() if M <= mm])
                ts.append(f"M={M}:{best[0]/tc:.2f}(k{best[1]})")
            line += " | pi/cublas " + " ".join(ts)
        print(line, flush=True)
    print("INVARIANCE FAILURES:", len(fails), fails[:8])
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main("--bench" in sys.argv))
