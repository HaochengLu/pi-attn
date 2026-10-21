import os, sys, torch
from torch.utils.cpp_extension import load
HERE = os.path.dirname(os.path.abspath(__file__))
K_DIR = os.path.join(HERE, "../../kernels")
sys.path.insert(0, K_DIR)


def burst(fn, Ws, X, n=64, reps=5):
    ts = []
    for _ in range(reps + 1):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for i in range(n):
            fn(X, Ws[i % len(Ws)])
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000 / n)
    return sorted(ts[1:])[reps // 2]

import test_gemm as G
ext = load(name="pi_seg_probe", sources=[os.path.join(HERE, "seg_torch.cu")], extra_include_paths=[HERE, K_DIR],
           extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_89,code=sm_89", "--expt-relaxed-constexpr"])
SM = 114
shapes = {"0.6B down": (1024, 3072), "0.6B o": (1024, 2048), "1.7B down": (2048, 6144), "1.7B gateup": (12288, 2048),
          "8B qkv": (6144, 4096), "8B o": (4096, 4096), "8B down": (4096, 12288), "8B gateup": (24576, 4096)}
POOL = 192 << 20
for name, (N, K) in shapes.items():
    copies = max(2, POOL // (N * K * 2))
    Ws = [(torch.randn(N, K, device="cuda") * 0.02).bfloat16() for _ in range(copies)]
    X = torch.randn(1, K, device="cuda").bfloat16()
    cb = burst(lambda x, w: torch.matmul(x, w.T), Ws, X)
    best_cur = min(burst(lambda x, w: G.ext.linear(x, w, k, False), Ws, X) for k in (20, 21, 30, 31) if K % (256 if k in (21, 30) else 128) == 0)
    out = []
    for S in (1, 2, 4, 8):
        for cfg, bk, stg in ((0, 128, 6), (1, 256, 3), (2, 64, 8)):
            if K % (S * bk):
                continue
            items = N // 8 * S
            for warps in (4, 8):
                if stg * (16 + 8 * warps) * (bk + 8) * 2 > 101376:
                    continue
                grid = min(SM, (items + warps - 1) // warps) if items >= SM else items
                grid = max(1, min(grid, SM))
                ws = torch.empty(S, 1, N, device="cuda")
                cnt = torch.zeros(N // 8, dtype=torch.int32, device="cuda")
                y = ext.seg(X, Ws[0], ws, cnt, S, grid, warps, cfg)
                assert cnt.count_nonzero().item() == 0
                if S == 1:
                    assert torch.equal(y, G.ext.linear(X, Ws[0], 20 if K % 128 == 0 else 1, False))
                t = burst(lambda x, w: ext.seg(x, w, ws, cnt, S, grid, warps, cfg), Ws, X)
                out.append((t, S, bk, warps))
    out.sort()
    print(f"{name:12s} cublas {cb:6.2f} cur {best_cur:6.2f} ({best_cur/cb:.2f}) | " + " ".join(f"S{s}/bk{b}/w{wp} {t:6.2f}" for t, s, b, wp in out[:4]) + f" | best/cublas {out[0][0]/cb:.2f}", flush=True)
    del Ws
