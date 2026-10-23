"""Probe 18: segment-parallel tile shapes for 0.6B's S = 2 projections at mid M, graph-timed
bursts (gemm_burst.burst) against cuBLAS. Ids 56, 57, 60, 61 are prototype tiles."""
import sys
sys.path.insert(0, "kernels")
import torch
import test_gemm as G
from gemm_burst import burst

POOL = 600 << 20
IDS = [48, 49, 12, 13, 14, 15, 56, 57, 60, 61]
for M in [int(a) for a in sys.argv[1:]] or [32, 64, 128, 192, 256, 384, 512]:
    for name in ("0.6B o", "0.6B down"):
        N, K = G.SHAPES[name]
        copies = max(4, POOL // (N * K * 2))
        Ws = [(torch.randn(N, K, device="cuda") * 0.02).bfloat16() for _ in range(copies)]
        X = torch.randn(M, K, device="cuda").bfloat16()
        cb = burst(lambda x, w: torch.matmul(x, w.T), Ws, X)
        ref = torch.matmul(X, Ws[0].T)
        res = {}
        for k in IDS:
            try:
                y = G.ext.linear(X, Ws[0], k, False)
            except RuntimeError:
                continue
            if k >= 56:  # prototype: must equal id 13's bits (segment-sequential CUTLASS)
                assert torch.equal(y, G.ext.linear(X, Ws[0], 13, False)), (k, M, name)
            res[k] = burst(lambda x, w, k=k: G.ext.linear(x, w, k, False), Ws, X)
        print(f"M={M:4d} {name:10s} cublas {cb:6.2f} | " + " ".join(f"k{k} {t:6.2f}" for k, t in sorted(res.items(), key=lambda kv: kv[1])), flush=True)
        del Ws; torch.cuda.empty_cache()
