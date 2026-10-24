"""Stream-K with TN n8 tiles per chain (ids 60..64) vs the shipped candidates, graph bursts."""
import sys, torch
import test_gemm as G
from gemm_burst import burst, POOL
NEW = [60, 61, 62, 63, 64, 65, 66, 70, 71, 72, 73]
for M in [int(a) for a in sys.argv[1:]] or (1, 8):
    for name in ("0.6B o", "0.6B down", "1.7B o", "1.7B down", "4B o", "4B down", "8B o", "8B down", "1.7B qkv"):
        N, K = G.SHAPES[name]
        copies = max(4, POOL // (N * K * 2))
        Ws = [(torch.randn(N, K, device="cuda") * 0.02).bfloat16() for _ in range(copies)]
        X = torch.randn(M, K, device="cuda").bfloat16()
        cb = burst(lambda x, w: torch.matmul(x, w.T), Ws, X)
        ref = G.ext.linear(X, Ws[0], 1, False)
        res = {}
        for k in list(G.candidates(M, N, K)) + NEW:
            y = G.ext.linear(X, Ws[0], k, False)
            if y is None: continue
            assert torch.equal(y.view(torch.int16), ref.view(torch.int16)), (name, M, k)
            res[k] = burst(lambda x, w: G.ext.linear(x, w, k, False), Ws, X)
        top = sorted(res, key=res.get)[:4]
        new = " ".join(f"k{k} {res[k]:.2f}" for k in NEW if k in res)
        print(f"M={M} {name:10s} cublas {cb:6.2f} | best " + " ".join(f"k{k} {res[k]:.2f}" for k in top) + f" | new {new}", flush=True)
        del Ws; torch.cuda.empty_cache()
