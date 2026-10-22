"""Per-decode-step GEMM time for Qwen3 models: invariant GEMM (best kernel per
shape/M) vs cuBLAS, summed over all layers + lm_head."""
import torch

import test_gemm as G

MODELS = {  # layers, [(N, K) per layer], lm_head (N, K)
    "Qwen3-0.6B": (28, [(4096, 1024), (1024, 2048), (6144, 1024), (1024, 3072)], (151936, 1024)),
    "Qwen3-1.7B": (28, [(4096, 2048), (2048, 2048), (12288, 2048), (2048, 6144)], (151936, 2048)),
    "Qwen3-8B": (36, [(6144, 4096), (4096, 4096), (24576, 4096), (4096, 12288)], (151936, 4096)),
}
for name, (layers, shapes, head) in MODELS.items():
    for M in [1, 8, 32, 64, 128]:
        tc = tp = 0.0
        for (N, K), mult in [(s, layers) for s in shapes] + [(head, 1)]:
            W = (torch.randn(N, K, device="cuda") * 0.02).bfloat16()
            X = torch.randn(M, K, device="cuda").bfloat16()
            c = G.timeit(lambda: torch.matmul(X, W.T))
            cands = ([1, 2, 4, 6, 7, 8, 9] if M <= 64 else []) + [10, 11, 12, 13]
            p = min(G.timeit(lambda: G.ext.linear(X, W, k, False)) for k in cands)
            tc += c * mult
            tp += p * mult
        print(f"{name} M={M}: GEMMs per step cublas {tc:.0f}us invariant {tp:.0f}us ratio {tp/tc:.3f}", flush=True)
