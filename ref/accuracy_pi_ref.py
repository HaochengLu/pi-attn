"""Accuracy of PI-Attn (snapped / unsnapped) vs FA2-style attention against
float64, on identical rows. Reports mean and max abs error per configuration."""
import math
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from pi_ref import bf16, fa2_attention, pi_attention  # noqa: E402
from test_pi_ref import exact  # noqa: E402

torch.manual_seed(1)
D = 64
scale = 1 / math.sqrt(D)
for T, temp in [(333, 1.0), (333, 8.0), (600, 4.0)]:
    q, K, V = bf16(torch.randn(T, D) * temp), bf16(torch.randn(T, D)), bf16(torch.randn(T, D))
    pos = torch.arange(T)
    rows = torch.arange(0, T, 5)
    ref = exact(q[rows], pos[rows], K, V, scale)
    res = {
        "FA2 tile64": fa2_attention(q[rows], pos[rows], K, V, 64, scale),
        "PI C=64 snap": pi_attention(q[rows], pos[rows], K, V, 64, scale, True),
        "PI C=64 nosnap": pi_attention(q[rows], pos[rows], K, V, 64, scale, False),
        "PI C=16 snap": pi_attention(q[rows], pos[rows], K, V, 16, scale, True),
    }
    for name, out in res.items():
        e = (out - ref).abs()
        print(f"T={T} temp={temp} {name:15s} mean {e.mean():.3e}  max {e.max():.3e}", flush=True)
