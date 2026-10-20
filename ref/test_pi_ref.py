"""Phase-1 checks for the PI-Attn reference model.

1. Invariance: PI output for a row is bitwise identical whether the row is
   computed alone (decode), together with other rows (prefill / batch), or in
   arbitrary chunked-prefill groups, for snapped and unsnapped statistics.
2. Merge algebra: sequential merge vs global-max merge (equal only when
   snapped and without underflow); measured on normal and extreme scores.
3. Baseline: FA2-style running-max attention is NOT invariant to tile size,
   split points, or prefill-vs-decode.
4. Accuracy vs float64 attention.
"""
import itertools
import math
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from pi_ref import F32, bf16, fa2_attention, pi_attention  # noqa: E402

torch.manual_seed(0)
D = 64


def case(T, temp):
    q = bf16(torch.randn(T, D) * temp)
    K = bf16(torch.randn(T, D))
    V = bf16(torch.randn(T, D))
    return q, K, V


def exact(q, pos, K, V, scale):
    q, K, V = q.double(), K.double(), V.double()
    out = []
    for r in range(q.shape[0]):
        n = int(pos[r]) + 1
        s = (q[r] @ K[:n].T) * scale
        p = torch.softmax(s, 0)
        out.append(p @ V[:n])
    return torch.stack(out).float()


def bits_equal(a, b):
    return torch.equal(a.view(torch.int32), b.view(torch.int32))


def run():
    scale = 1 / math.sqrt(D)
    report = {}
    for T, C, temp, snap in itertools.product([150, 333], [16, 64], [1.0, 8.0], [True, False]):
        q, K, V = case(T, temp)
        pos = torch.arange(T)
        rows = torch.randperm(T)[:24]
        full = pi_attention(q, pos, K, V, C, scale, snap)                     # prefill, all rows
        single = torch.stack([pi_attention(q[r:r + 1], pos[r:r + 1], K[: r + 1], V[: r + 1], C, scale, snap)[0]
                              for r in rows.tolist()])                       # decode: cache only up to r
        groups = torch.tensor(sorted(torch.randperm(T - 1)[:4].add(1).tolist()))
        pieces = []
        for a, b in zip([0] + groups.tolist(), groups.tolist() + [T]):         # chunked prefill cuts
            pieces.append(pi_attention(q[a:b], pos[a:b], K[:b], V[:b], C, scale, snap))
        chunked = torch.cat(pieces)
        ok_decode = bits_equal(full[rows], single)
        ok_chunked = bits_equal(full, chunked)
        err = (full - exact(q, pos, K, V, scale)).abs().max().item()
        report[(T, C, temp, snap)] = (ok_decode, ok_chunked, err)
        print(f"PI   T={T} C={C} temp={temp} snap={snap}: decode==prefill {ok_decode}, "
              f"chunked==prefill {ok_chunked}, max|err vs fp64| {err:.3e}", flush=True)

    # Merge algebra: sequential vs global-max merge.
    for temp in [1.0, 8.0, 40.0, 200.0]:
        q, K, V = case(256, temp)
        pos = torch.full((32,), 255)
        qq = q[:32]
        for snap in [True, False]:
            a = pi_attention(qq, pos, K, V, 16, scale, snap, merge="seq")
            b = pi_attention(qq, pos, K, V, 16, scale, snap, merge="global")
            diff = (a.view(torch.int32) != b.view(torch.int32)).float().mean().item()
            print(f"MERGE temp={temp} snap={snap}: seq vs global element mismatch rate {diff:.4f}", flush=True)

    # Baseline FA2: tile size, split points, prefill vs decode.
    for temp in [1.0, 8.0]:
        q, K, V = case(333, temp)
        pos = torch.arange(333)
        rows = torch.arange(200, 333, 7)
        base = fa2_attention(q[rows], pos[rows], K, V, 64, scale)
        tile32 = fa2_attention(q[rows], pos[rows], K, V, 32, scale)
        split = fa2_attention(q[rows], pos[rows], K, V, 64, scale, split_points=(100, 190, 260))
        rate = lambda x: (x.view(torch.int32) != base.view(torch.int32)).any(1).float().mean().item()
        err = (base - exact(q[rows], pos[rows], K, V, scale)).abs().max().item()
        print(f"FA2  temp={temp}: rows differing tile64 vs tile32 {rate(tile32):.2f}, "
              f"no-split vs 3-split {rate(split):.2f}, max|err vs fp64| {err:.3e}", flush=True)

    bad = [k for k, v in report.items() if not (v[0] and v[1])]
    print("INVARIANCE FAILURES:", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(run())
