"""E7: per-layer decode step, exact v3 (one fused kernel) vs fla fused_recurrent, Qwen3.5-4B GDN shape
(HV=32, K=V=128), CUDA graphs. Recurrent: state read + written once per step (in-place-update traffic, the
strongest baseline). Exact: slots at random chunk rows (steady state, folds at their natural 1/64 rate)."""
import sys, torch, torch.nn.functional as F
from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
from gdn_exact3 import GDNExactDecode3, BT
from gdn_bench import graph_time

dev, H, HV, K, V = 'cuda', 32, 32, 128, 128
scale = K ** -0.5
NW = [int(x) for x in sys.argv[1:]] or [4]
print(f'{"B":>4} | {"recurrent graph/kern us":>24} | ' + ' | '.join(f'v3 nw={w} graph/kern' for w in NW) + ' | best/rec', flush=True)
for B in [1, 8, 32, 64, 128, 256]:
    gen = torch.Generator(device=dev).manual_seed(B)
    q = torch.randn(B, HV, K, device=dev, generator=gen).bfloat16()
    k = torch.randn(B, H, K, device=dev, generator=gen).bfloat16()
    v = torch.randn(B, HV, V, device=dev, generator=gen).bfloat16()
    g = F.logsigmoid(torch.randn(B, HV, device=dev, generator=gen) + 3).float()
    beta = torch.sigmoid(torch.randn(B, HV, device=dev, generator=gen)).bfloat16()
    state = torch.randn(B, HV, K, V, device=dev, generator=gen) * 0.1
    cu = torch.arange(B + 1, device=dev, dtype=torch.int32)
    rec = lambda: fused_recurrent_gated_delta_rule(q=q[None], k=k[None], v=v[None], g=g[None], beta=beta[None],
                                                   initial_state=state, output_final_state=True,
                                                   use_qk_l2norm_in_kernel=True, cu_seqlens=cu)[0]
    rg, rk = graph_time(rec)
    pos = torch.randint(0, 4096, (B,), device=dev, generator=gen).int()
    res = []
    for w in NW:
        dec = GDNExactDecode3(B, H, HV, K, V, dev, num_warps=w)
        dec.kbuf.copy_(F.normalize(torch.randn_like(dec.kbuf.float()), dim=-1).bfloat16()); dec.vbuf.normal_()
        dec.gbuf.copy_(F.logsigmoid(torch.randn_like(dec.gbuf) + 3)); dec.bbuf.uniform_(0, 1)
        dec.S.normal_(0, 0.1); dec.Hb.copy_(dec.S)
        res.append(graph_time(lambda dec=dec: dec.step(q, k, v, g, beta, pos, scale)))
    best = min(x[0] for x in res)
    print(f'{B:4d} | {rg:11.1f} / {rk:9.1f} | ' + ' | '.join(f'{a:9.1f} / {b:8.1f}' for a, b in res) + f' | {best / rg:5.2f}x', flush=True)
