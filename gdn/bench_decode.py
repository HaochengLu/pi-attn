"""Per-layer GDN decode step: exact_decode vs fla 0.5.0 fused_recurrent_gated_delta_rule, CUDA graphs.

Qwen3.5-4B linear-attention shape (HV=32 heads after HF's q/k repeat, K=V=128, fp32 state). The recurrent baseline
reads the state and writes the new one once per step (the traffic of an in-place update; no copy-back is counted)
and does its l2norm in-kernel. exact_decode includes its l2norm and buffer writes; slots sit at random absolute
positions (steady state: random chunk rows; the benchmark does not commit, so no folds are timed: a fold costs one
extra state read+write per 64 tokens). Reported: graph replay time per step (us), mean of 200 replays.  usage: PYTHONPATH=<fla> python gdn/bench_decode.py [nw1:nw2:BV ...]   (from gdn/)"""
import sys, torch, torch.nn.functional as F
from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
from exact_decode import GDNExactDecode
from timing import graph_time

dev, H, HV, K, V = 'cuda', 32, 32, 128, 128


def fill(dec, gen):
    dec.kbuf.copy_(F.normalize(torch.randn_like(dec.kbuf.float()), dim=-1).bfloat16()); dec.vbuf.normal_()
    dec.gbuf.copy_(F.logsigmoid(torch.randn_like(dec.gbuf) + 3)); dec.bbuf.uniform_(0, 1)
    dec.put_states(torch.arange(dec.N, device=dev, dtype=torch.int32), torch.randn(dec.N, HV, K, V, device=dev) * 0.1)


def inputs(B, T, gen):
    return (torch.randn(B, T, H, K, device=dev, generator=gen).bfloat16(), torch.randn(B, T, H, K, device=dev, generator=gen).bfloat16(),
            torch.randn(B, T, HV, V, device=dev, generator=gen).bfloat16(),
            F.logsigmoid(torch.randn(B, T, HV, device=dev, generator=gen) + 3).float(),
            torch.sigmoid(torch.randn(B, T, HV, device=dev, generator=gen)).bfloat16())


if __name__ == '__main__':
    cfgs = [tuple(int(x) for x in a.split(':')) for a in sys.argv[1:]] or [(2, 4, 64)]
    KT = 4
    print('decode: one token per slot per step | verify: a window of 4 tokens per slot (speculative verification);\n'
          'recurrent 4 = fla fused_recurrent over 4 tokens per slot, final state only (no per-token rollback copies)')
    print(f'{"B":>4} | {"rec 1":>7} | ' + ' | '.join(f'exact {a}:{b}:{c} 1' for a, b, c in cfgs) + f' | {"rec 4":>7} | '
          + ' | '.join(f'exact {a}:{b}:{c} 4' for a, b, c in cfgs) + ' | ratios (best exact / rec): 1 tok, 4 tok', flush=True)
    for B in [1, 8, 32, 64, 128, 256]:
        gen = torch.Generator(device=dev).manual_seed(B)
        q, k, v, g, beta = inputs(B, KT, gen)
        state = torch.randn(B, HV, K, V, device=dev, generator=gen) * 0.1
        def rec(T):                                  # varlen: T tokens per slot, inputs prepared outside the graph
            xs = [x[:, :T].reshape(1, B * T, *x.shape[2:]).contiguous() for x in (q, k, v, g, beta)]
            cu = torch.arange(0, B * T + 1, T, device=dev, dtype=torch.int32)
            return lambda: fused_recurrent_gated_delta_rule(q=xs[0], k=xs[1], v=xs[2], g=xs[3], beta=xs[4],
                                                            initial_state=state, output_final_state=True,
                                                            use_qk_l2norm_in_kernel=True, cu_seqlens=cu)[0]
        r1, r4 = graph_time(rec(1)), graph_time(rec(KT))
        pos = torch.randint(0, 4096, (B,), device=dev, generator=gen).int()
        e1, e4 = [], []
        for nw1, nw2, BV in cfgs:
            d1 = GDNExactDecode(B, H, HV, K, V, dev, BV=BV, nw1=nw1, nw2=nw2); fill(d1, gen)
            q1, k1, v1, g1, b1 = (x[:, 0].contiguous() for x in (q, k, v, g, beta))
            e1.append(graph_time(lambda d=d1: d.decode(q1, k1, v1, g1, b1, pos, K ** -0.5)))
            d4 = GDNExactDecode(B, H, HV, K, V, dev, KT=KT, BV=BV, nw1=nw1, nw2=nw2); fill(d4, gen)
            nrows = GDNExactDecode.window(pos, KT)
            e4.append(graph_time(lambda d=d4: d.step(q, k, v, g, beta, pos, nrows, K ** -0.5)))
        print(f'{B:4d} | {r1:7.1f} | ' + ' | '.join(f'{x:19.1f}' for x in e1) + f' | {r4:7.1f} | '
              + ' | '.join(f'{x:19.1f}' for x in e4) + f' | {min(e1) / r1:5.2f}x {min(e4) / r4:5.2f}x', flush=True)
