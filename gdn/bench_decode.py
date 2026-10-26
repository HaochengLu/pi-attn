"""Per-layer GDN decode step: exact_decode vs fla 0.5.0 fused_recurrent_gated_delta_rule, CUDA graphs.

Qwen3.5-4B linear-attention shape (HV=32 heads after HF's q/k repeat, K=V=128, fp32 state). The recurrent baseline
reads the state and writes the new one once per step (the traffic of an in-place update; no copy-back is counted)
and does its l2norm in-kernel. exact_decode includes its l2norm and buffer writes; slots sit at random absolute
positions (steady state: random chunk rows, folds at their natural 1-in-64 rate). Reported: graph replay time per
step (us), mean of 200 replays.  usage: PYTHONPATH=<fla>:gdn python gdn/bench_decode.py [nw1:nw2:BV ...]"""
import sys, torch, torch.nn.functional as F
from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
from exact_decode import GDNExactDecode

dev, H, HV, K, V = 'cuda', 32, 32, 128, 128


def graph_time(fn, iters=200):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr): fn()
    for _ in range(10): gr.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters): gr.replay()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1e3


if __name__ == '__main__':
    cfgs = [tuple(int(x) for x in a.split(':')) for a in sys.argv[1:]] or [(2, 4, 128)]
    print(f'{"B":>4} | {"recurrent":>9} | ' + ' | '.join(f'exact {a}:{b}:{c}' for a, b, c in cfgs) + ' | best/recurrent', flush=True)
    for B in [1, 8, 32, 64, 128, 256]:
        gen = torch.Generator(device=dev).manual_seed(B)
        q = torch.randn(B, H, K, device=dev, generator=gen).bfloat16()
        k = torch.randn(B, H, K, device=dev, generator=gen).bfloat16()
        v = torch.randn(B, HV, V, device=dev, generator=gen).bfloat16()
        g = F.logsigmoid(torch.randn(B, HV, device=dev, generator=gen) + 3).float()
        beta = torch.sigmoid(torch.randn(B, HV, device=dev, generator=gen)).bfloat16()
        state = torch.randn(B, HV, K, V, device=dev, generator=gen) * 0.1
        cu = torch.arange(B + 1, device=dev, dtype=torch.int32)
        rec = lambda: fused_recurrent_gated_delta_rule(q=q[None], k=k[None], v=v[None], g=g[None], beta=beta[None],
                                                       initial_state=state, output_final_state=True,
                                                       use_qk_l2norm_in_kernel=True, cu_seqlens=cu)[0]
        rt = graph_time(rec)
        pos = torch.randint(0, 4096, (B,), device=dev, generator=gen).int()
        res = []
        for nw1, nw2, BV in cfgs:
            dec = GDNExactDecode(B, H, HV, K, V, dev, BV=BV, nw1=nw1, nw2=nw2)
            dec.kbuf.copy_(F.normalize(torch.randn_like(dec.kbuf.float()), dim=-1).bfloat16()); dec.vbuf.normal_()
            dec.gbuf.copy_(F.logsigmoid(torch.randn_like(dec.gbuf) + 3)); dec.bbuf.uniform_(0, 1)
            dec.S.normal_(0, 0.1); dec.Hb.copy_(dec.S)
            res.append(graph_time(lambda dec=dec: dec.step(q, k, v, g, beta, pos, K ** -0.5)))
        print(f'{B:4d} | {rt:9.1f} | ' + ' | '.join(f'{x:15.1f}' for x in res) + f' | {min(res) / rt:5.2f}x', flush=True)
