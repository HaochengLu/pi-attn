"""E6: per-layer decode step cost, exact (GDNExactDecode) vs fla fused_recurrent, Qwen3.5-4B GDN shape
(HV=32 heads after q/k repeat, K=V=128). Both captured in CUDA graphs; reports graph-replay time per step and
summed kernel time from the profiler. Exact includes its l2norm of q,k and the row writes into the chunk buffer;
slots sit at random rows of their chunks (steady-state decode, includes folds at their natural 1/64 rate)."""
import sys, torch, torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
from fla.modules.l2norm import l2norm_fwd
from gdn_exact import GDNExactDecode, BT
from gdn_exact2 import GDNExactDecode2

dev, H, HV, K, V = 'cuda', 32, 32, 128, 128
scale = K ** -0.5

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
    t = e0.elapsed_time(e1) / iters * 1e3
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(50): gr.replay()
        torch.cuda.synchronize()
    kern = sum(e.device_time_total for e in p.key_averages() if e.device_type.name == 'CUDA') / 50
    return t, kern

print(f'{"B":>4} | {"recurrent graph/kern us":>24} | {"exact v1 graph/kern":>20} | {"exact v2 graph/kern":>20} | v2/rec', flush=True)
for B in [1, 8, 32, 64, 128, 256]:
    gen = torch.Generator(device=dev).manual_seed(B)
    q = torch.randn(1, B, HV, K, device=dev, generator=gen).bfloat16()
    k = torch.randn(1, B, H, K, device=dev, generator=gen).bfloat16()
    v = torch.randn(1, B, HV, V, device=dev, generator=gen).bfloat16()
    g = F.logsigmoid(torch.randn(1, B, HV, device=dev, generator=gen) + 3).float()
    beta = torch.sigmoid(torch.randn(1, B, HV, device=dev, generator=gen)).bfloat16()
    state = torch.randn(B, HV, K, V, device=dev, generator=gen) * 0.1
    cu = torch.arange(B + 1, device=dev, dtype=torch.int32)
    def rec():
        o, s = fused_recurrent_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, initial_state=state,
                                                output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
        return o                  # state read + written once (= an in-place update's traffic); no copy-back counted
    res = []
    for cls in (GDNExactDecode, GDNExactDecode2):
      dec = cls(B, H, HV, K, V, dev)
      dec.rows.copy_(torch.randint(0, BT, (B,), device=dev, generator=gen).int())
      dec.kbuf.copy_(F.normalize(torch.randn_like(dec.kbuf.float()), dim=-1).bfloat16()); dec.vbuf.normal_()
      dec.gbuf.copy_(F.logsigmoid(torch.randn_like(dec.gbuf) + 3)); dec.bbuf.uniform_(0, 1)
      dec.S.normal_(0, 0.1); dec.Hb.copy_(dec.S)
      def exact(dec=dec):
          qn = l2norm_fwd(q[0])[0]; kn = l2norm_fwd(k[0])[0]
          return dec.step(qn, kn, v[0], g[0], beta[0], scale)
      res.append(graph_time(exact))
    (rg, rk), (e1g, e1k), (e2g, e2k) = graph_time(rec), *res
    print(f'{B:4d} | {rg:11.1f} / {rk:9.1f} | {e1g:9.1f} / {e1k:8.1f} | {e2g:9.1f} / {e2k:8.1f} | {e2g / rg:5.2f}x', flush=True)
