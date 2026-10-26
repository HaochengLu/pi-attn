"""E3: per-layer decode-step cost of GDN replay (fla chunk on each sequence's partial chunk from its boundary
state) vs. today's fused_recurrent single step. Qwen3.5-4B linear-layer shapes: H=32, K=V=128.
Partial-chunk lengths are uniform in 1..64 (steady-state decode). Reports wall time per call (eager, includes
fla's host-side varlen prep) and summed CUDA kernel time from the profiler."""
import torch, torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

dev, H, K, V, C = 'cuda', 32, 128, 128, 64
g0 = torch.Generator(device=dev).manual_seed(0)

def mk(n):
    q = torch.randn(1, n, H, K, device=dev, generator=g0).bfloat16()
    k = torch.randn(1, n, H, K, device=dev, generator=g0).bfloat16()
    v = torch.randn(1, n, H, V, device=dev, generator=g0).bfloat16()
    g = F.logsigmoid(torch.randn(1, n, H, device=dev, generator=g0) + 3).float()
    beta = torch.sigmoid(torch.randn(1, n, H, device=dev, generator=g0)).bfloat16()
    return q, k, v, g, beta

def timeit(fn, iters=50):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters): fn()
    e1.record(); torch.cuda.synchronize()
    wall = e0.elapsed_time(e1) / iters * 1e3
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(iters): fn()
        torch.cuda.synchronize()
    kern = sum(e.device_time_total for e in p.key_averages()) / iters
    return wall, kern

print(f'{"B":>4} | {"recurrent wall/kern us":>24} | {"replay wall/kern us":>22} | kern ratio', flush=True)
for B in [1, 8, 32, 128]:
    h0 = torch.randn(B, H, K, V, device=dev, generator=g0) * 0.1
    x1 = mk(B)
    cu1 = torch.arange(B + 1, device=dev, dtype=torch.int32)
    rec = lambda: fused_recurrent_gated_delta_rule(q=x1[0], k=x1[1], v=x1[2], g=x1[3], beta=x1[4], initial_state=h0,
                                                   output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=cu1)
    lens = torch.randint(1, C + 1, (B,), generator=torch.Generator().manual_seed(B))
    xr = mk(int(lens.sum()))
    cur = torch.cat([torch.zeros(1, dtype=torch.int64), lens.cumsum(0)]).to(dev, torch.int32)
    rep = lambda: chunk_gated_delta_rule(*xr, initial_state=h0, output_final_state=True,
                                         use_qk_l2norm_in_kernel=True, cu_seqlens=cur)
    (rw, rk), (cw, ck) = timeit(rec), timeit(rep)
    print(f'{B:4d} | {rw:10.1f} / {rk:9.1f} | {cw:9.1f} / {ck:9.1f} | {ck / rk:6.2f}x', flush=True)
