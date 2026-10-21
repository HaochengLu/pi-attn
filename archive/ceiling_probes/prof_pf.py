import math, sys, torch
sys.path.insert(0, "/uniparser/users/luhaocheng/pi_attn/kernels")
import os; os.chdir("/uniparser/users/luhaocheng/pi_attn/kernels")
from bench_attention import PAGE, D, dev, ext, make_cache
from torch.profiler import profile, ProfilerActivity
for hq, (b, ql, pre) in [(32, (1, 128, 16384)), (16, (1, 128, 16384)), (32, (4, 2048, 2048))]:
    kv = ql + pre
    k, v, indptr, indices, last, lens = make_cache(b, kv, 8)
    q = torch.randn(b * ql, hq, D, device=dev, dtype=torch.bfloat16)
    qo = torch.arange(0, b + 1, device=dev, dtype=torch.int32) * ql
    f = lambda: ext.prefill(q, k, v, qo, indptr, indices, lens, ql, kv, 1 / math.sqrt(D), 0)
    for _ in range(5): f()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(20): f()
        torch.cuda.synchronize()
    print(f"hq={hq} b={b} q={ql} pre={pre}")
    for e in p.key_averages():
        if e.device_time_total > 0: print(f"   {e.key[:60]:60s} {e.device_time_total/e.count:8.1f}us x{e.count}")
