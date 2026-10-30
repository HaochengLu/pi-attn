"""Prefill kernel variants (Q in smem / registers x 4 / 8 warps per CTA) in the engine regime
(bench_attention.timeit: graph replay over a per-layer KV ring), planner splits, vs FlashInfer.
usage (from kernels/): python ../archive/ceiling_probes/a800/close/probe_prefill_warps.py"""
import math, subprocess, sys
import torch
sys.path.insert(0, '.')
import bench_attention as B

print('clock:', subprocess.run(['nvidia-smi', '--query-gpu=clocks.sm,clocks.applications.graphics', '--format=csv,noheader'],
                               capture_output=True, text=True).stdout.strip(), flush=True)
B.warm_gpu()
print('clock under load:', subprocess.run(['nvidia-smi', '--query-gpu=clocks.sm', '--format=csv,noheader'],
                                          capture_output=True, text=True).stdout.strip(), flush=True)
for hq, hkv in [(16, 8), (32, 8)]:
    for batch, ql, pre in B.PREFILL:
        kv_len = ql + pre
        kv, indptr, indices, last, lens = B.make_cache(batch, kv_len, hkv)
        q = torch.randn(batch * ql, hq, B.D, device='cuda', dtype=torch.bfloat16)
        qo = torch.arange(0, batch + 1, device='cuda', dtype=torch.int32) * ql
        ref = B.ext.prefill(q, *kv[0], qo, indptr, indices, lens, ql, kv_len, 1 / math.sqrt(B.D), 0, 2)
        line = f'hq={hq} b={batch} q={ql:4d} prefix={pre:5d}:'
        for var in (2, 1, 4, 3):
            out = B.ext.prefill(q, *kv[0], qo, indptr, indices, lens, ql, kv_len, 1 / math.sqrt(B.D), 0, var)
            assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), f'variant {var} bits differ'
            t = B.timeit(lambda i: B.ext.prefill(q, *kv[i], qo, indptr, indices, lens, ql, kv_len, 1 / math.sqrt(B.D), 0, var), len(kv))
            line += f' v{var}={t}'
        print(line, flush=True)
        del kv
        torch.cuda.empty_cache()
