"""Prefill kernel time on class A by grid order: ascending (asc), longest first (lf: row tiles descending in the slowest index) and descending (desc: row tile fastest), the device's variant and
plan, interleaved over ROUNDS rounds; us per call (bench_attention.timeit: CUDA graph over a per-layer KV ring).
FlashInfer's paged prefill for reference. Run from kernels/. usage: r5_prefill_bench.py [rounds]"""
import math, os, subprocess, sys
sys.path.insert(0, os.getcwd())
import flashinfer, torch
from bench_attention import PAGE, D, dev, ext, make_cache, timeit, warm_gpu
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
clock = lambda: subprocess.run("nvidia-smi --query-gpu=clocks.sm --format=csv,noheader".split(), capture_output=True, text=True).stdout.strip()
warm_gpu()
SHAPES = [(1, 64, 0), (1, 512, 0), (1, 1024, 0), (1, 2048, 0), (1, 4096, 0), (1, 8192, 0), (1, 2048, 2048), (1, 2048, 6144), (1, 512, 8192),
          (1, 128, 4096), (1, 128, 16384), (2, 1024, 0), (4, 512, 0), (8, 256, 0), (8, 512, 0), (8, 1024, 0), (16, 128, 0), (32, 64, 0),
          (32, 512, 0), (4, 2048, 2048), (16, 2048, 0)]
for hq, hkv in [(16, 8), (32, 8)]:
    for batch, q_len, prefix in SHAPES:
        kv_len = q_len + prefix
        kv, indptr, indices, last, lens = make_cache(batch, kv_len, hkv)
        q = torch.randn(batch * q_len, hq, D, device=dev, dtype=torch.bfloat16)
        qo = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * q_len
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.empty(256 << 20, device=dev, dtype=torch.uint8), "NHD")
        w.plan(qo.cpu(), indptr.cpu(), indices, last.cpu(), hq, hkv, D, PAGE, causal=True, q_data_type=torch.bfloat16)
        r = {"fi": [], "asc": [], "lf": [], "desc": []}
        def pi(order):
            ext.set_prefill_order(order)
            t = timeit(lambda i: ext.prefill(q, *kv[i], qo, indptr, indices, lens, q_len, kv_len, 1 / math.sqrt(D), 0, 0), len(kv))
            ext.set_prefill_order(0)
            return t
        for _ in range(ROUNDS):
            r["fi"].append(timeit(lambda i: w.run(q, kv[i]), len(kv)))
            r["asc"].append(pi(1)); r["lf"].append(pi(2)); r["desc"].append(pi(3))
        med = {k: sorted(v)[len(v) // 2] for k, v in r.items()}
        print(f"PRE hq={hq} b={batch:2d} q={q_len:4d} prefix={prefix:5d}: fi {med['fi']:8.1f} | asc {med['asc']:8.1f} ({med['asc']/med['fi']:.2f}) | lf {med['lf']:8.1f} ({med['lf']/med['fi']:.2f}) | desc {med['desc']:8.1f} ({med['desc']/med['fi']:.2f}) | lf/asc {med['lf']/med['asc']:.3f} desc/asc {med['desc']/med['asc']:.3f} | rounds asc {r['asc']} lf {r['lf']} desc {r['desc']} | {clock()}", flush=True)
        del kv
        torch.cuda.empty_cache()
