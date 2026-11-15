"""Prefill kernel time on class A (round 5): the shipped grid-order rule (prefill_longest_first: order 0) against ascending
(asc, the order before round 5) and longest first everywhere (lf); us per call, interleaved over ROUNDS rounds,
medians (bench_attention.timeit). FlashInfer's paged prefill for reference. Run from kernels/."""
import math, os, subprocess, sys
sys.path.insert(0, os.getcwd())
import flashinfer, torch
from bench_attention import PAGE, D, dev, ext, make_cache, timeit, warm_gpu
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
clock = lambda: subprocess.run("nvidia-smi --query-gpu=clocks.sm --format=csv,noheader".split(), capture_output=True, text=True).stdout.strip()
warm_gpu()
SHAPES = [(1, 64, 0), (1, 512, 0), (1, 1024, 0), (1, 2048, 0), (1, 4096, 0), (1, 8192, 0), (1, 2048, 2048), (1, 2048, 6144), (1, 2048, 14336),
          (1, 512, 8192), (1, 128, 4096), (1, 128, 16384), (2, 1024, 0), (2, 4096, 0), (3, 2048, 0), (4, 512, 0), (8, 256, 0), (8, 512, 0),
          (8, 1024, 0), (12, 1024, 0), (16, 128, 0), (16, 512, 0), (32, 64, 0), (32, 512, 0), (64, 128, 0), (4, 2048, 2048), (16, 2048, 0)]
for hq, hkv in [(16, 8), (32, 8)]:
    for batch, q_len, prefix in SHAPES:
        kv_len = q_len + prefix
        kv, indptr, indices, last, lens = make_cache(batch, kv_len, hkv)
        q = torch.randn(batch * q_len, hq, D, device=dev, dtype=torch.bfloat16)
        qo = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * q_len
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.empty(256 << 20, device=dev, dtype=torch.uint8), "NHD")
        w.plan(qo.cpu(), indptr.cpu(), indices, last.cpu(), hq, hkv, D, PAGE, causal=True, q_data_type=torch.bfloat16)
        r = {"fi": [], "asc": [], "lf": [], "rule": []}
        def pi(order):
            ext.set_prefill_order(order)
            t = timeit(lambda i: ext.prefill(q, *kv[i], qo, indptr, indices, lens, q_len, kv_len, 1 / math.sqrt(D), 0, 0), len(kv))
            ext.set_prefill_order(0)
            return t
        for _ in range(ROUNDS):
            r["fi"].append(timeit(lambda i: w.run(q, kv[i]), len(kv)))
            r["asc"].append(pi(1)); r["lf"].append(pi(2)); r["rule"].append(pi(0))
        m = {k: sorted(v)[len(v) // 2] for k, v in r.items()}
        mb = batch * (-(-kv_len // 512)) * 512 * hkv * 512 / 2**20
        print(f"PRE hq={hq} b={batch:2d} q={q_len:4d} prefix={prefix:5d} (K/V {mb:5.0f} MiB): fi {m['fi']:8.1f} | asc {m['asc']:8.1f} ({m['asc']/m['fi']:.2f}) | lf {m['lf']:8.1f} "
              f"| rule {m['rule']:8.1f} ({m['rule']/m['fi']:.2f}) | rule/asc {m['rule']/m['asc']:.3f} lf/asc {m['lf']/m['asc']:.3f} | {clock()}", flush=True)
        del kv
        torch.cuda.empty_cache()
