"""Attention time of the engine's prefill-phase batches (round 6): new prompts or chunks followed by decode rows
(requests that add one token). FlashInfer's paged prefill on the whole batch; PI as before round 6 (the whole batch
in one prefill launch); PI as the engine runs it now (the prompts through prefill, the one-token tail through the
decode kernel, its grid sized for the tail's longest request). The two PI outputs are checked bitwise. us per call,
medians over ROUNDS interleaved rounds (bench_attention.timeit: CUDA graph over a ring of per-layer K/V).
Run from kernels/. usage: mixed_batch_bench.py [rounds]   R6_MIX="n_new,q_new,prefix_new,n_dec,kv_dec;..."""
import math, os, subprocess, sys
sys.path.insert(0, os.getcwd())
import flashinfer, torch
from bench_attention import PAGE, D, dev, ext, timeit, warm_gpu
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
clock = lambda: subprocess.run("nvidia-smi --query-gpu=clocks.sm --format=csv,noheader".split(), capture_output=True, text=True).stdout.strip()
warm_gpu()
MIX = [(1, 2048, 0, 58, 2100), (1, 2048, 0, 1, 2100), (1, 2048, 0, 7, 2100), (2, 2048, 0, 62, 1500), (8, 2048, 0, 56, 1500), (1, 2048, 4096, 63, 1500),
       (1, 512, 0, 7, 700), (4, 512, 0, 4, 300), (8, 512, 0, 24, 700), (16, 512, 0, 48, 700), (4, 512, 0, 60, 700), (4, 512, 0, 124, 700),
       (1, 30, 0, 31, 600), (0, 0, 0, 64, 1500)]
if os.environ.get("R6_MIX"): MIX = [tuple(int(x) for x in s.split(",")) for s in os.environ["R6_MIX"].split(";")]
for hq, hkv in [(16, 8), (32, 8)]:
    for n_new, q_new, pre_new, n_dec, kv_dec in MIX:
        qlens = [q_new] * n_new + [1] * n_dec
        kvl = [q_new + pre_new] * n_new + [kv_dec] * n_dec
        pages_per = [(k + PAGE - 1) // PAGE for k in kvl]; n = sum(pages_per) + 1
        layers = min(28, -(-(160 << 20) // (2 * n * PAGE * hkv * D * 2)))
        kv = [(torch.randn(n, PAGE, hkv, D, device=dev, dtype=torch.bfloat16), torch.randn(n, PAGE, hkv, D, device=dev, dtype=torch.bfloat16)) for _ in range(layers)]
        ints = lambda v: torch.tensor(v, device=dev, dtype=torch.int32)
        perm = (torch.randperm(n - 1, device=dev).int() + 1).contiguous()
        indptr, qo, lens = ints([0] + list(torch.tensor(pages_per).cumsum(0))), ints([0] + list(torch.tensor(qlens).cumsum(0))), ints(kvl)
        last = ints([(k - 1) % PAGE + 1 for k in kvl])
        q = torch.randn(sum(qlens), hq, D, device=dev, dtype=torch.bfloat16)
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.empty(256 << 20, device=dev, dtype=torch.uint8), "NHD")
        w.plan(qo.cpu(), indptr.cpu(), perm, last.cpu(), hq, hkv, D, PAGE, causal=True, q_data_type=torch.bfloat16)
        mq, mk, nq = max(qlens), max(kvl), n_new * q_new
        groups = -(-max(kv_dec, 1) // ext.GROUP_KEYS)
        ws_o = torch.empty(max(n_dec, 1) * groups, hq, D, device=dev)
        ws_ml = torch.empty(max(n_dec, 1) * groups, hq, 2 * ext.DECODE_WARPS, device=dev)
        cnt = torch.zeros(max(n_dec, 1) * groups, hq, ext.DECODE_WARPS, dtype=torch.int32, device=dev)
        qo_n, ip_n, lens_n = qo[: n_new + 1].contiguous(), indptr[: n_new + 1].contiguous(), lens[:n_new].contiguous()
        ip_d, lens_d = indptr[n_new:].contiguous(), lens[n_new:].contiguous()
        q_n, q_d = q[:nq].contiguous(), q[nq:].contiguous()
        one = lambda i: ext.prefill(q, *kv[i], qo, indptr, perm, lens, mq, mk, 1 / math.sqrt(D), 0, 0)
        def split(i):
            out = []
            if n_new: out.append(ext.prefill(q_n, *kv[i], qo_n, ip_n, perm, lens_n, q_new, q_new + pre_new, 1 / math.sqrt(D), 0, 0))
            if n_dec: out.append(ext.decode(q_d, *kv[i], ip_d, perm, lens_d, ws_o, ws_ml, cnt, groups * ext.GROUP_KEYS, 1 / math.sqrt(D), kv_dec))
            return out
        assert torch.equal(one(0).view(torch.int16), torch.cat(split(0)).view(torch.int16)), "split bits differ"
        r = {"fi": [], "one": [], "split": []}
        for _ in range(ROUNDS):
            r["fi"].append(timeit(lambda i: w.run(q, kv[i]), len(kv)))
            r["one"].append(timeit(one, len(kv)))
            r["split"].append(timeit(split, len(kv)))
        m = {k: sorted(v)[len(v) // 2] for k, v in r.items()}
        print(f"MIX hq={hq} new {n_new:3d} x {q_new:4d} (+{pre_new:4d}) + one-token requests {n_dec:3d} @ {kv_dec:4d} keys: fi {m['fi']:8.1f} | one prefill launch {m['one']:8.1f} ({m['one']/m['fi']:.2f}) "
              f"| prefill + decode {m['split']:8.1f} ({m['split']/m['fi']:.2f}) | split/one {m['split']/m['one']:.3f} | {clock()}", flush=True)
        del kv, q, w
        torch.cuda.empty_cache()
