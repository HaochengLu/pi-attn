"""Prefill kernel time of one request (round 6): one launch against the wave split (prefill_wave_head: the first
`head` tokens, whose CTAs fill whole waves of SMs, then the rest as an extend), and against every other head that
is a whole number of waves or half-waves, to show the rule; us per call, medians over ROUNDS interleaved rounds
(bench_attention.timeit). The split's output is checked bitwise against the one launch. FlashInfer's paged prefill
for reference, in one launch and in the same two. Run from kernels/. usage: wave_split_bench.py [rounds]"""
import math, os, subprocess, sys
sys.path.insert(0, os.getcwd())
import flashinfer, torch
from bench_attention import PAGE, D, dev, ext, make_cache, timeit, warm_gpu
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
clock = lambda: subprocess.run("nvidia-smi --query-gpu=clocks.sm --format=csv,noheader".split(), capture_output=True, text=True).stdout.strip()
warm_gpu()
SMS = torch.cuda.get_device_properties(0).multi_processor_count
SHAPES = [(2048, 0), (2048, 2048), (2048, 6144), (2048, 14336), (2048, 28000), (1024, 0), (1024, 7168), (1280, 0), (1280, 8192), (1536, 8192), (1792, 8192),
          (4096, 0), (4096, 12288), (8192, 0), (1904, 14336), (704, 8192), (512, 8192)]
if os.environ.get("R6_SHAPES"): SHAPES = [tuple(int(x) for x in s.split(",")) for s in os.environ["R6_SHAPES"].split(";")]
for hq, hkv in [(16, 8), (32, 8)]:
    group = hq // hkv
    for q_len, prefix in SHAPES:
        kv_len = q_len + prefix
        kv, indptr, indices, last, lens = make_cache(1, kv_len, hkv)
        q = torch.randn(q_len, hq, D, device=dev, dtype=torch.bfloat16)
        ints = lambda *v: torch.tensor(v, device=dev, dtype=torch.int32)
        meta = {}  # per head: (q of the head, its qo_indptr and kv length, q of the rest, its qo_indptr), made outside graph capture
        def pi_split(i, head):
            if head not in meta:
                meta[head] = (q[:head].contiguous(), ints(0, head), ints(prefix + head), q[head:].contiguous(), ints(0, q_len - head))
            qh, qoh, lh, qr, qor = meta[head]
            a = ext.prefill(qh, *kv[i], qoh, indptr, indices, lh, head, prefix + head, 1 / math.sqrt(D), 0, 0)
            b = ext.prefill(qr, *kv[i], qor, indptr, indices, lens, q_len - head, kv_len, 1 / math.sqrt(D), 0, 0)
            return a, b
        qo_one = ints(0, q_len)
        pi_one = lambda i: ext.prefill(q, *kv[i], qo_one, indptr, indices, lens, q_len, kv_len, 1 / math.sqrt(D), 0, 0)
        rule = ext.prefill_wave_head(hq, hkv, q_len, prefix)
        per_cta = 128 // group
        ctas = -(-q_len * group // 128) * hkv
        heads = sorted({(w * SMS // 2 // hkv) * per_cta for w in range(1, 2 * ctas // SMS + 1)} - {0, q_len} | ({rule} if rule else set()))
        heads = [h for h in heads if 0 < h < q_len]
        for h in heads:
            assert torch.equal(torch.cat(pi_split(0, h)).view(torch.int16), pi_one(0).view(torch.int16)), "split bits differ"
        def fi(ql, pre):
            w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.empty(256 << 20, device=dev, dtype=torch.uint8), "NHD")
            n_pages = (pre + ql + PAGE - 1) // PAGE
            w.plan(ints(0, ql).cpu(), ints(0, n_pages).cpu(), indices[:n_pages].contiguous(), ints((pre + ql - 1) % PAGE + 1).cpu(), hq, hkv, D, PAGE, causal=True, q_data_type=torch.bfloat16)
            return w
        w_one = fi(q_len, prefix)
        r = {"fi": [], "one": []}; r.update({h: [] for h in heads})
        for _ in range(ROUNDS):
            r["fi"].append(timeit(lambda i: w_one.run(q, kv[i]), len(kv)))
            r["one"].append(timeit(pi_one, len(kv)))
            for h in heads: r[h].append(timeit(lambda i: pi_split(i, h), len(kv)))
        m = {k: sorted(v)[len(v) // 2] for k, v in r.items()}
        best = min(["one"] + heads, key=lambda k: m[k])
        chosen = m[rule] if rule else m["one"]
        print(f"WAVE hq={hq} q={q_len:5d} prefix={prefix:5d} ({ctas:4d} CTAs = {ctas / SMS:4.2f} waves): fi {m['fi']:8.1f} | one launch {m['one']:8.1f} ({m['one']/m['fi']:.2f}) | "
              + " ".join(f"head {h}: {m[h]:.1f} ({m[h]/m['one']:.3f})" for h in heads) + f" | rule: head {rule} {chosen:8.1f} ({chosen/m['fi']:.2f}; {chosen/m['one']:.3f} of one launch; best {best} {m[best]/m['one']:.3f}) | {clock()}", flush=True)
        del kv, q
        torch.cuda.empty_cache()
