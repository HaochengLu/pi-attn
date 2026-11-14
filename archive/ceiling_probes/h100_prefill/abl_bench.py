import math, os, sys
sys.path.insert(0, os.getcwd())
import flashinfer, torch
from bench_attention import PAGE, D, dev, ext, make_cache, timeit, warm_gpu
warm_gpu()
VARS = [1, 3, 4, 100, 101, 102, 104, 106, 108, 116, 107, 109, 114]
NAMES = {1: "mma4w", 3: "mma8w", 4: "mma8wQ", 100: "v5", 101: "nocopy", 102: "noS", 104: "noPV", 106: "noMMA", 108: "noSoft", 116: "noResc", 107: "softOnly", 109: "mmaOnly", 114: "copyOnly"}
for hq, hkv, batch, q_len, prefix in [(32, 8, 1, 4096, 0), (32, 8, 1, 1024, 0), (32, 8, 8, 1024, 0), (16, 8, 1, 2048, 0), (32, 8, 1, 128, 4096)]:
    kv_len = q_len + prefix
    kv, indptr, indices, last, lens = make_cache(batch, kv_len, hkv)
    q = torch.randn(batch * q_len, hq, D, device=dev, dtype=torch.bfloat16)
    qo = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * q_len
    fi = {}
    for be in ("auto", "fa2", "fa3"):
        try:
            w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.empty(256 << 20, device=dev, dtype=torch.uint8), "NHD", backend=be)
            w.plan(qo.cpu(), indptr.cpu(), indices, last.cpu(), hq, hkv, D, PAGE, causal=True, q_data_type=torch.bfloat16)
            fi[be] = timeit(lambda i: w.run(q, kv[i]), len(kv))
            if be == "auto": fi["auto_is"] = w._backend
            del w
        except Exception as e:
            fi[be] = f"err:{type(e).__name__}:{str(e)[:80]}"
    t = {}
    for v in VARS:
        try:
            t[v] = timeit(lambda i: ext.prefill(q, *kv[i], qo, indptr, indices, lens, q_len, kv_len, 1 / math.sqrt(D), 1 if v >= 100 else 0, v), len(kv))
        except Exception as e:
            t[v] = f"err:{str(e)[:60]}"
    print(f"PRE hq={hq} b={batch} q={q_len} prefix={prefix}: FI {fi} | " + " | ".join(f"{NAMES[v]} {x}" for v, x in t.items()), flush=True)
    del kv
    torch.cuda.empty_cache()
