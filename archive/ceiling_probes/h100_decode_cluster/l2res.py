"""decode_kernel and FlashInfer with the K/V of one layer only (L2-resident) against the DRAM ring."""
import math, os, sys
sys.path.insert(0, os.getcwd())
import torch, flashinfer
import bench_attention as B
ext, dev, D, PAGE, HQ, HKV = B.ext, "cuda", 128, 16, 16, 8
B.warm_gpu()
for batch in (1, 8):
    for kv in (96, 320, 640, 1280, 2176):
        kvs, indptr, indices, last, lens = B.make_cache(batch, kv, HKV)
        q = torch.randn(batch, HQ, D, device=dev, dtype=torch.bfloat16)
        ws = torch.empty(256 << 20, device=dev, dtype=torch.uint8)
        rows = batch * -(-kv // ext.GROUP_KEYS)
        ws_o = torch.empty(rows, HQ, D, device=dev); ws_ml = torch.empty(rows, HQ, 8, device=dev)
        cnt = torch.zeros(rows, HQ, 4, dtype=torch.int32, device=dev)
        w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")
        w.plan(indptr.cpu(), indices, last.cpu(), HQ, HKV, D, PAGE, q_data_type=torch.bfloat16)
        r = {}
        for name, ring in (("dram", kvs), ("l2", kvs[:1])):
            r["fi_" + name] = B.timeit(lambda i: w.run(q, ring[i]), len(ring))
            r["pi_" + name] = B.timeit(lambda i: ext.decode(q, *ring[i], indptr, indices, lens, ws_o, ws_ml, cnt, kv, 1 / math.sqrt(D)), len(ring))
        print(f"L2RES b={batch} kv={kv:4d} layers={len(kvs):2d}: " + " ".join(f"{k} {v:6.1f}" for k, v in r.items()), flush=True)
