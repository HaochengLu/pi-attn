"""The attention entries of eval/xgpu_hashes.py (same seeds, same calls) against
results/modal/H100_sm90/hashes_H100.json: decode with the planner (v0, told the longest request) and with every
decode kernel forced (PI_HASH_DECODE, default 1,2), prefill with the device's kernel."""
import hashlib, json, math, os, sys
import torch
sys.path.insert(0, os.getcwd())
import bench_attention as b
dev = "cuda"
ref = json.load(open("/pi/results/modal/H100_sm90/hashes_H100.json"))["attention"]
def h(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]
def cpu_randn(seed, *shape, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16).to(dev)
HQ, HKV, D, PAGE = 32, 8, 128, b.PAGE
bad = 0
for name, batch, q_len, prefix in (("prefill b1 q512", 1, 512, 0), ("prefill b4 q2048", 4, 2048, 0),
                                   ("extend b1 q128 prefix 4096", 1, 128, 4096), ("decode b8 kv 2048", 8, 1, 2047),
                                   ("decode b1 kv 32768", 1, 1, 32767)):
    kv = q_len + prefix
    pages = -(-kv // PAGE)
    k = cpu_randn(3000, batch * pages + 1, PAGE, HKV, D)
    v = cpu_randn(3001, batch * pages + 1, PAGE, HKV, D)
    q = cpu_randn(3002, batch * q_len, HQ, D)
    indices = torch.arange(1, batch * pages + 1, device=dev, dtype=torch.int32)
    indptr = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * pages
    lens = torch.full((batch,), kv, device=dev, dtype=torch.int32)
    scale = 1 / math.sqrt(D)
    if q_len == 1:
        rows = batch * -(-kv // b.ext.GROUP_KEYS)
        ws_o = torch.empty(rows, HQ, D, device=dev)
        ws_ml = torch.empty(rows, HQ, 2 * b.ext.DECODE_WARPS, device=dev)
        cnt = torch.zeros(rows, HQ, b.ext.DECODE_WARPS, dtype=torch.int32, device=dev)
        got = {}
        for var in [0] + [int(x) for x in os.environ.get("PI_HASH_DECODE", "1,2").split(",")]:
            b.ext.set_decode_variant(var)
            got[f"decode v{var}"] = h(b.ext.decode(q, k, v, indptr, indices, lens, ws_o, ws_ml, cnt, kv, scale, kv))
            assert cnt.count_nonzero().item() == 0
        b.ext.set_decode_variant(0)
    else:
        qo = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * q_len
        got = {"prefill v0": h(b.ext.prefill(q, k, v, qo, indptr, indices, lens, q_len, kv, scale, 0, 0))}
    ok = all(x == ref[name]["pi"] for x in got.values())
    bad += not ok
    print(f"HASH {name}: reference {ref[name]['pi']} | " + " ".join(f"{a}={x}" for a, x in got.items()) + f" | {'EQUAL' if ok else 'DIFFERENT'}", flush=True)
print("attention hashes equal to results/modal/H100_sm90/hashes_H100.json:", bad == 0, flush=True)
