"""Prefill kernel time of probe builds of the tree (each a set of -D flags; wrong bits allowed), interleaved over
rounds in one process, with FlashInfer's paged prefill for reference; us per call (bench_attention.timeit).
usage: r6_probe_bench.py rounds name[@tree][:-Dflag[,-Dflag]] ... (tree: a directory of $R6 with kernels/)   env: R6_SHAPES="b,q,prefix;..." R6_HQ="16,32" R6_VARIANT"""
import math, os, subprocess, sys
sys.path.insert(0, os.getcwd())
import flashinfer, torch
from torch.utils.cpp_extension import load
from build import cuda_build_flags, HERE
def build(name, flags):
    arch, cflags, includes, ldflags = cuda_build_flags(None)
    src = HERE
    if "@" in name: name, src = name.split("@"); src = os.path.join(os.environ["R6"], src, "kernels")
    return load(name=f"r6p_{name}", sources=[os.path.join(src, "pi_attn_torch.cu")], extra_include_paths=[src] + includes,
                extra_cuda_cflags=cflags + ["-lineinfo"] + flags, extra_ldflags=ldflags)
ROUNDS = int(sys.argv[1])
specs = [(a.split(":")[0], [f for f in a.split(":")[1].split(",") if f] if ":" in a else []) for a in sys.argv[2:]]
exts = {n.split("@")[0]: build(n, f) for n, f in specs}
specs = [(n.split("@")[0], f) for n, f in specs]
if os.environ.get("R6_BUILD_ONLY"): sys.exit(0)
os.environ["PI_ATTN_SUFFIX"] = "_r6pb"
from bench_attention import PAGE, D, dev, make_cache, timeit, warm_gpu
clock = lambda: subprocess.run("nvidia-smi --query-gpu=clocks.sm --format=csv,noheader".split(), capture_output=True, text=True).stdout.strip()
warm_gpu()
SHAPES = [tuple(int(x) for x in s.split(",")) for s in os.environ.get("R6_SHAPES", "1,8192,0;1,16384,0;32,512,0;16,2048,0;8,1024,0;1,2048,0;1,512,8192").split(";")]
VAR = int(os.environ.get("R6_VARIANT", "0")); SPL = int(os.environ.get("R6_SPLITS", "0"))
for hq in [int(x) for x in os.environ.get("R6_HQ", "16").split(",")]:
    hkv = 8
    for batch, q_len, prefix in SHAPES:
        kv_len = q_len + prefix
        kv, indptr, indices, last, lens = make_cache(batch, kv_len, hkv)
        q = torch.randn(batch * q_len, hq, D, device=dev, dtype=torch.bfloat16)
        qo = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * q_len
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.empty(256 << 20, device=dev, dtype=torch.uint8), "NHD")
        w.plan(qo.cpu(), indptr.cpu(), indices, last.cpu(), hq, hkv, D, PAGE, causal=True, q_data_type=torch.bfloat16)
        r = {"fi": []}; r.update({n: [] for n in exts})
        for _ in range(ROUNDS):
            r["fi"].append(timeit(lambda i: w.run(q, kv[i]), len(kv)))
            for n, e in exts.items():
                r[n].append(timeit(lambda i: e.prefill(q, *kv[i], qo, indptr, indices, lens, q_len, kv_len, 1 / math.sqrt(D), SPL, VAR), len(kv)))
        med = {k: sorted(v)[len(v) // 2] for k, v in r.items()}
        base = med[specs[0][0]]
        print(f"PRE hq={hq} b={batch:2d} q={q_len:5d} prefix={prefix:5d}: fi {med['fi']:8.1f} | " + " | ".join(f"{n} {med[n]:8.1f} ({med[n]/med['fi']:.2f}, {med[n]/base:.3f})" for n in exts) + f" | {clock()}", flush=True)
        del kv, q, w
        torch.cuda.empty_cache()
