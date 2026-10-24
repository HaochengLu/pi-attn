"""The shipped decode_kernel (engine HEAD, one K and one V slot) in probe_ring.py's harness and
shapes: built from `git show HEAD` copies as a separate extension. us per call."""
import math, os, subprocess, sys
import torch
from torch.utils.cpp_extension import load
from bench_attention import make_cache, warm_gpu, D
from probe_gate_dram import graph_time

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.join(HERE, "..", "logs", "old_decode_ext")
os.makedirs(OLD, exist_ok=True)
eng = os.path.join(HERE, "..", "llm_infer_rust")
for f in ("pi_attn_kernels.cu", "pi_attn.cuh"):
    open(os.path.join(OLD, f), "w").write(subprocess.check_output(["git", "-C", eng, "show", f"HEAD:src/kernels/pi/{f}"], text=True))
open(os.path.join(OLD, "pi_attn_torch.cu"), "w").write(subprocess.check_output(["git", "-C", os.path.join(HERE, ".."), "show", "HEAD:kernels/pi_attn_torch.cu"], text=True))
ext = load(name="pi_attn_old_decode", sources=[os.path.join(OLD, "pi_attn_torch.cu")], extra_include_paths=[OLD],
           extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_80,code=sm_80", "--expt-relaxed-constexpr"])
GRID = int(sys.argv[1])
BATCHES = [int(x) for x in sys.argv[2].split(",")]
KVS = [int(x) for x in sys.argv[3].split(",")]
warm_gpu()
hq, hkv = 16, 8
for batch in BATCHES:
    for kv in KVS:
        caches = [make_cache(batch, kv, hkv) for _ in range(28)]
        q = torch.randn(batch, hq, D, device="cuda", dtype=torch.bfloat16)
        rows = batch * -(-GRID // ext.GROUP_KEYS)
        ws_o = torch.empty(rows, hq, D, device="cuda")
        ws_ml = torch.empty(rows, hq, 2 * ext.DECODE_WARPS, device="cuda")
        cnt = torch.zeros(rows, hq, ext.DECODE_WARPS, dtype=torch.int32, device="cuda")
        fns = [lambda c=c: ext.decode(q, c[0], c[1], c[2], c[3], c[5], ws_o, ws_ml, cnt, GRID, 1 / math.sqrt(D)) for c in caches]
        print(f"b={batch:2d} kv={kv:5d}: old {graph_time(fns):6.1f}", flush=True)
        del caches; torch.cuda.empty_cache()
