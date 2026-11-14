"""H100 round 4: cluster (TMA multicast) variants of the wgmma tiles and the X-in-registers (wgmma RS) kernel
against the shipped ids and cuBLAS; cold 28-layer bursts (every launch streams its weight from DRAM).
  build | sanity cl | sanity rs | main [cl] [rs] [M ...]"""
import os, sys, time, torch
sys.path.insert(0, "/pi/kernels")
from torch.utils.cpp_extension import load
from build import cuda_build_flags
from tiles_r4 import CL, RS, BASE
HERE = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()
a, cflags, includes, ldflags = cuda_build_flags("90")
ext = load(name="probe_r4", sources=[os.path.join(HERE, "probe_r4.cu")], extra_include_paths=["/pi/kernels", HERE] + includes,
           extra_cuda_cflags=cflags, extra_ldflags=ldflags, verbose=bool(os.environ.get("V")))
print(f"probe ext build {time.time() - t0:.0f}s", flush=True)
cmd = sys.argv[1]
if cmd == "build": sys.exit(0)
SHAPES = {"qkv": (4096, 2048), "o": (2048, 2048), "gate_up": (12288, 2048), "down": (2048, 6144)}
torch.manual_seed(0)
eq = lambda y, r: y is not None and torch.equal(y.view(torch.int16), r.view(torch.int16))
if cmd == "sanity":
    fam = CL if sys.argv[2] == "cl" else RS
    w = (torch.randn(2048, 2048, device="cuda") * 0.02).bfloat16()
    for M in (256, 200):
        x = torch.randn(M, 2048, device="cuda").bfloat16()
        ref = ext.linear_r4(x, w, 91, 0)
        for k in fam:
            y = ext.linear_r4(x, w, k, 0); torch.cuda.synchronize()
            print("SANITY", sys.argv[2], k, "M", M, "refused" if y is None else ("equal" if eq(y, ref) else "MISMATCH"), flush=True)
    sys.exit(0)
fams = [f for f in ("cl", "rs") if f in sys.argv]
ids = {**(CL if "cl" in fams else {}), **(RS if "rs" in fams else {})}
LAYERS = 28
W = {n: [(torch.randn(N, K, device="cuda") * 0.02).bfloat16() for _ in range(LAYERS)] for n, (N, K) in SHAPES.items()}
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
def burst(fn, ws, reps=5):
    fn(ws[0]); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for w in ws: fn(w)
    ts = []
    for _ in range(reps + 1):
        flush.zero_()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000 / len(ws))
    return sorted(ts[1:])[reps // 2]
bad = {}; ran = 0
for n, (N, K) in SHAPES.items():
    for M in (64, 129, 192, 200, 256, 520, 1000, 2056):
        x = torch.randn(M, K, device="cuda").bfloat16()
        ref = ext.linear_r4(x, W[n][0], 91, 0)
        for k in ids:
            y = ext.linear_r4(x, W[n][0], k, 0)
            if y is None: continue
            ran += 1
            if not eq(y, ref): bad.setdefault(k, []).append((n, M))
    print("bitcheck", n, "done", flush=True)
print(f"BITCHECK {ran} launches of ids {sorted(ids)} vs id 91: mismatching ids {bad}", flush=True)
def ntiles(t, M, N): return -(-M // t[0]) * -(-N // t[1])
for M in [int(a) for a in sys.argv[2:] if a.isdigit()] or (192, 256, 512, 1024, 2048):
    tc = tb = tn = 0.0; line = []
    for n, (N, K) in SHAPES.items():
        x = torch.randn(M, K, device="cuda").bfloat16()
        cb = burst(lambda w: torch.matmul(x, w.T), W[n])
        res = {}
        for k, t in {**BASE, **ids}.items():
            nt = ntiles(t, M, N)
            if nt < 30 or (M <= 512 and nt > 300) or (M >= 1024 and t[0] * t[1] < 128 * 96): continue
            if ext.linear_r4(x, W[n][0], k, 0) is None: continue
            res[str(k)] = burst(lambda w: ext.linear_r4(x, w, k, 0), W[n], reps=3 if M >= 1024 else 5)
            if k in RS: res[f"{k}c"] = burst(lambda w: ext.linear_r4(x, w, k, 1), W[n], reps=3)
        base = min((k for k in res if int(k.rstrip("c")) < 200), key=res.get)
        new = min((k for k in res if not k.endswith("c")), key=res.get)
        print(f"  ALL M={M} {n} cublas {cb:.1f}: " + " ".join(f"k{k}:{v:.1f}" for k, v in res.items()), flush=True)
        line.append(f"{n} cublas {cb:.1f} shipped k{base}:{res[base]:.1f} ({res[base]/cb:.2f}) best k{new}:{res[new]:.1f} ({res[new]/cb:.2f})")
        tc += cb; tb += res[base]; tn += res[new]
    print(f"R4 M={M:4d} us/launch: " + " | ".join(line), flush=True)
    print(f"R4 M={M:4d} x28: cuBLAS {tc*LAYERS:.0f} us, shipped {tb*LAYERS:.0f} ({tb/tc:.3f}x), best {tn*LAYERS:.0f} ({tn/tc:.3f}x)", flush=True)
