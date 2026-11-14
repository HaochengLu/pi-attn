"""H100: Qwen3-1.7B projections, 28 independent layer copies in one CUDA graph (every launch streams its weight
from DRAM), cuBLAS vs the engine's candidate ids: best id before (ids <= 112) and after (all ids), per M.
Before that ("hash"), on the inputs of eval/xgpu_hashes.py, the SHA-256 of the outputs of ids 1 and 13 (the two
that eval/xgpu_hashes.py hashes) and of every wgmma id against the reference hashes
(results/modal/H100_sm90/hashes_H100.json). Run from kernels/."""
import hashlib, json, sys, torch
sys.path.insert(0, "/pi/kernels")
import test_gemm as G
LAYERS = 28
SMALL_NEW = (113, 114)  # 64-row cluster tiles: never contenders at M >= 1024
SHAPES = {"qkv": (4096, 2048), "o": (2048, 2048), "gate_up": (12288, 2048), "down": (2048, 6144)}
if "hash" in sys.argv:
    ref = json.load(open("/pi/results/modal/H100_sm90/hashes_H100.json"))["gemm"]
    def h(t): return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]
    def cpu_randn(seed, *shape, scale=1.0):
        g = torch.Generator().manual_seed(seed)
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16).to("cuda")
    ids = [1, 13] + [k for k in G.KERNELS if k >= 90]
    ran = {k: 0 for k in ids}; bad = []
    for i, (name, (N, K)) in enumerate(G.SHAPES.items()):
        W = cpu_randn(1000 + i, N, K, scale=K ** -0.5); X = cpu_randn(2000 + i, 1024, K)
        for M in (1, 8, 128, 1024):
            Xm = X[:M].contiguous()
            for k in ids:
                y = G.ext.linear(Xm, W, k, False)
                if y is None: continue
                ran[k] += 1
                if h(y) != ref[f"{name} M={M}"]["pi"]: bad.append((name, M, k))
        del W, X
    print("HASHX launches per id:", ran, flush=True)
    print(f"HASHX {sum(ran.values())} launches (ids 1, 13 and every wgmma id) over {len(ref)} cases; hashes differing from hashes_H100.json: {len(bad)} {bad[:8]}", flush=True)
torch.manual_seed(0)
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
for M in [int(a) for a in sys.argv[1:] if a.isdigit()] or (1, 8, 16, 32, 64, 128, 192, 256, 512, 1024, 2048, 2304, 4096, 8192):
    tc = tb = ta = 0.0; line = []
    for n, (N, K) in SHAPES.items():
        x = torch.randn(M, K, device="cuda").bfloat16()
        cb = burst(lambda w: torch.matmul(x, w.T), W[n])
        res = {}
        for k in G.candidates(M, N, K):
            tiles = None
            if M >= 1024 and (k < 90 and k not in (10, 11, 16, 69)): continue  # small mma tiles: never contenders up here
            if M >= 1024 and k in (94, 98, 99, 100, 101, 102, 103, 104, 105, 109) + SMALL_NEW: continue
            r = burst(lambda w: G.ext.linear(x, w, k, False), W[n], reps=3 if M >= 1024 else 5)
            res[k] = r
        old = min((k for k in res if k <= 112), key=res.get); new = min(res, key=res.get)
        line.append(f"{n} cublas {cb:.1f} before k{old}:{res[old]:.1f} ({res[old]/cb:.2f}) after k{new}:{res[new]:.1f} ({res[new]/cb:.2f})")
        tc += cb; tb += res[old]; ta += res[new]
    print(f"TABLE M={M:5d} us/launch: " + " | ".join(line), flush=True)
    print(f"TABLE M={M:5d} x28 layers: cuBLAS {tc*LAYERS:.0f} us, before {tb*LAYERS:.0f} ({tb/tc:.3f}x), after {ta*LAYERS:.0f} ({ta/tc:.3f}x)", flush=True)
