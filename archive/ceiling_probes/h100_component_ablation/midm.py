"""H100: Qwen3-1.7B projections, 28 independent layer copies, graph-captured bursts (every launch streams
its weight from DRAM), M = 64..2304: cuBLAS vs every PI candidate id; per-step sums."""
import sys, torch
sys.path.insert(0, "/pi/kernels")
import test_gemm as G
LAYERS = 28
SHAPES = {"qkv": (4096, 2048), "o": (2048, 2048), "gate_up": (12288, 2048), "down": (2048, 6144)}
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
for M in (64, 128, 192, 256, 512, 1024, 2048, 2304):
    tc = tp = 0.0; line = []
    for n, (N, K) in SHAPES.items():
        x = torch.randn(M, K, device="cuda").bfloat16()
        cb = burst(lambda w: torch.matmul(x, w.T), W[n])
        res = {}
        for k in G.candidates(M, N, K):
            try: res[k] = burst(lambda w: G.ext.linear(x, w, k, False), W[n])
            except Exception as ex: pass
        top = sorted(res, key=res.get)[:3]
        s90 = min((k for k in res if k >= 90), key=res.get, default=None)
        line.append(f"{n} cublas {cb:.1f} pi " + " ".join(f"k{k}:{res[k]:.1f}" for k in top) + f" ({res[top[0]]/cb:.2f})" + (f" best-wgmma k{s90}:{res[s90]:.1f}" if s90 is not None and s90 != top[0] else ""))
        tc += cb; tp += res[top[0]]
    print(f"MIDM M={M:4d} us/launch: " + " | ".join(line), flush=True)
    print(f"MIDM M={M:4d} x28 layers: cuBLAS {tc*LAYERS:.0f} us, PI best {tp*LAYERS:.0f} us, +{(tp-tc)*LAYERS:.0f} us ({tp/tc:.3f}x)", flush=True)
