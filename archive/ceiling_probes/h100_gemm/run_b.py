"""H100 run B: (1) register-only chain latency of wgmma (per tile config, 1 and 2 CTAs per SM) and of mma.sync;
(2) bit check of the new tile ids against ids 91 / 99; (3) cold 28-layer bursts of every wgmma id vs cuBLAS."""
import sys, torch
from probe_common import build, TILES, per_sm
ext = build()
SMS = 132
def timeit(fn, iters=20):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000)
    return sorted(ts)[len(ts) // 2]
if "chain" in sys.argv:
    out = torch.zeros(4096, device="cuda")
    for ctas, warps in ((132, 1), (132, 4), (132, 8), (132, 16), (264, 8)):
        t1, t2 = (timeit(lambda: ext.mma_chain(out, ctas, warps, st)) for st in (12800, 51200))
        print(f"CHAIN mma.sync m16n8k16: {ctas} CTAs x {warps} warps: {(t2 - t1) / 38400 * 1000:.1f} ns/step", flush=True)
    K1, K2 = 2048, 8192
    for k in sorted(TILES):
        bm, bn, st, bk = TILES[k]
        if bk != 64: continue
        for c in (1, 2, 4):
            if c > per_sm(k): continue
            N = bn * SMS * c
            w2 = torch.zeros(N, K2, device="cuda", dtype=torch.bfloat16)
            x2 = torch.zeros(bm, K2, device="cuda", dtype=torch.bfloat16)
            w1, x1 = w2[:, :K1].contiguous(), x2[:, :K1].contiguous()
            t1 = timeit(lambda: ext.linear90(x1, w1, k, 1)); t2 = timeit(lambda: ext.linear90(x2, w2, k, 1))
            slope = (t2 - t1) / ((K2 - K1) / 16) * 1000
            print(f"CHAIN wgmma k{k} {bm}x{bn}: {c} CTA/SM: {slope:.1f} ns/k16 step, intercept {t1 - slope * K1 / 16 / 1000:.1f} us"
                  f" | {bm * bn * 16 * c / slope / 1e3:.2f} TMAC/s/SM | 384 steps {t1 + (384 - 128) * slope / 1000:.1f} us", flush=True)
            del w1, w2, x1, x2
LAYERS = 28
SHAPES = {"qkv": (4096, 2048), "o": (2048, 2048), "gate_up": (12288, 2048), "down": (2048, 6144)}
torch.manual_seed(0)
W = {n: [(torch.randn(N, K, device="cuda") * 0.02).bfloat16() for _ in range(LAYERS)] for n, (N, K) in SHAPES.items()}
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
def burst(fn, ws, reps=3):
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
NEW = [k for k in TILES if k >= 103]
bad = 0
for n, (N, K) in SHAPES.items():
    for M in (1, 64, 128, 192, 200, 256, 2056):
        x = torch.randn(M, K, device="cuda").bfloat16()
        ref = ext.linear90(x, W[n][0], 91 if M > 64 else 99, 0)
        for k in NEW:
            y = ext.linear90(x, W[n][0], k, 0)
            if not torch.equal(y.view(torch.int16), ref.view(torch.int16)):
                bad += 1; print("BIT MISMATCH", n, M, k, flush=True)
print("BITCHECK new ids vs 91/99: mismatches", bad, flush=True)
def tiles(k, M, N):
    bm, bn, _, _ = TILES[k]
    return -(-M // bm) * -(-N // bn)
Ms = [int(a) for a in sys.argv[1:] if a.isdigit()] or (64, 128, 192, 256, 512, 2056, 2304)
for M in Ms:
    tc = to = tn = 0.0; line = []
    for n, (N, K) in SHAPES.items():
        x = torch.randn(M, K, device="cuda").bfloat16()
        cb = burst(lambda w: torch.matmul(x, w.T), W[n])
        res = {}
        for k in TILES:
            t = tiles(k, M, N)
            if t > 3000 or (M >= 512 and t < 40): continue
            res[k] = burst(lambda w: ext.linear90(x, w, k, 0), W[n])
        old = min((k for k in res if k < 103), key=res.get); top = sorted(res, key=res.get)[:4]
        line.append(f"{n} cublas {cb:.1f} old k{old}:{res[old]:.1f} ({res[old]/cb:.2f}) top " + " ".join(f"k{k}:{res[k]:.1f}" for k in top) + f" ({res[top[0]]/cb:.2f})")
        tc += cb; to += res[old]; tn += res[top[0]]
        print(f"  ALL M={M} {n}: " + " ".join(f"k{k}:{v:.1f}" for k, v in sorted(res.items())), flush=True)
    print(f"RUNB M={M:4d} us/launch: " + " | ".join(line), flush=True)
    print(f"RUNB M={M:4d} x28: cuBLAS {tc*LAYERS:.0f} us, old wgmma best {to*LAYERS:.0f} ({to/tc:.3f}x), all best {tn*LAYERS:.0f} ({tn/tc:.3f}x)", flush=True)
