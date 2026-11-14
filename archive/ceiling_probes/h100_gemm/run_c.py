"""H100 run C: deeper W pipelines (ids 120..128) and W-line touching by the idle producer warps (pf 1: blocking
byte loads, pf 2: prefetch.global.L2), cold 28-layer bursts vs cuBLAS; bit check of every (id, pf) against 91 / 99."""
import sys, torch
from probe_common import build, TILES
ext = build()
LAYERS = 28
SHAPES = {"qkv": (4096, 2048), "o": (2048, 2048), "gate_up": (12288, 2048), "down": (2048, 6144)}
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
bad = 0
for n, (N, K) in SHAPES.items():
    for M in (1, 64, 128, 200, 2056):
        x = torch.randn(M, K, device="cuda").bfloat16()
        ref = ext.linear90(x, W[n][0], 91 if M > 64 else 99, 0, 0)
        for k in TILES:
            for pf in (0, 1, 2):
                y = ext.linear90(x, W[n][0], k, 0, pf)
                if y is None or not torch.equal(y.view(torch.int16), ref.view(torch.int16)):
                    bad += 1; print("BIT MISMATCH", n, M, k, pf, flush=True)
print("BITCHECK every id x pf vs 91/99: mismatches", bad, flush=True)
def tiles(k, M, N):
    bm, bn, _, _ = TILES[k]
    return -(-M // bm) * -(-N // bn)
for M in [int(a) for a in sys.argv[1:]] or (8, 64, 128, 192, 256, 512, 2304):
    tc = to = tn = 0.0; line = []
    for n, (N, K) in SHAPES.items():
        x = torch.randn(M, K, device="cuda").bfloat16()
        cb = burst(lambda w: torch.matmul(x, w.T), W[n])
        res = {}
        for k in TILES:
            t = tiles(k, M, N)
            if t > 2000 or (M >= 512 and t < 40) or (M <= 64 and TILES[k][0] > 64 and k not in (123, 124)): continue
            for pf in (0, 1, 2):
                res[(k, pf)] = burst(lambda w: ext.linear90(x, w, k, 0, pf), W[n])
        old = min((kp for kp in res if kp[0] < 103 and kp[1] == 0), key=res.get); top = sorted(res, key=res.get)[:4]
        nopf = min((kp for kp in res if kp[1] == 0), key=res.get)
        line.append(f"{n} cublas {cb:.1f} old k{old[0]}:{res[old]:.1f} ({res[old]/cb:.2f}) best-nopf k{nopf[0]}:{res[nopf]:.1f} ({res[nopf]/cb:.2f}) top "
                    + " ".join(f"k{k}p{p}:{res[(k, p)]:.1f}" for k, p in top) + f" ({res[top[0]]/cb:.2f})")
        tc += cb; to += res[old]; tn += res[top[0]]
        print(f"  ALL M={M} {n}: " + " ".join(f"k{k}:" + "/".join(f"{res[(k, p)]:.1f}" for p in (0, 1, 2)) for k in TILES if (k, 0) in res), flush=True)
    print(f"RUNC M={M:4d} us/launch: " + " | ".join(line), flush=True)
    print(f"RUNC M={M:4d} x28: cuBLAS {tc*LAYERS:.0f} us, old wgmma best {to*LAYERS:.0f} ({to/tc:.3f}x), all best {tn*LAYERS:.0f} ({tn/tc:.3f}x)", flush=True)
