"""H100 run D: split X / W pipelines (ids 129..140: W stages of their own, 512..3072 columns in flight) against the
one-pipeline tiles of run C and cuBLAS, cold 28-layer bursts; bit check of every id against 91 / 99."""
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
    for M in (1, 7, 64, 65, 128, 200, 1000, 2056):
        x = torch.randn(M, K, device="cuda").bfloat16()
        ref = ext.linear90(x, W[n][0], 91 if M > 64 else 99)
        for k in TILES:
            y = ext.linear90(x, W[n][0], k)
            if y is None or not torch.equal(y.view(torch.int16), ref.view(torch.int16)):
                bad += 1; print("BIT MISMATCH", n, M, k, flush=True)
    print("bitcheck", n, "done", flush=True)
print("BITCHECK every id vs 91/99: mismatches", bad, flush=True)
def tiles(k, M, N):
    return -(-M // TILES[k][0]) * -(-N // TILES[k][1])
OLD = (90, 91, 99, 101)
for M in [int(a) for a in sys.argv[1:]] or (8, 64, 128, 192, 256, 512, 1024):
    tc = to = tn = 0.0; line = []
    for n, (N, K) in SHAPES.items():
        x = torch.randn(M, K, device="cuda").bfloat16()
        cb = burst(lambda w: torch.matmul(x, w.T), W[n])
        res = {}
        for k in TILES:
            t = tiles(k, M, N)
            if t > 1500 or (M >= 512 and t < 40) or (M <= 64 and TILES[k][0] > 64): continue
            res[k] = burst(lambda w: ext.linear90(x, w, k), W[n])
        old = min((k for k in res if k in OLD), key=res.get); top = sorted(res, key=res.get)[:4]
        one = min((k for k in res if TILES[k][4] == 0), key=res.get)
        line.append(f"{n} cublas {cb:.1f} old k{old}:{res[old]:.1f} ({res[old]/cb:.2f}) one-pipe k{one}:{res[one]:.1f} ({res[one]/cb:.2f}) top "
                    + " ".join(f"k{k}:{res[k]:.1f}" for k in top) + f" ({res[top[0]]/cb:.2f})")
        tc += cb; to += res[old]; tn += res[top[0]]
        print(f"  ALL M={M} {n}: " + " ".join(f"k{k}:{v:.1f}" for k, v in sorted(res.items())), flush=True)
    print(f"RUND M={M:4d} us/launch: " + " | ".join(line), flush=True)
    print(f"RUND M={M:4d} x28: cuBLAS {tc*LAYERS:.0f} us, old ids {to*LAYERS:.0f} ({to/tc:.3f}x), all best {tn*LAYERS:.0f} ({tn/tc:.3f}x)", flush=True)
