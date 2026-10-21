import os, sys, torch
from torch.utils.cpp_extension import load
os.makedirs("/tmp/bwp_build", exist_ok=True)
H = os.path.dirname(os.path.abspath(__file__))
e = load(name="bwp", sources=[os.path.join(H, "bw_probe.cu")],
         extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_89,code=sm_89"],
         build_directory="/tmp/bwp_build")
BURST = 32

def burst(fn, copies, reps=7):
    ts = []
    for _ in range(reps):
        s, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for i in range(BURST): fn(copies[i % len(copies)])
        en.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(en) * 1000 / BURST)
    ts.sort(); return ts[len(ts) // 2]

shapes = {"8B down": (4096, 12288), "8B o": (4096, 4096), "8B qkv": (6144, 4096), "8B gateup": (24576, 4096),
          "1.7B down": (2048, 6144), "1.7B gateup": (12288, 2048)}
if __name__ != "__main__": shapes = {}
for name, (N, K) in shapes.items():
    nb = N * K * 2
    copies = [(torch.randn(N, K, device="cuda") * 0.02).bfloat16() for _ in range(max(4, (600 << 20) // nb))]
    X = torch.randn(1, K, device="cuda").bfloat16()
    gb = lambda us: nb / us / 1e3
    cb = burst(lambda W: torch.matmul(X, W.T), copies)
    res = {"cublas": cb}
    for bps, th, un in [(4, 256, 4), (8, 256, 4), (4, 256, 8), (16, 128, 4)]:
        res[f"lin b{bps} t{th} u{un}"] = burst(lambda W: e.run(W, 0, bps, th, un), copies)
    for bk, s in [(64, 6), (128, 4), (128, 6), (256, 3), (256, 4), (512, 2), (512, 3)]:
        for rows in (32, 40, 48):
            if s * rows * (bk + 8) * 2 > 99 * 1024: continue
            res[f"tile bk{bk} s{s} r{rows}"] = burst(lambda W: e.run(W, 1, bk, s, rows), copies)
    for bk, s in [(64, 8), (128, 4), (128, 6), (256, 3), (256, 4), (512, 2), (512, 3)]:
        for wp in (4, 5, 6, 8):
            if wp * s * 8 * (bk + 8) * 2 > 99 * 1024: continue
            res[f"warp bk{bk} s{s} w{wp}"] = burst(lambda W: e.run(W, 2, bk, s, wp), copies)
    for d in (1, 2, 4):
        for wp, cps in [(4, 1), (8, 1), (4, 2), (8, 2), (16, 1)]:
            res[f"reg d{d} w{wp} c{cps}"] = burst(lambda W: e.run(W, 3, d, cps, wp), copies)
    best = sorted(res.items(), key=lambda kv: kv[1])
    print(f"== {name} {nb/2**20:.0f}MB cublas {cb:.1f}us ({gb(cb):.0f} GB/s)")
    print("   " + " | ".join(f"{k} {v:.1f} ({gb(v):.0f})" for k, v in best[:10]))
    for pref in ("lin", "tile", "warp", "reg"):
        k, v = min(((k, v) for k, v in res.items() if k.startswith(pref)), key=lambda kv: kv[1])
        print(f"   best {pref}: {k} {v:.1f}us {gb(v):.0f} GB/s ratio {v/cb:.2f}")
    sys.stdout.flush()
    del copies; torch.cuda.empty_cache()
