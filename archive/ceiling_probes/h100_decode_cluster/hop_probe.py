import os, torch
from torch.utils.cpp_extension import load
HERE = os.path.dirname(os.path.abspath(__file__))
ext = load(name="hop_probe", sources=[os.path.join(HERE, "hop_probe.cu")],
           extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a"])
sink = torch.zeros(16, device="cuda")


def timeit(fn, calls=20, iters=20):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls):
            fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / calls)
    ts.sort()
    return ts[len(ts) // 2] * 1000


a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
for _ in range(300):
    a @ a
torch.cuda.synchronize()
names = {0: "flag poll", 1: "mbarrier try_wait", 2: "mbarrier test_wait spin"}
for mode in (0, 1, 2):
    for n in (2, 8):
        for waiters in (1, 32, 128):
            base = timeit(lambda: ext.ring(sink, 1, n, 0, mode, waiters))
            t = timeit(lambda: ext.ring(sink, 1, n, 64, mode, waiters))
            t16 = timeit(lambda: ext.ring(sink, 16, n, 64, mode, waiters))
            print(f"RING {names[mode]:24s} n={n} waiters={waiters:3d}: 0 rounds {base:6.2f} us, 64 rounds {t:7.2f} us"
                  f" -> {(t - base) / (64 * n) * 1000:6.1f} ns per hand-off (16 clusters: {(t16 - base) / (64 * n) * 1000:6.1f})", flush=True)
for n in (2, 8):
    for kb in (0, 4, 16, 24):
        if n * kb > 200:
            continue
        t = timeit(lambda: ext.push(sink, 1, n, kb * 1024))
        print(f"PUSH n={n} {kb:2d} KB from each of {n - 1} CTAs into CTA 0: {t:6.2f} us", flush=True)
