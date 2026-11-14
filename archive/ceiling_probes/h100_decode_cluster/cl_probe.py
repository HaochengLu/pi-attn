import os, torch
from torch.utils.cpp_extension import load
HERE = os.path.dirname(os.path.abspath(__file__))
ext = load(name="cl_probe", sources=[os.path.join(HERE, "cl_probe.cu")],
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


def run(tag, clusters, n, spin=0, syncs=0, smem=1024, ydim=1):
    try:
        t = timeit(lambda: ext.probe(sink, clusters, n, spin, syncs, smem, ydim))
        print(f"{tag:28s} clusters={clusters:5d} n={n:2d} spin={spin:6d} syncs={syncs} smem={smem:6d} y={ydim}: {t:7.2f} us", flush=True)
    except Exception as e:
        print(f"{tag} clusters={clusters} n={n}: ERR {str(e)[:80]}", flush=True)


for spin in (0, 1000, 4000):
    run("spin calib 1 CTA", 1, 0, spin)
for x in (8, 16, 40, 128, 320, 1280):
    for n in (0, 1, 2, 4, 8, 16):
        run("empty", x, n)
for x in (16, 40, 320):
    for n in (0, 2, 8):
        run("spin", x, n, 4000)
        run("spin big smem", x, n, 4000, 0, 60000)
for x in (16, 40):
    for n in (2, 8):
        for syncs in (1, 2, 8):
            run("syncs", x, n, 0, syncs)
for x in (16, 40, 320):
    for n in (0, 8):
        run("empty ydim8", x, n, 0, 0, 1024, 8)
for n in (0, 8):   # same CTA count, regular vs cluster
    run("empty same CTAs", 128 if n else 1024, n)
