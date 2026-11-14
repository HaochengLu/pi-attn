import os, torch
from torch.utils.cpp_extension import load
HERE = os.path.dirname(os.path.abspath(__file__))
ext = load(name="copy_probe", sources=[os.path.join(HERE, "copy_probe.cu")],
           extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a"])
sink = torch.zeros(16, device="cuda")
TILES = 32  # one fold group's K and V


def timeit(fn, layers, calls=28, iters=20):
    fn(0); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(calls):
            fn(i % layers)
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
names = ["cp.async 8 half lines", "cp.async 4 whole lines", "ld + st.shared", "bulk 256 B per key", "bulk 8 KB per tile"]
for ctas, layers in ((1, 1), (1, 28), (16, 1), (16, 28), (128, 28)):
    srcs = [torch.randint(0, 100, (ctas * TILES * 32 * 2048,), dtype=torch.uint8, device="cuda") for _ in range(layers)]
    tag = "L2" if layers == 1 else "DRAM"
    base = timeit(lambda i: ext.copy(srcs[i], sink, ctas, 0, 4, 0, 1), layers)
    print(f"--- {ctas} CTAs, {tag}: empty launch {base:.2f} us; below: us for 32 tile loads (256 KB per CTA), launch subtracted", flush=True)
    for mode in range(5):
        row = []
        for warps in ((4, 8, 16) if mode < 3 else (4,)):
            for depth in (1, 2, 4, 8, 16):
                try:
                    t = timeit(lambda i: ext.copy(srcs[i], sink, ctas, mode, warps, TILES, depth), layers) - base
                    row.append(f"w{warps}d{depth} {t:5.1f}")
                except Exception as e:
                    row.append(f"w{warps}d{depth} ERR")
        print(f"COPY {ctas:3d} {tag:4s} {names[mode]:24s}: " + " | ".join(row), flush=True)
    del srcs
