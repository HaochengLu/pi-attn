import os, sys, torch
from torch.utils.cpp_extension import load
HERE = os.path.dirname(os.path.abspath(__file__))
CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
ARCH = sys.argv[1] if len(sys.argv) > 1 else "90a"
ext = load(name="rev_probe", sources=[os.path.join(HERE, "rev_probe.cu")],
           extra_cuda_cflags=["-O3", "-std=c++17", f"-gencode=arch=compute_{ARCH},code=sm_{ARCH}"],
           extra_ldflags=[f"-L{os.path.join(CUDA_HOME, 'lib64', 'stubs')}", "-lcuda"])
print(torch.cuda.get_device_name(0), flush=True)

# (2) mma.sync operand symmetry
torch.manual_seed(0)
bad = tot = 0
for trial, (sx, sc) in enumerate(((1, 1), (3, 100), (1, 1e-3), (30, 1e4), (1, 0))):
    n = 20000
    ex = torch.randint(-6, 7, (n, 16, 16), device="cuda").float() if trial >= 2 else torch.zeros(n, 16, 16, device="cuda")
    x = (torch.randn(n, 16, 16, device="cuda") * sx * torch.exp2(ex)).bfloat16().contiguous()
    y = (torch.randn(n, 8, 16, device="cuda") * sx * torch.exp2(ex[:, :8])).bfloat16().contiguous()
    c = (torch.randn(n, 16, 8, device="cuda") * sc).contiguous()
    o0 = torch.zeros(n, 16, 8, device="cuda"); o1 = torch.zeros(n, 8, 8, device="cuda")
    ext.sym(x, y, c, o0, o1); torch.cuda.synchronize()
    ref = o0[:, :8, :].transpose(1, 2).contiguous()
    ne = (ref.view(torch.int32) != o1.view(torch.int32)).sum().item()
    # sanity: the kernel computes what it should (float64 reference, loose)
    r64 = c.double() + x.double() @ y.double().transpose(1, 2)
    err = ((o0.double() - r64).abs() / (r64.abs() + 1e-3)).max().item()
    bad += ne; tot += ref.numel()
    print(f"SYM trial {trial}: {ne} of {ref.numel()} elements differ (A/B swapped); max rel err vs f64 {err:.2e}", flush=True)
print(f"SYM TOTAL: {bad} of {tot} differ", flush=True)
if ARCH != "90a":
    sys.exit(0)

sink = torch.zeros(16, device="cuda")
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

# correctness of the TMA page boxes: same smem image as cp.async, shuffled pages
src = torch.randint(0, 256, (4 * 32 * 32 * 2048,), dtype=torch.uint8, device="cuda")
for depth in (1, 4, 16):
    s0 = torch.zeros(4, dtype=torch.int64, device="cuda"); s5 = torch.zeros(4, dtype=torch.int64, device="cuda")
    ext.check(src, s0, 4, 0, 32, depth, 1); ext.check(src, s5, 4, 5, 32, depth, 1); torch.cuda.synchronize()
    print(f"CHECK depth {depth}: TMA page boxes == cp.async: {torch.equal(s0, s5)} {s0.tolist()[:2]}", flush=True)
del src

a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
for _ in range(300):
    a @ a
torch.cuda.synchronize()
names = {0: "cp.async 16 B (shipped mapping)", 4: "bulk 8 KB contiguous", 5: "TMA box page 16 keys", 6: "TMA box 32 keys"}
for ctas, layers, TILES in ((1, 28, 32), (1, 28, 16), (16, 28, 32), (16, 28, 16), (16, 1, 32), (64, 28, 32), (128, 28, 32)):
    srcs = [torch.randint(0, 100, (ctas * TILES * 32 * 2048,), dtype=torch.uint8, device="cuda") for _ in range(layers)]
    tag = "L2" if layers == 1 else "DRAM"
    base = timeit(lambda i: ext.copy(srcs[i], sink, ctas, 0, 4, 0, 1, 0), layers)
    print(f"--- {ctas} CTAs, {tag}, {TILES} tile loads ({TILES * 8} KB per CTA): empty launch {base:.2f} us; below launch subtracted", flush=True)
    for mode, perm in ((0, 0), (0, 1), (4, 0), (5, 0), (5, 1), (6, 0)):
        row = []
        for warps in ((4, 16) if mode == 0 else (4,)):
            for depth in (1, 2, 4, 8, 16):
                t = timeit(lambda i: ext.copy(srcs[i], sink, ctas, mode, warps, TILES, depth, perm), layers) - base
                row.append(f"w{warps}d{depth} {t:5.1f}")
        print(f"COPY {ctas:3d} {tag:4s} t{TILES} {names[mode]:32s} perm{perm}: " + " | ".join(row), flush=True)
    del srcs
