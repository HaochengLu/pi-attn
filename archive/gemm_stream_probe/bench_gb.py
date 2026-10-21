import os, sys, torch
from torch.utils.cpp_extension import load
K = "/uniparser/users/luhaocheng/pi_attn/llm_infer_rust/src/kernels/pi"
H = os.path.dirname(os.path.abspath(__file__))
e = load(name="gbp", sources=[os.path.join(H, "gbal_probe.cu")], extra_include_paths=[K],
         extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_89,code=sm_89", "--expt-relaxed-constexpr"],
         build_directory="/tmp/gbp_build")
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
def cold(fn, iters=15):
    ts = []
    for _ in range(iters + 3):
        flush.zero_()
        s, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); en.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(en) * 1000)
    ts = sorted(ts[3:]); return ts[len(ts) // 2]
for name, (N, Kd) in {"8B down": (4096, 12288), "8B o": (4096, 4096), "8B gateup": (24576, 4096), "1.7B o": (2048, 2048), "1.7B gateup": (12288, 2048), "1.7B lm": (151936, 2048)}.items():
    W = (torch.randn(N, Kd, device="cuda") * 0.02).bfloat16(); X = torch.randn(1, Kd, device="cuda").bfloat16()
    cb = cold(lambda: torch.matmul(X, W.T))
    r = []
    ref = torch.matmul(X.float(), W.float().T)
    for cfg in (0, 1, 2, 3):
        for warps, cps in [(8, 1), (16, 1), (4, 2), (8, 2)]:
            try:
                err = (e.run(X, W, warps, cfg, cps).float() - ref).abs().max().item()
                r.append(f"c{cfg}w{warps}x{cps} {cold(lambda: e.run(X, W, warps, cfg, cps)):.1f}{'' if err < 0.05 else '!'}")
            except Exception as ex:
                r.append(f"c{cfg}w{warps}x{cps} ERR")
    print(f"{name}: cublas {cb:.1f} | " + " ".join(r), flush=True)
