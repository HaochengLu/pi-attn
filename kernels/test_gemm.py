"""M-invariance, accuracy and speed of the invariant GEMM vs torch/cuBLAS.

Invariance: for every projection of the four Qwen3 models (LM heads included), every row
of Y must be bitwise equal whether computed with M = 1, any other M up to 4096, any kernel
id the engine can choose (gemm_dispatch.cuh), and with garbage in the other rows. Each id
must also refuse exactly the shapes `usable` says it cannot run, so the tuner never picks
an untested (id, shape, M)."""
import os
import sys

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
major, minor = torch.cuda.get_device_capability()
ext = load(name=f"pi_gemm_sm{major}{minor}", sources=[os.path.join(HERE, "gemm_torch.cu")],
           extra_include_paths=[HERE, os.path.join(os.path.dirname(__import__('flashinfer').__file__), 'data', 'cutlass', 'include')],
           extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"],
           build_directory=os.environ.get("PI_ATTN_BUILD_DIR"))
dev = "cuda"
torch.manual_seed(0)
SHAPES = {  # name: (N, K), every projection of Qwen3 0.6B/1.7B/4B/8B
    "0.6B qkv": (4096, 1024), "0.6B o": (1024, 2048), "0.6B gateup": (6144, 1024), "0.6B down": (1024, 3072),
    "0.6B lm_head": (151936, 1024), "1.7B qkv": (4096, 2048), "1.7B o": (2048, 2048),
    "1.7B gateup": (12288, 2048), "1.7B down": (2048, 6144), "1.7B lm_head": (151936, 2048),
    "4B qkv": (6144, 2560), "4B o": (2560, 4096), "4B gateup": (19456, 2560), "4B down": (2560, 9728),
    "4B lm_head": (151936, 2560),
    "8B qkv": (6144, 4096), "8B o": (4096, 4096), "8B gateup": (24576, 4096), "8B down": (4096, 12288),
    "8B lm_head": (151936, 4096)}
KERNELS = tuple(ext.small_kernels()) + tuple(ext.large_kernels())  # the engine's candidates
SMALL = set(ext.small_kernels())


def gemm_segments(N, K):  # mirrors gemm_inv.cuh
    return 2 if N <= 1024 and K >= 2048 and K % 512 == 0 else 1


SEGPAR_TILE = {48: (32, 64), 49: (64, 64)}  # segment-parallel ids: (TBM, TBN)


def segpar_fits(k, M, N, K):  # mirrors CutlassGemm::run_segpar: S > 1, fragments fit g_skt_acc, tiles <= 1024
    tm, tn = SEGPAR_TILE[k]
    tiles = -(-M // tm) * -(-N // tn)
    return gemm_segments(N, K) > 1 and tiles <= 1024 and tiles * gemm_segments(N, K) * tm * tn <= 256 * 64 * 128


BK = {20: 128, 21: 256, 22: 256, 24: 128, 25: 128, 30: 256, 31: 128, 32: 128, 33: 128,
      40: 128, 41: 128, 42: 128, 43: 256, 51: 512, 53: 1024, 54: 128}  # K granularity (default 64)
MAX_M = {30: 16, 31: 16, 32: 32, 33: 64, 40: 8, 41: 8, 42: 8, 43: 8, 51: 8, 53: 8, 54: 16, 55: 16}
STREAMK = {40, 41, 42, 43, 44, 45, 46, 47, 50, 51, 53}  # S = 1 shapes only
SEGSPLIT = {54, 55}  # S > 1 shapes only


def usable(k, M, N, K):
    """Whether kernel id k runs this shape (written independently of gemm_dispatch.cuh)."""
    if k in SEGPAR_TILE:
        return segpar_fits(k, M, N, K)
    segmented = gemm_segments(N, K) > 1
    return (K % BK.get(k, 64) == 0 and M <= MAX_M.get(k, 1 << 30)
            and not (k in STREAMK and segmented) and not (k in SEGSPLIT and not segmented))


def candidates(M, N, K):
    """The ids the engine's tuner tries for this shape in an M bucket that holds M."""
    return [k for k in KERNELS if usable(k, M, N, K) and (M <= 64 or k not in SMALL)]


def timeit(fn, iters=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000


def main(bench):
    fails = []
    for name, (N, K) in SHAPES.items():
        W = (torch.randn(N, K, device=dev) * 0.02).bfloat16()
        X = torch.randn(4096, K, device=dev).bfloat16()
        ref = ext.linear(X[:1].contiguous(), W, 1, False)[0]
        for M in [1, 2, 7, 8, 9, 16, 17, 32, 33, 64, 65, 200, 1000, 4096]:
            Xm = X[:M].clone()
            if M > 1:
                Xm[1:] = torch.randn_like(Xm[1:]) * 100  # other rows must not matter
            full = ext.linear(Xm, W, 13, False)  # every row of every kernel must match it too
            for kern in KERNELS:
                y = ext.linear(Xm, W, kern, False)
                if (y is not None) != usable(kern, M, N, K):
                    fails.append((name, M, kern, "refused" if y is None else "ran but should refuse"))
                elif y is not None and (not torch.equal(y[0].view(torch.int16), ref.view(torch.int16))
                                        or not torch.equal(y.view(torch.int16), full.view(torch.int16))):
                    fails.append((name, M, kern))
        segs = gemm_segments(N, K)
        if segs > 1:  # y = round(((+0 + p_0) + p_1) + ...), p_s = the unsegmented chain over segment s
            Xs = X[:64].contiguous()
            ks = K // segs
            tot = torch.zeros(64, N, device=dev)
            pad = max(0, 1056 - N)  # zero rows widen the weight to one segment; columns are independent
            for sg in range(segs):
                Wp = torch.cat([W[:, sg * ks:(sg + 1) * ks], W.new_zeros(pad, ks)]).contiguous()
                assert gemm_segments(Wp.shape[0], ks) == 1
                tot = tot + ext.linear(Xs[:, sg * ks:(sg + 1) * ks], Wp, 1, True)[:, :N]
            for kern in (1, 20, 33, 11, 13, 14, 15, 48, 49):
                if not torch.equal(ext.linear(Xs, W, kern, False).view(torch.int16), tot.bfloat16().view(torch.int16)):
                    fails.append((name, "segments", kern))
        y64 = (X[:256].double() @ W.double().T)
        e_pi = (ext.linear(X[:256].contiguous(), W, 0, True).double() - y64).abs().max().item()
        e_cb = (torch.matmul(X[:256], W.T).double() - y64).abs().max().item()
        e_pi_b = (ext.linear(X[:256].contiguous(), W, 0, False).double() - y64).abs().max().item()
        line = f"{name:13s} N={N:6d} K={K:5d}: max|err| pi(fp32 out) {e_pi:.2e} pi(bf16) {e_pi_b:.2e} cublas(bf16) {e_cb:.2e}"
        if bench:
            ts = []
            for M in [1, 8, 32, 64, 128, 256, 1024, 4096]:
                Xm = X[:M].contiguous()
                tc = timeit(lambda: torch.matmul(Xm, W.T))
                best = min((timeit(lambda: ext.linear(Xm, W, kk, False)), kk) for kk in candidates(M, N, K))
                ts.append(f"M={M}:{best[0]/tc:.2f}(k{best[1]})")
            line += " | pi/cublas " + " ".join(ts)
        print(line, flush=True)
    print("INVARIANCE FAILURES:", len(fails), fails[:8])
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main("--bench" in sys.argv))
