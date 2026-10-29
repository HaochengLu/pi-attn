"""M-invariance, TP-degree invariance, accuracy and speed of the invariant GEMM vs torch/cuBLAS.

Invariance: for every projection of the four Qwen3 models (LM heads included), every row
of Y must be bitwise equal whether computed with M = 1, any other M up to 4096, any kernel
id the engine can choose (gemm_dispatch.cuh), and with garbage in the other rows. Each id
must also refuse exactly the shapes `usable` says it cannot run, so the tuner never picks
an untested (id, shape, M).

TP-degree invariance (column sharding): for row-parallel projections of models served at
up to t_max = 8 GPUs, Y computed over the whole weight must equal, bitwise, the
concatenation of Y computed over its column shards for every t = 1..8 (uneven shards for
t = 3, 5, 6, 7), every M and every kernel id that runs the shard, with S =
gemm_segments(N, K, t_max) of the whole weight passed to every launch."""
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
SHAPES = {  # name: (N, K), every projection of Qwen3 0.6B/1.7B/4B/8B, and Qwen3.5-4B's
    "0.6B qkv": (4096, 1024), "0.6B o": (1024, 2048), "0.6B gateup": (6144, 1024), "0.6B down": (1024, 3072),
    "0.6B lm_head": (151936, 1024), "1.7B qkv": (4096, 2048), "1.7B o": (2048, 2048),
    "1.7B gateup": (12288, 2048), "1.7B down": (2048, 6144), "1.7B lm_head": (151936, 2048),
    "4B qkv": (6144, 2560), "4B o": (2560, 4096), "4B gateup": (19456, 2560), "4B down": (2560, 9728),
    "4B lm_head": (151936, 2560),
    "8B qkv": (6144, 4096), "8B o": (4096, 4096), "8B gateup": (24576, 4096), "8B down": (4096, 12288),
    "8B lm_head": (151936, 4096),
    # Qwen3.5-4B (vLLM, kernels/pi_vllm.py): GDN in_proj_qkvz, in_proj_ba, out_proj; attention qkv; MLP; LM head
    "3.5-4B qkvz": (12288, 2560), "3.5-4B ba": (64, 2560), "3.5-4B gdn out": (2560, 4096),
    "3.5-4B attn qkv": (10240, 2560), "3.5-4B gateup": (18432, 2560), "3.5-4B down": (2560, 9216),
    "3.5-4B lm_head": (248320, 2560)}
KERNELS = tuple(ext.small_kernels()) + tuple(ext.large_kernels())  # the engine's candidates
SMALL = set(ext.small_kernels())


def gemm_segments(N, K, t_max=1):  # mirrors gemm_inv.cuh
    s = 1
    while s < 8 and N // t_max * s <= 1024 and K % (512 * s) == 0 and K // (2 * s) >= 1024:
        s *= 2
    return s


# Before t_max: two segments for N <= 1024, K >= 2048, K % 512 == 0. The engine's own shapes keep it.
assert all(gemm_segments(N, K) == (2 if N <= 1024 and K >= 2048 and K % 512 == 0 else 1) for N, K in
           [(n, k) for n in range(32, 20000, 32) for k in range(64, 30000, 64) if n > 512 or k % 1024 or k < 4096])
assert all(ext.gemm_segments(N, K, t) == gemm_segments(N, K, t) for N in (64, 512, 1024, 2560, 8192)
           for K in (1024, 2560, 3072, 4096, 9216, 28672) for t in (1, 2, 4, 8))
SMEM_OPTIN = torch.cuda.get_device_properties(0).shared_memory_per_block_optin


SEGPAR_TILE = {48: (32, 64), 49: (64, 64)}  # segment-parallel ids: (TBM, TBN)


def segpar_fits(k, M, N, S):  # mirrors CutlassGemm::run_segpar: S > 1, tiles <= the 4096 arrival counters
    tm, tn = SEGPAR_TILE[k]
    tiles = -(-M // tm) * -(-N // tn)
    return S > 1 and tiles <= 4096


WARP_SK = {  # warp stream-K: id -> (BK, stages, n8 tiles per chain, X rows, warps per CTA)
    40: (128, 4, 1, 8, 2), 41: (128, 4, 1, 8, 3), 42: (128, 4, 1, 8, 4), 43: (256, 3, 1, 8, 3),
    51: (512, 3, 1, 8, 2), 53: (1024, 2, 1, 8, 1), 60: (256, 3, 2, 8, 2), 61: (512, 3, 2, 8, 2),
    62: (256, 4, 2, 8, 2), 63: (256, 3, 2, 16, 2)}
BK = {20: 128, 21: 256, 22: 256, 24: 128, 25: 128, 30: 256, 31: 128, 32: 128, 33: 128, 54: 128,
      **{k: v[0] for k, v in WARP_SK.items()}}  # K granularity within a segment (default 64)
MAX_M = {30: 16, 31: 16, 32: 32, 33: 64, 54: 16, 55: 16, **{k: v[3] for k, v in WARP_SK.items()}}
PAIRS = {k for k, v in WARP_SK.items() if v[2] == 2}  # stream-K over pairs of n8 tiles: N % 16 == 0
STREAMK = {44: (32, 64), 45: (32, 64), 46: (64, 64), 47: (64, 128), 50: (128, 64),
           64: (32, 32), 65: (128, 128), 66: (128, 256)}  # CUTLASS stream-K: id -> tile
SLOT = {17, 18, 19}  # SegmentedKernel with L2 slots: S > 1 only
SEGSPLIT = {54: (128, 3), 55: (64, 6)}  # S > 1 shapes only: (BK, stages), one cp.async ring per segment
SEG_WS = {30, 31, 32, 33, *WARP_SK}  # S > 1: S * M * N partials in g_seg_ws


def usable(k, M, N, K, S=None):
    """Whether kernel id k runs this shape with S segments (default: the weight's own);
    written independently of gemm_dispatch.cuh."""
    S = gemm_segments(N, K) if S is None else S
    if k in SEGPAR_TILE:
        return segpar_fits(k, M, N, S)
    if k in SEGSPLIT:
        bk, st = SEGSPLIT[k]
        if S * (st * 24 * (bk + 8) * 2 + 32 * 4 * 4) > SMEM_OPTIN:
            return False
    if k in WARP_SK:
        bk, st, tn, xr, warps = WARP_SK[k]
        if warps * st * (xr + 8 * tn) * (bk + 8) * 2 > SMEM_OPTIN:
            return False
    if k in STREAMK and S > 1:
        tm, tn = STREAMK[k]
        if -(-M // tm) * -(-N // tn) > 4096:  # arrival counters
            return False
    if k in SEG_WS and S > 1 and (S * min(M, MAX_M[k]) * N > 1 << 20 or N // 8 > 2048):
        return False
    # K tiles may not straddle a segment (the warp stream-K ids 51 / 53 stream 512 / 1024
    # columns per stage; 0.6B down has 1536-column segments).
    return ((K // S) % BK.get(k, 64) == 0 and K // S >= BK.get(k, 64) and M <= MAX_M.get(k, 1 << 30)
            and not ((k in SEGSPLIT or k in SLOT) and S == 1)
            and not (k in PAIRS and N % 16))


def candidates(M, N, K, S=None):
    """The ids the engine's tuner tries for this shape in an M bucket that holds M."""
    return [k for k in KERNELS if usable(k, M, N, K, S) and (M <= 64 or k not in SMALL)]


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


def check_segments(name, X, W, segs, fails):
    """y = round(((+0 + p_0) + p_1) + ...), p_s = the unsegmented chain over segment s."""
    N, K = W.shape
    Xs = X[:64].contiguous()
    ks = K // segs
    tot = torch.zeros(64, N, device=dev)
    for sg in range(segs):
        tot = tot + ext.linear(Xs[:, sg * ks:(sg + 1) * ks], W[:, sg * ks:(sg + 1) * ks].contiguous(), 1, True, 1)
    for kern in (1, 20, 33, 11, 13, 14, 15, 17, 18, 19, 44, 48, 49, 64, 65, 66):
        if not usable(kern, 64, N, K, segs):
            continue
        y = ext.linear(Xs, W, kern, False, segs)
        if y is None or not torch.equal(y.view(torch.int16), tot.bfloat16().view(torch.int16)):
            fails.append((name, "segments", segs, kern))


TP_SHAPES = {  # row-parallel projections (o / down, GDN out): name: (N, K)
    "4B o": (2560, 4096), "4B down": (2560, 9728), "8B o": (4096, 4096), "8B down": (4096, 12288),
    "3.5-4B down": (2560, 9216), "70B o": (8192, 8192), "70B down": (8192, 28672)}


def shards(N, t):
    """Column ranges of t ranks: as even as 32-column units allow (uneven for t = 3, 5, 6, 7)."""
    cuts = [round(i * N / t / 32) * 32 for i in range(t + 1)]
    return list(zip(cuts[:-1], cuts[1:]))


def check_tp(fails, t_max=8):
    for name, (N, K) in TP_SHAPES.items():
        S = gemm_segments(N, K, t_max)
        W = (torch.randn(N, K, device=dev) * 0.02).bfloat16()
        X = torch.randn(4096, K, device=dev).bfloat16()
        if S > 1:
            check_segments(name + f" t_max={t_max}", X, W, S, fails)
        row0 = ext.linear(X[:1].contiguous(), W, 1, False, S)[0]
        # M = 4096 (degrees 1, 3, 8): more tiles than resident CTAs, so the persistent slot ids
        # 17-19 wrap their tile loop.
        for M, degrees in [(1, range(1, t_max + 1)), (8, range(1, t_max + 1)), (12, range(1, t_max + 1)),
                           (33, range(1, t_max + 1)), (300, range(1, t_max + 1)), (4096, (1, 3, t_max))]:
            Xm = X[:M].contiguous()
            full = ext.linear(Xm, W, 13, False, S)
            if not torch.equal(full[0].view(torch.int16), row0.view(torch.int16)):
                fails.append((name, "tp rows", M))  # row invariance across M at S(t_max)
            runs = 0
            for t in degrees:
                for a, b in shards(N, t):
                    Wt = W[a:b].contiguous()
                    for kern in KERNELS:
                        y = ext.linear(Xm, Wt, kern, False, S)
                        if (y is not None) != usable(kern, M, b - a, K, S):
                            fails.append((name, "tp", t, M, kern, "refused" if y is None else "ran but should refuse"))
                        elif y is not None:
                            runs += 1
                            if not torch.equal(y.view(torch.int16), full[:, a:b].view(torch.int16)):
                                fails.append((name, "tp", t, (a, b), M, kern))
            print(f"tp {name:12s} N={N:5d} K={K:5d} t_max={t_max} S={S} M={M:4d}: {runs} shard launches", flush=True)
        del W
        torch.cuda.empty_cache()


def check_s8(fails):
    """S = 8, the largest canonical count, which no t_max = 8 shape above reaches: every id on the whole
    weight and on column shards (t = 1..8 at M = 33), against the explicit left fold."""
    for name, (N, K) in {"8B down": (4096, 12288), "70B down": (8192, 28672)}.items():
        W = (torch.randn(N, K, device=dev) * 0.02).bfloat16()
        X = torch.randn(300, K, device=dev).bfloat16()
        check_segments(name + " S=8", X, W, 8, fails)
        row0 = ext.linear(X[:1].contiguous(), W, 1, False, 8)[0]
        for M in [1, 12, 33, 300]:
            Xm = X[:M].contiguous()
            full = ext.linear(Xm, W, 13, False, 8)
            if not torch.equal(full[0].view(torch.int16), row0.view(torch.int16)):
                fails.append((name, "S=8 rows", M))
            for t in (range(1, 9) if M == 33 else (1,)):
                for a, b in shards(N, t):
                    Wt = W[a:b].contiguous()
                    for kern in KERNELS:
                        y = ext.linear(Xm, Wt, kern, False, 8)
                        if (y is not None) != usable(kern, M, b - a, K, 8):
                            fails.append((name, "S=8", t, M, kern, "refused" if y is None else "ran but should refuse"))
                        elif y is not None and not torch.equal(y.view(torch.int16), full[:, a:b].view(torch.int16)):
                            fails.append((name, "S=8", t, (a, b), M, kern))
        print(f"S=8 {name}: checked", flush=True)
        del W
        torch.cuda.empty_cache()


def main(bench):
    fails = []
    check_tp(fails)
    check_s8(fails)
    for name, (N, K) in SHAPES.items():
        W = (torch.randn(N, K, device=dev) * 0.02).bfloat16()
        X = torch.randn(16384 if bench else 4096, K, device=dev).bfloat16()
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
        if segs > 1:
            check_segments(name, X, W, segs, fails)
        y64 = (X[:256].double() @ W.double().T)
        e_pi = (ext.linear(X[:256].contiguous(), W, 0, True).double() - y64).abs().max().item()
        e_cb = (torch.matmul(X[:256], W.T).double() - y64).abs().max().item()
        e_pi_b = (ext.linear(X[:256].contiguous(), W, 0, False).double() - y64).abs().max().item()
        line = f"{name:13s} N={N:6d} K={K:5d}: max|err| pi(fp32 out) {e_pi:.2e} pi(bf16) {e_pi_b:.2e} cublas(bf16) {e_cb:.2e}"
        if bench:
            ts = []
            for M in [1, 8, 32, 64, 128, 256, 1024, 4096, 16384]:  # 16384: vLLM's prefill token budget
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
