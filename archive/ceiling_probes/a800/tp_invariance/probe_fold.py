"""K segments with the running total folded through an L2 slot instead of a second register tile.
For each S, builds three variants of SegmentedKernel from probe_segs.py's S build: FOLD=0 (as the engine: the
total is a second accumulator fragment), FOLD=1 (fp32 total in a per-tile global slot, folded element by element
at each segment end; the same __fadd_rn sequence, so it must give FOLD=0's bits), FOLD=2 (int64 fixed-point total,
q = rne(p * 2^28) per segment, summed exactly; timing only: its final conversion double-rounds).
Times the best CUTLASS segmented id at large M against the S=1 build's best id.
usage (from logs/tpprobe, after probe_segs.py): python probe_fold.py"""
import os, shutil, sys
import torch
from torch.utils.cpp_extension import load
import flashinfer

HERE = os.path.dirname(os.path.abspath(__file__))
major, minor = torch.cuda.get_device_capability()
CUT = os.path.join(os.path.dirname(flashinfer.__file__), 'data', 'cutlass', 'include')

OLD_HEAD = """    typename Mma::FragmentC total;
    total.clear();
    for (int s = 0; s < segs; ++s) {"""
NEW_HEAD = """#if PI_FOLD == 0
    typename Mma::FragmentC total;
    total.clear();
#else
    constexpr int kThreads = Base::kThreadCount, kTile = Mma::Shape::kM * Mma::Shape::kN;
    const int64_t tile_id = tile.m() + (int64_t)tile.n() * params.grid_tiled_shape.m();
#if PI_FOLD == 1
    float* slot = (float*)g_fold_ws + tile_id * kTile;
#else
    long long* slot = (long long*)g_fold_ws + tile_id * kTile;
#endif
    typename Mma::FragmentC acc;
#endif
    for (int s = 0; s < segs; ++s) {"""
OLD_ACC = """      Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
      typename Mma::FragmentC acc;
      acc.clear();
      mma(seg_k / Mma::Shape::kK, acc, iterator_A, iterator_B, acc);
      cutlass::arch::cp_async_wait<0>();
      __syncthreads();  // the next segment's prologue (or the epilogue) reuses shared memory
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Mma::FragmentC::kElements; ++i) total[i] = __fadd_rn(total[i], acc[i]);
    }"""
NEW_ACC = """      Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
#if PI_FOLD == 0
      typename Mma::FragmentC acc;
#endif
      acc.clear();
      mma(seg_k / Mma::Shape::kK, acc, iterator_A, iterator_B, acc);
      cutlass::arch::cp_async_wait<0>();
      __syncthreads();  // the next segment's prologue (or the epilogue) reuses shared memory
#if PI_FOLD == 0
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Mma::FragmentC::kElements; ++i) total[i] = __fadd_rn(total[i], acc[i]);
#else
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Mma::FragmentC::kElements; ++i) {
        const int64_t j = (int64_t)i * kThreads + thread_idx;
#if PI_FOLD == 1
        const float v = s == 0 ? __fadd_rn(0.f, acc[i]) : __fadd_rn(__ldcg(slot + j), acc[i]);
        if (s < segs - 1) __stcg(slot + j, v); else acc[i] = v;
#else
        const long long q = __float2ll_rn(acc[i] * 268435456.f), v = s == 0 ? q : __ldcg(slot + j) + q;
        if (s < segs - 1) __stcg(slot + j, v); else acc[i] = __ll2float_rn(v) * (1.f / 268435456.f);
#endif
      }
#endif
    }
#if PI_FOLD != 0
    const typename Mma::FragmentC& total = acc;
#endif"""
OLD_LAUNCH = """    cutlass::Kernel<Segmented><<<swizzle.get_grid_shape(grid_shape), Kernel::kThreadCount, smem, stream>>>(params);"""
NEW_LAUNCH = """#if PI_FOLD != 0
    if (!fold_ws((size_t)grid_shape.m() * grid_shape.n() * TBM * TBN * 8)) return cudaErrorMemoryAllocation;
#endif
    cutlass::Kernel<Segmented><<<swizzle.get_grid_shape(grid_shape), Kernel::kThreadCount, smem, stream>>>(params);"""


FOLD_WS = """// One workspace for every instantiation (g_fold_ws is one symbol per translation unit).
static bool fold_ws(size_t need) {
  static void* ws = nullptr;
  static size_t cap = 0;
  if (need <= cap) return true;
  cudaDeviceSynchronize();
  cudaFree(ws);
  if (cudaMalloc(&ws, need) != cudaSuccess) return false;
  cap = need;
  return cudaMemcpyToSymbol(g_fold_ws, &ws, sizeof(ws)) == cudaSuccess;
}
"""


def build(S, fold):
    src, dst = os.path.join(HERE, f'S{S}'), os.path.join(HERE, f'S{S}_f{fold}')
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        if f.endswith('.cuh') or f.endswith('.cu'):
            shutil.copy(os.path.join(src, f), dst)
    p = os.path.join(dst, 'gemm_cutlass.cuh')
    s = open(p).read()
    for a, b in [(OLD_HEAD, NEW_HEAD), (OLD_ACC, NEW_ACC), (OLD_LAUNCH, NEW_LAUNCH),
                 ('namespace pi_gemm {\n', f'#define PI_FOLD {fold}\nnamespace pi_gemm {{\nstatic __device__ void* g_fold_ws;\n' + FOLD_WS)]:
        assert s.count(a) == 1, a[:40]
        s = s.replace(a, b)
    open(p, 'w').write(s)
    return load(name=f'pi_gemm_probe_S{S}_f{fold}', sources=[os.path.join(dst, 'gemm_torch.cu')],
                extra_include_paths=[dst, CUT], build_directory=dst,
                extra_cuda_cflags=['-O3', '-std=c++17', '--expt-relaxed-constexpr', f'-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}'])


def timeit(fn, iters=10):
    fn(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1000


def best(ext, x, W, ids):
    r = None
    for k in ids:
        print('RUN', getattr(ext, '__name__', ext), x.shape[0], W.shape, k, file=sys.stderr, flush=True)
        if ext.linear(x, W, k, False) is None:
            continue
        t = timeit(lambda: ext.linear(x, W, k, False))
        r = (t, k) if r is None or t < r[0] else r
    return r


base = load(name='pi_gemm_probe_S1', sources=[os.path.join(HERE, 'S1', 'gemm_torch.cu')], build_directory=os.path.join(HERE, 'S1'),
            extra_include_paths=[os.path.join(HERE, 'S1'), CUT],
            extra_cuda_cflags=['-O3', '-std=c++17', '--expt-relaxed-constexpr', f'-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}'])
CUTLASS_IDS = [10, 11, 12, 13, 14, 15, 16]
exts = {(S, f): build(S, f) for S in [2, 4, 8] for f in [0, 1, 2]}
SHAPES = [('qwen3.5-4B o/gdn-out', 2560, 4096), ('qwen3.5-4B down', 2560, 9216), ('qwen3-8B o', 4096, 4096),
          ('qwen3-8B down', 4096, 12288), ('70B o', 8192, 8192), ('70B down', 8192, 28672)]
torch.manual_seed(0)
for name, N, K in SHAPES:
    W = torch.randn(N, K, device='cuda').bfloat16() * 0.02
    for M in [2048, 8192]:
        X = torch.randn(M, K, device='cuda').bfloat16()
        b = best(base, X, W, list(base.large_kernels()))
        line = f'{name:22s} M={M:5d}: S=1 {b[0]:.0f}us(k{b[1]})'
        for S in [2, 4, 8]:
            if K % (S * 256):
                continue
            ref = exts[(S, 0)].linear(X, W, 10, False)
            for f in [0, 1, 2]:
                e = exts[(S, f)]
                r = best(e, X, W, CUTLASS_IDS)
                outs = [e.linear(X, W, k, False) for k in CUTLASS_IDS]
                outs = [o for o in outs if o is not None]
                same_ids = all(torch.equal(o.view(torch.int16), outs[0].view(torch.int16)) for o in outs)
                same_reg = torch.equal(outs[0].view(torch.int16), ref.view(torch.int16))
                tag = ['reg', 'slot32', 'slot64'][f]
                line += f' | S={S} {tag} {r[0] / b[0]:.2f}x(k{r[1]}){"" if same_ids else " IDS-DIFFER"}{"" if f == 2 or same_reg else " NOT-REG-BITS"}'
        print(line, flush=True)
    del W
    torch.cuda.empty_cache()
