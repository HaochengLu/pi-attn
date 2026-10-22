// Why the first SegParallelKernel was slow: the same kernel (32x64x64, 5 stages) with
// stages switched off, and with its arrival flag either in a static __shared__ variable
// (the first version) or broadcast with __syncthreads_or (the engine's version).
// mode 0: full kernel; 1: mainloop + epilogue of the CTA's own accumulator only (wrong
// values on S > 1 shapes; on S = 1 shapes this is exactly the plain CUTLASS tile's work).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "gemm_cutlass.cuh"
using namespace pi_gemm;

template <typename Base, bool kStaticFlag>
struct ProbeKernel : Base {
  using typename Base::Epilogue;
  using typename Base::Mma;
  using typename Base::OutputOp;
  using typename Base::Params;
  using typename Base::SharedStorage;
  static constexpr int kTile = Mma::Shape::kM * Mma::Shape::kN;
  CUTLASS_DEVICE void run(Params const& params, SharedStorage& shared_storage, int tiles, int mode) {
    const int M = params.problem_size.m(), N = params.problem_size.n(), K = params.problem_size.k();
    const int tiles_m = (M + Mma::Shape::kM - 1) / Mma::Shape::kM;
    const int segs = gemm_segments(N, K), seg_k = K / segs;
    const int u = blockIdx.x % tiles, s = blockIdx.x / tiles;
    const int m0 = (u % tiles_m) * Mma::Shape::kM, n0 = (u / tiles_m) * Mma::Shape::kN;
    const int k0 = s * seg_k, k1 = k0 + seg_k;
    const int thread_idx = threadIdx.x, warp_idx = cutlass::canonical_warp_idx_sync(), lane_idx = threadIdx.x % 32;
    constexpr int kThreads = Base::kThreadCount;
    typename Mma::IteratorA iterator_A(params.params_A, params.ref_A.data(), {M, k1}, thread_idx, cutlass::MatrixCoord{m0, k0});
    typename Mma::IteratorB iterator_B(params.params_B, params.ref_B.data(), {k1, N}, thread_idx, cutlass::MatrixCoord{k0, n0});
    typename Mma::FragmentC acc;
    acc.clear();
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
    mma(seg_k / Mma::Shape::kK, acc, iterator_A, iterator_B, acc);
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();
    typename Mma::FragmentC total = acc;
    if (mode == 0) {
      float* frags = g_skt_acc + (int64_t)u * segs * kTile;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Mma::FragmentC::kElements; ++i) frags[s * kTile + i * kThreads + thread_idx] = acc[i];
      __threadfence();
      __syncthreads();
      bool last;
      if constexpr (kStaticFlag) {
        __shared__ bool flag;
        if (thread_idx == 0) {
          flag = atomicAdd(g_seg_count + u, 1) == segs - 1;
          if (flag) g_seg_count[u] = 0;
        }
        __syncthreads();
        last = flag;
      } else {
        bool mine = false;
        if (thread_idx == 0) {
          mine = atomicAdd(g_seg_count + u, 1) == segs - 1;
          if (mine) g_seg_count[u] = 0;
        }
        last = __syncthreads_or(mine);
      }
      if (!last) return;
      __threadfence();
      total.clear();
      for (int t = 0; t < segs; ++t) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < Mma::FragmentC::kElements; ++i)
          total[i] = __fadd_rn(total[i], t == s ? acc[i] : __ldcg(frags + t * kTile + i * kThreads + thread_idx));
      }
    }
    OutputOp output_op(params.output_op);
    cutlass::MatrixCoord offset(m0, n0);
    typename Epilogue::OutputTileIterator iterator_C(params.params_C, params.ref_C.data(), params.problem_size.mn(), thread_idx, offset);
    typename Epilogue::OutputTileIterator iterator_D(params.params_D, params.ref_D.data(), params.problem_size.mn(), thread_idx, offset);
    Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);
    epilogue(output_op, iterator_D, total, iterator_C);
  }
};

template <typename Kernel>
__global__ void __launch_bounds__(Kernel::kThreadCount) probe_kernel(typename Kernel::Params params, int tiles, int mode) {
  extern __shared__ __align__(16) char probe_smem[];
  Kernel op;
  op.run(params, *reinterpret_cast<typename Kernel::SharedStorage*>(probe_smem), tiles, mode);
}

template <bool kStaticFlag>
at::Tensor run_probe(at::Tensor x, at::Tensor w, int mode) {
  using G = CutlassGemm<cutlass::bfloat16_t, 32, 64, 64, 16, 32, 5>;
  using Kernel = typename G::Gemm::GemmKernel;
  using P = ProbeKernel<Kernel, kStaticFlag>;
  const int M = x.size(0), K = x.size(1), N = w.size(0), segs = gemm_segments(N, K);
  auto y = at::empty({M, N}, x.options());
  const int tiles = ((M + 31) / 32) * (N / 64);
  const int smem = (int)sizeof(typename Kernel::SharedStorage);
  static const bool attr = cudaFuncSetAttribute(probe_kernel<P>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess;
  TORCH_CHECK(attr);
  typename G::Gemm::ThreadblockSwizzle swizzle;
  const cutlass::gemm::GemmCoord problem(M, N, K);
  const auto grid_shape = swizzle.get_tiled_shape(problem, {32, 64, 64}, 1);
  typename Kernel::Params params(problem, grid_shape, {(cutlass::bfloat16_t*)x.data_ptr(), K}, {(cutlass::bfloat16_t*)w.data_ptr(), K},
                                 {(cutlass::bfloat16_t*)y.data_ptr(), N}, {(cutlass::bfloat16_t*)y.data_ptr(), N}, {1.f, 0.f});
  probe_kernel<P><<<tiles * segs, Kernel::kThreadCount, smem, at::cuda::getCurrentCUDAStream()>>>(params, tiles, mode);
  const cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, cudaGetErrorString(e));
  return y;
}

at::Tensor run(at::Tensor x, at::Tensor w, int64_t mode, bool static_flag) {
  return static_flag ? run_probe<true>(x, w, (int)mode) : run_probe<false>(x, w, (int)mode);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
