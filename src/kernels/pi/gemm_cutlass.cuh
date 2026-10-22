// CUTLASS 2.x sm80 tensor-op GEMMs (bf16 x bf16 -> fp32 accumulate), no split-K,
// for large M. Instantiated with several tile shapes; the per-element k order is
// ascending k16 steps in every instantiation, which test_gemm.py checks is
// bitwise identical to the hand-written small-M kernel. Shapes with K segments
// (gemm_segments > 1) run SegmentedKernel: the CTA runs the CUTLASS mainloop once
// per segment from a zero accumulator and folds it into a running total that
// starts at +0, the canonical sequence of gemm_inv.cuh.
// Unsegmented shapes can also run StreamKKernel: deterministic stream-K over the CUTLASS
// tiles, for mid M where the tiles fill the SMs unevenly (see run_streamk below).
#pragma once
#include <algorithm>

#include <cutlass/cutlass.h>
#include <cutlass/arch/memory_sm80.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>

#include "gemm_inv.cuh"

namespace pi_gemm {

template <typename Base>
struct SegmentedKernel : Base {
  using typename Base::Epilogue;
  using typename Base::Mma;
  using typename Base::OutputOp;
  using typename Base::Params;
  using typename Base::SharedStorage;
  using typename Base::ThreadblockSwizzle;

  CUTLASS_DEVICE void operator()(Params const& params, SharedStorage& shared_storage) {
    ThreadblockSwizzle swizzle;
    cutlass::gemm::GemmCoord tile = swizzle.get_tile_offset(params.swizzle_log_tile);
    if (params.grid_tiled_shape.m() <= tile.m() || params.grid_tiled_shape.n() <= tile.n()) return;
    const int M = params.problem_size.m(), N = params.problem_size.n(), K = params.problem_size.k();
    const int segs = gemm_segments(N, K), seg_k = K / segs;
    const int thread_idx = threadIdx.x, warp_idx = cutlass::canonical_warp_idx_sync(), lane_idx = threadIdx.x % 32;
    typename Mma::FragmentC total;
    total.clear();
    for (int s = 0; s < segs; ++s) {
      const int k0 = s * seg_k, k1 = k0 + seg_k;
      typename Mma::IteratorA iterator_A(params.params_A, params.ref_A.data(), {M, k1}, thread_idx,
                                         cutlass::MatrixCoord{tile.m() * Mma::Shape::kM, k0});
      typename Mma::IteratorB iterator_B(params.params_B, params.ref_B.data(), {k1, N}, thread_idx,
                                         cutlass::MatrixCoord{k0, tile.n() * Mma::Shape::kN});
      Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
      typename Mma::FragmentC acc;
      acc.clear();
      mma(seg_k / Mma::Shape::kK, acc, iterator_A, iterator_B, acc);
      cutlass::arch::cp_async_wait<0>();
      __syncthreads();  // the next segment's prologue (or the epilogue) reuses shared memory
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Mma::FragmentC::kElements; ++i) total[i] = __fadd_rn(total[i], acc[i]);
    }
    OutputOp output_op(params.output_op);
    cutlass::MatrixCoord offset(tile.m() * Mma::Shape::kM, tile.n() * Mma::Shape::kN);
    typename Epilogue::OutputTileIterator iterator_C(params.params_C, params.ref_C.data(), params.problem_size.mn(),
                                                     thread_idx, offset);
    typename Epilogue::OutputTileIterator iterator_D(params.params_D, params.ref_D.data(), params.problem_size.mn(),
                                                     thread_idx, offset);
    Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);
    epilogue(output_op, iterator_D, total, iterator_C);
  }
};

// ------------------------------------------------------- stream-K over CUTLASS tiles ----
// As gemm_streamk_kernel (gemm_inv.cuh) at CTA granularity: the (tile, k-tile) steps of
// all output tiles are split into equal contiguous ranges over W <= min(tiles, SMs * CPS)
// resident CTAs. Every range covers at least one tile, so a piece either starts a chain
// from zero or ends it, never both receives and hands off. A tile cut between CTAs w and
// w + 1 stays one canonical chain: w runs the first k tiles and stores its fp32
// accumulator fragments, w + 1 loads them and continues the same MMA chain (the
// mainloop's source accumulator), so the bits are those of the uncut chain. Each CTA
// runs its handed-off prefix first and its continuation last; it only waits on CTA
// w - 1's prefix, which never waits. Tiles are numbered M-fastest, so neighbouring CTAs
// share weight columns in L2. S = 1 shapes only. The handoff state is per device and
// per translation unit, as g_sk_acc; every launch leaves the flags zero.
constexpr int kSktMaxWorkers = 256, kSktMaxTile = 64 * 128;
static __device__ float g_skt_acc[kSktMaxWorkers * kSktMaxTile];
static __device__ int g_skt_flag[kSktMaxWorkers];

template <typename Base>
struct StreamKKernel : Base {
  using typename Base::Epilogue;
  using typename Base::Mma;
  using typename Base::OutputOp;
  using typename Base::Params;
  using typename Base::SharedStorage;
  static constexpr int kTile = Mma::Shape::kM * Mma::Shape::kN;
  static_assert(kTile <= kSktMaxTile, "stream-K tile larger than its handoff slot");

  CUTLASS_DEVICE void run(Params const& params, SharedStorage& shared_storage, int workers) {
    const int M = params.problem_size.m(), N = params.problem_size.n(), K = params.problem_size.k();
    const int tiles_m = (M + Mma::Shape::kM - 1) / Mma::Shape::kM, kt_n = K / Mma::Shape::kK;
    const int64_t total = (int64_t)tiles_m * ((N + Mma::Shape::kN - 1) / Mma::Shape::kN) * kt_n;
    const int wid = blockIdx.x;
    const int64_t s0 = wid * total / workers, s1 = (wid + 1) * total / workers;
    const int thread_idx = threadIdx.x, warp_idx = cutlass::canonical_warp_idx_sync(), lane_idx = threadIdx.x % 32;
    constexpr int kThreads = Base::kThreadCount;

    // Tile u, k tiles [ka, kb): from zero (ka == 0) or from CTA wid - 1's accumulator;
    // writes y (kb == kt_n) or hands the accumulator to CTA wid + 1.
    auto piece = [&](int u, int ka, int kb) {
      const int m0 = (u % tiles_m) * Mma::Shape::kM, n0 = (u / tiles_m) * Mma::Shape::kN;
      const int k0 = ka * Mma::Shape::kK, k1 = kb * Mma::Shape::kK;
      typename Mma::IteratorA iterator_A(params.params_A, params.ref_A.data(), {M, k1}, thread_idx,
                                         cutlass::MatrixCoord{m0, k0});
      typename Mma::IteratorB iterator_B(params.params_B, params.ref_B.data(), {k1, N}, thread_idx,
                                         cutlass::MatrixCoord{k0, n0});
      typename Mma::FragmentC acc;
      acc.clear();
      if (ka > 0) {
        if (thread_idx == 0) {
          while (atomicAdd(g_skt_flag + wid - 1, 0) == 0) {}
          g_skt_flag[wid - 1] = 0;  // ready for the next launch
        }
        __syncthreads();
        __threadfence();
        const float* src = g_skt_acc + (int64_t)(wid - 1) * kSktMaxTile;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < Mma::FragmentC::kElements; ++i) acc[i] = __ldcg(src + i * kThreads + thread_idx);
      }
      Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
      mma(kb - ka, acc, iterator_A, iterator_B, acc);
      cutlass::arch::cp_async_wait<0>();
      __syncthreads();  // the next piece's prologue (or the epilogue) reuses shared memory
      if (kb < kt_n) {
        float* dst = g_skt_acc + (int64_t)wid * kSktMaxTile;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < Mma::FragmentC::kElements; ++i) dst[i * kThreads + thread_idx] = acc[i];
        __threadfence();
        __syncthreads();
        if (thread_idx == 0) atomicExch(g_skt_flag + wid, 1);
      } else {
        OutputOp output_op(params.output_op);
        cutlass::MatrixCoord offset(m0, n0);
        typename Epilogue::OutputTileIterator iterator_C(params.params_C, params.ref_C.data(),
                                                         params.problem_size.mn(), thread_idx, offset);
        typename Epilogue::OutputTileIterator iterator_D(params.params_D, params.ref_D.data(),
                                                         params.problem_size.mn(), thread_idx, offset);
        Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);
        epilogue(output_op, iterator_D, acc, iterator_C);
        __syncthreads();
      }
    };

    const int64_t h_end = min(s1, (s0 + kt_n - 1) / kt_n * kt_n);  // end of the continuation
    const int64_t tail0 = max(h_end, s1 / kt_n * kt_n);            // start of the handed-off prefix
    if (tail0 < s1) piece((int)(tail0 / kt_n), 0, (int)(s1 - tail0));
    for (int64_t s = h_end; s < tail0; s += kt_n) piece((int)(s / kt_n), 0, kt_n);
    if (s0 < h_end) piece((int)(s0 / kt_n), (int)(s0 % kt_n), (int)((h_end - 1) % kt_n) + 1);
  }
};

template <typename Kernel>
__global__ void __launch_bounds__(Kernel::kThreadCount) streamk_tiles_kernel(typename Kernel::Params params,
                                                                             int workers) {
  extern __shared__ __align__(16) char skt_smem[];
  Kernel op;
  op.run(params, *reinterpret_cast<typename Kernel::SharedStorage*>(skt_smem), workers);
}

template <typename OutT, int TBM, int TBN, int TBK, int WM, int WN, int STAGES>
struct CutlassGemm {
  using Element = cutlass::bfloat16_t;
  using OutElement = std::conditional_t<sizeof(OutT) == 2, cutlass::bfloat16_t, float>;
  // Y^T[N,M] = W[N,K] * X^T[K,M]: use row-major X (MxK) as A and column-major W (KxN) as B.
  using Gemm = cutlass::gemm::device::Gemm<
      Element, cutlass::layout::RowMajor,      // A = X [M,K]
      Element, cutlass::layout::ColumnMajor,   // B = W^T, i.e. W [N,K] row-major
      OutElement, cutlass::layout::RowMajor,   // C = Y [M,N]
      float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<TBM, TBN, TBK>, cutlass::gemm::GemmShape<WM, WN, TBK>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<OutElement, 128 / cutlass::sizeof_bits<OutElement>::value,
                                                   float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, STAGES, 8, 8,
      /*SplitKSerial=*/false>;
  using Segmented = SegmentedKernel<typename Gemm::GemmKernel>;
  using StreamK = StreamKKernel<typename Gemm::GemmKernel>;

  static cudaError_t run(const void* x, int64_t ldx, const void* w, void* y, int64_t ldy, int M, int N,
                         int K, cudaStream_t stream) {
    if (gemm_segments(N, K) > 1) return run_segmented(x, ldx, w, y, ldy, M, N, K, stream);
    typename Gemm::Arguments args({M, N, K}, {(const Element*)x, ldx}, {(const Element*)w, K},
                                  {(OutElement*)y, ldy}, {(OutElement*)y, ldy}, {1.f, 0.f});
    Gemm op;
    auto st = op.can_implement(args);
    if (st != cutlass::Status::kSuccess) return cudaErrorInvalidValue;
    st = op.initialize(args, nullptr, stream);
    if (st != cutlass::Status::kSuccess) return cudaErrorInvalidValue;
    st = op(stream);
    return st == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorLaunchFailure;
  }

  // Stream-K over this tile shape with up to CPS CTAs per SM (all resident).
  static cudaError_t run_streamk(const void* x, int64_t ldx, const void* w, void* y, int64_t ldy, int M, int N,
                                 int K, int sms, int cps, cudaStream_t stream) {
    using Kernel = typename Gemm::GemmKernel;
    if (gemm_segments(N, K) != 1 || K % TBK) return cudaErrorInvalidValue;
    const int smem = (int)sizeof(typename Kernel::SharedStorage);
    static const bool attr_ok =
        smem < (48 << 10) ||
        cudaFuncSetAttribute(streamk_tiles_kernel<StreamK>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) ==
            cudaSuccess;
    static const int resident = [&] {
      int r = 0;
      return attr_ok && cudaOccupancyMaxActiveBlocksPerMultiprocessor(&r, streamk_tiles_kernel<StreamK>,
                                                                      Kernel::kThreadCount, smem) == cudaSuccess
                 ? r
                 : 0;
    }();
    if (resident < cps) return cudaErrorInvalidValue;  // CTAs that cannot all be resident could deadlock
    const int tiles = ((M + TBM - 1) / TBM) * ((N + TBN - 1) / TBN);
    const int workers = std::min({tiles, sms * cps, kSktMaxWorkers});
    typename Gemm::ThreadblockSwizzle swizzle;
    const cutlass::gemm::GemmCoord problem(M, N, K);
    const cutlass::gemm::GemmCoord grid_shape = swizzle.get_tiled_shape(problem, {TBM, TBN, TBK}, 1);
    typename Kernel::Params params(problem, grid_shape, {(Element*)x, ldx}, {(Element*)w, K},
                                   {(OutElement*)y, ldy}, {(OutElement*)y, ldy}, {1.f, 0.f});
    streamk_tiles_kernel<StreamK><<<workers, Kernel::kThreadCount, smem, stream>>>(params, workers);
    return cudaGetLastError();
  }

  static cudaError_t run_segmented(const void* x, int64_t ldx, const void* w, void* y, int64_t ldy, int M, int N,
                                   int K, cudaStream_t stream) {
    using Kernel = typename Gemm::GemmKernel;
    if ((K / gemm_segments(N, K)) % TBK) return cudaErrorInvalidValue;
    typename Gemm::ThreadblockSwizzle swizzle;
    const cutlass::gemm::GemmCoord problem(M, N, K);
    const cutlass::gemm::GemmCoord grid_shape = swizzle.get_tiled_shape(problem, {TBM, TBN, TBK}, 1);
    typename Kernel::Params params(problem, grid_shape, {(Element*)x, ldx}, {(Element*)w, K},
                                   {(OutElement*)y, ldy}, {(OutElement*)y, ldy}, {1.f, 0.f});
    const int smem = (int)sizeof(typename Kernel::SharedStorage);
    if (smem >= (48 << 10) &&
        cudaFuncSetAttribute(cutlass::Kernel<Segmented>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess)
      return cudaErrorInvalidValue;
    cutlass::Kernel<Segmented><<<swizzle.get_grid_shape(grid_shape), Kernel::kThreadCount, smem, stream>>>(params);
    return cudaGetLastError();
  }
};

}  // namespace pi_gemm
