// CUTLASS 2.x sm80 tensor-op GEMMs (bf16 x bf16 -> fp32 accumulate), no split-K,
// for large M. Instantiated with several tile shapes; the per-element k order is
// ascending k16 steps in every instantiation, which test_gemm.py checks is
// bitwise identical to the hand-written small-M kernel. Shapes with K segments
// (gemm_segments > 1) run SegmentedKernel: the CTA runs the CUTLASS mainloop once
// per segment from a zero accumulator and folds it into a running total that
// starts at +0, the canonical sequence of gemm_inv.cuh.
#pragma once
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
