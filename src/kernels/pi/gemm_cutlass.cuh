// CUTLASS 2.x sm80 tensor-op GEMMs (bf16 x bf16 -> fp32 accumulate), no split-K,
// for large M. Instantiated with several tile shapes; the per-element k order is
// ascending k16 steps in every instantiation, which test_gemm.py checks is
// bitwise identical to the hand-written small-M kernel.
#pragma once
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>

namespace pi_gemm {

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

  static cudaError_t run(const void* x, int64_t ldx, const void* w, void* y, int64_t ldy, int M, int N,
                         int K, cudaStream_t stream) {
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
};

}  // namespace pi_gemm
