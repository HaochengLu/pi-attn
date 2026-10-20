// PyTorch bindings for the M-invariant GEMM (tests and benchmarks).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "gemm_inv.cuh"
#include "gemm_cutlass.cuh"

using namespace pi_gemm;

// kernel: 0 = auto, 1..4 = small with G = kernel, 5 = hand-written large,
// 10.. = CUTLASS tiles (10: 128x256x32 s3, 11: 128x128x32 s4, 12: 64x128x32 s4, 13: 64x64x32 s5).
at::Tensor linear(at::Tensor x, at::Tensor w, int64_t kernel, bool out_fp32) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16);
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1) && w.is_contiguous());
  TORCH_CHECK(x.stride(1) == 1, "x rows must be contiguous");
  const c10::cuda::CUDAGuard guard(x.device());
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(K % 64 == 0 && N % 32 == 0, "K must be a multiple of 64 and N of 32");
  TORCH_CHECK(x.data_ptr() != w.data_ptr());
  auto y = at::empty({M, N}, x.options().dtype(out_fp32 ? at::kFloat : at::kBFloat16));
  if (M == 0) return y;
  int k = (int)kernel;
  if (k == 0) k = M <= 16 ? 1 : M <= 32 ? 2 : M <= 64 ? 4 : 5;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto run = [&](auto* out) {
    using OutT = std::remove_pointer_t<decltype(out)>;
    auto small = [&](auto kern, int G, int W, int BK = kSmallBK, int ST = kSmallStages) {
      TORCH_CHECK(K % BK == 0, "K must be a multiple of BK for this variant");
      const size_t smem = (size_t)ST * (16 * G + 8 * W) * (BK + 8) * sizeof(__nv_bfloat16);
      cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
      kern<<<dim3(N / (8 * W), (M + 16 * G - 1) / (16 * G)), W * 32, smem, stream>>>(
          (const __nv_bfloat16*)x.data_ptr(), x.stride(0), (const __nv_bfloat16*)w.data_ptr(), out, N, M, N, K);
    };
    // 1..4: small kernel, 4 warps, G = 1, 2, 4, 4; 6..9: small kernel W = 1 or 2 warps:
    // 6 = (G1, W1), 7 = (G1, W2), 8 = (G2, W1), 9 = (G4, W2).
    if (k == 1) small(gemm_small_kernel<1, 4, OutT>, 1, 4);
    else if (k == 2) small(gemm_small_kernel<2, 4, OutT>, 2, 4);
    else if (k == 3 || k == 4) small(gemm_small_kernel<4, 4, OutT>, 4, 4);
    else if (k == 6) small(gemm_small_kernel<1, 1, OutT>, 1, 1);
    else if (k == 7) small(gemm_small_kernel<1, 2, OutT>, 1, 2);
    else if (k == 8) small(gemm_small_kernel<2, 1, OutT>, 2, 1);
    else if (k == 9) small(gemm_small_kernel<4, 2, OutT>, 4, 2);
    // Deep-pipeline variants for weight streaming (decode): (G, W, BK, STAGES).
    else if (k == 20) small(gemm_small_kernel<1, 4, OutT, 128, 6>, 1, 4, 128, 6);
    else if (k == 21) small(gemm_small_kernel<1, 4, OutT, 256, 3>, 1, 4, 256, 3);
    else if (k == 22) small(gemm_small_kernel<1, 2, OutT, 256, 4>, 1, 2, 256, 4);
    else if (k == 23) small(gemm_small_kernel<1, 1, OutT, 256, 6>, 1, 1, 256, 6);
    else if (k == 24) small(gemm_small_kernel<2, 4, OutT, 128, 5>, 2, 4, 128, 5);
    else if (k == 25) small(gemm_small_kernel<4, 4, OutT, 128, 3>, 4, 4, 128, 3);
    else if (k >= 10 && k < 20) {
      cudaError_t e = cudaSuccess;
      const void* xp = x.data_ptr();
      const void* wp = w.data_ptr();
      if (k == 10) e = CutlassGemm<OutT, 128, 256, 32, 64, 64, 3>::run(xp, x.stride(0), wp, out, N, M, N, K, stream);
      else if (k == 11) e = CutlassGemm<OutT, 128, 128, 32, 64, 64, 4>::run(xp, x.stride(0), wp, out, N, M, N, K, stream);
      else if (k == 12) e = CutlassGemm<OutT, 64, 128, 32, 32, 64, 4>::run(xp, x.stride(0), wp, out, N, M, N, K, stream);
      else e = CutlassGemm<OutT, 64, 64, 32, 32, 32, 5>::run(xp, x.stride(0), wp, out, N, M, N, K, stream);
      TORCH_CHECK(e == cudaSuccess, "cutlass gemm failed for kernel ", k);
    } else {
      const size_t smem = kLargeSmemElems * sizeof(__nv_bfloat16);
      cudaFuncSetAttribute(gemm_large_kernel<OutT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
      gemm_large_kernel<OutT><<<dim3((N + kLargeBN - 1) / kLargeBN, (M + kLargeBM - 1) / kLargeBM), 256, smem, stream>>>(
          (const __nv_bfloat16*)x.data_ptr(), x.stride(0), (const __nv_bfloat16*)w.data_ptr(), out, N, M, N, K);
    }
  };
  if (out_fp32) run((float*)y.data_ptr()); else run((__nv_bfloat16*)y.data_ptr());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("linear", &linear); }
