// Candidate CUTLASS configurations for mid-M GEMMs, launched directly (not through run_gemm).
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "gemm_dispatch.cuh"

using namespace pi_gemm;

#define CFGS(X)                                         \
  X(0, CutlassGemm<__nv_bfloat16, 128, 64, 64, 64, 32, 5>::run)    \
  X(1, CutlassGemm<__nv_bfloat16, 128, 64, 64, 64, 32, 4>::run)    \
  X(2, CutlassGemm<__nv_bfloat16, 128, 64, 32, 64, 32, 6>::run)    \
  X(3, CutlassGemm<__nv_bfloat16, 64, 128, 64, 32, 64, 4>::run)    \
  X(4, CutlassGemm<__nv_bfloat16, 128, 128, 32, 64, 64, 4, 4>::run) \
  X(5, CutlassGemm<__nv_bfloat16, 64, 64, 32, 32, 32, 5, 4>::run)   \
  X(6, CutlassGemm<__nv_bfloat16, 128, 128, 64, 64, 64, 3>::run)    \
  X(7, CutlassGemm<__nv_bfloat16, 64, 64, 64, 32, 32, 5>::run)      \
  X(8, CutlassGemm<__nv_bfloat16, 128, 256, 64, 64, 64, 3>::run)    \
  X(9, CutlassGemm<__nv_bfloat16, 256, 128, 32, 64, 64, 3>::run)

c10::optional<at::Tensor> run(at::Tensor x, at::Tensor w, int64_t cfg, int64_t segs) {
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  auto y = at::empty({M, N}, x.options());
  auto st = at::cuda::getCurrentCUDAStream();
  cudaError_t err = cudaErrorInvalidValue;
#define RUN(i, ...) \
  if (cfg == i) err = __VA_ARGS__(x.data_ptr(), x.stride(0), w.data_ptr(), y.data_ptr(), N, M, N, K, (int)segs, st);
  CFGS(RUN)
  if (err == cudaErrorInvalidValue) return c10::nullopt;
  TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run);
  m.def("count", [] { return 10; });
}
