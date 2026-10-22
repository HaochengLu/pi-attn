// PERF PROBE: extra CUTLASS tile shapes for mid M (64..512) on narrow-N projections.
// Every instantiation keeps the canonical per-element chain (ascending k16 MMAs).
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "gemm_cutlass.cuh"
using namespace pi_gemm;

#define CFGS(X) \
  X(0, 32, 64, 32, 32, 32, 5) X(1, 64, 32, 32, 32, 32, 5) X(2, 32, 32, 32, 32, 32, 6) \
  X(3, 64, 64, 64, 32, 32, 4) X(4, 64, 64, 32, 32, 32, 8) X(5, 32, 64, 64, 16, 32, 5) \
  X(6, 64, 32, 64, 32, 32, 5) X(7, 32, 32, 64, 32, 32, 6) X(8, 64, 64, 64, 32, 32, 6) \
  X(9, 64, 64, 32, 16, 32, 6) X(10, 128, 64, 32, 32, 32, 5) X(11, 64, 128, 32, 32, 32, 4)

at::Tensor run(at::Tensor x, at::Tensor w, int64_t cfg) {
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  auto y = at::empty({M, N}, x.options());
  auto st = at::cuda::getCurrentCUDAStream();
  cudaError_t e = cudaErrorInvalidValue;
  switch (cfg) {
#define CASE(i, a, b, c, d, f, s) \
  case i: e = CutlassGemm<__nv_bfloat16, a, b, c, d, f, s>::run(x.data_ptr(), K, w.data_ptr(), y.data_ptr(), N, M, N, K, st); break;
    CFGS(CASE)
  }
  TORCH_CHECK(e == cudaSuccess, "cfg ", cfg, ": ", cudaGetErrorString(e));
  return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
