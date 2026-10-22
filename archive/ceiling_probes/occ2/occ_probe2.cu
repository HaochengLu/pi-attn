// PERF PROBE: 8-warp and two-CTAs-per-SM CUTLASS tiles for mid M (128..1024), each run
// as a plain/segment-sequential tile (mode 0), segment-parallel (mode 1, S > 1 shapes)
// or stream-K with one CTA per SM (mode 2, S = 1 shapes). Every instantiation keeps the
// canonical per-element chain.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "gemm_cutlass.cuh"
using namespace pi_gemm;

#define CFGS(X) \
  X(0, 64, 64, 64, 32, 32, 3) X(1, 128, 64, 32, 32, 32, 4) X(2, 64, 128, 32, 32, 32, 4) \
  X(3, 128, 128, 32, 64, 32, 3) X(4, 128, 64, 64, 64, 32, 3) X(5, 64, 64, 32, 32, 32, 6) \
  X(6, 128, 128, 32, 32, 64, 4) X(7, 64, 128, 64, 32, 32, 3)

at::Tensor run(at::Tensor x, at::Tensor w, int64_t cfg, int64_t mode) {
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  auto y = at::empty({M, N}, x.options());
  auto st = at::cuda::getCurrentCUDAStream();
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, x.get_device());
  cudaError_t e = cudaErrorInvalidValue;
  switch (cfg) {
#define CASE(i, a, b, c, d, f, s)                                                                                \
  case i: {                                                                                                      \
    using G = CutlassGemm<__nv_bfloat16, a, b, c, d, f, s>;                                                     \
    if (mode == 0) e = G::run(x.data_ptr(), K, w.data_ptr(), y.data_ptr(), N, M, N, K, st);                    \
    if (mode == 1) e = G::run_segpar(x.data_ptr(), K, w.data_ptr(), y.data_ptr(), N, M, N, K, st);             \
    if constexpr (a * b <= kSktMaxTile) /* stream-K handoff slot */                                            \
      if (mode == 2) e = G::run_streamk(x.data_ptr(), K, w.data_ptr(), y.data_ptr(), N, M, N, K, sms, 1, st);  \
    break;                                                                                                       \
  }
    CFGS(CASE)
  }
  if (e != cudaSuccess) return at::Tensor();  // this mode does not run this shape
  return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
