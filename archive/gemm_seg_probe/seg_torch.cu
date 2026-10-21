#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "gemm_seg.cuh"
using namespace pi_gemm;
at::Tensor seg(at::Tensor x, at::Tensor w, at::Tensor ws, at::Tensor cnt, int64_t S, int64_t grid, int64_t warps, int64_t cfg) {
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  auto y = at::empty({M, N}, x.options());
  auto st = at::cuda::getCurrentCUDAStream();
  cudaError_t e;
  auto* yp = (__nv_bfloat16*)y.data_ptr();
  if (cfg == 0) e = run_seg<1, 128, 6>(x.data_ptr(), K, w.data_ptr(), yp, N, ws.data_ptr<float>(), cnt.data_ptr<int>(), M, N, K, S, grid, warps, st);
  else if (cfg == 1) e = run_seg<1, 256, 3>(x.data_ptr(), K, w.data_ptr(), yp, N, ws.data_ptr<float>(), cnt.data_ptr<int>(), M, N, K, S, grid, warps, st);
  else e = run_seg<1, 64, 8>(x.data_ptr(), K, w.data_ptr(), yp, N, ws.data_ptr<float>(), cnt.data_ptr<int>(), M, N, K, S, grid, warps, st);
  TORCH_CHECK(e == cudaSuccess, cudaGetErrorString(e));
  return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("seg", &seg); }
