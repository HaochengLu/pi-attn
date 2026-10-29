// Mid-M probe: CUTLASS stream-K tiles at higher CTAs per SM.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "gemm_dispatch.cuh"
using namespace pi_gemm;
c10::optional<at::Tensor> run(at::Tensor x, at::Tensor w, int64_t cfg, int64_t segs, int64_t cps) {
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  auto y = at::empty({M, N}, x.options());
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, x.get_device());
  auto st = at::cuda::getCurrentCUDAStream();
  cudaError_t e = cudaErrorInvalidValue;
  const void* xp = x.data_ptr(); const void* wp = w.data_ptr(); void* yp = y.data_ptr();
  switch (cfg) {
    case 0: e = CutlassGemm<__nv_bfloat16, 64, 64, 64, 32, 32, 4>::run_streamk(xp, K, wp, yp, N, M, N, K, (int)segs, sms, (int)cps, st); break;
    case 1: e = CutlassGemm<__nv_bfloat16, 128, 64, 32, 64, 32, 4>::run_streamk(xp, K, wp, yp, N, M, N, K, (int)segs, sms, (int)cps, st); break;
    case 2: break;  // 128x128: larger than the stream-K handoff slot
    case 3: e = CutlassGemm<__nv_bfloat16, 64, 128, 32, 32, 64, 4>::run_streamk(xp, K, wp, yp, N, M, N, K, (int)segs, sms, (int)cps, st); break;
    case 4: e = CutlassGemm<__nv_bfloat16, 128, 64, 64, 64, 32, 3>::run_streamk(xp, K, wp, yp, N, M, N, K, (int)segs, sms, (int)cps, st); break;
  }
  if (e == cudaErrorInvalidValue) return c10::nullopt;
  TORCH_CHECK(e == cudaSuccess, cudaGetErrorString(e));
  return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
