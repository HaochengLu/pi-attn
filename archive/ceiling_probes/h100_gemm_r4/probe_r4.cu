// Slim extension for the round-4 H100 GEMM probes: the wgmma ids of gemm_sm90.cuh, cluster (TMA multicast)
// variants of its tiles (ids 200..), and the X-in-registers kernel of gemm_sm90_rs.cuh (ids 300..).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <set>

#include "gemm_sm90_rs.cuh"

c10::optional<at::Tensor> linear_r4(at::Tensor x, at::Tensor w, int64_t kernel, int64_t mode) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16);
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1) && w.is_contiguous() && x.stride(1) == 1);
  const c10::cuda::CUDAGuard guard(x.device());
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  auto y = at::empty({M, N}, x.options());
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, x.get_device());
  auto set = [](auto kern, size_t smem) {
    static std::set<const void*> done;
    if (done.insert((const void*)kern).second)
      cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  };
  using O = __nv_bfloat16;
  using namespace pi_gemm90;
  const void* xp = x.data_ptr();
  const void* wp = w.data_ptr();
  O* yp = (O*)y.data_ptr();
  const int64_t ldx = x.stride(0);
  auto st = at::cuda::getCurrentCUDAStream();
  cudaError_t err = cudaErrorInvalidValue;
#define CL(id, BM, BN, ST, BK, CX, CY) \
  case id: if (K % BK == 0) err = Sm90Gemm<BM, BN, ST, false, O, BK, CX, CY>::run(xp, ldx, wp, yp, N, M, N, K, 1, sms, set, st); break;
#define RS(id, BM, BN, ST, BK) \
  case id: if (K % BK == 0) err = Sm90GemmRS<BM, BN, ST, BK>::run(xp, ldx, wp, yp, N, M, N, K, (int)mode, sms, set, st); break;
  switch ((int)kernel) {
#include "tiles_r4.inc"
    default:
      if (K % sm90_kernel_bk((int)kernel) == 0) err = run_gemm_sm90((int)kernel, xp, ldx, wp, yp, N, M, N, K, 1, sms, set, st);
  }
  if (err == cudaErrorInvalidValue) return c10::nullopt;
  TORCH_CHECK(err == cudaSuccess, "gemm kernel ", kernel, " failed: ", cudaGetErrorString(err));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("linear_r4", &linear_r4); }
