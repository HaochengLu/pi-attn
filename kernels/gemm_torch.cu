// PyTorch bindings for the M-invariant GEMM (tests and benchmarks). Kernels are launched
// through the engine's own id table (gemm_dispatch.cuh, symlinked from the engine), so
// what is tested is what the engine's tuner chooses from.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "gemm_dispatch.cuh"

using namespace pi_gemm;

// y = x w^T with kernel id `kernel` (0 = 13; every id gives the same bits) and S = segs
// K segments (-1: gemm_segments of w's shape; a column shard of a weight passes the whole
// weight's S). Returns None when the id refuses the shape (cudaErrorInvalidValue).
// test_gemm.py checks every None against its own expectation, so a launch failure reported
// with that code still fails it.
c10::optional<at::Tensor> linear(at::Tensor x, at::Tensor w, int64_t kernel, bool out_fp32, int64_t segs) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16);
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1) && w.is_contiguous());
  TORCH_CHECK(x.stride(1) == 1, "x rows must be contiguous");
  const c10::cuda::CUDAGuard guard(x.device());
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(K % 64 == 0 && N % 32 == 0, "K must be a multiple of 64 and N of 32");
  TORCH_CHECK(x.data_ptr() != w.data_ptr());
  auto y = at::empty({M, N}, x.options().dtype(out_fp32 ? at::kFloat : at::kBFloat16));
  if (M == 0) return y;
  GemmDevice dev;
  cudaDeviceGetAttribute(&dev.sms, cudaDevAttrMultiProcessorCount, x.get_device());
  cudaDeviceGetAttribute(&dev.smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, x.get_device());
  const int k = kernel == 0 ? 13 : (int)kernel, s = segs < 0 ? gemm_segments(N, K) : (int)segs;
  auto stream = at::cuda::getCurrentCUDAStream();
  const cudaError_t err =
      out_fp32 ? run_gemm(k, x.data_ptr(), x.stride(0), w.data_ptr(), (float*)y.data_ptr(), N, M, N, K, s, dev, stream)
               : run_gemm(k, x.data_ptr(), x.stride(0), w.data_ptr(), (__nv_bfloat16*)y.data_ptr(), N, M, N, K, s, dev,
                          stream);
  if (err == cudaErrorInvalidValue) return c10::nullopt;
  TORCH_CHECK(err == cudaSuccess, "gemm kernel ", k, " failed: ", cudaGetErrorString(err));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear", &linear, py::arg("x"), py::arg("w"), py::arg("kernel"), py::arg("out_fp32"), py::arg("segs") = -1);
  m.def("gemm_segments", [](int N, int K, int t_max) { return gemm_segments(N, K, t_max); }, py::arg("N"), py::arg("K"),
        py::arg("t_max") = 1);
  m.def("small_kernels", [] { return small_kernels(); }, "ids the engine tunes for M buckets <= 64");
  m.def("large_kernels", [] { return large_kernels(); }, "ids the engine tunes for every M bucket");
}
