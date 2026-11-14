// Slim extension for the H100 GEMM probes: only the wgmma ids (bf16 output), with the register-only chain mode
// of gemm_sm90.cuh (PI_GEMM90_PROBE) and a register-only mma.sync chain.
#define PI_GEMM90_PROBE 1
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <set>

#include "gemm_sm90.cuh"

c10::optional<at::Tensor> linear90(at::Tensor x, at::Tensor w, int64_t kernel, int64_t probe, int64_t pf) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16);
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1) && w.is_contiguous() && x.stride(1) == 1);
  const c10::cuda::CUDAGuard guard(x.device());
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  auto y = at::empty({M, N}, x.options());
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, x.get_device());
  if (K % pi_gemm90::sm90_kernel_bk((int)kernel)) return c10::nullopt;
  pi_gemm90::g_probe = (int)probe;
  pi_gemm90::g_pf = (int)pf;
  auto set = [](auto kern, size_t smem) {
    static std::set<const void*> done;
    if (done.insert((const void*)kern).second)
      cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  };
  const cudaError_t err = pi_gemm90::run_gemm_sm90((int)kernel, x.data_ptr(), x.stride(0), w.data_ptr(),
                                                   (__nv_bfloat16*)y.data_ptr(), N, M, N, K, 1, sms, set,
                                                   at::cuda::getCurrentCUDAStream());
  pi_gemm90::g_probe = 0;
  if (err == cudaErrorInvalidValue) return c10::nullopt;
  TORCH_CHECK(err == cudaSuccess, "gemm kernel ", kernel, " failed: ", cudaGetErrorString(err));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// Every warp: one m16n8k16 chain of `steps` dependent mma.sync steps on register operands.
__global__ void mma_chain_kernel(int steps, float* out) {
  uint32_t h = (threadIdx.x + 977 * blockIdx.x) * 2654435761u;
  uint32_t a[4], b[2];
  for (int i = 0; i < 6; ++i) {
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    const __nv_bfloat162 v = __floats2bfloat162_rn((float)(h & 0xffff) / 32768.f - 1.f, (float)(h >> 16) / 32768.f - 1.f);
    (i < 4 ? a[i] : b[i - 4]) = *reinterpret_cast<const uint32_t*>(&v);
  }
  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
  for (int s = 0; s < steps; ++s)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  if (threadIdx.x == 0) out[blockIdx.x] = c0 + c1 + c2 + c3;
}

void mma_chain(at::Tensor out, int64_t ctas, int64_t warps, int64_t steps) {
  const c10::cuda::CUDAGuard guard(out.device());
  TORCH_CHECK(out.numel() >= ctas && out.scalar_type() == at::kFloat);
  mma_chain_kernel<<<(int)ctas, (int)warps * 32, 0, at::cuda::getCurrentCUDAStream()>>>((int)steps, (float*)out.data_ptr());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear90", &linear90);
  m.def("mma_chain", &mma_chain);
}
