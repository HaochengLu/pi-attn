// PERF PROBE: streaming GEMV with X staged in the per-warp ring (not used by the engine).
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "gemm_inv.cuh"
namespace pi_gemm {
// Per-warp ring stage: XR rows of X (XR = 8 for M <= 8, else 16) then 8 rows of W.
template <int BK, int STAGES, int XR>
constexpr int pwarp_elems() { return STAGES * (XR + 8) * (BK + 8); }

template <typename OutT, int BK, int STAGES, int XR>
__global__ void probe_stream_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx,
                                    const __nv_bfloat16* __restrict__ w, OutT* __restrict__ y,
                                    int64_t ldy, int M, int N, int K) {
  constexpr int kStride = BK + 8, kStage = (XR + 8) * kStride;
  extern __shared__ __align__(16) __nv_bfloat16 smem[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
  __nv_bfloat16* ring = smem + warp * pwarp_elems<BK, STAGES, XR>();
  const int units = N / 8;
  const int u_end = (int)((int64_t)(blockIdx.x + 1) * units / gridDim.x);
  const int ktiles = K / BK;
  const int g = lane >> 2, t = lane & 3, mat = lane >> 3, r8 = lane & 7;
  for (int u = (int)((int64_t)blockIdx.x * units / gridDim.x) + warp; u < u_end; u += warps) {
    const __nv_bfloat16* wu = w + (int64_t)u * 8 * K;
    auto issue = [&](int kt) {
      __nv_bfloat16* st = ring + (kt % STAGES) * kStage;
      constexpr int kVec = BK / 8;
      for (int i = lane; i < (XR + 8) * kVec; i += 32) {
        const int r = i / kVec, v = i % kVec;
        if (r < XR) {
          if (r < M) cp_async16(st + r * kStride + v * 8, x + (int64_t)r * ldx + kt * BK + v * 8, 16);
        } else {
          cp_async16(st + r * kStride + v * 8, wu + (int64_t)(r - XR) * K + kt * BK + v * 8, 16);
        }
      }
    };
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
      if (s < ktiles) issue(s);
      cp_async_commit();
    }
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int kt = 0; kt < ktiles; ++kt) {
      cp_async_wait<STAGES - 2>();
      __syncwarp();
      if (kt + STAGES - 1 < ktiles) issue(kt + STAGES - 1);
      cp_async_commit();
      const __nv_bfloat16* sx = ring + (kt % STAGES) * kStage;
      const __nv_bfloat16* sw = sx + XR * kStride;
#pragma unroll
      for (int ks = 0; ks < BK / 16; ks += 2) {
        uint32_t b[4];
        ldmatrix_x4(b, sw + r8 * kStride + ks * 16 + mat * 8);
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
          const int c0 = (ks + kk) * 16 + t * 2;
          uint32_t a[4];
          a[0] = g < M ? *reinterpret_cast<const uint32_t*>(sx + g * kStride + c0) : 0u;
          a[2] = g < M ? *reinterpret_cast<const uint32_t*>(sx + g * kStride + c0 + 8) : 0u;
          if constexpr (XR == 16) {
            a[1] = g + 8 < M ? *reinterpret_cast<const uint32_t*>(sx + (g + 8) * kStride + c0) : 0u;
            a[3] = g + 8 < M ? *reinterpret_cast<const uint32_t*>(sx + (g + 8) * kStride + c0 + 8) : 0u;
          } else {
            a[1] = a[3] = 0u;
          }
          mma_bf16(acc, a, b[2 * kk], b[2 * kk + 1]);
        }
      }
    }
    cp_async_wait<0>();
    __syncwarp();
    const int n = u * 8 + t * 2;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int m = g + h * 8;
      if (m >= M) continue;
      *reinterpret_cast<__nv_bfloat162*>(y + (int64_t)m * ldy + n) = __floats2bfloat162_rn(acc[2 * h], acc[2 * h + 1]);
    }
  }
}
}  // namespace pi_gemm
using namespace pi_gemm;
at::Tensor run(at::Tensor x, at::Tensor w, int64_t warps, int64_t cfg, int64_t ctas_per_sm) {
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  auto y = at::empty({M, N}, x.options());
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, x.get_device());
  const int units = N / 8, grid = std::min<int>(sms * ctas_per_sm, units);
  const int W = std::min<int>(warps, (units + grid - 1) / grid);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto go = [&](auto kern, size_t elems) {
    const size_t smem = W * elems * 2;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kern<<<grid, W * 32, smem, stream>>>((const __nv_bfloat16*)x.data_ptr(), x.stride(0),
        (const __nv_bfloat16*)w.data_ptr(), (__nv_bfloat16*)y.data_ptr(), N, M, N, K);
  };
  if (cfg == 0) go(probe_stream_kernel<__nv_bfloat16, 256, 4, 8>, pwarp_elems<256, 4, 8>());
  else if (cfg == 1) go(probe_stream_kernel<__nv_bfloat16, 256, 6, 8>, pwarp_elems<256, 6, 8>());
  else if (cfg == 2) go(probe_stream_kernel<__nv_bfloat16, 512, 3, 8>, pwarp_elems<512, 3, 8>());
  else go(probe_stream_kernel<__nv_bfloat16, 128, 8, 8>, pwarp_elems<128, 8, 8>());
  return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
