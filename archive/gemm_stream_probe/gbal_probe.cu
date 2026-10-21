// PERF PROBE: balanced lockstep small-M GEMM (not used by the engine). CTA c owns the
// contiguous n8 units [c U / grid, (c + 1) U / grid); its warps (one per unit, all in
// lockstep) share each X stage, as in gemm_small_kernel; extra warps loop.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "gemm_inv.cuh"
namespace pi_gemm {
template <typename OutT, int BK, int STAGES>
__global__ void bal_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx, const __nv_bfloat16* __restrict__ w,
                           OutT* __restrict__ y, int64_t ldy, int M, int N, int K) {
  constexpr int kStride = BK + 8;
  extern __shared__ __align__(16) __nv_bfloat16 smem[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
  const int stage_elems = (16 + 8 * warps) * kStride;
  const int units = N / 8;
  const int u0 = (int)((int64_t)blockIdx.x * units / gridDim.x), u1 = (int)((int64_t)(blockIdx.x + 1) * units / gridDim.x);
  const int ktiles = K / BK;
  const int g = lane >> 2, t = lane & 3, mat = lane >> 3, r8 = lane & 7;
  for (int base = u0; base < u1; base += warps) {
    const int nrows = min(warps, u1 - base) * 8;  // weight rows this pass
    auto issue = [&](int kt) {
      __nv_bfloat16* st = smem + (kt % STAGES) * stage_elems;
      constexpr int kVec = BK / 8;
      for (int i = threadIdx.x; i < (16 + nrows) * kVec; i += blockDim.x) {
        const int r = i / kVec, v = i % kVec;
        if (r < 16) {
          if (r < M) cp_async16(st + r * kStride + v * 8, x + (int64_t)r * ldx + kt * BK + v * 8, 16);
        } else {
          cp_async16(st + r * kStride + v * 8, w + ((int64_t)base * 8 + r - 16) * K + kt * BK + v * 8, 16);
        }
      }
    };
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
      if (s < ktiles) issue(s);
      cp_async_commit();
    }
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    const bool active = warp * 8 < nrows;
    for (int kt = 0; kt < ktiles; ++kt) {
      cp_async_wait<STAGES - 2>();
      __syncthreads();
      if (kt + STAGES - 1 < ktiles) issue(kt + STAGES - 1);
      cp_async_commit();
      if (!active) continue;
      const __nv_bfloat16* sx = smem + (kt % STAGES) * stage_elems;
      const __nv_bfloat16* sw = sx + (16 + warp * 8) * kStride;
#pragma unroll
      for (int ks = 0; ks < BK / 16; ks += 2) {
        uint32_t b[4];
        ldmatrix_x4(b, sw + r8 * kStride + ks * 16 + mat * 8);
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
          const int c0 = (ks + kk) * 16 + t * 2;
          uint32_t a[4];
          a[0] = g < M ? *reinterpret_cast<const uint32_t*>(sx + g * kStride + c0) : 0u;
          a[1] = g + 8 < M ? *reinterpret_cast<const uint32_t*>(sx + (g + 8) * kStride + c0) : 0u;
          a[2] = g < M ? *reinterpret_cast<const uint32_t*>(sx + g * kStride + c0 + 8) : 0u;
          a[3] = g + 8 < M ? *reinterpret_cast<const uint32_t*>(sx + (g + 8) * kStride + c0 + 8) : 0u;
          mma_bf16(acc, a, b[2 * kk], b[2 * kk + 1]);
        }
      }
    }
    cp_async_wait<0>();
    __syncthreads();
    if (active) {
      const int n = (base + warp) * 8 + t * 2;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int m = g + h * 8;
        if (m < M) *reinterpret_cast<__nv_bfloat162*>(y + (int64_t)m * ldy + n) = __floats2bfloat162_rn(acc[2 * h], acc[2 * h + 1]);
      }
    }
  }
}
}  // namespace pi_gemm
using namespace pi_gemm;
at::Tensor run(at::Tensor x, at::Tensor w, int64_t max_warps, int64_t cfg, int64_t ctas_per_sm) {
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  auto y = at::empty({M, N}, x.options());
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, x.get_device());
  const int units = N / 8, grid = std::min<int>(sms * ctas_per_sm, units);
  const int W = std::min<int>(max_warps, (units + grid - 1) / grid);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto go = [&](auto kern, int bk, int st) {
    const size_t smem = (size_t)st * (16 + 8 * W) * (bk + 8) * 2;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kern<<<grid, W * 32, smem, stream>>>((const __nv_bfloat16*)x.data_ptr(), x.stride(0),
        (const __nv_bfloat16*)w.data_ptr(), (__nv_bfloat16*)y.data_ptr(), N, M, N, K);
  };
  if (cfg == 0) go(bal_kernel<__nv_bfloat16, 256, 4>, 256, 4);
  else if (cfg == 1) go(bal_kernel<__nv_bfloat16, 128, 6>, 128, 6);
  else if (cfg == 2) go(bal_kernel<__nv_bfloat16, 256, 3>, 256, 3);
  else go(bal_kernel<__nv_bfloat16, 128, 8>, 128, 8);
  return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
