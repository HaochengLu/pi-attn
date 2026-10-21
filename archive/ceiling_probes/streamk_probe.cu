// PERF PROBE: deterministic stream-K for decode GEMMs (M <= 8, S = 1 shapes).
// The canonical per-element chain (acc = 0; acc = mma(x[k16], w[k16], acc) for k16
// ascending) is cut between two warps at an arbitrary k step: the first warp hands its
// fp32 accumulator to the second, which continues the same chain. Work = (n8 unit,
// k step) pairs, split evenly over all warps, so every SM streams the same bytes.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "gemm_inv.cuh"

namespace pi_gemm {
constexpr int kSkMaxWorkers = 4096;
static __device__ float g_sk_acc[kSkMaxWorkers * 128];
static __device__ int g_sk_flag[kSkMaxWorkers];

template <int BK, int STAGES>
__global__ void gemm_streamk_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx,
                                    const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ y,
                                    int64_t ldy, int M, int N, int K, int workers) {
  constexpr int kStride = BK + 8, kStage = 16 * kStride;  // 8 X rows, then 8 W rows
  extern __shared__ __align__(16) __nv_bfloat16 smem[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int wid = blockIdx.x * (blockDim.x >> 5) + warp;
  if (wid >= workers) return;
  __nv_bfloat16* ring = smem + warp * STAGES * kStage;
  const int units = N / 8, kt_n = K / BK;
  const int64_t total = (int64_t)units * kt_n;
  const int64_t s0 = wid * total / workers, s1 = (wid + 1) * total / workers;
  const int g = lane >> 2, t = lane & 3, mat = lane >> 3, r8 = lane & 7;

  // One piece: unit u, k tiles [ka, kb). in_pred: start from worker wid-1's accumulator;
  // kb == kt_n: write y, else hand the accumulator to worker wid+1.
  auto piece = [&](int u, int ka, int kb) {
    auto issue = [&](int kt) {
      __nv_bfloat16* st = ring + (kt % STAGES) * kStage;
      constexpr int kVec = BK / 8;
#pragma unroll
      for (int i = lane; i < 16 * kVec; i += 32) {
        const int r = i / kVec, v = i % kVec, kc = kt * BK + v * 8;
        if (r >= 8) cp_async16(st + r * kStride + v * 8, w + ((int64_t)u * 8 + r - 8) * K + kc, 16);
        else if (r < M) cp_async16(st + r * kStride + v * 8, x + (int64_t)r * ldx + kc, 16);
        else cp_async16(st + r * kStride + v * 8, x, 0);
      }
    };
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
      if (ka + s < kb) issue(ka + s);
      cp_async_commit();
    }
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    if (ka > 0) {  // continue worker wid-1's chain
      if (lane == 0) {
        while (atomicAdd(g_sk_flag + wid - 1, 0) == 0) {}
        g_sk_flag[wid - 1] = 0;  // ready for the next launch
      }
      __syncwarp();
      __threadfence();
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[i] = __ldcg(g_sk_acc + (wid - 1) * 128 + lane * 4 + i);
    }
    for (int kt = ka; kt < kb; ++kt) {
      cp_async_wait<STAGES - 2>();
      __syncwarp();
      if (kt + STAGES - 1 < kb) issue(kt + STAGES - 1);
      cp_async_commit();
      const __nv_bfloat16* sx = ring + (kt % STAGES) * kStage;
      const __nv_bfloat16* sw = sx + 8 * kStride;
#pragma unroll
      for (int ks = 0; ks < BK / 16; ks += 2) {
        uint32_t b[4];
        ldmatrix_x4(b, sw + r8 * kStride + ks * 16 + mat * 8);
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
          uint32_t a[4];
          ldmatrix_x4(a, sx + r8 * kStride + (ks + kk) * 16 + (mat >> 1) * 8);
          a[1] = a[3] = 0u;  // rows 8..15 (M <= 8): zero, as the zero-filled rows of the other kernels
          mma_bf16(acc, a, b[2 * kk], b[2 * kk + 1]);
        }
      }
      __syncwarp();
    }
    cp_async_wait<0>();
    __syncwarp();
    if (kb < kt_n) {  // hand off
#pragma unroll
      for (int i = 0; i < 4; ++i) g_sk_acc[wid * 128 + lane * 4 + i] = acc[i];
      __threadfence();
      __syncwarp();
      if (lane == 0) atomicExch(g_sk_flag + wid, 1);
    } else if (g < M) {
      *reinterpret_cast<__nv_bfloat162*>(y + (int64_t)g * ldy + u * 8 + t * 2) = __floats2bfloat162_rn(acc[0], acc[1]);
    }
  };

  // Order: the prefix piece handed to wid+1 first, whole units, the continuation last.
  const int64_t h_end = min(s1, (s0 + kt_n - 1) / kt_n * kt_n);  // end of the continuation piece
  const int64_t tail0 = max(h_end, s1 / kt_n * kt_n);            // start of the prefix piece
  if (tail0 < s1) piece((int)(tail0 / kt_n), 0, (int)(s1 - tail0));
  for (int64_t s = h_end; s < tail0; s += kt_n) piece((int)(s / kt_n), 0, kt_n);
  if (s0 < h_end) piece((int)(s0 / kt_n), (int)(s0 % kt_n), (int)((h_end - 1) % kt_n) + 1);
}

template <int BK, int STAGES>
void launch(at::Tensor x, at::Tensor w, at::Tensor y, int warps, int wpsm) {
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  const int workers = std::min({N / 8, sms * wpsm, kSkMaxWorkers});
  const size_t smem = (size_t)warps * STAGES * 16 * (BK + 8) * 2;
  cudaFuncSetAttribute(gemm_streamk_kernel<BK, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  gemm_streamk_kernel<BK, STAGES><<<(workers + warps - 1) / warps, warps * 32, smem, at::cuda::getCurrentCUDAStream()>>>(
      (const __nv_bfloat16*)x.data_ptr(), x.stride(0), (const __nv_bfloat16*)w.data_ptr(),
      (__nv_bfloat16*)y.data_ptr(), y.stride(0), M, N, K, workers);
}
}  // namespace pi_gemm

// cfg: 0 = BK128/S4, 1 = BK256/S3, 2 = BK128/S6, 3 = BK64/S8. warps per CTA; wpsm workers per SM.
at::Tensor run(at::Tensor x, at::Tensor w, int64_t cfg, int64_t warps, int64_t wpsm) {
  TORCH_CHECK(x.size(0) <= 8 && x.size(1) == w.size(1) && w.size(0) % 8 == 0);
  auto y = at::empty({x.size(0), w.size(0)}, x.options());
  if (cfg == 0) pi_gemm::launch<128, 4>(x, w, y, warps, wpsm);
  else if (cfg == 1) pi_gemm::launch<256, 3>(x, w, y, warps, wpsm);
  else if (cfg == 2) pi_gemm::launch<128, 6>(x, w, y, warps, wpsm);
  else pi_gemm::launch<64, 8>(x, w, y, warps, wpsm);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
