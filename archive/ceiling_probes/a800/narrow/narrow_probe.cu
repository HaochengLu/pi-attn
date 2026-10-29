// Probe: GEMM for narrow N, long K, M <= 128 (decode / short prefill on column shards).
// CTA (x, y, z) = W warps x TN n8 tiles = 8 W TN output columns, segment y of S, rows
// [16 G z, 16 G (z + 1)). The warps share each X stage (cp.async ring of STAGES x BK); each
// warp keeps G x TN accumulators. Per segment the k order is gemm_small_kernel's; with S > 1
// the partial goes to ws and the last CTA of (n tile, m tile) to arrive folds the S partials
// into +0 in ascending order, as gemm_balanced_kernel does, so the bits are the engine's.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "gemm_dispatch.cuh"

using namespace pi_gemm;

template <int G, int TN, int W, int BK, int STAGES>
__global__ void __launch_bounds__(W * 32) narrow_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx,
                                                        const __nv_bfloat16* __restrict__ w,
                                                        __nv_bfloat16* __restrict__ y, int M, int N, int K,
                                                        int segs, float* ws, int* counters) {
  constexpr int kRows = 16 * G, kCols = 8 * TN * W, kStride = BK + 8, kStage = (kRows + kCols) * kStride;
  extern __shared__ __align__(16) __nv_bfloat16 smem[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n0 = blockIdx.x * kCols, seg = blockIdx.y, m0 = blockIdx.z * kRows;
  const int seg_k = K / segs, k0 = seg * seg_k, ktiles = seg_k / BK;
  auto issue = [&](int kt) {
    __nv_bfloat16* st = smem + (kt % STAGES) * kStage;
    constexpr int kVec = BK / 8;
    for (int i = threadIdx.x; i < (kRows + kCols) * kVec; i += W * 32) {
      const int r = i / kVec, v = i % kVec, kc = k0 + kt * BK + v * 8;
      if (r >= kRows) cp_async16(st + r * kStride + v * 8, w + (int64_t)(n0 + r - kRows) * K + kc, 16);
      else if (m0 + r < M) cp_async16(st + r * kStride + v * 8, x + (int64_t)(m0 + r) * ldx + kc, 16);
      else cp_async16(st + r * kStride + v * 8, x, 0);
    }
  };
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < ktiles) issue(s);
    cp_async_commit();
  }
  float acc[G][TN][4];
#pragma unroll
  for (int g = 0; g < G; ++g)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[g][j][0] = acc[g][j][1] = acc[g][j][2] = acc[g][j][3] = 0.f;
  const int mat = lane >> 3, r8 = lane & 7;
  for (int kt = 0; kt < ktiles; ++kt) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    if (kt + STAGES - 1 < ktiles) issue(kt + STAGES - 1);
    cp_async_commit();
    const __nv_bfloat16* sx = smem + (kt % STAGES) * kStage;
    const __nv_bfloat16* sw = sx + (kRows + warp * 8 * TN) * kStride;
#pragma unroll
    for (int ks = 0; ks < BK / 16; ks += 2) {
      uint32_t b[TN][4];
#pragma unroll
      for (int j = 0; j < TN; ++j) ldmatrix_x4(b[j], sw + (j * 8 + r8) * kStride + ks * 16 + mat * 8);
#pragma unroll
      for (int kk = 0; kk < 2; ++kk)
#pragma unroll
        for (int g = 0; g < G; ++g) {
          uint32_t a[4];
          ldmatrix_x4(a, sx + (g * 16 + (mat & 1) * 8 + r8) * kStride + (ks + kk) * 16 + (mat >> 1) * 8);
#pragma unroll
          for (int j = 0; j < TN; ++j) mma_bf16(acc[g][j], a, b[j][2 * kk], b[j][2 * kk + 1]);
        }
    }
  }
  cp_async_wait<0>();
  const int gr = lane >> 2, t = lane & 3;
  if (segs > 1) {
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
      for (int j = 0; j < TN; ++j)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int m = m0 + g * 16 + gr + h * 8, n = n0 + (warp * TN + j) * 8 + t * 2;
          if (m < M) *reinterpret_cast<float2*>(ws + ((int64_t)seg * M + m) * N + n) = make_float2(acc[g][j][2 * h], acc[g][j][2 * h + 1]);
        }
    __threadfence();
    __syncthreads();
    int last = 0;
    if (threadIdx.x == 0) {
      const int c = blockIdx.x + gridDim.x * blockIdx.z;
      last = atomicAdd(counters + c, 1) == segs - 1;
      if (last) counters[c] = 0;
    }
    if (!__syncthreads_or(last)) return;
    __threadfence();
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
      for (int j = 0; j < TN; ++j)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int m = m0 + g * 16 + gr + h * 8, n = n0 + (warp * TN + j) * 8 + t * 2;
          if (m >= M) continue;
          float2 tot = make_float2(0.f, 0.f);
          for (int q = 0; q < segs; ++q) {
            const float2 p = __ldcg(reinterpret_cast<const float2*>(ws + ((int64_t)q * M + m) * N + n));
            tot.x = add(tot.x, p.x);
            tot.y = add(tot.y, p.y);
          }
          acc[g][j][2 * h] = tot.x;
          acc[g][j][2 * h + 1] = tot.y;
        }
  }
#pragma unroll
  for (int g = 0; g < G; ++g)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int m = m0 + g * 16 + gr + h * 8, n = n0 + (warp * TN + j) * 8 + t * 2;
        if (m < M) *reinterpret_cast<__nv_bfloat162*>(y + (int64_t)m * N + n) = __floats2bfloat162_rn(acc[g][j][2 * h], acc[g][j][2 * h + 1]);
      }
}

static float* g_ws = nullptr;
static int* g_cnt = nullptr;

template <int G, int TN, int W, int BK, int STAGES>
bool launch(at::Tensor x, at::Tensor w, at::Tensor y, int segs) {
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  constexpr int kCols = 8 * TN * W;
  if (N % kCols || K % segs || (K / segs) % BK || M > 16 * G * 65535) return false;
  if (!g_ws) {
    cudaMalloc(&g_ws, (size_t)8 * 128 * 16384 * 4);
    cudaMalloc(&g_cnt, 1 << 20);
    cudaMemset(g_cnt, 0, 1 << 20);
  }
  if ((int64_t)segs * M * N > (int64_t)8 * 128 * 16384) return false;
  const size_t smem = (size_t)STAGES * (16 * G + kCols) * (BK + 8) * 2;
  if (smem > 160 * 1024) return false;
  cudaFuncSetAttribute(narrow_kernel<G, TN, W, BK, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  dim3 grid(N / kCols, segs, (M + 16 * G - 1) / (16 * G));
  narrow_kernel<G, TN, W, BK, STAGES><<<grid, W * 32, smem, at::cuda::getCurrentCUDAStream()>>>(
      (const __nv_bfloat16*)x.data_ptr(), x.stride(0), (const __nv_bfloat16*)w.data_ptr(),
      (__nv_bfloat16*)y.data_ptr(), M, N, K, segs, g_ws, g_cnt);
  return true;
}

// Configs: (G, TN, W, BK, STAGES)
#define CONFIGS(X)                                                                                   \
  X(1, 1, 2, 128, 6) X(1, 1, 4, 128, 6) X(1, 2, 2, 128, 6) X(1, 1, 2, 256, 4) X(1, 2, 1, 256, 4)       \
  X(2, 1, 2, 128, 5) X(2, 1, 4, 128, 5) X(2, 2, 2, 128, 5) X(2, 1, 2, 256, 3) X(2, 2, 1, 256, 4)       \
  X(4, 1, 2, 128, 4) X(4, 1, 4, 128, 4) X(4, 2, 2, 128, 4) X(4, 1, 2, 64, 6) X(4, 2, 1, 128, 5)        \
  X(8, 1, 2, 64, 4) X(8, 1, 4, 64, 4) X(8, 2, 2, 64, 4) X(8, 1, 2, 128, 3) X(8, 2, 1, 64, 5)

std::vector<std::string> configs() {
  std::vector<std::string> v;
#define NAME(G, TN, W, BK, ST) v.push_back(#G "," #TN "," #W "," #BK "," #ST);
  CONFIGS(NAME)
  return v;
}

c10::optional<at::Tensor> narrow(at::Tensor x, at::Tensor w, int64_t cfg, int64_t segs) {
  auto y = at::empty({x.size(0), w.size(0)}, x.options());
  int i = 0;
  bool ok = false, hit = false;
#define RUN(G, TN, W, BK, ST) \
  if (i++ == cfg) { hit = true; ok = launch<G, TN, W, BK, ST>(x, w, y, (int)segs); }
  CONFIGS(RUN)
  TORCH_CHECK(hit, "no such config");
  if (!ok) return c10::nullopt;
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("narrow", &narrow);
  m.def("configs", &configs);
}
