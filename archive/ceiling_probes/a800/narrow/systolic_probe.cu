// Probe: systolic relay GEMM for narrow N, long K, M <= 64.
// Each K segment is cut into P slices of `slice` columns; CTA (q, s, j) owns slice j of segment s
// and output column group q (Nq columns). Its X slice (16 G rows x slice) is loaded into shared
// memory once. Warp w handles n8 blocks b = w, w + NW, ... of the group in order; for each block it
// waits for stage j - 1's fp32 accumulators (relay), runs the MMA chain over its slice (ascending
// k16, W streamed through a per-warp cp.async ring that runs across blocks), and hands the
// accumulators to stage j + 1, or, at the last stage, writes the segment's partial. Every element's
// chain is the canonical one (ascending k16 per segment, segments left-folded from +0 by the last
// arrival), so the bits are the engine's. All CTAs must be resident (grid <= SMs): stages wait on
// lower stages only, which never wait on higher ones.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "gemm_dispatch.cuh"

using namespace pi_gemm;

template <int G, int NW, int BK, int ST>
__global__ void __launch_bounds__(NW * 32, 1) systolic_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx,
                                                             const __nv_bfloat16* __restrict__ w,
                                                             __nv_bfloat16* __restrict__ y, int M, int N, int K,
                                                             int segs, int slice, int P, int Nq, float* hand,
                                                             int* flags, float* ws, int* counters) {
  constexpr int kRows = 16 * G, kWStride = BK + 8, kWStage = 8 * kWStride;
  extern __shared__ __align__(16) __nv_bfloat16 smem[];
  const int xstride = slice + 8;
  __nv_bfloat16* sx = smem;
  __nv_bfloat16* ring = smem + kRows * xstride + (threadIdx.x >> 5) * ST * kWStage;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int j = blockIdx.x % P, s = (blockIdx.x / P) % segs, q = blockIdx.x / (P * segs);
  const int kseg = K / segs, kbase = s * kseg + j * slice, n0 = q * Nq, blocks = Nq / 8;
  const int kt_n = slice / BK;
  // X slice, once.
  for (int i = threadIdx.x; i < kRows * (slice / 8); i += NW * 32) {
    const int r = i / (slice / 8), v = i % (slice / 8);
    if (r < M) cp_async16(sx + r * xstride + v * 8, x + (int64_t)r * ldx + kbase + v * 8, 16);
    else cp_async16(sx + r * xstride + v * 8, x, 0);
  }
  cp_async_commit();
  // This warp's (block, k tile) steps, W streamed through its ring across blocks.
  const int my_blocks = warp < blocks ? (blocks - warp + NW - 1) / NW : 0, steps = my_blocks * kt_n;
  auto issue = [&](int u) {
    const int b = warp + (u / kt_n) * NW, kt = u % kt_n;
    __nv_bfloat16* st = ring + (u % ST) * kWStage;
    constexpr int kVec = BK / 8;
#pragma unroll
    for (int i = lane; i < 8 * kVec; i += 32) {
      const int r = i / kVec, v = i % kVec;
      cp_async16(st + r * kWStride + v * 8, w + (int64_t)(n0 + b * 8 + r) * K + kbase + kt * BK + v * 8, 16);
    }
  };
#pragma unroll
  for (int u = 0; u < ST - 1; ++u) {
    if (u < steps) issue(u);
    cp_async_commit();
  }
  cp_async_wait<0>();  // the X slice (and the first W stages)
  __syncthreads();
  const int mat = lane >> 3, r8 = lane & 7, gr = lane >> 2, t = lane & 3;
  float acc[G][4];
  for (int u = 0; u < steps; ++u) {
    const int bi = u / kt_n, b = warp + bi * NW, kt = u % kt_n;
    if (kt == 0) {
      if (j == 0) {
#pragma unroll
        for (int g = 0; g < G; ++g) acc[g][0] = acc[g][1] = acc[g][2] = acc[g][3] = 0.f;
      } else {
        const int64_t slot = (((int64_t)q * segs + s) * P + (j - 1)) * blocks + b;
        if (lane == 0) {
          while (atomicAdd(flags + slot, 0) == 0) {}
          flags[slot] = 0;  // ready for the next launch
        }
        __syncwarp();
        __threadfence();
        const float* src = hand + slot * (G * 128);
#pragma unroll
        for (int g = 0; g < G; ++g)
#pragma unroll
          for (int i = 0; i < 4; ++i) acc[g][i] = __ldcg(src + g * 128 + lane * 4 + i);
      }
    }
    cp_async_wait<ST - 2>();
    __syncwarp();
    if (u + ST - 1 < steps) issue(u + ST - 1);
    cp_async_commit();
    const __nv_bfloat16* sw = ring + (u % ST) * kWStage;
#pragma unroll
    for (int ks = 0; ks < BK / 16; ks += 2) {
      uint32_t bf[4];
      ldmatrix_x4(bf, sw + r8 * kWStride + ks * 16 + mat * 8);
#pragma unroll
      for (int kk = 0; kk < 2; ++kk)
#pragma unroll
        for (int g = 0; g < G; ++g) {
          uint32_t a[4];
          ldmatrix_x4(a, sx + (g * 16 + (mat & 1) * 8 + r8) * xstride + kt * BK + (ks + kk) * 16 + (mat >> 1) * 8);
          mma_bf16(acc[g], a, bf[2 * kk], bf[2 * kk + 1]);
        }
    }
    __syncwarp();
    if (kt == kt_n - 1) {
      if (j < P - 1) {
        const int64_t slot = (((int64_t)q * segs + s) * P + j) * blocks + b;
        float* dst = hand + slot * (G * 128);
#pragma unroll
        for (int g = 0; g < G; ++g)
#pragma unroll
          for (int i = 0; i < 4; ++i) __stcg(dst + g * 128 + lane * 4 + i, acc[g][i]);
        __threadfence();
        __syncwarp();
        if (lane == 0) atomicExch(flags + slot, 1);
      } else {
        const int n = n0 + b * 8 + t * 2;
        if (segs > 1) {
#pragma unroll
          for (int g = 0; g < G; ++g)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const int m = g * 16 + gr + h * 8;
              if (m < M) *reinterpret_cast<float2*>(ws + ((int64_t)s * M + m) * N + n) = make_float2(acc[g][2 * h], acc[g][2 * h + 1]);
            }
          __threadfence();
          __syncwarp();
          int last = 0;
          if (lane == 0) {
            const int c = n / 8;
            last = atomicAdd(counters + c, 1) == segs - 1;
            if (last) counters[c] = 0;
          }
          if (!__shfl_sync(0xffffffffu, last, 0)) continue;
          __threadfence();
#pragma unroll
          for (int g = 0; g < G; ++g)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const int m = g * 16 + gr + h * 8;
              if (m >= M) continue;
              float2 tot = make_float2(0.f, 0.f);
              for (int qq = 0; qq < segs; ++qq) {
                const float2 p = __ldcg(reinterpret_cast<const float2*>(ws + ((int64_t)qq * M + m) * N + n));
                tot.x = add(tot.x, p.x);
                tot.y = add(tot.y, p.y);
              }
              acc[g][2 * h] = tot.x;
              acc[g][2 * h + 1] = tot.y;
            }
        }
#pragma unroll
        for (int g = 0; g < G; ++g)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int m = g * 16 + gr + h * 8;
            if (m < M) *reinterpret_cast<__nv_bfloat162*>(y + (int64_t)m * N + n) = __floats2bfloat162_rn(acc[g][2 * h], acc[g][2 * h + 1]);
          }
      }
    }
  }
  cp_async_wait<0>();
}

static float *g_hand = nullptr, *g_ws = nullptr;
static int *g_flags = nullptr, *g_cnt = nullptr;
constexpr size_t kHandFloats = (size_t)32 << 20, kFlags = 1 << 20;

template <int G, int NW, int BK, int ST>
bool launch(at::Tensor x, at::Tensor w, at::Tensor y, int segs, int slice, int Q, int sms) {
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  if (M > 16 * G || K % segs || (K / segs) % slice || slice % BK || N % (Q * 8)) return false;
  const int P = K / segs / slice, Nq = N / Q, blocks = Nq / 8, ctas = Q * segs * P;
  if (ctas > sms || blocks < 1) return false;
  if (!g_hand) {
    cudaMalloc(&g_hand, kHandFloats * 4);
    cudaMalloc(&g_flags, kFlags * 4);
    cudaMemset(g_flags, 0, kFlags * 4);
    cudaMalloc(&g_ws, (size_t)8 * 64 * 16384 * 4);
    cudaMalloc(&g_cnt, 1 << 20);
    cudaMemset(g_cnt, 0, 1 << 20);
  }
  if ((size_t)ctas * blocks * G * 128 > kHandFloats || (size_t)ctas * blocks > kFlags) return false;
  const size_t smem = ((size_t)16 * G * (slice + 8) + (size_t)NW * ST * 8 * (BK + 8)) * 2;
  if (smem > 163 * 1024) return false;
  cudaFuncSetAttribute(systolic_kernel<G, NW, BK, ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  systolic_kernel<G, NW, BK, ST><<<ctas, NW * 32, smem, at::cuda::getCurrentCUDAStream()>>>(
      (const __nv_bfloat16*)x.data_ptr(), x.stride(0), (const __nv_bfloat16*)w.data_ptr(),
      (__nv_bfloat16*)y.data_ptr(), M, N, K, segs, slice, P, Nq, g_hand, g_flags, g_ws, g_cnt);
  return true;
}

#define CONFIGS(X)                                                                                  \
  X(1, 8, 128, 4) X(1, 16, 128, 4) X(1, 8, 256, 3) X(2, 8, 128, 4) X(2, 16, 128, 3) X(4, 8, 128, 4)  \
  X(4, 16, 64, 4) X(4, 8, 64, 6)

std::vector<std::string> configs() {
  std::vector<std::string> v;
#define NAME(G, NW, BK, ST) v.push_back(#G "," #NW "," #BK "," #ST);
  CONFIGS(NAME)
  return v;
}

c10::optional<at::Tensor> systolic(at::Tensor x, at::Tensor w, int64_t cfg, int64_t segs, int64_t slice, int64_t Q) {
  auto y = at::empty({x.size(0), w.size(0)}, x.options());
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, x.get_device());
  int i = 0;
  bool ok = false, hit = false;
#define RUN(G, NW, BK, ST) \
  if (i++ == cfg) { hit = true; ok = launch<G, NW, BK, ST>(x, w, y, (int)segs, (int)slice, (int)Q, sms); }
  CONFIGS(RUN)
  TORCH_CHECK(hit, "no such config");
  if (!ok) return c10::nullopt;
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("systolic", &systolic);
  m.def("configs", &configs);
}
