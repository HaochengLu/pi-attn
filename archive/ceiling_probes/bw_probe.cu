// PERF PROBE: achievable weight-streaming bandwidth per access pattern (no MMA).
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cstdint>

__device__ __forceinline__ void cp16(void* s, const void* g) {
  unsigned a = (unsigned)__cvta_generic_to_shared(s);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(a), "l"(g));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// A: linear ldg.128, grid-stride, UNROLL loads in flight per thread.
template <int UNROLL>
__global__ void lin_kernel(const uint4* __restrict__ w, int64_t n16, float* out) {
  uint32_t s = 0;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (; i + (UNROLL - 1) * stride < n16; i += UNROLL * stride) {
    uint4 v[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) v[u] = __ldcs(w + i + u * stride);
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) s ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  for (; i < n16; i += stride) { uint4 v = __ldcs(w + i); s ^= v.x ^ v.y ^ v.z ^ v.w; }
  if (s == 0x12345678u) out[0] = 1.f;
}

// B: balanced pattern. CTA owns contiguous rows [r0, r1) (multiple of 8, `rows` per pass),
// streams K in BK-column stages of all pass rows through a STAGES ring, __syncthreads per stage.
template <int BK, int STAGES>
__global__ void tile_kernel(const __nv_bfloat16* __restrict__ w, int N, int K, int rows_per_pass, float* out) {
  extern __shared__ __align__(16) __nv_bfloat16 sm[];
  const int stride = BK + 8, stage = rows_per_pass * stride;
  const int units = N / 8;
  const int u0 = (int)((int64_t)blockIdx.x * units / gridDim.x), u1 = (int)((int64_t)(blockIdx.x + 1) * units / gridDim.x);
  uint32_t s = 0;
  const int kt_n = K / BK;
  for (int u = u0; u < u1; u += rows_per_pass / 8) {
    const int nr = min(rows_per_pass, (u1 - u) * 8);
    auto issue = [&](int kt) {
      __nv_bfloat16* st = sm + (kt % STAGES) * stage;
      constexpr int kVec = BK / 8;
      for (int i = threadIdx.x; i < nr * kVec; i += blockDim.x) {
        const int r = i / kVec, v = i % kVec;
        cp16(st + r * stride + v * 8, w + ((int64_t)u * 8 + r) * K + kt * BK + v * 8);
      }
    };
    for (int q = 0; q < STAGES - 1; ++q) { if (q < kt_n) issue(q); commit(); }
    for (int kt = 0; kt < kt_n; ++kt) {
      wait<STAGES - 2>();
      __syncthreads();
      if (kt + STAGES - 1 < kt_n) issue(kt + STAGES - 1);
      commit();
      const __nv_bfloat16* st = sm + (kt % STAGES) * stage;
      s ^= *reinterpret_cast<const uint32_t*>(st + (threadIdx.x % nr) * stride + (threadIdx.x / nr) % (BK / 2) * 2);
    }
    wait<0>();
    __syncthreads();
  }
  if (s == 0x12345678u) out[0] = 1.f;
}

// C: per-warp ring, no CTA barriers; warp owns one n8 unit at a time.
template <int BK, int STAGES>
__global__ void warp_kernel(const __nv_bfloat16* __restrict__ w, int N, int K, float* out) {
  extern __shared__ __align__(16) __nv_bfloat16 sm[];
  const int stride = BK + 8, stage = 8 * stride;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
  __nv_bfloat16* ring = sm + warp * STAGES * stage;
  const int units = N / 8;
  const int u0 = (int)((int64_t)blockIdx.x * units / gridDim.x), u1 = (int)((int64_t)(blockIdx.x + 1) * units / gridDim.x);
  uint32_t s = 0;
  const int kt_n = K / BK;
  for (int u = u0 + warp; u < u1; u += warps) {
    auto issue = [&](int kt) {
      __nv_bfloat16* st = ring + (kt % STAGES) * stage;
      constexpr int kVec = BK / 8;
      for (int i = lane; i < 8 * kVec; i += 32) {
        const int r = i / kVec, v = i % kVec;
        cp16(st + r * stride + v * 8, w + ((int64_t)u * 8 + r) * K + kt * BK + v * 8);
      }
    };
    for (int q = 0; q < STAGES - 1; ++q) { if (q < kt_n) issue(q); commit(); }
    for (int kt = 0; kt < kt_n; ++kt) {
      wait<STAGES - 2>();
      __syncwarp();
      if (kt + STAGES - 1 < kt_n) issue(kt + STAGES - 1);
      commit();
      s ^= *reinterpret_cast<const uint32_t*>(ring + (kt % STAGES) * stage + (lane & 7) * stride + (lane >> 3) * 2);
    }
    wait<0>();
    __syncwarp();
  }
  if (s == 0x12345678u) out[0] = 1.f;
}

// D: register streaming, warp owns one n8 unit: lane l reads row l/4, 16B chunks at
// k = kt*128 + (l%4)*8 + j*32 (j < 4), i.e. each row 256B contiguous per step, DEPTH steps in flight.
template <int DEPTH>
__global__ void reg_kernel(const __nv_bfloat16* __restrict__ w, int N, int K, float* out) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
  const int units = N / 8;
  const int u0 = (int)((int64_t)blockIdx.x * units / gridDim.x), u1 = (int)((int64_t)(blockIdx.x + 1) * units / gridDim.x);
  uint32_t s = 0;
  for (int u = u0 + warp; u < u1; u += warps) {
    const uint4* row = reinterpret_cast<const uint4*>(w + ((int64_t)u * 8 + lane / 4) * K) + (lane % 4);
    const int steps = K / 128;  // 128 bf16 = 16 uint4 per row per step; lane takes 4 of them
    for (int kt = 0; kt < steps; kt += DEPTH) {
      uint4 v[DEPTH][4];
#pragma unroll
      for (int d = 0; d < DEPTH; ++d)
#pragma unroll
        for (int j = 0; j < 4; ++j) v[d][j] = (kt + d < steps) ? __ldcs(row + (kt + d) * 16 + j * 4) : make_uint4(0, 0, 0, 0);
#pragma unroll
      for (int d = 0; d < DEPTH; ++d)
#pragma unroll
        for (int j = 0; j < 4; ++j) s ^= v[d][j].x ^ v[d][j].w;
    }
  }
  if (s == 0x12345678u) out[0] = 1.f;
}

static float* g_out() {
  static torch::Tensor t;
  if (!t.defined()) t = torch::zeros({1}, torch::dtype(torch::kFloat).device(torch::kCUDA));
  return t.data_ptr<float>();
}

void run(torch::Tensor w, int64_t kind, int64_t a, int64_t b, int64_t c) {
  auto st = at::cuda::getCurrentCUDAStream();
  const int N = w.size(0), K = w.size(1);
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  const auto* wp = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
  if (kind == 0) {  // a = blocks per SM, b = threads, c = unroll
    const int64_t n16 = w.numel() * 2 / 16;
    if (c == 4) lin_kernel<4><<<sms * a, b, 0, st>>>((const uint4*)wp, n16, g_out());
    else lin_kernel<8><<<sms * a, b, 0, st>>>((const uint4*)wp, n16, g_out());
  } else if (kind == 1) {  // a = BK, b = stages, c = rows per pass ; threads 128
#define T(BK, S)                                                                                  \
  if (a == BK && b == S) {                                                                        \
    size_t smem = (size_t)S * c * (BK + 8) * 2;                                                   \
    cudaFuncSetAttribute(tile_kernel<BK, S>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); \
    tile_kernel<BK, S><<<sms, 128, smem, st>>>(wp, N, K, (int)c, g_out());                        \
  }
    T(64, 4) T(64, 6) T(128, 4) T(128, 6) T(256, 3) T(256, 4) T(512, 2) T(512, 3)
#undef T
  } else if (kind == 2) {  // a = BK, b = stages, c = warps per CTA ; one CTA per SM
#define T(BK, S)                                                                                  \
  if (a == BK && b == S) {                                                                        \
    size_t smem = (size_t)c * S * 8 * (BK + 8) * 2;                                               \
    cudaFuncSetAttribute(warp_kernel<BK, S>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); \
    warp_kernel<BK, S><<<sms, 32 * c, smem, st>>>(wp, N, K, g_out());                             \
  }
    T(64, 4) T(64, 8) T(128, 4) T(128, 6) T(256, 3) T(256, 4) T(512, 2) T(512, 3)
#undef T
  } else {  // a = depth, c = warps per CTA, b = CTAs per SM
    const int grid = sms * b;
    if (a == 1) reg_kernel<1><<<grid, 32 * c, 0, st>>>(wp, N, K, g_out());
    if (a == 2) reg_kernel<2><<<grid, 32 * c, 0, st>>>(wp, N, K, g_out());
    if (a == 4) reg_kernel<4><<<grid, 32 * c, 0, st>>>(wp, N, K, g_out());
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
