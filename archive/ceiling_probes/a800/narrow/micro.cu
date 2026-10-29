// Micro-benchmarks for the narrow-N GEMM floors on this GPU.
// mma_latency: one warp, `steps` dependent mma.m16n8k16 (chains independent accumulators interleaved).
// stream: every warp streams its own 8 rows x len bf16 region (rows `ld` apart, as an n8 weight
// tile over K) through a cp.async ring of ST stages x BK columns; with `mma` the warp also runs the
// dependent MMA chain over each stage (A fragment from a fixed 16-row smem tile), as a GEMM chain.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "gemm_dispatch.cuh"

using namespace pi_gemm;

__global__ void mma_latency_kernel(float* out, int steps, int chains, long long* cycles) {
  uint32_t a[4] = {0x3f803f80u, 0x3f803f80u, 0x3f803f80u, 0x3f803f80u}, b0 = 0x3f803f80u, b1 = 0x3f803f80u;
  float acc[4][4] = {};
  const long long t0 = clock64();
  for (int i = 0; i < steps; ++i) {
#pragma unroll
    for (int c = 0; c < 4; ++c)
      if (c < chains) mma_bf16(acc[c], a, b0, b1);
  }
  const long long t1 = clock64();
  float s = 0.f;
  for (int c = 0; c < 4; ++c) s += acc[c][0] + acc[c][1] + acc[c][2] + acc[c][3];
  out[threadIdx.x] = s;
  if (threadIdx.x == 0) *cycles = t1 - t0;
}

template <int BK, int ST, bool MMA>
__global__ void stream_kernel(const __nv_bfloat16* __restrict__ w, int64_t ld, int len, float* out) {
  constexpr int kStride = BK + 8, kStage = 8 * kStride;
  extern __shared__ __align__(16) __nv_bfloat16 smem[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int64_t wid = (int64_t)blockIdx.x * (blockDim.x >> 5) + warp;
  __nv_bfloat16* ring = smem + 16 * kStride + warp * ST * kStage;  // [0, 16 rows): a fixed A tile
  const __nv_bfloat16* src = w + wid * 8 * ld;
  const int kt_n = len / BK;
  auto issue = [&](int kt) {
    __nv_bfloat16* st = ring + (kt % ST) * kStage;
    constexpr int kVec = BK / 8;
#pragma unroll
    for (int i = lane; i < 8 * kVec; i += 32) cp_async16(st + (i / kVec) * kStride + (i % kVec) * 8, src + (int64_t)(i / kVec) * ld + kt * BK + (i % kVec) * 8, 16);
  };
  for (int i = threadIdx.x; i < 16 * kStride; i += blockDim.x) smem[i] = __float2bfloat16(0.01f);
  __syncthreads();
#pragma unroll
  for (int s = 0; s < ST - 1; ++s) {
    if (s < kt_n) issue(s);
    cp_async_commit();
  }
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  const int mat = lane >> 3, r8 = lane & 7;
  for (int kt = 0; kt < kt_n; ++kt) {
    cp_async_wait<ST - 2>();
    __syncwarp();
    if (kt + ST - 1 < kt_n) issue(kt + ST - 1);
    cp_async_commit();
    const __nv_bfloat16* sw = ring + (kt % ST) * kStage;
    if constexpr (MMA) {
#pragma unroll
      for (int ks = 0; ks < BK / 16; ks += 2) {
        uint32_t b[4], a[4];
        ldmatrix_x4(b, sw + r8 * kStride + ks * 16 + mat * 8);
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
          ldmatrix_x4(a, smem + ((mat & 1) * 8 + r8) * kStride + (ks + kk) * 16 + (mat >> 1) * 8);
          mma_bf16(acc, a, b[2 * kk], b[2 * kk + 1]);
        }
      }
    } else {
      acc[0] += __bfloat162float(sw[lane]);
    }
    __syncwarp();
  }
  cp_async_wait<0>();
  out[wid * 32 + lane] = acc[0] + acc[1] + acc[2] + acc[3];
}

double mma_latency(int64_t steps, int64_t chains) {
  auto out = torch::empty({32}, torch::dtype(torch::kFloat).device(torch::kCUDA));
  auto cyc = torch::empty({1}, torch::dtype(torch::kLong).device(torch::kCUDA));
  mma_latency_kernel<<<1, 32>>>(out.data_ptr<float>(), (int)steps, (int)chains, (long long*)cyc.data_ptr());
  cudaDeviceSynchronize();
  return (double)cyc.item<int64_t>() / steps;
}

#define STREAM_CONFIGS(X) X(128, 4) X(128, 8) X(256, 3) X(256, 4) X(256, 6) X(512, 2) X(512, 3) X(512, 4) X(1024, 2) X(1024, 3)

std::vector<std::string> stream_configs() {
  std::vector<std::string> v;
#define SN(BK, ST) v.push_back(#BK "x" #ST);
  STREAM_CONFIGS(SN)
  return v;
}

// w: [warps_total * 8, ld]; streams `len` columns per row. Returns false if the config does not fit.
bool stream(at::Tensor w, int64_t len, int64_t cfg, int64_t warps_per_cta, bool mma) {
  const int64_t rows = w.size(0), ld = w.size(1);
  const int warps_total = rows / 8, ctas = warps_total / warps_per_cta;
  auto out = torch::empty({warps_total * 32}, torch::dtype(torch::kFloat).device(torch::kCUDA));
  int i = 0;
  bool ok = false;
#define SR(BK, ST)                                                                                           \
  if (i++ == cfg) {                                                                                          \
    const size_t smem = ((size_t)16 * (BK + 8) + (size_t)warps_per_cta * ST * 8 * (BK + 8)) * 2;             \
    if (smem <= 163 * 1024 && len % BK == 0) {                                                               \
      auto k = mma ? stream_kernel<BK, ST, true> : stream_kernel<BK, ST, false>;                            \
      cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);                       \
      k<<<ctas, warps_per_cta * 32, smem, at::cuda::getCurrentCUDAStream()>>>(                               \
          (const __nv_bfloat16*)w.data_ptr(), ld, (int)len, out.data_ptr<float>());                          \
      ok = true;                                                                                             \
    }                                                                                                        \
  }
  STREAM_CONFIGS(SR)
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return ok;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mma_latency", &mma_latency);
  m.def("stream", &stream);
  m.def("stream_configs", &stream_configs);
}
