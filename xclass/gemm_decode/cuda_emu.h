// cuda_emu.h -- functional CPU emulation of the CUDA subset used by xcp_kernel.cuh (one OS thread per GPU thread).
// Purpose: run the real kernel source against the exact CPU reference before any GPU time is spent. The tensor
// core is modelled as an EXACT fused sum (double); a result that is not an fp32 value is counted as a violation of
// Theorem T1 (emu::t1_violations) -- so the emulation also checks exactness on the kernel's own operand flow.
#pragma once
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __restrict__
#define __launch_bounds__(x)

struct uint2 { uint32_t x, y; };
struct uint4 { uint32_t x, y, z, w; };
struct float2 { float x, y; };
static inline uint2 make_uint2(uint32_t x, uint32_t y) { return {x, y}; }
static inline uint4 make_uint4(uint32_t x, uint32_t y, uint32_t z, uint32_t w) { return {x, y, z, w}; }
static inline float2 make_float2(float x, float y) { return {x, y}; }
struct __nv_bfloat16 { uint16_t b; };
struct __nv_bfloat162 { uint16_t x, y; };
static inline uint16_t emu_f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1)) >> 16); }
static inline __nv_bfloat162 __floats2bfloat162_rn(float a, float b) { return {emu_f2bf(a), emu_f2bf(b)}; }

typedef int cudaError_t;
typedef void* cudaStream_t;
enum { cudaSuccess = 0, cudaErrorInvalidValue = 1 };
enum { cudaDevAttrMultiProcessorCount, cudaDevAttrMaxSharedMemoryPerBlockOptin, cudaFuncAttributeMaxDynamicSharedMemorySize };

namespace emu {
struct Barrier {
  std::mutex m; std::condition_variable cv; int n, waiting = 0; long gen = 0;
  explicit Barrier(int n_) : n(n_) {}
  void wait() {
    std::unique_lock<std::mutex> l(m);
    const long g = gen;
    if (++waiting == n) { waiting = 0; ++gen; cv.notify_all(); }
    else cv.wait(l, [&] { return gen != g; });
  }
};
struct Warp {
  Barrier b{32};
  uint32_t sh[32];
  uint32_t a[32][4], b0[32], b1[32];
  float d[32][4];
};
struct Cta {
  std::vector<unsigned char> smem; Barrier* bar; std::vector<Warp>* warps;
};
struct Idx { unsigned x = 0, y = 0, z = 0; };
inline int g_sms = 5, g_smem_max = 99 * 1024;
inline std::atomic<long> t1_violations{0};
inline thread_local Cta* tl_cta = nullptr;
inline thread_local Warp* tl_warp = nullptr;
template <class F> void launch(int grid, int block, size_t smem, F&& f);
}  // namespace emu
inline thread_local emu::Idx threadIdx, blockIdx, gridDim, blockDim;

namespace emu {
template <class F> void launch(int grid, int block, size_t smem, F&& f) {
  std::vector<std::thread> th;
  std::vector<Cta> ctas(grid);
  std::vector<std::vector<Warp>> warps(grid);
  std::vector<Barrier*> bars(grid);
  for (int c = 0; c < grid; ++c) {
    ctas[c].smem.assign(smem + 64, 0xAB);
    bars[c] = new Barrier(block);
    warps[c] = std::vector<Warp>(block / 32);
    ctas[c].bar = bars[c]; ctas[c].warps = &warps[c];
  }
  for (int c = 0; c < grid; ++c)
    for (int t = 0; t < block; ++t)
      th.emplace_back([&, c, t] {
        threadIdx.x = t; blockIdx.x = c; gridDim.x = grid; blockDim.x = block;
        tl_cta = &ctas[c]; tl_warp = &warps[c][t / 32];
        f();
      });
  for (auto& t : th) t.join();
  for (auto b : bars) delete b;
}
inline void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  Warp& W = *tl_warp; const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  for (int i = 0; i < 4; ++i) { W.a[lane][i] = a[i]; W.d[lane][i] = d[i]; }
  W.b0[lane] = b0; W.b1[lane] = b1;
  W.b.wait();
  auto bf = [](uint32_t pair, int hi) { const uint32_t u = (hi ? (pair >> 16) : (pair & 0xFFFF)) << 16; float f; memcpy(&f, &u, 4); return (double)f; };
  auto A = [&](int row, int k) { const int l = (row & 7) * 4 + ((k & 7) >> 1); return bf(W.a[l][(row >> 3) + 2 * (k >> 3)], k & 1); };
  auto B = [&](int k, int n) { const int l = n * 4 + ((k & 7) >> 1); return bf(k < 8 ? W.b0[l] : W.b1[l], k & 1); };
  float out[4];
  for (int i = 0; i < 4; ++i) {
    const int row = g + 8 * (i >> 1), col = 2 * t + (i & 1);
    double s = W.d[lane][i];
    for (int k = 0; k < 16; ++k) s += A(row, k) * B(k, col);
    out[i] = (float)s;
    if ((double)out[i] != s) t1_violations++;
  }
  W.b.wait();
  for (int i = 0; i < 4; ++i) d[i] = out[i];
}
}  // namespace emu

static inline float __uint_as_float(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static inline uint32_t __float_as_uint(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static inline float __fadd_rn(float a, float b) { volatile float r = a + b; return r; }
static inline float __fsub_rn(float a, float b) { volatile float r = a - b; return r; }
static inline uint32_t __byte_perm(uint32_t x, uint32_t y, uint32_t s) {
  const uint64_t v = (uint64_t)x | ((uint64_t)y << 32); uint32_t r = 0;
  for (int i = 0; i < 4; ++i) r |= (uint32_t)((v >> (8 * ((s >> (4 * i)) & 7))) & 0xFF) << (8 * i);
  return r;
}
static inline float __shfl_xor_sync(uint32_t, float v, int mask) {
  emu::Warp& W = *emu::tl_warp; const int lane = threadIdx.x & 31;
  memcpy(&W.sh[lane], &v, 4);
  W.b.wait();
  float r; memcpy(&r, &W.sh[lane ^ mask], 4);
  W.b.wait();
  return r;
}
static inline void __syncthreads() { emu::tl_cta->bar->wait(); }
static inline void __syncwarp() {}
static inline void __threadfence() {}
static inline int atomicExch(int* p, int v) { return __atomic_exchange_n(p, v, __ATOMIC_SEQ_CST); }
static inline int atomicAdd(int* p, int v) { return __atomic_fetch_add(p, v, __ATOMIC_SEQ_CST); }
template <class T> static inline T __ldcg(const T* p) { return *p; }
using std::min;

static inline cudaError_t cudaDeviceGetAttribute(int* v, int attr, int) { *v = attr == cudaDevAttrMultiProcessorCount ? emu::g_sms : emu::g_smem_max; return cudaSuccess; }
template <class T> static inline cudaError_t cudaMalloc(T** p, size_t bytes) { *p = (T*)malloc(bytes); return cudaSuccess; }
static inline cudaError_t cudaMemset(void* p, int v, size_t bytes) { memset(p, v, bytes); return cudaSuccess; }
template <class K> static inline cudaError_t cudaFuncSetAttribute(K, int, int) { return cudaSuccess; }
static inline cudaError_t cudaGetLastError() { return cudaSuccess; }
