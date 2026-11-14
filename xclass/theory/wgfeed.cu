// wgfeed.cu -- which smem feed protocol makes wgmma (operands written with st.shared) reliable on H100?
// Native chained GEMM (X [tiles][T][64][16], W [tiles][T][8][16]) on mma.sync vs wgmma with several protocols.
//   nvcc -O2 -std=c++17 -arch=sm_90a wgfeed.cu -o wgfeed && ./wgfeed
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "xcdata.h"
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(2); } } while (0)
__device__ __forceinline__ uint32_t pk(uint16_t lo, uint16_t hi) { return (uint32_t)lo | ((uint32_t)hi << 16); }
__global__ void k_mma(const uint16_t* X, const uint16_t* W, float* out, int T) {
  const int tile = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, k0 = (lane & 3) * 2, r0 = warp * 16 + g, r1 = r0 + 8;
  float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
  for (int t = 0; t < T; ++t) {
    const uint16_t *x = X + ((size_t)tile * T + t) * 1024, *w = W + ((size_t)tile * T + t) * 128;
    const uint32_t a0 = pk(x[r0 * 16 + k0], x[r0 * 16 + k0 + 1]), a1 = pk(x[r1 * 16 + k0], x[r1 * 16 + k0 + 1]), a2 = pk(x[r0 * 16 + k0 + 8], x[r0 * 16 + k0 + 9]),
                   a3 = pk(x[r1 * 16 + k0 + 8], x[r1 * 16 + k0 + 9]), b0 = pk(w[g * 16 + k0], w[g * 16 + k0 + 1]), b1 = pk(w[g * 16 + k0 + 8], w[g * 16 + k0 + 9]);
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  }
  float* o = out + (size_t)tile * 512;
  o[r0 * 8 + k0] = d0; o[r0 * 8 + k0 + 1] = d1; o[r1 * 8 + k0] = d2; o[r1 * 8 + k0 + 1] = d3;
}
__device__ __forceinline__ int smem_off(int r, int k) { return (r >> 3) * 256 + (k >> 3) * 128 + (r & 7) * 16 + (k & 7) * 2; }
__device__ __forceinline__ uint64_t make_desc(const void* p) {
  const uint64_t addr = (uint64_t)__cvta_generic_to_shared(p);
  return ((addr >> 4) & 0x3FFF) | ((uint64_t)(128 >> 4) << 16) | ((uint64_t)(256 >> 4) << 32);
}
// FEED (compile-time): 0 st; bar; fence          1 st; bar; bar; fence        2 st; membar.cta; bar; fence
//                      3 st; fence; bar; fence   4 st(B first); bar; fence    5 st; bar; readback(ld.shared) ; fence
//                      8 cp.async.ca 16-byte chunks; cp.async.wait_group 0; bar; fence
//                      6 st; bar; fence; ~64 dependent integer ops delay; (wgmma)   7 double buffering: step t uses buffer t & 1
template <int FEED> __global__ void __launch_bounds__(128) k_wg(const uint16_t* X, const uint16_t* W, float* out, int T, unsigned* sink) {
  __shared__ __align__(128) unsigned char sA[2][2048];
  __shared__ __align__(128) unsigned char sB[2][2048];
  const int tile = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, k0 = (lane & 3) * 2, r0 = warp * 16 + g, r1 = r0 + 8;
  float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f; unsigned junk = 0;
  for (int t = 0; t < T; ++t) {
    const int buf = FEED == 7 ? (t & 1) : 0;
    const uint64_t da = make_desc(sA[buf]), db = make_desc(sB[buf]);
    const uint16_t *x = X + ((size_t)tile * T + t) * 1024, *w = W + ((size_t)tile * T + t) * 128;
    if (FEED == 8) {   // cp.async feed: one 16-byte chunk (8 bf16 along k) per thread for A, threads 0..15 for B
      const int r = threadIdx.x >> 1, h = threadIdx.x & 1;
      asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"((unsigned)__cvta_generic_to_shared(sA[buf] + (r >> 3) * 256 + h * 128 + (r & 7) * 16)), "l"(x + r * 16 + h * 8) : "memory");
      if (threadIdx.x < 16) asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"((unsigned)__cvta_generic_to_shared(sB[buf] + h * 128 + r * 16)), "l"(w + r * 16 + h * 8) : "memory");
      asm volatile("cp.async.commit_group;\n" ::: "memory");
      asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    } else {
      if (FEED == 4) *(uint16_t*)(sB[buf] + smem_off(threadIdx.x >> 4, threadIdx.x & 15)) = w[threadIdx.x];
      for (int i = threadIdx.x; i < 1024; i += 128) *(uint16_t*)(sA[buf] + smem_off(i >> 4, i & 15)) = x[i];
      if (FEED != 4) *(uint16_t*)(sB[buf] + smem_off(threadIdx.x >> 4, threadIdx.x & 15)) = w[threadIdx.x];
    }
    if (FEED == 2) __threadfence_block();
    if (FEED == 3) asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    __syncthreads();
    if (FEED == 1) __syncthreads();
    if (FEED == 5) { junk += *(volatile uint16_t*)(sB[buf] + smem_off((threadIdx.x + 37) >> 4 & 7, threadIdx.x & 15)); junk += *(volatile uint16_t*)(sA[buf] + smem_off((threadIdx.x + 517) >> 4 & 63, threadIdx.x & 15)); }
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    if (FEED == 6) { volatile unsigned v = threadIdx.x; for (int q = 0; q < 16; ++q) v = v * 1664525u + 1013904223u; junk += v; }
    asm volatile("" : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) :: "memory");
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, p, 1, 1, 0, 0;\n}\n"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) : "l"(da), "l"(db), "r"(1));
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    asm volatile("" : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) :: "memory");
    __syncthreads();
  }
  float* o = out + (size_t)tile * 512;
  o[r0 * 8 + k0] = d0; o[r0 * 8 + k0 + 1] = d1; o[r1 * 8 + k0] = d2; o[r1 * 8 + k0 + 1] = d3;
  if (junk == 0xDEADBEEF) sink[0] = junk;
}
template <int FEED> static void run(const uint16_t* X, const uint16_t* W, float* o, int tiles, int T, unsigned* s) { k_wg<FEED><<<tiles, 128>>>(X, W, o, T, s); }
int main() {
  const int tiles = 512, T = 64;
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0)); printf("GPU %s sm_%d%d wgmma feed protocols, tiles %d T %d\n", p.name, p.major, p.minor, tiles, T);
  XData d = xc_make(0, tiles, T); const size_t nout = (size_t)tiles * 512;
  uint16_t *dX, *dW; float* dO; unsigned* dS;
  CK(cudaMalloc(&dX, d.X.size() * 2)); CK(cudaMalloc(&dW, d.W.size() * 2)); CK(cudaMalloc(&dO, nout * 4)); CK(cudaMalloc(&dS, 64));
  CK(cudaMemcpy(dX, d.X.data(), d.X.size() * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dW, d.W.data(), d.W.size() * 2, cudaMemcpyHostToDevice));
  std::vector<uint32_t> ref(nout), o(nout);
  k_mma<<<tiles, 128>>>(dX, dW, dO, T); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(ref.data(), dO, nout * 4, cudaMemcpyDeviceToHost));
  void (*runs[9])(const uint16_t*, const uint16_t*, float*, int, int, unsigned*) = {run<0>, run<1>, run<2>, run<3>, run<4>, run<5>, run<6>, run<7>, run<8>};
  const char* nm[9] = {"st; bar; fence", "st; bar; bar; fence", "st; membar.cta; bar; fence", "st; fence; bar; fence", "st (B first); bar; fence", "st; bar; ld.shared readback; fence",
                       "st; bar; fence; delay", "st into alternating buffers; bar; fence", "cp.async; wait_group; bar; fence"};
  for (int round = 0; round < 3; ++round)
    for (int f = (round ? 8 : 0); f < 9; f += 8) {
      size_t total = 0, tilesbad = 0; const int reps = 20;
      for (int rep = 0; rep < reps; ++rep) {
        CK(cudaMemset(dO, 0xFF, nout * 4)); runs[f](dX, dW, dO, tiles, T, dS); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(o.data(), dO, nout * 4, cudaMemcpyDeviceToHost));
        for (int t = 0; t < tiles; ++t) { size_t mm = 0; for (int i = 0; i < 512; ++i) mm += o[(size_t)t * 512 + i] != ref[(size_t)t * 512 + i]; total += mm; tilesbad += mm != 0; }
      }
      printf("FEED %d %-42s: %zu mismatching outputs, %zu bad tile-runs of %d (%d wgmma steps)\n", f, nm[f], total, tilesbad, reps * tiles, reps * tiles * T);
    }
  return 0;
}
