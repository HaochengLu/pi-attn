// HMMA m16n8k16 bf16->f32 on one SM: latency (one accumulator chain) and throughput (NACC independent chains,
// round-robin) by warps per SM. Prints cycles per HMMA per warp and per SM at the measured SM clock.
// build: nvcc -O3 -arch=sm_80 mma_rate.cu -o mma_rate ; run: ./mma_rate <clock MHz>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
template <int NACC, int FMAS>
__global__ void k(int iters, float* out, uint32_t seed) {
  uint32_t a[4], b0, b1;
  for (int i = 0; i < 4; ++i) a[i] = 0x3c003c00u + ((threadIdx.x * 2654435761u + i * 40503u + seed) & 0x00ff00ffu);
  b0 = 0x3a003a00u + (threadIdx.x & 0xff); b1 = 0x39003900u + (threadIdx.x & 0x7f);
  float d[NACC][4];
  for (int n = 0; n < NACC; ++n) d[n][0] = d[n][1] = d[n][2] = d[n][3] = 0.f;
  float f[8] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
  const float sc = 0.999f + 1e-9f * seed;
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int n = 0; n < NACC; ++n) {
      mma(d[n], a, b0, b1);
#pragma unroll
      for (int j = 0; j < FMAS; ++j) f[j & 7] = __fmaf_rn(f[j & 7], sc, 0.5f);
    }
  }
  float s = 0;
  for (int n = 0; n < NACC; ++n) s += d[n][0] + d[n][1] + d[n][2] + d[n][3];
  for (int j = 0; j < 8; ++j) s += f[j];
  if (threadIdx.x == 0) out[blockIdx.x] = s;
}
template <int NACC, int FMAS>
double run(int warps, int ctas, int iters) {
  float* out; cudaMalloc(&out, ctas * 4);
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  double best = 1e30;
  for (int r = 0; r < 5; ++r) {
    cudaEventRecord(s); k<NACC, FMAS><<<ctas, warps * 32>>>(iters, out, r); cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e); if (r && ms < best) best = ms;
  }
  cudaFree(out);
  return best;
}
int main(int argc, char** argv) {
  const double mhz = argc > 1 ? atof(argv[1]) : 1410.0;
  const int sms = 108, iters = 20000;
  printf("clock %.0f MHz; cycles per HMMA: per warp (latency view) | per SM (throughput view)\n", mhz);
#define ROW(NACC, FMAS) for (int w : {1, 2, 4, 8, 16}) { double ms = run<NACC, FMAS>(w, sms, iters); double cyc = ms * 1e-3 * mhz * 1e6; \
    printf("acc=%2d fma/mma=%d warps/SM=%2d: %8.3f ms  %6.2f cyc/HMMA/warp  %6.2f cyc/HMMA/SM\n", NACC, FMAS, w, ms, cyc / (iters * NACC), cyc / (iters * NACC * (double)w)); }
  ROW(1, 0) ROW(2, 0) ROW(4, 0) ROW(8, 0) ROW(16, 0) ROW(4, 2) ROW(4, 4) ROW(4, 8) ROW(16, 2) ROW(16, 4) ROW(16, 8)
  return 0;
}
