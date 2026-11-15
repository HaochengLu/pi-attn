// Throughput of shared-memory fragment loads on one SM: ldmatrix.x4 (4 registers, 512 bytes per warp) against
// ld.shared of 16 bytes per thread (4 registers, 512 bytes per warp), independent loads consumed DELAY loads
// later (so the load latency is hidden), alone and with one HMMA per load; cycles per load per warp and per
// SM sub-partition, by warps per SM.  build: nvcc -O3 -arch=sm_80 smem_rate.cu -o smem_rate
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
__device__ __forceinline__ void ldm(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void lds(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// KIND 0: ldmatrix.x4, 1: ld.shared 16 B; MMA: HMMAs per load (0, 1, 2), fed by the load of DELAY iterations ago
template <int KIND, int MMA>
__global__ void k(int iters, float* out, uint32_t seed) {
  extern __shared__ __align__(16) uint16_t sm[];
  for (int i = threadIdx.x; i < 32 * 1024; i += blockDim.x) sm[i] = 0x3c00 + ((i * 7 + seed) & 255);
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const uint32_t base = (uint32_t)__cvta_generic_to_shared(sm);
  // ldmatrix: row lane % 8 of matrix lane / 8, swizzle-free rows 256 bytes apart; ld.shared: 16 bytes per thread
  // ldmatrix: row lane % 8 of matrix lane / 8 in a swizzled [8][128] bf16 tile (chunk ^ row, as the kernels' swz)
  const uint32_t a_m = base + (lane & 7) * 256 + (((lane >> 3) ^ (lane & 7)) << 4), a_l = base + lane * 16 + (threadIdx.x >> 5) * 512;
  constexpr int DELAY = 4;
  uint32_t ring[DELAY][4], a[4] = {0x3c003c00u, 0x3c003c00u, 0x3c003c00u, 0x3c003c00u};
  for (int i = 0; i < DELAY; ++i) for (int j = 0; j < 4; ++j) ring[i][j] = 0x3c003c00u;
  float d[8][4]; uint32_t acc = seed;
  for (int n = 0; n < 8; ++n) d[n][0] = d[n][1] = d[n][2] = d[n][3] = 0.f;
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      uint32_t r[4];
      const uint32_t off = ((it * 8 + n) & 7) * 2048;
      if (KIND == 0) ldm(r, a_m + off); else lds(r, a_l + off);
      uint32_t (&old)[4] = ring[n % DELAY];
      if (MMA >= 1) mma(d[n], a, old[0], old[1]);
      if (MMA >= 2) mma(d[(n + 4) & 7], a, old[2], old[3]);
      if (MMA == 0) acc += old[0] ^ old[1] ^ old[2] ^ old[3];
#pragma unroll
      for (int j = 0; j < 4; ++j) old[j] = r[j];
    }
  }
  float s = (float)acc;
  for (int n = 0; n < 8; ++n) s += d[n][0] + d[n][1] + d[n][2] + d[n][3];
  if (threadIdx.x == 0) out[blockIdx.x] = s;
}
template <int KIND, int MMA>
double run(int warps, int ctas, int iters) {
  float* out; cudaMalloc(&out, ctas * 4);
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  double best = 1e30;
  for (int r = 0; r < 5; ++r) {
    cudaFuncSetAttribute(k<KIND, MMA>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024);
    cudaEventRecord(s); k<KIND, MMA><<<ctas, warps * 32, 64 * 1024>>>(iters, out, r); cudaEventRecord(e); cudaEventSynchronize(e);
    if (cudaGetLastError() != cudaSuccess) { printf("launch failed\n"); exit(1); }
    float ms; cudaEventElapsedTime(&ms, s, e); if (r && ms < best) best = ms;
  }
  cudaFree(out);
  return best;
}
int main(int argc, char** argv) {
  const double mhz = argc > 1 ? atof(argv[1]) : 1410.0;
  const int sms = 108, iters = 20000;
  run<0, 2>(8, sms, 200000);
#define ROW(K, M, name) for (int w : {4, 8, 16}) { double ms = run<K, M>(w, sms, iters); double cyc = ms * 1e-3 * mhz * 1e6 / (iters * 8.0); \
    printf("SMEM %-34s warps/SM=%2d: %7.2f cycles per load per warp, %6.2f per load per sub-partition\n", name, w, cyc, cyc * 4 / w); }
  ROW(0, 0, "ldmatrix.x4") ROW(1, 0, "ld.shared 16 B") ROW(0, 1, "ldmatrix.x4 + 1 HMMA") ROW(1, 1, "ld.shared 16 B + 1 HMMA")
  ROW(0, 2, "ldmatrix.x4 + 2 HMMA") ROW(1, 2, "ld.shared 16 B + 2 HMMA")
  return 0;
}
