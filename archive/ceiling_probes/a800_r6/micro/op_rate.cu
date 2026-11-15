// Cycles per warp instruction on one SM by kind (16 independent chains, unrolled), by warps per SM.
// kinds: 0 FFMA, 1 integer add/xor, 2 MUFU.EX2, 3 ldmatrix x4 (smem), 4 F2FP pack, 5 FMNMX, 6 HMMA (16 acc),
// 7 = 1 HMMA + 4 FFMA + 4 int, 8 = 1 HMMA + 8 int, 9 = 1 HMMA + 2 LDSM + 4 FFMA
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float ex2(float x) { float y; asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ void ldm(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ uint32_t pack(float lo, float hi) { uint32_t v; asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(v) : "f"(hi), "f"(lo)); return v; }
template <int KIND>
__global__ void k(int iters, float* out, uint32_t seed) {
  __shared__ __align__(16) uint16_t sm[16 * 1024];
  for (int i = threadIdx.x; i < 16 * 1024; i += blockDim.x) sm[i] = 0x3c00 + (i & 255);
  __syncthreads();
  const uint32_t base = (uint32_t)__cvta_generic_to_shared(sm) + (threadIdx.x & 31) * 16;
  uint32_t a[4], b0, b1, u[16], r[4] = {0, 0, 0, 0};
  for (int i = 0; i < 4; ++i) a[i] = 0x3c003c00u + ((threadIdx.x * 2654435761u + i * 40503u + seed) & 0x00ff00ffu);
  b0 = 0x3a003a00u + (threadIdx.x & 0xff); b1 = 0x39003900u + (threadIdx.x & 0x7f);
  float d[16][4], f[16];
  for (int n = 0; n < 16; ++n) { d[n][0] = d[n][1] = d[n][2] = d[n][3] = 0.f; f[n] = 1.f + n + threadIdx.x; u[n] = n * 77 + threadIdx.x + seed; }
  const float sc = 0.999f + 1e-9f * seed; const uint32_t ku = 0x9e3779b9u + seed;
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int n = 0; n < 16; ++n) {
      if (KIND == 0) f[n] = __fmaf_rn(f[n], sc, 0.5f);
      if (KIND == 1) u[n] = (u[n] + ku) ^ (u[n] >> 3);  // 3 instructions
      if (KIND == 2) f[n] = ex2(f[n]) ;
      if (KIND == 3) { ldm(r, base + ((it * 16 + n) & 63) * 512); u[n] += r[0] ^ r[1] ^ r[2] ^ r[3]; }  // 1 LDSM + ~4 int
      if (KIND == 4) { u[n] = pack(f[n], f[(n + 1) & 15]); f[n] = __uint_as_float(u[n] << 16); }  // F2FP + shift
      if (KIND == 5) f[n] = fmaxf(f[n], f[(n + 5) & 15] * 1.0f);
      if (KIND >= 6) mma(d[n], a, b0, b1);
      if (KIND == 7) { for (int j = 0; j < 4; ++j) { f[(n + j) & 15] = __fmaf_rn(f[(n + j) & 15], sc, 0.5f); u[(n + j) & 15] += ku; } }
      if (KIND == 8) { for (int j = 0; j < 8; ++j) u[(n + j) & 15] += ku; }
      if (KIND == 9) { ldm(r, base + ((it * 16 + n) & 63) * 512); a[0] ^= r[0] & 1; ldm(r, base + ((it * 16 + n + 7) & 63) * 512); b0 ^= r[1] & 1;
                       for (int j = 0; j < 4; ++j) f[(n + j) & 15] = __fmaf_rn(f[(n + j) & 15], sc, 0.5f); }
    }
  }
  float s = 0;
  for (int n = 0; n < 16; ++n) s += d[n][0] + d[n][1] + d[n][2] + d[n][3] + f[n] + (float)u[n];
  if (threadIdx.x == 0) out[blockIdx.x] = s + r[0];
}
template <int KIND>
double run(int warps, int ctas, int iters) {
  float* out; cudaMalloc(&out, ctas * 4);
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  double best = 1e30;
  for (int r = 0; r < 5; ++r) {
    cudaEventRecord(s); k<KIND><<<ctas, warps * 32>>>(iters, out, r); cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e); if (r && ms < best) best = ms;
  }
  cudaFree(out);
  return best;
}
int main(int argc, char** argv) {
  const double mhz = argc > 1 ? atof(argv[1]) : 1410.0;
  const int sms = 108, iters = 20000;
  run<6>(8, sms, 200000);  // warm the clock
  const char* names[] = {"FFMA", "int x3", "EX2", "LDSM+4int", "F2FP+shl", "FMNMX+mul", "HMMA", "HMMA+4FFMA+4int", "HMMA+8int", "HMMA+2LDSM+4FFMA+2int"};
#define ROW(K) for (int w : {1, 4, 8, 16}) { double ms = run<K>(w, sms, iters); double cyc = ms * 1e-3 * mhz * 1e6; \
    printf("%-22s warps/SM=%2d: %8.3f ms  %6.2f cyc/iter-slot/warp  %6.2f cyc per slot per SMSP-pair(=x4/w)\n", names[K], w, ms, cyc / (iters * 16.0), cyc / (iters * 16.0) * 4 / w); }
  ROW(0) ROW(1) ROW(2) ROW(3) ROW(4) ROW(5) ROW(6) ROW(7) ROW(8) ROW(9)
  return 0;
}
