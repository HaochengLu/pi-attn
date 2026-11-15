// Latency of one instruction kind on a lone warp (one dependent chain), and the cost of the same chain with 2 and
// 4 warps per SM sub-partition (8 / 16 warps per SM). cycles per instruction at the given clock.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
__device__ __forceinline__ float ex2(float x) { float y; asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ uint32_t pack(float lo, float hi) { uint32_t v; asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(v) : "f"(hi), "f"(lo)); return v; }
template <int KIND>
__global__ void k(int iters, float* out, uint32_t seed) {
  float f = 1.f + threadIdx.x * 1e-3f, g = 0.5f; uint32_t u = threadIdx.x + seed;
  const float sc = 0.999f + 1e-9f * seed;
  __shared__ float sm[1024];
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int n = 0; n < 16; ++n) {
      if (KIND == 0) f = __fmaf_rn(f, sc, 0.5f);
      if (KIND == 1) f = ex2(f) * 0.25f;                      // EX2 + FMUL
      if (KIND == 2) f = __shfl_xor_sync(0xffffffffu, f, 1) + 1.f;   // SHFL + FADD
      if (KIND == 3) f = fmaxf(f, g) , g = f * sc;            // FMNMX + FMUL
      if (KIND == 4) f = __uint_as_float(pack(f, f) << 16) + 1.f;  // F2FP + SHL + FADD
      if (KIND == 5) f = __fadd_rn(f, sc);
      if (KIND == 6) u = (u ^ (u >> 7)) + 0x9e3779b9u;         // 2-3 int
      if (KIND == 7) { sm[threadIdx.x] = f; f = sm[threadIdx.x ^ 1] + 1.f; }  // STS + LDS + FADD
      if (KIND == 8) { __syncthreads(); f = __fadd_rn(f, sc); }  // BAR + FADD
    }
  }
  if (threadIdx.x == 0) out[blockIdx.x] = f + g + (float)u;
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
  run<0>(8, sms, 2000000);
  const char* names[] = {"FFMA", "EX2+FMUL", "SHFL+FADD", "FMNMX+FMUL", "F2FP+SHL+FADD", "FADD", "int x2-3", "STS+LDS+FADD", "BAR+FADD"};
#define ROW(K) for (int w : {4, 8, 16}) { double ms = run<K>(w, sms, iters); double cyc = ms * 1e-3 * mhz * 1e6; \
    printf("%-16s warps/SM=%2d: %8.3f ms  %6.2f cyc per link per warp\n", names[K], w, ms, cyc / (iters * 16.0)); }
  ROW(0) ROW(1) ROW(2) ROW(3) ROW(4) ROW(5) ROW(6) ROW(7) ROW(8)
  return 0;
}
