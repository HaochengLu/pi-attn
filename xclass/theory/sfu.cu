// sfu.cu -- red team: are the special-function-unit approximations and the CUDA libm routines bit-identical across
// GPU generations? Every op is evaluated on all 2^32 fp32 bit patterns; the outputs are hashed in 256 chunks
// (chunk = top 8 input bits). Two-argument ops use y = bits(mix(x)) for every x.
//   nvcc -O2 -std=c++17 -arch=sm_80 sfu.cu -o sfu && ./sfu
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(2); } } while (0)
#define PTX1(name, ins) __device__ __forceinline__ float name(float x) { float y; asm volatile(ins " %0, %1;\n" : "=f"(y) : "f"(x)); return y; }
#define PTX2(name, ins) __device__ __forceinline__ float name(float x, float z) { float y; asm volatile(ins " %0, %1, %2;\n" : "=f"(y) : "f"(x), "f"(z)); return y; }
PTX1(p_ex2_ftz, "ex2.approx.ftz.f32") PTX1(p_ex2, "ex2.approx.f32") PTX1(p_lg2_ftz, "lg2.approx.ftz.f32") PTX1(p_lg2, "lg2.approx.f32")
PTX1(p_rsqrt_ftz, "rsqrt.approx.ftz.f32") PTX1(p_rsqrt, "rsqrt.approx.f32") PTX1(p_sqrt_ftz, "sqrt.approx.ftz.f32") PTX1(p_rcp_ftz, "rcp.approx.ftz.f32")
PTX1(p_sin_ftz, "sin.approx.ftz.f32") PTX1(p_cos_ftz, "cos.approx.ftz.f32") PTX1(p_tanh, "tanh.approx.f32") PTX1(p_sqrt_rn, "sqrt.rn.f32") PTX1(p_rcp_rn, "rcp.rn.f32")
PTX2(p_div_approx, "div.approx.ftz.f32") PTX2(p_div_full, "div.full.f32") PTX2(p_div_rn, "div.rn.f32")
__device__ __forceinline__ uint64_t mix(uint64_t z) { z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; return z ^ (z >> 31); }
static const char* NAMES[] = {"ex2.approx.ftz", "ex2.approx", "lg2.approx.ftz", "lg2.approx", "rsqrt.approx.ftz", "rsqrt.approx", "sqrt.approx.ftz", "rcp.approx.ftz",
  "sin.approx.ftz", "cos.approx.ftz", "tanh.approx", "sqrt.rn", "rcp.rn", "libm expf", "libm logf", "libm tanhf", "libm erff", "libm rsqrtf", "sigmoid 1/(1+expf(-x))",
  "silu x/(1+expf(-x))", "libm exp2f", "libm log2f", "libm sinf", "libm cosf", "div.approx.ftz(x,y)", "div.full(x,y)", "div.rn(x,y)", "libm powf(x,y)", "fma.rn(x,y,x) [fmaf]", "softmax-term ex2.approx.ftz(x*log2e)"};
const int NOPS = 30;
template <int OP> __device__ __forceinline__ float apply(float x, float y) {
  switch (OP) {
    case 0: return p_ex2_ftz(x); case 1: return p_ex2(x); case 2: return p_lg2_ftz(x); case 3: return p_lg2(x); case 4: return p_rsqrt_ftz(x); case 5: return p_rsqrt(x);
    case 6: return p_sqrt_ftz(x); case 7: return p_rcp_ftz(x); case 8: return p_sin_ftz(x); case 9: return p_cos_ftz(x); case 10: return p_tanh(x); case 11: return p_sqrt_rn(x);
    case 12: return p_rcp_rn(x); case 13: return expf(x); case 14: return logf(x); case 15: return tanhf(x); case 16: return erff(x); case 17: return rsqrtf(x);
    case 18: return 1.0f / (1.0f + expf(-x)); case 19: return x / (1.0f + expf(-x)); case 20: return exp2f(x); case 21: return log2f(x); case 22: return sinf(x); case 23: return cosf(x);
    case 24: return p_div_approx(x, y); case 25: return p_div_full(x, y); case 26: return p_div_rn(x, y); case 27: return powf(x, y); case 28: return fmaf(x, y, x);
    default: return p_ex2_ftz(__fmul_rn(x, 1.4426950408889634f));
  }
}
template <int OP> __global__ void k_op(unsigned long long* chunk) {
  const uint32_t base = ((uint32_t)blockIdx.x * 256u + threadIdx.x) << 8;
  uint64_t s = 0;
  for (uint32_t i = 0; i < 256; ++i) {
    const uint32_t xb = base + i; const float x = __uint_as_float(xb), y = __uint_as_float((uint32_t)mix(xb + 0x9E3779B97F4A7C15ull));
    s += mix(((uint64_t)xb << 32) | __float_as_uint(apply<OP>(x, y)));
  }
  atomicAdd(chunk + (base >> 24), (unsigned long long)s);
}
template <int OP> static void run(unsigned long long* d) { k_op<OP><<<65536, 256>>>(d); }
int main() {
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  printf("GPU %s sm_%d%d  SFU probe (compiled for %s)\n", p.name, p.major, p.minor, ARCHNAME);
  unsigned long long* d; CK(cudaMalloc(&d, 256 * 8)); unsigned long long h[256];
  void (*runs[NOPS])(unsigned long long*) = {run<0>, run<1>, run<2>, run<3>, run<4>, run<5>, run<6>, run<7>, run<8>, run<9>, run<10>, run<11>, run<12>, run<13>, run<14>, run<15>,
    run<16>, run<17>, run<18>, run<19>, run<20>, run<21>, run<22>, run<23>, run<24>, run<25>, run<26>, run<27>, run<28>, run<29>};
  for (int op = 0; op < NOPS; ++op) {
    CK(cudaMemset(d, 0, 256 * 8)); runs[op](d); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(h, d, 256 * 8, cudaMemcpyDeviceToHost));
    unsigned long long tot = 0; for (int i = 0; i < 256; ++i) tot = tot * 0x100000001B3ull + h[i];
    printf("SFU %2d %-38s %016llx\n", op, NAMES[op], tot);
    printf("SFUCHUNK %2d", op); for (int i = 0; i < 256; ++i) printf(" %016llx", h[i]); printf("\n");
  }
  return 0;
}
