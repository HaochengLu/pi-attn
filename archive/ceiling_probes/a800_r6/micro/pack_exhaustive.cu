// cvt.rn.bf16x2.f32 of two raw floats against the pair of cvt.rn.bf16.f32 results converted back to float and packed
// (online_step before round 6): every one of the 2^32 bit patterns in each half, the other half a hashed pattern.
// build: nvcc -O3 -arch=sm_80 pack_exhaustive.cu -o pack_exhaustive
#include <cstdio>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float round_bf16(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__global__ void k(uint32_t base, unsigned long long* bad, unsigned long long* nan_in) {
  const uint32_t x = base + blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = x * 2654435761u + 0x9e3779b9u; y ^= y >> 15; y *= 0x85ebca6bu; y ^= y >> 13;
  const float fx = __uint_as_float(x), fy = __uint_as_float(y);
  const float px = round_bf16(fx), py = round_bf16(fy);
  const uint32_t w_old_a = pack_bf16(px, py), w_new_a = pack_bf16(fx, fy);
  const uint32_t w_old_b = pack_bf16(py, px), w_new_b = pack_bf16(fy, fx);
  // the floats the row sum adds: old = px, py; new = halves of the word
  const bool sum_a = __float_as_uint(px) == (w_new_a << 16) && __float_as_uint(py) == (w_new_a & 0xffff0000u);
  const bool sum_b = __float_as_uint(py) == (w_new_b << 16) && __float_as_uint(px) == (w_new_b & 0xffff0000u);
  if (w_old_a != w_new_a || w_old_b != w_new_b || !sum_a || !sum_b) atomicAdd(bad, 1ull);
  if (fx != fx) atomicAdd(nan_in, 1ull);
}
int main() {
  unsigned long long *bad, *nan_in;
  cudaMalloc(&bad, 8); cudaMalloc(&nan_in, 8); cudaMemset(bad, 0, 8); cudaMemset(nan_in, 0, 8);
  const uint32_t chunk = 1u << 24;
  for (uint64_t b = 0; b < (1ull << 32); b += chunk) k<<<chunk / 256, 256>>>((uint32_t)b, bad, nan_in);
  unsigned long long hb, hn;
  cudaMemcpy(&hb, bad, 8, cudaMemcpyDeviceToHost); cudaMemcpy(&hn, nan_in, 8, cudaMemcpyDeviceToHost);
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  printf("PACK %s: 4294967296 first operands (each also as the second operand of a hashed pattern), %llu NaN inputs: %llu mismatches\n", p.name, hn, hb);
  return hb != 0;
}
