// Log-probability of a chosen token under the softmax of its logits row, fp32.
//
// One CTA of kTokenLogprobThreads per row, so a row's value never depends on the
// batch. The operation sequence depends only on the vocabulary size: thread t
// reads columns t, t + kTokenLogprobThreads, ... in ascending order, first for
// the exact row max, then for the sum of 2^((x - max) log2 e); the per-thread
// sums are added in a fixed shared-memory tree, and
//   logprob = (x[token] - max) - log2(sum) ln 2.
// Only explicitly rounded operations and approximate-transcendental instructions
// are used, so the bits do not depend on the compiler's contraction choices: the
// serving engine and a trainer that link this header compute the same values on the
// same GPU architecture (ex2/lg2.approx are hardware approximations, A2 of the paper).
#pragma once
#include <cuda_bf16.h>
#include <cstdint>

namespace pi_logprob {

constexpr int kTokenLogprobThreads = 1024;

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float lg2(float x) {
  float y;
  asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float load(const float* p) { return *p; }
__device__ __forceinline__ float load(const __nv_bfloat16* p) { return __bfloat162float(*p); }

// logits: [rows, vocab] with row stride ld; tokens: [rows] (a negative token yields NaN).
template <typename T>
__global__ void __launch_bounds__(kTokenLogprobThreads)
    token_logprob_kernel(const T* __restrict__ logits, int64_t ld, int vocab, const int64_t* __restrict__ tokens,
                         float* __restrict__ out) {
  __shared__ float red[kTokenLogprobThreads];
  const int t = threadIdx.x;
  const T* x = logits + blockIdx.x * ld;
  float m = -INFINITY;
  for (int v = t; v < vocab; v += kTokenLogprobThreads) m = fmaxf(m, load(x + v));
  red[t] = m;
  __syncthreads();
  for (int s = kTokenLogprobThreads / 2; s > 0; s >>= 1) {
    if (t < s) red[t] = fmaxf(red[t], red[t + s]);
    __syncthreads();
  }
  m = red[0];
  __syncthreads();
  constexpr float kLog2e = 1.4426950408889634f;
  float sum = 0.f;
  for (int v = t; v < vocab; v += kTokenLogprobThreads) sum = __fadd_rn(sum, ex2(__fmul_rn(__fsub_rn(load(x + v), m), kLog2e)));
  red[t] = sum;
  __syncthreads();
  for (int s = kTokenLogprobThreads / 2; s > 0; s >>= 1) {
    if (t < s) red[t] = __fadd_rn(red[t], red[t + s]);
    __syncthreads();
  }
  if (t == 0) {
    const int64_t tok = tokens[blockIdx.x];
    constexpr float kLn2 = 0.6931471805599453f;
    out[blockIdx.x] = tok < 0 || tok >= vocab ? __int_as_float(0x7fc00000)
                                              : __fsub_rn(__fsub_rn(load(x + tok), m), __fmul_rn(lg2(red[0]), kLn2));
  }
}

// bf16 (is_fp32 == 0) or fp32 logits. Returns the launch error, if any.
inline cudaError_t token_logprob(const void* logits, int is_fp32, int64_t ld, int rows, int vocab,
                                 const int64_t* tokens, float* out, cudaStream_t stream) {
  if (rows == 0) return cudaSuccess;
  if (is_fp32)
    token_logprob_kernel<float><<<rows, kTokenLogprobThreads, 0, stream>>>(static_cast<const float*>(logits), ld, vocab,
                                                                            tokens, out);
  else
    token_logprob_kernel<__nv_bfloat16><<<rows, kTokenLogprobThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits), ld, vocab, tokens, out);
  return cudaGetLastError();
}

}  // namespace pi_logprob
