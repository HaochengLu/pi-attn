// Partition-Invariant Attention (PI-Attn), sm80+ (bf16, head_dim 128).
//
// Contract: KV positions are cut into chunks of kChunk keys at absolute
// positions. A chunk partial (m, l, o) for one query row is a pure function of
// (q row, chunk K/V, visible-key mask). The output is the canonical fold of the
// partials in ascending chunk order. The prefill kernel folds in registers; the
// decode path writes partials and the merge kernel folds them. Both use
// `chunk_partial` and `fold_step` below, so for any batch composition, split
// count, chunked-prefill cut or prefix-cache state the result is bitwise equal.
#pragma once
#include <cuda_bf16.h>
#include <cstdint>

namespace pi_attn {

constexpr int kHeadDim = 128;
constexpr int kChunk = 64;          // keys per chunk (multiple of page size)
constexpr int kRowStride = kHeadDim + 8;  // smem row padding (bf16), conflict-free
constexpr int kNTilesS = kChunk / 8;      // 8 n-tiles of S per chunk
constexpr int kKStepsQK = kHeadDim / 16;  // 8 k-steps over head_dim
constexpr int kKStepsPV = kChunk / 16;    // 4 k-steps over keys
constexpr int kNTilesO = kHeadDim / 8;    // 16 n-tiles of O
constexpr int kChunkSmemElems = 2 * kChunk * kRowStride;  // K and V

// Exact, contraction-free primitives shared by every kernel.
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float sub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float add(float a, float b) { return __fadd_rn(a, b); }

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float round_bf16(float x) {
  return __bfloat162float(__float2bfloat16_rn(x));
}

__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Query fragment for 16 rows x 128 dims in mma A layout. Row r of the tile is
// row_ptr(r) (nullptr = padding row, all zeros).
struct QFrag {
  uint32_t a[kKStepsQK][4];
};

template <typename RowPtr>
__device__ __forceinline__ void load_q(QFrag& q, RowPtr row_ptr, int lane) {
  const int g = lane >> 2, t = lane & 3;
  const __nv_bfloat16* r0 = row_ptr(g);
  const __nv_bfloat16* r1 = row_ptr(g + 8);
#pragma unroll
  for (int ks = 0; ks < kKStepsQK; ++ks) {
    const int c0 = ks * 16 + t * 2, c1 = c0 + 8;
    q.a[ks][0] = r0 ? *reinterpret_cast<const uint32_t*>(r0 + c0) : 0u;
    q.a[ks][1] = r1 ? *reinterpret_cast<const uint32_t*>(r1 + c0) : 0u;
    q.a[ks][2] = r0 ? *reinterpret_cast<const uint32_t*>(r0 + c1) : 0u;
    q.a[ks][3] = r1 ? *reinterpret_cast<const uint32_t*>(r1 + c1) : 0u;
  }
}

// Per-thread view of one chunk partial for its two rows (g and g+8).
struct Partial {
  float m[2];
  float l[2];
  float o[kNTilesO][4];  // mma C layout: [n-tile][c0..c3]
};

// Chunk partial for 16 rows. sK/sV hold the chunk ([kChunk][kRowStride]); keys
// that do not exist are zero-filled by the loader. `visible(row, key)` says
// whether key index (0..kChunk) is visible to tile row (0..15).
template <typename Visible>
__device__ __forceinline__ void chunk_partial(Partial& p, const QFrag& q, const __nv_bfloat16* sK,
                                              const __nv_bfloat16* sV, float scale_log2,
                                              Visible visible, int lane) {
  const int g = lane >> 2, t = lane & 3;
  float s[kNTilesS][4];
#pragma unroll
  for (int n = 0; n < kNTilesS; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
  // S = Q K^T, accumulating over head_dim in ascending k-steps.
#pragma unroll
  for (int ks = 0; ks < kKStepsQK; ++ks) {
#pragma unroll
    for (int n = 0; n < kNTilesS; ++n) {
      const __nv_bfloat16* kr = sK + (n * 8 + g) * kRowStride + ks * 16 + t * 2;
      mma_bf16(s[n], q.a[ks], *reinterpret_cast<const uint32_t*>(kr),
               *reinterpret_cast<const uint32_t*>(kr + 8));
    }
  }
  // Scale, mask, chunk-local row max (max is exact, order-free).
  float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
  for (int n = 0; n < kNTilesS; ++n) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int row = g + (i >> 1) * 8, key = n * 8 + t * 2 + (i & 1);
      const float v = visible(row, key) ? mul(s[n][i], scale_log2) : -INFINITY;
      s[n][i] = v;
      mx[i >> 1] = fmaxf(mx[i >> 1], v);
    }
  }
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
    mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
  }
  // P = bf16(exp2(s - m)); l = row sum of rounded P in a fixed order.
  float lsum[2] = {0.f, 0.f};
  uint32_t pa[kKStepsPV][4];
#pragma unroll
  for (int n = 0; n < kNTilesS; ++n) {
    float pv[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float m = mx[i >> 1] == -INFINITY ? 0.f : mx[i >> 1];
      pv[i] = round_bf16(ex2(sub(s[n][i], m)));
      lsum[i >> 1] = add(lsum[i >> 1], pv[i]);
    }
    // C layout of n-tiles (2j, 2j+1) is the A layout of k-step j.
    const int j = n >> 1, h = (n & 1) * 2;
    pa[j][h + 0] = pack_bf16(pv[0], pv[1]);
    pa[j][h + 1] = pack_bf16(pv[2], pv[3]);
  }
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    lsum[r] = add(lsum[r], __shfl_xor_sync(0xffffffffu, lsum[r], 1));
    lsum[r] = add(lsum[r], __shfl_xor_sync(0xffffffffu, lsum[r], 2));
    p.m[r] = mx[r];
    p.l[r] = lsum[r];
  }
  // O = P V from a zero accumulator, ascending k-steps over keys.
#pragma unroll
  for (int n = 0; n < kNTilesO; ++n) p.o[n][0] = p.o[n][1] = p.o[n][2] = p.o[n][3] = 0.f;
  // pa[j] = {a0a1 (row g, keys 16j+2t..), a2a3 (row g+8), a4a5 (row g, keys +8), a6a7 (row g+8, +8)}
#pragma unroll
  for (int j = 0; j < kKStepsPV; ++j) {
    uint32_t a[4] = {pa[j][0], pa[j][1], pa[j][2], pa[j][3]};
#pragma unroll
    for (int n = 0; n < kNTilesO; ++n) {
      const int d = n * 8 + g, k0 = j * 16 + t * 2;
      const __nv_bfloat16* v0 = sV + k0 * kRowStride + d;
      __nv_bfloat162 b0, b1;
      b0.x = v0[0];
      b0.y = v0[kRowStride];
      b1.x = v0[8 * kRowStride];
      b1.y = v0[9 * kRowStride];
      mma_bf16(p.o[n], a, *reinterpret_cast<uint32_t*>(&b0), *reinterpret_cast<uint32_t*>(&b1));
    }
  }
  // Canonicalize signed zeros so masked 0*v products cannot leave a -0.
#pragma unroll
  for (int n = 0; n < kNTilesO; ++n)
#pragma unroll
    for (int i = 0; i < 4; ++i) p.o[n][i] = add(p.o[n][i], 0.f);
}

// One canonical fold step on scalars. Every fold (registers or merge kernel)
// goes through this function. A partial with m == -inf is the identity.
struct FoldScalars {
  float alpha, beta;
};
__device__ __forceinline__ FoldScalars fold_scalars(float& M, float& L, float m_c, float l_c) {
  if (m_c == -INFINITY) return {1.f, 0.f};
  const float Mn = fmaxf(M, m_c);
  const float alpha = ex2(sub(M, Mn));
  const float beta = ex2(sub(m_c, Mn));
  L = __fmaf_rn(L, alpha, mul(l_c, beta));
  M = Mn;
  return {alpha, beta};
}
__device__ __forceinline__ float fold_elem(float o, float o_c, FoldScalars f) {
  if (f.beta == 0.f && f.alpha == 1.f) return o;
  return __fmaf_rn(o, f.alpha, mul(o_c, f.beta));
}
__device__ __forceinline__ __nv_bfloat16 finish(float o, float L) {
  return __float2bfloat16_rn(__fdiv_rn(o, L));
}

}  // namespace pi_attn
