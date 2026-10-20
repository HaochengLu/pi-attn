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

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_u32(p)));
}

// 16-byte async copy; src_bytes == 0 zero-fills the destination without reading.
__device__ __forceinline__ void cp_async16(void* dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(smem_u32(dst)), "l"(src),
               "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

// Stage 1 of a chunk partial: S = Q K^T, chunk-local max, P = bf16(exp2(s - m)),
// row sums. Register contents of every fragment are identical whichever kernel
// calls this, so the op sequence per row is identical.
struct PFrag {
  uint32_t a[kKStepsPV][4];
};

// `all_visible` is a hint that every (row, key) of the tile is visible; the
// mask is then skipped. It never changes a value, only avoids the test.
template <typename Visible>
__device__ __forceinline__ void chunk_scores(PFrag& pf, float (&m)[2], float (&l)[2],
                                             const QFrag& q, const __nv_bfloat16* sK,
                                             float scale_log2, Visible visible, int lane,
                                             bool all_visible = false) {
  const int g = lane >> 2, t = lane & 3;
  float s[kNTilesS][4];
#pragma unroll
  for (int n = 0; n < kNTilesS; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
  // ldmatrix.x4 address: matrices {n lo-d, n hi-d, n+1 lo-d, n+1 hi-d}.
  const int mat = lane >> 3, r8 = lane & 7;
#pragma unroll
  for (int ks = 0; ks < kKStepsQK; ++ks) {
#pragma unroll
    for (int n = 0; n < kNTilesS; n += 2) {
      uint32_t b[4];
      ldmatrix_x4(b, sK + ((n + (mat >> 1)) * 8 + r8) * kRowStride + ks * 16 + (mat & 1) * 8);
      mma_bf16(s[n], q.a[ks], b[0], b[1]);
      mma_bf16(s[n + 1], q.a[ks], b[2], b[3]);
    }
  }
  float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
  for (int n = 0; n < kNTilesS; ++n) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int row = g + (i >> 1) * 8, key = n * 8 + t * 2 + (i & 1);
      const float v = (all_visible || visible(row, key)) ? mul(s[n][i], scale_log2) : -INFINITY;
      s[n][i] = v;
      mx[i >> 1] = fmaxf(mx[i >> 1], v);
    }
  }
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
    mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
  }
  float lsum[2] = {0.f, 0.f};
#pragma unroll
  for (int n = 0; n < kNTilesS; ++n) {
    float pv[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float mm = mx[i >> 1] == -INFINITY ? 0.f : mx[i >> 1];
      pv[i] = round_bf16(ex2(sub(s[n][i], mm)));
      lsum[i >> 1] = add(lsum[i >> 1], pv[i]);
    }
    // C layout of n-tiles (2j, 2j+1) is the A layout of k-step j.
    const int j = n >> 1, h = (n & 1) * 2;
    pf.a[j][h + 0] = pack_bf16(pv[0], pv[1]);
    pf.a[j][h + 1] = pack_bf16(pv[2], pv[3]);
  }
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    lsum[r] = add(lsum[r], __shfl_xor_sync(0xffffffffu, lsum[r], 1));
    lsum[r] = add(lsum[r], __shfl_xor_sync(0xffffffffu, lsum[r], 2));
    m[r] = mx[r];
    l[r] = lsum[r];
  }
}

// Stage 2: O = P V from a zero accumulator, ascending k-steps over keys; signed
// zeros canonicalized so masked 0*v products cannot leave a -0.
__device__ __forceinline__ void chunk_values(float (&o)[kNTilesO][4], const PFrag& pf,
                                             const __nv_bfloat16* sV, int lane) {
#pragma unroll
  for (int n = 0; n < kNTilesO; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  // ldmatrix.x4.trans address: matrices {keys lo n, keys hi n, keys lo n+1, keys hi n+1}.
  const int mat = lane >> 3, r8 = lane & 7;
#pragma unroll
  for (int j = 0; j < kKStepsPV; ++j) {
#pragma unroll
    for (int n = 0; n < kNTilesO; n += 2) {
      uint32_t b[4];
      ldmatrix_x4_trans(b, sV + (j * 16 + (mat & 1) * 8 + r8) * kRowStride + (n + (mat >> 1)) * 8);
      mma_bf16(o[n], pf.a[j], b[0], b[1]);
      mma_bf16(o[n + 1], pf.a[j], b[2], b[3]);
    }
  }
#pragma unroll
  for (int n = 0; n < kNTilesO; ++n)
#pragma unroll
    for (int i = 0; i < 4; ++i) o[n][i] = add(o[n][i], 0.f);
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
