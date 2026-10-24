// Partition-Invariant Attention (PI-Attn), sm80+ (bf16, head_dim 128).
//
// Contract: every query row runs one canonical operation sequence that depends
// only on its position and the keys it sees. Keys are cut into tiles of kTile
// positions and fold groups of kGroupTiles tiles at absolute positions. Within a
// group the row runs an online softmax over the group's tiles in ascending
// order, from the empty state at the group start (`online_step`: P rounded to
// bf16 against the running max, o += P V on the tensor cores). The group
// partials (m, d, o) are folded in ascending group order (`fold_scalars` /
// `fold_elem`) and normalised by `finish`. Prefill folds the groups in
// registers or, split over groups, writes group partials that a merge kernel
// folds; decode runs each group in one CTA, the tile split across warps without
// changing any element's sequence, and its merge folds the groups. A row's bits
// therefore do not depend on batching, chunked-prefill cuts, prefix caching,
// split counts, or whether it is prefilled or decoded.
#pragma once
#include <cuda_bf16.h>
#include <cstdint>

namespace pi_attn {

constexpr int kHeadDim = 128;
constexpr int kTile = 32;                        // keys per online-softmax step
constexpr int kGroupTiles = 16;                  // tiles per fold group
constexpr int kGroupKeys = kTile * kGroupTiles;  // 512 keys per fold group
constexpr int kTileNS = kTile / 8;               // 4 n-tiles of S per tile
constexpr int kTileKS = kTile / 16;              // 2 key k-steps of P V per tile
constexpr int kKStepsQK = kHeadDim / 16;         // 8 k-steps over head_dim
constexpr int kNTilesO = kHeadDim / 8;           // 16 n-tiles of O

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

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
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

// Query A-fragments for 16 rows x 128 dims held in registers. Row r of the tile
// is row_ptr(r) (nullptr = padding row, all zeros).
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

// One canonical fold step. Every fold (registers or merge kernel) goes through
// these functions. A partial with m == -inf is the identity.
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
// Identity partials (m == -inf) leave the state untouched; this branch is part
// of the canonical sequence (all kernels take it).
__device__ __forceinline__ float fold_elem(float o, float o_c, FoldScalars f) {
  if (f.beta == 0.f && f.alpha == 1.f) return o;
  return __fmaf_rn(o, f.alpha, mul(o_c, f.beta));
}

__device__ __forceinline__ __nv_bfloat16 finish(float o, float L) {
  return __float2bfloat16_rn(__fdiv_rn(o, L));
}


// Swizzled [rows][128] bf16: 16-byte chunk j of row r at r * 16 + (j ^ (r & 7)).
__device__ __forceinline__ int swz(int row, int chunk) { return row * kHeadDim + ((chunk ^ (row & 7)) << 3); }
// Byte offset of chunk j of row r. In a 256-byte-aligned tile, chunk j ^ x of row r
// (x < 16) is at swz_bytes(r, j) ^ (x << 4): a per-thread address computed once
// serves every k-step with one XOR.
__device__ __forceinline__ uint32_t swz_bytes(int row, int chunk) { return (uint32_t)swz(row, chunk) * 2; }

// Online-softmax state of the two rows (g, g + 8) of one 16-row group.
struct OnlineState {
  float m[2], d[2], o[kNTilesO][4];
  __device__ __forceinline__ void reset() {
    m[0] = m[1] = -INFINITY;
    d[0] = d[1] = 0.f;
#pragma unroll
    for (int n = 0; n < kNTilesO; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  }
};

// One canonical online step on a tile's scaled+masked scores v (C layout,
// kTileNS n-tiles). Returns P as A fragments (kTileKS k-steps) and rescales o
// with fma(o, scale, +0), so o never holds -0: a fully masked step (prefill rows
// before the tile) then leaves the state bit-identical, as if skipped.
// Row max exact; P = bf16(2^(v - m_new)); d = fma(d, 2^(m - m_new), sum P) with
// the sum in the fixed order (own keys n = 0.., then lane xor 1, xor 2).
__device__ __forceinline__ void online_step(OnlineState& st, float (&v)[kTileNS][4],
                                            uint32_t (&pa)[kTileKS][4]) {
  float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
  for (int n = 0; n < kTileNS; ++n)
#pragma unroll
    for (int i = 0; i < 4; ++i) mx[i >> 1] = fmaxf(mx[i >> 1], v[n][i]);
  float mm[2], scale[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
    mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
    const float mn = fmaxf(st.m[r], mx[r]);
    mm[r] = mn == -INFINITY ? 0.f : mn;
    scale[r] = ex2(sub(st.m[r], mm[r]));
    st.m[r] = mn;
  }
  float ls[2] = {0.f, 0.f};
#pragma unroll
  for (int n = 0; n < kTileNS; ++n) {
    float p[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      p[i] = round_bf16(ex2(sub(v[n][i], mm[i >> 1])));
      ls[i >> 1] = add(ls[i >> 1], p[i]);
    }
    pa[n >> 1][(n & 1) * 2 + 0] = pack_bf16(p[0], p[1]);
    pa[n >> 1][(n & 1) * 2 + 1] = pack_bf16(p[2], p[3]);
  }
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    ls[r] = add(ls[r], __shfl_xor_sync(0xffffffffu, ls[r], 1));
    ls[r] = add(ls[r], __shfl_xor_sync(0xffffffffu, ls[r], 2));
    st.d[r] = __fmaf_rn(st.d[r], scale[r], ls[r]);
  }
#pragma unroll
  for (int n = 0; n < kNTilesO; ++n)
#pragma unroll
    for (int i = 0; i < 4; ++i) st.o[n][i] = __fmaf_rn(st.o[n][i], scale[i >> 1], 0.f);  // -0 -> +0
}

// Thread tid of 128 copies key tid / 4 of a tile, 16-byte chunks tid % 4 + 4 i
// (i = 0..3; a warp's copy covers 64 contiguous bytes of each of its 8 keys), so
// it reads one page-table entry and forms one address per tile.
static_assert(kTile * 4 == 128, "one key per 4 threads of a 128-thread CTA");

// Element offset of this thread's key of tile t in a cache (-1 = position >=
// kv_len), computed ahead of the copies so the page-table read is off the critical path.
__device__ __forceinline__ int64_t tile_row(const int32_t* pages, int page_size, int page_shift, int t, int kv_len,
                                            int num_kv_heads, int kvh, int tid) {
  const int pos = t * kTile + (tid >> 2);
  if (pos >= kv_len) return -1;
  const int slot = page_shift >= 0 ? (pages[pos >> page_shift] << page_shift) + (pos & (page_size - 1))
                                   : pages[pos / page_size] * page_size + pos % page_size;
  return ((int64_t)slot * num_kv_heads + kvh) * kHeadDim;
}

// Async load of one kTile x 128 K or V tile (swizzled), 128 threads; a key with
// row -1 is zero-filled without a global read.
__device__ __forceinline__ void issue_tile(__nv_bfloat16* tile, const __nv_bfloat16* cache, int64_t row, int tid) {
  const int key = tid >> 2, c0 = tid & 3;
  const __nv_bfloat16* src = cache + (row >= 0 ? row : 0) + c0 * 8;
#pragma unroll
  for (int i = 0; i < 4; ++i) cp_async16(tile + swz(key, c0 + 4 * i), src + 32 * i, row >= 0 ? 16 : 0);
}

}  // namespace pi_attn
