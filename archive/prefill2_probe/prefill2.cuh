// Prefill v2 (FlashInfer-style scaffold). Canonical per-row sequence: keys cut
// into 32-key tiles at absolute positions; within each absolute 512-key fold
// group an online softmax over the group's tiles in ascending order (state
// reset at the group start); group partials folded in ascending order.
// CTA = 4 warps x 32 rows (two 16-row groups per warp share every K/V
// fragment), Q tile and single K/V tiles in 128B-swizzled smem (48 KB), so two
// CTAs fit per SM. The cross-group total lives in an f32 scratch (L2), touched
// once per 512 keys.
#pragma once
#include "pi_attn.cuh"

namespace pi_attn {

constexpr int kTile = 32;                    // keys per online-softmax step
constexpr int kGroupTiles = 16;              // tiles per fold group (512 keys)
constexpr int kTileNS = kTile / 8;           // 4 n-tiles of S per tile
constexpr int kTileKS = kTile / 16;          // 2 key k-steps of P V per tile
constexpr int kP2Warps = 4;

// Swizzled [rows][128] bf16: 16-byte chunk j of row r at r * 16 + (j ^ (r & 7)).
__device__ __forceinline__ int swz(int row, int chunk) { return row * kHeadDim + ((chunk ^ (row & 7)) << 3); }

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

// Cache slots of the 4 keys this thread copies for tile t (-1 = position >= kv_len),
// computed ahead of the copies so the page-table reads are off the critical path.
__device__ __forceinline__ void tile_slots(int (&slot)[kTile / 8], const int32_t* pages, int page_size,
                                           int page_shift, int t, int kv_len, int tid) {
#pragma unroll
  for (int i = 0; i < kTile / 8; ++i) {
    const int pos = t * kTile + (tid >> 4) + 8 * i;
    if (pos >= kv_len) slot[i] = -1;
    else if (page_shift >= 0) slot[i] = (pages[pos >> page_shift] << page_shift) + (pos & (page_size - 1));
    else slot[i] = pages[pos / page_size] * page_size + pos % page_size;
  }
}

// Async load of one kTile x 128 K or V tile (swizzled), 128 threads; keys with
// slot -1 are zero-filled without a global read.
__device__ __forceinline__ void issue_tile2(__nv_bfloat16* tile, const __nv_bfloat16* cache,
                                            const int (&slot)[kTile / 8], int num_kv_heads, int kvh, int tid) {
  const int chunk = tid & 15;
  const int64_t row_stride = (int64_t)num_kv_heads * kHeadDim;
  const __nv_bfloat16* head_base = cache + (int64_t)kvh * kHeadDim + chunk * 8;
#pragma unroll
  for (int i = 0; i < kTile / 8; ++i) {
    const int key = (tid >> 4) + 8 * i;
    cp_async16(tile + swz(key, chunk), slot[i] >= 0 ? head_base + (int64_t)slot[i] * row_stride : cache,
               slot[i] >= 0 ? 16 : 0);
  }
}

// grid = (row tiles of 64 * GR, num_kv_heads, batch * kv_splits), 128 threads;
// each warp owns GR 16-row groups that share every K/V fragment. kv_splits == 1:
// the CTA folds all groups and writes out; the cross-group total lives in
// registers (GR == 1) or in ws_tot [rows, Hq, 128] f32 (GR == 2, touched once per
// group). kv_splits > 1: CTA split s covers a contiguous range of groups and
// writes group partials to ws_o / ws_ml [row][head][group]; prefill_merge_kernel
// folds them.
template <int GR>
constexpr int prefill2_smem_bytes() { return (64 * GR + 2 * kTile) * kHeadDim * 2; }

template <int GR>
__global__ void __launch_bounds__(kP2Warps * 32, 2)
    prefill2_kernel(const __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ out,
                    const __nv_bfloat16* __restrict__ k_cache, const __nv_bfloat16* __restrict__ v_cache,
                    const int32_t* __restrict__ qo_indptr, const int32_t* __restrict__ kv_page_indptr,
                    const int32_t* __restrict__ kv_page_indices, int64_t table_stride,
                    const int32_t* __restrict__ kv_lens, const int32_t* __restrict__ prefix_lens,
                    int num_qo_heads, int num_kv_heads, int page_size, int page_shift, float scale_log2,
                    int kv_splits, float* __restrict__ ws_o, float2* __restrict__ ws_ml, int max_groups,
                    float* __restrict__ ws_tot) {
  constexpr int kRows = kP2Warps * 16 * GR, kWarpRows = 16 * GR;
  extern __shared__ __align__(16) __nv_bfloat16 smem[];
  __nv_bfloat16* sQ = smem;
  __nv_bfloat16* sK = sQ + kRows * kHeadDim;
  __nv_bfloat16* sV = sK + kTile * kHeadDim;
  const int b = blockIdx.z / kv_splits, split = blockIdx.z % kv_splits;
  const int kvh = blockIdx.y, tile = blockIdx.x;
  const int group = num_qo_heads / num_kv_heads;
  const int q0 = qo_indptr[b], q_len = qo_indptr[b + 1] - q0;
  const int rows_total = q_len * group;
  if (tile * kRows >= rows_total) return;
  const int kv_len = kv_lens ? kv_lens[b] : prefix_lens[b] + q_len, prefix = kv_len - q_len;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t4 = lane & 3, mat = lane >> 3, r8 = lane & 7;
  const int32_t* pages = request_pages(kv_page_indices, kv_page_indptr, table_stride, b);
  const int row0 = tile * kRows;
  const int last_pos = prefix + (min(row0 + kRows, rows_total) - 1) / group;
  const int warp_first = row0 + warp * kWarpRows;
  // Position of each of this thread's rows (row group gi, half r); -1 = padding.
  int rpos[GR][2];
#pragma unroll
  for (int gi = 0; gi < GR; ++gi)
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int i = warp_first + gi * 16 + g + r * 8;
      rpos[gi][r] = i < rows_total ? prefix + i / group : -1;
    }
  const int warp_min_pos = warp_first < rows_total ? prefix + warp_first / group : INT_MAX;
  const int warp_max_pos = prefix + (min(warp_first + kWarpRows, rows_total) - 1) / group;
  const int tile_groups = last_pos / (kTile * kGroupTiles) + 1;
  const int per_split = (tile_groups + kv_splits - 1) / kv_splits;
  const int g_begin = split * per_split, g_end = min(tile_groups, g_begin + per_split);
  if (g_begin >= g_end) return;
  const int t_begin = g_begin * kGroupTiles;
  const int t_last = min(g_end * kGroupTiles, last_pos / kTile + 1) - 1;

#pragma unroll
  for (int i = 0; i < kRows * 16 / 128; ++i) {
    const int idx = tid + i * 128, r = idx >> 4, chunk = idx & 15, qi = row0 + r;
    const __nv_bfloat16* src = q;
    int bytes = 0;
    if (qi < rows_total) {
      src = q + (int64_t)(q0 + qi / group) * num_qo_heads * kHeadDim + (kvh * group + qi % group) * kHeadDim + chunk * 8;
      bytes = 16;
    }
    cp_async16(sQ + swz(r, chunk), src, bytes);
  }
  int slot[kTile / 8];
  tile_slots(slot, pages, page_size, page_shift, t_begin, kv_len, tid);
  issue_tile2(sK, k_cache, slot, num_kv_heads, kvh, tid);
  cp_async_commit();  // Q + K(t_begin)
  issue_tile2(sV, v_cache, slot, num_kv_heads, kvh, tid);
  cp_async_commit();

  OnlineState st[GR];
#pragma unroll
  for (int gi = 0; gi < GR; ++gi) st[gi].reset();
  float tM[GR][2], tL[GR][2];
  float tO[GR == 1 ? kNTilesO : 1][4];  // register total (GR == 1)
  bool have_tot = false;

  auto finish_group = [&](int fg, bool last) {
#pragma unroll
    for (int gi = 0; gi < GR; ++gi)
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const int i = warp_first + gi * 16 + g + r * 8;
        const bool valid = i < rows_total;
        const int64_t rh = valid ? (int64_t)(q0 + i / group) * num_qo_heads + kvh * group + i % group : 0;
        if (kv_splits > 1) {
          if (!valid || fg * kTile * kGroupTiles > rpos[gi][r]) continue;
          const int64_t slot = rh * max_groups + fg;
          if (t4 == 0) ws_ml[slot] = make_float2(st[gi].m[r], st[gi].d[r]);
#pragma unroll
          for (int n = 0; n < kNTilesO; ++n)
            *reinterpret_cast<float2*>(ws_o + slot * kHeadDim + n * 8 + t4 * 2) =
                make_float2(st[gi].o[n][2 * r], st[gi].o[n][2 * r + 1]);
          continue;
        }
        FoldScalars f = {0.f, 1.f};
        if (have_tot) f = fold_scalars(tM[gi][r], tL[gi][r], st[gi].m[r], st[gi].d[r]);
        else { tM[gi][r] = st[gi].m[r]; tL[gi][r] = st[gi].d[r]; }
        if constexpr (GR == 1) {
#pragma unroll
          for (int n = 0; n < kNTilesO; ++n)
#pragma unroll
            for (int e = 0; e < 2; ++e)
              tO[n][2 * r + e] = have_tot ? fold_elem(tO[n][2 * r + e], st[gi].o[n][2 * r + e], f) : st[gi].o[n][2 * r + e];
          if (last && valid) {
#pragma unroll
            for (int n = 0; n < kNTilesO; ++n) {
              __nv_bfloat162 ob;
              ob.x = finish(tO[n][2 * r], tL[gi][r]);
              ob.y = finish(tO[n][2 * r + 1], tL[gi][r]);
              *reinterpret_cast<__nv_bfloat162*>(out + rh * kHeadDim + n * 8 + t4 * 2) = ob;
            }
          }
        } else {
          if (!valid) continue;
          float* tot = ws_tot + rh * kHeadDim;
#pragma unroll
          for (int n = 0; n < kNTilesO; ++n) {
            float2 o2 = make_float2(st[gi].o[n][2 * r], st[gi].o[n][2 * r + 1]);
            if (have_tot) {
              const float2 tv = *reinterpret_cast<const float2*>(tot + n * 8 + t4 * 2);
              o2.x = fold_elem(tv.x, o2.x, f);
              o2.y = fold_elem(tv.y, o2.y, f);
            }
            if (last) {
              __nv_bfloat162 ob;
              ob.x = finish(o2.x, tL[gi][r]);
              ob.y = finish(o2.y, tL[gi][r]);
              *reinterpret_cast<__nv_bfloat162*>(out + rh * kHeadDim + n * 8 + t4 * 2) = ob;
            } else {
              *reinterpret_cast<float2*>(tot + n * 8 + t4 * 2) = o2;
            }
          }
        }
      }
    have_tot = true;
#pragma unroll
    for (int gi = 0; gi < GR; ++gi) st[gi].reset();
  };

  for (int t = t_begin; t <= t_last; ++t) {
    if (t + 1 <= t_last) tile_slots(slot, pages, page_size, page_shift, t + 1, kv_len, tid);
    cp_async_wait<1>();  // K(t) (and Q) landed
    __syncthreads();
    const int key0 = t * kTile;
    const bool active = key0 <= warp_max_pos;
    uint32_t pa[GR][kTileKS][4];
    if (active) {
      float s[GR][kTileNS][4];
#pragma unroll
      for (int gi = 0; gi < GR; ++gi)
#pragma unroll
        for (int n = 0; n < kTileNS; ++n) s[gi][n][0] = s[gi][n][1] = s[gi][n][2] = s[gi][n][3] = 0.f;
#pragma unroll
      for (int ks = 0; ks < kKStepsQK; ++ks) {
        uint32_t a[GR][4];
#pragma unroll
        for (int gi = 0; gi < GR; ++gi)
          ldmatrix_x4(a[gi], sQ + swz(warp * kWarpRows + gi * 16 + (mat & 1) * 8 + r8, ks * 2 + (mat >> 1)));
#pragma unroll
        for (int n = 0; n < kTileNS; n += 2) {
          uint32_t bk[4];
          ldmatrix_x4(bk, sK + swz((n + (mat >> 1)) * 8 + r8, ks * 2 + (mat & 1)));
#pragma unroll
          for (int gi = 0; gi < GR; ++gi) {
            mma_bf16(s[gi][n], a[gi], bk[0], bk[1]);
            mma_bf16(s[gi][n + 1], a[gi], bk[2], bk[3]);
          }
        }
      }
      const bool all_vis = key0 + kTile - 1 <= warp_min_pos;
#pragma unroll
      for (int gi = 0; gi < GR; ++gi) {
#pragma unroll
        for (int n = 0; n < kTileNS; ++n)
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int key = key0 + n * 8 + t4 * 2 + (i & 1);
            s[gi][n][i] = (all_vis || key <= rpos[gi][i >> 1]) ? mul(s[gi][n][i], scale_log2) : -INFINITY;
          }
        online_step(st[gi], s[gi], pa[gi]);
      }
    }
    __syncthreads();  // all warps done with K(t)
    if (t + 1 <= t_last) issue_tile2(sK, k_cache, slot, num_kv_heads, kvh, tid);
    cp_async_commit();
    cp_async_wait<1>();  // V(t) landed
    __syncthreads();
    if (active) {
#pragma unroll
      for (int j = 0; j < kTileKS; ++j)
#pragma unroll
        for (int n = 0; n < kNTilesO; n += 2) {
          uint32_t bv[4];
          ldmatrix_x4_trans(bv, sV + swz(j * 16 + (mat & 1) * 8 + r8, n + (mat >> 1)));
#pragma unroll
          for (int gi = 0; gi < GR; ++gi) {
            mma_bf16(st[gi].o[n], pa[gi][j], bv[0], bv[1]);
            mma_bf16(st[gi].o[n + 1], pa[gi][j], bv[2], bv[3]);
          }
        }
    }
    __syncthreads();  // all warps done with V(t)
    if (t + 1 <= t_last) issue_tile2(sV, v_cache, slot, num_kv_heads, kvh, tid);
    cp_async_commit();
    if (t % kGroupTiles == kGroupTiles - 1 || t == t_last) finish_group(t / kGroupTiles, t == t_last && kv_splits == 1);
  }
  cp_async_wait<0>();
}

// Decode v2. grid = (max_groups, num_kv_heads, batch), 4 warps. One CTA runs the
// canonical online sequence of fold group fg for the GQA rows of (b, kvh) and
// writes the group partial (m, d, o) at ws row b * max_groups + fg; the decode
// merge folds the groups. The tile is split across warps without changing any
// row's sequence: warp w computes S n-tile w (each element's head_dim k-steps in
// ascending order) and O n-tiles [4w, 4w + 4) (each element: fma rescale, then
// the two key k-steps in ascending order); the tile max is exact under any
// split; P is rounded elementwise; warp 0 sums P from smem in exactly
// online_step's order.
constexpr int kD2Warps = 4;
constexpr int decode2_smem_bytes() { return (2 * kTile * kHeadDim + 16 * (kTile + 8)) * 2 + (kD2Warps * 16 + 16) * 4; }

__global__ void __launch_bounds__(kD2Warps * 32)
    decode2_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k_cache,
                   const __nv_bfloat16* __restrict__ v_cache, const int32_t* __restrict__ kv_page_indptr,
                   const int32_t* __restrict__ kv_page_indices, int64_t table_stride,
                   const int32_t* __restrict__ kv_lens, float* __restrict__ ws_o, float2* __restrict__ ws_ml,
                   int max_groups, int num_qo_heads, int num_kv_heads, int page_size, int page_shift,
                   float scale_log2) {
  static_assert(kTileNS == kD2Warps && kNTilesO % kD2Warps == 0, "one S n-tile per warp");
  constexpr int kON = kNTilesO / kD2Warps, kPStride = kTile + 8;
  extern __shared__ __align__(16) __nv_bfloat16 smem[];
  __nv_bfloat16* sK = smem;
  __nv_bfloat16* sV = sK + kTile * kHeadDim;
  __nv_bfloat16* sP = sV + kTile * kHeadDim;  // [16 rows][kTile] rounded P
  float* sMax = reinterpret_cast<float*>(sP + 16 * kPStride);  // [warps][16]
  float* sD = sMax + kD2Warps * 16;                             // [16] (unused slot kept for alignment)
  const int fg = blockIdx.x, kvh = blockIdx.y, b = blockIdx.z;
  const int kv_len = kv_lens[b], pos = kv_len - 1;
  const int n_tiles = (kv_len + kTile - 1) / kTile;
  const int t_begin = fg * kGroupTiles;
  if (t_begin >= n_tiles) return;
  const int t_end = min(n_tiles, t_begin + kGroupTiles);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t4 = lane & 3, mat = lane >> 3, r8 = lane & 7;
  const int group = num_qo_heads / num_kv_heads;
  const int32_t* pages = request_pages(kv_page_indices, kv_page_indptr, table_stride, b);
  int slot[kTile / 8];
  tile_slots(slot, pages, page_size, page_shift, t_begin, kv_len, tid);
  issue_tile2(sK, k_cache, slot, num_kv_heads, kvh, tid);
  cp_async_commit();
  issue_tile2(sV, v_cache, slot, num_kv_heads, kvh, tid);
  cp_async_commit();
  QFrag qf;
  const int64_t q_stride = (int64_t)num_qo_heads * kHeadDim;
  load_q(qf, [&](int r) -> const __nv_bfloat16* {
           return r < group ? q + b * q_stride + (kvh * group + r) * kHeadDim : nullptr;
         }, lane);
  float M[2] = {-INFINITY, -INFINITY}, D = 0.f, Dh = 0.f;  // D/Dh: rows g, g+8 (warp 0)
  float O[kON][4];
#pragma unroll
  for (int n = 0; n < kON; ++n) O[n][0] = O[n][1] = O[n][2] = O[n][3] = 0.f;

  for (int t = t_begin; t < t_end; ++t) {
    if (t + 1 < t_end) tile_slots(slot, pages, page_size, page_shift, t + 1, kv_len, tid);
    cp_async_wait<1>();  // K(t) landed
    __syncthreads();
    float s[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int ks = 0; ks < kKStepsQK; ++ks) {
      uint32_t bk[4];
      ldmatrix_x4(bk, sK + swz(warp * 8 + r8, ks * 2 + (mat & 1)));
      mma_bf16(s, qf.a[ks], bk[0], bk[1]);
    }
    float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int row = g + (i >> 1) * 8, key = t * kTile + warp * 8 + t4 * 2 + (i & 1);
      s[i] = (row < group && key <= pos) ? mul(s[i], scale_log2) : -INFINITY;
      mx[i >> 1] = fmaxf(mx[i >> 1], s[i]);
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
      mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
    }
    if (t4 == 0) {
      sMax[warp * 16 + g] = mx[0];
      sMax[warp * 16 + g + 8] = mx[1];
    }
    __syncthreads();  // all warps done with K(t), maxima published
    if (t + 1 < t_end) issue_tile2(sK, k_cache, slot, num_kv_heads, kvh, tid);
    cp_async_commit();
    float mm[2], scale[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      float tmax = -INFINITY;
#pragma unroll
      for (int w = 0; w < kD2Warps; ++w) tmax = fmaxf(tmax, sMax[w * 16 + g + r * 8]);
      const float mn = fmaxf(M[r], tmax);
      mm[r] = mn == -INFINITY ? 0.f : mn;
      scale[r] = ex2(sub(M[r], mm[r]));
      M[r] = mn;
    }
    {
      float p[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) p[i] = round_bf16(ex2(sub(s[i], mm[i >> 1])));
      const int key = warp * 8 + t4 * 2;
      *reinterpret_cast<uint32_t*>(sP + g * kPStride + key) = pack_bf16(p[0], p[1]);
      *reinterpret_cast<uint32_t*>(sP + (g + 8) * kPStride + key) = pack_bf16(p[2], p[3]);
    }
    cp_async_wait<1>();  // V(t) landed
    __syncthreads();
    if (warp == 0) {  // d in exactly online_step's order
      float ls[2] = {0.f, 0.f};
#pragma unroll
      for (int n = 0; n < kTileNS; ++n) {
        const __nv_bfloat162 p0 = *reinterpret_cast<const __nv_bfloat162*>(sP + g * kPStride + n * 8 + t4 * 2);
        const __nv_bfloat162 p1 = *reinterpret_cast<const __nv_bfloat162*>(sP + (g + 8) * kPStride + n * 8 + t4 * 2);
        ls[0] = add(ls[0], __low2float(p0));
        ls[0] = add(ls[0], __high2float(p0));
        ls[1] = add(ls[1], __low2float(p1));
        ls[1] = add(ls[1], __high2float(p1));
      }
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        ls[r] = add(ls[r], __shfl_xor_sync(0xffffffffu, ls[r], 1));
        ls[r] = add(ls[r], __shfl_xor_sync(0xffffffffu, ls[r], 2));
      }
      D = __fmaf_rn(D, scale[0], ls[0]);
      Dh = __fmaf_rn(Dh, scale[1], ls[1]);
    }
    uint32_t pa[kTileKS][4];
#pragma unroll
    for (int j = 0; j < kTileKS; ++j) {
      const int k0 = j * 16 + t4 * 2;
      pa[j][0] = *reinterpret_cast<const uint32_t*>(sP + g * kPStride + k0);
      pa[j][1] = *reinterpret_cast<const uint32_t*>(sP + (g + 8) * kPStride + k0);
      pa[j][2] = *reinterpret_cast<const uint32_t*>(sP + g * kPStride + k0 + 8);
      pa[j][3] = *reinterpret_cast<const uint32_t*>(sP + (g + 8) * kPStride + k0 + 8);
    }
#pragma unroll
    for (int n = 0; n < kON; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) O[n][i] = __fmaf_rn(O[n][i], scale[i >> 1], 0.f);
#pragma unroll
    for (int j = 0; j < kTileKS; ++j)
#pragma unroll
      for (int n = 0; n < kON; n += 2) {
        uint32_t bv[4];
        ldmatrix_x4_trans(bv, sV + swz(j * 16 + (mat & 1) * 8 + r8, warp * kON + n + (mat >> 1)));
        mma_bf16(O[n], pa[j], bv[0], bv[1]);
        mma_bf16(O[n + 1], pa[j], bv[2], bv[3]);
      }
    __syncthreads();  // all warps done with V(t) and P
    if (t + 1 < t_end) issue_tile2(sV, v_cache, slot, num_kv_heads, kvh, tid);
    cp_async_commit();
  }
  cp_async_wait<0>();
  const int64_t slot0 = ((int64_t)b * max_groups + fg) * num_qo_heads + kvh * group + g;
  const int64_t slot1 = slot0 + 8;
#pragma unroll
  for (int n = 0; n < kON; ++n) {
    const int col = (warp * kON + n) * 8 + t4 * 2;
    if (g < group) *reinterpret_cast<float2*>(ws_o + slot0 * kHeadDim + col) = make_float2(O[n][0], O[n][1]);
    if (g + 8 < group) *reinterpret_cast<float2*>(ws_o + slot1 * kHeadDim + col) = make_float2(O[n][2], O[n][3]);
  }
  if (warp == 0 && t4 == 0) {
    if (g < group) ws_ml[slot0] = make_float2(M[0], D);
    if (g + 8 < group) ws_ml[slot1] = make_float2(M[1], Dh);
  }
  (void)sD;
}

}  // namespace pi_attn
