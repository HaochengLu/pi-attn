// PI-Attn kernels: paged prefill / extend, decode (one CTA per fold group), merges.
// KV cache layout per layer: [num_pages, page_size, num_kv_heads, 128] bf16.
#include <algorithm>
#include <mutex>
#include <map>
#include <queue>
#include <tuple>
#include <vector>

#include "pi_attn.cuh"

namespace pi_attn {

// Pages of request b: CSR (kv_page_indptr) or, when kv_page_indptr is null, row b
// of a dense page table with row stride table_stride.
__device__ __forceinline__ const int32_t* request_pages(const int32_t* kv_page_indices,
                                                       const int32_t* kv_page_indptr,
                                                       int64_t table_stride, int b) {
  return kv_page_indices + (kv_page_indptr ? (int64_t)kv_page_indptr[b] : (int64_t)b * table_stride);
}

// Prefill / extend. grid = (row tiles of 64, num_kv_heads, batch * kv_splits), 4
// warps of 16 rows. The rows of a CTA are the (token, query head) pairs of one kv
// head's GQA group, token-major (row r: token r / group, head kv_head * group +
// r % group), so each K/V tile is loaded once per kv head. Token i of request b
// sits at position kv_len[b] - q_len + i. kv_splits == 1: the CTA folds every
// group into a running total and writes out; otherwise CTA split s covers a
// contiguous range of groups and writes, per row and group, the group partial
// (ws_o / ws_ml [row][head][group]) that prefill_merge_kernel folds.
//
// Software pipeline: iteration t computes S(t + 1) = Q K(t + 1)^T while it runs
// tile t's online step and P V(t), so the S MMA chains overlap the softmax; S(t + 1)
// depends on nothing of tile t, and every row still runs its online steps in tile
// order. K and V each have two smem slots; iteration t loads K(t + 2) and V(t + 1),
// a full iteration ahead, behind one barrier. Every warp runs every tile of its
// range: a fully masked online step is an exact no-op (see online_step).
//
// QREG (sm_80): Q fragments live in registers and the fp32 folded totals (touched
// once per fold group) in smem: 80 KB, two CTAs per SM on sm_80's 164 KB. Otherwise
// Q is read from smem per tile and the totals sit in registers: 48 KB, which fits
// two CTAs in sm_89's 100 KB. Both run every row's identical sequence.
constexpr int kPrefillWarps = 4, kPrefillRows = kPrefillWarps * 16;
template <bool QREG>
constexpr int prefill_smem_bytes() {
  return (kPrefillRows + 4 * kTile) * kHeadDim * 2 + (QREG ? kPrefillRows * kHeadDim * 4 : 0);
}
// Which variant a device runs: QREG on sm_80, where it measures faster
// (results/a800/kernels/prefill_configs.txt); elsewhere the 48 KB one, the only one
// of which two CTAs fit in sm_89's 100 KB.
inline bool prefill_qreg(int cc_major, int cc_minor) { return cc_major == 8 && cc_minor == 0; }

template <bool QREG>
__global__ void __launch_bounds__(kPrefillWarps * 32, 2)
    prefill_kernel(const __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ out,
                    const __nv_bfloat16* __restrict__ k_cache, const __nv_bfloat16* __restrict__ v_cache,
                    const int32_t* __restrict__ qo_indptr, const int32_t* __restrict__ kv_page_indptr,
                    const int32_t* __restrict__ kv_page_indices, int64_t table_stride,
                    const int32_t* __restrict__ kv_lens, const int32_t* __restrict__ prefix_lens,
                    int num_qo_heads, int num_kv_heads, int page_size, int page_shift, float scale_log2,
                    int kv_splits, float* __restrict__ ws_o, float2* __restrict__ ws_ml, int max_groups) {
  constexpr int kRows = kPrefillRows;
  extern __shared__ __align__(16) __nv_bfloat16 smem[];
  __nv_bfloat16* sQ = smem;
  __nv_bfloat16* sK = sQ + kRows * kHeadDim;  // slots 0, 1 (kTile x 128 each)
  __nv_bfloat16* sV = sK + 2 * kTile * kHeadDim;  // slots 0, 1
  float* sTot = reinterpret_cast<float*>(sV + 2 * kTile * kHeadDim);  // QREG: [warp][n][i][lane]
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
  const int warp_first = row0 + warp * 16;
  // Position of this thread's rows g, g + 8; -1 = padding.
  int rpos[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int i = warp_first + g + r * 8;
    rpos[r] = i < rows_total ? prefix + i / group : -1;
  }
  const int tile_groups = last_pos / kGroupKeys + 1;
  const int per_split = (tile_groups + kv_splits - 1) / kv_splits;
  const int g_begin = split * per_split, g_end = min(tile_groups, g_begin + per_split);
  if (g_begin >= g_end) return;
  const int t_begin = g_begin * kGroupTiles;
  const int t_last = min(g_end * kGroupTiles, last_pos / kTile + 1) - 1;

  // ldmatrix addresses at k-step 0; a k-step XORs its chunk offset (see swz_bytes:
  // the kernel has no static smem, so the dynamic block and every tile in it are
  // 256-byte aligned).
  const uint32_t q_addr = smem_u32(sQ) + swz_bytes(warp * 16 + (mat & 1) * 8 + r8, mat >> 1);
  const uint32_t k_addr = smem_u32(sK) + swz_bytes((mat >> 1) * 8 + r8, mat & 1);
  const uint32_t v_addr = smem_u32(sV) + swz_bytes((mat & 1) * 8 + r8, mat >> 1);
  constexpr uint32_t kSlotBytes = kTile * kHeadDim * 2;
  // S = Q K^T of the tile in K slot `slot`, each element's k-steps in ascending order.
  uint32_t qa[QREG ? kKStepsQK : 1][4];
  auto qk = [&](float (&s)[kTileNS][4], int slot) {
#pragma unroll
    for (int n = 0; n < kTileNS; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < kKStepsQK; ++ks) {
      uint32_t a[4];
      if constexpr (QREG) {
#pragma unroll
        for (int i = 0; i < 4; ++i) a[i] = qa[ks][i];
      } else {
        ldmatrix_x4(a, q_addr ^ (ks << 5));
      }
#pragma unroll
      for (int n = 0; n < kTileNS; n += 2) {
        uint32_t bk[4];
        ldmatrix_x4(bk, ((k_addr + slot * kSlotBytes) ^ (ks << 5)) + n * 8 * kHeadDim * 2);
        mma_bf16(s[n], a, bk[0], bk[1]);
        mma_bf16(s[n + 1], a, bk[2], bk[3]);
      }
    }
  };

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
  // Tile t uses K slot and V slot t & 1 (t_begin is even). row1 / row2: this
  // thread's cache rows of tiles t + 1 and t + 2, each read an iteration before use.
  auto row_of = [&](int t) { return tile_row(pages, page_size, page_shift, t, kv_len, num_kv_heads, kvh, tid); };
  int64_t row1 = row_of(t_begin);
  issue_tile(sK, k_cache, row1, tid);
  cp_async_commit();  // Q + K(t_begin)
  issue_tile(sV, v_cache, row1, tid);
  row1 = row_of(t_begin + 1);
  int64_t row2 = row_of(t_begin + 2);
  if (t_begin + 1 <= t_last) issue_tile(sK + kTile * kHeadDim, k_cache, row1, tid);
  cp_async_commit();  // V(t_begin) + K(t_begin + 1)

  // st: the current group; the folded total of the finished groups is (tot_m,
  // tot_d, tot_o): o in registers, or (QREG) in this thread's smem slots.
  OnlineState st;
  st.reset();
  float tot_m[2] = {-INFINITY, -INFINITY}, tot_d[2] = {0.f, 0.f}, tot_reg[QREG ? 1 : kNTilesO][4];
  auto tot_o = [&](int n, int i) -> float& {
    if constexpr (QREG) return sTot[((warp * kNTilesO + n) * 4 + i) * 32 + lane];
    else return tot_reg[n][i];
  };
#pragma unroll
  for (int n = 0; n < kNTilesO; ++n)
#pragma unroll
    for (int i = 0; i < 4; ++i) tot_o(n, i) = 0.f;

  // Group fg is finished: fold it into the total (kv_splits == 1; on the last
  // group also normalise and write out) or write it as a group partial.
  auto finish_group = [&](int fg, bool last) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int i = warp_first + g + r * 8;
      if (kv_splits > 1) {
        if (i >= rows_total || fg * kGroupKeys > rpos[r]) continue;
        const int64_t slot = ((int64_t)(q0 + i / group) * num_qo_heads + kvh * group + i % group) * max_groups + fg;
        if (t4 == 0) ws_ml[slot] = make_float2(st.m[r], st.d[r]);
#pragma unroll
        for (int n = 0; n < kNTilesO; ++n)
          *reinterpret_cast<float2*>(ws_o + slot * kHeadDim + n * 8 + t4 * 2) = make_float2(st.o[n][2 * r], st.o[n][2 * r + 1]);
        continue;
      }
      const FoldScalars f = fold_scalars(tot_m[r], tot_d[r], st.m[r], st.d[r]);
#pragma unroll
      for (int n = 0; n < kNTilesO; ++n)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          float& o = tot_o(n, 2 * r + e);
          o = fold_elem(o, st.o[n][2 * r + e], f);
        }
      if (last && i < rows_total) {
        __nv_bfloat16* dst = out + ((int64_t)(q0 + i / group) * num_qo_heads + kvh * group + i % group) * kHeadDim;
#pragma unroll
        for (int n = 0; n < kNTilesO; ++n) {
          __nv_bfloat162 ob;
          ob.x = finish(tot_o(n, 2 * r), tot_d[r]);
          ob.y = finish(tot_o(n, 2 * r + 1), tot_d[r]);
          *reinterpret_cast<__nv_bfloat162*>(dst + n * 8 + t4 * 2) = ob;
        }
      }
    }
    st.reset();
  };

  // One pipelined iteration: s holds S(t) on entry; sn receives S(t + 1).
  auto step = [&](int t, float (&s)[kTileNS][4], float (&sn)[kTileNS][4]) {
    const int slot = t & 1;
    cp_async_wait<0>();  // V(t), K(t + 1)
    __syncthreads();     // ... for all threads; everyone is done with V(t - 1) and K(t)
    if (t + 1 <= t_last) issue_tile(sV + (slot ^ 1) * kTile * kHeadDim, v_cache, row1, tid);
    if (t + 2 <= t_last) issue_tile(sK + slot * kTile * kHeadDim, k_cache, row2, tid);
    cp_async_commit();
    row1 = row2;
    row2 = row_of(t + 3);
    qk(sn, slot ^ 1);  // past t_last: stale slot data, never used
    const int key0 = t * kTile;
#pragma unroll
    for (int n = 0; n < kTileNS; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int key = key0 + n * 8 + t4 * 2 + (i & 1);
        s[n][i] = key <= rpos[i >> 1] ? mul(s[n][i], scale_log2) : -INFINITY;
      }
    uint32_t pa[kTileKS][4];
    online_step(st, s, pa);
#pragma unroll
    for (int j = 0; j < kTileKS; ++j)
#pragma unroll
      for (int n = 0; n < kNTilesO; n += 2) {
        uint32_t bv[4];
        ldmatrix_x4_trans(bv, ((v_addr + slot * kSlotBytes) ^ (n << 4)) + j * 16 * kHeadDim * 2);
        mma_bf16(st.o[n], pa[j], bv[0], bv[1]);
        mma_bf16(st.o[n + 1], pa[j], bv[2], bv[3]);
      }
    if (t % kGroupTiles == kGroupTiles - 1 || t == t_last) finish_group(t / kGroupTiles, t == t_last);
  };

  float sa[kTileNS][4], sb[kTileNS][4];  // S of the current and next tile, alternating
  cp_async_wait<1>();  // Q, K(t_begin)
  __syncthreads();
  if constexpr (QREG) {
#pragma unroll
    for (int ks = 0; ks < kKStepsQK; ++ks) ldmatrix_x4(qa[ks], q_addr ^ (ks << 5));
  }
  qk(sa, 0);
  for (int t = t_begin; t <= t_last; t += 2) {
    step(t, sa, sb);
    if (t + 1 <= t_last) step(t + 1, sb, sa);
  }
  cp_async_wait<0>();
}

// Warp-pair prefill (sm_80). Same grid, rows and canonical sequence as prefill_kernel,
// with each 32-row half of the CTA run by a pair of warps: warp A holds Q in registers
// and runs S and online_softmax for both 16-row m-tiles; warp B holds o for them and
// runs the rescale and P V. Every K / V fragment loaded from smem then feeds two
// MMAs (prefill_kernel: one), halving smem reads per MMA, which bound the 16-row-warp
// kernel on sm_80's full-rate fp32-accumulate MMAs. A works one tile ahead of B: in
// step k, A computes S(k + 1) and the online step of tile k, writing P(k) and the
// row scales to smem slot k & 1, while B rescales and multiplies tile k - 1 from slot
// (k - 1) & 1; the step's barrier orders the hand-off. At a group's last tile A
// folds (m, d) into the totals and passes each row's fold scalars (and, at the end,
// the total d) to B, which folds o into the fp32 totals in smem. Slots: K and V two
// each (K(k + 2) and V(k) load in step k); Q is staged in the V slots before V(t_begin).
constexpr int kPairRows = 32;
constexpr int prefill_pair_smem_bytes() {
  return 4 * kTile * kHeadDim * 2                              // K, V slots
         + 2 * 2 * (2 * kTileKS * 4 * 32) * 4                  // P: [slot][pair][m-tile][k-step][i][lane]
         + 2 * kPrefillRows * 4                                // row scales: [slot][row]
         + 3 * kPrefillRows * 4                                // fold alpha, beta, total d: [3][row]
         + kPrefillRows * kHeadDim * 4;                        // fp32 o totals: [pair][m-tile][n][i][lane]
}

__global__ void __launch_bounds__(kPrefillWarps * 32, 2)
    prefill_pair_kernel(const __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ out,
                        const __nv_bfloat16* __restrict__ k_cache, const __nv_bfloat16* __restrict__ v_cache,
                        const int32_t* __restrict__ qo_indptr, const int32_t* __restrict__ kv_page_indptr,
                        const int32_t* __restrict__ kv_page_indices, int64_t table_stride,
                        const int32_t* __restrict__ kv_lens, const int32_t* __restrict__ prefix_lens,
                        int num_qo_heads, int num_kv_heads, int page_size, int page_shift, float scale_log2,
                        int kv_splits, float* __restrict__ ws_o, float2* __restrict__ ws_ml, int max_groups) {
  constexpr int kRows = kPrefillRows, kMT = kPairRows / 16;
  extern __shared__ __align__(16) __nv_bfloat16 smem[];
  __nv_bfloat16* sK = smem;                       // slots 0, 1
  __nv_bfloat16* sV = sK + 2 * kTile * kHeadDim;  // slots 0, 1; Q staging (64 rows = both slots) first
  uint32_t* sP = reinterpret_cast<uint32_t*>(sV + 2 * kTile * kHeadDim);
  float* sScale = reinterpret_cast<float*>(sP + 2 * 2 * kMT * kTileKS * 4 * 32);
  float* sFold = sScale + 2 * kRows;
  float* sTot = sFold + 3 * kRows;
  const int b = blockIdx.z / kv_splits, split = blockIdx.z % kv_splits;
  const int kvh = blockIdx.y, tile = blockIdx.x;
  const int group = num_qo_heads / num_kv_heads;
  const int q0 = qo_indptr[b], q_len = qo_indptr[b + 1] - q0;
  const int rows_total = q_len * group;
  if (tile * kRows >= rows_total) return;
  const int kv_len = kv_lens ? kv_lens[b] : prefix_lens[b] + q_len, prefix = kv_len - q_len;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t4 = lane & 3, mat = lane >> 3, r8 = lane & 7;
  const int pair = warp >> 1;
  const bool is_a = (warp & 1) == 0;
  const int32_t* pages = request_pages(kv_page_indices, kv_page_indptr, table_stride, b);
  const int row0 = tile * kRows;
  const int last_pos = prefix + (min(row0 + kRows, rows_total) - 1) / group;
  const int pair_first = row0 + pair * kPairRows;
  // CTA row (0..63) and position of this thread's rows g, g + 8 of m-tile mt; -1 = padding.
  auto lrow = [&](int mt, int r) { return pair * kPairRows + mt * 16 + g + r * 8; };
  int rpos[kMT][2];
#pragma unroll
  for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int i = pair_first + mt * 16 + g + r * 8;
      rpos[mt][r] = i < rows_total ? prefix + i / group : -1;
    }
  const int tile_groups = last_pos / kGroupKeys + 1;
  const int per_split = (tile_groups + kv_splits - 1) / kv_splits;
  const int g_begin = split * per_split, g_end = min(tile_groups, g_begin + per_split);
  if (g_begin >= g_end) return;
  const int t_begin = g_begin * kGroupTiles;
  const int t_last = min(g_end * kGroupTiles, last_pos / kTile + 1) - 1;
  constexpr uint32_t kSlotBytes = kTile * kHeadDim * 2;

#pragma unroll
  for (int i = 0; i < kRows * 16 / 128; ++i) {
    const int idx = tid + i * 128, r = idx >> 4, chunk = idx & 15, qi = row0 + r;
    const __nv_bfloat16* src = q;
    int bytes = 0;
    if (qi < rows_total) {
      src = q + (int64_t)(q0 + qi / group) * num_qo_heads * kHeadDim + (kvh * group + qi % group) * kHeadDim + chunk * 8;
      bytes = 16;
    }
    cp_async16(sV + swz(r, chunk), src, bytes);
  }
  // Tile t uses K slot and V slot t & 1 (t_begin is even). rows[i]: this thread's cache
  // row of tile k + i, each read a step before its first use.
  auto row_of = [&](int t) { return tile_row(pages, page_size, page_shift, t, kv_len, num_kv_heads, kvh, tid); };
  int64_t rows[3] = {row_of(t_begin), row_of(t_begin + 1), row_of(t_begin + 2)};
  issue_tile(sK, k_cache, rows[0], tid);
  if (t_begin + 1 <= t_last) issue_tile(sK + kTile * kHeadDim, k_cache, rows[1], tid);
  cp_async_commit();  // Q, K(t_begin), K(t_begin + 1)
  // Step k's barrier and loads, the same in both roles: K(k + 2) into the slot K(k) left
  // in step k - 1, V(k) into the slot V(k - 2) left in step k - 1.
  auto step_begin = [&](int k) {
    cp_async_wait<0>();  // K(k + 1), V(k - 1)
    __syncthreads();
    if (k + 2 <= t_last) issue_tile(sK + (k & 1) * kTile * kHeadDim, k_cache, rows[2], tid);
    if (k <= t_last) issue_tile(sV + (k & 1) * kTile * kHeadDim, v_cache, rows[0], tid);
    cp_async_commit();
    rows[0] = rows[1];
    rows[1] = rows[2];
    rows[2] = row_of(k + 3);
  };
  const auto group_end = [&](int t) { return t % kGroupTiles == kGroupTiles - 1 || t == t_last; };

  if (is_a) {
    // Warp A: Q fragments of both m-tiles in registers; S one tile ahead.
    uint32_t qa[kMT][kKStepsQK][4];
    const uint32_t q_addr = smem_u32(sV) + swz_bytes(pair * kPairRows + (mat & 1) * 8 + r8, mat >> 1);
    const uint32_t k_addr = smem_u32(sK) + swz_bytes((mat >> 1) * 8 + r8, mat & 1);
    auto qk = [&](float (&s)[kMT][kTileNS][4], int slot) {
#pragma unroll
      for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
        for (int n = 0; n < kTileNS; ++n) s[mt][n][0] = s[mt][n][1] = s[mt][n][2] = s[mt][n][3] = 0.f;
#pragma unroll
      for (int ks = 0; ks < kKStepsQK; ++ks)
#pragma unroll
        for (int n = 0; n < kTileNS; n += 2) {
          uint32_t bk[4];
          ldmatrix_x4(bk, ((k_addr + slot * kSlotBytes) ^ (ks << 5)) + n * 8 * kHeadDim * 2);
#pragma unroll
          for (int mt = 0; mt < kMT; ++mt) {
            mma_bf16(s[mt][n], qa[mt][ks], bk[0], bk[1]);
            mma_bf16(s[mt][n + 1], qa[mt][ks], bk[2], bk[3]);
          }
        }
    };
    float m[kMT][2], d[kMT][2], tot_m[kMT][2], tot_d[kMT][2];
#pragma unroll
    for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
      for (int r = 0; r < 2; ++r) m[mt][r] = tot_m[mt][r] = -INFINITY, d[mt][r] = tot_d[mt][r] = 0.f;
    // Tile k: S in s, S(k + 1) into sn.
    auto a_step = [&](int k, float (&s)[kMT][kTileNS][4], float (&sn)[kMT][kTileNS][4]) {
      step_begin(k);
      if (k > t_last) return;
      qk(sn, (k + 1) & 1);  // past t_last: stale slot data, never used
#pragma unroll
      for (int mt = 0; mt < kMT; ++mt) {
#pragma unroll
        for (int n = 0; n < kTileNS; ++n)
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int key = k * kTile + n * 8 + t4 * 2 + (i & 1);
            s[mt][n][i] = key <= rpos[mt][i >> 1] ? mul(s[mt][n][i], scale_log2) : -INFINITY;
          }
        uint32_t pa[kTileKS][4];
        float scale[2];
        online_softmax(m[mt], d[mt], s[mt], pa, scale);
        uint32_t* p = sP + ((((k & 1) * 2 + pair) * kMT + mt) * kTileKS * 4) * 32 + lane;
#pragma unroll
        for (int j = 0; j < kTileKS; ++j)
#pragma unroll
          for (int i = 0; i < 4; ++i) p[(j * 4 + i) * 32] = pa[j][i];
        if (t4 == 0)
#pragma unroll
          for (int r = 0; r < 2; ++r) sScale[(k & 1) * kRows + lrow(mt, r)] = scale[r];
      }
      if (!group_end(k)) return;
      // Group k / kGroupTiles is finished: its partial (split mode) or the fold scalars.
#pragma unroll
      for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const int i = pair_first + mt * 16 + g + r * 8;
          if (kv_splits > 1) {
            if (t4 == 0 && i < rows_total && (k / kGroupTiles) * kGroupKeys <= rpos[mt][r])
              ws_ml[((int64_t)(q0 + i / group) * num_qo_heads + kvh * group + i % group) * max_groups + k / kGroupTiles] =
                  make_float2(m[mt][r], d[mt][r]);
          } else {
            const FoldScalars f = fold_scalars(tot_m[mt][r], tot_d[mt][r], m[mt][r], d[mt][r]);
            if (t4 == 0) {
              sFold[lrow(mt, r)] = f.alpha;
              sFold[kRows + lrow(mt, r)] = f.beta;
              sFold[2 * kRows + lrow(mt, r)] = tot_d[mt][r];
            }
          }
          m[mt][r] = -INFINITY;
          d[mt][r] = 0.f;
        }
    };
    float sa[kMT][kTileNS][4], sb[kMT][kTileNS][4];
    cp_async_wait<0>();  // Q, K(t_begin), K(t_begin + 1)
    __syncthreads();
#pragma unroll
    for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
      for (int ks = 0; ks < kKStepsQK; ++ks) ldmatrix_x4(qa[mt][ks], (q_addr + mt * 16 * kHeadDim * 2) ^ (ks << 5));
    qk(sa, 0);
    for (int k = t_begin; k <= t_last + 1; k += 2) {
      a_step(k, sa, sb);
      if (k + 1 <= t_last + 1) a_step(k + 1, sb, sa);
    }
  } else {
    // Warp B: o of both m-tiles in registers, tile k - 1 in step k; fp32 totals in smem.
    const uint32_t v_addr = smem_u32(sV) + swz_bytes((mat & 1) * 8 + r8, mat >> 1);
    float o[kMT][kNTilesO][4];
#pragma unroll
    for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
      for (int n = 0; n < kNTilesO; ++n) o[mt][n][0] = o[mt][n][1] = o[mt][n][2] = o[mt][n][3] = 0.f;
    auto tot = [&](int mt, int n, int i) -> float& { return sTot[(((pair * kMT + mt) * kNTilesO + n) * 4 + i) * 32 + lane]; };
#pragma unroll
    for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
      for (int n = 0; n < kNTilesO; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) tot(mt, n, i) = 0.f;
    cp_async_wait<0>();
    __syncthreads();  // A's Q read
    for (int k = t_begin; k <= t_last + 1; ++k) {
      step_begin(k);
      const int t = k - 1;
      if (t < t_begin) continue;
#pragma unroll
      for (int mt = 0; mt < kMT; ++mt) {
        float scale[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) scale[r] = sScale[(t & 1) * kRows + lrow(mt, r)];
        rescale_o(o[mt], scale);
      }
      uint32_t pa[kMT][kTileKS][4];
#pragma unroll
      for (int mt = 0; mt < kMT; ++mt) {
        const uint32_t* p = sP + ((((t & 1) * 2 + pair) * kMT + mt) * kTileKS * 4) * 32 + lane;
#pragma unroll
        for (int j = 0; j < kTileKS; ++j)
#pragma unroll
          for (int i = 0; i < 4; ++i) pa[mt][j][i] = p[(j * 4 + i) * 32];
      }
#pragma unroll
      for (int j = 0; j < kTileKS; ++j)
#pragma unroll
        for (int n = 0; n < kNTilesO; n += 2) {
          uint32_t bv[4];
          ldmatrix_x4_trans(bv, ((v_addr + (t & 1) * kSlotBytes) ^ (n << 4)) + j * 16 * kHeadDim * 2);
#pragma unroll
          for (int mt = 0; mt < kMT; ++mt) {
            mma_bf16(o[mt][n], pa[mt][j], bv[0], bv[1]);
            mma_bf16(o[mt][n + 1], pa[mt][j], bv[2], bv[3]);
          }
        }
      if (!group_end(t)) continue;
      const int fg = t / kGroupTiles;
#pragma unroll
      for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const int i = pair_first + mt * 16 + g + r * 8;
          if (kv_splits > 1) {
            if (i >= rows_total || fg * kGroupKeys > rpos[mt][r]) continue;
            const int64_t slot = ((int64_t)(q0 + i / group) * num_qo_heads + kvh * group + i % group) * max_groups + fg;
#pragma unroll
            for (int n = 0; n < kNTilesO; ++n)
              *reinterpret_cast<float2*>(ws_o + slot * kHeadDim + n * 8 + t4 * 2) =
                  make_float2(o[mt][n][2 * r], o[mt][n][2 * r + 1]);
            continue;
          }
          const FoldScalars f = {sFold[lrow(mt, r)], sFold[kRows + lrow(mt, r)]};
#pragma unroll
          for (int n = 0; n < kNTilesO; ++n)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
              float& to = tot(mt, n, 2 * r + e);
              to = fold_elem(to, o[mt][n][2 * r + e], f);
            }
          if (t == t_last && i < rows_total) {
            const float L = sFold[2 * kRows + lrow(mt, r)];
            __nv_bfloat16* dst = out + ((int64_t)(q0 + i / group) * num_qo_heads + kvh * group + i % group) * kHeadDim;
#pragma unroll
            for (int n = 0; n < kNTilesO; ++n) {
              __nv_bfloat162 ob;
              ob.x = finish(tot(mt, n, 2 * r), L);
              ob.y = finish(tot(mt, n, 2 * r + 1), L);
              *reinterpret_cast<__nv_bfloat162*>(dst + n * 8 + t4 * 2) = ob;
            }
          }
        }
#pragma unroll
      for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
        for (int n = 0; n < kNTilesO; ++n) o[mt][n][0] = o[mt][n][1] = o[mt][n][2] = o[mt][n][3] = 0.f;
    }
  }
  cp_async_wait<0>();
}


// Prefill merge (split mode). grid = (num_qo_heads, total q rows), 128 threads.
// Folds the row's group partials in ascending order (the tree's second level).
__global__ void __launch_bounds__(kHeadDim)
    prefill_merge_kernel(const float* __restrict__ ws_o, const float2* __restrict__ ws_ml,
                         const int32_t* __restrict__ qo_indptr, const int32_t* __restrict__ kv_lens,
                         const int32_t* __restrict__ prefix_lens, int batch,
                         __nv_bfloat16* __restrict__ out, int num_qo_heads, int max_groups) {
  const int h = blockIdx.x, row = blockIdx.y, d = threadIdx.x;
  int b = 0;
  while (b + 1 < batch && qo_indptr[b + 1] <= row) ++b;
  const int q_len = qo_indptr[b + 1] - qo_indptr[b];
  const int prefix = kv_lens ? kv_lens[b] - q_len : prefix_lens[b];
  const int pos = prefix + (row - qo_indptr[b]);
  const int n_groups = pos / kGroupKeys + 1;
  const int64_t base = ((int64_t)row * num_qo_heads + h) * max_groups;
  float M = -INFINITY, L = 0.f, O = 0.f;
  for (int fg = 0; fg < n_groups; ++fg) {
    const float2 ml = ws_ml[base + fg];
    const FoldScalars f = fold_scalars(M, L, ml.x, ml.y);
    O = fold_elem(O, ws_o[(base + fg) * kHeadDim + d], f);
  }
  out[((int64_t)row * num_qo_heads + h) * kHeadDim + d] = finish(O, L);
}


// Causal prefill makespan in key-tile units for `sp` splits: CTAs are dispatched in
// launch order (tile fastest) to the earliest free of `slots` slots; a CTA costs the
// key tiles it runs plus a fixed 4, and split mode adds the partial-workspace write and
// merge (0.02 per 1000 row-head-groups). Fitted on results/kernels/prefill_configs.txt.
inline double prefill_makespan(int batch, int num_qo_heads, int num_kv_heads, int max_q_len, int max_kv_len,
                        int sp, int64_t slots) {
  const int group = num_qo_heads / std::max(num_kv_heads, 1), rows = max_q_len * group;
  const int row_tiles = (rows + kPrefillRows - 1) / (kPrefillRows), prefix = max_kv_len - max_q_len;
  std::priority_queue<double, std::vector<double>, std::greater<double>> free_at;
  for (int64_t i = 0; i < slots; ++i) free_at.push(0.0);
  double end = 0.0;
  for (int z = 0; z < batch * sp; ++z)
    for (int y = 0; y < num_kv_heads; ++y)
      for (int x = 0; x < row_tiles; ++x) {
        const int last_pos = prefix + (std::min((x + 1) * kPrefillRows, rows) - 1) / group;
        const int tg = last_pos / kGroupKeys + 1, per = (tg + sp - 1) / sp, gb = (z % sp) * per;
        const int ge = std::min(tg, gb + per);
        if (gb >= ge) continue;
        const int work = std::min(ge * kGroupTiles, last_pos / kTile + 1) - gb * kGroupTiles + 4;
        const double t = free_at.top() + work;
        free_at.pop();
        free_at.push(t);
        end = std::max(end, t);
      }
  if (sp > 1) end += 0.02e-3 * batch * max_q_len * num_qo_heads * ((max_kv_len - 1) / kGroupKeys + 1);
  return end;
}

// Prefill planner: split count over fold groups and max_groups for the split
// workspace. Splits never change bits, only speed.
inline void prefill_plan(int batch, int num_qo_heads, int num_kv_heads, int max_q_len, int max_kv_len,
                         int sms, int* splits, int* max_groups) {
  const int group = num_qo_heads / std::max(num_kv_heads, 1);
  const int64_t rows = (int64_t)max_q_len * group;
  const int64_t ctas = (rows + kPrefillRows - 1) / (kPrefillRows) * num_kv_heads * batch;
  const int64_t slots = 2 * (int64_t)sms;
  *max_groups = (max_kv_len - 1) / kGroupKeys + 1;
  const int64_t tiles = (max_kv_len + kTile - 1) / kTile;
  *splits = 1;
  // Split mode keeps one f32 partial per (row, head, fold group); stay within the
  // same 256 MiB a FlashInfer float workspace typically takes.
  const int64_t ws_bytes = (int64_t)batch * max_q_len * num_qo_heads * *max_groups * (kHeadDim + 2) * 4;
  if (ws_bytes > (256ll << 20)) return;
  if (ctas < slots) {
    // Row tiles alone leave SM slots idle (short extends over long prefixes): a
    // wave model over every split count (a CTA's time ~ the tiles it runs).
    int64_t best = INT64_MAX;
    for (int sp = 1; sp <= *max_groups; ++sp) {
      const int64_t groups_per = (*max_groups + sp - 1) / sp;
      const int64_t per = std::min<int64_t>(tiles, groups_per * kGroupTiles);
      const int64_t waves = (ctas * sp + slots - 1) / slots;
      const int64_t cost = waves * per + (sp > 1 ? per / 8 : 0);  // + workspace / merge
      if (cost < best) {
        best = cost;
        *splits = sp;
      }
    }
  } else if (ctas <= 4 * slots) {
    // A few waves: the causal tail (late row tiles run the most key tiles) can
    // still leave SMs idle; simulate the dispatch for 1, 2, 4, 8 splits. The
    // simulation costs more host time than a short kernel, so decisions are
    // memoised per shape, with the prefix rounded up to whole fold groups.
    const int kv_key = *max_groups * kGroupKeys;
    const auto key = std::make_tuple(batch, num_qo_heads, num_kv_heads, max_q_len, kv_key);
    static std::mutex mu;
    static std::map<std::tuple<int, int, int, int, int>, int> memo;
    std::lock_guard<std::mutex> lock(mu);
    auto it = memo.find(key);
    if (it != memo.end()) {
      *splits = it->second;
      return;
    }
    const int q_key = std::min(max_q_len, kv_key);
    double best = prefill_makespan(batch, num_qo_heads, num_kv_heads, q_key, kv_key, 1, slots);
    for (int sp = 2; sp <= 8 && sp <= *max_groups; sp *= 2) {
      const double c = prefill_makespan(batch, num_qo_heads, num_kv_heads, q_key, kv_key, sp, slots);
      if (c < best) {
        best = c;
        *splits = sp;
      }
    }
    if (memo.size() < 4096) memo.emplace(key, *splits);
  }
}


// Decode. grid = (max_groups, num_kv_heads, batch), 4 warps. One CTA runs the
// canonical online sequence of fold group fg for the GQA rows of (b, kvh) and
// writes the group partial (m, d, o) at ws row b * max_groups + fg. A request
// with one group finishes in place (the first fold into the identity is exact);
// otherwise, per warp slice of head_dim, the last warp to finish, found with a
// per-(b, kvh, warp) counter that it resets, folds the groups in ascending
// order, so no merge launch is needed and which CTA folds never changes a value. The tile is split across
// warps without changing any row's sequence: warp w computes S n-tile w (each element's head_dim k-steps in
// ascending order) and O n-tiles [4w, 4w + 4) (each element: fma rescale, then
// the two key k-steps in ascending order); the tile max is exact under any
// split; P is rounded elementwise; warp 0 sums P from smem in exactly
// online_step's order.
constexpr int kDecodeWarps = 4;
constexpr int decode_smem_bytes() { return (2 * kTile * kHeadDim + 16 * (kTile + 8)) * 2 + kDecodeWarps * 16 * 4; }

// A decode CTA's group partial (this thread: rows g, g + 8; O n-tiles [4w, 4w + 4)):
// written out directly for a one-group request; otherwise each warp publishes its
// dims of the partial with its own copy of (m, l), counts its arrival, and the last
// warp w to arrive folds dims [32w, 32w + 32) of every head in ascending group order.
// No CTA barrier follows: a CTA parked on an atomic round trip through a busy L2
// holds its SM slot.
constexpr int kDecodeON = kNTilesO / kDecodeWarps;
__device__ __forceinline__ void decode_finish(const float (&M)[2], float l0, float l1, const float (&O)[kDecodeON][4],
                                              int b, int fg, int kvh, int n_tiles, int group, int max_groups,
                                              int num_qo_heads, int num_kv_heads, float* __restrict__ ws_o,
                                              float2* __restrict__ ws_ml, int* __restrict__ counters,
                                              __nv_bfloat16* __restrict__ out) {
  constexpr int kON = kDecodeON;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
  const int n_groups = (n_tiles + kGroupTiles - 1) / kGroupTiles;
  const int head0 = kvh * group;
  if (n_groups == 1) {
    __nv_bfloat16* o0 = out + ((int64_t)b * num_qo_heads + head0 + g) * kHeadDim;
    __nv_bfloat16* o1 = o0 + 8 * kHeadDim;
#pragma unroll
    for (int n = 0; n < kON; ++n) {
      const int col = (warp * kON + n) * 8 + t4 * 2;
      if (g < group) *reinterpret_cast<__nv_bfloat162*>(o0 + col) = __halves2bfloat162(finish(O[n][0], l0), finish(O[n][1], l0));
      if (g + 8 < group)
        *reinterpret_cast<__nv_bfloat162*>(o1 + col) = __halves2bfloat162(finish(O[n][2], l1), finish(O[n][3], l1));
    }
    return;
  }
  const int64_t slot0 = ((int64_t)b * max_groups + fg) * num_qo_heads + head0 + g;
  const int64_t slot1 = slot0 + 8;
#pragma unroll
  for (int n = 0; n < kON; ++n) {
    const int col = (warp * kON + n) * 8 + t4 * 2;
    if (g < group) *reinterpret_cast<float2*>(ws_o + slot0 * kHeadDim + col) = make_float2(O[n][0], O[n][1]);
    if (g + 8 < group) *reinterpret_cast<float2*>(ws_o + slot1 * kHeadDim + col) = make_float2(O[n][2], O[n][3]);
  }
  if (t4 == 0) {
    if (g < group) ws_ml[slot0 * kDecodeWarps + warp] = make_float2(M[0], l0);
    if (g + 8 < group) ws_ml[slot1 * kDecodeWarps + warp] = make_float2(M[1], l1);
  }
  __threadfence();  // this lane's partial visible device-wide before the arrival
  __syncwarp();
  int last = 0;
  if (lane == 0) {
    int* cnt = counters + ((int64_t)b * num_kv_heads + kvh) * kDecodeWarps + warp;
    last = atomicAdd(cnt, 1) == n_groups - 1;
    if (last) *cnt = 0;  // ready for the next launch
  }
  if (!__shfl_sync(0xffffffffu, last, 0)) return;
  __threadfence();
  // Fold: lane p takes head p / 8, dims 32w + 4 (p % 8) + [0, 4); loads run
  // kFoldAhead groups ahead.
  constexpr int kFoldAhead = 8;
  for (int p = lane; p < group * 8; p += 32) {
    const int h = p >> 3, d0 = warp * 32 + (p & 7) * 4;
    float Mf = -INFINITY, L = 0.f, Of[4] = {0.f, 0.f, 0.f, 0.f};
    const int64_t slot_h = (int64_t)b * max_groups * num_qo_heads + head0 + h;
    for (int c0 = 0; c0 < n_groups; c0 += kFoldAhead) {
      float2 ml[kFoldAhead];
      float4 oc[kFoldAhead];
#pragma unroll
      for (int i = 0; i < kFoldAhead; ++i) {
        if (c0 + i >= n_groups) break;
        const int64_t slot = slot_h + (int64_t)(c0 + i) * num_qo_heads;
        ml[i] = __ldcg(ws_ml + slot * kDecodeWarps + warp);
        oc[i] = __ldcg(reinterpret_cast<const float4*>(ws_o + slot * kHeadDim + d0));
      }
#pragma unroll
      for (int i = 0; i < kFoldAhead; ++i) {
        if (c0 + i >= n_groups) break;
        const FoldScalars f = fold_scalars(Mf, L, ml[i].x, ml[i].y);
        Of[0] = fold_elem(Of[0], oc[i].x, f);
        Of[1] = fold_elem(Of[1], oc[i].y, f);
        Of[2] = fold_elem(Of[2], oc[i].z, f);
        Of[3] = fold_elem(Of[3], oc[i].w, f);
      }
    }
    __nv_bfloat16* o = out + ((int64_t)b * num_qo_heads + head0 + h) * kHeadDim + d0;
    reinterpret_cast<__nv_bfloat162*>(o)[0] = __halves2bfloat162(finish(Of[0], L), finish(Of[1], L));
    reinterpret_cast<__nv_bfloat162*>(o)[1] = __halves2bfloat162(finish(Of[2], L), finish(Of[3], L));
  }
}

__global__ void __launch_bounds__(kDecodeWarps * 32)
    decode_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k_cache,
                   const __nv_bfloat16* __restrict__ v_cache, const int32_t* __restrict__ kv_page_indptr,
                   const int32_t* __restrict__ kv_page_indices, int64_t table_stride,
                   const int32_t* __restrict__ kv_lens, float* __restrict__ ws_o, float2* __restrict__ ws_ml,
                   int* __restrict__ counters, __nv_bfloat16* __restrict__ out, int max_groups,
                   int num_qo_heads, int num_kv_heads, int page_size, int page_shift, float scale_log2) {
  static_assert(kTileNS == kDecodeWarps && kNTilesO % kDecodeWarps == 0, "one S n-tile per warp");
  constexpr int kON = kNTilesO / kDecodeWarps, kPStride = kTile + 8;
  extern __shared__ __align__(16) __nv_bfloat16 smem[];
  __nv_bfloat16* sK = smem;
  __nv_bfloat16* sV = sK + kTile * kHeadDim;
  __nv_bfloat16* sP = sV + kTile * kHeadDim;  // [16 rows][kTile] rounded P
  float* sMax = reinterpret_cast<float*>(sP + 16 * kPStride);  // [warps][16]
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
  int64_t row;
  row = tile_row(pages, page_size, page_shift, t_begin, kv_len, num_kv_heads, kvh, tid);
  issue_tile(sK, k_cache, row, tid);
  cp_async_commit();
  issue_tile(sV, v_cache, row, tid);
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
    if (t + 1 < t_end) row = tile_row(pages, page_size, page_shift, t + 1, kv_len, num_kv_heads, kvh, tid);
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
    if (t + 1 < t_end) issue_tile(sK, k_cache, row, tid);
    cp_async_commit();
    float mm[2], scale[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      float tmax = -INFINITY;
#pragma unroll
      for (int w = 0; w < kDecodeWarps; ++w) tmax = fmaxf(tmax, sMax[w * 16 + g + r * 8]);
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
    if (t + 1 < t_end) issue_tile(sV, v_cache, row, tid);
    cp_async_commit();
  }
  cp_async_wait<0>();
  float* sL = sMax;  // the loop is done with sMax: publish warp 0's row sums
  if (warp == 0 && t4 == 0) {
    sL[g] = D;
    sL[g + 8] = Dh;
  }
  __syncthreads();
  const float l0 = sL[g], l1 = sL[g + 8];
  decode_finish(M, l0, l1, O, b, fg, kvh, n_tiles, group, max_groups, num_qo_heads, num_kv_heads, ws_o, ws_ml,
                counters, out);
}


// Decode for small grids (few requests x kv heads): the same canonical sequence as
// decode_kernel, with the group's tiles in three phases, every value computed
// exactly as online_step does. Only the d and o updates are serial; S is independent
// per tile and the running max is an exact prefix max. So:
// 1. warp w computes S of tiles w, w + 4, ... (all 16 rows, each element's
//    head_dim k-steps in ascending order) and publishes each tile's row maxima;
// 2. from the prefix maxima each tile's owner computes its P (rounded against
//    online_step's running max after the tile) and row sum (online_step's order);
// 3. every warp runs the d / o chain over the tiles in order: d = fma(d, scale,
//    sum P); for O n-tiles [4w, 4w + 4), fma rescale, then the two key k-steps.
// The group's K and then V tiles stream through a ring of RING slots, four per step,
// RING / 4 - 1 steps ahead, from cache rows resolved at kernel start. With one CTA per
// SM this keeps a group's 256 KB in flight where decode_kernel's one-tile-ahead
// stream waited out a DRAM round trip per tile; its smem (150 KB) is why it is used
// only when the grid's requests x kv heads fit in one wave (decode_deep). Launch
// decode_deep_grid CTAs.
constexpr int kDecodeDeepRing = 16, kDecodeDeepMaxBatch = 256;
template <int RING>
constexpr int decode_deep_smem_bytes() {
  return RING * kTile * kHeadDim * 2 + kGroupTiles * (kTileKS * 4 * 32 * 4 + 2 * 16 * 4 + kTile * 8) +
         (kDecodeDeepMaxBatch + kDecodeWarps) * 4;
}
inline int decode_deep_grid(int max_groups, int num_kv_heads, int batch, int sms) {
  return (int)std::min<int64_t>(sms, (int64_t)max_groups * num_kv_heads * batch);
}
inline bool decode_deep(int cc_major, int cc_minor, int batch, int num_kv_heads, int sms) {
  return cc_major == 8 && cc_minor == 0 && batch * num_kv_heads <= sms && batch <= kDecodeDeepMaxBatch;
}

template <int RING>
__global__ void __launch_bounds__(kDecodeWarps * 32, 1)
    decode_deep_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k_cache,
                        const __nv_bfloat16* __restrict__ v_cache, const int32_t* __restrict__ kv_page_indptr,
                        const int32_t* __restrict__ kv_page_indices, int64_t table_stride,
                        const int32_t* __restrict__ kv_lens, float* __restrict__ ws_o, float2* __restrict__ ws_ml,
                        int* __restrict__ counters, __nv_bfloat16* __restrict__ out, int max_groups,
                        int num_qo_heads, int num_kv_heads, int page_size, int page_shift, float scale_log2,
                        int batch) {
  static_assert(kDecodeWarps == 4 && kGroupTiles % 4 == 0 && RING >= 8 && RING % 4 == 0,
                "four tiles per step, at least two steps in flight");
  constexpr int kON = kDecodeON, kSteps = kGroupTiles / 4;
  extern __shared__ __align__(16) __nv_bfloat16 smem[];
  __nv_bfloat16* sRing = smem;  // RING slots of kTile x 128
  uint32_t* sP = reinterpret_cast<uint32_t*>(sRing + RING * kTile * kHeadDim);  // [tile][j][i][lane] P A-fragments
  float* sMx = reinterpret_cast<float*>(sP + kGroupTiles * kTileKS * 4 * 32);  // [tile][16 rows] tile row max
  float* sLs = sMx + kGroupTiles * 16;                                         // [tile][16 rows] tile row sum of P
  int64_t* sRow = reinterpret_cast<int64_t*>(sLs + kGroupTiles * 16);          // [tile][key] cache row, -1 past kv_len
  int* sLen = reinterpret_cast<int*>(sRow + kGroupTiles * kTile);              // [request] kv length, then [warp] max
  // Persistent: CTA c runs work items c, c + gridDim.x, ... of (group, request, kv
  // head), group slowest, so a capture-sized grid (max_groups from the page table)
  // costs one loop step per absent group instead of one CTA launch at one CTA per SM.
  // Loop only over the groups the longest request has.
  int len = 0;
  for (int i = threadIdx.x; i < batch; i += kDecodeWarps * 32) {
    sLen[i] = kv_lens[i];
    len = max(len, sLen[i]);
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) len = max(len, __shfl_xor_sync(0xffffffffu, len, o));
  if ((threadIdx.x & 31) == 0) sLen[kDecodeDeepMaxBatch + (threadIdx.x >> 5)] = len;
  __syncthreads();
#pragma unroll
  for (int w = 0; w < kDecodeWarps; ++w) len = max(len, sLen[kDecodeDeepMaxBatch + w]);
  const int units = batch * num_kv_heads, groups = min(max_groups, (len + kGroupKeys - 1) / kGroupKeys);
  for (int item = blockIdx.x; item < groups * units; item += gridDim.x) {
    const int fg = item / units, b = item % units / num_kv_heads, kvh = item % num_kv_heads;
    const int kv_len = sLen[b], pos = kv_len - 1;
    const int n_tiles = (kv_len + kTile - 1) / kTile;
    const int t_begin = fg * kGroupTiles;
    if (t_begin >= n_tiles) continue;
    __syncthreads();  // all warps are done with the previous item's smem
    const int nt = min(n_tiles, t_begin + kGroupTiles) - t_begin, steps = (nt + 3) / 4;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t4 = lane & 3, mat = lane >> 3, r8 = lane & 7;
    const int group = num_qo_heads / num_kv_heads;
    const int32_t* pages = request_pages(kv_page_indices, kv_page_indptr, table_stride, b);
    // Every key's cache row up front: independent page-table reads, one round trip.
    for (int i = tid; i < nt * kTile; i += kDecodeWarps * 32) {
      const int p = t_begin * kTile + i;
      int64_t row = -1;
      if (p < kv_len) {
        const int slot = page_shift >= 0 ? (pages[p >> page_shift] << page_shift) + (p & (page_size - 1))
                                         : pages[p / page_size] * page_size + p % page_size;
        row = ((int64_t)slot * num_kv_heads + kvh) * kHeadDim;
      }
      sRow[i] = row;
    }
    __syncthreads();
    // Stream item i: K of tile i (i < nt), then V of tile i - nt; item i sits in slot
    // i % RING. One commit group per item (empty past the stream), so a step that
    // releases its slots and refills to `start + RING` items leaves at most RING - 4
    // groups beyond its own four pending.
    auto slot = [&](int item) { return sRing + (item % RING) * kTile * kHeadDim; };
    int next = 0;
    auto issue_to = [&](int end) {
      for (; next < end; ++next) {
        if (next < 2 * nt) {
          const int u = next < nt ? next : next - nt;
          issue_tile(slot(next), next < nt ? k_cache : v_cache, sRow[u * kTile + (tid >> 2)], tid);
        }
        cp_async_commit();
      }
    };
    issue_to(RING);
    QFrag qf;
    const int64_t q_stride = (int64_t)num_qo_heads * kHeadDim;
    load_q(qf, [&](int r) -> const __nv_bfloat16* {
             return r < group ? q + b * q_stride + (kvh * group + r) * kHeadDim : nullptr;
           }, lane);
    // ldmatrix offsets at k-step 0 (XOR the chunk offset, see swz_bytes; slots are
    // 256-byte aligned): K for all 4 S n-tiles, V for this warp's O n-tiles.
    const uint32_t k_off = swz_bytes((mat >> 1) * 8 + r8, mat & 1);
    const uint32_t v_off = swz_bytes((mat & 1) * 8 + r8, warp * kON + (mat >> 1));

    // Phase 1: step r, warp w: S of tile u = 4r + w, masked and scaled.
    float s[kSteps][kTileNS][4];
#pragma unroll
    for (int r = 0; r < kSteps; ++r) {
      if (r >= steps) break;
      __syncthreads();  // all warps are done with the previous step's slots
      issue_to(4 * r + RING);
      cp_async_wait<RING - 4>();  // items < 4r + 4 landed (this thread's copies) ...
      __syncthreads();     // ... and everyone's
      const int u = 4 * r + warp;
      if (u >= nt) continue;
      const uint32_t kb = smem_u32(slot(u)) + k_off;
#pragma unroll
      for (int n = 0; n < kTileNS; ++n) s[r][n][0] = s[r][n][1] = s[r][n][2] = s[r][n][3] = 0.f;
#pragma unroll
      for (int ks = 0; ks < kKStepsQK; ++ks)
#pragma unroll
        for (int n = 0; n < kTileNS; n += 2) {
          uint32_t bk[4];
          ldmatrix_x4(bk, (kb ^ (ks << 5)) + n * 8 * kHeadDim * 2);
          mma_bf16(s[r][n], qf.a[ks], bk[0], bk[1]);
          mma_bf16(s[r][n + 1], qf.a[ks], bk[2], bk[3]);
        }
      float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
      for (int n = 0; n < kTileNS; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int row = g + (i >> 1) * 8, key = (t_begin + u) * kTile + n * 8 + t4 * 2 + (i & 1);
          s[r][n][i] = (row < group && key <= pos) ? mul(s[r][n][i], scale_log2) : -INFINITY;
          mx[i >> 1] = fmaxf(mx[i >> 1], s[r][n][i]);
        }
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 1));
        mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 2));
      }
      if (t4 == 0) {
        sMx[u * 16 + g] = mx[0];
        sMx[u * 16 + g + 8] = mx[1];
      }
    }

    // Phases 2 and 3, step r: V of tiles 4r .. 4r + 3.
    float M[2] = {-INFINITY, -INFINITY}, D[2] = {0.f, 0.f};  // rows g, g + 8
    float O[kON][4];
#pragma unroll
    for (int n = 0; n < kON; ++n) O[n][0] = O[n][1] = O[n][2] = O[n][3] = 0.f;
#pragma unroll
    for (int r = 0; r < kSteps; ++r) {
      if (r >= steps) break;
      __syncthreads();
      issue_to(nt + 4 * r + RING);
      cp_async_wait<RING - 4>();
      __syncthreads();  // V tiles landed; on step 0 also every tile's row maxima
      if (r == 0) {
        // Phase 2: P and row sums of this warp's tiles, against online_step's running
        // max after the tile: the prefix max of the tile maxima.
        float run[2] = {-INFINITY, -INFINITY};  // prefix max before tile 4q + warp
        for (int q2 = 0; q2 < kSteps; ++q2) {
          const int u = 4 * q2 + warp;
          if (q2 >= steps || u >= nt) break;
          for (int v = (q2 ? u - 4 : 0); v < u; ++v)
#pragma unroll
            for (int h = 0; h < 2; ++h) run[h] = fmaxf(run[h], sMx[v * 16 + g + h * 8]);
          float mm[2];
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const float mn = fmaxf(run[h], sMx[u * 16 + g + h * 8]);
            mm[h] = mn == -INFINITY ? 0.f : mn;
          }
          float ls[2] = {0.f, 0.f};
          uint32_t pa[kTileKS][4];
#pragma unroll
          for (int q = 0; q < kSteps; ++q) {  // s[q2] with a compile-time index
            if (q != q2) continue;
#pragma unroll
            for (int n = 0; n < kTileNS; ++n) {
              float p[4];
#pragma unroll
              for (int i = 0; i < 4; ++i) {
                p[i] = round_bf16(ex2(sub(s[q][n][i], mm[i >> 1])));
                ls[i >> 1] = add(ls[i >> 1], p[i]);
              }
              pa[n >> 1][(n & 1) * 2 + 0] = pack_bf16(p[0], p[1]);
              pa[n >> 1][(n & 1) * 2 + 1] = pack_bf16(p[2], p[3]);
            }
          }
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            ls[h] = add(ls[h], __shfl_xor_sync(0xffffffffu, ls[h], 1));
            ls[h] = add(ls[h], __shfl_xor_sync(0xffffffffu, ls[h], 2));
          }
#pragma unroll
          for (int j = 0; j < kTileKS; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) sP[((u * kTileKS + j) * 4 + i) * 32 + lane] = pa[j][i];
          if (t4 == 0) {
            sLs[u * 16 + g] = ls[0];
            sLs[u * 16 + g + 8] = ls[1];
          }
        }
        __syncthreads();  // P and row sums published
      }
      // Phase 3: the serial d / o chain over this step's tiles.
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int t = 4 * r + e;
        if (t >= nt) break;
        float scale[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const float mn = fmaxf(M[h], sMx[t * 16 + g + h * 8]);
          const float mm = mn == -INFINITY ? 0.f : mn;
          scale[h] = ex2(sub(M[h], mm));
          M[h] = mn;
          D[h] = __fmaf_rn(D[h], scale[h], sLs[t * 16 + g + h * 8]);
        }
        uint32_t pa[kTileKS][4];
#pragma unroll
        for (int j = 0; j < kTileKS; ++j)
#pragma unroll
          for (int i = 0; i < 4; ++i) pa[j][i] = sP[((t * kTileKS + j) * 4 + i) * 32 + lane];
#pragma unroll
        for (int n = 0; n < kON; ++n)
#pragma unroll
          for (int i = 0; i < 4; ++i) O[n][i] = __fmaf_rn(O[n][i], scale[i >> 1], 0.f);
        const uint32_t vb = smem_u32(slot(nt + t)) + v_off;
#pragma unroll
        for (int j = 0; j < kTileKS; ++j)
#pragma unroll
          for (int n = 0; n < kON; n += 2) {
            uint32_t bv[4];
            ldmatrix_x4_trans(bv, (vb ^ (n << 4)) + j * 16 * kHeadDim * 2);
            mma_bf16(O[n], pa[j], bv[0], bv[1]);
            mma_bf16(O[n + 1], pa[j], bv[2], bv[3]);
          }
      }
    }
    cp_async_wait<0>();
    const float l0 = D[0], l1 = D[1];
    decode_finish(M, l0, l1, O, b, fg, kvh, n_tiles, group, max_groups, num_qo_heads, num_kv_heads, ws_o, ws_ml,
                  counters, out);
  }
}

}  // namespace pi_attn
