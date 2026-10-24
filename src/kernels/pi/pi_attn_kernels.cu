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
  // Each warp publishes its dims of the partial with its own copy of (m, l),
  // counts its arrival, and the last warp w to arrive folds dims [32w, 32w + 32)
  // of every head. No CTA barrier follows: a CTA parked on an atomic round trip
  // through a busy L2 holds its SM slot.
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

}  // namespace pi_attn
