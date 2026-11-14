// sm_90a (Hopper) prefill / extend kernel: prefill_kernel<QREG = false, WARPS = 4> of pi_attn_kernels.cu with
// its tensor-core work issued as wgmma. Included at the end of pi_attn_kernels.cu in sm_90a builds (PI_SM90A).
// Everything that defines the bits is shared with the other kernels or copied from prefill_kernel: the row
// map, the masks, online_step / fold / finish of pi_attn.cuh, the 512-key groups, the split partials. Changed:
//   S = Q K^T : 32 mma.sync m16n8k16 per warp -> 8 wgmma m64n32k16 (Q and K in shared memory), from +0
//   O += P V  : 32 mma.sync m16n8k16 per warp -> 2 wgmma m64n128k16 (P in registers, V in shared memory read
//                                                transposed), O the accumulator
// and the shared-memory layouts and cp.async destinations these need. On Hopper a wgmma k16 step returns the
// bits of the mma.sync step for every element (measured), and a warp group's register fragments have the
// layout of this kernel's per-warp fragments (lane (g, t) holds rows g, g + 8 and columns 8 n + 2 t, + 1),
// so online_step, its lane shuffles and its fixed row-sum order included, runs unchanged on them.
// wgmma is asynchronous: S(t + 1) is issued, tile t's online step runs on the CUDA cores meanwhile, then
// P V(t) is issued and both are awaited. Operands are fed with cp.async (the existing K/V double buffering).
#pragma once
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/pipeline/pipeline.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/barrier.h>

namespace pi_wg {
using BF = cutlass::bfloat16_t;

// +0 for a wgmma accumulator register, produced by its own add on an opaque register. Zeroing the 64 O registers
// with the constant makes them copies of one shared constant register, and ptxas (12.6) then serialises every
// wgmma of the function (C7511: one HGMMA per arrive / wait instead of a chain). (+0) + (+0) is exactly +0.
__device__ __forceinline__ float zero_reg() {
  float z = 0.f;
  asm volatile("" : "+f"(z));
  return __fadd_rn(z, z);
}
using TileShape_QKD = cute::Shape<cute::_64, cute::_32, cute::_128>;
using TileShape_PDV = cute::Shape<cute::_64, cute::_128, cute::_32>;
using TiledMmaQK = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<BF, BF, float, TileShape_QKD>()));
using TiledMmaPV = decltype(cute::make_tiled_mma(
    cute::GMMA::rs_op_selector<BF, BF, float, TileShape_PDV, cute::GMMA::Major::K, cute::GMMA::Major::MN>()));
using SmemLayoutAtomQ =
    decltype(cutlass::gemm::collective::detail::ss_smem_selector<cute::GMMA::Major::K, BF, cute::_64, cute::_128>());
using SmemLayoutQ = decltype(cute::tile_to_shape(SmemLayoutAtomQ{}, cute::Shape<cute::_64, cute::_128>{}));
using SmemLayoutAtomK =
    decltype(cutlass::gemm::collective::detail::ss_smem_selector<cute::GMMA::Major::K, BF, cute::_32, cute::_128>());
// [key][d][slot]: two 32-key tiles each of K and V
using SmemLayoutK = decltype(cute::tile_to_shape(SmemLayoutAtomK{}, cute::Shape<cute::_32, cute::_128, cute::_2>{}));
using SmemLayoutV = SmemLayoutK;
using SmemLayoutVt = decltype(cute::composition(
    SmemLayoutV{}, cute::make_ordered_layout(cute::Shape<cute::_128, cute::_32, cute::_2>{}, cute::Step<cute::_2, cute::_1, cute::_3>{})));

struct Smem {
  cute::array_aligned<BF, cute::cosize_v<SmemLayoutQ>> q;
  cute::array_aligned<BF, cute::cosize_v<SmemLayoutK>> k;
  cute::array_aligned<BF, cute::cosize_v<SmemLayoutV>> v;
};

// Shared-memory address of 8 consecutive head dims (one 16-byte chunk) of a row / key.
__device__ __forceinline__ void* q_ptr(Smem& sm, int row, int col) {
  auto sQ = cute::make_tensor(cute::make_smem_ptr(sm.q.data()), SmemLayoutQ{});
  return &sQ(row, col);
}
__device__ __forceinline__ void* k_ptr(Smem& sm, int slot, int key, int col) {
  auto sK = cute::make_tensor(cute::make_smem_ptr(sm.k.data()), SmemLayoutK{});
  return &sK(key, col, slot);
}
__device__ __forceinline__ void* v_ptr(Smem& sm, int slot, int key, int col) {
  auto sV = cute::make_tensor(cute::make_smem_ptr(sm.v.data()), SmemLayoutV{});
  return &sV(key, col, slot);
}

// ((2, 2, N / 8), MMA_M, MMA_N) -> ((2, 2, 2), MMA_M, (N / 16, MMA_N)) (FlashAttention-3 / FlashInfer)
template <typename Layout>
__device__ __forceinline__ auto acc_to_aregs(Layout acc_layout) {
  using X = cute::Underscore;
  auto l = cute::logical_divide(cute::get<0>(acc_layout), cute::Shape<X, X, cute::_2>{});
  return cute::make_layout(cute::make_layout(cute::get<0>(l), cute::get<1>(l), cute::get<2, 0>(l)),
                           cute::get<1>(acc_layout), cute::make_layout(cute::get<2, 1>(l), cute::get<2>(acc_layout)));
}

// Issue S = +0 + Q K(slot)^T: 8 ascending k16 steps over the head dim (asynchronous; wait0 completes it).
__device__ __forceinline__ void qk_issue(Smem& sm, int tid, float (&s)[4][4], int slot) {
  using namespace cute;
  Tensor sQ = make_tensor(make_smem_ptr(sm.q.data()), SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(sm.k.data()), SmemLayoutK{});
  TiledMmaQK mma_qk;
  auto tqk = mma_qk.get_thread_slice(tid);
  Tensor tSrQ = tqk.partition_fragment_A(sQ);
  Tensor tSrK = tqk.partition_fragment_B(sK);
  auto lay = partition_fragment_C(mma_qk, Shape<_64, _32>{}).layout();
  Tensor tS = make_tensor(make_rmem_ptr<float>(&s[0][0]), lay);
  static_assert(decltype(size(tS))::value == 16);
  CUTE_UNROLL
  for (int i = 0; i < 16; ++i) tS(i) = 0.f;
  warpgroup_fence_operand(tS);
  warpgroup_arrive();
  CUTE_UNROLL
  for (int kb = 0; kb < size<2>(tSrQ); ++kb) cute::gemm(mma_qk, tSrQ(_, _, kb), tSrK(_, _, kb, slot), tS);
  warpgroup_commit_batch();
}

// Issue O = O + P V(slot): 2 ascending k16 steps over the tile's keys (asynchronous).
__device__ __forceinline__ void pv_issue(Smem& sm, int tid, uint32_t (&pa)[2][4], float (&o)[16][4], int slot) {
  using namespace cute;
  Tensor sVt = make_tensor(make_smem_ptr(sm.v.data()), SmemLayoutVt{});
  TiledMmaQK mma_qk;
  TiledMmaPV mma_pv;
  auto tpv = mma_pv.get_thread_slice(tid);
  Tensor tOrV = tpv.partition_fragment_B(sVt);
  auto lay_s = partition_fragment_C(mma_qk, Shape<_64, _32>{}).layout();
  auto lay_o = partition_fragment_C(mma_pv, Shape<_64, _128>{}).layout();
  Tensor tOrP = make_tensor(make_rmem_ptr<BF>(reinterpret_cast<BF*>(&pa[0][0])), acc_to_aregs(lay_s));
  Tensor tOrO = make_tensor(make_rmem_ptr<float>(&o[0][0]), lay_o);
  static_assert(decltype(size(tOrO))::value == 64 && decltype(size(tOrP))::value == 16);
  warpgroup_fence_operand(tOrP);
  warpgroup_fence_operand(tOrO);
  warpgroup_arrive();
  CUTE_UNROLL
  for (int kb = 0; kb < size<2>(tOrP); ++kb) cute::gemm(mma_pv, tOrP(_, _, kb), tOrV(_, _, kb, slot), tOrO);
  warpgroup_commit_batch();
}

// Every issued wgmma has completed; its accumulators may be read.
template <int NS, int NO>
__device__ __forceinline__ void wait0(float (&s)[NS][4], float (&o)[NO][4]) {
  cute::warpgroup_wait<0>();
#pragma unroll
  for (int n = 0; n < NS; ++n)
#pragma unroll
    for (int i = 0; i < 4; ++i) cute::warpgroup_fence_operand(s[n][i]);
#pragma unroll
  for (int n = 0; n < NO; ++n)
#pragma unroll
    for (int i = 0; i < 4; ++i) cute::warpgroup_fence_operand(o[n][i]);
}
}  // namespace pi_wg

namespace pi_attn {

constexpr int prefill90_smem_bytes() { return (int)sizeof(pi_wg::Smem); }

// Arguments, grid and split semantics of prefill_kernel (64 rows, 128 threads = one warp group per CTA).
__global__ void __launch_bounds__(128, 2)
    prefill90_kernel(const __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ out,
                     const __nv_bfloat16* __restrict__ k_cache, const __nv_bfloat16* __restrict__ v_cache,
                     const int32_t* __restrict__ qo_indptr, const int32_t* __restrict__ kv_page_indptr,
                     const int32_t* __restrict__ kv_page_indices, int64_t table_stride,
                     const int32_t* __restrict__ kv_lens, const int32_t* __restrict__ prefix_lens,
                     int num_qo_heads, int num_kv_heads, int page_size, int page_shift, float scale_log2,
                     int kv_splits, float* __restrict__ ws_o, float2* __restrict__ ws_ml, int max_groups) {
  constexpr int kRows = 64, kThreads = 128;
  extern __shared__ char smem_buf[];
  pi_wg::Smem& sm = *reinterpret_cast<pi_wg::Smem*>(smem_buf);
  const int b = blockIdx.z / kv_splits, split = blockIdx.z % kv_splits;
  const int kvh = blockIdx.y, tile = blockIdx.x;
  const int group = num_qo_heads / num_kv_heads;
  const int q0 = qo_indptr[b], q_len = qo_indptr[b + 1] - q0;
  const int rows_total = q_len * group;
  if (tile * kRows >= rows_total) return;
  const int kv_len = kv_lens ? kv_lens[b] : prefix_lens[b] + q_len, prefix = kv_len - q_len;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t4 = lane & 3;
  const int32_t* pages = request_pages(kv_page_indices, kv_page_indptr, table_stride, b);
  const int row0 = tile * kRows;
  const int last_pos = prefix + (min(row0 + kRows, rows_total) - 1) / group;
  const int warp_first = row0 + warp * 16;
  int rpos[2];  // position of this thread's rows g, g + 8; -1 = padding
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

#pragma unroll
  for (int i = 0; i < kRows * 16 / kThreads; ++i) {
    const int idx = tid + i * kThreads, r = idx >> 4, chunk = idx & 15, qi = row0 + r;
    const __nv_bfloat16* src = q;
    int bytes = 0;
    if (qi < rows_total) {
      src = q + (int64_t)(q0 + qi / group) * num_qo_heads * kHeadDim + (kvh * group + qi % group) * kHeadDim + chunk * 8;
      bytes = 16;
    }
    cp_async16(pi_wg::q_ptr(sm, r, chunk * 8), src, bytes);
  }
  // Thread tid copies key tid / 4, 16-byte chunks tid % 4 + 4 i (as issue_tile); row -1 zero-fills.
  auto issue_k = [&](int slot, int64_t row) {
    const int key = tid >> 2, c0 = tid & 3;
    const __nv_bfloat16* src = k_cache + (row >= 0 ? row : 0) + c0 * 8;
#pragma unroll
    for (int i = 0; i < 4; ++i) cp_async16(pi_wg::k_ptr(sm, slot, key, (c0 + 4 * i) * 8), src + 32 * i, row >= 0 ? 16 : 0);
  };
  auto issue_v = [&](int slot, int64_t row) {
    const int key = tid >> 2, c0 = tid & 3;
    const __nv_bfloat16* src = v_cache + (row >= 0 ? row : 0) + c0 * 8;
#pragma unroll
    for (int i = 0; i < 4; ++i) cp_async16(pi_wg::v_ptr(sm, slot, key, (c0 + 4 * i) * 8), src + 32 * i, row >= 0 ? 16 : 0);
  };
  auto row_of = [&](int t) -> int64_t {
    return tile_row(pages, page_size, page_shift, t, kv_len, num_kv_heads, kvh, tid);
  };
  // Tile t uses K slot and V slot t & 1 (t_begin is even). row1 / row2: this thread's cache rows of tiles
  // t + 1 and t + 2, each read an iteration before use.
  int64_t row1 = row_of(t_begin);
  issue_k(0, row1);
  cp_async_commit();  // Q + K(t_begin)
  issue_v(0, row1);
  row1 = row_of(t_begin + 1);
  int64_t row2 = row_of(t_begin + 2);
  if (t_begin + 1 <= t_last) issue_k(1, row1);
  cp_async_commit();  // V(t_begin) + K(t_begin + 1)

  // st: the current group; the folded total of the finished groups is (tot_m, tot_d, tot_o).
  OnlineState st;
  st.reset();
#pragma unroll
  for (int n = 0; n < kNTilesO; ++n)
#pragma unroll
    for (int i = 0; i < 4; ++i) st.o[n][i] = pi_wg::zero_reg();
  float tot_m[2] = {-INFINITY, -INFINITY}, tot_d[2] = {0.f, 0.f}, tot_o[kNTilesO][4];
#pragma unroll
  for (int n = 0; n < kNTilesO; ++n)
#pragma unroll
    for (int i = 0; i < 4; ++i) tot_o[n][i] = 0.f;

  // Group fg is finished: fold it into the total (kv_splits == 1; on the last group also normalise and
  // write out) or write it as a group partial.
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
          float& o = tot_o[n][2 * r + e];
          o = fold_elem(o, st.o[n][2 * r + e], f);
        }
      if (last && i < rows_total) {
        __nv_bfloat16* dst = out + ((int64_t)(q0 + i / group) * num_qo_heads + kvh * group + i % group) * kHeadDim;
#pragma unroll
        for (int n = 0; n < kNTilesO; ++n) {
          __nv_bfloat162 ob;
          ob.x = finish(tot_o[n][2 * r], tot_d[r]);
          ob.y = finish(tot_o[n][2 * r + 1], tot_d[r]);
          *reinterpret_cast<__nv_bfloat162*>(dst + n * 8 + t4 * 2) = ob;
        }
      }
    }
    st.m[0] = st.m[1] = -INFINITY;
    st.d[0] = st.d[1] = 0.f;
#pragma unroll
    for (int n = 0; n < kNTilesO; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) st.o[n][i] = pi_wg::zero_reg();
  };

  // One pipelined iteration: s holds S(t) on entry; sn receives S(t + 1).
  auto step = [&](int t, float (&s)[kTileNS][4], float (&sn)[kTileNS][4]) {
    const int slot = t & 1;
    cp_async_wait<0>();  // V(t), K(t + 1)
    __syncthreads();     // ... for all threads; every wgmma on V(t - 1) and K(t) has completed (wait0)
    if (t + 1 <= t_last) issue_v(slot ^ 1, row1);
    if (t + 2 <= t_last) issue_k(slot, row2);
    cp_async_commit();
    row1 = row2;
    row2 = row_of(t + 3);
    pi_wg::qk_issue(sm, tid, sn, slot ^ 1);  // asynchronous; past t_last: stale slot data, never used
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
    pi_wg::pv_issue(sm, tid, pa, st.o, slot);
    pi_wg::wait0(sn, st.o);
    const bool close = t % kGroupTiles == kGroupTiles - 1 || t == t_last;  // the tile ends its fold group
    if (close) finish_group(t / kGroupTiles, t == t_last);
  };

  float sa[kTileNS][4], sb[kTileNS][4];  // S of the current and next tile, alternating
  float none[1][4] = {{0.f, 0.f, 0.f, 0.f}};
  cp_async_wait<1>();  // Q, K(t_begin)
  __syncthreads();
  pi_wg::qk_issue(sm, tid, sa, 0);
  pi_wg::wait0(sa, none);
  for (int t = t_begin; t <= t_last; t += 2) {
    step(t, sa, sb);
    if (t + 1 <= t_last) step(t + 1, sb, sa);
  }
  cp_async_wait<0>();
}


// ---------------------------------------------------------------------------------------------------------
// Decode on Hopper: decode_kernel's program for a fold group on 16 warps instead of 4.
//
// decode_kernel is bound by what its 4 warps execute in sequence, not by memory: a tile costs two load
// latencies (0.85 us with the K/V in L2 as from DRAM), and with a whole group's tiles in flight the 4 warps
// still take 0.37 us per tile for S, 0.13 for P and 0.22 for the o step (measurements and the designs that
// lost: archive/ceiling_probes/h100_decode_cluster). The contract serialises little of a group: S of a key
// depends on that key only, the running max is a prefix max of exact tile maxima, scale_t and P_t are
// elementwise functions of S and the max, the row sums of P_t are independent per tile, and only d_t and o_t
// are chains, o_t independently per head dim. So here
//   - the 16 warps issue the group's K and V in three cp.async groups (K of the 16 tiles; V of 8; V of the other 8
//     into K slots once S is done) and wait three times;
//   - 4 tiles of S run at once (warp 4 a + n: tiles = a mod 4, S n-tile n, decode_kernel's mma sequence);
//     4 warps form the exact tile maxima and one of them the running max and the scales, then all form P of
//     their own tiles;
//   - the o chain runs on 8 warps, two O n-tiles each: per tile fma rescale, then the two key k-steps;
//   - meanwhile 4 of the other warps form the row sums of P in online_step's order, and one of them the d
//     chain.
// Every element runs decode_kernel's instruction sequence on the same operands: the same bits. The group
// partial, the arrival counters and the fold are decode_kernel's, per 32-dim quarter of the head (two warps).
//
// A CTA holds 220 KB of shared memory, one per SM, so the grid has no CTA that only returns: per (request, kv
// head) it has `slots` CTAs (decode90_slots: as many as fit one wave), and CTA j runs the request's fold
// groups j, j + slots, ... one after the other. Which CTA runs a group changes no value. Where it runs instead
// of decode_kernel: decode90_plan (pi_attn_kernels.cu).
// 16 warps (128 registers per thread: with 20 warps the loop over groups spilled). Warps 0-7 run the o chain,
// warps 8-11 the tile maxima and the row sums, warp 8 the running max and the d chain.
constexpr int kD90Warps = 16, kD90OWarps = 8, kD90DWarp = 8;
// CTAs per (request, kv head): one per fold group while batch * num_kv_heads * slots stays within one CTA per
// SM; 0 = the launch does not fit (decode_kernel runs it). decode90_plan only takes launches with a CTA per
// group; the tests also run the others.
inline int decode90_slots(int batch, int num_kv_heads, int max_groups, int sms) {
  return std::min(max_groups, sms / std::max(batch * num_kv_heads, 1));
}
inline dim3 decode90_grid(int slots, int num_kv_heads, int batch) { return decode_grid(slots, num_kv_heads, batch); }
constexpr int decode90_smem_bytes() {
  return 24 * kTile * kHeadDim * 2                    // 16 K and 8 V tile slots
         + kGroupTiles * 16 * (kTile + 8) * 2         // P of every tile
         + kGroupTiles * (kDecodeWarps + 4) * 16 * 4  // per tile: maxima per S n-tile, max, m, scale, row sum of P
         + 2 * 16 * 4;                                // d and m of the group
}

// Tile slot of V(lt): slot 16 + lt for the first 8 tiles, K slot lt - 8 for the others.
__device__ __forceinline__ __nv_bfloat16* d90_v_slot(__nv_bfloat16* sKV, int lt) {
  return sKV + (lt < 8 ? 16 + lt : lt - 8) * (kTile * kHeadDim);
}

// Fold group fg of (request b, kv head kvh), by all the CTA's threads.
__device__ __forceinline__ void decode90_group(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache, const int32_t* __restrict__ pages, int kv_len, float* __restrict__ ws_o,
    float2* __restrict__ ws_ml, int* __restrict__ counters, __nv_bfloat16* __restrict__ out, int max_groups,
    int num_qo_heads, int num_kv_heads, int page_size, int page_shift, float scale_log2, int b, int kvh, int fg) {
  static_assert(kTileNS == 4 && kGroupTiles == 16 && kD90Warps == 16 && kNTilesO == 2 * kD90OWarps && kDecodeWarps == 4,
                "4 S n-tiles per tile, 4 tiles at once, two O n-tiles per warp, 4 quarters");
  constexpr int kPStride = kTile + 8, kPTile = 16 * kPStride, kTileElems = kTile * kHeadDim;
  extern __shared__ __align__(16) __nv_bfloat16 smem[];
  __nv_bfloat16* sKV = smem;                             // [24][kTile][128], swz per tile: K(lt) in slot lt,
                                                         // V(lt) in slot 16 + lt (lt < 8) or lt - 8
  __nv_bfloat16* sP = sKV + 24 * kTileElems;             // [16 tiles][16 rows][kPStride]
  float* sMax = reinterpret_cast<float*>(sP + kGroupTiles * kPTile);  // [16 tiles][4 n-tiles][16 rows]
  float* sTMax = sMax + kGroupTiles * 4 * 16;            // [16 tiles][16 rows] tile max
  float* sMM = sTMax + kGroupTiles * 16;                 // ... running max after the tile (0 if -inf)
  float* sScale = sMM + kGroupTiles * 16;                // ... scale
  float* sLS = sScale + kGroupTiles * 16;                // ... row sum of P
  float* sL = sLS + kGroupTiles * 16;                    // [16 rows] d of the group
  float* sMF = sL + 16;                                  // [16 rows] m of the group
  const int pos = kv_len - 1;
  const int n_tiles = (kv_len + kTile - 1) / kTile;
  const int n_groups = (n_tiles + kGroupTiles - 1) / kGroupTiles;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t4 = lane & 3, mat = lane >> 3, r8 = lane & 7;
  const int set = warp >> 2, wi = warp & 3;  // S / copies: tiles = set mod 4, S n-tile wi
  const bool ow = warp < kD90OWarps, dw = set == kD90DWarp / 4;
  const int ctid = tid & 127;                // issue_tile's thread index within the set
  const int group = num_qo_heads / num_kv_heads;
  const int t_begin = fg * kGroupTiles;
  const int nt = min(n_tiles - t_begin, kGroupTiles);  // tiles of this group

  // Loads, group 1: K of the warp set's 4 tiles. Group 2: V of tiles 0-7.
  QFrag qf;
  int64_t rows[4];
  {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int lt = set + 4 * i;
      rows[i] = lt < nt ? tile_row(pages, page_size, page_shift, t_begin + lt, kv_len, num_kv_heads, kvh, ctid) : -1;
      if (lt < nt) issue_tile(sKV + lt * kTileElems, k_cache, rows[i], ctid);
    }
  }
  cp_async_commit();
  {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int lt = set + 4 * i;
      if (lt < nt) issue_tile(d90_v_slot(sKV, lt), v_cache, rows[i], ctid);
    }
    const int64_t q_stride = (int64_t)num_qo_heads * kHeadDim;
    load_q(qf, [&](int r) -> const __nv_bfloat16* {
             return r < group ? q + b * q_stride + (kvh * group + r) * kHeadDim : nullptr;
           }, lane);
  }
  cp_async_commit();

  // S of the warp's 4 tiles (tiles set + 4 i), their mma chains interleaved; the warp's maxima of its 8 keys.
  float s[4][4];
  cp_async_wait<1>();  // group 1
  __syncthreads();
  {
#pragma unroll
    for (int i = 0; i < 4; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < kKStepsQK; ++ks)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int lt = set + 4 * i;
        if (lt < nt) {
          uint32_t bk[4];
          ldmatrix_x4(bk, sKV + lt * kTileElems + swz(wi * 8 + r8, ks * 2 + (mat & 1)));
          mma_bf16(s[i], qf.a[ks], bk[0], bk[1]);
        }
      }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int lt = set + 4 * i;
      if (lt < nt) {
        float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int row = g + (e >> 1) * 8, key = (t_begin + lt) * kTile + wi * 8 + t4 * 2 + (e & 1);
          s[i][e] = (row < group && key <= pos) ? mul(s[i][e], scale_log2) : -INFINITY;
          mx[e >> 1] = fmaxf(mx[e >> 1], s[i][e]);
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
          mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
        }
        if (t4 == 0) {
          sMax[(lt * 4 + wi) * 16 + g] = mx[0];
          sMax[(lt * 4 + wi) * 16 + g + 8] = mx[1];
        }
      }
    }
  }
  __syncthreads();  // maxima published; all warps done with K: slots 0-7 take V of tiles 8-15 (group 3)
  {
#pragma unroll
    for (int i = 2; i < 4; ++i) {
      const int lt = set + 4 * i;
      if (lt < nt) issue_tile(d90_v_slot(sKV, lt), v_cache, rows[i], ctid);
    }
  }
  cp_async_commit();
  if (dw && t4 == 0) {  // exact tile maxima: warp 8 + a takes the tiles = a mod 4
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int lt = wi + 4 * i;
      if (lt < nt) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          float tmax = -INFINITY;
#pragma unroll
          for (int n = 0; n < 4; ++n) tmax = fmaxf(tmax, sMax[(lt * 4 + n) * 16 + g + r * 8]);
          sTMax[lt * 16 + g + r * 8] = tmax;
        }
      }
    }
  }
  __syncthreads();
  if (warp == kD90DWarp && t4 == 0) {  // the running max and the scale of every tile, the group's m
    float M[2] = {-INFINITY, -INFINITY};
    for (int lt = 0; lt < nt; ++lt)
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const float mn = fmaxf(M[r], sTMax[lt * 16 + g + r * 8]);
        const float mm = mn == -INFINITY ? 0.f : mn;
        sMM[lt * 16 + g + r * 8] = mm;
        sScale[lt * 16 + g + r * 8] = ex2(sub(M[r], mm));
        M[r] = mn;
      }
    sMF[g] = M[0], sMF[g + 8] = M[1];
  }
  __syncthreads();
  {  // P of the warp's own tiles, as decode_kernel forms it (rows >= group: m = -inf, P = +0)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int lt = set + 4 * i;
      if (lt < nt) {
        const float mm[2] = {sMM[lt * 16 + g], sMM[lt * 16 + g + 8]};
        float p[4];  // rounded to bf16 by the pack below: the bits decode_kernel stores
#pragma unroll
        for (int e = 0; e < 4; ++e) p[e] = ex2(sub(s[i][e], mm[e >> 1]));
        const int key = wi * 8 + t4 * 2;
        *reinterpret_cast<uint32_t*>(sP + lt * kPTile + g * kPStride + key) = pack_bf16(p[0], p[1]);
        *reinterpret_cast<uint32_t*>(sP + lt * kPTile + (g + 8) * kPStride + key) = pack_bf16(p[2], p[3]);
      }
    }
  }
  cp_async_wait<1>();  // group 2
  __syncthreads();     // ... for all threads; P published

  // Warps 8-11: row sums of P of the tiles = warp mod 4 in exactly online_step's order (own keys n = 0..,
  // then lane xor 1, xor 2), while the o chain runs.
  if (dw) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int lt = wi + 4 * i;
      if (lt < nt) {
        const __nv_bfloat16* pt = sP + lt * kPTile;
        float ls[2] = {0.f, 0.f};
#pragma unroll
        for (int n = 0; n < kTileNS; ++n) {
          const __nv_bfloat162 p0 = *reinterpret_cast<const __nv_bfloat162*>(pt + g * kPStride + n * 8 + t4 * 2);
          const __nv_bfloat162 p1 = *reinterpret_cast<const __nv_bfloat162*>(pt + (g + 8) * kPStride + n * 8 + t4 * 2);
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
        if (t4 == 0) {
          sLS[lt * 16 + g] = ls[0];
          sLS[lt * 16 + g + 8] = ls[1];
        }
      }
    }
  }
  // The o chain on warps 0-7, two O n-tiles each: per tile decode_kernel's fma rescale and two key k-steps. P
  // comes as A fragments through ldmatrix (the registers decode_kernel loads word by word).
  float O0[4] = {0.f, 0.f, 0.f, 0.f}, O1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int lt = 0; lt < kGroupTiles; ++lt) {
    if (lt == 8) {
      cp_async_wait<0>();  // group 3
      __syncthreads();     // ... for all threads; row sums published
    }
    if (ow && lt < nt) {
      const float sc0 = sScale[lt * 16 + g], sc1 = sScale[lt * 16 + g + 8];
      O0[0] = __fmaf_rn(O0[0], sc0, 0.f), O0[1] = __fmaf_rn(O0[1], sc0, 0.f);
      O0[2] = __fmaf_rn(O0[2], sc1, 0.f), O0[3] = __fmaf_rn(O0[3], sc1, 0.f);
      O1[0] = __fmaf_rn(O1[0], sc0, 0.f), O1[1] = __fmaf_rn(O1[1], sc0, 0.f);
      O1[2] = __fmaf_rn(O1[2], sc1, 0.f), O1[3] = __fmaf_rn(O1[3], sc1, 0.f);
      const __nv_bfloat16* pt = sP + lt * kPTile + ((mat & 1) * 8 + r8) * kPStride + (mat >> 1) * 8;
      const __nv_bfloat16* vt = d90_v_slot(sKV, lt);
#pragma unroll
      for (int j = 0; j < kTileKS; ++j) {
        uint32_t pa[4], bv[4];
        ldmatrix_x4(pa, pt + j * 16);
        ldmatrix_x4_trans(bv, vt + swz(j * 16 + (mat & 1) * 8 + r8, warp * 2 + (mat >> 1)));
        mma_bf16(O0, pa, bv[0], bv[1]);
        mma_bf16(O1, pa, bv[2], bv[3]);
      }
    }
  }
  if (warp == kD90DWarp && t4 == 0) {  // d = fma(d, scale, row sum), tiles in order
    float D0 = 0.f, D1 = 0.f;
    for (int lt = 0; lt < nt; ++lt) {
      D0 = __fmaf_rn(D0, sScale[lt * 16 + g], sLS[lt * 16 + g]);
      D1 = __fmaf_rn(D1, sScale[lt * 16 + g + 8], sLS[lt * 16 + g + 8]);
    }
    sL[g] = D0, sL[g + 8] = D1;
  }
  __syncthreads();

  // The group is done: output or group partial as in decode_kernel, a warp per two O n-tiles.
  const float l0 = sL[g], l1 = sL[g + 8];
  const int head0 = kvh * group, quarter = warp >> 1;
  if (n_groups == 1) {
    if (!ow) return;
    __nv_bfloat16* o0 = out + ((int64_t)b * num_qo_heads + head0 + g) * kHeadDim;
    __nv_bfloat16* o1 = o0 + 8 * kHeadDim;
    const int col = warp * 16 + t4 * 2;
    if (g < group) {
      *reinterpret_cast<__nv_bfloat162*>(o0 + col) = __halves2bfloat162(finish(O0[0], l0), finish(O0[1], l0));
      *reinterpret_cast<__nv_bfloat162*>(o0 + col + 8) = __halves2bfloat162(finish(O1[0], l0), finish(O1[1], l0));
    }
    if (g + 8 < group) {
      *reinterpret_cast<__nv_bfloat162*>(o1 + col) = __halves2bfloat162(finish(O0[2], l1), finish(O0[3], l1));
      *reinterpret_cast<__nv_bfloat162*>(o1 + col + 8) = __halves2bfloat162(finish(O1[2], l1), finish(O1[3], l1));
    }
    return;
  }
  // A quarter of the head dim (two warps) plays decode_kernel's warp: its dims of the partial, its copy of
  // (m, l), its arrival counter, and the last arrival folds it.
  if (ow) {
    const int64_t slot0 = ((int64_t)b * max_groups + fg) * num_qo_heads + head0 + g;
    const int64_t slot1 = slot0 + 8;
    const int col = warp * 16 + t4 * 2;
    if (g < group) {
      *reinterpret_cast<float2*>(ws_o + slot0 * kHeadDim + col) = make_float2(O0[0], O0[1]);
      *reinterpret_cast<float2*>(ws_o + slot0 * kHeadDim + col + 8) = make_float2(O1[0], O1[1]);
    }
    if (g + 8 < group) {
      *reinterpret_cast<float2*>(ws_o + slot1 * kHeadDim + col) = make_float2(O0[2], O0[3]);
      *reinterpret_cast<float2*>(ws_o + slot1 * kHeadDim + col + 8) = make_float2(O1[2], O1[3]);
    }
    if ((warp & 1) == 0 && t4 == 0) {
      if (g < group) ws_ml[slot0 * kDecodeWarps + quarter] = make_float2(sMF[g], l0);
      if (g + 8 < group) ws_ml[slot1 * kDecodeWarps + quarter] = make_float2(sMF[g + 8], l1);
    }
    __threadfence();  // this lane's part of the partial visible device-wide before the arrival
  }
  __syncthreads();  // ... and every lane's
  if (!ow || (warp & 1)) return;
  int last = 0;
  if (lane == 0) {
    int* cnt = counters + ((int64_t)b * num_kv_heads + kvh) * kDecodeWarps + quarter;
    last = atomicAdd(cnt, 1) == n_groups - 1;
    if (last) *cnt = 0;  // ready for the next launch
  }
  if (!__shfl_sync(0xffffffffu, last, 0)) return;
  __threadfence();
  constexpr int kFoldAhead = 8;
  for (int p = lane; p < group * 8; p += 32) {
    const int h = p >> 3, d0 = quarter * 32 + (p & 7) * 4;
    float Mf = -INFINITY, L = 0.f, Of[4] = {0.f, 0.f, 0.f, 0.f};
    const int64_t slot_h = (int64_t)b * max_groups * num_qo_heads + head0 + h;
    for (int c0 = 0; c0 < n_groups; c0 += kFoldAhead) {
      float2 ml[kFoldAhead];
      float4 oc[kFoldAhead];
#pragma unroll
      for (int i = 0; i < kFoldAhead; ++i) {
        if (c0 + i >= n_groups) break;
        const int64_t slot = slot_h + (int64_t)(c0 + i) * num_qo_heads;
        ml[i] = __ldcg(ws_ml + slot * kDecodeWarps + quarter);
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

// grid = decode90_grid(slots, num_kv_heads, batch): decode_grid with `slots` CTAs per (request, kv head).
__global__ void __launch_bounds__(kD90Warps * 32)
    decode90_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k_cache,
                    const __nv_bfloat16* __restrict__ v_cache, const int32_t* __restrict__ kv_page_indptr,
                    const int32_t* __restrict__ kv_page_indices, int64_t table_stride,
                    const int32_t* __restrict__ kv_lens, float* __restrict__ ws_o, float2* __restrict__ ws_ml,
                    int* __restrict__ counters, __nv_bfloat16* __restrict__ out, int max_groups,
                    int num_qo_heads, int num_kv_heads, int page_size, int page_shift, float scale_log2,
                    bool group_fastest, int slots) {
  const int kvh = group_fastest ? blockIdx.y : blockIdx.x, b = group_fastest ? 0 : blockIdx.y;
  const int kv_len = kv_lens[b];
  const int n_groups = (kv_len + kGroupKeys - 1) / kGroupKeys;
  const int32_t* pages = request_pages(kv_page_indices, kv_page_indptr, table_stride, b);
  for (int fg = group_fastest ? blockIdx.x : blockIdx.z; fg < n_groups; fg += slots)
    decode90_group(q, k_cache, v_cache, pages, kv_len, ws_o, ws_ml, counters, out, max_groups, num_qo_heads,
                   num_kv_heads, page_size, page_shift, scale_log2, b, kvh, fg);
}

}  // namespace pi_attn

// ---------------------------------------------------------------------------------------------------------
// Warp-specialised prefill (prefill90w_kernel<NWG>), the device's prefill kernel on Hopper but for the
// shortest sequences (prefill90w_wg): the same row program, one CTA per SM.
//   threads 0..127   producer: gathers the paged K / V into two-stage rings (a stage: kStageKeys keys = two
//                    contract tiles) with cp.async; a stage is published / recycled through mbarriers
//   threads 128..    NWG consumer warp groups of 64 rows each: 64 NWG rows per CTA share one K / V stream
// A consumer warp group runs its tiles in order, each as prefill90_kernel does: S for a whole stage in one
// m64n64k16 chain, then per tile the online step and P V, waiting on its own tensor-core work only. The time
// of a tile is the CUDA-core work of the 32-key online step, and a second warp group in the SM does not slow
// the first (archive/ceiling_probes/h100_prefill), so the kernel packs up to three into an SM: the most its
// registers hold, with the folded totals in shared memory and one S fragment in registers. Overlapping tile
// t + 1's softmax with tile t's P V in flight measured slower, and so did ping-pong between the warp groups.
namespace pi_ws {
using BF = cutlass::bfloat16_t;
constexpr int kStageTiles = 2, kStageKeys = kStageTiles * pi_attn::kTile, kStages = 2;
static_assert(kStageKeys == 64 && pi_attn::kGroupTiles % kStageTiles == 0);
using TiledMmaQK = decltype(cute::make_tiled_mma(
    cute::GMMA::ss_op_selector<BF, BF, float, cute::Shape<cute::_64, cute::_64, cute::_128>>()));
using SmemLayoutAtomK =
    decltype(cutlass::gemm::collective::detail::ss_smem_selector<cute::GMMA::Major::K, BF, cute::_64, cute::_128>());
// [key][d][stage]
using SmemLayoutK = decltype(cute::tile_to_shape(SmemLayoutAtomK{}, cute::Shape<cute::_64, cute::_128, cute::Int<kStages>>{}));
using SmemLayoutVt = decltype(cute::composition(
    SmemLayoutK{}, cute::make_ordered_layout(cute::Shape<cute::_128, cute::_64, cute::Int<kStages>>{}, cute::Step<cute::_2, cute::_1, cute::_3>{})));
using PipeState = cutlass::PipelineState<kStages>;  // (stage index, phase) of a ring position
using MBar = cutlass::arch::ClusterBarrier;

// A ring of kStages K or V stages. full[i]: the stage's copies have landed (producer -> consumers);
// empty[i]: every wgmma that read the stage has completed (consumers -> producer).
struct Ring {
  uint64_t full[kStages], empty[kStages];
};

template <int NWG>  // consumer warp groups
struct Smem {
  cute::array_aligned<BF, cute::cosize_v<pi_wg::SmemLayoutQ>> q[NWG];  // per consumer warp group
  cute::array_aligned<BF, cute::cosize_v<SmemLayoutK>> k;
  cute::array_aligned<BF, cute::cosize_v<SmemLayoutK>> v;
  float tot[NWG][pi_attn::kNTilesO * 4][128];  // folded totals: [warp group][element of the thread][thread]
  Ring ring_k, ring_v;
};

__device__ __forceinline__ void* q_ptr(BF* q, int row, int col) {
  auto sQ = cute::make_tensor(cute::make_smem_ptr(q), pi_wg::SmemLayoutQ{});
  return &sQ(row, col);
}
__device__ __forceinline__ void* kv_ptr(BF* base, int stage, int key, int col) {
  auto sK = cute::make_tensor(cute::make_smem_ptr(base), SmemLayoutK{});
  return &sK(key, col, stage);
}

// Issue S = +0 + Q K(stage)^T for the stage's 64 keys: 8 ascending k16 steps over the head dim.
__device__ __forceinline__ void qk_issue(BF* q, BF* k, int ct, float (&s)[2 * pi_attn::kTileNS][4], int stage) {
  using namespace cute;
  Tensor sQ = make_tensor(make_smem_ptr(q), pi_wg::SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(k), SmemLayoutK{});
  TiledMmaQK mma_qk;
  auto tqk = mma_qk.get_thread_slice(ct);
  Tensor tSrQ = tqk.partition_fragment_A(sQ);
  Tensor tSrK = tqk.partition_fragment_B(sK);
  auto lay = partition_fragment_C(mma_qk, Shape<_64, _64>{}).layout();
  Tensor tS = make_tensor(make_rmem_ptr<float>(&s[0][0]), lay);
  static_assert(decltype(size(tS))::value == 32);
  CUTE_UNROLL
  for (int i = 0; i < 32; ++i) tS(i) = 0.f;
  warpgroup_fence_operand(tS);
  warpgroup_arrive();
  CUTE_UNROLL
  for (int kb = 0; kb < size<2>(tSrQ); ++kb) cute::gemm(mma_qk, tSrQ(_, _, kb), tSrK(_, _, kb, stage), tS);
  warpgroup_commit_batch();
}

// Issue O = O + P V for contract tile j of the stage: 2 ascending k16 steps over the tile's keys.
__device__ __forceinline__ void pv_issue(BF* v, int ct, uint32_t (&pa)[2][4], float (&o)[16][4], int stage, int j) {
  using namespace cute;
  Tensor sVt = make_tensor(make_smem_ptr(v), SmemLayoutVt{});
  pi_wg::TiledMmaQK mma_s;
  pi_wg::TiledMmaPV mma_pv;
  auto tpv = mma_pv.get_thread_slice(ct);
  Tensor tOrV = tpv.partition_fragment_B(sVt);
  auto lay_s = partition_fragment_C(mma_s, Shape<_64, _32>{}).layout();
  auto lay_o = partition_fragment_C(mma_pv, Shape<_64, _128>{}).layout();
  Tensor tOrP = make_tensor(make_rmem_ptr<BF>(reinterpret_cast<BF*>(&pa[0][0])), pi_wg::acc_to_aregs(lay_s));
  Tensor tOrO = make_tensor(make_rmem_ptr<float>(&o[0][0]), lay_o);
  static_assert(decltype(size(tOrO))::value == 64 && decltype(size(tOrP))::value == 16);
  static_assert(decltype(size<2>(tOrV))::value == 4);
  warpgroup_fence_operand(tOrP);
  warpgroup_fence_operand(tOrO);
  warpgroup_arrive();
  CUTE_UNROLL
  for (int kb = 0; kb < size<2>(tOrP); ++kb) cute::gemm(mma_pv, tOrP(_, _, kb), tOrV(_, _, 2 * j + kb, stage), tOrO);
  warpgroup_commit_batch();
}

// Every issued wgmma has completed; x may be read.
template <int N>
__device__ __forceinline__ void wait0(float (&x)[N][4]) {
  cute::warpgroup_wait<0>();
#pragma unroll
  for (int n = 0; n < N; ++n)
#pragma unroll
    for (int i = 0; i < 4; ++i) cute::warpgroup_fence_operand(x[n][i]);
}
__device__ __forceinline__ void ring_init(Ring& r, int producer_arrivals, int consumer_arrivals) {
#pragma unroll
  for (int i = 0; i < kStages; ++i) {
    MBar::init(&r.full[i], producer_arrivals);
    MBar::init(&r.empty[i], consumer_arrivals);
  }
}
}  // namespace pi_ws

namespace pi_attn {

// online_step of pi_attn.cuh up to, and without, its last loop (o = fma(o, scale, +0)), which the caller runs:
// the same operations in the same order on (m, d), P and scale. P is rounded to bf16 once per pair, straight
// into the A-fragment word (cvt.rn.bf16x2.f32 rounds each half as cvt.rn.bf16.f32 does); the rounded values the
// row sum adds are the word's halves.
__device__ __forceinline__ void online_scores(float (&m)[2], float (&d)[2], float (*v)[4], uint32_t (&pa)[kTileKS][4],
                                              float (&scale)[2]) {
  float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
  for (int n = 0; n < kTileNS; ++n)
#pragma unroll
    for (int i = 0; i < 4; ++i) mx[i >> 1] = fmaxf(mx[i >> 1], v[n][i]);
  float mm[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
    mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
    const float mn = fmaxf(m[r], mx[r]);
    mm[r] = mn == -INFINITY ? 0.f : mn;
    scale[r] = ex2(sub(m[r], mm[r]));
    m[r] = mn;
  }
  float ls[2] = {0.f, 0.f};
#pragma unroll
  for (int n = 0; n < kTileNS; ++n) {
    float e[4], p[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) e[i] = ex2(sub(v[n][i], mm[i >> 1]));
    const uint32_t w0 = pack_bf16(e[0], e[1]), w1 = pack_bf16(e[2], e[3]);
    p[0] = __uint_as_float(w0 << 16);
    p[1] = __uint_as_float(w0 & 0xffff0000u);
    p[2] = __uint_as_float(w1 << 16);
    p[3] = __uint_as_float(w1 & 0xffff0000u);
#pragma unroll
    for (int i = 0; i < 4; ++i) ls[i >> 1] = add(ls[i >> 1], p[i]);
    pa[n >> 1][(n & 1) * 2 + 0] = w0;
    pa[n >> 1][(n & 1) * 2 + 1] = w1;
  }
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    ls[r] = add(ls[r], __shfl_xor_sync(0xffffffffu, ls[r], 1));
    ls[r] = add(ls[r], __shfl_xor_sync(0xffffffffu, ls[r], 2));
    d[r] = __fmaf_rn(d[r], scale[r], ls[r]);
  }
}

// Rows and threads of a CTA with nwg consumer warp groups (2 or 3).
constexpr int prefill90w_rows(int nwg) { return 64 * nwg; }
constexpr int prefill90w_threads(int nwg) { return 128 * (1 + nwg); }
template <int NWG>
constexpr int prefill90w_smem_bytes() { return (int)sizeof(pi_ws::Smem<NWG>); }
// Which prefill kernel a Hopper device runs for a shape: prefill90w_kernel with the returned number of consumer
// warp groups, or (0) prefill90_kernel. Three warp groups when the 192-row CTAs fill the SMs twice over;
// otherwise two, which makes more and shorter CTAs for short prompts and short extends; prefill90_kernel for at
// most 128 new tokens inside one fold group, where a CTA runs a few tiles and its start-up (384 threads, the
// rings) costs more than they do (archive/ceiling_probes/h100_prefill). max_groups: prefill_plan's.
inline int prefill90w_wg(int batch, int num_qo_heads, int num_kv_heads, int max_q_len, int max_groups, int sms) {
  if (max_groups == 1 && max_q_len <= 4 * kTile) return 0;
  const int64_t rows = (int64_t)max_q_len * (num_qo_heads / std::max(num_kv_heads, 1));
  const int64_t ctas = (rows + 191) / 192 * num_kv_heads * batch;
  return ctas >= 2 * (int64_t)sms ? 3 : 2;
}
// The grid of a prefill90w launch: longest first. A causal row tile runs as many key tiles as its position, and
// CTAs are dispatched in grid order (x fastest): with the row tiles descending in the slowest index the long
// CTAs start first and the short ones fill the SMs behind them (ascending, the last SMs run the longest CTAs
// alone: 15-60% more wall time on one-request prefills).
inline dim3 prefill90w_grid(int row_tiles, int num_kv_heads, int batch_splits) {
  return dim3(num_kv_heads, batch_splits, row_tiles);
}

// Arguments and split semantics of prefill_kernel; grid = prefill90w_grid(row tiles of prefill90w_rows(NWG), ..).
template <int NWG>
__global__ void __launch_bounds__(128 * (1 + NWG), 1)
    prefill90w_kernel(const __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ out,
                      const __nv_bfloat16* __restrict__ k_cache, const __nv_bfloat16* __restrict__ v_cache,
                      const int32_t* __restrict__ qo_indptr, const int32_t* __restrict__ kv_page_indptr,
                      const int32_t* __restrict__ kv_page_indices, int64_t table_stride,
                      const int32_t* __restrict__ kv_lens, const int32_t* __restrict__ prefix_lens,
                      int num_qo_heads, int num_kv_heads, int page_size, int page_shift, float scale_log2,
                      int kv_splits, float* __restrict__ ws_o, float2* __restrict__ ws_ml, int max_groups) {
  constexpr int kRows = 64 * NWG, kStageKeys = pi_ws::kStageKeys, kWG = NWG;
  using pi_ws::MBar;
  using pi_ws::PipeState;
  using pi_ws::Ring;
  extern __shared__ char smem_buf[];
  pi_ws::Smem<NWG>& sm = *reinterpret_cast<pi_ws::Smem<NWG>*>(smem_buf);
  const int b = blockIdx.y / kv_splits, split = blockIdx.y % kv_splits;
  const int kvh = blockIdx.x, tile = gridDim.z - 1 - blockIdx.z;
  const int group = num_qo_heads / num_kv_heads;
  const int q0 = qo_indptr[b], q_len = qo_indptr[b + 1] - q0;
  const int rows_total = q_len * group;
  if (tile * kRows >= rows_total) return;
  const int kv_len = kv_lens ? kv_lens[b] : prefix_lens[b] + q_len, prefix = kv_len - q_len;
  const int32_t* pages = request_pages(kv_page_indices, kv_page_indptr, table_stride, b);
  const int row0 = tile * kRows;
  const int last_pos = prefix + (min(row0 + kRows, rows_total) - 1) / group;
  const int tile_groups = last_pos / kGroupKeys + 1;
  const int per_split = (tile_groups + kv_splits - 1) / kv_splits;
  const int g_begin = split * per_split, g_end = min(tile_groups, g_begin + per_split);
  if (g_begin >= g_end) return;
  const int t_begin = g_begin * kGroupTiles;
  const int t_last = min(g_end * kGroupTiles, last_pos / kTile + 1) - 1;
  const int h_begin = t_begin / pi_ws::kStageTiles, h_last = t_last / pi_ws::kStageTiles;  // stages
  const int wg = threadIdx.x >> 7, ct = threadIdx.x & 127;

  // Arrivals per stage: every producer thread, when its copies have landed; one lane of every consumer warp.
  if (threadIdx.x == 0) {
    pi_ws::ring_init(sm.ring_k, 128, kWG * 4);
    pi_ws::ring_init(sm.ring_v, 128, kWG * 4);
  }
  cutlass::arch::fence_barrier_init();
  __syncthreads();

  // Registers per thread the CTA is launched with: 65536 / threads rounded down to 8, that is 128 (three
  // consumer warp groups) or 168 (two); the producer keeps 40 and each consumer warp group takes
  // 152 (40 + 3 * 152 <= 4 * 128) or 232 (40 + 2 * 232 = 3 * 168).
  static_assert(kWG == 2 || kWG == 3, "register split");
  if (wg == 0) {
    // Producer. Thread ct copies key ct / 2 of a stage, 16-byte chunks ct % 2 + 2 i; a key at or past kv_len
    // is zero-filled without a global read. Order: K(h), V(h), K(h + 1), .., as the consumers take them.
    cutlass::arch::warpgroup_reg_dealloc<40>();
    const int key = ct >> 1, c0 = ct & 1;
    auto row_of = [&](int h) -> int64_t {
      const int pos = h * kStageKeys + key;
      if (h > h_last || pos >= kv_len) return -1;
      const int slot = page_shift >= 0 ? (pages[pos >> page_shift] << page_shift) + (pos & (page_size - 1))
                                       : pages[pos / page_size] * page_size + pos % page_size;
      return ((int64_t)slot * num_kv_heads + kvh) * kHeadDim;
    };
    auto load = [&](Ring& ring, PipeState& st, pi_ws::BF* dst, const __nv_bfloat16* cache, int64_t row) {
      MBar::wait(&ring.empty[st.index()], st.phase());  // the stage is free
      const __nv_bfloat16* src = cache + (row >= 0 ? row : 0) + c0 * 8;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        cp_async16(pi_ws::kv_ptr(dst, st.index(), key, (c0 + 2 * i) * 8), src + 16 * i, row >= 0 ? 16 : 0);
      cutlass::arch::cpasync_barrier_arrive_noinc(&ring.full[st.index()]);  // arrives when the copies land
      ++st;
    };
    PipeState wk(0, 1, 0), wv(0, 1, 0);  // phase 1: a stage's first wait for "free" passes
    int64_t row = row_of(h_begin);
    for (int h = h_begin; h <= h_last; ++h) {
      const int64_t next = row_of(h + 1);
      load(sm.ring_k, wk, sm.k.data(), k_cache, row);
      load(sm.ring_v, wv, sm.v.data(), v_cache, row);
      row = next;
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
    return;
  }

  // Consumers: warp group wgc holds rows [wgc * 64, wgc * 64 + 64) of the CTA's row tile.
  cutlass::arch::warpgroup_reg_alloc<kWG == 3 ? 152 : 232>();
  const int wgc = wg - 1, warp = ct >> 5, lane = ct & 31;
  const int g = lane >> 2, t4 = lane & 3;
  const int wg_first = row0 + wgc * 64, warp_first = wg_first + warp * 16;
  int rpos[2];  // position of this thread's rows g, g + 8; -1 = padding
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int i = warp_first + g + r * 8;
    rpos[r] = i < rows_total ? prefix + i / group : -1;
  }
  // Every row of the warp group sees every key up to this position (-1: the group has padding rows).
  const int wg_min_pos = wg_first + 63 < rows_total ? prefix + wg_first / group : -1;

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int idx = ct + i * 128, r = idx >> 4, chunk = idx & 15, qi = wg_first + r;
    const __nv_bfloat16* src = q;
    int bytes = 0;
    if (qi < rows_total) {
      src = q + (int64_t)(q0 + qi / group) * num_qo_heads * kHeadDim + (kvh * group + qi % group) * kHeadDim + chunk * 8;
      bytes = 16;
    }
    cp_async16(pi_ws::q_ptr(sm.q[wgc].data(), r, chunk * 8), src, bytes);
  }
  asm volatile("cp.async.wait_all;" ::: "memory");
  cutlass::arch::NamedBarrier::sync(128, wgc);  // this warp group's Q is in shared memory

  // (m, d, o): the current group. The folded total of the finished groups: (tot_m, tot_d) and, in this
  // thread's shared-memory slots, tot, which is written by the first fold that is not the last (stored): until
  // then the total is +0, and the last fold's result goes straight to the output.
  float m[2] = {-INFINITY, -INFINITY}, d[2] = {0.f, 0.f}, o[kNTilesO][4];
  float tot_m[2] = {-INFINITY, -INFINITY}, tot_d[2] = {0.f, 0.f};
  float* tot = &sm.tot[wgc][0][ct];
  bool stored = false;
#pragma unroll
  for (int n = 0; n < kNTilesO; ++n)
#pragma unroll
    for (int i = 0; i < 4; ++i) o[n][i] = pi_wg::zero_reg();

  // Group fg is finished: fold it into the total (kv_splits == 1; on the last group also normalise and
  // write out) or write it as a group partial.
  auto finish_group = [&](int fg, bool last) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int i = warp_first + g + r * 8;
      if (kv_splits > 1) {
        if (i >= rows_total || fg * kGroupKeys > rpos[r]) continue;
        const int64_t slot = ((int64_t)(q0 + i / group) * num_qo_heads + kvh * group + i % group) * max_groups + fg;
        if (t4 == 0) ws_ml[slot] = make_float2(m[r], d[r]);
#pragma unroll
        for (int n = 0; n < kNTilesO; ++n)
          *reinterpret_cast<float2*>(ws_o + slot * kHeadDim + n * 8 + t4 * 2) = make_float2(o[n][2 * r], o[n][2 * r + 1]);
        continue;
      }
      const FoldScalars f = fold_scalars(tot_m[r], tot_d[r], m[r], d[r]);
      const bool write = last && i < rows_total;
      __nv_bfloat16* dst = out + ((int64_t)(q0 + i / group) * num_qo_heads + kvh * group + i % group) * kHeadDim;
#pragma unroll
      for (int n = 0; n < kNTilesO; ++n) {
        float folded[2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          float& to = tot[(n * 4 + 2 * r + e) * 128];
          folded[e] = fold_elem(stored ? to : 0.f, o[n][2 * r + e], f);
          if (!last) to = folded[e];
        }
        if (write) {
          __nv_bfloat162 ob;
          ob.x = finish(folded[0], tot_d[r]);
          ob.y = finish(folded[1], tot_d[r]);
          *reinterpret_cast<__nv_bfloat162*>(dst + n * 8 + t4 * 2) = ob;
        }
      }
    }
    stored = kv_splits == 1;
    m[0] = m[1] = -INFINITY;
    d[0] = d[1] = 0.f;
#pragma unroll
    for (int n = 0; n < kNTilesO; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) o[n][i] = pi_wg::zero_reg();
  };

  float s[2 * kTileNS][4];  // S of the current stage
  PipeState rk, rv;         // the K and V stage to take
  auto ring_release = [&](Ring& ring, PipeState& st) {
    if (lane == 0) MBar::arrive(&ring.empty[st.index()]);
    ++st;
  };
  // Tile t = 2 h + j: scale and mask S as prefill_kernel, the online step, o += P V.
  auto tile_step = [&](int t, int j) {
    float (*v)[4] = &s[j * kTileNS];
    uint32_t pa[kTileKS][4];
    float sc[2];
    const int key0 = t * kTile;
    if (key0 + kTile - 1 <= wg_min_pos) {  // no row of the warp group masks a key of this tile
#pragma unroll
      for (int n = 0; n < kTileNS; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) v[n][i] = mul(v[n][i], scale_log2);
    } else {
#pragma unroll
      for (int n = 0; n < kTileNS; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int key = key0 + n * 8 + t4 * 2 + (i & 1);
          v[n][i] = key <= rpos[i >> 1] ? mul(v[n][i], scale_log2) : -INFINITY;
        }
    }
    online_scores(m, d, v, pa, sc);
#pragma unroll
    for (int n = 0; n < kNTilesO; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) o[n][i] = __fmaf_rn(o[n][i], sc[i >> 1], 0.f);  // -0 -> +0
    pi_ws::pv_issue(sm.v.data(), ct, pa, o, rv.index(), j);
    pi_ws::wait0(o);
    if (t % kGroupTiles == kGroupTiles - 1 || t == t_last) finish_group(t / kGroupTiles, t == t_last);
  };

  for (int h = h_begin; h <= h_last; ++h) {
    MBar::wait(&sm.ring_k.full[rk.index()], rk.phase());
    pi_ws::qk_issue(sm.q[wgc].data(), sm.k.data(), ct, s, rk.index());
    pi_ws::wait0(s);
    ring_release(sm.ring_k, rk);
    MBar::wait(&sm.ring_v.full[rv.index()], rv.phase());
    tile_step(h * pi_ws::kStageTiles, 0);
    if (h * pi_ws::kStageTiles + 1 <= t_last) tile_step(h * pi_ws::kStageTiles + 1, 1);
    ring_release(sm.ring_v, rv);
  }
}

}  // namespace pi_attn
