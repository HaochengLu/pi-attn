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

namespace pi_wg {
using BF = cutlass::bfloat16_t;
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
    st.reset();
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
    if (t % kGroupTiles == kGroupTiles - 1 || t == t_last) finish_group(t / kGroupTiles, t == t_last);
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

}  // namespace pi_attn
