// Decode on 16 warps for sm_80: decode_kernel's program for a whole fold group per CTA, with the mma operands
// swapped.
//
// decode_kernel is bound by what its 4 warps execute in sequence (a tile at a time behind three barriers), and
// the contract serialises little of a group: S of a key depends on that key only, the running max is a prefix max
// of exact tile maxima, scale_t and P_t are elementwise functions of S and the max, the row sums of P_t are
// independent per tile, and only d_t and o_t are chains, o_t independently per head dim. decode90_kernel
// (pi_attn_sm90.cuh) runs a group that way on Hopper. This kernel also takes the padding out of the mma:
// mma.sync m16n8k16 is symmetric in its operands - D[i][j] of (A = X, B = Y, C) equals D'[j][i] of (A = Y, B = X,
// C^T) bit for bit (1.69e9 elements on an A800, special values included: archive/ceiling_probes/a800_r5/
// mma_operand_swap; the tensor-core models of paper section 7 group the products by k alone) - so the KEYS take
// the 16-row operand and the GQA rows of a kv head, at most 8, the 8 columns:
//   S^T [32 keys x 8 heads] = K Q^T: 2 mma chains per tile (16 keys each) where decode_kernel runs 4 (8 keys by
//       16 rows, 16 - group of them padding); a warp runs a whole tile and has its exact tile maximum itself;
//   o^T [128 dims x 8 heads] = V^T P^T: per tile and 16 dims one accumulator of 2 key k-steps where decode_kernel
//       runs two (8 dims each).
// Every element still runs decode_kernel's instruction sequence on the same operands, the two mma operands
// exchanged: head_dim k-steps ascending from +0 for S; fma rescale, then the two key k-steps ascending, for o; P
// rounded elementwise; the row sums and the d chain formed from P in shared memory in online_step's order. The
// same bits. The group partial, the arrival counters and the fold are decode_kernel's, per 32-dim quarter of
// the head (two warps).
//   - the 16 warps issue the group's K in one cp.async group (16 tile slots, 128 KB), and V into the same slots
//     once S is done (tiles 0-7, then 8-15);
//   - warp t forms S^T, the tile maximum and then P^T of tile t; warp 8 the running max, the scales and the d
//     chain; warps 8-11 the row sums of P; warps 0-7 the o chain, 16 dims each.
// A CTA holds 138 KB of shared memory, one per SM; needs a GQA group of at most 8. The CTAs of a launch:
// decode16_layout; where it runs: decode_plan (pi_attn_kernels.cu). The unswapped 16-warp kernel and the timings
// of both: archive/ceiling_probes/a800_r5/decode16.
#pragma once
namespace pi_attn {
// Warps 0-7 run the o chain, warps 8-11 the row sums, warp 8 the running max and the d chain.
constexpr int kD16Warps = 16, kD16OWarps = 8, kD16DWarp = 8;
constexpr int decode16_smem_bytes() {
  return 16 * kTile * kHeadDim * 2      // 16 tile slots: K, then V
         + kGroupTiles * kTile * 8 * 2  // P^T of every tile: [32 keys][8 heads]
         + kGroupTiles * 4 * 8 * 4      // per tile and head: tile max, m, scale, row sum of P
         + 2 * 8 * 4;                   // d and m of the group
}

// Tile slot of V(lt): K(lt)'s, once S is done.
__device__ __forceinline__ __nv_bfloat16* d16_v_slot(__nv_bfloat16* sKV, int lt) {
  return sKV + lt * (kTile * kHeadDim);
}

// Fold group fg of (request b, kv head kvh), by all the CTA's threads.
__device__ __forceinline__ void decode16_group(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache, const int32_t* __restrict__ pages, int kv_len, float* __restrict__ ws_o,
    float2* __restrict__ ws_ml, int* __restrict__ counters, __nv_bfloat16* __restrict__ out, int max_groups,
    int num_qo_heads, int num_kv_heads, int page_size, int page_shift, float scale_log2, int b, int kvh, int fg) {
  static_assert(kTile == 32 && kGroupTiles == 16 && kD16Warps == 16 && kHeadDim == 16 * kD16OWarps && kDecodeWarps == 4,
                "a tile per warp, 16 dims per o warp, 4 quarters");
  constexpr int kPTile = kTile * 8, kTileElems = kTile * kHeadDim;
  extern __shared__ __align__(16) __nv_bfloat16 smem[];
  __nv_bfloat16* sKV = smem;                             // [16][kTile][128], swz per tile: K(lt), then V(lt)
  __nv_bfloat16* sP = sKV + 16 * kTileElems;             // [16 tiles][32 keys][8 heads]
  float* sTMax = reinterpret_cast<float*>(sP + kGroupTiles * kPTile);  // [16 tiles][8 heads] tile max
  float* sMM = sTMax + kGroupTiles * 8;                  // ... running max after the tile (0 if -inf)
  float* sScale = sMM + kGroupTiles * 8;                 // ... scale
  float* sLS = sScale + kGroupTiles * 8;                 // ... row sum of P
  float* sL = sLS + kGroupTiles * 8;                     // [8 heads] d of the group
  float* sMF = sL + 8;                                   // [8 heads] m of the group
  const int pos = kv_len - 1;
  const int n_tiles = (kv_len + kTile - 1) / kTile;
  const int n_groups = (n_tiles + kGroupTiles - 1) / kGroupTiles;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t4 = lane & 3, mat = lane >> 3, r8 = lane & 7;
  const int set = warp >> 2, wi = warp & 3;  // copies: tiles = set mod 4
  const bool ow = warp < kD16OWarps, dw = set == kD16DWarp / 4;
  const int ctid = tid & 127;                // issue_tile's thread index within the set
  const int group = num_qo_heads / num_kv_heads;
  const int t_begin = fg * kGroupTiles;
  const int nt = min(n_tiles - t_begin, kGroupTiles);  // tiles of this group

  // Loads, group 1: K of the warp set's 4 tiles.
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
  // Q as B fragments: column n = g is head g (zero beyond the group), rows k = the 16 dims of a k-step.
  uint32_t qb[kKStepsQK][2];
  {
    const __nv_bfloat16* qrow = g < group ? q + ((int64_t)b * num_qo_heads + kvh * group + g) * kHeadDim : nullptr;
#pragma unroll
    for (int ks = 0; ks < kKStepsQK; ++ks) {
      qb[ks][0] = qrow ? *reinterpret_cast<const uint32_t*>(qrow + ks * 16 + t4 * 2) : 0u;
      qb[ks][1] = qrow ? *reinterpret_cast<const uint32_t*>(qrow + ks * 16 + 8 + t4 * 2) : 0u;
    }
  }

  // S^T of tile `warp`: s[mt][i] = S[head 2 t4 + (i & 1)][key 16 mt + g + 8 (i >> 1)], its k-steps ascending.
  float s[2][4];
  cp_async_wait<0>();  // group 1
  __syncthreads();
  if (warp < nt) {
    const int lt = warp;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) s[mt][0] = s[mt][1] = s[mt][2] = s[mt][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < kKStepsQK; ++ks)
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        uint32_t ak[4];
        ldmatrix_x4(ak, sKV + lt * kTileElems + swz(mt * 16 + (mat & 1) * 8 + r8, ks * 2 + (mat >> 1)));
        mma_bf16(s[mt], ak, qb[ks][0], qb[ks][1]);
      }
    float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int head = 2 * t4 + (e & 1), key = (t_begin + lt) * kTile + mt * 16 + g + (e >> 1) * 8;
        s[mt][e] = (head < group && key <= pos) ? mul(s[mt][e], scale_log2) : -INFINITY;
        mx[e & 1] = fmaxf(mx[e & 1], s[mt][e]);
      }
#pragma unroll
    for (int r = 0; r < 2; ++r) {  // the exact tile maximum: over the 8 lanes of a t4
      mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 4));
      mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 8));
      mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 16));
    }
    if (g == 0) {
      sTMax[lt * 8 + 2 * t4] = mx[0];
      sTMax[lt * 8 + 2 * t4 + 1] = mx[1];
    }
  }
  __syncthreads();  // maxima published; all warps done with K: the slots take V (tiles 0-7: group 2, 8-15: group 3)
  {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int lt = set + 4 * i;
      if (lt < nt) issue_tile(d16_v_slot(sKV, lt), v_cache, rows[i], ctid);
    }
  }
  cp_async_commit();
  {
#pragma unroll
    for (int i = 2; i < 4; ++i) {
      const int lt = set + 4 * i;
      if (lt < nt) issue_tile(d16_v_slot(sKV, lt), v_cache, rows[i], ctid);
    }
  }
  cp_async_commit();
  if (warp == kD16DWarp && lane < 8) {  // the running max and the scale of every tile, the group's m; head = lane
    float M = -INFINITY;
    for (int lt = 0; lt < nt; ++lt) {
      const float mn = fmaxf(M, sTMax[lt * 8 + lane]);
      const float mm = mn == -INFINITY ? 0.f : mn;
      sMM[lt * 8 + lane] = mm;
      sScale[lt * 8 + lane] = ex2(sub(M, mm));
      M = mn;
    }
    sMF[lane] = M;
  }
  __syncthreads();
  if (warp < nt) {  // P^T of the warp's tile, as decode_kernel forms P (heads >= group: m = -inf, P = +0)
    const int lt = warp;
    const float mm[2] = {sMM[lt * 8 + 2 * t4], sMM[lt * 8 + 2 * t4 + 1]};
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      float p[4];  // rounded to bf16 by the pack below: the bits decode_kernel stores
#pragma unroll
      for (int e = 0; e < 4; ++e) p[e] = ex2(sub(s[mt][e], mm[e & 1]));
      __nv_bfloat16* pk = sP + lt * kPTile + (mt * 16 + g) * 8 + 2 * t4;
      *reinterpret_cast<uint32_t*>(pk) = pack_bf16(p[0], p[1]);
      *reinterpret_cast<uint32_t*>(pk + 8 * 8) = pack_bf16(p[2], p[3]);
    }
  }
  cp_async_wait<1>();  // group 2
  __syncthreads();     // ... for all threads; P published

  // Warps 8-11: row sums of P of the tiles = warp mod 4 in exactly online_step's order (own keys n = 0..,
  // then lane xor 1, xor 2), head = g, while the o chain runs.
  if (dw) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int lt = wi + 4 * i;
      if (lt < nt) {
        const __nv_bfloat16* pt = sP + lt * kPTile + g;
        float ls = 0.f;
#pragma unroll
        for (int n = 0; n < kTileNS; ++n) {
          ls = add(ls, __bfloat162float(pt[(n * 8 + t4 * 2) * 8]));
          ls = add(ls, __bfloat162float(pt[(n * 8 + t4 * 2 + 1) * 8]));
        }
        ls = add(ls, __shfl_xor_sync(0xffffffffu, ls, 1));
        ls = add(ls, __shfl_xor_sync(0xffffffffu, ls, 2));
        if (t4 == 0) sLS[lt * 8 + g] = ls;
      }
    }
  }
  // The o chain on warps 0-7, 16 dims each: O[i] = o[head 2 t4 + (i & 1)][dim 16 warp + g + 8 (i >> 1)]; per tile
  // decode_kernel's fma rescale and two key k-steps.
  float O[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int lt = 0; lt < kGroupTiles; ++lt) {
    if (lt == 8) {
      cp_async_wait<0>();  // group 3
      __syncthreads();     // ... for all threads; row sums published
    }
    if (ow && lt < nt) {
      const float sc0 = sScale[lt * 8 + 2 * t4], sc1 = sScale[lt * 8 + 2 * t4 + 1];
      O[0] = __fmaf_rn(O[0], sc0, 0.f), O[1] = __fmaf_rn(O[1], sc1, 0.f);
      O[2] = __fmaf_rn(O[2], sc0, 0.f), O[3] = __fmaf_rn(O[3], sc1, 0.f);
      const __nv_bfloat16* vt = d16_v_slot(sKV, lt);
      uint32_t pb[4];  // P^T as B fragments of the two key k-steps
      ldmatrix_x4_trans(pb, sP + lt * kPTile + (mat * 8 + r8) * 8);
#pragma unroll
      for (int j = 0; j < kTileKS; ++j) {
        uint32_t av[4];  // V^T as A fragments: rows = 16 dims, k = 16 keys
        ldmatrix_x4_trans(av, vt + swz(j * 16 + (mat >> 1) * 8 + r8, warp * 2 + (mat & 1)));
        mma_bf16(O, av, pb[2 * j], pb[2 * j + 1]);
      }
    }
  }
  if (warp == kD16DWarp && lane < 8) {  // d = fma(d, scale, row sum), tiles in order
    float Dd = 0.f;
    for (int lt = 0; lt < nt; ++lt) Dd = __fmaf_rn(Dd, sScale[lt * 8 + lane], sLS[lt * 8 + lane]);
    sL[lane] = Dd;
  }
  __syncthreads();

  // The group is done: output or group partial as in decode_kernel.
  const int head0 = kvh * group, quarter = warp >> 1;
  if (n_groups == 1) {
    if (!ow) return;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int h = 2 * t4 + (i & 1), dim = warp * 16 + g + (i >> 1) * 8;
      if (h < group) out[((int64_t)b * num_qo_heads + head0 + h) * kHeadDim + dim] = finish(O[i], sL[h]);
    }
    return;
  }
  // A quarter of the head dim (two warps) plays decode_kernel's warp: its dims of the partial, its copy of
  // (m, l), its arrival counter, and the last arrival folds it.
  if (ow) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int h = 2 * t4 + (i & 1), dim = warp * 16 + g + (i >> 1) * 8;
      const int64_t slot = ((int64_t)b * max_groups + fg) * num_qo_heads + head0 + h;
      if (h < group) ws_o[slot * kHeadDim + dim] = O[i];
    }
    if ((warp & 1) == 0 && lane < 8 && lane < group) {
      const int64_t slot = ((int64_t)b * max_groups + fg) * num_qo_heads + head0 + lane;
      ws_ml[slot * kDecodeWarps + quarter] = make_float2(sMF[lane], sL[lane]);
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

// grid = decode16_grid(decode16_layout(..), num_kv_heads, batch): decode_grid's, with the helper rows after the
// requests in y.
__global__ void __launch_bounds__(kD16Warps * 32)
    decode16_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k_cache,
                    const __nv_bfloat16* __restrict__ v_cache, const int32_t* __restrict__ kv_page_indptr,
                    const int32_t* __restrict__ kv_page_indices, int64_t table_stride,
                    const int32_t* __restrict__ kv_lens, float* __restrict__ ws_o, float2* __restrict__ ws_ml,
                    int* __restrict__ counters, __nv_bfloat16* __restrict__ out, int max_groups,
                    int num_qo_heads, int num_kv_heads, int page_size, int page_shift, float scale_log2,
                    bool group_fastest, int batch, int slots, int helper_rows) {
  const int kvh = group_fastest ? blockIdx.y : blockIdx.x, y = group_fastest ? 0 : blockIdx.y;
  const int j = group_fastest ? blockIdx.x : blockIdx.z;
  const bool helper = y >= batch;  // helper row y - batch: the last groups of the requests y - batch (+ helper_rows)
  for (int i = 0;; ++i) {
    const int b = helper ? y - batch + i * helper_rows : y;
    if (helper && (i == 2 || b >= batch)) break;
    const int kv_len = kv_lens[b];
    const int n_groups = (kv_len + kGroupKeys - 1) / kGroupKeys;
    int fg = j + i * slots;
    if (helper) {
      fg = n_groups - 1;
      if (n_groups <= slots) continue;  // every group has a column CTA of its own
    } else if (fg >= n_groups || (fg == n_groups - 1 && n_groups > slots && b < 2 * helper_rows)) {
      break;
    }
    decode16_group(q, k_cache, v_cache, request_pages(kv_page_indices, kv_page_indptr, table_stride, b), kv_len,
                   ws_o, ws_ml, counters, out, max_groups, num_qo_heads, num_kv_heads, page_size, page_shift,
                   scale_log2, b, kvh, fg);
  }
}

}  // namespace pi_attn
