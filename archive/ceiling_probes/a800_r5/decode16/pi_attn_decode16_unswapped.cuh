// Decode on 16 warps for sm_80: decode_kernel's program for a whole fold group per CTA, as decode90_kernel runs
// it on Hopper (pi_attn_sm90.cuh, which explains the warp roles), in 160 KB of shared memory, which sm_80's 164 KB
// holds: the group's 16 K tiles land first, and their slots take the V tiles once S is done (decode90_kernel keeps
// 8 more slots and loads half of V with K). Every element runs decode_kernel's instruction sequence on the same
// operands: the same bits. The CTAs of a launch: decode16_layout; where it runs: decode_plan (pi_attn_kernels.cu).
#pragma once
namespace pi_attn {
constexpr int kD16Warps = 16, kD16OWarps = 8, kD16DWarp = 8;
constexpr int decode16_smem_bytes() {
  return 16 * kTile * kHeadDim * 2                    // 16 tile slots: K, then V
         + kGroupTiles * 16 * (kTile + 8) * 2         // P of every tile
         + kGroupTiles * (kDecodeWarps + 4) * 16 * 4  // per tile: maxima per S n-tile, max, m, scale, row sum of P
         + 2 * 16 * 4;                                // d and m of the group
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
  static_assert(kTileNS == 4 && kGroupTiles == 16 && kD16Warps == 16 && kNTilesO == 2 * kD16OWarps && kDecodeWarps == 4,
                "4 S n-tiles per tile, 4 tiles at once, two O n-tiles per warp, 4 quarters");
  constexpr int kPStride = kTile + 8, kPTile = 16 * kPStride, kTileElems = kTile * kHeadDim;
  extern __shared__ __align__(16) __nv_bfloat16 smem[];
  __nv_bfloat16* sKV = smem;                             // [16][kTile][128], swz per tile: K(lt), then V(lt)
  __nv_bfloat16* sP = sKV + 16 * kTileElems;             // [16 tiles][16 rows][kPStride]
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
  const bool ow = warp < kD16OWarps, dw = set == kD16DWarp / 4;
  const int ctid = tid & 127;                // issue_tile's thread index within the set
  const int group = num_qo_heads / num_kv_heads;
  const int t_begin = fg * kGroupTiles;
  const int nt = min(n_tiles - t_begin, kGroupTiles);  // tiles of this group

  // Loads, group 1: K of the warp set's 4 tiles.
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
    const int64_t q_stride = (int64_t)num_qo_heads * kHeadDim;
    load_q(qf, [&](int r) -> const __nv_bfloat16* {
             return r < group ? q + b * q_stride + (kvh * group + r) * kHeadDim : nullptr;
           }, lane);
  }

  // S of the warp's 4 tiles (tiles set + 4 i), their mma chains interleaved; the warp's maxima of its 8 keys.
  float s[4][4];
  cp_async_wait<0>();  // group 1
  __syncthreads();
  {
#pragma unroll
    for (int i = 0; i < 4; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < kKStepsQK; ++ks)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int lt = set + 4 * i;
#ifdef PI_PROBE_HALF_MMA
        if (lt < nt && i < 2) {
#else
        if (lt < nt) {
#endif
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
  if (warp == kD16DWarp && t4 == 0) {  // the running max and the scale of every tile, the group's m
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
      const __nv_bfloat16* vt = d16_v_slot(sKV, lt);
#pragma unroll
      for (int j = 0; j < kTileKS; ++j) {
        uint32_t pa[4], bv[4];
        ldmatrix_x4(pa, pt + j * 16);
        ldmatrix_x4_trans(bv, vt + swz(j * 16 + (mat & 1) * 8 + r8, warp * 2 + (mat >> 1)));
        mma_bf16(O0, pa, bv[0], bv[1]);
#ifndef PI_PROBE_HALF_MMA
        mma_bf16(O1, pa, bv[2], bv[3]);
#endif
      }
    }
  }
  if (warp == kD16DWarp && t4 == 0) {  // d = fma(d, scale, row sum), tiles in order
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
