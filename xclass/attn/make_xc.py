"""Derive the cross-class (XC) kernel sources from the repo's PI-Attn kernels by exact text patches.
-DXC=0: the stock program. -DXC=1: the exact-partial program; -DXC_QB=128|32|16: head dims per exact
QK block (128: one chain, no promotion; 32 / 16: fp32 sum of 4 / 8 exact block partials)."""
import sys
src, dst = sys.argv[1], sys.argv[2]

def patch(s, old, new, count=1):
    assert s.count(old) == count, (s.count(old), old[:80])
    return s.replace(old, new)

# ---------------------------------------------------------------- pi_attn.cuh
h = open(f"{src}/pi_attn.cuh").read()
h = patch(h, "namespace pi_attn {\n", """#ifndef XC
#define XC 0
#endif
#ifndef XC_QB
#define XC_QB 128
#endif

namespace pi_attn {
""")
h = patch(h, """__device__ __forceinline__ uint32_t smem_u32(const void* p) {""",
"""// 4-byte async copy (the value scales ride in the V tile's copy group); src_bytes == 0 zero-fills.
__device__ __forceinline__ void cp_async4(void* dst, const void* src, int src_bytes) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))), "l"(src),
               "r"(src_bytes));
}
// d = a b (accumulator input zero): starts an exact partial without zeroing registers.
__device__ __forceinline__ void mma_bf16_z(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  const float z = 0.f;
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(z));
}
constexpr int kQBlockSteps = XC_QB / 16;  // k-steps per exact QK block
// S of one n-tile. XC_QB < 128: block b = ks / kQBlockSteps accumulates exactly in e[b & 1] from
// zero, and s is the fp32 sum of the block partials in ascending order. The add of block b is
// issued one k-step later (qk_promote, after that step's MMAs) or at the end (qk_finish), so no
// add waits on the MMA it reads. XC_QB == 128 or the stock program: one chain in s.
__device__ __forceinline__ void qk_step(float (&s)[4], float (&e)[2][4], int ks, const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if XC && XC_QB < 128
  if (ks % kQBlockSteps == 0) mma_bf16_z(e[(ks / kQBlockSteps) & 1], a, b0, b1);
  else mma_bf16(e[(ks / kQBlockSteps) & 1], a, b0, b1);
#elif XC
  if (ks == 0) mma_bf16_z(s, a, b0, b1);
  else mma_bf16(s, a, b0, b1);
#else
  mma_bf16(s, a, b0, b1);  // stock: the caller zeroes s
#endif
}
__device__ __forceinline__ void qk_promote(float (&s)[4], float (&e)[2][4], int ks) {
#if XC && XC_QB < 128
  if (ks % kQBlockSteps == 0 && ks > 0) {
    const int prev = ks / kQBlockSteps - 1;
#pragma unroll
    for (int i = 0; i < 4; ++i) s[i] = prev == 0 ? e[0][i] : __fadd_rn(s[i], e[prev & 1][i]);
  }
#endif
}
__device__ __forceinline__ void qk_finish(float (&s)[4], float (&e)[2][4]) {
#if XC && XC_QB < 128
#pragma unroll
  for (int i = 0; i < 4; ++i) s[i] = __fadd_rn(s[i], e[(128 / XC_QB - 1) & 1][i]);
#endif
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {""")
h = patch(h, """__device__ __forceinline__ void online_step(OnlineState& st, float (&v)[kTileNS][4],
                                            uint32_t (&pa)[kTileKS][4]) {""",
"""#if XC
// XC weights of one 32-key tile, rows (g, g + 8) of this thread. In: w[n][i] = p = 2^(s - m) in fp32
// (i >> 1 = row, keys n * 8 + 2 t + (i & 1)); sv[n] = the two keys' value scales 2^ev (the cache
// holds V~ = V / 2^ev on a 2^-11 grid, |V~| <= 1). ls = the row's sum of p (what d accumulates), in
// the stock order. Out: w = p 2^ev rounded (nearest even) to a multiple of qw = 2^(E - 12), E =
// exponent of (1 + 2^-7) * (the row's tile sum of p 2^ev); the caller packs w to bf16 (nearest even,
// still a multiple of qw). Then sum_k W^_k < 2^13 qw, so every partial of sum_k W^_k V~_kj is an
// integer below 2^24 in units of qw 2^-11: the tile's tensor-core sum is exact. qw is at least 2^-63.
__device__ __forceinline__ void xc_weights(float (&w)[kTileNS][4], const float2 (&sv)[kTileNS], float (&ls)[2]) {
  float lw[2] = {0.f, 0.f};
  ls[0] = ls[1] = 0.f;
#pragma unroll
  for (int n = 0; n < kTileNS; ++n)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      ls[i >> 1] = add(ls[i >> 1], w[n][i]);
      w[n][i] = mul(w[n][i], (i & 1) ? sv[n].y : sv[n].x);
      lw[i >> 1] = add(lw[i >> 1], w[n][i]);
    }
  float C[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    ls[r] = add(ls[r], __shfl_xor_sync(0xffffffffu, ls[r], 1));
    ls[r] = add(ls[r], __shfl_xor_sync(0xffffffffu, ls[r], 2));
    lw[r] = add(lw[r], __shfl_xor_sync(0xffffffffu, lw[r], 1));
    lw[r] = add(lw[r], __shfl_xor_sync(0xffffffffu, lw[r], 2));
    const uint32_t eb = __float_as_uint(mul(lw[r], 1.0078125f)) & 0x7f800000u;
    C[r] = fmaxf(__uint_as_float(eb + (11u << 23)), 0x1p-40f);  // 2^23 qw
  }
#pragma unroll
  for (int n = 0; n < kTileNS; ++n)
#pragma unroll
    for (int i = 0; i < 4; ++i) w[n][i] = sub(add(w[n][i], C[i >> 1]), C[i >> 1]);
}
__device__ __forceinline__ void xc_pack(const float (&w)[kTileNS][4], uint32_t (&pa)[kTileKS][4]) {
#pragma unroll
  for (int n = 0; n < kTileNS; ++n) {
    pa[n >> 1][(n & 1) * 2 + 0] = pack_bf16(w[n][0], w[n][1]);
    pa[n >> 1][(n & 1) * 2 + 1] = pack_bf16(w[n][2], w[n][3]);
  }
}
// XC online step: as below, but the weights go through xc_weights and o is not rescaled here: the
// caller computes the tile's exact partial acc on the tensor cores from zero and folds
// o = fma(o, scale, acc).
__device__ __forceinline__ void online_step(OnlineState& st, float (&v)[kTileNS][4],
                                            uint32_t (&pa)[kTileKS][4], const float2 (&sv)[kTileNS],
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
    const float mn = fmaxf(st.m[r], mx[r]);
    mm[r] = mn == -INFINITY ? 0.f : mn;
    scale[r] = ex2(sub(st.m[r], mm[r]));
    st.m[r] = mn;
  }
#pragma unroll
  for (int n = 0; n < kTileNS; ++n)
#pragma unroll
    for (int i = 0; i < 4; ++i) v[n][i] = ex2(sub(v[n][i], mm[i >> 1]));
  float ls[2];
  xc_weights(v, sv, ls);
  xc_pack(v, pa);
#pragma unroll
  for (int r = 0; r < 2; ++r) st.d[r] = __fmaf_rn(st.d[r], scale[r], ls[r]);
}
#else
__device__ __forceinline__ void online_step(OnlineState& st, float (&v)[kTileNS][4],
                                            uint32_t (&pa)[kTileKS][4]) {""")
h = patch(h, """    for (int i = 0; i < 4; ++i) st.o[n][i] = __fmaf_rn(st.o[n][i], scale[i >> 1], 0.f);  // -0 -> +0
}
""", """    for (int i = 0; i < 4; ++i) st.o[n][i] = __fmaf_rn(st.o[n][i], scale[i >> 1], 0.f);  // -0 -> +0
}
#endif
""")
open(f"{dst}/pi_attn.cuh", "w").write(h)

# ------------------------------------------------------------ pi_attn_kernels.cu
k = open(f"{src}/pi_attn_kernels.cu").read()
# --- prefill: smem, signature
k = patch(k, "  return (WARPS * 16 + 4 * kTile) * kHeadDim * 2 + (QREG ? WARPS * 16 * kHeadDim * 4 : 0);",
             "  return (WARPS * 16 + 4 * kTile) * kHeadDim * 2 + 2 * kTile * 4 + (QREG ? WARPS * 16 * kHeadDim * 4 : 0);")
k = patch(k, """                    const __nv_bfloat16* __restrict__ k_cache, const __nv_bfloat16* __restrict__ v_cache,
                    const int32_t* __restrict__ qo_indptr, const int32_t* __restrict__ kv_page_indptr,""",
"""                    const __nv_bfloat16* __restrict__ k_cache, const __nv_bfloat16* __restrict__ v_cache,
                    const float* __restrict__ sv_cache,
                    const int32_t* __restrict__ qo_indptr, const int32_t* __restrict__ kv_page_indptr,""")
k = patch(k, "  float* sTot = reinterpret_cast<float*>(sV + 2 * kTile * kHeadDim);  // QREG: [warp][n][i][lane]",
"""  float* sSV = reinterpret_cast<float*>(sV + 2 * kTile * kHeadDim);  // [slot][key] value scales 2^ev
  float* sTot = sSV + 2 * kTile;  // QREG: [warp][n][i][lane]""")
# --- prefill: QK
k = patch(k, """#pragma unroll
    for (int n = 0; n < kTileNS; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < kKStepsQK; ++ks) {""", """    float e[kTileNS][2][4];
#if !XC
#pragma unroll
    for (int n = 0; n < kTileNS; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
#endif
#pragma unroll
    for (int ks = 0; ks < kKStepsQK; ++ks) {""")
k = patch(k, """        mma_bf16(s[n], a, bk[0], bk[1]);
        mma_bf16(s[n + 1], a, bk[2], bk[3]);
      }""", """        qk_step(s[n], e[n], ks, a, bk[0], bk[1]);
        qk_step(s[n + 1], e[n + 1], ks, a, bk[2], bk[3]);
      }
#pragma unroll
      for (int n = 0; n < kTileNS; ++n) qk_promote(s[n], e[n], ks);""")
k = patch(k, """        qk_step(s[n + 1], e[n + 1], ks, a, bk[2], bk[3]);
      }
#pragma unroll
      for (int n = 0; n < kTileNS; ++n) qk_promote(s[n], e[n], ks);
    }
""", """        qk_step(s[n + 1], e[n + 1], ks, a, bk[2], bk[3]);
      }
#pragma unroll
      for (int n = 0; n < kTileNS; ++n) qk_promote(s[n], e[n], ks);
    }
#pragma unroll
    for (int n = 0; n < kTileNS; ++n) qk_finish(s[n], e[n]);
""")
# --- prefill: value-scale loads alongside the V tiles
k = patch(k, """  int64_t row1 = row_of(t_begin);
  if (copier) issue_tile(sK, k_cache, row1, tid);
  cp_async_commit();  // Q + K(t_begin)
  if (copier) issue_tile(sV, v_cache, row1, tid);
""", """  int64_t row1 = row_of(t_begin);
  // Value scale of this thread's key of the tile whose V goes to `slot` (keys past kv_len: 0; their
  // weights are 0), copied in the V tile's async group.
  auto load_sv = [&](int slot, int64_t row) {
#if XC
    if (copier && (tid & 3) == 0)
      cp_async4(sSV + slot * kTile + (tid >> 2), sv_cache + (row >= 0 ? row / kHeadDim : 0), row >= 0 ? 4 : 0);
#endif
  };
  if (copier) issue_tile(sK, k_cache, row1, tid);
  cp_async_commit();  // Q + K(t_begin)
  if (copier) issue_tile(sV, v_cache, row1, tid);
  load_sv(0, row1);
""")
k = patch(k, """    if (copier && t + 1 <= t_last) issue_tile(sV + (slot ^ 1) * kTile * kHeadDim, v_cache, row1, tid);
    if (copier && t + 2 <= t_last) issue_tile(sK + slot * kTile * kHeadDim, k_cache, row2, tid);
    cp_async_commit();
    row1 = row2;""", """    float2 svk[kTileNS];
#if XC
#pragma unroll
    for (int n = 0; n < kTileNS; ++n) svk[n] = *reinterpret_cast<const float2*>(sSV + slot * kTile + n * 8 + t4 * 2);
    // sSV[slot ^ 1] is rewritten below, sSV[slot] only after the next barrier.
#endif
    if (copier && t + 1 <= t_last) {
      issue_tile(sV + (slot ^ 1) * kTile * kHeadDim, v_cache, row1, tid);
      load_sv(slot ^ 1, row1);
    }
    if (copier && t + 2 <= t_last) issue_tile(sK + slot * kTile * kHeadDim, k_cache, row2, tid);
    cp_async_commit();
    row1 = row2;""")
k = patch(k, """    uint32_t pa[kTileKS][4];
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
""", """    uint32_t pa[kTileKS][4];
#if XC
    static_assert(kTileKS == 2, "one exact partial per tile: two chained key k-steps");
    float sc[2];
    online_step(st, s, pa, svk, sc);
    static_assert(kNTilesO % 8 == 0, "two batches of 8 n-tiles");
#pragma unroll
    for (int n0 = 0; n0 < kNTilesO; n0 += 8) {
      // The tile's exact partials of 8 n-tiles: both key k-steps of all of them are issued before
      // the folds, so no instruction waits on the MMA just before it.
      float acc[8][4];
#pragma unroll
      for (int n = 0; n < 8; n += 2) {
        uint32_t bv[4];
        ldmatrix_x4_trans(bv, ((v_addr + slot * kSlotBytes) ^ ((n0 + n) << 4)));
        mma_bf16_z(acc[n], pa[0], bv[0], bv[1]);
        mma_bf16_z(acc[n + 1], pa[0], bv[2], bv[3]);
      }
#pragma unroll
      for (int n = 0; n < 8; n += 2) {
        uint32_t bv[4];
        ldmatrix_x4_trans(bv, ((v_addr + slot * kSlotBytes) ^ ((n0 + n) << 4)) + 16 * kHeadDim * 2);
        mma_bf16(acc[n], pa[1], bv[0], bv[1]);
        mma_bf16(acc[n + 1], pa[1], bv[2], bv[3]);
      }
#pragma unroll
      for (int n = 0; n < 8; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) st.o[n0 + n][i] = __fmaf_rn(st.o[n0 + n][i], sc[i >> 1], acc[n][i]);
    }
#else
    (void)svk;
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
#endif
""")
# --- decode
k = patch(k, "constexpr int decode_smem_bytes() { return (2 * kTile * kHeadDim + 16 * (kTile + 8)) * 2 + kDecodeWarps * 16 * 4; }",
             "constexpr int decode_smem_bytes() { return (2 * kTile * kHeadDim + 16 * (kTile + 8)) * 2 + kDecodeWarps * 16 * 4 + kTile * 4 + 16 * kTile * 4; }")
k = patch(k, """                   const __nv_bfloat16* __restrict__ v_cache, const int32_t* __restrict__ kv_page_indptr,
                   const int32_t* __restrict__ kv_page_indices, int64_t table_stride,
                   const int32_t* __restrict__ kv_lens, float* __restrict__ ws_o, float2* __restrict__ ws_ml,""",
"""                   const __nv_bfloat16* __restrict__ v_cache, const float* __restrict__ sv_cache,
                   const int32_t* __restrict__ kv_page_indptr,
                   const int32_t* __restrict__ kv_page_indices, int64_t table_stride,
                   const int32_t* __restrict__ kv_lens, float* __restrict__ ws_o, float2* __restrict__ ws_ml,""")
k = patch(k, "  float* sMax = reinterpret_cast<float*>(sP + 16 * kPStride);  // [warps][16]",
"""  float* sMax = reinterpret_cast<float*>(sP + 16 * kPStride);  // [warps][16]
  float* sSV = sMax + kDecodeWarps * 16;  // [key] value scales 2^ev of the V tile in sV
  float* sPf = sSV + kTile;  // XC: [16 rows][kTile] fp32 p
  auto load_sv = [&](int64_t r) {
#if XC
    if ((threadIdx.x & 3) == 0) cp_async4(sSV + (threadIdx.x >> 2), sv_cache + (r >= 0 ? r / kHeadDim : 0), r >= 0 ? 4 : 0);
#endif
  };""")
k = patch(k, """  issue_tile(sV, v_cache, row, tid);
  cp_async_commit();
  QFrag qf;""", """  issue_tile(sV, v_cache, row, tid);
  load_sv(row);
  cp_async_commit();
  QFrag qf;""")
k = patch(k, """    float s[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int ks = 0; ks < kKStepsQK; ++ks) {
      uint32_t bk[4];
      ldmatrix_x4(bk, sK + swz(warp * 8 + r8, ks * 2 + (mat & 1)));
      mma_bf16(s, qf.a[ks], bk[0], bk[1]);""", """    float s[4], e[2][4];
#if !XC
    s[0] = s[1] = s[2] = s[3] = 0.f;
#endif
#pragma unroll
    for (int ks = 0; ks < kKStepsQK; ++ks) {
      uint32_t bk[4];
      ldmatrix_x4(bk, sK + swz(warp * 8 + r8, ks * 2 + (mat & 1)));
      qk_step(s, e, ks, qf.a[ks], bk[0], bk[1]);
      qk_promote(s, e, ks);""")
k = patch(k, """      qk_promote(s, e, ks);
    }
""", """      qk_promote(s, e, ks);
    }
    qk_finish(s, e);
""")
k = patch(k, """    {
      float p[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) p[i] = round_bf16(ex2(sub(s[i], mm[i >> 1])));
      const int key = warp * 8 + t4 * 2;
      *reinterpret_cast<uint32_t*>(sP + g * kPStride + key) = pack_bf16(p[0], p[1]);
      *reinterpret_cast<uint32_t*>(sP + (g + 8) * kPStride + key) = pack_bf16(p[2], p[3]);
    }""", """#if XC
    {
      const int key = warp * 8 + t4 * 2;
      *reinterpret_cast<float2*>(sPf + g * kTile + key) = make_float2(ex2(sub(s[0], mm[0])), ex2(sub(s[1], mm[0])));
      *reinterpret_cast<float2*>(sPf + (g + 8) * kTile + key) = make_float2(ex2(sub(s[2], mm[1])), ex2(sub(s[3], mm[1])));
    }
#else
    {
      float p[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) p[i] = round_bf16(ex2(sub(s[i], mm[i >> 1])));
      const int key = warp * 8 + t4 * 2;
      *reinterpret_cast<uint32_t*>(sP + g * kPStride + key) = pack_bf16(p[0], p[1]);
      *reinterpret_cast<uint32_t*>(sP + (g + 8) * kPStride + key) = pack_bf16(p[2], p[3]);
    }
#endif""")
k = patch(k, """    if (warp == 0) {  // d in exactly online_step's order
      float ls[2] = {0.f, 0.f};""", """#if XC
    uint32_t pa[kTileKS][4];
    {  // every warp runs xc_weights on the row's 32 weights exactly as online_step does
      float w[kTileNS][4], ls[2];
      float2 svk[kTileNS];
#pragma unroll
      for (int n = 0; n < kTileNS; ++n) {
        const float2 p0 = *reinterpret_cast<const float2*>(sPf + g * kTile + n * 8 + t4 * 2);
        const float2 p1 = *reinterpret_cast<const float2*>(sPf + (g + 8) * kTile + n * 8 + t4 * 2);
        w[n][0] = p0.x;
        w[n][1] = p0.y;
        w[n][2] = p1.x;
        w[n][3] = p1.y;
        svk[n] = *reinterpret_cast<const float2*>(sSV + n * 8 + t4 * 2);
      }
      xc_weights(w, svk, ls);
      xc_pack(w, pa);
      if (warp == 0) {
        D = __fmaf_rn(D, scale[0], ls[0]);
        Dh = __fmaf_rn(Dh, scale[1], ls[1]);
      }
    }
    {
      float acc[kON][4];
#pragma unroll
      for (int n = 0; n < kON; n += 2) {
        uint32_t bv[4];
        ldmatrix_x4_trans(bv, sV + swz((mat & 1) * 8 + r8, warp * kON + n + (mat >> 1)));
        mma_bf16_z(acc[n], pa[0], bv[0], bv[1]);
        mma_bf16_z(acc[n + 1], pa[0], bv[2], bv[3]);
      }
#pragma unroll
      for (int n = 0; n < kON; n += 2) {
        uint32_t bv[4];
        ldmatrix_x4_trans(bv, sV + swz(16 + (mat & 1) * 8 + r8, warp * kON + n + (mat >> 1)));
        mma_bf16(acc[n], pa[1], bv[0], bv[1]);
        mma_bf16(acc[n + 1], pa[1], bv[2], bv[3]);
      }
#pragma unroll
      for (int n = 0; n < kON; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) O[n][i] = __fmaf_rn(O[n][i], scale[i >> 1], acc[n][i]);
    }
#else
    if (warp == 0) {  // d in exactly online_step's order
      float ls[2] = {0.f, 0.f};""")
k = patch(k, """        mma_bf16(O[n], pa[j], bv[0], bv[1]);
        mma_bf16(O[n + 1], pa[j], bv[2], bv[3]);
      }
    __syncthreads();  // all warps done with V(t) and P
    if (t + 1 < t_end) issue_tile(sV, v_cache, row, tid);""", """        mma_bf16(O[n], pa[j], bv[0], bv[1]);
        mma_bf16(O[n + 1], pa[j], bv[2], bv[3]);
      }
#endif
    __syncthreads();  // all warps done with V(t), its scales and P
    if (t + 1 < t_end) {
      issue_tile(sV, v_cache, row, tid);
      load_sv(row);
    }""")
# --- decode, one group: the in-place finish must give the bits of the first fold into the identity,
# which turns a -0 partial (a rescale that underflowed to -0 plus an all-zero tile partial) into +0.
k = patch(k, """      if (g < group) *reinterpret_cast<__nv_bfloat162*>(o0 + col) = __halves2bfloat162(finish(O[n][0], l0), finish(O[n][1], l0));
      if (g + 8 < group)
        *reinterpret_cast<__nv_bfloat162*>(o1 + col) = __halves2bfloat162(finish(O[n][2], l1), finish(O[n][3], l1));""",
"""#if XC
#pragma unroll
      for (int i = 0; i < 4; ++i) O[n][i] = add(O[n][i], 0.f);  // -0 -> +0, as fold_elem(0, O, first fold)
#endif
      if (g < group) *reinterpret_cast<__nv_bfloat162*>(o0 + col) = __halves2bfloat162(finish(O[n][0], l0), finish(O[n][1], l0));
      if (g + 8 < group)
        *reinterpret_cast<__nv_bfloat162*>(o1 + col) = __halves2bfloat162(finish(O[n][2], l1), finish(O[n][3], l1));""")
open(f"{dst}/pi_attn_kernels.cu", "w").write(k)

# ------------------------------------------------------------ pi_attn_torch.cu
t = open(f"{src}/pi_attn_torch.cu").read()
t = patch(t, """at::Tensor prefill(at::Tensor q, at::Tensor k_cache, at::Tensor v_cache, at::Tensor qo_indptr,""",
             """at::Tensor prefill(at::Tensor q, at::Tensor k_cache, at::Tensor v_cache, at::Tensor sv_cache, at::Tensor qo_indptr,""")
t = patch(t, """  check_inputs(q, k_cache, v_cache, {qo_indptr, kv_page_indptr, kv_page_indices, kv_lens});
  const c10::cuda::CUDAGuard guard(q.device());
  auto out = at::empty_like(q);
  const int batch = kv_lens.size(0), hq = q.size(1), hkv = k_cache.size(2);
  TORCH_CHECK(qo_indptr.size(0) == batch + 1 && kv_page_indptr.size(0) == batch + 1);""",
"""  check_inputs(q, k_cache, v_cache, {qo_indptr, kv_page_indptr, kv_page_indices, kv_lens});
  check_sv(sv_cache, k_cache);
  const c10::cuda::CUDAGuard guard(q.device());
  auto out = at::empty_like(q);
  const int batch = kv_lens.size(0), hq = q.size(1), hkv = k_cache.size(2);
  TORCH_CHECK(qo_indptr.size(0) == batch + 1 && kv_page_indptr.size(0) == batch + 1);""")
t = patch(t, """static int page_shift_of(int page_size) {""", """// sv_cache: [pages, page_size, Hkv] f32 value scales 2^ev (XC builds; ignored by XC=0).
static void check_sv(const at::Tensor& sv, const at::Tensor& k_cache) {
  TORCH_CHECK(sv.is_cuda() && sv.scalar_type() == at::kFloat && sv.is_contiguous() && sv.dim() == 3 &&
              sv.size(0) == k_cache.size(0) && sv.size(1) == k_cache.size(1) && sv.size(2) == k_cache.size(2),
              "sv_cache: contiguous f32 CUDA [pages, page_size, Hkv]");
}

static int page_shift_of(int page_size) {""")
t = patch(t, """        (const __nv_bfloat16*)k_cache.data_ptr(), (const __nv_bfloat16*)v_cache.data_ptr(),
        qo_indptr.data_ptr<int32_t>(), kv_page_indptr.data_ptr<int32_t>(), kv_page_indices.data_ptr<int32_t>(),""",
"""        (const __nv_bfloat16*)k_cache.data_ptr(), (const __nv_bfloat16*)v_cache.data_ptr(), sv_cache.data_ptr<float>(),
        qo_indptr.data_ptr<int32_t>(), kv_page_indptr.data_ptr<int32_t>(), kv_page_indices.data_ptr<int32_t>(),""")
t = patch(t, """at::Tensor decode(at::Tensor q, at::Tensor k_cache, at::Tensor v_cache, at::Tensor kv_page_indptr,""",
             """at::Tensor decode(at::Tensor q, at::Tensor k_cache, at::Tensor v_cache, at::Tensor sv_cache, at::Tensor kv_page_indptr,""")
t = patch(t, """  check_inputs(q, k_cache, v_cache, {kv_page_indptr, kv_page_indices, kv_lens});
  const c10::cuda::CUDAGuard guard(q.device());""", """  check_inputs(q, k_cache, v_cache, {kv_page_indptr, kv_page_indices, kv_lens});
  check_sv(sv_cache, k_cache);
  const c10::cuda::CUDAGuard guard(q.device());""")
t = patch(t, """      (const __nv_bfloat16*)v_cache.data_ptr(), kv_page_indptr.data_ptr<int32_t>(),""",
             """      (const __nv_bfloat16*)v_cache.data_ptr(), sv_cache.data_ptr<float>(), kv_page_indptr.data_ptr<int32_t>(),""")
t = patch(t, """  m.def("prefill", &prefill, py::arg("q"), py::arg("k_cache"), py::arg("v_cache"), py::arg("qo_indptr"),""",
             """  m.def("prefill", &prefill, py::arg("q"), py::arg("k_cache"), py::arg("v_cache"), py::arg("sv_cache"), py::arg("qo_indptr"),""")
t = patch(t, """  m.attr("GROUP_KEYS") = kGroupKeys;""", """  m.attr("GROUP_KEYS") = kGroupKeys;
  m.attr("XC") = XC;
  m.attr("XC_QB") = XC_QB;""")
open(f"{dst}/pi_attn_torch.cu", "w").write(t)
print("ok")
