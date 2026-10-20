// PI-Attn kernels: paged prefill (register fold), decode partials, merge.
// KV cache layout per layer: [num_pages, page_size, num_kv_heads, 128] bf16.
#include "pi_attn.cuh"

namespace pi_attn {

// Async-load `rows` rows of a K/V chunk tile (or a Q tile). Rows at positions
// >= kv_len are zero-filled without a global read, so stale cache contents can
// never reach the arithmetic.
__device__ __forceinline__ void issue_tile(__nv_bfloat16* tile, const __nv_bfloat16* cache,
                                           const int32_t* pages, int page_size, int num_kv_heads,
                                           int kvh, int c, int kv_len, int tid, int nthreads) {
  constexpr int kVecPerRow = kHeadDim / 8;
  for (int i = tid; i < kChunk * kVecPerRow; i += nthreads) {
    const int key = i / kVecPerRow, vec = i % kVecPerRow;
    const int pos = c * kChunk + key;
    const __nv_bfloat16* src = cache;
    int bytes = 0;
    if (pos < kv_len) {
      const int64_t page = pages[pos / page_size];
      src = cache + ((page * page_size + pos % page_size) * num_kv_heads + kvh) * kHeadDim + vec * 8;
      bytes = 16;
    }
    cp_async16(tile + key * kRowStride + vec * 8, src, bytes);
  }
}

// Prefill smem: [Q tile if G == 2] + K double buffer + VB V buffers.
template <int NWARPS, int G, int VB>
constexpr int prefill_smem_elems() {
  return (G == 2 ? NWARPS * 16 * G * kRowStride : 0) + (2 + VB) * kTileElems;
}

// Prefill / extend. grid = (q tiles of 16*G*NWARPS rows, num_qo_heads, batch).
// Query row i of request b sits at absolute position kv_len[b] - q_len + i.
// NWARPS and G (16-row groups per warp sharing K/V fragments) only change how
// work is distributed, never a row's operation sequence.
template <int NWARPS, int G, int VB>
__global__ void __launch_bounds__(NWARPS * 32)
    prefill_kernel(const __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ out,
                   const __nv_bfloat16* __restrict__ k_cache,
                   const __nv_bfloat16* __restrict__ v_cache, const int32_t* __restrict__ qo_indptr,
                   const int32_t* __restrict__ kv_page_indptr,
                   const int32_t* __restrict__ kv_page_indices, const int32_t* __restrict__ kv_lens,
                   int num_qo_heads, int num_kv_heads, int page_size, float scale_log2) {
  constexpr int kWarpRows = 16 * G;
  constexpr int kCtaRows = NWARPS * kWarpRows;
  constexpr int kQElems = G == 2 ? kCtaRows * kRowStride : 0;
  extern __shared__ __align__(16) __nv_bfloat16 smem[];
  __nv_bfloat16* sQ = smem;
  __nv_bfloat16* sKbuf = smem + kQElems;
  __nv_bfloat16* sVbuf = smem + kQElems + 2 * kTileElems;

  const int b = blockIdx.z, h = blockIdx.y, tile = blockIdx.x;
  const int q0 = qo_indptr[b], q_len = qo_indptr[b + 1] - q0;
  if (tile * kCtaRows >= q_len) return;
  const int kv_len = kv_lens[b], prefix = kv_len - q_len;
  const int kvh = h / (num_qo_heads / num_kv_heads);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row_base = tile * kCtaRows + warp * kWarpRows;
  const int64_t q_stride = (int64_t)num_qo_heads * kHeadDim;
  const int32_t* pages = kv_page_indices + kv_page_indptr[b];
  const int last_row = min(tile * kCtaRows + kCtaRows, q_len) - 1;
  const int last_chunk = (prefix + last_row) / kChunk;
  const int warp_last = min(row_base + kWarpRows, q_len) - 1;  // last valid row of this warp

  auto sK = [&](int c) { return sKbuf + (c & 1) * kTileElems; };
  auto sV = [&](int c) { return sVbuf + (VB == 2 ? (c & 1) * kTileElems : 0); };
  auto issue_k = [&](int c) {
    issue_tile(sK(c), k_cache, pages, page_size, num_kv_heads, kvh, c, kv_len, threadIdx.x, blockDim.x);
    cp_async_commit();
  };
  auto issue_v = [&](int c) {
    issue_tile(sV(c), v_cache, pages, page_size, num_kv_heads, kvh, c, kv_len, threadIdx.x, blockDim.x);
    cp_async_commit();
  };

  QFrag qf;  // used when G == 1
  if constexpr (G == 2) {
    constexpr int kVecPerRow = kHeadDim / 8;
    for (int i = threadIdx.x; i < kCtaRows * kVecPerRow; i += blockDim.x) {
      const int r = i / kVecPerRow, vec = i % kVecPerRow, qi = tile * kCtaRows + r;
      const __nv_bfloat16* src = q;
      int bytes = 0;
      if (qi < q_len) {
        src = q + (q0 + qi) * q_stride + h * kHeadDim + vec * 8;
        bytes = 16;
      }
      cp_async16(sQ + r * kRowStride + vec * 8, src, bytes);
    }
    // Q joins the first K group.
  } else {
    load_q(qf,
           [&](int r) -> const __nv_bfloat16* {
             const int i = row_base + r;
             return i < q_len ? q + (q0 + i) * q_stride + h * kHeadDim : nullptr;
           },
           lane);
  }
  issue_k(0);
  issue_v(0);

  float M[G][2], L[G][2], O[G][kNTilesO][4];
#pragma unroll
  for (int grp = 0; grp < G; ++grp) {
    M[grp][0] = M[grp][1] = -INFINITY;
    L[grp][0] = L[grp][1] = 0.f;
#pragma unroll
    for (int n = 0; n < kNTilesO; ++n) O[grp][n][0] = O[grp][n][1] = O[grp][n][2] = O[grp][n][3] = 0.f;
  }
  const int mat = lane >> 3, r8 = lane & 7;

  for (int c = 0; c <= last_chunk; ++c) {
    if constexpr (VB == 2) {
      if (c + 1 <= last_chunk) { issue_k(c + 1); issue_v(c + 1); }
      else { cp_async_commit(); cp_async_commit(); }
      cp_async_wait<2>();  // K(c) and V(c) (and Q) landed
    } else {
      if (c + 1 <= last_chunk) issue_k(c + 1); else cp_async_commit();
      cp_async_wait<2>();  // K(c) (and Q) landed
    }
    __syncthreads();
    // Warps whose rows all precede this chunk would only fold identities.
    const bool active = c * kChunk <= prefix + warp_last;
    PFrag pf[G];
    FoldScalars fs[G][2];
    if (active) {
      float s[G][kNTilesS][4];
      chunk_qk<G>(s,
                  [&](int grp, int ks, uint32_t(&a)[4]) {
                    if constexpr (G == 2) {
                      ldmatrix_x4(a, sQ + (warp * kWarpRows + grp * 16 + (mat & 1) * 8 + r8) * kRowStride +
                                         ks * 16 + (mat >> 1) * 8);
                    } else {
                      a[0] = qf.a[ks][0]; a[1] = qf.a[ks][1]; a[2] = qf.a[ks][2]; a[3] = qf.a[ks][3];
                    }
                  },
                  sK(c), lane);
#pragma unroll
      for (int grp = 0; grp < G; ++grp) {
        const int gbase = row_base + grp * 16;
        float m[2], l[2];
        scores_to_p(s[grp], pf[grp], m, l, scale_log2,
                    [&](int r, int key) {
                      const int i = gbase + r;
                      return i < q_len && c * kChunk + key <= prefix + i;
                    },
                    gbase + 15 < q_len && (c + 1) * kChunk - 1 <= prefix + gbase, lane);
#pragma unroll
        for (int r = 0; r < 2; ++r) fs[grp][r] = fold_scalars(M[grp][r], L[grp][r], m[r], l[r]);
      }
    }
    if constexpr (VB == 1) {
      cp_async_wait<1>();  // V(c) landed
      __syncthreads();
    }
    if (active) {
      chunk_pv<G>(pf, sV(c), lane, [&](int grp, int n, const float(&o)[4]) {
#pragma unroll
        for (int i = 0; i < 4; ++i) O[grp][n][i] = fold_elem(O[grp][n][i], o[i], fs[grp][i >> 1]);
      });
    }
    __syncthreads();
    if constexpr (VB == 1) {
      if (c + 1 <= last_chunk) issue_v(c + 1); else cp_async_commit();
    }
  }
  cp_async_wait<0>();

  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int grp = 0; grp < G; ++grp) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int i = row_base + grp * 16 + g + r * 8;
      if (i >= q_len) continue;
      __nv_bfloat16* o = out + (q0 + i) * q_stride + h * kHeadDim;
#pragma unroll
      for (int n = 0; n < kNTilesO; ++n) {
        __nv_bfloat162 v;
        v.x = finish(O[grp][n][2 * r + 0], L[grp][r]);
        v.y = finish(O[grp][n][2 * r + 1], L[grp][r]);
        *reinterpret_cast<__nv_bfloat162*>(o + n * 8 + t * 2) = v;
      }
    }
  }
}

template <int DW, int KVB>
constexpr int decode_smem_elems() {
  return DW * KVB * kTileElems;
}

// Decode partials. grid = (num_kv_heads, batch, split CTAs). Each warp owns one
// chunk at a time (chunks strided over all warps of the request) and writes the
// partial of the GQA group's query rows. The split count only changes who
// computes a partial, never the partial. With KVB == 2, K of the next chunk
// streams in during this chunk's P V and V of the next chunk while the partial
// is written; with KVB == 1 one tile is reused for K then V (less smem, more
// resident warps).
template <int DW, int KVB>
__global__ void __launch_bounds__(DW * 32)
    decode_partial_kernel(const __nv_bfloat16* __restrict__ q,
                          const __nv_bfloat16* __restrict__ k_cache,
                          const __nv_bfloat16* __restrict__ v_cache,
                          const int32_t* __restrict__ kv_page_indptr,
                          const int32_t* __restrict__ kv_page_indices,
                          const int32_t* __restrict__ kv_lens, float* __restrict__ ws_o,
                          float2* __restrict__ ws_ml, int num_qo_heads, int num_kv_heads,
                          int page_size, int max_chunks, float scale_log2) {
  extern __shared__ __align__(16) __nv_bfloat16 smem[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  __nv_bfloat16* sK = smem + warp * KVB * kTileElems;
  __nv_bfloat16* sV = sK + (KVB - 1) * kTileElems;
  const int kvh = blockIdx.x, b = blockIdx.y;
  const int group = num_qo_heads / num_kv_heads;
  const int kv_len = kv_lens[b];
  const int n_chunks = (kv_len + kChunk - 1) / kChunk;
  const int pos = kv_len - 1;  // the decoded token; its KV is already in the cache
  const int32_t* pages = kv_page_indices + kv_page_indptr[b];
  const int64_t q_stride = (int64_t)num_qo_heads * kHeadDim;
  const int stride = gridDim.z * DW;
  int c = blockIdx.z * DW + warp;
  if (c >= n_chunks) return;
  auto load = [&](__nv_bfloat16* tile, const __nv_bfloat16* cache, int chunk) {
    issue_tile(tile, cache, pages, page_size, num_kv_heads, kvh, chunk, kv_len, lane, 32);
    cp_async_commit();
  };

  load(sK, k_cache, c);
  if constexpr (KVB == 2) load(sV, v_cache, c);

  QFrag qf;
  load_q(qf,
         [&](int r) -> const __nv_bfloat16* {
           return r < group ? q + b * q_stride + (kvh * group + r) * kHeadDim : nullptr;
         },
         lane);
  const int g = lane >> 2, t = lane & 3;
  for (; c < n_chunks; c += stride) {
    const int next = c + stride;
    if constexpr (KVB == 2) cp_async_wait<1>(); else cp_async_wait<0>();  // K(c) landed
    __syncwarp();
    float s[1][kNTilesS][4];
    chunk_qk<1>(s,
                [&](int, int ks, uint32_t(&a)[4]) {
                  a[0] = qf.a[ks][0]; a[1] = qf.a[ks][1]; a[2] = qf.a[ks][2]; a[3] = qf.a[ks][3];
                },
                sK, lane);
    PFrag pf[1];
    float m[2], l[2];
    scores_to_p(s[0], pf[0], m, l, scale_log2,
                [&](int r, int key) { return r < group && c * kChunk + key <= pos; },
                false, lane);
    __syncwarp();
    if constexpr (KVB == 2) {
      if (next < n_chunks) load(sK, k_cache, next); else cp_async_commit();
      cp_async_wait<1>();  // V(c) landed; K(next) may be in flight
    } else {
      load(sV, v_cache, c);
      cp_async_wait<0>();
    }
    __syncwarp();
    const int64_t slot0 = ((int64_t)b * num_qo_heads + kvh * group + g) * max_chunks + c;
    const int64_t slot1 = slot0 + 8 * (int64_t)max_chunks;
    chunk_pv<1>(pf, sV, lane, [&](int, int n, const float(&o)[4]) {
      if (g < group)
        *reinterpret_cast<float2*>(ws_o + slot0 * kHeadDim + n * 8 + t * 2) = make_float2(o[0], o[1]);
      if (g + 8 < group)
        *reinterpret_cast<float2*>(ws_o + slot1 * kHeadDim + n * 8 + t * 2) = make_float2(o[2], o[3]);
    });
    if (t == 0) {
      if (g < group) ws_ml[slot0] = make_float2(m[0], l[0]);
      if (g + 8 < group) ws_ml[slot1] = make_float2(m[1], l[1]);
    }
    __syncwarp();
    if constexpr (KVB == 2) {
      if (next < n_chunks) load(sV, v_cache, next); else cp_async_commit();
    } else {
      if (next < n_chunks) load(sK, k_cache, next);
    }
  }
  cp_async_wait<0>();
}

// Merge. grid = (num_qo_heads, batch), 128 threads (one per head dim). Partials
// are prefetched in batches; the fold itself is the canonical sequential one.
__global__ void __launch_bounds__(kHeadDim)
    decode_merge_kernel(const float* __restrict__ ws_o, const float2* __restrict__ ws_ml,
                        const int32_t* __restrict__ kv_lens, __nv_bfloat16* __restrict__ out,
                        int num_qo_heads, int max_chunks) {
  constexpr int kBatch = 8;
  const int h = blockIdx.x, b = blockIdx.y, d = threadIdx.x;
  const int n_chunks = (kv_lens[b] + kChunk - 1) / kChunk;
  const int64_t base = ((int64_t)b * num_qo_heads + h) * max_chunks;
  float M = -INFINITY, L = 0.f, O = 0.f;
  for (int c0 = 0; c0 < n_chunks; c0 += kBatch) {
    float2 ml[kBatch];
    float ov[kBatch];
#pragma unroll
    for (int i = 0; i < kBatch; ++i) {
      if (c0 + i < n_chunks) {
        ml[i] = ws_ml[base + c0 + i];
        ov[i] = ws_o[(base + c0 + i) * kHeadDim + d];
      }
    }
#pragma unroll
    for (int i = 0; i < kBatch; ++i) {
      if (c0 + i < n_chunks) {
        const FoldScalars f = fold_scalars(M, L, ml[i].x, ml[i].y);
        O = fold_elem(O, ov[i], f);
      }
    }
  }
  out[((int64_t)b * num_qo_heads + h) * kHeadDim + d] = finish(O, L);
}

}  // namespace pi_attn
