// PI-Attn kernels: paged prefill (register fold), decode partials, merge.
// KV cache layout per layer: [num_pages, page_size, num_kv_heads, 128] bf16.
#include "pi_attn.cuh"

namespace pi_attn {

// Async-load one K or V chunk tile. Keys at positions >= kv_len are zero-filled
// without a global read, so stale cache contents can never reach the
// arithmetic. nthreads is a multiple of 16, so each thread copies a fixed
// 16-byte column of every (nthreads / 16)-th key; page_shift = log2(page_size)
// (or -1 when page_size is not a power of two) avoids integer division.
// Pages of request b: CSR (kv_page_indptr) or, when kv_page_indptr is null, row b
// of a dense page table with row stride table_stride.
__device__ __forceinline__ const int32_t* request_pages(const int32_t* kv_page_indices,
                                                       const int32_t* kv_page_indptr,
                                                       int64_t table_stride, int b) {
  return kv_page_indices + (kv_page_indptr ? (int64_t)kv_page_indptr[b] : (int64_t)b * table_stride);
}

__device__ __forceinline__ void issue_tile(__nv_bfloat16* tile, const __nv_bfloat16* cache,
                                           const int32_t* pages, int page_size, int page_shift,
                                           int num_kv_heads, int kvh, int c, int kv_len, int tid,
                                           int nthreads) {
  constexpr int kVecPerRow = kHeadDim / 8;
  const int vec = tid % kVecPerRow, key_step = nthreads / kVecPerRow;
  const int64_t row_stride = (int64_t)num_kv_heads * kHeadDim;
  const __nv_bfloat16* head_base = cache + (int64_t)kvh * kHeadDim + vec * 8;
  for (int key = tid / kVecPerRow; key < kChunk; key += key_step) {
    const int pos = c * kChunk + key;
    const __nv_bfloat16* src = cache;
    int bytes = 0;
    if (pos < kv_len) {
      int64_t slot;
      if (page_shift >= 0) slot = ((int64_t)pages[pos >> page_shift] << page_shift) + (pos & (page_size - 1));
      else slot = (int64_t)pages[pos / page_size] * page_size + pos % page_size;
      src = head_base + slot * row_stride;
      bytes = 16;
    }
    cp_async16(tile + key * kRowStride + vec * 8, src, bytes);
  }
}

// Prefill smem: [Q tile if QS] + K double buffer + VB V buffers. QS (Q in smem,
// fetched with ldmatrix) frees the 32 registers of register-resident Q.
template <int NWARPS, int G, int VB, bool QS = (G == 2)>
constexpr int prefill_smem_elems() {
  return (QS ? NWARPS * 16 * G * kRowStride : 0) + (2 + VB) * kTileElems;
}

// Prefill / extend. grid = (row tiles of 16*G*NWARPS rows, num_kv_heads, batch * kv_splits).
// The rows of a CTA are the (token, query head) pairs of one kv head's GQA
// group, token-major: row r is token r / group, head kv_head * group + r % group,
// so each K/V tile is loaded once per kv head. Token i of request b sits at
// absolute position kv_len[b] - q_len + i. NWARPS, G (16-row groups per warp
// sharing K/V fragments) and the packing only change how work is distributed,
// never a row's operation sequence. With kv_splits == 1
// the CTA runs the whole two-level fold and writes the output; otherwise CTA
// split s covers a contiguous range of fold groups and writes, per row and
// group, the group partial (ws_o / ws_ml indexed [row][head][group]); the
// prefill merge kernel then folds the groups.
template <int NWARPS, int G, int VB, bool QS = (G == 2)>
__global__ void __launch_bounds__(NWARPS * 32)
    prefill_kernel(const __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ out,
                   const __nv_bfloat16* __restrict__ k_cache,
                   const __nv_bfloat16* __restrict__ v_cache, const int32_t* __restrict__ qo_indptr,
                   const int32_t* __restrict__ kv_page_indptr,
                   const int32_t* __restrict__ kv_page_indices, int64_t table_stride,
                   const int32_t* __restrict__ kv_lens, const int32_t* __restrict__ prefix_lens,
                   int num_qo_heads, int num_kv_heads, int page_size, int page_shift,
                   float scale_log2, int kv_splits, float* __restrict__ ws_o,
                   float2* __restrict__ ws_ml, int max_groups) {
  constexpr int kWarpRows = 16 * G;
  constexpr int kCtaRows = NWARPS * kWarpRows;
  static_assert(G == 1 || QS, "two row groups per warp need Q in smem");
  constexpr int kQElems = QS ? kCtaRows * kRowStride : 0;
  extern __shared__ __align__(16) __nv_bfloat16 smem[];
  __nv_bfloat16* sQ = smem;
  __nv_bfloat16* sKbuf = smem + kQElems;
  __nv_bfloat16* sVbuf = smem + kQElems + 2 * kTileElems;

  const int b = blockIdx.z / kv_splits, split = blockIdx.z % kv_splits;
  const int kvh = blockIdx.y, tile = blockIdx.x;
  const int group = num_qo_heads / num_kv_heads;
  const int q0 = qo_indptr[b], q_len = qo_indptr[b + 1] - q0;
  const int rows_total = q_len * group;
  if (tile * kCtaRows >= rows_total) return;
  // kv_lens, or prefix_lens + q_len when kv_lens is null.
  const int kv_len = kv_lens ? kv_lens[b] : prefix_lens[b] + q_len, prefix = kv_len - q_len;
  auto tok = [&](int r) { return r / group; };
  auto row_q = [&](int r) { return q + (int64_t)(q0 + r / group) * num_qo_heads * kHeadDim + (kvh * group + r % group) * kHeadDim; };
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row_base = tile * kCtaRows + warp * kWarpRows;
  const int32_t* pages = request_pages(kv_page_indices, kv_page_indptr, table_stride, b);
  const int last_tok = tok(min(tile * kCtaRows + kCtaRows, rows_total) - 1);
  const int warp_last_tok = tok(min(row_base + kWarpRows, rows_total) - 1);
  // Fold groups this CTA covers: [g_begin, g_end) of the tile's visible groups.
  const int tile_groups = (prefix + last_tok) / (kChunk * kGroupChunks) + 1;
  const int per_split = (tile_groups + kv_splits - 1) / kv_splits;
  const int g_begin = split * per_split, g_end = min(tile_groups, g_begin + per_split);
  if (g_begin >= g_end) return;
  const int c_begin = g_begin * kGroupChunks;
  const int c_last = min(g_end * kGroupChunks, (prefix + last_tok) / kChunk + 1) - 1;

  auto sK = [&](int c) { return sKbuf + (c & 1) * kTileElems; };
  auto sV = [&](int c) { return sVbuf + (VB == 2 ? (c & 1) * kTileElems : 0); };
  auto issue_k = [&](int c) {
    issue_tile(sK(c), k_cache, pages, page_size, page_shift, num_kv_heads, kvh, c, kv_len, threadIdx.x, blockDim.x);
    cp_async_commit();
  };
  auto issue_v = [&](int c) {
    issue_tile(sV(c), v_cache, pages, page_size, page_shift, num_kv_heads, kvh, c, kv_len, threadIdx.x, blockDim.x);
    cp_async_commit();
  };

  QFrag qf;  // used when !QS
  if constexpr (QS) {
    constexpr int kVecPerRow = kHeadDim / 8;
    for (int i = threadIdx.x; i < kCtaRows * kVecPerRow; i += blockDim.x) {
      const int r = i / kVecPerRow, vec = i % kVecPerRow, qi = tile * kCtaRows + r;
      const __nv_bfloat16* src = q;
      int bytes = 0;
      if (qi < rows_total) {
        src = row_q(qi) + vec * 8;
        bytes = 16;
      }
      cp_async16(sQ + r * kRowStride + vec * 8, src, bytes);
    }
    // Q joins the first K group.
  } else {
    load_q(qf,
           [&](int r) -> const __nv_bfloat16* {
             const int i = row_base + r;
             return i < rows_total ? row_q(i) : nullptr;
           },
           lane);
  }
  issue_k(c_begin);
  issue_v(c_begin);

  RowFold tot[G], grp[G];
#pragma unroll
  for (int gi = 0; gi < G; ++gi) {
    tot[gi].reset();
    grp[gi].reset();
  }
  const int mat = lane >> 3, r8 = lane & 7;
  const int g = lane >> 2, t = lane & 3;

  // Group partial of rows that have any visible chunk in group `fg` goes to the
  // workspace (split mode) or is absorbed into the running total.
  auto finish_group = [&](int fg) {
    if (kv_splits == 1) {
#pragma unroll
      for (int gi = 0; gi < G; ++gi) tot[gi].absorb(grp[gi]);
    } else {
#pragma unroll
      for (int gi = 0; gi < G; ++gi) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const int i = row_base + gi * 16 + g + r * 8;
          if (i >= rows_total || fg * kGroupChunks * kChunk > prefix + tok(i)) continue;
          const int64_t slot = ((int64_t)(q0 + tok(i)) * num_qo_heads + kvh * group + i % group) * max_groups + fg;
          if (t == 0) ws_ml[slot] = make_float2(grp[gi].M[r], grp[gi].L[r]);
          float* dst = ws_o + slot * kHeadDim;
#pragma unroll
          for (int n = 0; n < kNTilesO; ++n)
            *reinterpret_cast<float2*>(dst + n * 8 + t * 2) =
                make_float2(grp[gi].O[n][2 * r], grp[gi].O[n][2 * r + 1]);
        }
      }
    }
#pragma unroll
    for (int gi = 0; gi < G; ++gi) grp[gi].reset();
  };

  for (int c = c_begin; c <= c_last; ++c) {
    if constexpr (VB == 2) {
      if (c + 1 <= c_last) { issue_k(c + 1); issue_v(c + 1); }
      else { cp_async_commit(); cp_async_commit(); }
      cp_async_wait<2>();  // K(c) and V(c) (and Q) landed
    } else {
      if (c + 1 <= c_last) issue_k(c + 1); else cp_async_commit();
      cp_async_wait<2>();  // K(c) (and Q) landed
    }
    __syncthreads();
    // Warps whose rows all precede this chunk would only fold identities.
    const bool active = c * kChunk <= prefix + warp_last_tok;
    PFrag pf[G];
    FoldScalars fs[G][2];
    if (active) {
      float s[G][kNTilesS][4];
      chunk_qk<G>(s,
                  [&](int gi, int ks, uint32_t(&a)[4]) {
                    if constexpr (QS) {
                      ldmatrix_x4(a, sQ + (warp * kWarpRows + gi * 16 + (mat & 1) * 8 + r8) * kRowStride +
                                         ks * 16 + (mat >> 1) * 8);
                    } else {
                      a[0] = qf.a[ks][0]; a[1] = qf.a[ks][1]; a[2] = qf.a[ks][2]; a[3] = qf.a[ks][3];
                    }
                  },
                  sK(c), lane);
#pragma unroll
      for (int gi = 0; gi < G; ++gi) {
        const int gbase = row_base + gi * 16;
        float m[2], l[2];
        scores_to_p(s[gi], pf[gi], m, l, scale_log2,
                    [&](int r, int key) {
                      const int i = gbase + r;
                      return i < rows_total && c * kChunk + key <= prefix + tok(i);
                    },
                    gbase + 15 < rows_total && (c + 1) * kChunk - 1 <= prefix + tok(gbase), lane);
#pragma unroll
        for (int r = 0; r < 2; ++r) fs[gi][r] = fold_scalars(grp[gi].M[r], grp[gi].L[r], m[r], l[r]);
      }
    }
    if constexpr (VB == 1) {
      cp_async_wait<1>();  // V(c) landed
      __syncthreads();
    }
    if (active) {
      chunk_pv<G>(pf, sV(c), lane, [&](int gi, int n, const float(&o)[4]) {
#pragma unroll
        for (int i = 0; i < 4; ++i) grp[gi].O[n][i] = fold_elem(grp[gi].O[n][i], o[i], fs[gi][i >> 1]);
      });
    }
    if (c % kGroupChunks == kGroupChunks - 1 || c == c_last) finish_group(c / kGroupChunks);
    __syncthreads();
    if constexpr (VB == 1) {
      if (c + 1 <= c_last) issue_v(c + 1); else cp_async_commit();
    }
  }
  cp_async_wait<0>();
  if (kv_splits != 1) return;

#pragma unroll
  for (int gi = 0; gi < G; ++gi) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int i = row_base + gi * 16 + g + r * 8;
      if (i >= rows_total) continue;
      __nv_bfloat16* o = out + (row_q(i) - q);
#pragma unroll
      for (int n = 0; n < kNTilesO; ++n) {
        __nv_bfloat162 v;
        v.x = finish(tot[gi].O[n][2 * r + 0], tot[gi].L[r]);
        v.y = finish(tot[gi].O[n][2 * r + 1], tot[gi].L[r]);
        *reinterpret_cast<__nv_bfloat162*>(o + n * 8 + t * 2) = v;
      }
    }
  }
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
  const int n_groups = pos / (kChunk * kGroupChunks) + 1;
  const int64_t base = ((int64_t)row * num_qo_heads + h) * max_groups;
  float M = -INFINITY, L = 0.f, O = 0.f;
  for (int fg = 0; fg < n_groups; ++fg) {
    const float2 ml = ws_ml[base + fg];
    const FoldScalars f = fold_scalars(M, L, ml.x, ml.y);
    O = fold_elem(O, ws_o[(base + fg) * kHeadDim + d], f);
  }
  out[((int64_t)row * num_qo_heads + h) * kHeadDim + d] = finish(O, L);
}

template <int DW, int KVB>
constexpr int decode_smem_elems() {
  return DW * KVB * kTileElems;
}

// Compact decode workspace: request b's chunk partials occupy rows
// [chunk_offsets[b], chunk_offsets[b + 1]) of ws_o [capacity, Hq, 128] and
// ws_ml [capacity, Hq, 2], so the workspace scales with live KV, not with
// batch x max sequence length. One block computes the exclusive prefix sum.
__global__ void decode_chunk_offsets_kernel(const int32_t* __restrict__ kv_lens, int batch,
                                            int32_t* __restrict__ chunk_offsets) {
  if (threadIdx.x != 0) return;
  int32_t total = 0;
  for (int b = 0; b < batch; ++b) {
    chunk_offsets[b] = total;
    total += (kv_lens[b] + kChunk - 1) / kChunk;
  }
  chunk_offsets[batch] = total;
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
                          const int32_t* __restrict__ kv_page_indices, int64_t table_stride,
                          const int32_t* __restrict__ kv_lens, float* __restrict__ ws_o,
                          float2* __restrict__ ws_ml, const int32_t* __restrict__ chunk_offsets,
                          int capacity, int num_qo_heads, int num_kv_heads, int page_size,
                          int page_shift, float scale_log2) {
  extern __shared__ __align__(16) __nv_bfloat16 smem[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  __nv_bfloat16* sK = smem + warp * KVB * kTileElems;
  __nv_bfloat16* sV = sK + (KVB - 1) * kTileElems;
  const int kvh = blockIdx.x, b = blockIdx.y;
  const int group = num_qo_heads / num_kv_heads;
  const int kv_len = kv_lens[b];
  const int n_chunks = (kv_len + kChunk - 1) / kChunk;
  const int pos = kv_len - 1;  // the decoded token; its KV is already in the cache
  const int32_t* pages = request_pages(kv_page_indices, kv_page_indptr, table_stride, b);
  const int64_t q_stride = (int64_t)num_qo_heads * kHeadDim;
  const int stride = gridDim.z * DW;
  int c = blockIdx.z * DW + warp;
  if (c >= n_chunks) return;
  auto load = [&](__nv_bfloat16* tile, const __nv_bfloat16* cache, int chunk) {
    issue_tile(tile, cache, pages, page_size, page_shift, num_kv_heads, kvh, chunk, kv_len, lane, 32);
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
    // Never write past the workspace even if lengths and capacity disagree.
    const int64_t row = (int64_t)chunk_offsets[b] + c;
    const bool in_bounds = row < capacity;
    const int64_t slot0 = row * num_qo_heads + kvh * group + g;
    const int64_t slot1 = slot0 + 8;
    chunk_pv<1>(pf, sV, lane, [&](int, int n, const float(&o)[4]) {
      if (in_bounds && g < group)
        *reinterpret_cast<float2*>(ws_o + slot0 * kHeadDim + n * 8 + t * 2) = make_float2(o[0], o[1]);
      if (in_bounds && g + 8 < group)
        *reinterpret_cast<float2*>(ws_o + slot1 * kHeadDim + n * 8 + t * 2) = make_float2(o[2], o[3]);
    });
    if (t == 0 && in_bounds) {
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

// Decode merge. grid = (num_qo_heads, batch), 128 threads (one per head dim).
// Chunk partials are prefetched in batches and reduced with the canonical
// two-level tree: sequential within each fold group, then across groups.
__global__ void __launch_bounds__(kHeadDim)
    decode_merge_kernel(const float* __restrict__ ws_o, const float2* __restrict__ ws_ml,
                        const int32_t* __restrict__ kv_lens,
                        const int32_t* __restrict__ chunk_offsets, int capacity,
                        __nv_bfloat16* __restrict__ out, int num_qo_heads) {
  constexpr int kBatch = 8;
  static_assert(kGroupChunks % kBatch == 0, "prefetch batches must not straddle fold groups");
  const int h = blockIdx.x, b = blockIdx.y, d = threadIdx.x;
  const int64_t first = chunk_offsets[b];
  const int64_t want = (kv_lens[b] + kChunk - 1) / kChunk, room = capacity - first;
  const int n_chunks = (int)(want < room ? want : room);
  float tM = -INFINITY, tL = 0.f, tO = 0.f;  // total
  float gM = -INFINITY, gL = 0.f, gO = 0.f;  // current fold group
  for (int c0 = 0; c0 < n_chunks; c0 += kBatch) {
    float2 ml[kBatch];
    float ov[kBatch];
#pragma unroll
    for (int i = 0; i < kBatch; ++i) {
      if (c0 + i < n_chunks) {
        const int64_t slot = (first + c0 + i) * num_qo_heads + h;
        ml[i] = ws_ml[slot];
        ov[i] = ws_o[slot * kHeadDim + d];
      }
    }
#pragma unroll
    for (int i = 0; i < kBatch; ++i) {
      const int c = c0 + i;
      if (c < n_chunks) {
        const FoldScalars f = fold_scalars(gM, gL, ml[i].x, ml[i].y);
        gO = fold_elem(gO, ov[i], f);
        if (c % kGroupChunks == kGroupChunks - 1 || c == n_chunks - 1) {
          const FoldScalars fg = fold_scalars(tM, tL, gM, gL);
          tO = fold_elem(tO, gO, fg);
          gM = -INFINITY; gL = 0.f; gO = 0.f;
        }
      }
    }
  }
  out[((int64_t)b * num_qo_heads + h) * kHeadDim + d] = finish(tO, tL);
}

}  // namespace pi_attn
