// PI-Attn kernels: paged prefill (register fold), decode partials, merge.
// KV cache layout per layer: [num_pages, page_size, num_kv_heads, 128] bf16.
#include "pi_attn.cuh"

namespace pi_attn {

constexpr int kTileElems = kChunk * kRowStride;  // one K or V chunk tile in smem

// Async-load one K or V chunk tile. Keys at positions >= kv_len are zero-filled
// (no global read), so stale cache contents can never reach the arithmetic.
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

template <int R>
__device__ __forceinline__ void fold_rows(float (&M)[2], float (&L)[2], float (&O)[kNTilesO][4],
                                          const float (&m)[2], const float (&l)[2],
                                          const float (&o)[kNTilesO][4]) {
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const FoldScalars f = fold_scalars(M[r], L[r], m[r], l[r]);
#pragma unroll
    for (int n = 0; n < kNTilesO; ++n) {
      O[n][2 * r + 0] = fold_elem(O[n][2 * r + 0], o[n][2 * r + 0], f);
      O[n][2 * r + 1] = fold_elem(O[n][2 * r + 1], o[n][2 * r + 1], f);
    }
  }
}

constexpr int kPrefillSmemElems = 4 * kTileElems;  // K and V, double-buffered

// Prefill / extend. grid = (q tiles of 16*NWARPS rows, num_qo_heads, batch).
// Query row i of request b sits at absolute position kv_len[b] - q_len + i.
// NWARPS only changes how many 16-row groups share a K/V tile, never a row's ops.
template <int NWARPS>
__global__ void __launch_bounds__(NWARPS * 32)
    prefill_kernel(const __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ out,
                   const __nv_bfloat16* __restrict__ k_cache,
                   const __nv_bfloat16* __restrict__ v_cache, const int32_t* __restrict__ qo_indptr,
                   const int32_t* __restrict__ kv_page_indptr,
                   const int32_t* __restrict__ kv_page_indices, const int32_t* __restrict__ kv_lens,
                   int num_qo_heads, int num_kv_heads, int page_size, float scale_log2) {
  constexpr int kPrefillRows = NWARPS * 16;
  extern __shared__ __align__(16) __nv_bfloat16 smem[];
  const int b = blockIdx.z, h = blockIdx.y, tile = blockIdx.x;
  const int q0 = qo_indptr[b], q_len = qo_indptr[b + 1] - q0;
  if (tile * kPrefillRows >= q_len) return;
  const int kv_len = kv_lens[b], prefix = kv_len - q_len;
  const int kvh = h / (num_qo_heads / num_kv_heads);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row_base = tile * kPrefillRows + warp * 16;
  const int64_t q_stride = (int64_t)num_qo_heads * kHeadDim;
  const int32_t* pages = kv_page_indices + kv_page_indptr[b];
  const int last_row = min(tile * kPrefillRows + kPrefillRows, q_len) - 1;
  const int last_chunk = (prefix + last_row) / kChunk;

  auto sK = [&](int c) { return smem + (c & 1) * 2 * kTileElems; };
  auto sV = [&](int c) { return smem + (c & 1) * 2 * kTileElems + kTileElems; };
  auto issue = [&](int c) {
    issue_tile(sK(c), k_cache, pages, page_size, num_kv_heads, kvh, c, kv_len, threadIdx.x, blockDim.x);
    issue_tile(sV(c), v_cache, pages, page_size, num_kv_heads, kvh, c, kv_len, threadIdx.x, blockDim.x);
    cp_async_commit();
  };
  issue(0);

  QFrag qf;
  load_q(qf,
         [&](int r) -> const __nv_bfloat16* {
           const int i = row_base + r;
           return i < q_len ? q + (q0 + i) * q_stride + h * kHeadDim : nullptr;
         },
         lane);

  float M[2] = {-INFINITY, -INFINITY}, L[2] = {0.f, 0.f};
  float O[kNTilesO][4];
#pragma unroll
  for (int n = 0; n < kNTilesO; ++n) O[n][0] = O[n][1] = O[n][2] = O[n][3] = 0.f;

  for (int c = 0; c <= last_chunk; ++c) {
    if (c + 1 <= last_chunk) issue(c + 1); else cp_async_commit();
    cp_async_wait<1>();
    __syncthreads();
    // Warps whose rows all precede this chunk would only fold identities.
    if (c * kChunk <= prefix + min(row_base + 15, q_len - 1)) {
      PFrag pf;
      float m[2], l[2], o[kNTilesO][4];
      chunk_scores(pf, m, l, qf, sK(c), scale_log2,
                   [&](int r, int key) {
                     const int i = row_base + r;
                     return i < q_len && c * kChunk + key <= prefix + i;
                   },
                   lane, row_base + 15 < q_len && (c + 1) * kChunk - 1 <= prefix + row_base);
      chunk_values(o, pf, sV(c), lane);
      fold_rows<2>(M, L, O, m, l, o);
    }
    __syncthreads();
  }
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int i = row_base + g + r * 8;
    if (i >= q_len) continue;
    __nv_bfloat16* o = out + (q0 + i) * q_stride + h * kHeadDim;
#pragma unroll
    for (int n = 0; n < kNTilesO; ++n) {
      __nv_bfloat162 v;
      v.x = finish(O[n][2 * r + 0], L[r]);
      v.y = finish(O[n][2 * r + 1], L[r]);
      *reinterpret_cast<__nv_bfloat162*>(o + n * 8 + t * 2) = v;
    }
  }
}

constexpr int kDecodeWarps = 2;
constexpr int kDecodeSmemElems = kDecodeWarps * 2 * kTileElems;

// Decode partials. grid = (num_kv_heads, batch, split CTAs). Each warp owns one
// chunk at a time (chunks strided over all warps of the request) and writes the
// partial of the GQA group's query rows. The split count only changes who
// computes a partial, never the partial. K of the next chunk streams in while
// this chunk's P V runs; V of the next chunk while the partial is written.
__global__ void __launch_bounds__(kDecodeWarps * 32)
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
  __nv_bfloat16* sK = smem + warp * 2 * kTileElems;
  __nv_bfloat16* sV = sK + kTileElems;
  const int kvh = blockIdx.x, b = blockIdx.y;
  const int group = num_qo_heads / num_kv_heads;
  const int kv_len = kv_lens[b];
  const int n_chunks = (kv_len + kChunk - 1) / kChunk;
  const int pos = kv_len - 1;  // the decoded token; its KV is already in the cache
  const int32_t* pages = kv_page_indices + kv_page_indptr[b];
  const int64_t q_stride = (int64_t)num_qo_heads * kHeadDim;
  const int stride = gridDim.z * kDecodeWarps;
  int c = blockIdx.z * kDecodeWarps + warp;
  if (c >= n_chunks) return;

  issue_tile(sK, k_cache, pages, page_size, num_kv_heads, kvh, c, kv_len, lane, 32);
  cp_async_commit();
  issue_tile(sV, v_cache, pages, page_size, num_kv_heads, kvh, c, kv_len, lane, 32);
  cp_async_commit();

  QFrag qf;
  load_q(qf,
         [&](int r) -> const __nv_bfloat16* {
           return r < group ? q + b * q_stride + (kvh * group + r) * kHeadDim : nullptr;
         },
         lane);
  const int g = lane >> 2, t = lane & 3;
  for (; c < n_chunks; c += stride) {
    const int next = c + stride;
    cp_async_wait<1>();  // K(c) landed; V(c) may be in flight
    __syncwarp();
    PFrag pf;
    float m[2], l[2], o[kNTilesO][4];
    chunk_scores(pf, m, l, qf, sK, scale_log2,
                 [&](int r, int key) { return r < group && c * kChunk + key <= pos; }, lane);
    __syncwarp();
    if (next < n_chunks) issue_tile(sK, k_cache, pages, page_size, num_kv_heads, kvh, next, kv_len, lane, 32);
    cp_async_commit();
    cp_async_wait<1>();  // V(c) landed; K(next) may be in flight
    __syncwarp();
    chunk_values(o, pf, sV, lane);
    __syncwarp();
    if (next < n_chunks) issue_tile(sV, v_cache, pages, page_size, num_kv_heads, kvh, next, kv_len, lane, 32);
    cp_async_commit();
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int row = g + r * 8;
      if (row >= group) continue;
      const int64_t slot = ((int64_t)b * num_qo_heads + kvh * group + row) * max_chunks + c;
      if (t == 0) ws_ml[slot] = make_float2(m[r], l[r]);
      float* dst = ws_o + slot * kHeadDim;
#pragma unroll
      for (int n = 0; n < kNTilesO; ++n)
        *reinterpret_cast<float2*>(dst + n * 8 + t * 2) = make_float2(o[n][2 * r], o[n][2 * r + 1]);
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
