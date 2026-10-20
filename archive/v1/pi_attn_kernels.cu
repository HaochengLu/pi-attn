// PI-Attn kernels: paged prefill (register fold), decode partials, merge.
// KV cache layout per layer: [num_pages, page_size, num_kv_heads, 128] bf16.
#include "pi_attn.cuh"

namespace pi_attn {

// Load chunk `c` of one request/kv-head into smem. Keys at positions >= kv_len
// are zero-filled so stale cache contents can never reach the arithmetic.
__device__ __forceinline__ void load_chunk(__nv_bfloat16* sK, __nv_bfloat16* sV,
                                           const __nv_bfloat16* k_cache,
                                           const __nv_bfloat16* v_cache,
                                           const int32_t* pages, int page_size, int num_kv_heads,
                                           int kvh, int c, int kv_len, int tid, int nthreads) {
  constexpr int kVecPerRow = kHeadDim / 8;  // 16-byte vectors
  for (int i = tid; i < kChunk * kVecPerRow; i += nthreads) {
    const int key = i / kVecPerRow, vec = i % kVecPerRow;
    const int pos = c * kChunk + key;
    uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
    if (pos < kv_len) {
      const int64_t page = pages[pos / page_size];
      const int64_t off =
          ((page * page_size + pos % page_size) * num_kv_heads + kvh) * kHeadDim + vec * 8;
      kv = *reinterpret_cast<const uint4*>(k_cache + off);
      vv = *reinterpret_cast<const uint4*>(v_cache + off);
    }
    *reinterpret_cast<uint4*>(sK + key * kRowStride + vec * 8) = kv;
    *reinterpret_cast<uint4*>(sV + key * kRowStride + vec * 8) = vv;
  }
}

constexpr int kPrefillWarps = 4;
constexpr int kPrefillRows = kPrefillWarps * 16;

// Prefill / extend. grid = (q tiles of 64 rows, num_qo_heads, batch).
// Query row i of request b sits at absolute position kv_len[b] - q_len + i.
__global__ void __launch_bounds__(kPrefillWarps * 32)
    prefill_kernel(const __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ out,
                   const __nv_bfloat16* __restrict__ k_cache,
                   const __nv_bfloat16* __restrict__ v_cache, const int32_t* __restrict__ qo_indptr,
                   const int32_t* __restrict__ kv_page_indptr,
                   const int32_t* __restrict__ kv_page_indices, const int32_t* __restrict__ kv_lens,
                   int num_qo_heads, int num_kv_heads, int page_size, float scale_log2) {
  extern __shared__ __align__(16) __nv_bfloat16 smem[];
  __nv_bfloat16* sK = smem;
  __nv_bfloat16* sV = smem + kChunk * kRowStride;
  const int b = blockIdx.z, h = blockIdx.y, tile = blockIdx.x;
  const int q0 = qo_indptr[b], q_len = qo_indptr[b + 1] - q0;
  if (tile * kPrefillRows >= q_len) return;
  const int kv_len = kv_lens[b], prefix = kv_len - q_len;
  const int kvh = h / (num_qo_heads / num_kv_heads);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row_base = tile * kPrefillRows + warp * 16;  // first q index of this warp
  const int64_t q_stride = (int64_t)num_qo_heads * kHeadDim;

  QFrag qf;
  load_q(qf,
         [&](int r) -> const __nv_bfloat16* {
           const int i = row_base + r;
           return i < q_len ? q + (q0 + i) * q_stride + h * kHeadDim : nullptr;
         },
         lane);

  const int g = lane >> 2;
  float M[2] = {-INFINITY, -INFINITY}, L[2] = {0.f, 0.f};
  float O[kNTilesO][4];
#pragma unroll
  for (int n = 0; n < kNTilesO; ++n) O[n][0] = O[n][1] = O[n][2] = O[n][3] = 0.f;

  const int last_row = min(tile * kPrefillRows + kPrefillRows, q_len) - 1;
  const int last_chunk = (prefix + last_row) / kChunk;
  const int32_t* pages = kv_page_indices + kv_page_indptr[b];
  for (int c = 0; c <= last_chunk; ++c) {
    __syncthreads();
    load_chunk(sK, sV, k_cache, v_cache, pages, page_size, num_kv_heads, kvh, c, kv_len,
               threadIdx.x, blockDim.x);
    __syncthreads();
    Partial p;
    chunk_partial(p, qf, sK, sV, scale_log2,
                  [&](int r, int key) {
                    const int i = row_base + r;
                    return i < q_len && c * kChunk + key <= prefix + i;
                  },
                  lane);
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const FoldScalars f = fold_scalars(M[r], L[r], p.m[r], p.l[r]);
#pragma unroll
      for (int n = 0; n < kNTilesO; ++n) {
        O[n][2 * r + 0] = fold_elem(O[n][2 * r + 0], p.o[n][2 * r + 0], f);
        O[n][2 * r + 1] = fold_elem(O[n][2 * r + 1], p.o[n][2 * r + 1], f);
      }
    }
  }
  const int t = lane & 3;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int i = row_base + g + r * 8;
    if (i >= q_len) continue;
    __nv_bfloat16* o = out + (q0 + i) * q_stride + h * kHeadDim;
#pragma unroll
    for (int n = 0; n < kNTilesO; ++n) {
      o[n * 8 + t * 2 + 0] = finish(O[n][2 * r + 0], L[r]);
      o[n * 8 + t * 2 + 1] = finish(O[n][2 * r + 1], L[r]);
    }
  }
}

constexpr int kDecodeWarps = 2;

// Decode partials. grid = (num_kv_heads, batch, split CTAs). Each warp owns one
// chunk at a time (chunks are strided over all warps of the request), computes
// the partial for the GQA group's query rows and writes it to the workspace.
// The split count is free: it only changes who computes each partial.
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
  __nv_bfloat16* sK = smem + warp * kChunkSmemElems;
  __nv_bfloat16* sV = sK + kChunk * kRowStride;
  const int kvh = blockIdx.x, b = blockIdx.y;
  const int group = num_qo_heads / num_kv_heads;
  const int kv_len = kv_lens[b];
  const int n_chunks = (kv_len + kChunk - 1) / kChunk;
  const int pos = kv_len - 1;  // the decoded token; its KV is already in the cache
  const int32_t* pages = kv_page_indices + kv_page_indptr[b];
  const int64_t q_stride = (int64_t)num_qo_heads * kHeadDim;

  QFrag qf;
  load_q(qf,
         [&](int r) -> const __nv_bfloat16* {
           return r < group ? q + b * q_stride + (kvh * group + r) * kHeadDim : nullptr;
         },
         lane);
  const int g = lane >> 2, t = lane & 3;
  const int stride = gridDim.z * kDecodeWarps;
  for (int c = blockIdx.z * kDecodeWarps + warp; c < n_chunks; c += stride) {
    __syncwarp();
    load_chunk(sK, sV, k_cache, v_cache, pages, page_size, num_kv_heads, kvh, c, kv_len, lane, 32);
    __syncwarp();
    Partial p;
    chunk_partial(p, qf, sK, sV, scale_log2,
                  [&](int r, int key) { return r < group && c * kChunk + key <= pos; }, lane);
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int row = g + r * 8;
      if (row >= group) continue;
      const int64_t slot = ((int64_t)b * num_qo_heads + kvh * group + row) * max_chunks + c;
      if (t == 0) ws_ml[slot] = make_float2(p.m[r], p.l[r]);
      float* o = ws_o + slot * kHeadDim;
#pragma unroll
      for (int n = 0; n < kNTilesO; ++n) {
        o[n * 8 + t * 2 + 0] = p.o[n][2 * r + 0];
        o[n * 8 + t * 2 + 1] = p.o[n][2 * r + 1];
      }
    }
  }
}

// Merge. grid = (num_qo_heads, batch), 128 threads (one per head dim).
__global__ void __launch_bounds__(kHeadDim)
    decode_merge_kernel(const float* __restrict__ ws_o, const float2* __restrict__ ws_ml,
                        const int32_t* __restrict__ kv_lens, __nv_bfloat16* __restrict__ out,
                        int num_qo_heads, int max_chunks) {
  const int h = blockIdx.x, b = blockIdx.y, d = threadIdx.x;
  const int n_chunks = (kv_lens[b] + kChunk - 1) / kChunk;
  const int64_t base = ((int64_t)b * num_qo_heads + h) * max_chunks;
  float M = -INFINITY, L = 0.f, O = 0.f;
  for (int c = 0; c < n_chunks; ++c) {
    const float2 ml = ws_ml[base + c];
    const FoldScalars f = fold_scalars(M, L, ml.x, ml.y);
    O = fold_elem(O, ws_o[(base + c) * kHeadDim + d], f);
  }
  out[((int64_t)b * num_qo_heads + h) * kHeadDim + d] = finish(O, L);
}

}  // namespace pi_attn
