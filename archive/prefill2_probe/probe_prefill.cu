// PERF PROBE ONLY (not invariance-tested): prefill variants.
// MODE 0 = current kernel; 1 = online softmax within each fold group (MMA into O),
// groups folded into a register total; 2 = as 1 with one accumulator (upper bound).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include "pi_attn_kernels.cu"
static int page_shift_of(int page_size) {
  int s = 0;
  while ((1 << s) < page_size) ++s;
  return (1 << s) == page_size ? s : -1;
}
namespace pi_attn {
template <int NWARPS, int G, int VB, bool QS, int MODE>
__global__ void __launch_bounds__(NWARPS * 32)
    probe_kernel(const __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ out,
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
    if (MODE == 2) return;
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
    if (active && !(MODE == 1 || MODE == 2)) {
      float s[G][kNTilesS][4];
      if (MODE == 6 || MODE == 7) {
#pragma unroll
        for (int gi = 0; gi < G; ++gi)
#pragma unroll
          for (int n = 0; n < kNTilesS; ++n) s[gi][n][0] = s[gi][n][1] = s[gi][n][2] = s[gi][n][3] = 0.f;
      } else
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
                    MODE == 3 || (gbase + 15 < rows_total && (c + 1) * kChunk - 1 <= prefix + tok(gbase)), lane);
#pragma unroll
        for (int r = 0; r < 2; ++r) fs[gi][r] = fold_scalars(grp[gi].M[r], grp[gi].L[r], m[r], l[r]);
      }
    }
    if constexpr (VB == 1) {
      if (!(MODE == 1 || MODE == 2)) {
        cp_async_wait<1>();  // V(c) landed
        __syncthreads();
      }
    }
    if (active && !(MODE == 1 || MODE == 2 || MODE == 5 || MODE == 7)) {
      chunk_pv<G>(pf, sV(c), lane, [&](int gi, int n, const float(&o)[4]) {
#pragma unroll
        for (int i = 0; i < 4; ++i) grp[gi].O[n][i] = fold_elem(grp[gi].O[n][i], o[i], fs[gi][i >> 1]);
      });
    }
    if (active && MODE != 0) {  // online softmax within the fold group, MMA into O
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
        float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int n = 0; n < kNTilesS; ++n)
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int r = g + (i >> 1) * 8, key = n * 8 + t * 2 + (i & 1);
            const bool vis = gbase + r < rows_total && c * kChunk + key <= prefix + tok(gbase + r);
            const float v = vis ? mul(s[gi][n][i], scale_log2) : -INFINITY;
            s[gi][n][i] = v;
            mx[i >> 1] = fmaxf(mx[i >> 1], v);
          }
        float alpha[2], mm[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
          mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
          const float mn = fmaxf(grp[gi].M[r], mx[r]);
          mm[r] = mn == -INFINITY ? 0.f : mn;
          alpha[r] = ex2(sub(grp[gi].M[r], mm[r]));
          grp[gi].M[r] = mn;
        }
        float lsum[2] = {0.f, 0.f};
#pragma unroll
        for (int n = 0; n < kNTilesS; ++n) {
          float pv[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            pv[i] = round_bf16(ex2(sub(s[gi][n][i], mm[i >> 1])));
            lsum[i >> 1] = add(lsum[i >> 1], pv[i]);
          }
          const int j = n >> 1, h = (n & 1) * 2;
          pf[gi].a[j][h + 0] = pack_bf16(pv[0], pv[1]);
          pf[gi].a[j][h + 1] = pack_bf16(pv[2], pv[3]);
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          lsum[r] = add(lsum[r], __shfl_xor_sync(0xffffffffu, lsum[r], 1));
          lsum[r] = add(lsum[r], __shfl_xor_sync(0xffffffffu, lsum[r], 2));
          grp[gi].L[r] = __fmaf_rn(grp[gi].L[r], alpha[r], lsum[r]);
        }
#pragma unroll
        for (int n = 0; n < kNTilesO; ++n)
#pragma unroll
          for (int i = 0; i < 4; ++i) grp[gi].O[n][i] = mul(grp[gi].O[n][i], alpha[i >> 1]);
      }
    }
    if constexpr (VB == 1) {
      if (MODE != 0) {
        cp_async_wait<1>();
        __syncthreads();
      }
    }
    if (active && MODE != 0) {
#pragma unroll
      for (int j = 0; j < kKStepsPV; ++j) {
#pragma unroll
        for (int n = 0; n < kNTilesO; n += 2) {
          uint32_t bv[4];
          ldmatrix_x4_trans(bv, sV(c) + (j * 16 + (mat & 1) * 8 + r8) * kRowStride + (n + (mat >> 1)) * 8);
#pragma unroll
          for (int gi = 0; gi < G; ++gi) {
            mma_bf16(grp[gi].O[n], pf[gi].a[j], bv[0], bv[1]);
            mma_bf16(grp[gi].O[n + 1], pf[gi].a[j], bv[2], bv[3]);
          }
        }
      }
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
        const RowFold& fin = MODE == 2 ? grp[gi] : tot[gi];
        v.x = finish(fin.O[n][2 * r + 0], fin.L[r]);
        v.y = finish(fin.O[n][2 * r + 1], fin.L[r]);
        *reinterpret_cast<__nv_bfloat162*>(o + n * 8 + t * 2) = v;
      }
    }
  }
}

}  // namespace pi_attn
using namespace pi_attn;

at::Tensor probe(at::Tensor q, at::Tensor k_cache, at::Tensor v_cache, at::Tensor qo_indptr,
                 at::Tensor kv_page_indptr, at::Tensor kv_page_indices, at::Tensor kv_lens,
                 int64_t max_q_len, double sm_scale, int64_t cfg, int64_t mode) {
  const c10::cuda::CUDAGuard guard(q.device());
  auto out = at::empty_like(q);
  const int batch = kv_lens.size(0), hq = q.size(1), hkv = k_cache.size(2), page_size = k_cache.size(1);
  const int group = hq / hkv;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto launch = [&](auto kernel, int warps, int rows, size_t smem) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    const int tiles = (int)((max_q_len * group + rows - 1) / rows);
    kernel<<<dim3(tiles, hkv, batch), warps * 32, smem, stream>>>(
        (const __nv_bfloat16*)q.data_ptr(), (__nv_bfloat16*)out.data_ptr(),
        (const __nv_bfloat16*)k_cache.data_ptr(), (const __nv_bfloat16*)v_cache.data_ptr(),
        qo_indptr.data_ptr<int32_t>(), kv_page_indptr.data_ptr<int32_t>(),
        kv_page_indices.data_ptr<int32_t>(), (int64_t)0, kv_lens.data_ptr<int32_t>(), nullptr, hq, hkv,
        page_size, page_shift_of(page_size), (float)(sm_scale * 1.4426950408889634), 1, nullptr, nullptr, 1);
  };
#define P_(NW, G, VB, QS, M) launch(probe_kernel<NW, G, VB, QS, M>, NW, NW * 16 * G, \
    prefill_smem_elems<NW, G, VB, QS>() * sizeof(__nv_bfloat16))
#define PM_(NW, G, VB, QS) switch (mode) { case 1: P_(NW, G, VB, QS, 1); break; case 2: P_(NW, G, VB, QS, 2); break; case 3: P_(NW, G, VB, QS, 3); break; case 5: P_(NW, G, VB, QS, 5); break; case 6: P_(NW, G, VB, QS, 6); break; case 7: P_(NW, G, VB, QS, 7); break; default: P_(NW, G, VB, QS, 0); }
  // cfg 2 = 8 warps x 16 rows (engine), 7 = 4 warps x 16 rows, 8 = 4 warps x 32 rows (Q in smem, 1 V tile)
  if (cfg == 7) { PM_(4, 1, 2, false); }
  else if (cfg == 8) { PM_(4, 2, 1, true); }
  else { PM_(8, 1, 2, false); }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("probe", &probe); }
