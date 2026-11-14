// Reviewer probe (H100). (1) Intake of a fold group's K and V by one CTA, as copy_probe.cu of
// h100_decode_cluster, plus what that probe did not measure: cp.async.bulk.tensor boxes over the PAGED layout
// (a 2-D tensor map [slot][2048 B], box 256 B x 16 slots = one page of one kv head, 4 KB per instruction),
// with the pages of a CTA shuffled. Modes:
//   0 cp.async 16 B, decode_kernel's mapping            5 TMA box 256 B x 16 keys (one page), a thread per box
//   4 cp.async.bulk 8 KB contiguous (not paged)          6 TMA box 256 B x 32 keys (page size 32)
// (2) Is mma.sync m16n8k16 symmetric in its operands: D[i][j] of (A = X, B = Y, C) against D'[j][i] of
// (A = Y padded, B = X rows 0..7, C^T)? If so decode could put the KEYS in the 16-row operand.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_bf16.h>

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src));
}
__device__ __forceinline__ void bulk(void* dst, const void* src, int bytes, void* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void tma2d(void* dst, const CUtensorMap* tm, int c0, int c1, void* bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               ::"r"(smem_u32(dst)), "l"(tm), "r"(c0), "r"(c1), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void bar_init(void* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void bar_expect(void* bar, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes));
}
__device__ __forceinline__ void bar_wait(void* bar, int phase) {
  asm volatile("{\n.reg .pred P1;\nLAB_WAIT:\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
               "@P1 bra DONE;\nbra LAB_WAIT;\nDONE:\n}" ::"r"(smem_u32(bar)), "r"(phase));
}

constexpr int kTileBytes = 32 * 256, kKeyStride = 2048;

__global__ void __launch_bounds__(512) copy_kernel(const __grid_constant__ CUtensorMap tm, const char* src, int mode,
                                                   int tiles, int depth, int perm, float* sink) {
  extern __shared__ __align__(128) char sm[];
  char* bar = sm;
  char* buf = sm + 128;  // [depth][kTileBytes]
  const int tid = threadIdx.x, sets = blockDim.x / 128, set = tid / 128, t = tid % 128;
  const int cta_key0 = blockIdx.x * tiles * 32, npages = tiles * 2;
  src += (size_t)cta_key0 * kKeyStride;
  // slot (within the CTA's region) of key `key` of tile `tile`: pages of 16 keys, shuffled if perm
  auto slot = [&](int tile, int key) {
    const int page = tile * 2 + (key >> 4);
    return (perm ? (page * 37 + 11) % npages : page) * 16 + (key & 15);
  };
  if (mode >= 3 && tid == 0) bar_init(bar, 1);
  __syncthreads();
  int phase = 0;
  for (int t0 = 0; t0 < tiles; t0 += depth) {
    if (mode >= 3 && tid == 0) bar_expect(bar, min(depth, tiles - t0) * kTileBytes);
    for (int d = 0; d < depth && t0 + d < tiles; ++d) {
      const int tile = t0 + d;
      char* td = buf + d * kTileBytes;
      if (mode == 4) {
        if (tid == 0) bulk(td, src + (size_t)tile * kTileBytes, kTileBytes, bar);
        continue;
      }
      if (mode == 5) {
        for (int pg = 0; pg < 2; ++pg)
          if (tid == d * 2 + pg) tma2d(td + pg * 4096, &tm, 0, cta_key0 + slot(tile, pg * 16), bar);
        continue;
      }
      if (mode == 6) {
        if (tid == d) tma2d(td, &tm, 0, cta_key0 + tile * 32, bar);
        continue;
      }
      if (tile % sets != set) continue;
      const int key = t >> 2, c0 = t & 3;
      const char* ks = src + (size_t)slot(tile, key) * kKeyStride;
#pragma unroll
      for (int i = 0; i < 4; ++i) cp16(td + key * 256 + (c0 + 4 * i) * 16, ks + (c0 + 4 * i) * 16);
    }
    if (mode >= 3) {
      bar_wait(bar, phase);
      phase ^= 1;
    } else {
      asm volatile("cp.async.commit_group;");
      asm volatile("cp.async.wait_group 0;");
    }
    __syncthreads();
  }
  if (tid == 0 && buf[5] == 123 && buf[kTileBytes - 3] == 77) sink[0] = 1.f;
}

// Checksum of what a mode delivered: the smem image of every depth batch, summed with position weights.
__global__ void __launch_bounds__(512) check_kernel(const __grid_constant__ CUtensorMap tm, const char* src, int mode,
                                                    int tiles, int depth, int perm, long long* sums) {
  extern __shared__ __align__(128) char sm[];
  char* bar = sm;
  char* buf = sm + 128;
  const int tid = threadIdx.x, t = tid % 128;
  const int cta_key0 = blockIdx.x * tiles * 32, npages = tiles * 2;
  src += (size_t)cta_key0 * kKeyStride;
  auto slot = [&](int tile, int key) {
    const int page = tile * 2 + (key >> 4);
    return (perm ? (page * 37 + 11) % npages : page) * 16 + (key & 15);
  };
  if (tid == 0) bar_init(bar, 1);
  __syncthreads();
  int phase = 0;
  long long sum = 0;
  for (int t0 = 0; t0 < tiles; t0 += depth) {
    if (mode >= 3 && tid == 0) bar_expect(bar, min(depth, tiles - t0) * kTileBytes);
    for (int d = 0; d < depth && t0 + d < tiles; ++d) {
      const int tile = t0 + d;
      char* td = buf + d * kTileBytes;
      if (mode == 5) {
        for (int pg = 0; pg < 2; ++pg)
          if (tid == d * 2 + pg) tma2d(td + pg * 4096, &tm, 0, cta_key0 + slot(tile, pg * 16), bar);
        continue;
      }
      if (tid >= 128) continue;
      const int key = t >> 2, c0 = t & 3;
      const char* ks = src + (size_t)slot(tile, key) * kKeyStride;
#pragma unroll
      for (int i = 0; i < 4; ++i) cp16(td + key * 256 + (c0 + 4 * i) * 16, ks + (c0 + 4 * i) * 16);
    }
    if (mode >= 3) {
      bar_wait(bar, phase);
      phase ^= 1;
    } else {
      asm volatile("cp.async.commit_group;");
      asm volatile("cp.async.wait_group 0;");
    }
    __syncthreads();
    if (tid == 0)
      for (int d = 0; d < depth && t0 + d < tiles; ++d)
        for (int i = 0; i < kTileBytes; ++i)
          sum += (long long)(unsigned char)buf[d * kTileBytes + i] * ((t0 + d) * 8192 + i + 1);
    __syncthreads();
  }
  if (tid == 0) sums[blockIdx.x] = sum;
}

static CUtensorMap make_map(const void* src, int64_t keys, int box_keys) {
  CUtensorMap tm;
  cuuint64_t gdim[2] = {(cuuint64_t)kKeyStride, (cuuint64_t)keys}, gstride[1] = {(cuuint64_t)kKeyStride};
  cuuint32_t box[2] = {256, (cuuint32_t)box_keys}, es[2] = {1, 1};
  auto r = cuTensorMapEncodeTiled(&tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(src), gdim, gstride, box, es,
                                  CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                  CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled ", (int)r);
  return tm;
}

void copy(at::Tensor src, at::Tensor sink, int64_t ctas, int64_t mode, int64_t warps, int64_t tiles, int64_t depth,
          int64_t perm) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const int smem = 128 + depth * kTileBytes;
  TORCH_CHECK(src.numel() >= ctas * tiles * 32 * kKeyStride);
  CUtensorMap tm = make_map(src.data_ptr(), std::max<int64_t>(ctas * tiles * 32, 32), mode == 6 ? 32 : 16);
  cudaFuncSetAttribute(copy_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  copy_kernel<<<ctas, warps * 32, smem, stream>>>(tm, (const char*)src.data_ptr(), (int)mode, (int)tiles, (int)depth,
                                                  (int)perm, sink.data_ptr<float>());
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}
void check(at::Tensor src, at::Tensor sums, int64_t ctas, int64_t mode, int64_t tiles, int64_t depth, int64_t perm) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const int smem = 128 + depth * kTileBytes;
  CUtensorMap tm = make_map(src.data_ptr(), ctas * tiles * 32, 16);
  cudaFuncSetAttribute(check_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  check_kernel<<<ctas, 128, smem, stream>>>(tm, (const char*)src.data_ptr(), (int)mode, (int)tiles, (int)depth, (int)perm,
                                            (long long*)sums.data_ptr());
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

// ---- (2) mma.sync operand symmetry. x [n][16][16] bf16, y [n][8][16] bf16, c [n][16][8] f32.
// out0 [n][16][8]: D = C + X Y^T (A = X, B = Y).  out1 [n][8][8]: D'[i][j] = C[j][i] + Y_i . X_j (A = Y, B = X[0..8)).
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__global__ void sym_kernel(const __nv_bfloat16* x, const __nv_bfloat16* y, const float* c, float* out0, float* out1) {
  const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
  x += (size_t)blockIdx.x * 256, y += (size_t)blockIdx.x * 128, c += (size_t)blockIdx.x * 128;
  out0 += (size_t)blockIdx.x * 128, out1 += (size_t)blockIdx.x * 64;
  auto w = [](const __nv_bfloat16* p) { return *reinterpret_cast<const uint32_t*>(p); };
  {
    uint32_t a[4] = {w(x + g * 16 + 2 * t), w(x + (g + 8) * 16 + 2 * t), w(x + g * 16 + 8 + 2 * t), w(x + (g + 8) * 16 + 8 + 2 * t)};
    float d[4] = {c[g * 8 + 2 * t], c[g * 8 + 2 * t + 1], c[(g + 8) * 8 + 2 * t], c[(g + 8) * 8 + 2 * t + 1]};
    mma(d, a, w(y + g * 16 + 2 * t), w(y + g * 16 + 8 + 2 * t));
    out0[g * 8 + 2 * t] = d[0], out0[g * 8 + 2 * t + 1] = d[1];
    out0[(g + 8) * 8 + 2 * t] = d[2], out0[(g + 8) * 8 + 2 * t + 1] = d[3];
  }
  {
    uint32_t a[4] = {w(y + g * 16 + 2 * t), 0u, w(y + g * 16 + 8 + 2 * t), 0u};  // rows 8..15: zero
    float d[4] = {c[(2 * t) * 8 + g], c[(2 * t + 1) * 8 + g], 0.f, 0.f};         // D'[g][2t] starts from C[2t][g]
    mma(d, a, w(x + g * 16 + 2 * t), w(x + g * 16 + 8 + 2 * t));                 // B column n = g: X row g
    out1[g * 8 + 2 * t] = d[0], out1[g * 8 + 2 * t + 1] = d[1];
  }
}
void sym(at::Tensor x, at::Tensor y, at::Tensor c, at::Tensor out0, at::Tensor out1) {
  auto stream = at::cuda::getCurrentCUDAStream();
  sym_kernel<<<x.size(0), 32, 0, stream>>>((const __nv_bfloat16*)x.data_ptr(), (const __nv_bfloat16*)y.data_ptr(),
                                           c.data_ptr<float>(), out0.data_ptr<float>(), out1.data_ptr<float>());
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("copy", &copy);
  m.def("check", &check);
  m.def("sym", &sym);
}
