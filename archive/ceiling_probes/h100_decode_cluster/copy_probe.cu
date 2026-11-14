// How fast can one CTA take in a fold group's K and V (32 tile loads of 32 keys x 256 B, key rows 2048 B
// apart as with 8 kv heads)? Modes:
//   0 cp.async 16 B, decode_kernel's mapping (a warp instruction: 8 keys x 64 B = 8 half lines)
//   1 cp.async 16 B, a warp instruction = 2 keys x 256 B (4 whole lines)
//   2 ld.global 16 B to a register + st.shared, mapping of mode 1
//   3 cp.async.bulk, one 256 B copy per key (32 threads issue)
//   4 cp.async.bulk, one 8 KB copy per tile from a contiguous source (the TMA ceiling)
// `depth` tiles are issued before each wait; warps > 4: each further 4 warps take every other tile.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src));
}
__device__ __forceinline__ void bulk(void* dst, const void* src, int bytes, void* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
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

__global__ void __launch_bounds__(512) copy_kernel(const char* src, int mode, int tiles, int depth, float* sink) {
  extern __shared__ __align__(128) char sm[];
  char* bar = sm;
  char* buf = sm + 128;  // [depth][kTileBytes]
  const int tid = threadIdx.x, sets = blockDim.x / 128, set = tid / 128, t = tid % 128;
  src += (size_t)blockIdx.x * tiles * 32 * kKeyStride;
  if (mode >= 3 && tid == 0) bar_init(bar, 1);
  __syncthreads();
  int phase = 0;
  for (int t0 = 0; t0 < tiles; t0 += depth) {
    if (mode >= 3 && tid == 0) bar_expect(bar, min(depth, tiles - t0) * kTileBytes);
    for (int d = 0; d < depth && t0 + d < tiles; ++d) {
      const int tile = t0 + d;
      const char* ts = src + (size_t)tile * 32 * kKeyStride;
      char* td = buf + d * kTileBytes;
      if (mode == 4) {
        if (tid == 0) bulk(td, src + (size_t)tile * kTileBytes, kTileBytes, bar);
        continue;
      }
      if (mode == 3) {
        if (tid < 32) bulk(td + tid * 256, ts + (size_t)tid * kKeyStride, 256, bar);
        continue;
      }
      if (tile % sets != set) continue;
      if (mode == 0) {
        const int key = t >> 2, c0 = t & 3;
#pragma unroll
        for (int i = 0; i < 4; ++i) cp16(td + key * 256 + (c0 + 4 * i) * 16, ts + (size_t)key * kKeyStride + (c0 + 4 * i) * 16);
      } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int idx = t + 128 * i, key = idx >> 4, ch = idx & 15;
          if (mode == 1) cp16(td + key * 256 + ch * 16, ts + (size_t)key * kKeyStride + ch * 16);
          else *reinterpret_cast<int4*>(td + key * 256 + ch * 16) = *reinterpret_cast<const int4*>(ts + (size_t)key * kKeyStride + ch * 16);
        }
      }
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

void copy(at::Tensor src, at::Tensor sink, int64_t ctas, int64_t mode, int64_t warps, int64_t tiles, int64_t depth) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const int smem = 128 + depth * kTileBytes;
  TORCH_CHECK(src.numel() >= ctas * tiles * 32 * kKeyStride);
  cudaFuncSetAttribute(copy_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  copy_kernel<<<ctas, warps * 32, smem, stream>>>((const char*)src.data_ptr(), (int)mode, (int)tiles, (int)depth, sink.data_ptr<float>());
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("copy", &copy); }
