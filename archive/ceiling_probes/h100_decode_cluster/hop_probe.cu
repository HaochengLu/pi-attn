// Inter-CTA communication probes on a cluster of n CTAs:
//   ring : a token goes round the cluster `rounds` times; mode 0 = plain remote store of a counter, polled by
//          the receiver (volatile load, cluster fences); mode 1 = remote mbarrier arrive + blocking try_wait;
//          mode 2 = remote mbarrier arrive + test_wait spin. Time / (n * rounds) = one hand-off.
//   push : CTAs 1..n-1 each store `bytes` of their shared memory into CTA 0's (16 B per thread per store), then
//          arrive; CTA 0 waits. Time against bytes = DSMEM store throughput.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cooperative_groups.h>

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void fence_cluster() { asm volatile("fence.acq_rel.cluster;"); }
__device__ __forceinline__ void bar_init(void* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\nfence.mbarrier_init.release.cluster;" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void bar_arrive_remote(void* bar, int rank) {
  asm volatile("{\n.reg .b32 remAddr32;\nmapa.shared::cluster.u32 remAddr32, %0, %1;\n"
               "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n}" ::"r"(smem_u32(bar)), "r"(rank));
}
__device__ __forceinline__ void bar_try_wait(void* bar, int phase) {
  asm volatile("{\n.reg .pred P1;\nLAB_WAIT:\nmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1;\n"
               "@P1 bra DONE;\nbra LAB_WAIT;\nDONE:\n}" ::"r"(smem_u32(bar)), "r"(phase));
}
__device__ __forceinline__ void bar_test_wait(void* bar, int phase) {
  asm volatile("{\n.reg .pred P1;\nLAB_WAIT:\nmbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1;\n"
               "@P1 bra DONE;\nbra LAB_WAIT;\nDONE:\n}" ::"r"(smem_u32(bar)), "r"(phase));
}

// threads: 32 (one warp per CTA does everything; lane 0 signals).
__global__ void __launch_bounds__(128) ring_kernel(int n, int rounds, int mode, int waiters, float* sink) {
  extern __shared__ __align__(16) uint64_t sm[];
  auto cl = cooperative_groups::this_cluster();
  const int c = blockIdx.x % n, tid = threadIdx.x;
  volatile int* flag = reinterpret_cast<volatile int*>(sm + 2);
  if (tid == 0) {
    bar_init(sm, 1);
    *flag = 0;
  }
  cl.sync();
  if (tid >= waiters) return;  // `waiters` threads wait; thread 0 signals
  for (int r = 0; r < rounds; ++r) {
    if (c > 0 || r > 0) {  // wait for the token of this round (CTA 0 starts round 0)
      const int want = c > 0 ? r + 1 : r;
      if (mode == 0) {
        while (*flag < want) {}
        fence_cluster();
      } else if (mode == 1) {
        bar_try_wait(sm, (want - 1) & 1);
      } else {
        bar_test_wait(sm, (want - 1) & 1);
      }
    }
    if (tid == 0) {
      const int next = (c + 1) % n;
      if (mode == 0) {
        int* rf = cl.map_shared_rank(reinterpret_cast<int*>(sm + 2), next);
        fence_cluster();
        *rf = r + 1;
      } else {
        bar_arrive_remote(sm, next);
      }
    }
  }
  // CTA 0 receives the last token so that nobody exits while it may still be written to
  if (c == 0) {
    if (mode == 0) { while (*flag < rounds) {} }
    else if (mode == 1) bar_try_wait(sm, (rounds - 1) & 1);
    else bar_test_wait(sm, (rounds - 1) & 1);
  }
  if (tid == 0 && sm[2] == 0xdeadbeefull) sink[0] = 1.f;
}

__global__ void __launch_bounds__(128) push_kernel(int n, int bytes, float* sink) {
  extern __shared__ __align__(16) uint64_t sm[];
  auto cl = cooperative_groups::this_cluster();
  const int c = blockIdx.x % n, tid = threadIdx.x;
  float4* data = reinterpret_cast<float4*>(sm + 2);  // [n][bytes / 16]
  if (tid == 0) bar_init(sm, (n - 1) * 128);
  for (int i = tid; i < bytes / 16; i += 128) data[i] = make_float4(c, i, 1.f, 2.f);
  cl.sync();
  if (c > 0) {
    float4* dst = cl.map_shared_rank(data, 0) + (size_t)(c - 1) * (bytes / 16);
    for (int i = tid; i < bytes / 16; i += 128) dst[i] = data[i];
    bar_arrive_remote(sm, 0);
  } else {
    bar_try_wait(sm, 0);
    float acc = 0.f;
    for (int i = tid; i < (n - 1) * (bytes / 16); i += 128) acc += data[bytes / 16 + i].y;
    if (acc == -1.f) sink[0] = acc;
  }
}

static void launch(void* kernel_sel, int clusters, int n, int smem, at::Tensor sink, int a, int b, int c2, int mode) {
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(n * clusters, 1, 1);
  cfg.blockDim = dim3(128);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = n;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaError_t err;
  if (mode == 0) {
    cudaFuncSetAttribute(ring_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    err = cudaLaunchKernelEx(&cfg, ring_kernel, n, a, b, c2, sink.data_ptr<float>());
  } else {
    cudaFuncSetAttribute(push_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    err = cudaLaunchKernelEx(&cfg, push_kernel, n, a, sink.data_ptr<float>());
  }
  TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
}

void ring(at::Tensor sink, int64_t clusters, int64_t n, int64_t rounds, int64_t mode, int64_t waiters) {
  launch(nullptr, clusters, n, 1024, sink, rounds, mode, waiters, 0);
}
void push(at::Tensor sink, int64_t clusters, int64_t n, int64_t bytes) {
  launch(nullptr, clusters, n, 64 + n * bytes, sink, bytes, 0, 0, 1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ring", &ring);
  m.def("push", &push);
}
