// Cluster dispatch probe: how long a grid of `clusters` clusters of `n` CTAs takes when each CTA does nothing,
// spins, or runs cluster barriers.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cooperative_groups.h>

__global__ void __launch_bounds__(128) probe_kernel(int spin, int syncs, float* sink) {
  float x = threadIdx.x;
  if (syncs) {
    auto cl = cooperative_groups::this_cluster();
    for (int i = 0; i < syncs; ++i) cl.sync();
  }
  for (int i = 0; i < spin; ++i) x = __fmaf_rn(x, 1.0001f, 1.f);
  if (x == -1.f) sink[0] = x;
}

void probe(at::Tensor sink, int64_t clusters, int64_t n, int64_t spin, int64_t syncs, int64_t smem, int64_t ydim) {
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaFuncSetAttribute(probe_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  if (n > 8) cudaFuncSetAttribute(probe_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  cudaLaunchConfig_t cfg = {};
  const int cn = n > 0 ? n : 1;
  cfg.gridDim = dim3(cn * (clusters / ydim), ydim, 1);
  cfg.blockDim = dim3(128);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = cn;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = n > 0 ? 1 : 0;
  const cudaError_t err = cudaLaunchKernelEx(&cfg, probe_kernel, (int)spin, (int)syncs, sink.data_ptr<float>());
  TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("probe", &probe); }
