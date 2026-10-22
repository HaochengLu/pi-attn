#include <torch/extension.h>
#include "gemm_cutlass.cuh"
using namespace pi_gemm;
template <int A, int B, int C, int D, int E, int S>
std::string one() {
  using G = CutlassGemm<__nv_bfloat16, A, B, C, D, E, S>;
  using K = typename G::Gemm::GemmKernel;
  int smem = sizeof(typename K::SharedStorage);
  auto e1 = cudaFuncSetAttribute(streamk_tiles_kernel<typename G::StreamK>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  int r = -1;
  auto e2 = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&r, streamk_tiles_kernel<typename G::StreamK>, K::kThreadCount, smem);
  cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, streamk_tiles_kernel<typename G::StreamK>);
  return std::to_string(A) + "x" + std::to_string(B) + "x" + std::to_string(C) + " s" + std::to_string(S) + " smem " + std::to_string(smem) +
         " regs " + std::to_string(fa.numRegs) + " threads " + std::to_string(K::kThreadCount) + " resident " + std::to_string(r) + " e " + std::to_string(e1) + "/" + std::to_string(e2);
}
std::string run() { return one<32, 64, 64, 16, 32, 5>() + "\n" + one<32, 64, 64, 16, 32, 4>() + "\n" + one<32, 64, 64, 16, 32, 3>() + "\n" + one<64, 64, 64, 32, 32, 4>() + "\n" + one<64, 128, 32, 32, 64, 4>(); }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
