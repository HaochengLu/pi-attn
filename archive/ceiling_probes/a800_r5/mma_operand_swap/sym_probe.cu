// A800 round 5: is mma.sync m16n8k16 (bf16 x bf16 + f32) symmetric in its operands on sm_80?
// D[i][j] of (A = X, B = Y, C) against D'[j][i] of (A = Y, B = X, C^T), bit for bit, on single steps and on
// chains of `steps` k16 steps (the accumulator carried through, as S = Q K^T over head_dim and o += P V do).
// x [n][steps][16][16] bf16 (16 rows of A), y [n][steps][8][16] bf16 (8 columns of B), c [n][16][8] f32.
// out0 [n][16][8]: A = X, B = Y.  out1 [n][16][8], written transposed back: A = Y (rows 8..15 zero), B = X rows
// 0..7 and, in a second mma, X rows 8..15, so all 128 elements of D are compared.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda_bf16.h>

__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__global__ void sym_kernel(const __nv_bfloat16* x, const __nv_bfloat16* y, const float* c, float* out0, float* out1,
                           int steps) {
  const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
  x += (size_t)blockIdx.x * steps * 256, y += (size_t)blockIdx.x * steps * 128, c += (size_t)blockIdx.x * 128;
  out0 += (size_t)blockIdx.x * 128, out1 += (size_t)blockIdx.x * 128;
  auto w = [](const __nv_bfloat16* p) { return *reinterpret_cast<const uint32_t*>(p); };
  {
    float d[4] = {c[g * 8 + 2 * t], c[g * 8 + 2 * t + 1], c[(g + 8) * 8 + 2 * t], c[(g + 8) * 8 + 2 * t + 1]};
    for (int s = 0; s < steps; ++s) {
      const __nv_bfloat16* xs = x + s * 256;
      const __nv_bfloat16* ys = y + s * 128;
      uint32_t a[4] = {w(xs + g * 16 + 2 * t), w(xs + (g + 8) * 16 + 2 * t), w(xs + g * 16 + 8 + 2 * t), w(xs + (g + 8) * 16 + 8 + 2 * t)};
      mma(d, a, w(ys + g * 16 + 2 * t), w(ys + g * 16 + 8 + 2 * t));
    }
    out0[g * 8 + 2 * t] = d[0], out0[g * 8 + 2 * t + 1] = d[1];
    out0[(g + 8) * 8 + 2 * t] = d[2], out0[(g + 8) * 8 + 2 * t + 1] = d[3];
  }
  for (int h = 0; h < 2; ++h) {  // X rows 8 h .. 8 h + 7 as the 8 columns of B
    // D'[m = y row g][n = x row 8 h + 2 t (+1)] starts from C[8 h + 2 t (+1)][g]
    float d[4] = {c[(8 * h + 2 * t) * 8 + g], c[(8 * h + 2 * t + 1) * 8 + g], 0.f, 0.f};
    for (int s = 0; s < steps; ++s) {
      const __nv_bfloat16* xs = x + s * 256 + h * 128;
      const __nv_bfloat16* ys = y + s * 128;
      uint32_t a[4] = {w(ys + g * 16 + 2 * t), 0u, w(ys + g * 16 + 8 + 2 * t), 0u};
      mma(d, a, w(xs + g * 16 + 2 * t), w(xs + g * 16 + 8 + 2 * t));
    }
    out1[(8 * h + 2 * t) * 8 + g] = d[0], out1[(8 * h + 2 * t + 1) * 8 + g] = d[1];
  }
}
void sym(at::Tensor x, at::Tensor y, at::Tensor c, at::Tensor out0, at::Tensor out1, int64_t steps) {
  auto stream = at::cuda::getCurrentCUDAStream();
  sym_kernel<<<c.size(0), 32, 0, stream>>>((const __nv_bfloat16*)x.data_ptr(), (const __nv_bfloat16*)y.data_ptr(),
                                           c.data_ptr<float>(), out0.data_ptr<float>(), out1.data_ptr<float>(), (int)steps);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("sym", &sym); }
