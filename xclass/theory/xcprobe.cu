// xcprobe.cu -- GPU implementation of the XC canonical program (xc.h) on mma.sync (all classes) and wgmma (Hopper),
// plus the native chained mma.sync for contrast. Prints one FNV hash per (dataset, program) and dumps raw outputs.
//   nvcc -O2 -std=c++17 -arch=sm_80 xcprobe.cu -o xcprobe            (class A)
//   nvcc -O2 -std=c++17 -arch=sm_90a -DWGMMA xcprobe.cu -o xcprobe   (Hopper)
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "xc.h"
#include "xcdata.h"
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(2); } } while (0)
__device__ __forceinline__ uint32_t pk(uint16_t lo, uint16_t hi) { return (uint32_t)lo | ((uint32_t)hi << 16); }
__device__ __forceinline__ void mma16(float& d0, float& d1, float& d2, float& d3, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// pre-transform: one thread per group of 16 contiguous elements
__global__ void k_slice(const uint16_t* in, uint16_t* s0, uint16_t* s1, size_t ngroups, int G, int s, int* bad) {
  const size_t g = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= ngroups) return;
  uint16_t x[16], a[16], b[16];
  for (int k = 0; k < G; ++k) x[k] = in[g * G + k];
  if (xc_slice(x, G, s, a, b)) atomicAdd(bad, 1);
  for (int k = 0; k < G; ++k) { s0[g * G + k] = a[k]; s1[g * G + k] = b[k]; }
}
// native: chained accumulator in the tensor core (SA = SB = 0) ; XC: zero accumulator per MMA + IEEE fp32 fold
__global__ void k_gemm(const uint16_t* X0, const uint16_t* X1, const uint16_t* W0, const uint16_t* W1, float* out, int T, int SA, int SB, int mode) {
  const int tile = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, k0 = (lane & 3) * 2;
  const int r0 = warp * 16 + g, r1 = r0 + 8;
  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
  for (int t = 0; t < T; ++t) {
    const size_t xo = ((size_t)tile * T + t) * 1024, wo = ((size_t)tile * T + t) * 128;
    if (SA == 0) {
      const uint16_t *x = X0 + xo, *w = W0 + wo;
      mma16(c0, c1, c2, c3, pk(x[r0 * 16 + k0], x[r0 * 16 + k0 + 1]), pk(x[r1 * 16 + k0], x[r1 * 16 + k0 + 1]), pk(x[r0 * 16 + k0 + 8], x[r0 * 16 + k0 + 9]),
            pk(x[r1 * 16 + k0 + 8], x[r1 * 16 + k0 + 9]), pk(w[g * 16 + k0], w[g * 16 + k0 + 1]), pk(w[g * 16 + k0 + 8], w[g * 16 + k0 + 9]));
      continue;
    }
    if (mode == 2) {   // XR: two zero-padded k8 MMAs, each from a zero accumulator
      const uint16_t *x = X0 + xo, *w = W0 + wo;
      const uint32_t a0 = pk(x[r0 * 16 + k0], x[r0 * 16 + k0 + 1]), a1 = pk(x[r1 * 16 + k0], x[r1 * 16 + k0 + 1]), a2 = pk(x[r0 * 16 + k0 + 8], x[r0 * 16 + k0 + 9]),
                     a3 = pk(x[r1 * 16 + k0 + 8], x[r1 * 16 + k0 + 9]), b0 = pk(w[g * 16 + k0], w[g * 16 + k0 + 1]), b1 = pk(w[g * 16 + k0 + 8], w[g * 16 + k0 + 9]);
      for (int h = 0; h < 2; ++h) {
        float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
        if (h == 0) mma16(d0, d1, d2, d3, a0, a1, 0u, 0u, b0, 0u); else mma16(d0, d1, d2, d3, 0u, 0u, a2, a3, 0u, b1);
        c0 = __fadd_rn(c0, d0); c1 = __fadd_rn(c1, d1); c2 = __fadd_rn(c2, d2); c3 = __fadd_rn(c3, d3);
      }
      continue;
    }
    for (int j = 0; j < SA; ++j) for (int l = 0; l < SB; ++l) {
      if (mode == 1 && j == 1 && l == 1) continue;
      const uint16_t *x = (j ? X1 : X0) + xo, *w = (l ? W1 : W0) + wo;
      float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
      mma16(d0, d1, d2, d3, pk(x[r0 * 16 + k0], x[r0 * 16 + k0 + 1]), pk(x[r1 * 16 + k0], x[r1 * 16 + k0 + 1]), pk(x[r0 * 16 + k0 + 8], x[r0 * 16 + k0 + 9]),
            pk(x[r1 * 16 + k0 + 8], x[r1 * 16 + k0 + 9]), pk(w[g * 16 + k0], w[g * 16 + k0 + 1]), pk(w[g * 16 + k0 + 8], w[g * 16 + k0 + 9]));
      c0 = __fadd_rn(c0, d0); c1 = __fadd_rn(c1, d1); c2 = __fadd_rn(c2, d2); c3 = __fadd_rn(c3, d3);
    }
  }
  float* o = out + (size_t)tile * 512;
  o[r0 * 8 + k0] = c0; o[r0 * 8 + k0 + 1] = c1; o[r1 * 8 + k0] = c2; o[r1 * 8 + k0 + 1] = c3;
}
#ifdef WGMMA
__device__ __forceinline__ int smem_off(int r, int k) { return (r >> 3) * 256 + (k >> 3) * 128 + (r & 7) * 16 + (k & 7) * 2; }
__device__ __forceinline__ uint64_t make_desc(const void* p) {
  const uint64_t addr = (uint64_t)__cvta_generic_to_shared(p);
  return ((addr >> 4) & 0x3FFF) | ((uint64_t)(128 >> 4) << 16) | ((uint64_t)(256 >> 4) << 32);
}
__global__ void __launch_bounds__(128) k_gemm_wg(const uint16_t* X0, const uint16_t* X1, const uint16_t* W0, const uint16_t* W1, float* out, int T, int SA, int SB, int mode) {
  __shared__ __align__(128) unsigned char sA[2048];
  __shared__ __align__(128) unsigned char sB[2048];
  const int tile = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, k0 = (lane & 3) * 2;
  const int r0 = warp * 16 + g, r1 = r0 + 8;
  const uint64_t da = make_desc(sA), db = make_desc(sB);
  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
  const int nj = mode == 2 ? 2 : (SA ? SA : 1), nl = mode == 2 ? 1 : (SB ? SB : 1);   // XR: j is the k8 half
  for (int t = 0; t < T; ++t) {
    const size_t xo = ((size_t)tile * T + t) * 1024, wo = ((size_t)tile * T + t) * 128;
    for (int j = 0; j < nj; ++j) for (int l = 0; l < nl; ++l) {
      if (mode == 1 && j == 1 && l == 1) continue;
      const uint16_t *x = ((j && mode != 2) ? X1 : X0) + xo, *w = (l ? W1 : W0) + wo;
      {   // cp.async feed (st.shared feeding was measured to be racy against wgmma on H100, see wgfeed.cu)
        const int r = threadIdx.x >> 1, h = threadIdx.x & 1;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"((unsigned)__cvta_generic_to_shared(sA + (r >> 3) * 256 + h * 128 + (r & 7) * 16)), "l"(x + r * 16 + h * 8) : "memory");
        if (threadIdx.x < 16) asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"((unsigned)__cvta_generic_to_shared(sB + h * 128 + r * 16)), "l"(w + r * 16 + h * 8) : "memory");
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
      }
      __syncthreads();
      float d0 = SA ? 0.f : c0, d1 = SA ? 0.f : c1, d2 = SA ? 0.f : c2, d3 = SA ? 0.f : c3;
      asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
      asm volatile("" : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) :: "memory");
      asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
                   "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, p, 1, 1, 0, 0;\n}\n"
                   : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) : "l"(da), "l"(db), "r"(1));
      asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
      asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
      asm volatile("" : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) :: "memory");
      __syncthreads();
      if (SA) { c0 = __fadd_rn(c0, d0); c1 = __fadd_rn(c1, d1); c2 = __fadd_rn(c2, d2); c3 = __fadd_rn(c3, d3); }
      else { c0 = d0; c1 = d1; c2 = d2; c3 = d3; }
    }
  }
  float* o = out + (size_t)tile * 512;
  o[r0 * 8 + k0] = c0; o[r0 * 8 + k0 + 1] = c1; o[r1 * 8 + k0] = c2; o[r1 * 8 + k0 + 1] = c3;
}
#endif
static uint64_t fnv(const std::vector<uint32_t>& v) {
  uint64_t h = 1469598103934665603ull; const unsigned char* p = (const unsigned char*)v.data();
  for (size_t i = 0; i < v.size() * 4; ++i) h = (h ^ p[i]) * 1099511628211ull;
  return h;
}
int main(int argc, char** argv) {
  const int tiles = argc > 1 ? atoi(argv[1]) : 256, T = argc > 2 ? atoi(argv[2]) : 64;
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  printf("GPU %s sm_%d%d  XC probe tiles %d T %d (K = %d)\n", p.name, p.major, p.minor, tiles, T, 16 * T);
  // programs: {SA, SB, sA, sB}; SA = 0 is the native chained MMA
  const int prog[][5] = {{0, 0, 0, 0, 0}, {1, 1, 2, 2, 0}, {2, 1, 2, 2, 0}, {2, 2, 2, 2, 1}, {2, 2, 2, 2, 0}, {1, 1, 5, 5, 2}};
  FILE* f = fopen("xc_out.bin", "wb");
  for (int which = 0; which < 4; ++which) {
    XData d = xc_make(which, tiles, T);
    const size_t nx = d.X.size(), nw = d.W.size(), nout = (size_t)tiles * 512;
    uint16_t *dX, *dW, *dX0, *dX1, *dW0, *dW1; float* dO; int* dBad;
    CK(cudaMalloc(&dX, nx * 2)); CK(cudaMalloc(&dX0, nx * 2)); CK(cudaMalloc(&dX1, nx * 2));
    CK(cudaMalloc(&dW, nw * 2)); CK(cudaMalloc(&dW0, nw * 2)); CK(cudaMalloc(&dW1, nw * 2)); CK(cudaMalloc(&dO, nout * 4)); CK(cudaMalloc(&dBad, 4));
    CK(cudaMemcpy(dX, d.X.data(), nx * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dW, d.W.data(), nw * 2, cudaMemcpyHostToDevice));
    for (auto& pr : prog) {
      std::vector<uint32_t> o(nout); int bad = 0;
      if (pr[0] == 0) k_gemm<<<tiles, 128>>>(dX, dX, dW, dW, dO, T, 0, 0, 0);
      else {
        CK(cudaMemset(dBad, 0, 4));
        const int G = pr[4] == 2 ? 8 : 16;
        k_slice<<<(unsigned)((nx / G + 255) / 256), 256>>>(dX, dX0, dX1, nx / G, G, pr[2], dBad);
        k_slice<<<(unsigned)((nw / G + 255) / 256), 256>>>(dW, dW0, dW1, nw / G, G, pr[3], dBad);
        k_gemm<<<tiles, 128>>>(dX0, dX1, dW0, dW1, dO, T, pr[0], pr[1], pr[4]);
        CK(cudaDeviceSynchronize()); CK(cudaMemcpy(&bad, dBad, 4, cudaMemcpyDeviceToHost));
      }
      CK(cudaDeviceSynchronize()); CK(cudaMemcpy(o.data(), dO, nout * 4, cudaMemcpyDeviceToHost));
      printf("XCHASH data %d prog SA%d SB%d sA%d sB%d mode%d mma.sync %016llx out-of-domain %d\n", which, pr[0], pr[1], pr[2], pr[3], pr[4], (unsigned long long)fnv(o), bad);
      fwrite(o.data(), 4, nout, f);
#ifdef WGMMA
      for (int rep = 0; rep < 3 && pr[4] != 2; ++rep) {
        std::vector<uint32_t> ow(nout); CK(cudaMemset(dO, 0xFF, nout * 4));
        if (pr[0] == 0) k_gemm_wg<<<tiles, 128>>>(dX, dX, dW, dW, dO, T, 0, 0, 0); else k_gemm_wg<<<tiles, 128>>>(dX0, dX1, dW0, dW1, dO, T, pr[0], pr[1], pr[4]);
        CK(cudaDeviceSynchronize()); CK(cudaMemcpy(ow.data(), dO, nout * 4, cudaMemcpyDeviceToHost));
        size_t mm = 0; for (size_t i = 0; i < nout; ++i) mm += ow[i] != o[i];
        printf("XCHASH data %d prog SA%d SB%d sA%d sB%d mode%d wgmma rep %d %016llx mismatches vs mma.sync %zu\n", which, pr[0], pr[1], pr[2], pr[3], pr[4], rep, (unsigned long long)fnv(ow), mm);
      }
#endif
    }
    cudaFree(dX); cudaFree(dX0); cudaFree(dX1); cudaFree(dW); cudaFree(dW0); cudaFree(dW1); cudaFree(dO); cudaFree(dBad);
  }
  fclose(f);
  return 0;
}
