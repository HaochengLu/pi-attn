// Minimal wgmma kernels (one warp group, 128 threads). -DMODE=.. -DZS=0/1 -DACC_T=0/1
#include <cuda_bf16.h>
#include <cstdint>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/barrier.h>
#ifndef MODE
#define MODE 0
#endif
#ifndef ZS
#define ZS 0
#endif
#ifndef ACC_T
#define ACC_T 0
#endif
#ifndef NREG
#define NREG 0
#endif
namespace mn {
using namespace cute;
using BF = cutlass::bfloat16_t;
using TiledMmaQK = decltype(make_tiled_mma(GMMA::ss_op_selector<BF, BF, float, Shape<_64, _64, _128>>()));
using TiledMmaS32 = decltype(make_tiled_mma(GMMA::ss_op_selector<BF, BF, float, Shape<_64, _32, _128>>()));
using TiledMmaPV = decltype(make_tiled_mma(
    GMMA::rs_op_selector<BF, BF, float, Shape<_64, _128, _32>, GMMA::Major::K, GMMA::Major::MN>()));
using AtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, BF, _64, _128>());
using LQ = decltype(tile_to_shape(AtomQ{}, Shape<_64, _128>{}));
using LK = decltype(tile_to_shape(AtomQ{}, Shape<_64, _128, _4>{}));
using LVt = decltype(composition(LK{}, make_ordered_layout(Shape<_128, _64, _4>{}, Step<_2, _1, _3>{})));
struct Smem {
  array_aligned<BF, cosize_v<LQ>> q;
  array_aligned<BF, cosize_v<LK>> k;
  array_aligned<BF, cosize_v<LK>> v;
};
template <typename Layout>
__device__ __forceinline__ auto acc_to_aregs(Layout acc_layout) {
  using X = Underscore;
  auto l = logical_divide(get<0>(acc_layout), Shape<X, X, _2>{});
  return make_layout(make_layout(get<0>(l), get<1>(l), get<2, 0>(l)), get<1>(acc_layout),
                     make_layout(get<2, 1>(l), get<2>(acc_layout)));
}
__device__ __forceinline__ void qk_issue(Smem& sm, int ct, float (&s)[8][4], int stage) {
  Tensor sQ = make_tensor(make_smem_ptr(sm.q.data()), LQ{});
  Tensor sK = make_tensor(make_smem_ptr(sm.k.data()), LK{});
  TiledMmaQK mma_qk;
  auto tqk = mma_qk.get_thread_slice(ct);
  Tensor tSrQ = tqk.partition_fragment_A(sQ);
  Tensor tSrK = tqk.partition_fragment_B(sK);
  auto lay = partition_fragment_C(mma_qk, Shape<_64, _64>{}).layout();
  Tensor tS = make_tensor(make_rmem_ptr<float>(&s[0][0]), lay);
#if ZS
  mma_qk.accumulate_ = GMMA::ScaleOut::Zero;
#else
  CUTE_UNROLL
  for (int i = 0; i < 32; ++i) tS(i) = 0.f;
#endif
  warpgroup_fence_operand(tS);
  warpgroup_arrive();
  CUTE_UNROLL
  for (int kb = 0; kb < size<2>(tSrQ); ++kb) {
    cute::gemm(mma_qk, tSrQ(_, _, kb), tSrK(_, _, kb, stage), tS);
    mma_qk.accumulate_ = GMMA::ScaleOut::One;
  }
  warpgroup_commit_batch();
}
__device__ __forceinline__ void pv_issue(Smem& sm, int ct, uint32_t (&pa)[2][4], float (&o)[16][4], int stage, int j) {
  Tensor sVt = make_tensor(make_smem_ptr(sm.v.data()), LVt{});
  TiledMmaS32 mma_s;
  TiledMmaPV mma_pv;
  auto tpv = mma_pv.get_thread_slice(ct);
  Tensor tOrV = tpv.partition_fragment_B(sVt);
  auto lay_s = partition_fragment_C(mma_s, Shape<_64, _32>{}).layout();
  auto lay_o = partition_fragment_C(mma_pv, Shape<_64, _128>{}).layout();
  Tensor tOrP = make_tensor(make_rmem_ptr<BF>(reinterpret_cast<BF*>(&pa[0][0])), acc_to_aregs(lay_s));
  Tensor tOrO = make_tensor(make_rmem_ptr<float>(&o[0][0]), lay_o);
  warpgroup_fence_operand(tOrP);
  warpgroup_fence_operand(tOrO);
  warpgroup_arrive();
  CUTE_UNROLL
  for (int kb = 0; kb < size<2>(tOrP); ++kb) cute::gemm(mma_pv, tOrP(_, _, kb), tOrV(_, _, 2 * j + kb, stage), tOrO);
  warpgroup_commit_batch();
}
template <int N>
__device__ __forceinline__ void fence(float (&x)[N][4]) {
#pragma unroll
  for (int n = 0; n < N; ++n)
#pragma unroll
    for (int i = 0; i < 4; ++i) warpgroup_fence_operand(x[n][i]);
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ void make_p(float (*v)[4], uint32_t (&pa)[2][4]) {
#pragma unroll
  for (int n = 0; n < 4; ++n) {
    pa[n >> 1][(n & 1) * 2 + 0] = pack_bf16(v[n][0], v[n][1]);
    pa[n >> 1][(n & 1) * 2 + 1] = pack_bf16(v[n][2], v[n][3]);
  }
}
template <int N>
__device__ __forceinline__ float sum(float (&x)[N][4]) {
  float a = 0.f;
#pragma unroll
  for (int n = 0; n < N; ++n)
#pragma unroll
    for (int i = 0; i < 4; ++i) a += x[n][i];
  return a;
}

#ifndef XL
#define XL 0
#endif
#ifndef LB
#define LB 128
#endif
__global__ void __launch_bounds__(LB) mini(float* out, const float* in, int n) {
#if XL
  float xl[XL];
#pragma unroll
  for (int i = 0; i < XL; ++i) xl[i] = in[threadIdx.x * XL + i];
#endif
  extern __shared__ char buf[];
  Smem& sm = *reinterpret_cast<Smem*>(buf);
  const int ct = threadIdx.x;
#if NREG
  cutlass::arch::warpgroup_reg_alloc<NREG>();
#endif
  float acc = 0.f;
#if MODE == 0 && ACC_T
  // CUTLASS / FlashAttention-3 style: the accumulator is a partition_fragment_C tensor
  Tensor sQ = make_tensor(make_smem_ptr(sm.q.data()), LQ{});
  Tensor sK = make_tensor(make_smem_ptr(sm.k.data()), LK{});
  TiledMmaQK mma_qk;
  auto tqk = mma_qk.get_thread_slice(ct);
  Tensor tSrQ = tqk.partition_fragment_A(sQ);
  Tensor tSrK = tqk.partition_fragment_B(sK);
  for (int it = 0; it < n; ++it) {
    __syncthreads();
    Tensor tS = partition_fragment_C(mma_qk, Shape<_64, _64>{});
    warpgroup_fence_operand(tS);
    warpgroup_arrive();
    mma_qk.accumulate_ = GMMA::ScaleOut::Zero;
    CUTE_UNROLL
    for (int kb = 0; kb < size<2>(tSrQ); ++kb) {
      cute::gemm(mma_qk, tSrQ(_, _, kb), tSrK(_, _, kb, it & 3), tS);
      mma_qk.accumulate_ = GMMA::ScaleOut::One;
    }
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tS);
    CUTE_UNROLL
    for (int i = 0; i < 32; ++i) acc += tS(i);
  }
#elif MODE == 0
  // S only, one accumulator, issue + wait per iteration
  float sa[8][4];
  for (int it = 0; it < n; ++it) {
    __syncthreads();
    qk_issue(sm, ct, sa, it & 3);
    warpgroup_wait<0>();
    fence(sa);
    acc += sum(sa);
  }
#elif MODE == 1
  // S only, alternating accumulators: S(it + 1) in flight while S(it) is consumed
  float sa[8][4], sb[8][4];
  qk_issue(sm, ct, sa, 0);
  warpgroup_wait<0>();
  fence(sa);
  auto step = [&](int it, float (&s)[8][4], float (&sn)[8][4]) {
    __syncthreads();
    qk_issue(sm, ct, sn, (it + 1) & 3);
    acc += sum(s);
    warpgroup_wait<0>();
    fence(sn);
  };
  for (int it = 0; it < n; it += 2) {
    step(it, sa, sb);
    if (it + 1 < n) step(it + 1, sb, sa);
  }
#elif MODE == 2
  // S then P V, each awaited
  float sa[8][4], o[16][4];
  uint32_t pa[2][4];
#pragma unroll
  for (int i = 0; i < 16; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  for (int it = 0; it < n; ++it) {
    __syncthreads();
    qk_issue(sm, ct, sa, it & 3);
    warpgroup_wait<0>();
    fence(sa);
    make_p(&sa[0], pa);
    pv_issue(sm, ct, pa, o, it & 3, 0);
    warpgroup_wait<0>();
    fence(o);
  }
  acc = sum(o);
#elif MODE == 3
  // prefill90_kernel's overlap: S(it + 1) issued, P(it) built on the CUDA cores, P V(it) issued, one wait
  float sa[8][4], sb[8][4], o[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  qk_issue(sm, ct, sa, 0);
  warpgroup_wait<0>();
  fence(sa);
  auto step = [&](int it, float (&s)[8][4], float (&sn)[8][4]) {
    __syncthreads();
    qk_issue(sm, ct, sn, (it + 1) & 3);
    uint32_t pa[2][4];
    make_p(&s[0], pa);
    pv_issue(sm, ct, pa, o, it & 3, 0);
    warpgroup_wait<0>();
    fence(sn);
    fence(o);
#if XL
    if ((it & 15) == 15) {
#pragma unroll
      for (int i = 0; i < XL; ++i) xl[i] = xl[i] * 0.5f + o[i & 15][(i >> 4) & 3];
    }
#endif
  };
  for (int it = 0; it < n; it += 2) {
    step(it, sa, sb);
    if (it + 1 < n) step(it + 1, sb, sa);
  }
  acc = sum(o);
#elif MODE == 4
  // P V only
  float o[16][4];
  uint32_t pa[2][4];
#pragma unroll
  for (int i = 0; i < 16; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  for (int it = 0; it < n; ++it) {
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 8; ++i) pa[i >> 2][i & 3] = (uint32_t)(it * 77 + i + ct);
    pv_issue(sm, ct, pa, o, it & 3, 0);
    warpgroup_wait<0>();
    fence(o);
  }
  acc = sum(o);
#endif
#if XL
#pragma unroll
  for (int i = 0; i < XL; ++i) acc += xl[i];
#endif
  out[ct] = acc;
}
}  // namespace mn
