// sm_90a (Hopper) kernels of the canonical GEMM program (gemm_inv.cuh), with wgmma and TMA. Compiled only
// when the build targets sm_90a (PI_SM90A; gemm_dispatch.cuh registers the ids on compute capability 9.0).
//   y[m, n] = round((((+0 + p_0) + p_1) + ...) + p_{S-1}),  p_s = one fp32 chain from +0 over the segment's
//   k16 steps in ascending order: wgmma.m64nNk16 bf16 -> fp32 with scale_d = 1 on every step.
// On Hopper one wgmma k16 step returns the bits of mma.sync.m16n8k16 for every element (measured; the
// kernels here are checked bitwise against the mma.sync ids by kernels/test_gemm.py on the device).
// From CUTLASS 3.x / 4.x: the sm90 TMA / GMMA warp-specialized collective for shared storage, the TMA
// descriptors and the producer (load). The consumer loop and the epilogue are ours:
//  - the first step accumulates into an explicit +0 (CUTLASS's mma() issues it with ScaleOut::Zero, D = A B
//    instead of D = +0 + A B);
//  - the chain restarts and the segment totals fold with __fadd_rn at the canonical segment boundaries;
//  - no split-K or stream-K: every (m tile, n tile) is computed whole by one CTA. CTAs are persistent, CTA b
//    computes tiles b, b + gridDim.x, ... of a fixed walk (which CTA computes an element is free in the
//    contract); the producer streams the next tile's K stages while the consumers run the epilogue, which
//    writes the rounded tile to shared memory and hands it to one asynchronous TMA store. The walk goes in
//    blocks `swz` n tiles wide so that one wave of CTAs shares its X rows and W columns in L2.
// One CTA = 1 producer warp group + BM / 64 consumer warp groups (64 rows x BN columns each).
// A launch refuses (cudaErrorInvalidValue, before launching) x, w, y that are not 16-byte aligned, row
// strides that are not multiples of 8 elements (TMA), a K remainder, and a segment count its kind does not run.
#pragma once
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/pipeline/pipeline.hpp>

namespace pi_gemm90 {

// SEG: S > 1 kernels keep the segment totals in a second register fragment (BN <= 128).
template <int BM, int BN, int ST, bool SEG, typename OutT>
struct Sm90Gemm {
  static_assert(BM == 64 || BM == 128, "one or two consumer warp groups");
  static constexpr int kBK = 64;
  using TileShape = cute::Shape<cute::Int<BM>, cute::Int<BN>, cute::Int<kBK>>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using Schedule = std::conditional_t<BM == 128, cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                                      cutlass::gemm::KernelTmaWarpSpecialized>;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::bfloat16_t, cutlass::layout::RowMajor, 8,     // A = X [M, K], K contiguous
      cutlass::bfloat16_t, cutlass::layout::ColumnMajor, 8,  // B = W^T [K, N] column-major = W [N, K] row-major
      float, TileShape, ClusterShape, cutlass::gemm::collective::StageCount<ST>, Schedule>::CollectiveOp;
  using TiledMma = typename Mainloop::TiledMma;
  using Pipeline = typename Mainloop::MainloopPipeline;
  using PipelineState = typename Mainloop::PipelineState;
  struct SharedStorage {
    typename Mainloop::SharedStorage main;
    alignas(128) OutT epi[BM * BN];  // the output tile, row-major
  };
  static constexpr int kWG = BM / 64;
  static constexpr int kThreads = 128 * (1 + kWG);

  struct Params {
    typename Mainloop::Params mainloop;
    int M, N, K, segs, swz;
    cute::TmaDescriptor tma_y;  // [M][N] of OutT, box BM x BN, no swizzle
  };

  static __device__ __forceinline__ void store2(OutT* p, float a, float b) {
    if constexpr (sizeof(OutT) == 2) *reinterpret_cast<__nv_bfloat162*>(p) = __floats2bfloat162_rn(a, b);
    else *reinterpret_cast<float2*>(p) = make_float2(a, b);
  }

  // Tile t of the walk -> (m tile, n tile): blocks of swz n tiles (the last block takes the remainder), within
  // a block m-major. A bijection of [0, tiles_m * tiles_n).
  static __device__ __forceinline__ void tile_of(int t, int tiles_m, int tiles_n, int swz, int& mt, int& nt) {
    const int per = tiles_m * swz, full = tiles_n / swz;
    if (t < full * per) {
      const int r = t % per;
      mt = r / swz;
      nt = (t / per) * swz + r % swz;
    } else {
      const int rem = tiles_n - full * swz, r = t - full * per;
      mt = r / rem;
      nt = full * swz + r % rem;
    }
  }

  static __device__ __forceinline__ void device(Params const& params, char* smem_buf) {
    using namespace cute;
    SharedStorage& ss = *reinterpret_cast<SharedStorage*>(smem_buf);
    const int thread_idx = threadIdx.x, lane_idx = thread_idx % 32;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const bool producer = cutlass::canonical_warp_group_idx() == 0;
    if (warp_idx == 0 && cute::elect_one_sync()) Mainloop::prefetch_tma_descriptors(params.mainloop);

    typename Pipeline::Params pp;
    if (producer && warp_idx == 0) pp.role = Pipeline::ThreadCategory::Producer;
    if (!producer) pp.role = Pipeline::ThreadCategory::Consumer;
    pp.is_leader = (thread_idx % 128) == 0;
    pp.num_consumers = kWG * 128;
    pp.transaction_bytes = params.mainloop.tma_transaction_bytes;
    Pipeline pipeline(ss.main.pipeline, pp, ClusterShape{});
    __syncthreads();

    const int M = params.M, N = params.N, K = params.K;
    Mainloop mainloop;
    auto load_inputs = mainloop.load_init(make_shape(M, N, K, 1), params.mainloop);
    Tensor gA_mkl = get<0>(load_inputs);
    const int tiles_m = (M + BM - 1) / BM, tiles_n = (N + BN - 1) / BN, swz = params.swz;
    const int tiles = tiles_m * tiles_n;
    auto k_tile_iter = cute::make_coord_iterator(shape<3>(gA_mkl));
    const int k_tile_count = K / kBK;

    if (producer) {
      if (warp_idx == 0) {  // one stream of K tiles in ascending order per output tile, segments included
        PipelineState write_state = cutlass::make_producer_start_state<Pipeline>();
        for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
          int mt, nt;
          tile_of(t, tiles_m, tiles_n, swz, mt, nt);
          auto blk_coord = make_coord(mt, nt, _, 0);
          mainloop.load(params.mainloop, pipeline, write_state, load_inputs, blk_coord, k_tile_iter, k_tile_count,
                        lane_idx, 0, ss.main.tensors);
          write_state.advance(k_tile_count);
        }
        mainloop.load_tail(pipeline, write_state);
      }
      return;
    }

    // ---- consumer: the canonical chain ----
    const int mma_thread = thread_idx - 128;
    TiledMma tiled_mma;  // accumulate_ == ScaleOut::One: every step is D = D + A * B
    Tensor accum = partition_fragment_C(tiled_mma, take<0, 2>(TileShape{}));
    Tensor sA = make_tensor(make_smem_ptr(ss.main.tensors.smem_A.data()), typename Mainloop::SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(ss.main.tensors.smem_B.data()), typename Mainloop::SmemLayoutB{});
    Layout wg_layout = make_layout(Int<kWG>{}, Int<128>{});
    const int wg = __shfl_sync(0xFFFFFFFF, mma_thread / 128, 0);
    auto wg_mma = tiled_mma.get_slice(wg_layout(wg));
    Tensor tCsA = wg_mma.partition_A(sA);
    Tensor tCsB = wg_mma.partition_B(sB);
    Tensor tCrA = wg_mma.make_fragment_A(tCsA);
    Tensor tCrB = wg_mma.make_fragment_B(tCsB);
    static_assert(decltype(size<2>(tCrA))::value == kBK / 16, "k16 steps per stage");

    const int segs = SEG ? params.segs : 1, seg_tiles = k_tile_count / segs;
    float total[SEG ? decltype(size(accum))::value : 1];
    auto thr_mma = tiled_mma.get_thread_slice(mma_thread);
    Tensor tCcD = thr_mma.partition_C(make_identity_tensor(make_shape(Int<BM>{}, Int<BN>{})));  // tile-relative (m, n)
    PipelineState read_state, release_state;
    bool stored = false;
    for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
      int mt, nt;
      tile_of(t, tiles_m, tiles_n, swz, mt, nt);
      if constexpr (SEG) {
        CUTE_UNROLL
        for (int i = 0; i < size(accum); ++i) total[i] = 0.f;
      }
      for (int s = 0; s < segs; ++s) {
        clear(accum);  // +0
        warpgroup_fence_operand(accum);
        bool pending = false;
        CUTLASS_PRAGMA_NO_UNROLL
        for (int kt = 0; kt < seg_tiles; ++kt) {
          auto barrier_token = pipeline.consumer_try_wait(read_state);
          pipeline.consumer_wait(read_state, barrier_token);
          const int read_stage = read_state.index();
          warpgroup_fence_operand(accum);
          warpgroup_arrive();
          CUTE_UNROLL
          for (int k_block = 0; k_block < size<2>(tCrA); ++k_block)  // ascending k16 steps
            cute::gemm(tiled_mma, tCrA(_, _, k_block, read_stage), tCrB(_, _, k_block, read_stage), accum);
          warpgroup_commit_batch();
          if (pending) {  // the previous stage's steps are done: free its buffer
            warpgroup_wait<1>();
            warpgroup_fence_operand(accum);
            pipeline.consumer_release(release_state);
            ++release_state;
          }
          pending = true;
          ++read_state;
        }
        warpgroup_wait<0>();
        warpgroup_fence_operand(accum);
        pipeline.consumer_release(release_state);
        ++release_state;
        if constexpr (SEG) {
          CUTE_UNROLL
          for (int i = 0; i < size(accum); ++i) total[i] = __fadd_rn(total[i], accum(i));
        }
      }

      // ---- epilogue: round to nearest even; elements i, i + 1 are one row's adjacent columns (n even) ----
      // The previous tile's TMA store has read epi (issued a whole mainloop ago; the wait is a formality).
      if (mma_thread == 0 && stored) cute::tma_store_wait<0>();
      asm volatile("bar.sync %0, %1;" :: "r"(1), "r"(kWG * 128) : "memory");  // the consumers only
      CUTE_UNROLL
      for (int i = 0; i < size(accum); i += 2) {
        auto c = tCcD(i);
        OutT* p = ss.epi + get<0>(c) * BN + get<1>(c);
        if constexpr (SEG) store2(p, total[i], total[i + 1]);
        else store2(p, accum(i), accum(i + 1));
      }
      cute::tma_store_fence();
      asm volatile("bar.sync %0, %1;" :: "r"(1), "r"(kWG * 128) : "memory");
      if (mma_thread == 0) {  // out-of-range rows / columns of an edge tile are not written
        cute::SM90_TMA_STORE_2D::copy(&params.tma_y, ss.epi, nt * BN, mt * BM);
        cute::tma_store_arrive();
      }
      stored = true;
    }
    if (mma_thread == 0 && stored) cute::tma_store_wait<0>();
  }

  template <typename SetSmem>
  static cudaError_t run(const void* x, int64_t ldx, const void* w, OutT* y, int64_t ldy, int M, int N, int K,
                         int segs, int sms, SetSmem&& set_smem, cudaStream_t stream);
};

// The TMA descriptors live in the kernel parameters: they must be __grid_constant__ (a thread-local copy of
// a descriptor is not a legal TMA operand).
template <class G>
__global__ void __launch_bounds__(G::kThreads) sm90_gemm_kernel(CUTLASS_GRID_CONSTANT typename G::Params const params) {
  extern __shared__ char smem_buf[];
  G::device(params, smem_buf);
}

// set_smem(kernel, bytes): the caller's once-per-device cudaFuncSetAttribute (gemm_dispatch.cuh).
template <int BM, int BN, int ST, bool SEG, typename OutT>
template <typename SetSmem>
cudaError_t Sm90Gemm<BM, BN, ST, SEG, OutT>::run(const void* x, int64_t ldx, const void* w, OutT* y, int64_t ldy,
                                                 int M, int N, int K, int segs, int sms, SetSmem&& set_smem,
                                                 cudaStream_t stream) {
  using Self = Sm90Gemm<BM, BN, ST, SEG, OutT>;
  if (segs < 1 || K % (kBK * segs) || (!SEG && segs != 1) || (SEG && segs < 2) || N % 8 || ldy % 8 || ldx % 8 ||
      ((uintptr_t)x & 15) || ((uintptr_t)w & 15) || ((uintptr_t)y & 15) || sms < 1)
    return cudaErrorInvalidValue;
  using EA = cutlass::bfloat16_t;
  typename Mainloop::Arguments margs{(const EA*)x, cute::make_stride(int64_t(ldx), cute::Int<1>{}, int64_t(0)),
                                     (const EA*)w, cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0))};
  Params params{Mainloop::to_underlying_arguments(cute::make_shape(M, N, K, 1), margs, nullptr), M, N, K, segs, 1, {}};
  const int smem = (int)sizeof(SharedStorage);
  const int tiles_m = (M + BM - 1) / BM, tiles_n = (N + BN - 1) / BN;
  // Persistent CTAs: as many as are resident at once (shared memory and threads per SM); more would only queue.
  const int per_sm = std::max(1, std::min((228 << 10) / smem, 2048 / kThreads));
  const int grid = std::min(tiles_m * tiles_n, sms * per_sm);
  // One wave of CTAs = (grid / swz) m tiles x swz n tiles; its X rows + W columns are fewest at BM a = BN b.
  params.swz = std::max(1, std::min(tiles_n, (int)std::lround(std::sqrt((double)grid * BM / BN))));
  const cuuint64_t gdim[2] = {(cuuint64_t)N, (cuuint64_t)M}, gstride[1] = {(cuuint64_t)ldy * sizeof(OutT)};
  const cuuint32_t box[2] = {BN, BM}, estride[2] = {1, 1};
  if (cuTensorMapEncodeTiled(&params.tma_y, sizeof(OutT) == 2 ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 : CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                             2, y, gdim, gstride, box, estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                             CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
    return cudaErrorInvalidValue;
  set_smem(sm90_gemm_kernel<Self>, (size_t)smem);
  sm90_gemm_kernel<Self><<<grid, kThreads, smem, stream>>>(params);
  return cudaGetLastError();
}

// Ids 90..94: S = 1 tiles 128x256, 128x128, 64x128, 64x256, 64x64; 95..97: S > 1 tiles 128x128, 64x128, 64x64.
// Stage counts leave room for the epilogue tile (2 bytes per element for bf16 output, 4 for fp32).
inline bool is_sm90_kernel(int k) { return k >= 90 && k <= 97; }
template <typename OutT, typename SetSmem>
cudaError_t run_gemm_sm90(int k, const void* x, int64_t ldx, const void* w, OutT* y, int64_t ldy, int M, int N, int K,
                          int segs, int sms, SetSmem&& ss, cudaStream_t st) {
  constexpr bool h = sizeof(OutT) == 2;
  switch (k) {
    case 90: return Sm90Gemm<128, 256, h ? 3 : 2, false, OutT>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
    case 91: return Sm90Gemm<128, 128, h ? 5 : 4, false, OutT>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
    case 92: return Sm90Gemm<64, 128, 3, false, OutT>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
    case 93: return Sm90Gemm<64, 256, h ? 4 : 3, false, OutT>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
    case 94: return Sm90Gemm<64, 64, 4, false, OutT>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
    case 95: return Sm90Gemm<128, 128, h ? 5 : 4, true, OutT>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
    case 96: return Sm90Gemm<64, 128, 3, true, OutT>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
    case 97: return Sm90Gemm<64, 64, 4, true, OutT>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
    default: return cudaErrorInvalidValue;
  }
}

}  // namespace pi_gemm90
