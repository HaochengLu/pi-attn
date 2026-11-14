// Probe (round 4): the canonical GEMM chain with X in registers (wgmma RS: A from registers, B = W from shared
// memory), so that shared memory holds W stages only. Each consumer thread loads its 8 X elements of a k16 step
// straight from global memory (X is small and L2-resident), one K stage ahead of the wgmma that reads them.
// mode 1: no X loads at all (the registers keep a constant): the ceiling of any scheme that keeps X out of
// shared memory. Same chain per element as gemm_sm90.cuh: +0, then the k16 steps in ascending order, S = 1.
#pragma once
#include "gemm_sm90.cuh"

namespace pi_gemm90 {

template <int BM, int BN, int ST, int BK>
struct Sm90GemmRS {
  using BF = cutlass::bfloat16_t;
  using OutT = __nv_bfloat16;
  static constexpr int kBK = BK, kWG = BM / 64, kThreads = 128 * (1 + kWG), kSteps = BK / 16;
  using TileShape = cute::Shape<cute::Int<BM>, cute::Int<BN>, cute::Int<BK>>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, BF, cutlass::layout::RowMajor, 8, BF,
      cutlass::layout::ColumnMajor, 8, float, TileShape, ClusterShape, cutlass::gemm::collective::StageCount<ST>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;  // for W: descriptor and stage layout
  using TiledMma = decltype(cute::make_tiled_mma(
      cute::GMMA::rs_op_selector<BF, BF, float, cute::Shape<cute::_64, cute::Int<BN>, cute::Int<BK>>, cute::GMMA::Major::K,
                                 cute::GMMA::Major::K>(),
      cute::Layout<cute::Shape<cute::Int<kWG>, cute::_1, cute::_1>>{}));
  using Pipeline = typename Mainloop::MainloopPipeline;
  using PipelineState = typename Mainloop::PipelineState;
  struct SharedStorage {
    alignas(128) BF b[cute::cosize_v<typename Mainloop::SmemLayoutB>];
    typename Pipeline::SharedStorage pipeline;
    alignas(128) OutT epi[BM * BN];
  };
  struct Params {
    typename Mainloop::Params mainloop;
    const uint32_t* x;
    int64_t ldx;
    int M, N, K, swz, mode;
    cute::TmaDescriptor tma_y;
  };

  static __device__ __forceinline__ void device(Params const& params, char* smem_buf) {
    using namespace cute;
    SharedStorage& ss = *reinterpret_cast<SharedStorage*>(smem_buf);
    const int thread_idx = threadIdx.x;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const bool producer = cutlass::canonical_warp_group_idx() == 0;
    if (warp_idx == 0 && cute::elect_one_sync()) cute::prefetch_tma_descriptor(params.mainloop.tma_load_b.get_tma_descriptor());
    typename Pipeline::Params pp;
    if (producer && warp_idx == 0) pp.role = Pipeline::ThreadCategory::Producer;
    if (!producer) pp.role = Pipeline::ThreadCategory::Consumer;
    pp.is_leader = (thread_idx % 128) == 0;
    pp.num_consumers = kWG * 128;
    pp.transaction_bytes = BN * BK * 2;
    Pipeline pipeline(ss.pipeline, pp, ClusterShape{});
    __syncthreads();

    const int M = params.M, N = params.N, K = params.K, swz = params.swz;
    const int tiles_m = (M + BM - 1) / BM, tiles_n = (N + BN - 1) / BN, tiles = tiles_m * tiles_n;
    const int k_tile_count = K / BK;
    Tensor sB = make_tensor(make_smem_ptr(&ss.b[0]), typename Mainloop::SmemLayoutB{});

    if (producer) {
      if (warp_idx == 0 && cute::elect_one_sync()) {
        Mainloop mainloop;
        Tensor gB_nkl = get<1>(mainloop.load_init(make_shape(M, N, K, 1), params.mainloop));
        auto tma_b = params.mainloop.tma_load_b.get_slice(0);
        Tensor tBsB = tma_b.partition_D(sB);
        PipelineState write = cutlass::make_producer_start_state<Pipeline>();
        for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
          int mt, nt;
          Sm90Gemm<BM, BN, 2, false, OutT, BK>::tile_of(t, tiles_m, tiles_n, swz, mt, nt);
          Tensor tBgB = tma_b.partition_S(gB_nkl(_, _, nt, _, 0));
          for (int kb = 0; kb < k_tile_count; ++kb) {
            pipeline.producer_acquire(write);
            copy(params.mainloop.tma_load_b.with(*pipeline.producer_get_barrier(write), 0), tBgB(_, _, _, kb),
                 tBsB(_, _, _, write.index()));
            ++write;
          }
        }
        pipeline.producer_tail(write);
      }
      return;
    }

    const int mma_thread = thread_idx - 128;
    TiledMma tiled_mma;
    Tensor accum = partition_fragment_C(tiled_mma, take<0, 2>(TileShape{}));
    Layout wg_layout = make_layout(Int<kWG>{}, Int<128>{});
    const int wg = __shfl_sync(0xFFFFFFFF, mma_thread / 128, 0);
    auto wg_mma = tiled_mma.get_slice(wg_layout(wg));
    Tensor tCsB = wg_mma.partition_B(sB);
    Tensor tCrB = wg_mma.make_fragment_B(tCsB);
    auto thr_mma = tiled_mma.get_thread_slice(mma_thread);
    Tensor tCcD = thr_mma.partition_C(make_identity_tensor(make_shape(Int<BM>{}, Int<BN>{})));
    // (m, k) within the tile's K stage of this thread's A values: (8 values, 1, k16 steps); values 2i, 2i + 1 are
    // one row's adjacent columns = one 32-bit register.
    Tensor tCcA = thr_mma.partition_A(make_identity_tensor(make_shape(Int<BM>{}, Int<BK>{})));
    static_assert(decltype(size<0>(tCcA))::value == 8 && decltype(size<2>(tCcA))::value == kSteps);
    uint32_t cur[kSteps * 4], nxt[kSteps * 4];
    auto frag_layout = make_layout(make_shape(Int<8>{}, Int<1>{}, Int<kSteps>{}));
    Tensor tCrA = make_tensor(make_rmem_ptr<BF>(reinterpret_cast<BF*>(&cur[0])), frag_layout);
    const uint32_t* const x = params.x;
    const int64_t ldx = params.ldx;
    const int mode = params.mode;
    auto load_x = [&](uint32_t* dst, int mt, int kt) {  // the X registers of K stage kt of row tile mt
      CUTE_UNROLL
      for (int kb = 0; kb < kSteps; ++kb) {
        CUTE_UNROLL
        for (int i = 0; i < 4; ++i) {
          auto c = tCcA(2 * i, 0, kb);
          int row = mt * BM + get<0>(c);
          row = row < M ? row : M - 1;
          dst[kb * 4 + i] = *reinterpret_cast<const uint32_t*>(reinterpret_cast<const char*>(x) +
                                                              2 * (row * ldx + (int64_t)kt * BK + get<1>(c)));
        }
      }
    };
    if (mode == 1) {
      CUTE_UNROLL
      for (int i = 0; i < kSteps * 4; ++i) cur[i] = 0x3c003c00u + 0x00010001u * (uint32_t)((thread_idx + i) & 63);
    }
    PipelineState read_state;
    bool stored = false;
    for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
      int mt, nt;
      Sm90Gemm<BM, BN, 2, false, OutT, BK>::tile_of(t, tiles_m, tiles_n, swz, mt, nt);
      if (mode == 0) load_x(cur, mt, 0);
      clear(accum);
      warpgroup_fence_operand(accum);
      CUTLASS_PRAGMA_NO_UNROLL
      for (int kt = 0; kt < k_tile_count; ++kt) {
        auto barrier_token = pipeline.consumer_try_wait(read_state);
        pipeline.consumer_wait(read_state, barrier_token);
        const int read_stage = read_state.index();
        warpgroup_fence_operand(tCrA);
        warpgroup_fence_operand(accum);
        warpgroup_arrive();
        CUTE_UNROLL
        for (int kb = 0; kb < kSteps; ++kb)  // ascending k16 steps
          cute::gemm(tiled_mma, tCrA(_, _, kb), tCrB(_, _, kb, read_stage), accum);
        warpgroup_commit_batch();
        if (mode == 0 && kt + 1 < k_tile_count) load_x(nxt, mt, kt + 1);  // while the steps run
        warpgroup_wait<0>();
        warpgroup_fence_operand(accum);
        warpgroup_fence_operand(tCrA);
        pipeline.consumer_release(read_state);
        ++read_state;
        if (mode == 0 && kt + 1 < k_tile_count) {
          CUTE_UNROLL
          for (int i = 0; i < kSteps * 4; ++i) cur[i] = nxt[i];
        }
      }
      if (mma_thread == 0 && stored) cute::tma_store_wait<0>();
      asm volatile("bar.sync %0, %1;" :: "r"(1), "r"(kWG * 128) : "memory");
      CUTE_UNROLL
      for (int i = 0; i < size(accum); i += 2) {
        auto c = tCcD(i);
        *reinterpret_cast<__nv_bfloat162*>(ss.epi + get<0>(c) * BN + get<1>(c)) = __floats2bfloat162_rn(accum(i), accum(i + 1));
      }
      cute::tma_store_fence();
      asm volatile("bar.sync %0, %1;" :: "r"(1), "r"(kWG * 128) : "memory");
      if (mma_thread == 0) {
        cute::SM90_TMA_STORE_2D::copy(&params.tma_y, ss.epi, nt * BN, mt * BM);
        cute::tma_store_arrive();
      }
      stored = true;
    }
    if (mma_thread == 0 && stored) cute::tma_store_wait<0>();
  }

  template <typename SetSmem>
  static cudaError_t run(const void* x, int64_t ldx, const void* w, OutT* y, int64_t ldy, int M, int N, int K, int mode,
                         int sms, SetSmem&& set_smem, cudaStream_t stream) {
    using Self = Sm90GemmRS<BM, BN, ST, BK>;
    if (K % BK || N % 8 || ldy % 8 || ldx % 8 || ((uintptr_t)x & 15) || ((uintptr_t)w & 15) || ((uintptr_t)y & 15))
      return cudaErrorInvalidValue;
    typename Mainloop::Arguments margs{(const BF*)x, cute::make_stride(int64_t(ldx), cute::Int<1>{}, int64_t(0)),
                                       (const BF*)w, cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0))};
    Params params{Mainloop::to_underlying_arguments(cute::make_shape(M, N, K, 1), margs, nullptr),
                  (const uint32_t*)x, ldx, M, N, K, 1, mode, {}};
    const int smem = (int)sizeof(SharedStorage);
    if (smem > 232448) return cudaErrorInvalidValue;
    const int tiles_m = (M + BM - 1) / BM, tiles_n = (N + BN - 1) / BN;
    const int grid = std::min(tiles_m * tiles_n, sms);
    params.swz = std::max(1, std::min(tiles_n, (int)std::lround(std::sqrt((double)grid * BM / BN))));
    const cuuint64_t gdim[2] = {(cuuint64_t)N, (cuuint64_t)M}, gstride[1] = {(cuuint64_t)ldy * sizeof(OutT)};
    const cuuint32_t box[2] = {BN, BM}, estride[2] = {1, 1};
    if (cuTensorMapEncodeTiled(&params.tma_y, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, y, gdim, gstride, box, estride,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
      return cudaErrorInvalidValue;
    set_smem(sm90_gemm_kernel<Self>, (size_t)smem);
    sm90_gemm_kernel<Self><<<grid, kThreads, smem, stream>>>(params);
    return cudaGetLastError();
  }
};

}  // namespace pi_gemm90
