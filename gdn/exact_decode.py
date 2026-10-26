"""Exact GDN decode: each decoded token's output is, bitwise, the row that fla 0.5.0's chunk_gated_delta_rule
(use_qk_l2norm_in_kernel=True, its default use_exp2 path) produces for that token in a full training forward,
for one token per step or for a window of several (speculative verification), with free rollback.

State per (slot, value head). The fp32 state is kept only at absolute 64-token boundaries (S), with the bf16 copy fla's
fwd_h feeds to every intra-chunk dot (Hb). Every intermediate of fla's chunk program is row-causal inside a chunk, so
the current chunk's rows are cached as fla materialises them between its kernels:
  kbuf, vbuf, gbuf, bbuf  the chunk's inputs (k after l2norm; g raw; beta in its own dtype)
  qn                      l2norm(q) rows          sg   cumsum(g) * log2(e) (chunk_local_cumsum)
  AiF                     (I+A)^-1 rows in fp32: diagonal 16x16 blocks before the +I (as fla's forward substitution
                          holds them), off-diagonal blocks final; only the blocks later rows read are stored
  Abf                     (I+A)^-1 rows in bf16 (fla's stored A)   Uc, Wc, VNc   u, w, v_new rows (bf16)
Rows at or beyond the window's end hold garbage (earlier chunks, rejected drafts); every consumer masks them, and all
cached values are finite, so they contribute exact zeros. This invariant is also the rollback.

A step covers chunk rows r0 .. r0 + nrows - 1 (nrows <= KT, never past the chunk's 64th row):
  K0 (windows of several tokens only; one program per V block x slot x head, exits at once unless pending): if the
     previous step's accepted tokens completed a chunk, fold it into S/Hb (fwd_h's 64-row arithmetic from the cached
     u, w rows) before this window overwrites its rows. Plain decode accepts every token and folds in K2 instead.
  K1 (one program per slot x head, never reads the state), row by row: l2norm of q, k (fla l2norm_fwd_kernel), the
     chunk-buffer row, cumsum(g), the row of (I+A)^-1 (kkt tiles, the single forward-substitution iteration for that
     row, block merge from cached final blocks) and the rows of u, w (recompute_w_u) on the 16-row tile of the row's
     sub-block.
  K2 (one program per V block x slot x head): v_new = u - w bf16(S) for the window's rows (fwd_h), then their output
     rows (chunk_fwd_kernel_o), on one tile of 16 rows (plain decode) or 32 rows (a window of up to 16 tokens starting
     anywhere in a sub-block); bf16(S) is read once per window. Plain decode then folds a completed chunk.
fla's fused_recurrent reads and writes the fp32 state every token; here the state is written once per 64 tokens.

Why bitwise: every expression mirrors the fla source it is named after, with the same operand dtypes, the same
tl.dot shapes per row (a row of an MMA tile does not depend on the other rows) and the same accumulation forms.
Pinned on the way: fla's chunk_fwd_kernel_o epilogue is compiled with an fma contraction, written out here; and the
token's rows reach the dots through memory (a restructuring that spliced them in from registers changed the bits).
Triton's layout choices are global to a kernel, so even "pure data movement" edits can change bits: test_exact.py
is the arbiter after any edit. Verified on A800 (sm_80), Triton 3.7.1, fla 0.5.0, for the configurations it tests.

Contract scope: the GDN core (l2norm, cumsum, chunk program, state). The gate/beta computation, conv1d and output norm
around it, fla's use_gate_in_kernel path and context-parallel prefill are the caller's to match.
gdn/exact_decode_ref.py is the simplest (stateless, slower) implementation of the same core contract."""
import torch
import triton
import triton.language as tl

BT, BC = 64, 16
RCP_LN2 = 1.4426950216                       # fla.ops.utils.constant.RCP_LN2 (fla scales the cumsum by it)


@triton.jit
def _blk(ptr, r0, c0, ld, nrow, BC: tl.constexpr):
    """[BC, BC] fp32 tile at (r0, c0) of a row-major matrix with leading dim ld; rows >= nrow read as 0."""
    o = tl.arange(0, BC)
    return tl.load(ptr + (r0 + o)[:, None] * ld + (c0 + o)[None, :], mask=(o < nrow)[:, None], other=0.)


@triton.jit
def _put_row(ptr, r, c0, ld, x, i, BC: tl.constexpr):
    """Store row i of the [BC, BC] tile x to row r, columns c0..c0+BC of a row-major matrix."""
    o = tl.arange(0, BC)
    tl.store(ptr + r * ld + (c0 + o)[None, :] + o[:, None] * 0, x.to(ptr.dtype.element_ty), mask=(o == i)[:, None])


@triton.jit
def _rstd(x, eps):
    """fla l2norm_fwd_kernel's rstd on a [R, D] tile: 1 / sqrt(sum(x*x, 1) + eps) in fp32 (y = x * rstd is then
    elementwise, so it can be applied per 64-wide slice)."""
    b_x = x.to(tl.float32)
    return 1 / tl.sqrt(tl.sum(b_x * b_x, 1) + eps)


@triton.jit
def _kkt3(k, ks0, kj0, nrow_s, H: tl.constexpr, K: tl.constexpr, BC: tl.constexpr):
    """dot(k_s, k_j^T) summed over 64-wide K slices (fla's kkt accumulation)."""
    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    o = tl.arange(0, BC)
    for i_k in tl.static_range(K // 64):
        o_k = i_k * 64 + tl.arange(0, 64)
        b_ks = tl.load(k + (ks0 + o)[:, None] * (H*K) + o_k[None, :], mask=(o < nrow_s)[:, None], other=0.)
        b_kj = tl.load(k + (kj0 + o)[:, None] * (H*K) + o_k[None, :])
        b_A += tl.dot(b_ks, tl.trans(b_kj))
    return b_A


@triton.jit
def _gdn_k0(
    pending, kbuf, Uc, Wc, sgc, S, Hb,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
):
    """Fold the previous chunk into the state if its 64th row was accepted (pending): fwd_h's update by one complete
    chunk, from the cached u, w rows, on the fp32 state and its bf16 copy, for one 64-wide V block (per-element
    identical to fla's BV split). fla updates the state from the fp32 u - w h before v_new is rounded to bf16, so it is
    recomputed here rather than read from VNc. Runs before K1, which overwrites the chunk's rows."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n = i_nh // HV
    if tl.load(pending + i_n) == 0:
        return
    k = kbuf + (i_n * BT * H + (i_nh % HV) // (HV // H)) * K
    Uc += i_nh * BT * V
    Wc += i_nh * BT * K
    sg = sgc + i_nh * BT
    S_ = S + i_nh * K * V
    Hb_ = Hb + i_nh * K * V
    o_t = tl.arange(0, BT)
    b_h1 = tl.load(tl.make_block_ptr(Hb_, (K, V), (V, 1), (0, i_v * 64), (64, 64), (1, 0)))
    b_vf = tl.dot(tl.load(tl.make_block_ptr(Wc, (BT, K), (K, 1), (0, 0), (BT, 64), (1, 0))), b_h1)
    if K > 64:
        b_h2 = tl.load(tl.make_block_ptr(Hb_, (K, V), (V, 1), (64, i_v * 64), (64, 64), (1, 0)))
        b_vf += tl.dot(tl.load(tl.make_block_ptr(Wc, (BT, K), (K, 1), (0, 64), (BT, 64), (1, 0))), b_h2)
    b_vf = tl.load(tl.make_block_ptr(Uc, (BT, V), (V, 1), (0, i_v * 64), (BT, 64), (1, 0))) - b_vf
    b_glast = tl.load(sg + (BT - 1)).to(tl.float32)
    b_gl = tl.load(sg + o_t).to(tl.float32)
    b_vf = b_vf * tl.where(o_t < BT, tl.math.exp2(b_glast - b_gl), 0)[:, None]
    b_glast = tl.math.exp2(b_glast)
    b_vf = b_vf.to(kbuf.dtype.element_ty)
    for i_k in tl.static_range(K // 64):
        p_s = tl.make_block_ptr(S_, (K, V), (V, 1), (i_k * 64, i_v * 64), (64, 64), (1, 0))
        b_s = tl.load(p_s).to(tl.float32)
        b_s *= b_glast
        b_s += tl.dot(tl.load(tl.make_block_ptr(k, (K, BT), (1, H*K), (i_k * 64, 0), (64, BT), (0, 1))), b_vf)
        tl.store(p_s, b_s)
        tl.store(tl.make_block_ptr(Hb_, (K, V), (V, 1), (i_k * 64, i_v * 64), (64, 64), (1, 0)),
                 b_s.to(Hb_.dtype.element_ty))


@triton.jit
def _k1_row(
    r, j, i_n, i_h, i_hk, i_nh, q_in, k_in, v_in, g_in, b_in, k, v, g, beta, qn, AiF, Abf, Uc, Wc, sg,
    gscale, eps, KT,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr,
):
    """One token (chunk row r, window index j): everything K1 computes for it. Rows < r of the chunk caches are
    final (earlier steps or earlier iterations of this window)."""
    T = r + 1
    s = r // BC
    i = r - s * BC
    tb = s * BC
    o_i = tl.arange(0, BC)
    o_t = tl.arange(0, BT)
    o_64 = tl.arange(0, 64)
    o_K = tl.arange(0, K)
    o_V = tl.arange(0, V)
    m_row = (o_i == i)[:, None]

    # ---- token: l2norm(q), l2norm(k) (fla l2norm_fwd_kernel), chunk-buffer row -----------------------------------
    # With HV > H every value head of a key head writes the same normalised k row (identical bytes) and reads it back
    # after its own barrier.
    p_q = q_in + ((i_n * KT + j) * H + i_hk) * K
    p_k = k_in + ((i_n * KT + j) * H + i_hk) * K
    rq = _rstd(tl.load(p_q + o_K[None, :] + o_i[:, None] * 0), eps)
    rk = _rstd(tl.load(p_k + o_K[None, :] + o_i[:, None] * 0), eps)
    for i_k in tl.static_range(K // 64):
        b_qx = tl.load(p_q + i_k * 64 + o_64[None, :] + o_i[:, None] * 0)
        tl.store(qn + (i_nh * BT + r) * K + i_k * 64 + o_64[None, :] + o_i[:, None] * 0,
                 (b_qx.to(tl.float32) * rq[:, None]).to(b_qx.dtype), mask=(o_i == 0)[:, None])
        b_kx = tl.load(p_k + i_k * 64 + o_64[None, :] + o_i[:, None] * 0)
        tl.store(k + r * (H*K) + i_k * 64 + o_64[None, :] + o_i[:, None] * 0,
                 (b_kx.to(tl.float32) * rk[:, None]).to(b_kx.dtype), mask=(o_i == 0)[:, None])
    tl.store(v + r * (HV*V) + o_V, tl.load(v_in + ((i_n * KT + j) * HV + i_h) * V + o_V))
    tl.store(g + r * HV, tl.load(g_in + (i_n * KT + j) * HV + i_h))
    tl.store(beta + r * HV, tl.load(b_in + (i_n * KT + j) * HV + i_h))
    tl.debug_barrier()

    # ---- chunk_local_cumsum_scalar_kernel ----------------------------------------------------------------------
    b_s = tl.load(tl.make_block_ptr(g, (T,), (HV,), (0,), (BT,), (0,)), boundary_check=(0,)).to(tl.float32)
    b_gc = tl.cumsum(b_s, axis=0)
    b_gc *= gscale
    tl.store(tl.make_block_ptr(sg, (T,), (1,), (0,), (BT,), (0,)), b_gc.to(tl.float32), boundary_check=(0,))
    tl.debug_barrier()

    # ---- kkt_solve: row i of sub-block s of (I+A)^-1 ------------------------------------------------------------
    nrow = i + 1
    m_ts = (tb + o_i) < T
    b_gs = tl.load(sg + tb + o_i, mask=m_ts, other=0.).to(tl.float32)
    b_bs = tl.load(beta + (tb + o_i) * HV, mask=m_ts, other=0.).to(tl.float32)
    m_d = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    b_Ass = _kkt3(k, tb, tb, nrow, H, K, BC)
    b_Ass *= tl.where(m_d & m_ts[:, None] & m_ts[None, :], tl.math.exp2(b_gs[:, None] - b_gs[None, :]), 0.)
    b_Ass = b_Ass * b_bs[:, None]
    b_Ai = tl.where((o_i < i)[:, None], _blk(AiF, tb, tb, BT, i, BC), -b_Ass)
    if i >= 2:
        b_a = tl.sum(tl.where((o_i == i)[:, None], -b_Ass, 0.), 0)
        b_a = tl.where(o_i < i, b_a, 0.)
        b_a = b_a + tl.sum(b_a[:, None] * b_Ai, 0)
        b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
    _put_row(AiF, r, tb, BT, b_Ai, i, BC)
    b_Aiss = b_Ai + m_I
    _put_row(Abf, r, tb, BT, b_Aiss, i, BC)
    if s >= 1:
        b_g0 = tl.load(sg + o_i).to(tl.float32)
        b_Ai00 = _blk(AiF, 0, 0, BT, BC, BC) + m_I
        b_As0 = _kkt3(k, tb, 0, nrow, H, K, BC)
        b_As0 *= tl.where(m_ts[:, None], tl.math.exp2(b_gs[:, None] - b_g0[None, :]), 0.)
        b_As0 = b_As0 * b_bs[:, None]
        if s == 1:
            b_Ai10 = -tl.dot(tl.dot(b_Aiss, b_As0, input_precision='tf32'), b_Ai00, input_precision='tf32')
            _put_row(AiF, r, 0, BT, b_Ai10, i, BC)
            _put_row(Abf, r, 0, BT, b_Ai10, i, BC)
        else:
            b_g1 = tl.load(sg + BC + o_i).to(tl.float32)
            b_Ai11 = _blk(AiF, BC, BC, BT, BC, BC) + m_I
            b_Ai10 = _blk(AiF, BC, 0, BT, BC, BC)
            b_As1 = _kkt3(k, tb, BC, nrow, H, K, BC)
            b_As1 *= tl.where(m_ts[:, None], tl.math.exp2(b_gs[:, None] - b_g1[None, :]), 0.)
            b_As1 = b_As1 * b_bs[:, None]
            if s == 2:
                b_Ai21 = -tl.dot(tl.dot(b_Aiss, b_As1, input_precision='tf32'), b_Ai11, input_precision='tf32')
                b_Ai20 = -tl.dot(b_Aiss, tl.dot(b_As0, b_Ai00, input_precision='tf32') +
                                 tl.dot(b_As1, b_Ai10, input_precision='tf32'), input_precision='tf32')
                _put_row(AiF, r, BC, BT, b_Ai21, i, BC)
                _put_row(AiF, r, 0, BT, b_Ai20, i, BC)
                _put_row(Abf, r, BC, BT, b_Ai21, i, BC)
                _put_row(Abf, r, 0, BT, b_Ai20, i, BC)
            else:
                b_g2 = tl.load(sg + 2 * BC + o_i).to(tl.float32)
                b_Ai22 = _blk(AiF, 2 * BC, 2 * BC, BT, BC, BC) + m_I
                b_Ai20 = _blk(AiF, 2 * BC, 0, BT, BC, BC)
                b_Ai21 = _blk(AiF, 2 * BC, BC, BT, BC, BC)
                b_As2 = _kkt3(k, tb, 2 * BC, nrow, H, K, BC)
                b_As2 *= tl.where(m_ts[:, None], tl.math.exp2(b_gs[:, None] - b_g2[None, :]), 0.)
                b_As2 = b_As2 * b_bs[:, None]
                b_Ai32 = -tl.dot(tl.dot(b_Aiss, b_As2, input_precision='tf32'), b_Ai22, input_precision='tf32')
                b_Ai31 = -tl.dot(b_Aiss, tl.dot(b_As1, b_Ai11, input_precision='tf32') +
                                 tl.dot(b_As2, b_Ai21, input_precision='tf32'), input_precision='tf32')
                b_Ai30 = -tl.dot(b_Aiss, tl.dot(b_As0, b_Ai00, input_precision='tf32') +
                                 tl.dot(b_As1, b_Ai10, input_precision='tf32') +
                                 tl.dot(b_As2, b_Ai20, input_precision='tf32'), input_precision='tf32')
                # sub-block 3's off-diagonal blocks are read by no later row: Abf only
                _put_row(Abf, r, 2 * BC, BT, b_Ai32, i, BC)
                _put_row(Abf, r, BC, BT, b_Ai31, i, BC)
                _put_row(Abf, r, 0, BT, b_Ai30, i, BC)
    tl.debug_barrier()

    # ---- recompute_w_u: rows of u, w for the token (16-row tile, row i stored) --------------------------------
    m_rs = (o_i <= i)[:, None]
    b_b = tl.load(tl.make_block_ptr(beta, (T,), (HV,), (0,), (BT,), (0,)), boundary_check=(0,))
    b_A = tl.load(Abf + (tb + o_i)[:, None] * BT + o_t[None, :], mask=m_rs, other=0.)
    b_vin = tl.load(tl.make_block_ptr(v, (T, V), (HV*V, 1), (0, 0), (BT, V), (1, 0)), boundary_check=(0, 1))
    b_vb = (b_vin * b_b[:, None]).to(b_vin.dtype)
    b_u = tl.dot(b_A, b_vb, allow_tf32=False).to(b_vin.dtype)
    tl.store(Uc + (tb + o_i)[:, None] * V + o_V[None, :], b_u, mask=m_row)
    b_ge = tl.math.exp2(tl.load(tl.make_block_ptr(sg, (T,), (1,), (0,), (BT,), (0,)), boundary_check=(0,)))
    for i_k in tl.static_range(K // 64):
        b_kk = tl.load(tl.make_block_ptr(k, (T, K), (H*K, 1), (0, i_k * 64), (BT, 64), (1, 0)), boundary_check=(0, 1))
        b_kb = b_kk * b_b[:, None]
        b_kb *= b_ge[:, None]
        b_w = tl.dot(b_A, b_kb.to(b_kk.dtype)).to(b_kk.dtype)
        tl.store(Wc + (tb + o_i)[:, None] * K + i_k * 64 + o_64[None, :], b_w, mask=m_row)




@triton.jit
def _gdn_k1(
    q_in, k_in, v_in, g_in, b_in, pos, nrows, pending, kbuf, vbuf, gbuf, bbuf, qn, AiF, Abf, Uc, Wc, sgc,
    gscale, eps, KT,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr,
    SINGLE: tl.constexpr,
):
    """Rows pos % 64 .. pos % 64 + nrows - 1 of one (slot, head); the slot's head-0 program then resets the fold flag
    (K0 has run)."""
    i_nh = tl.program_id(0)
    i_n, i_h = i_nh // HV, i_nh % HV
    i_hk = i_h // (HV // H)
    k = kbuf + (i_n * BT * H + i_hk) * K
    v = vbuf + (i_n * BT * HV + i_h) * V
    g = gbuf + i_n * BT * HV + i_h
    beta = bbuf + i_n * BT * HV + i_h
    AiF += i_nh * BT * BT
    Abf += i_nh * BT * BT
    Uc += i_nh * BT * V
    Wc += i_nh * BT * K
    sg = sgc + i_nh * BT
    r0 = tl.load(pos + i_n) % BT
    nr = tl.load(nrows + i_n)
    if SINGLE:                          # one token: no loop (a runtime loop costs registers and occupancy)
        _k1_row(r0, 0, i_n, i_h, i_hk, i_nh, q_in, k_in, v_in, g_in, b_in, k, v, g, beta, qn, AiF, Abf, Uc, Wc,
                sg, gscale, eps, KT, H, HV, K, V, BT, BC)
    else:
        for j in range(nr):
            _k1_row(r0 + j, j, i_n, i_h, i_hk, i_nh, q_in, k_in, v_in, g_in, b_in, k, v, g, beta, qn, AiF, Abf, Uc,
                    Wc, sg, gscale, eps, KT, H, HV, K, V, BT, BC)
            tl.debug_barrier()
    if i_h == 0:
        tl.store(pending + i_n, 0)                                   # K0 has run


@triton.jit
def _gdn_k2(
    pos, nrows, kbuf, qn, Hb, o, Uc, Wc, VNc, sgc,
    S, scale, KT,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr,
    BT: tl.constexpr, BC: tl.constexpr, TILE: tl.constexpr, EAGER_FOLD: tl.constexpr,
):
    """Window rows r0 .. r1 = r0 + nrows - 1 of one (V block, slot, head) on one TILE-row tile starting at r0's
    sub-block (16 rows for one token, 32 for a window of up to 16): v_new = u - w bf16(S) for the window's rows (fwd_h),
    then their output rows (chunk_fwd_kernel_o), which need the v_new rows of the whole window. Tile rows outside the
    window are zero (MMA rows are independent)."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // HV, i_nh % HV
    r0 = tl.load(pos + i_n) % BT
    nr = tl.load(nrows + i_n)
    r1 = r0 + nr - 1
    T = r1 + 1
    k = kbuf + (i_n * BT * H + i_h // (HV // H)) * K
    Uc += i_nh * BT * V
    Wc += i_nh * BT * K
    VNc += i_nh * BT * V
    sg = sgc + i_nh * BT
    Hb_ = Hb + i_nh * K * V
    o_i = tl.arange(0, BC)
    o_t = tl.arange(0, BT)
    o_64 = tl.arange(0, 64)
    o_v = i_v * BV + tl.arange(0, BV)
    tb = (r0 // BC) * BC                       # the window lies in rows tb .. tb + TILE - 1 (TILE 16: one row)
    o_w = tl.arange(0, TILE)
    rows = tb + o_w
    m_win = ((rows >= r0) & (rows <= r1))[:, None]

    # ---- fwd_h's v_new = u - w bf16(S) and chunk_fwd_kernel_o's q bf16(S), q k^T, on the same K slices --------------
    m_t = o_t < T
    m_ts = rows < T
    b_o = tl.zeros([TILE, BV], dtype=tl.float32)
    b_Aq = tl.zeros([TILE, BT], dtype=tl.float32)
    for i_k in tl.static_range(K // 64):
        b_w = tl.load(Wc + rows[:, None] * K + i_k * 64 + o_64[None, :], mask=m_win, other=0.)
        b_h = tl.load(tl.make_block_ptr(Hb_, (K, V), (V, 1), (i_k * 64, i_v * BV), (64, BV), (1, 0)))
        b_q = tl.load(qn + (i_nh * BT + rows)[:, None] * K + i_k * 64 + o_64[None, :], mask=m_win, other=0.)
        b_kt = tl.load(tl.make_block_ptr(k, (K, T), (1, H*K), (i_k * 64, 0), (64, BT), (0, 1)), boundary_check=(0, 1))
        if i_k == 0:
            b_v = tl.dot(b_w, b_h)                                  # fwd_h: b_v = dot(w1, h1); b_v += dot(w2, h2)
            b_h1 = b_h
        else:
            b_v += tl.dot(b_w, b_h)
            b_h2 = b_h
        b_o += tl.dot(b_q, b_h)                                     # chunk_fwd_kernel_o: zeros += dot over K
        b_Aq += tl.dot(b_q, b_kt)
    b_u = tl.load(Uc + rows[:, None] * V + o_v[None, :], mask=m_win, other=0.)
    b_v = b_u.to(tl.float32) - b_v
    tl.store(VNc + rows[:, None] * V + o_v[None, :], b_v.to(VNc.dtype.element_ty), mask=m_win)
    tl.debug_barrier()

    # ---- chunk_fwd_kernel_o epilogue for the window's rows (needs the v_new rows of the whole window) -------------
    b_gs = tl.load(sg + rows, mask=m_ts, other=0.).to(tl.float32)
    b_gall = tl.load(tl.make_block_ptr(sg, (T,), (1,), (0,), (BT,), (0,)), boundary_check=(0,))
    b_o = b_o * tl.math.exp2(b_gs)[:, None]
    b_Aq = b_Aq * tl.math.exp2(b_gs[:, None] - b_gall[None, :])
    m_A = (rows[:, None] >= o_t[None, :]) & (m_ts[:, None] & m_t)
    b_Aq = tl.where(m_A, b_Aq, 0)
    b_vn = tl.load(tl.make_block_ptr(VNc, (T, V), (V, 1), (0, i_v * BV), (BT, BV), (1, 0)), boundary_check=(0, 1))
    b_o = tl.fma(tl.dot(b_Aq.to(b_vn.dtype), b_vn), scale, b_o * scale)      # = fla's PTX for its epilogue
    tl.store(o + ((i_n * KT + (rows - r0)) * HV + i_h)[:, None] * V + o_v[None, :], b_o.to(o.dtype.element_ty),
             mask=m_win)

    # ---- plain decode folds a completed chunk right away (fwd_h's 64-row arithmetic from the cached u, w rows) ------
    if EAGER_FOLD:
        if T == BT:
            S_ = S + i_nh * K * V
            b_vf = tl.dot(tl.load(tl.make_block_ptr(Wc, (BT, K), (K, 1), (0, 0), (BT, 64), (1, 0))), b_h1)
            if K > 64:
                b_vf += tl.dot(tl.load(tl.make_block_ptr(Wc, (BT, K), (K, 1), (0, 64), (BT, 64), (1, 0))), b_h2)
            b_vf = tl.load(tl.make_block_ptr(Uc, (BT, V), (V, 1), (0, i_v * BV), (BT, BV), (1, 0))) - b_vf
            b_glast = tl.load(sg + (BT - 1)).to(tl.float32)
            b_gl = tl.load(sg + o_t).to(tl.float32)
            b_vf = b_vf * tl.where(m_t, tl.math.exp2(b_glast - b_gl), 0)[:, None]
            b_glast = tl.math.exp2(b_glast)
            b_vf = b_vf.to(kbuf.dtype.element_ty)
            for i_k in tl.static_range(K // 64):
                p_s = tl.make_block_ptr(S_, (K, V), (V, 1), (i_k * 64, i_v * BV), (64, BV), (1, 0))
                b_s = tl.load(p_s).to(tl.float32)
                b_s *= b_glast
                b_s += tl.dot(tl.load(tl.make_block_ptr(k, (K, BT), (1, H*K), (i_k * 64, 0), (64, BT), (0, 1))), b_vf)
                tl.store(p_s, b_s)
                tl.store(tl.make_block_ptr(Hb_, (K, V), (V, 1), (i_k * 64, i_v * BV), (64, BV), (1, 0)),
                         b_s.to(Hb_.dtype.element_ty))


class GDNExactDecode:
    """Exact decode state for N slots of one GDN layer (K in {64, 128}, V % 64 == 0, V % BV == 0).

    A step processes a window of up to KT consecutive tokens per slot (1 for plain decode; 1 + draft length for
    speculative verification), starting at the slot's absolute position `pos`, truncated at the slot's next 64-token
    boundary (`window`). Its outputs are the trainer forward's rows for those tokens, whatever the window size.
    `commit(pos, accepted)` records how many of the window's tokens were kept; the caller then advances
    pos += accepted. Rejected tokens leave nothing behind: rows beyond the accepted ones are overwritten by the next
    window, and the state is folded only when a chunk's 64th row has been accepted (lazily, at the start of the next
    step), so rollback is free. With KT == 1 use decode(): every token is accepted and folds happen in the step."""

    def __init__(self, N, H, HV, K, V, device, KT=1, dtype=torch.bfloat16, beta_dtype=None, BV=64, nw1=2, nw2=4,
                 eps=1e-6):
        assert K in (64, 128) and V % 64 == 0 and V % BV == 0 and 1 <= KT <= BC
        self.N, self.H, self.HV, self.K, self.V, self.KT = N, H, HV, K, V, KT
        self.BV, self.nw1, self.nw2, self.eps = BV, nw1, nw2, eps
        z = lambda *s, dt=dtype: torch.zeros(*s, device=device, dtype=dt)
        self.S, self.Hb = z(N, HV, K, V, dt=torch.float32), z(N, HV, K, V)
        self.kbuf, self.vbuf = z(N, BT, H, K), z(N, BT, HV, V)
        self.gbuf, self.bbuf = z(N, BT, HV, dt=torch.float32), z(N, BT, HV, dt=beta_dtype or dtype)
        self.AiF, self.Abf = z(N * HV, BT, BT, dt=torch.float32), z(N * HV, BT, BT)
        self.Uc, self.Wc, self.VNc = z(N * HV, BT, V), z(N * HV, BT, K), z(N * HV, BT, V)
        self.sg, self.qn = z(N * HV, BT, dt=torch.float32), z(N * HV, BT, K)
        self.pending = torch.zeros(N, device=device, dtype=torch.int32)
        self._ones = torch.ones(N, device=device, dtype=torch.int32)

    @staticmethod
    def window(pos, k):
        """Rows a step may process: min(k, tokens left in the slot's current chunk)."""
        return torch.clamp(BT - pos % BT, max=k).to(torch.int32)

    def _launch(self, n0, ns, q, k, v, g, beta, pos, nrows, scale, KT):
        """KT == 1 (plain decode, every token accepted): K1 + K2 with a 16-row tile and the fold done eagerly by K2 on
        the chunk's 64th row. KT > 1 (windows that may be partly rejected): K0 (lazy fold of an accepted chunk) + K1 +
        K2 with a 32-row tile."""
        HV = self.HV
        sl, hs = slice(n0, n0 + ns), slice(n0 * HV, (n0 + ns) * HV)
        o = torch.empty(ns, KT, HV, self.V, device=q.device, dtype=v.dtype)
        if KT > 1:
            _gdn_k0[(self.V // 64, ns * HV)](
                self.pending[sl], self.kbuf[sl], self.Uc[hs], self.Wc[hs], self.sg[hs], self.S[sl], self.Hb[sl],
                H=self.H, HV=HV, K=self.K, V=self.V, BT=BT, num_warps=4)
        _gdn_k1[(ns * HV,)](
            q, k, v, g, beta, pos, nrows, self.pending[sl], self.kbuf[sl], self.vbuf[sl], self.gbuf[sl], self.bbuf[sl],
            self.qn[hs], self.AiF[hs], self.Abf[hs], self.Uc[hs], self.Wc[hs], self.sg[hs],
            RCP_LN2, self.eps, KT, H=self.H, HV=HV, K=self.K, V=self.V, BT=BT, BC=BC, SINGLE=KT == 1,
            num_warps=self.nw1)
        _gdn_k2[(self.V // self.BV, ns * HV)](
            pos, nrows, self.kbuf[sl], self.qn[hs], self.Hb[sl], o, self.Uc[hs], self.Wc[hs], self.VNc[hs],
            self.sg[hs], self.S[sl], scale, KT, H=self.H, HV=HV, K=self.K, V=self.V, BV=self.BV, BT=BT, BC=BC,
            TILE=BC if KT == 1 else 2 * BC, EAGER_FOLD=KT == 1, num_warps=self.nw2)
        return o

    def step(self, q, k, v, g, beta, pos, nrows, scale):
        """q, k [N,KT,H,K] raw (normalised in-kernel), v [N,KT,HV,V], g [N,KT,HV] raw log-gate, beta [N,KT,HV];
        pos [N] int32 absolute position of the window's first token; nrows [N] int32 = window(pos, KT).
        Returns o [N,KT,HV,V] (rows >= nrows undefined). Three launches (K0 fold, K1, K2); graph-capturable."""
        N, KT, H, HV, K, V = self.N, self.KT, self.H, self.HV, self.K, self.V
        for x, shape in ((q, (N, KT, H, K)), (k, (N, KT, H, K)), (v, (N, KT, HV, V)), (g, (N, KT, HV)), (beta, (N, KT, HV))):
            assert tuple(x.shape) == shape and x.is_contiguous(), (tuple(x.shape), shape)
        # fla's bits depend on these dtypes (beta enters recompute_w_u unconverted; g must be the fp32 log-gate)
        assert beta.dtype == self.bbuf.dtype and g.dtype == torch.float32 and k.dtype == self.kbuf.dtype
        assert self.KT > 1, 'KT == 1: use decode()'
        return self._launch(0, self.N, q, k, v, g, beta, pos, nrows, scale, self.KT)

    def commit(self, pos, accepted):
        """accepted [N] int32 (0..nrows) tokens of the window just stepped are kept; the caller then sets
        pos += accepted. Marks a fold for the next step when the window's acceptance completed a chunk."""
        self.pending |= ((accepted > 0) & (pos % BT + accepted == BT)).to(torch.int32)

    def decode(self, q, k, v, g, beta, pos, scale):
        """Plain one-token decode (needs KT == 1): q, k [N,H,K], v [N,HV,V], g, beta [N,HV]; returns o [N,HV,V].
        Every token is accepted, so a completed chunk is folded in the same step. The caller then increments pos."""
        assert self.KT == 1
        return self._launch(0, self.N, q[:, None], k[:, None], v[:, None], g[:, None], beta[:, None], pos,
                            self._ones, scale, 1)[:, 0]

    def load(self, n, S, b, q, k, v, g, beta):
        """Slot n resumes at its boundary state S [HV,K,V] fp32 taken at absolute position b (multiple of 64), then
        replays the r = P - b < 64 partial-chunk tokens (raw q, k [r,H,K], v [r,HV,V], g, beta [r,HV]) in windows of
        up to 16 to fill the row caches (outputs discarded)."""
        assert b % BT == 0 and k.shape[0] < BT
        self.S[n] = S
        self.Hb[n] = S.to(self.Hb.dtype)
        self.pending[n] = 0
        dev = k.device
        for j0 in range(0, k.shape[0], BC):
            w = min(BC, k.shape[0] - j0)
            pos = torch.tensor([b + j0], device=dev, dtype=torch.int32)
            x = [t[None, j0:j0 + w].contiguous() for t in (q, k, v, g, beta)]
            self._launch(n, 1, *x, pos, torch.tensor([w], device=dev, dtype=torch.int32), 1.0, w)
