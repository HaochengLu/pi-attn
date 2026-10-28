"""Exact GDN decode: each decoded token's output is, bitwise, the row that fla 0.5.0's chunk_gated_delta_rule
(use_qk_l2norm_in_kernel=True, its default use_exp2 path) produces for that token in a full training forward,
for one token per step or for a window of several (speculative verification), with free rollback.

State per (slot, value head). The fp32 state S is kept only at absolute 64-token boundaries, as the bf16 copy fla's
fwd_h feeds to every intra-chunk dot (Hb, read every step) plus what bf16 drops (Slo: S's lower 16 bits; Sup: one bit
per element, packed 32 rows to a word, set where rounding to bf16 went up), read and written only by folds: S's bits are
(Hb's bits - Sup) << 16 | Slo, 2.06 bytes per element beside Hb instead of 4 for a separate fp32 S (get_states,
put_states convert). Every intermediate of fla's chunk program is row-causal inside a chunk, so the current chunk's rows
are cached as fla materialises them between its kernels:
  kbuf, vbuf, gbuf, bbuf  the chunk's inputs (k after l2norm; g raw; beta in its own dtype)
  sg                      cumsum(g) * log2(e) (chunk_local_cumsum)
  AiF                     (I+A)^-1 rows in fp32: diagonal 16x16 blocks before the +I (as fla's forward substitution
                          holds them), off-diagonal blocks final; only the blocks later rows read are stored
  Abf                     (I+A)^-1 rows in bf16 (fla's stored A)            VNc   v_new rows (bf16)
The l2norm(q), u and w rows are read only for the window that computes them (they go through a per-step scratch), and
a fold recomputes the whole chunk's u, w from the cached rows.
Rows at or beyond the window's end hold garbage (earlier chunks, rejected drafts); every consumer masks them, and all
cached values are finite, so they contribute exact zeros. This invariant is also the rollback.

A step covers chunk rows r0 .. r0 + nrows - 1 (nrows <= KT, never past the chunk's 64th row):
  K0 (windows of several tokens only; one program per V block x slot x head, exits at once unless pending): if the
     previous step's accepted tokens completed a chunk, fold it into S/Hb (fwd_h's 64-row arithmetic, from u, w
     recomputed by recompute_w_u's 64-row arithmetic) before this window overwrites its rows. Plain decode accepts
     every token and folds in K2 instead.
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
Pinned on the way: fla's chunk_fwd_kernel_o epilogue is compiled with an fma contraction whose form depends on the
Triton version, written out here per version (_O_EPILOGUE_FORM); and the
token's rows reach the dots through memory (a restructuring that spliced them in from registers changed the bits).
Triton's layout choices are global to a kernel, so even "pure data movement" edits can change bits: test_exact.py
is the arbiter after any edit. Verified with fla 0.5.0 on A800 (sm_80) and RTX 4090 D (sm_89), Triton 3.7.1 and 3.6.0,
for the configurations it tests, with fla's odd autotune configs removed (pin_fla_configs).

Contract scope: the GDN core (l2norm, cumsum, chunk program, state). The gate/beta computation, conv1d and output norm
around it, fla's use_gate_in_kernel path and context-parallel prefill are the caller's to match.
gdn/exact_decode_ref.py is the simplest (stateless, slower) implementation of the same core contract."""
import torch
import triton
import triton.language as tl

BT, BC = 64, 16
RCP_LN2 = 1.4426950216                       # fla.ops.utils.constant.RCP_LN2 (fla scales the cumsum by it)

# fla's chunk_fwd_kernel_o computes `b_o * scale + tl.dot(b_A, b_v) * scale`, and LLVM contracts one of the two
# products into an fma; which one depends on the Triton version (read off fla's compiled PTX):
#   Triton 3.7.1: fma(dot, scale, b_o * scale)        Triton 3.6.0: fma(b_o, scale, dot * scale)
# The epilogue below is written out in the matching form. Other versions are refused until test_exact.py has been
# run under them and this table extended.
_O_EPILOGUE_FORM = {'3.7.1': 0, '3.6.0': 1}
if triton.__version__ not in _O_EPILOGUE_FORM:
    raise RuntimeError(f'exact GDN decode: Triton {triton.__version__} unverified; run gdn/test_exact.py, then add '
                       f'its chunk_fwd_kernel_o epilogue form to _O_EPILOGUE_FORM')
O_FORM = tl.constexpr(_O_EPILOGUE_FORM[triton.__version__])

# GPUs (compute capability) on which test_exact.py passes for each Triton version above; others are refused.
_VALIDATED_GPUS = {(8, 0), (8, 9)}                    # A800, RTX 4090 D

# fla autotune configs whose bits differ from the other configs of the same kernel, and from this file's kernels
# (e4_autotune*.log, at the Qwen3.5-4B GDN shape; e14b_o_config_pins.log: test_exact.py with each config pinned). fla tunes per shape (l2norm's key includes the row count,
# and the varlen and fixed-length variants of a kernel are tuned separately), so with them in the list one forward
# can run two programs: e10 (alone vs packed rows differ on the RTX 4090) and e14 (varlen vs fixed-length).
_FLA_ODD_CONFIGS = {
    'l2norm_fwd_kernel': lambda c, cap, tv: (c.kwargs['BT'], c.num_warps) in ((8, 8), (8, 16), (16, 16)),
    'chunk_fwd_kernel_o': lambda c, cap, tv: tv == '3.6.0' and (c.kwargs['BK'], c.kwargs['BV']) == (64, 64),
}


def pin_fla_configs():
    """Remove _FLA_ODD_CONFIGS from fla's autotune lists, so that fla's forward is one program whatever it tunes to.
    The sampler (exact_gdn.install) and the trainer both have to call it before fla's first use."""
    from fla.modules import l2norm
    from fla.ops.common import chunk_o
    cap, tv = torch.cuda.get_device_capability(), triton.__version__
    for mod, name in ((l2norm, 'l2norm_fwd_kernel'), (chunk_o, 'chunk_fwd_kernel_o')):
        kern = getattr(mod, name)
        while not hasattr(kern, 'configs'):                          # autotuner under fla's decorators
            kern = kern.fn
        kern.configs = [c for c in kern.configs if not _FLA_ODD_CONFIGS[name](c, cap, tv)]
        kern.cache.clear()


@triton.jit
def _o_epilogue(b_o, b_dot, scale, FORM: tl.constexpr):
    """fla's `b_o * scale + dot * scale` with the fp-contraction fla gets under the installed Triton (O_FORM)."""
    if FORM == 0:
        return tl.fma(b_dot, scale, b_o * scale)
    else:
        return tl.fma(b_o, scale, b_dot * scale)


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
def _state_tile(b_h, Slo, Sup, k0, c0, V: tl.constexpr, BVX: tl.constexpr):
    """The fp32 state's [64, BVX] tile at rows k0.. (a multiple of 64), columns c0.., from its bf16 copy b_h (the same
    tile of Hb) and the stored rest (see state_layout): S's bits are (Hb's bits - up) << 16 | lo, since bf16(S) is S's
    upper half rounded to nearest, i.e. that half or the next one (up = 1). up bits are packed 32 rows to a word, so the
    tile's are two rows of words, broadcast down the rows."""
    o_r = tl.arange(0, 64)
    o_c = c0 + tl.arange(0, BVX)
    lo = tl.load(Slo + (k0 + o_r)[:, None] * V + o_c[None, :]).to(tl.int32) & 0xFFFF
    w0 = tl.load(Sup + (k0 // 32) * V + o_c)[None, :]
    w1 = tl.load(Sup + (k0 // 32 + 1) * V + o_c)[None, :]
    up = (tl.where((o_r < 32)[:, None], w0, w1) >> (o_r % 32)[:, None]) & 1
    hb = b_h.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    return (((hb - up) << 16) | lo).to(tl.float32, bitcast=True)


@triton.jit
def _state_put(b_s, Hb, Slo, Sup, k0, c0, V: tl.constexpr, BVX: tl.constexpr):
    """Store the fp32 state tile b_s ([64, BVX] at rows k0.., columns c0..) as Hb = bf16(b_s), lo = b_s's lower 16
    bits and one bit per element saying whether bf16 rounding went up, packed 32 rows to a word (_state_tile)."""
    o_r = tl.arange(0, 64)
    o_c = c0 + tl.arange(0, BVX)
    b_hb = b_s.to(Hb.dtype.element_ty)
    tl.store(Hb + (k0 + o_r)[:, None] * V + o_c[None, :], b_hb)
    bits = b_s.to(tl.int32, bitcast=True)
    tl.store(Slo + (k0 + o_r)[:, None] * V + o_c[None, :], (bits & 0xFFFF).to(tl.int16))
    up = ((b_hb.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF) != ((bits >> 16) & 0xFFFF)).to(tl.int32)
    up = up << (o_r % 32)[:, None]                                    # distinct bits: their sum is their or
    tl.store(Sup + (k0 // 32) * V + o_c, tl.sum(tl.where((o_r < 32)[:, None], up, 0), axis=0))
    tl.store(Sup + (k0 // 32 + 1) * V + o_c, tl.sum(tl.where((o_r < 32)[:, None], 0, up), axis=0))


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
def _fold_vf(k, v, beta, Abf, sg, b_h1, b_h2, c0, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
             V: tl.constexpr, BT: tl.constexpr, BVX: tl.constexpr):
    """fwd_h's fp32 u - w bf16(S) of a completed chunk, columns c0 .. c0 + BVX (b_h1, b_h2: the bf16 state's two 64-row
    halves of those columns), with u, w recomputed from the cached chunk inputs and (I+A)^-1 by recompute_w_u's
    arithmetic on the whole 64-row tile: K1 computed each of these rows on a 16-row tile, and a row of an MMA tile does
    not depend on the other rows, so these are the rows K2 used (and fla's). (I+A)^-1 is read lower-triangular: K1
    never writes the blocks right of a row's diagonal block (they never contribute to a row), and whatever they hold
    must not enter the fold."""
    o_t = tl.arange(0, BT)
    b_b = tl.load(beta + o_t * HV)
    b_A = tl.load(Abf + o_t[:, None] * BT + o_t[None, :], mask=o_t[None, :] <= o_t[:, None], other=0.)
    b_vin = tl.load(tl.make_block_ptr(v, (BT, V), (HV*V, 1), (0, c0), (BT, BVX), (1, 0)))
    b_vb = (b_vin * b_b[:, None]).to(b_vin.dtype)
    b_u = tl.dot(b_A, b_vb, allow_tf32=False).to(b_vin.dtype)
    b_ge = tl.math.exp2(tl.load(sg + o_t))
    b_kk = tl.load(tl.make_block_ptr(k, (BT, K), (H*K, 1), (0, 0), (BT, 64), (1, 0)))
    b_kb = b_kk * b_b[:, None]
    b_kb *= b_ge[:, None]
    b_vf = tl.dot(tl.dot(b_A, b_kb.to(b_kk.dtype)).to(b_kk.dtype), b_h1)
    if K > 64:
        b_kk = tl.load(tl.make_block_ptr(k, (BT, K), (H*K, 1), (0, 64), (BT, 64), (1, 0)))
        b_kb = b_kk * b_b[:, None]
        b_kb *= b_ge[:, None]
        b_vf += tl.dot(tl.dot(b_A, b_kb.to(b_kk.dtype)).to(b_kk.dtype), b_h2)
    return b_u - b_vf


@triton.jit
def _gdn_k0(
    pending, kbuf, vbuf, bbuf, Abf, sgc, Slo, Sup, Hb, slots,
    sSl: tl.constexpr, sSu: tl.constexpr, sHb: tl.constexpr, skb: tl.constexpr, svb: tl.constexpr,
    sgb: tl.constexpr, sbb: tl.constexpr, sAi: tl.constexpr, sAb: tl.constexpr, sVN: tl.constexpr, ssg: tl.constexpr,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
):
    """Fold the previous chunk into the state if its 64th row was accepted (pending): fwd_h's update by one complete
    chunk on the fp32 state and its bf16 copy, for one 64-wide V block (per-element identical to fla's BV split), from
    u - w h (_fold_vf: fla updates the state from the fp32 value, before v_new is rounded to bf16). Runs before K1,
    which overwrites the chunk's rows."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // HV, i_nh % HV
    slot = tl.load(slots + i_n).to(tl.int64)
    if slot < 0:                        # padding entry of a CUDA-graph batch (engine's PAD_SLOT_ID)
        return
    if tl.load(pending + slot) == 0:
        return
    k = kbuf + slot * skb + (i_h // (HV // H)) * K
    sg = sgc + slot * ssg + i_h * BT
    Slo_ = Slo + slot * sSl + i_h * K * V
    Sup_ = Sup + slot * sSu + i_h * K * (V // 32)
    Hb_ = Hb + slot * sHb + i_h * K * V
    o_t = tl.arange(0, BT)
    b_h1 = tl.load(tl.make_block_ptr(Hb_, (K, V), (V, 1), (0, i_v * 64), (64, 64), (1, 0)))
    b_h2 = b_h1
    if K > 64:
        b_h2 = tl.load(tl.make_block_ptr(Hb_, (K, V), (V, 1), (64, i_v * 64), (64, 64), (1, 0)))
    b_vf = _fold_vf(k, vbuf + slot * svb + i_h * V, bbuf + slot * sbb + i_h, Abf + slot * sAb + i_h * BT * BT, sg,
                    b_h1, b_h2, i_v * 64, H, HV, K, V, BT, 64)
    b_glast = tl.load(sg + (BT - 1)).to(tl.float32)
    b_gl = tl.load(sg + o_t).to(tl.float32)
    b_vf = b_vf * tl.where(o_t < BT, tl.math.exp2(b_glast - b_gl), 0)[:, None]
    b_glast = tl.math.exp2(b_glast)
    b_vf = b_vf.to(kbuf.dtype.element_ty)
    for i_k in tl.static_range(K // 64):
        if i_k == 0:
            b_s = _state_tile(b_h1, Slo_, Sup_, 0, i_v * 64, V, 64)
        else:
            b_s = _state_tile(b_h2, Slo_, Sup_, 64, i_v * 64, V, 64)
        b_s *= b_glast
        b_s += tl.dot(tl.load(tl.make_block_ptr(k, (K, BT), (1, H*K), (i_k * 64, 0), (64, BT), (0, 1))), b_vf)
        _state_put(b_s, Hb_, Slo_, Sup_, i_k * 64, i_v * 64, V, 64)


@triton.jit
def _k1_row(
    r, j, xr, gr, i_h, i_hk, q_in, k_in, v_in, g_in, b_in, sqk, sv, k, v, g, beta, sq, AiF, Abf, su, sw, sg,
    gscale, eps,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr,
):
    """One token (chunk row r, window index j; its inputs are row xr + j of q, k, v and row gr + j of g, beta):
    everything K1 computes for it. Rows < r of the chunk caches are final (earlier steps or earlier iterations of this
    window)."""
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
    p_q = q_in + (xr + j) * sqk + i_hk * K
    p_k = k_in + (xr + j) * sqk + i_hk * K
    rq = _rstd(tl.load(p_q + o_K[None, :] + o_i[:, None] * 0), eps)
    rk = _rstd(tl.load(p_k + o_K[None, :] + o_i[:, None] * 0), eps)
    for i_k in tl.static_range(K // 64):
        b_qx = tl.load(p_q + i_k * 64 + o_64[None, :] + o_i[:, None] * 0)
        tl.store(sq + j * K + i_k * 64 + o_64[None, :] + o_i[:, None] * 0,
                 (b_qx.to(tl.float32) * rq[:, None]).to(b_qx.dtype), mask=(o_i == 0)[:, None])
        b_kx = tl.load(p_k + i_k * 64 + o_64[None, :] + o_i[:, None] * 0)
        tl.store(k + r * (H*K) + i_k * 64 + o_64[None, :] + o_i[:, None] * 0,
                 (b_kx.to(tl.float32) * rk[:, None]).to(b_kx.dtype), mask=(o_i == 0)[:, None])
    tl.store(v + r * (HV*V) + o_V, tl.load(v_in + (xr + j) * sv + i_h * V + o_V))
    tl.store(g + r * HV, tl.load(g_in + (gr + j) * HV + i_h))
    tl.store(beta + r * HV, tl.load(b_in + (gr + j) * HV + i_h))
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

    # ---- recompute_w_u: rows of u, w for the token (16-row tile, row i stored for K2) ---------------------------
    m_rs = (o_i <= i)[:, None]
    b_b = tl.load(tl.make_block_ptr(beta, (T,), (HV,), (0,), (BT,), (0,)), boundary_check=(0,))
    b_A = tl.load(Abf + (tb + o_i)[:, None] * BT + o_t[None, :], mask=m_rs, other=0.)
    b_vin = tl.load(tl.make_block_ptr(v, (T, V), (HV*V, 1), (0, 0), (BT, V), (1, 0)), boundary_check=(0, 1))
    b_vb = (b_vin * b_b[:, None]).to(b_vin.dtype)
    b_u = tl.dot(b_A, b_vb, allow_tf32=False).to(b_vin.dtype)
    tl.store(su + j * V + o_V[None, :] + o_i[:, None] * 0, b_u, mask=m_row)
    b_ge = tl.math.exp2(tl.load(tl.make_block_ptr(sg, (T,), (1,), (0,), (BT,), (0,)), boundary_check=(0,)))
    for i_k in tl.static_range(K // 64):
        b_kk = tl.load(tl.make_block_ptr(k, (T, K), (H*K, 1), (0, i_k * 64), (BT, 64), (1, 0)), boundary_check=(0, 1))
        b_kb = b_kk * b_b[:, None]
        b_kb *= b_ge[:, None]
        b_w = tl.dot(b_A, b_kb.to(b_kk.dtype)).to(b_kk.dtype)
        tl.store(sw + j * K + i_k * 64 + o_64[None, :] + o_i[:, None] * 0, b_w, mask=m_row)




@triton.jit
def _gdn_k1(
    q_in, k_in, v_in, g_in, b_in, xrow, grow, sqk, sv, pos, nrows, pending, kbuf, vbuf, gbuf, bbuf, scr_q, scr_u,
    scr_w, AiF, Abf, sgc, slots,
    sSl: tl.constexpr, sSu: tl.constexpr, sHb: tl.constexpr, skb: tl.constexpr, svb: tl.constexpr,
    sgb: tl.constexpr, sbb: tl.constexpr, sAi: tl.constexpr, sAb: tl.constexpr, sVN: tl.constexpr, ssg: tl.constexpr,
    gscale, eps,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr,
    KT: tl.constexpr, SINGLE: tl.constexpr,
):
    """Rows pos % 64 .. pos % 64 + nrows - 1 of one (slot, head), the window's tokens being rows xrow .. of q, k, v
    (row stride sqk, sv) and rows grow .. of g, beta; the window's l2norm(q), u and w rows, which only K2 reads, go to
    the step's scratch ([n, HV, KT, K or V]); the slot's head-0 program then resets the fold flag (K0 has run)."""
    i_nh = tl.program_id(0)
    i_n, i_h = i_nh // HV, i_nh % HV
    i_hk = i_h // (HV // H)
    slot = tl.load(slots + i_n).to(tl.int64)
    if slot < 0:                        # padding entry of a CUDA-graph batch (engine's PAD_SLOT_ID)
        return
    k = kbuf + slot * skb + i_hk * K
    v = vbuf + slot * svb + i_h * V
    g = gbuf + slot * sgb + i_h
    beta = bbuf + slot * sbb + i_h
    AiF += slot * sAi + i_h * BT * BT
    Abf += slot * sAb + i_h * BT * BT
    sg = sgc + slot * ssg + i_h * BT
    sq = scr_q + (i_n * HV + i_h) * KT * K
    su = scr_u + (i_n * HV + i_h) * KT * V
    sw = scr_w + (i_n * HV + i_h) * KT * K
    r0 = tl.load(pos + i_n) % BT
    nr = tl.load(nrows + i_n)
    xr = tl.load(xrow + i_n).to(tl.int64)
    gr = tl.load(grow + i_n).to(tl.int64)
    if SINGLE:                          # one token: no loop (a runtime loop costs registers and occupancy)
        _k1_row(r0, 0, xr, gr, i_h, i_hk, q_in, k_in, v_in, g_in, b_in, sqk, sv, k, v, g, beta, sq, AiF, Abf, su,
                sw, sg, gscale, eps, H, HV, K, V, BT, BC)
    else:
        for j in range(nr):
            _k1_row(r0 + j, j, xr, gr, i_h, i_hk, q_in, k_in, v_in, g_in, b_in, sqk, sv, k, v, g, beta, sq, AiF,
                    Abf, su, sw, sg, gscale, eps, H, HV, K, V, BT, BC)
            tl.debug_barrier()
    if i_h == 0:
        tl.store(pending + slot, 0)                                  # K0 has run


@triton.jit
def _gdn_k2(
    pos, nrows, kbuf, vbuf, bbuf, Abf, scr_q, scr_u, scr_w, Hb, o, orow, VNc, sgc, Slo, Sup, slots,
    sSl: tl.constexpr, sSu: tl.constexpr, sHb: tl.constexpr, skb: tl.constexpr, svb: tl.constexpr,
    sgb: tl.constexpr, sbb: tl.constexpr, sAi: tl.constexpr, sAb: tl.constexpr, sVN: tl.constexpr, ssg: tl.constexpr,
    scale, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr,
    BT: tl.constexpr, BC: tl.constexpr, KT: tl.constexpr, TILE: tl.constexpr, EAGER_FOLD: tl.constexpr,
):
    """Window rows r0 .. r1 = r0 + nrows - 1 of one (V block, slot, head) on one TILE-row tile starting at r0's
    sub-block (16 rows for one token, 32 for a window of up to 16): v_new = u - w bf16(S) for the window's rows (fwd_h),
    then their output rows (chunk_fwd_kernel_o), which need the v_new rows of the whole window. Tile rows outside the
    window are zero (MMA rows are independent). Output row j of the window goes to row orow + j of o [rows, HV, V]."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // HV, i_nh % HV
    r0 = tl.load(pos + i_n) % BT
    nr = tl.load(nrows + i_n)
    r1 = r0 + nr - 1
    T = r1 + 1
    slot = tl.load(slots + i_n).to(tl.int64)
    if slot < 0:                        # padding entry of a CUDA-graph batch (engine's PAD_SLOT_ID)
        return
    k = kbuf + slot * skb + (i_h // (HV // H)) * K
    VNc += slot * sVN + i_h * BT * V
    sg = sgc + slot * ssg + i_h * BT
    sq = scr_q + (i_n * HV + i_h) * KT * K - r0 * K             # scratch rows by chunk row (window row j = row - r0)
    su = scr_u + (i_n * HV + i_h) * KT * V - r0 * V
    sw = scr_w + (i_n * HV + i_h) * KT * K - r0 * K
    Hb_ = Hb + slot * sHb + i_h * K * V
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
        b_w = tl.load(sw + rows[:, None] * K + i_k * 64 + o_64[None, :], mask=m_win, other=0.)
        b_h = tl.load(tl.make_block_ptr(Hb_, (K, V), (V, 1), (i_k * 64, i_v * BV), (64, BV), (1, 0)))
        b_q = tl.load(sq + rows[:, None] * K + i_k * 64 + o_64[None, :], mask=m_win, other=0.)
        b_kt = tl.load(tl.make_block_ptr(k, (K, T), (1, H*K), (i_k * 64, 0), (64, BT), (0, 1)), boundary_check=(0, 1))
        if i_k == 0:
            b_v = tl.dot(b_w, b_h)                                  # fwd_h: b_v = dot(w1, h1); b_v += dot(w2, h2)
            b_h1 = b_h
        else:
            b_v += tl.dot(b_w, b_h)
            b_h2 = b_h
        b_o += tl.dot(b_q, b_h)                                     # chunk_fwd_kernel_o: zeros += dot over K
        b_Aq += tl.dot(b_q, b_kt)
    b_u = tl.load(su + rows[:, None] * V + o_v[None, :], mask=m_win, other=0.)
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
    b_o = _o_epilogue(b_o, tl.dot(b_Aq.to(b_vn.dtype), b_vn), scale, O_FORM)
    tl.store(o + ((tl.load(orow + i_n).to(tl.int64) + rows - r0) * HV + i_h)[:, None] * V + o_v[None, :],
             b_o.to(o.dtype.element_ty),
             mask=m_win)

    # ---- plain decode folds a completed chunk right away (fwd_h's 64-row arithmetic, _fold_vf) ---------------------
    if EAGER_FOLD:
        if T == BT:
            Slo_ = Slo + slot * sSl + i_h * K * V
            Sup_ = Sup + slot * sSu + i_h * K * (V // 32)
            if K == 64:
                b_h2 = b_h1
            b_vf = _fold_vf(k, vbuf + slot * svb + i_h * V, bbuf + slot * sbb + i_h, Abf + slot * sAb + i_h * BT * BT,
                            sg, b_h1, b_h2, i_v * BV, H, HV, K, V, BT, BV)
            b_glast = tl.load(sg + (BT - 1)).to(tl.float32)
            b_gl = tl.load(sg + o_t).to(tl.float32)
            b_vf = b_vf * tl.where(m_t, tl.math.exp2(b_glast - b_gl), 0)[:, None]
            b_glast = tl.math.exp2(b_glast)
            b_vf = b_vf.to(kbuf.dtype.element_ty)
            for i_k in tl.static_range(K // 64):
                if i_k == 0:
                    b_s = _state_tile(b_h1, Slo_, Sup_, 0, i_v * BV, V, BV)
                else:
                    b_s = _state_tile(b_h2, Slo_, Sup_, 64, i_v * BV, V, BV)
                b_s *= b_glast
                b_s += tl.dot(tl.load(tl.make_block_ptr(k, (K, BT), (1, H*K), (i_k * 64, 0), (64, BT), (0, 1))), b_vf)
                _state_put(b_s, Hb_, Slo_, Sup_, i_k * 64, i_v * BV, V, BV)


@triton.jit
def _gdn_state_io(S, Hb, Slo, Sup, slots, sSl: tl.constexpr, sSu: tl.constexpr, sHb: tl.constexpr,
                  HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, PUT: tl.constexpr):
    """Rows i_k * 64 .. of entry i_n's head i_h: S [n, HV, K, V] fp32 (contiguous) to the slot's stored state (PUT)
    or back."""
    i_nh, i_k = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // HV, i_nh % HV
    slot = tl.load(slots + i_n).to(tl.int64)
    Hb_ = Hb + slot * sHb + i_h * K * V
    Slo_ = Slo + slot * sSl + i_h * K * V
    Sup_ = Sup + slot * sSu + i_h * K * (V // 32)
    p_s = tl.make_block_ptr(S + i_nh.to(tl.int64) * K * V, (K, V), (V, 1), (i_k * 64, 0), (64, V), (1, 0))
    if PUT:
        _state_put(tl.load(p_s), Hb_, Slo_, Sup_, i_k * 64, 0, V, V)
    else:
        b_h = tl.load(tl.make_block_ptr(Hb_, (K, V), (V, 1), (i_k * 64, 0), (64, V), (1, 0)))
        tl.store(p_s, _state_tile(b_h, Slo_, Sup_, i_k * 64, 0, V, V))


def state_layout(H, HV, K, V, dtype=torch.bfloat16, beta_dtype=None):
    """Per-slot caches as (name, dtype, shape) and each one's offset in 32-bit words inside one slot's storage, plus
    the slot's total size in words. One slot = one sequence of one GDN layer."""
    bufs = [('Slo', torch.int16, (HV, K, V)), ('Sup', torch.int32, (HV, K // 32, V)), ('Hb', dtype, (HV, K, V)),
            ('kbuf', dtype, (BT, H, K)),
            ('vbuf', dtype, (BT, HV, V)), ('gbuf', torch.float32, (BT, HV)), ('bbuf', beta_dtype or dtype, (BT, HV)),
            ('AiF', torch.float32, (HV, BT, BT)), ('Abf', dtype, (HV, BT, BT)), ('VNc', dtype, (HV, BT, V)),
            ('sg', torch.float32, (HV, BT))]
    out, words = [], 0
    for name, dt, shape in bufs:
        n = 1
        for d in shape:
            n *= d
        nw = (n * torch.tensor([], dtype=dt).element_size() + 3) // 4
        out.append((name, dt, shape, words))
        words += (nw + 3) // 4 * 4                        # 16-byte aligned buffers
    return out, words


class GDNExactDecode:
    """Exact decode state for N slots of one GDN layer (K in {64, 128}, V % 64 == 0, V % BV == 0).

    A step processes a window of up to KT consecutive tokens per slot (1 for plain decode; 1 + draft length for
    speculative verification), starting at the slot's absolute position `pos`, truncated at the slot's next 64-token
    boundary (`window`). Its outputs are the trainer forward's rows for those tokens, whatever the window size.
    `commit(pos, accepted)` records how many of the window's tokens were kept; the caller then advances
    pos += accepted. Rejected tokens leave nothing behind: rows beyond the accepted ones are overwritten by the next
    window, and the state is folded only when a chunk's 64th row has been accepted (lazily, at the start of the next
    step), so rollback is free. With KT == 1 use decode(): every token is accepted and folds happen in the step.

    All caches of a slot live in one row of `storage` ([N, words] fp32, see state_layout), so an engine can keep them
    in its own per-sequence state pool; `slots` arguments then name the pool rows a batch uses (default: 0..N-1).
    step_rows() is step()/decode() for such an engine: it reads each window's tokens where they lie in the engine's
    activations (row indices, any row stride) and writes its output rows in place, so a step needs no gathers."""

    def __init__(self, N, H, HV, K, V, device, KT=1, dtype=torch.bfloat16, beta_dtype=None, BV=64, nw1=2, nw2=4,
                 eps=1e-6, storage=None):
        assert K in (64, 128) and V % 64 == 0 and V % BV == 0 and 1 <= KT <= BC
        cap = torch.cuda.get_device_capability(device)
        if cap not in _VALIDATED_GPUS:
            raise RuntimeError(f'exact GDN decode: sm_{cap[0]}{cap[1]} unverified; run gdn/test_exact.py there, then add it '
                               f'to _VALIDATED_GPUS')
        self.N, self.H, self.HV, self.K, self.V, self.KT = N, H, HV, K, V, KT
        self.BV, self.nw1, self.nw2, self.eps = BV, nw1, nw2, eps
        layout, words = state_layout(H, HV, K, V, dtype, beta_dtype)
        if storage is None:
            storage = torch.zeros(N, words, device=device, dtype=torch.float32)
        assert storage.dtype == torch.float32 and storage.shape[1] >= words and storage.stride(1) == 1
        self.storage = storage
        for name, dt, shape, off in layout:
            n = 1
            for d in shape:
                n *= d
            raw = storage[:, off:off + (n * torch.tensor([], dtype=dt).element_size() + 3) // 4]
            setattr(self, name, raw.view(dt)[:, :n].unflatten(1, shape))
        # fold flags per slot, plus one entry that padded batch entries (slot < 0) write to
        self.pending = torch.zeros(storage.shape[0] + 1, device=device, dtype=torch.int32)
        self._ones = torch.ones(N, device=device, dtype=torch.int32)
        self._slots = torch.arange(N, device=device, dtype=torch.int32)
        # per-slot element strides of every cache, in the order the kernels take them (compile-time constants there)
        self._st = tuple(getattr(self, n).stride(0) for n in
                         ('Slo', 'Sup', 'Hb', 'kbuf', 'vbuf', 'gbuf', 'bbuf', 'AiF', 'Abf', 'VNc', 'sg'))
        self._rows = {}

    @staticmethod
    def window(pos, k):
        """Rows a step may process: min(k, tokens left in the slot's current chunk)."""
        return torch.clamp(BT - pos % BT, max=k).to(torch.int32)

    def _rows_of(self, n, KT, device):
        """Row i * KT for entries i = 0..n-1 (the [n, KT, ...] layout), int32; cached."""
        r = self._rows.get((n, KT))
        if r is None:
            r = self._rows[(n, KT)] = torch.arange(0, n * KT, KT, device=device, dtype=torch.int32)
        return r

    def _launch(self, slots, q, k, v, g, beta, pos, nrows, scale, KT, rows=None, rows_g=None, out=None, rows_out=None):
        """Batch entry i uses cache slot slots[i]; its window's token j is row rows[i] + j of q, k [R, H*K] and v
        [R, HV*V] (any row stride, unit column stride) and row rows_g[i] + j of g, beta [R', HV] (contiguous), and its
        output row j goes to row rows_out[i] + j of out [R'', HV, V] (contiguous; rows past the window are not
        written). Defaults: rows i * KT, and a new out [n * KT, HV, V]. slots, pos, nrows, rows*: int32, unit stride
        (an engine's column view of its block table, e.g. vLLM's with speculative decoding, must be compacted first).
        KT == 1 (plain decode, every token accepted): K1 + K2 with a 16-row tile and the fold done eagerly by K2 on the
        chunk's 64th row. KT > 1 (windows that may be partly rejected): K0 (lazy fold of an accepted chunk) + K1 + K2
        with a 32-row tile."""
        HV, n = self.HV, slots.shape[0]
        if rows is None:
            rows = self._rows_of(n, KT, q.device)
        if out is None:
            out, rows_out = torch.empty(n * KT, HV, self.V, device=q.device, dtype=v.dtype), self._rows_of(n, KT, q.device)
        if KT > 1:
            self.fold(slots)
        # the window's l2norm(q), u, w rows: K1 writes them, only K2 reads them
        scr = torch.empty(n * HV * KT * (2 * self.K + self.V), device=q.device, dtype=self.kbuf.dtype)
        scr_q, scr_w, scr_u = scr.split([n * HV * KT * self.K] * 2 + [n * HV * KT * self.V])
        _gdn_k1[(n * HV,)](
            q, k, v, g, beta, rows, rows if rows_g is None else rows_g, q.stride(0), v.stride(0), pos, nrows,
            self.pending, self.kbuf, self.vbuf, self.gbuf, self.bbuf, scr_q, scr_u, scr_w, self.AiF, self.Abf,
            self.sg, slots, *self._st, RCP_LN2, self.eps, H=self.H, HV=HV, K=self.K, V=self.V, BT=BT, BC=BC,
            KT=KT, SINGLE=KT == 1, num_warps=self.nw1)
        _gdn_k2[(self.V // self.BV, n * HV)](
            pos, nrows, self.kbuf, self.vbuf, self.bbuf, self.Abf, scr_q, scr_u, scr_w, self.Hb, out, rows_out,
            self.VNc, self.sg, self.Slo, self.Sup, slots, *self._st, scale, H=self.H, HV=HV, K=self.K, V=self.V,
            BV=self.BV, BT=BT, BC=BC, KT=KT, TILE=BC if KT == 1 else 2 * BC, EAGER_FOLD=KT == 1, num_warps=self.nw2)
        return out

    def _check(self, q, k, v, g, beta, n, KT):
        H, HV, K, V = self.H, self.HV, self.K, self.V
        for x, shape in ((q, (n, KT, H, K)), (k, (n, KT, H, K)), (v, (n, KT, HV, V)), (g, (n, KT, HV)), (beta, (n, KT, HV))):
            assert tuple(x.shape) == shape and x.is_contiguous(), (tuple(x.shape), shape)
        self._check_dtypes(k, g, beta)

    def _check_dtypes(self, k, g, beta):
        # fla's bits depend on these dtypes (beta enters recompute_w_u unconverted; g must be the fp32 log-gate)
        assert beta.dtype == self.bbuf.dtype and g.dtype == torch.float32 and k.dtype == self.kbuf.dtype

    def _state_slots(self, slots, distinct):
        """slots as contiguous int32, checked on the device without a host sync: in range and, for a store,
        distinct (two programs storing one slot would race on its packed Sup words)."""
        slots, = self._i32(torch.as_tensor(slots, device=self.Hb.device))
        if slots.numel():
            s = slots.sort().values
            torch._assert_async((s[0] >= 0) & (s[-1] < self.Hb.shape[0]) & ((s[1:] != s[:-1]).all() | (not distinct)))
        return slots

    @staticmethod
    def _i32(*ts):
        return [t if t.dtype == torch.int32 and t.is_contiguous() else t.to(torch.int32).contiguous() for t in ts]

    def step(self, q, k, v, g, beta, pos, nrows, scale, slots=None):
        """q, k [n,KT,H,K] raw (normalised in-kernel), v [n,KT,HV,V], g [n,KT,HV] raw log-gate, beta [n,KT,HV];
        pos [n] int32 absolute position of the window's first token; nrows [n] int32 = window(pos, KT).
        Returns o [n,KT,HV,V] (rows >= nrows undefined). Three launches (K0 fold, K1, K2); graph-capturable."""
        slots = self._slots if slots is None else slots
        n, KT = slots.shape[0], self.KT
        assert KT > 1, 'KT == 1: use decode()'
        self._check(q, k, v, g, beta, n, KT)
        x = [t.view(n * KT, -1) for t in (q, k, v, g, beta)]
        return self._launch(*self._i32(slots), *x, *self._i32(pos, nrows), scale, KT).view(n, KT, self.HV, self.V)

    def step_rows(self, q, k, v, g, beta, rows, rows_g, pos, nrows, scale, slots, out, rows_out, KT=None):
        """step() (KT > 1, default this object's KT) or decode() (KT == 1, nrows 1) on tokens read where they lie and
        outputs written where they go: entry i's token j is row rows[i] + j of q, k [R, H*K] and v [R, HV*V] (any row
        stride, e.g. column slices of an engine's fused qkv activations) and row rows_g[i] + j of g, beta [R', HV]; its
        output is row rows_out[i] + j of out [R'', HV, V] (rows past the window are left as they are). slots, pos,
        nrows, rows*: int32 [n], unit stride."""
        KT = self.KT if KT is None else KT
        H, HV, K, V = self.H, self.HV, self.K, self.V
        assert (q.shape[1] == k.shape[1] == H * K and v.shape[1] == HV * V and q.stride(1) == k.stride(1) == 1
                and k.stride(0) == q.stride(0) and v.stride(1) == 1 and g.is_contiguous() and beta.is_contiguous()
                and g.shape[1] == HV and out.is_contiguous() and tuple(out.shape[1:]) == (HV, V))
        self._check_dtypes(k, g, beta)
        return self._launch(slots, q, k, v, g, beta, pos, nrows, scale, KT, rows, rows_g, out, rows_out)

    def fold(self, slots):
        """Fold the completed chunk of every slot marked pending into its state now (K0; normally the start of the slot's
        next window). For a slot whose next step is not a window, e.g. an engine's prefill continuing it. The caller
        clears pending."""
        slots, = self._i32(slots)
        _gdn_k0[(self.V // 64, slots.shape[0] * self.HV)](
            self.pending, self.kbuf, self.vbuf, self.bbuf, self.Abf, self.sg, self.Slo, self.Sup, self.Hb, slots,
            *self._st,
            H=self.H, HV=self.HV, K=self.K, V=self.V, BT=BT, num_warps=4)

    def commit(self, pos, accepted, slots=None):
        """accepted [n] int32 (0..nrows) tokens of the window just stepped are kept; the caller then sets
        pos += accepted. Marks a fold for the next step when the window's acceptance completed a chunk."""
        slots = self._slots if slots is None else slots
        idx = torch.where(slots < 0, self.pending.shape[0] - 1, slots).long()
        self.pending[idx] |= (pos % BT + accepted == BT).to(torch.int32)

    def decode(self, q, k, v, g, beta, pos, scale, slots=None):
        """Plain one-token decode (needs KT == 1): q, k [n,H,K], v [n,HV,V], g, beta [n,HV]; returns o [n,HV,V].
        Every token is accepted, so a completed chunk is folded in the same step. The caller then increments pos."""
        assert self.KT == 1
        slots = self._slots if slots is None else slots
        n = slots.shape[0]
        self._check(*[t[:, None] for t in (q, k, v, g, beta)], n, 1)
        ones = self._ones if n == self.N else torch.ones_like(slots)
        x = [t.view(n, -1) for t in (q, k, v, g, beta)]
        return self._launch(*self._i32(slots), *x, *self._i32(pos), ones, scale, 1)

    def get_states(self, slots):
        """The fp32 boundary states [n, HV, K, V] of slots ([n]; the stored form is Hb, Slo, Sup)."""
        slots = self._state_slots(slots, False)
        S = torch.empty(slots.shape[0], self.HV, self.K, self.V, device=self.Hb.device, dtype=torch.float32)
        _gdn_state_io[(slots.shape[0] * self.HV, self.K // 64)](S, self.Hb, self.Slo, self.Sup, slots, *self._st[:3],
                                                                HV=self.HV, K=self.K, V=self.V, PUT=False)
        return S

    def put_states(self, slots, S):
        """Store the fp32 boundary states S [n, HV, K, V] into slots ([n], distinct)."""
        slots = self._state_slots(slots, True)
        assert S.dtype == torch.float32 and tuple(S.shape) == (slots.shape[0], self.HV, self.K, self.V)
        _gdn_state_io[(slots.shape[0] * self.HV, self.K // 64)](S.contiguous(), self.Hb, self.Slo, self.Sup, slots,
                                                                *self._st[:3], HV=self.HV, K=self.K, V=self.V,
                                                                PUT=True)

    def set_state(self, slot, S):
        """Slot `slot` starts from the boundary state S [HV,K,V] fp32."""
        self.put_states(torch.tensor([slot], device=S.device, dtype=torch.int32), S[None])
        self.pending[slot] = 0

    def replay(self, slot, p, q, k, v, g, beta):
        """Append tokens at absolute positions p .. p+r-1 of slot `slot`, all inside one 64-token chunk (rows before p
        of that chunk are already cached), to the row caches, in windows of up to 16 (outputs discarded).
        q, k [r,H,K] raw, v [r,HV,V], g, beta [r,HV]."""
        r = k.shape[0]
        assert r == 0 or p // BT == (p + r - 1) // BT, 'replay must stay inside one chunk'
        dev = k.device
        sl = torch.tensor([slot], device=dev, dtype=torch.int32)
        for j0 in range(0, r, BC):
            w = min(BC, r - j0, BT - (p + j0) % BT)
            x = [t[None, j0:j0 + w].contiguous() for t in (q, k, v, g, beta)]
            self._check(*x, 1, w)
            self._launch(sl, *[t.view(w, -1) for t in x], torch.tensor([p + j0], device=dev, dtype=torch.int32),
                         torch.tensor([w], device=dev, dtype=torch.int32), 1.0, w)

    def load(self, n, S, b, q, k, v, g, beta):
        """Slot n resumes at its boundary state S [HV,K,V] fp32 taken at absolute position b (multiple of 64), then
        replays the r = P - b < 64 partial-chunk tokens (raw q, k [r,H,K], v [r,HV,V], g, beta [r,HV])."""
        assert b % BT == 0 and k.shape[0] < BT
        self.set_state(n, S)
        self.replay(n, b, q, k, v, g, beta)
