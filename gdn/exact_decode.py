"""Exact GDN decode: each decode step returns, bitwise, the row that fla 0.5.0's chunk_gated_delta_rule
(use_qk_l2norm_in_kernel=True, its default use_exp2 path) produces for that token in a full training forward.

Per slot the fp32 state is kept only at absolute 64-token boundaries (S), with the bf16 copy fla's fwd_h feeds to
every intra-chunk dot (Hb). Every intermediate of fla's chunk program is row-causal inside a chunk, so a step for
the token at chunk row r (sub-block s = r // 16, i = r % 16) computes only row r and reuses cached rows < r:
  K1 (one program per slot x head, never reads the state): l2norm of q, k (fla l2norm_fwd_kernel), the token's row
     of the chunk buffers, cumsum(g) (chunk_local_cumsum), the row of (I + A)^-1 (kkt_solve: kkt tiles, the one
     forward-substitution iteration for row i, block merge from cached final blocks), and the rows of u, w
     (recompute_w_u) on the 16-row tile of sub-block s.
  K2 (one program per V-block x slot x head): v_new row = u - w bf16(S) (fwd_h), the output row (chunk_fwd_kernel_o),
     and on the 64th row the fold of the chunk into S with fwd_h's 64-row arithmetic from the cached u, w rows.
Per step K2 reads bf16(S) once and nothing writes the state except the fold; fla's fused_recurrent reads and writes
the fp32 state every step.

Why bitwise: every expression mirrors the fla source it is named after, with the same operand dtypes, the same
tl.dot shapes per row (a row of an MMA tile does not depend on the other rows) and the same accumulation forms.
Two traps found on the way, both pinned here: LLVM fp-contraction of chunk_fwd_kernel_o's epilogue (written out as
fla's PTX does it, an explicit fma), and dot operands produced in registers through casts or splices, which Triton
compiles to different numerics than operands loaded from memory (so the token's rows go through memory, with a
barrier). Triton's layout choices are global to a kernel, so restructuring "pure data movement" can still change
the bits; test_exact.py is the arbiter after any edit. Verified on A800 (sm_80), Triton 3.7.1, fla 0.5.0.
gdn/exact_decode_ref.py is the simplest (stateless, slower) implementation of the same contract."""
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
def _gdn_k1(
    q_in, k_in, v_in, g_in, b_in, pos, kbuf, vbuf, gbuf, bbuf, qn, AiF, Abf, Uc, Wc, sgc,
    gscale, eps,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr,
):
    i_nh = tl.program_id(0)
    i_n, i_h = i_nh // HV, i_nh % HV
    i_hk = i_h // (HV // H)
    r = tl.load(pos + i_n) % BT
    T = r + 1
    s = r // BC
    i = r - s * BC
    tb = s * BC
    k = kbuf + (i_n * BT * H + i_hk) * K
    v = vbuf + (i_n * BT * HV + i_h) * V
    g = gbuf + i_n * BT * HV + i_h
    beta = bbuf + i_n * BT * HV + i_h
    AiF += i_nh * BT * BT
    Abf += i_nh * BT * BT
    Uc += i_nh * BT * V
    Wc += i_nh * BT * K
    sg = sgc + i_nh * BT
    o_i = tl.arange(0, BC)
    o_t = tl.arange(0, BT)
    o_64 = tl.arange(0, 64)
    o_K = tl.arange(0, K)
    o_V = tl.arange(0, V)
    m_row = (o_i == i)[:, None]

    # ---- token: l2norm(q), l2norm(k) (fla l2norm_fwd_kernel), chunk-buffer row -----------------------------------
    p_q = q_in + (i_n * H + i_hk) * K
    p_k = k_in + (i_n * H + i_hk) * K
    rq = _rstd(tl.load(p_q + o_K[None, :] + o_i[:, None] * 0), eps)
    rk = _rstd(tl.load(p_k + o_K[None, :] + o_i[:, None] * 0), eps)
    for i_k in tl.static_range(K // 64):
        b_qx = tl.load(p_q + i_k * 64 + o_64[None, :] + o_i[:, None] * 0)
        tl.store(qn + i_nh * K + i_k * 64 + o_64[None, :] + o_i[:, None] * 0,
                 (b_qx.to(tl.float32) * rq[:, None]).to(b_qx.dtype), mask=(o_i == 0)[:, None])
        b_kx = tl.load(p_k + i_k * 64 + o_64[None, :] + o_i[:, None] * 0)
        tl.store(k + r * (H*K) + i_k * 64 + o_64[None, :] + o_i[:, None] * 0,
                 (b_kx.to(tl.float32) * rk[:, None]).to(b_kx.dtype), mask=(o_i == 0)[:, None])
    tl.store(v + r * (HV*V) + o_V, tl.load(v_in + (i_n * HV + i_h) * V + o_V))
    tl.store(g + r * HV, tl.load(g_in + i_n * HV + i_h))
    tl.store(beta + r * HV, tl.load(b_in + i_n * HV + i_h))
    tl.debug_barrier()

    # ---- chunk_local_cumsum_scalar_kernel ----------------------------------------------------------------------
    b_s = tl.load(tl.make_block_ptr(g, (T,), (HV,), (0,), (BT,), (0,)), boundary_check=(0,)).to(tl.float32)
    b_gc = tl.cumsum(b_s, axis=0)
    b_gc *= gscale
    tl.store(tl.make_block_ptr(sg, (T,), (1,), (0,), (BT,), (0,)), b_gc.to(tl.float32), boundary_check=(0,))
    tl.debug_barrier()

    # ---- kkt_solve row (identical to v2) -------------------------------------------------------------------------
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
def _gdn_k2(
    pos, kbuf, qn, S, Hb, o, Uc, Wc, VNc, sgc,
    scale,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr,
    BT: tl.constexpr, BC: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // HV, i_nh % HV
    r = tl.load(pos + i_n) % BT
    T = r + 1
    s = r // BC
    i = r - s * BC
    tb = s * BC
    k = kbuf + (i_n * BT * H + i_h // (HV // H)) * K
    Uc += i_nh * BT * V
    Wc += i_nh * BT * K
    VNc += i_nh * BT * V
    sg = sgc + i_nh * BT
    Hb_ = Hb + i_nh * K * V
    S_ = S + i_nh * K * V
    o_i = tl.arange(0, BC)
    o_t = tl.arange(0, BT)
    o_64 = tl.arange(0, 64)
    o_v = i_v * BV + tl.arange(0, BV)
    m_row = (o_i == i)[:, None]
    m_t = o_t < T
    m_ts = (tb + o_i) < T

    # ---- fwd_h: v_new row = u - w bf16(S) (tile rows other than i are zero; MMA rows are independent) ----------
    b_v = tl.zeros([BC, BV], dtype=tl.float32)
    b_o = tl.zeros([BC, BV], dtype=tl.float32)
    b_Aq = tl.zeros([BC, BT], dtype=tl.float32)
    for i_k in tl.static_range(K // 64):
        b_w = tl.load(Wc + (tb + o_i)[:, None] * K + i_k * 64 + o_64[None, :], mask=m_row, other=0.)
        b_h = tl.load(tl.make_block_ptr(Hb_, (K, V), (V, 1), (i_k * 64, i_v * BV), (64, BV), (1, 0)))
        b_q = tl.load(qn + i_nh * K + i_k * 64 + o_64[None, :] + o_i[:, None] * 0, mask=m_row, other=0.)
        b_kt = tl.load(tl.make_block_ptr(k, (K, T), (1, H*K), (i_k * 64, 0), (64, BT), (0, 1)), boundary_check=(0, 1))
        if i_k == 0:
            b_v = tl.dot(b_w, b_h)                                  # fwd_h: b_v = dot(w1, h1); b_v += dot(w2, h2)
        else:
            b_v += tl.dot(b_w, b_h)
        b_o += tl.dot(b_q, b_h)                                     # chunk_fwd_kernel_o: zeros += dot over K
        b_Aq += tl.dot(b_q, b_kt)
    b_u = tl.load(Uc + (tb + o_i)[:, None] * V + o_v[None, :], mask=m_row, other=0.)
    b_v = b_u.to(tl.float32) - b_v
    tl.store(VNc + (tb + o_i)[:, None] * V + o_v[None, :], b_v.to(VNc.dtype.element_ty), mask=m_row)
    tl.debug_barrier()

    # ---- chunk_fwd_kernel_o: row i ---------------------------------------------------------------------------------
    b_gs = tl.load(sg + tb + o_i, mask=m_ts, other=0.).to(tl.float32)
    b_gall = tl.load(tl.make_block_ptr(sg, (T,), (1,), (0,), (BT,), (0,)), boundary_check=(0,))
    b_o = b_o * tl.math.exp2(b_gs)[:, None]
    b_Aq = b_Aq * tl.math.exp2(b_gs[:, None] - b_gall[None, :])
    m_A = ((tb + o_i)[:, None] >= o_t[None, :]) & (m_ts[:, None] & m_t)
    b_Aq = tl.where(m_A, b_Aq, 0)
    b_vn = tl.load(tl.make_block_ptr(VNc, (T, V), (V, 1), (0, i_v * BV), (BT, BV), (1, 0)), boundary_check=(0, 1))
    b_o = tl.fma(tl.dot(b_Aq.to(b_vn.dtype), b_vn), scale, b_o * scale)     # = fla's contraction, see v1
    b_orow = tl.sum(tl.where(m_row, b_o, 0.), 0)
    tl.store(o + (i_n * HV + i_h) * V + o_v, b_orow.to(o.dtype.element_ty))

    # ---- fold a complete chunk (fwd_h's 64-row arithmetic from the cached u, w rows) ---------------------------
    if T == BT:
        b_h1 = tl.load(tl.make_block_ptr(Hb_, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)))
        b_vf = tl.dot(tl.load(tl.make_block_ptr(Wc, (BT, K), (K, 1), (0, 0), (BT, 64), (1, 0))), b_h1)
        if K > 64:
            b_h2 = tl.load(tl.make_block_ptr(Hb_, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)))
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
            tl.store(tl.make_block_ptr(Hb_, (K, V), (V, 1), (i_k * 64, i_v * BV), (64, BV), (1, 0)), b_s.to(Hb.dtype.element_ty))


class GDNExactDecode:
    """Exact decode state for N slots of one GDN layer (K in {64, 128}, V % BV == 0). The caller owns the per-slot
    absolute positions `pos` (index of the token being decoded) and advances them after each step."""

    def __init__(self, N, H, HV, K, V, device, dtype=torch.bfloat16, BV=64, nw1=4, nw2=4, eps=1e-6):
        assert K in (64, 128) and V % BV == 0
        self.N, self.H, self.HV, self.K, self.V, self.BV, self.nw1, self.nw2, self.eps = N, H, HV, K, V, BV, nw1, nw2, eps
        z = lambda *s, dt=dtype: torch.zeros(*s, device=device, dtype=dt)
        self.S, self.Hb = z(N, HV, K, V, dt=torch.float32), z(N, HV, K, V)
        self.kbuf, self.vbuf = z(N, BT, H, K), z(N, BT, HV, V)
        self.gbuf, self.bbuf = z(N, BT, HV, dt=torch.float32), z(N, BT, HV)
        self.AiF, self.Abf = z(N * HV, BT, BT, dt=torch.float32), z(N * HV, BT, BT)
        self.Uc, self.Wc, self.VNc = z(N * HV, BT, V), z(N * HV, BT, K), z(N * HV, BT, V)
        self.sg, self.qn = z(N * HV, BT, dt=torch.float32), z(N * HV, K)

    def _launch(self, n0, ns, q, k, v, g, beta, pos, scale):
        HV = self.HV
        sl, hs = slice(n0, n0 + ns), slice(n0 * HV, (n0 + ns) * HV)
        o = torch.empty(ns, HV, self.V, device=q.device, dtype=v.dtype)
        _gdn_k1[(ns * HV,)](
            q, k, v, g, beta, pos, self.kbuf[sl], self.vbuf[sl], self.gbuf[sl], self.bbuf[sl], self.qn[hs],
            self.AiF[hs], self.Abf[hs], self.Uc[hs], self.Wc[hs], self.sg[hs], RCP_LN2, self.eps,
            H=self.H, HV=HV, K=self.K, V=self.V, BT=BT, BC=BC, num_warps=self.nw1)
        _gdn_k2[(self.V // self.BV, ns * HV)](
            pos, self.kbuf[sl], self.qn[hs], self.S[sl], self.Hb[sl], o, self.Uc[hs], self.Wc[hs], self.VNc[hs],
            self.sg[hs], scale, H=self.H, HV=HV, K=self.K, V=self.V, BV=self.BV, BT=BT, BC=BC, num_warps=self.nw2)
        return o

    def step(self, q, k, v, g, beta, pos, scale):
        """q, k [N,H,K] raw (normalised in-kernel), v [N,HV,V], g [N,HV] raw log-gate, beta [N,HV],
        pos [N] int32 absolute token index. Returns o [N,HV,V]. Two launches; graph-capturable."""
        return self._launch(0, self.N, q, k, v, g, beta, pos, scale)

    def load(self, n, S, b, q, k, v, g, beta):
        """Slot n resumes at its boundary state S [HV,K,V] fp32 taken at absolute position b (multiple of 64), then
        replays the partial-chunk tokens (raw q, k [r,H,K], v [r,HV,V], g, beta [r,HV]) to fill the row caches."""
        assert b % BT == 0
        self.S[n] = S
        self.Hb[n] = S.to(self.Hb.dtype)
        for j in range(k.shape[0]):
            pos = torch.tensor([b + j], device=k.device, dtype=torch.int32)
            self._launch(n, 1, q[j:j + 1].contiguous(), k[j:j + 1].contiguous(), v[j:j + 1].contiguous(),
                         g[j:j + 1].contiguous(), beta[j:j + 1].contiguous(), pos, 1.0)
