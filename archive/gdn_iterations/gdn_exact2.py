"""Exact GDN decode, v2 (row-incremental). Same contract as gdn_exact.py (v1): every decode step returns, bitwise,
the row fla 0.5.0's chunk_gated_delta_rule (use_exp2) produces for that token in a full forward.

Why it can be incremental: every intermediate of fla's chunk program is row-causal inside a 64-token chunk, so once a
row of (I+A)^-1, w, u, v_new is computed it never changes. A step for the token at chunk row r (sub-block s = r//16,
i = r%16) therefore only computes row r, reusing cached rows < r:
  A_sj (j <= s)  16x16 tiles of beta*kk^T*gate for the rows of sub-block s          (kkt_solve, step 1-2)
  forward substitution: the single iteration for row i of the diagonal block       (kkt_solve, step 3)
  block merge: row i of Ai_sj (j < s) from cached final blocks of earlier rows     (kkt_solve, step 4)
  w, u, v_new, o on the 16-row tile of sub-block s                                 (recompute_w_u, fwd_h, chunk_o)
and folds the chunk into the fp32 state on r == 63 with fwd_h's 64-row arithmetic from the cached u, w rows.

Per step it reads bf16(S) once (no state write) plus the chunk's cached rows; fla's fused_recurrent reads and writes
the fp32 state every step. Caches per (slot, head): AiF [64,64] fp32 (diagonal blocks stored before +I, as fla's
loop holds them), Abf [64,64] bf16 (fla's stored A), u/w/v_new rows [64,*] bf16.

Bitwise equality relies on (a) every expression mirroring the fla source with the same operand dtypes and
(b) a row of an MMA tile not depending on the other rows (M=16 here vs M=64 in fla). Both are checked by e5."""
import torch
import triton
import triton.language as tl

BT, BC = 64, 16
RCP_LN2 = 1.4426950216                       # fla.ops.utils.constant.RCP_LN2


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
def _kkt(k, ks0, kj0, nrow_s, H: tl.constexpr, K: tl.constexpr, BC: tl.constexpr):
    """kkt_solve step 1 for one block: sum over 64-wide K slices of dot(k_s, k_j^T) (fla accumulates the same way)."""
    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    o = tl.arange(0, BC)
    for i_k in range(tl.cdiv(K, 64)):
        o_k = i_k * 64 + tl.arange(0, 64)
        b_ks = tl.load(k + (ks0 + o)[:, None] * (H*K) + o_k[None, :], mask=(o < nrow_s)[:, None], other=0.)
        b_kj = tl.load(k + (kj0 + o)[:, None] * (H*K) + o_k[None, :])
        b_A += tl.dot(b_ks, tl.trans(b_kj))
    return b_A


@triton.jit
def _gdn_exact_decode2_kernel(
    q, kbuf, vbuf, gbuf, bbuf, rows, S, Hb, o, AiF, Abf, Uc, Wc, VNc, scr_g,
    scale, gscale,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr,
):
    i_nh = tl.program_id(0)
    i_n, i_h = i_nh // HV, i_nh % HV
    r = tl.load(rows + i_n)
    T = r + 1
    s = r // BC
    i = r - s * BC
    tb = s * BC
    k = kbuf + (i_n * BT * H + i_h // (HV // H)) * K
    v = vbuf + (i_n * BT * HV + i_h) * V
    g = gbuf + i_n * BT * HV + i_h
    beta = bbuf + i_n * BT * HV + i_h
    AiF += i_nh * BT * BT
    Abf += i_nh * BT * BT
    Uc += i_nh * BT * V
    Wc += i_nh * BT * K
    VNc += i_nh * BT * V
    sg = scr_g + i_nh * BT
    Hb_ = Hb + i_nh * K * V
    S_ = S + i_nh * K * V
    o_i = tl.arange(0, BC)
    o_t = tl.arange(0, BT)
    m_t = o_t < T

    # ---- chunk_local_cumsum_scalar_kernel ----------------------------------------------------------------------
    b_s = tl.load(tl.make_block_ptr(g, (T,), (HV,), (0,), (BT,), (0,)), boundary_check=(0,)).to(tl.float32)
    b_gc = tl.cumsum(b_s, axis=0)
    b_gc *= gscale
    tl.store(tl.make_block_ptr(sg, (T,), (1,), (0,), (BT,), (0,)), b_gc.to(tl.float32), boundary_check=(0,))
    tl.debug_barrier()

    # ---- kkt_solve for the rows of sub-block s ------------------------------------------------------------------
    nrow = i + 1                                                    # valid rows of sub-block s
    m_ts = (tb + o_i) < T
    b_gs = tl.load(sg + tb + o_i, mask=m_ts, other=0.).to(tl.float32)
    b_bs = tl.load(beta + (tb + o_i) * HV, mask=m_ts, other=0.).to(tl.float32)
    m_d = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    b_Ass = _kkt(k, tb, tb, nrow, H, K, BC)
    b_Ass *= tl.where(m_d & m_ts[:, None] & m_ts[None, :], tl.math.exp2(b_gs[:, None] - b_gs[None, :]), 0.)
    b_Ass = b_Ass * b_bs[:, None]
    # forward substitution: rows < i are final (cached, before +I); rows >= i still hold -A, as in fla's loop
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
        b_As0 = _kkt(k, tb, 0, nrow, H, K, BC)
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
            b_As1 = _kkt(k, tb, BC, nrow, H, K, BC)
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
                b_As2 = _kkt(k, tb, 2 * BC, nrow, H, K, BC)
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

    # ---- recompute_w_u on the 16-row tile of sub-block s (row i is the new one) ---------------------------------
    m_rs = (o_i <= i)[:, None]
    b_b = tl.load(tl.make_block_ptr(beta, (T,), (HV,), (0,), (BT,), (0,)), boundary_check=(0,))
    b_A = tl.load(Abf + (tb + o_i)[:, None] * BT + o_t[None, :], mask=m_rs, other=0.)
    o_v = tl.arange(0, V)
    b_vin = tl.load(tl.make_block_ptr(v, (T, V), (HV*V, 1), (0, 0), (BT, V), (1, 0)), boundary_check=(0, 1))
    b_vb = (b_vin * b_b[:, None]).to(b_vin.dtype)
    b_u = tl.dot(b_A, b_vb, allow_tf32=False).to(b_vin.dtype)
    b_ge = tl.math.exp2(tl.load(tl.make_block_ptr(sg, (T,), (1,), (0,), (BT,), (0,)), boundary_check=(0,)))
    b_k1 = tl.load(tl.make_block_ptr(k, (T, K), (H*K, 1), (0, 0), (BT, 64), (1, 0)), boundary_check=(0, 1))
    b_kb1 = b_k1 * b_b[:, None]
    b_kb1 *= b_ge[:, None]
    b_w1 = tl.dot(b_A, b_kb1.to(b_k1.dtype)).to(b_k1.dtype)
    o_64 = tl.arange(0, 64)
    m_row = (o_i == i)[:, None]
    tl.store(Uc + (tb + o_i)[:, None] * V + o_v[None, :], b_u, mask=m_row)
    tl.store(Wc + (tb + o_i)[:, None] * K + o_64[None, :], b_w1, mask=m_row)
    if K > 64:
        b_k2 = tl.load(tl.make_block_ptr(k, (T, K), (H*K, 1), (0, 64), (BT, 64), (1, 0)), boundary_check=(0, 1))
        b_kb2 = b_k2 * b_b[:, None]
        b_kb2 *= b_ge[:, None]
        b_w2 = tl.dot(b_A, b_kb2.to(b_k2.dtype)).to(b_k2.dtype)
        tl.store(Wc + (tb + o_i)[:, None] * K + 64 + o_64[None, :], b_w2, mask=m_row)

    # ---- fwd_h: v_new on the tile -------------------------------------------------------------------------------
    b_h1 = tl.load(tl.make_block_ptr(Hb_, (K, V), (V, 1), (0, 0), (64, V), (1, 0)), boundary_check=(0, 1))
    b_v = tl.dot(b_w1, b_h1)
    if K > 64:
        b_h2 = tl.load(tl.make_block_ptr(Hb_, (K, V), (V, 1), (64, 0), (64, V), (1, 0)), boundary_check=(0, 1))
        b_v += tl.dot(b_w2, b_h2)
    b_v = tl.where(m_rs, b_u, 0.).to(tl.float32) - b_v
    tl.store(VNc + (tb + o_i)[:, None] * V + o_v[None, :], b_v.to(VNc.dtype.element_ty), mask=m_row)
    tl.debug_barrier()

    # ---- chunk_fwd_kernel_o: row i of the tile ------------------------------------------------------------------
    q_ = q + (i_n * HV + i_h) * K
    b_o = tl.zeros([BC, V], dtype=tl.float32)
    b_Aq = tl.zeros([BC, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, 64)):
        o_k = i_k * 64 + tl.arange(0, 64)
        b_q = tl.load(q_ + o_k[None, :] + o_i[:, None] * 0, mask=m_row, other=0.)
        b_kt = tl.load(tl.make_block_ptr(k, (K, T), (1, H*K), (i_k * 64, 0), (64, BT), (0, 1)), boundary_check=(0, 1))
        b_hk = tl.load(tl.make_block_ptr(Hb_, (K, V), (V, 1), (i_k * 64, 0), (64, V), (1, 0)), boundary_check=(0, 1))
        b_o += tl.dot(b_q, b_hk)
        b_Aq += tl.dot(b_q, b_kt)
    b_gall = tl.load(tl.make_block_ptr(sg, (T,), (1,), (0,), (BT,), (0,)), boundary_check=(0,))
    b_o = b_o * tl.math.exp2(b_gs)[:, None]
    b_Aq = b_Aq * tl.math.exp2(b_gs[:, None] - b_gall[None, :])
    m_A = ((tb + o_i)[:, None] >= o_t[None, :]) & (m_ts[:, None] & m_t)
    b_Aq = tl.where(m_A, b_Aq, 0)
    b_vn = tl.load(tl.make_block_ptr(VNc, (T, V), (V, 1), (0, 0), (BT, V), (1, 0)), boundary_check=(0, 1))
    b_o = tl.fma(tl.dot(b_Aq.to(b_vn.dtype), b_vn), scale, b_o * scale)     # = fla's contraction, see v1
    b_orow = tl.sum(tl.where(m_row, b_o, 0.), 0)
    tl.store(o + (i_n * HV + i_h) * V + o_v, b_orow.to(o.dtype.element_ty))

    # ---- fold a complete chunk: fwd_h's 64-row arithmetic from the cached u, w rows ----------------------------
    if T == BT:
        b_wf1 = tl.load(tl.make_block_ptr(Wc, (BT, K), (K, 1), (0, 0), (BT, 64), (1, 0)))
        b_vf = tl.dot(b_wf1, b_h1)
        if K > 64:
            b_wf2 = tl.load(tl.make_block_ptr(Wc, (BT, K), (K, 1), (0, 64), (BT, 64), (1, 0)))
            b_vf += tl.dot(b_wf2, b_h2)
        b_vf = tl.load(tl.make_block_ptr(Uc, (BT, V), (V, 1), (0, 0), (BT, V), (1, 0))) - b_vf
        b_glast = tl.load(sg + (BT - 1)).to(tl.float32)
        b_gl = tl.load(sg + o_t).to(tl.float32)
        b_vf = b_vf * tl.where(m_t, tl.math.exp2(b_glast - b_gl), 0)[:, None]
        b_glast = tl.math.exp2(b_glast)
        b_vf = b_vf.to(kbuf.dtype.element_ty)
        p_s1 = tl.make_block_ptr(S_, (K, V), (V, 1), (0, 0), (64, V), (1, 0))
        b_s1 = tl.load(p_s1).to(tl.float32)
        b_s1 *= b_glast
        b_s1 += tl.dot(tl.load(tl.make_block_ptr(k, (K, BT), (1, H*K), (0, 0), (64, BT), (0, 1))), b_vf)
        tl.store(p_s1, b_s1)
        tl.store(tl.make_block_ptr(Hb_, (K, V), (V, 1), (0, 0), (64, V), (1, 0)), b_s1.to(Hb.dtype.element_ty))
        if K > 64:
            p_s2 = tl.make_block_ptr(S_, (K, V), (V, 1), (64, 0), (64, V), (1, 0))
            b_s2 = tl.load(p_s2).to(tl.float32)
            b_s2 *= b_glast
            b_s2 += tl.dot(tl.load(tl.make_block_ptr(k, (K, BT), (1, H*K), (64, 0), (64, BT), (0, 1))), b_vf)
            tl.store(p_s2, b_s2)
            tl.store(tl.make_block_ptr(Hb_, (K, V), (V, 1), (64, 0), (64, V), (1, 0)), b_s2.to(Hb.dtype.element_ty))


class GDNExactDecode2:
    """Row-incremental exact decode state for N slots of one GDN layer (K in {64, 128}, V <= 128)."""

    def __init__(self, N, H, HV, K, V, device, dtype=torch.bfloat16, num_warps=4):
        assert K in (64, 128) and V <= 128
        self.N, self.H, self.HV, self.K, self.V, self.num_warps = N, H, HV, K, V, num_warps
        z = lambda *s, dt=dtype: torch.zeros(*s, device=device, dtype=dt)
        self.S, self.Hb = z(N, HV, K, V, dt=torch.float32), z(N, HV, K, V)
        self.kbuf, self.vbuf = z(N, BT, H, K), z(N, BT, HV, V)
        self.gbuf, self.bbuf = z(N, BT, HV, dt=torch.float32), z(N, BT, HV)
        self.AiF, self.Abf = z(N * HV, BT, BT, dt=torch.float32), z(N * HV, BT, BT)
        self.Uc, self.Wc, self.VNc = z(N * HV, BT, V), z(N * HV, BT, K), z(N * HV, BT, V)
        self.scr_g = z(N * HV, BT, dt=torch.float32)
        self.rows = torch.zeros(N, device=device, dtype=torch.int32)
        self.idx = torch.arange(N, device=device)

    def load(self, n, S, k, v, g, beta):
        """Slot n resumes at boundary state S [HV,K,V] fp32; the partial-chunk rows (k normalised [r,H,K], v, raw g,
        beta) are replayed through the step kernel to fill the row caches (outputs discarded)."""
        self.S[n] = S
        self.Hb[n] = S.to(self.Hb.dtype)
        self.rows[n] = 0
        for j in range(k.shape[0]):
            self._step_slots([n], torch.zeros(1, self.HV, self.K, device=k.device, dtype=k.dtype),
                             k[j:j + 1], v[j:j + 1], g[j:j + 1], beta[j:j + 1], 1.0)

    def _step_slots(self, slots, q, k, v, g, beta, scale):
        self._scale = scale
        sl = torch.tensor(slots, device=q.device)
        r = self.rows[sl].long()
        self.kbuf[sl, r] = k; self.vbuf[sl, r] = v; self.gbuf[sl, r] = g; self.bbuf[sl, r] = beta
        o = torch.empty(len(slots), self.HV, self.V, device=q.device, dtype=v.dtype)
        for j, n in enumerate(slots):
            self._launch(q[j:j + 1], o[j:j + 1], n, 1)
        self.rows[sl] = ((self.rows[sl] + 1) % BT).int()
        return o

    def _launch(self, q, o, n0, nslots):
        HV = self.HV
        sl = slice(n0, n0 + nslots)
        hs = slice(n0 * HV, (n0 + nslots) * HV)
        _gdn_exact_decode2_kernel[(nslots * HV,)](
            q, self.kbuf[sl], self.vbuf[sl], self.gbuf[sl], self.bbuf[sl], self.rows[sl], self.S[sl], self.Hb[sl], o,
            self.AiF[hs], self.Abf[hs], self.Uc[hs], self.Wc[hs], self.VNc[hs], self.scr_g[hs],
            self._scale, RCP_LN2, H=self.H, HV=HV, K=self.K, V=self.V, BT=BT, BC=BC, num_warps=self.num_warps)

    def step(self, q, k, v, g, beta, scale):
        """One token for every slot: q [N,HV,K], k [N,H,K] (l2-normalised), v [N,HV,V], g [N,HV], beta [N,HV].
        Returns o [N,HV,V]. Graph-capturable."""
        self._scale = scale
        self.kbuf[self.idx, self.rows] = k
        self.vbuf[self.idx, self.rows] = v
        self.gbuf[self.idx, self.rows] = g
        self.bbuf[self.idx, self.rows] = beta
        o = torch.empty(self.N, self.HV, self.V, device=q.device, dtype=v.dtype)
        self._launch(q, o, 0, self.N)
        self.rows += 1
        self.rows.masked_fill_(self.rows == BT, 0)
        return o
