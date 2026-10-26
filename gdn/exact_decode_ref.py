"""Reference exact GDN decode (stateless, simplest form of the contract in exact_decode.py): one decode step == the row fla's chunk_gated_delta_rule (0.5.0, use_exp2) would produce for that
token in a full forward, bitwise.

Each sequence slot keeps:
  S   [N, HV, K, V] fp32  state at its last absolute 64-token boundary (what fla carries between chunks)
  Hb  [N, HV, K, V] bf16  bf16(S): the copy fla's fwd_h stores per chunk and feeds to every intra-chunk dot
  kbuf/vbuf/gbuf/bbuf     the current partial chunk's inputs (rows 0..r), k already l2-normalised
A step writes the new token into row r and runs one kernel that re-executes fla's chunk program on rows 0..r:
  cumsum(g)*log2(e) -> kkt + forward substitution + block merge (I+A)^-1 -> w, u -> v_new = u - w Hb -> o row r.
On r == 63 the same kernel folds the chunk into S exactly like fwd_h's chunk loop and refreshes Hb.
Every expression below is copied from the fla kernel it mirrors (named in the comments) with the same operand
dtypes and the same tl.dot shapes, so per-element arithmetic is identical; tests/e5 checks it bitwise.

This version recomputes the whole partial chunk every step (stateless apart from the inputs): it is the
readable statement of the algorithm and the cross-check for exact_decode.py, not the fast path."""
import torch
import triton
import triton.language as tl

BT, BC = 64, 16
RCP_LN2 = 1.4426950216                       # fla.ops.utils.constant.RCP_LN2 (asserted equal at import below)


@triton.jit
def _gdn_exact_decode_kernel(
    q, kbuf, vbuf, gbuf, bbuf, rows, S, Hb, o, scr_A, scr_g,
    scale, gscale,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr,
    BT: tl.constexpr, BC: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // HV, i_nh % HV
    NV: tl.constexpr = V // BV
    T = tl.load(rows + i_n) + 1                               # rows 0..T-1 of this chunk are valid; new row = T-1
    i_hk = i_h // (HV // H)
    k = kbuf + (i_n * BT * H + i_hk) * K                       # fla layout [T, H, K] with bos = n*64
    v = vbuf + (i_n * BT * HV + i_h) * V
    g = gbuf + i_n * BT * HV + i_h
    beta = bbuf + i_n * BT * HV + i_h
    sA = scr_A + (i_nh * NV + i_v) * BT * BT
    sg = scr_g + (i_nh * NV + i_v) * BT
    o_i = tl.arange(0, BC)

    # ---- chunk_local_cumsum_scalar_kernel (HAS_SCALE, scale = RCP_LN2), stored fp32 ---------------------------
    p_s = tl.make_block_ptr(g, (T,), (HV,), (0,), (BT,), (0,))
    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_gc = tl.cumsum(b_s, axis=0)
    b_gc *= gscale                                            # passed as a runtime fp32 arg, like fla's scale
    p_sg = tl.make_block_ptr(sg, (T,), (1,), (0,), (BT,), (0,))
    tl.store(p_sg, b_gc.to(tl.float32), boundary_check=(0,))
    tl.debug_barrier()

    # ---- chunk_gated_delta_rule_fwd_kkt_solve_kernel (USE_G, USE_EXP2), BK = 64 -------------------------------
    i_tc0, i_tc1, i_tc2, i_tc3 = 0, BC, 2 * BC, 3 * BC
    m_tc0 = (i_tc0 + o_i) < T
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T
    b_b0 = tl.load(tl.make_block_ptr(beta, (T,), (HV,), (i_tc0,), (BC,), (0,)), boundary_check=(0,)).to(tl.float32)
    b_b1 = tl.load(tl.make_block_ptr(beta, (T,), (HV,), (i_tc1,), (BC,), (0,)), boundary_check=(0,)).to(tl.float32)
    b_b2 = tl.load(tl.make_block_ptr(beta, (T,), (HV,), (i_tc2,), (BC,), (0,)), boundary_check=(0,)).to(tl.float32)
    b_b3 = tl.load(tl.make_block_ptr(beta, (T,), (HV,), (i_tc3,), (BC,), (0,)), boundary_check=(0,)).to(tl.float32)
    b_g0 = tl.load(tl.make_block_ptr(sg, (T,), (1,), (i_tc0,), (BC,), (0,)), boundary_check=(0,)).to(tl.float32)
    b_g1 = tl.load(tl.make_block_ptr(sg, (T,), (1,), (i_tc1,), (BC,), (0,)), boundary_check=(0,)).to(tl.float32)
    b_g2 = tl.load(tl.make_block_ptr(sg, (T,), (1,), (i_tc2,), (BC,), (0,)), boundary_check=(0,)).to(tl.float32)
    b_g3 = tl.load(tl.make_block_ptr(sg, (T,), (1,), (i_tc3,), (BC,), (0,)), boundary_check=(0,)).to(tl.float32)

    b_A00 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A11 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A22 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A33 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A32 = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, 64)):
        b_k0 = tl.load(tl.make_block_ptr(k, (T, K), (H*K, 1), (i_tc0, i_k * 64), (BC, 64), (1, 0)), boundary_check=(0, 1))
        b_A00 += tl.dot(b_k0, tl.trans(b_k0))
        if i_tc1 < T:
            b_k1 = tl.load(tl.make_block_ptr(k, (T, K), (H*K, 1), (i_tc1, i_k * 64), (BC, 64), (1, 0)), boundary_check=(0, 1))
            b_A11 += tl.dot(b_k1, tl.trans(b_k1))
            b_A10 += tl.dot(b_k1, tl.trans(b_k0))
            if i_tc2 < T:
                b_k2 = tl.load(tl.make_block_ptr(k, (T, K), (H*K, 1), (i_tc2, i_k * 64), (BC, 64), (1, 0)), boundary_check=(0, 1))
                b_A22 += tl.dot(b_k2, tl.trans(b_k2))
                b_A20 += tl.dot(b_k2, tl.trans(b_k0))
                b_A21 += tl.dot(b_k2, tl.trans(b_k1))
                if i_tc3 < T:
                    b_k3 = tl.load(tl.make_block_ptr(k, (T, K), (H*K, 1), (i_tc3, i_k * 64), (BC, 64), (1, 0)), boundary_check=(0, 1))
                    b_A33 += tl.dot(b_k3, tl.trans(b_k3))
                    b_A30 += tl.dot(b_k3, tl.trans(b_k0))
                    b_A31 += tl.dot(b_k3, tl.trans(b_k1))
                    b_A32 += tl.dot(b_k3, tl.trans(b_k2))

    m_d = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    b_A00 *= tl.where(m_d & m_tc0[:, None] & m_tc0[None, :], tl.math.exp2(b_g0[:, None] - b_g0[None, :]), 0.)
    b_A11 *= tl.where(m_d & m_tc1[:, None] & m_tc1[None, :], tl.math.exp2(b_g1[:, None] - b_g1[None, :]), 0.)
    b_A22 *= tl.where(m_d & m_tc2[:, None] & m_tc2[None, :], tl.math.exp2(b_g2[:, None] - b_g2[None, :]), 0.)
    b_A33 *= tl.where(m_d & m_tc3[:, None] & m_tc3[None, :], tl.math.exp2(b_g3[:, None] - b_g3[None, :]), 0.)
    b_A10 *= tl.where(m_tc1[:, None] & m_tc0[None, :], tl.math.exp2(b_g1[:, None] - b_g0[None, :]), 0.)
    b_A20 *= tl.where(m_tc2[:, None] & m_tc0[None, :], tl.math.exp2(b_g2[:, None] - b_g0[None, :]), 0.)
    b_A21 *= tl.where(m_tc2[:, None] & m_tc1[None, :], tl.math.exp2(b_g2[:, None] - b_g1[None, :]), 0.)
    b_A30 *= tl.where(m_tc3[:, None] & m_tc0[None, :], tl.math.exp2(b_g3[:, None] - b_g0[None, :]), 0.)
    b_A31 *= tl.where(m_tc3[:, None] & m_tc1[None, :], tl.math.exp2(b_g3[:, None] - b_g1[None, :]), 0.)
    b_A32 *= tl.where(m_tc3[:, None] & m_tc2[None, :], tl.math.exp2(b_g3[:, None] - b_g2[None, :]), 0.)
    b_A00 = b_A00 * b_b0[:, None]
    b_A11 = b_A11 * b_b1[:, None]
    b_A22 = b_A22 * b_b2[:, None]
    b_A33 = b_A33 * b_b3[:, None]
    b_A10 = b_A10 * b_b1[:, None]
    b_A20 = b_A20 * b_b2[:, None]
    b_A21 = b_A21 * b_b2[:, None]
    b_A30 = b_A30 * b_b3[:, None]
    b_A31 = b_A31 * b_b3[:, None]
    b_A32 = b_A32 * b_b3[:, None]

    b_Ai00 = -b_A00
    b_Ai11 = -b_A11
    b_Ai22 = -b_A22
    b_Ai33 = -b_A33
    for i in range(2, min(BC, T - i_tc0)):
        b_a00 = tl.sum(tl.where((o_i == i)[:, None], -b_A00, 0.), 0)
        b_a00 = tl.where(o_i < i, b_a00, 0.)
        b_a00 = b_a00 + tl.sum(b_a00[:, None] * b_Ai00, 0)
        b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
    for i in range(2, min(BC, T - i_tc1)):
        b_a11 = tl.sum(tl.where((o_i == i)[:, None], -b_A11, 0.), 0)
        b_a11 = tl.where(o_i < i, b_a11, 0.)
        b_a11 = b_a11 + tl.sum(b_a11[:, None] * b_Ai11, 0)
        b_Ai11 = tl.where((o_i == i)[:, None], b_a11, b_Ai11)
    for i in range(2, min(BC, T - i_tc2)):
        b_a22 = tl.sum(tl.where((o_i == i)[:, None], -b_A22, 0.), 0)
        b_a22 = tl.where(o_i < i, b_a22, 0.)
        b_a22 = b_a22 + tl.sum(b_a22[:, None] * b_Ai22, 0)
        b_Ai22 = tl.where((o_i == i)[:, None], b_a22, b_Ai22)
    for i in range(2, min(BC, T - i_tc3)):
        b_a33 = tl.sum(tl.where((o_i == i)[:, None], -b_A33, 0.), 0)
        b_a33 = tl.where(o_i < i, b_a33, 0.)
        b_a33 = b_a33 + tl.sum(b_a33[:, None] * b_Ai33, 0)
        b_Ai33 = tl.where((o_i == i)[:, None], b_a33, b_Ai33)
    b_Ai00 += m_I
    b_Ai11 += m_I
    b_Ai22 += m_I
    b_Ai33 += m_I

    b_Ai10 = -tl.dot(tl.dot(b_Ai11, b_A10, input_precision='tf32'), b_Ai00, input_precision='tf32')
    b_Ai21 = -tl.dot(tl.dot(b_Ai22, b_A21, input_precision='tf32'), b_Ai11, input_precision='tf32')
    b_Ai32 = -tl.dot(tl.dot(b_Ai33, b_A32, input_precision='tf32'), b_Ai22, input_precision='tf32')
    b_Ai20 = -tl.dot(b_Ai22, tl.dot(b_A20, b_Ai00, input_precision='tf32') +
                     tl.dot(b_A21, b_Ai10, input_precision='tf32'), input_precision='tf32')
    b_Ai31 = -tl.dot(b_Ai33, tl.dot(b_A31, b_Ai11, input_precision='tf32') +
                     tl.dot(b_A32, b_Ai21, input_precision='tf32'), input_precision='tf32')
    b_Ai30 = -tl.dot(b_Ai33, tl.dot(b_A30, b_Ai00, input_precision='tf32') +
                     tl.dot(b_A31, b_Ai10, input_precision='tf32') +
                     tl.dot(b_A32, b_Ai20, input_precision='tf32'), input_precision='tf32')

    # stored in k.dtype exactly as fla's A (zeros beyond row T-1 are re-created by the masked load below)
    tl.store(tl.make_block_ptr(sA, (T, BT), (BT, 1), (i_tc0, 0), (BC, BC), (1, 0)), b_Ai00.to(sA.dtype.element_ty), boundary_check=(0, 1))
    tl.store(tl.make_block_ptr(sA, (T, BT), (BT, 1), (i_tc1, 0), (BC, BC), (1, 0)), b_Ai10.to(sA.dtype.element_ty), boundary_check=(0, 1))
    tl.store(tl.make_block_ptr(sA, (T, BT), (BT, 1), (i_tc1, BC), (BC, BC), (1, 0)), b_Ai11.to(sA.dtype.element_ty), boundary_check=(0, 1))
    tl.store(tl.make_block_ptr(sA, (T, BT), (BT, 1), (i_tc2, 0), (BC, BC), (1, 0)), b_Ai20.to(sA.dtype.element_ty), boundary_check=(0, 1))
    tl.store(tl.make_block_ptr(sA, (T, BT), (BT, 1), (i_tc2, BC), (BC, BC), (1, 0)), b_Ai21.to(sA.dtype.element_ty), boundary_check=(0, 1))
    tl.store(tl.make_block_ptr(sA, (T, BT), (BT, 1), (i_tc2, 2*BC), (BC, BC), (1, 0)), b_Ai22.to(sA.dtype.element_ty), boundary_check=(0, 1))
    tl.store(tl.make_block_ptr(sA, (T, BT), (BT, 1), (i_tc3, 0), (BC, BC), (1, 0)), b_Ai30.to(sA.dtype.element_ty), boundary_check=(0, 1))
    tl.store(tl.make_block_ptr(sA, (T, BT), (BT, 1), (i_tc3, BC), (BC, BC), (1, 0)), b_Ai31.to(sA.dtype.element_ty), boundary_check=(0, 1))
    tl.store(tl.make_block_ptr(sA, (T, BT), (BT, 1), (i_tc3, 2*BC), (BC, BC), (1, 0)), b_Ai32.to(sA.dtype.element_ty), boundary_check=(0, 1))
    tl.store(tl.make_block_ptr(sA, (T, BT), (BT, 1), (i_tc3, 3*BC), (BC, BC), (1, 0)), b_Ai33.to(sA.dtype.element_ty), boundary_check=(0, 1))
    tl.debug_barrier()

    # ---- recompute_w_u_fwd_kernel (USE_G, USE_EXP2), BK = BV = 64 ---------------------------------------------
    b_b = tl.load(tl.make_block_ptr(beta, (T,), (HV,), (0,), (BT,), (0,)), boundary_check=(0,))
    b_A = tl.load(tl.make_block_ptr(sA, (T, BT), (BT, 1), (0, 0), (BT, BT), (1, 0)), boundary_check=(0, 1))
    b_A = tl.where((tl.arange(0, BT) < T)[None, :], b_A, 0.).to(b_A.dtype)   # fla's A columns >= T are zeros
    b_vin = tl.load(tl.make_block_ptr(v, (T, V), (HV*V, 1), (0, i_v * BV), (BT, BV), (1, 0)), boundary_check=(0, 1))
    b_vb = (b_vin * b_b[:, None]).to(b_vin.dtype)
    b_u = tl.dot(b_A, b_vb, allow_tf32=False).to(b_vin.dtype)               # u stored in v.dtype
    b_ge = tl.math.exp2(tl.load(tl.make_block_ptr(sg, (T,), (1,), (0,), (BT,), (0,)), boundary_check=(0,)))
    b_kin1 = tl.load(tl.make_block_ptr(k, (T, K), (H*K, 1), (0, 0), (BT, 64), (1, 0)), boundary_check=(0, 1))
    b_kb1 = b_kin1 * b_b[:, None]
    b_kb1 *= b_ge[:, None]
    b_w1 = tl.dot(b_A, b_kb1.to(b_kin1.dtype)).to(b_kin1.dtype)             # w stored in k.dtype
    if K > 64:
        b_kin2 = tl.load(tl.make_block_ptr(k, (T, K), (H*K, 1), (0, 64), (BT, 64), (1, 0)), boundary_check=(0, 1))
        b_kb2 = b_kin2 * b_b[:, None]
        b_kb2 *= b_ge[:, None]
        b_w2 = tl.dot(b_A, b_kb2.to(b_kin2.dtype)).to(b_kin2.dtype)

    # ---- chunk_gated_delta_rule_fwd_kernel_h_blockdim64: v_new for this chunk --------------------------------
    S_ = S + i_nh * K * V
    Hb_ = Hb + i_nh * K * V
    b_h1 = tl.load(tl.make_block_ptr(Hb_, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)), boundary_check=(0, 1))
    b_v = tl.dot(b_w1, b_h1)
    if K > 64:
        b_h2 = tl.load(tl.make_block_ptr(Hb_, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)), boundary_check=(0, 1))
        b_v += tl.dot(b_w2, b_h2)
    rmask = (tl.arange(0, BT) < T)[:, None]
    b_v = tl.where(rmask, b_u, 0.).to(tl.float32) - b_v                     # u loaded with boundary_check
    b_vnew = b_v.to(vbuf.dtype.element_ty)                                    # v_new stored in u.dtype

    # ---- chunk_fwd_kernel_o: row T-1 only (q rows other than T-1 are zero; MMA rows are independent) ---------
    o_t = tl.arange(0, BT)
    is_row = o_t == (T - 1)
    q_ = q + (i_n * H + i_hk) * K
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_Aq = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, 64)):
        o_k = i_k * 64 + tl.arange(0, 64)
        b_q = tl.load(q_ + o_k[None, :] + o_t[:, None] * 0, mask=is_row[:, None], other=0.)
        b_kt = tl.load(tl.make_block_ptr(k, (K, T), (1, H*K), (i_k * 64, 0), (64, BT), (0, 1)), boundary_check=(0, 1))
        b_hk = tl.load(tl.make_block_ptr(Hb_, (K, V), (V, 1), (i_k * 64, i_v * BV), (64, BV), (1, 0)), boundary_check=(0, 1))
        b_o += tl.dot(b_q, b_hk)
        b_Aq += tl.dot(b_q, b_kt)
    b_gq = tl.load(tl.make_block_ptr(sg, (T,), (1,), (0,), (BT,), (0,)), boundary_check=(0,))
    b_o = b_o * tl.math.exp2(b_gq)[:, None]
    b_Aq = b_Aq * tl.math.exp2(b_gq[:, None] - b_gq[None, :])
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_Aq = tl.where(m_A, b_Aq, 0)
    # fla's PTX for `b_o * scale + tl.dot(b_A, b_v) * scale` (LLVM fp-contraction): fma(scale, dot, b_o * scale).
    # Written out so this kernel's contraction cannot drift with the surrounding code.
    b_o = tl.fma(tl.dot(b_Aq.to(b_vnew.dtype), b_vnew), scale, b_o * scale)
    b_orow = tl.sum(tl.where(is_row[:, None], b_o, 0.), 0)
    o_v = i_v * BV + tl.arange(0, BV)
    tl.store(o + (i_n * HV + i_h) * V + o_v, b_orow.to(o.dtype.element_ty))

    # ---- fold on a complete chunk: fwd_h's state update, then the bf16 copy fwd_h stores per chunk -----------
    if T == BT:
        b_glast = tl.load(sg + (BT - 1)).to(tl.float32)
        b_gl = tl.load(tl.make_block_ptr(sg, (T,), (1,), (0,), (BT,), (0,)), boundary_check=(0,)).to(tl.float32)
        b_vf = b_v * tl.where(m_t, tl.math.exp2(b_glast - b_gl), 0)[:, None]
        b_glast = tl.math.exp2(b_glast)
        b_vf = b_vf.to(kbuf.dtype.element_ty)
        p_s1 = tl.make_block_ptr(S_, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_s1 = tl.load(p_s1, boundary_check=(0, 1)).to(tl.float32)
        b_s1 *= b_glast
        b_s1 += tl.dot(tl.load(tl.make_block_ptr(k, (K, T), (1, H*K), (0, 0), (64, BT), (0, 1)), boundary_check=(0, 1)), b_vf)
        tl.store(p_s1, b_s1, boundary_check=(0, 1))
        tl.store(tl.make_block_ptr(Hb_, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)), b_s1.to(Hb.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_s2 = tl.make_block_ptr(S_, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_s2 = tl.load(p_s2, boundary_check=(0, 1)).to(tl.float32)
            b_s2 *= b_glast
            b_s2 += tl.dot(tl.load(tl.make_block_ptr(k, (K, T), (1, H*K), (64, 0), (64, BT), (0, 1)), boundary_check=(0, 1)), b_vf)
            tl.store(p_s2, b_s2, boundary_check=(0, 1))
            tl.store(tl.make_block_ptr(Hb_, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)), b_s2.to(Hb.dtype.element_ty), boundary_check=(0, 1))


class GDNExactDecodeRef:
    """Decode state for N sequence slots of one GDN layer (K <= 128, V % 64 == 0)."""

    def __init__(self, N, H, HV, K, V, device, dtype=torch.bfloat16, BV=64, num_warps=4):
        assert K in (64, 128) and V % BV == 0
        self.N, self.H, self.HV, self.K, self.V, self.BV, self.num_warps = N, H, HV, K, V, BV, num_warps
        z = lambda *s, dt=dtype: torch.zeros(*s, device=device, dtype=dt)
        self.S, self.Hb = z(N, HV, K, V, dt=torch.float32), z(N, HV, K, V)
        self.kbuf, self.vbuf = z(N, BT, H, K), z(N, BT, HV, V)
        self.gbuf, self.bbuf = z(N, BT, HV, dt=torch.float32), z(N, BT, HV)
        self.rows = torch.zeros(N, device=device, dtype=torch.int32)       # index the next token will occupy
        NV = V // BV
        self.scr_A, self.scr_g = z(N * HV * NV, BT, BT), z(N * HV * NV, BT, dt=torch.float32)
        self.idx = torch.arange(N, device=device)

    def load(self, n, S, k, v, g, beta):
        """Slot n resumes at a boundary state S [HV,K,V] fp32 with partial-chunk rows k [r,H,K] (normalised),
        v [r,HV,V], g [r,HV] (raw log-gates), beta [r,HV]."""
        r = k.shape[0]
        assert r < BT
        self.S[n] = S
        self.Hb[n] = S.to(self.Hb.dtype)
        self.kbuf[n].zero_(); self.vbuf[n].zero_(); self.gbuf[n].zero_(); self.bbuf[n].zero_()
        self.kbuf[n, :r], self.vbuf[n, :r], self.gbuf[n, :r], self.bbuf[n, :r] = k, v, g, beta
        self.rows[n] = r

    def step(self, q, k, v, g, beta, scale):
        """One token for every slot: q, k [N,H,K] (both l2-normalised), v [N,HV,V], g [N,HV], beta [N,HV].
        Returns o [N,HV,V]. Graph-capturable: no host reads."""
        self.kbuf[self.idx, self.rows] = k
        self.vbuf[self.idx, self.rows] = v
        self.gbuf[self.idx, self.rows] = g
        self.bbuf[self.idx, self.rows] = beta
        o = torch.empty(self.N, self.HV, self.V, device=q.device, dtype=v.dtype)
        _gdn_exact_decode_kernel[(self.V // self.BV, self.N * self.HV)](
            q, self.kbuf, self.vbuf, self.gbuf, self.bbuf, self.rows, self.S, self.Hb, o, self.scr_A, self.scr_g,
            scale, RCP_LN2, H=self.H, HV=self.HV, K=self.K, V=self.V, BV=self.BV, BT=BT, BC=BC,
            num_warps=self.num_warps)
        self.rows += 1
        self.rows.masked_fill_(self.rows == BT, 0)                     # chunk folded in-kernel; start a new one
        return o


try:
    from fla.ops.utils.constant import RCP_LN2 as _fla_rcp
    assert float(_fla_rcp) == RCP_LN2, (_fla_rcp, RCP_LN2)
except ImportError:
    pass
