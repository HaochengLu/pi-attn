"""Bitwise test: exact GDN decode vs fla's chunk_gated_delta_rule full forward (the trainer's program).

For each head layout, 8 slots each get their own random sequence and their own prefill length P (so slots sit at
different rows of their chunks, including boundaries); the prefill is fla's chunk over [0, b) for the boundary state
plus the partial chunk [b, P) loaded into the decoder; then all slots decode token by token to T together.
Checked: every decoded row against the full forward's row, and the fp32 state at every 64-token boundary against
fla's final state of the prefix. Both implementations: exact_decode (fast) and exact_decode_ref (stateless).
Speculative mode (exact_decode): windows of up to 4 or 16 tokens per step (16: also how exact_gdn replays a prefill), a random number of them accepted per slot,
one random slot idle per step; rejected positions carry wrong-draft inputs. Accepted rows must equal the trainer's rows bitwise and the
rejected drafts must leave no trace in later rows or states.
"rows" cases: the same through step_rows, inputs read and outputs written in place in an engine-like layout (in_place).
"dirty" cases: the slots' storage starts as finite garbage, not zeros (cells no step writes, e.g. (I+A)^-1 right of a
row's diagonal block, must not matter).

usage: PYTHONPATH=<fla> python gdn/test_exact.py            (exit status 1 on any mismatch)"""
import sys, torch, torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.modules.l2norm import l2norm_fwd
from exact_decode import GDNExactDecode, BT, pin_fla_configs
from exact_decode_ref import GDNExactDecodeRef

dev, T = 'cuda', 700
pin_fla_configs()                                   # the reference is fla's forward with its odd configs removed
PS = [1, 63, 64, 65, 130, 200, 255, 333]


def seq(i, H, HV, K=128, V=128, beta_dtype=torch.bfloat16):
    gen = torch.Generator(device=dev).manual_seed(100 + i)
    q = torch.randn(1, T, H, K, device=dev, generator=gen).bfloat16()
    k = torch.randn(1, T, H, K, device=dev, generator=gen).bfloat16()
    v = torch.randn(1, T, HV, V, device=dev, generator=gen).bfloat16()
    g = F.logsigmoid(torch.randn(1, T, HV, device=dev, generator=gen) + 3).float()
    beta = torch.sigmoid(torch.randn(1, T, HV, device=dev, generator=gen)).to(beta_dtype)
    return q, k, v, g, beta


def chunk(q, k, v, g, beta):
    return chunk_gated_delta_rule(q, k, v, g, beta, output_final_state=True, use_qk_l2norm_in_kernel=True)


def state(dec, n):
    """Slot n's fp32 boundary state (exact_decode stores it as bf16 copy + rest; exact_decode_ref as S)."""
    if isinstance(dec, GDNExactDecodeRef):
        return dec.S[n]
    return dec.get_states(torch.tensor([n], device=dev, dtype=torch.int32))[0]


def run_state_codec(HV=4, K=128, V=128):
    """put_states then get_states returns every fp32 bit pattern unchanged, and the stored bf16 copy is bf16(S):
    random finite words, plus the cases the code is about (bf16 ties, rounding up across a power of two, zeros,
    subnormals, the largest finite values)."""
    gen = torch.Generator(device=dev).manual_seed(7)
    w = torch.randint(-2**31, 2**31, (2, HV, K, V), device=dev, generator=gen, dtype=torch.int64).to(torch.int32)
    w = torch.where(((w >> 23) & 0xFF) == 0xFF, w & ~(1 << 30), w)                       # finite only
    special = torch.tensor([0, -2**31, 0x00008000, 0x00018000, 0x3F808000, 0x3F818000, 0x3F7FFFFF, 0x3F7F8000,
                            -0x40808000, -0x40818000, 0x00000001, 0x007FFFFF, 0x7F7FFFFF, -0x00808001, 0x7F7F8000],
                           device=dev, dtype=torch.int64).to(torch.int32)
    w.view(-1)[:special.numel()] = special
    w.view(-1)[special.numel():2 * special.numel()] = special
    S = w.view(torch.float32)
    dec = GDNExactDecode(3, HV, HV, K, V, dev)
    dec.storage.copy_(dirty(dec.storage))
    slots = torch.tensor([2, 0], device=dev, dtype=torch.int32)
    dec.put_states(slots, S)
    back = dec.get_states(slots)
    ok = torch.equal(back.view(torch.int32), w) and torch.equal(dec.Hb[slots.long()].view(torch.int16),
                                                                 S.to(torch.bfloat16).view(torch.int16))
    print(f'state codec HV={HV} K={K} V={V}: {S.numel()} fp32 words round-trip {ok}')
    return ok


def dirty(storage):
    """A slot storage full of finite garbage, as words other code left behind (finite read as bf16 or as fp32: every
    bf16 half is a normal sample). The kernels read cache cells they did not write only where the value meets an exact
    zero, so finite garbage must not change a bit."""
    s = torch.empty_like(storage)
    s.view(torch.bfloat16).normal_()
    return s


def in_place(dec, x, pos, nrows, scale, KT):
    """dec.step_rows on window inputs x = (q, k, v, g, beta), each [N, KT, ...], laid out as an engine has them: q|k|v
    side by side in the rows of one wider activation buffer (row stride larger than q's), the slots' windows in
    reverse order with a garbage row between them; gate and beta at other row offsets, outputs to other rows of a
    larger buffer. Returns o [N, KT, HV, V] gathered from there."""
    N, HK, HVV, HV, V = x[0].shape[0], x[0][0, 0].numel(), x[2][0, 0].numel(), x[2].shape[2], x[2].shape[3]
    gap, R = KT + 1, N * (KT + 1)
    rows = torch.tensor([(N - 1 - n) * gap for n in range(N)], device=dev, dtype=torch.int32)
    buf = torch.randn(R, 2 * HK + HVV + 8, device=dev).to(x[0].dtype)
    gb = [torch.randn(R + 3, HV, device=dev).to(t.dtype) for t in x[3:]]
    tix = (rows[:, None] + torch.arange(KT, device=dev)).long()          # [N, KT]
    buf[tix, :HK], buf[tix, HK:2 * HK], buf[tix, 2 * HK:2 * HK + HVV] = x[0].flatten(2), x[1].flatten(2), x[2].flatten(2)
    for t, src in zip(gb, x[3:]):
        t[tix + 3] = src
    out = torch.zeros(R + 7, HV, V, device=dev, dtype=x[2].dtype)
    dec.step_rows(buf[:, :HK], buf[:, HK:2 * HK], buf[:, 2 * HK:2 * HK + HVV], gb[0], gb[1], rows, rows + 3,
                  pos, nrows, scale, dec._slots, out, rows + 7, KT=KT)
    return out[tix + 7]


def run(impl, H, HV, K=128, V=128, cfg=(2, 4, 64), beta_dtype=torch.bfloat16, inplace=False, garbage=False):
    N, scale = len(PS), K ** -0.5
    seqs = [seq(i, H, HV, K, V, beta_dtype) for i in range(N)]
    ref = [chunk(*s) for s in seqs]
    qn = [l2norm_fwd(s[0])[0] for s in seqs]              # the reference implementation takes normalised q, k
    kn = [l2norm_fwd(s[1])[0] for s in seqs]
    nw1, nw2, BV = cfg
    dec = GDNExactDecode(N, H, HV, K, V, dev, beta_dtype=beta_dtype, BV=BV, nw1=nw1, nw2=nw2) \
        if impl == 'exact_decode' else GDNExactDecodeRef(N, H, HV, K, V, dev)
    if garbage:
        dec = GDNExactDecode(N, H, HV, K, V, dev, beta_dtype=beta_dtype, BV=BV, nw1=nw1, nw2=nw2,
                             storage=dirty(dec.storage))
    for n, P in enumerate(PS):
        b = BT * (P // BT)
        q, k, v, g, beta = seqs[n]
        Sb = chunk(q[:, :b], k[:, :b], v[:, :b], g[:, :b], beta[:, :b])[1][0] if b else torch.zeros(HV, K, V, device=dev)
        if impl == 'exact_decode':
            dec.load(n, Sb, b, q[0, b:P], k[0, b:P], v[0, b:P], g[0, b:P], beta[0, b:P])
        else:
            dec.load(n, Sb, kn[n][0, b:P], v[0, b:P], g[0, b:P], beta[0, b:P])
    rows = bad_rows = bad_states = states = 0
    for s in range(T - max(PS)):
        ts = [P + s for P in PS]
        at = lambda xs, j: torch.stack([xs[n][j][0, t] for n, t in enumerate(ts)])
        if impl == 'exact_decode' and inplace:
            pos = torch.tensor(ts, device=dev, dtype=torch.int32)
            o = in_place(dec, [at(seqs, j)[:, None] for j in range(5)], pos, torch.ones_like(pos), scale, 1)[:, 0]
        elif impl == 'exact_decode':
            o = dec.decode(at(seqs, 0), at(seqs, 1), at(seqs, 2), at(seqs, 3), at(seqs, 4),
                           torch.tensor(ts, device=dev, dtype=torch.int32), scale)
        else:
            o = dec.step(torch.stack([qn[n][0, t] for n, t in enumerate(ts)]),
                         torch.stack([kn[n][0, t] for n, t in enumerate(ts)]), at(seqs, 2), at(seqs, 3), at(seqs, 4), scale)
        ro = torch.stack([ref[n][0][0, t] for n, t in enumerate(ts)])
        bad_rows += int((o != ro).flatten(1).any(1).sum()); rows += N
        for n, t in enumerate(ts):
            b = t + 1                     # plain decode folds a completed chunk in the step that completes it
            if b % BT == 0 and b > PS[n]:
                q, k, v, g, beta = seqs[n]
                Sref = chunk(q[:, :b], k[:, :b], v[:, :b], g[:, :b], beta[:, :b])[1][0]
                bad_states += int(not torch.equal(state(dec, n), Sref)); states += 1
    tag = f'{impl} {nw1}:{nw2}:{BV}{" rows" if inplace else " dirty" if garbage else ""}' if impl == 'exact_decode' else impl
    print(f'{tag:24s} H={H:2d} HV={HV:2d} K={K:3d} V={V:3d} beta {str(beta_dtype)[6:]:8s}: rows {rows} mismatching {bad_rows}; '
          f'boundary states {states} mismatching {bad_states}')
    return bad_rows == 0 and bad_states == 0


def run_spec(H, HV, KT=4, K=128, V=128, inplace=False, garbage=False):
    """Speculative windows: KT tokens per slot per step, the first `acc` of them the true next tokens, the rest
    wrong drafts (the same positions of another slot's sequence); acc is random in 1..window. Each step one random
    slot is idle (nrows = 0, nothing accepted), as in a batch where it has nothing to verify."""
    N, scale = len(PS), K ** -0.5
    seqs = [seq(i, H, HV, K, V) for i in range(N)]
    ref = [chunk(*s) for s in seqs]
    dec = GDNExactDecode(N, H, HV, K, V, dev, KT=KT)
    if garbage:
        dec = GDNExactDecode(N, H, HV, K, V, dev, KT=KT, storage=dirty(dec.storage))
    for n, P in enumerate(PS):
        b = BT * (P // BT)
        q, k, v, g, beta = seqs[n]
        Sb = chunk(q[:, :b], k[:, :b], v[:, :b], g[:, :b], beta[:, :b])[1][0] if b else torch.zeros(HV, K, V, device=dev)
        dec.load(n, Sb, b, q[0, b:P], k[0, b:P], v[0, b:P], g[0, b:P], beta[0, b:P])
    gen = torch.Generator(device=dev).manual_seed(7)
    pos = torch.tensor(PS, device=dev, dtype=torch.int32)
    rows = bad_rows = bad_states = states = 0
    while int(pos.max()) + KT <= T:
        folding = [n for n in range(N) if int(dec.pending[n])]          # folded at the start of this step
        nrows = GDNExactDecode.window(pos, KT)
        acc = torch.minimum(torch.randint(1, KT + 1, (N,), device=dev, generator=gen).int(), nrows)
        idle = int(torch.randint(0, N, (1,), device=dev, generator=gen))
        nrows[idle] = 0
        acc[idle] = 0
        keep = torch.arange(KT, device=dev)[None, :] < acc[:, None]      # [N, KT]
        win = []
        for x in range(5):                                               # q, k, v, g, beta
            true = torch.stack([seqs[n][x][0, int(pos[n]):int(pos[n]) + KT] for n in range(N)])
            draft = torch.stack([seqs[(n + 1) % N][x][0, int(pos[n]):int(pos[n]) + KT] for n in range(N)])
            win.append(torch.where(keep.view(N, KT, *([1] * (true.dim() - 2))), true, draft))
        o = in_place(dec, win, pos, nrows, scale, KT) if inplace else dec.step(*win, pos, nrows, scale)
        for n in range(N):
            t0, a = int(pos[n]), int(acc[n])
            bad_rows += int((o[n, :a] != ref[n][0][0, t0:t0 + a]).flatten(1).any(1).sum()); rows += a
        for n in folding:
            b = BT * (int(pos[n]) // BT)
            q, k, v, g, beta = seqs[n]
            Sref = chunk(q[:, :b], k[:, :b], v[:, :b], g[:, :b], beta[:, :b])[1][0]
            bad_states += int(not torch.equal(state(dec, n), Sref)); states += 1
        dec.commit(pos, acc)
        pos = pos + acc
    print(f'spec window KT={KT:<2d}{" rows" if inplace else " dirty" if garbage else "     "}   H={H:2d} HV={HV:2d} K={K:3d} V={V:3d}: accepted rows {rows} mismatching {bad_rows}; '
          f'boundary states {states} mismatching {bad_states}')
    return bad_rows == 0 and bad_states == 0


def run_fold(H=16, HV=32, KT=5, K=128, V=128):
    """A one-token window completes a chunk and the slot's next step is not a window (exact_gdn: a prompt cut into a
    one-token chunk, then prefill continuing it): commit + fold() must give fla's state at the boundary, and decoding
    on from it the trainer's rows."""
    q, k, v, g, beta = seq(0, H, HV, K, V)
    ref_o = chunk(q, k, v, g, beta)[0]
    scale = K ** -0.5
    dec = GDNExactDecode(2, H, HV, K, V, dev, KT=KT)
    dec.load(0, torch.zeros(HV, K, V, device=dev), 0, q[0, :63], k[0, :63], v[0, :63], g[0, :63], beta[0, :63])
    slots, pos = torch.tensor([0], device=dev, dtype=torch.int32), torch.tensor([63], device=dev, dtype=torch.int32)
    pad = lambda t: torch.cat([t[:, 63:64], t[:, 63:64].new_zeros(1, KT - 1, *t.shape[2:])], 1).contiguous()
    o = dec.step(pad(q), pad(k), pad(v), pad(g), pad(beta), pos, torch.ones_like(pos), scale, slots=slots)
    dec.commit(pos, torch.ones_like(pos), slots=slots)
    dec.fold(slots)
    dec.pending.zero_()
    S64 = chunk(q[:, :64], k[:, :64], v[:, :64], g[:, :64], beta[:, :64])[1][0]
    ok_row, ok_state = torch.equal(o[0, 0], ref_o[0, 63]), torch.equal(state(dec, 0), S64)
    one = GDNExactDecode(2, H, HV, K, V, dev, KT=1, storage=dec.storage)
    one.pending = dec.pending
    bad = 0
    for t in range(64, 70):
        o1 = one.decode(q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t], torch.tensor([t], device=dev, dtype=torch.int32),
                        scale, slots=slots)
        bad += int(not torch.equal(o1[0], ref_o[0, t]))
    print(f'fold without a following window H={H:2d} HV={HV:2d}: row 63 equal {ok_row}; state at 64 equal {ok_state}; '
          f'rows 64..69 mismatching {bad}')
    return ok_row and ok_state and bad == 0


if __name__ == '__main__':
    cases = [('exact_decode', dict(H=H, HV=HV, cfg=cfg)) for cfg in ((2, 4, 64), (2, 4, 128))
             for H, HV in ((32, 32), (16, 16), (16, 32))]
    cases += [('exact_decode', dict(H=16, HV=16, K=64, V=64)), ('exact_decode', dict(H=16, HV=32, beta_dtype=torch.float32))]
    cases += [('exact_decode', dict(H=16, HV=32, inplace=True)), ('exact_decode', dict(H=16, HV=32, garbage=True))]
    cases += [('exact_decode_ref', dict(H=H, HV=HV)) for H, HV in ((32, 32), (16, 32))]
    ok = run_state_codec() and run_state_codec(HV=2, K=64, V=64)
    ok = all([run(impl, **kw) for impl, kw in cases]) and ok
    ok = all([run_spec(H, HV, KT) for H, HV, KT in ((32, 32, 4), (16, 32, 4), (32, 32, 16), (16, 32, 16))]) and ok
    ok = all([run_spec(16, 32, KT, inplace=True) for KT in (5, 16)]) and ok
    ok = all([run_spec(16, 32, KT, garbage=True) for KT in (5, 16)]) and ok
    ok = run_fold() and ok
    print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)
