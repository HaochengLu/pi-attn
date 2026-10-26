"""Bitwise test: exact GDN decode vs fla's chunk_gated_delta_rule full forward (the trainer's program).

For each head layout, 8 slots each get their own random sequence and their own prefill length P (so slots sit at
different rows of their chunks, including boundaries); the prefill is fla's chunk over [0, b) for the boundary state
plus the partial chunk [b, P) loaded into the decoder; then all slots decode token by token to T together.
Checked: every decoded row against the full forward's row, and the fp32 state at every 64-token boundary against
fla's final state of the prefix. Both implementations: exact_decode (fast) and exact_decode_ref (stateless).
Speculative mode (exact_decode): windows of up to 4 or 16 tokens per step, a random number of them accepted per slot,
one random slot idle per step; rejected positions carry wrong-draft inputs. Accepted rows must equal the trainer's rows bitwise and the
rejected drafts must leave no trace in later rows or states.

usage: PYTHONPATH=<fla> python gdn/test_exact.py            (exit status 1 on any mismatch)"""
import sys, torch, torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.modules.l2norm import l2norm_fwd
from exact_decode import GDNExactDecode, BT
from exact_decode_ref import GDNExactDecodeRef

dev, T = 'cuda', 700
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


def run(impl, H, HV, K=128, V=128, cfg=(2, 4, 64), beta_dtype=torch.bfloat16):
    N, scale = len(PS), K ** -0.5
    seqs = [seq(i, H, HV, K, V, beta_dtype) for i in range(N)]
    ref = [chunk(*s) for s in seqs]
    qn = [l2norm_fwd(s[0])[0] for s in seqs]              # the reference implementation takes normalised q, k
    kn = [l2norm_fwd(s[1])[0] for s in seqs]
    nw1, nw2, BV = cfg
    dec = GDNExactDecode(N, H, HV, K, V, dev, beta_dtype=beta_dtype, BV=BV, nw1=nw1, nw2=nw2) \
        if impl == 'exact_decode' else GDNExactDecodeRef(N, H, HV, K, V, dev)
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
        if impl == 'exact_decode':
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
                bad_states += int(not torch.equal(dec.S[n], Sref)); states += 1
    tag = f'{impl} {nw1}:{nw2}:{BV}' if impl == 'exact_decode' else impl
    print(f'{tag:24s} H={H:2d} HV={HV:2d} K={K:3d} V={V:3d} beta {str(beta_dtype)[6:]:8s}: rows {rows} mismatching {bad_rows}; '
          f'boundary states {states} mismatching {bad_states}')
    return bad_rows == 0 and bad_states == 0


def run_spec(H, HV, KT=4, K=128, V=128):
    """Speculative windows: KT tokens per slot per step, the first `acc` of them the true next tokens, the rest
    wrong drafts (the same positions of another slot's sequence); acc is random in 1..window. Each step one random
    slot is idle (nrows = 0, nothing accepted), as in a batch where it has nothing to verify."""
    N, scale = len(PS), K ** -0.5
    seqs = [seq(i, H, HV, K, V) for i in range(N)]
    ref = [chunk(*s) for s in seqs]
    dec = GDNExactDecode(N, H, HV, K, V, dev, KT=KT)
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
        o = dec.step(*win, pos, nrows, scale)
        for n in range(N):
            t0, a = int(pos[n]), int(acc[n])
            bad_rows += int((o[n, :a] != ref[n][0][0, t0:t0 + a]).flatten(1).any(1).sum()); rows += a
        for n in folding:
            b = BT * (int(pos[n]) // BT)
            q, k, v, g, beta = seqs[n]
            Sref = chunk(q[:, :b], k[:, :b], v[:, :b], g[:, :b], beta[:, :b])[1][0]
            bad_states += int(not torch.equal(dec.S[n], Sref)); states += 1
        dec.commit(pos, acc)
        pos = pos + acc
    print(f'spec window KT={KT}          H={H:2d} HV={HV:2d} K={K:3d} V={V:3d}: accepted rows {rows} mismatching {bad_rows}; '
          f'boundary states {states} mismatching {bad_states}')
    return bad_rows == 0 and bad_states == 0


if __name__ == '__main__':
    cases = [('exact_decode', dict(H=H, HV=HV, cfg=cfg)) for cfg in ((2, 4, 64), (2, 4, 128))
             for H, HV in ((32, 32), (16, 16), (16, 32))]
    cases += [('exact_decode', dict(H=16, HV=16, K=64, V=64)), ('exact_decode', dict(H=16, HV=32, beta_dtype=torch.float32))]
    cases += [('exact_decode_ref', dict(H=H, HV=HV)) for H, HV in ((32, 32), (16, 32))]
    ok = all([run(impl, **kw) for impl, kw in cases])
    ok = all([run_spec(H, HV, KT) for H, HV, KT in ((32, 32, 4), (16, 32, 4), (32, 32, 16))]) and ok
    print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)
