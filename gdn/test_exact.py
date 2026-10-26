"""Bitwise test: exact GDN decode vs fla's chunk_gated_delta_rule full forward (the trainer's program).

For each head layout, 8 slots each get their own random sequence and their own prefill length P (so slots sit at
different rows of their chunks, including boundaries); the prefill is fla's chunk over [0, b) for the boundary state
plus the partial chunk [b, P) loaded into the decoder; then all slots decode token by token to T together.
Checked: every decoded row against the full forward's row, and the fp32 state at every 64-token boundary against
fla's final state of the prefix. Both implementations: exact_decode (fast) and exact_decode_ref (stateless).

usage: PYTHONPATH=<fla> python gdn/test_exact.py            (exit status 1 on any mismatch)"""
import sys, torch, torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.modules.l2norm import l2norm_fwd
from exact_decode import GDNExactDecode, BT
from exact_decode_ref import GDNExactDecodeRef

dev, K, V, T = 'cuda', 128, 128, 700
PS = [1, 63, 64, 65, 130, 200, 255, 333]


def seq(i, H, HV):
    gen = torch.Generator(device=dev).manual_seed(100 + i)
    q = torch.randn(1, T, H, K, device=dev, generator=gen).bfloat16()
    k = torch.randn(1, T, H, K, device=dev, generator=gen).bfloat16()
    v = torch.randn(1, T, HV, V, device=dev, generator=gen).bfloat16()
    g = F.logsigmoid(torch.randn(1, T, HV, device=dev, generator=gen) + 3).float()
    beta = torch.sigmoid(torch.randn(1, T, HV, device=dev, generator=gen)).bfloat16()
    return q, k, v, g, beta


def chunk(q, k, v, g, beta):
    return chunk_gated_delta_rule(q, k, v, g, beta, output_final_state=True, use_qk_l2norm_in_kernel=True)


def run(impl, H, HV):
    N, scale = len(PS), K ** -0.5
    seqs = [seq(i, H, HV) for i in range(N)]
    ref = [chunk(*s) for s in seqs]
    qn = [l2norm_fwd(s[0])[0] for s in seqs]              # the reference implementation takes normalised q, k
    kn = [l2norm_fwd(s[1])[0] for s in seqs]
    dec = GDNExactDecode(N, H, HV, K, V, dev) if impl == 'exact_decode' else GDNExactDecodeRef(N, H, HV, K, V, dev)
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
            o = dec.step(at(seqs, 0), at(seqs, 1), at(seqs, 2), at(seqs, 3), at(seqs, 4),
                         torch.tensor(ts, device=dev, dtype=torch.int32), scale)
        else:
            o = dec.step(torch.stack([qn[n][0, t] for n, t in enumerate(ts)]),
                         torch.stack([kn[n][0, t] for n, t in enumerate(ts)]), at(seqs, 2), at(seqs, 3), at(seqs, 4), scale)
        ro = torch.stack([ref[n][0][0, t] for n, t in enumerate(ts)])
        bad_rows += int((o != ro).flatten(1).any(1).sum()); rows += N
        for n, t in enumerate(ts):
            if (t + 1) % BT == 0:
                q, k, v, g, beta = seqs[n]
                Sref = chunk(q[:, :t + 1], k[:, :t + 1], v[:, :t + 1], g[:, :t + 1], beta[:, :t + 1])[1][0]
                bad_states += int(not torch.equal(dec.S[n], Sref)); states += 1
    print(f'{impl:17s} H={H:2d} HV={HV:2d}: rows {rows} mismatching {bad_rows}; boundary states {states} mismatching {bad_states}')
    return bad_rows == 0 and bad_states == 0


if __name__ == '__main__':
    ok = all(run(impl, H, HV) for impl in ('exact_decode', 'exact_decode_ref') for H, HV in ((32, 32), (16, 16), (16, 32)))
    print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)
