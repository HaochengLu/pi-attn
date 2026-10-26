"""E1: is fla's chunked Gated DeltaNet program row-causal and resumable bitwise?

If yes, decode can execute the trainer's own chunk program (absolute 64-token chunks) and match prefill and
the trainer bitwise. Checks:
  prefix : chunk(x[:L]) rows == chunk(x[:T]) rows                 (later tokens never touch earlier rows)
  resume : chunk(x[64c:L] | S_c) rows == chunk(x[:T]) rows         (chunked prefill / decode from a boundary state)
  decode : token-by-token replay of the partial chunk == prefill   (what a decode step would compute)
  pack   : varlen packing (cu_seqlens, trainer style) == alone
  recur  : fla's fused_recurrent (today's decode kernel) vs chunk   (the mismatch being removed)"""
import torch, torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

torch.manual_seed(0)
dev = 'cuda'
B, T, H, K, V = 1, 1024, 16, 128, 128

def inputs(T, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    q = torch.randn(1, T, H, K, device=dev, generator=g).to(torch.bfloat16)
    k = torch.randn(1, T, H, K, device=dev, generator=g).to(torch.bfloat16)
    v = torch.randn(1, T, H, V, device=dev, generator=g).to(torch.bfloat16)
    gg = F.logsigmoid(torch.randn(1, T, H, device=dev, generator=g) + 3).float()      # decay log-gates, mostly near 0
    beta = torch.sigmoid(torch.randn(1, T, H, device=dev, generator=g)).to(torch.bfloat16)
    return q, k, v, gg, beta

def run(q, k, v, g, beta, h0=None, cu=None):
    o, s = chunk_gated_delta_rule(q, k, v, g, beta, initial_state=h0, output_final_state=True,
                                  use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
    return o, s

def same(a, b):
    return torch.equal(a, b), (a.float() - b.float()).abs().max().item()

q, k, v, g, beta = inputs(T, 1)
o_full, s_full = run(q, k, v, g, beta)

bad = []
for L in [1, 2, 31, 63, 64, 65, 100, 127, 128, 129, 200, 257, 383, 511, 512, 513, 700, 1000]:
    o, _ = run(q[:, :L], k[:, :L], v[:, :L], g[:, :L], beta[:, :L])
    eq, d = same(o, o_full[:, :L])
    if not eq: bad.append((L, d))
print('prefix  mismatching L:', bad if bad else 'none')

bad = []
for c in [1, 2, 5, 9]:
    _, s_c = run(q[:, :64 * c], k[:, :64 * c], v[:, :64 * c], g[:, :64 * c], beta[:, :64 * c])
    for L in [64 * c + 1, 64 * c + 17, 64 * c + 63, 64 * c + 64, 64 * c + 200]:
        o, _ = run(q[:, 64 * c:L], k[:, 64 * c:L], v[:, 64 * c:L], g[:, 64 * c:L], beta[:, 64 * c:L], h0=s_c)
        eq, d = same(o, o_full[:, 64 * c:L])
        if not eq: bad.append((c, L, d))
print('resume  mismatching (c, L):', bad if bad else 'none')

c = 3
_, s_c = run(q[:, :64 * c], k[:, :64 * c], v[:, :64 * c], g[:, :64 * c], beta[:, :64 * c])
bad = []
for L in range(64 * c + 1, 64 * c + 65):
    o, s = run(q[:, 64 * c:L], k[:, 64 * c:L], v[:, 64 * c:L], g[:, 64 * c:L], beta[:, 64 * c:L], h0=s_c)
    eq, d = same(o[:, -1], o_full[:, L - 1])
    if not eq: bad.append((L, d))
_, s_next = run(q[:, :64 * (c + 1)], k[:, :64 * (c + 1)], v[:, :64 * (c + 1)], g[:, :64 * (c + 1)], beta[:, :64 * (c + 1)])
print('decode  mismatching L:', bad if bad else 'none', '| state fold equal:', torch.equal(s, s_next))

lens = [300, 64, 517, 143]
xs = [inputs(n, 10 + i) for i, n in enumerate(lens)]
cat = [torch.cat([x[j] for x in xs], 1) for j in range(5)]
cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device=dev, dtype=torch.int32)
o_pack, _ = run(*cat, cu=cu)
bad = []
for i, (x, n) in enumerate(zip(xs, lens)):
    o, _ = run(*x)
    eq, d = same(o, o_pack[:, cu[i]:cu[i + 1]])
    if not eq: bad.append((i, n, d))
print('pack    mismatching seqs:', bad if bad else 'none')

o_rec, s_rec = fused_recurrent_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, output_final_state=True,
                                                use_qk_l2norm_in_kernel=True)
print('recur   nan in output:', torch.isnan(o_rec).any().item(), '| nan in state:', torch.isnan(s_rec).any().item())
eq, d = same(o_rec, o_full)
rel = ((o_rec.float() - o_full.float()).norm() / o_full.float().norm()).item()
print(f'recur   fused_recurrent == chunk: {eq}, max |diff| {d:.3e}, rel l2 {rel:.3e}, rows differing '
      f'{(o_rec != o_full).any(-1).any(-1).float().mean().item()*100:.1f}%')
