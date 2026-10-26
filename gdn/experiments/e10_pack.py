"""E10: is stock fla's GDN forward batch-invariant at trainer batch sizes? One fresh process: a 1024-token sequence
alone vs the same sequence packed (cu_seqlens) with other sequences into a ~40k-token trainer micro-batch.
fla's l2norm autotune key includes NB = cdiv(rows, 65536) (rows = tokens x heads), so the two runs are tuned
separately; E4 showed num_warps 8/16 change the bits. Prints whether the sequence's output rows differ."""
import sys, torch, torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
import fla.modules.l2norm as L
dev, HV, K = 'cuda', 32, 128
gen = torch.Generator(device=dev).manual_seed(int(sys.argv[1]))
def mk(n):
    return (torch.randn(1, n, HV, K, device=dev, generator=gen).bfloat16(), torch.randn(1, n, HV, K, device=dev, generator=gen).bfloat16(),
            torch.randn(1, n, HV, K, device=dev, generator=gen).bfloat16(),
            F.logsigmoid(torch.randn(1, n, HV, device=dev, generator=gen) + 3).float(),
            torch.sigmoid(torch.randn(1, n, HV, device=dev, generator=gen)).bfloat16())
x = mk(1024); others = mk(39 * 1024)
o_alone = chunk_gated_delta_rule(*x, use_qk_l2norm_in_kernel=True)[0]
packed = [torch.cat([a, b], 1) for a, b in zip(x, others)]
cu = torch.tensor([0, 1024] + [1024 * (j + 2) for j in range(39)], device=dev, dtype=torch.int32)
o_pack = chunk_gated_delta_rule(*packed, use_qk_l2norm_in_kernel=True, cu_seqlens=cu)[0][:, :1024]
at = L.l2norm_fwd_kernel
while not hasattr(at, 'cache'): at = at.fn
picks = {str(k[1]): (v.kwargs['BT'], v.num_warps) for k, v in at.cache.items()}
rows = (o_alone != o_pack).flatten(2).any(-1).float().mean().item()
print(f'seed {sys.argv[1]}: l2norm picks (NB: (BT, warps)) {picks}  rows differing alone-vs-packed {rows*100:.1f}%')
