"""E4b: which l2norm_fwd config does fla's autotune actually pick for realistic row counts (tokens x 32 heads, D=128)?
Run in fresh processes with FLA_CACHE_RESULTS=0 so every run benchmarks anew."""
import torch, fla.modules.l2norm as L
at = L.l2norm_fwd_kernel
while not hasattr(at, 'cache'): at = at.fn
for tokens in [1, 64, 1024, 2048, 2049, 8192, 32768, 131072]:
    x = torch.randn(tokens * 32, 128, device='cuda', dtype=torch.bfloat16)
    L.l2norm_fwd(x)
    torch.cuda.synchronize()
    picks = {k: (v.kwargs.get('BT'), v.num_warps) for k, v in at.cache.items()}
    print(tokens, picks[max(picks, key=lambda k: str(k) in str(list(at.cache.keys())[-1]))] if picks else None,
          'keys', len(picks), flush=True)
print('final cache:', {str(k)[:40]: (v.kwargs.get('BT'), v.num_warps) for k, v in at.cache.items()})
