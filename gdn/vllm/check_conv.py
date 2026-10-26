"""Is vLLM 0.17.1's causal conv1d prefill kernel (causal_conv1d_fn) bitwise equal to its decode kernel
(causal_conv1d_update) on the same sequence? Qwen3.5-4B GDN conv: dim 8192, width 4, bf16, silu.
Conv state laid out as vLLM's GDN layer passes it: (slots, state_len, dim) storage, transposed to (slots, dim, state_len)."""
import torch
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
dev, dim, W, T, P = 'cuda', 8192, 4, 300, 100
g0 = torch.Generator(device=dev).manual_seed(0)
x = torch.randn(dim, T, device=dev, generator=g0).bfloat16()
w = torch.randn(dim, W, device=dev, generator=g0).bfloat16() * 0.3
store = torch.zeros(2, W - 1, dim, device=dev, dtype=torch.bfloat16)
cs = store.transpose(-1, -2)
qsl = lambda n: torch.tensor([0, n], device=dev, dtype=torch.int32)
full = causal_conv1d_fn(x.clone(), w, None, conv_states=cs, query_start_loc=qsl(T),
                        cache_indices=torch.tensor([0], device=dev, dtype=torch.int32),
                        has_initial_state=torch.tensor([False], device=dev), activation='silu')
pre = causal_conv1d_fn(x[:, :P].clone().contiguous(), w, None, conv_states=cs, query_start_loc=qsl(P),
                       cache_indices=torch.tensor([1], device=dev, dtype=torch.int32),
                       has_initial_state=torch.tensor([False], device=dev), activation='silu')
dec = []
for t in range(P, T):
    y = causal_conv1d_update(x[:, t][None].clone(), cs, w, None, 'silu',
                             conv_state_indices=torch.tensor([1], device=dev, dtype=torch.int32))
    dec.append(y[0])
dec = torch.stack(dec, 1)
print(f'prefix prefill == full rows: {torch.equal(pre, full[:, :P])}; decode == full rows: '
      f'{torch.equal(dec, full[:, P:])} (rows differing {(dec != full[:, P:]).any(0).float().mean().item() * 100:.1f}%, '
      f'max |d| {(dec.float() - full[:, P:].float()).abs().max().item():.3e})')
