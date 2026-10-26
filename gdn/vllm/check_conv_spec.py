"""Is vLLM 0.17.1's speculative conv1d path (causal_conv1d_update over windows of 1 + drafts, rolled back through
num_accepted_tokens) bitwise equal to its prefill conv on the accepted tokens? Qwen3.5-4B conv: dim 8192, width 4,
4 drafts (conv state length 3 + 4). Windows of 1..5 tokens (fewer drafts near a 64-token boundary or without a
match), as vLLM passes them: max_query_len 5; accepted tokens are the true sequence, rejected ones random.
Then the order a prompt takes in exact_gdn when the token budget cuts it into one-token chunks: prefill conv, one-token
updates through the speculative path (num_accepted_tokens 1), prefill conv again from the state they left."""
import torch
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
dev, dim, W, D, T, P = 'cuda', 8192, 4, 4, 400, 100
g0 = torch.Generator(device=dev).manual_seed(0)
x = torch.randn(dim, T, device=dev, generator=g0).bfloat16()
w = torch.randn(dim, W, device=dev, generator=g0).bfloat16() * 0.3
store = torch.zeros(2, W - 1 + D, dim, device=dev, dtype=torch.bfloat16)
cs = store.transpose(-1, -2)
i32 = lambda *v: torch.tensor(list(v), device=dev, dtype=torch.int32)
full = causal_conv1d_fn(x.clone(), w, None, conv_states=cs, query_start_loc=i32(0, T), cache_indices=i32(0),
                        has_initial_state=torch.tensor([False], device=dev), activation='silu')
causal_conv1d_fn(x[:, :P].clone().contiguous(), w, None, conv_states=cs, query_start_loc=i32(0, P),
                 cache_indices=i32(1), has_initial_state=torch.tensor([False], device=dev), activation='silu')
t, acc_prev, bad, rows = P, 1, 0, 0
gen = torch.Generator().manual_seed(1)
while t + 1 + D <= T:
    n = int(torch.randint(1, 2 + D, (1,), generator=gen))           # window length
    a = int(torch.randint(1, n + 1, (1,), generator=gen))           # tokens of this window that will be accepted
    win = x[:, t:t + n].clone()
    win[:, a:] = torch.randn(dim, n - a, device=dev, generator=g0).bfloat16()
    y = causal_conv1d_update(win.transpose(0, 1).contiguous(), cs, w, None, 'silu', conv_state_indices=i32(1),
                             num_accepted_tokens=i32(acc_prev), query_start_loc=i32(0, n),
                             max_query_len=1 + D, validate_data=False)
    bad += int((y[:a].transpose(0, 1) != full[:, t:t + a]).any(0).sum()); rows += a
    t, acc_prev = t + a, a
print(f'speculative conv windows: accepted rows {rows}, rows differing from the prefill conv {bad}')

# prompt: prefill chunk, two one-token chunks through the speculative update, prefill chunk resuming from that state
causal_conv1d_fn(x[:, :P].clone().contiguous(), w, None, conv_states=cs, query_start_loc=i32(0, P),
                 cache_indices=i32(1), has_initial_state=torch.tensor([False], device=dev), activation='silu')
ys = [causal_conv1d_update(x[:, t:t + 1].transpose(0, 1).contiguous(), cs, w, None, 'silu', conv_state_indices=i32(1),
                           num_accepted_tokens=i32(1), query_start_loc=i32(0, 1), max_query_len=1 + D,
                           validate_data=False).transpose(0, 1) for t in (P, P + 1)]
yp = causal_conv1d_fn(x[:, P + 2:P + 40].clone().contiguous(), w, None, conv_states=cs, query_start_loc=i32(0, 38),
                      cache_indices=i32(1), has_initial_state=torch.tensor([True], device=dev), activation='silu')
y = torch.cat(ys + [yp], 1)
print(f'prefill, two one-token speculative updates, prefill: rows differing from the full prefill conv '
      f'{int((y != full[:, P:P + 40]).any(0).sum())} of 40')
