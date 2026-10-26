"""Production baseline: vLLM 0.17.1's GDN decode kernel (its fla copy, fused_recurrent_gated_delta_rule as called by
qwen3_next.py / qwen3_5.py), Qwen3.5-4B shape (HV=32, K=V=128, fp32 state pool), CUDA graphs, same timing as
bench_decode.py. Run in env/venv-vllm17.
  decode   one token per slot, state updated in place in the pool (ssm_state_indices), l2norm in-kernel
  verify 4 one window of 4 tokens per slot as vLLM's speculative path runs it: a pool slot per token, so the state
           after every draft token is written (that is its rollback), num_accepted_tokens selecting the start state
usage: python gdn/bench_vllm_gdn.py"""
import torch, torch.nn.functional as F
from vllm.model_executor.layers.fla.ops import fused_recurrent_gated_delta_rule
import vllm
from timing import graph_time

dev, H, HV, K, V, T = 'cuda', 32, 32, 128, 128, 4
print(f'vllm {vllm.__version__} torch {torch.__version__}')
print(f'{"B":>4} | {"decode us":>9} | {"verify 4 us":>11}', flush=True)
for B in [1, 8, 32, 64, 128, 256]:
    gen = torch.Generator(device=dev).manual_seed(B)
    mk = lambda n: (torch.randn(1, n, H, K, device=dev, generator=gen).bfloat16(),
                    torch.randn(1, n, H, K, device=dev, generator=gen).bfloat16(),
                    torch.randn(1, n, HV, V, device=dev, generator=gen).bfloat16(),
                    F.logsigmoid(torch.randn(1, n, HV, device=dev, generator=gen) + 3).float(),
                    torch.sigmoid(torch.randn(1, n, HV, device=dev, generator=gen)).bfloat16())
    q1, k1, v1, g1, b1 = mk(B)
    pool1 = torch.randn(B, HV, V, K, device=dev, generator=gen) * 0.1
    idx1 = torch.arange(B, device=dev, dtype=torch.int32)
    cu1 = torch.arange(B + 1, device=dev, dtype=torch.int32)
    dec = lambda: fused_recurrent_gated_delta_rule(q=q1, k=k1, v=v1, g=g1, beta=b1, initial_state=pool1,
                                                   inplace_final_state=True, cu_seqlens=cu1, ssm_state_indices=idx1,
                                                   use_qk_l2norm_in_kernel=True)[0]
    q4, k4, v4, g4, b4 = mk(B * T)
    pool4 = torch.randn(B * T, HV, V, K, device=dev, generator=gen) * 0.1
    idx4 = torch.arange(B * T, device=dev, dtype=torch.int32).view(B, T)
    cu4 = torch.arange(0, B * T + 1, T, device=dev, dtype=torch.int32)
    acc = torch.randint(1, T + 1, (B,), device=dev, generator=gen).int()
    ver = lambda: fused_recurrent_gated_delta_rule(q=q4, k=k4, v=v4, g=g4, beta=b4, initial_state=pool4,
                                                   inplace_final_state=True, cu_seqlens=cu4, ssm_state_indices=idx4,
                                                   num_accepted_tokens=acc, use_qk_l2norm_in_kernel=True)[0]
    print(f'{B:4d} | {graph_time(dec):9.1f} | {graph_time(ver):11.1f}', flush=True)
