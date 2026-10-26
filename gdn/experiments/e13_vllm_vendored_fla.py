"""E13: vLLM 0.17.1's vendored fla chunk_gated_delta_rule vs fla 0.5.0's, bitwise (same process, same inputs, each with its own state layout). Run in env/venv-vllm17 with PYTHONPATH=env/fla_pkgs."""
import torch, torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as vllm_chunk
dev = 'cuda'
for H, HV, T in [(16, 32, 1000), (32, 32, 517), (16, 16, 64)]:
    g0 = torch.Generator(device=dev).manual_seed(T)
    q = torch.randn(1, T, H, 128, device=dev, generator=g0).bfloat16(); k = torch.randn(1, T, H, 128, device=dev, generator=g0).bfloat16()
    v = torch.randn(1, T, HV, 128, device=dev, generator=g0).bfloat16()
    g = F.logsigmoid(torch.randn(1, T, HV, device=dev, generator=g0) + 3).float()
    beta = torch.sigmoid(torch.randn(1, T, HV, device=dev, generator=g0)).bfloat16()
    h0 = torch.randn(1, HV, 128, 128, device=dev, generator=g0) * 0.1
    o1, s1 = fla_chunk(q, k, v, g, beta, initial_state=h0, output_final_state=True, use_qk_l2norm_in_kernel=True)
    # vLLM keeps the state as [N, HV, V, K] (fla 0.5.0: [N, HV, K, V])
    o2, s2 = vllm_chunk(q=q, k=k, v=v, g=g, beta=beta, initial_state=h0.transpose(-1, -2).contiguous(),
                        output_final_state=True, use_qk_l2norm_in_kernel=True)
    s2 = s2.transpose(-1, -2)
    print(f'H={H} HV={HV} T={T}: o equal {torch.equal(o1, o2)} (max |d| {(o1.float()-o2.float()).abs().max().item():.2e}), '
          f'state equal {torch.equal(s1, s2)} (max |d| {(s1 - s2).abs().max().item():.2e})')
