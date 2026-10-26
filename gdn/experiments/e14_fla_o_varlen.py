"""E14: is fla 0.5.0's chunk_fwd_kernel_o (the output stage of chunk_gated_delta_rule) the same program in its
fixed-length and varlen (cu_seqlens) variants? Same inputs (random, Qwen3.5-4B head layout H=16 / HV=32, K=V=128),
the earlier stages run once; the output kernel runs both ways on one sequence. Prints the differing elements, with
fla as installed and after exact_decode.pin_fla_configs() (which removes the output kernel's config whose bits differ
on sm_89 with Triton 3.6.0; the two variants are tuned separately and can pick different configs).
usage: python e14_fla_o_varlen.py [T] [SEEDS]"""
import sys, torch, torch.nn.functional as F
import os, triton
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exact_decode import pin_fla_configs
from fla.modules.l2norm import l2norm_fwd
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import chunk_fwd_o
T = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
dev, H, HV, K, V = 'cuda', 16, 32, 128, 128
print(f'GPU {torch.cuda.get_device_name()}  Triton {triton.__version__}  T={T}')
def compare(label):
    total = 0
    for seed in range(SEEDS):
        gen = torch.Generator(device=dev).manual_seed(seed)
        r = lambda *s: torch.randn(*s, device=dev, generator=gen)
        q, k = l2norm_fwd(r(1, T, H, K).bfloat16())[0], l2norm_fwd(r(1, T, H, K).bfloat16())[0]
        v = r(1, T, HV, V).bfloat16()
        g = chunk_local_cumsum(F.logsigmoid(r(1, T, HV) + 3).float(), chunk_size=64, scale=RCP_LN2)
        beta = torch.sigmoid(r(1, T, HV)).bfloat16()
        w, u, _ = chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=g, beta=beta, use_exp2=True)
        h, vn, _ = chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, g=g, initial_state=None, output_final_state=False,
                                                use_exp2=True)
        cu = torch.tensor([0, T], device=dev, dtype=torch.int32)
        ci = torch.stack([torch.zeros(T // 64, dtype=torch.int32), torch.arange(T // 64, dtype=torch.int32)], 1).to(dev)
        o_fix = chunk_fwd_o(q=q, k=k, v=vn, h=h, g=g, scale=K ** -0.5, use_exp2=True)
        o_var = chunk_fwd_o(q=q, k=k, v=vn, h=h, g=g, scale=K ** -0.5, cu_seqlens=cu, chunk_indices=ci, use_exp2=True)
        d = int((o_fix != o_var).sum())
        total += d
        print(f'{label} seed {seed}: output elements differing fixed-length vs varlen {d} of {o_fix.numel()}')
    print(f'{label}: fixed-length and varlen output kernels', 'agree' if total == 0 else 'DIFFER')


compare('fla as installed     ')
pin_fla_configs()
compare('after pin_fla_configs')
