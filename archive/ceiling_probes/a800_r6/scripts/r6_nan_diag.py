import math, os, sys, torch
sys.path.insert(0, os.getcwd())
from torch.utils.cpp_extension import load
from build import cuda_build_flags
def build(tree):
    src = os.path.join(os.environ["R6"], tree, "kernels")
    arch, cflags, includes, ldflags = cuda_build_flags(None)
    return load(name="r6eq_" + tree.replace("/", "_"), sources=[os.path.join(src, "pi_attn_torch.cu")], extra_include_paths=[src] + includes,
                extra_cuda_cflags=cflags + ["-lineinfo"], extra_ldflags=ldflags)
e = build(sys.argv[1])
dev, PAGE, D = "cuda", 16, 128
SPECIAL = torch.tensor([0.0, -0.0, float("inf"), float("-inf"), float("nan"), 1e-40, -1e-40, 3.38e38, -3.38e38, 1.2e-38, -1.2e-38, 9.2e-41], dtype=torch.float32)
gen = torch.Generator(device=dev)
for seed in range(6):
  for ql in (64, 100, 512):
    gen.manual_seed(seed)
    def data(shape):
        x = torch.randn(shape, generator=gen, device=dev)
        m = torch.rand(shape, generator=gen, device=dev) < 0.02
        idx = torch.randint(0, len(SPECIAL), shape, generator=gen, device=dev)
        return torch.where(m, SPECIAL.to(dev)[idx], x).to(torch.bfloat16)
    hq, hkv = 16, 8
    n = (ql + 15) // 16 + 1
    k, v, q = data((n, PAGE, hkv, D)), data((n, PAGE, hkv, D)), data((ql, hq, D))
    perm = (torch.randperm(n - 1, generator=gen, device=dev).int() + 1).contiguous()
    indptr = torch.tensor([0, n - 1], device=dev, dtype=torch.int32); qo = torch.tensor([0, ql], device=dev, dtype=torch.int32)
    lens = torch.tensor([ql], device=dev, dtype=torch.int32)
    outs = {var: e.prefill(q, k, v, qo, indptr, perm, lens, ql, ql, 1 / math.sqrt(D), 1, var) for var in (1, 2, 3, 4)}
    b = lambda t: t.contiguous().view(torch.int16)
    for var in (2, 3, 4):
        d = b(outs[var]) != b(outs[1])
        if d.any():
            a, c = outs[1][d].float(), outs[var][d].float()
            rows = d.any(-1).nonzero()
            print(f"seed {seed} q={ql} var {var} vs 1: {int(d.sum())} elements differ; both NaN: {int((a.isnan() & c.isnan()).sum())}; ref bits {sorted(set(b(outs[1])[d].tolist()))[:6]} var bits {sorted(set(b(outs[var])[d].tolist()))[:6]}; rows(token,head) {rows[:6].tolist()} of {len(rows)}")
        else: print(f"seed {seed} q={ql} var {var} vs 1: equal")
