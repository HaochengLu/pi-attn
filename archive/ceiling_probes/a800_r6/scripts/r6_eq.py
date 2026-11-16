"""Bit equality of two builds of the prefill and decode kernels: r6_eq.py <ref tree> <new tree> [seeds]
(trees: directories of $R6 with kernels/). Every prefill variant the GPU runs, split counts, grid orders 1-3 and both loop variants of the new build,
variable lengths, prefixes; inputs: Gaussian at several magnitudes and special values injected into Q, K and V
(+-0, subnormals, largest and smallest normals; in Q and K also +-inf and NaN). Each launch of the new build is
compared with the same launch of the reference build and with the reference build's first launch of the shape.
Regime nonfinite_v (+-inf and NaN in V too) is compared launch against launch only: a masked key's P = 0 times a
non-finite V is NaN, so with non-finite V a row's output depends on which later keys share its CTA, in both builds."""
import math, os, sys
import torch
from torch.utils.cpp_extension import load
sys.path.insert(0, os.getcwd())
from build import cuda_build_flags
def build(tree):
    src = os.path.join(os.environ["R6"], tree, "kernels")
    arch, cflags, includes, ldflags = cuda_build_flags(None)
    return load(name="r6eq_" + tree.replace("/", "_"), sources=[os.path.join(src, "pi_attn_torch.cu")], extra_include_paths=[src] + includes,
                extra_cuda_cflags=cflags + ["-lineinfo"], extra_ldflags=ldflags)
ref, new = build(sys.argv[1]), build(sys.argv[2])
SEEDS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
dev, PAGE, D = "cuda", 16, 128
SPECIAL = torch.tensor([0.0, -0.0, float("inf"), float("-inf"), float("nan"), 1e-40, -1e-40, 3.38e38, -3.38e38, 1.2e-38, -1.2e-38, 9.2e-41], dtype=torch.float32)
FINITE = SPECIAL[~(SPECIAL.isnan() | SPECIAL.isinf())]
def data(shape, regime, gen, finite=False):
    x = torch.randn(shape, generator=gen, device=dev)
    if regime == "small": x = x * 1e-3
    if regime == "large": x = x * 40
    if regime == "tiny": x = x * 1e-18
    if regime == "huge": x = x * 1e18
    if regime in ("special2", "special30", "nonfinite_v"):
        sp = FINITE if finite else SPECIAL
        m = torch.rand(shape, generator=gen, device=dev) < (0.3 if regime == "special30" else 0.02)
        idx = torch.randint(0, len(sp), shape, generator=gen, device=dev)
        x = torch.where(m, sp.to(dev)[idx], x)
    return x.to(torch.bfloat16)
bits = lambda t: t.contiguous().view(torch.int16)
total = bad = 0
SHAPES = [([1], [0], 16, 8), ([33], [0], 16, 8), ([64], [0], 16, 8), ([512], [0], 16, 8), ([700], [0], 32, 8), ([1300], [0], 16, 8), ([2048], [0], 16, 8),
          ([97], [1000], 16, 8), ([512], [3000], 16, 8), ([64], [5000], 32, 8), ([300, 17, 801, 64], [0, 500, 33, 2000], 16, 8),
          ([129] * 9, [0] * 9, 16, 8), ([40] * 30, [7] * 30, 16, 8), ([257, 31], [511, 513], 8, 8), ([200, 90], [100, 1500], 24, 8), ([1100], [0], 4, 4), ([4500], [0], 16, 8)]
for seed in range(SEEDS):
    gen = torch.Generator(device=dev); gen.manual_seed(1234 + seed)
    for regime in ["normal", "small", "large", "tiny", "huge", "special2", "special30", "nonfinite_v"]:
        for qlens, prefixes, hq, hkv in SHAPES:
            batch = len(qlens); kvl = [q + p for q, p in zip(qlens, prefixes)]
            pages_per = [(k + PAGE - 1) // PAGE for k in kvl]; n = sum(pages_per) + 1
            k = data((n, PAGE, hkv, D), regime, gen); v = data((n, PAGE, hkv, D), "normal" if regime in ("tiny", "huge") else regime, gen, finite=regime != "nonfinite_v")
            q = data((sum(qlens), hq, D), regime, gen)
            perm = (torch.randperm(n - 1, generator=gen, device=dev).int() + 1).contiguous()
            indptr = torch.tensor([0] + list(torch.tensor(pages_per).cumsum(0)), device=dev, dtype=torch.int32)
            qo = torch.tensor([0] + list(torch.tensor(qlens).cumsum(0)), device=dev, dtype=torch.int32)
            lens = torch.tensor(kvl, device=dev, dtype=torch.int32)
            mq, mk = max(qlens), max(kvl)
            base = None
            for var in (0, 1, 2, 3, 4):
                for sp in (0, 1, 2, 3, 7):
                    for order in (1, 2, 3) if hasattr(new, 'set_prefill_loop') else (1, 2):
                        outs = []
                        for e in (ref, new):
                            e.set_prefill_order(order if hasattr(e, 'set_prefill_loop') or order < 3 else 1)  # the reference build has no order 3
                            if hasattr(e, "set_prefill_loop"): e.set_prefill_loop(1 + (total + order) % 2)  # masked loop alone / with the unmasked loop
                            outs.append(bits(e.prefill(q, k, v, qo, indptr, perm, lens, mq, mk, 1 / math.sqrt(D), sp, var)))
                        if base is None: base = outs[0]
                        total += 1
                        if not (torch.equal(outs[0], outs[1]) and (regime == "nonfinite_v" or torch.equal(outs[1], base))):
                            bad += 1
                            if bad <= 10: print("MISMATCH", regime, qlens[:4], prefixes[:4], hq, var, sp, order, int((outs[0] != outs[1]).sum()), int((outs[1] != base).sum()), flush=True)
        print(f"seed {seed} regime {regime}: {total} launch pairs so far, {bad} mismatches", flush=True)
print(f"EQ prefill {sys.argv[2]} vs {sys.argv[1]} (variants 0-4, splits 0/1/2/3/7, grid orders 1-3 and both loop variants of the new build, 8 input regimes, {len(SHAPES)} shapes, {SEEDS} seeds): {total} launch pairs, each also against the reference build's first launch of the shape, {bad} mismatches")
sys.exit(1 if bad else 0)
