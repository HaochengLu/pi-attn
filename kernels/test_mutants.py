"""Negative controls: the invariance checks must catch kernels that break the canonical program.

Attention: each mutant is a copy of the shipped kernel sources with one deliberate change to
one computation path (the shipped files are not touched). test_invariance's checks run on the
mutant build, and the check that compares the mutated path with the others must fail.

GEMM: the M-invariance check of test_gemm run on cuBLAS (torch.matmul), which picks its
algorithm by M, and the segment-fold check run with a wrong segment count, must both fail.

The unmutated kernels pass the same checks in test_invariance.py and test_gemm.py.
"""
import os
import shutil
import sys

import torch
from torch.utils.cpp_extension import load

import test_invariance as T

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = ("pi_attn_torch.cu", "pi_attn_kernels.cu", "pi_attn.cuh")

# name: (file, shipped text, mutated text, checks that must fail)
MUTANTS = {
    # The split-prefill merge folds a row's fold groups in descending order.
    "merge_reversed": ("pi_attn_kernels.cu",
                       "    const float2 ml = ws_ml[base + fg];\n"
                       "    const FoldScalars f = fold_scalars(M, L, ml.x, ml.y);\n"
                       "    O = fold_elem(O, ws_o[(base + fg) * kHeadDim + d], f);",
                       "    const int rg = n_groups - 1 - fg;\n"
                       "    const float2 ml = ws_ml[base + rg];\n"
                       "    const FoldScalars f = fold_scalars(M, L, ml.x, ml.y);\n"
                       "    O = fold_elem(O, ws_o[(base + rg) * kHeadDim + d], f);",
                       {"split"}),
    # The decode merge folds with two roundings (o * alpha, then + o_c * beta) instead of one fma:
    # a one-ulp fp32 difference that reaches the bf16 output only for a few elements in 10^5.
    "decode_fold_unfused": ("pi_attn_kernels.cu",
                            "        Of[0] = fold_elem(Of[0], oc[i].x, f);\n"
                            "        Of[1] = fold_elem(Of[1], oc[i].y, f);\n"
                            "        Of[2] = fold_elem(Of[2], oc[i].z, f);\n"
                            "        Of[3] = fold_elem(Of[3], oc[i].w, f);",
                            "        auto unfused = [&](float o, float c) {\n"
                            "          return (f.beta == 0.f && f.alpha == 1.f) ? o : __fadd_rn(__fmul_rn(o, f.alpha), __fmul_rn(c, f.beta));\n"
                            "        };\n"
                            "        Of[0] = unfused(Of[0], oc[i].x);\n"
                            "        Of[1] = unfused(Of[1], oc[i].y);\n"
                            "        Of[2] = unfused(Of[2], oc[i].z);\n"
                            "        Of[3] = unfused(Of[3], oc[i].w);",
                            {"path"}),
    # Prefill (groups folded in registers) closes a fold group every 256 keys instead of 512.
    "prefill_group_256": ("pi_attn_kernels.cu",
                          "if (t % kGroupTiles == kGroupTiles - 1 || t == t_last)",
                          "if (t % (kGroupTiles / 2) == kGroupTiles / 2 - 1 || t == t_last)",
                          {"path"}),
}


def build_mutant(name, file, old, new):
    d = os.path.join(os.environ.get("TMPDIR", "/tmp"), "pi_mutants", name)
    os.makedirs(d, exist_ok=True)
    for f in SOURCES:
        shutil.copyfile(os.path.join(HERE, f), os.path.join(d, f))
    path = os.path.join(d, file)
    src = open(path).read()
    assert src.count(old) == 1, f"{name}: the shipped text to mutate must occur exactly once in {file}"
    open(path, "w").write(src.replace(old, new))
    major, minor = torch.cuda.get_device_capability()
    return load(name=f"pi_attn_mutant_{name}", sources=[os.path.join(d, "pi_attn_torch.cu")],
                extra_include_paths=[d], build_directory=d,
                extra_cuda_cflags=["-O3", "-std=c++17", f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}",
                                   "--expt-relaxed-constexpr"])


def attention_mutants():
    bad = []
    shipped = T.ext
    for name, (file, old, new, must_fail) in MUTANTS.items():
        T.ext = build_mutant(name, file, old, new)
        T.KVSPLITS, T.VARIANT = 1, 0
        T.COUNTERS.zero_()
        fails = T.run_checks()
        caught = {f[0].split()[0] for f in fails}
        print(f"mutant {name}: {len(fails)} failed checks, kinds {sorted(caught)}", flush=True)
        if not must_fail <= caught:
            bad.append((name, "not caught by", sorted(must_fail - caught)))
    T.ext = shipped
    return bad


def gemm_controls():
    import test_gemm as G
    bad = []
    # cuBLAS through the M-invariance check: row 0 alone vs inside M rows, for every projection.
    caught = []
    for name, (N, K) in G.SHAPES.items():
        W = (torch.randn(N, K, device="cuda") * 0.02).bfloat16()
        X = torch.randn(4096, K, device="cuda").bfloat16()
        ref = torch.matmul(X[:1], W.T)[0]
        if any(not torch.equal(torch.matmul(X[:M], W.T)[0].view(torch.int16), ref.view(torch.int16))
               for M in [2, 7, 8, 9, 16, 17, 32, 33, 64, 65, 200, 1000, 4096]):
            caught.append(name)
    print(f"cuBLAS M-invariance control: {len(caught)}/{len(G.SHAPES)} projections vary with M", flush=True)
    if not caught:
        bad.append(("cuBLAS", "M-invariance check never fails"))
    # Segment fold with the wrong count: S = 1 where the weight's canonical S is 2.
    N, K = G.SHAPES["0.6B down"]
    assert G.gemm_segments(N, K) == 2
    W = (torch.randn(N, K, device="cuda") * 0.02).bfloat16()
    X = torch.randn(64, K, device="cuda").bfloat16()
    fails = []
    ks = K // 2
    tot = torch.zeros(64, N, device="cuda")
    for sg in range(2):
        tot = tot + G.ext.linear(X[:, sg * ks:(sg + 1) * ks].contiguous(), W[:, sg * ks:(sg + 1) * ks].contiguous(), 1, True, 1)
    y = G.ext.linear(X, W, 1, False, 1)
    print(f"wrong segment count control: S=1 result equals the S=2 fold: {torch.equal(y.view(torch.int16), tot.bfloat16().view(torch.int16))}", flush=True)
    if torch.equal(y.view(torch.int16), tot.bfloat16().view(torch.int16)):
        bad.append(("segments", "S=1 indistinguishable from the S=2 fold"))
    return bad


def main():
    bad = attention_mutants() + gemm_controls()
    print("UNCAUGHT MUTANTS:", len(bad), bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
