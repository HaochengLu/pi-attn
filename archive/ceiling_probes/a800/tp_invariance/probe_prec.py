"""E4: accuracy of the candidate K-combines of a row-parallel GEMM against an fp64 reference.
y = X W^T with K cut into S leaves (contiguous K ranges). Each leaf partial p_s is an fp32 GEMM (TF32 off).
Schemes (output rounded to bf16 in every case):
  chain    one fp32 chain over all of K (today's S=1 canonical)
  fold32   fp32 left fold of the S leaves, +0 first (the in-kernel segment fold)
  fix64    sum of q_s = clamp(rne(p_s * 2^F), +-(2^59-1)) in int64, times 2^-F
  ring16   what a bf16 TP all-reduce does: each leaf rounded to bf16, summed in bf16 (ring order)
Reports, per scheme: bf16 outputs different from rne_bf16(fp64), and the max error in bf16 ulps of the reference.
Activations: gaussian, and gaussian with 1% of the K channels scaled 100x (outlier channels, as in LLM residuals).
usage: python probe_prec.py"""
import torch

torch.backends.cuda.matmul.allow_tf32 = False
F, QMAX = 28, 2 ** 59 - 1


def ulp_bf16(y):  # bf16 ulp at |y| (normal range)
    e = torch.frexp(y.abs().clamp_min(2.0 ** -126))[1] - 1
    return torch.ldexp(torch.ones_like(y), e - 7)


def run(X, W, S):
    M, K = X.shape
    ref = X.double() @ W.double().T
    ks = K // S
    P = [X[:, s * ks:(s + 1) * ks].float() @ W[:, s * ks:(s + 1) * ks].float().T for s in range(S)]
    out = {'chain': (X.float() @ W.float().T)}
    acc = torch.zeros_like(P[0])
    for p in P:
        acc = acc + p
    out['fold32'] = acc
    q = sum((p.double() * 2.0 ** F).round().clamp(-QMAX, QMAX).long() for p in P)
    out['fix64'] = q.double() * 2.0 ** -F
    r = P[0].bfloat16()
    for p in P[1:]:
        r = r + p.bfloat16()
    out['ring16'] = r
    rb = ref.bfloat16()
    u = ulp_bf16(rb.double())
    res = {}
    for k, y in out.items():
        yb = y.bfloat16() if y.dtype != torch.bfloat16 else y
        res[k] = ((yb != rb).float().mean().item(), ((yb.double() - ref).abs() / u).max().item())
    return res


torch.manual_seed(0)
M = 1024
for name, N, K in [('qwen3.5-4B down', 2560, 9216), ('70B o', 8192, 8192), ('70B down', 8192, 28672)]:
    W = (torch.randn(N, K, device='cuda') * 0.02).bfloat16()
    for act in ['gauss', 'outlier']:
        X = torch.randn(M, K, device='cuda')
        if act == 'outlier':
            X[:, torch.randperm(K, device='cuda')[:K // 100]] *= 100
        X = X.bfloat16()
        for S in [2, 8]:
            r = run(X, W, S)
            print(f'{name:16s} {act:7s} S={S}: ' + '  '.join(f'{k} {d * 100:.2f}%/{m:.2f}ulp' for k, (d, m) in r.items()), flush=True)
