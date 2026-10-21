"""Bitwise invariance of prefill2/decode2 (probe): full prefill vs chunked prefill over a
cached prefix in other pages vs split mode vs decode of each probed position, in mixed batches."""
import math, os, random, sys
import torch
from torch.utils.cpp_extension import load
K = "/uniparser/users/luhaocheng/pi_attn/kernels"
sys.path.insert(0, K)
import test_invariance as T
H = os.path.dirname(os.path.abspath(__file__))
p2 = load(name="pi_p2", sources=[os.path.join(H, "p2_ext.cu")], extra_include_paths=[K, H],
          extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_89,code=sm_89", "--expt-relaxed-constexpr"],
          build_directory="/tmp/pi_p2_build")
S = T.SCALE


GR = int(os.environ.get('P2_GR', '2'))


def pf(pool, reqs, splits=1):
    q = torch.cat([r[0] for r in reqs]).contiguous()
    m = T.meta([r[1] for r in reqs], [r[2] for r in reqs], [r[0].shape[0] for r in reqs])
    out = p2.prefill2(q, pool.k, pool.v, m["qo_indptr"], m["kv_page_indptr"], m["kv_page_indices"], m["kv_lens"],
                      max(r[0].shape[0] for r in reqs), max(r[2] for r in reqs), S, splits, GR)
    return list(out.split([r[0].shape[0] for r in reqs]))


def dec(pool, reqs):
    q = torch.cat([r[0] for r in reqs]).contiguous()
    m = T.meta([r[1] for r in reqs], [r[2] for r in reqs])
    return list(p2.decode2(q, pool.k, pool.v, m["kv_page_indptr"], m["kv_page_indices"], m["kv_lens"],
                           max(r[2] for r in reqs), S).split(1))


fails = []
for hq, hkv, lens in [(16, 8, [1, 31, 32, 33, 64, 511, 512, 513, 1000, 2049, 5000]), (32, 8, [7, 512, 1537, 4100]),
                      (16, 1, [300, 1030]), (32, 4, [2047, 33])]:
    T.HQ, T.HKV = hq, hkv
    torch.manual_seed(hq + hkv)
    random.seed(hq * hkv)
    pool = T.Pool(pages=sum((L + 15) // 16 for L in lens) * 2 + 64)
    seqs, pages = {}, {}
    for L in lens:
        seqs[L] = T.seq(L)
        pages[L] = pool.alloc(L)
        pool.write(pages[L], 0, seqs[L][1], seqs[L][2])
    alone = {L: pf(pool, [(seqs[L][0], pages[L], L)])[0] for L in lens}
    order = lens[:]
    random.shuffle(order)
    for sp in (1, 3):
        for L, o in zip(order, pf(pool, [(seqs[L][0], pages[L], L) for L in order], sp)):
            if not T.same(o, alone[L]):
                fails.append(("batch/split", hq, hkv, sp, L))
    for L in lens:
        if L < 2:
            continue
        p2g = pool.alloc(L)
        pool.write(p2g, 0, seqs[L][1], seqs[L][2])
        cuts = sorted(random.sample(range(1, L), min(3, L - 1)))
        outs, st = [], 0
        for e in cuts + [L]:
            outs.append(pf(pool, [(seqs[L][0][st:e].contiguous(), p2g, e)], random.choice([1, 2]))[0])
            st = e
        if not T.same(torch.cat(outs), alone[L]):
            fails.append(("chunk+prefix", hq, hkv, L))
        pool.free.extend(p2g)
    probes = [(L, p) for L in lens for p in sorted({0, L // 2, L - 1, random.randrange(L), min(L - 1, 511), min(L - 1, 512)})]
    random.shuffle(probes)
    for i in range(0, len(probes), 7):
        grp = probes[i:i + 7]
        for (L, p), o in zip(grp, dec(pool, [(seqs[L][0][p:p + 1].contiguous(), pages[L], p + 1) for L, p in grp])):
            if not T.same(o[0], alone[L][p]):
                fails.append(("decode/path", hq, hkv, L, p))
    Lr = max(l for l in lens if l <= 4100)
    q, k, v = seqs[Lr]
    ref = T.exact(q, k, v).float()
    e = (alone[Lr].float() - ref).abs()
    print(f"hq{hq}/hkv{hkv}: fails so far {len(fails)} | err vs fp64 (L={Lr}) mean {e.mean():.3e} max {e.max():.3e}", flush=True)
print("checks failed:", len(fails), fails[:10])
sys.exit(1 if fails else 0)
