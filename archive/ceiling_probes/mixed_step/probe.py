"""Probe 21: the mixed steps of the 0.6B 512/256/c128 run. There the engine puts new prompts
and running decode rows in one prefill-phase batch, and both backends run every row,
decode rows included (q = 1), through their prefill path. Shapes: P new 512-token prompts
plus D decode rows at 512-766 keys, 16 query and 8 KV heads. Also timed: the same decode
rows through the decode kernel instead. Run from kernels/."""
import itertools
import math
import sys

import flashinfer
import torch

sys.path.insert(0, ".")
import bench_attention as b  # noqa: E402

HQ, HKV = 16, 8


def batch(n_prompts, n_dec):
    torch.manual_seed(0)
    q_lens = [512] * n_prompts + [1] * n_dec
    kv_lens = [512] * n_prompts + [512 + 2 * i for i in range(n_dec)]
    pages = [(n + b.PAGE - 1) // b.PAGE for n in kv_lens]
    n = sum(pages) + 1
    k = torch.randn(n, b.PAGE, HKV, b.D, device=b.dev, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    indices = (torch.arange(1, n, device=b.dev)).int().contiguous()
    cum = lambda xs: torch.tensor([0] + list(itertools.accumulate(xs)), device=b.dev, dtype=torch.int32)
    indptr, qo = cum(pages), cum(q_lens)
    last = torch.tensor([(x - 1) % b.PAGE + 1 for x in kv_lens], device=b.dev, dtype=torch.int32)
    lens = torch.tensor(kv_lens, device=b.dev, dtype=torch.int32)
    q = torch.randn(sum(q_lens), HQ, b.D, device=b.dev, dtype=torch.bfloat16)
    return q, k, v, qo, indptr, indices, last, lens, max(q_lens), max(kv_lens)


ws = torch.empty(256 << 20, device=b.dev, dtype=torch.uint8)
scale = 1 / math.sqrt(b.D)
for n_prompts, n_dec in ((0, 128), (1, 127), (2, 126), (4, 124), (1, 32), (4, 0)):
    q, k, v, qo, indptr, indices, last, lens, mq, mk = batch(n_prompts, n_dec)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
    w.plan(qo.cpu(), indptr.cpu(), indices, last.cpu(), HQ, HKV, b.D, b.PAGE, causal=True, q_data_type=torch.bfloat16)
    fi = b.timeit(lambda: w.run(q, (k, v)))
    pi = b.timeit(lambda: b.ext.prefill(q, k, v, qo, indptr, indices, lens, mq, mk, scale, 0))
    line = f"{n_prompts} x 512-token prompts + {n_dec:3d} decode rows: FI prefill {fi} us, PI prefill {pi} us ({pi / fi:.2f}x)"
    if n_dec:  # the decode rows alone through the decode kernel
        s = n_prompts
        dq = q[qo[s]:].contiguous()
        dlens = lens[s:].contiguous()
        dindptr = (indptr[s:] - indptr[s]).contiguous()
        dind = indices[indptr[s]:].contiguous()
        rows = n_dec * -(-mk // b.ext.GROUP_KEYS)
        ws_o = torch.empty(rows, HQ, b.D, device=b.dev)
        ws_ml = torch.empty(rows, HQ, 2 * b.ext.DECODE_WARPS, device=b.dev)
        cnt = torch.zeros(rows, HQ, b.ext.DECODE_WARPS, dtype=torch.int32, device=b.dev)
        dec = b.timeit(lambda: b.ext.decode(dq, k, v, dindptr, dind, dlens, ws_o, ws_ml, cnt, mk, scale))
        line += f" | decode rows alone via PI decode kernel {dec} us"
    print(line, flush=True)
