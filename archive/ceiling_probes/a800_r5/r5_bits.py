"""Round-5 bit checks of the class-A scheduling variants (run from kernels/):
  prefill grid order: the device's choice (0), ascending (1) and longest first (2), every prefill variant this GPU runs and the
  device's choice, forced split counts and the planner, single requests, mixed batches, extends over prefixes;
  test_invariance.run_checks() with each order forced;
  decode16_kernel (decode variant 3) against decode_kernel (dec_r4_bench.equal), if PI_R5_DECODE16=1."""
import math, os, random, sys
sys.path.insert(0, os.getcwd())
import torch
import test_invariance as T
ext, dev, D, PAGE = T.ext, "cuda", 128, 16
torch.manual_seed(1); random.seed(1)
print(torch.cuda.get_device_name(0), "VARIANTS", T.VARIANTS, flush=True)

def run(q, k, v, qo, indptr, idx, lens, mq, mk, sp, var, order):
    ext.set_prefill_order(order)
    out = ext.prefill(q, k, v, qo, indptr, idx, lens, mq, mk, 1 / math.sqrt(D), sp, var)
    ext.set_prefill_order(0)
    return out

n = bad = 0
for hq, hkv in ((16, 8), (32, 8), (8, 8), (16, 1)):
    for shape in ([(1, 0)], [(33, 0)], [(64, 0)], [(65, 0)], [(517, 0)], [(2049, 0)], [(4100, 0)], [(128, 4096)], [(512, 8192)], [(1, 5000)],
                  [(2048, 2048)] * 4, [(1024, 0)] * 8, [(512, 0)] * 32, [(700, 0), (3, 0), (2049, 100), (64, 1300), (1, 0), (513, 511)],
                  [(q, random.randint(0, 3000)) for q in random.sample(range(1, 900), 12)]):
        batch = len(shape)
        qls = [s[0] for s in shape]; kls = [s[0] + s[1] for s in shape]
        pp = [-(-kl // PAGE) for kl in kls]
        npg = sum(pp) + 1
        k = torch.randn(npg, PAGE, hkv, D, device=dev).bfloat16()
        v = torch.randn(npg, PAGE, hkv, D, device=dev).bfloat16()
        q = (torch.randn(sum(qls), hq, D, device=dev) * 3).bfloat16()
        perm = (torch.randperm(npg - 1, device=dev).int() + 1).contiguous()
        t = lambda x: torch.tensor(x, device=dev, dtype=torch.int32)
        indptr = t([0] + list(torch.tensor(pp).cumsum(0).tolist())); qo = t([0] + list(torch.tensor(qls).cumsum(0).tolist()))
        lens = t(kls)
        ref = run(q, k, v, qo, indptr, perm, lens, max(qls), max(kls), 1, T.VARIANTS[0], 1)
        for var in [0] + T.VARIANTS:
            for sp in (0, 1, 2, 3, 7):
                for order in (0, 1, 2):
                    out = run(q, k, v, qo, indptr, perm, lens, max(qls), max(kls), sp, var, order)
                    n += 1
                    if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
                        bad += 1
                        print("MISMATCH prefill", hq, hkv, shape[:3], var, sp, order, flush=True)
print(f"EQ prefill grid order (the device's choice, ascending and longest first, variants {[0] + T.VARIANTS}, splits 0/1/2/3/7): {n} launches against the ascending one-split reference, {bad} mismatches", flush=True)

for order in (0, 1, 2):
    ext.set_prefill_order(order)
    T.COUNTERS.zero_()
    if os.environ.get("PI_R5_DECODE16"):
        T.DECODE_VARIANTS = [1, 3, 4]
    fails = T.run_checks()
    print(f"test_invariance.run_checks with prefill order {order}, DECODE_VARIANTS {T.DECODE_VARIANTS}: checks failed: {len(fails)} {fails[:5]}", flush=True)
ext.set_prefill_order(0)
