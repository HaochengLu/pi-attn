"""E4: does fla's autotune choice change the bits of the GDN chunk forward?

Each Triton autotuner on the chunk_gated_delta_rule forward path is pinned to one config at a time (the others keep
their own choice) and the output / final state are hashed. More than one distinct hash for an autotuner means the
trainer's forward is not a single program: which bits it produces depends on a timing benchmark (and, with
FLA_CACHE_RESULTS=1, on whatever got cached on that machine first)."""
import sys, hashlib, torch, torch.nn.functional as F, triton
import fla.ops.gated_delta_rule.chunk_fwd as m1, fla.ops.gated_delta_rule.wy_fast as m2
import fla.ops.common.chunk_delta_h as m3, fla.ops.common.chunk_o as m4, fla.ops.utils.cumsum as m5
import fla.modules.l2norm as m6
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

dev = 'cuda'
T, H, HV, K, V = int(sys.argv[1]) if len(sys.argv) > 1 else 1024, 16, 32, 128, 128     # Qwen3.5-4B GDN shape
gen = torch.Generator(device=dev).manual_seed(0)
q = torch.randn(1, T, HV, K, device=dev, generator=gen).bfloat16()
k = torch.randn(1, T, HV, K, device=dev, generator=gen).bfloat16()
v = torch.randn(1, T, HV, V, device=dev, generator=gen).bfloat16()
g = F.logsigmoid(torch.randn(1, T, HV, device=dev, generator=gen) + 3).float()
beta = torch.sigmoid(torch.randn(1, T, HV, device=dev, generator=gen)).bfloat16()

def run():
    o, s = chunk_gated_delta_rule(q, k, v, g, beta, output_final_state=True, use_qk_l2norm_in_kernel=True)
    torch.cuda.synchronize()
    return hashlib.sha1(o.view(torch.int16).cpu().numpy().tobytes() + s.view(torch.int32).cpu().numpy().tobytes()).hexdigest()[:10]

tuners = []
for mod in (m1, m2, m3, m4, m5, m6):
    for name in dir(mod):
        obj = getattr(mod, name)
        while obj is not None and not isinstance(obj, triton.runtime.autotuner.Autotuner):
            obj = getattr(obj, 'fn', None) if isinstance(obj, triton.runtime.KernelInterface) else None
        if obj is not None and all(obj is not t for _, t in tuners):
            tuners.append((f'{mod.__name__.split(".")[-1]}.{name}', obj))
base = run()
print('baseline', base)
for name, at in tuners:
    configs = list(at.configs)
    if len(configs) < 2: continue
    hashes = {}
    for c in configs:
        at.configs = [c]; at.cache.clear()
        try:
            h = run()
        except Exception as e:
            h = 'ERR:' + type(e).__name__
        hashes.setdefault(h, []).append(str(c).replace('num_ctas: 1, ', '').replace(', maxnreg: None', ''))
    at.configs = configs; at.cache.clear()
    print(f'{name:55s} configs {len(configs):2d}  distinct outputs {len(hashes)}', flush=True)
    if len(hashes) > 1:
        for h, cs in hashes.items(): print(f'      {h}: {cs[0]}{" (+%d more)" % (len(cs)-1) if len(cs) > 1 else ""}')
