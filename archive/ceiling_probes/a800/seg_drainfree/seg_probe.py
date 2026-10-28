"""Probe: cost of K segments at prefill M, with the old segmented mainloop (drain and refill per segment) and the
drain-free one. Builds: base (current rule, these shapes S = 1), s2_old / s2_new (every K >= 2048 shape S = 2).
Best time over the large-M kernel ids, warm weights (compute-bound M)."""
import glob, os, sys, torch
from torch.utils.cpp_extension import load
D = os.path.dirname(os.path.abspath(__file__))
CUT = glob.glob(os.path.join(os.environ['ROOT'], 'env/venv/lib/python3*/site-packages/flashinfer/data/cutlass/include'))[0]
major, minor = torch.cuda.get_device_capability()
exts = {}
for v in ('base', 's2_old', 's2_new'):
    d = os.path.join(D, v)
    os.makedirs(os.path.join(d, 'build'), exist_ok=True)
    exts[v] = load(name=f'segprobe_{v}', sources=[os.path.join(d, 'gemm_torch.cu')], extra_include_paths=[d, CUT],
                   extra_cuda_cflags=['-O3', '-std=c++17', '--expt-relaxed-constexpr',
                                      f'-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}'],
                   build_directory=os.path.join(d, 'build'))
IDS = [10, 11, 12, 13, 14, 15, 44, 45, 46, 47, 48, 49, 50]


def t(fn, iters=20):
    fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters): fn()
        e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000 / iters)
    return sorted(ts)[2]


SHAPES = {'4B o': (2560, 4096), '4B down': (2560, 9728), '8B o': (4096, 4096), '8B down': (4096, 12288)}
for name, (N, K) in SHAPES.items():
    W = (torch.randn(N, K, device='cuda') * 0.02).bfloat16()
    for M in (512, 1024, 2048, 4096):
        X = torch.randn(M, K, device='cuda').bfloat16()
        row = []
        for v, ext in exts.items():
            best = min((t(lambda: ext.linear(X, W, k, False)), k) for k in IDS if ext.linear(X, W, k, False) is not None)
            row.append(f'{v} {best[0]:8.1f}us k{best[1]}')
        cb = t(lambda: torch.matmul(X, W.T))
        print(f'{name:8s} M={M:5d} cublas {cb:8.1f}us | ' + ' | '.join(row), flush=True)
