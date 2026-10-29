"""Hardware floors for narrow-N decode GEMMs on this GPU (micro.cu).
usage (from kernels/): python ../logs/narrow/probe_micro.py"""
import os, sys
import torch
from torch.utils.cpp_extension import load
sys.path.insert(0, '.')
from bench_tp import timeit

HERE = os.path.dirname(os.path.abspath(__file__))
major, minor = torch.cuda.get_device_capability()
ext = load(name='micro_probe', sources=[os.path.join(HERE, 'micro.cu')], build_directory=HERE,
           extra_include_paths=[os.path.abspath('.'), os.path.join(os.path.dirname(__import__('flashinfer').__file__), 'data', 'cutlass', 'include')],
           extra_cuda_cflags=['-O3', '-std=c++17', '--expt-relaxed-constexpr', f'-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}'])
clk = torch.cuda.get_device_properties(0).clock_rate if hasattr(torch.cuda.get_device_properties(0), 'clock_rate') else 1410000
print('dependent mma.m16n8k16 cycles per step (chains interleaved):',
      {c: round(ext.mma_latency(100000, c), 1) for c in (1, 2, 3, 4)}, flush=True)
SMS = torch.cuda.get_device_properties(0).multi_processor_count
CFG = ext.stream_configs()
TOTAL = 64 << 20  # bytes streamed per launch
for per_sm in (1, 2, 3, 4, 8):
    warps = SMS * per_sm
    length = (TOTAL // (warps * 16)) // 1024 * 1024
    ws = [torch.randn(warps * 8, length, device='cuda').bfloat16() for _ in range(4)]  # 4 x 64 MB: cold
    for wpc in (1, 2, 4):
        if per_sm % wpc and wpc > per_sm:
            continue
        line = f'{per_sm} warps/SM ({wpc}/CTA), {8 * length * 2 // 1024} KB per warp:'
        for mma in (False, True):
            best = []
            for i, c in enumerate(CFG):
                if not ext.stream(ws[0], length, i, wpc, mma):
                    continue
                t = timeit(lambda w: ext.stream(w, length, i, wpc, mma), ws)
                best.append((warps * 8 * length * 2 / t / 1e3, c))
            best.sort(reverse=True)
            line += f" {'mma' if mma else 'copy'}: " + ' '.join(f'{c}={g:.0f}GB/s' for g, c in best[:4])
        print(line, flush=True)
    del ws
    torch.cuda.empty_cache()
