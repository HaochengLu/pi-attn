import os, sys, time, torch
from torch.utils.cpp_extension import load
sys.path.insert(0, "/pi/kernels")
from build import cuda_build_flags
HERE = os.path.dirname(os.path.abspath(__file__))
def build(arch=None):
    t0 = time.time()
    a, cflags, includes, ldflags = cuda_build_flags(arch)
    ext = load(name="probe90", sources=[os.path.join(HERE, os.environ.get("PROBE_SRC", "probe90.cu"))], extra_include_paths=["/pi/kernels"] + includes,
               extra_cuda_cflags=cflags, extra_ldflags=ldflags, verbose=bool(os.environ.get("V")))
    print(f"probe ext build {time.time() - t0:.0f}s", flush=True)
    return ext
# id: (BM, BN, X stages, BK, W stages (0: one pipeline))
TILES = {90: (128, 256, 3, 64, 0), 91: (128, 128, 5, 64, 0), 99: (64, 16, 8, 64, 0), 101: (64, 32, 4, 128, 0),
         107: (128, 192, 4, 64, 0), 111: (192, 96, 4, 64, 0), 112: (192, 192, 3, 64, 0), 113: (192, 128, 4, 64, 0),
         120: (64, 16, 8, 128, 0), 121: (64, 32, 8, 128, 0), 122: (64, 64, 6, 128, 0), 124: (128, 32, 8, 64, 0),
         125: (128, 64, 8, 64, 0), 126: (128, 96, 6, 64, 0), 128: (192, 32, 7, 64, 0),
         129: (64, 16, 4, 64, 24), 130: (64, 16, 4, 64, 48), 131: (64, 32, 4, 64, 24), 132: (128, 16, 4, 64, 24),
         133: (128, 32, 4, 64, 24), 134: (192, 16, 3, 64, 24), 135: (128, 64, 4, 64, 16), 136: (192, 32, 3, 64, 16),
         137: (64, 64, 4, 64, 16), 138: (64, 96, 4, 64, 12), 139: (192, 64, 3, 64, 12), 140: (128, 128, 3, 64, 8)}
