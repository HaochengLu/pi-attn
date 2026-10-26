"""E14b: test_exact.py with fla's chunk_fwd_kernel_o pinned to each of its autotune configs in turn (OCFG=index), to find
which configs are the program exact_decode reproduces. usage: (from gdn/) OCFG=i PYTHONPATH=<fla>:. python experiments/e14b_o_config_pins.py"""
import sys, os, runpy
from fla.ops.common import chunk_o
kern = chunk_o.chunk_fwd_kernel_o
while not hasattr(kern, 'configs'): kern = kern.fn
i = int(os.environ['OCFG'])
c = kern.configs[i]
print(f'chunk_fwd_kernel_o pinned to config {i}: {c.kwargs} num_warps {c.num_warps}')
import exact_decode
exact_decode.pin_fla_configs = lambda: None          # keep the pinned config (do not filter it away)
kern.configs = [c]
sys.argv = ['test_exact.py']
runpy.run_path('test_exact.py', run_name='__main__')
