"""archive/ceiling_probes/h100_decode_r4/dec_r4_bench.py's bench() on the round-4 report grid (run from kernels/)."""
import sys
src = open("/work/dec_r4_bench.py").read()
sys.argv = ["dec_r4_bench.py"]
exec(compile(src, "dec_r4_bench.py", "exec"))
for batch in (1, 2, 4, 8, 16, 32, 64):
    for kv in (96, 640, 2176, 4200):
        bench(batch, kv, "DEC")
