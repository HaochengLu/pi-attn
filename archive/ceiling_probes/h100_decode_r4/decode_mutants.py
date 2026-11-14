"""The decode mutants of kernels/test_mutants.py only (no GEMM controls). Run from kernels/."""
import os, sys
sys.path.insert(0, os.getcwd())
os.environ["PI_MUTANTS"] = "decode_fold_unfused,decode90_d_unfused"
import test_mutants as M
bad = M.attention_mutants()
print("UNCAUGHT DECODE MUTANTS:", len(bad), bad)
sys.exit(1 if bad else 0)
