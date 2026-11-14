# Across tensor-core classes (paper §7)

This directory holds the programs and probes behind §7 of the paper. They are research prototypes, not part of the engine.

Each subdirectory keeps the raw logs from the runs the paper cites. The logs were produced on Modal GPUs: A100-80GB, A10/A10G, L4 and H100, with CUDA 12.6 and the repository's `requirements/engine.txt` environment. Run every command from inside its own subdirectory, on the GPU you want to test.

## `theory/`: tensor-core models, the exact-lattice programs, SFU identity

| File | What it is |
|---|---|
| `tc_models.py`, `model.h` | Bit-exact models of one m16n8k16 bf16 step for class A (sm_80/86/89, two groups of 8, F=24) and class H (sm_90, one group of 16, F=25) |
| `mmaprobe.cu`, `gen.h`, `predict.cpp`, `fit.cpp`, `r1check.cpp` | Calibration probe, case generator and model checks |
| `xc.h`, `xc_ref.py`, `xc_cpu.cpp`, `xcprobe.cu` | The XC programs: specification, an exact rational reference, a C++ reference, and a GPU implementation on `mma.sync` and, with `-DWGMMA`, on `wgmma` |
| `sfu.cu` | Hashes 30 SFU, IEEE and libm operations over all 2^32 inputs |
| `wg_diag.cu`, `wgfeed.cu` | The `st.shared` → `wgmma` feed race, and the `cp.async` feed that avoids it |
| `log_*.txt`, `sfu_*.txt`, `vectors_*.txt`, `xc_cpu_256_64.txt` | Measured outputs |

```
nvcc -O2 -std=c++17 -arch=sm_80 xcprobe.cu -o xcprobe && ./xcprobe 256 64     # sm_90a -DWGMMA on H100
nvcc -O2 -std=c++17 -arch=sm_80 -DARCHNAME='"sm_80"' sfu.cu -o sfu && ./sfu
g++ -O2 -std=c++17 xc_cpu.cpp -o xc_cpu && ./xc_cpu 256 64                  # must print the GPU hashes
```

## `gemm/`: I4, an integer-tensor-core GEMM

- `gx.cu` holds the kernels and `drv.py` the driver and torch reference. Run `python drv.py --tasks ident,acc,speed`.
- `id_*.txt` hold the identity hashes per GPU; `out_*.log` hold the full runs.

## `gemm_decode/`: portable decode GEMM (TRIF, P21C, P11C)

- `xcp.h` is the program specification with an exact CPU reference.
- `xcp_kernel.cuh` holds the small-M and tile kernels, and `xcp.cu` the driver.
- `xcp_cpu.cpp` and `cuda_emu.h` run the kernel source on a CPU; `xcp_ref.py` is a rational reference.

```
nvcc -O3 -std=c++17 --expt-relaxed-constexpr -arch=sm_80 xcp.cu -o xcp -lcublas
./xcp verify          # hashes per case and program; compare with identity_table.txt
./xcp bench           # timings against cuBLAS
```

`identity_table.txt` shows the same hash for every case on L4, A10, A100 and H100, and on the CPU reference. The `table_*.txt` files hold the timings; the `log_*_final.txt` files hold the raw runs.

## `attn/`: XC attention

- `make_xc.py <src> <dst>` writes XC variants of `kernels/pi_attn*.cu*` into `<dst>`.
- `xc_build.py <xc> <qk_block>` builds one variant.
- `xc_run.py` measures hashes and accuracy, `xc_inv.py` runs the fold-group invariance checks, and `xc_bench.py` times the kernels.
- `cmp_hash.py` compares the `logs4_all_*.txt` files from two GPUs.
- `tc_exact_probe.cu` checks where tensor-core sums stop being exact.
