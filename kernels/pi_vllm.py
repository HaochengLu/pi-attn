"""PI-Attn's row-invariant GEMM as the dense projections of vLLM 0.17.1's batch-invariant mode.

With VLLM_BATCH_INVARIANT=1 on sm_80/sm_89, vLLM routes aten::mm, addmm, matmul and linear to one Triton kernel
(batch_invariant.matmul_persistent: 128x128x64 tiles, no split-K), which keeps each output row independent of the
batch but runs a batch of 1 through a 128-row tile. install() replaces that function: a projection y = x W^T
(bf16, W [N, K] contiguous with K % 64 == 0 and N % 32 == 0, no bias) runs the engine's GEMM (pi_launch.cu: fixed k
order per output element, tile tuned per weight shape and M bucket; every tile gives the same bits), anything else
runs vLLM's kernel as before. STATS counts eager calls by path ('pi', or why vLLM's kernel ran); code torch.compile
traces is not counted, so selfcheck also reports the weight shapes the PI GEMM ran (shapes()).
The bits differ from vLLM's kernel's (a different, equally batch-invariant k order).
usage: from the engine process, after `import vllm` and before the model's first forward: pi_vllm.install()."""
import glob
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_PI = os.path.join(HERE, '..', 'llm_infer_rust', 'src', 'kernels', 'pi')
STATS = {}
_ext = None


def ext():
    """The extension, JIT-built for the local GPU (CUTLASS headers from the engine venv's FlashInfer, as test_gemm)."""
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load
        major, minor = torch.cuda.get_device_capability()
        root = os.path.join(HERE, '..', 'env', 'venv', 'lib')
        cutlass = glob.glob(os.path.join(root, 'python3*', 'site-packages', 'flashinfer', 'data', 'cutlass', 'include'))
        assert cutlass, 'CUTLASS headers not found under env/venv (FlashInfer)'
        _ext = load(name=f'pi_vllm_sm{major}{minor}', sources=[os.path.join(HERE, 'pi_vllm_torch.cu')],
                    extra_include_paths=[ENGINE_PI, cutlass[0]],
                    extra_cuda_cflags=['-O3', '-std=c++17', '--expt-relaxed-constexpr',
                                       f'-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}'],
                    build_directory=os.environ.get('PI_ATTN_BUILD_DIR'))
    return _ext


@torch.library.custom_op('pi_attn::linear', mutates_args=())
def linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """y = x w^T on the PI GEMM, as an op torch.compile keeps opaque (vLLM compiles the code that calls it)."""
    return ext().linear(x, w)


@linear.register_fake
def _(x, w):
    return x.new_empty(x.shape[0], w.shape[0])


def shapes():
    """(N, K) of every weight the PI GEMM has run."""
    return ext().shapes()


def _reason(a, b, bias):
    """Why a @ b cannot run as x W^T on the PI GEMM (None if it can): b must be W.t() of a contiguous W [N, K]."""
    if bias is not None:
        return 'bias'
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        return f'dtype {a.dtype}'
    if a.dim() != 2 or b.dim() != 2 or a.stride(1) != 1:
        return 'layout of x'
    K, N = b.shape
    if b.stride() != (1, K):
        return 'b is not W.t()'
    if K % 64 or N % 32:
        return f'shape N={N} K={K}'
    return None


def install():
    """Route vLLM's batch-invariant matmuls to the PI GEMM (see the module docstring)."""
    from vllm.model_executor.layers import batch_invariant as bi
    assert bi.vllm_is_batch_invariant(), 'PI GEMM in vLLM replaces the batch-invariant mode kernel: set VLLM_BATCH_INVARIANT=1'
    if getattr(bi.matmul_persistent, 'pi_gemm', False):
        return
    vllm_mm = bi.matmul_persistent
    ext()                                                            # build before the model compiles

    def matmul_persistent(a, b, bias=None):
        why = _reason(a, b, bias)
        if not torch.compiler.is_compiling():                        # torch.compile refuses the side effect
            key = 'pi' if why is None else why
            STATS[key] = STATS.get(key, 0) + 1
        if why is None:
            return linear(a, b.t())
        return vllm_mm(a, b, bias=bias)

    matmul_persistent.pi_gemm = True
    bi.matmul_persistent = matmul_persistent
