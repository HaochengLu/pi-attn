"""How many requests of a given maximum length vLLM 0.17.1's pool holds, stock vs exact GDN, for Qwen3.5-4B at
gpu_memory_utilization 0.85: vLLM's own "Maximum concurrency" estimate, at 4K, 32K and 128K tokens per request (the
exact GDN state is a fixed cost per request, the attention KV cache grows with its length). Each engine is built in
its own process (eager mode: no CUDA graphs, only the sizing matters).
usage: python gdn/vllm/pool_capacity.py MODEL_DIR   (from the repo root)"""
import os, re, subprocess, sys


def build(model, exact, max_len):
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:0] = [here, os.path.dirname(here)]
    import exact_gdn
    if exact:
        exact_gdn.install()
    from vllm import LLM
    LLM(model=model, dtype='bfloat16', gpu_memory_utilization=0.85, max_model_len=max_len,
        enable_prefix_caching=False, mamba_ssm_cache_dtype='float32', enforce_eager=True)


def main():
    model = sys.argv[1]
    for max_len in (4096, 32768, 131072):
        conc = []
        for exact in (0, 1):
            log = subprocess.run([sys.executable, __file__, '--build', model, str(exact), str(max_len)],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
            m = re.findall(r'Maximum concurrency for [0-9,]+ tokens per request: ([0-9.]+)x', log)
            conc.append(float(m[-1]) if m else float('nan'))
        print(f'{max_len:>6} tokens per request: stock {conc[0]:.2f} requests, exact {conc[1]:.2f} '
              f'(exact / stock {conc[1] / conc[0]:.2f})')


if __name__ == '__main__':
    if sys.argv[1] == '--build':
        build(sys.argv[2], sys.argv[3] == '1', int(sys.argv[4]))
    else:
        main()
