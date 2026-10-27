"""Does MTP drafting change the prompt log-probabilities vLLM 0.17.1 returns? Scores 16 fixed sequences (228-783
tokens of the model README) with prompt_logprobs, one greedy token each, batch-invariant dense ops, piecewise CUDA
graphs, with stock and with exact GDN, with and without MTP drafting (3 drafts), whole and chunked prefill (a token
budget of 256 per step), each in its own engine process; then counts the positions whose log-probability is equal
between configurations. Scoring is prefill only, so drafting should change nothing.
usage: python gdn/vllm/check_mtp_prompt_logprobs.py MODEL_DIR   (Qwen3.5; from the repo root)"""
import os, subprocess, sys, tempfile


def score(model, exact, mtp, chunk, out):
    import torch
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ.setdefault('VLLM_DISABLE_COMPILE_CACHE', '1')
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:0] = [here, os.path.dirname(here)]
    import exact_gdn
    if exact:
        exact_gdn.install()
    else:
        exact_gdn.fix_vllm_mixed_spec_decodes()
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    ids = AutoTokenizer.from_pretrained(model)(open(f'{model}/README.md').read()).input_ids
    seqs = [ids[200 * i:200 * i + 228 + 37 * i] for i in range(16)]
    kw = {'attention_backend': 'FLASH_ATTN', 'compilation_config': {'cudagraph_mode': 'PIECEWISE'}}
    if mtp:
        kw['speculative_config'] = {'method': 'mtp', 'num_speculative_tokens': 3}
    if chunk:
        kw.update(max_num_batched_tokens=chunk, max_num_seqs=min(chunk, 256))
    llm = LLM(model=model, dtype='bfloat16', gpu_memory_utilization=0.6, max_model_len=4096, seed=0,
              enable_prefix_caching=False, mamba_ssm_cache_dtype='float32', **kw)
    sc = llm.generate([{'prompt_token_ids': s} for s in seqs], SamplingParams(temperature=0, max_tokens=1,
                                                                               prompt_logprobs=0))
    torch.save([[None if d is None else list(d.values())[0].logprob for d in o.prompt_logprobs] for o in sc], out)


def main():
    import torch
    model, tmp = sys.argv[1], tempfile.mkdtemp()
    env = dict(os.environ, VLLM_BATCH_INVARIANT='1')
    runs = {}
    for exact in (0, 1):
        for mtp in (0, 1):
            for chunk in (0, 256):
                out = f'{tmp}/{exact}{mtp}{chunk}.pt'
                subprocess.run([sys.executable, __file__, '--score', model, str(exact), str(mtp), str(chunk), out],
                               env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                runs[exact, mtp, chunk] = torch.load(out)
    eq = lambda a, b: (sum(x == y for ra, rb in zip(a, b) for x, y in zip(ra, rb)), sum(len(r) for r in a))
    for exact in (0, 1):
        for chunk in (0, 256):
            print(f'{"exact" if exact else "stock"} GDN, {"chunked (256)" if chunk else "whole"} prefill: prompt '
                  'log-probs equal without and with MTP drafting at %d/%d positions' % eq(runs[exact, 0, chunk],
                                                                                         runs[exact, 1, chunk]))


if __name__ == '__main__':
    if sys.argv[1] == '--score':
        score(sys.argv[2], sys.argv[3] == '1', sys.argv[4] == '1', int(sys.argv[5]), sys.argv[6])
    else:
        main()
