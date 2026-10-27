"""Offline serving throughput of vLLM 0.17.1 on a hybrid model, stock vs exact GDN (exact_gdn.py).

N requests, each a P-token slice of the model README, greedy, NEW output tokens (ignore_eos), all submitted at once
(max_num_seqs = N); reports the wall time of llm.generate after an identical warm-up generate, and output
tokens/s.
usage: [VLLM_BATCH_INVARIANT=1] [CUDAGRAPH_MODE=PIECEWISE] python gdn/vllm/throughput.py MODEL_DIR [N] [P] [NEW]
       [--exact] [--spec] [--mtp=K]   (--spec: n-gram drafting, 4 drafts; --mtp=K: the model's MTP head drafting K
       tokens, with vLLM's async scheduling; both with the vLLM fixes of fix_vllm_mixed_spec_decodes)"""
import os, sys, time


def main():
    model = sys.argv[1]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    P = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    NEW = int(sys.argv[4]) if len(sys.argv) > 4 else 256
    exact = '--exact' in sys.argv
    mtp = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--mtp=')), None)
    spec = '--spec' in sys.argv or mtp is not None
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'               # patches must live in the model's process
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:0] = [here, os.path.dirname(here)]                     # gdn/vllm (exact_gdn), gdn (exact_decode)
    import exact_gdn
    if exact:
        exact_gdn.install()
    elif spec:
        exact_gdn.fix_vllm_mixed_spec_decodes()                      # stock vLLM 0.17.1 needs it for n-gram drafts
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    ids = tok(open(f'{model}/README.md').read()).input_ids
    prompts = [{'prompt_token_ids': ids[(7 * i) % (len(ids) - P):(7 * i) % (len(ids) - P) + P]} for i in range(N)]
    kw = {'attention_backend': 'FLASH_ATTN'} if os.environ.get('VLLM_BATCH_INVARIANT') == '1' else {}
    if mtp is not None:
        kw['speculative_config'] = {'method': 'mtp', 'num_speculative_tokens': mtp}
    elif spec:
        kw['speculative_config'] = {'method': 'ngram', 'num_speculative_tokens': 4, 'prompt_lookup_max': 3,
                                    'prompt_lookup_min': 2}
    if os.environ.get('CUDAGRAPH_MODE'):
        kw['compilation_config'] = {'cudagraph_mode': os.environ['CUDAGRAPH_MODE']}
    llm = LLM(model=model, dtype='bfloat16', gpu_memory_utilization=0.85, max_model_len=P + NEW + 16, seed=0,
              enable_prefix_caching=False, mamba_ssm_cache_dtype='float32', max_num_seqs=N, **kw)
    sp = SamplingParams(temperature=0, max_tokens=NEW, ignore_eos=True)
    # warm-up: the full request set once (first-use kernel compilation and autotuning happen here), then timed
    llm.generate(prompts, sp)
    t = time.time()
    out = llm.generate(prompts, sp)
    dt = time.time() - t
    n_out = sum(len(o.outputs[0].token_ids) for o in out)
    print(f'RESULT exact={exact} spec={"mtp" + str(mtp) if mtp is not None else spec} batch_invariant={os.environ.get("VLLM_BATCH_INVARIANT", "0")} N={N} P={P} NEW={NEW}: '
          f'{dt:.2f} s, {n_out / dt:.1f} output tok/s')


if __name__ == '__main__':
    main()
