"""In-engine prefill/decode consistency for a hybrid model in vLLM 0.17.1.

Greedy-generate NEW tokens for each prompt (decode path), then score prompt + generated tokens in one prefill and read
prompt_logprobs (prefill path). Counts the generated tokens whose decode-time fp32 log-probability equals the
prefill-time one bitwise, and whose greedy token equals the prefill's argmax. Dense ops can be made batch-invariant
with VLLM_BATCH_INVARIANT=1 (FLASH_ATTN backend); with them on, what remains is the GDN layers.
usage: [VLLM_BATCH_INVARIANT=1] python gdn/vllm/selfcheck.py MODEL_DIR [N_PROMPTS] [NEW] [--exact] [--spec]
       [--prefill-chunk=T] [--short] [--aligned] [--one-token-chunks] [--preempt]
(--exact installs gdn/vllm/exact_gdn.py's patch first; --spec turns on n-gram speculative decoding with 4 drafts;
CUDAGRAPH_MODE=PIECEWISE selects vLLM's piecewise CUDA graphs, needed with --spec: stock vLLM 0.17.1 fails to capture
full graphs for Qwen3.5 with n-gram drafts; --prefill-chunk=T caps the tokens per step at T, so prompts are prefilled
in chunks, continuing from the state a previous chunk left, and with T a multiple of 64 chunks end on chunk boundaries;
--short uses prompts of 1..1+3(N-1) tokens, generated after a batch of long prompts, so new requests whose prompt ends
inside the first chunk take over slots that hold a finished request's state; --aligned uses prompts of 64, 128, 192, 256
tokens, so prefill ends on a 64-token boundary and decoding starts a new chunk; --one-token-chunks uses N-1 prompts of
2 tokens and one of 200 with N+1 tokens per step, so while the short requests decode (served first) the long prompt
advances a few tokens per step, including one-token chunks that complete a 64-token chunk; --preempt preempts the
running request with the most generated tokens every 23rd scheduler step during generation through vLLM's preemption
routine (a synthetic trigger and victim; the request's blocks are freed and it is recomputed from its first token,
prompt and generated tokens as a prefill), and prints how many).
Also prints a hash of every generated token and the bits of its log-probability, to compare runs."""
import hashlib, os, struct, sys


def main():
    model = sys.argv[1]
    n_prompts = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    new = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    exact = '--exact' in sys.argv
    spec = '--spec' in sys.argv
    chunk = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--prefill-chunk=')), None)
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'               # patches must live in the model's process
    # vLLM reuses torch.compile artifacts from its cache when the config hash matches; an artifact compiled earlier can
    # be a different program for the dense ops (seen: Qwen3.5-4B, same config, two hashes depending on the cache), so
    # bit-level comparisons compile afresh
    os.environ.setdefault('VLLM_DISABLE_COMPILE_CACHE', '1')
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
    text = open(f'{model}/README.md').read()
    ids = tok(text).input_ids
    short, aligned, trickle = '--short' in sys.argv, '--aligned' in sys.argv, '--one-token-chunks' in sys.argv
    preempt = {'step': 0, 'count': 0} if '--preempt' in sys.argv else None
    plen = lambda i: ((200 if i == n_prompts - 1 else 2) if trickle else 1 + 3 * i if short
                      else 64 * (1 + i % 4) if aligned else 100 + 37 * i)
    prompts = [ids[200 * i:200 * i + plen(i)] for i in range(n_prompts)]
    if trickle:
        chunk = n_prompts + 1
    kw = {'attention_backend': 'FLASH_ATTN'} if os.environ.get('VLLM_BATCH_INVARIANT') == '1' else {}
    if spec:
        kw['speculative_config'] = {'method': 'ngram', 'num_speculative_tokens': 4, 'prompt_lookup_max': 3,
                                    'prompt_lookup_min': 2}
    if chunk:
        kw.update(max_num_batched_tokens=chunk, max_num_seqs=min(chunk, 256))
    if os.environ.get('CUDAGRAPH_MODE'):                             # e.g. PIECEWISE: vLLM 0.17.1's full-graph capture
        kw['compilation_config'] = {'cudagraph_mode': os.environ['CUDAGRAPH_MODE']}   # fails for Qwen3.5 + n-gram
    llm = LLM(model=model, dtype='bfloat16', gpu_memory_utilization=0.6, max_model_len=4096, seed=0,
              enable_prefix_caching=False, mamba_ssm_cache_dtype='float32', **kw)
    if short:                    # leave finished requests' state in every slot of the pool (vLLM hands out unused ones first)
        llm.generate([{'prompt_token_ids': ids[7 * i:7 * i + 300]} for i in range(400)],
                     SamplingParams(temperature=0, max_tokens=2, ignore_eos=True))
    if preempt is not None:
        import time
        from vllm.v1.core.sched.interface import PauseState
        from vllm.v1.core.sched.scheduler import Scheduler
        orig_schedule = Scheduler.schedule

        def schedule(self):
            """As vLLM's scheduler does when it runs out of blocks: a running request is preempted (the one with the
            most generated tokens, so its recompute runs past its prompt; vLLM's priority policy also picks any running
            request), and no waiting request is scheduled in that step (the preempted one resumes in a later step)."""
            preempt['step'] += 1
            if preempt['step'] % 23 or len(self.running) < 2 or self._pause_state != PauseState.UNPAUSED:
                return orig_schedule(self)
            req = max(self.running, key=lambda r: r.num_output_tokens)
            self.running.remove(req)
            self._preempt_request(req, time.monotonic())
            preempt['count'] += 1
            self._pause_state = PauseState.PAUSED_NEW
            try:
                return orig_schedule(self)
            finally:
                self._pause_state = PauseState.UNPAUSED
        Scheduler.schedule = schedule
    gen = llm.generate([{'prompt_token_ids': p} for p in prompts],
                       SamplingParams(temperature=0, max_tokens=new, logprobs=0, ignore_eos=True))
    if preempt is not None:
        Scheduler.schedule = orig_schedule
        print('PREEMPTIONS', preempt['count'])
    full = [p + list(g.outputs[0].token_ids) for p, g in zip(prompts, gen)]
    sc = llm.generate([{'prompt_token_ids': f} for f in full],
                      SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=0))
    h = hashlib.sha256()
    for g in gen:
        for i, t in enumerate(g.outputs[0].token_ids):
            h.update(struct.pack('<id', t, g.outputs[0].logprobs[i][t].logprob))
    eq = tot = arg_eq = 0
    for p, g, s in zip(prompts, gen, sc):
        toks = g.outputs[0].token_ids
        for i, t in enumerate(toks):
            d = g.outputs[0].logprobs[i][t].logprob
            pl = s.prompt_logprobs[len(p) + i]
            pf = pl[t].logprob
            eq += d == pf
            arg_eq += max(pl.values(), key=lambda x: x.logprob) is pl[t] or pl[t].rank == 1
            tot += 1
    print(f'RESULT exact={exact} spec={spec} batch_invariant={os.environ.get("VLLM_BATCH_INVARIANT", "0")}: decode '
          f'log-prob == prefill log-prob bitwise for {eq}/{tot} generated tokens; greedy token is prefill argmax for '
          f'{arg_eq}/{tot}; tokens+log-probs sha256 {h.hexdigest()[:16]}')
    if exact and exact_gdn.DEBUG:
        print('DEBUG', exact_gdn.DEBUG_STATS)


if __name__ == '__main__':
    main()
