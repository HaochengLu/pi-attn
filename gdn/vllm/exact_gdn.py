"""Exact GDN for vLLM 0.17.1 (Qwen3.5 / Qwen3-Next), installed by monkeypatching before the engine is built.

Per GDN layer:
  * state: each sequence's slot in vLLM's mamba state pool (the "ssm" tensor) holds gdn/exact_decode.py's caches
    (fp32 state at the last 64-token boundary, its bf16 copy and the current chunk's rows; state_layout), so vLLM
    sizes and allocates them itself and the memory cost of exactness shows up as fewer concurrent sequences.
  * prefill (incl. chunked prefill continuing a sequence): fla 0.5.0's chunk program, the trainer's (vLLM's vendored
    copy is a different one, e13), up to each prompt chunk's last 64-token boundary, in two varlen calls: the chunks
    that complete a partly cached chunk (the cached rows packed in front of the new tokens, from the sequence's
    boundary state) and then the whole chunks after them, in place in the batch's rows; the rows after the last
    boundary are replayed through exact_decode, which gives them (bitwise fla's rows) and fills the row caches.
  * decode: exact_decode (bitwise the row fla's chunked forward gives that token), in decode-only and mixed batches
    alike (stock vLLM uses its recurrent kernel for decode-only batches and its chunk kernel for mixed ones). Its
    kernels read q, k, v from the conv output and write the layer output in place (step_rows).
  * speculative decoding (n-gram drafter): each request's window [last token, drafts...] goes through an exact_decode
    window, so every accepted token gets the bits it gets in plain decode. The previous window's acceptance is
    committed at the start of the next step (rows past it are overwritten later; a chunk is folded into the state only
    once its 64th row is accepted). The proposer is patched to cut drafts at the next 64-token boundary, which a window
    never crosses. One-token entries (no draft metadata in the step, or a one-token prompt chunk) are plain decodes,
    after a fold their slot's last window may have left.
  * memory: one state block per request also with speculative decoding (vLLM allocates 1 + drafts, a state per draft
    to roll back to; the request's slot is column 0 of its block table either way).
  * positions: a request's current position is seq_len - query_len from the runner's metadata. Per slot, ws and wn
    record the last step: a window (its first position and length; its acceptance is pos - ws, committed at the
    slot's next step), or a step that kept every token (the next position, wn = 0). Each layer commits its slots'
    previous windows in one launch (_advance); the index plans of a step are built once and shared by the layers.
    No step waits for the device (indices go host-to-device from pinned memory, nothing comes back), except the rare
    speculative-engine step with a one-token prompt chunk among longer ones (vLLM's prefill conv without metadata).
Unchanged: projections, conv1d (vLLM's prefill, decode and speculative conv kernels agree bitwise: check_conv*.py),
gating, norm.

vLLM 0.17.1 issues met on the way (they affect stock speculative decoding of hybrid models too; see
fix_vllm_mixed_spec_decodes): decode requests without drafts are marked like prefill requests, so a batch mixing them
with speculative ones kills the engine on an assertion and their conv state is not rolled back; in a step in which no
request has a draft vLLM sends no speculative metadata and runs its plain conv on the decode requests (alone, or with
prefill requests in the same batch), which takes the previous window's rejected drafts as conv history (handled here
by rolling the conv state back with the acceptance derived from positions); with speculative decoding the mamba block
table has 1 + drafts columns, so a request's slot column is a strided view (exact_decode compacts it); a sequence
whose first chunk has one token (a one-token prompt, or a prompt the token budget cuts that way) is run as a decode
from whatever state its slot holds (such slots are cleared here); vLLM's count of
accepted tokens is stale for a request its scheduler left out of a step, so the speculative conv update is rolled back
by the acceptance derived from positions here, as GDN is.

Not handled: prefix caching, tensor parallelism (asserted), drafters other than n-gram (their drafts are not cut at
the 64-token boundary; a window crossing it trips a device-side assertion), and full CUDA graphs with speculative
decoding (stock vLLM 0.17.1 fails to capture them for Qwen3.5; use PIECEWISE).
Use: VLLM_ENABLE_V1_MULTIPROCESSING=0 (the patch must live in the process that runs the model), PYTHONPATH with
env/fla_pkgs and gdn/, mamba_ssm_cache_dtype='float32', enable_prefix_caching=False; call install() before LLM().
Decode equals the engine's prefill bitwise only with batch-invariant dense ops as well (VLLM_BATCH_INVARIANT=1,
attention_backend='FLASH_ATTN'); without them GDN is exact and the dense ops are not.
EXACT_GDN_DEBUG=1 cross-checks every speculative window and every fold against fla's chunk program (slow)."""
import itertools
import os

import torch
import triton
import triton.language as tl

BT = 64
DEBUG = os.environ.get('EXACT_GDN_DEBUG') == '1'
DEBUG_STATS = {}


def install():
    import numpy as np
    import vllm.model_executor.layers.mamba.mamba_utils as mu
    import vllm.model_executor.models.qwen3_next as qn
    from vllm.v1.spec_decode.ngram_proposer import NgramProposer
    from exact_decode import state_layout, pin_fla_configs

    orig_shape = mu.MambaStateShapeCalculator.gated_delta_net_state_shape.__func__

    def gated_delta_net_state_shape(cls, tp_world_size, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
                                    conv_kernel_size, num_spec=0):
        conv_shape, _ = orig_shape(cls, tp_world_size, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
                                   conv_kernel_size, num_spec)
        assert tp_world_size == 1, 'exact GDN: tensor parallelism not handled yet'
        _, words = state_layout(num_k_heads, num_v_heads, head_k_dim, head_v_dim)
        return conv_shape, (words,)

    mu.MambaStateShapeCalculator.gated_delta_net_state_shape = classmethod(gated_delta_net_state_shape)
    pin_fla_configs()                                                # fla's forward as one program (prefill)

    orig_spec = qn.Qwen3NextGatedDeltaNet.get_kv_cache_spec

    def get_kv_cache_spec(self, vllm_config):
        """One state block per request: vLLM's speculative path keeps a state per draft token (num_speculative_blocks
        extra blocks) to roll back to; exact_decode rolls back in place, so they would only cut concurrency."""
        import dataclasses
        return dataclasses.replace(orig_spec(self, vllm_config), num_speculative_blocks=0)

    qn.Qwen3NextGatedDeltaNet.get_kv_cache_spec = get_kv_cache_spec
    fix_vllm_mixed_spec_decodes()
    qn.Qwen3NextGatedDeltaNet._forward_core = _forward_core_exact

    orig_propose = NgramProposer.propose

    def propose(self, sampled_token_ids, num_tokens_no_spec, token_ids_cpu, slot_mappings=None):
        """Drafts for request i verify in a window starting at its last sampled token, position
        num_tokens_no_spec[i] - 1; cut them so the window ends at or before the 64-token boundary."""
        drafts = orig_propose(self, sampled_token_ids, num_tokens_no_spec, token_ids_cpu, slot_mappings)
        for i, d in enumerate(drafts):
            room = BT - 1 - (int(num_tokens_no_spec[i]) - 1) % BT
            if len(d) > room:
                drafts[i] = d[:room]
        return drafts

    NgramProposer.propose = propose


def fix_vllm_mixed_spec_decodes():
    """vLLM 0.17.1 bug fix, applied to stock and exact runs alike. With n-gram drafting some decode requests have no
    draft in a step (no match). The runner marks them like prefill requests (-1 in num_decode_draft_tokens), so they
    become non-speculative decodes in a batch with speculative ones, which the GDN metadata builder asserts never
    happens (`num_decodes: x, num_spec_decodes: y`), killing the engine; and as non-speculative decodes their conv
    state would not be rolled back by the previous window's acceptance (the plain conv update takes the rejected drafts
    as history). The fix: the runner marks every request past its prompt as a speculative request with 0 or more drafts
    (it knows prompt lengths; the builder does not), and the builder's assertion is dropped for what remains, a
    one-token chunk of a prompt (query length 1, so vLLM counts it as a decode). For the exact layer the metadata also
    keeps the batch's seq_lens and query_start_loc (the runner's persistent buffers, so valid inside CUDA graphs), the
    host copies of the non-speculative entries' query_start_loc and of every request's computed tokens (the runner's),
    and, in mixed batches, the batch index of every speculative and non-speculative entry: the exact layer derives
    each request's absolute position from them instead of the accepted counts, which vLLM passes only in steps that
    have drafts."""
    import inspect, textwrap
    import vllm.v1.attention.backends.gdn_attn as ga
    import vllm.v1.worker.gpu_model_runner as gmr
    B = ga.GDNAttentionMetadataBuilder
    if getattr(B, '_exact_fixed', False):
        return
    R = gmr.GPUModelRunner
    src = textwrap.dedent(inspect.getsource(R._prepare_inputs))
    old = "num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)"
    assert src.count(old) == 1, 'vLLM runner changed; re-check the fix'
    src = src.replace(old, "num_decode_draft_tokens = np.where(self.input_batch.num_computed_tokens_cpu[:num_reqs] >= "
                           "self.input_batch.num_prompt_tokens[:num_reqs], 0, -1).astype(np.int32)")
    ns = {}
    exec(compile(src, gmr.__file__, 'exec'), gmr.__dict__, ns)
    R._prepare_inputs = ns['_prepare_inputs']
    src = textwrap.dedent(inspect.getsource(B.build))
    old_assert = """    assert not (num_decodes > 0 and num_spec_decodes > 0), (
        f"num_decodes: {num_decodes}, num_spec_decodes: {num_spec_decodes}"
    )
"""
    assert old_assert in src, 'vLLM GDN builder changed; re-check the fix'
    src = src.replace(old_assert, "")
    # a batch whose decode requests all have zero drafts stays on the speculative path
    import re
    src, n_sum = re.subn(r"num_decode_draft_tokens_cpu\[num_decode_draft_tokens_cpu >= 0\]\s*\.sum\(\)\s*\.item\(\)\s*== 0",
                         "(num_decode_draft_tokens_cpu >= 0).sum().item() == 0", src)
    assert n_sum == 1, 'vLLM GDN builder changed; re-check the fix'
    src = src.replace("    return attn_metadata\n", """    attn_metadata.seq_lens_all = m.seq_lens
    attn_metadata.qsl_all = m.query_start_loc
    attn_metadata.non_spec_qsl_cpu = non_spec_query_start_loc_cpu
    mixed = spec_sequence_masks is not None and (num_prefills > 0 or num_decodes > 0)
    _dev = m.query_start_loc.device
    _mask = spec_sequence_masks_cpu[:m.num_reqs] if mixed else None
    attn_metadata.spec_req_idx = torch.nonzero(_mask).flatten().pin_memory().to(_dev, non_blocking=True) if mixed else None
    attn_metadata.non_spec_req_idx_cpu = torch.nonzero(~_mask).flatten() if mixed else None
    attn_metadata.non_spec_req_idx = (attn_metadata.non_spec_req_idx_cpu.pin_memory().to(_dev, non_blocking=True)
                                      if mixed else None)
    attn_metadata.num_computed_cpu = m._num_computed_tokens_cpu
    return attn_metadata
""")
    ns = {}
    exec(compile(src, ga.__file__, 'exec'), ga.__dict__, ns)
    B.build = ns['build']
    B._exact_fixed = True


def _decoders(layer, storage, num_spec):
    """The layer's exact_decode objects over vLLM's pool (rebuilt if vLLM reallocates it): one for single tokens
    (decode, replay) and, with speculative decoding, one for windows of 1 + num_spec tokens, sharing the storage and
    the fold flags; plus the slot counters ws, wn (_advance_windows)."""
    from exact_decode import GDNExactDecode
    st = getattr(layer, '_exact', None)
    if st is None or st['one'].storage.data_ptr() != storage.data_ptr():
        assert storage.dtype == torch.float32, 'exact GDN needs mamba_ssm_cache_dtype=float32'
        n, dev = storage.shape[0], storage.device
        assert 1 + num_spec <= 16, 'exact GDN: windows of at most 16 tokens (num_speculative_tokens <= 15)'
        mk = lambda KT: GDNExactDecode(n, layer.num_k_heads, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim,
                                       dev, KT=KT, BV=128, nw1=2, nw2=4, storage=storage)
        st = {'one': mk(1), 'ws': torch.zeros(n, device=dev, dtype=torch.int32),
              'wn': torch.zeros(n, device=dev, dtype=torch.int32)}
        if num_spec:
            st['win'] = mk(1 + num_spec)
            st['win'].pending = st['one'].pending
        layer._exact = st
    return st


def _positions(md, req_idx, n):
    """Absolute position of the first token each of n batch entries feeds this step: the request's computed tokens,
    seq_len - query_len (req_idx maps entries to batch requests; None means entries 0..n-1)."""
    r = torch.arange(n, device=md.seq_lens_all.device) if req_idx is None else req_idx
    r = r.clamp(max=md.seq_lens_all.shape[0] - 1)                  # CUDA-graph padding entries (slot -1) beyond it
    return (md.seq_lens_all[r] - (md.qsl_all[r + 1] - md.qsl_all[r])).to(torch.int32)


@triton.jit
def _advance_windows(slots, pos, nrows, ws, wn, pending, acc, n, ACCEPTED: tl.constexpr, BT: tl.constexpr,
                     BLOCK: tl.constexpr):
    """Entry i (slot s = slots[i]; skipped if s < 0, CUDA-graph padding) feeds tokens from position pos[i] on. If s's
    last step was a window (wn[s] > 0) of the same sequence (pos > 0), pos - ws[s] of its tokens were accepted:
    acc[i] = that (else 1, as after a plain step), and s's fold flag is set if they completed a chunk. Then this step
    is recorded: ws[s], wn[s] = pos, nrows[i] (a window, committed at s's next step), or with ACCEPTED (every token
    of the step is kept: plain decode, prompt chunk) the next position and no window, pos + nrows[i], 0."""
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inb = o < n
    s = tl.load(slots + o, mask=inb, other=-1).to(tl.int64)
    m = inb & (s >= 0)
    p = tl.load(pos + o, mask=m, other=0)
    w0 = tl.load(ws + s, mask=m, other=0)
    prev = m & (tl.load(wn + s, mask=m, other=0) > 0) & (p > 0)
    a = tl.where(prev, p - w0, 1)
    tl.store(acc + o, a, mask=inb)
    tl.store(pending + s, tl.full([BLOCK], 1, tl.int32), mask=prev & (w0 % BT + a == BT))
    nr = tl.load(nrows + o, mask=m, other=0)
    if ACCEPTED:
        tl.store(ws + s, p + nr, mask=m)
        tl.store(wn + s, tl.zeros([BLOCK], tl.int32), mask=m)
    else:
        tl.store(ws + s, p, mask=m)
        tl.store(wn + s, nr, mask=m)


def _advance(st, slots, pos, nrows, accepted):
    """Commit each slot's previous window and record this step's (_advance_windows); one launch, no host sync.
    slots, pos, nrows: contiguous int32. Returns the previous window's acceptance per entry (1 if none), which the
    speculative conv update rolls back by."""
    n = slots.shape[0]
    acc = torch.empty_like(pos)
    _advance_windows[(triton.cdiv(n, 256),)](slots, pos, nrows, st['ws'], st['wn'], st['one'].pending, acc, n,
                                            ACCEPTED=accepted, BT=BT, BLOCK=256)
    return acc


def _forward_core_exact(self, mixed_qkv, b, a, core_attn_out):
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    from vllm.model_executor.models.qwen3_next import fused_gdn_gating

    fc = get_forward_context()
    md = fc.attn_metadata
    if md is None:                                                   # profile run
        return
    md = md[self.prefix]
    kv = self.kv_cache[fc.virtual_engine]
    conv_state = kv[0].transpose(-1, -2)
    st = _decoders(self, kv[1], self.num_spec)
    n_tok = md.num_actual_tokens
    mixed_qkv, b, a = mixed_qkv[:n_tok], b[:n_tok], a[:n_tok]
    conv_w = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
    spec = md.spec_sequence_masks is not None
    only_spec = spec and md.num_prefills == 0 and md.num_decodes == 0

    # 1. this step's entries (plans shared by the layers) and each slot's previous window, committed (_advance):
    #    speculative windows; one-token entries (decodes, one-token prompt chunks), whose slot is cleared if they start
    #    a sequence; prompt chunks (_prefill)
    if spec:
        sp = _spec_plan(md, not only_spec)
        acc_s = _advance(st, sp['slots'], sp['pos'], sp['nrows'], False)
    x_non = None if only_spec else (mixed_qkv if not spec else mixed_qkv.index_select(0, md.non_spec_token_indx))
    ns = _split_non_spec(md, self.num_spec) if x_non is not None and md.num_decodes + md.num_prefills > 0 else None
    one = None
    if ns is not None:
        n_one = md.num_decodes if ns['one'] is None else ns['one'].numel()
        if n_one:
            _reset_fresh(st, md, ns, kv[0])
            one = _one_plan(md, ns, n_one)
            # speculative engine: the previous step may have been a window; these steps keep every token
            one['acc'] = _advance(st, one['slots'], one['pos'], one['ones'], True) if self.num_spec else None

    # 2. conv1d and gating, exactly as stock vLLM (its prefill and decode conv kernels agree bitwise), the speculative
    #    update rolled back by the acceptance derived from positions (as GDN is): vLLM's own count is stale for a
    #    request it did not schedule in some step (its persistent batch drops and re-adds it)
    if spec:
        nsd = md.num_spec_decodes
        x_spec = mixed_qkv if only_spec else mixed_qkv.index_select(0, md.spec_token_indx)
        x_spec = causal_conv1d_update(x_spec, conv_state, conv_w, self.conv1d.bias, self.activation,
                                      conv_state_indices=sp['slots'][:nsd], num_accepted_tokens=acc_s[:nsd],
                                      query_start_loc=md.spec_query_start_loc,
                                      max_query_len=1 + self.num_spec, validate_data=False)
    if ns is not None:
        if md.num_prefills == 0:                                     # one-token entries only (no host sync)
            x_non = _conv_one(self, x_non, conv_state, conv_w, one)
        elif self.num_spec == 0 or one is None:
            # no rollback to do (no speculation, or prompt chunks only): vLLM's prefill conv over every entry, one-token
            # ones included, with its metadata (host-built: no device-to-host copy)
            x_non = causal_conv1d_fn(x_non.transpose(0, 1), conv_w, self.conv1d.bias, activation=self.activation,
                                     conv_states=conv_state, has_initial_state=md.has_initial_state,
                                     cache_indices=md.non_spec_state_indices_tensor,
                                     query_start_loc=md.non_spec_query_start_loc, metadata=md).transpose(0, 1)
        else:                            # one-token entries rolled back, prompt chunks (this path syncs: rare)
            x_out = torch.empty_like(x_non)
            if one is not None:
                x_out[ns['tok_one']] = _conv_one(self, x_non[ns['tok_one']], conv_state, conv_w, one)
            if ns['long'].numel():
                x_out[ns['tok_long']] = causal_conv1d_fn(
                    x_non[ns['tok_long']].transpose(0, 1), conv_w, self.conv1d.bias, activation=self.activation,
                    conv_states=conv_state, has_initial_state=md.has_initial_state[ns['long']],
                    cache_indices=md.non_spec_state_indices_tensor[ns['long']], query_start_loc=ns['qsl_long'],
                    metadata=None).transpose(0, 1)
            x_non = x_out
    g, beta = fused_gdn_gating(self.A_log, a, b, self.dt_bias)       # [1, L, HV] fp32, [1, L, HV] b.dtype
    scale = self.head_k_dim ** -0.5
    out = core_attn_out[:n_tok]

    # 3. the GDN core (speculative windows: read from the conv output and written to the layer output in place)
    if spec:
        _spec_windows(self, st, sp, x_spec, g[0], beta[0], scale, out)
    if ns is not None:
        g_n, b_n = (g, beta) if not spec else (g.index_select(1, md.non_spec_token_indx),
                                               beta.index_select(1, md.non_spec_token_indx))
        o_n = torch.zeros(x_non.shape[0], self.num_v_heads, self.head_v_dim, device=x_non.device, dtype=out.dtype)
        _non_spec(self, st, md, x_non, g_n[0], b_n[0], scale, o_n, ns, one)
        if spec:
            out.index_copy_(0, md.non_spec_token_indx, o_n)
        else:
            out.copy_(o_n)


def _spec_plan(md, mixed):
    """The speculative entries of this step, built once and shared by every layer (cached on the metadata): slots
    (the block table's slot column, compacted), window positions and lengths, and each window's first row among the
    speculative tokens (the conv output) and in the batch (gate, beta and the layer output; the same rows unless the
    batch is mixed)."""
    sp = md.__dict__.get('_exact_sp')
    if sp is not None:
        return sp
    nsd = md.spec_state_indices_tensor.shape[0]                      # spec requests (incl. CUDA-graph padding)
    slots = md.spec_state_indices_tensor[:, 0].to(torch.int32).contiguous()
    qsl = md.spec_query_start_loc[:nsd + 1]
    nrows = (qsl[1:] - qsl[:-1]).to(torch.int32)
    pos = _positions(md, md.spec_req_idx, nsd)
    # a window never runs past its chunk's 64th row: the n-gram proposer is cut to that (install); a drafter that is
    # not would get rows the kernels do not compute, so fail loudly (device-side, no host sync)
    torch._assert_async(((nrows <= BT - pos % BT) | (slots < 0)).all())
    rows = qsl[:-1].to(torch.int32)
    sp = {'slots': slots, 'pos': pos, 'nrows': nrows, 'rows': rows,
          'rows_b': md.qsl_all[md.spec_req_idx].to(torch.int32) if mixed else rows}
    md._exact_sp = sp
    return sp


def _spec_windows(self, st, sp, x, g, beta, scale, out):
    """Speculative requests: verify this step's windows (the previous windows are committed by _advance). x: the
    speculative tokens after conv1d, q | k | v per row; g, beta [rows, HV] and out [rows, HV, V] over the batch."""
    dec = st['win']
    q, k, v = x[:, :self.key_dim], x[:, self.key_dim:2 * self.key_dim], x[:, 2 * self.key_dim:]
    folds = _debug_fold_refs(dec, sp['slots'], scale) if DEBUG else None
    dec.step_rows(q, k, v, g, beta, sp['rows'], sp['rows_b'], sp['pos'], sp['nrows'], scale, sp['slots'], out,
                  sp['rows_b'])
    if DEBUG:                            # the windows' inputs and outputs gathered [n, KT, ...]
        j = torch.arange(dec.KT, device=x.device)
        gx = (sp['rows'][:, None] + j).clamp(max=x.shape[0] - 1).long()
        gb = (sp['rows_b'][:, None] + j).clamp(max=g.shape[0] - 1).long()
        H, K, HV, V = self.num_k_heads, self.head_k_dim, self.num_v_heads, self.head_v_dim
        xs = (q[gx].unflatten(2, (H, K)), k[gx].unflatten(2, (H, K)), v[gx].unflatten(2, (HV, V)), g[gb], beta[gb])
        _debug_check(dec, sp['slots'], sp['pos'], sp['nrows'], xs, scale, out[gb], folds)


def _split_non_spec(md, num_spec):
    """The non-speculative entries, split into one-token entries (decodes, or one-token chunks of a prompt: run as
    decodes) and longer ones (prompt chunks: prefill). vLLM orders a batch by query length <= its reorder threshold:
    1 without speculation, so the one-token entries come first (ns['one'] is None: the first num_decodes entries, rows
    0..); 1 + drafts with speculation, so a short prompt chunk can precede a one-token entry and the entries are split
    by query length. Computed once per step (cached on the metadata the layers share), from the builder's host copy of
    query_start_loc (no sync)."""
    ns = md.__dict__.get('_exact_ns')
    if ns is not None:
        return ns
    ns = {'one': None}
    if md.num_prefills > 0:
        n = md.num_decodes + md.num_prefills
        qsl = md.non_spec_qsl_cpu[:n + 1].tolist()
        lens = [qsl[i + 1] - qsl[i] for i in range(n)]
        if num_spec == 0:
            long = list(range(md.num_decodes, n))
            parts = {'long': long}
        else:
            one = [i for i in range(n) if lens[i] == 1]
            long = [i for i in range(n) if lens[i] > 1]
            parts = {'long': long, 'one': one, 'tok_one': [qsl[i] for i in one],
                     'tok_long': [r for i in long for r in range(qsl[i], qsl[i + 1])],
                     'qsl_long': [0] + list(itertools.accumulate(lens[i] for i in long))}
        ns.update(_h2d(parts, md.non_spec_query_start_loc.device), long_host=long, qsl_host=qsl)
        if num_spec:
            ns['qsl_long'] = ns['qsl_long'].to(torch.int32)
    md._exact_ns = ns
    return ns


_STAGING = {'bufs': [None] * 4, 'events': [None] * 4, 'next': 0}


def _h2d(parts, dev):
    """Index arrays {name: host ints} to the device as int64 tensors, in one copy that does not stall the host: a copy
    from pageable memory first waits for the stream, so the data goes through a pinned staging buffer, one of a ring
    of four, each reused once the copy that last read it has run (pinning a new buffer per copy costs ~1 ms)."""
    import numpy as np
    arrs = [np.asarray(a, dtype=np.int64).ravel() for a in parts.values()]
    flat = torch.from_numpy(np.concatenate(arrs)) if arrs else torch.zeros(0, dtype=torch.int64)
    i = _STAGING['next']
    _STAGING['next'] = (i + 1) % len(_STAGING['bufs'])
    if _STAGING['events'][i] is not None:
        _STAGING['events'][i].synchronize()
    buf = _STAGING['bufs'][i]
    if buf is None or buf.numel() < flat.numel():
        buf = _STAGING['bufs'][i] = torch.empty(max(2 * flat.numel(), 1 << 16), dtype=torch.int64, pin_memory=True)
    buf[:flat.numel()].copy_(flat)
    out = buf[:flat.numel()].to(dev, non_blocking=True)
    _STAGING['events'][i] = torch.cuda.Event()
    _STAGING['events'][i].record()
    return dict(zip(parts, torch.split(out, [a.size for a in arrs])))


def _one_entries(md, ns, n_rows):
    """(slots, batch request indices, count) of the one-token entries (n_rows: their count when they are the first
    entries)."""
    slots = md.non_spec_state_indices_tensor
    if ns['one'] is None:
        n = n_rows
        return slots[:n], (None if md.non_spec_req_idx is None else md.non_spec_req_idx[:n]), n
    ent = ns['one']
    return slots[ent], (ent if md.non_spec_req_idx is None else md.non_spec_req_idx[ent]), ent.numel()


def _one_plan(md, ns, n_rows):
    """The one-token entries' slots (compacted int32), positions and rows among the non-speculative tokens, plus the
    per-entry constants their updates take; built once per step (cached with ns). Per layer, 'acc' is added: the
    previous window's acceptance."""
    p = ns.get('one_plan')
    if p is None:
        sl, req, n = _one_entries(md, ns, n_rows)
        pos = _positions(md, req, n)
        qsl = torch.arange(n + 1, device=pos.device, dtype=torch.int32)
        p = ns['one_plan'] = {'slots': sl.to(torch.int32).contiguous(), 'pos': pos, 'ones': torch.ones_like(pos),
                              'qsl': qsl, 'rows': qsl[:n] if ns['one'] is None else ns['tok_one'].to(torch.int32)}
    return dict(p)


@triton.jit
def _clear_new_sequences(slots, req, seq_lens, qsl, n_req, S, sS, nS, Hb, sHb, nHb, conv, sconv, nconv, pending, ws, wn,
                         HAS_REQ: tl.constexpr, BLOCK: tl.constexpr):
    """For each one-token entry i: if it is at position 0 (seq_len - query_len of its request), clear its slot: boundary
    state and its bf16 copy, conv history, fold flag, window counters. Other entries return at once."""
    i = tl.program_id(0)
    slot = tl.load(slots + i).to(tl.int64)
    if slot < 0:                                   # CUDA-graph padding
        return
    if HAS_REQ:
        r = tl.load(req + i).to(tl.int64)
    else:
        r = i.to(tl.int64)
    r = tl.minimum(r, n_req - 1)
    if tl.load(seq_lens + r) - (tl.load(qsl + r + 1) - tl.load(qsl + r)) != 0:
        return
    o = tl.arange(0, BLOCK)
    for s0 in range(0, nS, BLOCK):
        tl.store(S + slot * sS + s0 + o, tl.zeros([BLOCK], S.dtype.element_ty), mask=s0 + o < nS)
    for s0 in range(0, nHb, BLOCK):
        tl.store(Hb + slot * sHb + s0 + o, tl.zeros([BLOCK], Hb.dtype.element_ty), mask=s0 + o < nHb)
    for s0 in range(0, nconv, BLOCK):
        tl.store(conv + slot * sconv + s0 + o, tl.zeros([BLOCK], conv.dtype.element_ty), mask=s0 + o < nconv)
    tl.store(pending + slot, 0)
    tl.store(ws + slot, 0)
    tl.store(wn + slot, 0)


def _reset_fresh(st, md, ns, conv_rows):
    """A one-token entry at position 0 starts a new sequence (a prompt cut into a first chunk of one token by the token
    budget, or a one-token prompt) and runs as a decode, from whatever its slot holds: vLLM does not clear mamba slots
    for new requests. Such slots get the zero state, an empty conv history and no previous window. One launch per
    layer, graph-capturable, no host sync; the row caches need no clearing (a sequence reads only rows it wrote)."""
    sl, req, n = _one_entries(md, ns, md.num_decodes)
    dec = st['one']
    assert conv_rows[0].is_contiguous() and dec.S[0].is_contiguous() and dec.Hb[0].is_contiguous()
    _clear_new_sequences[(n,)](
        sl.to(torch.int32).contiguous(), req if req is not None else sl, md.seq_lens_all, md.qsl_all,
        md.seq_lens_all.shape[0], dec.S, dec.S.stride(0), dec.S[0].numel(), dec.Hb, dec.Hb.stride(0), dec.Hb[0].numel(),
        conv_rows, conv_rows.stride(0), conv_rows[0].numel(), dec.pending, st['ws'], st['wn'],
        HAS_REQ=req is not None, BLOCK=1024)


def _conv_one(self, x, conv_state, conv_w, one):
    """conv1d of the one-token entries x [n, dim]. In a speculative engine their previous step may have been a window
    with rejected drafts, while vLLM's plain conv update takes the conv history as it stands (vLLM sends no speculative
    metadata in a step without drafts, and one-token prompt chunks are never speculative): roll the history back by
    the previous window's acceptance (one['acc'], from positions; 1 when the last step was not a window)."""
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    if one['acc'] is None:
        return causal_conv1d_update(x, conv_state, conv_w, self.conv1d.bias, self.activation,
                                    conv_state_indices=one['slots'], validate_data=True)
    return causal_conv1d_update(x, conv_state, conv_w, self.conv1d.bias, self.activation,
                                conv_state_indices=one['slots'], num_accepted_tokens=one['acc'],
                                query_start_loc=one['qsl'], max_query_len=1 + self.num_spec, validate_data=False)


def _non_spec(self, st, md, x, g, beta, scale, out, ns, one):
    """Non-speculative entries: one-token entries (plain decodes: every token is kept, and a chunk is folded on its
    64th row in the same step) and prompt chunks (prefill)."""
    dec = st['one']
    prefill = bool(ns.get('long_host'))
    if prefill:                          # vLLM's prefill conv output is column-major: copied to [1, L, H, K] ...
        q, k, v = self.rearrange_mixed_qkv(x)
        qkv = (q[0].flatten(1), k[0].flatten(1), v[0].flatten(1))
    else:                                # the conv update's output, q | k | v per row, read in place
        qkv = (x[:, :self.key_dim], x[:, self.key_dim:2 * self.key_dim], x[:, 2 * self.key_dim:])
    if prefill:                          # first: it writes every row of out, the one-token entries' rows below
        _prefill(self, st, md, qkv, g, beta, scale, out, ns)
    if one is not None:
        if 'win' in st:                  # a chunk that the previous window's acceptance completed (marked by _advance)
            dec.fold(one['slots'])
        dec.step_rows(*qkv, g, beta, one['rows'], one['rows'], one['pos'], one['ones'], scale, one['slots'], out,
                      one['rows'])


def _prefill_plan(md, ns, L):
    """Index plan of _prefill for this step's prompt chunks (L: rows of the non-speculative tokens). It does not
    depend on the layer, so it is built once per step (cached on the metadata the layers share), from host data the
    builder has: positions (the runner's num_computed_tokens) and token offsets (non_spec_qsl_cpu); one host-to-device
    copy, no device-to-host copy."""
    import numpy as np
    pf = md.__dict__.get('_exact_pf')
    if pf is not None:
        return pf
    W = 16
    long = ns['long_host']
    req = long if md.non_spec_req_idx_cpu is None else md.non_spec_req_idx_cpu[long].tolist()
    assert md.num_computed_cpu is not None, 'exact GDN: the runner passes no host num_computed_tokens'
    p0 = md.num_computed_cpu[req].numpy().astype(np.int64)
    qh = ns['qsl_host']
    s0 = np.array([qh[i] for i in long], dtype=np.int64)
    e0 = np.array([qh[i + 1] for i in long], dtype=np.int64)
    end = p0 + e0 - s0
    b0, b1 = p0 - p0 % BT, end - end % BT
    bc = np.where(p0 % BT == 0, p0, b0 + BT)                         # first boundary at or after p0
    reach = b1 > b0                                                  # entries reaching a new boundary
    head = np.nonzero(reach & (p0 % BT != 0))[0]                     # ... completing a partly cached chunk
    body = np.nonzero(reach & (b1 > bc))[0]                          # ... with whole chunks bc .. b1 after it
    fresh = np.nonzero(p0 == 0)[0]
    # replay: (entry, first position, first token, rows) of each entry's rows after its last boundary
    r_first = np.where(reach, b1, p0)
    rp = np.nonzero(end > r_first)[0]
    r_pos, r_tok, r_n = r_first[rp], s0[rp] + r_first[rp] - p0[rp], end[rp] - r_first[rp]
    nwin = -(-int(r_n.max()) // W) if rp.size else 0
    w0 = np.arange(nwin)[:, None] * W                                # [nwin, 1]
    r_nr = np.clip(r_n[None, :] - w0, 0, W)                          # [nwin, nr] rows per window
    # head chunks, packed: the slot's cached rows 0 .. c-1, then the new tokens p0 .. bc-1 (64 rows each)
    nh = head.size
    c = (p0 - b0)[head]
    seg, off = np.repeat(np.arange(nh), BT), np.tile(np.arange(BT), nh)
    is_new = off >= c[seg]
    tok = s0[head][seg] + off - c[seg]
    # body chunks, in place in the batch's rows: rows bs .. be-1 of each body entry are a sequence of the varlen call,
    # the rows between them (heads, tails, one-token entries, padding) filler sequences whose outputs are not used
    bs, be = s0 + bc - p0, s0 + b1 - p0
    cuts, kind = [0], []
    for i in body:
        if bs[i] > cuts[-1]:
            cuts += [bs[i]]
            kind += [-1]
        cuts += [be[i]]
        kind += [i]
    if body.size and cuts[-1] < L:
        cuts += [L]
        kind += [-1]
    kind, b_cu = np.array(kind, dtype=np.int64), np.array(cuts, dtype=np.int64)
    nck = -(-np.diff(b_cu) // BT)
    b_ci = np.stack([np.repeat(np.arange(nck.size), nck), np.arange(nck.sum()) - np.repeat(np.cumsum(nck) - nck, nck)], 1)
    b_seq = np.nonzero(kind >= 0)[0]                                 # sequence of each body entry
    hidx = np.full(p0.size, -1)
    hidx[head] = np.arange(nh)
    from_s = (hidx[body] < 0) & (p0[body] > 0)                       # body starts from the slot's state
    from_h = hidx[body] >= 0                                         # ... from its head chunk's final state
    head_only = head[b1[head] == bc[head]]
    # one host-to-device copy of every index (entries are mapped to slots on the device)
    parts = {'fresh': fresh, 'h_ent': head, 'h_new': is_new, 'h_gi': np.where(is_new, tok, 0),
             'h_cs': np.where(is_new, 0, head[seg]), 'h_cr': np.where(is_new, 0, off), 'h_cu': np.arange(nh + 1) * BT,
             'h_ci': np.stack([np.arange(nh), np.zeros(nh, np.int64)], 1).ravel(),
             'h_oi': np.nonzero(is_new)[0], 'h_od': tok[is_new], 'b_cu': b_cu, 'b_ci': b_ci.ravel(), 'b_seq': b_seq,
             'b_ent': body, 'bs_seq': b_seq[from_s], 'bs_ent': body[from_s], 'bh_seq': b_seq[from_h],
             'bh_h': hidx[body[from_h]], 'ho_ent': head_only, 'ho_h': hidx[head_only],
             'r_ent': rp, 'r_pos': r_pos[None, :] + w0, 'r_nr': r_nr, 'r_row': r_tok[None, :] + w0, 'p0': p0,
             'qlen': e0 - s0}
    d = _h2d(parts, md.qsl_all.device)
    slots = md.non_spec_state_indices_tensor[ns['long']].to(torch.int32)
    sl = slots.long()
    i32 = torch.int32
    some = lambda key, v: v if parts[key].size else None
    pf = {'slots': slots, 'p0': d['p0'].to(i32), 'qlen': d['qlen'].to(i32), 'nwin': nwin, 'nh': nh, 'nb': body.size,
          'fresh': some('fresh', sl[d['fresh']]),
          'h_slot': sl[d['h_ent']], 'h_new': d['h_new'].bool(), 'h_gi': d['h_gi'], 'h_cs': sl[d['h_cs']],
          'h_cr': d['h_cr'], 'h_cu': d['h_cu'].to(i32), 'h_ci': d['h_ci'].view(-1, 2).to(i32), 'h_oi': d['h_oi'],
          'h_od': d['h_od'],
          'b_cu': d['b_cu'].to(i32), 'b_ci': d['b_ci'].view(-1, 2).to(i32), 'nseq': kind.size, 'b_seq': d['b_seq'],
          'b_slot': sl[d['b_ent']], 'bs_seq': some('bs_seq', d['bs_seq']), 'bs_slot': sl[d['bs_ent']],
          'bh_seq': some('bh_seq', d['bh_seq']), 'bh_h': d['bh_h'], 'ho_slot': some('ho_ent', sl[d['ho_ent']]),
          'ho_h': d['ho_h']}
    if nwin:
        pf.update(r_slot=slots[d['r_ent']], r_pos=d['r_pos'].view(nwin, -1).to(i32),
                  r_nr=d['r_nr'].view(nwin, -1).to(i32), r_row=d['r_row'].view(nwin, -1).to(i32))
    md._exact_pf = pf
    return pf


def _prefill(self, st, md, qkv, g, beta, scale, out, ns):
    """The prompt chunks (entries ns['long']) in one go. Entry i adds tokens at absolute positions p0 .. end-1 (rows
    s0 .. of the batch); its slot holds the state at b0 = 64 * (p0 // 64) and the rows b0 .. p0-1 (a new request:
    p0 = 0, and its slot is cleared). With bc the first boundary at or after p0 and b1 = 64 * (end // 64), fla's chunk
    program runs on the chunks up to b1, in two varlen calls: the head chunks b0 .. bc of the entries that complete a
    partly cached chunk (cached rows in front of the new tokens, packed; 64 rows each), then the body chunks bc .. b1
    of every entry, in place in the batch's rows (other rows go through as filler sequences), each from its head's
    final state, its slot's, or zero. They give the rows p0 .. b1-1 and the fp32 state at b1. The rows after the last
    boundary (b1 .. end-1, or p0 .. end-1 for an entry within one chunk) are replayed through exact_decode in windows of
    16 rows, which fills the row caches and gives those rows: decode rows are bitwise fla's rows, so every request gets
    the rows of the single forward over b0 .. end (row-causality and resumability: e1_premise). qkv: q, k, v [L, H*K]
    .. contiguous; g, beta [L, HV]; out [L, HV, V], every row of which is written here or after (one-token entries).
    The index plan is shared by the layers (_prefill_plan)."""
    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd
    from fla.modules.l2norm import l2norm_fwd
    dec = st['one']
    pf = _prefill_plan(md, ns, qkv[0].shape[0])
    slots = pf['slots']
    if DEBUG:                            # the host positions against the device's, and how often a window precedes
        add = lambda key, v: DEBUG_STATS.update({key: DEBUG_STATS.get(key, 0) + int(v)})
        req = ns['long'] if md.non_spec_req_idx is None else md.non_spec_req_idx[ns['long']]
        add('prefill_positions_bad', not torch.equal(_positions(md, req, slots.shape[0]), pf['p0']))
        add('prefill_after_window', ((st['wn'][slots.long()] > 0) & (pf['p0'] > 0)).sum())
    if 'win' in st:
        # speculative engine: record the chunk (no window to commit at the slot's next step); a window before it
        # (none in practice: one-token entries are plain decodes, and a prompt chunk follows prompt chunks) would be
        # committed, and the chunk its acceptance completed folded, here
        _advance(st, slots, pf['p0'], pf['qlen'], True)
        dec.fold(slots)
        dec.pending.index_fill_(0, slots.long(), 0)
    if pf['fresh'] is not None:     # a new request starts from the zero state (index_fill_: assigning a Python scalar
        dec.S.index_fill_(0, pf['fresh'], 0)       # through an index copies it from the host, which syncs)
        dec.Hb.index_fill_(0, pf['fresh'], 0)
    H, K, HV, V = self.num_k_heads, self.head_k_dim, self.num_v_heads, self.head_v_dim
    if pf['nh'] or pf['nb']:
        qn, kn = (l2norm_fwd(t.unflatten(1, (H, K)))[0] for t in qkv[:2])
        vv = qkv[2].unflatten(1, (HV, V))
    if pf['nh']:
        new, gi, cs, cr = pf['h_new'], pf['h_gi'], pf['h_cs'], pf['h_cr']
        sel = lambda newv, cachev: torch.where(new.view(-1, *([1] * (newv.dim() - 1))), newv[gi],
                                               cachev[cs, cr])[None].contiguous()
        Qh = torch.where(new[:, None, None], qn[gi], 0.)[None].contiguous()   # cached rows: outputs unused
        _, oH, _, SH, _, _ = chunk_gated_delta_rule_fwd(
            Qh, sel(kn, dec.kbuf), sel(vv, dec.vbuf), sel(g, dec.gbuf), sel(beta, dec.bbuf), scale,
            dec.S[pf['h_slot']].contiguous(), True, cu_seqlens=pf['h_cu'], chunk_indices=pf['h_ci'])
    if pf['nb']:
        init = torch.zeros(pf['nseq'], HV, K, V, device=qn.device, dtype=torch.float32)
        if pf['bs_seq'] is not None:
            init.index_copy_(0, pf['bs_seq'], dec.S[pf['bs_slot']])
        if pf['bh_seq'] is not None:
            init.index_copy_(0, pf['bh_seq'], SH[pf['bh_h']])
        _, oB, _, SB, _, _ = chunk_gated_delta_rule_fwd(
            qn[None], kn[None], vv[None], g[None], beta[None], scale, init, True, cu_seqlens=pf['b_cu'],
            chunk_indices=pf['b_ci'])
        out.copy_(oB[0])                 # the body rows; the other rows are overwritten below and after
        SB = SB[pf['b_seq']]
        dec.S[pf['b_slot']] = SB
        dec.Hb[pf['b_slot']] = SB.to(dec.Hb.dtype)
    if pf['nh']:
        out.index_copy_(0, pf['h_od'], oH[0, pf['h_oi']].to(out.dtype))
        if pf['ho_slot'] is not None:    # no body chunks after the head: its final state is the state at b1
            SH = SH[pf['ho_h']]
            dec.S[pf['ho_slot']] = SH
            dec.Hb[pf['ho_slot']] = SH.to(dec.Hb.dtype)
    for w in range(pf['nwin']):                                      # rows read and written in place
        dec.step_rows(*qkv, g, beta, pf['r_row'][w], pf['r_row'][w], pf['r_pos'][w], pf['r_nr'][w], scale,
                      pf['r_slot'], out, pf['r_row'][w], KT=16)


def _debug_fold_refs(dec, slots, scale):
    """EXACT_GDN_DEBUG: for every slot about to be folded (pending), fla's chunk program over the cached chunk rows."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    refs = {}
    for sl in slots.tolist():
        if sl >= 0 and int(dec.pending[sl]):
            c = lambda t: t[sl][None].contiguous()
            _, S1 = chunk_gated_delta_rule(torch.zeros_like(c(dec.kbuf)), c(dec.kbuf), c(dec.vbuf), c(dec.gbuf),
                                           c(dec.bbuf), scale=scale, initial_state=dec.S[sl][None].clone(),
                                           output_final_state=True, use_qk_l2norm_in_kernel=False)
            refs[sl] = S1[0]
    return refs


def _debug_check(dec, slots, pos, nrows, xs, scale, o, folds):
    """EXACT_GDN_DEBUG: every fold of this step, and every window's rows against fla's chunk program run from the
    slot's boundary state over its cached rows and the window; counts in DEBUG_STATS."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.modules.l2norm import l2norm_fwd
    count = lambda key, bad: DEBUG_STATS.update({key: DEBUG_STATS.get(key, 0) + 1,
                                                 key + '_bad': DEBUG_STATS.get(key + '_bad', 0) + bad})
    for sl, S1 in folds.items():
        count('folds', not torch.equal(dec.S[sl], S1))
    for i, sl in enumerate(slots.tolist()):
        if sl < 0:
            continue
        p, n = int(pos[i]), int(nrows[i])
        r = p % BT
        qi, ki, vi, gi, bi = (t[i, :n] for t in xs)
        qn, kn = l2norm_fwd(qi)[0], l2norm_fwd(ki)[0]
        cat = lambda old, new: torch.cat([old, new], 0)[None].contiguous()
        ref, _ = chunk_gated_delta_rule(cat(qn.new_zeros(r, *qn.shape[1:]), qn), cat(dec.kbuf[sl, :r], kn),
                                        cat(dec.vbuf[sl, :r], vi), cat(dec.gbuf[sl, :r], gi),
                                        cat(dec.bbuf[sl, :r], bi), scale=scale,
                                        initial_state=dec.S[sl][None].clone(), output_final_state=True,
                                        use_qk_l2norm_in_kernel=False)
        count('windows', not torch.equal(ref[0, r:], o[i, :n]))
