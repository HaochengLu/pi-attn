"""Exact GDN for vLLM 0.17.1 (Qwen3.5 / Qwen3-Next), installed by monkeypatching before the engine is built.

Per GDN layer:
  * state: each sequence's slot in vLLM's mamba state pool (the "ssm" tensor) holds gdn/exact_decode.py's caches
    (fp32 state at the last 64-token boundary, its bf16 copy and the current chunk's rows; state_layout), so vLLM
    sizes and allocates them itself and the memory cost of exactness shows up as fewer concurrent sequences.
  * prefill (incl. chunked prefill continuing a sequence): fla 0.5.0's chunk program, the trainer's (vLLM's vendored
    copy is a different one, e13), resumed from the sequence's boundary state with the cached rows of its partial
    chunk in front, varlen over the prefill requests, up to their last 64-token boundary; the rows after it are
    replayed through exact_decode, which gives them (bitwise fla's rows) and fills the row caches.
  * decode: exact_decode.decode (bitwise the row fla's chunked forward gives that token), in decode-only and mixed
    batches alike (stock vLLM uses its recurrent kernel for decode-only batches and its chunk kernel for mixed ones).
  * speculative decoding (n-gram drafter): each request's window [last token, drafts...] goes through
    exact_decode.step, so every accepted token gets the bits it gets in plain decode. The previous window's acceptance
    is committed at the start of the next step (rows past it are overwritten later; a chunk is folded into the state
    only once its 64th row is accepted). The proposer is patched to cut drafts at the next 64-token boundary, which a
    window never crosses.
  * memory: one state block per request also with speculative decoding (vLLM allocates 1 + drafts, a state per draft
    to roll back to; the request's slot is column 0 of its block table either way).
  * positions: per slot ws (absolute position of the last window's first token) and wn (its length). A request's
    current position is seq_len - query_len from the runner's metadata, its last window's acceptance pos - ws.
Unchanged: projections, conv1d (vLLM's prefill, decode and speculative conv kernels agree bitwise: check_conv*.py),
gating, norm.

vLLM 0.17.1 issues met on the way (they affect stock speculative decoding of hybrid models too; see
fix_vllm_mixed_spec_decodes): decode requests without drafts are marked like prefill requests, so a batch mixing them
with speculative ones kills the engine on an assertion and their conv state is not rolled back; in a step in which no
request has a draft vLLM sends no speculative metadata and runs its plain conv on the decode requests (alone, or with
prefill requests in the same batch), which takes the previous window's rejected drafts as conv history (handled here
by rolling the conv state back with the acceptance derived from positions); with speculative decoding the mamba block
table has 1 + drafts columns, so a request's slot column is a strided view (exact_decode compacts it); a one-token
prompt is run as a decode from whatever state its slot holds (refused in vLLM's input validation).

Not handled: prefix caching, tensor parallelism (asserted), drafters other than n-gram (their drafts are not cut at
the 64-token boundary; a window crossing it trips a device-side assertion), and full CUDA graphs with speculative
decoding (stock vLLM 0.17.1 fails to capture them for Qwen3.5; use PIECEWISE).
Use: VLLM_ENABLE_V1_MULTIPROCESSING=0 (the patch must live in the process that runs the model), PYTHONPATH with
env/fla_pkgs and gdn/, mamba_ssm_cache_dtype='float32', enable_prefix_caching=False; call install() before LLM().
Decode equals the engine's prefill bitwise only with batch-invariant dense ops as well (VLLM_BATCH_INVARIANT=1,
attention_backend='FLASH_ATTN'); without them GDN is exact and the dense ops are not.
EXACT_GDN_DEBUG=1 cross-checks every speculative window and every fold against fla's chunk program (slow)."""
import os

import torch

BT = 64
DEBUG = os.environ.get('EXACT_GDN_DEBUG') == '1'
DEBUG_STATS = {}


def install():
    import numpy as np
    import vllm.model_executor.layers.mamba.mamba_utils as mu
    import vllm.model_executor.models.qwen3_next as qn
    from vllm.v1.spec_decode.ngram_proposer import NgramProposer
    from vllm.v1.engine.input_processor import InputProcessor
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

    orig_validate = InputProcessor._validate_prompt_len

    def _validate_prompt_len(self, prompt_len, prompt_type):
        """vLLM 0.17.1 runs a one-token prompt as a decode (query length 1), from whatever state its slot holds (mamba
        slots are not cleared for new requests); the request is refused where vLLM refuses an empty prompt, so the
        caller gets the error and the engine runs on."""
        if prompt_len == 1 and prompt_type == 'decoder':
            raise ValueError('exact GDN: prompts of one token are not supported (vLLM would decode them from a stale '
                             'state slot)')
        return orig_validate(self, prompt_len, prompt_type)

    InputProcessor._validate_prompt_len = _validate_prompt_len


def fix_vllm_mixed_spec_decodes():
    """vLLM 0.17.1 bug fix, applied to stock and exact runs alike. With n-gram drafting some decode requests have no
    draft in a step (no match). The runner marks them like prefill requests (-1 in num_decode_draft_tokens), so they
    become non-speculative decodes in a batch with speculative ones, which the GDN metadata builder asserts never
    happens (`num_decodes: x, num_spec_decodes: y`), killing the engine; and as non-speculative decodes their conv
    state would not be rolled back by the previous window's acceptance (the plain conv update takes the rejected drafts
    as history). The fix: the runner marks every request past its prompt as a speculative request with 0 or more drafts
    (it knows prompt lengths; the builder does not), and the builder's assertion is dropped for what remains, a
    one-token chunk of a prompt (query length 1, so vLLM counts it as a decode). For the exact layer the metadata also
    keeps the batch's seq_lens and query_start_loc (the runner's persistent buffers, so valid inside CUDA graphs) and,
    in mixed batches, the batch index of every speculative and non-speculative entry: the exact layer derives each
    request's absolute position from them instead of the accepted counts, which vLLM passes only in steps that have
    drafts."""
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
    mixed = spec_sequence_masks is not None and (num_prefills > 0 or num_decodes > 0)
    _dev = m.query_start_loc.device
    attn_metadata.spec_req_idx = torch.nonzero(spec_sequence_masks_cpu[:m.num_reqs]).flatten().to(_dev) if mixed else None
    attn_metadata.non_spec_req_idx = torch.nonzero(~spec_sequence_masks_cpu[:m.num_reqs]).flatten().to(_dev) if mixed else None
    return attn_metadata
""")
    ns = {}
    exec(compile(src, ga.__file__, 'exec'), ga.__dict__, ns)
    B.build = ns['build']
    B._exact_fixed = True


def _decoders(layer, storage, num_spec):
    """The layer's exact_decode objects over vLLM's pool (rebuilt if vLLM reallocates it): one for single tokens
    (decode, replay) and, with speculative decoding, one for windows of 1 + num_spec tokens, sharing the storage and
    the fold flags; plus the slot counters ws, wn (one extra entry receives padded CUDA-graph entries, slot -1)."""
    from exact_decode import GDNExactDecode
    st = getattr(layer, '_exact', None)
    if st is None or st['one'].storage.data_ptr() != storage.data_ptr():
        assert storage.dtype == torch.float32, 'exact GDN needs mamba_ssm_cache_dtype=float32'
        n, dev = storage.shape[0], storage.device
        assert 1 + num_spec <= 16, 'exact GDN: windows of at most 16 tokens (num_speculative_tokens <= 15)'
        mk = lambda KT: GDNExactDecode(n, layer.num_k_heads, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim,
                                       dev, KT=KT, BV=128, nw1=2, nw2=4, storage=storage)
        st = {'one': mk(1), 'ws': torch.zeros(n + 1, device=dev, dtype=torch.int32),
              'wn': torch.zeros(n + 1, device=dev, dtype=torch.int32)}
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


def _slot_index(slots, counters):
    """Counter index per slot; padded CUDA-graph entries (slot < 0) go to the extra last entry."""
    return torch.where(slots < 0, counters.shape[0] - 1, slots).long()


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

    # 1. conv1d and gating, exactly as stock vLLM (its prefill and decode conv kernels agree bitwise)
    if spec:
        x_spec = mixed_qkv if only_spec else mixed_qkv.index_select(0, md.spec_token_indx)
        x_spec = causal_conv1d_update(x_spec, conv_state, conv_w, self.conv1d.bias, self.activation,
                                      conv_state_indices=md.spec_state_indices_tensor[:, 0][:md.num_spec_decodes],
                                      num_accepted_tokens=md.num_accepted_tokens,
                                      query_start_loc=md.spec_query_start_loc,
                                      max_query_len=1 + self.num_spec, validate_data=False)
    x_non = None if only_spec else (mixed_qkv if not spec else mixed_qkv.index_select(0, md.non_spec_token_indx))
    nd, npf = md.num_decodes, md.num_prefills
    if nd > 0 and self.num_spec > 0:
        # a speculative engine in a step in which no request has drafts (so vLLM sends none of its speculative
        # metadata): vLLM would run its plain conv on the decode requests, which takes the previous window's rejected
        # drafts as history; roll the conv state back by the previous window's acceptance instead (decodes come first)
        sl = md.non_spec_state_indices_tensor[:nd]
        idx = _slot_index(sl, st['ws'])
        pos = _positions(md, None if md.non_spec_req_idx is None else md.non_spec_req_idx[:nd], nd)
        acc = torch.where(st['wn'][idx] > 0, pos - st['ws'][idx], 1).to(torch.int32)
        x_dec = causal_conv1d_update(x_non[:nd], conv_state, conv_w, self.conv1d.bias, self.activation,
                                     conv_state_indices=sl, num_accepted_tokens=acc,
                                     query_start_loc=torch.arange(nd + 1, device=x_non.device, dtype=torch.int32),
                                     max_query_len=1 + self.num_spec, validate_data=False)
        if npf > 0:                                                  # the prefill requests after them
            qsl = md.non_spec_query_start_loc[nd:nd + npf + 1]
            x_pre = causal_conv1d_fn(x_non[nd:qsl[-1]].transpose(0, 1), conv_w, self.conv1d.bias,
                                     activation=self.activation, conv_states=conv_state,
                                     has_initial_state=md.has_initial_state[nd:nd + npf],
                                     cache_indices=md.non_spec_state_indices_tensor[nd:nd + npf],
                                     query_start_loc=qsl - nd, metadata=None).transpose(0, 1)
            x_dec = torch.cat([x_dec, x_pre, x_non[qsl[-1]:]])
        x_non = x_dec
    elif npf > 0:
        x_non = causal_conv1d_fn(x_non.transpose(0, 1), conv_w, self.conv1d.bias, activation=self.activation,
                                 conv_states=conv_state, has_initial_state=md.has_initial_state,
                                 cache_indices=md.non_spec_state_indices_tensor,
                                 query_start_loc=md.non_spec_query_start_loc, metadata=md).transpose(0, 1)
    elif nd > 0:
        x_non = causal_conv1d_update(x_non, conv_state, conv_w, self.conv1d.bias, self.activation,
                                     conv_state_indices=md.non_spec_state_indices_tensor[:md.num_actual_tokens],
                                     validate_data=True)
    g, beta = fused_gdn_gating(self.A_log, a, b, self.dt_bias)       # [1, L, HV] fp32, [1, L, HV] b.dtype
    scale = self.head_k_dim ** -0.5
    out = core_attn_out[:n_tok]

    if spec:
        g_s, b_s = (g, beta) if only_spec else (g.index_select(1, md.spec_token_indx),
                                                beta.index_select(1, md.spec_token_indx))
        o_s = _spec_windows(self, st, md, x_spec, g_s[0], b_s[0], scale)
        if only_spec:
            out.copy_(o_s)
        else:
            out.index_copy_(0, md.spec_token_indx, o_s)
    if x_non is not None and (md.num_prefills > 0 or md.num_decodes > 0):
        g_n, b_n = (g, beta) if not spec else (g.index_select(1, md.non_spec_token_indx),
                                               beta.index_select(1, md.non_spec_token_indx))
        o_n = torch.zeros(x_non.shape[0], self.num_v_heads, self.head_v_dim, device=x_non.device, dtype=out.dtype)
        _non_spec(self, st, md, x_non, g_n[0], b_n[0], scale, o_n)
        if spec:
            out.index_copy_(0, md.non_spec_token_indx, o_n)
        else:
            out.copy_(o_n)


def _spec_windows(self, st, md, x, g, beta, scale):
    """Speculative requests: commit the previous window's acceptance, then verify this window."""
    dec, ws, wn = st['win'], st['ws'], st['wn']
    KT = dec.KT
    q, k, v = self.rearrange_mixed_qkv(x)                            # [1, L, H, K] ...
    nsd = md.spec_state_indices_tensor.shape[0]                      # spec requests (incl. CUDA-graph padding)
    slots = md.spec_state_indices_tensor[:, 0]
    qsl = md.spec_query_start_loc[:nsd + 1]
    nrows = (qsl[1:] - qsl[:-1]).to(torch.int32)
    idx = _slot_index(slots, ws)
    pos = _positions(md, md.spec_req_idx, nsd)
    acc = torch.where(wn[idx] > 0, pos - ws[idx], 0)              # tokens of the previous window that were kept
    dec.commit(ws[idx], acc, slots=slots)
    # a window never runs past its chunk's 64th row: the n-gram proposer is cut to that (install); a drafter that is
    # not would get rows the kernels do not compute, so fail loudly (device-side, no host sync)
    torch._assert_async(((nrows <= BT - pos % BT) | (slots < 0)).all())
    L = q.shape[1]
    gather = (qsl[:-1, None] + torch.arange(KT, device=q.device)[None, :]).clamp(max=max(L - 1, 0)).long()
    pick = lambda t: t[gather].contiguous()                          # [nsd, KT, ...]
    folds = _debug_fold_refs(dec, slots, scale) if DEBUG else None
    o = dec.step(pick(q[0]), pick(k[0]), pick(v[0]), pick(g), pick(beta), pos, nrows, scale, slots=slots)
    if DEBUG:
        _debug_check(dec, slots, pos, nrows, gather, q, k, v, g, beta, scale, o, folds)
    ws[idx] = pos
    wn[idx] = nrows
    valid = torch.arange(KT, device=q.device)[None, :] < nrows[:, None]
    dest = torch.where(valid, gather, L).flatten()                  # rows outside a window go to a dummy row L
    out = torch.zeros(L + 1, self.num_v_heads, self.head_v_dim, device=q.device, dtype=o.dtype)
    out.index_copy_(0, dest, o.flatten(0, 1))
    return out[:L]


def _non_spec(self, st, md, x, g, beta, scale, out):
    """Plain decode requests (one token each, every token accepted) and prefill requests."""
    dec, ws, wn = st['one'], st['ws'], st['wn']
    q, k, v = self.rearrange_mixed_qkv(x)
    nd, npf = md.num_decodes, md.num_prefills
    slots = md.non_spec_state_indices_tensor
    if nd > 0 and 'win' in st:                                       # speculative engine, a step without drafts:
        win = st['win']                                              # one-token windows on the window path
        sl = slots[:nd]
        idx = _slot_index(sl, ws)
        pos = _positions(md, None if md.non_spec_req_idx is None else md.non_spec_req_idx[:nd], nd)
        acc = torch.where(wn[idx] > 0, pos - ws[idx], 0)
        win.commit(ws[idx], acc, slots=sl)
        pad = lambda t: torch.cat([t[:nd, None], t[:nd, None].new_zeros(nd, win.KT - 1, *t.shape[1:])], 1).contiguous()
        ones = torch.ones_like(pos)
        out[:nd] = win.step(pad(q[0]), pad(k[0]), pad(v[0]), pad(g), pad(beta), pos, ones, scale, slots=sl)[:, 0]
        ws[idx] = pos
        wn[idx] = ones
    elif nd > 0:                                                     # vLLM orders decode requests first
        sl = slots[:nd]
        idx = _slot_index(sl, ws)
        pos = ws[idx] + wn[idx]
        out[:nd] = dec.decode(q[0, :nd], k[0, :nd], v[0, :nd], g[:nd], beta[:nd], pos, scale, slots=sl)
        ws[idx] = pos
        wn[idx] = torch.ones_like(pos)                               # a device tensor: capturable in a CUDA graph
    if npf > 0:
        _prefill(self, st, md, q, k, v, g, beta, scale, out, nd, npf)


def _prefill(self, st, md, q, k, v, g, beta, scale, out, nd, npf):
    """Prefill requests nd .. nd+npf-1 in one go. Request i adds tokens at absolute positions p0 .. end-1; its slot
    holds the state at b0 = 64 * (p0 // 64) and the rows b0 .. p0-1 (a new request: p0 = 0, and its slot is cleared).
    For the requests that reach a new boundary b1 = 64 * (end // 64), fla's chunk program runs, varlen over them, on
    b0 .. b1 (cached rows in front of the new tokens), giving their rows p0 .. b1-1 and the fp32 state at b1. The rows
    after the last boundary (b1 .. end-1, or p0 .. end-1 for a request within one chunk) are replayed through
    exact_decode in windows of 16 rows, which fills the row caches and gives those rows: decode rows are bitwise fla's
    rows, so every request gets the rows of the single forward over b0 .. end (row-causality and resumability,
    e1_premise). Host work is vectorized over requests, with one device-to-host and one host-to-device copy."""
    import numpy as np
    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd
    from fla.modules.l2norm import l2norm_fwd
    dec, ws, wn = st['one'], st['ws'], st['wn']
    dev = q.device
    W = 16
    slots_d = md.non_spec_state_indices_tensor[nd:nd + npf]
    init_d = md.has_initial_state[nd:nd + npf]
    qsl = md.non_spec_query_start_loc[nd:nd + npf + 1]
    sl_l = slots_d.long()
    if 'win' in st:
        # speculative engine: a one-token chunk of a prompt runs as a one-token window (vLLM counts it as a decode),
        # whose acceptance would be committed at the slot's next window; commit it here (all of it, prompt tokens) and
        # fold the chunk it may have completed before reading the boundary state
        st['win'].commit(ws[sl_l], torch.where(init_d, wn[sl_l], 0), slots=slots_d)
        if DEBUG:
            DEBUG_STATS['prefill_folds'] = DEBUG_STATS.get('prefill_folds', 0) + int((dec.pending[sl_l] * init_d).sum())
        dec.fold(slots_d)
    p0_d = torch.where(init_d, ws[sl_l] + wn[sl_l], 0)
    s0, e0, slot, p0, init = torch.stack([qsl[:-1], qsl[1:], slots_d, p0_d.to(slots_d.dtype),
                                          init_d.to(slots_d.dtype)]).cpu().numpy().astype(np.int64)  # host sync
    end = p0 + e0 - s0
    b0, b1 = p0 - p0 % BT, end - end % BT
    cross = np.nonzero(b1 > b0)[0]                                   # requests reaching a new boundary
    fresh = np.nonzero(init == 0)[0]
    # replay: (slot, first position, first token, rows) of each request's rows after its last boundary
    r_first = np.where(b1 > b0, b1, p0)
    rp = np.nonzero(end > r_first)[0]
    r_slot, r_pos, r_tok, r_n = slot[rp], r_first[rp], s0[rp] + r_first[rp] - p0[rp], end[rp] - r_first[rp]
    nwin = -(-int(r_n.max()) // W) if rp.size else 0
    w0 = np.arange(nwin)[:, None] * W                                # [nwin, 1]
    r_nr = np.clip(r_n[None, :] - w0, 0, W)                          # [nwin, nr] rows per window
    j = np.arange(W)
    r_tix = r_tok[None, :, None] + np.minimum(w0[:, :, None] + j, np.maximum(r_n[None, :, None] - 1, 0))
    r_valid = j[None, None, :] < r_nr[:, :, None]
    # chunk segments: cached rows 0 .. c-1 of the slot, then the new tokens up to b1
    c, m = (p0 - b0)[cross], (b1 - p0)[cross]
    L = c + m
    cu = np.r_[0, np.cumsum(L)]
    seg = np.repeat(np.arange(cross.size), L)
    off = np.arange(cu[-1]) - cu[seg]
    is_new = off >= c[seg]
    tok = s0[cross][seg] + off - c[seg]
    nck = -(-L // BT)
    ci = np.stack([np.repeat(np.arange(cross.size), nck), np.arange(nck.sum()) - np.repeat(np.cumsum(nck) - nck, nck)], 1)
    # one host-to-device copy of every index
    parts = {'fresh': slot[fresh], 'cross': slot[cross], 'cu': cu, 'ci': ci.ravel(), 'is_new': is_new,
             'gi': np.where(is_new, tok, 0), 'cs': np.where(is_new, 0, slot[cross][seg]),
             'cr': np.where(is_new, 0, off), 'oi': np.nonzero(is_new)[0], 'od': tok[is_new],
             'r_slot': r_slot, 'r_pos': r_pos[None, :] + w0, 'r_nr': r_nr, 'r_tix': r_tix,
             'r_src': np.nonzero(r_valid.reshape(nwin, -1))[1] if nwin else np.zeros(0, np.int64),
             'r_dst': r_tix[r_valid], 'r_cnt': r_valid.reshape(nwin, -1).sum(1) if nwin else np.zeros(0, np.int64),
             'end': end}
    flat = torch.from_numpy(np.concatenate([np.asarray(a, np.int64).ravel() for a in parts.values()])).to(dev)
    sizes = [np.asarray(a).size for a in parts.values()]
    d = dict(zip(parts, torch.split(flat, sizes)))
    r_cnt = parts['r_cnt'].tolist()

    if fresh.size:                                                   # a new request starts from the zero state
        dec.S[d['fresh']] = 0
        dec.Hb[d['fresh']] = 0
    if cross.size:
        qn, kn = l2norm_fwd(q[0])[0], l2norm_fwd(k[0])[0]
        new = d['is_new'].bool()
        gi, cs, cr = d['gi'], d['cs'], d['cr']
        sel = lambda newv, cachev: torch.where(new.view(-1, *([1] * (newv.dim() - 1))), newv[gi],
                                               cachev[cs, cr])[None].contiguous()
        Qs = torch.where(new[:, None, None], qn[gi], 0.)[None].contiguous()   # cached rows: outputs unused
        slA = d['cross']
        _, oA, _, SA, _, _ = chunk_gated_delta_rule_fwd(
            Qs, sel(kn, dec.kbuf), sel(v[0], dec.vbuf), sel(g, dec.gbuf), sel(beta, dec.bbuf), scale,
            dec.S[slA].contiguous(), True, cu_seqlens=d['cu'].to(torch.int32),
            chunk_indices=d['ci'].view(-1, 2).to(torch.int32))
        out.index_copy_(0, d['od'], oA[0, d['oi']].to(out.dtype))
        dec.S[slA] = SA
        dec.Hb[slA] = SA.to(dec.Hb.dtype)
    dec.pending[sl_l] = 0
    if nwin:
        rs = d['r_slot'].to(torch.int32)
        pos_w, nr_w, tix_w = d['r_pos'].view(nwin, -1), d['r_nr'].view(nwin, -1), d['r_tix'].view(nwin, rp.size, W)
        src, dst = torch.split(d['r_src'], r_cnt), torch.split(d['r_dst'], r_cnt)
        for w in range(nwin):
            xs = [t[tix_w[w]].contiguous() for t in (q[0], k[0], v[0], g, beta)]
            o = dec._launch(rs, *xs, pos_w[w].to(torch.int32), nr_w[w].to(torch.int32), scale, W)
            out.index_copy_(0, dst[w], o.flatten(0, 1)[src[w]])
    ws[sl_l] = d['end'].to(torch.int32)
    wn[sl_l] = 0


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


def _debug_check(dec, slots, pos, nrows, gather, q, k, v, g, beta, scale, o, folds):
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
        rows = gather[i, :n]
        qn, kn = l2norm_fwd(q[0, rows])[0], l2norm_fwd(k[0, rows])[0]
        cat = lambda old, new: torch.cat([old, new], 0)[None].contiguous()
        ref, _ = chunk_gated_delta_rule(cat(qn.new_zeros(r, *qn.shape[1:]), qn), cat(dec.kbuf[sl, :r], kn),
                                        cat(dec.vbuf[sl, :r], v[0, rows]), cat(dec.gbuf[sl, :r], g[rows]),
                                        cat(dec.bbuf[sl, :r], beta[rows]), scale=scale,
                                        initial_state=dec.S[sl][None].clone(), output_final_state=True,
                                        use_qk_l2norm_in_kernel=False)
        count('windows', not torch.equal(ref[0, r:], o[i, :n]))
