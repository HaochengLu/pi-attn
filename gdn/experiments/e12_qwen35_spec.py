"""E12: speculative decoding that is invisible, end to end on Qwen3.5.

Greedy generation three ways, all with row-invariant dense ops (ROWINV stand-in from e2) and exact_decode in every
GDN layer:
  plain   one token per step (exact_decode.decode)
  spec    prompt-lookup drafting (the continuation of the last earlier occurrence of the current 2-gram, up to D
          tokens), one verification forward per step over [current token, drafts] (exact_decode.step windows of up
          to 1 + D), greedy acceptance plus the bonus token; rejected drafts are rolled back by exact_decode.commit,
          cropping the attention KV cache and the conv history
  trainer one teacher-forced forward over prompt + generated tokens (HF, no cache: fla's chunk program)
Reported: whether the token sequences and every generated token's fp32 log-probability are bitwise equal across the
three, the acceptance statistics, and the number of target forwards.
usage: python e12_qwen35_spec.py MODEL_DIR [P] [NEW] [D]"""
import os, sys
os.environ['ROWINV'] = '1'
model_dir = sys.argv[1]
P = int(sys.argv[2]) if len(sys.argv) > 2 else 200
NEW = int(sys.argv[3]) if len(sys.argv) > 3 else 300
D = int(sys.argv[4]) if len(sys.argv) > 4 else 4
sys.argv = [sys.argv[0], model_dir, str(P + NEW), str(P)]
src = open('e2_qwen35.py').read()
exec(src[:src.index("# ---- record the trainer forward's GDN core")])   # setup, ROWINV patches, model, ids
from exact_decode import GDNExactDecode, BT as BT4
orig_conv = M.causal_conv1d_fn

full_attn = [i for i, t in enumerate(model.config.text_config.layer_types if hasattr(model.config, 'text_config')
                                      else model.config.layer_types) if t == 'full_attention']


def gdn_forward(self, hidden_states, cache_params=None, attention_mask=None, **kw):
    """Qwen3_5GatedDeltaNet.forward with the core replaced: prefill = fla's chunk program over the prompt; afterwards
    every call is a window of n tokens at position st['t'] through exact_decode (KT = 1 + D)."""
    st = self._ex
    bsz, n, _ = hidden_states.shape
    x = self.in_proj_qkv(hidden_states).transpose(1, 2)
    z = self.in_proj_z(hidden_states).reshape(bsz, n, -1, self.head_v_dim)
    bb, a = self.in_proj_b(hidden_states), self.in_proj_a(hidden_states)
    st['x'] = x if st['x'] is None else torch.cat([st['x'], x], -1)
    y = orig_conv(st['x'], self.conv1d.weight.squeeze(1), self.conv1d.bias, activation=self.activation)[:, :, -n:]
    q, k, v = torch.split(y.transpose(1, 2), [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    q = q.reshape(bsz, n, -1, self.head_k_dim); k = k.reshape(bsz, n, -1, self.head_k_dim)
    v = v.reshape(bsz, n, -1, self.head_v_dim)
    beta = bb.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    rep = self.num_v_heads // self.num_k_heads
    if rep > 1:
        q, k = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)
    t0 = st['t']
    if t0 == 0:                                                   # prefill: the trainer's own chunk program
        core, _ = chunk(q, k, v, g, beta)
        b = BT4 * (n // BT4)
        Sb = chunk(q[:, :b], k[:, :b], v[:, :b], g[:, :b], beta[:, :b])[1][0] if b else \
            torch.zeros(self.num_v_heads, self.head_k_dim, self.head_v_dim, device=q.device)
        st['dec'] = GDNExactDecode(1, q.shape[2], self.num_v_heads, self.head_k_dim, self.head_v_dim, q.device,
                                   KT=st['KT'], BV=128)
        st['dec'].load(0, Sb, b, q[0, b:n], k[0, b:n], v[0, b:n], g[0, b:n], beta[0, b:n])
    else:
        KT = st['KT']
        pad = lambda u: torch.cat([u, u.new_zeros(u.shape[0], KT - n, *u.shape[2:])], 1).contiguous()
        pos = torch.tensor([t0], device=q.device, dtype=torch.int32)
        nrows = torch.tensor([n], device=q.device, dtype=torch.int32)
        core = st['dec'].step(pad(q), pad(k), pad(v), pad(g), pad(beta), pos, nrows, self.head_k_dim ** -0.5)[:, :n]
    st['t'] = t0 + n
    core = self.norm(core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim)).reshape(bsz, n, -1)
    return self.out_proj(core)


def generate(spec):
    """Greedy generation of NEW tokens after the P-token prompt. Returns tokens, per-token fp32 log-probs, forwards."""
    KT = 1 + D if spec else 2                     # plain decode uses windows of one token (KT >= 2 selects step())
    for m in gdn:
        m._ex = {'x': None, 't': 0, 'dec': None, 'KT': KT}
    saved, M.Qwen3_5GatedDeltaNet.forward = M.Qwen3_5GatedDeltaNet.forward, gdn_forward
    seq = ids[0, :P].tolist()
    toks, lps, forwards, accepted_drafts, proposed = [], [], 0, 0, 0
    with torch.no_grad():
        o = model(ids[:, :P], use_cache=True); pkv = o.past_key_values; forwards += 1
        lp = o.logits[0, -1].float().log_softmax(-1)
        cur = int(lp.argmax()); toks.append(cur); lps.append(float(lp[cur])); seq.append(cur)
        while len(toks) < NEW:
            t = len(seq) - 1                          # position of `cur`, not yet fed
            drafts = []
            if spec:
                for j in range(len(seq) - 3, -1, -1):
                    if seq[j] == seq[-2] and seq[j + 1] == seq[-1]:
                        drafts = seq[j + 2:j + 2 + D]
                        break
            room = min(BT4 - t % BT4, NEW - len(toks))   # never past the chunk's 64th row, never past NEW
            drafts = drafts[:max(0, room - 1)]
            win = [cur] + drafts
            o = model(torch.tensor([win], device=dev), past_key_values=pkv, use_cache=True); forwards += 1
            pkv = o.past_key_values
            L = o.logits[0].float().log_softmax(-1)
            a = 1
            while a < len(win) and int(L[a - 1].argmax()) == win[a]:
                a += 1                                  # drafts win[1..a-1] accepted
            proposed += len(drafts); accepted_drafts += a - 1
            for i in range(1, a):
                toks.append(win[i]); lps.append(float(L[i - 1][win[i]])); seq.append(win[i])
            if len(toks) < NEW:
                cur = int(L[a - 1].argmax()); toks.append(cur); lps.append(float(L[a - 1][cur])); seq.append(cur)
            for m in gdn:                               # roll back what was not accepted
                st = m._ex
                st['dec'].commit(torch.tensor([t], device=dev, dtype=torch.int32),
                                 torch.tensor([a], device=dev, dtype=torch.int32))
                st['t'] = t + a
                st['x'] = st['x'][..., :t + a]
            if a < len(win):
                for li in full_attn:
                    pkv.layers[li].crop(-(len(win) - a))
    M.Qwen3_5GatedDeltaNet.forward = saved
    return toks[:NEW], lps[:NEW], forwards, accepted_drafts, proposed


def trainer(toks):
    with torch.no_grad():
        full = torch.cat([ids[:, :P], torch.tensor([toks], device=dev)], 1)
        L = model(full).logits[0, P - 1:-1].float().log_softmax(-1)
    return [int(L[i].argmax()) for i in range(len(toks))], [float(L[i][toks[i]]) for i in range(len(toks))]


tp, lp_p, fw_p, _, _ = generate(spec=False)
ts, lp_s, fw_s, acc, prop = generate(spec=True)
tt_greedy, lp_t = trainer(tp)
same_tok = tp == ts
same_lp_ps = lp_p == lp_s
same_lp_pt = lp_p == lp_t and tt_greedy == tp
print(f'model {model_dir.rstrip("/").split("/")[-1]}  prompt {P}  new {NEW}  drafts <= {D}')
print(f'tokens plain == spec: {same_tok}   log-probs plain == spec (bitwise): {same_lp_ps}   '
      f'plain == trainer forward (tokens and log-probs, bitwise): {same_lp_pt}')
print(f'target forwards: plain {fw_p}, spec {fw_s}; drafts accepted {acc}/{prop}; '
      f'tokens per spec forward {NEW / fw_s:.2f}')
if not (same_tok and same_lp_ps and same_lp_pt):
    first = next((i for i in range(NEW) if tp[i] != ts[i] or lp_p[i] != lp_s[i] or lp_p[i] != lp_t[i]), None)
    print('first difference at generated token', first)
