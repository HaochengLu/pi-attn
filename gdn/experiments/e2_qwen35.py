"""E2: on real Qwen3.5 activations, how far is today's GDN decode from the trainer's forward, and does
replaying the trainer's chunk program on the absolute 64-token grid close the gap bitwise?

The trainer forward is HF Qwen3_5 with no cache (fla chunk_gated_delta_rule + torch causal conv). We record
each GDN layer's inputs during that forward, then compare, per layer:
  A core : fla fused_recurrent token-by-token after a P-token chunk prefill (what HF/vLLM-style decode runs)
           vs. replay: chunk(x[b:t+1] | S_b) with b = 64*floor(t/64), S_b from chunk(x[:b])
  B conv : causal_conv1d_update token-by-token vs. the full causal conv, and full-conv prefix truncation
  C e2e  : teacher-forced logprob of the next token, full forward vs. prefill P + cached decode, with the
           default GDN decode and with GDN decode replaced by replay (conv over the full history + chunk replay).
           What remains in the replay variant is the non-GDN mismatch (GEMM row count, full attention).
With ROWINV=1, every Linear, RMSNorm reduction and the full-attention layers run per row with the row's own shapes (a stand-in
for PI-Attn's row-invariant GEMM/attention), on both the trainer forward and decode, so C isolates GDN.
usage: [ROWINV=1] python e2_qwen35.py MODEL_DIR [T] [P]"""
import os, sys, json, torch, torch.nn.functional as F
import transformers.models.qwen3_5.modeling_qwen3_5 as M
from transformers import AutoModelForCausalLM, AutoTokenizer
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

path = sys.argv[1]
T = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
P = int(sys.argv[3]) if len(sys.argv) > 3 else 300
C = 64
dev = 'cuda'
torch.manual_seed(0)

ROWINV = os.environ.get('ROWINV') == '1'
if ROWINV:
    def rowinv_linear(self, x):
        rows = x.reshape(-1, x.shape[-1])
        out = torch.cat([F.linear(rows[i:i + 1], self.weight, self.bias) for i in range(rows.shape[0])])
        return out.reshape(*x.shape[:-1], -1)
    torch.nn.Linear.forward = rowinv_linear

    def rowinv_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        """query [B,H,q,D], key/value [B,Hkv,kv,D]; row i sits at absolute position kv-q+i, attends to keys [0, pos]."""
        key, value = M.repeat_kv(key, module.num_key_value_groups), M.repeat_kv(value, module.num_key_value_groups)
        nq, nk = query.shape[2], key.shape[2]
        outs = []
        for i in range(nq):
            p = nk - nq + i
            w = torch.matmul(query[:, :, i:i + 1], key[:, :, :p + 1].transpose(2, 3)) * scaling
            w = F.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
            outs.append(torch.matmul(w, value[:, :, :p + 1]))
        return torch.cat(outs, 2).transpose(1, 2).contiguous(), None
    def rowinv_mean(x):
        """Mean over the last dim by an elementwise halving tree (zero-padded to a power of two): the order is
        fixed by the width alone, never by how many rows are in the tensor (torch's mean kernel splits by shape)."""
        n = x.shape[-1]
        w = 1 << (n - 1).bit_length()
        x = F.pad(x, (0, w - n))
        while x.shape[-1] > 1:
            h = x.shape[-1] // 2
            x = x[..., :h] + x[..., h:]
        return x / n
    M.Qwen3_5RMSNorm._norm = lambda self, x: x * torch.rsqrt(rowinv_mean(x.pow(2)) + self.eps)
    def rowinv_gated_norm(self, hidden_states, gate):
        dt = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        hidden_states = hidden_states * torch.rsqrt(rowinv_mean(hidden_states.pow(2)) + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(dt)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(dt)
    M.Qwen3_5RMSNormGated.forward = rowinv_gated_norm
    from transformers import AttentionInterface
    AttentionInterface.register('rowinv', rowinv_attention)
print('ROWINV', ROWINV)
print('bound chunk impl:', getattr(M.torch_chunk_gated_delta_rule, '__wrapped__', None), flush=True)
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
                                             attn_implementation='rowinv' if ROWINV else None).to(dev).eval()
text = open(f'{path}/README.md').read()
ids = tok(text, return_tensors='pt').input_ids[:, :T].to(dev)
assert ids.shape[1] == T, ids.shape
gdn = [m for m in model.modules() if isinstance(m, M.Qwen3_5GatedDeltaNet)]
print(f'model {path.rstrip("/").split("/")[-1]}  T={T} P={P}  GDN layers={len(gdn)}', flush=True)

def chunk(q, k, v, g, beta, h0=None):
    return chunk_gated_delta_rule(q, k, v, g, beta, initial_state=h0, output_final_state=True,
                                  use_qk_l2norm_in_kernel=True)

# ---- record the trainer forward's GDN core and conv inputs -----------------------------------------------
rec_core, rec_conv = [], []
orig_chunk, orig_conv = M.torch_chunk_gated_delta_rule, M.causal_conv1d_fn
def rec_chunk_fn(q, k, v, g=None, beta=None, **kw):
    rec_core.append((q, k, v, g, beta))
    return orig_chunk(q, k, v, g=g, beta=beta, **kw)
def rec_conv_fn(x, w, b=None, activation=None, **kw):
    rec_conv.append((x, w, b, activation))
    return orig_conv(x, w, b, activation=activation, **kw)
M.torch_chunk_gated_delta_rule, M.causal_conv1d_fn = rec_chunk_fn, rec_conv_fn
with torch.no_grad():
    logits_train = model(ids).logits.float()
M.torch_chunk_gated_delta_rule, M.causal_conv1d_fn = orig_chunk, orig_conv
assert len(rec_core) == len(gdn) == len(rec_conv), (len(rec_core), len(rec_conv))

def rowstats(o, ref):
    """o, ref: [T', H, V]. Bitwise-equal row fraction and relative error over rows."""
    d = (o.float() - ref.float())
    eq = (d == 0).flatten(1).all(1)
    rel = d.flatten(1).norm(dim=1) / ref.float().flatten(1).norm(dim=1).clamp_min(1e-30)
    return eq.float().mean().item(), rel.mean().item(), rel.max().item()

# ---- A: GDN core ------------------------------------------------------------------------------------------
print('\nA core  layer | recurrent decode: eq-rows  mean-rel  max-rel | replay: eq-rows', flush=True)
A = []
with torch.no_grad():
    for li, (q, k, v, g, beta) in enumerate(rec_core):
        o_ref, _ = chunk(q, k, v, g, beta)
        _, s = chunk(q[:, :P], k[:, :P], v[:, :P], g[:, :P], beta[:, :P])
        o_rec = []
        for t in range(P, T):
            o, s = fused_recurrent_gated_delta_rule(q=q[:, t:t + 1], k=k[:, t:t + 1], v=v[:, t:t + 1],
                                                    g=g[:, t:t + 1], beta=beta[:, t:t + 1], initial_state=s,
                                                    output_final_state=True, use_qk_l2norm_in_kernel=True)
            o_rec.append(o)
        o_rec = torch.cat(o_rec, 1)
        o_rep, states = [], {}
        for t in range(P, T):
            b = C * (t // C)
            if b not in states:
                states[b] = chunk(q[:, :b], k[:, :b], v[:, :b], g[:, :b], beta[:, :b])[1] if b else None
            o, _ = chunk(q[:, b:t + 1], k[:, b:t + 1], v[:, b:t + 1], g[:, b:t + 1], beta[:, b:t + 1], states[b])
            o_rep.append(o[:, -1:])
        o_rep = torch.cat(o_rep, 1)
        r = rowstats(o_rec[0], o_ref[0, P:]) + (rowstats(o_rep[0], o_ref[0, P:])[0],)
        A.append(r)
        print(f'        {li:5d} | {r[0]*100:6.1f}%  {r[1]:.2e}  {r[2]:.2e} | {r[3]*100:6.1f}%', flush=True)

# ---- B: causal conv ---------------------------------------------------------------------------------------
print('\nB conv  layer | update decode: eq-rows  max|d| | full-conv prefix truncation eq-rows', flush=True)
Bs = []
with torch.no_grad():
    for li, (x, w, b, act) in enumerate(rec_conv):
        y_ref = orig_conv(x, w, b, activation=act)                         # [1, D, T]
        Kw = w.shape[-1]
        state = F.pad(x[:, :, :P], (Kw, 0))[:, :, -Kw:].clone()             # HF conv state holds the last K inputs
        ys = [M.causal_conv1d_update(x[:, :, t:t + 1], state, w, b, act) for t in range(P, T)]
        y_upd = torch.cat(ys, -1)
        eq_upd = (y_upd == y_ref[:, :, P:]).all(1).float().mean().item()
        md = (y_upd.float() - y_ref[:, :, P:].float()).abs().max().item()
        eq_pre = sum(torch.equal(orig_conv(x[:, :, :L], w, b, activation=act), y_ref[:, :, :L])
                     for L in range(P, T, 37)) / len(range(P, T, 37))
        Bs.append((eq_upd, md, eq_pre))
        print(f'        {li:5d} | {eq_upd*100:6.1f}%  {md:.2e} | {eq_pre*100:6.1f}%', flush=True)

# ---- C: end-to-end ----------------------------------------------------------------------------------------
class Replay:
    """GDN decode that re-executes the trainer program: conv over the full raw history (rows of the full conv),
    core via chunk replay from the absolute 64-token boundary state. Experiment-only (O(T) conv per step)."""
    def __init__(self, m):
        self.m, self.x, self.qkvgb, self.states = m, None, None, {}

def replay_forward(self, hidden_states, cache_params=None, attention_mask=None, **kw):
    r = self._replay
    bsz, n, _ = hidden_states.shape
    x = self.in_proj_qkv(hidden_states).transpose(1, 2)
    z = self.in_proj_z(hidden_states).reshape(bsz, n, -1, self.head_v_dim)
    bb, a = self.in_proj_b(hidden_states), self.in_proj_a(hidden_states)
    r.x = x if r.x is None else torch.cat([r.x, x], -1)
    t1 = r.x.shape[-1]
    y = orig_conv(r.x, self.conv1d.weight.squeeze(1), self.conv1d.bias, activation=self.activation)[:, :, -n:]
    q, k, v = torch.split(y.transpose(1, 2), [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    q = q.reshape(bsz, n, -1, self.head_k_dim); k = k.reshape(bsz, n, -1, self.head_k_dim)
    v = v.reshape(bsz, n, -1, self.head_v_dim)
    beta = bb.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    rep = self.num_v_heads // self.num_k_heads
    if rep > 1:
        q, k = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)
    new = (q, k, v, g, beta)
    r.qkvgb = new if r.qkvgb is None else tuple(torch.cat([o, nn_], 1) for o, nn_ in zip(r.qkvgb, new))
    t0 = t1 - n
    outs = []
    # rows t0..t1-1: each block of rows sharing a boundary b is one chunk call from S_b (prefill = one call per block)
    t = t0
    while t < t1:
        b = C * (t // C)
        e = min(t1, b + C)
        if b not in r.states:
            r.states[b] = chunk(*(u[:, :b] for u in r.qkvgb))[1] if b else None
        o, _ = chunk(*(u[:, b:e] for u in r.qkvgb), r.states[b])
        outs.append(o[:, t - b:])
        t = e
    core = torch.cat(outs, 1)
    core = self.norm(core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim)).reshape(bsz, n, -1)
    return self.out_proj(core)

def decode_logprobs(replay):
    if replay:
        for m in gdn: m._replay = Replay(m)
        M.Qwen3_5GatedDeltaNet.forward, saved = replay_forward, M.Qwen3_5GatedDeltaNet.forward
    out = []
    with torch.no_grad():
        o = model(ids[:, :P], use_cache=True)
        pkv = o.past_key_values
        out.append(o.logits[:, -1:].float())
        for t in range(P, T - 1):
            o = model(ids[:, t:t + 1], past_key_values=pkv, use_cache=True)
            pkv = o.past_key_values
            out.append(o.logits.float())
    if replay:
        M.Qwen3_5GatedDeltaNet.forward = saved
    return torch.cat(out, 1)                                                # logits for positions P-1 .. T-2

def compare(lg):
    ref = logits_train[:, P - 1:T - 1]
    tgt = ids[:, P:T]
    lp, lr = lg.log_softmax(-1), ref.log_softmax(-1)
    dlp = (lp.gather(-1, tgt[..., None]) - lr.gather(-1, tgt[..., None])).abs()
    kl = (lr.exp() * (lr - lp)).sum(-1)
    flips = (lg.argmax(-1) != ref.argmax(-1)).float().mean().item()
    eq = (lg == ref).all(-1).float().mean().item()
    return dict(eq_rows=eq, mean_abs_dlogp=dlp.mean().item(), max_abs_dlogp=dlp.max().item(),
                mean_kl=kl.mean().item(), argmax_flip=flips)

print('\nC e2e (teacher-forced next-token logprob, decode vs trainer forward)', flush=True)
res = {}
for name, rp in [('default', False), ('gdn_replay', True)]:
    res[name] = compare(decode_logprobs(rp))
    print(f'  {name:11s}', json.dumps({k: float(f"{v:.4g}") for k, v in res[name].items()}), flush=True)
