"""E9: Qwen3.5 end to end with the real exact-decode kernel (gdn/exact_decode.py) in the GDN layers.
Row-invariant dense ops (ROWINV stand-in from e2) on both sides; the trainer forward is HF with no cache (fla chunk).
Sampler: prefill P tokens (fla chunk over the prompt, row-causal so its rows are the trainer's), the slot's boundary
state S_b and the partial chunk loaded into GDNExactDecode, then one token at a time through dec.step.
Reports teacher-forced next-token logprob agreement, decode vs trainer forward.
usage: python e9_qwen35_v4.py MODEL_DIR [T] [P]"""
import os, sys
os.environ['ROWINV'] = '1'
src = open('e2_qwen35.py').read()
exec(src[:src.index("print('\\nA core")])                               # setup, ROWINV patches, trainer forward
exec(src.split("def decode_logprobs")[0].split("# ---- C: end-to-end")[1].split("\n", 1)[1].split("class Replay")[0])
exec("def compare" + src.split("def compare")[1].split("print('\\nC e2e")[0])
from exact_decode import GDNExactDecode, BT as BT4

def v4_forward(self, hidden_states, cache_params=None, attention_mask=None, **kw):
    st = self._v4
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
    if n > 1:                                                     # prefill: the trainer's own chunk program
        assert t0 == 0
        core, _ = chunk(q, k, v, g, beta)
        b = BT4 * (n // BT4)
        Sb = chunk(q[:, :b], k[:, :b], v[:, :b], g[:, :b], beta[:, :b])[1][0] if b else \
            torch.zeros(self.num_v_heads, self.head_k_dim, self.head_v_dim, device=q.device)
        if st['dec'] is None:
            st['dec'] = GDNExactDecode(1, q.shape[2], self.num_v_heads, self.head_k_dim, self.head_v_dim, q.device,
                                        BV=128, nw1=2, nw2=4)
        st['dec'].load(0, Sb, b, q[0, b:n], k[0, b:n], v[0, b:n], g[0, b:n], beta[0, b:n])
    else:
        pos = torch.tensor([t0], device=q.device, dtype=torch.int32)
        core = st['dec'].decode(q[:, 0].contiguous(), k[:, 0].contiguous(), v[:, 0].contiguous(), g[:, 0].contiguous(),
                                beta[:, 0].contiguous(), pos, self.head_k_dim ** -0.5)[None]
    st['t'] = t0 + n
    core = self.norm(core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim)).reshape(bsz, n, -1)
    return self.out_proj(core)

def decode_v4():
    for m in gdn: m._v4 = {'x': None, 't': 0, 'dec': None}
    saved, M.Qwen3_5GatedDeltaNet.forward = M.Qwen3_5GatedDeltaNet.forward, v4_forward
    out = []
    with torch.no_grad():
        o = model(ids[:, :P], use_cache=True); pkv = o.past_key_values
        out.append(o.logits[:, -1:].float())
        for t in range(P, T - 1):
            o = model(ids[:, t:t + 1], past_key_values=pkv, use_cache=True); pkv = o.past_key_values
            out.append(o.logits.float())
    M.Qwen3_5GatedDeltaNet.forward = saved
    return torch.cat(out, 1)

r = compare(decode_v4())
print('E9 exact_decode vs trainer forward:', json.dumps({k: float(f"{v:.4g}") for k, v in r.items()}), flush=True)
