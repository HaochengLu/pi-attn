"""Model-level TP-degree invariance: Qwen3 (real weights) on t emulated tensor-parallel ranks, one GPU.

Layout (no cross-rank reduction anywhere): every projection is column-parallel. Rank r holds the rows
of each weight for its output columns and the full K; the layer input is all-gathered first.
  attention: QKV rows of the rank's heads (kv heads split evenly; t > num_kv_heads not used here),
             per-head q/k norm and RoPE, PI-Attn prefill/decode over the rank's heads and its slice of
             the paged KV cache; all-gather of the attention output; o_proj column shard; all-gather.
  MLP:       gate/up column shards, SiLU(gate) * up, all-gather of the intermediate, down column shard,
             all-gather.
  LM head:   vocab column shards, all-gather of the logits, greedy argmax on the full row.
Every GEMM passes S = gemm_segments(N, K, T_MAX) of the whole weight (gemm_inv.cuh) and cycles
through the kernel ids the engine's tuner may pick for that launch's shape (test_gemm.candidates), so
ranks of different widths run different kernels as they would after tuning. An all-gather is emulated
by concatenating the ranks' outputs (a gather copies bits). RMSNorm sums squares with a fixed pairwise
tree over the hidden dimension, so a row's result does not depend on the tensor's shape.

Prompts are random token ids (4 prompts, 37-1029 tokens). Checks: prefill logits of every prompt's
last token and the logits of every decode step, for t = 2, 3, 4, 8, are bitwise equal to t = 1.
Negative controls, Megatron row-parallel o/down at t = 2 (each rank's partial product over K / 2):
partials summed in bf16 as a bf16 all-reduce does, and partials in fp32 summed in fp32 then rounded
(only the summation order differs from t = 1). Both must differ, so the check can see a
reduction-order change.
usage: python test_tp_model.py [model_dir] (default ../models/Qwen3-8B)"""
import json
import math
import os
import sys

import torch
from safetensors import safe_open

import test_gemm as G
from build import build

T_MAX = 8
PAGE = 16
DECODE_STEPS = 8
PROMPTS = [37, 200, 513, 1029]
dev = "cuda"
attn = build()


def load(model_dir):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    w = {}
    for f in sorted(os.listdir(model_dir)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(model_dir, f), "pt", device=dev) as st:
                for k in st.keys():
                    w[k] = st.get_tensor(k)
    if "lm_head.weight" not in w:
        w["lm_head.weight"] = w["model.embed_tokens.weight"]
    return cfg, w


def tree_sum(x):
    """Sum over the last dimension (a power of two) by halving: one fixed order for every row."""
    assert x.shape[-1] & (x.shape[-1] - 1) == 0, "tree_sum needs a power-of-two dimension"
    while x.shape[-1] > 1:
        h = x.shape[-1] // 2
        x = x[..., :h] + x[..., h:]
    return x


def rms_norm(x, weight, eps):
    xf = x.float()
    r = torch.rsqrt(tree_sum(xf * xf) / x.shape[-1] + eps)
    return (xf * r * weight.float()).bfloat16()


def rope(x, pos, theta):
    """x [T, H, D] bf16, pos [T]: rotate-half RoPE in fp32, rounded to bf16."""
    d = x.shape[-1]
    inv = 1.0 / theta ** (torch.arange(0, d, 2, device=dev, dtype=torch.float64) / d)
    ang = (pos.double()[:, None] * inv[None]).float()
    cos, sin = torch.cat([ang.cos()] * 2, -1)[:, None], torch.cat([ang.sin()] * 2, -1)[:, None]
    xf = x.float()
    rot = torch.cat([-xf[..., d // 2:], xf[..., : d // 2]], -1)
    return (xf * cos + rot * sin).bfloat16()


def cuts(n, t, unit):
    c = [round(i * n / t / unit) * unit for i in range(t + 1)]
    return list(zip(c[:-1], c[1:]))


class Model:
    def __init__(self, cfg, w):
        self.cfg, self.w = cfg, w
        self.H, self.HQ, self.HKV, self.D = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        self.L, self.eps, self.theta = cfg["num_hidden_layers"], cfg["rms_norm_eps"], cfg["rope_theta"]
        self.S = {}
        self.calls = 0

    def linear(self, x, name, rows=None):
        """x @ W[rows]^T with the whole weight's canonical S (rows: an index tensor or a (start, end) range)."""
        W = self.w[name]
        if name not in self.S:
            self.S[name] = G.gemm_segments(W.shape[0], W.shape[1], T_MAX)
        if rows is not None:
            W = W[rows[0]:rows[1]] if isinstance(rows, tuple) else W.index_select(0, rows)
        ids = G.candidates(x.shape[0], W.shape[0], W.shape[1], self.S[name])
        self.calls += 1
        return G.ext.linear(x.contiguous(), W.contiguous(), ids[self.calls % len(ids)], False, self.S[name])

    def heads(self, t):
        """Per rank: (q heads range, kv heads range)."""
        g = self.HQ // self.HKV
        return [((a * g, b * g), (a, b)) for a, b in cuts(self.HKV, t, 1)]

    def new_cache(self, pages):
        return [(torch.zeros(pages, PAGE, self.HKV, self.D, device=dev, dtype=torch.bfloat16),
                 torch.zeros(pages, PAGE, self.HKV, self.D, device=dev, dtype=torch.bfloat16)) for _ in range(self.L)]

    def forward(self, ids, pos, seqs, cache, t, decode, row_parallel=False):
        """ids/pos [T]; seqs: per request (q_len, kv_len, page list). Returns logits [T, V]."""
        T, D = ids.shape[0], self.D
        x = self.w["model.embed_tokens.weight"][ids]
        qo, kvp, kvi, kvl = [0], [0], [], []
        for q_len, kv_len, pages in seqs:
            qo.append(qo[-1] + q_len)
            kvi += pages
            kvp.append(len(kvi))
            kvl.append(kv_len)
        i32 = lambda a: torch.tensor(a, device=dev, dtype=torch.int32)
        qo, kvp, kvi, kvl = i32(qo), i32(kvp), i32(kvi), i32(kvl)
        slot = torch.cat([torch.tensor([pages[p // PAGE] * PAGE + p % PAGE for p in range(kv_len - q_len, kv_len)])
                          for q_len, kv_len, pages in seqs]).to(dev)
        shards = self.heads(t)
        for l in range(self.L):
            p = f"model.layers.{l}."
            h = rms_norm(x, self.w[p + "input_layernorm.weight"], self.eps)
            kc, vc = cache[l]
            outs = []
            for (qa, qb), (ka, kb) in shards:
                rq = torch.arange(qa * D, qb * D, device=dev)
                q = self.linear(h, p + "self_attn.q_proj.weight", rq).view(T, qb - qa, D)
                k = self.linear(h, p + "self_attn.k_proj.weight", (ka * D, kb * D)).view(T, kb - ka, D)
                v = self.linear(h, p + "self_attn.v_proj.weight", (ka * D, kb * D)).view(T, kb - ka, D)
                q = rope(rms_norm(q, self.w[p + "self_attn.q_norm.weight"], self.eps), pos, self.theta)
                k = rope(rms_norm(k, self.w[p + "self_attn.k_norm.weight"], self.eps), pos, self.theta)
                kc.view(-1, self.HKV, D)[slot, ka:kb] = k
                vc.view(-1, self.HKV, D)[slot, ka:kb] = v
                ks, vs = kc[:, :, ka:kb].contiguous(), vc[:, :, ka:kb].contiguous()
                if decode:
                    rows = len(seqs) * -(-int(kvl.max()) // attn.GROUP_KEYS)
                    ws_o = torch.empty(rows, qb - qa, D, device=dev)
                    ws_ml = torch.empty(rows, qb - qa, 2 * attn.DECODE_WARPS, device=dev)
                    cnt = torch.zeros(rows * (qb - qa) * attn.DECODE_WARPS, device=dev, dtype=torch.int32)
                    o = attn.decode(q.contiguous(), ks, vs, kvp, kvi, kvl, ws_o, ws_ml, cnt, int(kvl.max()), 1 / math.sqrt(D))
                else:
                    o = attn.prefill(q.contiguous(), ks, vs, qo, kvp, kvi, kvl, max(s[0] for s in seqs),
                                     max(s[1] for s in seqs), 1 / math.sqrt(D), 0)
                outs.append(o.reshape(T, -1))
            a = torch.cat(outs, 1)  # all-gather of the attention output (heads in rank order = global order)
            x = x + self.project(a, p + "self_attn.o_proj.weight", t, row_parallel)
            h = rms_norm(x, self.w[p + "post_attention_layernorm.weight"], self.eps)
            I = self.w[p + "mlp.gate_proj.weight"].shape[0]
            mid = []
            for a_, b_ in cuts(I, t, 32):
                g_ = self.linear(h, p + "mlp.gate_proj.weight", (a_, b_)).float()
                u_ = self.linear(h, p + "mlp.up_proj.weight", (a_, b_)).float()
                mid.append((g_ * torch.sigmoid(g_) * u_).bfloat16())
            x = x + self.project(torch.cat(mid, 1), p + "mlp.down_proj.weight", t, row_parallel)
        h = rms_norm(x, self.w["model.norm.weight"], self.eps)
        V = self.w["lm_head.weight"].shape[0]
        return torch.cat([self.linear(h, "lm_head.weight", r) for r in cuts(V, t, 32)], 1)

    def project(self, x, name, t, row_parallel):
        """Row-parallel-in-Megatron projection. Column layout: shards of the output columns, all-gathered.
        row_parallel (negative control): K split over t ranks, partials summed in rank order, in bf16
        ("bf16") or in fp32 and rounded once ("fp32")."""
        N, K = self.w[name].shape
        if not row_parallel:
            return torch.cat([self.linear(x, name, r) for r in cuts(N, t, 32)], 1)
        W, y = self.w[name], None
        fp32 = row_parallel == "fp32"
        for a, b in cuts(K, t, 64):
            part = G.ext.linear(x[:, a:b].contiguous(), W[:, a:b].contiguous(), 0, fp32, 1)
            y = part if y is None else y + part
        return y.bfloat16()


def run(m, t, row_parallel=False):
    """Prefill the prompts as one batch, then DECODE_STEPS greedy steps for all of them."""
    torch.manual_seed(0)
    prompts = [torch.randint(0, 150000, (n,), device=dev) for n in PROMPTS]
    page_ids, nxt = [], 1  # page 0 unused
    for n in PROMPTS:
        k = -(-(n + DECODE_STEPS) // PAGE)
        page_ids.append(list(range(nxt, nxt + k)))
        nxt += k
    cache = m.new_cache(nxt)
    ids = torch.cat(prompts)
    pos = torch.cat([torch.arange(n, device=dev) for n in PROMPTS])
    seqs = [(n, n, pg) for n, pg in zip(PROMPTS, page_ids)]
    logits = m.forward(ids, pos, seqs, cache, t, decode=False, row_parallel=row_parallel)
    last = torch.tensor([sum(PROMPTS[:i + 1]) - 1 for i in range(len(PROMPTS))], device=dev)
    out = [logits[last]]
    tok = out[-1].float().argmax(-1)
    lens = list(PROMPTS)
    for _ in range(DECODE_STEPS):
        pos = torch.tensor(lens, device=dev)
        seqs = [(1, n + 1, pg) for n, pg in zip(lens, page_ids)]
        out.append(m.forward(tok, pos, seqs, cache, t, decode=True, row_parallel=row_parallel))
        tok = out[-1].float().argmax(-1)
        lens = [n + 1 for n in lens]
    return out


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "Qwen3-8B")
    cfg, w = load(model_dir)
    m = Model(cfg, w)
    ref = run(m, 1)
    toks = torch.stack([o.float().argmax(-1) for o in ref])
    print(f"{os.path.basename(os.path.normpath(model_dir))} t_max={T_MAX}: S per weight "
          f"{sorted(set((k.split('.')[-2], s) for k, s in m.S.items()))}")
    print("greedy tokens (t=1), one column per prompt:", toks.T.tolist())
    fails = 0
    for t in [2, 3, 4, 8]:
        got = run(m, t)
        bad = [i for i, (a, b) in enumerate(zip(ref, got)) if not torch.equal(a.view(torch.int16), b.view(torch.int16))]
        print(f"t={t}: {len(ref)} logit sets ({len(PROMPTS)} prompts: prefill + {DECODE_STEPS} decode steps), "
              f"bitwise different: {bad}", flush=True)
        fails += len(bad)
    for mode in ("bf16", "fp32"):
        ctl = run(m, 2, row_parallel=mode)
        diff = [(a.view(torch.int16) != b.view(torch.int16)).float().mean().item() for a, b in zip(ref, ctl)]
        print(f"negative control, Megatron row-parallel o/down at t=2, {mode} partials: fraction of logits "
              f"that differ per step {[f'{d:.3f}' for d in diff]}")
        if all(d == 0 for d in diff):
            print("negative control did not differ: the check cannot see reduction-order changes")
            fails += 1
    print("TP INVARIANCE FAILURES:", fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
