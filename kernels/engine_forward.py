"""The serving engine's dense Qwen3 forward in PyTorch, as a trainer runs it over packed sequences.

Every operation is the engine's own kernel, in the engine's order (llm_infer_rust/src/models/qwen3/model.rs):
embedding gather; per layer RMSNorm / fused add-RMSNorm, packed QKV projection, per-head q/k RMSNorm,
half-split RoPE from the engine's fp32 cos/sin table, PI-Attn causal prefill over a paged KV cache,
o projection, fused add-RMSNorm, packed gate/up projection, SiLU-and-mul, down projection; final fused
add-RMSNorm and the LM head. Projections are the PI GEMM (kernels/gemm_torch.cu) with the weight's
canonical segment count; the row-wise layer kernels are the engine's fused_ops.cu (engine_ops_torch.cu).
Since each of these computes a row's bits independently of the other rows, the packing of sequences into
a micro-batch, and prefill vs the engine's decode, do not change a token's logits or log-probability.
Forward only: a trainer would back-propagate through its own differentiable copy."""
import json
import math
import os

import torch
from safetensors import safe_open
from torch.utils.cpp_extension import load

import test_gemm as G
from build import build

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(HERE, "..", "llm_infer_rust")
PAGE = 16
dev = "cuda"
attn = build()


def build_engine_ops():
    major, minor = torch.cuda.get_device_capability()
    fi = os.path.join(os.path.dirname(__import__("flashinfer").__file__), "data")
    return load(
        name=f"engine_ops_sm{major}{minor}",
        sources=[os.path.join(HERE, "engine_ops_torch.cu"), os.path.join(ENGINE, "src/layers/fused_ops.cu"),
                 os.path.join(ENGINE, "src/layers/fused_ops_bridge.cpp")],
        extra_include_paths=[HERE, os.path.join(ENGINE, "third_party/sglang"), os.path.join(ENGINE, "third_party/sglang/include"),
                             os.path.join(fi, "include"), os.path.join(fi, "cccl/libcudacxx/include"),
                             os.path.join(fi, "cccl/cub"), os.path.join(fi, "cccl/thrust")],
        # fused_ops.cu as build.rs compiles it (without torch's half / bf16 operator restrictions)
        extra_cuda_cflags=["-O3", "-std=c++20", "--expt-relaxed-constexpr", "-UNDEBUG", f"-DSGL_CUDA_ARCH={major}{minor}0",
                           "-DSGLANG_USE_FLASHINFER_NORM", f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}",
                           "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
                           "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_OPERATORS__"],
        extra_cflags=["-O3", "-std=c++20"],
        build_directory=os.environ.get("PI_ATTN_BUILD_DIR"))


ops = build_engine_ops()


def linear(x, w):
    # Kernel id 0 (= 13) rather than the engine's tuned id: every id gives the same bits.
    return G.ext.linear(x.contiguous(), w, 0, False)


class EngineQwen3:
    def __init__(self, model_dir):
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        w = {}
        for f in sorted(os.listdir(model_dir)):
            if f.endswith(".safetensors"):
                with safe_open(os.path.join(model_dir, f), "pt", device=dev) as st:
                    for k in st.keys():
                        w[k] = st.get_tensor(k)
        self.cfg, self.eps = cfg, cfg["rms_norm_eps"]
        self.hq, self.hkv, self.hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        self.embed = w["model.embed_tokens.weight"]
        self.lm_head = w.get("lm_head.weight", self.embed)
        self.norm = w["model.norm.weight"]
        self.layers = []
        for i in range(cfg["num_hidden_layers"]):
            p = f"model.layers.{i}."
            a = p + "self_attn."
            self.layers.append(dict(
                ln1=w[p + "input_layernorm.weight"], ln2=w[p + "post_attention_layernorm.weight"],
                qkv=torch.cat([w[a + "q_proj.weight"], w[a + "k_proj.weight"], w[a + "v_proj.weight"]]).contiguous(),
                o=w[a + "o_proj.weight"], q_norm=w[a + "q_norm.weight"], k_norm=w[a + "k_norm.weight"],
                gate_up=torch.cat([w[p + "mlp.gate_proj.weight"], w[p + "mlp.up_proj.weight"]]).contiguous(),
                down=w[p + "mlp.down_proj.weight"]))
        # HalfSplitRope::new on CUDA: fp32 frequencies, fp32 cos/sin cache.
        inv_freq = (torch.arange(0, self.hd, 2, dtype=torch.float32, device=dev)
                    * (-(math.log(cfg["rope_theta"]) / self.hd))).exp()
        freqs = torch.arange(cfg["max_position_embeddings"], dtype=torch.float32, device=dev).unsqueeze(-1) * inv_freq.unsqueeze(0)
        self.cos, self.sin = freqs.cos(), freqs.sin()

    @torch.no_grad()
    def logits(self, seqs):
        """seqs: token-id lists, packed into one micro-batch. Returns bf16 logits [sum(len), vocab]."""
        lens = [len(s) for s in seqs]
        T = sum(lens)
        ids = torch.tensor([t for s in seqs for t in s], device=dev)
        positions = torch.cat([torch.arange(n, device=dev) for n in lens])
        pages = [-(-n // PAGE) for n in lens]
        page_start = [sum(pages[:i]) for i in range(len(seqs))]
        # Each sequence in its own contiguous pages: (page, slot) of every token.
        slot = torch.cat([(page_start[i] * PAGE + torch.arange(n, device=dev)) for i, n in enumerate(lens)])
        i32 = lambda x: torch.tensor(x, device=dev, dtype=torch.int32)
        qo_indptr = i32([0] + list(torch.tensor(lens).cumsum(0)))
        kv_indptr = i32([0] + list(torch.tensor(pages).cumsum(0)))
        kv_indices = torch.arange(sum(pages), device=dev, dtype=torch.int32)
        kv_lens = i32(lens)
        k_cache = torch.zeros(sum(pages) * PAGE, self.hkv, self.hd, device=dev, dtype=torch.bfloat16)
        v_cache = torch.zeros_like(k_cache)
        qw, kw = self.hq * self.hd, self.hkv * self.hd

        hidden = self.embed.index_select(0, ids)
        residual = None
        for L in self.layers:
            if residual is None:
                x, residual = ops.rms_norm(hidden, L["ln1"], self.eps), hidden
            else:
                ops.add_rms_norm_(hidden, residual, L["ln1"], self.eps)
                x = hidden
            packed = linear(x, L["qkv"])
            q = packed[:, :qw].view(T, self.hq, self.hd)
            k = packed[:, qw:qw + kw].view(T, self.hkv, self.hd)
            v = packed[:, qw + kw:].view(T, self.hkv, self.hd)
            ops.qk_norm_(q, k, L["q_norm"], L["k_norm"], self.eps)
            ops.rope_(q, k, positions, self.cos, self.sin)
            k_cache[slot], v_cache[slot] = k, v
            att = attn.prefill(q.contiguous(), k_cache.view(-1, PAGE, self.hkv, self.hd),
                               v_cache.view(-1, PAGE, self.hkv, self.hd), qo_indptr, kv_indptr, kv_indices, kv_lens,
                               max(lens), max(lens), 1.0 / math.sqrt(self.hd), 0, 0)
            out = linear(att.view(T, qw), L["o"])
            ops.add_rms_norm_(out, residual, L["ln2"], self.eps)
            hidden = linear(ops.silu_and_mul(linear(out, L["gate_up"])), L["down"])
        ops.add_rms_norm_(hidden, residual, self.norm, self.eps)
        return linear(hidden, self.lm_head)

    def token_logprobs(self, seqs, starts):
        """Log-probability of seqs[i][j] for j >= starts[i] (from the logits at j - 1): list of f32 tensors."""
        logits = self.logits(seqs)
        out, off = [], 0
        for s, st in zip(seqs, starts):
            rows = torch.arange(off + st - 1, off + len(s) - 1, device=dev)
            out.append(ops.token_logprob(logits.index_select(0, rows), torch.tensor(s[st:], device=dev)))
            off += len(s)
        return out
