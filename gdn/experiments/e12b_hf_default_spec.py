"""E12b: the status quo. Stock HF Qwen3.5 (fla kernels: recurrent decode for one token, chunk from the recurrent state
for several; cuBLAS/SDPA dense ops): greedy-generate NEW tokens one at a time, then re-score the same tokens the way
a speculative verifier sees them (windows of W tokens through the cache, every token accepted, so no rollback is
involved). Reports how many of the generated tokens get a bitwise-equal log-probability and the size of the gap.
usage: python e12b_hf_default_spec.py MODEL_DIR [P] [NEW] [W]"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
path = sys.argv[1]
P = int(sys.argv[2]) if len(sys.argv) > 2 else 200
NEW = int(sys.argv[3]) if len(sys.argv) > 3 else 300
W = int(sys.argv[4]) if len(sys.argv) > 4 else 5
dev = 'cuda'
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(dev).eval()
ids = tok(open(f'{path}/README.md').read(), return_tensors='pt').input_ids[:, :P].to(dev)
with torch.no_grad():
    o = model(ids, use_cache=True); pkv = o.past_key_values
    lp = o.logits[0, -1].float().log_softmax(-1)
    toks = [int(lp.argmax())]; lp_plain = [float(lp[toks[0]])]
    while len(toks) < NEW:
        o = model(torch.tensor([[toks[-1]]], device=dev), past_key_values=pkv, use_cache=True); pkv = o.past_key_values
        lp = o.logits[0, -1].float().log_softmax(-1)
        toks.append(int(lp.argmax())); lp_plain.append(float(lp[toks[-1]]))
    o = model(ids, use_cache=True); pkv = o.past_key_values                    # same prefill, then windows
    lp_win = [float(o.logits[0, -1].float().log_softmax(-1)[toks[0]])]
    i = 0                                        # toks[i] is the next token to feed
    while len(lp_win) < NEW:
        win = toks[i:i + W]
        o = model(torch.tensor([win], device=dev), past_key_values=pkv, use_cache=True); pkv = o.past_key_values
        L = o.logits[0].float().log_softmax(-1)
        for j in range(len(win)):
            if len(lp_win) < NEW:
                lp_win.append(float(L[j][toks[i + j + 1]]))
        i += len(win)
eq = sum(a == b for a, b in zip(lp_plain, lp_win))
d = [abs(a - b) for a, b in zip(lp_plain, lp_win)]
print(f'model {path.rstrip("/").split("/")[-1]} (stock HF): window {W} verification vs one-token decode, {NEW} tokens: '
      f'log-probs bitwise equal {eq}/{NEW}; mean |dlogp| {sum(d) / NEW:.3e}, max {max(d):.3e}')
