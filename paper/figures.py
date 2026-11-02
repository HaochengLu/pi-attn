"""Figures for the PI-Attn paper. Every value is read from results/ or copied from a table the
paper prints next to the figure (which itself traces to results/). Run: python figures.py <results_dir> <out_dir>."""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np

RES = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

# Validated categorical palette (dataviz reference, light mode); colour follows the entity.
C = {
    "PI": "#2a78d6", "original": "#eb6834", "vllm17": "#1baf7a", "vllm17bi": "#eda100",
    "sglang": "#e87ba4", "sglangdet": "#008300", "violet": "#6250d6", "red": "#e34948",
}
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]
ORANGE_RAMP = ["#fdeee7", "#f9cdb8", "#f4a985", "#eb6834", "#c4501f", "#8f3a16"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.5,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42, "axes.titleweight": "semibold", "axes.titlecolor": INK, "axes.titlelocation": "left",
})


def hgrid(ax):
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=220)
    plt.close(fig)


def box(ax, x, y, w, h, fc, ec=None, lw=0.0, r=0.012, z=2, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec or fc, lw=lw,
                       zorder=z, alpha=alpha)
    ax.add_patch(p)
    return p


# ------------------------------------------------------------------ Figure 1: attention program
def fig_attention():
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.06, 1.0)
    ax.axis("off")
    x0, x1 = 0.20, 0.985
    n_tiles = 48  # three 512-key fold groups of 16 tiles of 32 keys
    tw = (x1 - x0) / n_tiles
    gcol = [BLUE_RAMP[1], BLUE_RAMP[2], BLUE_RAMP[1]]
    lab = dict(fontsize=7.6, color=INK, va="center", weight="semibold")
    sub = dict(fontsize=6.7, color=INK2, va="center")
    yk = 0.82
    ax.text(0.0, yk + 0.035, "keys", **lab)
    ax.text(0.0, yk - 0.025, "absolute positions; tile = 32", **sub)
    for t in range(n_tiles):
        box(ax, x0 + t * tw + 0.0012, yk, tw - 0.0024, 0.07, gcol[t // 16], r=0.002)
    for g in range(3):
        xa = x0 + g * 16 * tw
        ax.plot([xa + 0.004, xa + 16 * tw - 0.004], [yk + 0.11] * 2, color=AXIS, lw=0.8)
        ax.text(xa + 8 * tw, yk + 0.15, f"fold group {g}: 512 keys", ha="center",
                fontsize=6.9, color=INK2)
    yc = 0.60
    ax.text(0.0, yc + 0.04, "canonical program", **lab)
    ax.text(0.0, yc - 0.02, "one query row", **sub)
    for g in range(3):
        xa = x0 + g * 16 * tw
        box(ax, xa + 0.004, yc, 16 * tw - 0.008, 0.07, "#f0efec", r=0.006)
        ax.text(xa + 8 * tw, yc + 0.035, "16 online-softmax steps", ha="center", va="center",
                fontsize=6.7, color=INK2)
    yf = 0.47
    ax.plot([x0, x1 - 0.01], [yf, yf], color=C["PI"], lw=1.1, zorder=2)
    ax.annotate("", xy=(x1, yf), xytext=(x1 - 0.02, yf),
                arrowprops=dict(arrowstyle="-|>", color=C["PI"], lw=1.1, mutation_scale=8))
    for g in range(3):
        cx = x0 + (g + 0.5) * 16 * tw
        ax.plot([cx, cx], [yc - 0.004, yf + 0.03], color=MUTED, lw=0.7)
        box(ax, cx - 0.022, yf - 0.03, 0.044, 0.06, SURF, ec=C["PI"], lw=1.0, r=0.01, z=3)
        ax.text(cx, yf, f"fold {g}", ha="center", va="center", fontsize=6.4, color=C["PI"], zorder=4)
    ax.text(0.0, yf + 0.02, "fixed-form fold", **lab)
    ax.text(0.0, yf - 0.04, "ascending groups, then O / L", **sub)
    rows = [
        ("prefill", "FlashAttention-2 loop, state restarts every 512 keys", [(0, 48)], C["PI"]),
        ("decode", "one CTA per (request, KV head, group)", [(0, 16), (16, 32), (32, 48)], C["vllm17"]),
        ("chunked / cached", "any query cut, any prefix-cache hit", [(0, 16), (16, 48)], C["violet"]),
    ]
    yy = 0.30
    for name, desc, spans, col in rows:
        ax.text(0.0, yy + 0.04, name, **lab)
        ax.text(0.0, yy - 0.02, desc, **dict(sub, fontsize=6.2))
        for (a, b) in spans:
            box(ax, x0 + a * tw + 0.004, yy, (b - a) * tw - 0.008, 0.05, col, r=0.006, alpha=0.9)
        yy -= 0.13
    ax.text(0.5, -0.05, "Every path runs the same per-row sequence: same operations, same operands, same order.",
            ha="center", fontsize=7.4, color=INK, style="italic")
    save(fig, "fig1_attention_program")


# ------------------------------------------------------------------ Figure 2: GEMM program
def fig_gemm():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("a  Fixed: one element's operation sequence", fontsize=8.5)
    # chain of k16 steps across S = 2 segments
    n = 16
    x0, x1 = 0.03, 0.97
    w = (x1 - x0) / n
    for s in range(2):
        y = 0.62 - s * 0.30
        ax.text(x0, y + 0.13, f"segment {s}:  acc = 0;  acc = mma(x, w, acc), k16 ascending",
                fontsize=6.8, color=INK2)
        for i in range(n):
            box(ax, x0 + i * w + 0.003, y, w - 0.006, 0.08, BLUE_RAMP[2 + s], r=0.004)
        for i in range(n - 1):
            ax.annotate("", xy=(x0 + (i + 1) * w + 0.002, y + 0.04), xytext=(x0 + (i + 1) * w - 0.004, y + 0.04),
                        arrowprops=dict(arrowstyle="-", color=SURF, lw=0.8))
        ax.text(x1 + 0.005, y + 0.04, f"$p_{s}$", fontsize=8, color=C["PI"], va="center")
    ax.text(0.5, 0.08, r"$y = \mathrm{round}((+0 + p_0) + p_1)$,   $S = S(N, K, t_{\max})$: weight shape only",
            ha="center", fontsize=7.8, color=INK)

    ax = axes[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("b  Free: everything the tuner may choose", fontsize=8.5)
    items = [
        ("output tiling", "small-M warps, CUTLASS 32×64 … 128×256"),
        ("k-stage depth / pipeline", "32- or 64-deep stages, cp.async rings"),
        ("stream-K cut", "fp32 accumulator handed on, chain unbroken"),
        ("segment placement", "same CTA, separate CTAs, or warps"),
        ("who folds", "last arrival folds in ascending order"),
    ]
    y = 0.80
    for head, sub in items:
        box(ax, 0.02, y - 0.055, 0.035, 0.07, C["PI"], r=0.006)
        ax.text(0.08, y, head, fontsize=7.6, color=INK, va="center", weight="semibold")
        ax.text(0.08, y - 0.075, sub, fontsize=6.7, color=INK2, va="center")
        y -= 0.175
    save(fig, "fig2_gemm_program")


# ------------------------------------------------------------------ Figure 3: determinism matrix
def fig_determinism():
    checks = ["batch\nc4", "batch\nc16", "shared\nprefix", "decode vs\nre-prefill", "seeded\nre-prefill",
              "repeat\n×200", "seeded\nsample c4", "seeded\nsample c16", "prefix\nfull hit", "prefix\npartial hit"]
    # requests differing / total (repeat: distinct completions - 1 over 199). From the Section 7.1 table.
    rows = [
        ("Original (FlashInfer)", [16, 16, 32, 24, 24, 168, 16, 16, 5, 6]),
        ("SGLang 0.5.5", [16, 16, 32, 24, 24, 32, 16, 16, 8, 8]),
        ("SGLang deterministic", [5, 5, 30, 24, 24, 0, 4, 6, None, None]),
        ("vLLM 0.17.1", [16, 16, 32, 24, 24, 166, 16, 16, 8, 6]),
        ("vLLM 0.17.1 batch-invariant", [0, 0, 0, 0, 8, 0, 0, 0, 0, 0]),
        ("PI-Attn (--deterministic)", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    ]
    tot = [16, 16, 32, 24, 24, 199, 16, 16, 8, 8]
    frac = np.array([[np.nan if v is None else v / t for v, t in zip(r[1], tot)] for r in rows])
    cmap = LinearSegmentedColormap.from_list("o", ["#fbf3ee", "#f4a985", "#eb6834", "#a8431a"])
    fig, ax = plt.subplots(figsize=(7.0, 2.55))
    m = np.ma.masked_invalid(frac)
    cm = cmap.copy()
    cm.set_bad("#f0efec")
    ax.imshow(m, cmap=cm, vmin=0, vmax=1, aspect="auto")
    for i, (name, vals) in enumerate(rows):
        for j, v in enumerate(vals):
            if v is None:
                txt, col = "no hits", MUTED
            else:
                txt = f"{v}" if j != 5 else f"{v + 1}"
                col = "#ffffff" if frac[i, j] > 0.55 else INK
                if v == 0:
                    col = C["PI"] if i == len(rows) - 1 else INK2
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.6, color=col,
                    weight="bold" if i == len(rows) - 1 else "normal")
    ax.set_xticks(range(len(checks)), checks, fontsize=7)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7.8)
    ax.get_yticklabels()[-1].set_color(C["PI"])
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(checks)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color=SURF, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("requests whose tokens or fp32 log-probabilities differ (repeat ×200: distinct completions)",
                  fontsize=7.4, color=INK2)
    save(fig, "fig3_determinism")


# ------------------------------------------------------------------ Figure 4: baselines
def fig_baselines():
    wl = ["64/256\nc1", "512/256\nc8", "512/256\nc32", "2048/256\nc16", "4096/128\nc4"]
    d = {
        "original": [283, 1894, 5014, 1878, 539],
        "PI": [298, 1790, 4951, 1863, 542],
        "sglang": [253, 1768, 4865, 1837, 530],
        "sglangdet": [73, 554, 1993, 843, 192],
        "vllm17": [270, 1842, 4890, 1885, 553],
        "vllm17bi": [80, 609, 2228, 921, 223],
    }
    order = [("PI", "PI-Attn (deterministic)"), ("vllm17bi", "vLLM 0.17.1 batch-invariant (deterministic)"),
             ("sglangdet", "SGLang deterministic"), ("vllm17", "vLLM 0.17.1"), ("sglang", "SGLang 0.5.5")]
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    base = np.array(d["original"], float)
    nb = len(order)
    width = 0.15
    x = np.arange(len(wl))
    for k, (key, label) in enumerate(order):
        vals = np.array(d[key]) / base
        xs = x + (k - (nb - 1) / 2) * width
        ax.bar(xs, vals, width * 0.86, color=C[key], label=label, zorder=3)
        if key in ("PI", "vllm17bi"):
            for xi, v in zip(xs, vals):
                ax.text(xi, v + 0.035, f"{v:.2f}", ha="center", va="bottom", fontsize=6.2, color=INK2,
                        bbox=dict(fc=SURF, ec="none", pad=0.4), zorder=5)
    ax.axhline(1.0, color=INK2, lw=0.7, zorder=2)
    ax.set_xticks(x, wl)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("output throughput\n(original engine = 1)")
    hgrid(ax)
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.22), handlelength=1.0, columnspacing=1.2)
    ax.set_xlabel("Qwen3-1.7B workload (input / output tokens, concurrency), A800")
    save(fig, "fig4_baselines")


# ------------------------------------------------------------------ Figure 5: end to end
def fig_e2e():
    rows = [l for l in (RES / "a800/e2e_matrix/SUMMARY.md").read_text().splitlines() if l.startswith("| ") and "model" not in l and "---" not in l]
    models = ["0.6B", "1.7B", "4B", "8B"]
    wls = ["64/256 c1", "512/256 c8", "512/256 c32", "2048/256 c16", "4096/128 c4"]
    M = np.full((4, 5), np.nan)
    for r in rows:
        c = [x.strip() for x in r.strip("|").split("|")]
        if c[1] != "0":
            continue
        M[models.index(c[0]), wls.index(c[2])] = float(c[5].rstrip("%"))
    assert not np.isnan(M).any()
    fig = plt.figure(figsize=(7.0, 2.55))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.32)
    ax = fig.add_subplot(gs[0])
    cmap = LinearSegmentedColormap.from_list("div", ["#c4501f", "#f4a985", "#f0efec", "#86b6ef", "#1c5cab"])
    norm = TwoSlopeNorm(vmin=-15, vcenter=0, vmax=15)
    ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
    for i in range(4):
        for j in range(5):
            v = M[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7.4,
                    color="#ffffff" if abs(v) > 10 else INK)
    ax.set_xticks(range(5), [w.replace(" ", "\n") for w in wls], fontsize=7)
    ax.set_yticks(range(4), ["Qwen3-" + m for m in models])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 5), minor=True)
    ax.set_yticks(np.arange(-0.5, 4), minor=True)
    ax.grid(which="minor", color=SURF, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_title("a  Δ output throughput vs. original engine (%)", fontsize=8.3)

    ax = fig.add_subplot(gs[1])
    L17 = [(64, 8.8), (512, -0.6), (2048, 0.1), (4096, 2.9), (16384, 12.2), (30000, 18.3)]
    L8 = [(64, 3.6), (512, 3.6), (2048, 5.2), (4096, 6.3), (16384, 13.2)]
    for pts, key, lab in [(L17, "PI", "Qwen3-1.7B"), (L8, "violet", "Qwen3-8B")]:
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=C[key], lw=1.6, marker="o", ms=4.2, mec=SURF, mew=1.0, label=lab, zorder=3)
        ax.text(xs[-1] * 1.18, ys[-1] + (0.0 if key == "PI" else 0.0), lab, color=INK2, fontsize=7, va="center")
    ax.set_xscale("log")
    ax.set_xticks([64, 512, 2048, 4096, 16384, 30000], ["64", "512", "2K", "4K", "16K", "30K"])
    for t in ax.get_xticklabels():
        if t.get_text() in ("2K", "16K"):
            t.set_horizontalalignment("right")
        if t.get_text() in ("4K", "30K"):
            t.set_horizontalalignment("left")
    ax.minorticks_off()
    ax.axhline(0, color=AXIS, lw=0.7)
    ax.set_xlim(45, 80000)
    ax.set_ylabel("Δ time to first token (%)")
    ax.set_xlabel("prompt length (tokens)")
    hgrid(ax)
    ax.set_title("b  Prefill pays: TTFT vs. prompt length", fontsize=8.3)
    save(fig, "fig5_e2e")


# ------------------------------------------------------------------ Figure 6: rollout
def fig_rollout():
    models = ["0.6B", "1.7B", "4B", "8B"]
    pi_eq, hf_eq, hf_off, toks = [], [], [], []
    for m in models:
        d = json.loads((RES / f"a800/rollout/Qwen3-{m}.json").read_text())
        n = d["pi alone"]["tokens"]
        toks.append(n)
        pi_eq.append(min(d[k]["bitwise_equal"] for k in ("pi alone", "pi packing 0", "pi packing 1")) / n * 100)
        hf_eq.append(d["hf alone"]["bitwise_equal"] / n * 100)
        hf_off.append(d["hf alone"]["ratio_frac_outside_1pct"] * 100)
    assert sum(toks) == 55593, sum(toks)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.25), gridspec_kw={"wspace": 0.38})
    x = np.arange(4)
    w = 0.34
    ax = axes[0]
    ax.bar(x - w / 2, pi_eq, w * 0.9, color=C["PI"], label="PI trainer forward (any packing)", zorder=3)
    ax.bar(x + w / 2, hf_eq, w * 0.9, color=C["original"], label="HF bf16 trainer forward", zorder=3)
    for xi, v in zip(x, hf_eq):
        ax.text(xi + w / 2, v + 2, f"{v:.1f}", ha="center", fontsize=6.6, color=INK2)
    for xi in x:
        ax.text(xi - w / 2, 102, "100", ha="center", fontsize=6.6, color=C["PI"], weight="bold")
    ax.set_xticks(x, models)
    ax.set_xlabel("Qwen3")
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("tokens with log-prob\nbitwise equal to engine (%)")
    hgrid(ax)
    ax.set_title("a  Trainer log-prob = rollout log-prob", fontsize=8.3)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.36), ncol=2, handlelength=1.0)
    ax = axes[1]
    ax.bar(x - w / 2, [0] * 4, w * 0.9, color=C["PI"], zorder=3)
    ax.bar(x + w / 2, hf_off, w * 0.9, color=C["original"], zorder=3)
    for xi, v in zip(x, hf_off):
        ax.text(xi + w / 2, v + 0.5, f"{v:.1f}", ha="center", fontsize=6.6, color=INK2)
    for xi in x:
        ax.text(xi - w / 2, 0.5, "0", ha="center", fontsize=6.6, color=C["PI"], weight="bold")
    ax.set_xticks(x, models)
    ax.set_xlabel("Qwen3")
    ax.set_ylim(0, 26)
    ax.set_ylabel("tokens with |importance\nratio − 1| > 1% (%)")
    hgrid(ax)
    ax.set_title("b  Importance-ratio noise", fontsize=8.3)
    save(fig, "fig6_rollout")


# ------------------------------------------------------------------ Figure 7: batch-1 breakdown
def fig_breakdown():
    txt = (RES / "a800/nsys/b1_compare.txt").read_text()
    models = ["0.6B", "1.7B", "4B", "8B"]
    comps = [("projections", "projections", "violet"), ("attention", "attention", "PI"),
             ("other kernels", "other kernels", "vllm17"), ("between kernels", "between kernels", "vllm17bi")]
    vals = {c[0]: [] for c in comps}
    for blk in txt.split("== ")[1:]:
        for key, _, _ in comps:
            m = re.search(rf"^\s+{key}\s+[\d.]+ ->\s+[\d.]+\s+([+-][\d.]+)", blk, re.M)
            vals[key].append(float(m.group(1)))
    fig, ax = plt.subplots(figsize=(7.0, 2.3))
    x = np.arange(4)
    w = 0.19
    for k, (key, lab, col) in enumerate(comps):
        xs = x + (k - 1.5) * w
        ax.bar(xs, vals[key], w * 0.86, color=C[col], label=lab, zorder=3)
    ax.axhline(0, color=INK2, lw=0.7, zorder=4)
    ax.set_xticks(x, ["Qwen3-" + m for m in models])
    ax.set_ylabel("PI − original\n(µs per decode step)")
    hgrid(ax)
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), handlelength=1.0)
    ax.set_ylim(-760, 160)
    ax.text(3.45, 120, "above 0: PI slower", fontsize=6.6, color=MUTED, ha="right")
    save(fig, "fig7_breakdown")


# ------------------------------------------------------------------ Figure 8: cross-GPU
def fig_xgpu():
    cats = ["GEMM kernel\ncases (80)", "attention kernel\ncases (5)", "GSM8K gens,\n1.7B (1319)", "GSM8K gens,\n8B (500)"]
    pi = [80 / 80, 5 / 5, 1.0, 1.0]
    # baseline: cuBLAS 14/80, FlashInfer 1/5, original engine (worst of c1/c16) 910/1319, 343/500 differ
    base = [14 / 80, 1 / 5, (1319 - 910) / 1319, (500 - 343) / 500]
    base_lab = ["cuBLAS", "FlashInfer", "original engine", "original engine"]
    cnt_p = dict(zip(np.arange(4)[::-1], ["80 / 80", "5 / 5", "1319 / 1319", "500 / 500"]))
    cnt_b = dict(zip(np.arange(4)[::-1], ["14 / 80", "1 / 5", "409–413 / 1319", "157–175 / 500"]))
    fig, ax = plt.subplots(figsize=(7.0, 1.95))
    y = np.arange(4)[::-1]
    h = 0.34
    ax.barh(y + h / 2, [v * 100 for v in pi], h * 0.88, color=C["PI"], label="PI-Attn", zorder=3)
    ax.barh(y - h / 2, [v * 100 for v in base], h * 0.88, color=C["original"], label="stock kernels / engine",
            zorder=3)
    for yi, v, bl in zip(y, base, base_lab):
        ax.text(v * 100 + 1.2, yi - h / 2, f"{bl}: {cnt_b[yi]}", va="center", fontsize=6.8, color=INK2)
        ax.text(101.2, yi + h / 2, cnt_p[yi], va="center", fontsize=6.8, color=C["PI"], weight="bold")
    ax.set_yticks(y, cats)
    ax.set_xlim(0, 118)
    ax.set_xlabel("bitwise identical between A800 (sm_80) and RTX 4090 D (sm_89), %")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, handlelength=1.0)
    save(fig, "fig8_xgpu")


# ------------------------------------------------------------------ Figure 9: GDN self-check
def fig_gdn():
    cfg = ["stock GDN", "exact GDN", "stock GDN,\nbatch-inv. dense", "exact GDN,\nbatch-inv. dense"]
    v08 = [274, 264, 311, 2048]
    v4 = [120, 131, 181, 2048]
    fig, ax = plt.subplots(figsize=(7.0, 2.0))
    x = np.arange(4)
    w = 0.34
    ax.bar(x - w / 2, np.array(v08) / 20.48, w * 0.88, color=C["violet"], label="Qwen3.5-0.8B", zorder=3)
    ax.bar(x + w / 2, np.array(v4) / 20.48, w * 0.88, color=C["vllm17"], label="Qwen3.5-4B", zorder=3)
    for xi, a, b in zip(x, v08, v4):
        ax.text(xi - w / 2, a / 20.48 + 2, f"{a}", ha="center", fontsize=6.6, color=INK2)
        ax.text(xi + w / 2, b / 20.48 + 2, f"{b}", ha="center", fontsize=6.6, color=INK2)
    ax.set_xticks(x, cfg)
    ax.set_ylim(0, 112)
    ax.set_ylabel("decode log-prob = prefill\nlog-prob, bitwise (% of 2048)")
    hgrid(ax)
    ax.legend(loc="upper left", handlelength=1.0)
    save(fig, "fig9_gdn")


for f in (fig_attention, fig_gemm, fig_determinism, fig_baselines, fig_e2e, fig_rollout, fig_breakdown, fig_xgpu,
          fig_gdn):
    f()
print("ok")
