"""Publication-grade figures for the PI-Attn paper.

Architectural diagrams (Figures 2 and 3) are compiled via Tectonic + TikZ with
Libertinus Serif/Math and generous node padding so labels never crowd borders.
Empirical charts use STIX Two serif typography and read directly from ../results/.

Usage:
    python figures.py <results_dir> <out_dir> [func_names...]
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import fitz  # PyMuPDF for vector PDF -> crisp PNG rendering
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np

RES = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

# Restraint-first academic palette (OSDI / SOSP / MLSys aesthetic)
NAVY = "#1B3A6B"       # PI-Attn primary
NAVY_FILL = "#244C8A"
TEAL = "#2A6F78"       # Secondary accent
TEAL_FILL = "#35828C"
RUST = "#B54434"       # Non-deterministic mismatch / vLLM batch-inv
RUST_FILL = "#C85240"
AMBER = "#B87D2B"      # SGLang determ / secondary baseline
AMBER_FILL = "#CC8E35"
SLATE = "#5B6777"      # Stock non-deterministic baselines
SLATE_FILL = "#738091"
SLATE_LIGHT = "#E6E9EF"

INK = "#111827"
INK2 = "#374151"
MUTED = "#6B7280"
GRID = "#E5E7EB"
SPINE = "#6B7280"
SURF = "#FFFFFF"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.4,
    "axes.titlesize": 8.8,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "axes.labelsize": 8.2,
    "axes.labelcolor": INK,
    "axes.edgecolor": SPINE,
    "axes.linewidth": 0.65,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.color": SPINE,
    "ytick.color": SPINE,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.8,
    "ytick.major.size": 2.8,
    "xtick.major.pad": 3.5,
    "ytick.major.pad": 3.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.55,
    "grid.linestyle": "--",
    "legend.frameon": False,
    "legend.fontsize": 7.3,
    "figure.dpi": 180,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
})


def hgrid(ax):
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)


def pdf_to_png(pdf_path: Path, png_path: Path, dpi: int = 280):
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    pix.save(str(png_path))
    doc.close()


def save(fig, name: str):
    pdf_path = OUT / f"{name}.pdf"
    png_path = OUT / f"{name}.png"
    fig.savefig(pdf_path)
    plt.close(fig)
    pdf_to_png(pdf_path, png_path)


def compile_tikz(tex_body: str, name: str):
    full_tex = r"""\documentclass[tikz,border=5pt]{standalone}
\usepackage{libertinus-otf}
\usepackage{unicode-math}
\setmathfont{LibertinusMath-Regular.otf}
\setmonofont[Scale=0.85]{Inconsolatazi4-Regular.otf}
\usepackage{xcolor}
\definecolor{navy}{HTML}{1B3A6B}
\definecolor{navyfill}{HTML}{EBF1FA}
\definecolor{navymid}{HTML}{C6D8F0}
\definecolor{teal}{HTML}{23646E}
\definecolor{tealfill}{HTML}{E8F3F5}
\definecolor{rust}{HTML}{A63A2B}
\definecolor{rustfill}{HTML}{FBECE9}
\definecolor{slate}{HTML}{475569}
\definecolor{slatefill}{HTML}{F1F5F9}
\definecolor{ink}{HTML}{111827}
\definecolor{ink2}{HTML}{374151}
\definecolor{muted}{HTML}{6B7280}
\definecolor{border}{HTML}{94A3B8}
\usetikzlibrary{arrows.meta,positioning,calc,decorations.pathreplacing,fit,backgrounds}
\begin{document}
""" + tex_body + "\n\\end{document}\n"
    with tempfile.TemporaryDirectory() as tmp:
        tex_file = Path(tmp) / f"{name}.tex"
        pdf_tmp = Path(tmp) / f"{name}.pdf"
        tex_file.write_text(full_tex)
        res = subprocess.run(
            ["tectonic", "-X", "compile", str(tex_file)],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            raise RuntimeError(f"TikZ compile failed for {name}:\n{res.stderr}\n{res.stdout}")
        pdf_out = OUT / f"{name}.pdf"
        shutil.copy2(pdf_tmp, pdf_out)
        pdf_to_png(pdf_out, OUT / f"{name}.png", dpi=300)


def read_a800_r6_17b_greedy():
    txt = (RES / "a800_r6/e2e_matrix/SUMMARY.md").read_text().splitlines()
    fi, pi = [], []
    wls = ["64/256 c1", "512/256 c8", "512/256 c32", "2048/256 c16", "4096/128 c4"]
    rows = {}
    for line in txt:
        if line.startswith("| 1.7B | 0 |"):
            cols = [c.strip() for c in line.strip("|").split("|")]
            rows[cols[2]] = (float(cols[3]), float(cols[4]))
    for w in wls:
        f_v, p_v = rows[w]
        fi.append(f_v)
        pi.append(p_v)
    return np.array(fi), np.array(pi)


# ------------------------------------------------------------------ Figure 0 (Figure 1): Teaser
def fig_teaser():
    fig, axes = plt.subplots(
        1, 3, figsize=(7.05, 2.35),
        gridspec_kw={"width_ratios": [1.58, 1.0, 0.92], "wspace": 0.50},
    )

    # Panel (a): Zero-tax determinism on BOTH A800 and H100 (Qwen3-1.7B, 5 workloads)
    ax = axes[0]
    fi_a, pi_a = read_a800_r6_17b_greedy()
    vllm_a = np.array([270.0, 1842.0, 4890.0, 1885.0, 553.0])
    vllm_bi_a = np.array([80.0, 609.0, 2228.0, 921.0, 223.0])
    sgl_a = np.array([253.0, 1768.0, 4865.0, 1837.0, 530.0])
    sgl_det_a = np.array([73.0, 554.0, 1993.0, 843.0, 192.0])

    fi_h = np.array([436.0, 2998.0, 8537.0, 3375.0, 974.0])
    pi_h = np.array([455.0, 2946.0, 8624.0, 3400.0, 966.0])
    vllm_h = np.array([442.0, 3066.0, 8738.0, 3648.0, 1081.0])
    vllm_bi_h = np.array([149.0, 1168.0, 4442.0, 1889.0, 467.0])
    sgl_h = np.array([407.0, 2758.0, 7812.0, 3296.0, 956.0])
    sgl_det_h = np.array([243.0, 1710.0, 5446.0, 2311.0, 577.0])

    def gmean(r):
        return float(np.exp(np.mean(np.log(r))))

    groups = [
        ("c=1\n64/256", (pi_a / fi_a)[0], (vllm_bi_a / vllm_a)[0], (sgl_det_a / sgl_a)[0]),
        ("c=8\n512/256", (pi_a / fi_a)[1], (vllm_bi_a / vllm_a)[1], (sgl_det_a / sgl_a)[1]),
        ("c=32\n512/256", (pi_a / fi_a)[2], (vllm_bi_a / vllm_a)[2], (sgl_det_a / sgl_a)[2]),
        ("A800\n(5 wls)", gmean(pi_a / fi_a), gmean(vllm_bi_a / vllm_a), gmean(sgl_det_a / sgl_a)),
        ("H100\n(5 wls)", gmean(pi_h / fi_h), gmean(vllm_bi_h / vllm_h), gmean(sgl_det_h / sgl_h)),
    ]
    x = np.arange(len(groups))
    w = 0.235
    specs = [
        (1, "PI-Attn", NAVY_FILL, NAVY),
        (2, "vLLM batch-inv.", RUST_FILL, RUST),
        (3, "SGLang determ.", AMBER_FILL, AMBER),
    ]
    for k, (idx, lab, fc, ec) in enumerate(specs):
        vals = [g[idx] for g in groups]
        bars = ax.bar(
            x + (k - 1) * w, vals, w * 0.86,
            color=fc, edgecolor=ec, linewidth=0.65, label=lab, zorder=3,
        )
        if k == 0:
            for bx, v in zip(bars, vals):
                y_txt = max(v + 0.035, 1.04)
                ax.text(
                    bx.get_x() + bx.get_width() / 2, y_txt,
                    f"{v:.2f}×", ha="center", va="bottom",
                    fontsize=6.3, fontweight="bold", color=NAVY,
                )
    ax.axvline(2.53, color=GRID, lw=0.85, ls="-", zorder=1)
    ax.axhline(1.0, color=INK2, lw=0.75, ls=(0, (3, 2)), zorder=2)
    ax.set_xticks(x, [g[0] for g in groups], fontsize=6.8)
    ax.set_ylim(0, 1.44)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["0.0×", "0.5×", "1.0×"])
    ax.set_ylabel("Throughput vs. stock\nnon-determ. engine", labelpad=4)
    hgrid(ax)
    ax.legend(
        loc="upper left", bbox_to_anchor=(-0.02, 1.03),
        ncol=3, fontsize=6.4, handlelength=0.85, columnspacing=0.65, handletextpad=0.35,
    )
    ax.set_title("(a) Full-speed determinism", pad=13)

    # Panel (b): Rollout = trainer across 0.6B - 8B
    ax = axes[1]
    models = ["0.6B", "1.7B", "4B", "8B"]
    hf = [
        json.loads((RES / f"a800/rollout/Qwen3-{m}.json").read_text())["hf alone"]["ratio_frac_outside_1pct"] * 100
        for m in models
    ]
    xm = np.arange(4)
    bw = 0.33
    ax.bar(xm - bw / 2, [0.0] * 4, bw * 0.86, color=NAVY_FILL, edgecolor=NAVY, lw=0.65, label="PI-Attn trainer", zorder=3)
    ax.bar(xm + bw / 2, hf, bw * 0.86, color=RUST_FILL, edgecolor=RUST, lw=0.65, label="Standard BF16", zorder=3)
    for xi, hv in zip(xm, hf):
        ax.text(xi - bw / 2, 0.7, "0%", ha="center", va="bottom", fontsize=6.3, fontweight="bold", color=NAVY)
        ax.text(xi + bw / 2, hv + 0.7, f"{hv:.1f}%", ha="center", va="bottom", fontsize=6.2, color=INK2)
    ax.set_xticks(xm, models, fontsize=7.1)
    ax.set_xlabel("Qwen3 model scale", fontsize=7.5, labelpad=3)
    ax.set_ylim(0, 30)
    ax.set_yticks([0, 10, 20])
    ax.set_ylabel("Tokens with $|r_t - 1| > 1\\%$", labelpad=4)
    hgrid(ax)
    ax.legend(loc="upper right", bbox_to_anchor=(1.02, 1.03), fontsize=6.4, handlelength=0.85)
    ax.set_title("(b) Zero RL mismatch", pad=13)

    # Panel (c): Cross-GPU bitwise identity
    ax = axes[2]
    cats = ["Kernels\n(5 GPUs)", "Rollouts\n(A10G)", "GSM8K\n(4090 D)"]
    pi_v = [100.0, 100.0, 100.0]
    st_v = [39 / 108 * 100, 0.0, (1319 - 910) / 1319 * 100]
    xc = np.arange(3)
    cw = 0.33
    ax.bar(xc - cw / 2, pi_v, cw * 0.86, color=NAVY_FILL, edgecolor=NAVY, lw=0.65, label="PI-Attn", zorder=3)
    ax.bar(xc + cw / 2, st_v, cw * 0.86, color=SLATE_FILL, edgecolor=SLATE, lw=0.65, label="Stock", zorder=3)
    for xi, sv in zip(xc, st_v):
        ax.text(xi - cw / 2, 103.0, "100%", ha="center", va="bottom", fontsize=6.2, fontweight="bold", color=NAVY)
        ax.text(xi + cw / 2, sv + 2.5, f"{sv:.0f}%", ha="center", va="bottom", fontsize=6.1, color=INK2)
    ax.set_xticks(xc, cats, fontsize=6.6)
    ax.set_ylim(0, 140)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel("Bitwise identical\nto A800 (%)", labelpad=4)
    hgrid(ax)
    ax.legend(
        loc="upper left", bbox_to_anchor=(-0.02, 1.03),
        ncol=2, fontsize=6.4, handlelength=0.85, columnspacing=0.75, handletextpad=0.35,
    )
    ax.set_title("(c) Cross-GPU identity", pad=13)

    save(fig, "fig0_teaser")


# ------------------------------------------------------------------ Figure 1 (Figure 2): Canonical Attention (TikZ)
def fig_attention():
    tex = r"""
\begin{tikzpicture}[
    font=\small,
    >=Stealth,
    lbl/.style={font=\footnotesize\bfseries\color{ink}, anchor=west},
    sublbl/.style={font=\scriptsize\color{ink2}, anchor=west},
    foldnode/.style={draw=navy, line width=0.65pt, fill=white, rounded corners=1.5pt,
                     inner xsep=6pt, inner ysep=4pt, minimum height=0.52cm,
                     font=\scriptsize\bfseries\color{navy}}
]
\def\xL{0.0}
\def\xS{4.10}
\def\GW{3.52}
\def\GSTEP{3.76}
\def\TOTW{11.04}

% 1. Absolute KV positions
\node[lbl] at (\xL, 4.18) {1. Canonical KV Grid};
\node[sublbl] at (\xL, 3.80) {Absolute positions; tile $= 32$};

\foreach \g/\gname in {0/{Group $G_0$ ($0..511$)}, 1/{Group $G_1$ ($512..1023$)}, 2/{Group $G_2$ ($1024..1535$)}} {
    \pgfmathsetmacro{\gx}{\xS + \g * \GSTEP}
    \draw[decorate, decoration={brace, amplitude=3.5pt, raise=2.5pt}, draw=ink2, line width=0.55pt]
        (\gx, 4.24) -- (\gx + \GW, 4.24) node[midway, above=5pt, font=\scriptsize\bfseries\color{ink}] {\gname};
    \foreach \t in {0,...,15} {
        \pgfmathsetmacro{\tx}{\gx + \t * (\GW / 16.0)}
        \pgfmathsetmacro{\tw}{(\GW / 16.0) - 0.028}
        \ifnum\g=2
            \ifnum\t>10
                \filldraw[fill=slatefill, draw=border, line width=0.38pt] (\tx, 3.72) rectangle (\tx + \tw, 4.18);
            \else
                \filldraw[fill=navymid, draw=navy, line width=0.38pt] (\tx, 3.72) rectangle (\tx + \tw, 4.18);
            \fi
        \else
            \filldraw[fill=navymid, draw=navy, line width=0.38pt] (\tx, 3.72) rectangle (\tx + \tw, 4.18);
        \fi
    }
}
% Query position marker p
\pgfmathsetmacro{\px}{\xS + 2 * \GSTEP + 11 * (\GW / 16.0) - 0.014}
\draw[rust, line width=0.85pt, densely dashed] (\px, 3.50) -- (\px, 4.30);
\node[font=\scriptsize\bfseries\color{rust}, anchor=west] at (\px + 0.08, 3.42) {$j > p$: exact no-op};

% 2. Intra-group online softmax
\node[lbl] at (\xL, 2.86) {2. Intra-Group Scan};
\node[sublbl] at (\xL, 2.48) {Reset $(-\infty, 0, 0)$ every 512 keys};

\foreach \g in {0,1,2} {
    \pgfmathsetmacro{\gx}{\xS + \g * \GSTEP}
    \filldraw[fill=navyfill, draw=navy, line width=0.58pt, rounded corners=1pt] (\gx, 2.38) rectangle (\gx + \GW, 3.00);
    \node[font=\scriptsize\color{ink}] at (\gx + 0.5*\GW, 2.66)
        {$(-\infty,0,0) \xrightarrow{\;16\text{ tiles}\;} (m_{\g}, d_{\g}, o_{\g})$};
    \draw[-{Stealth[length=1.6mm]}, draw=ink2, line width=0.5pt] (\gx + 0.5*\GW, 3.70) -- (\gx + 0.5*\GW, 3.02);
}

% 3. Fixed-order inter-group fold
\node[lbl] at (\xL, 1.60) {3. Ascending Fold};
\node[sublbl] at (\xL, 1.22) {Associativity boundary fixed};

\node[font=\scriptsize\color{ink2}, anchor=west, inner xsep=2pt] (init) at (\xS, 1.42) {$(-\infty,0,0)$};
\foreach \g in {0,1,2} {
    \pgfmathsetmacro{\cx}{\xS + \g * \GSTEP + 0.54 * \GW}
    \node[foldnode] (F\g) at (\cx, 1.42) {Fold $G_{\g}$};
    \draw[-{Stealth[length=1.6mm]}, draw=navy, line width=0.55pt]
        (\cx, 2.38) -- (F\g.north);
}
\node[font=\scriptsize\bfseries\color{navy}, anchor=east, inner xsep=2pt] (out) at (\xS + \TOTW, 1.42) {$O / L$};
\draw[-{Stealth[length=1.6mm]}, draw=navy, line width=0.65pt] (init.east) -- (F0.west);
\draw[-{Stealth[length=1.6mm]}, draw=navy, line width=0.65pt] (F0.east) -- (F1.west);
\draw[-{Stealth[length=1.6mm]}, draw=navy, line width=0.65pt] (F1.east) -- (F2.west);
\draw[-{Stealth[length=1.6mm]}, draw=navy, line width=0.65pt] (F2.east) -- (out.west);

% Divider
\draw[draw=border, line width=0.55pt, densely dotted] (\xL, 0.78) -- (\xS + \TOTW, 0.78);
\node[font=\scriptsize\itshape\color{ink2}, fill=white, inner xsep=6pt] at (\xS + 0.5*\TOTW - 0.3, 0.78)
    {Schedule-free kernel realizations executing the identical per-row floating-point sequence};

% 4. Three execution paths with generous bar height (0.48 cm) and vertical spacing (0.70 cm)
% Prefill
\node[lbl] at (\xL, 0.20) {Prefill (FA2 / \texttt{wgmma})};
\filldraw[fill=navyfill, draw=navy, line width=0.58pt, rounded corners=1pt] (\xS, -0.04) rectangle (\xS + \TOTW, 0.44);
\node[font=\scriptsize\color{navy}] at (\xS + 0.5*\TOTW, 0.20)
    {Streaming CTA loop per query tile; state resets to $(-\infty,0,0)$ every 512 keys and folds into $(M,L,O)$};

% Decode
\node[lbl] at (\xL, -0.50) {Split-KV Decode};
\foreach \g/\ctatxt in {0/{CTA $(\text{req}, h, G_0)$}, 1/{CTA $(\text{req}, h, G_1)$}, 2/{CTA $(\text{req}, h, G_2)$}} {
    \pgfmathsetmacro{\gx}{\xS + \g * \GSTEP}
    \filldraw[fill=tealfill, draw=teal, line width=0.58pt, rounded corners=1pt] (\gx, -0.74) rectangle (\gx + \GW, -0.26);
    \node[font=\scriptsize\color{teal}] at (\gx + 0.5*\GW, -0.50) {\ctatxt};
}
\node[font=\scriptsize\color{teal}, anchor=west] at (\xS + \TOTW + 0.12, -0.50) {$\to$ fold};

% Chunked / Prefix-cached
\node[lbl] at (\xL, -1.20) {Cached / Chunked};
\filldraw[fill=slatefill, draw=slate, line width=0.58pt, rounded corners=1pt] (\xS, -1.44) rectangle (\xS + 5.10, -0.96);
\node[font=\scriptsize\color{slate}] at (\xS + 2.55, -1.20) {Radix prefix cache hit (paged KV)};
\filldraw[fill=rustfill, draw=rust, line width=0.58pt, rounded corners=1pt] (\xS + 5.26, -1.44) rectangle (\xS + \TOTW, -0.96);
\node[font=\scriptsize\color{rust}] at (\xS + 8.15, -1.20) {Chunked extend / mixed-batch decode tail};

\end{tikzpicture}
"""
    compile_tikz(tex, "fig1_attention_program")


# ------------------------------------------------------------------ Figure 2 (Figure 3): Canonical GEMM (TikZ)
def fig_gemm():
    tex = r"""
\begin{tikzpicture}[
    font=\small,
    >=Stealth,
    paneltitle/.style={font=\footnotesize\bfseries\color{ink}, anchor=west},
    kstep0/.style={draw=navy, line width=0.55pt, fill=navyfill, rounded corners=1pt,
                   minimum width=0.86cm, minimum height=0.46cm, inner xsep=5pt, inner ysep=3pt,
                   font=\scriptsize\color{navy}},
    kstep1/.style={draw=teal, line width=0.55pt, fill=tealfill, rounded corners=1pt,
                   minimum width=0.86cm, minimum height=0.46cm, inner xsep=5pt, inner ysep=3pt,
                   font=\scriptsize\color{teal}},
    ctanode/.style={line width=0.6pt, rounded corners=1.2pt,
                    minimum width=1.18cm, minimum height=0.48cm, inner xsep=5pt, inner ysep=3pt,
                    font=\scriptsize\bfseries},
    tpnode/.style={draw=teal, line width=0.55pt, fill=tealfill, rounded corners=1pt,
                   minimum width=0.72cm, minimum height=0.62cm, inner sep=4pt,
                   font=\scriptsize\color{teal}}
]
% ==================== Panel (a): Fixed Canonical Chain ====================
\node[paneltitle] at (0.0, 3.45) {(a) Fixed Numerical Program (per element $y_{m,n}$)};

\node[font=\scriptsize\color{ink2}, anchor=west] at (0.0, 2.95)
    {Weight shape $(N, K, t_{\max})$ fixes $S$ segments of ascending $k16$ steps:};

% Segment 0 (spacious TikZ nodes with wide inner padding)
\node[font=\scriptsize\bfseries\color{navy}, anchor=west] at (0.0, 2.42) {Segment $s{=}0$:};
\node[font=\scriptsize\color{ink2}, anchor=west] at (1.85, 2.42) {$\mathrm{acc}_0 \leftarrow +0$};

\node[kstep0, anchor=west] (s0a) at (0.0, 1.84) {$k_0$};
\node[kstep0, right=0.22cm of s0a] (s0b) {$k_1$};
\node[font=\scriptsize\color{navy}, right=0.20cm of s0b] (s0dots) {$\cdots$};
\node[kstep0, right=0.20cm of s0dots] (s0c) {$k_{L{-}2}$};
\node[kstep0, right=0.22cm of s0c] (s0d) {$k_{L{-}1}$};
\node[font=\scriptsize\bfseries\color{navy}, right=0.22cm of s0d] (p0) {$p_0 \in \text{fp32}$};

\draw[-{Stealth[length=1.4mm]}, draw=navy, line width=0.5pt] (s0a) -- (s0b);
\draw[-{Stealth[length=1.4mm]}, draw=navy, line width=0.5pt] (s0b) -- (s0dots);
\draw[-{Stealth[length=1.4mm]}, draw=navy, line width=0.5pt] (s0dots) -- (s0c);
\draw[-{Stealth[length=1.4mm]}, draw=navy, line width=0.5pt] (s0c) -- (s0d);
\draw[-{Stealth[length=1.5mm]}, draw=navy, line width=0.6pt] (s0d) -- (p0);

% Segment 1
\node[font=\scriptsize\bfseries\color{teal}, anchor=west] at (0.0, 1.25) {Segment $s{=}1$:};
\node[font=\scriptsize\color{ink2}, anchor=west] at (1.85, 1.25) {$\mathrm{acc}_1 \leftarrow +0$};

\node[kstep1, anchor=west] (s1a) at (0.0, 0.67) {$k_L$};
\node[kstep1, right=0.22cm of s1a] (s1b) {$k_{L{+}1}$};
\node[font=\scriptsize\color{teal}, right=0.20cm of s1b] (s1dots) {$\cdots$};
\node[kstep1, right=0.20cm of s1dots] (s1c) {$k_{2L{-}2}$};
\node[kstep1, right=0.22cm of s1c] (s1d) {$k_{2L{-}1}$};
\node[font=\scriptsize\bfseries\color{teal}, right=0.22cm of s1d] (p1) {$p_1 \in \text{fp32}$};

\draw[-{Stealth[length=1.4mm]}, draw=teal, line width=0.5pt] (s1a) -- (s1b);
\draw[-{Stealth[length=1.4mm]}, draw=teal, line width=0.5pt] (s1b) -- (s1dots);
\draw[-{Stealth[length=1.4mm]}, draw=teal, line width=0.5pt] (s1dots) -- (s1c);
\draw[-{Stealth[length=1.4mm]}, draw=teal, line width=0.5pt] (s1c) -- (s1d);
\draw[-{Stealth[length=1.5mm]}, draw=teal, line width=0.6pt] (s1d) -- (p1);

% Fold equation box
\filldraw[fill=slatefill, draw=border, line width=0.55pt, rounded corners=1.5pt] (0.0, -0.18) rectangle (6.65, 0.30);
\node[font=\scriptsize\color{ink}] at (3.325, 0.06)
    {Canonical output: \ $y_{m,n} = \mathrm{bf16}\!\left(\big((+0 + p_0) + p_1\big) + \cdots + p_{S-1}\right)$};

% Vertical separator with wide gutters on both sides
\draw[draw=border, line width=0.6pt] (7.15, -0.18) -- (7.15, 3.55);

% ==================== Panel (b): Schedule-Free Kernel Space ====================
\def\xB{7.58}
\def\xTxt{10.92}
\node[paneltitle] at (\xB, 3.45) {(b) Schedule-Free Execution Space (0 bit drift)};

% Sub-diagram 1: M x N tiling & Hopper 2x1 cluster
\filldraw[fill=navyfill, draw=navy, line width=0.55pt] (\xB, 2.25) rectangle (\xB + 2.94, 2.97);
\draw[draw=navy, line width=0.38pt]
    (\xB + 0.49, 2.25) -- (\xB + 0.49, 2.97)
    (\xB + 0.98, 2.25) -- (\xB + 0.98, 2.97)
    (\xB + 1.47, 2.25) -- (\xB + 1.47, 2.97)
    (\xB + 1.96, 2.25) -- (\xB + 1.96, 2.97)
    (\xB + 2.45, 2.25) -- (\xB + 2.45, 2.97)
    (\xB, 2.49) -- (\xB + 2.94, 2.49)
    (\xB, 2.73) -- (\xB + 2.94, 2.73);
\draw[rust, line width=1.0pt] (\xB + 0.49, 2.49) rectangle (\xB + 1.47, 2.97);
\node[font=\scriptsize\bfseries\color{ink}, anchor=west] at (\xTxt, 2.86) {1. Arbitrary $M{\times}N$ Tiling \& Clusters};
\node[font=\scriptsize\color{ink2}, anchor=west] at (\xTxt, 2.54) {Warp streaming ($M{\le}8$), CUTLASS tiles,};
\node[font=\scriptsize\color{ink2}, anchor=west] at (\xTxt, 2.24) {Hopper $2{\times}1$ TMA multicast \& \texttt{wgmma}.};

% Sub-diagram 2: Deterministic Stream-K hand-off (wide 0.74 cm arrow gap for fp32 label!)
\node[ctanode, minimum width=1.10cm, draw=navy, fill=navyfill, text=navy, anchor=west] (cw0) at (\xB, 1.40) {CTA $w$};
\node[ctanode, minimum width=1.12cm, draw=rust, fill=rustfill, text=rust, anchor=east] (cw1) at (\xB + 2.94, 1.40) {CTA $w{+}1$};
\draw[-{Stealth[length=1.6mm]}, draw=rust, line width=0.75pt] (cw0.east) -- node[above=2.5pt, font=\tiny\bfseries\color{rust}] {\texttt{fp32}} (cw1.west);
\node[font=\scriptsize\bfseries\color{ink}, anchor=west] at (\xTxt, 1.68) {2. Deterministic Stream-K};
\node[font=\scriptsize\color{ink2}, anchor=west] at (\xTxt, 1.36) {Exact \texttt{fp32} accumulator hand-off mid-chain;};
\node[font=\scriptsize\color{ink2}, anchor=west] at (\xTxt, 1.06) {SM load balance with zero split-K merge.};

% Sub-diagram 3: TP-degree invariance (spacious tpnode boxes!)
\node[tpnode, anchor=west] (tp0) at (\xB, 0.22) {$r_0$};
\node[tpnode, anchor=west] (tp1) at (\xB + 0.90, 0.22) {$r_1$};
\node[tpnode, anchor=east] (tp2) at (\xB + 2.94, 0.22) {$r_{t{-}1}$};
\node[font=\scriptsize\color{teal}] at ($ (tp1.east)!0.5!(tp2.west) $) {$\cdots$};
\node[font=\scriptsize\bfseries\color{ink}, anchor=west] at (\xTxt, 0.44) {3. TP-Degree Invariance ($t \le t_{\max}$)};
\node[font=\scriptsize\color{ink2}, anchor=west] at (\xTxt, 0.10) {Column-parallel whole-$K$ shards; bit-exact \texttt{all-gather}.};

\end{tikzpicture}
"""
    compile_tikz(tex, "fig2_gemm_program")


# ------------------------------------------------------------------ Figure 3 (Figure 4): detcheck matrix
def fig_determinism():
    checks = [
        "Batch\nc=4", "Batch\nc=16", "Shared\nprefix", "Decode vs.\nre-prefill",
        "Seeded\nre-prefill", "Repeat\n×200", "Seeded\nsample c=4",
        "Seeded\nsample c=16", "Prefix\nfull hit", "Prefix\npart. hit",
    ]
    rows = [
        ("Stock engine (FlashInfer + cuBLAS)", [16, 16, 32, 24, 24, 168, 16, 16, 5, 6]),
        ("SGLang 0.5.5 (stock)", [16, 16, 32, 24, 24, 32, 16, 16, 8, 8]),
        ("vLLM 0.17.1 (stock)", [16, 16, 32, 24, 24, 166, 16, 16, 8, 6]),
        ("SGLang 0.5.5 deterministic", [5, 5, 30, 24, 24, 0, 4, 6, None, None]),
        ("vLLM 0.17.1 batch-invariant", [0, 0, 0, 0, 8, 0, 0, 0, 0, 0]),
        ("PI-Attn (--deterministic)", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    ]
    tot = [16, 16, 32, 24, 24, 199, 16, 16, 8, 8]
    frac = np.array([[np.nan if v is None else v / t for v, t in zip(r[1], tot)] for r in rows])

    cmap = LinearSegmentedColormap.from_list(
        "det", ["#F0F6FA", "#F8DDD6", "#E08875", "#B54434"]
    )
    fig, ax = plt.subplots(figsize=(7.05, 2.55))
    m = np.ma.masked_invalid(frac)
    cm = cmap.copy()
    cm.set_bad("#ECEEF2")
    ax.imshow(m, cmap=cm, vmin=0, vmax=1, aspect="auto")

    for i, (_, vals) in enumerate(rows):
        is_pi = i == len(rows) - 1
        for j, v in enumerate(vals):
            if is_pi:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1.0, 1.0,
                                           facecolor="#DCE8F8", edgecolor="none", zorder=2))
            if v is None:
                txt, col, wt = "disabled", MUTED, "normal"
            elif v == 0:
                txt = "1 / 200" if j == 5 else f"0 / {tot[j]}"
                col = NAVY if is_pi else TEAL
                wt = "bold" if is_pi else "normal"
            else:
                txt = f"{v + 1} uniq" if j == 5 else f"{v} / {tot[j]}"
                col = "#FFFFFF" if frac[i, j] > 0.52 else INK
                wt = "normal"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.0, color=col, fontweight=wt, zorder=4)

    ax.set_xticks(range(len(checks)), checks, fontsize=6.9)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7.6)
    ax.get_yticklabels()[-1].set_color(NAVY)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.tick_params(axis="x", length=0, pad=5)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(checks)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color=SURF, linewidth=2.0, linestyle="-")
    ax.tick_params(which="minor", length=0)
    ax.add_patch(plt.Rectangle((-0.48, len(rows) - 1.48), len(checks) - 0.04, 0.96,
                               fill=False, edgecolor=NAVY, linewidth=1.25, zorder=5, clip_on=False))
    ax.set_xlabel(
        "Requests diverging in token IDs or fp32 log-probability bits (Repeat ×200: distinct outputs observed; 1 / 200 = exact)",
        fontsize=7.2, color=INK2, labelpad=8,
    )
    save(fig, "fig3_determinism")


# ------------------------------------------------------------------ Figure 4 (Figure 6): A800 & H100 Cross-Engine Baselines
def fig_baselines():
    wl = ["64/256\nc=1", "512/256\nc=8", "512/256\nc=32", "2048/256\nc=16", "4096/128\nc=4"]
    fi_a, pi_a = read_a800_r6_17b_greedy()
    a800 = {
        "stock": fi_a,
        "PI": pi_a,
        "vllm": np.array([270.0, 1842.0, 4890.0, 1885.0, 553.0]),
        "vllm_bi": np.array([80.0, 609.0, 2228.0, 921.0, 223.0]),
        "sgl": np.array([253.0, 1768.0, 4865.0, 1837.0, 530.0]),
        "sgl_det": np.array([73.0, 554.0, 1993.0, 843.0, 192.0]),
    }
    h100 = {
        "stock": np.array([436.0, 2998.0, 8537.0, 3375.0, 974.0]),
        "PI": np.array([455.0, 2946.0, 8624.0, 3400.0, 966.0]),
        "vllm": np.array([442.0, 3066.0, 8738.0, 3648.0, 1081.0]),
        "vllm_bi": np.array([149.0, 1168.0, 4442.0, 1889.0, 467.0]),
        "sgl": np.array([407.0, 2758.0, 7812.0, 3296.0, 956.0]),
        "sgl_det": np.array([243.0, 1710.0, 5446.0, 2311.0, 577.0]),
    }
    series = [
        ("PI", "PI-Attn (deterministic)", NAVY_FILL, NAVY, None),
        ("vllm_bi", "vLLM 0.17.1 batch-invariant", RUST_FILL, RUST, None),
        ("sgl_det", "SGLang 0.5.5 deterministic", AMBER_FILL, AMBER, None),
        ("vllm", "vLLM 0.17.1 (non-determ.)", SLATE_LIGHT, SLATE, "//"),
        ("sgl", "SGLang 0.5.5 (non-determ.)", "#F3F4F6", MUTED, ".."),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.50), sharey=True, gridspec_kw={"wspace": 0.16})
    x = np.arange(len(wl))
    w = 0.152
    for ax, data, title in [
        (axes[0], a800, "(a) NVIDIA A800 (Ampere sm_80)"),
        (axes[1], h100, "(b) NVIDIA H100 (Hopper sm_90)"),
    ]:
        base = data["stock"]
        for k, (key, lab, fc, ec, hat) in enumerate(series):
            vals = data[key] / base
            xs = x + (k - 2) * w
            ax.bar(
                xs, vals, w * 0.86, color=fc, edgecolor=ec, linewidth=0.6,
                hatch=hat, label=lab, zorder=3,
            )
        ax.axhline(1.0, color=INK2, lw=0.75, ls=(0, (3, 2)), zorder=2)
        ax.set_xticks(x, wl, fontsize=6.9)
        ax.set_ylim(0, 1.28)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
        ax.set_yticklabels(["0.0×", "0.25×", "0.50×", "0.75×", "1.00×", "1.25×"])
        hgrid(ax)
        ax.set_title(title, pad=8)
    axes[0].set_ylabel("Throughput (stock engine = 1.00×)", labelpad=5)
    axes[0].legend(
        ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.34),
        fontsize=7.1, handlelength=1.15, columnspacing=1.15,
    )
    save(fig, "fig4_baselines")


# ------------------------------------------------------------------ Figure 5: End-to-End Performance & Prefill Scaling
def fig_e2e():
    rows = [
        l for l in (RES / "a800_r6/e2e_matrix/SUMMARY.md").read_text().splitlines()
        if l.startswith("| ") and "model" not in l and "---" not in l
    ]
    models = ["0.6B", "1.7B", "4B", "8B"]
    wls = ["64/256 c1", "512/256 c8", "512/256 c32", "2048/256 c16", "4096/128 c4"]
    M = np.full((4, 5), np.nan)
    for r in rows:
        c = [x.strip() for x in r.strip("|").split("|")]
        if c[1] != "0":
            continue
        M[models.index(c[0]), wls.index(c[2])] = float(c[5].rstrip("%"))
    assert not np.isnan(M).any()

    fig = plt.figure(figsize=(7.05, 2.50))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.0], wspace=0.36)

    # Panel (a): 4x5 End-to-End Matrix
    ax = fig.add_subplot(gs[0])
    cmap = LinearSegmentedColormap.from_list(
        "div", ["#A63A2B", "#E8B4A8", "#F4F5F7", "#9BBCE0", "#1B3A6B"]
    )
    norm = TwoSlopeNorm(vmin=-16, vcenter=0, vmax=16)
    ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
    for i in range(4):
        for j in range(5):
            v = M[i, j]
            col = "#FFFFFF" if abs(v) > 9.5 else INK
            wt = "bold" if v >= 5.0 else "normal"
            ax.text(j, i, f"{v:+.1f}%", ha="center", va="center", fontsize=7.2, color=col, fontweight=wt)
    ax.set_xticks(range(5), [w.replace(" c", "\nc=") for w in wls], fontsize=6.9)
    ax.set_yticks(range(4), ["Qwen3-" + m for m in models], fontsize=7.6)
    ax.tick_params(axis="x", length=0, pad=5)
    ax.tick_params(axis="y", length=0, pad=5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 5), minor=True)
    ax.set_yticks(np.arange(-0.5, 4), minor=True)
    ax.grid(which="minor", color=SURF, linewidth=2.0, linestyle="-")
    ax.tick_params(which="minor", length=0)
    ax.set_title("(a) End-to-end Δ throughput on A800 (+3.3%)", pad=8)

    # Panel (b): Long-context prefill TTFT scaling (512 to 30K tokens)
    ax = fig.add_subplot(gs[1])

    def parse_ttft_ratio(base_dir: Path, model: str):
        txt = (base_dir / "e2e_extended/SUMMARY.md").read_text().split(f"### Qwen3-{model}")[1].split("###")[0]
        pts = []
        for l in txt.splitlines():
            c = [x.strip() for x in l.strip("|").split("|")]
            if len(c) == 5 and c[1] == "TTFT ms" and c[0].count("/") == 2:
                plen = int(c[0].split("/")[0])
                if plen < 512:
                    continue
                fi_med = float(c[2].split()[0])
                pi_med = float(c[3].split()[0])
                pi_lo, pi_hi = [float(z) for z in c[3].split("[")[1].rstrip("]").split("–")]
                pts.append((plen, (pi_med / fi_med - 1.0) * 100.0,
                            (pi_lo / fi_med - 1.0) * 100.0, (pi_hi / fi_med - 1.0) * 100.0))
        return pts

    a17 = parse_ttft_ratio(RES / "a800_r6", "1.7B")
    a8 = parse_ttft_ratio(RES / "a800_r6", "8B")
    r17 = parse_ttft_ratio(RES, "1.7B")

    curves = [
        (a8, RUST, "s", "-", r"Qwen3-8B (A800, $\leq$16K)"),
        (a17, NAVY, "o", "-", "Qwen3-1.7B (A800)"),
        (r17, TEAL, "^", "--", "Qwen3-1.7B (RTX 4090 D)"),
    ]
    for pts, col, mk, ls, lab in curves:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ylo = [p[1] - p[2] for p in pts]
        yhi = [p[3] - p[1] for p in pts]
        ax.errorbar(
            xs, ys, yerr=[ylo, yhi], color=col, lw=1.4, ls=ls,
            marker=mk, ms=4.2, mec=SURF, mew=0.7, capsize=2.2, elinewidth=0.8,
            label=lab, zorder=3,
        )

    ax.set_xscale("log")
    ax.set_xticks([512, 2048, 4096, 16384, 30000], ["512", "2K", "4K", "16K", "30K"])
    ax.minorticks_off()
    ax.axhline(0.0, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=2)
    ax.set_xlim(360, 45000)
    ax.set_ylim(-6.0, 14.5)
    ax.set_yticks([-5, 0, 5, 10])
    ax.set_yticklabels(["−5%", "0%", "+5%", "+10%"])
    ax.set_ylabel("Δ Time to first token (%)", labelpad=4)
    ax.set_xlabel("Prompt length (tokens)", labelpad=4)
    hgrid(ax)
    ax.legend(loc="upper left", fontsize=6.8, handlelength=1.6)
    ax.set_title("(b) Long-prompt TTFT scaling (512–30K)", pad=8)

    save(fig, "fig5_e2e")


# ------------------------------------------------------------------ Figure 6 (Figure 8): Rollout = Trainer (Appendix)
def fig_rollout():
    models = ["0.6B", "1.7B", "4B", "8B"]
    pi_eq, hf_eq, hf_rmin, hf_rmax, toks = [], [], [], [], []
    for m in models:
        d = json.loads((RES / f"a800/rollout/Qwen3-{m}.json").read_text())
        n = d["pi alone"]["tokens"]
        toks.append(n)
        pi_eq.append(min(d[k]["bitwise_equal"] for k in ("pi alone", "pi packing 0", "pi packing 1")) / n * 100)
        hf_eq.append(d["hf alone"]["bitwise_equal"] / n * 100)
        hf_rmin.append(min(d["hf alone"]["ratio_min"], d["hf batch 8"]["ratio_min"]))
        hf_rmax.append(max(d["hf alone"]["ratio_max"], d["hf batch 8"]["ratio_max"]))
    assert sum(toks) == 55593, sum(toks)

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.25), gridspec_kw={"wspace": 0.36})
    x = np.arange(4)
    w = 0.34

    ax = axes[0]
    ax.bar(x - w / 2, pi_eq, w * 0.86, color=NAVY_FILL, edgecolor=NAVY, lw=0.65,
           label="PI-Attn trainer forward (any micro-batch packing)", zorder=3)
    ax.bar(x + w / 2, hf_eq, w * 0.86, color=SLATE_FILL, edgecolor=SLATE, lw=0.65,
           label="Standard HuggingFace BF16 forward", zorder=3)
    for xi, v in zip(x, hf_eq):
        ax.text(xi - w / 2, 102.5, "100%", ha="center", fontsize=6.6, fontweight="bold", color=NAVY)
        ax.text(xi + w / 2, v + 2.5, f"{v:.1f}%", ha="center", fontsize=6.5, color=INK2)
    ax.set_xticks(x, ["Qwen3-" + m for m in models], fontsize=7.2)
    ax.set_ylim(0, 120)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Bitwise-identical log-probs (%)", labelpad=4)
    hgrid(ax)
    ax.set_title("(a) Exact rollout–trainer bit match", pad=8)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.30), ncol=2, fontsize=7.2, handlelength=1.1)

    ax = axes[1]
    ax.axhline(1.0, color=NAVY, lw=1.0, ls="--", zorder=2)
    for xi, rlo, rhi in zip(x, hf_rmin, hf_rmax):
        ax.plot([xi + 0.12, xi + 0.12], [rlo, rhi], color=RUST, lw=2.2, zorder=3)
        ax.scatter([xi + 0.12, xi + 0.12], [rlo, rhi], color=RUST_FILL, edgecolors=RUST, s=18, zorder=4)
        ax.text(xi + 0.12, rhi + 0.06, f"{rhi:.2f}×", ha="center", va="bottom", fontsize=6.3, color=RUST)
        ax.text(xi + 0.12, rlo - 0.06, f"{rlo:.2f}×", ha="center", va="top", fontsize=6.3, color=INK2)
        ax.scatter([xi - 0.14], [1.0], color=NAVY_FILL, edgecolors=NAVY, marker="D", s=22, zorder=5)
        ax.text(xi - 0.14, 1.07, "1.00×", ha="center", va="bottom", fontsize=6.2, fontweight="bold", color=NAVY)
    ax.set_xticks(x, ["Qwen3-" + m for m in models], fontsize=7.2)
    ax.set_xlim(-0.5, 3.55)
    ax.set_ylim(0.28, 2.82)
    ax.set_yticks([0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_yticklabels(["0.5×", "1.0×", "1.5×", "2.0×", "2.5×"])
    ax.set_ylabel("Observed importance ratio $r_t$ range", labelpad=4)
    hgrid(ax)
    ax.set_title("(b) Extreme importance-ratio drift ($[r_{\\min}, r_{\\max}]$)", pad=8)
    save(fig, "fig6_rollout")


# ------------------------------------------------------------------ Figure 7 (Figure 10): Batch-1 Step Breakdown
def fig_breakdown():
    txt = (RES / "a800/nsys/b1_compare.txt").read_text()
    models = ["0.6B", "1.7B", "4B", "8B"]
    comps = [
        ("attention", "Attention", NAVY_FILL, NAVY),
        ("projections", "Projections (GEMM)", TEAL_FILL, TEAL),
        ("other kernels", "Norm / RoPE / MLP", SLATE_FILL, SLATE),
        ("between kernels", "Inter-kernel launch", AMBER_FILL, AMBER),
    ]
    vals = {c[0]: [] for c in comps}
    for blk in txt.split("== ")[1:]:
        for key, _, _, _ in comps:
            m = re.search(rf"^\s+{key}\s+[\d.]+ ->\s+[\d.]+\s+([+-][\d.]+)", blk, re.M)
            vals[key].append(float(m.group(1)))

    fig, ax = plt.subplots(figsize=(7.05, 2.25))
    x = np.arange(4)
    w = 0.185
    for k, (key, lab, fc, ec) in enumerate(comps):
        xs = x + (k - 1.5) * w
        ax.bar(xs, vals[key], w * 0.86, color=fc, edgecolor=ec, lw=0.6, label=lab, zorder=3)
    ax.axhline(0, color=INK, lw=0.75, zorder=4)
    ax.set_xticks(x, ["Qwen3-" + m for m in models])
    ax.set_ylabel("PI-Attn − Stock (µs / decode step)", labelpad=5)
    hgrid(ax)
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0.0, 1.20), fontsize=7.1, handlelength=1.0)
    ax.set_ylim(-750, 155)
    ax.text(3.42, 95, r"$\uparrow$ slower    $\downarrow$ faster (PI-Attn saves time)", fontsize=6.9, color=INK2, ha="right")
    save(fig, "fig7_breakdown")


# ------------------------------------------------------------------ Figure 8 (Figure 9): Cross-GPU Identity
def fig_xgpu():
    cats = [
        "GEMM suite\n(108 shapes, A10G/L4)",
        "Attention suite\n(5 shapes, A10G/L4)",
        "GSM8K generations\nQwen3-1.7B (4090 D)",
        "GSM8K generations\nQwen3-8B (4090 D)",
    ]
    pi = [100.0, 100.0, 100.0, 100.0]
    base = [39 / 108 * 100, 0.0, (1319 - 910) / 1319 * 100, (500 - 343) / 500 * 100]
    cnt_p = ["108 / 108", "5 / 5", "1319 / 1319", "500 / 500"]
    cnt_b = ["cuBLAS (L4): 39 / 108", "FlashInfer (L4): 0 / 5", "Stock engine: 409 / 1319", "Stock engine: 157 / 500"]

    fig, ax = plt.subplots(figsize=(7.05, 2.10))
    y = np.arange(4)[::-1]
    h = 0.34
    ax.barh(y + h / 2, pi, h * 0.86, color=NAVY_FILL, edgecolor=NAVY, lw=0.65, label="PI-Attn", zorder=3)
    ax.barh(y - h / 2, base, h * 0.86, color=SLATE_FILL, edgecolor=SLATE, lw=0.65, label="Stock kernels / engine", zorder=3)
    for yi, bv, cp, cb in zip(y, base, cnt_p, cnt_b):
        ax.text(101.5, yi + h / 2, cp, va="center", fontsize=6.9, color=NAVY, fontweight="bold")
        ax.text(bv + 1.5, yi - h / 2, cb, va="center", fontsize=6.7, color=INK2)
    ax.set_yticks(y, cats, fontsize=7.2)
    ax.set_xlim(0, 124)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Bitwise-identical outputs relative to NVIDIA A800 (sm_80)", labelpad=4)
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.04), ncol=2, fontsize=7.2, handlelength=1.0)
    save(fig, "fig8_xgpu")


# ------------------------------------------------------------------ Figure 9 (Figure 7): Exact GDN Decode & Throughput
def fig_gdn():
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.30), gridspec_kw={"width_ratios": [1.08, 1.0], "wspace": 0.38})

    # Panel (a): Bitwise agreement (2048 tokens)
    ax = axes[0]
    cfg = ["Stock\nGDN", "Exact GDN\nonly", "Stock GDN +\nbatch-inv.", "Exact GDN +\nbatch-inv."]
    v08 = [274, 264, 311, 2048]
    v4 = [120, 131, 181, 2048]
    x = np.arange(4)
    w = 0.36
    ax.bar(x - w / 2, np.array(v08) / 20.48, w * 0.84, color=NAVY_FILL, edgecolor=NAVY, lw=0.65, label="Qwen3.5-0.8B", zorder=3)
    ax.bar(x + w / 2, np.array(v4) / 20.48, w * 0.84, color=TEAL_FILL, edgecolor=TEAL, lw=0.65, label="Qwen3.5-4B", zorder=3)
    for xi, a, b in zip(x, v08, v4):
        wt = "bold" if a == 2048 else "normal"
        dx = 0.03 if a == 2048 else 0.0
        ax.text(xi - w / 2 - dx, a / 20.48 + 2.5, "100%" if a == 2048 else f"{a/20.48:.0f}%",
                ha="center", fontsize=6.2, color=NAVY if a == 2048 else INK2, fontweight=wt)
        ax.text(xi + w / 2 + dx, b / 20.48 + 2.5, "100%" if b == 2048 else f"{b/20.48:.0f}%",
                ha="center", fontsize=6.2, color=TEAL if b == 2048 else INK2, fontweight=wt)
    ax.set_xticks(x, cfg, fontsize=6.8)
    ax.set_ylim(0, 122)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Decode = prefill log-prob\nbitwise (% of 2,048 tokens)", labelpad=4)
    hgrid(ax)
    ax.legend(loc="upper left", fontsize=6.8, handlelength=0.95)
    ax.set_title("(a) Bitwise exactness in vLLM 0.17.1", pad=8)

    # Panel (b): Relative throughput in vLLM 0.17.1 on Qwen3.5-4B
    ax = axes[1]
    wls = ["c=1\ndecode", "c=8\ndecode", "c=256\nserving", "c=256\n+ MTP spec"]
    stock_tok = np.array([139.8, 937.4, 6439.0, 5246.0])
    vllm_bi_tok = np.array([38.0, 290.2, 5336.0, 4573.0])
    pi_tok = np.array([136.4, 966.4, 6501.0, 5651.0])

    xb = np.arange(4)
    bw = 0.34
    r_pi = pi_tok / stock_tok
    r_bi = vllm_bi_tok / stock_tok
    ax.bar(xb - bw / 2, r_pi, bw * 0.85, color=NAVY_FILL, edgecolor=NAVY, lw=0.65,
           label="Exact GDN + PI-Attn GEMM", zorder=3)
    ax.bar(xb + bw / 2, r_bi, bw * 0.85, color=RUST_FILL, edgecolor=RUST, lw=0.65,
           label="Exact GDN + vLLM batch-inv.", zorder=3)
    for xi, vp, vb in zip(xb, r_pi, r_bi):
        y_p = max(vp + 0.035, 1.035) if vp >= 0.95 else vp + 0.03
        ax.text(xi - bw / 2, y_p, f"{vp:.2f}×", ha="center", fontsize=6.3, fontweight="bold", color=NAVY)
        ax.text(xi + bw / 2, vb + 0.03, f"{vb:.2f}×", ha="center", fontsize=6.2, color=INK2)
    ax.axhline(1.0, color=INK2, lw=0.75, ls=(0, (3, 2)), zorder=2)
    ax.set_xticks(xb, wls, fontsize=6.8)
    ax.set_ylim(0, 1.50)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["0.0×", "0.5×", "1.0×"])
    ax.set_ylabel("Throughput vs. stock vLLM", labelpad=4)
    hgrid(ax)
    ax.legend(loc="upper left", fontsize=6.5, handlelength=0.95)
    ax.set_title("(b) Qwen3.5-4B throughput (vLLM)", pad=8)

    save(fig, "fig9_gdn")


if __name__ == "__main__":
    all_figs = (
        fig_teaser, fig_attention, fig_gemm, fig_determinism,
        fig_baselines, fig_e2e, fig_rollout, fig_breakdown, fig_xgpu, fig_gdn,
    )
    for fn in all_figs:
        if not sys.argv[3:] or fn.__name__ in sys.argv[3:]:
            fn()
    print("all figures generated ok")
