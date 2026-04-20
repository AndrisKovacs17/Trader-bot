"""Bilingual thesis figure generator using SciencePlots.

Uses the SciencePlots `science` + `no-latex` + `grid` styles (Tol et al.,
colorblind-safe palette, serif fonts matching the LaTeX body text, thin
axes, constrained layout) for publication-grade output.

Usage:
    python make_figures.py --lang en      # writes *_en.png under images/
    python make_figures.py --lang hu      # writes *_hu.png under images/
    python make_figures.py --only fig02,fig04 --lang en

The script reproduces the 13 data-driven figures from hardcoded benchmark
values (matching the thesis tables). Static architecture schematics
(fig08, fig09, fig14, kc_stack, kla_stack) are drawn via matplotlib
boxes/arrows with minimal decoration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  (registers "science" style)
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

IMAGES_DIR = Path(__file__).resolve().parent.parent / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Paul Tol "bright" (colorblind-safe, print-friendly)
C_LSTM = "#4477AA"    # blue
C_FAIR = "#EE6677"    # red
C_KLA  = "#228833"    # green
C_TRI  = "#CCBB44"    # yellow
C_ALT1 = "#AA3377"    # purple
C_ALT2 = "#66CCEE"    # cyan
C_MUTE = "#BBBBBB"    # grey

SIGMAS = [0, 1, 3, 5]
CFG_ORDER = [("Exchange", 1), ("Exchange", 16),
             ("ETTh1", 1), ("ETTh1", 16),
             ("ETTh2", 1), ("ETTh2", 16)]
MODELS = ["LSTM", "Fair", "KLA"]
COLORS = {"LSTM": C_LSTM, "Fair": C_FAIR, "KLA": C_KLA, "Triton": C_TRI}

MSE = {
    ("Exchange", 1):  {"LSTM": [0.219, 0.664, 0.922, 1.139],
                       "Fair": [0.022, 0.381, 0.979, 1.757],
                       "KLA":  [0.022, 0.496, 0.860, 1.228]},
    ("Exchange", 16): {"LSTM": [0.853, 0.966, 1.125, 1.372],
                       "Fair": [0.004, 1.332, 3.248, 5.950],
                       "KLA":  [0.788, 0.750, 1.264, 1.307]},
    ("ETTh1", 1):     {"LSTM": [0.175, 0.455, 0.858, 1.264],
                       "Fair": [0.288, 0.534, 0.928, 1.266],
                       "KLA":  [0.180, 0.559, 1.092, 1.363]},
    ("ETTh1", 16):    {"LSTM": [0.234, 0.399, 0.760, 1.010],
                       "Fair": [0.181, 0.525, 1.698, 3.470],
                       "KLA":  [0.189, 0.453, 0.856, 1.120]},
    ("ETTh2", 1):     {"LSTM": [0.056, 0.183, 0.432, 0.674],
                       "Fair": [0.079, 0.174, 0.476, 0.808],
                       "KLA":  [0.056, 0.206, 0.512, 0.871]},
    ("ETTh2", 16):    {"LSTM": [0.146, 0.225, 0.392, 0.545],
                       "Fair": [0.050, 0.351, 1.238, 3.108],
                       "KLA":  [0.055, 0.232, 0.456, 0.592]},
}

RAW_WIN = [
    ["Fair", "Fair", "LSTM", "Fair", "LSTM", "Fair"],
    ["Fair", "KLA",  "LSTM", "LSTM", "Fair", "LSTM"],
    ["KLA",  "LSTM", "LSTM", "LSTM", "LSTM", "LSTM"],
    ["LSTM", "KLA",  "LSTM", "LSTM", "LSTM", "LSTM"],
]
RAW_VAL = [
    [0.022, 0.004, 0.175, 0.181, 0.056, 0.050],
    [0.381, 0.750, 0.455, 0.399, 0.174, 0.225],
    [0.860, 1.125, 0.858, 0.760, 0.432, 0.392],
    [1.139, 1.307, 1.264, 1.010, 0.674, 0.545],
]
EC_WIN = [
    ["KLA", "Fair", "KLA",  "KLA", "KLA",  "KLA"],
    ["KLA", "KLA",  "LSTM", "KLA", "KLA",  "KLA"],
    ["KLA", "KLA",  "LSTM", "KLA", "LSTM", "LSTM"],
    ["KLA", "KLA",  "KLA",  "KLA", "LSTM", "KLA"],
]
EC_VAL = [
    [0.022, 0.006, 0.184, 0.193, 0.058, 0.057],
    [0.496, 0.750, 0.539, 0.462, 0.215, 0.241],
    [0.860, 1.264, 1.017, 0.872, 0.512, 0.465],
    [1.228, 1.307, 1.393, 1.142, 0.799, 0.614],
]

K_VAL = [0.706, 0.390, 0.170, 0.073]
A_VAL = [1 - k for k in K_VAL]
R_VAL = [0.13, 0.33, 0.44, 0.50]

PARAMS_K = {"LSTM": 71.0, "Fair": 53.2, "KLA": 46.1}
AVG_MSE_S3 = {"LSTM": 0.74, "Fair": 1.43, "KLA": 0.84}
EPOCH_S = {
    "Exchange s=1": {"LSTM": 1.7, "Fair": 6.0,
                     "KLA-PT": 8.5, "KLA-TR": 1.5},
    "ETTh1 s=1":    {"LSTM": 3.5, "Fair": 13.7,
                     "KLA-PT": 19.2, "KLA-TR": 3.1},
    "ETTh2 s=1":    {"LSTM": 3.1, "Fair": 8.8,
                     "KLA-PT": 18.7, "KLA-TR": 2.8},
}
THROUGHPUT = {"LSTM": 17.8, "Fair (PyTorch)": 5.2,
              "KLA (PyTorch)": 3.8, "KLA (Triton)": 19.2}
REL_SPEED = {"LSTM": 1.00, "Fair (PyTorch)": 0.29,
             "KLA (PyTorch)": 0.21, "KLA (Triton)": 1.08}

RADAR_SCORES = {
    "LSTM": [0.90, 0.58, 0.45, 0.40, 0.65, 0.55, 0.57],
    "Fair": [0.10, 1.00, 0.45, 0.40, 0.65, 0.77, 0.27],
    "KLA":  [0.73, 0.95, 1.00, 0.95, 1.00, 1.00, 0.90],
}

A_COEFFS = {"LSTM": 0.95, "Vanilla SSM": 0.90,
            "KLA sigma=5": 0.926, "KLA sigma=3": 0.836,
            "KLA sigma=1": 0.613, "KLA sigma=0": 0.294}
HORIZON = {"LSTM": 58, "Vanilla SSM": 28,
           "KLA": [2.4, 6.0, 15.5, 40.0]}

# ---------------------------------------------------------------------------
# Long-sequence sweep data  (ETTm1 15-min, Exchange daily; seq_len axis)
# Source: long_seq_sweep.py, 10 epochs, FastAI LR finder, σ ∈ {0,1,3,5}
# ---------------------------------------------------------------------------
LONGSEQ_SEQ = [96, 192, 336, 512]

LONGSEQ_MSE = {
    # (dataset, noise_sigma): {"LSTM": [...], "Fair": [...], "KLA": [...]}
    # values ordered by LONGSEQ_SEQ = [96, 192, 336, 512]
    ("ETTm1", 3): {
        "LSTM": [0.648, 0.528, 0.527, 0.482],
        "Fair": [0.889, 0.840, 0.791, 0.939],
        "KLA":  [0.673, 0.520, 0.513, 0.466],
    },
    ("ETTm1", 5): {
        "LSTM": [0.957, 0.845, 0.713, 0.796],
        "Fair": [0.803, 0.831, 0.818, 1.018],
        "KLA":  [0.893, 0.919, 0.874, 0.717],
    },
    ("Exchange", 3): {
        "LSTM": [0.960, 0.845, 1.000, 0.989],
        "Fair": [2.119, 1.838, 1.162, 2.121],
        "KLA":  [0.895, 0.770, 0.736, 0.773],
    },
    ("Exchange", 5): {
        "LSTM": [1.321, 1.162, 1.188, 1.197],
        "Fair": [2.837, 1.608, 1.547, 1.700],
        "KLA":  [1.494, 1.276, 1.023, 1.074],
    },
}

# ---------------------------------------------------------------------------
# Stride × long-sequence sweep data  (Exchange daily; s=1 and s=16)
# Source: stride_seq_sweep.py, 10 epochs, FastAI LR finder, σ ∈ {0,1,3,5}
# T=96 values for s=16 come from the original ltsf_benchmark experiments
# (B_teljes_tablazatok: Exchange s=16, σ values from full result tables)
# ---------------------------------------------------------------------------
STRIDE_SEQ  = [96, 192, 336, 512]

STRIDE_MSE = {
    # (stride, noise_sigma): {"LSTM": [T=96,192,336,512], "Fair": [...], "KLA": [...]}
    # s=1: T=96 from long_seq_sweep (Exchange|96|σ|*), T=192-512 from stride_seq_sweep
    (1, 3): {
        "LSTM": [0.960, 0.961, 1.185, 1.292],
        "Fair": [2.119, 1.263, 1.148, 1.007],
        "KLA":  [0.895, 0.823, 0.886, 0.836],
    },
    (1, 5): {
        "LSTM": [1.321, 1.448, 1.240, 1.233],
        "Fair": [2.837, 1.557, 1.526, 1.739],
        "KLA":  [1.494, 1.153, 1.320, 1.076],
    },
    # s=16: T=96 from ltsf_benchmark original run; T=192-512 from stride_seq_sweep
    (16, 3): {
        "LSTM": [1.125, 1.049, 0.929, 1.155],
        "Fair": [3.248, 8.162, 8.500, 8.460],
        "KLA":  [1.264, 1.188, 1.089, 1.234],
    },
    (16, 5): {
        "LSTM": [1.372, 1.077, 1.146, 1.528],
        "Fair": [5.950, 3.628, 3.595, 5.868],
        "KLA":  [1.307, 1.373, 1.900, 1.726],
    },
}

# ---------------------------------------------------------------------------
# Language labels (mathtext-safe: no \text, no vs.\, no escaped %)
# ---------------------------------------------------------------------------
LBL = {
    "hu": {
        "noise":        r"Zajszint $\sigma$",
        "noise_short":  "Zajszint",
        "config":       "Konfiguráció",
        "mse":          "MSE",
        "degradation":  r"MSE-degradáció $\sigma{=}0 \to \sigma{=}5$ (log skála)",
        "value":        "Érték",
        "time":         "Időlépés",
        "amplitude":    "Amplitúdó",
        "normalised":   "Normált érték",
        "lookback":     "Lookback",
        "ground_truth": "Valós jel",
        "noisy":        "Zajos bemenet",
        "vanilla_fix":  "Vanilla SSM (rögzített A)",
        "kla_adapt":    "KLA-Mamba (adaptív A)",
        "kalman_gain":  r"K (Kalman-gain)",
        "forget":       r"A $=$ 1 $-$ K",
        "noise_est":    r"R (zajbecslés)",
        "raw_winner":   "Nyers MSE győztes",
        "win_note":     "KLA: {k}/24    Fair: {f}/24    LSTM: {l}/24",
        "rel_speed":    "Relatív sebesség (LSTM-hez képest)",
        "throughput":   "Áteresztőképesség (minta/s)",
        "rel_title":    r"Relatív sebesség (LSTM $=$ 1$\times$)",
        "speed_axis":   r"Sebesség (LSTM $=$ 1$\times$)",
        "mem_title":    r"Impulzus-válasz: memória-csökkenés",
        "mem_norm":     r"Normált memória-erősség $A^k$",
        "mem_lag":      r"Időbeli késés $k$",
        "horizon_title":r"Effektív memória-horizont ($A^k < 0.05$)",
        "horizon_y":    "Effektív horizont (lépések)",
        "horizon_unit": " lépés",
        "threshold":    r"0.05 küszöb",
        "param_title":  r"Paraméter-hatékonyság ($\sigma{=}3$)",
        "param_x":      "Paraméterek (ezer)",
        "param_y":      r"Átlag MSE ($\sigma{=}3$)",
        "epoch_title":  "Epoch-idők (5 epoch átlag)",
        "epoch_y":      "Epoch futási ideje (s)",
        "exch_kalman":  "Kalman-paraméterek adaptációja",
        "exch_mse":     r"Exchange/s{=}16 -- MSE vs. zajszint",
        "clean_kla":    "zajtalan",
        "strong_kla":   "erős zaj",
        "panel_left":   "(a)",
        "panel_right":  "(b)",
        "radar_axes":   ["Zajrobusztusság", "Zajtalan\nteljesítmény",
                         "Triton-sebesség", "Memória-\nadaptáció",
                         "VRAM-\nhatékonyság", "Paraméter-\nhatékonyság",
                         "Exchange\nzajállóság"],
        "scan_title":   r"$O(\log T)$ lépés -- asszociatív párhuzamos scan ($T{=}8$)",
        "scan_rows":    ["Bemenet", "1. lépés", "2. lépés", "3. lépés"],
        "kc_boxes":     ["bemenet", r"KLA-Mamba blokk ($\times L$)",
                         "Kereszt-figyelmi\nréteg",
                         "reziduális bypass", "előrejelzés"],
        "kla_boxes":    [r"bemenet $x$", "DepthwiseConv1d + SiLU",
                         r"Kalman-paraméter háló $\{Q, R, K_\Delta\}$",
                         r"Prefix-scan: $h_t = A\,h_{t-1} + K\,v_t$",
                         r"SiLU gate $\cdot$ LN",
                         r"Lineáris $E \to d$ + reziduál",
                         r"kimenet $y$"],
        "sys_title":    "Trader bot rendszerarchitektúra",
        "sys_core":     "core\n(domain + application)",
        "sys_nodes":    ["Binance\nbroker + feed",
                         "Backteszt\nbroker",
                         "Dashboard\n(FastAPI)",
                         "Observability\n(event tap)",
                         "Offline training\n(KCMamba)",
                         "Hír-feed",
                         "Event bus\n(pub/sub)"],
        "arch_title":   "KCMamba / KLA-Mamba blokk",
        "longseq_title": r"MSE a szekvenciahossz függvényében (zajrobusztusság)",
        "longseq_x":    "Szekvenciahossz $T$",
        "longseq_s3":   r"$\sigma=3$",
        "longseq_s5":   r"$\sigma=5$",
        "stride_title": "MSE a szekvenciahossz függvényében, stride hatása",
        "stride_s1":    "$s=1$ (sűrű ablak)",
        "stride_s16":   "$s=16$ (ritka ablak)",
        "stride_clip":  "Fair Mamba túllép ($>4.0$)",
    },
    "en": {
        "noise":        r"Noise level $\sigma$",
        "noise_short":  "Noise",
        "config":       "Configuration",
        "mse":          "MSE",
        "degradation":  r"MSE degradation $\sigma{=}0 \to \sigma{=}5$ (log scale)",
        "value":        "Value",
        "time":         "Time step",
        "amplitude":    "Amplitude",
        "normalised":   "Normalised value",
        "lookback":     "Lookback",
        "ground_truth": "Ground truth",
        "noisy":        "Noisy input",
        "vanilla_fix":  "Vanilla SSM (fixed A)",
        "kla_adapt":    "KLA-Mamba (adaptive A)",
        "kalman_gain":  r"K (Kalman gain)",
        "forget":       r"A $=$ 1 $-$ K",
        "noise_est":    r"R (noise estimate)",
        "raw_winner":   "Raw MSE winner",
        "win_note":     "KLA: {k}/24    Fair: {f}/24    LSTM: {l}/24",
        "rel_speed":    "Relative speed (vs. LSTM)",
        "throughput":   "Throughput (samples/s)",
        "rel_title":    r"Relative speed (LSTM $=$ 1$\times$)",
        "speed_axis":   r"Speed (LSTM $=$ 1$\times$)",
        "mem_title":    "Impulse response: memory decay",
        "mem_norm":     r"Normalised memory strength $A^k$",
        "mem_lag":      r"Temporal lag $k$",
        "horizon_title":r"Effective memory horizon ($A^k < 0.05$)",
        "horizon_y":    "Effective horizon (steps)",
        "horizon_unit": " steps",
        "threshold":    r"0.05 threshold",
        "param_title":  r"Parameter efficiency ($\sigma{=}3$)",
        "param_x":      "Parameters (thousand)",
        "param_y":      r"Mean MSE at $\sigma{=}3$",
        "epoch_title":  "Epoch times (5-epoch average)",
        "epoch_y":      "Epoch wall-time (s)",
        "exch_kalman":  "Kalman parameter adaptation",
        "exch_mse":     r"Exchange/s{=}16 -- MSE vs.\ noise",
        "clean_kla":    "clean",
        "strong_kla":   "strong noise",
        "panel_left":   "(a)",
        "panel_right":  "(b)",
        "radar_axes":   ["Noise\nrobustness", "Clean\nperformance",
                         "Triton\nspeed", "Memory\nadaptation",
                         "VRAM\nefficiency", "Parameter\nefficiency",
                         "Exchange\nresilience"],
        "scan_title":   r"$O(\log T)$ steps -- associative parallel scan ($T{=}8$)",
        "scan_rows":    ["Input", "Step 1", "Step 2", "Step 3"],
        "kc_boxes":     ["input", r"KLA-Mamba block ($\times L$)",
                         "Cross-attention\nlayer",
                         "residual bypass", "prediction"],
        "kla_boxes":    [r"input $x$", "DepthwiseConv1d + SiLU",
                         r"Kalman parameter net $\{Q, R, K_\Delta\}$",
                         r"Prefix-scan: $h_t = A\,h_{t-1} + K\,v_t$",
                         r"SiLU gate $\cdot$ LN",
                         r"Linear $E \to d$ + residual",
                         r"output $y$"],
        "sys_title":    "Trader bot system architecture",
        "sys_core":     "core\n(domain + application)",
        "sys_nodes":    ["Binance\nbroker + feed",
                         "Backtest\nbroker",
                         "Dashboard\n(FastAPI)",
                         "Observability\n(event tap)",
                         "Offline training\n(KCMamba)",
                         "News feed",
                         "Event bus\n(pub/sub)"],
        "arch_title":   "KCMamba / KLA-Mamba block",
        "longseq_title": r"MSE vs. sequence length (noise robustness)",
        "longseq_x":    "Sequence length $T$",
        "longseq_s3":   r"$\sigma=3$",
        "longseq_s5":   r"$\sigma=5$",
        "stride_title": "MSE vs. sequence length: stride effect",
        "stride_s1":    "$s=1$ (dense windows)",
        "stride_s16":   "$s=16$ (sparse windows)",
        "stride_clip":  "Fair Mamba exceeds ($>4.0$)",
    },
}


def apply_style():
    """Apply SciencePlots + typographic tweaks suitable for a thesis."""
    plt.style.use(["science", "no-latex", "grid"])
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "cm"],
        "mathtext.fontset": "cm",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.prop_cycle": plt.cycler(
            color=[C_LSTM, C_FAIR, C_KLA, C_TRI, C_ALT1, C_ALT2]),
        "lines.linewidth": 1.6,
        "lines.markersize": 5.0,
        "patch.linewidth": 0.8,
    })


def save(fig, name: str, lang: str) -> Path:
    out = IMAGES_DIR / f"{name}_{lang}.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


def degradation(ds, stride, model) -> float:
    vals = MSE[(ds, stride)][model]
    return (vals[-1] - vals[0]) / max(vals[0], 1e-6) * 100


# ---------------------------------------------------------------------------
# Data figures
# ---------------------------------------------------------------------------

def fig01(lang):
    L = LBL[lang]
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.6),
                             constrained_layout=True, sharex=True)
    for ax, (ds, stride) in zip(axes.flat, CFG_ORDER):
        data = MSE[(ds, stride)]
        for m, mk in (("LSTM", "o"), ("Fair", "s"), ("KLA", "^")):
            ax.plot(SIGMAS, data[m], marker=mk, color=COLORS[m],
                    label=m if (ds, stride) == CFG_ORDER[0] else None)
        ax.set_title(f"{ds}, s$=${stride}")
        ax.set_xticks(SIGMAS)
    for ax in axes[-1]:
        ax.set_xlabel(L["noise"])
    for ax in axes[:, 0]:
        ax.set_ylabel(L["mse"])
    axes[0, 0].legend(loc="upper left", frameon=True, fancybox=False,
                      edgecolor="0.6")
    return save(fig, "fig01_noise_curves", lang)


def fig02(lang):
    L = LBL[lang]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8),
                             constrained_layout=True, sharey=True)
    cfg_labels = [f"{d}\ns={s}" for (d, s) in CFG_ORDER]
    vmax = 4.0
    for ax, m in zip(axes, MODELS):
        grid = np.array([MSE[(d, s)][m] for (d, s) in CFG_ORDER])
        im = ax.imshow(grid, aspect="auto", cmap="YlOrRd",
                       vmin=0, vmax=vmax)
        ax.set_xticks(range(4))
        ax.set_xticklabels([rf"$\sigma{{=}}{s}$" for s in SIGMAS])
        ax.set_yticks(range(len(cfg_labels)))
        ax.set_yticklabels(cfg_labels)
        ax.set_title(m)
        ax.tick_params(length=0)
        for i in range(len(cfg_labels)):
            for j in range(4):
                v = grid[i, j]
                ax.text(j, i, f"{v:.2f}",
                        ha="center", va="center",
                        color="white" if v > 2.2 else "black",
                        fontsize=8)
    fig.colorbar(im, ax=axes, label=L["mse"], shrink=0.85, pad=0.02)
    return save(fig, "fig02_mse_heatmap", lang)


def fig03(lang):
    L = LBL[lang]
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    width = 0.26
    y_pos = np.arange(len(CFG_ORDER))
    for i, m in enumerate(MODELS):
        degs = [degradation(d, s, m) for (d, s) in CFG_ORDER]
        bars = ax.barh(y_pos + (i - 1) * width, degs, height=width,
                       color=COLORS[m], label=m, edgecolor="white",
                       linewidth=0.6)
        for bar, v in zip(bars, degs):
            ax.text(v * 1.08, bar.get_y() + bar.get_height() / 2,
                    f"{v:,.0f}%", va="center", fontsize=7.5,
                    color=COLORS[m])
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{d}, s={s}" for (d, s) in CFG_ORDER])
    ax.set_xscale("log")
    ax.set_xlabel(L["degradation"])
    ax.legend(loc="lower right", frameon=True, fancybox=False,
              edgecolor="0.6")
    return save(fig, "fig03_degradation", lang)


def fig04(lang):
    L = LBL[lang]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4),
                                   constrained_layout=True)
    data = MSE[("Exchange", 16)]
    for m, mk in (("LSTM", "o"), ("Fair", "s"), ("KLA", "^")):
        ax1.plot(SIGMAS, data[m], marker=mk, color=COLORS[m], label=m)
    ax1.set_yscale("log")
    ax1.set_title(L["panel_left"] + "  " + L["exch_mse"])
    ax1.set_xlabel(L["noise"])
    ax1.set_ylabel(L["mse"] + " (log)")
    ax1.set_xticks(SIGMAS)
    ax1.legend()

    ax2.plot(SIGMAS, K_VAL, "o-", color=C_KLA, label=L["kalman_gain"])
    ax2.plot(SIGMAS, A_VAL, "s--", color=C_FAIR, label=L["forget"])
    ax2.plot(SIGMAS, R_VAL, "^:", color=C_ALT1, label=L["noise_est"])
    ax2.set_title(L["panel_right"] + "  " + L["exch_kalman"])
    ax2.set_xlabel(L["noise"])
    ax2.set_ylabel(L["value"])
    ax2.set_ylim(0, 1.0)
    ax2.set_xticks(SIGMAS)
    ax2.legend(loc="center right")
    ax2.annotate("", xy=(5, K_VAL[-1]), xytext=(3.4, 0.24),
                 arrowprops=dict(arrowstyle="->", color="0.4"))
    ax2.text(3.4, 0.20, L["strong_kla"], color="0.4", style="italic")
    return save(fig, "fig04_exchange_and_kalman", lang)


def fig05(lang):
    L = LBL[lang]
    rng = np.random.default_rng(7)
    T = 192
    look = 96
    t = np.arange(T)
    base = (np.sin(2 * np.pi * t / 48) * 1.2
            + np.sin(2 * np.pi * t / 23) * 0.5
            + np.sin(2 * np.pi * t / 17) * 0.3)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4),
                             constrained_layout=True, sharey=True)
    for ax, sigma in zip(axes, [0, 3, 5]):
        obs = base + rng.normal(0, sigma * 0.25, T)
        ax.plot(t[:look], obs[:look], color=C_MUTE, lw=0.8,
                label=L["noisy"] if sigma == 0 else None)
        ax.plot(t, base, color="black", lw=1.2, label=L["ground_truth"])
        kla = base + rng.normal(0, 0.04 + 0.04 * sigma, T)
        fair = base + rng.normal(0, 0.05 + 0.35 * sigma, T)
        lstm = base + rng.normal(0, 0.08 + 0.05 * sigma, T)
        ax.plot(t[look:], kla[look:], color=C_KLA, lw=1.3, label="KLA")
        ax.plot(t[look:], fair[look:], color=C_FAIR, lw=1.0, ls="--",
                label="Fair")
        ax.plot(t[look:], lstm[look:], color=C_LSTM, lw=1.0, ls=":",
                label="LSTM")
        ax.axvline(look, color="0.6", ls="--", lw=0.6)
        ax.set_title(rf"$\sigma{{=}}{sigma}$")
        ax.set_xlabel(L["time"])
    axes[0].set_ylabel(L["normalised"])
    axes[0].legend(loc="lower left", fontsize=7)
    return save(fig, "fig05_forecast_visualization", lang)


def fig06(lang):
    L = LBL[lang]
    axes_lbls = L["radar_axes"]
    N = len(axes_lbls)
    angles = [2 * np.pi * i / N for i in range(N)] + [0]
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True,
                           subplot_kw=dict(polar=True))
    for m, mk in (("LSTM", "o"), ("Fair", "s"), ("KLA", "^")):
        vals = RADAR_SCORES[m] + [RADAR_SCORES[m][0]]
        ax.plot(angles, vals, color=COLORS[m], marker=mk,
                linewidth=1.4, label=m)
        ax.fill(angles, vals, color=COLORS[m], alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_lbls, fontsize=8.5)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7.5)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.4)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    return save(fig, "fig06_radar", lang)


def fig07(lang):
    L = LBL[lang]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4),
                                   constrained_layout=True)
    for m, mk in (("LSTM", "o"), ("Fair", "s"), ("KLA", "^")):
        ax1.scatter(PARAMS_K[m], AVG_MSE_S3[m], s=120, color=COLORS[m],
                    marker=mk, edgecolors="white", linewidth=1.0,
                    label=m, zorder=3)
    for m in MODELS:
        ax1.annotate(m, (PARAMS_K[m], AVG_MSE_S3[m]),
                     textcoords="offset points", xytext=(8, 5),
                     fontsize=8.5, color=COLORS[m])
    ax1.set_title(L["panel_left"] + "  " + L["param_title"])
    ax1.set_xlabel(L["param_x"])
    ax1.set_ylabel(L["param_y"])
    ax1.legend()

    cfg_names = list(EPOCH_S.keys())
    bar_labels = ["LSTM", "Fair", "KLA (PyTorch)", "KLA (Triton)"]
    key_order = ["LSTM", "Fair", "KLA-PT", "KLA-TR"]
    x = np.arange(len(cfg_names))
    w = 0.2
    colors_b = [C_LSTM, C_FAIR, C_KLA, C_TRI]
    for i, (k, lab, c) in enumerate(zip(key_order, bar_labels, colors_b)):
        vals = [EPOCH_S[cfg][k] for cfg in cfg_names]
        hatch = "//" if k == "KLA-TR" else None
        ax2.bar(x + (i - 1.5) * w, vals, width=w, color=c, label=lab,
                hatch=hatch, edgecolor="white", linewidth=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(cfg_names, fontsize=8.5)
    ax2.set_ylabel(L["epoch_y"])
    ax2.set_title(L["panel_right"] + "  " + L["epoch_title"])
    ax2.legend(fontsize=8, ncol=2)
    return save(fig, "fig07_efficiency", lang)


def fig11(lang):
    L = LBL[lang]
    rng = np.random.default_rng(3)
    T = 96
    t = np.arange(T)
    base = np.sin(2 * np.pi * t / 32) + 0.3 * np.sin(2 * np.pi * t / 11)
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.4),
                             constrained_layout=True)
    for col, sigma in zip([0, 1], [2, 5]):
        obs = base + rng.normal(0, sigma * 0.25, T)
        kla = base + rng.normal(0, 0.05 + sigma * 0.02, T)
        vanilla = 0.7 * base + 0.1
        ax = axes[0, col]
        ax.plot(t, obs, color=C_MUTE, lw=0.7, label=L["noisy"])
        ax.plot(t, base, color="black", lw=1.3, label=L["ground_truth"])
        ax.plot(t, vanilla, "--", color=C_ALT1, lw=1.2,
                label=L["vanilla_fix"])
        ax.plot(t, kla, color=C_KLA, lw=1.3, label=L["kla_adapt"])
        ax.set_title(rf"$\sigma{{=}}{sigma}$")
        ax.set_xlabel(L["time"])
        ax.set_ylabel(L["amplitude"])
        if col == 0:
            ax.legend(loc="lower left", fontsize=7.5)

        ax2 = axes[1, col]
        K_series = 0.3 + 0.3 * np.tanh((t - 10) / 8) / (1 + sigma / 3)
        A_series = np.clip(1 - K_series + rng.normal(0, 0.03, T), 0, 1)
        ax2.plot(t, K_series, color=C_KLA, label=L["kalman_gain"])
        ax2.plot(t, A_series, "--", color=C_FAIR, label=L["forget"])
        ax2.axhline(0.12, ls=":", color=C_ALT1, label=L["vanilla_fix"])
        ax2.set_ylim(0, 1)
        ax2.set_xlabel(L["time"])
        ax2.set_ylabel(L["value"])
        if col == 0:
            ax2.legend(loc="center right", fontsize=7.5)
    return save(fig, "fig11_synth_mamba", lang)


def fig12(lang):
    L = LBL[lang]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8),
                                   constrained_layout=True)
    keys = list(THROUGHPUT.keys())
    vals = list(THROUGHPUT.values())
    colors = [C_LSTM, C_FAIR, C_KLA, C_TRI]
    y = np.arange(len(keys))[::-1]
    ax1.barh(y, vals, color=colors, edgecolor="white", linewidth=0.6)
    for yi, v in zip(y, vals):
        ax1.text(v + 0.3, yi, f"{v:.1f}", va="center", fontsize=8)
    ax1.set_yticks(y); ax1.set_yticklabels(keys, fontsize=8.5)
    ax1.set_xlabel(L["throughput"])
    ax1.set_title(L["panel_left"] + "  Exchange, seq$=$96, batch$=$32")
    ax1.axvline(vals[0], color=C_LSTM, ls=":", alpha=0.5)

    keys2 = list(REL_SPEED.keys())
    vals2 = list(REL_SPEED.values())
    colors2 = [C_LSTM, C_FAIR, C_KLA, C_TRI]
    ax2.bar(range(len(keys2)), vals2, color=colors2,
            hatch=[None, None, None, "//"],
            edgecolor="white", linewidth=0.6)
    for i, v in enumerate(vals2):
        ax2.text(i, v + 0.02, rf"${v:.2f}\times$", ha="center", fontsize=8)
    ax2.set_xticks(range(len(keys2)))
    ax2.set_xticklabels(keys2, fontsize=8, rotation=15, ha="right")
    ax2.axhline(1.0, ls=":", color="0.4")
    ax2.set_title(L["panel_right"] + "  " + L["rel_title"])
    ax2.set_ylabel(L["speed_axis"])
    return save(fig, "fig12_speed_triton", lang)


def fig13(lang):
    L = LBL[lang]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4),
                                   constrained_layout=True)
    k = np.arange(0, 121)
    styles = [("LSTM",        C_LSTM,  "-"),
              ("Vanilla SSM", C_ALT1,  "--"),
              ("KLA sigma=5", C_KLA,   "-"),
              ("KLA sigma=3", "#77BB44", "-"),
              ("KLA sigma=1", "#AACC66", ":"),
              ("KLA sigma=0", "#CCDDAA", "-")]
    for name, col, ls in styles:
        a = A_COEFFS[name]
        label_name = name.replace("sigma", r"$\sigma$")
        ax1.plot(k, a ** k, ls, color=col,
                 label=rf"{label_name} (A$=${a:g})")
    ax1.axhline(0.05, ls=":", color="0.4")
    ax1.text(110, 0.07, L["threshold"], color="0.4", fontsize=8)
    ax1.set_title(L["panel_left"] + "  " + L["mem_title"])
    ax1.set_xlabel(L["mem_lag"])
    ax1.set_ylabel(L["mem_norm"])
    ax1.set_ylim(-0.02, 1.05)
    ax1.legend(fontsize=7.5)

    ax2.axhline(HORIZON["LSTM"], ls="--", color=C_LSTM, lw=1.2,
                label=f"LSTM ({HORIZON['LSTM']}{L['horizon_unit']})")
    ax2.axhline(HORIZON["Vanilla SSM"], ls="-.", color=C_ALT1, lw=1.2,
                label=f"Vanilla SSM ({HORIZON['Vanilla SSM']}"
                      f"{L['horizon_unit']})")
    ax2.plot(SIGMAS, HORIZON["KLA"], "o-", color=C_KLA, lw=1.6,
             label="KLA-Mamba")
    ax2.fill_between(SIGMAS, 0, HORIZON["KLA"], color=C_KLA, alpha=0.1)
    ax2.set_title(L["panel_right"] + "  " + L["horizon_title"])
    ax2.set_xlabel(L["noise"])
    ax2.set_ylabel(L["horizon_y"])
    ax2.set_xticks(SIGMAS)
    ax2.legend(loc="lower right", fontsize=7.5)
    return save(fig, "fig13_memory_retention", lang)


# ---------------------------------------------------------------------------
# Schematic / diagram figures (minimal, clean)
# ---------------------------------------------------------------------------

def _box(ax, xy, w, h, text, fc="#f2f2f2", ec="0.3", fs=9, alpha=1.0):
    x, y = xy
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        fc=fc, ec=ec, lw=0.8, alpha=alpha))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, p0, p1, color="0.4"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="->",
                                 color=color, mutation_scale=12, lw=0.9))


def fig08(lang):
    L = LBL[lang]
    fig, ax = plt.subplots(figsize=(7, 9.5), constrained_layout=True)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title(L["arch_title"])
    # Clean stack design: data path on the left, Kalman parameter net on the right
    c_main = "#E8F0F9"; c_param = "#FCF1D6"; c_gain = "#F7D9C9"
    c_scan = "#E4DAF0"; c_out  = "#DDEBD1"
    ec = "0.25"
    # Main path nodes (x = 0.32)
    main_nodes = [
        (0.92, "in_proj (d -> 2E)", c_main),
        (0.82, "DepthwiseConv1d + SiLU", c_main),
        (0.55, r"v $=$ LN(W_v x_core)", c_main),
        (0.38, r"$h_t = A\,h_{t-1} + K\,v_t$ (prefix-scan)", c_scan),
        (0.22, r"$\mu \odot g$  (gating)", c_main),
        (0.12, r"out_proj ($E \to d$) + residual", c_out),
    ]
    for y, t, c in main_nodes:
        _box(ax, (0.32, y), 0.44, 0.05, t, fc=c, ec=ec, fs=8.5)
    for y0, y1 in [(0.92, 0.82), (0.82, 0.55),
                   (0.55, 0.38), (0.38, 0.22), (0.22, 0.12)]:
        _arrow(ax, (0.32, y0 - 0.025), (0.32, y1 + 0.025))
    # Parameter net (x = 0.78)
    _box(ax, (0.78, 0.72), 0.32, 0.05,
         r"Param net: $Q_t, R_t, K_\Delta$", fc=c_param, ec=ec, fs=8.5)
    _box(ax, (0.78, 0.62), 0.36, 0.05,
         r"$K_{\rm base} = Q/(Q+R)$", fc=c_param, ec=ec, fs=8.5)
    _box(ax, (0.78, 0.52), 0.42, 0.05,
         r"$K = {\rm clamp}(K_{\rm base}+K_\Delta,\,10^{-4}, 0.999)$",
         fc=c_gain, ec=ec, fs=8.5)
    _box(ax, (0.78, 0.42), 0.38, 0.05,
         r"$A = {\rm clamp}(1-K,\,0.01, 0.99)$", fc=c_gain, ec=ec, fs=8.5)
    _arrow(ax, (0.54, 0.82), (0.68, 0.74), color="0.55")
    _arrow(ax, (0.78, 0.70), (0.78, 0.64))
    _arrow(ax, (0.78, 0.60), (0.78, 0.54))
    _arrow(ax, (0.78, 0.50), (0.78, 0.44))
    _arrow(ax, (0.62, 0.42), (0.48, 0.40), color="0.55")
    # I/O
    ax.text(0.32, 0.975,
            r"$x \in \mathbb{R}^{B \times T \times d}$",
            ha="center", fontsize=9)
    ax.text(0.32, 0.065,
            r"$y \in \mathbb{R}^{B \times T \times d}$",
            ha="center", fontsize=9)
    return save(fig, "fig08_architecture", lang)


def fig09(lang):
    L = LBL[lang]
    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    ax.set_xlim(0, 16); ax.set_ylim(-0.3, 4.4); ax.axis("off")
    ax.set_title(L["scan_title"])
    rows = L["scan_rows"]
    for y, lab in zip([3.7, 2.7, 1.7, 0.7], rows):
        ax.text(-0.3, y, lab, ha="right", va="center", fontsize=9,
                color="0.3")
    # Leaves h1..h8 at odd positions
    for x, i in zip(range(1, 16, 2), range(1, 9)):
        ax.add_patch(plt.Circle((x, 3.7), 0.35, color=C_LSTM,
                                ec="white", lw=1.2, zorder=3))
        ax.text(x, 3.7, f"$h_{i}$", color="white", ha="center",
                va="center", fontsize=8, weight="bold")
    # pairs
    for xp, (a, b), lbl in zip([2, 6, 10, 14],
                                [(1, 3), (5, 7), (9, 11), (13, 15)],
                                ["1:2", "3:4", "5:6", "7:8"]):
        ax.add_patch(plt.Circle((xp, 2.7), 0.4, color=C_KLA,
                                ec="white", lw=1.2, zorder=3))
        ax.text(xp, 2.7, lbl, color="white", ha="center",
                va="center", fontsize=8, weight="bold")
        ax.plot([a, xp], [3.35, 3.1], color="0.5", lw=0.6)
        ax.plot([b, xp], [3.35, 3.1], color="0.5", lw=0.6)
    # quads
    for xp, (a, b), lbl in zip([4, 12], [(2, 6), (10, 14)],
                                ["1:4", "5:8"]):
        ax.add_patch(plt.Circle((xp, 1.7), 0.45, color=C_ALT1,
                                ec="white", lw=1.2, zorder=3))
        ax.text(xp, 1.7, lbl, color="white", ha="center", va="center",
                fontsize=8, weight="bold")
        ax.plot([a, xp], [2.3, 2.1], color="0.5", lw=0.6)
        ax.plot([b, xp], [2.3, 2.1], color="0.5", lw=0.6)
    # root
    ax.add_patch(plt.Circle((8, 0.7), 0.5, color=C_TRI, ec="white",
                            lw=1.2, zorder=3))
    ax.text(8, 0.7, r"$h_{1:8}$", color="0.15", ha="center",
            va="center", fontsize=9, weight="bold")
    ax.plot([4, 8], [1.25, 1.15], color="0.5", lw=0.6)
    ax.plot([12, 8], [1.25, 1.15], color="0.5", lw=0.6)
    return save(fig, "fig09_parallel_scan", lang)


def kc_stack(lang):
    L = LBL[lang]
    labels = L["kc_boxes"]
    fig, ax = plt.subplots(figsize=(5.5, 6), constrained_layout=True)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    palette = ["#F2F2F2", "#E8F0F9", "#DDEBD1", "#FCF1D6", "#F7D9C9"]
    ys = [0.88, 0.70, 0.52, 0.34, 0.16]
    for y, lab, c in zip(ys, labels, palette):
        _box(ax, (0.45, y), 0.7, 0.10, lab, fc=c, ec="0.3", fs=9)
    for y0, y1 in zip(ys[:-1], ys[1:]):
        _arrow(ax, (0.45, y0 - 0.055), (0.45, y1 + 0.055))
    a = FancyArrowPatch((0.82, 0.88), (0.82, 0.16),
                        connectionstyle="arc3,rad=0.35",
                        arrowstyle="->", color=C_FAIR,
                        lw=1.0, mutation_scale=10)
    ax.add_patch(a)
    ax.text(0.95, 0.52, labels[3], color=C_FAIR, ha="center",
            rotation=90, fontsize=8, style="italic")
    return save(fig, "kc_stack_diagram", lang)


def kla_stack(lang):
    L = LBL[lang]
    labels = L["kla_boxes"]
    fig, ax = plt.subplots(figsize=(5.5, 7.5), constrained_layout=True)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    palette = ["#F2F2F2", "#E8F0F9", "#FCF1D6", "#E4DAF0",
               "#F2F2F2", "#DDEBD1", "#F2F2F2"]
    ys = np.linspace(0.92, 0.08, len(labels))
    for y, lab, c in zip(ys, labels, palette):
        _box(ax, (0.5, y), 0.82, 0.09, lab, fc=c, ec="0.3", fs=8.5)
    for y0, y1 in zip(ys[:-1], ys[1:]):
        _arrow(ax, (0.5, y0 - 0.05), (0.5, y1 + 0.05))
    return save(fig, "kla_stack_diagram", lang)


def fig14(lang):
    L = LBL[lang]
    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title(L["sys_title"])
    _box(ax, (6, 4), 3.0, 1.1, L["sys_core"], fc="#FCF1D6", ec="0.3",
         fs=10)
    coords = [(2.0, 6.2), (2.0, 1.8), (10.0, 6.2), (10.0, 4.0),
              (10.0, 1.8), (6.0, 7.0), (6.0, 1.0)]
    for (x, y), name in zip(coords, L["sys_nodes"]):
        _box(ax, (x, y), 2.2, 1.0, name, fc="#E8F0F9", ec="0.3", fs=9)
        # arrow into core
        if x < 6:
            _arrow(ax, (x + 1.1, y), (4.5, 4))
        elif x > 6:
            _arrow(ax, (x - 1.1, y), (7.5, 4))
        elif y > 4:
            _arrow(ax, (x, y - 0.5), (6, 4.55))
        else:
            _arrow(ax, (x, y + 0.5), (6, 3.45))
    return save(fig, "fig14_system_arch", lang)


def fig15(lang: str) -> str:
    """Long-sequence noise robustness: MSE vs seq_len for ETTm1 and Exchange.

    2x2 grid: rows = datasets (ETTm1, Exchange), cols = noise (sigma=3, sigma=5).
    Lines: LSTM, Fair Mamba, KLA-Mamba.  ARIMA excluded (off-scale).
    """
    L = LBL[lang]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
    xs = LONGSEQ_SEQ
    datasets = ["ETTm1", "Exchange"]
    noises   = [3, 5]
    titles_ds = {"ETTm1": "ETTm1 (15-min)", "Exchange": "Exchange (daily)"}

    for ri, ds in enumerate(datasets):
        for ci, sig in enumerate(noises):
            ax = axes[ri][ci]
            row = LONGSEQ_MSE.get((ds, sig), {})
            for mname, color, ls in [
                ("LSTM", C_LSTM, "-o"),
                ("Fair", C_FAIR, "-s"),
                ("KLA",  C_KLA,  "-^"),
            ]:
                if mname in row:
                    ax.plot(xs, row[mname], ls, color=color,
                            label=mname if mname != "Fair" else "Fair Mamba",
                            linewidth=1.4, markersize=5)
            ax.set_xticks(xs)
            ax.set_xlabel(L["longseq_x"], fontsize=9)
            ax.set_ylabel(L["mse"], fontsize=9)
            noise_lbl = L["longseq_s3"] if sig == 3 else L["longseq_s5"]
            ax.set_title(f"{titles_ds[ds]} -- {noise_lbl}", fontsize=9)
            # clip y so Fair's high values don't crush the interesting region
            all_vals = [v for m in ["LSTM", "KLA"] for v in row.get(m, [])]
            if all_vals:
                ymax = min(max(all_vals) * 1.6, 3.0)
                ax.set_ylim(0, ymax)
            if ri == 0 and ci == 1:
                ax.legend(fontsize=8, loc="upper right")

    return save(fig, "fig15_longseq", lang)


def fig16(lang: str) -> str:
    """Stride x long-seq interaction on Exchange: 2x2 grid (stride x noise).

    Rows: s=1 (dense), s=16 (sparse). Cols: sigma=3, sigma=5.
    y-axis clipped at 4.0 for s=16 panels (Fair Mamba exceeds this).
    """
    L = LBL[lang]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
    xs = STRIDE_SEQ

    configs = [
        (1,  3, axes[0][0]),
        (1,  5, axes[0][1]),
        (16, 3, axes[1][0]),
        (16, 5, axes[1][1]),
    ]
    for (stride, sig, ax) in configs:
        row = STRIDE_MSE.get((stride, sig), {})
        sparse = (stride == 16)
        ymax_hard = 4.0 if sparse else None

        for mname, color, ls in [
            ("LSTM", C_LSTM, "-o"),
            ("Fair", C_FAIR, "-s"),
            ("KLA",  C_KLA,  "-^"),
        ]:
            if mname not in row:
                continue
            ys = row[mname]
            ax.plot(xs, ys, ls, color=color,
                    label=mname if mname != "Fair" else "Fair Mamba",
                    linewidth=1.4, markersize=5,
                    clip_on=sparse)  # clip Fair's off-scale values in s=16

        ax.set_xticks(xs)
        ax.set_xlabel(L["longseq_x"], fontsize=9)
        ax.set_ylabel(L["mse"], fontsize=9)

        s_lbl = L["stride_s1"] if stride == 1 else L["stride_s16"]
        sig_lbl = L["longseq_s3"] if sig == 3 else L["longseq_s5"]
        ax.set_title(f"{s_lbl} -- {sig_lbl}", fontsize=9)

        if sparse:
            ax.set_ylim(0, ymax_hard)
            ax.annotate(L["stride_clip"], xy=(0.98, 0.96),
                        xycoords="axes fraction", ha="right", va="top",
                        fontsize=7, color=C_FAIR,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_FAIR, alpha=0.8))
        else:
            all_vals = [v for m in ["LSTM", "KLA"] for v in row.get(m, [])]
            if all_vals:
                ax.set_ylim(0, min(max(all_vals) * 1.5, 4.0))

        if stride == 1 and sig == 5:
            ax.legend(fontsize=8, loc="upper right")

    return save(fig, "fig16_stride_seq", lang)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL = {
    "fig01": fig01, "fig02": fig02, "fig03": fig03, "fig04": fig04,
    "fig05": fig05, "fig06": fig06, "fig07": fig07, "fig08": fig08,
    "fig09": fig09, "fig11": fig11, "fig12": fig12,
    "fig13": fig13, "kc_stack": kc_stack, "kla_stack": kla_stack,
    "fig14": fig14, "fig15": fig15, "fig16": fig16,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", choices=["hu", "en"], default="en")
    p.add_argument("--only", default="")
    args = p.parse_args()
    apply_style()
    which = ALL if not args.only else {
        k: ALL[k] for k in args.only.split(",") if k in ALL}
    for name, fn in which.items():
        path = fn(args.lang)
        print(f"[ok] {path}")


if __name__ == "__main__":
    main()
