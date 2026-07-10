#!/usr/bin/env python
"""Figures for the through-zero flanging writeup (through-zero-flanging.md).

    venv/bin/python design/writeups/figures/make_figures.py

See _style.py for the shared house aesthetic.
"""
import numpy as np
from _style import apply_style, figdir_of, finish, tidy, boxed_legend, K, G, L, FILL
import matplotlib.pyplot as plt

apply_style()
FIGDIR = figdir_of(__file__)


# =========================================================================
# Figure 1 — the comb.  |H(f)| = 2|cos(pi f Delta)|.  As Delta -> 0 the
# notches spread out and the response opens to a flat 2x (full reinforce).
# =========================================================================
def fig_comb():
    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    f = np.linspace(0, 6.0, 4000)        # kHz
    for delta, col, lab in [
        (0.50, K, r"$\Delta = 0.5\ \mathrm{ms}$"),
        (0.20, G, r"$\Delta = 0.2\ \mathrm{ms}$"),
    ]:
        ax.plot(f, 2 * np.abs(np.cos(np.pi * f * delta)), color=col, label=lab)
    ax.axhline(2.0, color=L, linestyle=(0, (1, 1.5)), linewidth=0.8)
    ax.text(0.12, 2.07, r"$\Delta \to 0$  (full reinforcement)", ha="left",
            va="bottom", color="0.45", fontsize=8.5)
    ax.annotate(r"$f_{1}=\dfrac{1}{2\Delta}$", xy=(1.0, 0.0), xytext=(1.55, 0.62),
                fontsize=9, color=K,
                arrowprops=dict(arrowstyle="-", color="0.4", linewidth=0.6))
    ax.set_xlim(0, 6.0)
    ax.set_ylim(0, 2.25)
    ax.set_xlabel(r"frequency  $f$  (kHz)")
    ax.set_ylabel(r"$|H(f)|$")
    ax.set_yticks([0, 1, 2])
    boxed_legend(ax, loc="upper right", handlelength=1.6, borderaxespad=0.4)
    tidy(ax)
    finish(fig, "comb", FIGDIR)


# =========================================================================
# Figure 2 — what "through zero" is.  Notch centre frequencies over one LFO
# cycle:  f_k(t) = (2k+1)/(2|Delta(t)|).  Ordinary flanging keeps Delta
# bounded above zero -> bounded bands.  Through-zero lets Delta cross zero
# -> every notch sweeps up through the ceiling.
# =========================================================================
def fig_notch_fan():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=True)
    t = np.linspace(0, 1, 1600)          # LFO cycles
    fmax = 8.0
    ks = range(6)

    d_ord = 0.8 + 0.4 * np.cos(2 * np.pi * t)          # in [0.4, 1.2] ms
    for k in ks:
        a1.plot(t, (2 * k + 1) / (2 * d_ord), color=K if k == 0 else G,
                linewidth=0.9 if k == 0 else 0.7)
    a1.set_title(r"ordinary flanger", fontsize=10)
    a1.text(0.5, 7.4, r"$\Delta(t)>0$  always", ha="center", fontsize=8.5,
            color="0.35")

    d_tz = 1.2 * np.sin(2 * np.pi * t)                 # crosses zero at t=0.5
    eps = 1e-3
    for k in ks:
        fk = (2 * k + 1) / (2 * np.abs(d_tz) + eps)
        fk[fk > fmax * 1.4] = np.nan
        a2.plot(t, fk, color=K if k == 0 else G,
                linewidth=0.9 if k == 0 else 0.7)
    a2.axvline(0.5, color=L, linestyle=(0, (1, 1.5)), linewidth=0.8)
    a2.set_title(r"through-zero flanger", fontsize=10)
    a2.text(0.5, 7.4, r"$\Delta\!\to\!0$", ha="center", fontsize=8.5, color="0.35",
            bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))

    for ax in (a1, a2):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, fmax)
        ax.set_xlabel(r"time  (LFO cycles)")
        ax.set_xticks([0, 0.5, 1.0])
        tidy(ax)
    a1.set_ylabel(r"notch frequency (kHz)")
    finish(fig, "notch-fan", FIGDIR)


# =========================================================================
# Figure 3 — the causality wall.  Both panels realise the SAME relative
# delay  Delta(t) = 2A sin(2 pi t / T).  Left: a causal stream reads only
# the past (d >= 0), so both taps ride a bias D >= A -> fixed latency.
# Right: a closed-form kernel evaluates x(tau - d) at any d -> latency 0.
# =========================================================================
def fig_causality_wall():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 3.1), sharey=True)
    t = np.linspace(0, 1.5, 1500)        # LFO cycles
    A = 1.0
    m = A * np.sin(2 * np.pi * t)

    # left: streaming, minimal bias D = A
    D = A
    a1.axhspan(-1.4, 0, facecolor=FILL, edgecolor="none")
    a1.text(0.75, -0.62, r"cannot read the future", ha="center", va="center",
            fontsize=8, color="0.45")
    a1.axhline(0, color="0.4", linewidth=0.6)
    a1.plot(t, D + m, color=K)
    a1.plot(t, D - m, color=G, linestyle=(0, (4, 2)))
    a1.axhline(D, color=L, linestyle=(0, (1, 1.5)), linewidth=0.8)
    a1.annotate("", xy=(0.06, 0), xytext=(0.06, D),
                arrowprops=dict(arrowstyle="<->", color="0.35", linewidth=0.7))
    a1.text(0.11, D / 2, r"latency $D\!=\!A$", ha="left", va="center",
            fontsize=8.5, color=K)
    a1.set_title(r"streaming buffer", fontsize=10)

    # right: Tropical, symmetric about zero
    a2.axhspan(-1.4, 0, facecolor=FILL, edgecolor="none")
    a2.text(0.75, -0.62, r"closed-form future (free to read)", ha="center",
            va="center", fontsize=8, color="0.45")
    a2.axhline(0, color="0.4", linewidth=0.6)
    a2.plot(t, m, color=K, label=r"$d_A(t)$")
    a2.plot(t, -m, color=G, linestyle=(0, (4, 2)), label=r"$d_B(t)$")
    a2.text(0.30, 0.10, r"latency $=0$", ha="left", va="bottom", fontsize=8.5,
            color=K)
    a2.set_title(r"Tropical kernel", fontsize=10)

    for ax in (a1, a2):
        ax.set_xlim(0, 1.5)
        ax.set_ylim(-1.4, 2.35)
        ax.set_xlabel(r"time  (LFO cycles)")
        ax.set_xticks([0, 0.5, 1.0, 1.5])
        ax.set_yticks([-1, 0, 1, 2])
        tidy(ax)
    boxed_legend(a2, loc="upper center", handlelength=1.8, borderaxespad=0.4,
                 labelspacing=0.3, ncol=2, columnspacing=1.4)
    a1.set_ylabel(r"delay  $d$  (ms)")
    finish(fig, "causality-wall", FIGDIR)


if __name__ == "__main__":
    fig_comb()
    fig_notch_fan()
    fig_causality_wall()
    print("wrote comb.svg, notch-fan.svg, causality-wall.svg to", FIGDIR)
