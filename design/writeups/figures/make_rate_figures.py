#!/usr/bin/env python
"""Figures for the rate-agnostic-consumers writeup (no-control-rate.md).

    venv/bin/python design/writeups/figures/make_rate_figures.py

See _style.py for the shared house aesthetic.
"""
import numpy as np
from _style import apply_style, figdir_of, finish, tidy, boxed_legend, K, G, L, FILL
import matplotlib.pyplot as plt

apply_style()
FIGDIR = figdir_of(__file__)


def _sig(t):
    """A smooth, slowly-varying signal (an envelope/LFO-ish thing)."""
    return 0.60 * np.sin(2 * np.pi * 0.5 * t) + 0.28 * np.sin(2 * np.pi * 0.25 * t + 0.6)


# =========================================================================
# Figure 1 — the rate hierarchy is a fact about buffers.  A block-processing
# engine only refreshes a "control-rate" signal at block boundaries and
# holds it in between, so the control tier is a lossy, quantized, delayed
# approximation of the signal -- and "audio rate" vs "control rate" hardens
# into a type with resampling glue at every seam.
# =========================================================================
def fig_rate_hierarchy():
    fig, ax = plt.subplots(figsize=(5.9, 2.7))
    t = np.linspace(0, 3.5, 2000)
    s = _sig(t)

    T = 0.5                                     # block length
    tb = np.arange(0, 3.5 + 1e-9, T)
    vb = _sig(tb)
    held = _sig(np.floor(t / T) * T)            # sample-and-hold

    ax.fill_between(t, s, held, color=FILL, edgecolor="none")
    for b in tb:
        ax.axvline(b, color=L, linestyle=(0, (1, 2)), linewidth=0.6)
    ax.plot(t, s, color=K)
    ax.step(np.append(tb, 3.5), np.append(vb, vb[-1]), where="post", color=G,
            linewidth=0.9)
    ax.plot(tb, vb, linestyle="none", marker="o", color=G, ms=2.6)

    ax.text(0.06, 0.80, r"the signal", ha="left", fontsize=9, color="0.2")
    ax.text(2.15, -0.63, r"control rate: one value per block, held", ha="center",
            fontsize=8.5, color="0.4")
    # one-block bracket
    ax.annotate("", xy=(1.0, 0.86), xytext=(1.5, 0.86),
                arrowprops=dict(arrowstyle="<->", color="0.4", linewidth=0.7))
    ax.text(1.25, 0.90, r"one block", ha="center", va="bottom", fontsize=8, color=K)

    ax.set_xlim(0, 3.5)
    ax.set_ylim(-0.95, 1.02)
    ax.set_xlabel(r"time  $\tau$")
    ax.set_ylabel(r"signal")
    ax.set_yticks([-0.5, 0, 0.5])
    tidy(ax)
    finish(fig, "rate-hierarchy", FIGDIR)


# =========================================================================
# Figure 2 — one signal, many consumers.  The closed-form signal is a single
# function; the audio DAC, a control target, and a video frame each sample
# it at their OWN rate, and every sample is the exact value on the same
# curve -- no held staircase, no rate conversion, no privileged clock.
# =========================================================================
def fig_many_consumers():
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    t = np.linspace(0, 3, 3000)
    ax.plot(t, _sig(t), color=L, linewidth=0.8)          # the function f(tau)

    def sample(step):
        ts = np.arange(0, 3 + 1e-9, step)
        return ts, _sig(ts)

    ta, va = sample(0.04)                                 # audio  (dense)
    tc, vc = sample(0.25)                                 # control (medium)
    tv, vv = sample(0.60)                                 # video  (sparse)

    ax.plot(ta, va, linestyle="none", marker=".", color=K, ms=2.2,
            label=r"audio  (48 kHz)")
    ax.plot(tc, vc, linestyle="none", marker="o", markerfacecolor="white",
            markeredgecolor=K, markeredgewidth=0.8, ms=4.2,
            label=r"control  (200 Hz)")
    ax.plot(tv, vv, linestyle="none", marker="s", color="0.35", ms=4.6,
            label=r"video  (60 Hz)")

    ax.text(0.05, -0.86, r"one function; each consumer samples the exact value at its own rate",
            ha="left", fontsize=8.5, color="0.35")

    ax.set_xlim(0, 3)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(r"time  $\tau$")
    ax.set_ylabel(r"$f(\tau)$")
    ax.set_yticks([-1, 0, 1])
    boxed_legend(ax, loc="upper right", handlelength=1.2, borderaxespad=0.4,
                 labelspacing=0.35)
    tidy(ax)
    finish(fig, "many-consumers", FIGDIR)


if __name__ == "__main__":
    fig_rate_hierarchy()
    fig_many_consumers()
    print("wrote rate-hierarchy.svg, many-consumers.svg to", FIGDIR)
