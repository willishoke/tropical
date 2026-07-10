#!/usr/bin/env python
"""Figures for the sawtooth/triangle-by-scrubbing writeup (sawtooth-is-a-line.md).

    venv/bin/python design/writeups/figures/make_waveform_figures.py

See _style.py for the shared house aesthetic.
"""
import numpy as np
from _style import apply_style, figdir_of, finish, tidy, K, G, L, FILL
import matplotlib.pyplot as plt

apply_style()
FIGDIR = figdir_of(__file__)


def _break(x, y, thresh=0.4):
    """Insert NaN at any large jump (a phase wrap or a square-wave edge), so the
    curve draws without a vertical seam at the discontinuity."""
    y = np.asarray(y, float)
    xs, ys = [x[0]], [y[0]]
    for i in range(1, len(y)):
        if abs(y[i] - y[i - 1]) > thresh:
            xs.append(x[i]); ys.append(np.nan)
        xs.append(x[i]); ys.append(y[i])
    return np.array(xs), np.array(ys)


# =========================================================================
# Figure 1 — the phase accumulator carries state.  phi[n] = phi[n-1] + inc
# (mod 1): the value at n is a function of the whole history before it, not
# of n alone.  You cannot read it anywhere; you have to step there.
# =========================================================================
def fig_accumulator():
    fig, ax = plt.subplots(figsize=(5.9, 2.7))
    inc = 0.09
    n = np.arange(0, 34)
    phi = np.mod(n * inc, 1.0)

    xr, yr = _break(n, phi)
    ax.plot(xr, yr, color=G, linewidth=0.8)
    ax.plot(n, phi, linestyle="none", marker="o", color=K, ms=2.8)

    # one "+ inc" step
    i = 3
    ax.annotate("", xy=(n[i + 1], phi[i + 1]), xytext=(n[i], phi[i]),
                arrowprops=dict(arrowstyle="->", color="0.3", linewidth=0.8))
    ax.text(n[i] + 0.5, phi[i] + 0.14, r"$+\,\mathrm{inc}$", ha="center",
            fontsize=8.5, color=K)

    # a wrap
    w = int(np.argmax(np.diff(phi) < -0.4)) + 1
    ax.annotate("", xy=(n[w], phi[w]), xytext=(n[w - 1], phi[w - 1] + 0.09),
                arrowprops=dict(arrowstyle="->", color="0.3", linewidth=0.8,
                                linestyle=(0, (2, 1.5))))
    ax.text(n[w] + 0.3, 0.60, r"$-\,1$  (wrap)", ha="left", fontsize=8.5, color=K)

    ax.text(0.2, 1.06, r"$\varphi[n] = \varphi[n-1] + \mathrm{inc}\ \ (\mathrm{mod}\ 1)$",
            ha="left", va="bottom", fontsize=9.5, color="0.15")

    ax.set_xlim(-0.5, 33.5)
    ax.set_ylim(-0.12, 1.18)
    ax.set_xlabel(r"sample index  $n$")
    ax.set_ylabel(r"phase  $\varphi$")
    ax.set_yticks([0, 1])
    tidy(ax)
    finish(fig, "accumulator", FIGDIR)


# =========================================================================
# Figure 2 — the sawtooth is a line.  saw(tau) = frac(f*tau): the line
# f*tau (top) read modulo one (bottom).  The value at tau is a function of
# tau alone, so it can be read anywhere -- backward, forward, jumped.
# =========================================================================
def fig_line_mod_one():
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.9, 3.5), sharex=True,
                                 gridspec_kw={"height_ratios": [1.25, 1],
                                              "hspace": 0.16})
    t = np.linspace(-1.15, 3.6, 3000)
    line = t                                    # f = 1
    saw = np.mod(t, 1.0)

    # top: the line, crossing the integers
    a1.plot(t, line, color=G)
    for k in (0, 1, 2, 3):
        a1.axhline(k, color=L, linestyle=(0, (1, 1.5)), linewidth=0.7)
    a1.text(-1.1, 3.05, r"the line  $f\,\tau$", ha="left", va="top", fontsize=9,
            color="0.3")
    a1.set_ylim(-1.35, 3.75)
    a1.set_yticks([0, 1, 2, 3])
    a1.set_ylabel(r"$f\,\tau$")
    tidy(a1)

    # bottom: the sawtooth = line mod 1, read at three arbitrary tau
    xr, yr = _break(t, saw)
    a2.plot(xr, yr, color=K)
    a2.axhline(0, color="0.55", linewidth=0.5)
    reads = [-0.6, 1.4, 3.1]
    labels = [r"backward", r"fractional", r"jumped"]
    for tau, lab in zip(reads, labels):
        v = np.mod(tau, 1.0)
        for ax in (a1, a2):
            ax.axvline(tau, color="0.55", linestyle=(0, (1, 2)), linewidth=0.6)
        a2.plot(tau, v, marker="o", color=K, ms=3.6)
        a2.annotate(lab, xy=(tau, v), xytext=(tau, 1.12), ha="center",
                    va="bottom", fontsize=8, color="0.35")
    a2.text(-1.1, 1.40, r"$\mathrm{saw}(\tau)=\mathrm{frac}(f\,\tau)$", ha="left",
            va="bottom", fontsize=9.5, color="0.15")
    a2.set_ylim(-0.12, 1.66)
    a2.set_yticks([0, 1])
    a2.set_ylabel(r"$\mathrm{saw}$")
    a2.set_xlabel(r"time  $\tau$  (periods)")
    a2.set_xlim(-1.15, 3.6)
    a2.set_xticks([-1, 0, 1, 2, 3])
    tidy(a2)
    finish(fig, "line-mod-one", FIGDIR)


# =========================================================================
# Figure 3 — one line, four waveforms.  The phase phi = frac(f*tau) is the
# sawtooth itself; triangle, square and sine are memoryless shapings of that
# one ramp.  The phase is the line; the waveform is a map of the phase.
# =========================================================================
def fig_shaping_family():
    fig, axes = plt.subplots(4, 1, figsize=(5.9, 4.4), sharex=True,
                             gridspec_kw={"hspace": 0.28})
    t = np.linspace(0, 2.5, 3000)
    phi = np.mod(t, 1.0)

    rows = [
        (phi,                               r"$\varphi=\mathrm{frac}(f\tau)$",
         r"phase $=$ sawtooth",            (-0.1, 1.1), [0, 1]),
        (1 - 2 * np.abs(2 * phi - 1),       r"$1-2\,|\,2\varphi-1\,|$",
         r"triangle",                      (-1.2, 1.2), [-1, 1]),
        (np.where(phi < 0.5, -1.0, 1.0),    r"$\mathrm{sign}(\varphi-\frac{1}{2})$",
         r"square",                        (-1.35, 1.35), [-1, 1]),
        (np.sin(2 * np.pi * phi),           r"$\sin(2\pi\varphi)$",
         r"sine",                          (-1.2, 1.2), [-1, 1]),
    ]
    for ax, (y, formula, name, ylim, yt) in zip(axes, rows):
        xr, yr = _break(t, y)               # NaN at wraps / square edges
        ax.plot(xr, yr, color=K)
        ax.text(-0.02, 0.5, name, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=9, color="0.25")
        ax.text(2.48, ylim[1] * 0.62, formula, ha="right", va="center",
                fontsize=8.5, color="0.4")
        ax.set_ylim(*ylim)
        ax.set_yticks(yt)
        tidy(ax)
    axes[0].set_title(r"one ramp $\varphi=\mathrm{frac}(f\tau)$, four memoryless shapings",
                      fontsize=10)
    axes[-1].set_xlabel(r"time  $\tau$  (periods)")
    axes[-1].set_xlim(0, 2.5)
    axes[-1].set_xticks([0, 0.5, 1, 1.5, 2, 2.5])
    finish(fig, "shaping-family", FIGDIR)


if __name__ == "__main__":
    fig_accumulator()
    fig_line_mod_one()
    fig_shaping_family()
    print("wrote accumulator.svg, line-mod-one.svg, shaping-family.svg to", FIGDIR)
