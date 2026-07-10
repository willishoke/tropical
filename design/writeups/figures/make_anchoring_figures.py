#!/usr/bin/env python
"""Figures for the phase-alignment-by-anchoring writeup (phase-anchoring.md).

    venv/bin/python design/writeups/figures/make_anchoring_figures.py

See _style.py for the shared house aesthetic.
"""
import numpy as np
from _style import apply_style, figdir_of, finish, tidy, K, G, L, FILL
import matplotlib.pyplot as plt

apply_style()
FIGDIR = figdir_of(__file__)


def _wave(phi, s=0.0):
    """A periodic wave whose harmonics grow with the morph parameter s in [0,1]."""
    return (np.sin(2 * np.pi * phi)
            + (0.14 + 0.55 * s) * np.sin(2 * np.pi * 2 * phi + 1.0)
            + (0.45 * s) * np.sin(2 * np.pi * 3 * phi + 0.3))


def _first_rising_cross(phi, y, level):
    """Phase of the first upward crossing of `level` (linear-interpolated)."""
    d = y - level
    for i in range(len(d) - 1):
        if d[i] <= 0 < d[i + 1]:
            t = -d[i] / (d[i + 1] - d[i])
            return phi[i] + t * (phi[i + 1] - phi[i])
    return phi[0]


# =========================================================================
# Figure 1 — the trigger is a search on the stream.  The sweep fires when
# the sampled signal crosses a level on a rising slope; the true crossing
# falls between two samples, so the sweep start is uncertain by up to one
# sample -- and the edge has to occur at all before anything is drawn.
# =========================================================================
def fig_trigger():
    fig, ax = plt.subplots(figsize=(5.9, 2.7))
    t = np.linspace(0.55, 2.05, 2000)
    level = 0.30
    ax.plot(t, np.sin(2 * np.pi * t), color=K)
    ax.axhline(level, color=L, linestyle=(0, (4, 2)), linewidth=0.8)
    ax.text(2.04, level + 0.04, r"trigger level", ha="right", va="bottom",
            fontsize=8.5, color="0.4")

    # a rising crossing near the middle, and the two samples straddling it
    tc = 1 + np.arcsin(level) / (2 * np.pi)         # exact crossing (rising)
    dt = 1.0 / 12                                   # coarse sample interval
    tl = np.floor(tc / dt) * dt
    tr = tl + dt
    for ts in (tl, tr):
        ax.plot(ts, np.sin(2 * np.pi * ts), marker="o", color=K, ms=3.4)
    ax.axvline(tc, color="0.4", linewidth=0.6, ymax=0.80)
    ax.annotate("", xy=(tl, level), xytext=(tr, level),
                arrowprops=dict(arrowstyle="<->", color="0.35", linewidth=0.7))
    ax.text((tl + tr) / 2, level - 0.10, r"$\pm$ one sample (jitter)", ha="center",
            va="top", fontsize=8.5, color=K)
    ax.text(tc + 0.03, 0.98, r"the trigger $waits$ for the crossing:", ha="left",
            va="top", fontsize=9, color="0.2")
    ax.text(tc + 0.03, 0.83, r"a search on the stream", ha="left",
            va="top", fontsize=9, color="0.2")

    ax.set_xlim(0.55, 2.05)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel(r"time  $\tau$  (periods)")
    ax.set_ylabel(r"signal")
    ax.set_yticks([-1, 0, 1])
    ax.set_xticks([0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    tidy(ax)
    finish(fig, "trigger", FIGDIR)


# =========================================================================
# Figure 2 — anchoring.  The phasor's phase is a KNOWN function of the
# sample index (an exact integer on Z/2^32), so the cycle boundary tau* is
# computed, not searched.  The scope fills its display with
# render_window(tau*, count) -- a random-access read of the closed-form
# kernel, pinned to the anchor.
# =========================================================================
def fig_anchor():
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.8, 3.6), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1.5],
                                              "hspace": 0.18})
    t = np.linspace(0, 3.2, 3000)
    phi = np.mod(t, 1.0)
    anchors = [0, 1, 2, 3]
    win_lo, win_hi = 1.0, 2.4                       # the render window

    # top: phasor phase (sawtooth), anchors at phi = 0
    a1.plot(t, phi, color=K)
    for a in anchors:
        a1.axvline(a, color=L, linewidth=0.7, linestyle=(0, (1, 1.5)))
        a1.plot(a, 0, marker="o", color=K, ms=3)
    a1.text(0.04, 0.80, r"phasor phase  $\varphi(n)=(\mathrm{incr}\cdot n \ \mathrm{mod}\ 2^{32})/2^{32}$",
            ha="left", va="center", fontsize=8.5, color="0.25")
    a1.annotate(r"$\tau^{*}$", xy=(1.0, 0.0), xytext=(1.12, 0.42), fontsize=10,
                color=K, arrowprops=dict(arrowstyle="-", color="0.4", linewidth=0.6))
    a1.set_ylim(-0.08, 1.08)
    a1.set_yticks([0, 1])
    a1.set_ylabel(r"$\varphi$")
    tidy(a1)

    # bottom: the output, with the render window pinned to the anchor tau*
    x = np.sin(2 * np.pi * t) + 0.3 * np.sin(2 * np.pi * 2 * t + 0.5)
    a2.axvspan(win_lo, win_hi, facecolor=FILL, edgecolor="none")
    a2.plot(t, x, color=K)
    a2.axvline(win_lo, color="0.4", linewidth=0.8)
    for a in anchors:
        a2.axvline(a, color=L, linewidth=0.7, linestyle=(0, (1, 1.5)))
    a2.text((win_lo + win_hi) / 2, -1.28, r"$\mathrm{render\_window}(\tau^{*},\ \mathrm{count})$",
            ha="center", va="bottom", fontsize=8.5, color=K)
    a2.set_ylim(-1.45, 1.45)
    a2.set_yticks([-1, 0, 1])
    a2.set_ylabel(r"signal")
    a2.set_xlabel(r"time  $\tau$  (periods)")
    a2.set_xlim(0, 3.2)
    a2.set_xticks([0, 1, 2, 3])
    tidy(a2)
    finish(fig, "anchor", FIGDIR)


# =========================================================================
# Figure 3 — lock the timebase, not the output.  Six successive frames of a
# morphing timbre, overlaid.  Left: aligned by a threshold crossing of the
# OUTPUT -- the crossing wanders as the shape morphs, so the frames smear.
# Right: aligned by the phasor's PHASE -- every frame pinned at the cycle
# start, so the timbre morphs in place and the display stays still.
# =========================================================================
def fig_lock_timebase():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.9), sharey=True)
    rng = np.random.default_rng(7)
    phi = np.linspace(-0.15, 1.25, 1400)
    dense = np.linspace(0, 1.25, 4000)
    level = 0.35
    N = 6

    for k in range(N):
        s = k / (N - 1)
        col = str(0.15 + 0.6 * s)                  # older frames lighter
        y = _wave(phi, s)
        # left: trigger on the output (crossing wanders) + a little jitter
        tc = _first_rising_cross(dense, _wave(dense, s), level)
        jit = rng.uniform(-0.02, 0.02)
        a1.plot(phi - tc + 0.15 + jit, y, color=col, linewidth=0.8)
        # right: anchor on the phasor phase (phi = 0 pinned)
        a2.plot(phi, y, color=col, linewidth=0.8)

    a1.axhline(level, color=L, linestyle=(0, (4, 2)), linewidth=0.7)
    a1.set_title(r"triggered on the output", fontsize=10)
    a1.text(0.5, -2.15, r"crossing wanders $\Rightarrow$ smear", ha="center",
            fontsize=8.5, color="0.35")

    a2.axvline(0.0, color="0.4", linewidth=0.6)
    a2.set_title(r"anchored on the phasor", fontsize=10)
    a2.text(0.55, -2.15, r"phase pinned $\Rightarrow$ morph in place", ha="center",
            fontsize=8.5, color="0.35")

    for ax in (a1, a2):
        ax.set_xlim(-0.2, 1.25)
        ax.set_ylim(-2.4, 2.4)
        ax.set_xlabel(r"display position")
        ax.set_xticks([])
        tidy(ax)
    a1.set_ylabel(r"signal")
    finish(fig, "lock-timebase", FIGDIR)


if __name__ == "__main__":
    fig_trigger()
    fig_anchor()
    fig_lock_timebase()
    print("wrote trigger.svg, anchor.svg, lock-timebase.svg to", FIGDIR)
