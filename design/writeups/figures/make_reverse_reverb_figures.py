#!/usr/bin/env python
"""Figures for the reverse reverb writeup (reverse-reverb.md).

    venv/bin/python design/writeups/figures/make_reverse_reverb_figures.py

See _style.py for the shared house aesthetic.
"""
import numpy as np
from _style import apply_style, figdir_of, finish, tidy, K, G, L, FILL
import matplotlib.pyplot as plt

apply_style()
FIGDIR = figdir_of(__file__)
TAU = 0.40   # reverb decay time constant (s), shared across figures


# =========================================================================
# Figure 1 — the phenomenon.  Amplitude envelope of a single hit through a
# forward reverb (a decaying tail AFTER the sound) versus a reverse reverb
# (a swell that grows INTO the sound, arriving before it).
# =========================================================================
def fig_pre_swell():
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    t = np.linspace(-1.35, 1.35, 2000)
    fwd = np.where(t >= 0, np.exp(-t / TAU), 0.0)
    rev = np.where(t <= 0, np.exp(t / TAU), 0.0)

    ax.axvspan(-1.35, 0, facecolor=FILL, edgecolor="none")
    ax.plot(t, fwd, color=G, linestyle=(0, (4, 2)))
    ax.plot(t, rev, color=K)
    ax.axvline(0, color="0.4", linewidth=0.6)

    ax.text(0.02, 1.06, r"the sound", ha="left", va="bottom", fontsize=8.5,
            color="0.35")
    ax.text(-0.60, 0.34, r"reverse reverb", ha="center", color=K, fontsize=9)
    ax.text(-0.66, 0.13, r"(swell arrives $before$)", ha="center", color="0.4",
            fontsize=8)
    ax.text(0.62, 0.34, r"forward reverb", ha="center", color="0.35", fontsize=9)
    ax.text(0.60, 0.13, r"(tail decays $after$)", ha="center", color="0.4",
            fontsize=8)

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(0, 1.18)
    ax.set_xlabel(r"time relative to the hit  (s)")
    ax.set_ylabel(r"amplitude envelope")
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    tidy(ax)
    finish(fig, "pre-swell", FIGDIR)


def _reflections(seed=3, n=26, span=1.5):
    """Deterministic reflection times (s>0) and signed gains, |g| ~ exp decay."""
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(span / n, size=n)
    s = np.cumsum(gaps) + 0.03
    s = s[s < span]
    signs = rng.choice([-1.0, 1.0], size=s.size)
    mag = np.exp(-s / TAU) * rng.uniform(0.45, 1.0, size=s.size)
    return s, signs * mag


# =========================================================================
# Figure 2 — the mechanism.  A reverb impulse response is a sum of shifted,
# decaying copies (taps).  Forward reverb: every tap sits AFTER the impulse
# (s >= 0, causal).  Reverse reverb is its time-reverse: every tap sits
# BEFORE the impulse (s <= 0, anti-causal) -- energy that precedes the sound.
# =========================================================================
def fig_impulse_mirror():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.9), sharey=True)
    s, g = _reflections()
    tt = np.linspace(0, 1.6, 500)

    # left: causal reverb
    a1.axhline(0, color="0.4", linewidth=0.6)
    a1.plot([0, 0], [0, 1], color=K, linewidth=1.4)          # direct sound
    a1.plot(0, 1, marker="o", color=K, ms=3)
    a1.vlines(s, 0, g, color=G, linewidth=0.7)
    a1.plot(s, g, linestyle="none", marker="o", color=G, ms=2.2)
    a1.plot(tt, np.exp(-tt / TAU), color=L, linestyle=(0, (1, 1.5)), linewidth=0.8)
    a1.plot(tt, -np.exp(-tt / TAU), color=L, linestyle=(0, (1, 1.5)), linewidth=0.8)
    a1.set_title(r"forward reverb $h(t)$  (causal)", fontsize=10)
    a1.text(0.82, 0.86, r"energy $after$", ha="center", fontsize=8.5, color="0.35")

    # right: anti-causal (time-reversed) reverb
    a2.axvspan(-1.65, 0, facecolor=FILL, edgecolor="none")
    a2.axhline(0, color="0.4", linewidth=0.6)
    a2.plot([0, 0], [0, 1], color=K, linewidth=1.4)          # direct sound
    a2.plot(0, 1, marker="o", color=K, ms=3)
    a2.vlines(-s, 0, g, color=G, linewidth=0.7)
    a2.plot(-s, g, linestyle="none", marker="o", color=G, ms=2.2)
    a2.plot(-tt, np.exp(-tt / TAU), color=L, linestyle=(0, (1, 1.5)), linewidth=0.8)
    a2.plot(-tt, -np.exp(-tt / TAU), color=L, linestyle=(0, (1, 1.5)), linewidth=0.8)
    a2.set_title(r"reverse reverb $h(-t)$  (anti-causal)", fontsize=10)
    a2.text(-0.82, 0.86, r"energy $before$", ha="center", fontsize=8.5, color="0.35")

    for ax in (a1, a2):
        ax.set_xlim(-1.65, 1.65)
        ax.set_ylim(-1.05, 1.15)
        ax.set_xlabel(r"time  $t$  (s)")
        ax.set_xticks([-1, 0, 1])
        tidy(ax)
    a1.set_ylabel(r"tap gain")
    finish(fig, "impulse-mirror", FIGDIR)


# =========================================================================
# Figure 3 — the latency wall (cf. through-zero's causality-wall).  Reverse
# reverb wants taps at s < 0 (reading x ahead of now).  A causal stream
# cannot; it must delay the whole signal by the entire tail length T so the
# taps land at s >= 0 -> latency = T (seconds).  A closed-form kernel reads
# x(tau + |s|) directly -> latency 0.
# =========================================================================
def fig_latency_wall():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 3.0), sharey=True)
    # a few representative reverse-reverb taps at s < 0 (ahead of now)
    s = np.array([-0.25, -0.55, -0.85, -1.15, -1.45])
    g = np.exp(s / TAU) * np.array([1, -1, 1, 1, -1]) * 0.9
    T = 1.45                                   # tail length = max reach ahead

    # left: reverse reverb wants taps in the future (gray); a stream can only
    # realise them by shoving the whole IR into the past by T (black).
    a1.axvspan(-1.75, 0, facecolor=FILL, edgecolor="none")
    a1.text(-0.87, -0.80, r"cannot read the future", ha="center", va="center",
            fontsize=8, color="0.45")
    a1.axhline(0, color="0.4", linewidth=0.6)
    a1.vlines(s, 0, g, color=L, linewidth=0.7)                 # wanted (unreadable)
    a1.plot(s, g, linestyle="none", marker="o", color=L, ms=2.2)
    a1.plot([0, 0], [0, 1], color=L, linewidth=1.2)            # wanted direct sound
    a1.plot(0, 1, marker="o", color=L, ms=2.6)
    a1.vlines(s + T, 0, g, color=K, linewidth=0.8)             # shifted (readable)
    a1.plot(s + T, g, linestyle="none", marker="o", color=K, ms=2.4)
    a1.plot([T, T], [0, 1], color=K, linewidth=1.4)            # realised direct sound
    a1.plot(T, 1, marker="o", color=K, ms=3)
    a1.annotate("", xy=(0, 1.16), xytext=(T, 1.16),
                arrowprops=dict(arrowstyle="<->", color="0.35", linewidth=0.7))
    a1.text(T / 2, 1.22, r"latency $=T$  (whole tail)", ha="center", va="bottom",
            fontsize=8.5, color=K)
    a1.set_title(r"streaming buffer", fontsize=10)

    # right: Tropical reads the forward taps directly
    a2.axvspan(-1.75, 0, facecolor=FILL, edgecolor="none")
    a2.text(-0.87, -0.78, r"closed-form future (free to read)", ha="center",
            va="center", fontsize=8, color="0.45")
    a2.axhline(0, color="0.4", linewidth=0.6)
    a2.vlines(s, 0, g, color=K, linewidth=0.8)
    a2.plot(s, g, linestyle="none", marker="o", color=K, ms=2.4)
    a2.plot([0, 0], [0, 1], color=K, linewidth=1.4)
    a2.plot(0, 1, marker="o", color=K, ms=3)
    a2.text(0.05, 1.22, r"latency $=0$", ha="left", va="bottom", fontsize=8.5,
            color=K)
    a2.set_title(r"Tropical kernel", fontsize=10)

    for ax in (a1, a2):
        ax.set_xlim(-1.75, 1.75)
        ax.set_ylim(-1.0, 1.5)
        ax.set_xlabel(r"tap time relative to now  (s)")
        ax.set_xticks([-1, 0, 1])
        tidy(ax)
    a1.set_ylabel(r"tap gain")
    finish(fig, "latency-wall", FIGDIR)


if __name__ == "__main__":
    fig_pre_swell()
    fig_impulse_mirror()
    fig_latency_wall()
    print("wrote pre-swell.svg, impulse-mirror.svg, latency-wall.svg to", FIGDIR)
