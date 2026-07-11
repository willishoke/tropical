"""Shared figure style for the writeups series.

House aesthetic: narrow precise grayscale lines, Computer-Modern serif,
high data-ink ratio (Tufte). SVG with glyph outlines embedded as paths, so
the files are self-contained and render identically without CM installed.

    python -m venv venv && venv/bin/pip install matplotlib numpy
    venv/bin/python design/writeups/figures/make_<writeup>_figures.py
"""
import os
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after backend selection)
import matplotlib.patheffects as pe  # noqa: E402

# grayscale palette
K = "0.0"      # black
G = "0.50"     # mid gray
L = "0.72"     # light gray
FILL = "0.90"  # region shading


def apply_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,       # avoid missing-glyph warn w/ cmr10
        "axes.formatter.use_mathtext": True,
        "font.size": 10,
        "axes.linewidth": 0.6,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "lines.linewidth": 0.9,
        "figure.dpi": 110,
        "svg.fonttype": "path",            # embed glyph outlines -> portable
        "svg.hashsalt": "tropical-writeups",  # deterministic element ids
        "legend.frameon": False,
        "legend.fontsize": 8.5,
    })


def figdir_of(script_file):
    return os.path.dirname(os.path.abspath(script_file))


def finish(fig, name, figdir):
    fig.savefig(os.path.join(figdir, name + ".svg"), bbox_inches="tight",
                pad_inches=0.04, transparent=True,
                metadata={"Date": None})     # deterministic: no timestamp
    plt.close(fig)


def tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def boxed_legend(ax, **kw):
    """A white-backed legend that reads cleanly over curves."""
    return ax.legend(frameon=True, framealpha=1.0, edgecolor="0.8",
                     fancybox=False, **kw)


def halo(*artists, lw=2.6):
    """Give text a thin white 'paper' halo so it reads over curves, axes, and
    shaded regions without being moved. This is the highest-leverage
    legibility fix in the series: it decouples whether a label is readable
    from where it sits, so annotations no longer have to dodge every curve.
    Deterministic (a plain stroke, no iterative solver), so byte-reproducible
    output is preserved. Pass any Text / Annotation artists.

        halo(ax.text(...))
        halo(a1, a2, a3)                     # several at once
        halo(ax.annotate(...), lw=3.0)       # wider halo over busy regions

    Only the text is stroked; an Annotation's arrow is left untouched. Use on
    inline labels and callouts, not on axis labels/ticks (already in the
    margins). Halos assume a light page, which these black-ink-on-transparent
    figures already require."""
    effs = [pe.withStroke(linewidth=lw, foreground="white"), pe.Normal()]
    for a in artists:
        a.set_path_effects(effs)
    return artists[0] if len(artists) == 1 else artists
