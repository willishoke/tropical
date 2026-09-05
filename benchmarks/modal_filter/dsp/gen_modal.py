#!/usr/bin/env python3
"""Faust side of the modal-filter comparison.

Mirrors tropical's `resonator` partial law exactly
(`Playground/Vocabulary.lean:resonatorBank`):

    partial k (1-indexed):  freq_k  = k * f0
                            sigma_k = decay * (1 + 0.4*k)
                            amp_k   = k^-1.1

and `filterPair`'s resonance law: Q = 0.55 * 80^res.

The SOURCE realizations differ by paradigm and are meant to: Faust's is the
idiomatic recurrent form (os.osc phase accumulators, one-pole exponential
envelopes p^n seeded by an impulse), tropical's is closed-form exp/cos in
absolute time. Metric 1 (the MARGINAL cost of the filter) differences the
source out per system; metric 2 reports totals with the paradigm difference
stated rather than hidden.

The sweep variant drives the cutoff from ABSOLUTE time (a sample counter over
ma.SR), so a rendering at 8x the rate passes through the same cutoff
trajectory and aligns at every 8th sample — that is what makes the
oversampled run a reference rather than a different piece of music.
"""
import math, pathlib, sys

DECAY = 4.0
RES = 0.5
Q = 0.55 * 80.0 ** RES
CENTER = 800.0


def f0_of(v: int) -> float:
    return 220.0 * (1.0 + 0.03 * v)          # detuned so no two voices fold


def voice(f0: float, m: int) -> str:
    parts = []
    for k in range(1, m + 1):
        amp = k ** -1.1
        sigma = DECAY * (1.0 + 0.4 * k)
        parts.append(f"({amp:.10f} * env({sigma:.6f}) * os.osc({k * f0:.6f}))")
    return "(" + " + ".join(parts) + ")"


HEAD = """import("stdfaust.lib");
impulse = 1 - 1';
env(s) = impulse : + ~ *(exp(-s / ma.SR));
"""


def bare(p: int, m: int) -> str:
    body = " + ".join(voice(f0_of(v), m) for v in range(p))
    return HEAD + f"process = {body};\n"


def filtered(p: int, m: int) -> str:
    body = " + ".join(
        f"({voice(f0_of(v), m)} : fi.svf.lp({CENTER:.1f}, {Q:.10f}))"
        for v in range(p))
    return HEAD + f"process = {body};\n"


def cf_voice(f0: float, m: int) -> str:
    """CLOSED-FORM source for the sweep variant only: exp/cos in absolute
    time, so a rendering at 8x the rate produces (near-)identical source
    values at common instants and the 1x-vs-8x difference isolates the
    FILTER. The recurrent source (os.osc accumulators, one-pole envelopes)
    decorrelates between rates over a 2 s render — measured at -4.5 dB even
    at a STATIC cutoff, swamping the filter artifact entirely."""
    parts = []
    for k in range(1, m + 1):
        amp = k ** -1.1
        sigma = DECAY * (1.0 + 0.4 * k)
        parts.append(f"({amp:.10f} * exp(-{sigma:.6f} * tsec)"
                     f" * cos(2.0 * ma.PI * {k * f0:.6f} * tsec))")
    return "(" + " + ".join(parts) + ")"


def swept(m: int, rate_hz: float, octaves: float) -> str:
    head = HEAD + (
        "counter = (+(1) ~ _) - 1;\n"
        "tsec = counter / ma.SR;\n"
        f"cutoff = {CENTER:.1f} * pow(2.0, {octaves:.4f} * "
        f"sin(2.0 * ma.PI * {rate_hz:.6f} * tsec));\n")
    body = f"({cf_voice(f0_of(0), m)} : fi.svf.lp(cutoff, {Q:.10f}))"
    return head + f"process = {body};\n"


if __name__ == "__main__":
    kind = sys.argv[1]
    out = pathlib.Path(sys.argv[2])
    if kind == "bare":
        out.write_text(bare(int(sys.argv[3]), int(sys.argv[4])))
    elif kind == "filtered":
        out.write_text(filtered(int(sys.argv[3]), int(sys.argv[4])))
    elif kind == "swept":
        out.write_text(swept(int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])))
    else:
        raise SystemExit(f"unknown kind {kind}")
