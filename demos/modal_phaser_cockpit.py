#!/usr/bin/env python3
"""Current-universe modal phaser design cockpit.

This file is intentionally a local, untracked promotion artifact.  It has two
independent evaluators:

* ``sequential_modal`` composes identity-plus-tail sections in the same shape
  as the production modal carrier.
* ``oracle_residues`` forms the high-precision rational product directly and
  takes residues at its distinct poles.  It does not call the sequential
  helper.

With ``--artifacts DIR`` the cockpit also writes response CSV/SVG files and
stereo WAV comparisons (current-universe on the left, a conventional recursive
digital all-pass cascade on the right).  The listening comparison is not the
correctness oracle.
"""

from __future__ import annotations

import argparse
import cmath
import csv
import math
import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import mpmath as mp


SAMPLE_RATE = 44_100
TAU = 2.0 * math.pi
CENTER_RANGE = (40.0, 4_000.0)
SWEEP_RANGE = (0.0, 3.0)
RATE_RANGE = (0.02, 8.0)
MIX_RANGE = (0.0, 1.0)

# Six distinct structural ratios.  Adjacent sections are half an octave apart,
# symmetric in log frequency, and the product of all ratios is exactly one.
RATIO_EXPONENTS = (-1.25, -0.75, -0.25, 0.25, 0.75, 1.25)
RATIOS = tuple(2.0**x for x in RATIO_EXPONENTS)
FOUR_RATIO_EXPONENTS = (-1.125, -0.375, 0.375, 1.125)
FOUR_RATIOS = tuple(2.0**x for x in FOUR_RATIO_EXPONENTS)


@dataclass(frozen=True)
class Mode:
    pole: complex
    residue: complex


def clamp(value: float, bounds: tuple[float, float]) -> float:
    return min(bounds[1], max(bounds[0], value))


def section_frequencies(
    coordinate: float,
    center: float,
    sweep: float,
    rate: float,
    ratios: Sequence[float] = RATIOS,
) -> tuple[float, ...]:
    center = clamp(center, CENTER_RANGE)
    sweep = clamp(sweep, SWEEP_RANGE)
    rate = clamp(rate, RATE_RANGE)
    octave_offset = sweep * math.sin(TAU * rate * coordinate)
    scale = 2.0**octave_offset
    return tuple(center * ratio * scale for ratio in ratios)


def sequential_modal(source: Sequence[Mode], poles: Sequence[float], mix: float) -> list[Mode]:
    """Production-shape identity-plus-tail composition in ordinary complex f64."""
    wet = list(source)
    for a in poles:
        p = complex(-a, 0.0)
        tail = complex(-2.0 * a, 0.0)
        # Evaluate the pre-section transfer at the new tail pole before updating.
        tail_residue = tail * sum(mode.residue / (p - mode.pole) for mode in wet)
        wet = [
            Mode(mode.pole, mode.residue * (1.0 + tail / (mode.pole - p)))
            for mode in wet
        ]
        wet.append(Mode(p, tail_residue))

    mix = clamp(mix, MIX_RANGE)
    out = []
    for dry, filtered in zip(source, wet[: len(source)]):
        out.append(Mode(dry.pole, (1.0 - mix) * dry.residue + mix * filtered.residue))
    out.extend(Mode(mode.pole, mix * mode.residue) for mode in wet[len(source) :])
    return out


def oracle_residues(
    source: Sequence[Mode], poles: Sequence[float], mix: float, dps: int = 80
) -> list[tuple[mp.mpc, mp.mpc]]:
    """Independent residues of X(s) * ((1-mix) + mix * product A_k(s))."""
    with mp.workdps(dps):
        mmix = mp.mpf(mix)
        src = [(mp.mpc(m.pole.real, m.pole.imag), mp.mpc(m.residue.real, m.residue.imag)) for m in source]
        aa = [mp.mpf(a) for a in poles]

        def allpass(s: mp.mpc) -> mp.mpc:
            value = mp.mpc(1)
            for a in aa:
                value *= (s - a) / (s + a)
            return value

        def source_at(s: mp.mpc) -> mp.mpc:
            return mp.fsum(residue / (s - pole) for pole, residue in src)

        result: list[tuple[mp.mpc, mp.mpc]] = []
        for pole, residue in src:
            gain = (1 - mmix) + mmix * allpass(pole)
            result.append((pole, residue * gain))
        for k, a in enumerate(aa):
            pole = mp.mpc(-a, 0)
            allpass_residue = -2 * a
            for j, other in enumerate(aa):
                if j != k:
                    allpass_residue *= (pole - other) / (pole + other)
            result.append((pole, mmix * allpass_residue * source_at(pole)))
        return result


def eval_modes(modes: Sequence[Mode], coordinate: float) -> complex:
    if coordinate <= 0.0:
        return 0j
    return sum(mode.residue * cmath.exp(mode.pole * coordinate) for mode in modes)


def eval_mp_modes(modes: Sequence[tuple[mp.mpc, mp.mpc]], coordinate: float) -> complex:
    if coordinate <= 0.0:
        return 0j
    with mp.workdps(80):
        t = mp.mpf(coordinate)
        value = mp.fsum(residue * mp.exp(pole * t) for pole, residue in modes)
        return complex(value)


def allpass_at(frequency: float, section_hz: Sequence[float]) -> complex:
    s = complex(0.0, TAU * frequency)
    value = 1.0 + 0.0j
    for frequency_hz in section_hz:
        a = TAU * frequency_hz
        value *= (s - a) / (s + a)
    return value


def phaser_at(frequency: float, section_hz: Sequence[float], mix: float) -> complex:
    return (1.0 - mix) + mix * allpass_at(frequency, section_hz)


def logarithmic_grid(lo: float, hi: float, count: int) -> list[float]:
    ratio = hi / lo
    return [lo * ratio ** (i / (count - 1)) for i in range(count)]


def local_notches(section_hz: Sequence[float], mix: float = 0.5) -> list[tuple[float, float]]:
    grid = logarithmic_grid(min(section_hz) / 8.0, max(section_hz) * 8.0, 40_001)
    mags = [abs(phaser_at(frequency, section_hz, mix)) for frequency in grid]
    return [
        (grid[i], mags[i])
        for i in range(1, len(grid) - 1)
        if mags[i] <= mags[i - 1] and mags[i] <= mags[i + 1]
    ]


def source_fixture() -> tuple[Mode, ...]:
    return (
        Mode(complex(-3.5, TAU * 173.0), complex(0.8, -0.15)),
        Mode(complex(-7.0, TAU * 431.0), complex(-0.2, 0.55)),
        Mode(complex(-12.0, TAU * 997.0), complex(0.35, 0.08)),
    )


def oracle_error_corpus() -> tuple[float, float]:
    worst_abs = 0.0
    worst_rel = 0.0
    source = source_fixture()
    coordinates = (1.0 / SAMPLE_RATE, 0.0007, 0.004, 0.021, 0.113, 0.731)
    snapshots = (
        (40.0, 0.0, 0.02, 0.0, -1234.25),
        (700.0, 1.5, 0.2, 0.5, 0.0),
        (4_000.0, 3.0, 8.0, 1.0, 1_000_000.125),
        (1_113.0, 2.25, 1.0, 0.37, -0.375),
    )
    for center, sweep, rate, mix, universe_t in snapshots:
        section_hz = section_frequencies(universe_t, center, sweep, rate)
        poles = tuple(TAU * f for f in section_hz)
        production = sequential_modal(source, poles, mix)
        oracle = oracle_residues(source, poles, mix)
        for coordinate in coordinates:
            actual = eval_modes(production, coordinate)
            expected = eval_mp_modes(oracle, coordinate)
            error = abs(actual - expected)
            worst_abs = max(worst_abs, error)
            worst_rel = max(worst_rel, error / max(abs(expected), 1.0e-12))
    return worst_abs, worst_rel


def ratio_separation_floor() -> tuple[float, float]:
    """Return the first close-ratio spacing whose f64 time error exceeds 1e-8."""
    source = (Mode(complex(-5.0, TAU * 313.0), complex(1.0, 0.2)),)
    for octave_gap in (
        1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5,
        3e-6, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8, 3e-9, 1e-9,
    ):
        ratios = (2.0 ** (-octave_gap / 2.0), 2.0 ** (octave_gap / 2.0))
        poles = tuple(TAU * 700.0 * ratio for ratio in ratios)
        production = sequential_modal(source, poles, 0.5)
        oracle = oracle_residues(source, poles, 0.5, dps=100)
        worst = max(
            abs(eval_modes(production, t) - eval_mp_modes(oracle, t))
            for t in (0.0001, 0.001, 0.01, 0.1, 0.5)
        )
        if worst > 1.0e-8:
            return octave_gap, worst
    return 0.0, 0.0


def bounded_universe_probe() -> tuple[float, float]:
    source = source_fixture()
    peak = 0.0
    energy = 0.0
    count = 0
    for universe_t in (-1.0e9 - 0.125, -10_000.5, -0.25, 0.0, 0.25, 10_000.5, 1.0e9 + 0.125):
        poles = tuple(TAU * f for f in section_frequencies(universe_t, 700.0, 3.0, 8.0))
        modes = sequential_modal(source, poles, 0.5)
        for coordinate in (1.0 / SAMPLE_RATE, 0.001, 0.01, 0.1, 1.0):
            value = eval_modes(modes, coordinate).real
            if not math.isfinite(value):
                raise AssertionError("non-finite current-universe sample")
            peak = max(peak, abs(value))
            energy += value * value
            count += 1
    return peak, math.sqrt(energy / count)


def response_statistics(section_hz: Sequence[float], mix: float) -> tuple[float, float]:
    grid = logarithmic_grid(20.0, 20_000.0, 16_385)
    values = [abs(phaser_at(frequency, section_hz, mix)) for frequency in grid]
    return max(values), math.sqrt(sum(value * value for value in values) / len(values))


def write_response_artifacts(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    section_hz = tuple(700.0 * ratio for ratio in RATIOS)
    grid = logarithmic_grid(20.0, 20_000.0, 4_097)
    rows = [(frequency, *(abs(phaser_at(frequency, section_hz, mix)) for mix in (0.0, 0.5, 1.0))) for frequency in grid]
    with (directory / "response.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("frequency_hz", "mix_0", "mix_0_5", "mix_1"))
        writer.writerows(rows)

    width, height, margin = 960, 420, 42
    points = []
    for frequency, _, magnitude, _ in rows:
        x = margin + (math.log10(frequency / 20.0) / 3.0) * (width - 2 * margin)
        db = max(-72.0, 20.0 * math.log10(max(magnitude, 1.0e-12)))
        y = margin + (-db / 72.0) * (height - 2 * margin)
        points.append(f"{x:.2f},{y:.2f}")
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#10151c"/>'
        f'<polyline fill="none" stroke="#6fe0d0" stroke-width="2" points="{" ".join(points)}"/>'
        '<text x="42" y="24" fill="#d9e2ec" font-family="monospace" font-size="15">six-section dry/wet response · center 700 Hz · mix 0.5</text>'
        '</svg>'
    )
    (directory / "response.svg").write_text(svg)


def current_universe_sample(t: float, rate: float, ratios: Sequence[float]) -> float:
    source = source_fixture()
    poles = tuple(TAU * f for f in section_frequencies(t, 700.0, 1.5, rate, ratios))
    return eval_modes(sequential_modal(source, poles, 0.5), t).real


def recursive_samples(seconds: float, rate: float, ratios: Sequence[float]) -> list[float]:
    count = int(seconds * SAMPLE_RATE)
    previous_x = [0.0] * len(ratios)
    previous_y = [0.0] * len(ratios)
    output: list[float] = []
    source = source_fixture()
    for index in range(count):
        t = index / SAMPLE_RATE
        dry = eval_modes(source, t).real
        value = dry
        frequencies = section_frequencies(t, 700.0, 1.5, rate, ratios)
        for k, frequency in enumerate(frequencies):
            tangent = math.tan(math.pi * min(frequency, SAMPLE_RATE * 0.49) / SAMPLE_RATE)
            q = (1.0 - tangent) / (1.0 + tangent)
            filtered = q * value + previous_x[k] - q * previous_y[k]
            previous_x[k], previous_y[k] = value, filtered
            value = filtered
        output.append(0.5 * dry + 0.5 * value)
    return output


def write_listening_wav(directory: Path, rate: float, ratios: Sequence[float], label: str) -> None:
    seconds = 1.5
    recursive = recursive_samples(seconds, rate, ratios)
    current = [current_universe_sample(i / SAMPLE_RATE, rate, ratios) for i in range(len(recursive))]
    peak = max(1.0, *(abs(value) for value in current), *(abs(value) for value in recursive))
    scale = 0.92 * 32767.0 / peak
    path = directory / f"compare_{label}_{rate:g}hz.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        frames = bytearray()
        for left, right in zip(current, recursive):
            frames.extend(struct.pack("<hh", round(left * scale), round(right * scale)))
        handle.writeframes(frames)


def print_report() -> None:
    section_hz = tuple(700.0 * ratio for ratio in RATIOS)
    four_hz = tuple(700.0 * ratio for ratio in FOUR_RATIOS)
    notches = local_notches(section_hz)
    four_notches = local_notches(four_hz)
    worst_abs, worst_rel = oracle_error_corpus()
    separation, separation_error = ratio_separation_floor()
    bounded_peak, bounded_rms = bounded_universe_probe()

    print("modal phaser cockpit")
    print("ratios:", ", ".join(f"{ratio:.12g}" for ratio in RATIOS))
    print("ratio exponents:", RATIO_EXPONENTS)
    print("notches @ center=700Hz, mix=.5:", ", ".join(f"{f:.3f}Hz ({m:.2e})" for f, m in notches))
    print("four-section fallback notches:", ", ".join(f"{f:.3f}Hz ({m:.2e})" for f, m in four_notches))
    for mix in (0.0, 0.5, 1.0):
        peak, rms = response_statistics(section_hz, mix)
        print(f"mix={mix:g}: peak={peak:.12g}, log-grid RMS={rms:.12g}")
    unit_error = max(abs(abs(allpass_at(f, section_hz)) - 1.0) for f in logarithmic_grid(20.0, 20_000.0, 16_385))
    print(f"wet all-pass unit-magnitude worst abs error: {unit_error:.3e}")
    print(f"sequential vs independent mp rational oracle: abs={worst_abs:.3e}, rel={worst_rel:.3e}")
    print(f"first two-section proximity over 1e-8 time error: {separation:.1e} octaves (error {separation_error:.3e})")
    print(f"far/reverse/hold universe probe: peak={bounded_peak:.9g}, RMS={bounded_rms:.9g}, finite=yes")

    if len(notches) != 3 or len(four_notches) != 2:
        raise SystemExit("notch-count contract failed")
    if unit_error > 2.0e-15 or worst_abs > 2.0e-9 or worst_rel > 2.0e-9:
        raise SystemExit("numerical contract failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, help="write local CSV/SVG/WAV diagnostics")
    args = parser.parse_args()
    print_report()
    if args.artifacts:
        write_response_artifacts(args.artifacts)
        for rate in (0.05, 0.2, 1.0, 8.0):
            write_listening_wav(args.artifacts, rate, RATIOS, "six")
            write_listening_wav(args.artifacts, rate, FOUR_RATIOS, "four")
        print(f"artifacts: {args.artifacts}")


if __name__ == "__main__":
    main()
