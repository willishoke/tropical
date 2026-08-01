#!/usr/bin/env python3
"""Fit dense stationary modal-room auditions to the Clouds gold reference.

This is deliberately a representation/capacity experiment, not a production
profile generator.  Each candidate is a legal frozen modal transfer: a sum of
slow/fast opposite-residue pairs.  The rows are grouped into progressively
later bloom shells so echo density builds instead of arriving fully formed.
"""

from __future__ import annotations

import json
import itertools
import math
import tempfile
from pathlib import Path

import numpy as np

from run import (
    SAMPLE_RATE,
    TWO_PI,
    active_rms,
    fft_convolve,
    rounded,
    shift,
    snare_modes,
)
from run_reference import (
    CLOUDS_COMMIT,
    DURATION_SECONDS,
    OUTPUT_SAMPLE_RATE,
    RENDER_SAMPLE_RATE,
    build_reference_binary,
    modal_signal,
    normalized_echo_density,
    parse_args,
    render_clouds,
    resample_linear,
    spectral_flatness,
    write_wav,
)


AUDITION_DIR = Path(__file__).resolve().parent
OUT_DIR = AUDITION_DIR / "match_out"
SAMPLE_COUNT = round(DURATION_SECONDS * SAMPLE_RATE)
PREDELAY_SECONDS = 0.022
FREQUENCY_EDGES = np.asarray(
    [20, 40, 80, 160, 315, 630, 1250, 2500, 5000, 8000, 12000, 16000, 20000, 21800],
    dtype=np.float64,
)
BLOOM_SHELLS = (
    # fraction, peak seconds, relative late energy
    (0.08, 0.018, 0.50),
    (0.17, 0.055, 0.78),
    (0.28, 0.140, 1.00),
    (0.47, 0.360, 1.18),
)
CANDIDATES = (
    ("modal_cloud_256", 256),
    ("modal_cloud_1024", 1024),
    ("modal_cloud_4096", 4096),
)
SMEAR_PROFILES = {
    "mutable_rate": ((0.0048, 0.173), (0.0031, 0.317)),
    "wide_2": ((0.0080, 0.230), (0.0050, 0.410)),
    "wide_3": ((0.0070, 0.230), (0.0048, 0.410), (0.0030, 0.670)),
}
DIFFUSER_DELAYS_32K = (113, 162, 241, 399)


def frame_band_energy(
    signal: np.ndarray,
    *,
    window_seconds: float = 0.080,
    hop_seconds: float = 0.040,
) -> tuple[np.ndarray, np.ndarray]:
    window_size = round(window_seconds * SAMPLE_RATE)
    hop_size = round(hop_seconds * SAMPLE_RATE)
    window = np.hanning(window_size)
    frequencies = np.fft.rfftfreq(window_size, 1.0 / SAMPLE_RATE)
    masks = [
        (frequencies >= lo) & (frequencies < hi)
        for lo, hi in zip(FREQUENCY_EDGES[:-1], FREQUENCY_EDGES[1:])
    ]
    centers: list[float] = []
    rows: list[list[float]] = []
    for begin in range(0, len(signal) - window_size + 1, hop_size):
        spectrum = np.fft.rfft(signal[begin:begin + window_size] * window)
        power = np.square(np.abs(spectrum))
        centers.append((begin + window_size / 2) / SAMPLE_RATE)
        rows.append([float(np.sum(power[mask])) for mask in masks])
    return np.asarray(centers), np.asarray(rows)


def nonnegative_lstsq(a: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tiny exhaustive NNLS for the four fixed bloom shells."""
    best = np.zeros(a.shape[1], dtype=np.float64)
    best_error = float(np.sum(np.square(y)))
    for width in range(1, a.shape[1] + 1):
        for active in itertools.combinations(range(a.shape[1]), width):
            trial, *_ = np.linalg.lstsq(a[:, active], y, rcond=None)
            if np.any(trial < 0.0):
                continue
            prediction = a[:, active] @ trial
            error = float(np.sum(np.square(prediction - y)))
            if error < best_error:
                best_error = error
                best[:] = 0.0
                best[list(active)] = trial
    return best


def fit_band_profile(target_impulse: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centers, energy = frame_band_energy(target_impulse)
    energy /= max(float(np.max(energy)), 1e-30)
    fit_region = (centers >= 0.02) & (centers <= 4.0)
    local_t = np.maximum(0.0, centers[fit_region] - PREDELAY_SECONDS)
    sigma_grid = np.geomspace(0.45, 28.0, 30)
    sigmas: list[float] = []
    shell_energy: list[np.ndarray] = []
    for band in range(energy.shape[1]):
        y = energy[fit_region, band]
        floor = max(float(np.max(y)) * 1e-6, 1e-14)
        weights = 1.0 / np.sqrt(y + floor)
        best_score = math.inf
        best_sigma = 1.0
        best_q = np.zeros(len(BLOOM_SHELLS), dtype=np.float64)
        for slow_sigma in sigma_grid:
            columns = []
            for _, peak_seconds, _ in BLOOM_SHELLS:
                fast_sigma = fast_sigma_for_peak(float(slow_sigma), peak_seconds)
                envelope = (
                    np.exp(-slow_sigma * local_t)
                    - np.exp(-fast_sigma * local_t)
                )
                columns.append(np.square(envelope))
            model = np.column_stack(columns)
            q = nonnegative_lstsq(model * weights[:, None], y * weights)
            prediction = model @ q
            audible = y > floor
            score = float(np.mean(np.square(
                np.log10(prediction[audible] + floor)
                - np.log10(y[audible] + floor)
            )))
            if score < best_score:
                best_score = score
                best_sigma = float(slow_sigma)
                best_q = q
        sigmas.append(best_sigma)
        shell_energy.append(best_q)
    return np.asarray(shell_energy), np.asarray(sigmas)


def fast_sigma_for_peak(slow_sigma: float, peak_seconds: float) -> float:
    """Solve peak=(log(fast/slow)/(fast-slow)) for fast > slow."""
    low = slow_sigma * (1.0 + 1e-9)
    high = max(low * 2.0, 2.0 / peak_seconds)
    def peak(fast: float) -> float:
        return math.log(fast / slow_sigma) / (fast - slow_sigma)
    while peak(high) > peak_seconds:
        high *= 2.0
    for _ in range(80):
        mid = 0.5 * (low + high)
        if peak(mid) > peak_seconds:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def integer_allocation(weights: np.ndarray, count: int) -> np.ndarray:
    exact = weights / np.sum(weights) * count
    result = np.floor(exact).astype(np.int64)
    remainder = count - int(np.sum(result))
    order = np.argsort(-(exact - result))
    result[order[:remainder]] += 1
    return result


def make_modal_room(
    cell_count: int,
    shell_energy: np.ndarray,
    band_sigmas: np.ndarray,
) -> tuple[list[list[float]], np.ndarray]:
    # Allocate cells by linear bandwidth.  Energy controls residues instead of
    # stealing frequency coverage, which is what keeps the late field diffuse.
    widths = np.diff(FREQUENCY_EDGES)
    band_counts = integer_allocation(widths, cell_count)
    rows: list[list[float]] = []
    group_spectra: dict[tuple[int, int], np.ndarray] = {}
    t = np.arange(SAMPLE_COUNT, dtype=np.float64) / SAMPLE_RATE
    bin_hz = SAMPLE_RATE / SAMPLE_COUNT
    occupied: set[int] = set()
    global_index = 0

    for band, count in enumerate(band_counts):
        if count == 0:
            continue
        lo = float(FREQUENCY_EDGES[band])
        hi = min(float(FREQUENCY_EDGES[band + 1]), SAMPLE_RATE * 0.495)
        shell_counts = integer_allocation(
            np.asarray([shell[0] for shell in BLOOM_SHELLS]), count
        )
        shell_for_local: list[int] = []
        for shell_index, shell_count in enumerate(shell_counts):
            shell_for_local.extend([shell_index] * int(shell_count))
        # Interleave shells across the frequency band rather than making each
        # bloom stage occupy a tell-tale contiguous pitch range.
        permutation = np.argsort(
            [((index + 1) * 0.6180339887498949) % 1.0 for index in range(count)]
        )
        shell_for_local = [shell_for_local[int(index)] for index in permutation]

        for local in range(count):
            jitter = ((global_index + 1) * 0.6180339887498949) % 1.0 - 0.5
            frequency = lo + (local + 0.5 + 0.72 * jitter) / count * (hi - lo)
            frequency_bin = max(1, min(SAMPLE_COUNT // 2 - 1, round(frequency / bin_hz)))
            while frequency_bin in occupied:
                frequency_bin += 1
            occupied.add(frequency_bin)
            frequency = frequency_bin * bin_hz
            shell_index = shell_for_local[local]
            _, peak_seconds, _ = BLOOM_SHELLS[shell_index]
            slow_sigma = float(band_sigmas[band])
            fast_sigma = fast_sigma_for_peak(slow_sigma, peak_seconds)
            phase = (2.399963229728653 * (global_index + 1) + 0.31 * jitter) % TWO_PI
            amplitude = math.sqrt(
                max(float(shell_energy[band, shell_index]), 1e-18)
                / max(1, int(shell_counts[shell_index]))
            )
            rows.append([
                rounded(frequency, 9),
                rounded(slow_sigma, 9),
                rounded(amplitude, 12),
                rounded(phase, 9),
            ])
            rows.append([
                rounded(frequency, 9),
                rounded(fast_sigma, 9),
                rounded(-amplitude, 12),
                rounded(phase, 9),
            ])

            key = (band, shell_index)
            spectrum = group_spectra.setdefault(
                key, np.zeros(SAMPLE_COUNT // 2 + 1, dtype=np.complex128)
            )
            spectrum[frequency_bin] += (
                0.5 * SAMPLE_COUNT * amplitude * np.exp(1j * phase)
            )
            global_index += 1

    impulse = np.zeros(SAMPLE_COUNT, dtype=np.float64)
    for (band, shell_index), spectrum in group_spectra.items():
        carrier = np.fft.irfft(spectrum, SAMPLE_COUNT)
        slow_sigma = float(band_sigmas[band])
        fast_sigma = fast_sigma_for_peak(
            slow_sigma, BLOOM_SHELLS[shell_index][1]
        )
        impulse += carrier * (
            np.exp(-slow_sigma * t) - np.exp(-fast_sigma * t)
        )
    return rows, shift(impulse, PREDELAY_SECONDS)


def discrete_convolve(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    wanted = len(signal) + len(kernel) - 1
    fft_size = 1 << (wanted - 1).bit_length()
    result = np.fft.irfft(
        np.fft.rfft(signal, fft_size) * np.fft.rfft(kernel, fft_size), fft_size
    )
    return result[:len(signal)]


def diffuser_kernel(gain: float = 0.70, repeats: int = 6) -> np.ndarray:
    """Finite closed-form projection of Clouds' four input allpasses."""
    result = np.ones(1, dtype=np.float64)
    for delay_32k in DIFFUSER_DELAYS_32K:
        delay = round(delay_32k * SAMPLE_RATE / RENDER_SAMPLE_RATE)
        stage = np.zeros(delay * repeats + 1, dtype=np.float64)
        stage[0] = -gain
        for repeat in range(1, repeats + 1):
            stage[repeat * delay] = (
                (1.0 - gain * gain) * math.pow(gain, repeat - 1)
            )
        result = np.convolve(result, stage)
    return result


def variable_read(signal: np.ndarray, offset_samples: np.ndarray) -> np.ndarray:
    coordinates = np.arange(len(signal), dtype=np.float64) + offset_samples
    lo = np.floor(coordinates).astype(np.int64)
    fraction = coordinates - lo
    valid = (lo >= 0) & (lo + 1 < len(signal))
    result = np.zeros(len(signal), dtype=np.float64)
    indices = lo[valid]
    result[valid] = (
        signal[indices] * (1.0 - fraction[valid])
        + signal[indices + 1] * fraction[valid]
    )
    return result


def swept_flange(signal: np.ndarray, depth_seconds: float, rate: float) -> np.ndarray:
    t = np.arange(len(signal), dtype=np.float64) / SAMPLE_RATE
    offset = depth_seconds * SAMPLE_RATE * np.sin(TWO_PI * rate * t)
    return (
        0.50 * signal
        + 0.25 * variable_read(signal, -offset)
        + 0.25 * variable_read(signal, offset)
    )


def apply_smear(
    signal: np.ndarray, stages: tuple[tuple[float, float], ...]
) -> np.ndarray:
    result = signal
    for depth_seconds, rate in stages:
        result = swept_flange(result, depth_seconds, rate)
    return result


def feature_summary(impulse: np.ndarray) -> dict[str, object]:
    centers = (0.020, 0.050, 0.100, 0.200, 0.500, 1.000)
    _, energy = frame_band_energy(impulse)
    normalized = energy / max(float(np.sum(energy)), 1e-30)
    return {
        "normalized_echo_density": {
            f"{round(center * 1000)}ms": rounded(
                normalized_echo_density(impulse, center), 4
            )
            for center in centers
        },
        "spectral_flatness": {
            "100_300ms": rounded(spectral_flatness(impulse, 0.1, 0.3), 6),
            "500_1000ms": rounded(spectral_flatness(impulse, 0.5, 1.0), 6),
        },
        "band_time_distribution": normalized.tolist(),
    }


def feature_distance(
    candidate: dict[str, object], target: dict[str, object]
) -> dict[str, float]:
    c_density = np.asarray(list(candidate["normalized_echo_density"].values()))
    t_density = np.asarray(list(target["normalized_echo_density"].values()))
    c_band = np.asarray(candidate["band_time_distribution"])
    t_band = np.asarray(target["band_time_distribution"])
    # Gain-invariant log-energy distance across time/frequency tiles.
    floor = max(float(np.max(t_band)) * 1e-7, 1e-14)
    audible = t_band > floor
    log_c = np.log10(c_band[audible] + floor)
    log_t = np.log10(t_band[audible] + floor)
    return {
        "echo_density_rmse": rounded(float(np.sqrt(np.mean((c_density - t_density) ** 2))), 6),
        "log_band_energy_rmse": rounded(float(np.sqrt(np.mean((log_c - log_t) ** 2))), 6),
    }


def main() -> None:
    args = parse_args()
    if OUTPUT_SAMPLE_RATE != SAMPLE_RATE:
        raise RuntimeError("room audition sample-rate mismatch")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tropical-clouds-match-") as temp:
        binary = Path(temp) / "clouds_reference"
        build_reference_binary(args.clouds_root, binary)
        source_count = round(DURATION_SECONDS * RENDER_SAMPLE_RATE)
        dry_32k = modal_signal(snare_modes(0), RENDER_SAMPLE_RATE, source_count)
        impulse_32k = np.zeros(source_count, dtype=np.float64)
        impulse_32k[0] = 1.0
        gold_wet_32k = render_clouds(binary, dry_32k)
        gold_impulse_32k = render_clouds(binary, impulse_32k)

    dry = resample_linear(dry_32k, RENDER_SAMPLE_RATE, SAMPLE_RATE)
    gold_wet_stereo = resample_linear(gold_wet_32k, RENDER_SAMPLE_RATE, SAMPLE_RATE)
    gold_impulse_stereo = resample_linear(
        gold_impulse_32k, RENDER_SAMPLE_RATE, SAMPLE_RATE
    )
    gold_wet = np.mean(gold_wet_stereo, axis=1)
    gold_impulse = np.mean(gold_impulse_stereo, axis=1)
    shell_energy, band_sigmas = fit_band_profile(gold_impulse)
    target_features = feature_summary(gold_impulse)

    tracks: dict[str, np.ndarray] = {"clouds_gold_wet_mono": gold_wet}
    profiles: dict[str, object] = {}
    results: dict[str, object] = {}
    candidate_impulses: dict[str, np.ndarray] = {}
    for candidate_id, cell_count in CANDIDATES:
        rows, impulse = make_modal_room(cell_count, shell_energy, band_sigmas)
        wet = fft_convolve(dry, impulse)
        candidate_features = feature_summary(impulse)
        results[candidate_id] = {
            "frequency_cells": cell_count,
            "room_rows": len(rows),
            "predelay_ms": PREDELAY_SECONDS * 1000,
            "features": {
                "normalized_echo_density": candidate_features["normalized_echo_density"],
                "spectral_flatness": candidate_features["spectral_flatness"],
            },
            "distance_to_clouds_mono_impulse": feature_distance(
                candidate_features, target_features
            ),
        }
        profiles[candidate_id] = {"room_modes": rows}
        tracks[f"{candidate_id}_wet"] = wet
        candidate_impulses[candidate_id] = impulse

    hybrid_impulses: dict[str, tuple[np.ndarray, tuple[tuple[float, float], ...]]] = {}
    for profile_name, stages in SMEAR_PROFILES.items():
        hybrid_impulses[f"modal_cloud_1024_smear_{profile_name}"] = (
            apply_smear(candidate_impulses["modal_cloud_1024"], stages), stages
        )
    mutable_stages = SMEAR_PROFILES["mutable_rate"]
    hybrid_impulses["modal_cloud_1024_diffused_smear_mutable_rate"] = (
        apply_smear(discrete_convolve(
            candidate_impulses["modal_cloud_1024"], diffuser_kernel()
        ), mutable_stages),
        mutable_stages,
    )
    for candidate_id, (impulse, smear_stages) in hybrid_impulses.items():
        wet = fft_convolve(dry, impulse)
        candidate_features = feature_summary(impulse)
        results[candidate_id] = {
            "base_frequency_cells": 1024,
            "base_room_rows": 2048,
            "diffuser": "four six-repeat finite allpass projections"
                if "diffused" in candidate_id else None,
            "smear_stages": [
                {"depth_ms": depth * 1000.0, "rate_hz": rate}
                for depth, rate in smear_stages
            ],
            "features": {
                "normalized_echo_density": candidate_features["normalized_echo_density"],
                "spectral_flatness": candidate_features["spectral_flatness"],
            },
            "distance_to_clouds_mono_impulse": feature_distance(
                candidate_features, target_features
            ),
        }
        profiles[candidate_id] = {
            "base": "modal_cloud_1024",
            "diffuser_delay_samples_at_32k": list(DIFFUSER_DELAYS_32K)
                if "diffused" in candidate_id else [],
            "diffuser_gain": 0.70 if "diffused" in candidate_id else None,
            "diffuser_repeats": 6 if "diffused" in candidate_id else None,
            "smear_stages": smear_stages,
        }
        tracks[f"{candidate_id}_wet"] = wet

    # One wet RMS target and mix ratio for every comparison.
    reference_rms = active_rms(gold_wet)
    normalized_tracks: dict[str, np.ndarray] = {}
    for name, wet in tracks.items():
        normalized_wet = wet * (reference_rms / max(active_rms(wet), 1e-30))
        normalized_tracks[name] = normalized_wet
        normalized_tracks[name.replace("_wet", "_mix")] = dry + normalized_wet * 1.15
    common_gain = math.pow(10.0, -1.0 / 20.0) / max(
        float(np.max(np.abs(track))) for track in normalized_tracks.values()
    )
    for name, track in normalized_tracks.items():
        write_wav(OUT_DIR / f"{name}.wav", track * common_gain)

    target_public = {
        "normalized_echo_density": target_features["normalized_echo_density"],
        "spectral_flatness": target_features["spectral_flatness"],
    }
    summary = {
        "reference": "Mutable Instruments Clouds mono fold-down",
        "source_commit": CLOUDS_COMMIT,
        "method": "modal capacity plus closed-form diffusion/smear experiment; spectral/decay projection and staged bloom shells",
        "target_features": target_public,
        "fitted_shell_energy": [
            [rounded(float(value), 8) for value in row]
            for row in shell_energy
        ],
        "fitted_band_sigma": [rounded(float(value), 6) for value in band_sigmas],
        "candidates": results,
        "common_headroom_gain": rounded(common_gain, 9),
    }
    (OUT_DIR / "match_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "modal_profiles.json").write_text(
        json.dumps(profiles, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
