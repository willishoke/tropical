#!/usr/bin/env python3
"""Fit the fixed-cost grouped room representation to the Clouds impulse.

This experiment keeps the twelve pole-lattice periods proven by
run_structured_recovery.py, but treats each periodic carrier as an offline-fit
asset instead of forcing its residues to come from the Freeverb-style teacher.
The runtime representation and analytic modal-source convolution are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np

from run import TWO_PI, active_rms, rounded, snare_modes
from run_modal_match import feature_distance, feature_summary
from run_reference import (
    CLOUDS_COMMIT,
    DURATION_SECONDS,
    OUTPUT_SAMPLE_RATE,
    RENDER_SAMPLE_RATE,
    build_reference_binary,
    modal_signal,
    render_clouds,
    resample_linear,
    write_wav,
)
from run_structured_recovery import (
    discrete_convolve,
    derive_modal_decomposition,
    scaled_geometric,
    synthesize_modal_impulse,
)


AUDITION_DIR = Path(__file__).resolve().parent
OUT_DIR = AUDITION_DIR / "grouped_fit_out"
SAMPLE_RATE = RENDER_SAMPLE_RATE
SAMPLE_COUNT = round(DURATION_SECONDS * SAMPLE_RATE)
FIT_SWEEPS = 14
NATIVE_SAMPLE_RATE = 44_100
NATIVE_PERIODS = (
    1116,
    1188,
    1277,
    1356,
    1422,
    1491,
    1557,
    1617,
    1112,
    882,
    682,
    450,
)
DECAY_LADDER_RT60 = (
    0.22,
    0.32,
    0.46,
    0.68,
    1.00,
    1.45,
    2.10,
    3.00,
    4.20,
    5.80,
    7.80,
    10.50,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clouds-root",
        type=Path,
        default=Path(os.environ.get(
            "CLOUDS_REFERENCE_ROOT", "/private/tmp/tropical-clouds-reference"
        )),
        help="official pichenettes/eurorack checkout at the frozen commit",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        choices=(RENDER_SAMPLE_RATE, NATIVE_SAMPLE_RATE),
        default=RENDER_SAMPLE_RATE,
        help="fit/evaluation rate after the single frozen-target resample",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR,
        help="directory for fitted carriers, summaries, and audition WAVs",
    )
    return parser.parse_args()


def configure(sample_rate: int, output_dir: Path) -> None:
    global OUT_DIR, SAMPLE_RATE, SAMPLE_COUNT
    OUT_DIR = output_dir.resolve()
    SAMPLE_RATE = sample_rate
    SAMPLE_COUNT = round(DURATION_SECONDS * SAMPLE_RATE)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha256(array: np.ndarray, dtype: str) -> str:
    canonical = np.ascontiguousarray(array, dtype=np.dtype(dtype))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def carrier_source_sha256(
    sample_rate: int,
    periods: tuple[int, ...],
    radii: tuple[float, ...],
    carriers: list[np.ndarray],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"tropical-grouped-room-carriers-v1\0")
    digest.update(np.asarray([sample_rate], dtype="<u4").tobytes())
    digest.update(np.asarray(periods, dtype="<u4").tobytes())
    digest.update(np.asarray(radii, dtype="<f8").tobytes())
    for carrier in carriers:
        digest.update(np.asarray(carrier, dtype="<f8").tobytes())
    return digest.hexdigest()


def to_output_rate(signal: np.ndarray) -> np.ndarray:
    if SAMPLE_RATE == OUTPUT_SAMPLE_RATE:
        return signal.copy()
    return resample_linear(signal, SAMPLE_RATE, OUTPUT_SAMPLE_RATE)


def radius_for_rt60(rt60_seconds: float) -> float:
    return math.exp(-math.log(1000.0) / (rt60_seconds * SAMPLE_RATE))


def fit_weights(target: np.ndarray) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Equalize log-time regions without chasing the quantization floor."""
    edges_seconds = (0.0, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 4.0, 8.0, 12.0)
    regions: list[tuple[int, int, float]] = []
    peak_rms = 0.0
    for begin_s, end_s in zip(edges_seconds[:-1], edges_seconds[1:]):
        begin = round(begin_s * SAMPLE_RATE)
        end = min(len(target), round(end_s * SAMPLE_RATE))
        rms = float(np.sqrt(np.mean(np.square(target[begin:end]))))
        peak_rms = max(peak_rms, rms)
        regions.append((begin, end, rms))

    weights_squared = np.zeros(len(target), dtype=np.float64)
    public: list[dict[str, float]] = []
    floor = peak_rms * 1e-3
    for (begin, end, rms), begin_s, end_s in zip(
        regions, edges_seconds[:-1], edges_seconds[1:]
    ):
        count = max(1, end - begin)
        effective_rms = max(rms, floor)
        weights_squared[begin:end] = 1.0 / (
            count * effective_rms * effective_rms
        )
        public.append({
            "begin_seconds": begin_s,
            "end_seconds": end_s,
            "target_rms": rms,
        })
    weights_squared /= max(float(np.mean(weights_squared)), 1e-30)
    return weights_squared, public


def synthesize_carriers(
    periods: tuple[int, ...], radii: tuple[float, ...], carriers: list[np.ndarray]
) -> np.ndarray:
    n = np.arange(SAMPLE_COUNT, dtype=np.int64)
    result = np.zeros(SAMPLE_COUNT, dtype=np.float64)
    for period, radius, carrier in zip(periods, radii, carriers):
        result += np.power(radius, n) * carrier[n % period]
    return result


def fit_carriers(
    target: np.ndarray,
    periods: tuple[int, ...],
    radii: tuple[float, ...],
    weights_squared: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, list[float]]:
    """Block coordinate descent for the exact sparse linear least-squares fit."""
    n = np.arange(len(target), dtype=np.int64)
    designs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    carriers: list[np.ndarray] = []
    for period, radius in zip(periods, radii):
        remainder = n % period
        envelope = np.power(radius, n)
        denominator = np.bincount(
            remainder,
            weights=weights_squared * envelope * envelope,
            minlength=period,
        )
        designs.append((remainder, envelope, denominator))
        carriers.append(np.zeros(period, dtype=np.float64))

    model = np.zeros(len(target), dtype=np.float64)
    history: list[float] = []
    target_energy = float(np.dot(weights_squared * target, target))
    for sweep in range(FIT_SWEEPS):
        indices = range(len(periods)) if sweep % 2 == 0 else range(len(periods) - 1, -1, -1)
        for index in indices:
            remainder, envelope, denominator = designs[index]
            old_carrier = carriers[index]
            old_contribution = envelope * old_carrier[remainder]
            residual_without = target - model + old_contribution
            numerator = np.bincount(
                remainder,
                weights=weights_squared * envelope * residual_without,
                minlength=len(old_carrier),
            )
            ridge = max(float(np.max(denominator)) * 1e-10, 1e-30)
            new_carrier = numerator / (denominator + ridge)
            model += envelope * (new_carrier - old_carrier)[remainder]
            carriers[index] = new_carrier
        error = target - model
        history.append(math.sqrt(
            float(np.dot(weights_squared * error, error))
            / max(target_energy, 1e-30)
        ))
    return carriers, model, history


def structured_modal_response(
    rows: list[list[float]],
    periods: tuple[int, ...],
    radii: tuple[float, ...],
    carriers: list[np.ndarray],
) -> np.ndarray:
    """Random-access analytic convolution of fitted carriers with modal rows."""
    n = np.arange(SAMPLE_COUNT, dtype=np.int64)
    total = np.zeros(SAMPLE_COUNT, dtype=np.complex128)
    for frequency, sigma, amplitude, phase in rows:
        pole = np.exp(complex(-sigma, TWO_PI * frequency) / SAMPLE_RATE)
        coefficient = amplitude * np.exp(1j * phase)
        response = np.zeros(SAMPLE_COUNT, dtype=np.complex128)
        for period, radius, carrier in zip(periods, radii, carriers):
            quotient = n // period
            remainder = n - quotient * period
            ratio = radius / pole
            prefix = np.cumsum(
                carrier * np.power(ratio, np.arange(period, dtype=np.int64))
            )
            prefix_at_remainder = prefix[remainder]
            ratio_period = ratio ** period
            through_remainder = scaled_geometric(
                pole, radius, period, n, quotient + 1, ratio_period
            )
            after_remainder = scaled_geometric(
                pole, radius, period, n, quotient, ratio_period
            )
            response += (
                prefix_at_remainder * through_remainder
                + (prefix[-1] - prefix_at_remainder) * after_remainder
            )
        total += coefficient * response
    return total.real


def best_gain_nrmse(candidate: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    denominator = float(np.dot(candidate, candidate))
    gain = float(np.dot(candidate, target) / denominator) if denominator else 0.0
    error = float(
        np.linalg.norm(gain * candidate - target)
        / max(np.linalg.norm(target), 1e-30)
    )
    return gain, error


def save_carriers(
    path: Path,
    sample_rate: int,
    periods: tuple[int, ...],
    radii: tuple[float, ...],
    carriers: list[np.ndarray],
) -> None:
    arrays: dict[str, np.ndarray] = {
        "sample_rate": np.asarray(sample_rate, dtype=np.int32),
        "periods": np.asarray(periods, dtype=np.int32),
        "radii": np.asarray(radii, dtype=np.float64),
    }
    arrays.update({f"carrier_{index}": carrier for index, carrier in enumerate(carriers)})
    np.savez_compressed(path, **arrays)


def save_stereo_carriers(
    path: Path,
    periods: tuple[int, ...],
    radii: tuple[float, ...],
    mid_carriers: list[np.ndarray],
    side_carriers: list[np.ndarray],
) -> None:
    arrays: dict[str, np.ndarray] = {
        "periods": np.asarray(periods, dtype=np.int32),
        "radii": np.asarray(radii, dtype=np.float64),
    }
    for index, (mid, side) in enumerate(zip(mid_carriers, side_carriers)):
        arrays[f"mid_carrier_{index}"] = mid
        arrays[f"side_carrier_{index}"] = side
    np.savez_compressed(path, **arrays)


def phase_scrambled_carriers(carriers: list[np.ndarray]) -> list[np.ndarray]:
    """Preserve per-lattice modal magnitudes while decorrelating stereo phase."""
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    result: list[np.ndarray] = []
    for group_index, carrier in enumerate(carriers):
        spectrum = np.fft.rfft(carrier)
        bins = np.arange(len(spectrum), dtype=np.float64)
        phase = golden_angle * (group_index + 1) * (bins + 1.0)
        scrambled = np.abs(spectrum) * np.exp(1j * phase)
        scrambled[0] = spectrum[0].real * (-1.0 if group_index % 2 else 1.0)
        if len(carrier) % 2 == 0:
            scrambled[-1] = spectrum[-1].real * (
                -1.0 if group_index % 3 else 1.0
            )
        result.append(np.fft.irfft(scrambled, len(carrier)))
    return result


def stereo_field_summary(signal: np.ndarray) -> dict[str, object]:
    result: dict[str, object] = {}
    for label, begin_s, end_s in (
        ("100_300ms", 0.1, 0.3),
        ("500_1000ms", 0.5, 1.0),
    ):
        begin = round(begin_s * SAMPLE_RATE)
        end = round(end_s * SAMPLE_RATE)
        left = signal[begin:end, 0]
        right = signal[begin:end, 1]
        denominator = math.sqrt(
            float(np.dot(left, left)) * float(np.dot(right, right))
        )
        correlation = float(np.dot(left, right) / denominator) if denominator else 0.0
        mid = 0.5 * (left + right)
        side = 0.5 * (left - right)
        side_mid_ratio = float(
            np.sqrt(np.mean(np.square(side)))
            / max(float(np.sqrt(np.mean(np.square(mid)))), 1e-30)
        )
        result[label] = {
            "left_right_correlation": rounded(correlation, 6),
            "side_to_mid_rms": rounded(side_mid_ratio, 6),
        }
    return result


def main() -> None:
    args = parse_args()
    configure(args.sample_rate, args.output_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tropical-clouds-grouped-fit-") as temp:
        binary = Path(temp) / "clouds_reference"
        build_reference_binary(args.clouds_root, binary)
        dry_rows = snare_modes(0)
        reference_count = round(DURATION_SECONDS * RENDER_SAMPLE_RATE)
        reference_impulse = np.zeros(reference_count, dtype=np.float64)
        reference_impulse[0] = 1.0
        reference_dry = modal_signal(
            dry_rows, RENDER_SAMPLE_RATE, reference_count
        )
        reference_impulse_stereo = render_clouds(binary, reference_impulse)
        reference_wet_stereo = render_clouds(binary, reference_dry)

    # Clouds is always rendered at its frozen 32 kHz native rate.  Resample the
    # complete target exactly once before fitting; every operation below runs
    # at SAMPLE_RATE.
    clouds_impulse_stereo = resample_linear(
        reference_impulse_stereo, RENDER_SAMPLE_RATE, SAMPLE_RATE
    )
    clouds_wet_stereo = resample_linear(
        reference_wet_stereo, RENDER_SAMPLE_RATE, SAMPLE_RATE
    )
    dry = modal_signal(dry_rows, SAMPLE_RATE, SAMPLE_COUNT)

    clouds_impulse = np.mean(clouds_impulse_stereo, axis=1)
    clouds_wet = np.mean(clouds_wet_stereo, axis=1)
    teacher_groups, teacher_direct = derive_modal_decomposition()
    reference_periods = tuple(
        group.delay if group.kind == "comb" else 2 * group.delay
        for group in teacher_groups
    )
    periods = (
        NATIVE_PERIODS if SAMPLE_RATE == NATIVE_SAMPLE_RATE else reference_periods
    )
    current_radii = tuple(
        group.radius ** (RENDER_SAMPLE_RATE / SAMPLE_RATE)
        for group in teacher_groups
    )
    decay_ladder_radii = tuple(radius_for_rt60(value) for value in DECAY_LADDER_RT60)
    reference_teacher_impulse = synthesize_modal_impulse(
        teacher_groups, teacher_direct, reference_count
    )
    teacher_impulse = resample_linear(
        reference_teacher_impulse, RENDER_SAMPLE_RATE, SAMPLE_RATE
    )
    teacher_wet = discrete_convolve(dry, teacher_impulse)
    weights_squared, target_regions = fit_weights(clouds_impulse)

    profiles = {
        "current_radii": current_radii,
        "decay_ladder": decay_ladder_radii,
    }
    impulses_32k: dict[str, np.ndarray] = {
        "clouds_gold": clouds_impulse,
        "freeverb_teacher": teacher_impulse,
    }
    wet_32k: dict[str, np.ndarray] = {
        "clouds_gold": clouds_wet,
        "freeverb_teacher": teacher_wet,
    }
    results: dict[str, object] = {}
    files: dict[str, object] = {}
    fitted_assets: dict[str, tuple[tuple[float, ...], list[np.ndarray]]] = {}

    for profile_name, radii in profiles.items():
        carriers, fitted_impulse, history = fit_carriers(
            clouds_impulse, periods, radii, weights_squared
        )
        convolution_wet = discrete_convolve(dry, fitted_impulse)
        structured_wet = structured_modal_response(
            dry_rows, periods, radii, carriers
        )
        exact_relative_error = float(
            np.linalg.norm(structured_wet - convolution_wet)
            / max(np.linalg.norm(convolution_wet), 1e-30)
        )
        impulse_44k = to_output_rate(fitted_impulse)
        gold_44k = to_output_rate(clouds_impulse)
        impulse_gain, impulse_nrmse = best_gain_nrmse(
            fitted_impulse, clouds_impulse
        )
        wet_gain, wet_nrmse = best_gain_nrmse(structured_wet, clouds_wet)
        profile_path = OUT_DIR / f"grouped_fit_{profile_name}.npz"
        save_carriers(profile_path, SAMPLE_RATE, periods, radii, carriers)
        files[profile_path.name] = {
            "sha256": sha256(profile_path),
            "canonical_carrier_source_sha256": carrier_source_sha256(
                SAMPLE_RATE, periods, radii, carriers
            ),
        }
        candidate_features = feature_summary(impulse_44k)
        gold_features = feature_summary(gold_44k)
        results[profile_name] = {
            "rt60_seconds": [
                rounded(-math.log(1000.0) / (SAMPLE_RATE * math.log(radius)), 6)
                for radius in radii
            ],
            "weighted_fit_nrmse_history": [rounded(value, 8) for value in history],
            "impulse_best_gain": rounded(impulse_gain, 9),
            "impulse_waveform_nrmse": rounded(impulse_nrmse, 6),
            "modal_source_vs_actual_clouds_best_gain": rounded(wet_gain, 9),
            "modal_source_vs_actual_clouds_waveform_nrmse": rounded(wet_nrmse, 6),
            "structured_source_vs_convolution_relative_l2": exact_relative_error,
            "features": {
                "normalized_echo_density": candidate_features["normalized_echo_density"],
                "spectral_flatness": candidate_features["spectral_flatness"],
            },
            "distance_to_clouds": feature_distance(
                candidate_features, gold_features
            ),
        }
        impulses_32k[f"grouped_fit_{profile_name}"] = fitted_impulse
        wet_32k[f"grouped_fit_{profile_name}"] = structured_wet
        fitted_assets[profile_name] = (radii, carriers)

    # Clouds modulation makes the exact side waveform deliberately
    # nonstationary, so a stationary waveform fit projects almost none of it.
    # Preserve the fitted mid-channel modal magnitudes and scramble only their
    # phases instead.  This targets the audible stereo statistics without
    # pretending to recover the LFO trajectory.
    mid_radii, mid_carriers = fitted_assets["current_radii"]
    clouds_side_impulse = 0.5 * (
        clouds_impulse_stereo[:, 0] - clouds_impulse_stereo[:, 1]
    )
    side_carriers = phase_scrambled_carriers(mid_carriers)
    fitted_side_impulse = synthesize_carriers(
        periods, mid_radii, side_carriers
    )
    fitted_mid_impulse = impulses_32k["grouped_fit_current_radii"]
    target_side_mid_ratio = float(
        np.linalg.norm(clouds_side_impulse)
        / max(np.linalg.norm(clouds_impulse), 1e-30)
    )
    diffuse_begin = round(0.1 * SAMPLE_RATE)
    diffuse_end = round(1.0 * SAMPLE_RATE)
    target_diffuse_side_mid_ratio = float(
        np.linalg.norm(clouds_side_impulse[diffuse_begin:diffuse_end])
        / max(
            np.linalg.norm(clouds_impulse[diffuse_begin:diffuse_end]),
            1e-30,
        )
    )
    fitted_side_scale = target_diffuse_side_mid_ratio * float(
        np.linalg.norm(fitted_mid_impulse[diffuse_begin:diffuse_end])
        / max(
            np.linalg.norm(fitted_side_impulse[diffuse_begin:diffuse_end]),
            1e-30,
        )
    )
    side_carriers = [carrier * fitted_side_scale for carrier in side_carriers]
    fitted_side_impulse *= fitted_side_scale
    fitted_stereo_impulse = np.column_stack((
        fitted_mid_impulse + fitted_side_impulse,
        fitted_mid_impulse - fitted_side_impulse,
    ))
    structured_mid_wet = wet_32k["grouped_fit_current_radii"]
    structured_side_wet = structured_modal_response(
        dry_rows, periods, mid_radii, side_carriers
    )
    fitted_stereo_wet = np.column_stack((
        structured_mid_wet + structured_side_wet,
        structured_mid_wet - structured_side_wet,
    ))
    stereo_gain, stereo_nrmse = best_gain_nrmse(
        fitted_stereo_impulse.reshape(-1), clouds_impulse_stereo.reshape(-1)
    )
    stereo_wet_gain, stereo_wet_nrmse = best_gain_nrmse(
        fitted_stereo_wet.reshape(-1), clouds_wet_stereo.reshape(-1)
    )
    side_convolution_wet = discrete_convolve(dry, fitted_side_impulse)
    side_exact_error = float(
        np.linalg.norm(structured_side_wet - side_convolution_wet)
        / max(np.linalg.norm(side_convolution_wet), 1e-30)
    )
    stereo_profile_path = OUT_DIR / "grouped_fit_current_radii_stereo.npz"
    save_stereo_carriers(
        stereo_profile_path,
        periods,
        mid_radii,
        mid_carriers,
        side_carriers,
    )
    files[stereo_profile_path.name] = {"sha256": sha256(stereo_profile_path)}
    results["current_radii"]["stereo"] = {
        "side_method": "deterministic residue-phase scrambling with target side/mid energy",
        "target_side_mid_full_ir_ratio": rounded(target_side_mid_ratio, 6),
        "target_side_mid_100_1000ms_ratio": rounded(
            target_diffuse_side_mid_ratio, 6
        ),
        "fitted_side_scale": rounded(fitted_side_scale, 9),
        "impulse_best_gain": rounded(stereo_gain, 9),
        "impulse_waveform_nrmse": rounded(stereo_nrmse, 6),
        "modal_source_vs_actual_clouds_best_gain": rounded(stereo_wet_gain, 9),
        "modal_source_vs_actual_clouds_waveform_nrmse": rounded(
            stereo_wet_nrmse, 6
        ),
        "structured_side_vs_convolution_relative_l2": side_exact_error,
        "clouds_field": stereo_field_summary(clouds_impulse_stereo),
        "fitted_field": stereo_field_summary(fitted_stereo_impulse),
    }
    impulses_32k["clouds_gold_stereo"] = clouds_impulse_stereo
    impulses_32k["grouped_fit_current_radii_stereo"] = fitted_stereo_impulse
    wet_32k["clouds_gold_stereo"] = clouds_wet_stereo
    wet_32k["grouped_fit_current_radii_stereo"] = fitted_stereo_wet

    # Impulses and source auditions are independently loudness-matched to their
    # Clouds controls, with one shared headroom gain inside each comparison set.
    for suffix, tracks, gold_name in (
        ("impulse", impulses_32k, "clouds_gold"),
        ("wet", wet_32k, "clouds_gold"),
    ):
        tracks_44k = {
            name: to_output_rate(track)
            for name, track in tracks.items()
        }
        target_rms = active_rms(tracks_44k[gold_name])
        normalized = {
            name: track * (target_rms / max(active_rms(track), 1e-30))
            for name, track in tracks_44k.items()
        }
        common_gain = math.pow(10.0, -1.0 / 20.0) / max(
            float(np.max(np.abs(track))) for track in normalized.values()
        )
        for name, track in normalized.items():
            path = OUT_DIR / f"{name}_{suffix}.wav"
            write_wav(path, track * common_gain)
            files[path.name] = {"sha256": sha256(path)}

    summary = {
        "question": "Can Clouds' lush fixed impulse be fitted without increasing grouped runtime cost?",
        "clouds_commit": CLOUDS_COMMIT,
        "reference_sample_rate": RENDER_SAMPLE_RATE,
        "fit_sample_rate": SAMPLE_RATE,
        "target_resampling": "one linear resample before optimizer",
        "target_mono_float64_le_sha256": array_sha256(clouds_impulse, "<f8"),
        "method": "weighted offline least squares over complete periodic pole-lattice carriers",
        "fit_sweeps": FIT_SWEEPS,
        "periods": periods,
        "runtime_cost": {
            "groups": len(periods),
            "metal_source_rows": len(dry_rows),
            "source_group_interactions_per_hit": len(periods) * len(dry_rows),
            "four_hit_interactions": 4 * len(periods) * len(dry_rows),
            "carrier_samples": sum(periods),
            "prefix_bytes_complex64": sum(periods) * len(dry_rows) * 8,
            "stereo_prefix_bytes_complex64": sum(periods) * len(dry_rows) * 2 * 8,
            "stereo_note": "same group/exponential evaluations; two prefix reads and accumulators",
        },
        "target_regions": target_regions,
        "profiles": results,
        "files": files,
    }
    summary_path = OUT_DIR / "grouped_fit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
