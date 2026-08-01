#!/usr/bin/env python3
"""Prove classic reverse reverb and a continuous temporal-position control.

The target operation is reverse(source) -> causal room -> reverse(result),
which equals the original source convolved with the anti-causal room kernel.
It is not the same as mirroring the already-composed forward wet signal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from run import TWO_PI, rounded, snare_modes
from run_grouped_clouds_fit import (
    SAMPLE_COUNT,
    SAMPLE_RATE,
    configure as configure_grouped_fit,
    structured_modal_response,
    synthesize_carriers,
)
from run_reference import OUTPUT_SAMPLE_RATE, resample_linear, write_wav


AUDITION_DIR = Path(__file__).resolve().parent
ASSET_PATH = AUDITION_DIR / "grouped_fit_out" / "grouped_fit_current_radii.npz"
STEREO_ASSET_PATH = (
    AUDITION_DIR / "grouped_fit_out" / "grouped_fit_current_radii_stereo.npz"
)
OUT_DIR = AUDITION_DIR / "reverse_position_out"
EVENT_SECONDS = 6.0
POSITION_VALUES = (-1.0, -0.5, 0.0, 0.5, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--asset", type=Path, default=ASSET_PATH)
    parser.add_argument("--stereo-asset", type=Path, default=STEREO_ASSET_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--mono",
        action="store_true",
        help="render the frozen production mono contract without side evidence",
    )
    return parser.parse_args()


def configure(sample_rate: int, asset: Path, output_dir: Path) -> None:
    global SAMPLE_RATE, SAMPLE_COUNT, ASSET_PATH, OUT_DIR
    SAMPLE_RATE = sample_rate
    SAMPLE_COUNT = round(12.0 * sample_rate)
    ASSET_PATH = asset.resolve()
    OUT_DIR = output_dir.resolve()
    configure_grouped_fit(sample_rate, output_dir)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repository_relative(path: Path) -> str:
    repository_root = AUDITION_DIR.parents[2]
    try:
        return str(path.resolve().relative_to(repository_root))
    except ValueError:
        return str(path.resolve())


def to_output_rate(signal: np.ndarray) -> np.ndarray:
    if SAMPLE_RATE == OUTPUT_SAMPLE_RATE:
        return signal.copy()
    return resample_linear(signal, SAMPLE_RATE, OUTPUT_SAMPLE_RATE)


def fft_convolve_full(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    wanted = len(a) + len(b) - 1
    fft_size = 1 << (wanted - 1).bit_length()
    result = np.fft.irfft(
        np.fft.rfft(a, fft_size) * np.fft.rfft(b, fft_size), fft_size
    )
    return result[:wanted]


def load_grouped_asset() -> tuple[tuple[int, ...], tuple[float, ...], list[np.ndarray]]:
    with np.load(ASSET_PATH) as data:
        if "sample_rate" in data and int(data["sample_rate"]) != SAMPLE_RATE:
            raise RuntimeError(
                f"grouped asset rate {int(data['sample_rate'])} != {SAMPLE_RATE}"
            )
        periods = tuple(int(value) for value in data["periods"])
        radii = tuple(float(value) for value in data["radii"])
        carriers = [
            np.asarray(data[f"carrier_{index}"], dtype=np.float64)
            for index in range(len(periods))
        ]
    return periods, radii, carriers


def load_stereo_side_asset() -> list[np.ndarray]:
    with np.load(STEREO_ASSET_PATH) as data:
        periods = data["periods"]
        return [
            np.asarray(data[f"side_carrier_{index}"], dtype=np.float64)
            for index in range(len(periods))
        ]


def cyclic_future_table(carrier: np.ndarray, z: complex) -> np.ndarray:
    """B[R] = sum_(s>=0)^P-1 carrier[(R+s) mod P] z^s."""
    period = len(carrier)
    powers = np.power(z, np.arange(period, dtype=np.int64))
    weighted = carrier * powers
    prefix = np.empty(period + 1, dtype=np.complex128)
    prefix[0] = 0.0j
    prefix[1:] = np.cumsum(weighted)
    starts = np.arange(period, dtype=np.int64)
    return np.power(z, -starts) * (
        (prefix[-1] - prefix[starts]) + (z ** period) * prefix[starts]
    )


def anti_causal_modal_response(
    rows: list[list[float]],
    periods: tuple[int, ...],
    radii: tuple[float, ...],
    carriers: list[np.ndarray],
    relative_samples: np.ndarray,
) -> np.ndarray:
    """Exact x * reverse(h) for causal analytic modal x and grouped h.

    For x(u) = C exp(lambda u / Fs) gate(u) and
    h[k] = r^k c[k mod P] gate(k),

      y_rev(u) = C exp(lambda (u + d) / Fs)
                 r^d B[d mod P] / (1 - (pr)^P),
      d = max(ceil(-u), 0).

    The infinite future sum converges because both source and room poles are
    stable.  Negative output coordinates therefore need no sequential buffer.
    """
    u = np.asarray(relative_samples, dtype=np.float64)
    pre_age = np.maximum(np.ceil(-u), 0.0).astype(np.int64)
    total = np.zeros(len(u), dtype=np.complex128)
    for frequency, sigma, amplitude, phase in rows:
        exponent = complex(-sigma, TWO_PI * frequency) / SAMPLE_RATE
        pole = np.exp(exponent)
        coefficient = amplitude * np.exp(1j * phase)
        response = np.zeros(len(u), dtype=np.complex128)
        pole_age = np.exp(exponent * (u + pre_age))
        for period, radius, carrier in zip(periods, radii, carriers):
            z = pole * radius
            future = cyclic_future_table(carrier, z)
            denominator = 1.0 - z ** period
            response += (
                pole_age
                * np.power(radius, pre_age)
                * future[pre_age % period]
                / denominator
            )
        total += coefficient * response
    return total.real


def scaled_geometric_real(
    exponent: complex,
    radius: float,
    period: int,
    u: np.ndarray,
    length: np.ndarray,
    ratio_period: complex,
) -> np.ndarray:
    p_to_u = np.exp(exponent * u)
    if abs(1.0 - ratio_period) < 1e-10:
        return length * p_to_u
    span = period * length
    return (
        p_to_u
        - np.power(radius, span) * np.exp(exponent * (u - span))
    ) / (1.0 - ratio_period)


def causal_modal_response(
    rows: list[list[float]],
    periods: tuple[int, ...],
    radii: tuple[float, ...],
    carriers: list[np.ndarray],
    relative_samples: np.ndarray,
) -> np.ndarray:
    """Exact grouped causal response at real scene coordinates."""
    u = np.asarray(relative_samples, dtype=np.float64)
    supported = u >= 0.0
    total = np.zeros(len(u), dtype=np.complex128)
    if not np.any(supported):
        return total.real
    active_u = u[supported]
    m = np.floor(active_u).astype(np.int64)
    active_total = np.zeros(len(active_u), dtype=np.complex128)
    for frequency, sigma, amplitude, phase in rows:
        exponent = complex(-sigma, TWO_PI * frequency) / SAMPLE_RATE
        pole = np.exp(exponent)
        coefficient = amplitude * np.exp(1j * phase)
        response = np.zeros(len(active_u), dtype=np.complex128)
        for period, radius, carrier in zip(periods, radii, carriers):
            quotient = m // period
            remainder = m - quotient * period
            ratio = radius / pole
            prefix = np.cumsum(
                carrier * np.power(ratio, np.arange(period, dtype=np.int64))
            )
            ratio_period = ratio ** period
            through_remainder = scaled_geometric_real(
                exponent,
                radius,
                period,
                active_u,
                quotient + 1,
                ratio_period,
            )
            after_remainder = scaled_geometric_real(
                exponent,
                radius,
                period,
                active_u,
                quotient,
                ratio_period,
            )
            prefix_at_remainder = prefix[remainder]
            response += (
                prefix_at_remainder * through_remainder
                + (prefix[-1] - prefix_at_remainder) * after_remainder
            )
        active_total += coefficient * response
    total[supported] = active_total
    return total.real


def modal_signal_at(rows: list[list[float]], ages: np.ndarray) -> np.ndarray:
    ages = np.asarray(ages, dtype=np.float64)
    result = np.zeros(len(ages), dtype=np.float64)
    supported = ages >= 0.0
    t = ages[supported] / SAMPLE_RATE
    for frequency, sigma, amplitude, phase in rows:
        result[supported] += amplitude * np.exp(-sigma * t) * np.cos(
            TWO_PI * frequency * t + phase
        )
    return result


def literal_causal_response(
    rows: list[list[float]], room_impulse: np.ndarray, coordinates: np.ndarray
) -> np.ndarray:
    result = np.zeros(len(coordinates), dtype=np.float64)
    for index, u in enumerate(np.asarray(coordinates, dtype=np.float64)):
        if u < 0.0:
            continue
        count = min(len(room_impulse), math.floor(u) + 1)
        k = np.arange(count, dtype=np.float64)
        result[index] = np.dot(
            room_impulse[:count], modal_signal_at(rows, u - k)
        )
    return result


def literal_reverse_response(
    rows: list[list[float]], room_impulse: np.ndarray, coordinates: np.ndarray
) -> np.ndarray:
    result = np.zeros(len(coordinates), dtype=np.float64)
    for index, u in enumerate(np.asarray(coordinates, dtype=np.float64)):
        begin = max(math.ceil(-u), 0)
        if begin >= len(room_impulse):
            continue
        k = np.arange(begin, len(room_impulse), dtype=np.float64)
        result[index] = np.dot(
            room_impulse[begin:], modal_signal_at(rows, u + k)
        )
    return result


def relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(candidate - reference)
        / max(np.linalg.norm(reference), 1e-30)
    )


def best_gain_nrmse(candidate: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    denominator = float(np.dot(candidate, candidate))
    gain = float(np.dot(candidate, target) / denominator) if denominator else 0.0
    error = float(
        np.linalg.norm(gain * candidate - target)
        / max(np.linalg.norm(target), 1e-30)
    )
    return gain, error


def equal_power_position(
    forward: np.ndarray, reverse: np.ndarray, position: np.ndarray | float
) -> np.ndarray:
    p = np.clip(position, -1.0, 1.0)
    if isinstance(p, np.ndarray) and forward.ndim == 2:
        p = p[:, None]
    forward_gain = np.sqrt(0.5 * (1.0 + p))
    reverse_gain = np.sqrt(0.5 * (1.0 - p))
    return forward_gain * forward + reverse_gain * reverse


def smoothstep(value: np.ndarray) -> np.ndarray:
    x = np.clip(value, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def main() -> None:
    args = parse_args()
    configure(args.sample_rate, args.asset, args.output_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    periods, radii, carriers = load_grouped_asset()
    side_carriers = None if args.mono else load_stereo_side_asset()
    rows = snare_modes(0)
    room_impulse = synthesize_carriers(periods, radii, carriers)
    source = np.zeros(SAMPLE_COUNT, dtype=np.float64)
    t = np.arange(SAMPLE_COUNT, dtype=np.float64) / SAMPLE_RATE
    for frequency, sigma, amplitude, phase in rows:
        source += amplitude * np.exp(-sigma * t) * np.cos(
            TWO_PI * frequency * t + phase
        )

    # Literal finite reverse(source)->room->reverse(result) reference.  Its
    # index zero is relative time -(len(room)-1).
    explicit_reverse = fft_convolve_full(source[::-1], room_impulse)[::-1]
    full_relative = np.arange(-(SAMPLE_COUNT - 1), SAMPLE_COUNT, dtype=np.int64)
    analytic_reverse = anti_causal_modal_response(
        rows, periods, radii, carriers, full_relative
    )
    reference_gain, reference_nrmse = best_gain_nrmse(
        analytic_reverse, explicit_reverse
    )
    reference_max_abs = float(np.max(np.abs(
        reference_gain * analytic_reverse - explicit_reverse
    )))

    integer_coordinates = np.asarray(
        [-2 * SAMPLE_RATE, -SAMPLE_RATE, -1001, -1, 0, 1, 997, SAMPLE_RATE],
        dtype=np.float64,
    )
    fractional_coordinates = np.asarray(
        [
            -SAMPLE_RATE - 0.75,
            -1234.5,
            -0.75,
            0.25,
            123.125,
            SAMPLE_RATE - 0.5,
        ],
        dtype=np.float64,
    )
    causal_integer_coordinates = integer_coordinates[integer_coordinates >= 0]
    causal_fractional_coordinates = fractional_coordinates[
        fractional_coordinates >= 0
    ]
    causal_integer_error = relative_l2(
        causal_modal_response(
            rows, periods, radii, carriers, causal_integer_coordinates
        ),
        literal_causal_response(rows, room_impulse, causal_integer_coordinates),
    )
    causal_fractional_error = relative_l2(
        causal_modal_response(
            rows, periods, radii, carriers, causal_fractional_coordinates
        ),
        literal_causal_response(
            rows, room_impulse, causal_fractional_coordinates
        ),
    )
    reverse_integer_error = relative_l2(
        anti_causal_modal_response(
            rows, periods, radii, carriers, integer_coordinates
        ),
        literal_reverse_response(rows, room_impulse, integer_coordinates),
    )
    reverse_fractional_error = relative_l2(
        anti_causal_modal_response(
            rows, periods, radii, carriers, fractional_coordinates
        ),
        literal_reverse_response(rows, room_impulse, fractional_coordinates),
    )

    side_validation: dict[str, float] = {}
    if side_carriers is not None:
        side_room_impulse = synthesize_carriers(periods, radii, side_carriers)
        explicit_side_reverse = fft_convolve_full(
            source[::-1], side_room_impulse
        )[::-1]
        analytic_side_reverse = anti_causal_modal_response(
            rows, periods, radii, side_carriers, full_relative
        )
        side_reference_gain, side_reference_nrmse = best_gain_nrmse(
            analytic_side_reverse, explicit_side_reverse
        )
        side_reference_max_abs = float(np.max(np.abs(
            side_reference_gain * analytic_side_reverse - explicit_side_reverse
        )))
        side_validation = {
            "stereo_side_analytic_vs_literal_best_gain": side_reference_gain,
            "stereo_side_analytic_vs_literal_nrmse": side_reference_nrmse,
            "stereo_side_analytic_vs_literal_max_abs": side_reference_max_abs,
        }

    # The incumbent direction path mirrors the whole composed forward output.
    # Quantify why that is not a substitute for classic reverse reverb.
    forward_full = fft_convolve_full(source, room_impulse)
    mirrored_composite = forward_full[::-1]
    mirror_gain, mirror_nrmse = best_gain_nrmse(
        mirrored_composite, explicit_reverse
    )

    event_sample = round(EVENT_SECONDS * SAMPLE_RATE)
    timeline = np.arange(SAMPLE_COUNT, dtype=np.int64)
    relative = timeline - event_sample
    forward_from_anchor = np.zeros(SAMPLE_COUNT, dtype=np.float64)
    forward_supported = relative >= 0
    forward_indices = relative[forward_supported]
    forward_from_anchor[forward_supported] = forward_full[forward_indices]
    reverse_from_anchor = anti_causal_modal_response(
        rows, periods, radii, carriers, relative
    )
    forward_stereo: np.ndarray | None = None
    reverse_stereo: np.ndarray | None = None
    if side_carriers is not None:
        forward_side = structured_modal_response(
            rows, periods, radii, side_carriers
        )
        forward_side_from_anchor = np.zeros(SAMPLE_COUNT, dtype=np.float64)
        forward_side_from_anchor[forward_supported] = forward_side[forward_indices]
        reverse_side_from_anchor = anti_causal_modal_response(
            rows, periods, radii, side_carriers, relative
        )
        forward_stereo = np.column_stack((
            forward_from_anchor + forward_side_from_anchor,
            forward_from_anchor - forward_side_from_anchor,
        ))
        reverse_stereo = np.column_stack((
            reverse_from_anchor + reverse_side_from_anchor,
            reverse_from_anchor - reverse_side_from_anchor,
        ))
    dry_from_anchor = np.zeros(SAMPLE_COUNT, dtype=np.float64)
    dry_from_anchor[forward_supported] = source[forward_indices]

    wet_tracks: dict[str, np.ndarray] = {}
    stereo_wet_tracks: dict[str, np.ndarray] = {}
    position_metrics: dict[str, object] = {}
    for position in POSITION_VALUES:
        label = str(position).replace("-", "minus_").replace(".", "p")
        name = f"position_{label}"
        wet_tracks[name] = equal_power_position(
            forward_from_anchor, reverse_from_anchor, position
        )
        if forward_stereo is not None and reverse_stereo is not None:
            stereo_wet_tracks[name] = equal_power_position(
                forward_stereo, reverse_stereo, position
            )
        energy = np.square(wet_tracks[name])
        total_energy = float(np.sum(energy))
        position_metrics[name] = {
            "pre_event_energy_fraction": rounded(
                float(np.sum(energy[relative < 0])) / max(total_energy, 1e-30),
                6,
            ),
            "energy_centroid_seconds_relative_to_event": rounded(
                float(np.dot(relative / SAMPLE_RATE, energy))
                / max(total_energy, 1e-30),
                6,
            ),
        }

    # A demonstrator gesture: while the next event is known but still in the
    # future, scrub from forward to reverse over 1.5 seconds.  This cannot revise
    # already-played audio; it changes the still-future event's pre-tail.
    scrub_begin = round(2.5 * SAMPLE_RATE)
    scrub_end = round(4.0 * SAMPLE_RATE)
    scrub_phase = (timeline - scrub_begin) / max(1, scrub_end - scrub_begin)
    scrub_position = 1.0 - 2.0 * smoothstep(scrub_phase)
    wet_tracks["position_scrub_forward_to_reverse"] = equal_power_position(
        forward_from_anchor, reverse_from_anchor, scrub_position
    )
    if forward_stereo is not None and reverse_stereo is not None:
        stereo_wet_tracks["position_scrub_forward_to_reverse"] = equal_power_position(
            forward_stereo, reverse_stereo, scrub_position
        )

    # One normalization for the complete wet/mix comparison set.
    reference_track = to_output_rate(wet_tracks["position_1p0"])
    reference_rms = float(np.sqrt(np.mean(np.square(reference_track))))
    rendered: dict[str, np.ndarray] = {}
    dry_44k = to_output_rate(dry_from_anchor)
    for name, wet in wet_tracks.items():
        wet_44k = to_output_rate(wet)
        wet_rms = float(np.sqrt(np.mean(np.square(wet_44k))))
        wet_44k *= reference_rms / max(wet_rms, 1e-30)
        rendered[f"{name}_wet"] = wet_44k
        rendered[f"{name}_mix"] = dry_44k + 1.15 * wet_44k
    if stereo_wet_tracks:
        stereo_reference = to_output_rate(stereo_wet_tracks["position_1p0"])
        stereo_reference_rms = float(
            np.sqrt(np.mean(np.square(stereo_reference)))
        )
        dry_stereo_44k = np.column_stack((dry_44k, dry_44k))
        for name, wet in stereo_wet_tracks.items():
            wet_44k = to_output_rate(wet)
            wet_rms = float(np.sqrt(np.mean(np.square(wet_44k))))
            wet_44k *= stereo_reference_rms / max(wet_rms, 1e-30)
            rendered[f"{name}_stereo_wet"] = wet_44k
            rendered[f"{name}_stereo_mix"] = dry_stereo_44k + 1.15 * wet_44k
    common_gain = math.pow(10.0, -1.0 / 20.0) / max(
        float(np.max(np.abs(track))) for track in rendered.values()
    )
    files: dict[str, object] = {}
    for name, track in rendered.items():
        path = OUT_DIR / f"{name}.wav"
        write_wav(path, track * common_gain)
        files[path.name] = {"sha256": sha256(path)}

    summary = {
        "question": "Is classic reverse-reverb position exact and economical for the fitted grouped room?",
        "operation": "reverse(source) -> causal room -> reverse(result) == source * anti_causal(room)",
        "asset": repository_relative(ASSET_PATH),
        "sample_rate": SAMPLE_RATE,
        "output_contract": "mono" if args.mono else "stereo evidence plus mono",
        "event_seconds": EVENT_SECONDS,
        "validation": {
            "analytic_vs_literal_finite_reverse_best_gain": reference_gain,
            "analytic_vs_literal_finite_reverse_nrmse": reference_nrmse,
            "analytic_vs_literal_finite_reverse_max_abs": reference_max_abs,
            "causal_integer_literal_relative_l2": causal_integer_error,
            "causal_fractional_literal_relative_l2": causal_fractional_error,
            "reverse_integer_literal_relative_l2": reverse_integer_error,
            "reverse_fractional_literal_relative_l2": reverse_fractional_error,
            "whole_composite_mirror_vs_classic_best_gain": mirror_gain,
            "whole_composite_mirror_vs_classic_nrmse": mirror_nrmse,
            **side_validation,
        },
        "cost": {
            "groups": len(periods),
            "source_rows": len(rows),
            "one_direction_interactions_per_hit": len(periods) * len(rows),
            "intermediate_position_interactions_per_hit_direct": 2 * len(periods) * len(rows),
            "four_hit_interactions_one_direction": 4 * len(periods) * len(rows),
            "four_hit_interactions_intermediate_direct": 8 * len(periods) * len(rows),
            "reverse_cyclic_prefix_complex64_bytes": sum(periods) * len(rows) * 8,
            "forward_reverse_prefix_complex64_bytes": (
                2 * sum(periods) * len(rows) * 8
            ),
            "cached_forward_reverse_16s_mono_float32_bytes": round(
                2 * 16 * OUTPUT_SAMPLE_RATE * 4
            ),
        },
        "position": {
            "minus_one": "classic reverse pre-tail",
            "zero": "equal-power two-sided bloom",
            "plus_one": "causal forward tail",
            "gesture_constraint": "a live write changes only current/future playback; an upcoming authored event is analytically available",
            "measured_static_positions": position_metrics,
        },
        "files": files,
    }
    summary_path = OUT_DIR / "reverse_position_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
