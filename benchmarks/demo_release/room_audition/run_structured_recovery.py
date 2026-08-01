#!/usr/bin/env python3
"""Feasibility study: recover a delay reverb in closed-form modal space.

The teacher is a deliberately transparent Freeverb-style LTI topology: eight
parallel feedback combs followed by four Schroeder allpasses.  Unlike the
earlier hand-authored room banks, every teacher pole and residue is derived
from the delay topology.  This lets the experiment answer three separate
questions without conflating them:

1. Does the complete modal decomposition reproduce the stateful teacher?
2. How much perceptual quality survives pole-pair truncation at useful orders?
3. Can a feedback comb acting on a known modal source be evaluated directly,
   in constant work per source mode and comb, without expanding its delay?
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import time
from dataclasses import dataclass
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
    parse_args,
    render_clouds,
    resample_linear,
    write_wav,
)


AUDITION_DIR = Path(__file__).resolve().parent
OUT_DIR = AUDITION_DIR / "recovery_out"
SAMPLE_RATE = RENDER_SAMPLE_RATE
SAMPLE_COUNT = round(DURATION_SECONDS * SAMPLE_RATE)
REFERENCE_RATE = 44_100
COMB_DELAYS_44K = (1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617)
ALLPASS_DELAYS_44K = (556, 441, 341, 225)
COMB_DELAYS = tuple(round(value * SAMPLE_RATE / REFERENCE_RATE) for value in COMB_DELAYS_44K)
ALLPASS_DELAYS = tuple(round(value * SAMPLE_RATE / REFERENCE_RATE) for value in ALLPASS_DELAYS_44K)
RT60_SECONDS = 4.0
ALLPASS_GAIN = 0.70
COMPLEX_MODE_BUDGETS = (64, 256, 1024, 3072)
LISTENING_VARIANTS = (
    ("energy", 1024),
    ("stratified", 1024),
    ("stratified", 3072),
)


@dataclass
class ModeGroup:
    kind: str
    delay: int
    radius: float
    poles: np.ndarray
    residues: np.ndarray


@dataclass
class ModePair:
    group_index: int
    indices: tuple[int, ...]
    mode_count: int
    frequency: float
    energy_score: float


def discrete_convolve(signal: np.ndarray, impulse: np.ndarray) -> np.ndarray:
    wanted = len(signal) + len(impulse) - 1
    fft_size = 1 << (wanted - 1).bit_length()
    result = np.fft.irfft(
        np.fft.rfft(signal, fft_size) * np.fft.rfft(impulse, fft_size), fft_size
    )
    return result[:len(signal)]


def comb_gains() -> tuple[float, ...]:
    per_sample = math.pow(10.0, -3.0 / (RT60_SECONDS * SAMPLE_RATE))
    return tuple(math.pow(per_sample, delay) for delay in COMB_DELAYS)


def output_weights() -> np.ndarray:
    signs = np.asarray((1, -1, 1, 1, -1, 1, -1, -1), dtype=np.float64)
    return signs / math.sqrt(len(signs))


def comb_transfer(w: complex) -> complex:
    total = 0.0j
    for delay, gain, weight in zip(COMB_DELAYS, comb_gains(), output_weights()):
        wd = w ** delay
        total += weight * wd / (1.0 - gain * wd)
    return total


def allpass_transfer(w: complex, delay: int) -> complex:
    wd = w ** delay
    return (ALLPASS_GAIN + wd) / (1.0 + ALLPASS_GAIN * wd)


def allpass_product(w: complex, skip: int | None = None) -> complex:
    result = 1.0 + 0.0j
    for index, delay in enumerate(ALLPASS_DELAYS):
        if index != skip:
            result *= allpass_transfer(w, delay)
    return result


def derive_modal_decomposition() -> tuple[list[ModeGroup], float]:
    groups: list[ModeGroup] = []
    for delay, gain, weight in zip(COMB_DELAYS, comb_gains(), output_weights()):
        radius = math.pow(gain, 1.0 / delay)
        k = np.arange(delay, dtype=np.float64)
        poles = radius * np.exp(1j * TWO_PI * k / delay)
        residues = np.empty(delay, dtype=np.complex128)
        for index, pole in enumerate(poles):
            w0 = 1.0 / pole
            residues[index] = weight / (gain * delay) * allpass_product(w0)
        groups.append(ModeGroup("comb", delay, radius, poles, residues))

    for ap_index, delay in enumerate(ALLPASS_DELAYS):
        radius = math.pow(ALLPASS_GAIN, 1.0 / delay)
        k = np.arange(delay, dtype=np.float64)
        poles = radius * np.exp(1j * math.pi * (2.0 * k + 1.0) / delay)
        residues = np.empty(delay, dtype=np.complex128)
        base_residue = (ALLPASS_GAIN * ALLPASS_GAIN - 1.0) / (
            ALLPASS_GAIN * delay
        )
        for index, pole in enumerate(poles):
            w0 = 1.0 / pole
            residues[index] = (
                base_residue
                * allpass_product(w0, skip=ap_index)
                * comb_transfer(w0)
            )
        groups.append(ModeGroup("allpass", delay, radius, poles, residues))

    direct_comb = sum(
        -weight / gain
        for gain, weight in zip(comb_gains(), output_weights())
    )
    direct_allpass = math.pow(1.0 / ALLPASS_GAIN, len(ALLPASS_DELAYS))
    return groups, direct_comb * direct_allpass


def stateful_comb_bank(signal: np.ndarray) -> np.ndarray:
    result = np.zeros_like(signal)
    for delay, gain, weight in zip(COMB_DELAYS, comb_gains(), output_weights()):
        branch = np.zeros_like(signal)
        for n in range(delay, len(signal)):
            branch[n] = signal[n - delay] + gain * branch[n - delay]
        result += weight * branch
    return result


def stateful_allpass(signal: np.ndarray, delay: int) -> np.ndarray:
    result = np.zeros_like(signal)
    gain = ALLPASS_GAIN
    for n in range(len(signal)):
        delayed_input = signal[n - delay] if n >= delay else 0.0
        delayed_output = result[n - delay] if n >= delay else 0.0
        result[n] = gain * signal[n] + delayed_input - gain * delayed_output
    return result


def stateful_teacher_impulse(sample_count: int) -> tuple[np.ndarray, np.ndarray]:
    impulse = np.zeros(sample_count, dtype=np.float64)
    impulse[0] = 1.0
    comb_only = stateful_comb_bank(impulse)
    result = comb_only
    for delay in ALLPASS_DELAYS:
        result = stateful_allpass(result, delay)
    return comb_only, result


def mode_pairs(groups: list[ModeGroup]) -> list[ModePair]:
    pairs: list[ModePair] = []
    for group_index, group in enumerate(groups):
        visited: set[int] = set()
        for index, pole in enumerate(group.poles):
            if index in visited:
                continue
            if group.kind == "comb":
                partner = (-index) % group.delay
            else:
                partner = group.delay - 1 - index
            indices = (index,) if partner == index else (index, partner)
            visited.update(indices)
            energy = sum(
                abs(group.residues[item]) ** 2 / (1.0 - abs(group.poles[item]) ** 2)
                for item in indices
            )
            frequency = abs(float(np.angle(pole)))
            if frequency > math.pi:
                frequency = TWO_PI - frequency
            pairs.append(ModePair(
                group_index,
                indices,
                len(indices),
                frequency,
                float(energy),
            ))
    return pairs


def select_energy(pairs: list[ModePair], budget: int) -> list[ModePair]:
    selected: list[ModePair] = []
    used = 0
    for pair in sorted(pairs, key=lambda item: item.energy_score, reverse=True):
        if used + pair.mode_count > budget:
            continue
        selected.append(pair)
        used += pair.mode_count
        if used >= budget:
            break
    return selected


def select_stratified(pairs: list[ModePair], budget: int) -> list[ModePair]:
    # One best-excited conjugate pair per linear-frequency stratum, then fill
    # any unused real-mode slots by energy.  This preserves the distribution
    # that the colorless-reverb literature identifies as perceptually critical.
    target_pairs = max(1, budget // 2)
    buckets: list[list[ModePair]] = [[] for _ in range(target_pairs)]
    for pair in pairs:
        bucket = min(target_pairs - 1, int(pair.frequency / math.pi * target_pairs))
        buckets[bucket].append(pair)
    chosen = [
        max(bucket, key=lambda item: item.energy_score)
        for bucket in buckets if bucket
    ]
    chosen_ids = {(pair.group_index, pair.indices) for pair in chosen}
    used = sum(pair.mode_count for pair in chosen)
    if used < budget:
        for pair in sorted(pairs, key=lambda item: item.energy_score, reverse=True):
            key = (pair.group_index, pair.indices)
            if key in chosen_ids or used + pair.mode_count > budget:
                continue
            chosen.append(pair)
            chosen_ids.add(key)
            used += pair.mode_count
            if used >= budget:
                break
    return chosen


def selection_masks(
    groups: list[ModeGroup], selected: list[ModePair] | None
) -> list[np.ndarray]:
    if selected is None:
        return [np.ones(len(group.poles), dtype=bool) for group in groups]
    masks = [np.zeros(len(group.poles), dtype=bool) for group in groups]
    for pair in selected:
        masks[pair.group_index][list(pair.indices)] = True
    return masks


def synthesize_modal_impulse(
    groups: list[ModeGroup],
    direct: float,
    sample_count: int,
    selected: list[ModePair] | None = None,
) -> np.ndarray:
    masks = selection_masks(groups, selected)
    result = np.zeros(sample_count, dtype=np.float64)
    residue_sum = 0.0 + 0.0j
    n = np.arange(sample_count, dtype=np.float64)
    for group, mask in zip(groups, masks):
        residues = np.where(mask, group.residues, 0.0j)
        residue_sum += np.sum(residues)
        if group.kind == "comb":
            period = group.delay
            spectrum = residues * period
        else:
            period = 2 * group.delay
            spectrum = np.zeros(period, dtype=np.complex128)
            spectrum[1::2] = residues * period
        carrier = np.fft.ifft(spectrum).real
        tiled = np.resize(carrier, sample_count)
        result += tiled * np.exp(math.log(group.radius) * n)

    # The full partial fraction direct term is exact.  A truncated pole subset
    # gets a one-sample feedthrough correction so its wet impulse still starts
    # continuously instead of exposing the cancellation formerly supplied by
    # the omitted residues.
    adjusted_direct = direct if selected is None else -float(residue_sum.real)
    result[0] += adjusted_direct
    return result


def group_carrier(group: ModeGroup) -> np.ndarray:
    """Collect one complete uniform pole lattice into a periodic real table."""
    if group.kind == "comb":
        period = group.delay
        spectrum = group.residues * period
    else:
        period = 2 * group.delay
        spectrum = np.zeros(period, dtype=np.complex128)
        spectrum[1::2] = group.residues * period
    return np.fft.ifft(spectrum).real


def scaled_geometric(
    pole: complex,
    radius: float,
    period: int,
    n: np.ndarray,
    length: np.ndarray,
    ratio_period: complex,
) -> np.ndarray:
    """Return p^n (1-Q^length)/(1-Q) without overflowing for |Q|>1."""
    if abs(1.0 - ratio_period) < 1e-10:
        return length * np.power(pole, n)
    span = period * length
    return (
        np.power(pole, n)
        - np.power(radius, span) * np.power(pole, n - span)
    ) / (1.0 - ratio_period)


def structured_teacher_modal_response(
    rows: list[list[float]],
    groups: list[ModeGroup],
    direct: float,
    sample_count: int,
) -> np.ndarray:
    """Exact zero-state teacher response, grouped by delay pole lattices.

    For one group h[k] = r^k c[k mod P] and one source mode x[n] = C p^n,
    convolution splits at n = KP+R into two geometric series.  The only
    group-specific data read at a coordinate is a prefix-table entry at R.
    """
    n = np.arange(sample_count, dtype=np.int64)
    total = np.zeros(sample_count, dtype=np.complex128)
    carriers = [(group, group_carrier(group)) for group in groups]
    for frequency, sigma, amplitude, phase in rows:
        pole = np.exp(complex(-sigma, TWO_PI * frequency) / SAMPLE_RATE)
        coefficient = amplitude * np.exp(1j * phase)
        response = direct * np.power(pole, n)
        for group, carrier in carriers:
            period = len(carrier)
            quotient = n // period
            remainder = n - quotient * period
            ratio = group.radius / pole
            powers = np.power(ratio, np.arange(period, dtype=np.int64))
            prefix = np.cumsum(carrier * powers)
            prefix_at_remainder = prefix[remainder]
            prefix_total = prefix[-1]
            ratio_period = ratio ** period
            through_remainder = scaled_geometric(
                pole,
                group.radius,
                period,
                n,
                quotient + 1,
                ratio_period,
            )
            after_remainder = scaled_geometric(
                pole,
                group.radius,
                period,
                n,
                quotient,
                ratio_period,
            )
            response += (
                prefix_at_remainder * through_remainder
                + (prefix_total - prefix_at_remainder) * after_remainder
            )
        total += coefficient * response
    return total.real


def structured_comb_modal_response(
    rows: list[list[float]], sample_count: int
) -> np.ndarray:
    n_all = np.arange(sample_count, dtype=np.int64)
    result = np.zeros(sample_count, dtype=np.float64)
    for delay, gain, weight in zip(COMB_DELAYS, comb_gains(), output_weights()):
        active_n = n_all[delay:]
        rounds = (active_n - delay) // delay
        remainder = (active_n - delay) - rounds * delay
        branch = np.zeros(len(active_n), dtype=np.complex128)
        for frequency, sigma, amplitude, phase in rows:
            pole = np.exp(complex(-sigma, TWO_PI * frequency) / SAMPLE_RATE)
            coefficient = amplitude * np.exp(1j * phase)
            denominator = 1.0 - gain * pole ** (-delay)
            first = pole ** (active_n - delay)
            last = np.power(gain, rounds + 1) * pole ** (remainder - delay)
            if abs(denominator) < 1e-10:
                # Smooth limit of the geometric quotient at exact resonance.
                geometric = (rounds + 1) * pole ** (active_n - delay)
            else:
                geometric = (first - last) / denominator
            branch += coefficient * geometric
        result[delay:] += weight * branch.real
    return result


def best_gain_error(candidate: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    denominator = float(np.dot(candidate, candidate))
    gain = float(np.dot(candidate, target) / denominator) if denominator > 0.0 else 0.0
    error = float(np.linalg.norm(gain * candidate - target) / np.linalg.norm(target))
    return gain, error


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}
    start = time.perf_counter()
    groups, direct = derive_modal_decomposition()
    timings["derive_modal_decomposition_seconds"] = time.perf_counter() - start
    total_modes = sum(len(group.poles) for group in groups)
    pairs = mode_pairs(groups)

    validation_count = min(SAMPLE_COUNT, 65_536)
    start = time.perf_counter()
    comb_stateful, teacher_stateful = stateful_teacher_impulse(validation_count)
    timings["stateful_teacher_65536_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    teacher_modal_validation = synthesize_modal_impulse(
        groups, direct, validation_count
    )
    timings["full_modal_teacher_65536_seconds"] = time.perf_counter() - start
    full_max_error = float(np.max(np.abs(
        teacher_modal_validation - teacher_stateful
    )))
    full_relative_error = float(
        np.linalg.norm(teacher_modal_validation - teacher_stateful)
        / np.linalg.norm(teacher_stateful)
    )

    # The modal teacher is now independently validated, so synthesize its full
    # audition horizon without paying a Python state loop for twelve seconds.
    teacher_impulse = synthesize_modal_impulse(groups, direct, SAMPLE_COUNT)
    dry_rows = snare_modes(0)
    dry_32k = modal_signal(dry_rows, SAMPLE_RATE, SAMPLE_COUNT)
    teacher_wet_32k = discrete_convolve(dry_32k, teacher_impulse)
    comb_wet_stateful = discrete_convolve(
        dry_32k,
        np.pad(comb_stateful, (0, SAMPLE_COUNT - validation_count)),
    )
    start = time.perf_counter()
    comb_wet_structured = structured_comb_modal_response(dry_rows, SAMPLE_COUNT)
    timings["structured_comb_modal_source_12s_seconds"] = time.perf_counter() - start
    # Compare only the interval supported by the validation impulse.  At 2.048 s
    # it is already well beyond the metal source and many comb round trips.
    supported = validation_count
    structured_max_error = float(np.max(np.abs(
        comb_wet_structured[:supported] - comb_wet_stateful[:supported]
    )))
    structured_relative_error = float(
        np.linalg.norm(comb_wet_structured[:supported] - comb_wet_stateful[:supported])
        / np.linalg.norm(comb_wet_stateful[:supported])
    )
    start = time.perf_counter()
    teacher_wet_structured = structured_teacher_modal_response(
        dry_rows, groups, direct, SAMPLE_COUNT
    )
    timings["structured_full_teacher_modal_source_12s_seconds"] = (
        time.perf_counter() - start
    )
    structured_teacher_max_error = float(np.max(np.abs(
        teacher_wet_structured - teacher_wet_32k
    )))
    structured_teacher_relative_error = float(
        np.linalg.norm(teacher_wet_structured - teacher_wet_32k)
        / np.linalg.norm(teacher_wet_32k)
    )

    with tempfile.TemporaryDirectory(prefix="tropical-clouds-recovery-") as temp:
        binary = Path(temp) / "clouds_reference"
        build_reference_binary(args.clouds_root, binary)
        clouds_wet_32k = render_clouds(binary, dry_32k)
        impulse = np.zeros(SAMPLE_COUNT, dtype=np.float64)
        impulse[0] = 1.0
        clouds_impulse_32k = render_clouds(binary, impulse)
    clouds_wet = np.mean(
        resample_linear(clouds_wet_32k, SAMPLE_RATE, OUTPUT_SAMPLE_RATE), axis=1
    )
    clouds_impulse = np.mean(
        resample_linear(clouds_impulse_32k, SAMPLE_RATE, OUTPUT_SAMPLE_RATE), axis=1
    )
    dry = resample_linear(dry_32k, SAMPLE_RATE, OUTPUT_SAMPLE_RATE)
    teacher_wet = resample_linear(teacher_wet_32k, SAMPLE_RATE, OUTPUT_SAMPLE_RATE)
    teacher_impulse_44k = resample_linear(
        teacher_impulse, SAMPLE_RATE, OUTPUT_SAMPLE_RATE
    )
    clouds_features = feature_summary(clouds_impulse)
    teacher_features = feature_summary(teacher_impulse_44k)

    all_pair_energy = sum(pair.energy_score for pair in pairs)
    selections: dict[tuple[str, int], list[ModePair]] = {}
    experiments: dict[str, object] = {}
    wet_tracks: dict[str, np.ndarray] = {
        "clouds_gold_wet_mono": clouds_wet,
        "freeverb_teacher_wet": teacher_wet,
    }
    impulse_tracks: dict[str, np.ndarray] = {
        "clouds_gold_impulse_mono": clouds_impulse,
        "freeverb_teacher_impulse": teacher_impulse_44k,
    }
    for budget in COMPLEX_MODE_BUDGETS:
        for strategy, selector in (
            ("energy", select_energy),
            ("stratified", select_stratified),
        ):
            selected = selector(pairs, budget)
            selections[(strategy, budget)] = selected
            actual_modes = sum(pair.mode_count for pair in selected)
            start = time.perf_counter()
            candidate_impulse_32k = synthesize_modal_impulse(
                groups, direct, SAMPLE_COUNT, selected
            )
            synth_seconds = time.perf_counter() - start
            gain, nrmse = best_gain_error(candidate_impulse_32k, teacher_impulse)
            candidate_44k = resample_linear(
                candidate_impulse_32k, SAMPLE_RATE, OUTPUT_SAMPLE_RATE
            )
            candidate_features = feature_summary(candidate_44k)
            key = f"{strategy}_{budget}"
            experiments[key] = {
                "requested_complex_modes": budget,
                "actual_complex_modes": actual_modes,
                "conjugate_groups": len(selected),
                "modal_energy_score_fraction": rounded(
                    sum(pair.energy_score for pair in selected) / all_pair_energy, 8
                ),
                "best_scalar_gain": rounded(gain, 9),
                "teacher_impulse_nrmse_after_gain": rounded(nrmse, 6),
                "distance_to_teacher": feature_distance(
                    candidate_features, teacher_features
                ),
                "distance_to_clouds": feature_distance(
                    candidate_features, clouds_features
                ),
                "features": {
                    "normalized_echo_density": candidate_features["normalized_echo_density"],
                    "spectral_flatness": candidate_features["spectral_flatness"],
                },
                "synthesis_seconds": rounded(synth_seconds, 6),
            }
            if (strategy, budget) in LISTENING_VARIANTS:
                candidate_wet_32k = discrete_convolve(dry_32k, candidate_impulse_32k)
                wet_tracks[f"freeverb_{strategy}_{budget}_wet"] = resample_linear(
                    candidate_wet_32k, SAMPLE_RATE, OUTPUT_SAMPLE_RATE
                )
                impulse_tracks[f"freeverb_{strategy}_{budget}_impulse"] = (
                    candidate_44k
                )

    # Match all wet tracks to the gold active RMS, use one mix ratio, then one
    # common headroom gain.  No candidate can win by being louder.
    target_rms = active_rms(clouds_wet)
    normalized_tracks: dict[str, np.ndarray] = {}
    for name, wet in wet_tracks.items():
        normalized_wet = wet * (target_rms / max(active_rms(wet), 1e-30))
        normalized_tracks[name] = normalized_wet
        normalized_tracks[name.replace("_wet", "_mix")] = dry + 1.15 * normalized_wet
    common_gain = math.pow(10.0, -1.0 / 20.0) / max(
        float(np.max(np.abs(track))) for track in normalized_tracks.values()
    )
    files: dict[str, object] = {}
    for name, track in normalized_tracks.items():
        path = OUT_DIR / f"{name}.wav"
        write_wav(path, track * common_gain)
        files[path.name] = {"sha256": sha256(path)}

    # Impulse responses form a separate loudness-matched set.  Keeping their
    # gain normalization independent from the source auditions makes the decay
    # and coloration directly comparable without embedding the dry Metal hit.
    impulse_target_rms = active_rms(impulse_tracks["clouds_gold_impulse_mono"])
    normalized_impulses = {
        name: impulse * (
            impulse_target_rms / max(active_rms(impulse), 1e-30)
        )
        for name, impulse in impulse_tracks.items()
    }
    impulse_common_gain = math.pow(10.0, -1.0 / 20.0) / max(
        float(np.max(np.abs(impulse)))
        for impulse in normalized_impulses.values()
    )
    for name, impulse in normalized_impulses.items():
        path = OUT_DIR / f"{name}.wav"
        write_wav(path, impulse * impulse_common_gain)
        files[path.name] = {"sha256": sha256(path)}

    summary = {
        "question": "Can a stateful delay reverb be recovered economically in Tropical's closed-form modal realm?",
        "teacher": {
            "topology": "8 parallel homogeneous feedback combs -> 4 Schroeder allpasses",
            "sample_rate": SAMPLE_RATE,
            "comb_delays": COMB_DELAYS,
            "allpass_delays": ALLPASS_DELAYS,
            "rt60_seconds": RT60_SECONDS,
            "allpass_gain": ALLPASS_GAIN,
            "total_exact_complex_modes": total_modes,
            "stateful_delay_samples": sum(COMB_DELAYS) + sum(ALLPASS_DELAYS),
        },
        "exactness": {
            "full_modal_vs_stateful_max_abs_65536": full_max_error,
            "full_modal_vs_stateful_relative_l2_65536": full_relative_error,
            "structured_comb_vs_stateful_max_abs_supported": structured_max_error,
            "structured_comb_vs_stateful_relative_l2_supported": structured_relative_error,
            "structured_comb_supported_samples": supported,
            "structured_full_teacher_vs_modal_convolution_max_abs": structured_teacher_max_error,
            "structured_full_teacher_vs_modal_convolution_relative_l2": structured_teacher_relative_error,
        },
        "cost_model": {
            "stateful_teacher_per_sample": "8 comb delay reads/writes + 4 allpass stages; sequential only",
            "full_diagonal_modal_per_sample": total_modes,
            "structured_comb_per_random_coordinate": f"{len(COMB_DELAYS)} combs x {len(dry_rows)} authored real modal rows",
            "structured_full_teacher_per_random_coordinate": f"{len(groups)} pole-lattice tables x {len(dry_rows)} authored real modal rows",
            "structured_full_teacher_table_samples": sum(
                group.delay if group.kind == "comb" else 2 * group.delay
                for group in groups
            ),
            "structured_full_teacher_prefix_complex_samples": sum(
                group.delay if group.kind == "comb" else 2 * group.delay
                for group in groups
            ) * len(dry_rows),
            "structured_full_teacher_prefix_bytes_complex64": sum(
                group.delay if group.kind == "comb" else 2 * group.delay
                for group in groups
            ) * len(dry_rows) * 8,
            "runtime_term_reduction_vs_diagonal_modes": rounded(
                total_modes / (len(groups) * len(dry_rows)), 3
            ),
        },
        "clouds_commit": CLOUDS_COMMIT,
        "clouds_features": {
            "normalized_echo_density": clouds_features["normalized_echo_density"],
            "spectral_flatness": clouds_features["spectral_flatness"],
        },
        "teacher_features": {
            "normalized_echo_density": teacher_features["normalized_echo_density"],
            "spectral_flatness": teacher_features["spectral_flatness"],
        },
        "experiments": experiments,
        "timings": {name: rounded(value, 6) for name, value in timings.items()},
        "common_headroom_gain": rounded(common_gain, 9),
        "files": files,
    }
    (OUT_DIR / "recovery_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
