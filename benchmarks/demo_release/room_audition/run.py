#!/usr/bin/env python3
"""Render the modal-pocket fixed-room listening gate with NumPy only."""

from __future__ import annotations

import hashlib
import json
import math
import wave
from pathlib import Path

import numpy as np


SAMPLE_RATE = 44_100
DURATION_SECONDS = 10.0
N_SAMPLES = round(SAMPLE_RATE * DURATION_SECONDS)
TWO_PI = 2.0 * math.pi
GOLDEN_ANGLE = 2.399963229728653
GOLDEN_RATIO = 1.618033988749895
RT60_LOG = math.log(1000.0)
SNARE_FREQUENCIES = (211, 433, 887, 1511, 2837, 5081)
OUT_DIR = Path(__file__).resolve().parent / "out"


CANDIDATES = (
    {
        "id": "foundry_24",
        "label": "Foundry 24",
        "pole_count": 24,
        "predelay_ms": 55,
        "early_spacing_ms": 37,
        "early_gain": 0.10,
        "early_feedback": 0.69,
        "late_residue_scale": 24.0,
    },
    {
        "id": "industrial_cathedral_32",
        "label": "Industrial Cathedral 32",
        "pole_count": 32,
        "predelay_ms": 78,
        "early_spacing_ms": 43,
        "early_gain": 0.10,
        "early_feedback": 0.69,
        "late_residue_scale": 24.0,
    },
    {
        "id": "tanker_48",
        "label": "Tanker 48",
        "pole_count": 48,
        "predelay_ms": 105,
        "early_spacing_ms": 53,
        "early_gain": 0.10,
        "early_feedback": 0.69,
        "late_residue_scale": 24.0,
    },
)


def rounded(value: float, places: int = 9) -> float:
    return round(value, places)


def snare_modes(hit_index: int = 0) -> list[list[float]]:
    """Byte-for-value transcription of playground/scene.js snareModes."""
    rows: list[list[float]] = []
    for index, frequency in enumerate(SNARE_FREQUENCIES):
        amplitude = 0.065 * math.pow(1 + index, -0.32)
        body_decay = 7.5 + 1.15 * index
        attack_decay = body_decay + 88 + 3.7 * index
        phase = (GOLDEN_ANGLE * (index + 11 * hit_index + 1)) % TWO_PI
        rows.append([
            frequency,
            rounded(body_decay, 6),
            rounded(amplitude, 9),
            rounded(phase, 9),
        ])
        rows.append([
            frequency,
            rounded(attack_decay, 6),
            rounded(-amplitude, 9),
            rounded(phase, 9),
        ])
    return rows


def modal_signal(rows: list[list[float]], sample_count: int = N_SAMPLES) -> np.ndarray:
    t = np.arange(sample_count, dtype=np.float64) / SAMPLE_RATE
    result = np.zeros(sample_count, dtype=np.float64)
    for frequency, sigma, amplitude, phase in rows:
        result += (
            amplitude
            * np.exp(-sigma * t)
            * np.cos(TWO_PI * frequency * t + phase)
        )
    return result


def incumbent_room_rows(count: int = 16, rt60: float = 8.0) -> list[list[float]]:
    rows: list[list[float]] = []
    sigma = 6.91 / rt60
    for index in range(count):
        u = index / max(1, count - 1)
        frequency = 60.0 * math.pow(6000.0 / 60.0, u)
        phase = TWO_PI * GOLDEN_RATIO * index
        rows.append([
            rounded(frequency),
            rounded(sigma, 9),
            1.0,
            rounded(phase % TWO_PI),
        ])
    return rows


def rt60_for_frequency(frequency: float) -> float:
    """Frozen three-band decay with short cosine shoulders between bands."""
    if frequency <= 260.0:
        return 8.5
    if frequency < 620.0:
        x = (frequency - 260.0) / (620.0 - 260.0)
        return 8.5 + (5.5 - 8.5) * (0.5 - 0.5 * math.cos(math.pi * x))
    if frequency <= 1800.0:
        return 5.5
    if frequency < 3600.0:
        x = (frequency - 1800.0) / (3600.0 - 1800.0)
        return 5.5 + (3.5 - 5.5) * (0.5 - 0.5 * math.cos(math.pi * x))
    return 3.5


def candidate_room_rows(pole_count: int, residue_scale: float) -> list[list[float]]:
    """Deterministic jittered cells, each an opposite-residue slow/fast pair."""
    assert pole_count % 2 == 0
    cell_count = pole_count // 2
    rows: list[list[float]] = []
    log_lo = math.log(90.0)
    log_span = math.log(8000.0 / 90.0)
    amplitude_base = residue_scale / math.sqrt(cell_count)
    for index in range(cell_count):
        # Stratified log coverage, but golden-ratio jitter prevents a log lattice.
        jitter = ((index + 1) * GOLDEN_RATIO) % 1.0 - 0.5
        u = (index + 0.5 + 0.62 * jitter) / cell_count
        u = min(0.995, max(0.005, u))
        frequency = math.exp(log_lo + log_span * u)
        rt60 = rt60_for_frequency(frequency)
        slow_sigma = RT60_LOG / rt60
        # This opposite pole reaches its maximum difference after 35–44 ms.
        fast_sigma = slow_sigma + 115.0
        spectral_tilt = math.pow(500.0 / max(180.0, frequency), 0.08)
        phase = (GOLDEN_ANGLE * (index + 1) + 0.37 * jitter) % TWO_PI
        amplitude = amplitude_base * spectral_tilt
        row = [
            rounded(frequency),
            rounded(slow_sigma, 9),
            rounded(amplitude, 9),
            rounded(phase),
        ]
        rows.append(row)
        rows.append([
            row[0],
            rounded(fast_sigma, 9),
            rounded(-amplitude, 9),
            row[3],
        ])
    return rows


def fft_convolve(signal: np.ndarray, impulse: np.ndarray) -> np.ndarray:
    wanted = len(signal) + len(impulse) - 1
    fft_size = 1 << (wanted - 1).bit_length()
    spectrum = np.fft.rfft(signal, fft_size) * np.fft.rfft(impulse, fft_size)
    # The modal calculus is continuous-time convolution; the sampled sum needs dt.
    return np.fft.irfft(spectrum, fft_size)[: len(signal)] / SAMPLE_RATE


def shift(signal: np.ndarray, seconds: float) -> np.ndarray:
    offset = round(seconds * SAMPLE_RATE)
    result = np.zeros_like(signal)
    if offset < len(signal):
        result[offset:] = signal[: len(signal) - offset]
    return result


def early_field(
    dry: np.ndarray,
    spacing_seconds: float,
    feedback: float,
    gain: float,
) -> np.ndarray:
    result = np.zeros_like(dry)
    for tap in range(7):
        result += gain * math.pow(feedback, tap) * shift(dry, tap * spacing_seconds)
    return result


def active_rms(signal: np.ndarray, seconds: float = 5.0) -> float:
    window = signal[: round(seconds * SAMPLE_RATE)]
    return float(np.sqrt(np.mean(np.square(window))))


def db_ratio(value: float, reference: float, power: bool = False) -> float:
    if value <= 0.0 or reference <= 0.0:
        return -300.0
    return (10.0 if power else 20.0) * math.log10(value / reference)


def window_rms(signal: np.ndarray, begin: float, end: float) -> float:
    a = round(begin * SAMPLE_RATE)
    b = round(end * SAMPLE_RATE)
    return float(np.sqrt(np.mean(np.square(signal[a:b]))))


def candidate_metrics(
    profile: dict[str, object],
    room_rows: list[list[float]],
    wet: np.ndarray,
) -> dict[str, object]:
    onset = float(profile["predelay_ms"]) / 1000.0
    first_energy = float(np.sum(np.square(
        wet[round(onset * SAMPLE_RATE):round((onset + 0.020) * SAMPLE_RATE)]
    )))
    early_energy = float(np.sum(np.square(
        wet[round((onset + 0.020) * SAMPLE_RATE):round((onset + 0.200) * SAMPLE_RATE)]
    )))
    onset_rms = window_rms(wet, onset, onset + 0.020)
    slow_rows = room_rows[::2]
    bands = {
        "low": [row for row in slow_rows if row[0] < 400.0],
        "mid": [row for row in slow_rows if 400.0 <= row[0] < 2500.0],
        "high": [row for row in slow_rows if row[0] >= 2500.0],
    }
    band_rt60 = {
        name: rounded(float(np.median([RT60_LOG / row[1] for row in rows])), 3)
        if rows else None
        for name, rows in bands.items()
    }
    return {
        "pole_count": len(room_rows),
        "frequency_cells": len(room_rows) // 2,
        "frequency_span_hz": [
            rounded(min(row[0] for row in slow_rows), 3),
            rounded(max(row[0] for row in slow_rows), 3),
        ],
        "predelay_ms": profile["predelay_ms"],
        "early_spacing_ms": profile["early_spacing_ms"],
        "authored_early_maxima": 7,
        "band_rt60_seconds": band_rt60,
        "energy_20_200ms_vs_first_20ms_db": rounded(
            db_ratio(early_energy, first_energy, power=True), 2
        ),
        "late_rms_1_2s_vs_onset_db": rounded(
            db_ratio(window_rms(wet, onset + 1.0, onset + 2.0), onset_rms), 2
        ),
        "late_rms_4_5s_vs_onset_db": rounded(
            db_ratio(window_rms(wet, onset + 4.0, onset + 5.0), onset_rms), 2
        ),
    }


def bloom_range_ms(rows: list[list[float]]) -> list[float]:
    peaks: list[float] = []
    for slow, fast in zip(rows[::2], rows[1::2]):
        slow_sigma = slow[1]
        fast_sigma = fast[1]
        peaks.append(1000.0 * math.log(fast_sigma / slow_sigma) / (fast_sigma - slow_sigma))
    return [rounded(min(peaks), 2), rounded(max(peaks), 2)]


def pcm24_bytes(signal: np.ndarray) -> bytes:
    assert np.all(np.isfinite(signal))
    assert float(np.max(np.abs(signal))) < 1.0
    quantized = np.rint(signal * ((1 << 23) - 1)).astype(np.int32)
    unsigned = quantized.astype(np.uint32)
    packed = np.empty((len(signal), 3), dtype=np.uint8)
    packed[:, 0] = unsigned & 0xFF
    packed[:, 1] = (unsigned >> 8) & 0xFF
    packed[:, 2] = (unsigned >> 16) & 0xFF
    return packed.tobytes()


def write_wav(path: Path, signal: np.ndarray) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(3)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm24_bytes(signal))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dry_rows = snare_modes(0)
    dry = modal_signal(dry_rows)
    dry_rms = active_rms(dry)

    tracks: dict[str, np.ndarray] = {"dry_metal": dry.copy()}
    raw_tracks: dict[str, np.ndarray] = {}
    profiles: dict[str, object] = {
        "sample_rate": SAMPLE_RATE,
        "source": {"id": "metal_hit_0", "modes": dry_rows},
        "candidates": {},
    }

    incumbent_rows = incumbent_room_rows()
    incumbent_wet = fft_convolve(dry, modal_signal(incumbent_rows))
    raw_tracks["incumbent_wet"] = incumbent_wet

    metrics: dict[str, object] = {
        "sample_rate": SAMPLE_RATE,
        "duration_seconds": DURATION_SECONDS,
        "matching": {
            "window_seconds": 5.0,
            "wet_to_dry_rms_in_mix": 0.75,
            "method": "one linear gain per track, then one common headroom gain; no limiter",
        },
        "incumbent": {
            "pole_count": len(incumbent_rows),
            "rt60_seconds": 8.0,
            "predelay_ms": 0,
            "early_taps": 0,
        },
        "candidates": {},
    }

    for profile in CANDIDATES:
        room_rows = candidate_room_rows(
            int(profile["pole_count"]), float(profile["late_residue_scale"])
        )
        late = fft_convolve(dry, modal_signal(room_rows))
        early = early_field(
            dry,
            float(profile["early_spacing_ms"]) / 1000.0,
            float(profile["early_feedback"]),
            float(profile["early_gain"]),
        )
        wet = shift(early + late, float(profile["predelay_ms"]) / 1000.0)
        raw_tracks[f"{profile['id']}_wet"] = wet
        candidate_summary = candidate_metrics(profile, room_rows, wet)
        candidate_summary["bloom_peak_ms_range"] = bloom_range_ms(room_rows)
        metrics["candidates"][str(profile["id"])] = candidate_summary
        profiles["candidates"][str(profile["id"])] = {
            **profile,
            "room_modes": room_rows,
        }

    # Match every wet-only file to the dry active RMS. Each mix gets the same
    # -2.5 dB wet/dry RMS ratio and is then matched to the same dry reference.
    for name, wet in raw_tracks.items():
        wet_gain = dry_rms / active_rms(wet)
        tracks[name] = wet * wet_gain
        mix = dry + wet * wet_gain * 0.75
        tracks[name.replace("_wet", "_mix")] = mix * (dry_rms / active_rms(mix))

    # One common gain preserves loudness equality while keeping every peak below
    # -1 dBFS. This is not peak normalization and does not alter any comparison.
    preliminary_peak = max(float(np.max(np.abs(signal))) for signal in tracks.values())
    common_gain = math.pow(10.0, -1.0 / 20.0) / preliminary_peak
    metrics["matching"]["common_headroom_gain"] = rounded(common_gain, 9)
    metrics["files"] = {}
    for name, signal in tracks.items():
        rendered = signal * common_gain
        path = OUT_DIR / f"{name}.wav"
        write_wav(path, rendered)
        metrics["files"][path.name] = {
            "rms_dbfs_0_5s": rounded(db_ratio(active_rms(rendered), 1.0), 3),
            "peak_dbfs": rounded(db_ratio(float(np.max(np.abs(rendered))), 1.0), 3),
            "sha256": file_sha256(path),
        }

    profiles_path = OUT_DIR / "profiles.json"
    profiles_path.write_text(json.dumps(profiles, indent=2) + "\n", encoding="utf-8")
    metrics["profiles_sha256"] = file_sha256(profiles_path)
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    markdown = [
        "# Fixed-room audition summary",
        "",
        "All WAVs use one linear per-track loudness gain and one shared headroom gain; no limiter.",
        "",
        "| Candidate | poles | predelay | early spacing | bloom peak | RT60 low/mid/high | 20–200 ms energy | 1–2 s RMS | 4–5 s RMS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in CANDIDATES:
        item = metrics["candidates"][str(profile["id"])]
        band = item["band_rt60_seconds"]
        markdown.append(
            f"| {profile['label']} | {item['pole_count']} | {item['predelay_ms']} ms | "
            f"{item['early_spacing_ms']} ms | {item['bloom_peak_ms_range'][0]}–"
            f"{item['bloom_peak_ms_range'][1]} ms | {band['low']}/{band['mid']}/{band['high']} s | "
            f"{item['energy_20_200ms_vs_first_20ms_db']} dB | "
            f"{item['late_rms_1_2s_vs_onset_db']} dB | "
            f"{item['late_rms_4_5s_vs_onset_db']} dB |"
        )
    markdown.extend([
        "",
        "The three objective columns are relative to the first 20 ms after wet onset.",
        "Each candidate authors seven separated early arrivals (direct plus six comb taps).",
        "",
    ])
    (OUT_DIR / "summary.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
