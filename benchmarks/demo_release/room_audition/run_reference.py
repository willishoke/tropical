#!/usr/bin/env python3
"""Render the official Mutable Instruments Clouds reverb as the room target."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from run import GOLDEN_ANGLE, SNARE_FREQUENCIES, TWO_PI, rounded, snare_modes


AUDITION_DIR = Path(__file__).resolve().parent
OUT_DIR = AUDITION_DIR / "reference_out"
RENDER_SAMPLE_RATE = 32_000
OUTPUT_SAMPLE_RATE = 44_100
DURATION_SECONDS = 12.0
CLOUDS_COMMIT = "08460a69a7e1f7a81c5a2abcc7189c9a6b7208d4"


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
    return parser.parse_args()


def modal_signal(
    rows: list[list[float]], sample_rate: int, sample_count: int
) -> np.ndarray:
    t = np.arange(sample_count, dtype=np.float64) / sample_rate
    result = np.zeros(sample_count, dtype=np.float64)
    for frequency, sigma, amplitude, phase in rows:
        result += (
            amplitude
            * np.exp(-sigma * t)
            * np.cos(TWO_PI * frequency * t + phase)
        )
    return result


def resample_linear(signal: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    target_count = round(len(signal) * target_rate / source_rate)
    source_x = np.arange(len(signal), dtype=np.float64)
    target_x = np.arange(target_count, dtype=np.float64) * source_rate / target_rate
    if signal.ndim == 1:
        return np.interp(target_x, source_x, signal)
    return np.column_stack([
        np.interp(target_x, source_x, signal[:, channel])
        for channel in range(signal.shape[1])
    ])


def build_reference_binary(clouds_root: Path, destination: Path) -> None:
    actual_commit = subprocess.check_output(
        ["git", "-C", str(clouds_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != CLOUDS_COMMIT:
        raise RuntimeError(
            f"Clouds checkout is {actual_commit}; expected frozen {CLOUDS_COMMIT}"
        )
    subprocess.run([
        "clang++", "-std=c++17", "-O2", "-DTEST",
        f"-I{clouds_root}",
        f"-I{clouds_root / 'stmlib'}",
        str(AUDITION_DIR / "clouds_reference.cpp"),
        "-o", str(destination),
    ], check=True)


def render_clouds(
    binary: Path,
    mono: np.ndarray,
    *,
    time: float = 0.98,
    diffusion: float = 0.70,
    lp: float = 0.82,
    input_gain: float = 0.20,
) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="tropical-clouds-") as temp:
        input_path = Path(temp) / "input.f32"
        output_path = Path(temp) / "output.f32"
        mono.astype(np.float32).tofile(input_path)
        subprocess.run([
            str(binary), str(input_path), str(output_path),
            str(time), str(diffusion), str(lp), str(input_gain), "1.0",
        ], check=True)
        raw = np.fromfile(output_path, dtype=np.float32)
    if raw.size != mono.size * 2:
        raise RuntimeError("Clouds reference returned an unexpected frame count")
    return raw.reshape(-1, 2).astype(np.float64)


def active_rms(signal: np.ndarray, seconds: float = 5.0) -> float:
    window = signal[: round(seconds * OUTPUT_SAMPLE_RATE)]
    return float(np.sqrt(np.mean(np.square(window))))


def pcm24_bytes(signal: np.ndarray) -> bytes:
    if not np.all(np.isfinite(signal)):
        raise RuntimeError("non-finite reference sample")
    if float(np.max(np.abs(signal))) >= 1.0:
        raise RuntimeError("reference exceeds PCM headroom")
    quantized = np.rint(signal * ((1 << 23) - 1)).astype(np.int32)
    unsigned = quantized.astype(np.uint32).reshape(-1)
    packed = np.empty((len(unsigned), 3), dtype=np.uint8)
    packed[:, 0] = unsigned & 0xFF
    packed[:, 1] = (unsigned >> 8) & 0xFF
    packed[:, 2] = (unsigned >> 16) & 0xFF
    return packed.tobytes()


def write_wav(path: Path, signal: np.ndarray) -> None:
    channels = 1 if signal.ndim == 1 else signal.shape[1]
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(3)
        wav.setframerate(OUTPUT_SAMPLE_RATE)
        wav.writeframes(pcm24_bytes(signal))


def normalized_echo_density(
    impulse: np.ndarray, center_seconds: float, window_ms: float = 20.0
) -> float:
    center = round(center_seconds * OUTPUT_SAMPLE_RATE)
    radius = round(window_ms * OUTPUT_SAMPLE_RATE / 2000.0)
    window = impulse[max(0, center - radius):center + radius]
    if not window.size:
        return 0.0
    sigma = float(np.sqrt(np.mean(np.square(window))))
    if sigma <= 1e-15:
        return 0.0
    # A Gaussian process has 31.73% of samples outside one standard deviation.
    return float(np.mean(np.abs(window) > sigma) / 0.31731050786291415)


def spectral_flatness(signal: np.ndarray, begin: float, end: float) -> float:
    a = round(begin * OUTPUT_SAMPLE_RATE)
    b = round(end * OUTPUT_SAMPLE_RATE)
    window = signal[a:b] * np.hanning(max(0, b - a))
    power = np.square(np.abs(np.fft.rfft(window))) + 1e-20
    return float(np.exp(np.mean(np.log(power))) / np.mean(power))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tropical-clouds-build-") as temp:
        binary = Path(temp) / "clouds_reference"
        build_reference_binary(args.clouds_root, binary)
        count_32k = round(DURATION_SECONDS * RENDER_SAMPLE_RATE)
        dry_32k = modal_signal(snare_modes(0), RENDER_SAMPLE_RATE, count_32k)
        impulse_32k = np.zeros(count_32k, dtype=np.float64)
        impulse_32k[0] = 1.0
        wet_32k = render_clouds(binary, dry_32k)
        impulse_wet_32k = render_clouds(binary, impulse_32k)

    dry = resample_linear(dry_32k, RENDER_SAMPLE_RATE, OUTPUT_SAMPLE_RATE)
    wet_stereo = resample_linear(wet_32k, RENDER_SAMPLE_RATE, OUTPUT_SAMPLE_RATE)
    impulse_stereo = resample_linear(
        impulse_wet_32k, RENDER_SAMPLE_RATE, OUTPUT_SAMPLE_RATE
    )
    wet_mono = np.mean(wet_stereo, axis=1)
    impulse_mono = np.mean(impulse_stereo, axis=1)

    wet_gain = active_rms(dry) / max(1e-15, active_rms(wet_mono))
    wet_stereo *= wet_gain
    wet_mono *= wet_gain
    mix_stereo = np.column_stack([dry, dry]) + wet_stereo * 1.15
    mix_mono = dry + wet_mono * 1.15
    tracks: dict[str, np.ndarray] = {
        "clouds_gold_wet_stereo": wet_stereo,
        "clouds_gold_mix_stereo": mix_stereo,
        "clouds_gold_wet_mono": wet_mono,
        "clouds_gold_mix_mono": mix_mono,
    }
    common_gain = math.pow(10.0, -1.0 / 20.0) / max(
        float(np.max(np.abs(track))) for track in tracks.values()
    )
    files: dict[str, object] = {}
    for name, track in tracks.items():
        path = OUT_DIR / f"{name}.wav"
        rendered = track * common_gain
        write_wav(path, rendered)
        files[path.name] = {
            "channels": 1 if track.ndim == 1 else track.shape[1],
            "rms_dbfs_0_5s": rounded(20.0 * math.log10(active_rms(rendered)), 3),
            "peak_dbfs": rounded(20.0 * math.log10(float(np.max(np.abs(rendered)))), 3),
            "sha256": sha256(path),
        }

    centers = (0.020, 0.050, 0.100, 0.200, 0.500, 1.000)
    metrics = {
        "reference": "Mutable Instruments Clouds reverb",
        "source_commit": CLOUDS_COMMIT,
        "render_sample_rate": RENDER_SAMPLE_RATE,
        "output_sample_rate": OUTPUT_SAMPLE_RATE,
        "parameters": {
            "time": 0.98,
            "diffusion": 0.70,
            "lp": 0.82,
            "input_gain": 0.20,
            "amount_for_wet_render": 1.0,
        },
        "mix_wet_to_dry_rms": 1.15,
        "common_headroom_gain": rounded(common_gain, 9),
        "mono_fold": "(left + right) / 2",
        "impulse": {
            "normalized_echo_density": {
                f"{round(center * 1000)}ms": rounded(
                    normalized_echo_density(impulse_mono, center), 4
                )
                for center in centers
            },
            "spectral_flatness": {
                "100_300ms": rounded(spectral_flatness(impulse_mono, 0.1, 0.3), 6),
                "500_1000ms": rounded(spectral_flatness(impulse_mono, 0.5, 1.0), 6),
            },
            "stereo_correlation_100ms_5s": rounded(float(np.corrcoef(
                impulse_stereo[round(0.1 * OUTPUT_SAMPLE_RATE):round(5 * OUTPUT_SAMPLE_RATE), 0],
                impulse_stereo[round(0.1 * OUTPUT_SAMPLE_RATE):round(5 * OUTPUT_SAMPLE_RATE), 1],
            )[0, 1]), 6),
        },
        "files": files,
    }
    (OUT_DIR / "reference_summary.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
