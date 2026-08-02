#!/usr/bin/env python3
"""Render three chord-derived pizzicato replacements for the Metal hits.

This is deliberately an offline listening gate.  It reads the accepted native
grouped-room asset, renders an exact production string bed through ``diffcli``,
and writes local WAV evidence without opening an audio device or changing the
scene graph/cache checked into the product.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import generate_grouped_room_scene_cache as scene_cache
import run_reverse_position as reverse_room
from run import GOLDEN_ANGLE, TWO_PI
from run_reference import write_wav


AUDITION_DIR = Path(__file__).resolve().parent
ROOT = AUDITION_DIR.parents[2]
SCENE_PATH = ROOT / "playground" / "scene.js"
ASSET_PATH = (
    AUDITION_DIR / "native_rate_out" / "grouped_fit_current_radii.npz"
)
OUT_DIR = AUDITION_DIR / "pizzicato_audition_out"
SAMPLE_RATE = 44_100
OUTPUT_SECONDS = 20.0
ROOM_SECONDS = 12.0
SOURCE_TAIL_SECONDS = 3.0
PREROLL_SECONDS = 1.0
PATTERN_BEATS = 16
AUTHORED_LEVEL = 0.72
PRESENCE = 0.82
MASTER_GAIN = scene_cache.MASTER_SINK_GAIN
INTEGRATED_GAIN = AUTHORED_LEVEL * PRESENCE * MASTER_GAIN
PIZZICATO_SECTION_GAIN = 3.0
WET_SEND_GAIN = 3.0
TARGET_PEAK = math.pow(10.0, -1.0 / 20.0)
POSITION_VALUES = (1.0, 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, default=ASSET_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repository_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def load_scene_score() -> dict[str, object]:
    script = r"""
const scene = require('./playground/scene.js')
process.stdout.write(JSON.stringify({
  sampleRate: scene.SAMPLE_RATE,
  stepSeconds: scene.STEP_SECONDS,
  patternSteps: scene.PATTERN_STEPS,
  chords: scene.CHORDS.map((chord) => ({
    label: chord.label,
    name: chord.name,
    ratios: chord.voices.map((voice) => scene.absoluteRatio(chord, voice)),
    frequencies: chord.voices.map((voice) => scene.voiceFrequency(chord, voice)),
  })),
}))
"""
    result = subprocess.run(
        ["node", "-e", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    score = json.loads(result.stdout)
    if score["sampleRate"] != SAMPLE_RATE or score["patternSteps"] != PATTERN_BEATS:
        raise RuntimeError("scene score no longer matches the audition contract")
    return score


def note_event(
    seconds: float,
    chord_index: int,
    voice_index: int,
    velocity: float,
) -> dict[str, float | int]:
    return {
        "seconds": round(seconds, 6),
        "chord_index": chord_index,
        "voice_index": voice_index,
        "velocity": velocity,
    }


def build_patterns() -> list[dict[str, object]]:
    rotating: list[dict[str, float | int]] = []
    dyads_and_ghosts: list[dict[str, float | int]] = []
    rakes: list[dict[str, float | int]] = []
    rotation = (0, 2, 4, 1)
    rotating_velocity = (1.0, 0.68, 0.78, 0.62)
    ghost_voices = (2, 4, 1)
    ghost_velocity = (0.44, 0.52, 0.40)

    for beat in range(PATTERN_BEATS):
        chord_index = beat // 4
        beat_in_chord = beat % 4
        seconds = PREROLL_SECONDS + beat
        rotating.append(note_event(
            seconds,
            chord_index,
            rotation[beat_in_chord],
            rotating_velocity[beat_in_chord],
        ))
        if beat_in_chord == 0:
            # A grounded open dyad, scaled per note so its summed onset does not
            # win the comparison merely by being two voices.
            dyads_and_ghosts.extend((
                note_event(seconds, chord_index, 0, 0.72),
                note_event(seconds, chord_index, 3, 0.64),
            ))
        else:
            dyads_and_ghosts.append(note_event(
                seconds,
                chord_index,
                ghost_voices[beat_in_chord - 1],
                ghost_velocity[beat_in_chord - 1],
            ))

    for chord_index in range(4):
        chord_start = PREROLL_SECONDS + 4 * chord_index
        for order, voice_index in enumerate(range(5)):
            rakes.append(note_event(
                chord_start + 0.034 * order,
                chord_index,
                voice_index,
                0.58,
            ))

    return [
        {
            "id": "01_rotating_single",
            "label": "rotating single note every beat",
            "events": rotating,
        },
        {
            "id": "02_downbeat_dyad_ghosts",
            "label": "downbeat dyad with single-note ghost beats",
            "events": dyads_and_ghosts,
        },
        {
            "id": "03_chord_tone_rake",
            "label": "one low-to-high chord-tone rake per chord",
            "events": rakes,
        },
    ]


def pizzicato_modes(
    frequency: float,
    velocity: float,
    voice_index: int,
) -> list[list[float]]:
    """A warm, click-free pluck shared by all three rhythmic candidates."""
    rows: list[list[float]] = []
    for partial in range(1, 6):
        amplitude = 0.040 * velocity / math.pow(partial, 1.42)
        slow_decay = 9.5 + 3.5 * (partial - 1)
        fast_decay = slow_decay + 118.0 + 24.0 * (partial - 1)
        # Keep a given chord voice timbrally stable on every recurrence.  The
        # paired rows cancel exactly at the anchor and share the same phase.
        phase = (GOLDEN_ANGLE * (voice_index + 3 * partial)) % TWO_PI
        for decay, signed_amplitude in (
            (slow_decay, amplitude),
            (fast_decay, -amplitude),
        ):
            rows.append([
                frequency * partial,
                decay,
                signed_amplitude,
                phase,
            ])
    return rows


def render_source(
    events: list[dict[str, float | int]],
    score: dict[str, object],
    sample_count: int,
) -> np.ndarray:
    signal = np.zeros(sample_count, dtype=np.float64)
    tail_count = round(SOURCE_TAIL_SECONDS * SAMPLE_RATE)
    chords = score["chords"]
    for event in events:
        anchor = round(float(event["seconds"]) * SAMPLE_RATE)
        count = min(tail_count, sample_count - anchor)
        if count <= 0:
            continue
        age = np.arange(count, dtype=np.float64) / SAMPLE_RATE
        chord = chords[int(event["chord_index"])]
        frequency = chord["frequencies"][int(event["voice_index"])]
        rows = pizzicato_modes(
            float(frequency),
            float(event["velocity"]),
            int(event["voice_index"]),
        )
        for mode_frequency, decay, amplitude, phase in rows:
            signal[anchor:anchor + count] += (
                amplitude
                * np.exp(-decay * age)
                * np.cos(TWO_PI * mode_frequency * age + phase)
            )
    return signal


def render_scene_bed(sample_count: int) -> np.ndarray:
    graph = scene_cache.export_scene()
    graph["taps"] = False
    by_id = {node["id"]: node for node in graph["nodes"]}
    by_id["body"]["in"]["in"] = ["veil"]
    by_id["levelControl"]["params"]["value"] = AUTHORED_LEVEL
    frames = math.ceil(sample_count / scene_cache.BUFFER)
    with tempfile.TemporaryDirectory(prefix="tropical-pizzicato-bed-") as temp:
        graph_path = Path(temp) / "scene.json"
        graph_path.write_bytes(scene_cache.canonical_json(graph))
        rendered = subprocess.run(
            [
                str(scene_cache.DIFFCLI),
                "render-graph",
                str(graph_path),
                "--frames",
                str(frames),
                "--buffer",
                str(scene_cache.BUFFER),
                "--start",
                "0",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
    bed = np.frombuffer(rendered, dtype="<f8")[:sample_count].copy()
    if bed.size != sample_count or not np.all(np.isfinite(bed)):
        raise RuntimeError("production string-bed render was incomplete or non-finite")
    return bed


def fft_convolve_spectra(
    source: np.ndarray,
    forward_spectrum: np.ndarray,
    reverse_spectrum: np.ndarray,
    fft_size: int,
    room_count: int,
    output_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    source_spectrum = np.fft.rfft(source, fft_size)
    forward = np.fft.irfft(
        source_spectrum * forward_spectrum, fft_size
    )[:output_count]
    reverse_full = np.fft.irfft(
        source_spectrum * reverse_spectrum, fft_size
    )
    reverse = reverse_full[room_count - 1:room_count - 1 + output_count]
    return forward, reverse


def measurements(signal: np.ndarray) -> dict[str, float | int]:
    peak = float(np.max(np.abs(signal)))
    rms = float(np.sqrt(np.mean(np.square(signal))))
    return {
        "sample_count": int(signal.size),
        "peak": peak,
        "peak_dbfs": 20.0 * math.log10(max(peak, 1e-30)),
        "rms": rms,
        "rms_dbfs": 20.0 * math.log10(max(rms, 1e-30)),
    }


def attack_metrics(score: dict[str, object]) -> dict[str, float]:
    event = [note_event(0.0, 0, 0, 1.0)]
    count = round(0.25 * SAMPLE_RATE)
    source = render_source(event, score, count)
    peak_index = int(np.argmax(np.abs(source)))
    peak = float(np.max(np.abs(source)))
    threshold_indices = np.flatnonzero(np.abs(source) >= peak * 0.01)
    return {
        "one_percent_onset_ms": 1000.0 * int(threshold_indices[0]) / SAMPLE_RATE,
        "peak_latency_ms": 1000.0 * peak_index / SAMPLE_RATE,
    }


def convolution_validation(
    score: dict[str, object],
    room_impulse: np.ndarray,
    forward_spectrum: np.ndarray,
    reverse_spectrum: np.ndarray,
    fft_size: int,
    room_count: int,
    source_count: int,
) -> dict[str, float]:
    """Check FFT indexing against the literal finite grouped-room operation."""
    event = note_event(PREROLL_SECONDS, 0, 0, 1.0)
    source = render_source([event], score, source_count)
    forward, reverse = fft_convolve_spectra(
        source,
        forward_spectrum,
        reverse_spectrum,
        fft_size,
        room_count,
        round(OUTPUT_SECONDS * SAMPLE_RATE),
    )
    chord = score["chords"][0]
    rows = pizzicato_modes(float(chord["frequencies"][0]), 1.0, 0)
    offsets = np.asarray(
        [-22_050, -4_410, -1, 0, 1, 441, 4_410, 22_050, 44_100],
        dtype=np.float64,
    )
    anchor = round(PREROLL_SECONDS * SAMPLE_RATE)
    indices = anchor + offsets.astype(np.int64)
    literal_forward = reverse_room.literal_causal_response(
        rows, room_impulse, offsets
    )
    literal_reverse = reverse_room.literal_reverse_response(
        rows, room_impulse, offsets
    )
    fft_forward = forward[indices]
    fft_reverse = reverse[indices]
    forward_error = reverse_room.relative_l2(fft_forward, literal_forward)
    reverse_error = reverse_room.relative_l2(fft_reverse, literal_reverse)
    if max(forward_error, reverse_error) > 1e-9:
        raise RuntimeError(
            "finite-room FFT alignment validation failed: "
            f"forward={forward_error}, reverse={reverse_error}"
        )
    if abs(source[anchor]) > 1e-12:
        raise RuntimeError("paired pizzicato rows do not cancel at their anchor")
    return {
        "causal_fft_vs_literal_relative_l2": forward_error,
        "reverse_fft_vs_literal_relative_l2": reverse_error,
        "paired_source_anchor_absolute": float(abs(source[anchor])),
        "causal_pre_anchor_max_absolute": float(np.max(np.abs(forward[:anchor]))),
    }


def position_name(position: float) -> str:
    return "plus_1" if position == 1.0 else "plus_0p5"


def main() -> None:
    args = parse_args()
    asset = args.asset.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    score = load_scene_score()
    patterns = build_patterns()
    output_count = round(OUTPUT_SECONDS * SAMPLE_RATE)
    room_count = round(ROOM_SECONDS * SAMPLE_RATE)
    source_count = output_count + room_count - 1

    reverse_room.configure(SAMPLE_RATE, asset, output_dir)
    periods, radii, carriers = reverse_room.load_grouped_asset()
    room_impulse = reverse_room.synthesize_carriers(periods, radii, carriers)
    if room_impulse.size != room_count:
        raise RuntimeError("grouped room did not render the expected 12-second impulse")
    convolution_count = source_count + room_count - 1
    fft_size = 1 << (convolution_count - 1).bit_length()
    forward_spectrum = np.fft.rfft(room_impulse, fft_size)
    reverse_spectrum = np.fft.rfft(room_impulse[::-1], fft_size)
    validation = convolution_validation(
        score,
        room_impulse,
        forward_spectrum,
        reverse_spectrum,
        fft_size,
        room_count,
        source_count,
    )
    scene_bed = render_scene_bed(output_count)

    unscaled: dict[str, np.ndarray] = {}
    pattern_summaries: dict[str, object] = {}
    for pattern in patterns:
        pattern_id = str(pattern["id"])
        events = pattern["events"]
        source = render_source(events, score, source_count)
        direct = (
            INTEGRATED_GAIN
            * PIZZICATO_SECTION_GAIN
            * source[:output_count]
        )
        forward, reverse = fft_convolve_spectra(
            source,
            forward_spectrum,
            reverse_spectrum,
            fft_size,
            room_count,
            output_count,
        )
        unscaled[f"{pattern_id}_dry"] = scene_bed + direct
        positions: dict[str, object] = {}
        for position in POSITION_VALUES:
            wet = (
                INTEGRATED_GAIN
                * PIZZICATO_SECTION_GAIN
                * WET_SEND_GAIN
                * reverse_room.equal_power_position(forward, reverse, position)
            )
            label = position_name(position)
            unscaled[f"{pattern_id}_position_{label}_wet"] = wet
            unscaled[f"{pattern_id}_position_{label}_mix"] = (
                scene_bed + direct + wet
            )
            energy = np.square(wet)
            positions[label] = {
                "pre_first_event_energy_fraction": float(
                    np.sum(energy[:round(PREROLL_SECONDS * SAMPLE_RATE)])
                    / max(float(np.sum(energy)), 1e-30)
                ),
            }
        pattern_summaries[pattern_id] = {
            "label": pattern["label"],
            "note_event_count": len(events),
            "events": events,
            "isolated_direct": measurements(direct),
            "positions": positions,
        }

    largest_peak = max(float(np.max(np.abs(track))) for track in unscaled.values())
    common_gain = min(1.0, TARGET_PEAK / max(largest_peak, 1e-30))
    files: dict[str, object] = {}
    for name, track in unscaled.items():
        rendered = common_gain * track
        if not np.all(np.isfinite(rendered)) or float(np.max(np.abs(rendered))) >= 1.0:
            raise RuntimeError(f"{name} failed finite/headroom validation")
        path = output_dir / f"{name}.wav"
        write_wav(path, rendered)
        files[path.name] = {
            **measurements(rendered),
            "sha256": sha256_file(path),
        }

    summary = {
        "schema": 1,
        "purpose": "offline replacement audition for the four metallic room hits",
        "integration_status": "audition only; production scene and room cache unchanged",
        "render_safety": "offline files only; no audio device opened",
        "sample_rate": SAMPLE_RATE,
        "output_seconds": OUTPUT_SECONDS,
        "preroll_seconds": PREROLL_SECONDS,
        "room_impulse_seconds": ROOM_SECONDS,
        "source_tail_seconds": SOURCE_TAIL_SECONDS,
        "scene": {
            "path": repository_relative(SCENE_PATH),
            "sha256": sha256_file(SCENE_PATH),
            "score": score,
            "bed": "native JIT render of the production veil branches only",
        },
        "room_asset": {
            "path": repository_relative(asset),
            "sha256": sha256_file(asset),
            "groups": len(periods),
        },
        "pizzicato": {
            "partials": 5,
            "body_decays_per_second": [9.5, 13.0, 16.5, 20.0, 23.5],
            "paired_attack_decay_offsets_per_second": [
                118.0, 142.0, 166.0, 190.0, 214.0
            ],
            "attack": attack_metrics(score),
        },
        "validation": validation,
        "gain": {
            "master": MASTER_GAIN,
            "presence": PRESENCE,
            "level": AUTHORED_LEVEL,
            "integrated_scene_path": INTEGRATED_GAIN,
            "pizzicato_section": PIZZICATO_SECTION_GAIN,
            "pizzicato_room_send": WET_SEND_GAIN,
            "common_headroom_gain": common_gain,
            "per_file_normalization_or_limiter": "none",
        },
        "patterns": pattern_summaries,
        "files": files,
    }
    summary_path = output_dir / "pizzicato_audition_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
