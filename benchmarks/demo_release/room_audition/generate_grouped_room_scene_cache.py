#!/usr/bin/env python3
"""Generate the bounded fixed-scene grouped-room production cache.

The source of truth is playground/scene.js.  The accepted native grouped
carrier room is convolved offline with every fixed modal impact in that graph;
the result is checked against the independent analytic grouped equations.  The
payload is exactly two little-endian float32 arrays: causal forward followed by
classic reverse. Runtime interpolation and POSITION mixing remain analytic.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np

import run_reverse_position as reverse_room


ROOT = Path(__file__).resolve().parents[3]
SCENE = ROOT / "playground/scene.js"
DIFFCLI = ROOT / "lean/.lake/build/bin/diffcli"
ROOM_FIT_ASSET = (
    ROOT
    / "benchmarks/demo_release/room_audition/native_rate_out/"
    "grouped_fit_current_radii.npz"
)
OUTPUT = (
    ROOT
    / "playground/assets/grouped-room/clouds-current-radii-mono-v1-scene-44100.f32le"
)
MANIFEST = OUTPUT.with_suffix(".json")
SAMPLE_RATE = 44100
SCENE_SECONDS = 16
SAMPLE_COUNT = SAMPLE_RATE * SCENE_SECONDS
BUFFER = 512
MASTER_SINK_GAIN = 3.7
PROFILE = "clouds-current-radii-mono-v1-scene-cache"
ROOM_IMPULSE_SECONDS = 20
SOURCE_TAIL_SECONDS = 4


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def export_scene() -> dict[str, object]:
    script = (
        "const s=require('./playground/scene.js');"
        "process.stdout.write(JSON.stringify({"
        "sampleRate:s.SAMPLE_RATE,sceneSeconds:s.SCENE_SECONDS,"
        "graph:s.buildSceneGraph()}))"
    )
    result = subprocess.run(
        ["node", "-e", script], cwd=ROOT, check=True, capture_output=True, text=True
    )
    exported = json.loads(result.stdout)
    if exported["sampleRate"] != SAMPLE_RATE or exported["sceneSeconds"] != SCENE_SECONDS:
        raise RuntimeError(f"scene coordinate changed: {exported}")
    return exported["graph"]


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def room_source_ids(source_graph: dict[str, object]) -> list[str]:
    impact = next(node for node in source_graph["nodes"] if node["id"] == "impact")
    source_ids = list(impact["in"]["in"])
    if not source_ids or len(source_ids) != len(set(source_ids)):
        raise RuntimeError(f"invalid fixed-room source list: {source_ids}")
    return source_ids


def load_room_fit() -> tuple[tuple[int, ...], tuple[float, ...], list[np.ndarray]]:
    with np.load(ROOM_FIT_ASSET) as data:
        if int(data["sample_rate"]) != SAMPLE_RATE:
            raise RuntimeError("grouped carrier room is not at the scene sample rate")
        periods = tuple(int(value) for value in data["periods"])
        radii = tuple(float(value) for value in data["radii"])
        carriers = [
            np.asarray(data[f"carrier_{index}"], dtype=np.float64)
            for index in range(len(periods))
        ]
    if not periods or len(periods) != len(radii) or len(periods) != len(carriers):
        raise RuntimeError("grouped carrier room has inconsistent tables")
    return periods, radii, carriers


def synthesize_room_impulse(
    periods: tuple[int, ...],
    radii: tuple[float, ...],
    carriers: list[np.ndarray],
) -> np.ndarray:
    count = round(ROOM_IMPULSE_SECONDS * SAMPLE_RATE)
    indices = np.arange(count, dtype=np.int64)
    impulse = np.zeros(count, dtype=np.float64)
    for period, radius, carrier in zip(periods, radii, carriers, strict=True):
        if len(carrier) != period:
            raise RuntimeError("grouped carrier period does not match its table")
        impulse += np.power(radius, indices) * carrier[indices % period]
    if not np.all(np.isfinite(impulse)):
        raise RuntimeError("grouped room impulse is non-finite")
    return impulse


def render_fixed_source(source_graph: dict[str, object], count: int) -> np.ndarray:
    by_id = {node["id"]: node for node in source_graph["nodes"]}
    source = np.zeros(count, dtype=np.float64)
    tail_count = round(SOURCE_TAIL_SECONDS * SAMPLE_RATE)
    for source_id in room_source_ids(source_graph):
        node = by_id[source_id]
        if node["kind"] != "string":
            raise RuntimeError(f"fixed-room source {source_id} is not modal")
        anchor = round(float(node["params"]["t"]) * SAMPLE_RATE)
        active_count = min(tail_count, count - anchor)
        age = np.arange(active_count, dtype=np.float64) / SAMPLE_RATE
        for frequency, sigma, amplitude, phase in node["params"]["modes"]:
            source[anchor:anchor + active_count] += (
                amplitude
                * np.exp(-sigma * age)
                * np.cos(2.0 * math.pi * frequency * age + phase)
            )
    if not np.all(np.isfinite(source)):
        raise RuntimeError("fixed modal source is non-finite")
    return source


def render_endpoints(
    source_graph: dict[str, object], room_impulse: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    source_count = SAMPLE_COUNT + len(room_impulse) - 1
    source = render_fixed_source(source_graph, source_count)
    convolution_count = source_count + len(room_impulse) - 1
    fft_size = 1 << (convolution_count - 1).bit_length()
    source_spectrum = np.fft.rfft(source, fft_size)
    forward = np.fft.irfft(
        source_spectrum * np.fft.rfft(room_impulse, fft_size), fft_size
    )[:SAMPLE_COUNT]
    reverse_full = np.fft.irfft(
        source_spectrum * np.fft.rfft(room_impulse[::-1], fft_size), fft_size
    )
    reverse = reverse_full[
        len(room_impulse) - 1:len(room_impulse) - 1 + SAMPLE_COUNT
    ]
    if not np.all(np.isfinite(forward)) or not np.all(np.isfinite(reverse)):
        raise RuntimeError("fixed-scene endpoint convolution is non-finite")
    return forward, reverse


def analytic_validation(
    source_graph: dict[str, object],
    periods: tuple[int, ...],
    radii: tuple[float, ...],
    carriers: list[np.ndarray],
    forward: np.ndarray,
    reverse: np.ndarray,
) -> dict[str, float | int]:
    reverse_room.configure(SAMPLE_RATE, ROOM_FIT_ASSET, MANIFEST.parent)
    sample_indices = np.asarray(
        [0, 2645, 2646, 4410, 44_100, 176_400, 352_800, 529_200, 701_190],
        dtype=np.int64,
    )
    by_id = {node["id"]: node for node in source_graph["nodes"]}
    analytic_forward = np.zeros(len(sample_indices), dtype=np.float64)
    analytic_reverse = np.zeros(len(sample_indices), dtype=np.float64)
    row_count = 0
    for source_id in room_source_ids(source_graph):
        node = by_id[source_id]
        rows = node["params"]["modes"]
        relative = sample_indices - round(float(node["params"]["t"]) * SAMPLE_RATE)
        analytic_forward += reverse_room.causal_modal_response(
            rows, periods, radii, carriers, relative
        )
        analytic_reverse += reverse_room.anti_causal_modal_response(
            rows, periods, radii, carriers, relative
        )
        row_count += len(rows)
    forward_error = reverse_room.relative_l2(
        forward[sample_indices], analytic_forward
    )
    reverse_error = reverse_room.relative_l2(
        reverse[sample_indices], analytic_reverse
    )
    max_absolute_error = max(
        float(np.max(np.abs(forward[sample_indices] - analytic_forward))),
        float(np.max(np.abs(reverse[sample_indices] - analytic_reverse))),
    )
    if max(forward_error, reverse_error) > 1e-8:
        raise RuntimeError(
            "fixed-scene convolution disagrees with analytic grouped oracle: "
            f"forward={forward_error}, reverse={reverse_error}"
        )
    return {
        "source_rows": row_count,
        "sample_coordinates_checked": len(sample_indices),
        "causal_fft_vs_infinite_analytic_relative_l2": forward_error,
        "reverse_fft_vs_infinite_analytic_relative_l2": reverse_error,
        "max_absolute_error": max_absolute_error,
    }


def quantization(candidate: np.ndarray, stored: np.ndarray) -> dict[str, float]:
    delta = stored.astype(np.float64) - candidate
    return {
        "relative_l2": float(
            np.linalg.norm(delta) / max(np.linalg.norm(candidate), 1e-30)
        ),
        "max_absolute_error": float(np.max(np.abs(delta))),
    }


def main() -> None:
    source_graph = export_scene()
    source_graph_hash = sha256_bytes(canonical_json(source_graph))
    source_ids = room_source_ids(source_graph)
    periods, radii, carriers = load_room_fit()
    room_impulse = synthesize_room_impulse(periods, radii, carriers)
    forward, reverse = render_endpoints(source_graph, room_impulse)
    validation = analytic_validation(
        source_graph, periods, radii, carriers, forward, reverse
    )
    forward_f32 = np.asarray(forward, dtype="<f4")
    reverse_f32 = np.asarray(reverse, dtype="<f4")
    payload = forward_f32.tobytes() + reverse_f32.tobytes()
    expected_bytes = 2 * SAMPLE_COUNT * np.dtype("<f4").itemsize
    if len(payload) != expected_bytes:
        raise RuntimeError(f"cache size {len(payload)} != {expected_bytes}")
    OUTPUT.write_bytes(payload)

    manifest = {
        "schema": 1,
        "profile": PROFILE,
        "sample_rate": SAMPLE_RATE,
        "scene_seconds": SCENE_SECONDS,
        "sample_count_per_arm": SAMPLE_COUNT,
        "scalar_format": "float32",
        "endianness": "little",
        "interpolation": "cyclic linear",
        "position_mix": "equal power: reverse=sqrt((1-p)/2), forward=sqrt((1+p)/2)",
        "source_scene": str(SCENE.relative_to(ROOT)),
        "source_graph_sha256": source_graph_hash,
        "source_room_fit": str(ROOM_FIT_ASSET.relative_to(ROOT)),
        "source_room_fit_sha256": sha256_file(ROOM_FIT_ASSET),
        "source_nodes": source_ids,
        "downstream_room_compensation": next(
            node for node in source_graph["nodes"]
            if node["id"] == "pizzicatoRoomCompensation"
        )["params"]["value"],
        "generation": {
            "method": "fixed modal score FFT-convolved with the accepted grouped carrier room and checked against infinite analytic grouped equations",
            "room_impulse_seconds": ROOM_IMPULSE_SECONDS,
            "source_tail_seconds": SOURCE_TAIL_SECONDS,
            "command": (
                "uv run --with numpy python "
                "benchmarks/demo_release/room_audition/"
                "generate_grouped_room_scene_cache.py"
            ),
        },
        "oracle": validation,
        "arrays": [
            {
                "role": "causal",
                "byte_offset": 0,
                "element_count": SAMPLE_COUNT,
                "peak": float(np.max(np.abs(forward_f32))),
                "float32_quantization": quantization(forward, forward_f32),
            },
            {
                "role": "classic_reverse",
                "byte_offset": SAMPLE_COUNT * 4,
                "element_count": SAMPLE_COUNT,
                "peak": float(np.max(np.abs(reverse_f32))),
                "float32_quantization": quantization(reverse, reverse_f32),
            },
        ],
        "asset": {
            "path": str(OUTPUT.relative_to(ROOT)),
            "byte_count": len(payload),
            "sha256": sha256_file(OUTPUT),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
