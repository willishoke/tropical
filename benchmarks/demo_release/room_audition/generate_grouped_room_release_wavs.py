#!/usr/bin/env python3
"""Render the fixed-scene cache release listening set.

The dry and static wet references are rendered through the native JIT graph.
The POSITION choreography mixes the same two immutable float32 arms with the
runtime's equal-power law, then is added to the JIT dry scene.  No limiter or
per-file normalization is applied: every WAV retains the authored demo gain.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import generate_grouped_room_scene_cache as cache_generator
from run_reference import write_wav


ROOT = cache_generator.ROOT
DIFFCLI = cache_generator.DIFFCLI
SAMPLE_RATE = cache_generator.SAMPLE_RATE
SAMPLE_COUNT = cache_generator.SAMPLE_COUNT
BUFFER = cache_generator.BUFFER
CACHE = cache_generator.OUTPUT
CACHE_MANIFEST = cache_generator.MANIFEST
OUTPUT = Path(__file__).resolve().parent / "release_out"
AUTHORED_LEVEL = 0.72
PRESENCE = 0.82
ROOM_LEVEL = 1.0
MASTER_GAIN = cache_generator.MASTER_SINK_GAIN


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def graph_for(position: float, room_level: float, wet_only: bool) -> dict[str, object]:
    graph = cache_generator.export_scene()
    graph["taps"] = False
    by_id = {node["id"]: node for node in graph["nodes"]}
    by_id["positionControl"]["params"]["value"] = position
    by_id["spaceLevel"]["params"]["value"] = room_level
    by_id["levelControl"]["params"]["value"] = AUTHORED_LEVEL
    if wet_only:
        # Preserve the integrated presence, listening-level, and master-gain
        # path while removing strings and the parallel dry pizzicato impact.
        by_id["body"]["in"]["in"] = ["wet"]
    return graph


def render(graph: dict[str, object]) -> np.ndarray:
    frames = math.ceil(SAMPLE_COUNT / BUFFER)
    with tempfile.TemporaryDirectory(prefix="tropical-room-release-") as tmp:
        graph_path = Path(tmp) / "scene.json"
        graph_path.write_bytes(cache_generator.canonical_json(graph))
        result = subprocess.run(
            [
                str(DIFFCLI),
                "render-graph",
                str(graph_path),
                "--frames",
                str(frames),
                "--buffer",
                str(BUFFER),
                "--start",
                "0",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
    rendered = np.frombuffer(result.stdout, dtype="<f8")[:SAMPLE_COUNT].copy()
    if rendered.size != SAMPLE_COUNT or not np.all(np.isfinite(rendered)):
        raise RuntimeError("native JIT scene render was incomplete or non-finite")
    return rendered


def smoothstep(value: np.ndarray) -> np.ndarray:
    x = np.clip(value, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def position_choreography() -> np.ndarray:
    seconds = np.arange(SAMPLE_COUNT, dtype=np.float64) / SAMPLE_RATE
    position = np.ones(SAMPLE_COUNT, dtype=np.float64)
    toward_reverse = smoothstep((seconds - 3.0) / 1.5)
    position = 1.0 - 2.0 * toward_reverse
    toward_forward = smoothstep((seconds - 7.5) / 1.5)
    position = np.where(seconds >= 7.5, -1.0 + 2.0 * toward_forward, position)
    return position


def equal_power(forward: np.ndarray, reverse: np.ndarray, position: np.ndarray) -> np.ndarray:
    forward_gain = np.sqrt(0.5 * (1.0 + np.clip(position, -1.0, 1.0)))
    reverse_gain = np.sqrt(0.5 * (1.0 - np.clip(position, -1.0, 1.0)))
    return forward_gain * forward + reverse_gain * reverse


def relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(candidate - reference)
        / max(np.linalg.norm(reference), 1e-30)
    )


def measurements(signal: np.ndarray) -> dict[str, float | int | str]:
    peak = float(np.max(np.abs(signal)))
    rms = float(np.sqrt(np.mean(np.square(signal))))
    return {
        "sample_count": int(signal.size),
        "peak": peak,
        "peak_dbfs": 20.0 * math.log10(max(peak, 1e-30)),
        "rms": rms,
        "rms_dbfs": 20.0 * math.log10(max(rms, 1e-30)),
    }


def main() -> None:
    initial_commit = git("rev-parse", "HEAD")
    initial_worktree = git("status", "--short")
    if initial_worktree:
        raise RuntimeError("release WAV generation requires a clean worktree")

    cache_manifest = json.loads(CACHE_MANIFEST.read_text(encoding="utf-8"))
    if sha256_file(CACHE) != cache_manifest["asset"]["sha256"]:
        raise RuntimeError("scene cache hash does not match its manifest")
    bases = np.fromfile(CACHE, dtype="<f4")
    if bases.size != 2 * SAMPLE_COUNT:
        raise RuntimeError("scene cache has the wrong element count")
    bases = bases.reshape(2, SAMPLE_COUNT).astype(np.float64)

    source_graph = cache_generator.export_scene()
    graph_hash = sha256_bytes(cache_generator.canonical_json(source_graph))
    if graph_hash != cache_manifest["source_graph_sha256"]:
        raise RuntimeError("current scene graph does not match the cache source graph")

    wet_forward = render(graph_for(1.0, ROOM_LEVEL, wet_only=True))
    wet_reverse = render(graph_for(-1.0, ROOM_LEVEL, wet_only=True))
    dry = render(graph_for(1.0, 0.0, wet_only=False))
    full_forward = render(graph_for(1.0, ROOM_LEVEL, wet_only=False))

    room_compensation = next(
        node for node in source_graph["nodes"]
        if node["id"] == "pizzicatoRoomCompensation"
    )["params"]["value"]
    integrated_wet_gain = (
        MASTER_GAIN
        * PRESENCE
        * AUTHORED_LEVEL
        * ROOM_LEVEL
        * room_compensation
    )
    expected_forward = integrated_wet_gain * bases[0]
    expected_reverse = integrated_wet_gain * bases[1]
    endpoint_parity = {
        "forward_relative_l2": relative_l2(wet_forward, expected_forward),
        "forward_max_absolute_error": float(np.max(np.abs(wet_forward - expected_forward))),
        "reverse_relative_l2": relative_l2(wet_reverse, expected_reverse),
        "reverse_max_absolute_error": float(np.max(np.abs(wet_reverse - expected_reverse))),
        "full_forward_vs_dry_plus_wet_relative_l2": relative_l2(
            full_forward, dry + wet_forward
        ),
        "full_forward_vs_dry_plus_wet_max_absolute_error": float(
            np.max(np.abs(full_forward - (dry + wet_forward)))
        ),
    }
    if max(
        endpoint_parity["forward_relative_l2"],
        endpoint_parity["reverse_relative_l2"],
        endpoint_parity["full_forward_vs_dry_plus_wet_relative_l2"],
    ) > 1e-9:
        raise RuntimeError(f"release render parity failed: {endpoint_parity}")

    position = position_choreography()
    wet_scrub = equal_power(wet_forward, wet_reverse, position)
    final_scene = dry + wet_scrub
    tracks = {
        "wet_position_plus_1.wav": wet_forward,
        "wet_position_minus_1.wav": wet_reverse,
        "wet_position_scrub.wav": wet_scrub,
        "final_scene_dry.wav": dry,
        "final_scene_position_plus_1.wav": full_forward,
        "final_scene_position_choreography.wav": final_scene,
    }
    max_peak = max(float(np.max(np.abs(signal))) for signal in tracks.values())
    if max_peak >= 1.0:
        raise RuntimeError(f"release listening set exceeds PCM headroom: {max_peak}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    files: dict[str, object] = {}
    for name, signal in tracks.items():
        path = OUTPUT / name
        write_wav(path, signal)
        files[name] = {
            **measurements(signal),
            "sha256": sha256_file(path),
        }

    summary = {
        "schema": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "commit": initial_commit,
        "worktree_status_at_start": initial_worktree,
        "sample_rate": SAMPLE_RATE,
        "scene_seconds": cache_generator.SCENE_SECONDS,
        "source_graph_sha256": graph_hash,
        "selected_path": "bounded fixed-scene causal/reverse cache",
        "cache": {
            "path": str(CACHE.relative_to(ROOT)),
            "sha256": cache_manifest["asset"]["sha256"],
        },
        "render_backend": "native JIT diffcli plus analytic equal-power cache mix",
        "gain": {
            "master": MASTER_GAIN,
            "presence": PRESENCE,
            "level": AUTHORED_LEVEL,
            "room": ROOM_LEVEL,
            "pizzicato_room_compensation": room_compensation,
            "integrated_wet": integrated_wet_gain,
            "normalization_or_limiter": "none",
        },
        "position_choreography": {
            "start": 1.0,
            "forward_to_reverse_seconds": [3.0, 4.5],
            "reverse_hold_pizzicato_beats_seconds": [5.06, 6.06, 7.06],
            "reverse_to_forward_seconds": [7.5, 9.0],
            "later_forward_pizzicato_beats_seconds": [9.06, 10.06],
            "curve": "smoothstep position into equal-power endpoint gains",
        },
        "endpoint_parity": endpoint_parity,
        "files": files,
        "subjective_headphone_monitor_review": "pending user approval",
    }
    summary_path = OUTPUT / "release_listening_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
