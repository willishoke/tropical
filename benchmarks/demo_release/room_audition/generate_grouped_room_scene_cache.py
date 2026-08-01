#!/usr/bin/env python3
"""Generate the bounded fixed-scene grouped-room fallback.

The source of truth is playground/scene.js.  Each endpoint is rendered through
the already-qualified direct JIT evaluator with the graph output temporarily
routed to the room bus, then divided by the plan's frozen master sink gain.
The payload is exactly two little-endian float32 arrays: forward followed by
classic reverse.  Runtime interpolation and POSITION mixing remain analytic.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SCENE = ROOT / "playground/scene.js"
DIFFCLI = ROOT / "lean/.lake/build/bin/diffcli"
DIRECT_MANIFEST = (
    ROOT / "playground/assets/grouped-room/clouds-current-radii-mono-v1-44100.json"
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


def direct_reference_graph(source_graph: dict[str, object]) -> dict[str, object]:
    """Replace the selected cache node with its four direct reference arms."""
    graph = json.loads(json.dumps(source_graph))
    out_node = next(node for node in graph["nodes"] if node["id"] == "out")
    nodes = [node for node in graph["nodes"] if node["id"] not in ("room", "out")]
    for number in range(1, 5):
        nodes.append(
            {
                "id": f"metalRoom{number}",
                "kind": "groupedroom",
                "params": {"profile": "clouds-current-radii-mono-v1"},
                "sel": {},
                "in": {
                    "in": [f"metalHit{number}"],
                    "position": ["positionControl"],
                },
            }
        )
    nodes.append(
        {
            "id": "room",
            "kind": "mix",
            "params": {},
            "sel": {},
            "in": {"in": [f"metalRoom{number}" for number in range(1, 5)]},
        }
    )
    nodes.append(out_node)
    graph["nodes"] = nodes
    return graph


def render_endpoint(source_graph: dict[str, object], position: float) -> np.ndarray:
    graph = direct_reference_graph(source_graph)
    graph["taps"] = False
    nodes = graph["nodes"]
    next(node for node in nodes if node["id"] == "positionControl")["params"][
        "value"
    ] = position
    next(node for node in nodes if node["id"] == "out")["in"]["in"] = ["room"]
    frames = math.ceil(SAMPLE_COUNT / BUFFER)
    with tempfile.TemporaryDirectory(prefix="tropical-room-cache-") as tmp:
        graph_path = Path(tmp) / "room-only.json"
        graph_path.write_bytes(canonical_json(graph))
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
        raise RuntimeError("direct endpoint render was incomplete or non-finite")
    return rendered / MASTER_SINK_GAIN


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
    forward = render_endpoint(source_graph, 1.0)
    reverse = render_endpoint(source_graph, -1.0)
    forward_f32 = np.asarray(forward, dtype="<f4")
    reverse_f32 = np.asarray(reverse, dtype="<f4")
    payload = forward_f32.tobytes() + reverse_f32.tobytes()
    expected_bytes = 2 * SAMPLE_COUNT * np.dtype("<f4").itemsize
    if len(payload) != expected_bytes:
        raise RuntimeError(f"cache size {len(payload)} != {expected_bytes}")
    OUTPUT.write_bytes(payload)

    direct_manifest = json.loads(DIRECT_MANIFEST.read_text(encoding="utf-8"))
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
        "source_direct_profile": direct_manifest["profile"],
        "source_direct_asset_sha256": direct_manifest["hashes"]["binary_sha256"],
        "generation": {
            "method": "room-bus endpoints rendered through direct JIT evaluator; master sink gain removed",
            "master_sink_gain_removed": MASTER_SINK_GAIN,
            "buffer_frames": BUFFER,
            "command": (
                "uv run --with numpy python "
                "benchmarks/demo_release/room_audition/"
                "generate_grouped_room_scene_cache.py"
            ),
        },
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
