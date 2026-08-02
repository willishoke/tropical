#!/usr/bin/env python3
"""Gate the fixed-scene cache against runtime reads and generation oracles."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import generate_grouped_room_scene_cache as generator


ROOT = generator.ROOT
DIFFCLI = generator.DIFFCLI
SAMPLE_RATE = generator.SAMPLE_RATE
SAMPLE_COUNT = generator.SAMPLE_COUNT
MASTER_GAIN = generator.MASTER_SINK_GAIN


def graph_for(position: float) -> dict[str, object]:
    graph = generator.export_scene()
    graph["taps"] = False
    next(node for node in graph["nodes"] if node["id"] == "positionControl")[
        "params"
    ]["value"] = position
    next(node for node in graph["nodes"] if node["id"] == "out")["in"]["in"] = [
        "room"
    ]
    return graph


def render(
    graph: dict[str, object],
    backend: str,
    start: int,
    count: int,
    velocity: float,
    tau_base: float,
) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="tropical-room-cache-probe-") as tmp:
        path = Path(tmp) / "graph.json"
        path.write_bytes(generator.canonical_json(graph))
        command = [
            str(DIFFCLI),
            "render-graph",
            str(path),
            "--frames",
            "1",
            "--buffer",
            str(count),
            "--start",
            str(start),
        ]
        if velocity != 1.0:
            command.extend(("--velocity", str(velocity)))
        if tau_base != 0.0:
            command.extend(("--tau-base", str(tau_base)))
        if backend == "metal":
            command.append("--metal")
        print(
            f"render {backend} start={start} count={count} velocity={velocity} tau={tau_base}",
            file=sys.stderr,
            flush=True,
        )
        result = subprocess.run(
            command, cwd=ROOT, capture_output=True, timeout=30
        )
        if result.returncode:
            raise RuntimeError(
                f"{backend} cache renderer failed: "
                + result.stderr.decode("utf-8", errors="replace").strip()
            )
    rendered = np.frombuffer(result.stdout, dtype="<f8")
    if rendered.size != count or not np.all(np.isfinite(rendered)):
        raise RuntimeError(f"{backend} cache render was incomplete or non-finite")
    return rendered


def cache_oracle(
    bases: np.ndarray,
    position: float,
    start: int,
    count: int,
    velocity: float,
    tau_base: float,
) -> np.ndarray:
    device = np.arange(start, start + count, dtype=np.float64)
    coordinate = tau_base * SAMPLE_RATE + velocity * device
    wrapped = np.mod(coordinate, SAMPLE_COUNT)
    base = np.floor(wrapped).astype(np.int64)
    fraction = wrapped - base
    following = (base + 1) % SAMPLE_COUNT
    forward = bases[0, base] + fraction * (bases[0, following] - bases[0, base])
    reverse = bases[1, base] + fraction * (bases[1, following] - bases[1, base])
    p = min(1.0, max(-1.0, position))
    return MASTER_GAIN * (
        math.sqrt(0.5 * (1.0 + p)) * forward
        + math.sqrt(0.5 * (1.0 - p)) * reverse
    )


def relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(candidate - reference)
        / max(np.linalg.norm(reference), 1e-30)
    )


def main() -> None:
    bases = np.fromfile(generator.OUTPUT, dtype="<f4").reshape(2, SAMPLE_COUNT)
    source_graph = generator.export_scene()
    periods, radii, carriers = generator.load_room_fit()
    room_impulse = generator.synthesize_room_impulse(periods, radii, carriers)
    generated_forward, generated_reverse = generator.render_endpoints(
        source_graph, room_impulse
    )
    generated_bases = np.stack((generated_forward, generated_reverse))
    analytic = generator.analytic_validation(
        source_graph,
        periods,
        radii,
        carriers,
        generated_forward,
        generated_reverse,
    )
    cases = (
        ("forward_integer", 1.0, 110000, 512, 1.0, 0.0),
        ("reverse_integer", -1.0, 250000, 512, 1.0, 0.0),
        ("midpoint_fractional", 0.0, 220000, 512, 0.5, 0.37 / SAMPLE_RATE),
        ("reverse_flow", -0.5, 300000, 512, -0.75, 500000.25 / SAMPLE_RATE + 0.75 * 300000 / SAMPLE_RATE),
        ("stopped_fractional", 0.5, 400000, 512, 0.0, 5.25),
    )
    results: dict[str, dict[str, float | None]] = {}
    for name, position, start, count, velocity, tau_base in cases:
        print(f"case {name}", file=sys.stderr, flush=True)
        cache_graph = graph_for(position)
        oracle = cache_oracle(
            bases, position, start, count, velocity, tau_base
        )
        generation_oracle = cache_oracle(
            generated_bases, position, start, count, velocity, tau_base
        )
        cache_jit = render(cache_graph, "jit", start, count, velocity, tau_base)
        cache_metal = (
            render(cache_graph, "metal", start, count, velocity, tau_base)
            if velocity == 1.0 and tau_base == 0.0
            else None
        )
        results[name] = {
            "cache_jit_vs_read_oracle": relative_l2(cache_jit, oracle),
            "cache_metal_vs_read_oracle": (
                relative_l2(cache_metal, oracle) if cache_metal is not None else None
            ),
            "cache_jit_vs_generation_oracle": relative_l2(
                cache_jit, generation_oracle
            ),
            "cache_vs_generation_max_absolute_error": float(
                np.max(np.abs(cache_jit - generation_oracle))
            ),
        }
    print(json.dumps({
        "analytic_generation": analytic,
        "runtime_cases": results,
    }, indent=2, sort_keys=True))
    failures: list[str] = []
    for name, measured in results.items():
        # The host writes velocity/tau as binary64, then the master clock
        # quantizes its increment and origin to Q32.32.  The read oracle uses
        # the requested decimals directly, so 1e-8 covers only that coordinate
        # quantization (observed worst case 4.7e-9).
        jit_error = measured["cache_jit_vs_read_oracle"]
        if jit_error is None or jit_error > 1e-8:
            failures.append(f"{name}: JIT read oracle")
        metal_error = measured["cache_metal_vs_read_oracle"]
        if metal_error is not None and metal_error > 1e-5:
            failures.append(f"{name}: Metal read oracle")
        # The independent generator retains binary64 until the final asset
        # quantization. This bound covers float32 storage plus Q32.32 clock
        # coordinate quantization across the FLOW matrix.
        if measured["cache_jit_vs_generation_oracle"] > 1e-6:
            failures.append(f"{name}: generation FLOW reference")
    if failures:
        raise SystemExit("scene-cache probe failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
