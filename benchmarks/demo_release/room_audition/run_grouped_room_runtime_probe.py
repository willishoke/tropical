#!/usr/bin/env python3
"""Compare the production groupedroom graph with its frozen-prefix oracle.

The probe drives the real Playground decoder, Plan-6 loader, JIT, and optionally
Metal.  It intentionally uses the production `.tgrm` payload rather than the
fit-time carrier archive, so the reference includes the approved complex64
prefix quantization.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ASSET = ROOT / "playground/assets/grouped-room/clouds-current-radii-mono-v1-44100.tgrm"
DIFFCLI = ROOT / "lean/.lake/build/bin/diffcli"
SAMPLE_RATE = 44100
TWO_PI = 6.283185307179586
MASTER_GAIN = 3.7
PERIODS = np.asarray(
    [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617, 1112, 882, 682, 450],
    dtype=np.int64,
)
RADII = np.asarray(
    [
        0.9999608411563025,
        0.9999608411563025,
        0.9999608411563025,
        0.9999608411563025,
        0.9999608411563025,
        0.9999608411563025,
        0.9999608411563025,
        0.9999608411563025,
        0.9993579933935869,
        0.9991915402588722,
        0.9989527279195296,
        0.9984134577393715,
    ],
    dtype=np.float64,
)
COORDINATES = (
    (211, 7.5, 0.065),
    (211, 95.5, -0.065),
    (433, 8.65, 0.055),
    (433, 100.35, -0.055),
    (887, 9.8, 0.047),
    (887, 105.2, -0.047),
    (1511, 10.95, 0.041),
    (1511, 110.05, -0.041),
    (2837, 12.1, 0.036),
    (2837, 114.9, -0.036),
    (5081, 13.25, 0.032),
    (5081, 119.75, -0.032),
)
GROUP_OFFSETS = np.concatenate(([0], np.cumsum(PERIODS[:-1]))).astype(np.int64)
PREFIX_VALUES_PER_SOURCE = int(np.sum(PERIODS))


def load_prefixes() -> tuple[np.ndarray, np.ndarray]:
    payload = ASSET.read_bytes()
    magic, version, header_bytes, sample_rate = struct.unpack_from("<4sHHI", payload, 0)
    if magic != b"TGRM" or version != 1 or header_bytes != 128 or sample_rate != SAMPLE_RATE:
        raise RuntimeError(
            f"unexpected grouped-room header: magic={magic!r} version={version} header={header_bytes} rate={sample_rate}"
        )
    shape = (len(COORDINATES), PREFIX_VALUES_PER_SOURCE)
    forward = np.frombuffer(payload, dtype="<c8", count=np.prod(shape), offset=576)
    reverse = np.frombuffer(payload, dtype="<c8", count=np.prod(shape), offset=1358976)
    return forward.reshape(shape).astype(np.complex128), reverse.reshape(shape).astype(np.complex128)


FORWARD_PREFIX, REVERSE_PREFIX = load_prefixes()


def scaled_geometric(
    exponent: complex,
    radius: float,
    period: int,
    u: np.ndarray,
    length: np.ndarray,
) -> np.ndarray:
    ratio_period = (radius / np.exp(exponent)) ** period
    p_to_u = np.exp(exponent * u)
    if abs(1.0 - ratio_period) < 1e-10:
        return length * p_to_u
    span = period * length
    return (
        p_to_u
        - np.power(radius, span) * np.exp(exponent * (u - span))
    ) / (1.0 - ratio_period)


def oracle(relative_samples: np.ndarray, position: float) -> np.ndarray:
    u = np.asarray(relative_samples, dtype=np.float64)
    causal_supported = u >= 0.0
    m = np.floor(np.maximum(u, 0.0)).astype(np.int64)
    d = np.maximum(np.ceil(-u), 0.0).astype(np.int64)
    forward_total = np.zeros(len(u), dtype=np.complex128)
    reverse_total = np.zeros(len(u), dtype=np.complex128)
    for source_index, (frequency, sigma, amplitude) in enumerate(COORDINATES):
        exponent = complex(-sigma, TWO_PI * frequency) / SAMPLE_RATE
        pole = np.exp(exponent)
        forward_response = np.zeros(len(u), dtype=np.complex128)
        reverse_response = np.zeros(len(u), dtype=np.complex128)
        for group_index, (period, radius, offset) in enumerate(
            zip(PERIODS, RADII, GROUP_OFFSETS)
        ):
            quotient = m // period
            remainder = m % period
            prefix = FORWARD_PREFIX[source_index, offset : offset + period]
            a_remainder = prefix[remainder]
            forward_response += (
                a_remainder
                * scaled_geometric(exponent, radius, int(period), u, quotient + 1)
                + (prefix[-1] - a_remainder)
                * scaled_geometric(exponent, radius, int(period), u, quotient)
            )
            future = REVERSE_PREFIX[source_index, offset : offset + period]
            z = pole * radius
            reverse_response += (
                np.exp(exponent * (u + d))
                * np.power(radius, d)
                * future[d % period]
                / (1.0 - z**period)
            )
        forward_total += amplitude * forward_response
        reverse_total += amplitude * reverse_response
    forward = np.where(causal_supported, forward_total.real, 0.0)
    reverse = reverse_total.real
    p = min(1.0, max(-1.0, position))
    return MASTER_GAIN * (
        math.sqrt(0.5 * (1.0 + p)) * forward
        + math.sqrt(0.5 * (1.0 - p)) * reverse
    )


def graph(anchor_seconds: float, position: float) -> dict[str, object]:
    modes = [[frequency, sigma, amplitude, 0] for frequency, sigma, amplitude in COORDINATES]
    return {
        "nodes": [
            {
                "id": "hit",
                "kind": "string",
                "params": {"t": anchor_seconds, "modes": modes},
                "sel": {},
                "in": {},
            },
            {
                "id": "position",
                "kind": "glideknob",
                "params": {"value": position},
                "sel": {},
                "in": {},
            },
            {
                "id": "room",
                "kind": "groupedroom",
                "params": {"profile": "clouds-current-radii-mono-v1"},
                "sel": {},
                "in": {"in": ["hit"], "position": ["position"]},
            },
            {
                "id": "outn",
                "kind": "out",
                "params": {},
                "sel": {},
                "in": {"in": ["room"]},
            },
        ],
        "out": "outn",
    }


def render(backend: str, anchor_seconds: float, position: float, start: int, count: int) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="tropical-groupedroom-") as tmp:
        graph_path = Path(tmp) / "graph.json"
        graph_path.write_text(json.dumps(graph(anchor_seconds, position)), encoding="utf-8")
        command = [
            str(DIFFCLI),
            "render-graph",
            str(graph_path),
            "--frames",
            "1",
            "--buffer",
            str(count),
            "--start",
            str(start),
        ]
        if backend == "metal":
            command.append("--metal")
        result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True)
        rendered = np.frombuffer(result.stdout, dtype="<f8")
        if len(rendered) != count:
            raise RuntimeError(
                f"{backend} returned {len(rendered)} samples, expected {count}; stderr={result.stderr.decode()}"
            )
        return rendered


def relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(candidate - reference)
        / max(np.linalg.norm(reference), 1e-30)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("jit", "metal", "both"), default="jit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backends = ("jit", "metal") if args.backend == "both" else (args.backend,)
    # Decimal anchors are the same values the graph decoder quantizes through
    # `litF`; multiplying by Fs here gives the oracle's real sample anchor.
    cases = (
        ("causal_integer", 0.0, 1.0, 0, 2048),
        ("reverse_integer", 1.0, -1.0, SAMPLE_RATE - 2048, 2048),
        ("causal_fractional", -0.000011337868, 1.0, 0, 512),
        ("reverse_fractional", 0.500011337868, -1.0, 22040, 512),
        ("midpoint_fractional", 0.500011337868, 0.0, 22040, 512),
    )
    summary: dict[str, dict[str, float]] = {}
    for backend in backends:
        summary[backend] = {}
        for name, anchor_seconds, position, start, count in cases:
            actual = render(backend, anchor_seconds, position, start, count)
            anchor_samples = anchor_seconds * SAMPLE_RATE
            coordinates = np.arange(start, start + count, dtype=np.float64) - anchor_samples
            reference = oracle(coordinates, position)
            summary[backend][name] = relative_l2(actual, reference)
    print(json.dumps(summary, indent=2, sort_keys=True))
    limits = {"jit": 1e-9, "metal": 1e-5}
    failures = [
        f"{backend}/{case}={error:.3e}>{limits[backend]:.1e}"
        for backend, results in summary.items()
        for case, error in results.items()
        if not math.isfinite(error) or error > limits[backend]
    ]
    if failures:
        raise SystemExit("grouped-room runtime oracle failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
