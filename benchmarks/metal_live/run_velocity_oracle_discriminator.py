#!/usr/bin/env python3
"""Reproduce the large-clock production-dispatch oracle incident without DAC."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import run as harness


BINARY = harness.ROOT / "build" / "metal_velocity_oracle_discriminator"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-build", action="store_true")
    options = parser.parse_args()
    if not options.skip_build:
        harness.run(["cmake", "--build", str(harness.ROOT / "build"), "-j4"])
    if not BINARY.exists():
        raise RuntimeError(f"missing discriminator binary: {BINARY}")

    with tempfile.TemporaryDirectory(
            prefix="tropical-velocity-oracle-") as directory:
        artifacts, _ = harness.prepare_heavy_graph(Path(directory))
        args = [
            str(BINARY),
            str(artifacts["ir"]),
            str(artifacts["coeff"]),
            str(artifacts["msl"]),
            str(artifacts["manifest"]),
            str(artifacts["reload_ir"]),
            str(artifacts["reload_coeff"]),
            str(artifacts["reload_msl"]),
            str(artifacts["reload_manifest"]),
        ]
        env = harness.benchmark_env({
            "TROPICAL_METAL_RENDER_TILE_FRAMES": None,
            "TROPICAL_KERNEL_CACHE_ROOT": str(Path(directory) / "run-cache"),
        })
        raw, wall_ns = harness.run(args, env)
    result = json.loads(raw)
    if (
        result.get("schema")
        != "tropical_metal_velocity_oracle_discriminator_2"
        or result.get("passed") is not True
    ):
        raise RuntimeError(f"velocity/oracle discriminator failed: {result}")
    result["wall_ns"] = wall_ns
    print(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
