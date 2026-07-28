#!/usr/bin/env python3
"""Non-interactive Apple Metal latency and soak qualification."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNTIME_BENCH = ROOT / "build" / "tropical_runtime_bench"
DIFFCLI = ROOT / "lean" / ".lake" / "build" / "bin" / "diffcli"
LATENCY = HERE / "fixtures"
MATRIX = ROOT / "benchmarks" / "current_baseline" / "fixtures" / "matrix.json"
DISCIPLINES = {"raw": 1, "glide": 3, "anchor": 2, "velocity": 2}


def run(args: list[str], env: dict[str, str] | None = None) -> tuple[bytes, int]:
    started = time.perf_counter_ns()
    proc = subprocess.run(
        args, cwd=ROOT, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    wall = time.perf_counter_ns() - started
    if proc.returncode:
        raise RuntimeError(
            f"{' '.join(args)} failed ({proc.returncode}):\n"
            + proc.stderr.decode("utf-8", "replace")
        )
    return proc.stdout, wall


def output(args: list[str]) -> str:
    try:
        return subprocess.run(
            args, cwd=ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False,
        ).stdout.strip()
    except OSError as error:
        return f"unavailable: {error}"


def percentile(values: list[int], q: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(q * len(ordered) + .999999) - 1))]


def summarize(values: list[int]) -> dict[str, int | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "median": int(statistics.median(values)) if values else None,
        "p95": percentile(values, .95),
        "p99": percentile(values, .99),
        "max": max(values) if values else None,
    }


def safe_hardware_manifest() -> dict[str, Any]:
    raw = output([
        "system_profiler", "-json", "SPHardwareDataType", "SPDisplaysDataType",
    ])
    try:
        decoded = json.loads(raw)
        hardware = decoded.get("SPHardwareDataType", [{}])[0]
        display = decoded.get("SPDisplaysDataType", [{}])[0]
        return {
            "model_identifier": hardware.get("machine_model"),
            "chip": hardware.get("chip_type"),
            "core_count": hardware.get("number_processors"),
            "memory": hardware.get("physical_memory"),
            "metal_device": display.get("sppci_model") or display.get("_name"),
            "metal_cores": display.get("sppci_cores"),
            "metal_support": display.get("spdisplays_metal"),
        }
    except (json.JSONDecodeError, IndexError, TypeError):
        return {"error": "system_profiler JSON unavailable"}


def manifest(mode: str) -> dict[str, Any]:
    toolchain = (ROOT / "lean" / "lean-toolchain").read_text().strip()
    return {
        "schema": "tropical_metal_qualification_environment_1",
        "mode": mode,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "commit": output(["git", "rev-parse", "HEAD"]),
        "git_status": output(["git", "status", "--short"]),
        "macos": output(["sw_vers"]),
        "hardware": safe_hardware_manifest(),
        "lean": output([
            str(Path.home() / ".elan" / "bin" / "elan"),
            "run", toolchain, "lean", "--version",
        ]),
        "llvm": output(["/opt/homebrew/opt/llvm/bin/llvm-config", "--version"]),
        "build": "RelWithDebInfo; TROPICAL_METAL=ON; TROPICAL_WASM_EMIT=ON",
        "sample_rate": 44100,
        "legacy_pipeline_default_depth": 3,
        "depth_precedence":
            "TROPICAL_METAL_PIPELINE_DEPTH overrides TROPICAL_METAL_PIPELINE=1",
        "cache":
            "TROPICAL_KERNEL_CACHE_ROOT points to a fresh harness-owned directory",
    }


def probe(args: list[str], env: dict[str, str]) -> dict[str, Any]:
    raw, wall = run(args, env)
    result = json.loads(raw)
    result["wall_ns"] = wall
    result["process_summary_ns"] = summarize(result["process_ns"])
    result["write_summary_ns"] = summarize(result["write_ns"])
    return result


def latency_args(buffer: int, metal: bool, write_count: int) -> list[str]:
    args = [
        str(RUNTIME_BENCH),
        "--ir", str(LATENCY / "latency.ll"),
        "--manifest", str(LATENCY / "latency.manifest.json"),
        "--buffer", str(buffer),
        "--blocks", "8",
        "--warmup", "2",
        "--slot", "0",
        "--slot-value", "0.75",
        "--write-count", str(write_count),
        "--latency",
        "--rss-every", "1",
    ]
    if metal:
        args += ["--msl", str(LATENCY / "latency.metal")]
    return args


def run_latency_matrix(stream, work: Path) -> None:
    rate = 44100
    for buffer in (128, 256, 512):
        jit_env = os.environ.copy()
        jit_env["TROPICAL_KERNEL_CACHE_ROOT"] = str(work / "cache" / f"jit-{buffer}")
        jit = probe(latency_args(buffer, False, 1), jit_env)
        if jit["latency_blocks"] != 0:
            raise RuntimeError(f"JIT latency canary at B={buffer} was not next-block")
        stream.write(json.dumps({
            "schema": "tropical_metal_latency_row_1",
            "backend": "jit",
            "buffer_length": buffer,
            "pipeline_depth": 0,
            "discipline": "raw",
            "predicted_seconds": 0,
            "observed_seconds": 0,
            "raw": jit,
        }, separators=(",", ":")) + "\n")
        stream.flush()

        for depth in (1, 2, 3):
            for discipline, write_count in DISCIPLINES.items():
                env = os.environ.copy()
                env["TROPICAL_METAL_PIPELINE_DEPTH"] = str(depth)
                env["TROPICAL_KERNEL_CACHE_ROOT"] = str(
                    work / "cache" / f"metal-{buffer}-{depth}-{discipline}"
                )
                result = probe(latency_args(buffer, True, write_count), env)
                observed = result["latency_blocks"]
                if result["pipeline_depth"] != depth or observed != depth:
                    raise RuntimeError(
                        f"stale/early future: B={buffer} D={depth} "
                        f"{discipline}, observed={observed}"
                    )
                stream.write(json.dumps({
                    "schema": "tropical_metal_latency_row_1",
                    "backend": "metal",
                    "buffer_length": buffer,
                    "pipeline_depth": depth,
                    "discipline": discipline,
                    "write_count": write_count,
                    "predicted_seconds": depth * buffer / rate,
                    "observed_seconds": observed * buffer / rate,
                    "detection_tolerance_samples": buffer,
                    "raw": result,
                }, separators=(",", ":")) + "\n")
                stream.flush()
                print(f"latency B={buffer} D={depth} {discipline}", file=os.sys.stderr)


def prepare_heavy_graph(work: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    fixtures = json.loads(MATRIX.read_text())["fixtures"]
    fixture = next(item for item in fixtures if item["name"] == "modal-bank-512")
    artifact = work / "modal-bank-512"
    artifact.mkdir(parents=True)
    graph = artifact / "graph.json"
    heavy_muted = {
        "nodes": [
            fixture["graph"]["nodes"][0],
            {"id": "mute", "kind": "knob", "params": {"value": 0}},
            {"id": "muted", "kind": "ring", "in": {"in": ["bank", "mute"]}},
            {"id": "out", "kind": "out", "in": {"in": ["muted"]}},
        ],
        "out": "out",
    }
    graph.write_text(json.dumps(heavy_muted, separators=(",", ":")))
    env = os.environ.copy()
    env["TROPICAL_STAGE0_DUMP"] = str(artifact)
    env["TROPICAL_KERNEL_CACHE_ROOT"] = str(work / "prepare-cache")
    run([
        str(DIFFCLI), "render-graph", str(graph), "--metal",
        "--frames", "0", "--buffer", "512",
    ], env)
    plan = json.loads((artifact / "manifest.json").read_text())
    sink = plan.get("sinks", [{}])[0]
    reference_slot = sink.get("inputs", [None])[0]
    param_slot = next(
        (i for i, name in enumerate(plan.get("slot_names", []))
         if isinstance(name, str) and name.startswith("param:bank.")),
        None,
    )
    if reference_slot is None or param_slot is None:
        raise RuntimeError("heavy graph did not expose reference/parameter slots")
    return {
        "ir": artifact / "audio.ll",
        "coeff": artifact / "coeff.ll",
        "msl": artifact / "audio.metal",
        "manifest": artifact / "manifest.json",
    }, {"reference_slot": reference_slot, "param_slot": param_slot}


def run_soak(stream, work: Path, duration: float, buffer: int, depth: int) -> None:
    artifacts, slots = prepare_heavy_graph(work)
    rate = 44100
    expected_blocks = max(1, int(duration * rate / buffer))
    rss_every = max(1, int(5 * rate / buffer))
    common = [
        str(RUNTIME_BENCH),
        "--ir", str(artifacts["ir"]),
        "--coeff", str(artifacts["coeff"]),
        "--msl", str(artifacts["msl"]),
        "--manifest", str(artifacts["manifest"]),
        "--buffer", str(buffer),
        "--duration-seconds", str(duration),
        "--realtime",
        "--warmup", "8",
        "--slot", str(slots["param_slot"]),
        "--slot-value", "0.75",
        "--write-every", str(max(1, int(rate / buffer))),
        "--rss-every", str(rss_every),
        "--jump-at", str(expected_blocks // 4),
        "--jump-index", str(1 << 40),
        "--reload-at", str(expected_blocks // 2),
    ]
    env = os.environ.copy()
    env["TROPICAL_METAL_PIPELINE_DEPTH"] = str(depth)
    env["TROPICAL_KERNEL_CACHE_ROOT"] = str(work / "soak-cache")
    offline_args = [
        str(RUNTIME_BENCH),
        "--ir", str(artifacts["ir"]),
        "--coeff", str(artifacts["coeff"]),
        "--msl", str(artifacts["msl"]),
        "--manifest", str(artifacts["manifest"]),
        "--buffer", str(buffer),
        "--blocks", "1000",
        "--warmup", "8",
        "--slot", str(slots["param_slot"]),
        "--slot-value", "0.75",
        "--write-every", "100",
        "--rss-every", "100",
    ]
    offline = probe(offline_args, env)
    result = probe(common + ["--dac"], env)
    rss = result["rss_bytes"]
    growth = rss[-1] - rss[1] if len(rss) > 2 else None
    measured_snapshots = [
        item for item in result["dac_snapshots"]
        if item["label"] not in {"startup_warmup_before_reset", "after_stop"}
    ]
    post_reset_underruns = max(
        (item["underrun_count"] for item in measured_snapshots), default=0
    )
    row = {
        "schema": "tropical_metal_soak_row_1",
        "fixture": "modal-bank-512",
        "duration_requested_seconds": duration,
        "buffer_length": buffer,
        "pipeline_depth": depth,
        "hot_swap_block": expected_blocks // 2,
        "clock_jump_block": expected_blocks // 4,
        "rss_post_warmup_growth_bytes": growth,
        "sink_contract":
            "live param-backed multiply defaults to zero; bank workload remains live",
        "dac_stats": result["dac_stats"],
        "post_reset_underruns": post_reset_underruns,
        "qualification_status":
            "blocked" if post_reset_underruns else "measured-window-pass",
        "offline_support": offline,
        "raw": result,
    }
    stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    stream.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("latency", "soak", "all"), default="latency")
    parser.add_argument("--duration-seconds", type=float, default=1800)
    parser.add_argument("--buffer", type=int, choices=(128, 256, 512), default=512)
    parser.add_argument("--depth", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-build", action="store_true")
    options = parser.parse_args()

    if not options.skip_build:
        run(["cmake", "--build", str(ROOT / "build"), "-j4"])
        run(["make", "lean", "JOBS=4"])
    if not RUNTIME_BENCH.exists() or not DIFFCLI.exists():
        raise RuntimeError("build artifacts are missing")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = options.output or HERE / "data" / f"{options.mode}-{stamp}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="tropical-metal-live-", dir=output_path.parent))
    try:
        with output_path.open("w") as stream:
            stream.write(json.dumps(manifest(options.mode), separators=(",", ":")) + "\n")
            if options.mode in {"latency", "all"}:
                run_latency_matrix(stream, work)
            if options.mode in {"soak", "all"}:
                run_soak(stream, work, options.duration_seconds,
                         options.buffer, options.depth)
    finally:
        shutil.rmtree(work)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
