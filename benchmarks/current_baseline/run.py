#!/usr/bin/env python3
"""Current-main compiler/runtime baseline.

The orchestrator intentionally measures subprocess walls and an artifact-only
native probe rather than changing production schemas. Every row retains raw
samples and labels inseparable frontend walls as coarse.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MATRIX = HERE / "fixtures" / "matrix.json"
DIFFCLI = ROOT / "lean" / ".lake" / "build" / "bin" / "diffcli"
RUNTIME_BENCH = ROOT / "build" / "tropical_runtime_bench"


def command(args: list[str], env: dict[str, str] | None = None) -> tuple[bytes, int]:
    started = time.perf_counter_ns()
    proc = subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    elapsed = time.perf_counter_ns() - started
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", "replace")
        raise RuntimeError(f"{' '.join(args)} failed ({proc.returncode}):\n{stderr}")
    return proc.stdout, elapsed


def text_command(args: list[str]) -> str:
    try:
        return subprocess.run(
            args, cwd=ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False
        ).stdout.strip()
    except OSError as error:
        return f"unavailable: {error}"


def percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(fraction * len(ordered) + 0.999999) - 1))
    return ordered[index]


def summary(values: list[int]) -> dict[str, int | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "median": int(statistics.median(values)) if values else None,
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def instruction_count(plan: Any) -> int:
    if isinstance(plan, list):
        return sum(instruction_count(item) for item in plan)
    if not isinstance(plan, dict):
        return 0
    count = 0
    for key, value in plan.items():
        if key in {"instructions", "preamble", "pre_input_instructions",
                   "state_evolution", "postamble"} and isinstance(value, list):
            count += len(value)
        count += instruction_count(value)
    return count


def graph_size(fixture: dict[str, Any], source: Path | None) -> tuple[int, int]:
    if fixture["class"] == "graph":
        encoded = json.dumps(fixture["graph"], separators=(",", ":")).encode()
        return len(fixture["graph"].get("nodes", [])), len(encoded)
    assert source is not None
    parsed = json.loads(source.read_text())
    decls = parsed.get("body", {}).get("decls", [])
    return len(decls), source.stat().st_size


def cache_snapshot() -> dict[str, Any]:
    ordinary = Path.home() / ".cache" / "tropical" / "kernels"
    if not ordinary.exists():
        return {"exists": False, "files": 0, "mtime_ns": None}
    files = [path for path in ordinary.rglob("*") if path.is_file()]
    return {
        "exists": True,
        "files": len(files),
        "mtime_ns": ordinary.stat().st_mtime_ns,
    }


def safe_hardware_manifest() -> dict[str, Any]:
    raw = text_command([
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


def environment_manifest() -> dict[str, Any]:
    toolchain = (ROOT / "lean" / "lean-toolchain").read_text().strip()
    return {
        "schema": "tropical_baseline_environment_1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "commit": text_command(["git", "rev-parse", "HEAD"]),
        "git_status": text_command(["git", "status", "--short"]),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "macos": text_command(["sw_vers"]),
        "hardware": safe_hardware_manifest(),
        "lean": text_command([
            str(Path.home() / ".elan" / "bin" / "elan"),
            "run", toolchain, "lean", "--version",
        ]),
        "llvm": text_command(["/opt/homebrew/opt/llvm/bin/llvm-config", "--version"]),
        "cmake_cache": text_command([
            "cmake", "-LA", "-N", str(ROOT / "build")
        ]),
        "sample_rate": 44100,
        "block_length": 512,
        "stage0": os.environ.get("TROPICAL_STAGE0", "default-on"),
        "banks_unroll": os.environ.get("TROPICAL_BANKS_UNROLL", "default-banked"),
        "ordinary_cache_before": cache_snapshot(),
    }


def artifact_metrics(manifest_path: Path, ir_path: Path, coeff_path: Path,
                     msl_path: Path | None) -> dict[str, Any]:
    plan = json.loads(manifest_path.read_text())
    ir = ir_path.read_bytes()
    coeff = coeff_path.read_bytes() if coeff_path.exists() else b""
    msl = msl_path.read_bytes() if msl_path and msl_path.exists() else b""
    return {
        "plan_instruction_count": instruction_count(plan),
        "register_count": plan.get("register_count"),
        "slot_count": plan.get("slot_count"),
        "array_slot_count": plan.get("array_slot_count"),
        "coeff_array_slot_count": len(plan.get("coeff_array_slots", [])),
        "llvm_bytes": len(ir),
        "llvm_lines": ir.count(b"\n"),
        "coefficient_llvm_bytes": len(coeff),
        "coefficient_llvm_lines": coeff.count(b"\n"),
        "msl_bytes": len(msl) if msl else None,
        "msl_lines": msl.count(b"\n") if msl else None,
        "wasm_bytes": None,
    }


def runtime_args(ir: Path, manifest: Path, coeff: Path, msl: Path | None,
                 blocks: int, slot: int | None) -> list[str]:
    args = [
        str(RUNTIME_BENCH),
        "--ir", str(ir),
        "--manifest", str(manifest),
        "--buffer", "512",
        "--blocks", str(blocks),
        "--warmup", "8",
        "--rate", "44100",
    ]
    if coeff.exists() and coeff.stat().st_size:
        args += ["--coeff", str(coeff)]
    if msl is not None:
        args += ["--msl", str(msl)]
    if slot is not None:
        args += ["--slot", str(slot), "--write-every", "10"]
    return args


def run_probe(args: list[str], cache_root: Path,
              extra_env: dict[str, str] | None = None) -> dict[str, Any]:
    env = os.environ.copy()
    env["TROPICAL_KERNEL_CACHE_ROOT"] = str(cache_root)
    if extra_env:
        env.update(extra_env)
    stdout, wall_ns = command(args, env)
    result = json.loads(stdout)
    result["process_summary_ns"] = summary(result["process_ns"])
    result["write_summary_ns"] = summary(result["write_ns"])
    result["subprocess_wall_ns"] = wall_ns
    return result


def prepare_artifacts(fixture: dict[str, Any], directory: Path,
                      metal: bool) -> tuple[dict[str, Path], dict[str, int | None]]:
    directory.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TROPICAL_STAGE0_DUMP"] = str(directory)
    env["TROPICAL_KERNEL_CACHE_ROOT"] = str(directory / "prepare-cache")
    timings: dict[str, int | None] = {
        "frontend_plan_total_ns": None,
        "artifact_prepare_total_ns": None,
        "stage0_split_ns": None,
        "plan_emit_ns": None,
    }

    if fixture["class"] == "program":
        source = ROOT / fixture["path"]
        plan_out, plan_ns = command([str(DIFFCLI), "compile", str(source)], env)
        manifest = directory / "manifest.json"
        manifest.write_bytes(plan_out)
        ir_out, ir_ns = command([str(DIFFCLI), "emit-ir", str(source)], env)
        (directory / "audio.ll").write_bytes(ir_out)
        (directory / "coeff.ll").write_bytes(b"")
        timings["frontend_plan_total_ns"] = plan_ns
        timings["artifact_prepare_total_ns"] = ir_ns
        if metal:
            msl_out, msl_ns = command([str(DIFFCLI), "emit-msl", str(source)], env)
            (directory / "audio.metal").write_bytes(msl_out)
            timings["msl_frontend_emit_total_ns"] = msl_ns
    else:
        graph_path = directory / "graph.json"
        graph_path.write_text(json.dumps(fixture["graph"], separators=(",", ":")))
        args = [
            str(DIFFCLI), "render-graph", str(graph_path),
            "--frames", "0", "--buffer", "512",
        ]
        if metal:
            args.append("--metal")
        _, total_ns = command(args, env)
        timings["artifact_prepare_total_ns"] = total_ns
        timings["topology_edit_to_publication_ns"] = total_ns

    paths = {
        "manifest": directory / "manifest.json",
        "ir": directory / "audio.ll",
        "coeff": directory / "coeff.ll",
        "msl": directory / "audio.metal",
    }
    return paths, timings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--runtime-blocks", type=int, default=100)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--metal", action="store_true")
    parser.add_argument("--wasm", action="store_true")
    options = parser.parse_args()

    if not options.skip_build:
        command(["cmake", "--build", str(ROOT / "build"), "-j4"])
        command(["make", "lean", "JOBS=4"])
    for required in (DIFFCLI, RUNTIME_BENCH):
        if not required.exists():
            raise RuntimeError(f"missing build artifact: {required}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = options.output or HERE / "data" / f"{options.suite}-{timestamp}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="tropical-baseline-", dir=output.parent))
    environment = environment_manifest()
    matrix = json.loads(MATRIX.read_text())["fixtures"]
    fixtures = [item for item in matrix if options.suite == "full" or item.get("smoke")]

    try:
        with output.open("w") as stream:
            stream.write(json.dumps(environment, separators=(",", ":")) + "\n")
            for fixture in fixtures:
                fixture_dir = work / fixture["name"]
                paths, timings = prepare_artifacts(fixture, fixture_dir, options.metal)
                plan = json.loads(paths["manifest"].read_text())
                slot_names = plan.get("slot_names", [])
                param_slot = next(
                    (index for index, name in enumerate(slot_names)
                     if isinstance(name, str) and name.startswith("param:")),
                    None,
                )
                source = ROOT / fixture["path"] if fixture["class"] == "program" else None
                nodes, source_bytes = graph_size(fixture, source)
                metrics = artifact_metrics(
                    paths["manifest"], paths["ir"], paths["coeff"],
                    paths["msl"] if options.metal else None,
                )

                if options.wasm and fixture["class"] == "program":
                    wasm = fixture_dir / "kernel.wasm"
                    _, wasm_ns = command([
                        str(DIFFCLI), "compile-wasm", str(source),
                        f"--out={wasm}",
                    ])
                    metrics["wasm_bytes"] = wasm.stat().st_size
                    timings["wasm_total_ns"] = wasm_ns

                runtime: dict[str, list[dict[str, Any]]] = {
                    "jit_cold": [], "jit_warm": [],
                    "metal_cold": [], "metal_warm": [],
                }
                for repeat in range(options.repeats):
                    cache_root = work / "cache" / fixture["name"] / str(repeat)
                    if cache_root.exists():
                        shutil.rmtree(cache_root)
                    jit_args = runtime_args(
                        paths["ir"], paths["manifest"], paths["coeff"],
                        None, options.runtime_blocks, param_slot,
                    )
                    runtime["jit_cold"].append(run_probe(jit_args, cache_root))
                    runtime["jit_warm"].append(run_probe(jit_args, cache_root))
                    if options.metal:
                        metal_args = runtime_args(
                            paths["ir"], paths["manifest"], paths["coeff"],
                            paths["msl"], options.runtime_blocks, param_slot,
                        )
                        metal_cache = work / "cache-metal" / fixture["name"] / str(repeat)
                        runtime["metal_cold"].append(run_probe(metal_args, metal_cache))
                        runtime["metal_warm"].append(run_probe(metal_args, metal_cache))

                row = {
                    "schema": "tropical_baseline_row_1",
                    "fixture": fixture["name"],
                    "fixture_class": fixture["class"],
                    "source_nodes": nodes,
                    "source_bytes": source_bytes,
                    "arena_node_count": None,
                    "bank_capacity": fixture.get("bank_capacity"),
                    "live_count": fixture.get("live_count"),
                    "nested": fixture.get("nested", False),
                    "flagship": fixture.get("flagship", False),
                    "metrics": metrics,
                    "timings": timings,
                    "runtime": runtime,
                    "cache_state": {
                        "root_env": "TROPICAL_KERNEL_CACHE_ROOT",
                        "cold": "fresh per-repeat benchmark-owned root",
                        "warm": "second process reusing that root",
                    },
                    "limitations": [
                        "frontend plan/lower/emit walls are coarse subprocess totals",
                        "arena node count is not exposed without invasive compiler instrumentation",
                        "Metal load includes the mandatory dual JIT load",
                    ],
                }
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                stream.flush()
                print(f"recorded {fixture['name']}", file=sys.stderr)

            ordinary_after = cache_snapshot()
            trailer = {
                "schema": "tropical_baseline_trailer_1",
                "ordinary_cache_after": ordinary_after,
                "ordinary_cache_unchanged":
                    ordinary_after == environment["ordinary_cache_before"],
            }
            stream.write(json.dumps(trailer, separators=(",", ":")) + "\n")
    finally:
        shutil.rmtree(work)

    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
