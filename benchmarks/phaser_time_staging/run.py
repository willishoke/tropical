#!/usr/bin/env python3
"""No-DAC qualification harness for absolute-time phaser staging.

The default matrix is a bounded smoke/decision run. ``--full`` expands the
requested partial/section/interval/rate/control matrix and renders one complete
LFO cycle per row; it can take a long time at 0.02 Hz. Every Metal invocation
uses diffcli's offline renderer or tropical_runtime_bench without ``--dac``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import struct
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DIFFCLI = ROOT / "lean" / ".lake" / "build" / "bin" / "diffcli"
RUNTIME_BENCH = ROOT / "build" / "tropical_runtime_bench"
RATE = 44_100
BUFFER = 128


def clean_env(overrides: dict[str, str]) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("TROPICAL_")}
    env.update(overrides)
    return env


def command_text(args: list[str]) -> str:
    try:
        return subprocess.run(
            args, cwd=ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False,
        ).stdout.strip()
    except OSError as error:
        return f"unavailable: {error}"


def run_checked(
    args: list[str], *, env: dict[str, str], timeout: float = 600.0,
) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        args, cwd=ROOT, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=timeout, check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"{' '.join(args)} failed ({result.returncode}):\n"
            + result.stderr.decode("utf-8", "replace")
        )
    return result


def graph(partials: int, sections: int, rate: float,
          center: float, sweep: float) -> dict[str, Any]:
    def node(identifier: str, kind: str, params: dict[str, float],
             inputs: dict[str, list[str]]) -> dict[str, Any]:
        return {"id": identifier, "kind": kind, "params": params,
                "sel": {}, "in": inputs}

    return {
        "nodes": [
            node("resonator", "resonator",
                 {"freq": 220.0, "decay": 4.0,
                  "partials": float(partials)}, {"addr": []}),
            node("phaser", "phaser",
                 {"center": center, "sweep": sweep, "rate": rate,
                  "mix": 0.5, "_benchmark_stages": float(sections)},
                 {"in": ["resonator"]}),
            node("out", "out", {}, {"in": ["phaser"]}),
        ],
        "out": "out",
        "taps": [],
    }


def iter_instructions(function: dict[str, Any]) -> Iterable[dict[str, Any]]:
    yield from function.get("preamble_instructions", [])
    for child in function.get("children", []):
        yield from child.get("pre_input_instructions", [])
        yield from iter_instructions(child)
    yield from function.get("instructions", [])


def file_metric(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"bytes": len(payload), "lines": payload.count(b"\n"),
            "sha256": hashlib.sha256(payload).hexdigest()}


def artifact_metrics(directory: Path) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_text())
    instructions = [instruction
                    for function in manifest.get("instance_functions", [])
                    for instruction in iter_instructions(function)]
    tags: dict[str, int] = {}
    for instruction in instructions:
        tag = instruction.get("tag", "?")
        tags[tag] = tags.get(tag, 0) + 1
    msl = (directory / "audio.metal").read_text()
    files = {name: file_metric(directory / filename) for name, filename in {
        "audio_llvm": "audio.ll", "coefficient_llvm": "coeff.ll",
        "tile_llvm": "tile.ll", "exact_llvm": "exact.ll",
        "exact_coefficient_llvm": "exact_coeff.ll", "msl": "audio.metal",
        "manifest": "manifest.json", "exact_manifest": "exact_manifest.json",
    }.items()}
    return {
        "admission": manifest.get("phaser_time_staging"),
        "interval_frames": manifest.get("tile_interval_frames"),
        "tile_array_slots": manifest.get("tile_array_slots", []),
        "coefficient_array_slots": manifest.get("coeff_array_slots", []),
        "audio_instruction_count": len(instructions),
        "audio_instruction_tags": dict(sorted(tags.items())),
        "audio_division_count": tags.get("Div", 0),
        "audio_register_count": manifest.get("register_count"),
        "msl_operation_occurrences": {
            operation: msl.count(f"{operation}(")
            for operation in ("sin", "cos", "exp", "pow", "sqrt", "log", "atan2")
        },
        "files": files,
    }


def emit_artifacts(label: str, graph_path: Path, interval: int,
                   output: Path) -> tuple[Path, dict[str, Any]]:
    directory = output / "artifacts" / label
    directory.mkdir(parents=True, exist_ok=True)
    env = clean_env({
        "TROPICAL_PHASER_TIME_STAGING": "1",
        "TROPICAL_PHASER_TIME_STAGING_HIGHER_ORDER": "1",
        "TROPICAL_PHASER_TIME_STAGING_INTERVAL": str(interval),
        "TROPICAL_METAL_RENDER_TILE_FRAMES": str(BUFFER),
        "TROPICAL_STAGE0": "1",
        "TROPICAL_STAGE0_DUMP": str(directory),
        "TROPICAL_KERNEL_CACHE_ROOT": str(output / "cache" / f"emit-{label}"),
    })
    completed = run_checked([
        str(DIFFCLI), "render-graph", str(graph_path), "--metal",
        "--frames", "1", "--buffer", str(BUFFER),
    ], env=env)
    metrics = artifact_metrics(directory)
    metrics["compiler_stderr"] = completed.stderr.decode("utf-8", "replace").strip()
    return directory, metrics


def decode_f64(payload: bytes) -> list[float]:
    if len(payload) % 8:
        raise RuntimeError(f"renderer returned {len(payload)} non-f64-aligned bytes")
    return list(struct.unpack(f"<{len(payload) // 8}d", payload))


def render(graph_path: Path, *, metal: bool, interval: int,
           start: int, samples: int, cache: Path) -> tuple[bytes, str]:
    frames = math.ceil(samples / BUFFER)
    env = clean_env({
        "TROPICAL_STAGE0": "1",
        "TROPICAL_KERNEL_CACHE_ROOT": str(cache),
    })
    if metal:
        env.update({
            "TROPICAL_PHASER_TIME_STAGING": "1",
            "TROPICAL_PHASER_TIME_STAGING_HIGHER_ORDER": "1",
            "TROPICAL_PHASER_TIME_STAGING_INTERVAL": str(interval),
            "TROPICAL_METAL_RENDER_TILE_FRAMES": str(BUFFER),
        })
    command = [str(DIFFCLI), "render-graph", str(graph_path),
               "--frames", str(frames), "--buffer", str(BUFFER),
               "--start", str(start)]
    if metal:
        command.append("--metal")
    completed = run_checked(command, env=env)
    return completed.stdout[:samples * 8], completed.stderr.decode("utf-8", "replace")


def error_metrics(exact_payload: bytes, staged_payload: bytes,
                  interval: int) -> dict[str, Any]:
    exact = decode_f64(exact_payload)
    staged = decode_f64(staged_payload)
    if len(exact) != len(staged):
        raise RuntimeError("exact/staged sample count mismatch")
    residual = [right - left for left, right in zip(exact, staged)]
    all_finite = all(math.isfinite(value)
                     for value in exact + staged + residual)
    signal_energy = sum(value * value for value in exact)
    error_energy = sum(value * value for value in residual)
    exact_diff = [abs(exact[i] - exact[i - 1]) for i in range(1, len(exact))]
    staged_diff = [abs(staged[i] - staged[i - 1]) for i in range(1, len(staged))]
    boundaries = range(interval, len(exact), interval)
    boundary_rows = []
    transient_radius = round(RATE * 0.005)
    transient_errors: list[float] = []
    for boundary in boundaries:
        exact_first = exact_diff[boundary - 1]
        staged_first = staged_diff[boundary - 1]
        boundary_finite = math.isfinite(exact_first) and math.isfinite(staged_first)
        boundary_rows.append({
            "sample": boundary,
            "exact_first_difference": exact_first if math.isfinite(exact_first) else None,
            "staged_first_difference": staged_first if math.isfinite(staged_first) else None,
            "excess": staged_first - exact_first if boundary_finite else None,
        })
        transient_errors.extend(abs(residual[i]) for i in range(
            max(0, boundary - transient_radius),
            min(len(residual), boundary + transient_radius + 1))
            if math.isfinite(residual[i]))
    return {
        "sample_count": len(exact),
        "all_finite": all_finite,
        "exact_peak": max((abs(value) for value in exact
                           if math.isfinite(value)), default=0.0),
        "staged_peak": (max((abs(value) for value in staged), default=0.0)
                        if all_finite else None),
        "max_abs_error": (max((abs(value) for value in residual), default=0.0)
                          if all_finite else None),
        "max_peak_relative_error": (
            max((abs(value) for value in residual), default=0.0)
            / max(max((abs(value) for value in exact), default=0.0), 1.0e-30)
            if all_finite else None),
        "rms_error": (math.sqrt(error_energy / len(residual))
                      if all_finite and residual else 0.0 if all_finite else None),
        "snr_db": (10.0 * math.log10(signal_energy / error_energy)
                   if all_finite and signal_energy > 0.0 and error_energy > 0.0
                   else 999.0 if all_finite and signal_energy > 0.0 else None),
        "boundary_max_first_difference_excess": (max(
            (row["excess"] for row in boundary_rows), default=0.0)
            if all_finite else None),
        "five_ms_transient_max_abs_error": (max(transient_errors, default=0.0)
                                             if all_finite else None),
        "boundaries": boundary_rows,
    }


def performance_probe(directory: Path, interval: int,
                      blocks: int, warmup: int, cache: Path) -> dict[str, Any]:
    env = clean_env({
        "TROPICAL_PHASER_TIME_STAGING": "1",
        "TROPICAL_PHASER_TIME_STAGING_HIGHER_ORDER": "1",
        "TROPICAL_PHASER_TIME_STAGING_INTERVAL": str(interval),
        "TROPICAL_METAL_RENDER_TILE_FRAMES": str(BUFFER),
        "TROPICAL_KERNEL_CACHE_ROOT": str(cache),
    })
    completed = run_checked([
        str(RUNTIME_BENCH),
        "--ir", str(directory / "audio.ll"),
        "--manifest", str(directory / "manifest.json"),
        "--coeff", str(directory / "coeff.ll"),
        "--tile", str(directory / "tile.ll"),
        "--msl", str(directory / "audio.metal"),
        "--exact-ir", str(directory / "exact.ll"),
        "--exact-coeff", str(directory / "exact_coeff.ll"),
        "--exact-manifest", str(directory / "exact_manifest.json"),
        "--buffer", str(BUFFER), "--rate", str(RATE),
        "--blocks", str(blocks), "--warmup", str(warmup),
    ], env=env)
    # The runtime reports C++ non-finite spellings when a falsification row
    # actually emits one. Preserve that evidence instead of losing the entire
    # run to Python's stricter lowercase-token parser.
    payload = completed.stdout.decode("utf-8", "replace")
    payload = re.sub(r"(?<=[:\[,])nan(?=[,\]}])", "NaN", payload)
    payload = re.sub(r"(?<=[:\[,])inf(?=[,\]}])", "Infinity", payload)
    payload = re.sub(r"(?<=[:\[,])-inf(?=[,\]}])", "-Infinity", payload)
    row = json.loads(payload)
    for key, value in row.items():
        if isinstance(value, float) and not math.isfinite(value):
            row[key] = None
    frames = row.get("metal_rendered_frames_measured", 0)
    audio_ns = frames * 1e9 / RATE
    row["metal_render_load_fraction"] = (
        row.get("metal_render_time_measured_ns", 0) / audio_ns
        if audio_ns else None)
    row["metal_render_ns_per_frame"] = (
        row.get("metal_render_time_measured_ns", 0) / frames
        if frames else None)
    row.pop("process_ns", None)
    row.pop("write_ns", None)
    return row


def performance_summary(probes: list[dict[str, Any]]) -> dict[str, Any]:
    loads = [probe["metal_render_load_fraction"] for probe in probes]
    return {
        "median_load_fraction": statistics.median(loads),
        "p99_load_fraction": max(loads),
        "zero_deadline_overruns": all(
            probe.get("overrun_count", 0) == 0 for probe in probes),
        "zero_materialization_failures": all(
            probe.get("metal_materialization_failure_count", 0) == 0
            for probe in probes),
        "zero_exact_fallbacks": all(
            probe.get("metal_exact_fallback_count", 0) == 0
            for probe in probes),
        "zero_nonfinite_output": all(
            probe.get("nonfinite_count", 0) == 0 for probe in probes),
        "zero_dispatch_starvation_tag_activation_callback_faults": all(
            all(probe.get(key, 0) == 0 for key in (
                "metal_dispatch_failure_count",
                "metal_render_starvation_count",
                "metal_epoch_tag_mismatch_count",
                "metal_activation_failure_count",
                "metal_callback_thread_violation_count"))
            for probe in probes),
    }


def smoke_matrix() -> list[tuple[int, int, int, float, float, float]]:
    return [
        (6, 6, 128, 0.2, 700.0, 1.5),
        (6, 6, 64, 8.0, 40.0, 3.0),
        (6, 6, 32, 8.0, 4000.0, 3.0),
        (32, 12, 64, 0.2, 700.0, 1.5),
        (32, 18, 32, 8.0, 700.0, 1.5),
    ]


def higher_order_representation_metrics(
    partials: int, sections: int, interval: int,
) -> dict[str, Any]:
    source_support = 10 if interval == 128 else 8
    weight_support = 6 if interval == 32 else 8
    output_floats = 8 + 2 * source_support * partials + interval
    primitive_floats = (4 * partials
                        + source_support * (sections + 1) + interval)
    return {
        "kind": "whole_tail_newton_phase_type_table",
        "source_support": source_support,
        "weight_support": weight_support,
        "tail_samples": interval,
        "published_semantic_floats": output_floats,
        "private_primitive_floats": primitive_floats,
        "packed_segment_floats": output_floats + primitive_floats,
    }


def full_matrix() -> list[tuple[int, int, int, float, float, float]]:
    controls = [(700.0, 1.5), (40.0, 0.0), (4000.0, 3.0)]
    return [(partials, sections, interval, rate, center, sweep)
            for partials in (6, 32)
            for sections in (6, 12, 18)
            for interval in (128, 64, 32)
            for rate in (0.02, 0.2, 8.0)
            for center, sweep in controls]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "data" / "latest")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--performance-repetitions", type=int, default=3)
    parser.add_argument("--blocks", type=int, default=384)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--samples", type=int, default=4096,
                        help="per-row smoke comparison length; --full uses at least one LFO cycle")
    parser.add_argument("--start", type=int, default=4096)
    parser.add_argument("--limit", type=int,
                        help="run only the first N selected rows")
    parser.add_argument("--case-index", type=int, action="append", default=[],
                        help="run only this zero-based matrix row (repeatable)")
    parser.add_argument("--performance-only", action="store_true",
                        help="reuse emitted rows and refresh only worker probes")
    parser.add_argument("--staged-audio-only", action="store_true",
                        help="reuse graphs/exact audio and refresh staged comparisons")
    args = parser.parse_args()
    if args.performance_only and args.staged_audio_only:
        parser.error("choose at most one refresh-only mode")
    if args.performance_repetitions < 1 or args.blocks < 1 or args.samples < 1:
        parser.error("repetitions, blocks, and samples must be positive")
    for binary in (DIFFCLI, RUNTIME_BENCH):
        if not binary.exists():
            parser.error(f"missing {binary}; run make build && make lean")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.performance_only:
        raw_path = output / "raw.json"
        if not raw_path.exists():
            parser.error("--performance-only requires an existing raw.json")
        result = json.loads(raw_path.read_text())
        rows = result.get("rows", [])
        if args.case_index:
            selected = set(args.case_index)
            rows = [row for row in rows if int(row.get("index", -1)) in selected]
            if len(rows) != len(selected):
                parser.error("a refresh case index is absent from raw.json")
        for row in rows:
            label = row["label"]
            interval = int(row["interval_frames"])
            probes = [performance_probe(
                output / "artifacts" / label, interval,
                args.blocks, args.warmup,
                output / "cache" / f"probe-{label}-{repetition}")
                for repetition in range(args.performance_repetitions)]
            row["performance_rows"] = probes
            row["performance_summary"] = performance_summary(probes)
            print(f"[performance] {label}", file=sys.stderr, flush=True)
        result["performance_recorded_at"] = datetime.now(timezone.utc).isoformat()
        result["performance_repetitions"] = args.performance_repetitions
        result["performance_blocks"] = args.blocks
        result["performance_warmup"] = args.warmup
        raw_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(raw_path)
        return 0
    if args.staged_audio_only:
        raw_path = output / "raw.json"
        if not raw_path.exists():
            parser.error("--staged-audio-only requires an existing raw.json")
        result = json.loads(raw_path.read_text())
        rows = result.get("rows", [])
        if args.case_index:
            selected = set(args.case_index)
            rows = [row for row in rows if int(row.get("index", -1)) in selected]
            if len(rows) != len(selected):
                parser.error("a refresh case index is absent from raw.json")
        for row in rows:
            label = row["label"]
            interval = int(row["interval_frames"])
            graph_path = output / "graphs" / f"{label}.json"
            staged_payload, staged_stderr = render(
                graph_path, metal=True, interval=interval,
                start=int(row["start_sample"]), samples=int(row["samples"]),
                cache=output / "cache" / f"staged-{label}")
            staged_path = output / row["audio_artifacts"]["staged"]
            exact_path = output / row["audio_artifacts"]["exact"]
            error_path = output / row["audio_artifacts"]["error"]
            staged_path.write_bytes(staged_payload)
            errors = error_metrics(exact_path.read_bytes(), staged_payload, interval)
            error_path.write_text(
                json.dumps(errors, indent=2, sort_keys=True) + "\n")
            row["error"] = errors
            row["staged_renderer_stderr"] = staged_stderr.strip()
            print(f"[staged-audio] {label}", file=sys.stderr, flush=True)
        result["staged_audio_recorded_at"] = datetime.now(timezone.utc).isoformat()
        raw_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(raw_path)
        return 0
    (output / "graphs").mkdir(exist_ok=True)
    (output / "audio").mkdir(exist_ok=True)
    matrix = full_matrix() if args.full else smoke_matrix()
    if args.case_index:
        invalid = [index for index in args.case_index
                   if index < 0 or index >= len(matrix)]
        if invalid:
            parser.error(f"case indices out of range: {invalid}")
        matrix = [matrix[index] for index in args.case_index]
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        matrix = matrix[:args.limit]
    result: dict[str, Any] = {
        "schema": "tropical_absolute_time_phaser_staging_3",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "commit": command_text(["git", "rev-parse", "HEAD"]),
        "git_status": command_text(["git", "status", "--short"]),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "sample_rate": RATE,
        "device_buffer_frames": BUFFER,
        "interpolation": "adaptive_source_chebyshev_8_or_10_weight_chebyshev_6_or_8_tail_table",
        "full_matrix": args.full,
        "physical_output_opened": False,
        "rows": [],
    }

    for index, (partials, sections, interval, rate, center, sweep) in enumerate(matrix):
        label = (f"p{partials}_s{sections}_i{interval}_r{rate:g}_"
                 f"c{center:g}_w{sweep:g}")
        graph_path = output / "graphs" / f"{label}.json"
        graph_path.write_text(json.dumps(
            graph(partials, sections, rate, center, sweep),
            separators=(",", ":")) + "\n")
        directory, artifacts = emit_artifacts(label, graph_path, interval, output)
        samples = max(args.samples,
                      math.ceil(RATE / rate)) if args.full else args.samples
        exact_payload, exact_stderr = render(
            graph_path, metal=False, interval=interval, start=args.start,
            samples=samples, cache=output / "cache" / f"exact-{label}")
        staged_payload, staged_stderr = render(
            graph_path, metal=True, interval=interval, start=args.start,
            samples=samples, cache=output / "cache" / f"staged-{label}")
        exact_path = output / "audio" / f"{label}.exact.f64le"
        staged_path = output / "audio" / f"{label}.staged.f64le"
        exact_path.write_bytes(exact_payload)
        staged_path.write_bytes(staged_payload)
        errors = error_metrics(exact_payload, staged_payload, interval)
        error_path = output / "audio" / f"{label}.error.json"
        error_path.write_text(json.dumps(errors, indent=2, sort_keys=True) + "\n")
        probes = [performance_probe(
            directory, interval, args.blocks, args.warmup,
            output / "cache" / f"probe-{label}-{repetition}")
            for repetition in range(args.performance_repetitions)]
        row = {
            "index": index, "label": label, "partials": partials,
            "sections": sections, "interval_frames": interval,
            "lfo_rate_hz": rate, "center_hz": center, "sweep_octaves": sweep,
            "start_sample": args.start, "samples": samples,
            "representation": higher_order_representation_metrics(
                partials, sections, interval),
            "artifacts": artifacts, "error": errors,
            "exact_renderer_stderr": exact_stderr.strip(),
            "staged_renderer_stderr": staged_stderr.strip(),
            "audio_artifacts": {
                "exact": str(exact_path.relative_to(output)),
                "staged": str(staged_path.relative_to(output)),
                "error": str(error_path.relative_to(output)),
            },
            "performance_rows": probes,
            "performance_summary": performance_summary(probes),
        }
        result["rows"].append(row)
        raw_path = output / "raw.json"
        raw_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"[{index + 1}/{len(matrix)}] {label}", file=sys.stderr, flush=True)

    print(output / "raw.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
