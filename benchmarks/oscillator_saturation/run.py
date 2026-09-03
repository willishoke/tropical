#!/usr/bin/env python3
"""Time-domain oscillator saturation sweep — JIT versus Metal.

The modal/reverb work measures flagship graphs that are heavy by construction.
This harness measures the opposite end: plain closed-form oscillators, the
cheapest real signal source the vocabulary has, swept by count until a backend
spends a stated fraction of its realtime budget producing one block.

Saturation is defined per block:

    saturation(N) = per-block production wall / block deadline
    block deadline = buffer_frames / rate

A backend's capacity at threshold T is the largest oscillator count whose
saturation stays at or below T. This harness BRACKETS that capacity between
adjacent measured counts and never reports an unmeasured count as measured;
an interpolated midpoint is recorded separately and labelled as an estimate.

Both backends are measured through the same `tropical_runtime_bench` block
loop over artifacts emitted by the same front end, so the only difference
between rows is the backend under test. A backend that refuses a count (Metal
rejects kernels exceeding its per-thread stack) is recorded as a result and
excluded from the bracket, never guessed at.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DIFFCLI = ROOT / "lean" / ".lake" / "build" / "bin" / "diffcli"
RUNTIME_BENCH = ROOT / "build" / "tropical_runtime_bench"

ROW_SCHEMA = "tropical_oscillator_saturation_1"

# Identical control surface to benchmarks/current_baseline: every inherited
# TROPICAL_* variable is removed, then these are set explicitly and recorded.
BASE_CONTROLS: dict[str, str | None] = {
    "TROPICAL_STAGE0": "1",
    "TROPICAL_BANKS_UNROLL": None,
    "TROPICAL_JIT_OPT_LEVEL": "O2",
    "TROPICAL_KERNEL_CACHE_DISABLE": "0",
    "TROPICAL_BACKEND": None,
    "TROPICAL_BUFFER_LENGTH": None,
    "TROPICAL_METAL_RENDER_TILE_FRAMES": None,
    "TROPICAL_STAGE0_DUMP": None,
}

DEFAULT_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024]
SMOKE_COUNTS = [1, 4, 16, 64]


# ---------------------------------------------------------------- primitives

def benchmark_env(overrides: dict[str, str | None]) -> dict[str, str]:
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith("TROPICAL_")
    }
    controls = BASE_CONTROLS | overrides
    env.update({k: v for k, v in controls.items() if v is not None})
    return env


class EmitTimeout(RuntimeError):
    """Artifact emission exceeded its wall budget for this oscillator count."""


def command(args: list[str], env: dict[str, str] | None = None,
            timeout: float | None = None) -> tuple[bytes, int]:
    started = time.perf_counter_ns()
    try:
        proc = subprocess.run(
            args, cwd=ROOT, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False, timeout=timeout,
        )
    except subprocess.TimeoutExpired as expired:
        raise EmitTimeout(
            f"{' '.join(args)} exceeded {timeout}s") from expired
    elapsed = time.perf_counter_ns() - started
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", "replace")
        raise RuntimeError(f"{' '.join(args)} failed ({proc.returncode}):\n{stderr}")
    return proc.stdout, elapsed


def text_command(args: list[str]) -> str:
    try:
        return subprocess.run(
            args, cwd=ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False,
        ).stdout.strip()
    except OSError as error:
        return f"unavailable: {error}"


def percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1,
                max(0, int(fraction * len(ordered) + 0.999999) - 1))
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


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------- fixture

def build_graph(count: int, morph: float, base_freq: float,
                spread: float) -> dict[str, Any]:
    """`count` independent `source` oscillators summed into one output.

    `source` is the vocabulary's plain time-domain oscillator (freq/morph/pm).
    Frequencies are spread so no two voices share a coefficient and the graph
    cannot collapse under common-subexpression elimination — the sweep must
    scale real oscillators, not deduplicated copies of one.
    """
    nodes: list[dict[str, Any]] = [
        {
            "id": f"osc{i}",
            "kind": "source",
            "params": {"freq": base_freq * (1.0 + spread * i), "morph": morph},
        }
        for i in range(count)
    ]
    nodes.append({
        "id": "mix",
        "kind": "mix",
        "in": {"in": [f"osc{i}" for i in range(count)]},
    })
    nodes.append({"id": "out", "kind": "out", "in": {"in": ["mix"]}})
    return {"nodes": nodes, "out": "out"}


def build_patch(count: int, base_freq: float, spread: float) -> dict[str, Any]:
    """`count` `FixedSinOsc` instances summed into a `SoftClip` sink.

    This is the `tropical_program_2` patch-bay spelling of the same question
    the graph fixture asks, and it is the one that reaches useful counts:
    `dac.out` must be a bare `ref`, so the sum is folded through instance
    inputs and terminated at a single shaper. `FixedSinOsc` is the minimal
    time-domain oscillator the stdlib registers.

    Unlike the graph fixture this route reaches `--mode`, so fused and
    per-instance-microkernel realizations are both measurable.
    """
    decls: list[dict[str, Any]] = [
        {
            "op": "instanceDecl",
            "name": f"osc{i}",
            "program": "FixedSinOsc",
            "inputs": {"freq": base_freq * (1.0 + spread * i)},
        }
        for i in range(count)
    ]

    def ref(i: int) -> dict[str, Any]:
        return {"op": "ref", "instance": f"osc{i}", "output": "sine"}

    expr: dict[str, Any] = ref(0)
    for i in range(1, count):
        expr = {"op": "add", "args": [expr, ref(i)]}
    decls.append({
        "op": "instanceDecl", "name": "sink", "program": "SoftClip",
        "inputs": {"input": expr, "drive": 1},
    })
    return {
        "schema": "tropical_program_2",
        "name": f"osc_saturation_{count}",
        "body": {
            "op": "block",
            "decls": decls,
            "assigns": [{
                "op": "outputAssign", "name": "dac.out",
                "expr": {"op": "ref", "instance": "sink", "output": "out"},
            }],
        },
    }


def emit_artifacts(count: int, directory: Path, metal: bool,
                   cache_root: Path, buffer_frames: int, timeout: float,
                   args: argparse.Namespace) -> tuple[dict[str, Path], int]:
    directory.mkdir(parents=True, exist_ok=True)
    env = benchmark_env({
        "TROPICAL_STAGE0_DUMP": str(directory),
        "TROPICAL_KERNEL_CACHE_ROOT": str(cache_root),
    })
    paths = {
        "manifest": directory / "manifest.json",
        "ir": directory / "audio.ll",
        "coeff": directory / "coeff.ll",
        "msl": directory / "audio.metal",
    }

    if args.fixture == "graph":
        source = directory / "graph.json"
        source.write_text(json.dumps(
            build_graph(count, args.morph, args.base_freq, args.spread),
            separators=(",", ":")))
        cmd = [str(DIFFCLI), "render-graph", str(source),
               "--frames", "0", "--buffer", str(buffer_frames)]
        if metal:
            cmd.append("--metal")
        _, emit_ns = command(cmd, env, timeout=timeout)
        return paths, emit_ns

    # patch route: compile at the requested mode, then dump artifacts by
    # loading the plan through the backend under test.
    source = directory / "patch.json"
    source.write_text(json.dumps(
        build_patch(count, args.base_freq, args.spread), separators=(",", ":")))
    plan, plan_ns = command(
        [str(DIFFCLI), "compile", str(source), f"--mode={args.mode}"],
        env, timeout=timeout)
    paths["manifest"].write_bytes(plan)
    verb = "render-metal" if metal else "render-bytes"
    _, dump_ns = command(
        [str(DIFFCLI), verb, str(paths["manifest"]),
         "--frames", "0", "--buffer", str(buffer_frames)],
        env, timeout=timeout)
    return paths, plan_ns + dump_ns


# --------------------------------------------------------------- measurement

def measure(paths: dict[str, Path], metal: bool, buffer_frames: int,
            blocks: int, warmup: int, rate: int,
            cache_root: Path) -> dict[str, Any]:
    args = [
        str(RUNTIME_BENCH),
        "--ir", str(paths["ir"]),
        "--manifest", str(paths["manifest"]),
        "--buffer", str(buffer_frames),
        "--blocks", str(blocks),
        "--warmup", str(warmup),
        "--rate", str(rate),
    ]
    if paths["coeff"].exists() and paths["coeff"].stat().st_size:
        args += ["--coeff", str(paths["coeff"])]
    if metal:
        args += ["--msl", str(paths["msl"])]
    env = benchmark_env({"TROPICAL_KERNEL_CACHE_ROOT": str(cache_root)})
    stdout, wall_ns = command(args, env)
    result = json.loads(stdout)
    result["subprocess_wall_ns"] = wall_ns
    return result


def duty(value_ns: int | None, deadline_ns: int) -> float | None:
    if value_ns is None or deadline_ns <= 0:
        return None
    return value_ns / deadline_ns


def sweep_point(count: int, backend: str, args: argparse.Namespace,
                workdir: Path, cache_root: Path) -> dict[str, Any]:
    metal = backend == "metal"
    directory = workdir / backend / f"n{count}"

    point: dict[str, Any] = {
        "oscillators": count,
        "backend": backend,
        "fixture": args.fixture,
        "mode": args.mode if args.fixture == "patch" else "fused",
        "emit_timed_out": False,
        "emit_failed": False,
        "emit_timeout_seconds": args.emit_timeout,
    }
    try:
        paths, emit_ns = emit_artifacts(
            count, directory, metal, cache_root, args.buffer,
            args.emit_timeout, args)
    except EmitTimeout as timeout:
        point["emit_timed_out"] = True
        point["emit_timeout_detail"] = str(timeout)
        return point
    except RuntimeError as failure:
        # A backend can refuse a count outright — Metal, for instance, rejects
        # a kernel that exceeds its per-thread stack. That is a RESULT (the
        # backend's structural ceiling), not a harness crash, so it is
        # recorded and the sweep continues.
        point["emit_failed"] = True
        point["emit_failure_detail"] = str(failure).strip().splitlines()[-1][:400]
        return point

    point["emit_ns"] = emit_ns
    point["artifact_bytes"] = {
        "audio_ir": paths["ir"].stat().st_size if paths["ir"].exists() else None,
        "coefficient_ir": (
            paths["coeff"].stat().st_size if paths["coeff"].exists() else None),
        "manifest": (
            paths["manifest"].stat().st_size if paths["manifest"].exists() else None),
        "msl": paths["msl"].stat().st_size if paths["msl"].exists() else None,
    }
    point["artifact_sha256"] = {
        "audio_ir": sha256_file(paths["ir"]),
        "msl": sha256_file(paths["msl"]) if metal else None,
    }

    # Adjacent measurements contend for the same cores and GPU queue; an
    # unsettled run inflates the tail without inflating the median, which
    # silently corrupts a p99-based capacity. Settle, then measure.
    time.sleep(args.settle_seconds)
    probe = measure(paths, metal, args.buffer, args.blocks, args.warmup,
                    args.rate, cache_root)

    process_ns = probe["process_ns"]
    deadline_ns = probe["deadline_ns"]
    stats = summary(process_ns)
    point.update({
        "reported_backend": probe["backend"],
        "deadline_ns": deadline_ns,
        "buffer_frames": probe["buffer_length"],
        "sample_rate": probe["sample_rate"],
        "metal_render_tile_frames": probe.get("metal_render_tile_frames"),
        "load_ns": probe.get("load_ns"),
        "overrun_count": probe.get("overrun_count"),
        "nonfinite_count": probe.get("nonfinite_count"),
        "process_summary_ns": stats,
        "saturation": {
            "median": duty(stats["median"], deadline_ns),
            "p95": duty(stats["p95"], deadline_ns),
            "p99": duty(stats["p99"], deadline_ns),
            "max": duty(stats["max"], deadline_ns),
        },
        "per_oscillator_median_ns": (
            stats["median"] / count if stats["median"] is not None else None),
    })
    return point


# ------------------------------------------------------------------ capacity

def capacity(points: list[dict[str, Any]], threshold: float,
             statistic: str) -> dict[str, Any]:
    """Bracket the largest oscillator count that stays within `threshold`.

    Reports the measured bracket only. The geometric midpoint is recorded as
    an explicitly labelled estimate and is never presented as a measurement.
    """
    usable = [
        p for p in points
        if not p["emit_timed_out"] and not p.get("emit_failed")
        and p.get("saturation", {}).get(statistic) is not None
    ]
    usable.sort(key=lambda p: p["oscillators"])
    if not usable:
        return {"statistic": statistic, "threshold": threshold,
                "status": "no_measurements"}

    below = [p for p in usable if p["saturation"][statistic] <= threshold]
    above = [p for p in usable if p["saturation"][statistic] > threshold]

    if not below:
        return {
            "statistic": statistic, "threshold": threshold,
            "status": "threshold_exceeded_at_smallest_measured_count",
            "smallest_measured": usable[0]["oscillators"],
            "smallest_measured_saturation": usable[0]["saturation"][statistic],
        }
    if not above:
        return {
            "statistic": statistic, "threshold": threshold,
            "status": "threshold_not_reached_within_sweep",
            "largest_measured": below[-1]["oscillators"],
            "largest_measured_saturation": below[-1]["saturation"][statistic],
        }

    lower = below[-1]
    upper = min(above, key=lambda p: p["oscillators"])
    return {
        "statistic": statistic,
        "threshold": threshold,
        "status": "bracketed",
        "capacity_lower_measured": lower["oscillators"],
        "capacity_lower_saturation": lower["saturation"][statistic],
        "capacity_upper_measured": upper["oscillators"],
        "capacity_upper_saturation": upper["saturation"][statistic],
        "interpolated_estimate": interpolate(
            lower["oscillators"], lower["saturation"][statistic],
            upper["oscillators"], upper["saturation"][statistic], threshold),
        "interpolated_estimate_note":
            "log-linear between the two bracketing measurements; an estimate, "
            "not a measured count",
    }


def interpolate(n_lo: int, s_lo: float, n_hi: int, s_hi: float,
                threshold: float) -> float | None:
    if s_hi <= s_lo or n_hi <= n_lo:
        return None
    import math
    frac = (threshold - s_lo) / (s_hi - s_lo)
    return math.exp(math.log(n_lo) + frac * (math.log(n_hi) - math.log(n_lo)))


def crossover(rows: dict[str, list[dict[str, Any]]],
              statistic: str) -> dict[str, Any]:
    """Smallest measured count at which Metal's saturation drops below JIT's."""
    jit = {p["oscillators"]: p for p in rows.get("jit", [])
           if not p["emit_timed_out"] and not p.get("emit_failed")}
    metal = {p["oscillators"]: p for p in rows.get("metal", [])
             if not p["emit_timed_out"] and not p.get("emit_failed")}
    shared = sorted(set(jit) & set(metal))
    if not shared:
        return {"status": "no_shared_measured_counts"}
    comparisons = []
    for n in shared:
        j = jit[n]["saturation"][statistic]
        m = metal[n]["saturation"][statistic]
        if j is None or m is None:
            continue
        comparisons.append({
            "oscillators": n, "jit": j, "metal": m,
            "metal_cheaper": m < j,
            "jit_over_metal": (j / m) if m > 0 else None,
        })
    winners = [c for c in comparisons if c["metal_cheaper"]]
    return {
        "statistic": statistic,
        "shared_counts": shared,
        "comparisons": comparisons,
        "status": "metal_wins_from" if winners else "jit_cheaper_at_every_measured_count",
        "metal_cheaper_from": winners[0]["oscillators"] if winners else None,
    }


# --------------------------------------------------------------- environment

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


def environment_manifest(args: argparse.Namespace) -> dict[str, Any]:
    toolchain = (ROOT / "lean" / "lean-toolchain").read_text().strip()
    return {
        "schema": "tropical_oscillator_saturation_environment_1",
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
        "sample_rate": args.rate,
        "buffer_frames": args.buffer,
        "blocks": args.blocks,
        "warmup": args.warmup,
        "settle_seconds": args.settle_seconds,
        "fixture": args.fixture,
        "mode": args.mode,
        "morph": args.morph,
        "base_freq": args.base_freq,
        "spread": args.spread,
        "saturation_threshold": args.saturation,
        "benchmark_controls": BASE_CONTROLS,
        "environment_policy":
            "all inherited TROPICAL_* variables removed before explicit controls",
        "measurement_boundary":
            "process_ns is one tropical_runtime_process_offline call: for jit the "
            "kernel computes the block; for metal the caller waits for the "
            "worker's exact next tile, so the wall is synchronous GPU production "
            "including per-dispatch submission cost",
    }


# ------------------------------------------------------------------ reporting

def print_table(rows: dict[str, list[dict[str, Any]]], threshold: float) -> None:
    print()
    print(f"{'N':>6}  {'backend':<7}  {'median':>10}  {'p99':>10}  "
          f"{'sat_med':>8}  {'sat_p99':>8}  {'ns/osc':>8}  {'emit':>8}")
    print("-" * 78)
    for backend in ("jit", "metal"):
        for point in sorted(rows.get(backend, []), key=lambda p: p["oscillators"]):
            n = point["oscillators"]
            if point["emit_timed_out"]:
                print(f"{n:>6}  {backend:<7}  {'emit timeout':>10}")
                continue
            if point.get("emit_failed"):
                print(f"{n:>6}  {backend:<7}  {'REFUSED':>10}  "
                      f"{point.get('emit_failure_detail', '')[:60]}")
                continue
            stats = point["process_summary_ns"]
            sat = point["saturation"]
            flag = "  <-- over" if sat["median"] and sat["median"] > threshold else ""
            print(f"{n:>6}  {backend:<7}  "
                  f"{stats['median']/1000:>9.1f}u  {stats['p99']/1000:>9.1f}u  "
                  f"{100*sat['median']:>7.2f}%  {100*sat['p99']:>7.2f}%  "
                  f"{point['per_oscillator_median_ns']:>8.1f}  "
                  f"{point['emit_ns']/1e9:>7.1f}s{flag}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["smoke", "full"], default="full")
    parser.add_argument("--counts", type=str, default=None,
                        help="comma-separated oscillator counts (overrides --suite)")
    parser.add_argument("--backend", choices=["jit", "metal", "both"],
                        default="both")
    parser.add_argument("--saturation", type=float, default=0.50,
                        help="saturation threshold defining capacity (default 0.50)")
    parser.add_argument("--statistic", choices=["median", "p95", "p99"],
                        default="median",
                        help="saturation statistic the capacity bracket uses")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--rate", type=int, default=44100)
    parser.add_argument("--blocks", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--emit-timeout", type=float, default=600.0,
                        help="per-count artifact emission budget in seconds")
    parser.add_argument("--fixture", choices=["patch", "graph"], default="patch",
                        help="patch: FixedSinOsc instances via tropical_program_2 "
                             "(reaches --mode, scales linearly). graph: playground "
                             "`source` nodes (heavier voice; see findings.md for "
                             "the compile-time pathology above ~32k instructions)")
    parser.add_argument("--mode", choices=["fused", "microkernel",
                                           "microkernel-deep"],
                        default="fused",
                        help="compilation mode; patch fixture only")
    parser.add_argument("--morph", type=float, default=0.0,
                        help="oscillator waveform morph (0 = sine); graph fixture only")
    parser.add_argument("--base-freq", type=float, default=55.0)
    parser.add_argument("--spread", type=float, default=0.017,
                        help="fractional frequency spread per voice; keeps "
                             "voices distinct under CSE")
    parser.add_argument("--out", type=str, default=None,
                        help="JSONL output path (default data/<tag>.jsonl)")
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    for required in (DIFFCLI, RUNTIME_BENCH):
        if not required.exists():
            print(f"missing {required}; run `make build` and `make lean`",
                  file=sys.stderr)
            return 2

    if args.counts:
        counts = [int(c) for c in args.counts.split(",") if c.strip()]
    else:
        counts = SMOKE_COUNTS if args.suite == "smoke" else DEFAULT_COUNTS
    backends = ["jit", "metal"] if args.backend == "both" else [args.backend]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = args.tag or f"{args.suite}-{stamp}"
    out_path = Path(args.out) if args.out else HERE / "data" / f"{tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    workdir = HERE / ".work" / tag
    cache_root = workdir / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    environment = environment_manifest(args)
    rows: dict[str, list[dict[str, Any]]] = {b: [] for b in backends}

    started = time.perf_counter_ns()
    sweep_error: str | None = None
    try:
      for backend in backends:
          for count in counts:
              print(f"[{backend}] N={count} ...", flush=True)
              point = sweep_point(count, backend, args, workdir, cache_root)
              rows[backend].append(point)
              if point["emit_timed_out"]:
                  print(f"[{backend}] N={count} emit exceeded "
                        f"{args.emit_timeout}s; skipping larger counts on this "
                        f"backend", flush=True)
                  break
              if point.get("emit_failed"):
                  print(f"[{backend}] N={count} REFUSED by backend: "
                        f"{point.get('emit_failure_detail', '')}; skipping larger "
                        f"counts on this backend", flush=True)
                  break
              sat = point["saturation"]["median"]
              print(f"[{backend}] N={count} saturation_median={100*sat:.2f}%",
                    flush=True)
    except BaseException as unexpected:            # noqa: BLE001 - evidence first
        # A long sweep is expensive; never discard measured points because a
        # later one failed in a way not yet modelled. Record and report.
        sweep_error = f"{type(unexpected).__name__}: {unexpected}"
        print(f"sweep aborted: {sweep_error}", file=sys.stderr)
    total_ns = time.perf_counter_ns() - started

    capacities = {
        backend: capacity(points, args.saturation, args.statistic)
        for backend, points in rows.items()
    }

    record = {
        "schema": ROW_SCHEMA,
        "tag": tag,
        "environment": environment,
        "counts_requested": counts,
        "backends": backends,
        "points": rows,
        "capacity": capacities,
        "crossover": crossover(rows, args.statistic),
        "total_wall_ns": total_ns,
        "sweep_error": sweep_error,
    }
    with out_path.open("w") as handle:
        handle.write(json.dumps(record) + "\n")

    print_table(rows, args.saturation)
    for backend, result in capacities.items():
        print(f"capacity[{backend}] @ {100*args.saturation:.0f}% "
              f"({args.statistic}): {json.dumps(result)}")
    print(f"crossover: {json.dumps(record['crossover'].get('status'))} "
          f"at N={record['crossover'].get('metal_cheaper_from')}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
