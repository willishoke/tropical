#!/usr/bin/env python3
"""Non-interactive Apple Metal latency and soak qualification."""

from __future__ import annotations

import argparse
import json
import math
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
LATENCY_PARAMS = {
    "raw": ("probe.raw", 0.25, 0.75),
    "glide": ("probe.glide", 0.25, 0.75),
    "anchor": ("probe.freq", 220.0, 440.0),
    "velocity": ("master.velocity", 1.0, 0.75),
}
BASE_CONTROLS: dict[str, str | None] = {
    "TROPICAL_STAGE0": "1",
    "TROPICAL_BANKS_UNROLL": None,
    "TROPICAL_JIT_OPT_LEVEL": "O2",
    "TROPICAL_KERNEL_CACHE_DISABLE": "0",
    "TROPICAL_BACKEND": None,
    "TROPICAL_BUFFER_LENGTH": None,
    "TROPICAL_METAL_PIPELINE": "0",
    "TROPICAL_METAL_PIPELINE_DEPTH": None,
    "TROPICAL_STAGE0_DUMP": None,
}


def benchmark_env(overrides: dict[str, str | None]) -> dict[str, str]:
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith("TROPICAL_")
    }
    controls = BASE_CONTROLS | overrides
    env.update({
        key: value for key, value in controls.items()
        if value is not None
    })
    return env


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


def summarize_callback_histogram(result: dict[str, Any]) -> dict[str, Any]:
    histogram = result.get("callback_histogram", {})
    counts = histogram.get("counts", [])
    width = histogram.get("bin_width_ns")
    overflow_floor = histogram.get("overflow_floor_ns")
    total = sum(counts)

    def quantile_upper_bound(q: float) -> int | None:
        if not counts or not width or total == 0:
            return None
        target = max(1, int(q * total + .999999) - 0)
        seen = 0
        for index, count in enumerate(counts):
            seen += count
            if seen >= target:
                if index == len(counts) - 1:
                    return overflow_floor
                return (index + 1) * width
        return None

    return {
        "count": total,
        "p50_upper_bound_ns": quantile_upper_bound(.50),
        "p95_upper_bound_ns": quantile_upper_bound(.95),
        "p99_upper_bound_ns": quantile_upper_bound(.99),
        "max_exact_ns": int(
            result.get("dac_stats", {}).get("max_callback_ms", 0) * 1e6
        ),
        "resolution_ns": width,
        "overflow_floor_ns": overflow_floor,
        "overflow_count": counts[-1] if counts else None,
    }


def analyze_rss(result: dict[str, Any], rate: int, buffer: int
                ) -> dict[str, Any]:
    event_blocks = result.get("event_blocks", {})
    hot_swap_progress = event_blocks.get("hot_swap_progress")
    settle_blocks = max(1, int(2 * rate / buffer))
    warmup_end = (
        hot_swap_progress + settle_blocks
        if isinstance(hot_swap_progress, int) else None
    )
    pairs = [
        (block, value)
        for block, value in zip(result.get("rss_blocks", []),
                                result.get("rss_bytes", []), strict=True)
        if warmup_end is None or block >= warmup_end
    ]
    values = [value for _, value in pairs]
    deltas = [right - left for left, right in zip(values, values[1:])]
    window = max(1, len(values) // 3)
    first_level = (
        int(statistics.median(values[:window])) if values else None)
    last_level = (
        int(statistics.median(values[-window:])) if values else None)
    net_growth = values[-1] - values[0] if len(values) >= 2 else None
    robust_level_growth = (
        last_level - first_level
        if first_level is not None and last_level is not None else None
    )
    material_allowance = (
        max(1 << 20, int(first_level * .01))
        if first_level is not None else None
    )
    material_growth = (
        material_allowance is not None
        and (
            (net_growth is not None and net_growth > material_allowance)
            or (
                robust_level_growth is not None
                and robust_level_growth > material_allowance
            )
        )
    )
    return {
        "sample_count": len(values),
        "warmup_end_block": warmup_end,
        "warmup_contract":
            "two seconds after observed hot-swap progress",
        "first_block": pairs[0][0] if pairs else None,
        "last_block": pairs[-1][0] if pairs else None,
        "growth_bytes": net_growth,
        "first_level_bytes": first_level,
        "last_level_bytes": last_level,
        "robust_level_growth_bytes": robust_level_growth,
        "material_growth_allowance_bytes": material_allowance,
        "increase_intervals": sum(delta > 0 for delta in deltas),
        "flat_intervals": sum(delta == 0 for delta in deltas),
        "decrease_intervals": sum(delta < 0 for delta in deltas),
        "material_net_or_level_growth": material_growth,
    }


def evaluate_acceptance(result: dict[str, Any],
                        rss_analysis: dict[str, Any],
                        buffer: int, rate: int, depth: int,
                        canary_amplitude_bound: float = 1e-12,
                        ) -> tuple[dict[str, bool], list[str]]:
    expected_labels = {
        "start",
        "post_2p40_clock_jump",
        "midpoint_after_hot_swap",
        "end",
    }
    labels = result.get("reference_labels", [])
    snr = result.get("reference_snr_db", [])
    energies = result.get("reference_signal_energy", [])
    callback = result.get("callback_summary_ns", {})
    p99 = callback.get("p99_upper_bound_ns")
    deadline_ns = buffer * 1e9 / rate
    event_completion = result.get("event_completion", {})
    histogram_count = callback.get("count")
    callback_count = result.get("dac_stats", {}).get("callback_count")
    event_blocks = result.get("event_blocks", {})
    start_block = event_blocks.get("start_reference")
    jump_request_block = event_blocks.get("clock_jump_request")
    jump_progress_block = event_blocks.get("clock_jump_progress")
    swap_publication_block = event_blocks.get("hot_swap_publication")
    swap_progress_block = event_blocks.get("hot_swap_progress")
    post_jump_block = event_blocks.get("post_jump_reference")
    midpoint_block = event_blocks.get("midpoint_reference")
    end_block = event_blocks.get("end_reference")
    duration_elapsed_block = event_blocks.get("duration_elapsed")
    end_preceding_write = event_blocks.get("end_preceding_write")
    last_write = event_blocks.get("last_write")
    writes_stopped = event_blocks.get("writes_stopped")
    measured_loop_ns = result.get("measured_loop_ns")
    measured_loop_callbacks = result.get("measured_loop_callback_count")
    expected_callbacks = (
        measured_loop_ns / (buffer * 1e9 / rate)
        if isinstance(measured_loop_ns, (int, float))
        and measured_loop_ns > 0 else None
    )
    callback_coverage = (
        measured_loop_callbacks / expected_callbacks
        if isinstance(measured_loop_callbacks, int)
        and expected_callbacks else None
    )
    bank_energy_fraction_lower_bounds = []
    for energy in energies:
        if not isinstance(energy, (int, float)) or energy <= 0:
            bank_energy_fraction_lower_bounds.append(0.0)
            continue
        total_rms = math.sqrt(energy / buffer)
        bank_rms_lower = max(0.0, total_rms - canary_amplitude_bound)
        bank_energy_fraction_lower_bounds.append(
            (bank_rms_lower / total_rms) ** 2 if total_rms else 0.0)
    applied_disciplines = {
        event.get("discipline")
        for event in result.get("param_events", [])
        if isinstance(event, dict)
    }

    gates = {
        "runtime_not_aborted": not result.get("dac_aborted", True),
        "negotiated_frames_match":
            result.get("negotiated_buffer_frames") == buffer,
        "observed_pipeline_depth_matches_requested":
            result.get("pipeline_depth") == depth,
        "zero_underruns":
            result.get("dac_stats", {}).get("underrun_count") == 0,
        "zero_callback_overruns":
            result.get("dac_stats", {}).get("overrun_count") == 0,
        "callback_p99_below_half_deadline":
            p99 is not None and p99 < deadline_ns / 2,
        "histogram_covers_all_measured_callbacks":
            histogram_count is not None and histogram_count == callback_count,
        "callback_rate_covers_measured_wall":
            callback_coverage is not None
            and .99 <= callback_coverage <= 1.01,
        "zero_sticky_device_continuity_events":
            result.get("disconnect_count") == 0
            and result.get("reconnect_success_count") == 0
            and result.get("reconnect_failure_count") == 0,
        "zero_runtime_ownership_failures":
            result.get("ownership_failure_count") == 0
            and result.get("reference_ownership_failure_count") == 0,
        "all_required_events_completed":
            all(event_completion.get(name, False) for name in (
                "baseline", "writes", "writes_stopped", "clock_jump", "hot_swap",
                "start_reference", "post_jump_reference",
                "midpoint_reference", "end_reference",
            )),
        "reference_checkpoint_order_valid":
            all(isinstance(value, int) for value in (
                event_blocks.get("baseline"),
                event_blocks.get("clock_jump_progress"),
                event_blocks.get("hot_swap_progress"),
                start_block, post_jump_block, midpoint_block, end_block,
            ))
            and start_block > event_blocks["baseline"]
            and post_jump_block > event_blocks["clock_jump_progress"]
            and midpoint_block > event_blocks["hot_swap_progress"]
            and end_block > midpoint_block,
        "clock_and_swap_clear_future_queue":
            all(isinstance(value, int) for value in (
                jump_request_block, jump_progress_block,
                swap_publication_block, swap_progress_block,
            ))
            and jump_progress_block >= jump_request_block + depth + 1
            and swap_progress_block >= swap_publication_block + depth + 1,
        "final_reference_after_write_stop_and_clear_of_future_queue":
            isinstance(end_block, int)
            and isinstance(last_write, int)
            and isinstance(writes_stopped, int)
            and isinstance(duration_elapsed_block, int)
            and end_preceding_write == last_write
            and last_write < writes_stopped <= end_block
            and end_block > duration_elapsed_block
            and end_block >= writes_stopped + depth + 1
            and end_block >= last_write + depth + 1,
        "callback_progress_not_stalled":
            not event_completion.get("callback_progress_stalled", True),
        "all_reference_checkpoints_present":
            set(labels) == expected_labels and len(snr) == len(expected_labels),
        "all_reference_snr_above_100db":
            len(snr) == len(expected_labels)
            and all(
                isinstance(value, (int, float))
                and math.isfinite(value)
                and value > 100.0
                for value in snr
            ),
        "bank_dominates_canary_at_every_checkpoint":
            len(bank_energy_fraction_lower_bounds) == len(expected_labels)
            and all(
                fraction >= .99
                for fraction in bank_energy_fraction_lower_bounds
            ),
        "replacement_ir_msl_distinct_and_midpoint_checked":
            result.get("reload_artifacts_distinct") is True
            and "midpoint_after_hot_swap" in labels,
        "all_production_param_disciplines_exercised":
            applied_disciplines == {"raw", "glide", "anchor", "velocity"},
        "rss_sampling_sufficient":
            rss_analysis.get("sample_count", 0) >= 3,
        "no_material_post_warmup_rss_growth":
            not rss_analysis["material_net_or_level_growth"],
    }
    return gates, [name for name, passed in gates.items() if not passed]


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
        "benchmark_controls": BASE_CONTROLS,
        "environment_policy":
            "all inherited TROPICAL_* variables removed before explicit controls",
    }


def probe(args: list[str], env: dict[str, str]) -> dict[str, Any]:
    raw, wall = run(args, env)
    result = json.loads(raw)
    result["wall_ns"] = wall
    result["process_summary_ns"] = summarize(result["process_ns"])
    result["write_summary_ns"] = summarize(result["write_ns"])
    result["callback_summary_ns"] = summarize_callback_histogram(result)
    return result


def write_failure_row(stream, *, mode: str, requested_mode: str,
                      duration: float, buffer: int, depth: int,
                      error: Exception) -> None:
    stream.write(json.dumps({
        "schema": "tropical_metal_qualification_failure_1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "requested_mode": requested_mode,
        "duration_requested_seconds": duration,
        "buffer_length": buffer,
        "pipeline_depth": depth,
        "exception_class": type(error).__name__,
        "exception_message": str(error),
        "qualification_status": "blocked",
    }, separators=(",", ":")) + "\n")
    stream.flush()


def latency_args(buffer: int, metal: bool, discipline: str) -> list[str]:
    name, low, high = LATENCY_PARAMS[discipline]
    args = [
        str(RUNTIME_BENCH),
        "--ir", str(LATENCY / "latency.ll"),
        "--manifest", str(LATENCY / f"latency.{discipline}.manifest.json"),
        "--buffer", str(buffer),
        "--blocks", "8",
        "--warmup", "2",
        "--param-event", f"{discipline},{name},{low},{high}",
        "--latency",
        "--rss-every", "1",
    ]
    if metal:
        args += ["--msl", str(LATENCY / "latency.metal")]
    return args


def run_latency_matrix(stream, work: Path) -> None:
    rate = 44100
    for buffer in (128, 256, 512):
        for discipline in LATENCY_PARAMS:
            jit_env = benchmark_env({
                "TROPICAL_KERNEL_CACHE_ROOT":
                    str(work / "cache" / f"jit-{buffer}-{discipline}"),
            })
            jit = probe(latency_args(buffer, False, discipline), jit_env)
            if jit.get("ownership_failure_count") != 0:
                raise RuntimeError(
                    f"JIT ownership failure at B={buffer} {discipline}")
            if jit["latency_blocks"] != 0:
                raise RuntimeError(
                    f"JIT latency canary at B={buffer} {discipline} "
                    "was not next-block")
            stream.write(json.dumps({
                "schema": "tropical_metal_latency_row_1",
                "backend": "jit",
                "buffer_length": buffer,
                "pipeline_depth": 0,
                "discipline": discipline,
                "dispatch": jit["param_events"],
                "predicted_seconds": 0,
                "observed_seconds": 0,
                "raw": jit,
            }, separators=(",", ":")) + "\n")
            stream.flush()

            for depth in (1, 2, 3):
                env = benchmark_env({
                    "TROPICAL_METAL_PIPELINE_DEPTH": str(depth),
                    "TROPICAL_KERNEL_CACHE_ROOT": str(
                        work / "cache" / f"metal-{buffer}-{depth}-{discipline}"
                    ),
                })
                result = probe(latency_args(buffer, True, discipline), env)
                observed = result["latency_blocks"]
                if result.get("ownership_failure_count") != 0:
                    raise RuntimeError(
                        f"Metal ownership failure at B={buffer} D={depth} "
                        f"{discipline}")
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
                    "dispatch": result["param_events"],
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
    bank = dict(fixture["graph"]["nodes"][0])
    bank["in"] = {"addr": ["probe_addr"]}
    heavy_bank = {
        "nodes": [
            bank,
            {"id": "probe_addr", "kind": "knob", "params": {"value": 0.05}},
            {"id": "bank_trim", "kind": "knob", "params": {"value": 1e-8}},
            {
                "id": "quiet_bank",
                "kind": "ring",
                "in": {"in": ["bank", "bank_trim"]},
            },
            {
                "id": "canary",
                "kind": "source",
                "params": {"freq": 55, "morph": 0},
            },
            {"id": "canary_trim", "kind": "knob", "params": {"value": 1e-12}},
            {
                "id": "quiet_canary",
                "kind": "ring",
                "in": {"in": ["canary", "canary_trim"]},
            },
            {
                "id": "mix",
                "kind": "mix",
                "in": {"in": ["quiet_bank", "quiet_canary"]},
            },
            {"id": "out", "kind": "out", "in": {"in": ["mix"]}},
        ],
        "out": "out",
    }

    replacement = json.loads(json.dumps(heavy_bank))
    replacement_bank = next(
        node for node in replacement["nodes"] if node["id"] == "bank")
    replacement_bank["params"]["partials"] = 511
    next(node for node in replacement["nodes"]
         if node["id"] == "probe_addr")["params"]["value"] = 0.075

    def emit(name: str, graph_value: dict[str, Any]) -> dict[str, Path]:
        artifact = work / name
        artifact.mkdir(parents=True)
        graph = artifact / "graph.json"
        graph.write_text(json.dumps(graph_value, separators=(",", ":")))
        env = benchmark_env({
            "TROPICAL_STAGE0_DUMP": str(artifact),
            "TROPICAL_KERNEL_CACHE_ROOT": str(work / f"{name}-prepare-cache"),
        })
        run([
            str(DIFFCLI), "render-graph", str(graph), "--metal",
            "--frames", "0", "--buffer", "512",
        ], env)
        return {
            "ir": artifact / "audio.ll",
            "coeff": artifact / "coeff.ll",
            "msl": artifact / "audio.metal",
            "manifest": artifact / "manifest.json",
        }

    artifacts = emit("modal-bank-512", heavy_bank)
    reload_artifacts = emit("modal-bank-511-replacement", replacement)
    if artifacts["ir"].read_bytes() == reload_artifacts["ir"].read_bytes():
        raise RuntimeError("replacement LLVM IR is not distinguishable")
    if artifacts["msl"].read_bytes() == reload_artifacts["msl"].read_bytes():
        raise RuntimeError("replacement MSL is not distinguishable")
    if (
        artifacts["coeff"].read_bytes()
        == reload_artifacts["coeff"].read_bytes()
    ):
        raise RuntimeError("replacement coefficient IR is not distinguishable")
    if (
        artifacts["manifest"].read_bytes()
        == reload_artifacts["manifest"].read_bytes()
    ):
        raise RuntimeError("replacement manifest is not distinguishable")

    plan = json.loads(artifacts["manifest"].read_text())
    sink = plan.get("sinks", [{}])[0]
    reference_slot = sink.get("inputs", [None])[0]
    expected = {
        "raw": "bank.freq",
        "glide": "canary.morph",
        "anchor": "canary.freq",
        "velocity": "master.velocity",
    }
    applied = {
        item["discipline"]: item["name"]
        for item in plan.get("param_disciplines", [])
        if item.get("name") in expected.values()
    }
    replacement_plan = json.loads(reload_artifacts["manifest"].read_text())
    replacement_sink = replacement_plan.get("sinks", [{}])[0]
    replacement_reference_slot = replacement_sink.get("inputs", [None])[0]
    replacement_applied = {
        item["discipline"]: item["name"]
        for item in replacement_plan.get("param_disciplines", [])
        if item.get("name") in expected.values()
    }
    if (
        reference_slot is None
        or applied != expected
        or replacement_reference_slot != reference_slot
        or replacement_applied != expected
    ):
        raise RuntimeError(
            "heavy graph discipline/reference preflight failed: "
            f"initial={applied}/{reference_slot}, "
            f"replacement={replacement_applied}/{replacement_reference_slot}")
    return artifacts | {
        "reload_ir": reload_artifacts["ir"],
        "reload_coeff": reload_artifacts["coeff"],
        "reload_msl": reload_artifacts["msl"],
        "reload_manifest": reload_artifacts["manifest"],
    }, {
        "reference_slot": reference_slot,
        "param_events": [
            ("raw", "bank.freq", 180.0, 260.0),
            ("glide", "canary.morph", 0.0, 0.5),
            ("anchor", "canary.freq", 55.0, 65.0),
            ("velocity", "master.velocity", 1.0, 0.75),
        ],
        "canary_amplitude_bound": 1e-12,
    }


def run_soak(stream, work: Path, duration: float, buffer: int, depth: int) -> None:
    artifacts, slots = prepare_heavy_graph(work)
    rate = 44100
    expected_blocks = max(1, int(duration * rate / buffer))
    rss_every = max(1, int(2 * rate / buffer))
    common = [
        str(RUNTIME_BENCH),
        "--ir", str(artifacts["ir"]),
        "--coeff", str(artifacts["coeff"]),
        "--msl", str(artifacts["msl"]),
        "--manifest", str(artifacts["manifest"]),
        "--reload-ir", str(artifacts["reload_ir"]),
        "--reload-coeff", str(artifacts["reload_coeff"]),
        "--reload-msl", str(artifacts["reload_msl"]),
        "--reload-manifest", str(artifacts["reload_manifest"]),
        "--buffer", str(buffer),
        "--duration-seconds", str(duration),
        "--realtime",
        "--warmup", "8",
        "--reference-slot", str(slots["reference_slot"]),
        "--write-every", str(max(1, int(rate / buffer))),
        "--write-stop-at", str(expected_blocks * 3 // 4),
        "--rss-every", str(rss_every),
        "--jump-at", str(expected_blocks // 4),
        "--jump-index", str(1 << 40),
        "--reload-at", str(expected_blocks // 2),
    ]
    for discipline, name, low, high in slots["param_events"]:
        common += [
            "--param-event",
            f"{discipline},{name},{low},{high}",
        ]
    env = benchmark_env({
        "TROPICAL_METAL_PIPELINE_DEPTH": str(depth),
        "TROPICAL_KERNEL_CACHE_ROOT": str(work / "soak-cache"),
    })
    offline_args = [
        str(RUNTIME_BENCH),
        "--ir", str(artifacts["ir"]),
        "--coeff", str(artifacts["coeff"]),
        "--msl", str(artifacts["msl"]),
        "--manifest", str(artifacts["manifest"]),
        "--buffer", str(buffer),
        "--blocks", "1000",
        "--warmup", "8",
        "--rss-every", "100",
    ]
    offline = probe(offline_args, env)
    result = probe(common + ["--dac"], env)
    rss_analysis = analyze_rss(result, rate, buffer)
    acceptance_gates, failure_reasons = evaluate_acceptance(
        result, rss_analysis, buffer, rate, depth,
        slots["canary_amplitude_bound"])
    bank_fraction_lower_bounds = []
    for energy in result.get("reference_signal_energy", []):
        total_rms = math.sqrt(max(0.0, energy) / buffer)
        bank_lower = max(
            0.0, total_rms - slots["canary_amplitude_bound"])
        bank_fraction_lower_bounds.append(
            (bank_lower / total_rms) ** 2 if total_rms else 0.0)
    expected_callback_count = (
        result["measured_loop_ns"] / (buffer * 1e9 / rate)
        if result.get("measured_loop_ns") else None)
    callback_coverage = (
        result["measured_loop_callback_count"] / expected_callback_count
        if expected_callback_count else None)
    measured_snapshots = [
        item for item in result["dac_snapshots"]
        if item["label"] not in {"startup_warmup_before_reset", "after_stop"}
    ]
    post_reset_underruns = max(
        (item["underrun_count"] for item in measured_snapshots), default=0
    )
    row = {
        "schema": "tropical_metal_soak_row_2",
        "fixture": "modal-bank-512",
        "duration_requested_seconds": duration,
        "buffer_length": buffer,
        "pipeline_depth": depth,
        "hot_swap_block": expected_blocks // 2,
        "clock_jump_block": expected_blocks // 4,
        "rss_post_warmup": rss_analysis,
        "sink_contract":
            "the addressed 512-mode bank is the reference signal behind a "
            "1e-8 trim; the 55 Hz clock canary is bounded at 1e-12. "
            "Every checkpoint gates a conservative >=99% bank-energy lower "
            "bound and >100 dB Metal/JIT SNR, so the canary cannot mask the bank",
        "bank_energy_fraction_lower_bounds":
            bank_fraction_lower_bounds,
        "callback_coverage": {
            "expected_from_measured_wall": expected_callback_count,
            "observed": result.get("measured_loop_callback_count"),
            "ratio": callback_coverage,
            "acceptance_interval": [0.99, 1.01],
        },
        "dac_stats": result["dac_stats"],
        "post_reset_underruns": post_reset_underruns,
        "acceptance_gates": acceptance_gates,
        "failure_reasons": failure_reasons,
        "unavailable_measurements": [
            "pipeline queue depth over time (configured depth only)",
            "Metal resource/object counts",
            "GPU execution and re-prime duration",
            "live inter-callback/block interval distribution",
        ],
        "qualification_status":
            "blocked" if failure_reasons else "measured-window-pass",
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
            stream.flush()
            active_mode = options.mode
            try:
                if options.mode in {"latency", "all"}:
                    active_mode = "latency"
                    run_latency_matrix(stream, work)
                if options.mode in {"soak", "all"}:
                    active_mode = "soak"
                    run_soak(stream, work, options.duration_seconds,
                             options.buffer, options.depth)
            except Exception as error:
                write_failure_row(
                    stream,
                    mode=active_mode,
                    requested_mode=options.mode,
                    duration=options.duration_seconds,
                    buffer=options.buffer,
                    depth=options.depth,
                    error=error,
                )
                raise
    finally:
        shutil.rmtree(work)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
