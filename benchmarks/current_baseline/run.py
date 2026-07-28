#!/usr/bin/env python3
"""Current-main compiler/runtime baseline.

The orchestrator intentionally measures subprocess walls and an artifact-only
native probe rather than changing production schemas. Every row retains raw
samples and labels inseparable frontend walls as coarse.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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
PRODUCT_GRAPH_CONTRACT = "TROPICAL_EXACT_PRODUCT_GRAPH_1"
PRODUCT_GRAPH_BEGIN = "// TROPICAL_EXACT_PRODUCT_GRAPH_BEGIN"
PRODUCT_GRAPH_END = "// TROPICAL_EXACT_PRODUCT_GRAPH_END"
PRODUCT_GRAPH_DECLARATION = "const GRAPH = "
BASELINE_MATRIX_SCHEMA = "tropical_baseline_matrix_2"
BASELINE_ROW_SCHEMA = "tropical_baseline_row_3"
ARTIFACT_PATH_KEYS = {
    "manifest": "manifest",
    "audio_ir": "ir",
    "coefficient_ir": "coeff",
    "msl": "msl",
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


def compile_summary(values: list[int]) -> dict[str, int | float | None]:
    result: dict[str, int | float | None] = summary(values)
    median = result["median"]
    result["range_over_median"] = (
        (max(values) - min(values)) / median
        if values and isinstance(median, int) and median > 0 else None
    )
    return result


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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def fixture_graph(fixture: dict[str, Any]) -> dict[str, Any]:
    inline = fixture.get("graph")
    if isinstance(inline, dict):
        return copy.deepcopy(inline)

    source = fixture.get("product_graph")
    if not isinstance(source, dict):
        raise RuntimeError(
            f"graph fixture {fixture.get('name')} has no graph source")
    if source.get("contract") != PRODUCT_GRAPH_CONTRACT:
        raise RuntimeError(
            f"unsupported product graph contract: {source.get('contract')}")
    source_path = ROOT / str(source.get("path", ""))
    text = source_path.read_text()
    if text.count(PRODUCT_GRAPH_BEGIN) != 1 or text.count(PRODUCT_GRAPH_END) != 1:
        raise RuntimeError(
            f"{source_path} must contain one exact-product graph marker pair")
    begin = text.index(PRODUCT_GRAPH_BEGIN) + len(PRODUCT_GRAPH_BEGIN)
    end = text.index(PRODUCT_GRAPH_END, begin)
    declaration = text[begin:end].strip()
    if not declaration.startswith(PRODUCT_GRAPH_DECLARATION):
        raise RuntimeError(
            f"{source_path} exact-product region must declare const GRAPH")
    encoded = declaration[len(PRODUCT_GRAPH_DECLARATION):].strip()
    try:
        graph = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"{source_path} exact-product GRAPH must remain strict JSON: "
            f"{error}") from error
    if not isinstance(graph, dict):
        raise RuntimeError(f"{source_path} exact-product GRAPH is not an object")
    return graph


def fixture_provenance(fixture: dict[str, Any]) -> dict[str, Any]:
    graph = fixture_graph(fixture)
    graph_bytes = json.dumps(
        graph, sort_keys=True, separators=(",", ":")).encode()
    source = fixture.get("product_graph")
    if isinstance(source, dict):
        source_path = ROOT / str(source["path"])
        return {
            "kind": "renderer_exact_product_graph",
            "path": str(source["path"]),
            "contract": source["contract"],
            "source_file_sha256": sha256_bytes(source_path.read_bytes()),
            "normalized_graph_sha256": sha256_bytes(graph_bytes),
        }
    return {
        "kind": "inline_matrix_graph",
        "path": str(MATRIX.relative_to(ROOT)),
        "normalized_graph_sha256": sha256_bytes(graph_bytes),
    }


def structural_edit_provenance(
        source_fixture: dict[str, Any],
        edited_fixture: dict[str, Any]) -> dict[str, Any]:
    source = fixture_provenance(source_fixture)
    graph = fixture_graph(edited_fixture)
    graph_bytes = json.dumps(
        graph, sort_keys=True, separators=(",", ":")).encode()
    return {
        "kind": "derived_structural_edit",
        "source_normalized_graph_sha256":
            source["normalized_graph_sha256"],
        "normalized_graph_sha256": sha256_bytes(graph_bytes),
        "normalized_graph": graph,
    }


def source_provenance(
        fixture: dict[str, Any], source: Path | None) -> dict[str, Any]:
    if fixture["class"] == "graph":
        return fixture_provenance(fixture)
    assert source is not None
    return {
        "kind": "program_file",
        "path": str(source.relative_to(ROOT)),
        "source_file_sha256": sha256_bytes(source.read_bytes()),
    }


def load_matrix() -> list[dict[str, Any]]:
    decoded = json.loads(MATRIX.read_text())
    if decoded.get("schema") != BASELINE_MATRIX_SCHEMA:
        raise RuntimeError(
            f"baseline matrix must use {BASELINE_MATRIX_SCHEMA}")
    fixtures = decoded.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        raise RuntimeError("baseline matrix fixtures must be a non-empty list")
    names = [fixture.get("name") for fixture in fixtures]
    if any(not isinstance(name, str) or not name for name in names):
        raise RuntimeError("every baseline fixture needs a name")
    if len(set(names)) != len(names):
        raise RuntimeError("baseline fixture names must be unique")
    flagship = [fixture for fixture in fixtures if fixture.get("flagship")]
    if len(flagship) != 1:
        raise RuntimeError("baseline matrix needs exactly one flagship")
    if not isinstance(flagship[0].get("product_graph"), dict):
        raise RuntimeError("flagship must consume the exact product graph")
    fixture_graph(flagship[0])
    return fixtures


def graph_size(fixture: dict[str, Any], source: Path | None) -> tuple[int, int]:
    if fixture["class"] == "graph":
        graph = fixture_graph(fixture)
        encoded = json.dumps(graph, separators=(",", ":")).encode()
        return len(graph.get("nodes", [])), len(encoded)
    assert source is not None
    parsed = json.loads(source.read_text())
    decls = parsed.get("body", {}).get("decls", [])
    return len(decls), source.stat().st_size


def cache_inventory() -> tuple[bool, dict[str, dict[str, Any]]]:
    ordinary = Path.home() / ".cache" / "tropical" / "kernels"
    if not ordinary.exists():
        return False, {}
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(ordinary.rglob("*")):
        if not path.is_file():
            continue
        try:
            value = path.read_bytes()
        except FileNotFoundError:
            continue
        inventory[str(path.relative_to(ordinary))] = {
            "bytes": len(value),
            "sha256": sha256_bytes(value),
        }
    return True, inventory


def cache_snapshot(
        exists: bool,
        inventory: dict[str, dict[str, Any]]) -> dict[str, Any]:
    encoded = json.dumps(
        inventory, sort_keys=True, separators=(",", ":")).encode()
    return {
        "exists": exists,
        "files": len(inventory),
        "bytes": sum(entry["bytes"] for entry in inventory.values()),
        "tree_sha256": sha256_bytes(encoded),
    }


def cache_changes(
        before: dict[str, dict[str, Any]],
        after: dict[str, dict[str, Any]]) -> dict[str, Any]:
    before_paths = set(before)
    after_paths = set(after)
    return {
        "added": [
            {"path": path, **after[path]}
            for path in sorted(after_paths - before_paths)
        ],
        "removed": [
            {"path": path, **before[path]}
            for path in sorted(before_paths - after_paths)
        ],
        "modified": [
            {"path": path, "before": before[path], "after": after[path]}
            for path in sorted(before_paths & after_paths)
            if before[path] != after[path]
        ],
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


def environment_manifest(
        ordinary_cache_before: dict[str, Any]) -> dict[str, Any]:
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
        "benchmark_controls": BASE_CONTROLS,
        "environment_policy":
            "all inherited TROPICAL_* variables removed before explicit controls",
        "stage0": "enabled (explicit TROPICAL_STAGE0=1)",
        "banks_unroll": "banked (TROPICAL_BANKS_UNROLL explicitly absent)",
        "ordinary_cache_before": ordinary_cache_before,
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


def artifact_digests(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for label, path_key in ARTIFACT_PATH_KEYS.items():
        path = paths[path_key]
        if not path.exists():
            continue
        value = path.read_bytes()
        result[label] = {
            "sha256": sha256_bytes(value),
            "bytes": len(value),
        }
    return result


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
    env = benchmark_env({
        "TROPICAL_KERNEL_CACHE_ROOT": str(cache_root),
    })
    if extra_env:
        env.update(extra_env)
    stdout, wall_ns = command(args, env)
    result = json.loads(stdout)
    result["process_summary_ns"] = summary(result["process_ns"])
    result["write_summary_ns"] = summary(result["write_ns"])
    result["subprocess_wall_ns"] = wall_ns
    return result


def prepare_artifacts(fixture: dict[str, Any], directory: Path,
                      metal: bool, cache_root: Path
                      ) -> tuple[dict[str, Path], dict[str, int | None]]:
    directory.mkdir(parents=True, exist_ok=True)
    env = benchmark_env({
        "TROPICAL_STAGE0_DUMP": str(directory),
        "TROPICAL_KERNEL_CACHE_ROOT": str(cache_root),
    })
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
        graph_path.write_text(json.dumps(
            fixture_graph(fixture), separators=(",", ":")))
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


def generation_series(fixture: dict[str, Any], directory: Path,
                      metal: bool, repeats: int
                      ) -> tuple[dict[str, Path], dict[str, Any]]:
    cold_samples: list[dict[str, int | None]] = []
    warm_samples: list[dict[str, int | None]] = []
    cold_digests: list[dict[str, Any]] = []
    warm_digests: list[dict[str, Any]] = []
    representative: dict[str, Path] | None = None
    reference_digests: dict[str, dict[str, Any]] | None = None
    emitted_bytes_stable = True

    for repeat in range(repeats):
        repeat_root = directory / f"repeat-{repeat}"
        cache_root = repeat_root / "cache"
        cold_paths, cold = prepare_artifacts(
            fixture, repeat_root / "cold", metal, cache_root)
        warm_paths, warm = prepare_artifacts(
            fixture, repeat_root / "warm", metal, cache_root)
        cold_samples.append(cold)
        warm_samples.append(warm)
        cold_digest = artifact_digests(cold_paths)
        warm_digest = artifact_digests(warm_paths)
        cold_digests.append({
            "repeat": repeat,
            "artifacts": cold_digest,
        })
        warm_digests.append({
            "repeat": repeat,
            "artifacts": warm_digest,
        })
        representative = warm_paths

        for current in (cold_digest, warm_digest):
            if reference_digests is None:
                reference_digests = current
            elif current != reference_digests:
                emitted_bytes_stable = False

    assert representative is not None
    timing_keys = sorted({
        key for sample in cold_samples + warm_samples for key in sample
    })
    summaries: dict[str, Any] = {}
    for key in timing_keys:
        summaries[key] = {
            "cold": compile_summary([
                int(sample[key]) for sample in cold_samples
                if sample.get(key) is not None
            ]),
            "warm": compile_summary([
                int(sample[key]) for sample in warm_samples
                if sample.get(key) is not None
            ]),
        }
    return representative, {
        "repeat_count": repeats,
        "cold": cold_samples,
        "warm": warm_samples,
        "summary_ns": summaries,
        "emitted_bytes_stable": emitted_bytes_stable,
        "artifact_digests": {
            "algorithm": "sha256",
            "expected_artifacts": [
                "manifest", "audio_ir", "coefficient_ir",
                *(["msl"] if metal else []),
            ],
            "cold": cold_digests,
            "warm": warm_digests,
        },
        "cold_cache":
            "fresh benchmark-owned root for each repeat",
        "warm_cache":
            "second full generation reusing the corresponding cold root",
    }


def validate_generation_evidence(
        generation: dict[str, Any], repeats: int, metal: bool) -> None:
    digests = generation.get("artifact_digests", {})
    expected = {
        "manifest", "audio_ir", "coefficient_ir",
        *(["msl"] if metal else []),
    }
    if digests.get("algorithm") != "sha256":
        raise RuntimeError("generation evidence must use SHA-256")
    if set(digests.get("expected_artifacts", [])) != expected:
        raise RuntimeError("generation evidence artifact set mismatch")

    all_artifacts: list[dict[str, Any]] = []
    for temperature in ("cold", "warm"):
        samples = digests.get(temperature)
        if not isinstance(samples, list) or len(samples) != repeats:
            raise RuntimeError(
                f"generation evidence needs {repeats} {temperature} samples")
        for repeat, sample in enumerate(samples):
            if sample.get("repeat") != repeat:
                raise RuntimeError(
                    f"{temperature} digest repeat indices are not contiguous")
            artifacts = sample.get("artifacts")
            if not isinstance(artifacts, dict) or set(artifacts) != expected:
                raise RuntimeError(
                    f"{temperature} repeat {repeat} artifact digest mismatch")
            for value in artifacts.values():
                digest = value.get("sha256")
                byte_count = value.get("bytes")
                if (
                    not isinstance(digest, str) or len(digest) != 64
                    or any(ch not in "0123456789abcdef" for ch in digest)
                    or not isinstance(byte_count, int) or byte_count < 0
                ):
                    raise RuntimeError("malformed retained artifact digest")
            all_artifacts.append(artifacts)

    stable = all(
        artifacts == all_artifacts[0] for artifacts in all_artifacts[1:])
    if generation.get("emitted_bytes_stable") is not stable:
        raise RuntimeError(
            "emitted_bytes_stable disagrees with retained artifact digests")


def flagship_edits(fixture: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not fixture.get("flagship"):
        return {}

    topology = copy.deepcopy(fixture)
    topology.pop("product_graph", None)
    topology["graph"] = fixture_graph(fixture)
    nodes = topology["graph"]["nodes"]
    mix_index = next(i for i, node in enumerate(nodes) if node["id"] == "mx")
    nodes.insert(mix_index, {
        "id": "r5",
        "kind": "resonator",
        "params": {"freq": 440, "decay": 4},
        "sel": {},
        "in": {"addr": ["adr"]},
    })
    mix = next(node for node in nodes if node["id"] == "mx")
    mix["in"]["in"].append("r5")
    topology["name"] = f"{fixture['name']}-topology-add-addressed-ring"

    partials = copy.deepcopy(fixture)
    partials.pop("product_graph", None)
    partials["graph"] = fixture_graph(fixture)
    r1 = next(node for node in partials["graph"]["nodes"]
              if node["id"] == "r1")
    if "partials" in r1["params"]:
        raise RuntimeError(
            "exact-product r1 must exercise the renderer's default partials")
    r1["params"]["partials"] = 7
    partials["name"] = (
        f"{fixture['name']}-structural-default-partials-6-to-7")

    return {
        "topology_add_addressed_ring": topology,
        "structural_default_partials_6_to_7": partials,
    }


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
    if options.repeats < 1:
        raise RuntimeError("--repeats must be positive")

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
    ordinary_before_exists, ordinary_before = cache_inventory()
    environment = environment_manifest(
        cache_snapshot(ordinary_before_exists, ordinary_before))
    matrix = load_matrix()
    fixtures = [item for item in matrix if options.suite == "full" or item.get("smoke")]

    try:
        with output.open("w") as stream:
            stream.write(json.dumps(environment, separators=(",", ":")) + "\n")
            for fixture in fixtures:
                fixture_dir = work / "generation" / fixture["name"]
                paths, generation = generation_series(
                    fixture, fixture_dir, options.metal, options.repeats)
                validate_generation_evidence(
                    generation, options.repeats, options.metal)
                structural_edits: dict[str, Any] = {}
                for edit_name, edited_fixture in flagship_edits(fixture).items():
                    _, edit_generation = generation_series(
                        edited_fixture, work / "edits" / edit_name,
                        options.metal, options.repeats)
                    validate_generation_evidence(
                        edit_generation, options.repeats, options.metal)
                    structural_edits[edit_name] = {
                        "fixture_provenance": structural_edit_provenance(
                            fixture, edited_fixture),
                        **edit_generation,
                    }
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
                    ], benchmark_env({
                        "TROPICAL_KERNEL_CACHE_ROOT":
                            str(fixture_dir / "wasm-cache"),
                    }))
                    metrics["wasm_bytes"] = wasm.stat().st_size
                    generation["wasm_total_ns"] = wasm_ns

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
                    "schema": BASELINE_ROW_SCHEMA,
                    "fixture": fixture["name"],
                    "fixture_class": fixture["class"],
                    "fixture_provenance":
                        source_provenance(fixture, source),
                    "source_nodes": nodes,
                    "source_bytes": source_bytes,
                    "arena_node_count": None,
                    "bank_capacity": fixture.get("bank_capacity"),
                    "live_count": fixture.get("live_count"),
                    "nested": fixture.get("nested", False),
                    "flagship": fixture.get("flagship", False),
                    "metrics": metrics,
                    "generation": generation,
                    "structural_edits": structural_edits,
                    "runtime": runtime,
                    "cache_state": {
                        "root_env": "TROPICAL_KERNEL_CACHE_ROOT",
                        "cold": "fresh per-repeat benchmark-owned root",
                        "warm": "second process reusing that root",
                    },
                    "benchmark_controls": {
                        **BASE_CONTROLS,
                        "TROPICAL_KERNEL_CACHE_ROOT":
                            "<per-repeat harness-owned root>",
                    },
                    "limitations": [
                        "frontend plan/lower/emit walls are coarse subprocess totals",
                        "arena node count is not exposed without invasive compiler instrumentation",
                        "Metal load includes the mandatory dual JIT load",
                        "the flagship structural-capacity edit makes the "
                        "renderer-default six partials explicit as seven; "
                        "decoded graph sel fields remain inert",
                    ],
                }
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                stream.flush()
                print(f"recorded {fixture['name']}", file=sys.stderr)

            ordinary_after_exists, ordinary_after = cache_inventory()
            unchanged = (
                ordinary_before_exists == ordinary_after_exists
                and ordinary_before == ordinary_after
            )
            trailer = {
                "schema": "tropical_baseline_trailer_1",
                "ordinary_cache_after":
                    cache_snapshot(ordinary_after_exists, ordinary_after),
                "ordinary_cache_unchanged": unchanged,
                "ordinary_cache_changes":
                    cache_changes(ordinary_before, ordinary_after),
            }
            stream.write(json.dumps(trailer, separators=(",", ":")) + "\n")
    finally:
        shutil.rmtree(work)

    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
