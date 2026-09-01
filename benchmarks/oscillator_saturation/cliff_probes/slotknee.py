#!/usr/bin/env python3
"""Vary slots-per-voice; locate the knee; test knee*slots_per_voice == 4095."""
import json, os, subprocess, sys, statistics, time, tempfile, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DIFFCLI = ROOT / "lean/.lake/build/bin/diffcli"
BENCH = ROOT / "build/tropical_runtime_bench"
WORK = Path(__file__).resolve().parents[1] / ".work" / "cliff" / "slotknee"

BASE = {"TROPICAL_STAGE0": "1", "TROPICAL_JIT_OPT_LEVEL": "O2",
        "TROPICAL_KERNEL_CACHE_DISABLE": "0"}

def env(extra):
    e = {k: v for k, v in os.environ.items() if not k.startswith("TROPICAL_")}
    e.update(BASE); e.update(extra); return e

def build_patch(count, stages, base_freq=55.0, spread=0.017):
    decls = []
    for i in range(count):
        decls.append({"op": "instanceDecl", "name": f"osc{i}",
                      "program": "FixedSinOsc",
                      "inputs": {"freq": base_freq * (1.0 + spread * i)}})
        src = {"op": "ref", "instance": f"osc{i}", "output": "sine"}
        for s in range(stages - 1):
            decls.append({"op": "instanceDecl", "name": f"shp{i}_{s}",
                          "program": "SoftClip",
                          "inputs": {"input": src, "drive": 1 + 0.001 * i}})
            src = {"op": "ref", "instance": f"shp{i}_{s}", "output": "out"}
    def leaf(i):
        return ({"op": "ref", "instance": f"shp{i}_{stages-2}", "output": "out"}
                if stages > 1 else
                {"op": "ref", "instance": f"osc{i}", "output": "sine"})
    expr = leaf(0)
    for i in range(1, count):
        expr = {"op": "add", "args": [expr, leaf(i)]}
    decls.append({"op": "instanceDecl", "name": "sink", "program": "SoftClip",
                  "inputs": {"input": expr, "drive": 1}})
    return {"schema": "tropical_program_2", "name": f"slotknee_{stages}_{count}",
            "body": {"op": "block", "decls": decls,
                     "assigns": [{"op": "outputAssign", "name": "dac.out",
                                  "expr": {"op": "ref", "instance": "sink",
                                           "output": "out"}}]}}

def run(count, stages, buffer=512, blocks=300, warmup=32, rate=44100,
        measure=True, settle=1.0):
    d = WORK / f"s{stages}" / f"n{count}"
    d.mkdir(parents=True, exist_ok=True)
    cache = WORK / "cache"; cache.mkdir(parents=True, exist_ok=True)
    e = env({"TROPICAL_STAGE0_DUMP": str(d), "TROPICAL_KERNEL_CACHE_ROOT": str(cache)})
    src = d / "patch.json"
    src.write_text(json.dumps(build_patch(count, stages), separators=(",", ":")))
    man = d / "manifest.json"
    plan = subprocess.run([str(DIFFCLI), "compile", str(src), "--mode=fused"],
                          cwd=ROOT, env=e, capture_output=True, check=True).stdout
    man.write_bytes(plan)
    subprocess.run([str(DIFFCLI), "render-bytes", str(man), "--frames", "0",
                    "--buffer", str(buffer)], cwd=ROOT, env=e,
                   capture_output=True, check=True)
    slot_count = json.loads(plan)["slot_count"]
    out = {"n": count, "stages": stages, "slot_count": slot_count,
           "text_bytes": (d / "audio.ll").stat().st_size}
    if not measure:
        return out
    time.sleep(settle)
    args = [str(BENCH), "--ir", str(d / "audio.ll"), "--manifest", str(man),
            "--buffer", str(buffer), "--blocks", str(blocks),
            "--warmup", str(warmup), "--rate", str(rate)]
    coeff = d / "coeff.ll"
    if coeff.exists() and coeff.stat().st_size:
        args += ["--coeff", str(coeff)]
    probe = json.loads(subprocess.run(
        args, cwd=ROOT, env=env({"TROPICAL_KERNEL_CACHE_ROOT": str(cache)}),
        capture_output=True, check=True).stdout)
    med = statistics.median(probe["process_ns"])
    out["median_ns"] = med
    out["ns_per_voice"] = med / count
    return out

if __name__ == "__main__":
    stages = int(sys.argv[1])
    counts = [int(c) for c in sys.argv[2].split(",")]
    measure = "--no-measure" not in sys.argv
    for c in counts:
        r = run(c, stages, measure=measure)
        spv = (r["slot_count"] - 3) / c
        print(json.dumps({**r, "slots_per_voice": spv,
                          "predicted_knee": 4095 / spv if spv else None}),
              flush=True)
