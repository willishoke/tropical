#!/usr/bin/env python3
"""Does the banked (bankSum region) family show the slot-index cliff?"""
import json, os, subprocess, sys, statistics, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
DIFFCLI = ROOT / "lean/.lake/build/bin/diffcli"
BENCH = ROOT / "build/tropical_runtime_bench"
WORK = Path(__file__).resolve().parents[1] / ".work" / "cliff" / "bankknee"
BASE = {"TROPICAL_STAGE0": "1", "TROPICAL_JIT_OPT_LEVEL": "O2",
        "TROPICAL_KERNEL_CACHE_DISABLE": "0"}
def env(extra):
    e = {k: v for k, v in os.environ.items() if not k.startswith("TROPICAL_")}
    e.update(BASE); e.update(extra); return e
def run(partials, buffer=512, blocks=300, warmup=32, rate=44100, settle=1.0):
    d = WORK / f"p{partials}"; d.mkdir(parents=True, exist_ok=True)
    cache = WORK / "cache"; cache.mkdir(parents=True, exist_ok=True)
    e = env({"TROPICAL_STAGE0_DUMP": str(d), "TROPICAL_KERNEL_CACHE_ROOT": str(cache)})
    g = {"nodes": [
            {"id": "bank", "kind": "resonator",
             "params": {"freq": 220, "decay": 4, "partials": partials}},
            {"id": "out", "kind": "out", "in": {"in": ["bank"]}}],
         "out": "out"}
    src = d / "graph.json"; src.write_text(json.dumps(g, separators=(",", ":")))
    subprocess.run([str(DIFFCLI), "render-graph", str(src), "--frames", "0",
                    "--buffer", str(buffer)], cwd=ROOT, env=e,
                   capture_output=True, check=True)
    man = json.loads((d / "manifest.json").read_text())
    out = {"partials": partials, "slot_count": man["slot_count"],
           "array_slot_count": man.get("array_slot_count"),
           "ll_bytes": (d / "audio.ll").stat().st_size}
    time.sleep(settle)
    args = [str(BENCH), "--ir", str(d / "audio.ll"),
            "--manifest", str(d / "manifest.json"), "--buffer", str(buffer),
            "--blocks", str(blocks), "--warmup", str(warmup), "--rate", str(rate)]
    coeff = d / "coeff.ll"
    if coeff.exists() and coeff.stat().st_size:
        args += ["--coeff", str(coeff)]
    probe = json.loads(subprocess.run(args, cwd=ROOT,
        env=env({"TROPICAL_KERNEL_CACHE_ROOT": str(cache)}),
        capture_output=True, check=True).stdout)
    med = statistics.median(probe["process_ns"])
    out["median_ns"] = med; out["ns_per_partial"] = med / partials
    return out
if __name__ == "__main__":
    for p in [int(x) for x in sys.argv[1].split(",")]:
        print(json.dumps(run(p)), flush=True)
