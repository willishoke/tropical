#!/usr/bin/env python3
"""PREFLIGHT: is the marginal cost of tropical's filter zero at AUDIO rate?

`design/modal-filter-comparison-handoff.local.md` proposes a full fixture
(P x M x sweep-rate, plus a quality-vs-sweep oracle). Its load-bearing
assumption is that a `filter` node composes into modes already being
evaluated rather than adding an audio-rate stage -- i.e. that the pole pair
from `filterPair` is coefficient-time work.

That assumption is cheap to test on its own, and everything expensive in the
fixture depends on it, so test it first.

The stage-0 split makes this almost direct: `TROPICAL_STAGE0_DUMP` writes the
once-per-block coefficient kernel (`coeff.ll`) and the per-sample audio kernel
(`audio.ll`) as separate modules. So "does the filter cost anything per
sample" is literally "does audio.ll's loop body grow".

Reported per M, with and without the filter:
  * audio loop-body IR instructions -- what every sample pays
  * coeff.ll IR instructions        -- what every BLOCK pays
  * median ns/block                 -- the wall clock, as a check on the count

Prediction under test: d(audio) ~ 0 and d(coeff) > 0. If d(audio) instead
grows with M, the partial-fraction expansion is adding modes at audio rate,
the handoff's premise is wrong, and the full fixture needs redesigning before
it is built.
"""
import json, os, re, statistics, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DIFFCLI = ROOT / "lean/.lake/build/bin/diffcli"
BENCH = ROOT / "build/tropical_runtime_bench"
WORK = Path(__file__).resolve().parent / ".work"

BASE = {"TROPICAL_STAGE0": "1", "TROPICAL_JIT_OPT_LEVEL": "O2",
        "TROPICAL_KERNEL_CACHE_DISABLE": "0"}


def env(**extra):
    e = {k: v for k, v in os.environ.items() if not k.startswith("TROPICAL_")}
    e.update(BASE)
    e.update(extra)
    return e


STAGES = {
    # `filter` reads its cutoff/resonance through `paramValue` and closes over
    # the live slot reads.  `reverb` declares a control and receives a FROZEN
    # value (`build := fun frozenRt60 => …`).  Measuring both separates "modal
    # linear stages are expensive" from "this node bypasses the freezing path".
    "filter": {"kind": "filter", "params": {"cutoff": 800, "resonance": 0.5}},
    "reverb": {"kind": "reverb", "params": {"rt60": 2, "dir": 0}},
}


def graph(partials: int, stage: "str | None") -> dict:
    nodes = [{"id": "bank", "kind": "resonator",
              "params": {"freq": 220, "decay": 4, "partials": partials}}]
    sink = "bank"
    if stage:
        spec = STAGES[stage]
        nodes.append({"id": "stg", "kind": spec["kind"],
                      "params": spec["params"], "in": {"in": ["bank"]}})
        sink = "stg"
    nodes.append({"id": "out", "kind": "out", "in": {"in": [sink]}})
    return {"nodes": nodes, "out": "out"}


def ir_loop_body(path: Path) -> int:
    """IR instructions inside the per-sample loop, i.e. what a sample pays."""
    if not path.exists() or not path.stat().st_size:
        return 0
    lines = path.read_text().splitlines()
    try:
        lo = next(i for i, l in enumerate(lines) if l.startswith("loop_body:"))
        hi = next(i for i, l in enumerate(lines) if l.startswith("loop_end:"))
    except StopIteration:
        return 0
    return sum(1 for l in lines[lo + 1:hi] if re.match(r"^\s+%?\S+.*=|^\s+(store|br|call) ", l))


def ir_total(path: Path) -> int:
    if not path.exists() or not path.stat().st_size:
        return 0
    return sum(1 for l in path.read_text().splitlines()
               if re.match(r"^\s+%?\S+.*=|^\s+(store|br|call) ", l))


def run(partials: int, stage: "str | None", buffer=512, blocks=300, warmup=32):
    tag = f"m{partials}{stage or 'plain'}"
    d = WORK / tag
    d.mkdir(parents=True, exist_ok=True)
    cache = WORK / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    src = d / "graph.json"
    src.write_text(json.dumps(graph(partials, stage), separators=(",", ":")))
    e = env(TROPICAL_STAGE0_DUMP=str(d), TROPICAL_KERNEL_CACHE_ROOT=str(cache))
    p = subprocess.run([str(DIFFCLI), "render-graph", str(src), "--frames", "0",
                        "--buffer", str(buffer)], cwd=str(ROOT), env=e,
                       capture_output=True)
    if p.returncode != 0:
        raise SystemExit(f"render-graph ({tag}): {p.stderr.decode()[:400]}")
    man = json.loads((d / "manifest.json").read_text())
    args = [str(BENCH), "--ir", str(d / "audio.ll"),
            "--manifest", str(d / "manifest.json"), "--buffer", str(buffer),
            "--blocks", str(blocks), "--warmup", str(warmup), "--rate", "44100"]
    coeff = d / "coeff.ll"
    if coeff.exists() and coeff.stat().st_size:
        args += ["--coeff", str(coeff)]
    time.sleep(1.0)
    q = subprocess.run(args, cwd=str(ROOT),
                       env=env(TROPICAL_KERNEL_CACHE_ROOT=str(cache)),
                       capture_output=True)
    if q.returncode != 0:
        raise SystemExit(f"bench ({tag}): {q.stderr.decode()[:400]}")
    return {"audio": ir_loop_body(d / "audio.ll"),
            "coeff": ir_total(coeff),
            "slots": man["slot_count"],
            "ns": statistics.median(json.loads(q.stdout)["process_ns"])}


if __name__ == "__main__":
    counts = [int(c) for c in (sys.argv[1] if len(sys.argv) > 1 else "4,16,64,256").split(",")]
    WORK.mkdir(parents=True, exist_ok=True)
    print(f"{'M':>5} {'stage':>7} {'audio':>7} {'d_audio':>8} {'/part':>7} "
          f"{'coeff':>7} {'d_coeff':>8} {'ns':>10} {'d_ns%':>8}")
    for m in counts:
        base = run(m, None)
        print(f"{m:>5} {'-':>7} {base['audio']:>7} {'':>8} {'':>7} "
              f"{base['coeff']:>7} {'':>8} {base['ns']:>10.0f} {'':>8}")
        for stage in ("filter", "reverb"):
            r = run(m, stage)
            da = r["audio"] - base["audio"]
            print(f"{m:>5} {stage:>7} {r['audio']:>7} {da:>8} {da/m:>7.1f} "
                  f"{r['coeff']:>7} {r['coeff']-base['coeff']:>8} "
                  f"{r['ns']:>10.0f} {100.0*(r['ns']-base['ns'])/base['ns']:>7.1f}%",
                  flush=True)
