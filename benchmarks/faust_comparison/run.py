#!/usr/bin/env python3
"""tropical vs Faust — voices per core at a saturation threshold.

A naive "N sines" race between these two systems measures the wrong thing.
Faust idiomatically uses a phase ACCUMULATOR (cheap: one add and a wrap);
tropical computes phase CLOSED FORM from an absolute coordinate, because that
is what buys scrubbing and click-free hot-swap. Comparing only those two
prices tropical's design choice and reports it as a deficiency.

So three Faust variants run, separating two axes:

  F1  wavetable  + accumulator   what a Faust user actually ships
  F2  libm sin   + accumulator   recurrent, sine held constant against F3
  F3  libm sin   + counter       CLOSED FORM: tropical's semantics

  F2 vs F3  isolates recurrence-vs-closed-form, sine implementation fixed.
  F1 vs tropical  is the real-world number: both use a cheap sine (Faust a
                  wavetable, tropical a fixed-point Q31 polynomial).

Faust output is fully unrolled (N sin calls in one expression inside the
sample loop), matching tropical's unrolled kernel, so neither side is
secretly looping where the other is not.

Both sides are measured identically: same block, rate, warmup, isolated
process, median over N blocks, saturation = median block wall / deadline.
Faust is compiled by the SAME LLVM the tropical JIT uses, so codegen quality
is not a hidden variable.
"""
from __future__ import annotations
import argparse, json, os, platform, statistics, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FAUST = "/opt/homebrew/bin/faust"
CXX = "/opt/homebrew/opt/llvm/bin/clang++"
RUNTIME_BENCH = ROOT / "build" / "tropical_runtime_bench"
DIFFCLI = ROOT / "lean" / ".lake" / "build" / "bin" / "diffcli"
ARCH = HERE / "bench_arch.cpp"
GEN = HERE / "dsp" / "gen_dsp.py"
VARIANTS = ["f1_wavetable_recurrent", "f2_libmsin_recurrent", "f3_libmsin_closedform"]
SCHEMA = "tropical_vs_faust_1"


def sh(args, timeout=3600, env=None):
    t = time.perf_counter_ns()
    p = subprocess.run(args, capture_output=True, timeout=timeout, env=env, cwd=str(ROOT))
    return p, time.perf_counter_ns() - t


def clean_env(**over):
    e = {k: v for k, v in os.environ.items() if not k.startswith("TROPICAL_")}
    e.update({k: v for k, v in over.items() if v is not None})
    return e


def faust_point(n: int, variant: str, work: Path, a) -> dict[str, Any]:
    work.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(GEN), str(n), str(work)], check=True,
                   capture_output=True)
    dsp = work / f"{variant}_{n}.dsp"
    cpp = work / f"{variant}_{n}.cpp"
    exe = work / f"{variant}_{n}"
    precision = "-double" if a.precision == "double" else "-single"
    ff = "double" if a.precision == "double" else "float"
    out: dict[str, Any] = {"variant": variant, "oscillators": n,
                           "precision": a.precision}
    # Faust aborts its own compilation after `-t` seconds (default 120) and
    # exits on SIGALRM with an empty stderr, which reads as a mystery failure.
    # A 2048-term `process` expression exceeds that default, so raise it and
    # record Faust's own compile wall as data rather than losing the row.
    p, ns = sh([FAUST, precision, "-t", str(int(a.faust_timeout)),
                "-a", str(ARCH), "-o", str(cpp), str(dsp)])
    if p.returncode == -14:
        out["error"] = f"faust: SIGALRM (self-timeout at -t {int(a.faust_timeout)}s)"
        return out
    if p.returncode != 0:
        out["error"] = "faust: " + p.stderr.decode()[:200]; return out
    out["faust_compile_ns"] = ns
    p, ns = sh([CXX, "-O3", "-std=c++17", f"-DFAUSTFLOAT={ff}",
                "-o", str(exe), str(cpp)])
    if p.returncode != 0:
        out["error"] = "clang: " + p.stderr.decode()[:200]; return out
    out["cxx_compile_ns"] = ns
    out["cpp_lines"] = len(cpp.read_text().splitlines())
    time.sleep(a.settle_seconds)
    p, _ = sh([str(exe), "--buffer", str(a.buffer), "--blocks", str(a.blocks),
               "--warmup", str(a.warmup), "--rate", str(a.rate), "--label", variant])
    if p.returncode != 0:
        out["error"] = "run: " + p.stderr.decode()[:200]; return out
    out.update(json.loads(p.stdout.decode()))
    out["per_voice_ns"] = out["median_ns"] / n
    return out


def tropical_point(n: int, work: Path, a) -> dict[str, Any]:
    """Same fixture as benchmarks/oscillator_saturation: N FixedSinOsc summed."""
    d = work / f"tropical_{n}"; d.mkdir(parents=True, exist_ok=True)
    patch = d / "patch.json"
    decls = [{"op": "instanceDecl", "name": f"osc{i}", "program": "FixedSinOsc",
              "inputs": {"freq": 55.0 * (1.0 + 0.017 * i)}} for i in range(n)]
    # The sum is a LEFT fold, matching how C++ parses Faust's `a + b + c`, so
    # neither side gets a differently shaped reduction. Built as text rather
    # than nested dicts: a 2048-deep object exceeds the JSON encoder's
    # recursion limit, and raising that limit risks the C stack instead.
    def ref(i: int) -> str:
        return f'{{"op":"ref","instance":"osc{i}","output":"sine"}}'
    expr_json = ("{\"op\":\"add\",\"args\":[" * (n - 1)) + ref(0) + "".join(
        f",{ref(i)}]}}" for i in range(1, n)) if n > 1 else ref(0)
    decls_json = ",".join(json.dumps(d, separators=(",", ":")) for d in decls)
    sink_json = ('{"op":"instanceDecl","name":"sink","program":"SoftClip",'
                 f'"inputs":{{"input":{expr_json},"drive":1}}}}')
    patch.write_text(
        '{"schema":"tropical_program_2","name":"faustcmp' + str(n) + '",'
        '"body":{"op":"block","decls":[' + decls_json + "," + sink_json + '],'
        '"assigns":[{"op":"outputAssign","name":"dac.out",'
        '"expr":{"op":"ref","instance":"sink","output":"out"}}]}}')
    env = clean_env(TROPICAL_STAGE0_DUMP=str(d),
                    TROPICAL_KERNEL_CACHE_ROOT=str(d / "cache"))
    out: dict[str, Any] = {"variant": "tropical_jit", "oscillators": n,
                           "precision": "double"}
    p, ns = sh([str(DIFFCLI), "compile", str(patch), "--mode=fused"], env=env)
    if p.returncode != 0:
        out["error"] = "compile: " + p.stderr.decode()[:200]; return out
    (d / "manifest.json").write_bytes(p.stdout)
    p, ns2 = sh([str(DIFFCLI), "render-bytes", str(d / "manifest.json"),
                 "--frames", "0", "--buffer", str(a.buffer)], env=env)
    if p.returncode != 0:
        out["error"] = "emit: " + p.stderr.decode()[:200]; return out
    out["compile_ns"] = ns + ns2
    args = [str(RUNTIME_BENCH), "--ir", str(d / "audio.ll"),
            "--manifest", str(d / "manifest.json"), "--buffer", str(a.buffer),
            "--blocks", str(a.blocks), "--warmup", str(a.warmup),
            "--rate", str(a.rate)]
    if (d / "coeff.ll").exists() and (d / "coeff.ll").stat().st_size:
        args += ["--coeff", str(d / "coeff.ll")]
    time.sleep(a.settle_seconds)
    p, _ = sh(args, env=clean_env(TROPICAL_KERNEL_CACHE_ROOT=str(d / "cache")))
    if p.returncode != 0:
        out["error"] = "bench: " + p.stderr.decode()[:200]; return out
    r = json.loads(p.stdout.decode())
    ps = sorted(r["process_ns"])
    med = statistics.median(ps)
    out.update({"median_ns": med, "p95_ns": ps[int(.95 * len(ps))],
                "p99_ns": ps[int(.99 * len(ps))], "max_ns": max(ps),
                "deadline_ns": r["deadline_ns"],
                "saturation_median": med / r["deadline_ns"],
                "saturation_p99": ps[int(.99 * len(ps))] / r["deadline_ns"],
                "per_voice_ns": med / n})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--counts", default="1,4,16,64,256,512,1024,1536,2048")
    ap.add_argument("--precision", choices=["double", "single"], default="double")
    ap.add_argument("--buffer", type=int, default=512)
    ap.add_argument("--rate", type=int, default=44100)
    ap.add_argument("--blocks", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=32)
    ap.add_argument("--settle-seconds", type=float, default=1.5)
    ap.add_argument("--saturation", type=float, default=0.50)
    ap.add_argument("--faust-timeout", type=float, default=1800,
                    help="faust -t; its default 120s aborts on large exprs")
    ap.add_argument("--skip-tropical", action="store_true")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    for tool in (Path(FAUST), Path(CXX), RUNTIME_BENCH, DIFFCLI):
        if not tool.exists():
            print(f"missing {tool}", file=sys.stderr); return 2

    counts = [int(c) for c in a.counts.split(",") if c.strip()]
    tag = a.tag or f"{a.precision}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    work = HERE / ".work" / tag
    rows: list[dict[str, Any]] = []
    for n in counts:
        for v in VARIANTS:
            r = faust_point(n, v, work, a)
            rows.append(r)
            print(f"[{v}] N={n}: " + (r["error"] if "error" in r else
                  f"sat={100*r['saturation_median']:.2f}% per_voice={r['per_voice_ns']:.0f}ns"),
                  flush=True)
        if not a.skip_tropical:
            r = tropical_point(n, work, a)
            rows.append(r)
            print(f"[tropical] N={n}: " + (r["error"] if "error" in r else
                  f"sat={100*r['saturation_median']:.2f}% per_voice={r['per_voice_ns']:.0f}ns"),
                  flush=True)

    out = HERE / "data" / f"{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema": SCHEMA, "tag": tag,
        "environment": {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                                     capture_output=True, text=True).stdout.strip(),
            "platform": platform.platform(), "machine": platform.machine(),
            "faust": subprocess.run([FAUST, "--version"], capture_output=True,
                                    text=True).stdout.strip().splitlines()[0:1],
            "cxx": subprocess.run([CXX, "--version"], capture_output=True,
                                  text=True).stdout.strip().splitlines()[0:1],
            "precision": a.precision, "buffer": a.buffer, "rate": a.rate,
            "blocks": a.blocks, "warmup": a.warmup,
            "note": "faust compiled by the same LLVM the tropical JIT uses",
        },
        "points": rows}) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
