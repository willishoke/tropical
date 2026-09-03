#!/usr/bin/env python3
"""Marginal per-voice machine-instruction count: tropical FixedSin vs Faust F3 libm sin.

Static instruction counts at a single N are contaminated by prologue, epilogue,
and the block loop.  Counting at two N and differencing cancels all of it:
what is left is exactly what one more voice costs to emit.

Faust side: compile F3 (libm sin, counter-driven -> closed form) exactly as
benchmarks/faust_comparison/run.py does, then otool the `compute` method.
tropical side: compile the same fixture, dump the post-IR-pipeline module the
codegen layer actually receives, and run llc at the SAME codegen opt level the
JIT now defaults to (None) so the object is the one that really runs.
"""
import json, os, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FAUST = "/opt/homebrew/bin/faust"
CXX = "/opt/homebrew/opt/llvm/bin/clang++"
LLC = "/opt/homebrew/opt/llvm/bin/llc"
ARCH = ROOT / "benchmarks/faust_comparison/bench_arch.cpp"
GEN = ROOT / "benchmarks/faust_comparison/dsp/gen_dsp.py"
DIFFCLI = ROOT / "lean/.lake/build/bin/diffcli"
WORK = ROOT / "benchmarks/faust_comparison/.work/sine"


def clean_env(**over):
    e = {k: v for k, v in os.environ.items() if not k.startswith("TROPICAL_")}
    e.update(over)
    return e


# ---------------------------------------------------------------- faust side

def faust_object(n: int, variant: str) -> Path:
    d = WORK / f"faust_{n}"
    d.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(GEN), str(n), str(d)], check=True,
                   capture_output=True)
    cpp, exe = d / f"{variant}_{n}.cpp", d / f"{variant}_{n}"
    if not exe.exists():
        subprocess.run([FAUST, "-double", "-t", "1800", "-a", str(ARCH),
                        "-o", str(cpp), str(d / f"{variant}_{n}.dsp")], check=True,
                       capture_output=True)
        subprocess.run([CXX, "-O3", "-std=c++17", "-DFAUSTFLOAT=double",
                        "-o", str(exe), str(cpp)], check=True, capture_output=True)
    return exe


def faust_compute_text(exe: Path) -> list[str]:
    """The instruction lines of the `compute` method, which holds the sample loop."""
    out = subprocess.run(["otool", "-tV", str(exe)], capture_output=True,
                         check=True).stdout.decode(errors="replace")
    lines = out.splitlines()
    # Symbols appear as `_ZN...compute...:` label lines; take the longest
    # function containing `compute` -- the DSP one, not the wrapper.
    bounds, cur = [], None
    for i, ln in enumerate(lines):
        if ln.endswith(":") and not ln.startswith((" ", "\t")) and "0x" not in ln[:4]:
            if cur is not None:
                bounds.append((cur[0], cur[1], i))
            cur = (ln[:-1], i)
    if cur is not None:
        bounds.append((cur[0], cur[1], len(lines)))
    cands = [b for b in bounds if "compute" in b[0]]
    if not cands:
        raise SystemExit("no compute symbol in " + str(exe))
    name, lo, hi = max(cands, key=lambda b: b[2] - b[1])
    return [l for l in lines[lo + 1:hi] if re.match(r"^[0-9a-f]{8,16}\t", l)]


# ------------------------------------------------------------- tropical side

def tropical_patch(n: int, d: Path) -> Path:
    patch = d / "patch.json"
    decls = [{"op": "instanceDecl", "name": f"osc{i}", "program": "FixedSinOsc",
              "inputs": {"freq": 55.0 * (1.0 + 0.017 * i)}} for i in range(n)]

    def ref(i):
        return f'{{"op":"ref","instance":"osc{i}","output":"sine"}}'
    expr = ('{"op":"add","args":[' * (n - 1)) + ref(0) + "".join(
        f",{ref(i)}]}}" for i in range(1, n)) if n > 1 else ref(0)
    decls_json = ",".join(json.dumps(x, separators=(",", ":")) for x in decls)
    sink = ('{"op":"instanceDecl","name":"sink","program":"SoftClip",'
            f'"inputs":{{"input":{expr},"drive":1}}}}')
    patch.write_text('{"schema":"tropical_program_2","name":"sineprobe' + str(n) + '",'
                     '"body":{"op":"block","decls":[' + decls_json + "," + sink + '],'
                     '"assigns":[{"op":"outputAssign","name":"dac.out",'
                     '"expr":{"op":"ref","instance":"sink","output":"out"}}]}}')
    return patch


def tropical_object(n: int) -> Path:
    d = WORK / f"trop_{n}"
    d.mkdir(parents=True, exist_ok=True)
    patch = tropical_patch(n, d)
    env = clean_env(TROPICAL_STAGE0="1", TROPICAL_JIT_OPT_LEVEL="O2",
                    TROPICAL_JIT_TRACE="1", TROPICAL_JIT_TRACE_DUMP=str(d),
                    TROPICAL_KERNEL_CACHE_DISABLE="1")
    man = d / "manifest.json"
    p = subprocess.run([str(DIFFCLI), "compile", str(patch), "--mode=fused"],
                       capture_output=True, env=env, cwd=str(ROOT))
    if p.returncode != 0:
        raise SystemExit("compile: " + p.stderr.decode()[:400])
    man.write_bytes(p.stdout)
    # render-bytes drives the JIT, which is what emits the trace dump.
    p = subprocess.run([str(DIFFCLI), "render-bytes", str(man), "--frames", "0",
                        "--buffer", "512"], capture_output=True, env=env, cwd=str(ROOT))
    if p.returncode != 0:
        raise SystemExit("render: " + p.stderr.decode()[:400])
    lls = sorted(d.glob("postpipeline_*.ll"), key=lambda q: q.stat().st_size)
    if not lls:
        raise SystemExit("no postpipeline dump in " + str(d))
    ll = lls[-1]          # the audio kernel is the big one
    obj = d / "kernel.o"
    subprocess.run([LLC, "-O0", "-filetype=obj", str(ll), "-o", str(obj)],
                   check=True, capture_output=True)
    return obj


def object_text(obj: Path) -> list[str]:
    out = subprocess.run(["otool", "-tV", str(obj)], capture_output=True,
                         check=True).stdout.decode(errors="replace")
    return [l for l in out.splitlines() if re.match(r"^[0-9a-f]{8,16}\t", l)]


def mnemonics(lines):
    ms = []
    for l in lines:
        parts = l.split("\t")
        if len(parts) >= 2 and parts[1].strip():
            ms.append(parts[1].split()[0] if parts[1].split() else "")
    return ms


def summarize(tag, lo_n, hi_n, lo, hi):
    from collections import Counter
    a, b = Counter(mnemonics(lo)), Counter(mnemonics(hi))
    dn = hi_n - lo_n
    total = (len(hi) - len(lo)) / dn
    print(f"\n=== {tag}: N={lo_n} -> N={hi_n} ===")
    print(f"  static instrs: {len(lo)} -> {len(hi)}   marginal per voice: {total:.2f}")
    deltas = sorted(((b[k] - a[k]) / dn, k) for k in set(a) | set(b))
    print("  per-voice mnemonic breakdown (>= 0.05):")
    for v, k in sorted(deltas, reverse=True):
        if abs(v) >= 0.05:
            print(f"    {v:8.2f}  {k}")
    return total


if __name__ == "__main__":
    LO, HI = 64, 128
    WORK.mkdir(parents=True, exist_ok=True)
    f_lo = faust_compute_text(faust_object(LO, "f3_libmsin_closedform"))
    f_hi = faust_compute_text(faust_object(HI, "f3_libmsin_closedform"))
    t_lo = object_text(tropical_object(LO))
    t_hi = object_text(tropical_object(HI))
    fv = summarize("Faust F3 (libm sin, closed form)", LO, HI, f_lo, f_hi)
    tv = summarize("tropical (FixedSin Q31)", LO, HI, t_lo, t_hi)
    print(f"\ntropical / F3 static instruction ratio per voice: {tv / fv:.2f}x")
