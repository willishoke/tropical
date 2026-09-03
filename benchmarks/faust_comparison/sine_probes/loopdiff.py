#!/usr/bin/env python3
"""Marginal per-voice cost of the PER-SAMPLE LOOP BODY only.

A whole-function instruction count is wrong here: both compilers hoist
block-invariant work (tropical's `freq*2^32 / sampleRate`, Faust's constant
folding) into a preheader that runs once per 512-frame block, not once per
sample.  Counting the function would price hoisted work as if it were hot.

So: emit assembly, split into basic blocks by label, and keep only the blocks
that belong to the innermost loop -- the ones reachable from a block whose
terminator branches BACKWARD.  Differencing two voice counts then cancels the
loop's own overhead and leaves the marginal per-voice per-sample cost.
"""
import json, os, re, subprocess, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FAUST, CXX = "/opt/homebrew/bin/faust", "/opt/homebrew/opt/llvm/bin/clang++"
LLC = "/opt/homebrew/opt/llvm/bin/llc"
ARCH = ROOT / "benchmarks/faust_comparison/bench_arch.cpp"
GEN = ROOT / "benchmarks/faust_comparison/dsp/gen_dsp.py"
DIFFCLI = ROOT / "lean/.lake/build/bin/diffcli"
WORK = ROOT / "benchmarks/faust_comparison/.work/sine"

LABEL = re.compile(r"^(L?[A-Za-z_.$][\w.$]*):")
INSTR = re.compile(r"^\s+([a-z][\w.]*)\b")
DIRECTIVE = re.compile(r"^\s+\.")


def clean_env(**over):
    e = {k: v for k, v in os.environ.items() if not k.startswith("TROPICAL_")}
    e.update(over)
    return e


def blocks(asm_text: str, func_hint: str):
    """[(label, [mnemonics], [branch targets])] for the function containing func_hint."""
    # strip trailing `; ...` comments: label lines carry one, which would
    # otherwise defeat the "ends with a colon" test.
    lines = [re.sub(r"\s*;.*$", "", l) for l in asm_text.splitlines()]
    # bound the function: from its label to the next `.cfi_endproc`
    start = next(i for i, l in enumerate(lines)
                 if func_hint in l and l.rstrip().endswith(":") and not l.startswith((" ", "\t")))
    raw = asm_text.splitlines()
    # clang emits `.cfi_endproc`; llc without frame info emits only the
    # `-- End function` marker comment.  Accept either.
    end = next(i for i in range(start + 1, len(raw))
               if ".cfi_endproc" in raw[i] or "-- End function" in raw[i])
    out, cur = [], ("__entry__", [], [])
    for l in lines[start + 1:end]:
        m = LABEL.match(l)
        if m:
            out.append(cur)
            cur = (m.group(1), [], [])
            continue
        if DIRECTIVE.match(l):
            continue
        mi = INSTR.match(l)
        if mi:
            cur[1].append(mi.group(1))
            if mi.group(1).startswith(("b", "cb", "tb")) and not mi.group(1).startswith("bl"):
                t = re.search(r"\b(L?[A-Za-z_.$][\w.$]*)\s*$", l)
                if t:
                    cur[2].append(t.group(1))
    out.append(cur)
    return out


def loop_body_mnemonics(asm_text: str, func_hint: str):
    bs = blocks(asm_text, func_hint)
    index = {name: i for i, (name, _, _) in enumerate(bs)}
    # a backward branch marks a loop; take every block from the target to the
    # branching block inclusive -- the loop's body in layout order.
    best = None
    for i, (name, ms, targets) in enumerate(bs):
        for t in targets:
            j = index.get(t)
            if j is not None and j <= i:
                span = sum(len(b[1]) for b in bs[j:i + 1])
                if best is None or span > best[0]:
                    best = (span, j, i)
    if best is None:
        raise SystemExit("no loop found in " + func_hint)
    _, j, i = best
    ms = []
    for b in bs[j:i + 1]:
        ms.extend(b[1])
    return ms


# ---------------------------------------------------------------- faust side

def faust_asm(n: int, variant: str) -> str:
    d = WORK / f"faust_{n}"
    d.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(GEN), str(n), str(d)], check=True,
                   capture_output=True)
    cpp, s = d / f"{variant}_{n}.cpp", d / f"{variant}_{n}.s"
    if not cpp.exists():
        subprocess.run([FAUST, "-double", "-t", "1800", "-a", str(ARCH),
                        "-o", str(cpp), str(d / f"{variant}_{n}.dsp")],
                       check=True, capture_output=True)
    if not s.exists():
        subprocess.run([CXX, "-O3", "-std=c++17", "-DFAUSTFLOAT=double", "-S",
                        "-o", str(s), str(cpp)], check=True, capture_output=True)
    return s.read_text()


# ------------------------------------------------------------- tropical side

def tropical_asm(n: int) -> str:
    d = WORK / f"trop_{n}"
    d.mkdir(parents=True, exist_ok=True)
    patch = d / "patch.json"
    decls = [{"op": "instanceDecl", "name": f"osc{i}", "program": "FixedSinOsc",
              "inputs": {"freq": 55.0 * (1.0 + 0.017 * i)}} for i in range(n)]

    def ref(i):
        return f'{{"op":"ref","instance":"osc{i}","output":"sine"}}'
    expr = ('{"op":"add","args":[' * (n - 1)) + ref(0) + "".join(
        f",{ref(i)}]}}" for i in range(1, n)) if n > 1 else ref(0)
    patch.write_text('{"schema":"tropical_program_2","name":"sineprobe' + str(n) + '",'
                     '"body":{"op":"block","decls":['
                     + ",".join(json.dumps(x, separators=(",", ":")) for x in decls)
                     + ',{"op":"instanceDecl","name":"sink","program":"SoftClip",'
                     f'"inputs":{{"input":{expr},"drive":1}}}}],'
                     '"assigns":[{"op":"outputAssign","name":"dac.out",'
                     '"expr":{"op":"ref","instance":"sink","output":"out"}}]}}')
    env = clean_env(TROPICAL_STAGE0="1", TROPICAL_JIT_OPT_LEVEL="O2",
                    TROPICAL_JIT_TRACE="1", TROPICAL_JIT_TRACE_DUMP=str(d),
                    TROPICAL_KERNEL_CACHE_DISABLE="1")
    man = d / "manifest.json"
    p = subprocess.run([str(DIFFCLI), "compile", str(patch), "--mode=fused"],
                       capture_output=True, env=env, cwd=str(ROOT))
    if p.returncode != 0:
        raise SystemExit("compile: " + p.stderr.decode()[:400])
    man.write_bytes(p.stdout)
    p = subprocess.run([str(DIFFCLI), "render-bytes", str(man), "--frames", "0",
                        "--buffer", "512"], capture_output=True, env=env, cwd=str(ROOT))
    if p.returncode != 0:
        raise SystemExit("render: " + p.stderr.decode()[:400])
    ll = sorted(d.glob("postpipeline_*.ll"), key=lambda q: q.stat().st_size)[-1]
    s = d / "kernel.s"
    subprocess.run([LLC, "-O0", "-filetype=asm", str(ll), "-o", str(s)],
                   check=True, capture_output=True)
    return s.read_text()


def report(tag, lo_n, hi_n, lo, hi):
    a, b, dn = Counter(lo), Counter(hi), hi_n - lo_n
    per = (len(hi) - len(lo)) / dn
    print(f"\n=== {tag}: loop body, N={lo_n} -> {hi_n} ===")
    print(f"  loop-body instrs: {len(lo)} -> {len(hi)}   PER VOICE PER SAMPLE: {per:.2f}")
    for v, k in sorted(((b[k] - a[k]) / dn, k) for k in set(a) | set(b))[::-1]:
        if abs(v) >= 0.05:
            print(f"    {v:8.2f}  {k}")
    return per


if __name__ == "__main__":
    LO, HI = 64, 128
    WORK.mkdir(parents=True, exist_ok=True)
    fl = loop_body_mnemonics(faust_asm(LO, "f3_libmsin_closedform"), "compute")
    fh = loop_body_mnemonics(faust_asm(HI, "f3_libmsin_closedform"), "compute")
    tl = loop_body_mnemonics(tropical_asm(LO), "tropical_k_")
    th = loop_body_mnemonics(tropical_asm(HI), "tropical_k_")
    fv = report("Faust F3 (libm sin, closed form)", LO, HI, fl, fh)
    tv = report("tropical (FixedSin Q31)", LO, HI, tl, th)
    print(f"\nper-voice per-sample loop-body instruction ratio: {tv/fv:.2f}x")
