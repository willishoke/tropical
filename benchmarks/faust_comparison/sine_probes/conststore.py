#!/usr/bin/env python3
"""Price the loop-invariant constant slot stores before building a fix.

Every instance input port gets a module slot and the emitter writes that slot
INSIDE the per-sample loop, including when the input is a compile-time literal.
Deleting those stores from the emitted IR is a legitimate upper bound on what
hoisting them would buy: the value is a constant and the address is a distinct
constant GEP into `%slots`, so no load in the kernel can observe the difference
(-O2 has already forwarded them), and the last write of each slot still happens
-- just once per block instead of once per sample.

Same harness as the cliff probes: rewrite an `audio.ll` dumped by
TROPICAL_STAGE0_DUMP and feed it to `tropical_runtime_bench --ir`.

  usage: conststore.py <stages> <comma-separated voice counts>
"""
import glob, json, os, pathlib, re, shutil, statistics, subprocess, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                       / "oscillator_saturation" / "cliff_probes"))
from slotknee import run as emit                                    # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[3]
WORK = ROOT / "benchmarks/oscillator_saturation/.work/cliff"
BENCH = str(ROOT / "build/tropical_runtime_bench")
LLVM = "/opt/homebrew/opt/llvm/bin"

CONST_STORE = re.compile(
    r"^\s+store double (?:[-0-9][0-9.eE+-]*|0x[0-9A-Fa-f]+), ptr %[\w.]+, align 8\s*$")


def strip_const_stores(src: pathlib.Path, dst: pathlib.Path) -> int:
    lines = src.read_text().splitlines(keepends=True)
    lo = next(i for i, l in enumerate(lines) if l.startswith("loop_body:"))
    hi = next(i for i, l in enumerate(lines) if l.startswith("loop_end:"))
    out, removed = [], 0
    for i, l in enumerate(lines):
        if lo < i < hi and CONST_STORE.match(l):
            removed += 1
            continue
        out.append(l)
    dst.write_text("".join(out))
    return removed


def bench(ir: str, man: str, tag: str) -> tuple[float, int]:
    cache = WORK / "cs" / tag
    shutil.rmtree(cache, ignore_errors=True)
    cache.mkdir(parents=True, exist_ok=True)
    e = {k: v for k, v in os.environ.items() if not k.startswith("TROPICAL_")}
    e.update({"TROPICAL_STAGE0": "1", "TROPICAL_JIT_OPT_LEVEL": "O2",
              "TROPICAL_KERNEL_CACHE_DISABLE": "0",
              "TROPICAL_KERNEL_CACHE_ROOT": str(cache)})
    time.sleep(1.0)
    p = json.loads(subprocess.run(
        [BENCH, "--ir", ir, "--manifest", man, "--buffer", "512", "--blocks",
         "300", "--warmup", "32", "--rate", "44100"],
        cwd=str(ROOT), env=e, capture_output=True, check=True).stdout)
    o = max(glob.glob(f"{cache}/*/*.o"), key=os.path.getsize)
    tb = int(re.search(r"__text\s+(\d+)", subprocess.run(
        [f"{LLVM}/llvm-size", "--format=sysv", o],
        capture_output=True, text=True).stdout).group(1))
    return statistics.median(p["process_ns"]), tb


if __name__ == "__main__":
    stages = int(sys.argv[1])
    counts = [int(c) for c in sys.argv[2].split(",")]
    print(f"{'N':>5} {'removed':>8} {'stock_text':>11} {'stock ns/v':>11} "
          f"{'strip_text':>11} {'strip ns/v':>11} {'delta':>7}")
    for n in counts:
        emit(n, stages, measure=False)
        d = WORK / "slotknee" / f"s{stages}" / f"n{n}"
        removed = strip_const_stores(d / "audio.ll", d / "nostores.ll")
        a, ta = bench(str(d / "audio.ll"), str(d / "manifest.json"), f"s{stages}n{n}a")
        b, tb = bench(str(d / "nostores.ll"), str(d / "manifest.json"), f"s{stages}n{n}b")
        print(f"{n:>5} {removed:>8} {ta:>11} {a/n:>11.1f} {tb:>11} {b/n:>11.1f} "
              f"{100*(b-a)/a:>6.1f}%", flush=True)
