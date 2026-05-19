#!/usr/bin/env python3
"""Two-stage LTO compile time: isolate the LTO link step.

Setup matches the incremental-rebuild story:
  Stage A (one-time per kernel, cached across edits):
      clang -c -flto[=thin] -O2 kernel.c -o kernel.bc
  Stage B (re-runs whenever the patch graph changes):
      clang -flto[=thin] -O2 *.bc -o out

The user-facing cost of "add an instance and rewire it" is roughly:
    1 × Stage-A (for the new kernel) + 1 × Stage-B (full LTO link).

Measures both FullLTO and ThinLTO. ThinLTO is the more incremental
flavour: per-bitcode summaries enable targeted cross-module inlining
without merging everything into a single module first.

For each N this script measures:
  mono_nolto         — clang -O2 mono.c main.c          (full-rebuild baseline)
  mono_lto           — clang -O2 -flto mono.c main.c    (full-rebuild w/ FullLTO)
  full_bc_total      — Σ Stage-A (FullLTO bitcodes)
  full_lto_link      — Stage B FullLTO link              (← the headline #1)
  thin_bc_total      — Σ Stage-A (ThinLTO bitcodes)
  thin_lto_link      — Stage B ThinLTO link              (← the headline #2)

Repeated 3× per measurement, min reported. Hard 15s per clang
invocation; (variant) timing out is skipped at larger N.
"""

import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE      = Path(__file__).parent
GEN       = HERE / "gen.py"
WORK      = HERE / "work"
RESULTS   = HERE / "data" / "results.csv"
TIMEOUT_S = 15.0
CC        = "/opt/homebrew/opt/llvm/bin/clang"
SIZES     = [16, 64, 256, 512, 1024, 2048, 4096]
REPEATS   = 3


def time_one(args):
    t0 = time.monotonic()
    try:
        r = subprocess.run([CC] + args, timeout=TIMEOUT_S, capture_output=True)
        elapsed = time.monotonic() - t0
        if r.returncode != 0:
            return (False, elapsed, "fail: " + r.stderr.decode()[:80])
        return (True, elapsed, "ok")
    except subprocess.TimeoutExpired:
        return (False, TIMEOUT_S, "timeout")


def time_min(args):
    best = None
    for _ in range(REPEATS):
        ok, elapsed, status = time_one(args)
        if not ok:
            return (False, elapsed, status)
        best = elapsed if best is None else min(best, elapsed)
    return (True, best, "ok")


def measure_monolithic(n, lto):
    outdir = WORK / f"mono_n{n}{'_lto' if lto else ''}"
    if outdir.exists():
        shutil.rmtree(outdir)
    subprocess.run([sys.executable, str(GEN), "monolithic", str(n), str(outdir)], check=True)
    args = ["-O2", "-Wno-everything"]
    if lto:
        args.append("-flto")
    args += [str(outdir / "monolithic.c"), str(outdir / "main.c"),
             "-o", str(outdir / "out")]
    return time_min(args)


def measure_separate_two_stage(n, lto_mode):
    """lto_mode is 'full' or 'thin'.

    Returns (bc_total_s_or_None, link_s_or_None, status).
    """
    outdir = WORK / f"sep_{lto_mode}_n{n}"
    if outdir.exists():
        shutil.rmtree(outdir)
    subprocess.run([sys.executable, str(GEN), "separate", str(n), str(outdir)], check=True)
    lto_flag = f"-flto={lto_mode}"

    bc_files = []
    bc_total = 0.0
    srcs = sorted((outdir / "kernels").glob("*.c")) + \
           [outdir / "scheduler.c", outdir / "main.c"]
    for src in srcs:
        bc = src.with_suffix(".bc")
        args = ["-c", lto_flag, "-O2", "-Wno-everything", str(src), "-o", str(bc)]
        ok, elapsed, status = time_one(args)
        if not ok:
            return (None, None, f"stage-A {status}")
        bc_total += elapsed
        bc_files.append(bc)

    link_args = [lto_flag, "-O2", "-Wno-everything"] + [str(bc) for bc in bc_files] + \
                ["-o", str(outdir / "out")]
    ok, link_elapsed, status = time_min(link_args)
    if not ok:
        return (bc_total, None, f"stage-B {status}")
    return (bc_total, link_elapsed, "ok")


def fmt(x):
    return f"{x*1000:>8.1f} ms" if x is not None else f"{'—':>11}"


def main():
    RESULTS.parent.mkdir(exist_ok=True)
    WORK.mkdir(exist_ok=True)
    rows = []
    skip_full, skip_thin = False, False

    header = (f"{'N':>5}  {'mono_nolto':>10}  {'mono_lto':>10}"
              f"  {'full_bc_total':>13}  {'full_link':>10}"
              f"  {'thin_bc_total':>13}  {'thin_link':>10}")
    print(header); print("-" * len(header))

    for n in SIZES:
        _, mn_elapsed, mn_status = measure_monolithic(n, False)
        _, ml_elapsed, ml_status = measure_monolithic(n, True)

        if skip_full:
            full_bc, full_link, full_status = None, None, "skipped"
        else:
            full_bc, full_link, full_status = measure_separate_two_stage(n, "full")
            if "timeout" in full_status:
                skip_full = True

        if skip_thin:
            thin_bc, thin_link, thin_status = None, None, "skipped"
        else:
            thin_bc, thin_link, thin_status = measure_separate_two_stage(n, "thin")
            if "timeout" in thin_status:
                skip_thin = True

        print(f"{n:>5}  {fmt(mn_elapsed)}  {fmt(ml_elapsed)}"
              f"  {fmt(full_bc)}  {fmt(full_link)}"
              f"  {fmt(thin_bc)}  {fmt(thin_link)}")

        def add(metric, val, status):
            rows.append([n, metric,
                         f"{val*1000:.1f}" if val is not None else "", status])
        add("mono_nolto",        mn_elapsed,                       mn_status)
        add("mono_lto",          ml_elapsed,                       ml_status)
        add("full_bc_total",     full_bc,                          full_status)
        add("full_bc_avg",       (full_bc / n) if full_bc else None, full_status)
        add("full_lto_link",     full_link,                        full_status)
        add("thin_bc_total",     thin_bc,                          thin_status)
        add("thin_bc_avg",       (thin_bc / n) if thin_bc else None, thin_status)
        add("thin_lto_link",     thin_link,                        thin_status)

        if WORK.exists():
            shutil.rmtree(WORK); WORK.mkdir()

    with RESULTS.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_kernels", "metric", "value_ms", "status"])
        w.writerows(rows)
    print(f"\nResults → {RESULTS}")


if __name__ == "__main__":
    main()
