#!/usr/bin/env python3
"""Generate C source for the LTO compile-time stress benchmark.

The generator is the "code-golf" surface: this short script produces
arbitrarily many kernels of source that expand to deep LLVM IR.

Usage:
    gen.py monolithic <N> <outdir>   # one TU with N inlined kernels
    gen.py separate   <N> <outdir>   # N kernel TUs + 1 scheduler TU

Generated sources are gitignored (work tree only).
"""

import sys
from pathlib import Path

HEADER = """#pragma once
#include <stddef.h>
typedef struct { double a, b, s; } K;
static inline double step(double x, K *k) {
    double y = k->a * x + k->b + 0.5 * k->s;
    k->s = y;
    return y;
}
"""

MAIN_TMPL = """#include "kernel.h"
extern void chain(const double *, double *, size_t, K *);
int main(void) {
    static K ks[%d];
    static double in[8], out[8];
    chain(in, out, 8, ks);
    return (int)out[0];
}
"""


def gen_monolithic(n, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "kernel.h").write_text(HEADER)
    lines = [
        '#include "kernel.h"',
        '__attribute__((noinline))',
        'void chain(const double *in, double *out, size_t n, K *ks) {',
        '    for (size_t i = 0; i < n; i++) {',
        '        double y = in[i];',
    ]
    lines.extend(f'        y = step(y, &ks[{k}]);' for k in range(n))
    lines += ['        out[i] = y;', '    }', '}', '']
    (outdir / "monolithic.c").write_text("\n".join(lines))
    (outdir / "main.c").write_text(MAIN_TMPL % n)


def gen_separate(n, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "kernel.h").write_text(HEADER)
    kdir = outdir / "kernels"
    kdir.mkdir(exist_ok=True)
    for k in range(n):
        (kdir / f"k_{k}.c").write_text(
            f'#include "../kernel.h"\n'
            f'double step_{k}(double x, K *k) {{ return step(x, k); }}\n'
        )
    sched = ['#include "kernel.h"']
    sched.extend(f'extern double step_{k}(double, K *);' for k in range(n))
    sched += [
        '__attribute__((noinline))',
        'void chain(const double *in, double *out, size_t n, K *ks) {',
        '    for (size_t i = 0; i < n; i++) {',
        '        double y = in[i];',
    ]
    sched.extend(f'        y = step_{k}(y, &ks[{k}]);' for k in range(n))
    sched += ['        out[i] = y;', '    }', '}', '']
    (outdir / "scheduler.c").write_text("\n".join(sched))
    (outdir / "main.c").write_text(MAIN_TMPL % n)


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    variant, n, outdir = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    {"monolithic": gen_monolithic, "separate": gen_separate}[variant](n, outdir)


if __name__ == "__main__":
    main()
