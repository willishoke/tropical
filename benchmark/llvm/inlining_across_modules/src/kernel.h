// kernel.h — shared definition of the per-sample kernel computation.
//
// Each kernel: y = a*x + b + 0.5 * state; state = y. Mirrors the
// dataflow shape of a simple recursive filter cell. The arithmetic is
// not meaningful as a DSP operation; it's chosen to exercise:
//
//   - A load of the input
//   - An fmuladd (a*x + b)
//   - A load of the persistent state slot
//   - A second fmuladd (... + 0.5 * state)
//   - A store back to the state slot
//   - Production of the output value
//
// These are the per-sample patterns tropical's tight inner loop has.
// If LLVM can inline the kernel across compilation boundaries and
// recover the same code it produces for the inlined-in-place version,
// the loop body should look identical post-optimization regardless of
// how the kernels were composed at the source level.

#ifndef BENCH_INLINING_KERNEL_H
#define BENCH_INLINING_KERNEL_H

#include <stddef.h>

// The kernel's per-instance configuration. `a` and `b` are
// compile-time-ish coefficients (treated as runtime data here so the
// optimizer can't constant-fold them away). `state` is the persistent
// register threaded across samples.
typedef struct {
    double a;
    double b;
    double state;
} kernel_state_t;

// One sample of the kernel computation, as a pure inline function.
// Defined in the header so it can be embedded as-is by the monolithic
// variant. Each compilation-strategy variant uses this in a
// different way (inlined, called via static function, called across
// module boundaries, etc.).
static inline double kernel_step(double x, kernel_state_t *k) {
    double y = k->a * x + k->b + 0.5 * k->state;
    k->state = y;
    return y;
}

// Tree shape: linear chain of N kernels. Output of kernel i becomes
// input of kernel i+1.
#define CHAIN_LEN 4

#endif // BENCH_INLINING_KERNEL_H
