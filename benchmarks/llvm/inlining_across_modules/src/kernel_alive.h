// kernel_alive.h — alive-gated kernel variant.
//
// Each kernel has an `alive` field. The kernel_step_alive function
// checks alive before computing; when alive ≤ 0.5, the kernel returns
// a held value (typically the last output state) without recomputing
// or updating its state. Mirrors tropical's active-set runtime
// semantics: "an asleep instance retains its last-written output and
// its internal state freezes."
//
// The question this variant exists to answer: does the constant-fold
// pattern that makes default-alive checks disappear under O2 survive
// across compilation-module boundaries (with and without LTO)?
//
// In the active-set spike:
//   WriteSlot(alive, 1.0); ... if (alive > 0.5) { kernel_body }
// folds to unconditional `kernel_body` via GVN — the optimizer sees
// the constant-1.0 store, forwards it to the load inside the alive
// check, folds the comparison to true, removes the conditional.
//
// That works when WriteSlot and the if-load live in the SAME LLVM
// function. We're testing whether it works across function-call and
// module-boundary configurations.

#ifndef BENCH_INLINING_KERNEL_ALIVE_H
#define BENCH_INLINING_KERNEL_ALIVE_H

#include <stddef.h>

// Alive-gated kernel state. `state` is the persistent register;
// `alive` is the (per-sample, written by the scheduler) gating signal.
typedef struct {
    double a;
    double b;
    double state;
    double alive;
} kernel_alive_state_t;

// One sample of the kernel computation, gated on `alive`. When alive
// is below threshold (0.5), the kernel SKIPS its body and returns the
// previous state value unchanged. When alive is above threshold, the
// kernel runs identically to the non-alive version.
//
// Inline form: when used in v1 (kernel + scheduler same TU), this is
// inlined at the call site. The (alive > 0.5) check then folds away
// if the scheduler wrote 1.0 to alive earlier in the same sample.
static inline double kernel_step_alive(double x, kernel_alive_state_t *k) {
    if (k->alive > 0.5) {
        double y = k->a * x + k->b + 0.5 * k->state;
        k->state = y;
        return y;
    }
    return k->state;
}

#endif // BENCH_INLINING_KERNEL_ALIVE_H
