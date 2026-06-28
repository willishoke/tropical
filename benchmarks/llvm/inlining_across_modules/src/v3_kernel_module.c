// v3_kernel_module.c — the kernel implementation as its own
// translation unit. Used by variant 3 (separate modules, no LTO) and
// variant 4 (separate modules, with LTO).
//
// The function is NON-static and has external linkage so the linker
// can resolve it. The `inline` keyword is not used — without LTO,
// inline functions in separate translation units would be invisible
// to the optimizer at the call site. With LTO, the optimizer can
// inspect the body and inline as it would for any external function.
//
// The contrast with v1 is: there, the call to `kernel_step` is
// inline-by-definition. Here, the call is a real function call until
// the linker (with LTO) chooses to inline it.

#include "kernel.h"

// Re-export as an external function. Cannot be `static inline` because
// the scheduler in a different translation unit needs to call it.
double kernel_step_external(double x, kernel_state_t *k) {
    double y = k->a * x + k->b + 0.5 * k->state;
    k->state = y;
    return y;
}
