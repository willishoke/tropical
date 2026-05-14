// v3_scheduler.c — variant 3 scheduler: chains kernels via calls to
// kernel_step_external in a separate translation unit.
//
// Without LTO, the optimizer at compile time of this file knows only
// that kernel_step_external is an external function (it's an external
// declaration, no body visible). It cannot inline; cannot reason about
// register state; cannot hoist loads. The result should be: a real
// function call per kernel per sample, with state loaded and stored
// through memory each call.
//
// With LTO (variant 4), the linker presents both translation units
// to LLVM together. The optimizer can then inline kernel_step_external
// and apply the full set of optimizations it applies to v1. We expect
// (and the experiment will measure) that LTO recovers most or all of
// v1's optimized code shape.

#include "kernel.h"

// External declaration — body lives in v3_kernel_module.c.
extern double kernel_step_external(double x, kernel_state_t *k);

void scheduler_v3(
    const double *input,
    double *output,
    size_t n,
    kernel_state_t *k1,
    kernel_state_t *k2,
    kernel_state_t *k3,
    kernel_state_t *k4
) {
    for (size_t i = 0; i < n; i++) {
        double y1 = kernel_step_external(input[i], k1);
        double y2 = kernel_step_external(y1,       k2);
        double y3 = kernel_step_external(y2,       k3);
        double y4 = kernel_step_external(y3,       k4);
        output[i] = y4;
    }
}
