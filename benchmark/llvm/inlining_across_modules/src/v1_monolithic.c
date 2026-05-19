// v1_monolithic.c — variant 1: the entire kernel chain compiled
// into ONE function body. This mirrors today's tropical JIT shape,
// where a session's instances become nested basic blocks inside a
// single LLVM function. Inlining is trivial because there are no
// function boundaries to cross.
//
// We expect the optimized IR to:
//   - hoist all kernel_state_t loads out of the loop
//   - thread state through phi nodes
//   - emit 4 fused multiply-adds per sample, with no intervening
//     memory ops
//   - store updated states back to memory ONCE after the loop
//
// This is the baseline against which the other variants are compared.

#include "kernel.h"

// `noinline` on the scheduler prevents LTO from inlining it into the
// timing loop in runner.c. Without this attribute, with -flto, the
// optimizer would inline scheduler_v1 into main's measurement loop and
// constant-propagate the kernel coefficients (which are stack-local in
// main). That conflates "kernel inlining into scheduler" with
// "scheduler inlining into runner," both of which give performance
// wins. With this attribute, the only LTO effect under measurement is
// the kernel-into-scheduler inlining (orthogonally to the scheduler's
// call from runner).
__attribute__((noinline))
void scheduler_v1(
    const double *input,
    double *output,
    size_t n,
    kernel_state_t *k1,
    kernel_state_t *k2,
    kernel_state_t *k3,
    kernel_state_t *k4
) {
    for (size_t i = 0; i < n; i++) {
        double y1 = kernel_step(input[i], k1);
        double y2 = kernel_step(y1,       k2);
        double y3 = kernel_step(y2,       k3);
        double y4 = kernel_step(y3,       k4);
        output[i] = y4;
    }
}
