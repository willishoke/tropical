// v3_kernel_module_alive.c — kernel as external function, with alive.
//
// The kernel checks its `alive` field; if 0, returns previous state.
// In the v3 (no LTO) variant, the scheduler can't see this function's
// body, so the alive check stays as a real runtime branch in the
// compiled output.
//
// In the v4 (with LTO) variant, the linker presents both this TU and
// the scheduler TU to LLVM together. The optimizer can inline this
// function into the scheduler, then propagate the constant 1.0 that
// the scheduler writes to alive, then fold the check away.
//
// THE TEST: does v4_alive perform identically to v1_alive? If yes,
// default-alive folding survives module boundaries under LTO. If no,
// we've found a limit on the operadic substrate's modular compilation
// story.

#include "kernel_alive.h"

double kernel_step_external_alive(double x, kernel_alive_state_t *k) {
    if (k->alive > 0.5) {
        double y = k->a * x + k->b + 0.5 * k->state;
        k->state = y;
        return y;
    }
    return k->state;
}
