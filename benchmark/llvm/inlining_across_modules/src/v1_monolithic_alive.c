// v1_monolithic_alive.c — variant 1, alive-aware, chain4.
//
// Each sample: write alive=1.0 to each kernel (the default-alive
// pattern), then chain the four kernels. We expect the optimizer to:
//   - See the constant 1.0 write to k->alive
//   - Forward it through GVN to the load inside kernel_step_alive
//   - Fold the (alive > 0.5) comparison to `true`
//   - Eliminate the conditional entirely
// resulting in identical optimized IR to v1_monolithic.c (no alive).
//
// If this works, the runtime cost of default-alive is zero. If the
// optimizer doesn't fold (for whatever reason), we'd see an
// observable runtime cost vs the no-alive baseline.

#include "kernel_alive.h"

__attribute__((noinline))
void scheduler_v1_alive(
    const double *input,
    double *output,
    size_t n,
    kernel_alive_state_t *k1,
    kernel_alive_state_t *k2,
    kernel_alive_state_t *k3,
    kernel_alive_state_t *k4
) {
    for (size_t i = 0; i < n; i++) {
        // Default-alive pattern: write 1.0 to each alive slot every
        // sample. LLVM's GVN should see this as constant and fold the
        // alive checks downstream.
        k1->alive = 1.0;
        k2->alive = 1.0;
        k3->alive = 1.0;
        k4->alive = 1.0;

        double y1 = kernel_step_alive(input[i], k1);
        double y2 = kernel_step_alive(y1,       k2);
        double y3 = kernel_step_alive(y2,       k3);
        double y4 = kernel_step_alive(y3,       k4);
        output[i] = y4;
    }
}
