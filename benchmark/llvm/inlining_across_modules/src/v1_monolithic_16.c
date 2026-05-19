// v1_monolithic_16.c — variant 1 at chain length 16 (Phaser16-scale).
//
// Same structure as v1_monolithic.c but with 16 kernels chained instead
// of 4. Tests how the per-kernel function-call cost (compared to v3_16)
// scales with chain length. If the cost scales linearly with kernel
// count, separate-modules without LTO becomes problematic for realistic
// audio DSP (a Phaser16 has 16 internal stages).

#include "kernel.h"

__attribute__((noinline))
void scheduler_v1_16(
    const double *input,
    double *output,
    size_t n,
    kernel_state_t *k0, kernel_state_t *k1, kernel_state_t *k2, kernel_state_t *k3,
    kernel_state_t *k4, kernel_state_t *k5, kernel_state_t *k6, kernel_state_t *k7,
    kernel_state_t *k8, kernel_state_t *k9, kernel_state_t *k10, kernel_state_t *k11,
    kernel_state_t *k12, kernel_state_t *k13, kernel_state_t *k14, kernel_state_t *k15
) {
    for (size_t i = 0; i < n; i++) {
        double y = input[i];
        y = kernel_step(y, k0);  y = kernel_step(y, k1);
        y = kernel_step(y, k2);  y = kernel_step(y, k3);
        y = kernel_step(y, k4);  y = kernel_step(y, k5);
        y = kernel_step(y, k6);  y = kernel_step(y, k7);
        y = kernel_step(y, k8);  y = kernel_step(y, k9);
        y = kernel_step(y, k10); y = kernel_step(y, k11);
        y = kernel_step(y, k12); y = kernel_step(y, k13);
        y = kernel_step(y, k14); y = kernel_step(y, k15);
        output[i] = y;
    }
}
