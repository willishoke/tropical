// v3_scheduler_16.c — variant 3 at chain length 16 (Phaser16-scale).
//
// Chains 16 calls to kernel_step_external. Without LTO, this should
// produce 16 actual function calls per sample. With LTO (v4_16), the
// external function gets inlined and the result should match v1_16.

#include "kernel.h"

extern double kernel_step_external(double x, kernel_state_t *k);

__attribute__((noinline))
void scheduler_v3_16(
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
        y = kernel_step_external(y, k0);  y = kernel_step_external(y, k1);
        y = kernel_step_external(y, k2);  y = kernel_step_external(y, k3);
        y = kernel_step_external(y, k4);  y = kernel_step_external(y, k5);
        y = kernel_step_external(y, k6);  y = kernel_step_external(y, k7);
        y = kernel_step_external(y, k8);  y = kernel_step_external(y, k9);
        y = kernel_step_external(y, k10); y = kernel_step_external(y, k11);
        y = kernel_step_external(y, k12); y = kernel_step_external(y, k13);
        y = kernel_step_external(y, k14); y = kernel_step_external(y, k15);
        output[i] = y;
    }
}
