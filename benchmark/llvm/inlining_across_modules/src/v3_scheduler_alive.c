// v3_scheduler_alive.c — alive-aware scheduler calling external kernel.

#include "kernel_alive.h"

extern double kernel_step_external_alive(double x, kernel_alive_state_t *k);

__attribute__((noinline))
void scheduler_v3_alive(
    const double *input,
    double *output,
    size_t n,
    kernel_alive_state_t *k1,
    kernel_alive_state_t *k2,
    kernel_alive_state_t *k3,
    kernel_alive_state_t *k4
) {
    for (size_t i = 0; i < n; i++) {
        k1->alive = 1.0;
        k2->alive = 1.0;
        k3->alive = 1.0;
        k4->alive = 1.0;

        double y1 = kernel_step_external_alive(input[i], k1);
        double y2 = kernel_step_external_alive(y1,       k2);
        double y3 = kernel_step_external_alive(y2,       k3);
        double y4 = kernel_step_external_alive(y3,       k4);
        output[i] = y4;
    }
}
