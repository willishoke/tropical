// phaser_compare.c — does the per-sample conditional inhibit
// LICM and phi-promotion of the 16 reg states, or are they
// inherently going to memory regardless?

#include <stdint.h>

// (a) FLAT: how the current JIT would emit Phaser16 in a single function
void phaser_flat(
    double * restrict slots,
    double * restrict regs,
    float  * restrict output,
    int32_t block_size
) {
    for (int32_t s = 0; s < block_size; s++) {
        double input = slots[0];
        double accum = input;
        for (int i = 0; i < 16; i++) {
            double delay = regs[i];
            double out = -0.5 * accum + delay;
            regs[i] = accum + 0.5 * out;
            accum = out;
        }
        slots[1] = accum;
        output[s] = (float) accum;
    }
}

// (b) ALWAYSINLINE no conditional
static inline __attribute__((always_inline))
void phaser16_b(double * restrict slots, double * restrict regs) {
    double input = slots[0];
    double accum = input;
    for (int i = 0; i < 16; i++) {
        double delay = regs[i];
        double out = -0.5 * accum + delay;
        regs[i] = accum + 0.5 * out;
        accum = out;
    }
    slots[1] = accum;
}

void phaser_inline_uncond(
    double * restrict slots,
    double * restrict regs,
    float  * restrict output,
    int32_t block_size
) {
    for (int32_t s = 0; s < block_size; s++) {
        phaser16_b(slots, regs);
        output[s] = (float) slots[1];
    }
}

// (c) ALWAYSINLINE WITH conditional
static inline __attribute__((always_inline))
void phaser16_c(double * restrict slots, double * restrict regs) {
    double input = slots[0];
    double accum = input;
    for (int i = 0; i < 16; i++) {
        double delay = regs[i];
        double out = -0.5 * accum + delay;
        regs[i] = accum + 0.5 * out;
        accum = out;
    }
    slots[1] = accum;
}

void phaser_inline_cond(
    double * restrict slots,
    double * restrict regs,
    float  * restrict output,
    int32_t block_size
) {
    for (int32_t s = 0; s < block_size; s++) {
        if (slots[2] > 0.5) phaser16_c(slots, regs);
        output[s] = (float) slots[1];
    }
}
