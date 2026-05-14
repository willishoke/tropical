// phaser.c — test the actual claim from our discussion:
// when a Phaser16 instance (composed of 16 internal allpass stages)
// is asleep, none of the 16 stages should compute.
//
// After strata, Phaser16's body is FLATTENED — all 16 stages are
// already inlined into Phaser16's function body. The 16 stages
// are not separate functions in the IR; they're just instructions.
// So the test is: when the conditional `if (alive_phaser)` is false,
// do all 16 stages' instructions disappear from the executed path?

#include <stdint.h>

static inline __attribute__((always_inline))
void phaser16(double * restrict slots, double * restrict regs) {
    double input = slots[0];
    double accum = input;
    // 16 allpass stages — each has a 1-sample delay + filter coef
    for (int i = 0; i < 16; i++) {
        double delay = regs[i];
        double coef = 0.5;
        double out = -coef * accum + delay;
        regs[i] = accum + coef * out;
        accum = out;
    }
    slots[1] = accum;
}

void scheduler(
    double * restrict slots,
    double * restrict regs,
    float  * restrict output,
    int32_t block_size
) {
    for (int32_t s = 0; s < block_size; s++) {
        if (slots[2] > 0.5) phaser16(slots, regs);
        output[s] = (float) slots[1];
    }
}
