// active_set_spike.c
//
// Experiment: does the proposed multi-function architecture
// (per-instance kernel functions + a scheduler with conditional dispatch)
// survive LLVM optimization?
//
// Three claims to verify:
//   1. alwaysinline functions are inlined into the scheduler
//   2. For ALWAYS-ON instances (alive constant true): slot loads
//      eliminated, code looks identical to a unified flat kernel
//   3. For DYNAMIC alive: conditional branches survive but inlined bodies
//      live in the branches; for loop-invariant alive: loop unswitching
//      may produce specialized variants

#include <stdint.h>
#include <stddef.h>

// ============================================================
// Variant A: All instances always alive (constant true)
//   Expected: scheduler optimizes to a flat unified kernel.
//   Slot load in filter should be eliminated (GVN sees the
//   just-written value from osc).
// ============================================================

static inline __attribute__((always_inline))
void osc_A(double * restrict slots, double * restrict regs) {
    double freq = slots[0];                  // freq input slot
    double phase = regs[0];
    double sample = phase;
    slots[1] = sample;                       // osc.out
    regs[0] = phase + freq * 0.0001;
}

static inline __attribute__((always_inline))
void filter_A(double * restrict slots, double * restrict regs) {
    double in = slots[1];                    // reads osc.out
    double state = regs[1];
    double out = in * 0.5 + state * 0.5;
    slots[2] = out;                          // filter.out
    regs[1] = out;
}

void scheduler_A_always_on(
    double * restrict slots,
    double * restrict regs,
    float  * restrict output,
    int32_t block_size
) {
    // alive is statically true here (literal 1). Both calls
    // unconditional. Expected: fully optimized like a unified kernel.
    for (int32_t s = 0; s < block_size; s++) {
        if (1) osc_A(slots, regs);
        if (1) filter_A(slots, regs);
        output[s] = (float) slots[2];
    }
}

// ============================================================
// Variant B: Dynamic alive — alive read from a slot each sample
//   Expected: conditional branches survive. Inlined bodies live
//   inside the branches. Filter's load of slots[1] can NOT be
//   eliminated because osc may not have run this sample.
// ============================================================

static inline __attribute__((always_inline))
void osc_B(double * restrict slots, double * restrict regs) {
    double freq = slots[0];
    double phase = regs[0];
    double sample = phase;
    slots[1] = sample;
    regs[0] = phase + freq * 0.0001;
}

static inline __attribute__((always_inline))
void filter_B(double * restrict slots, double * restrict regs) {
    double in = slots[1];
    double state = regs[1];
    double out = in * 0.5 + state * 0.5;
    slots[2] = out;
    regs[1] = out;
}

void scheduler_B_dynamic_alive(
    double * restrict slots,
    double * restrict regs,
    float  * restrict output,
    int32_t block_size
) {
    for (int32_t s = 0; s < block_size; s++) {
        // Alive read per-sample from the slot array. Could change
        // sample-to-sample. LLVM has to keep the branch.
        if (slots[3] > 0.5) osc_B(slots, regs);
        if (slots[4] > 0.5) filter_B(slots, regs);
        output[s] = (float) slots[2];
    }
}

// ============================================================
// Variant C: Loop-invariant alive — alive read once outside
// the sample loop. Expected: LLVM unswitches, producing
// specialized loop bodies (or hoists the branch entirely).
// ============================================================

static inline __attribute__((always_inline))
void osc_C(double * restrict slots, double * restrict regs) {
    double freq = slots[0];
    double phase = regs[0];
    double sample = phase;
    slots[1] = sample;
    regs[0] = phase + freq * 0.0001;
}

static inline __attribute__((always_inline))
void filter_C(double * restrict slots, double * restrict regs) {
    double in = slots[1];
    double state = regs[1];
    double out = in * 0.5 + state * 0.5;
    slots[2] = out;
    regs[1] = out;
}

void scheduler_C_per_block_alive(
    double * restrict slots,
    double * restrict regs,
    float  * restrict output,
    int32_t block_size
) {
    // Alive sampled ONCE at block start. Loop-invariant inside.
    int alive_osc    = slots[3] > 0.5;
    int alive_filter = slots[4] > 0.5;
    for (int32_t s = 0; s < block_size; s++) {
        if (alive_osc)    osc_C(slots, regs);
        if (alive_filter) filter_C(slots, regs);
        output[s] = (float) slots[2];
    }
}

// ============================================================
// Variant D: Mixed — osc always on, filter dynamically gated.
//   Tests whether GVN can eliminate the load of slots[1] in
//   filter even when filter is conditional (osc's write to
//   slots[1] dominates the filter call site).
// ============================================================

static inline __attribute__((always_inline))
void osc_D(double * restrict slots, double * restrict regs) {
    double freq = slots[0];
    double phase = regs[0];
    double sample = phase;
    slots[1] = sample;
    regs[0] = phase + freq * 0.0001;
}

static inline __attribute__((always_inline))
void filter_D(double * restrict slots, double * restrict regs) {
    double in = slots[1];                   // Should still see osc's write
    double state = regs[1];
    double out = in * 0.5 + state * 0.5;
    slots[2] = out;
    regs[1] = out;
}

void scheduler_D_mixed(
    double * restrict slots,
    double * restrict regs,
    float  * restrict output,
    int32_t block_size
) {
    for (int32_t s = 0; s < block_size; s++) {
        osc_D(slots, regs);                  // unconditional
        if (slots[4] > 0.5) filter_D(slots, regs);
        output[s] = (float) slots[2];
    }
}
