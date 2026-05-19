// current_vs_proposed.c — compare what O2 does with:
//   (a) a single flat function (mirrors current JIT output)
//   (b) the proposed multi-function form with alwaysinline + conditional dispatch
//
// Specifically check: does mem2reg+LICM phi-promote the reg state
// through the sample loop in either case? If yes, O2 is fine. If no,
// we need O3 or extra passes.

#include <stdint.h>

// ============================================================
// (a) CURRENT JIT FORM: single flat function, all instructions
//     inlined into one body. This mirrors what compile_resolved
//     produces today for a SinOsc → OnePole patch.
// ============================================================

void current_unified_kernel(
    double * restrict slots,   // slot values (inter-module)
    double * restrict regs,    // state registers (per-instance)
    float  * restrict output,
    int32_t block_size
) {
    // Inlined "instructions" the JIT would emit for: SinOsc → OnePole → out
    for (int32_t s = 0; s < block_size; s++) {
        // SinOsc instructions: read freq from slots[0], phase from regs[0]
        double freq  = slots[0];
        double phase = regs[0];
        double osc_out = phase;             // "sin" placeholder
        // Write osc output to slots[1] (the proposed slot-mode form)
        slots[1] = osc_out;
        // Update phase
        regs[0] = phase + freq * 0.0001;

        // OnePole instructions: read input from slots[1], state from regs[1]
        double fin  = slots[1];
        double fstate = regs[1];
        double fout = fin * 0.5 + fstate * 0.5;
        slots[2] = fout;
        regs[1] = fout;

        output[s] = (float) slots[2];
    }
}

// ============================================================
// (b) PROPOSED MULTI-FUNCTION FORM: separate alwaysinline functions
//     + scheduler with conditional dispatch.
// ============================================================

static inline __attribute__((always_inline))
void osc_instance(double * restrict slots, double * restrict regs) {
    double freq = slots[0];
    double phase = regs[0];
    slots[1] = phase;
    regs[0] = phase + freq * 0.0001;
}

static inline __attribute__((always_inline))
void onepole_instance(double * restrict slots, double * restrict regs) {
    double in = slots[1];
    double state = regs[1];
    double out = in * 0.5 + state * 0.5;
    slots[2] = out;
    regs[1] = out;
}

void proposed_scheduler_always_on(
    double * restrict slots,
    double * restrict regs,
    float  * restrict output,
    int32_t block_size
) {
    for (int32_t s = 0; s < block_size; s++) {
        if (1) osc_instance(slots, regs);
        if (1) onepole_instance(slots, regs);
        output[s] = (float) slots[2];
    }
}

void proposed_scheduler_dynamic(
    double * restrict slots,
    double * restrict regs,
    float  * restrict output,
    int32_t block_size
) {
    for (int32_t s = 0; s < block_size; s++) {
        if (slots[3] > 0.5) osc_instance(slots, regs);
        if (slots[4] > 0.5) onepole_instance(slots, regs);
        output[s] = (float) slots[2];
    }
}
