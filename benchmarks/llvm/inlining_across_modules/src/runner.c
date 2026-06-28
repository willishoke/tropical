// runner.c — driver that times the scheduler entry points.
//
// Links against whichever scheduler object file(s) we choose to
// build with. The variant selection happens at link time, not at
// runtime — each binary (v1_runner, v3_runner, etc.) links a
// different set of object files but uses identical timing logic.
//
// The variant being benchmarked is determined by a compile-time
// macro (BENCH_VARIANT) and the corresponding scheduler symbol it
// imports.

#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Chain length is selected at compile time via BENCH_CHAIN_LEN. The
// runner has two parallel sets of scheduler signatures and timing
// paths: 4-kernel (existing) and 16-kernel (Phaser16-scale).
#ifndef BENCH_CHAIN_LEN
#define BENCH_CHAIN_LEN 4
#endif

// ─── Variant + chain-length dispatch ───────────────────────────────────────
//
// Each chain-length has its own scheduler signature (fixed-arity in
// the kernel pointers). The chain-length is selected at compile time
// via BENCH_CHAIN_LEN; the variant is selected via BENCH_VARIANT_V*.
// Combining them gives 4 variants × 2 chain lengths = 8 binaries.

#ifdef BENCH_VARIANT_V1LTO
#define BENCH_VARIANT_V1
#define BENCH_LTO_SUFFIX "_lto"
#else
#define BENCH_LTO_SUFFIX ""
#endif

#if BENCH_CHAIN_LEN == 4

#ifdef BENCH_VARIANT_V1
extern void scheduler_v1(
    const double *, double *, size_t,
    kernel_state_t *, kernel_state_t *, kernel_state_t *, kernel_state_t *);
#define BENCH_NAME "v1_chain4" BENCH_LTO_SUFFIX
#define BENCH_FN scheduler_v1
#elif defined(BENCH_VARIANT_V3)
extern void scheduler_v3(
    const double *, double *, size_t,
    kernel_state_t *, kernel_state_t *, kernel_state_t *, kernel_state_t *);
#define BENCH_NAME "v3_chain4_nolto"
#define BENCH_FN scheduler_v3
#elif defined(BENCH_VARIANT_V4)
extern void scheduler_v3(
    const double *, double *, size_t,
    kernel_state_t *, kernel_state_t *, kernel_state_t *, kernel_state_t *);
#define BENCH_NAME "v4_chain4_lto"
#define BENCH_FN scheduler_v3
#else
#error "Define BENCH_VARIANT_V1/V1LTO/V3/V4"
#endif

#elif BENCH_CHAIN_LEN == 16

#ifdef BENCH_VARIANT_V1
extern void scheduler_v1_16(
    const double *, double *, size_t,
    kernel_state_t *, kernel_state_t *, kernel_state_t *, kernel_state_t *,
    kernel_state_t *, kernel_state_t *, kernel_state_t *, kernel_state_t *,
    kernel_state_t *, kernel_state_t *, kernel_state_t *, kernel_state_t *,
    kernel_state_t *, kernel_state_t *, kernel_state_t *, kernel_state_t *);
#define BENCH_NAME "v1_chain16" BENCH_LTO_SUFFIX
#define BENCH_FN scheduler_v1_16
#elif defined(BENCH_VARIANT_V3)
extern void scheduler_v3_16(
    const double *, double *, size_t,
    kernel_state_t *, kernel_state_t *, kernel_state_t *, kernel_state_t *,
    kernel_state_t *, kernel_state_t *, kernel_state_t *, kernel_state_t *,
    kernel_state_t *, kernel_state_t *, kernel_state_t *, kernel_state_t *,
    kernel_state_t *, kernel_state_t *, kernel_state_t *, kernel_state_t *);
#define BENCH_NAME "v3_chain16_nolto"
#define BENCH_FN scheduler_v3_16
#elif defined(BENCH_VARIANT_V4)
extern void scheduler_v3_16(
    const double *, double *, size_t,
    kernel_state_t *, kernel_state_t *, kernel_state_t *, kernel_state_t *,
    kernel_state_t *, kernel_state_t *, kernel_state_t *, kernel_state_t *,
    kernel_state_t *, kernel_state_t *, kernel_state_t *, kernel_state_t *,
    kernel_state_t *, kernel_state_t *, kernel_state_t *, kernel_state_t *);
#define BENCH_NAME "v4_chain16_lto"
#define BENCH_FN scheduler_v3_16
#else
#error "Define BENCH_VARIANT_V1/V1LTO/V3/V4"
#endif

#else
#error "BENCH_CHAIN_LEN must be 4 or 16"
#endif

static double seconds_since(const struct timespec *t0) {
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (t1.tv_sec - t0->tv_sec) + 1e-9 * (t1.tv_nsec - t0->tv_nsec);
}

int main(int argc, char **argv) {
    // Defaults — overridable via argv for sweeping.
    size_t buf_len = 4096;
    int    warmup_iters = 16;
    int    measured_iters = 256;

    if (argc > 1) buf_len        = (size_t)strtoul(argv[1], NULL, 10);
    if (argc > 2) warmup_iters   = atoi(argv[2]);
    if (argc > 3) measured_iters = atoi(argv[3]);

    double *input  = aligned_alloc(64, buf_len * sizeof(double));
    double *output = aligned_alloc(64, buf_len * sizeof(double));

    // Initialise the input with a deterministic but non-trivial signal
    // so the optimizer can't fold the whole computation to a constant.
    for (size_t i = 0; i < buf_len; i++) {
        input[i] = 0.5 * (double)((i * 1664525u + 1013904223u) >> 24) / 256.0;
    }

    // Kernel state — pick coefficients that don't trigger numerical
    // instability over the test duration. The state DC value matters
    // (cumulative drift would push us into infinity); a < 1 keeps it
    // bounded.
    kernel_state_t ks[BENCH_CHAIN_LEN];
    for (int i = 0; i < BENCH_CHAIN_LEN; i++) {
        // Decay a coefficient with kernel index so the chain is stable
        // (geometric decrease in gain through the chain).
        ks[i].a = 0.5 / ((double)(i + 1));
        ks[i].b = 0.01 * (double)(i + 1);
        ks[i].state = 0.0;
    }

#if BENCH_CHAIN_LEN == 4
#define CALL_BENCH() BENCH_FN(input, output, buf_len, \
    &ks[0], &ks[1], &ks[2], &ks[3])
#elif BENCH_CHAIN_LEN == 16
#define CALL_BENCH() BENCH_FN(input, output, buf_len, \
    &ks[0], &ks[1], &ks[2], &ks[3], &ks[4], &ks[5], &ks[6], &ks[7], \
    &ks[8], &ks[9], &ks[10], &ks[11], &ks[12], &ks[13], &ks[14], &ks[15])
#endif

    // Warm-up: prime caches; let CPU governor settle.
    for (int it = 0; it < warmup_iters; it++) {
        CALL_BENCH();
    }

    // Measured: time M iterations, report ns/sample (total samples =
    // M * buf_len). Reset states before each iteration so behavior
    // is reproducible across runs.
    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int it = 0; it < measured_iters; it++) {
        CALL_BENCH();
    }

    double elapsed_s = seconds_since(&t0);
    size_t total_samples = (size_t)measured_iters * buf_len;
    double ns_per_sample = (elapsed_s * 1e9) / (double)total_samples;

    // Read a value from `output` so the optimizer can't dead-code-
    // eliminate the whole loop.
    double sink = output[buf_len - 1];

    // CSV row: variant, buf_len, iters, ns_per_sample, sink
    // (Header printed only if BENCH_PRINT_HEADER is set in env.)
    if (getenv("BENCH_PRINT_HEADER")) {
        printf("variant,buf_len,iters,ns_per_sample,sink_value\n");
    }
    printf("%s,%zu,%d,%.3f,%.6f\n",
           BENCH_NAME, buf_len, measured_iters, ns_per_sample, sink);

    free(input);
    free(output);
    return 0;
}
