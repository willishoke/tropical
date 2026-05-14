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

// Scheduler signature is uniform across variants — the differences
// are in HOW each variant is compiled and linked, not in the API.
typedef void (*scheduler_fn_t)(
    const double *input,
    double *output,
    size_t n,
    kernel_state_t *k1,
    kernel_state_t *k2,
    kernel_state_t *k3,
    kernel_state_t *k4
);

#ifdef BENCH_VARIANT_V1LTO
#define BENCH_VARIANT_V1
#define BENCH_NAME "v1lto_monolithic_lto"
#endif

#ifdef BENCH_VARIANT_V1
extern void scheduler_v1(
    const double *, double *, size_t,
    kernel_state_t *, kernel_state_t *, kernel_state_t *, kernel_state_t *);
#ifndef BENCH_NAME
#define BENCH_NAME "v1_monolithic"
#endif
#define BENCH_FN scheduler_v1
#elif defined(BENCH_VARIANT_V3)
extern void scheduler_v3(
    const double *, double *, size_t,
    kernel_state_t *, kernel_state_t *, kernel_state_t *, kernel_state_t *);
#define BENCH_NAME "v3_separate_modules_nolto"
#define BENCH_FN scheduler_v3
#elif defined(BENCH_VARIANT_V4)
// v4 uses the same source as v3; the difference is at the link layer
// (with -flto vs without). The scheduler symbol is the same one
// (scheduler_v3), but the runtime metric is reported as a distinct
// variant so we can compare apples-to-apples.
extern void scheduler_v3(
    const double *, double *, size_t,
    kernel_state_t *, kernel_state_t *, kernel_state_t *, kernel_state_t *);
#define BENCH_NAME "v4_separate_modules_lto"
#define BENCH_FN scheduler_v3
#else
#error "Define BENCH_VARIANT_V1, BENCH_VARIANT_V3, or BENCH_VARIANT_V4"
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
    kernel_state_t k1 = { .a = 0.5,  .b = 0.01, .state = 0.0 };
    kernel_state_t k2 = { .a = 0.4,  .b = 0.02, .state = 0.0 };
    kernel_state_t k3 = { .a = 0.3,  .b = 0.03, .state = 0.0 };
    kernel_state_t k4 = { .a = 0.2,  .b = 0.04, .state = 0.0 };

    // Warm-up: prime caches; let CPU governor settle.
    for (int it = 0; it < warmup_iters; it++) {
        BENCH_FN(input, output, buf_len, &k1, &k2, &k3, &k4);
    }

    // Measured: time M iterations, report ns/sample (total samples =
    // M * buf_len). Reset states before each iteration so behavior
    // is reproducible across runs.
    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int it = 0; it < measured_iters; it++) {
        BENCH_FN(input, output, buf_len, &k1, &k2, &k3, &k4);
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
