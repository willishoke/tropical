// runner_alive.c — timing harness for the alive-aware variants.
//
// Parallel to runner.c but uses kernel_alive_state_t and the alive-
// aware scheduler signatures. Chain length 4 only (for now).
//
// Build-time variant selection via BENCH_VARIANT_V*_ALIVE. The
// BENCH_NAME baked into the output identifies the variant in the CSV.

#include "kernel_alive.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef BENCH_VARIANT_V1LTO_ALIVE
#define BENCH_VARIANT_V1_ALIVE
#define BENCH_LTO_SUFFIX "_lto"
#else
#define BENCH_LTO_SUFFIX ""
#endif

#ifdef BENCH_VARIANT_V1_ALIVE
extern void scheduler_v1_alive(
    const double *, double *, size_t,
    kernel_alive_state_t *, kernel_alive_state_t *,
    kernel_alive_state_t *, kernel_alive_state_t *);
#define BENCH_NAME "v1_chain4_alive" BENCH_LTO_SUFFIX
#define BENCH_FN scheduler_v1_alive
#elif defined(BENCH_VARIANT_V3_ALIVE)
extern void scheduler_v3_alive(
    const double *, double *, size_t,
    kernel_alive_state_t *, kernel_alive_state_t *,
    kernel_alive_state_t *, kernel_alive_state_t *);
#define BENCH_NAME "v3_chain4_alive_nolto"
#define BENCH_FN scheduler_v3_alive
#elif defined(BENCH_VARIANT_V4_ALIVE)
extern void scheduler_v3_alive(
    const double *, double *, size_t,
    kernel_alive_state_t *, kernel_alive_state_t *,
    kernel_alive_state_t *, kernel_alive_state_t *);
#define BENCH_NAME "v4_chain4_alive_lto"
#define BENCH_FN scheduler_v3_alive
#else
#error "Define BENCH_VARIANT_V1_ALIVE / V1LTO_ALIVE / V3_ALIVE / V4_ALIVE"
#endif

static double seconds_since(const struct timespec *t0) {
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (t1.tv_sec - t0->tv_sec) + 1e-9 * (t1.tv_nsec - t0->tv_nsec);
}

int main(int argc, char **argv) {
    size_t buf_len = 4096;
    int    warmup_iters = 16;
    int    measured_iters = 256;

    if (argc > 1) buf_len        = (size_t)strtoul(argv[1], NULL, 10);
    if (argc > 2) warmup_iters   = atoi(argv[2]);
    if (argc > 3) measured_iters = atoi(argv[3]);

    double *input  = aligned_alloc(64, buf_len * sizeof(double));
    double *output = aligned_alloc(64, buf_len * sizeof(double));

    for (size_t i = 0; i < buf_len; i++) {
        input[i] = 0.5 * (double)((i * 1664525u + 1013904223u) >> 24) / 256.0;
    }

    // Initial alive=0.0 — the scheduler writes 1.0 every sample, so
    // the initial value is irrelevant for the body. Matching the
    // active-set spike: slot defaults to 1.0 at allocation in
    // tropical's runtime; here we test the "scheduler-writes-1.0-
    // every-sample" pattern that GVN should be able to fold.
    kernel_alive_state_t ks[4];
    for (int i = 0; i < 4; i++) {
        ks[i].a = 0.5 / ((double)(i + 1));
        ks[i].b = 0.01 * (double)(i + 1);
        ks[i].state = 0.0;
        ks[i].alive = 0.0;
    }

    for (int it = 0; it < warmup_iters; it++) {
        BENCH_FN(input, output, buf_len, &ks[0], &ks[1], &ks[2], &ks[3]);
    }

    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int it = 0; it < measured_iters; it++) {
        BENCH_FN(input, output, buf_len, &ks[0], &ks[1], &ks[2], &ks[3]);
    }

    double elapsed_s = seconds_since(&t0);
    size_t total_samples = (size_t)measured_iters * buf_len;
    double ns_per_sample = (elapsed_s * 1e9) / (double)total_samples;
    double sink = output[buf_len - 1];

    if (getenv("BENCH_PRINT_HEADER")) {
        printf("variant,buf_len,iters,ns_per_sample,sink_value\n");
    }
    printf("%s,%zu,%d,%.3f,%.6f\n",
           BENCH_NAME, buf_len, measured_iters, ns_per_sample, sink);

    free(input);
    free(output);
    return 0;
}
