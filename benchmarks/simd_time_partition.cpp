// simd_time_partition.cpp
//
// Does partitioning a block of time across SIMD lanes beat computing samples one
// at a time, for a STATELESS per-sample kernel? A stateless kernel carries no
// inter-sample state, so each output sample is a pure function of its time index
// t — consecutive samples are independent and W of them can be computed in W
// lanes at once. Three paths:
//
//   R  t as a per-sample recurrence (a state slot, t += dt each sample) — one
//      sample at a time. The loop-carried state forbids vectorizing across time;
//      this is the baseline a stateful per-sample evaluator gives.
//   C  t as a closed form (t = n*dt), loop vectorizer left ON — once the
//      recurrence is gone the compiler can take the time axis itself.
//   B  t closed form + explicit SIMD: W t's per step, lanes split the block.
//
// Kernel: out(t) = sum_{k<K} g^k * sum_{m<M} A_m * sin(2*pi*f_m*(t + k*D)) —
// M sinusoidal partials summed at K offset positions, M*K sines per sample. M and
// K are the arithmetic-density knob (set via -DBENCH_M / -DBENCH_K).
//
// Method: each path is timed over TRIALS runs after WARMUP untimed runs, with the
// caches flushed before every run (a >LLC scratch write) so nothing starts hot.
// We report the median (robust to scheduler jitter), the min (best case), and the
// stddev. -ffp-contract=off pins every path to the same mul-add shape so C and B
// come out bit-identical.
//
// Build & run:  make run   (float + double + a single-partial "thin" variant)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef BENCH_T
#define BENCH_T float
#endif
#ifndef BENCH_M
#define BENCH_M 8
#endif
#ifndef BENCH_K
#define BENCH_K 8
#endif
using Float = BENCH_T;

static constexpr int    W      = 8;        // logical SIMD lanes (W consecutive t)
static constexpr int    M      = BENCH_M;  // partials (a within-sample parallel axis)
static constexpr int    K      = BENCH_K;  // offset positions
static constexpr int    N      = 1 << 20;  // samples per pass (multiple of W)
static constexpr int    WARMUP = 3;
static constexpr int    TRIALS = 15;
static constexpr double SR     = 48000.0;

typedef Float   VF __attribute__((vector_size(W * sizeof(Float))));
typedef int32_t VI __attribute__((vector_size(W * sizeof(int32_t))));

static inline Float to_nearest(Float x) { return (Float)(int32_t)(x + (Float)0.5); }
static inline VF    to_nearest(VF x) {
    VI i = __builtin_convertvector(x + (Float)0.5, VI);
    return __builtin_convertvector(i, VF);
}

// sin(2*pi*cycles): range-reduce to [-0.5,0.5] turns + odd Taylor-9. Identical
// ops for Float and VF, so scalar and vector agree to FP rounding (and to the bit
// with fp-contract off).
template <class T> static inline T sin_turns(T cycles) {
    T r  = cycles - to_nearest(cycles);
    T x  = r * (Float)6.283185307179586;
    T x2 = x * x;
    T p  = x2 * (Float)0.0 + (Float)(1.0 / 362880.0);  // broadcast leading coeff
    p = (Float)(-1.0 / 5040.0) + x2 * p;
    p = (Float)( 1.0 /  120.0) + x2 * p;
    p = (Float)(-1.0 /    6.0) + x2 * p;
    p = (Float)(  1.0        ) + x2 * p;
    return x * p;
}

static Float F[M], A[M], Gk[K], D;
static void init_kernel() {
    for (int m = 0; m < M; m++) { F[m] = (Float)(110.0 * (m + 1) * 1.0009); A[m] = (Float)(1.0 / (m + 1)); }
    Float g = (Float)0.72, gk = (Float)1.0;
    for (int k = 0; k < K; k++) { Gk[k] = gk; gk *= g; }
    D = (Float)0.045;  // 45 ms offset spacing
}

static inline Float kernel_scalar(Float t) {
    Float acc = 0;
    for (int k = 0; k < K; k++) {
        Float tk = t + (Float)k * D, partials = 0;
        for (int m = 0; m < M; m++) partials += A[m] * sin_turns(F[m] * tk);
        acc += Gk[k] * partials;
    }
    return acc;
}
static inline VF kernel_simd(VF t) {
    VF acc = t - t;                         // zero vector
    for (int k = 0; k < K; k++) {
        VF tk = t + (Float)k * D, partials = t - t;
        for (int m = 0; m < M; m++) partials += A[m] * sin_turns(F[m] * tk);
        acc += Gk[k] * partials;
    }
    return acc;
}

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// Evict the caches before each timed run: stream a buffer larger than any
// last-level cache so no path starts with its output (or coefficients) resident.
// The volatile sink defeats dead-store elimination of the eviction itself.
static char* g_flush = nullptr;
static constexpr size_t FLUSH_BYTES = 64u << 20;  // 64 MB > LLC
static volatile char g_sink = 0;
static void flush_caches() {
    char s = g_sink;
    for (size_t i = 0; i < FLUSH_BYTES; i += 64) { g_flush[i] = (char)(i ^ s); s ^= g_flush[i]; }
    g_sink = s;
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
static double vmin(const std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); }
static double stddev(const std::vector<double>& v) {
    double mean = 0; for (double x : v) mean += x; mean /= (double)v.size();
    double s = 0; for (double x : v) s += (x - mean) * (x - mean);
    return std::sqrt(s / (double)v.size());
}

int main() {
    init_kernel();
    g_flush = (char*)std::malloc(FLUSH_BYTES);
    Float* out_rec  = (Float*)std::aligned_alloc(64, N * sizeof(Float));
    Float* out_simd = (Float*)std::aligned_alloc(64, N * sizeof(Float));
    Float* out_auto = (Float*)std::aligned_alloc(64, N * sizeof(Float));
    const Float dt = (Float)(1.0 / SR);

    VF iota; for (int j = 0; j < W; j++) iota[j] = (Float)j;

    std::vector<double> ts_rec, ts_auto, ts_simd;
    for (int trial = 0; trial < WARMUP + TRIALS; trial++) {
        // R — stateful baseline: t as a per-sample recurrence through a state
        // slot. The loop-carried writeback forbids vectorizing across time, so it
        // runs one sample at a time.
        flush_caches();
        double tr = now_ms();
        { volatile Float t_state = 0; const Float vstep = dt;
          for (int n = 0; n < N; n++) {
              Float t = t_state;
              out_rec[n] = kernel_scalar(t);
              t_state = t + vstep;          // writeback through volatile → serial
          } }
        const double d_rec = now_ms() - tr;

        // C — closed-form t, loop vectorizer ON: with the recurrence gone, the
        // compiler can take the time axis on its own.
        flush_caches();
        double tc = now_ms();
        for (int n = 0; n < N; n++) out_auto[n] = kernel_scalar((Float)n * dt);
        const double d_auto = now_ms() - tc;

        // B — closed-form t + explicit SIMD: W t's per step, lanes split time.
        flush_caches();
        double tb = now_ms();
        for (int n = 0; n < N; n += W) {
            VF t = ((Float)n + iota) * dt;
            *reinterpret_cast<VF*>(&out_simd[n]) = kernel_simd(t);
        }
        const double d_simd = now_ms() - tb;

        if (trial >= WARMUP) { ts_rec.push_back(d_rec); ts_auto.push_back(d_auto); ts_simd.push_back(d_simd); }
    }

    double dbc = 0, drift = 0, sum = 0;
    for (int n = 0; n < N; n++) {
        dbc   = std::max(dbc,   std::fabs((double)out_simd[n] - (double)out_auto[n]));
        drift = std::max(drift, std::fabs((double)out_rec[n]  - (double)out_auto[n]));
        sum  += (double)out_auto[n];
    }

    const double mr = median(ts_rec), ma = median(ts_auto), mb = median(ts_simd);
    printf("element %-6s  kernel %dx%d = %d sin/sample  %d samples/pass  W=%d  (median of %d, %d warmup)\n\n",
           sizeof(Float) == 4 ? "float" : "double", M, K, M * K, N, W, TRIALS, WARMUP);
    printf("  R  recurrent t, one sample/time : %7.2f ms  (min %7.2f, sd %.2f)  %7.1f Msamp/s   (stateful baseline)\n",
           mr, vmin(ts_rec), stddev(ts_rec), N / mr / 1e3);
    printf("  C  closed-form t, auto-vec ON   : %7.2f ms  (min %7.2f, sd %.2f)  %7.1f Msamp/s   %.2fx vs R\n",
           ma, vmin(ts_auto), stddev(ts_auto), N / ma / 1e3, mr / ma);
    printf("  B  closed-form t, explicit SIMD : %7.2f ms  (min %7.2f, sd %.2f)  %7.1f Msamp/s   %.2fx vs R\n",
           mb, vmin(ts_simd), stddev(ts_simd), N / mb / 1e3, mr / mb);
    printf("\n  B==C bit-exact: %.1e   recurrent-t drift from closed form: %.1e   (checksum %.4f)\n",
           dbc, drift, sum);

    std::free(out_rec); std::free(out_simd); std::free(out_auto); std::free(g_flush);
    return 0;
}
