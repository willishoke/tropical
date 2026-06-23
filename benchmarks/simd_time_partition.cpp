// simd_time_partition.cpp
//
// Does partitioning a block of time across SIMD lanes beat computing samples one at
// a time, for a closed-form-in-τ kernel (no per-sample state)? Every output sample
// is an independent function of τ, so W consecutive samples can be computed in W
// lanes at once. Three paths:
//
//   R  τ as a per-sample recurrence (∫velocity through a state slot) — one sample at
//      a time, the way tropical's runtime and Faust evaluate TODAY. The loop-carried
//      state forbids vectorizing across time. This is the honest baseline.
//   C  τ unrolled to a closed form (the grade-aware move), loop vectorizer left ON —
//      once the recurrence is gone, the compiler can take the time axis itself.
//   B  τ unrolled + explicit SIMD: W τ's per step, lanes partition the time block.
//
// R is what an engine that doesn't know τ is control-rate emits; C and B both require
// the grade-aware τ-unroll first. The honest win is C/B over R.
//
// Kernel = the reversible-synth shape: a modal voice through K future taps,
//   out(τ) = Σ_{k<K} g^k · Σ_{m<M} A_m · sin(2π f_m (τ + k·D))
// Build & run:  make run   (builds float + double)

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <algorithm>

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

static constexpr int    W      = 8;        // logical SIMD lanes (W consecutive τ)
static constexpr int    M      = BENCH_M;  // modal partials  (a parallel axis the compiler can take)
static constexpr int    K      = BENCH_K;  // reverse-reverb taps
static constexpr int    N      = 1 << 20;  // samples per pass (multiple of W)
static constexpr int    TRIALS = 7;
static constexpr double SR     = 48000.0;

typedef Float   VF __attribute__((vector_size(W * sizeof(Float))));
typedef int32_t VI __attribute__((vector_size(W * sizeof(int32_t))));

static inline Float to_nearest(Float x) { return (Float)(int32_t)(x + (Float)0.5); }
static inline VF    to_nearest(VF x) {
    VI i = __builtin_convertvector(x + (Float)0.5, VI);
    return __builtin_convertvector(i, VF);
}

// sin(2π·cycles), range-reduce to [-0.5,0.5] turns + odd Taylor-9. identical ops
// for Float and VF, so scalar and vector agree to FP rounding (and to the bit with
// fp-contract off).
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
static void init_patch() {
    for (int m = 0; m < M; m++) { F[m] = (Float)(110.0 * (m + 1) * 1.0009); A[m] = (Float)(1.0 / (m + 1)); }
    Float g = (Float)0.72, gk = (Float)1.0;
    for (int k = 0; k < K; k++) { Gk[k] = gk; gk *= g; }
    D = (Float)0.045; // 45 ms tap spacing
}

static inline Float kernel_scalar(Float tau) {
    Float acc = 0;
    for (int k = 0; k < K; k++) {
        Float tk = tau + (Float)k * D, voice = 0;
        for (int m = 0; m < M; m++) voice += A[m] * sin_turns(F[m] * tk);
        acc += Gk[k] * voice;
    }
    return acc;
}
static inline VF kernel_simd(VF tau) {
    VF acc = tau - tau;                     // zero vector
    for (int k = 0; k < K; k++) {
        VF tk = tau + (Float)k * D, voice = tau - tau;
        for (int m = 0; m < M; m++) voice += A[m] * sin_turns(F[m] * tk);
        acc += Gk[k] * voice;
    }
    return acc;
}

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

int main() {
    init_patch();
    Float* out_rec  = (Float*)std::aligned_alloc(64, N * sizeof(Float));
    Float* out_simd = (Float*)std::aligned_alloc(64, N * sizeof(Float));
    Float* out_auto = (Float*)std::aligned_alloc(64, N * sizeof(Float));
    const Float dt = (Float)(1.0 / SR);

    VF iota; for (int j = 0; j < W; j++) iota[j] = (Float)j;

    double t_rec = 1e30, t_simd = 1e30, t_auto = 1e30;
    for (int trial = 0; trial < TRIALS; trial++) {
        // R — engine-faithful baseline: τ as a per-sample recurrence (∫velocity
        // through a state slot). The loop-carried state forbids vectorizing across
        // time, so it runs one sample at a time — what tropical and Faust do today.
        double r = now_ms();
        { volatile Float tau_state = 0; const Float vstep = dt;
          for (int n = 0; n < N; n++) {
              Float tau = tau_state;
              out_rec[n] = kernel_scalar(tau);
              tau_state = tau + vstep;          // writeback through volatile -> serial
          } }
        t_rec = std::min(t_rec, now_ms() - r);

        // C — grade-aware τ-unroll (closed form), loop vectorizer left ON: with the
        // recurrence gone, the compiler can take the time axis on its own.
        double c = now_ms();
        for (int n = 0; n < N; n++) out_auto[n] = kernel_scalar((Float)n * dt);
        t_auto = std::min(t_auto, now_ms() - c);

        // B — grade-aware τ-unroll + explicit SIMD: W τ's per step, lanes split time.
        double b = now_ms();
        for (int n = 0; n < N; n += W) {
            VF tau = ((Float)n + iota) * dt;
            *reinterpret_cast<VF*>(&out_simd[n]) = kernel_simd(tau);
        }
        t_simd = std::min(t_simd, now_ms() - b);
    }

    double dbc = 0, drift = 0, sum = 0;
    for (int n = 0; n < N; n++) {
        dbc   = std::max(dbc,   std::fabs((double)out_simd[n] - (double)out_auto[n]));
        drift = std::max(drift, std::fabs((double)out_rec[n]  - (double)out_auto[n]));
        sum  += (double)out_auto[n];
    }

    printf("element %-6s  kernel %dx%d = %d sin/sample  %d samples/pass  W=%d\n\n",
           sizeof(Float) == 4 ? "float" : "double", M, K, M * K, N, W);
    printf("  R  recurrent τ, one sample/time : %8.3f ms  %7.1f Msamp/s   (engine today)\n",
           t_rec, N / t_rec / 1e3);
    printf("  C  τ-unroll, auto-vec ON        : %8.3f ms  %7.1f Msamp/s   %.2fx vs R\n",
           t_auto, N / t_auto / 1e3, t_rec / t_auto);
    printf("  B  τ-unroll, explicit SIMD      : %8.3f ms  %7.1f Msamp/s   %.2fx vs R\n",
           t_simd, N / t_simd / 1e3, t_rec / t_simd);
    printf("\n  B==C bit-exact: %.1e   recurrent-τ drift from closed form: %.1e   (checksum %.4f)\n",
           dbc, drift, sum);

    std::free(out_rec); std::free(out_simd); std::free(out_auto);
    return 0;
}
