# benchmarks

Standalone microbenchmarks and LLVM spikes that inform the engine's architecture.
Nothing here depends on the engine: the microbenchmark is a single self-contained
translation unit you can `make run`, and `llvm/` is a playground of focused
experiments with recorded findings.

- **`simd_time_partition`** (below) — the grade-aware τ-unroll vs the recurrent
  baseline, on CPU.
- **`gpu_time_partition/`** — the same time-partition axis taken to the GPU (Apple
  Silicon / Metal, UMA): is a per-block dispatch realtime-feasible? See its `findings.md`.
- **`pure_kernel_partition/`** — liveness reuse versus a pure multi-kernel split when
  a graph's cooperative working set crosses Metal's per-kernel memory ceiling.
- **`llvm/`** — LLVM-level optimization spikes (active-set inlining, compile-time
  scaling, cross-module inlining) that backed the fractal-compilation choices. See
  `llvm/README.md` for the index.

## simd_time_partition

For a closed-form-in-τ kernel, every output sample is an independent function of τ, so
`W` consecutive samples can be computed in `W` SIMD lanes at once. The kernel is the
reversible-synth shape — a modal voice through `K` future taps,
`out(τ) = Σ_k gᵏ·Σ_m A_m·sin(2π f_m (τ + kD))`. Three paths:

```
R  τ as a per-sample recurrence (∫velocity through a state slot) — one sample at a
   time, the way tropical's runtime and Faust evaluate TODAY.  ← the honest baseline
C  τ unrolled to a closed form (the grade-aware move), loop vectorizer left ON
B  τ unrolled + explicit SIMD: W τ's per step, lanes partition the time block
```

`R` is the baseline because both engines keep `τ` recurrent, and a loop-carried
recurrence is exactly what forbids the compiler from vectorizing across time. `C` and
`B` both require the grade-aware `τ`-unroll first — that is the move under test.

```
make run        # builds float, double, and a single-oscillator (thin) variant
```

### Results — Apple M1 Pro (arm64 / NEON, 4-wide float)

Rich kernel — 8 partials × 8 taps (64 sin/sample):

```
                                   float               double
  R  recurrent τ (engine today)  39.9 ms             83.9 ms
  C  τ-unroll, auto-vec ON         "   (1.01x)          "   (1.01x)
  B  τ-unroll, explicit SIMD     33.5 ms  (1.19x)     67.8 ms  (1.24x)
```

Single oscillator — 1 partial × 1 tap, where time is the *only* parallel axis:

```
  R  recurrent τ (engine today)   3.87 ms
  C  τ-unroll, auto-vec ON        0.63 ms  (6.12x)
  B  τ-unroll, explicit SIMD      0.71 ms  (5.42x)
```

`B` and `C` are bit-identical. The float recurrent-`τ` drifts **6.8** from the closed
form over the 1M-sample render (`drift` line); double drifts 2.6e-6.

### What this shows (the fair read)

1. **The recurrent baseline already vectorizes *within* a sample** — across the 8
   partials. So on a partial-rich kernel the SIMD lanes are spent regardless, and the
   `τ`-unroll stacks only ~1.2× on top by *also* using the time axis. On a lean kernel
   (one voice) the recurrent form leaves the width entirely unused **and** pays the
   per-sample state round-trip, so the unroll reclaims ~5–6×.
2. **The win is the `τ`-unroll, not the hand-written SIMD.** Once `τ` is a closed form
   the compiler takes the time axis on its own (`C`); on the lean kernel it even beats
   the explicit version (`B`). The grade analysis — which licenses replacing the
   `τ = ∫velocity` recurrence with the control-rate closed form — is the whole move.
   An engine that keeps `τ` recurrent (tropical's runtime today, Faust) can't get any
   of this: there is no time axis for any backend to take.
3. **Accuracy, for free.** Phase accumulation in `float` decorrelates over a long
   render (drift 6.8 here); the closed form is exact at every sample. The closed form
   isn't just sometimes-faster — it's *correct* where the accumulator quietly isn't.

So the honest CPU read: grade-aware time-partition is a real but kernel-dependent win
(modest when partials soak the lanes, several-× when time is the only axis), plus a
real accuracy win. The dramatic parallelism — thousands of lanes, combinational
silicon — is the FPGA story this CPU bench structurally can't reach.
