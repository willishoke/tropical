# benchmarks

Standalone microbenchmarks for the engine's hot paths. No engine dependency —
each is a single self-contained translation unit you can `make run`.

## simd_time_partition

A **stateless** per-sample kernel carries no inter-sample state, so each output
sample is a pure function of its time index `t`. Consecutive samples are therefore
independent and can be computed in parallel — `W` of them in `W` SIMD lanes at
once. This benchmark measures the **ideal speedup of partitioning a block of time
across SIMD lanes** versus computing one sample at a time, as a function of the
kernel's arithmetic density.

The kernel is `out(t) = Σ_{k<K} gᵏ · Σ_{m<M} A_m · sin(2π f_m (t + k·D))` — `M`
sinusoidal partials summed at `K` offset positions, `M·K` sines per sample. `M`
and `K` are the density knob.

Three paths:

```
R  t carried as a per-sample recurrence (a state slot, t += dt each sample) —
   one sample at a time. The loop-carried state forbids vectorizing across time;
   this is the baseline a stateful per-sample evaluator gives.     ← baseline
C  t written as a closed form (t = n·dt), loop vectorizer left ON — with the
   recurrence gone, the compiler can take the time axis itself.
B  t closed form + explicit SIMD: W t's per step, lanes split the block.
```

`R` is the baseline because a loop-carried recurrence is exactly what stops the
compiler from vectorizing across time. `C` and `B` both require dropping the
recurrence first — that is the move under test.

### Method

Each path is timed over **15 trials after 3 warmup runs**, with the **caches
flushed** (a >LLC scratch write) before every run so nothing starts hot. We report
the **median** (robust to scheduler jitter), the **min** (best case), and the
stddev. `-ffp-contract=off` pins all three paths to the same mul-add shape, so `C`
and `B` come out **bit-identical** — the reported `B==C` difference is exactly 0.

```
make run        # builds float, double, and a single-partial (thin) variant
```

### Results — Apple M1 Pro (arm64 / NEON), AppleClang 17, `-O3`

Rich kernel — 8 partials × 8 offsets (64 sin/sample):

```
                                   float                    double
  R  recurrent t (baseline)      40.5 ms              86.3 ms
  C  closed-form, auto-vec ON    40.2 ms  (1.01x)     85.5 ms  (1.01x)
  B  closed-form, explicit SIMD  33.9 ms  (1.19x)     69.4 ms  (1.24x)
```

Single partial — 1 × 1, where time is the *only* parallel axis:

```
  R  recurrent t (baseline)       4.02 ms
  C  closed-form, auto-vec ON     0.65 ms  (6.21x)
  B  closed-form, explicit SIMD   0.70 ms  (5.76x)
```

`B` and `C` are bit-identical (`B==C = 0`). The recurrent-`t` path drifts **6.8**
from the closed form in float over the 1M-sample run (accumulated rounding in
`t += dt`); in double it drifts 2.6e-6. (Numbers are medians of 15 runs; absolute
times vary a few percent run-to-run with thermal state.)

### Reading it

1. **The recurrent baseline already vectorizes *within* a sample** — across the M
   partials, via the inner reduction loop. So on a partial-rich kernel the lanes
   are spent regardless, and partitioning time adds only ~1.2× on top. On a thin
   kernel (one partial) the recurrent form leaves the width unused **and** pays the
   per-sample state round-trip, so the closed form reclaims ~6×.
2. **The win is the closed form, not the hand-written SIMD.** Once `t` is a closed
   form the compiler takes the time axis on its own (`C`); on the thin kernel it
   even beats the explicit version (`B`). A stateful per-sample evaluator can get
   none of this — there is no time axis for any backend to take.
3. **Accuracy, for free.** Float `t`-accumulation decorrelates over a long run
   (drift 6.8 here); the closed form is exact at every sample. So the closed form
   isn't just sometimes-faster — it's *correct* where the accumulator quietly isn't.

So: SIMD time-partition is a real but kernel-dependent CPU win — modest when the
inner arithmetic already soaks the lanes, several-× when time is the only axis —
plus a free accuracy win. Large lane counts are a hardware-width question, not a
CPU one.

### Portability

The kernel uses GCC/Clang vector extensions (`__attribute__((vector_size))`), so
the same source targets any ISA: `W=8` float lanes lower to one 256-bit AVX
register on x86 (`-march=native`) or two 128-bit NEON registers on arm64
(`-mcpu=native`). The *shape* of the result — a modest win when partials soak the
lanes, a several-× win when time is the only axis — is portable; the magnitudes
above are arm64-only and unverified on x86.
