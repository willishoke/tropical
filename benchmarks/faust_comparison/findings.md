# tropical vs Faust — findings

**Date:** 2026-08-31
**Host:** Apple M1 Pro, macOS 26.3
**Faust:** 2.85.9 · **C++:** Homebrew LLVM 22.1.7 — the same LLVM the tropical JIT uses,
so codegen quality is not a hidden variable
**Rows:** [`data/double-m1pro-20260831.jsonl`](data/double-m1pro-20260831.jsonl),
[`data/n2048-fill.jsonl`](data/n2048-fill.jsonl)
**Fixture:** N detuned sine voices summed, f64, 512-frame block @ 44.1 kHz

## Why three Faust variants

A naive "N sines" race measures the wrong thing. Faust idiomatically uses a
phase ACCUMULATOR (one add, one wrap); tropical computes phase CLOSED FORM
from an absolute coordinate, because that is what buys scrubbing and
click-free hot-swap. Comparing only those two prices tropical's design choice
and reports it as a deficiency. So:

```
  F1  wavetable + accumulator   what a Faust user actually ships
  F2  libm sin  + accumulator   recurrent, sine held constant against F3
  F3  libm sin  + counter       CLOSED FORM: tropical's semantics
```

`F2 vs F3` isolates recurrence-vs-closed-form with the sine fixed. `F1 vs
tropical` is the real-world number. Faust's output is fully unrolled (N `sin`
calls in one expression inside the sample loop), matching tropical's unrolled
kernel, so neither side is secretly looping where the other is not.

## Results — per-voice cost (ns per voice per 512-frame block)

```
    N   tropical    F1 wavetable   F2 libm+accum   F3 libm+closed
   64       2318            1376            2183             2120
  256       2308             800            2093             1747
  512       2334             752            2224             1805
 1024       2382             750            2503             1926
 1536       4160             756            2614             2034
 2048       4564             754            2732             2126

  voices at 50% saturation (interpolated between measured points):
  tropical ~1450   F2 ~2125   F3 ~2700   F1 ~7700
```

## Three readings, and they do not agree

**1. tropical loses to idiomatic Faust, by ~3x at N=512 and ~6x at N=2048.**
F1 is flat at ~750 ns/voice from N=256 on. This is the headline loss.

The cause is almost certainly the sine kernel, not the architecture: Faust's
`os.osc` is an interpolated wavetable (one load plus a lerp), tropical's
`FixedSin` is a ~6-term fixed-point Q31 Horner polynomial with quadrant
folding (`EmitArrow/Numerics.lean`). Memory beats arithmetic at this size.

**2. tropical also trails Faust's OWN closed-form variant by 24-32%**
(2334 vs 1805 at N=512). Same semantics, same compiler, same unrolled shape —
so that gap is implementation quality, not paradigm. tropical's fixed-point
polynomial being slower than libm `sin` is a surprise worth its own
investigation.

**3. The architectural thesis is independently vindicated.** F3 beats F2 at
every count from 256 up — 17%, 19%, 23%, 22%, 22% — with the sine
implementation, the compiler, and the unrolled shape all held constant. Only
the recurrence differs. This reproduces `../simd_time_partition`'s result
against a competitor's compiler: a phase accumulator is a loop-carried
dependency that forbids vectorising across time, and removing it is worth
~20%. **Statelessness is not the tax paid for scrubbing; on this evidence it
is a discount.**

## Secondary observations

- **tropical's compile cliff reappears.** Per-voice cost jumps 2382 -> 4160
  (+75%) between N=1024 and N=1536, exactly where
  `../oscillator_saturation/findings.md` located it. Faust shows no such cliff
  (750 -> 756).
- **Faust's compiler self-aborts at N=2048.** It exits on SIGALRM with an empty
  stderr after its own default `-t 120` compile timeout; a 2048-term `process`
  expression exceeds it. The harness now passes a larger `-t` and records a
  self-timeout explicitly rather than losing the row to a mystery failure.

## What this fixture does NOT show

It is a bank of bare oscillators, which is not where tropical's design is
meant to pay. Every cost here is per-voice synthesis, and none of it exercises
composition — filters, modulation, or the modal algebra where a filter
composes into modes already being evaluated rather than adding a per-voice
stage. A fixture built around a modal source through a swept resonant filter,
reported as the MARGINAL cost of adding the filter, would cancel the
wavetable-vs-polynomial handicap that dominates this table and measure the
architecture instead. That is the natural next fixture; it is not built.

Read this table as "the cost of tropical's sine kernel at scale", not as "the
cost of tropical's design". The one line here that is genuinely about the
design is F2 vs F3, and it goes tropical's way.
