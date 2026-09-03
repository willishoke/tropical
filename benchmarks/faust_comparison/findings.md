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

The cause is the sine kernel, not the architecture — and the two kernels are
not solving the same problem. Faust's `os.osc` compiles to a **truncated**
lookup in a 65536-entry table (one load, NO interpolation: the emitted index
is `int(65536.0 * phase)`, clamped); tropical's `FixedSin` is a ~6-term
fixed-point Q31 Horner polynomial with quadrant folding
(`EmitArrow/Numerics.lean`). Measured against `libm`:

```
  tropical Q31 polynomial   max |error| 3.83e-9   -168.3 dB   ~28 bits
  Faust F1 truncated table  max |error| 9.59e-5    -80.4 dB   ~13 bits
                                          25017x    88.0 dB
```

Memory beats arithmetic at this size, but it is buying its speed with 88 dB of
accuracy, and the table's error is a staircase — harmonic distortion, not
noise. Chasing 750 ns/voice means choosing that trade, not closing a gap.
(Corrected 2026-09-02: this section previously described F1 as "an interpolated
wavetable (one load plus a lerp)". It interpolates nothing.)

**2. tropical also trails Faust's OWN closed-form variant by 24-32%**
(2334 vs 1805 at N=512). Same semantics, same compiler, same unrolled shape —
so that gap is implementation quality, not paradigm. tropical's fixed-point
polynomial being slower than libm `sin` is a surprise worth its own
investigation.

> **RESOLVED 2026-09-02 — and not by the polynomial.** The gap was eight
> instructions per voice per sample computing an identity: the Q32 phase was
> laundered through a float and back before reaching the sine. Removing it
> closes the gap to +3.7% / +0.4% / -2.3%. See
> [Reading 2 resolved](#reading-2-resolved-the-24-32-gap-to-f3-was-a-phase-round-trip).

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

## Reading 2 resolved: the 24-32% gap to F3 was a phase round trip

**Date:** 2026-09-02 · **Rows:** [`data/after-phasefix.jsonl`](data/after-phasefix.jsonl)
· **Probes:** [`sine_probes/`](sine_probes/) · closed
`design/fixedsin-vs-libm-handoff.local.md`

The gap is closed. F3 is the control: it is unchanged code, so its
reproducing the recorded baseline within 1-2% is what licenses comparing
the tropical column across runs at all.

```
              F3 baseline   F3 now     tropical baseline   tropical now      gap
  N=256          1747        1765            2308              1831     +32% -> +3.7%
  N=512          1805        1844            2334              1851     +29% -> +0.4%
  N=1024         1926        1948            2382              1903     +24% -> -2.3%
```

Read -2.3% as parity, not a win.

### It was never the polynomial

The handoff framed this as "an integer polynomial losing to a libm call,"
and ranked five hypotheses that all point at the sine kernel or its exit.
None of them was the cause. Differencing the per-sample loop body across
N=64 -> 128 (so preheader and loop overhead cancel) gave **47.89 arm64
instructions per voice per sample**, of which the Horner polynomial is
about 19: seven multiplies and seven subtractions, with every `>> 30`
folded into the following subtraction as a shifted-register operand
(`subs x27, x2, x27, asr #30`). That is as tight as the polynomial can be
emitted. Nothing in `EmitArrow/Numerics.lean` was worth changing.

Hypothesis 1 (vectorisation) is refuted directly: Faust's `.2d` operations
are SLP pairs in the surrounding arithmetic, and every voice still costs a
scalar `bl _sin`. Hypothesis 2 came closest -- it suspected the int->float
*exit* -- but the cost was a round trip on the way *in*, which is why
looking at `x / 2^30` (which LLVM does strength-reduce, correctly) found
nothing.

### The actual cause: eight instructions computing an identity

`buildFixedSinOsc` computed the Q32 phase in integers and then laundered it
through the float domain and back before handing it to the sine:

```
  ucvtf  d0, x9          ; int -> double
  fmul   d0, d0, d13     ; * 2^-32     -> normalized phase in [0,1)
  fcmp   d0, #0.0
  fcsel  d0, d0, d1, gt  ; clamp low
  fcmp   d0, d2
  fcsel  d0, d0, d2, mi  ; clamp high
  fmul   d0, d0, d12     ; * 2^32      -> back to Q32
  fcvtzs x9, d0          ; double -> int
```

`acc & 0xffffffff` lands in [0, 2^32), which is exact in a double (< 2^53);
both scalings are by a power of two, so exact; therefore the clamp can
never fire and the output equals the input, bit for bit. It was a
representation seam between "phase as a normalized float port" and "phase
as Q32" -- and not even a load-bearing one, since the phase OFFSET input is
already folded in upstream in the integer domain.

Removing it took the loop body to **39.89** instructions per voice per
sample. The delta is exactly the eight above and nothing else: `fcmp` 2->0,
`fcsel` 2->0, `ucvtf` 1->0, `fcvtzs` 1->0, `fmul` 3->1. A 16.7% instruction
cut bought ~20% of wall clock, slightly super-proportional because the
eight sat *on the serial dependency chain* -- the phase had to complete its
excursion before the sine could start.

**How bit-exactness was established.** Not by the frozen hashes: the
`tests/golden/stdlib/*.hash` goldens freeze wire+port STRUCTURE, so they
necessarily move when a builder emits fewer nodes, and they prove nothing
numerical. The proof is `bootstrap-sin`, which renders 2048 samples from an
independently built term that *still performs the round trip*
(`fixedSinOscTerm` in `Testing/ArrowFixtures.lean`, via `phasorPhaseSig`)
and compares them byte-for-byte against the stdlib generator that no longer
does. It passes unchanged, as do `fixedsin-longtau` (byte-exact at tau+2^30
samples) and `negative-clock` (exact at negative time) -- the large-tau and
negative-`acc` cases where the identity argument would break if it broke.

### Refuted: the dead constant slot stores are NOT a cost

Half the stores in the per-sample loop body write a compile-time constant to
a fixed slot address every sample -- 129 of 260 at N=64 -- because the
emitter gives every instance input a slot and writes it in the sample loop
whether or not it varies. That looked like the obvious next target, and like
a second appearance of the uniform-slot discipline behind the kernel-size
cliff's cause 1.

It is not a cost. Deleting the stores outright (`sine_probes/conststore.py`,
an upper bound on hoisting them) makes the kernel **slower**, reproducibly:

```
    N  removed  stock_text  stock ns/v  strip_text  strip ns/v   delta
   64       65       13040      1848.3       10616      2033.2   +10.0%
  128      129       25784      1793.6       20856      1960.6    +9.3%
  192      193       38560      1848.1       31096      1998.9    +8.2%
   -- rerun --
   64       65       13040      1945.0       10616      2034.5    +4.6%
  128      129       25784      1869.3       20856      1985.2    +6.2%
```

The strip is semantically valid: nothing in the module loads those slots,
so the stores are genuinely dead. `__text` drops ~19% and the kernel gets
SLOWER anyway. Whatever the mechanism -- scheduling and alignment shifting
under `CodeGenOptLevel::None`, where there is no scheduler to re-pack the
loop -- the conclusion is that "hoist the loop-invariant slot writes" is not
a promising optimisation at these sizes, and the intuition that fewer
instructions and less code must be faster is wrong here in both of its
halves.

This also cross-checks the cliff findings rather than contradicting them.
At N=64-192 the kernel is 13-38 KB of `__text`, far below the ~230 KB
per-sample code budget the cliff located; code size is not the binding
constraint down here, so shrinking it buys nothing.

### A trap for anyone counting Faust instructions

Static counts on the Faust side are inflated ~3x: clang emits three copies
of the frame loop (unrolled-by-2, a scalar fallback for the aliasing case,
and a remainder), so N=64 shows 192 `bl _sin`, not 64. Any per-voice figure
taken from `otool` output has to be divided by the number of loop copies,
which is why `sine_probes/loopdiff.py` differences loop bodies instead.

### What is still open

Reading 1 -- the ~2.4x loss to F1 at ~750 ns/voice -- is untouched, and is a
different question: memory versus arithmetic at 88 dB less accuracy (see
reading 1, corrected), not implementation quality. Reading 3 (statelessness is a discount, not a tax)
is unaffected; it was always an F2-vs-F3 comparison internal to Faust.

## What this fixture does NOT show

It is a bank of bare oscillators, which is not where tropical's design is
meant to pay. Every cost here is per-voice synthesis, and none of it exercises
composition — filters, modulation, or the modal algebra where a filter
composes into modes already being evaluated rather than adding a per-voice
stage. A fixture built around a modal source through a swept resonant filter,
reported as the MARGINAL cost of adding the filter, would cancel the
table-vs-polynomial handicap that dominates this table and measure the
architecture instead. That is the natural next fixture; it is not built.

Read this table as "the cost of tropical's sine kernel at scale", not as "the
cost of tropical's design". The one line here that is genuinely about the
design is F2 vs F3, and it goes tropical's way.
