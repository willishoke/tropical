# Preflight: what does tropical's filter actually cost at audio rate?

**Date:** 2026-09-03 · **Host:** Apple M1 Pro · **Probe:**
[`marginal.py`](marginal.py) · **Verdict: do not build the fixture yet.**

`design/modal-filter-comparison-handoff.local.md` proposes a P x M x sweep-rate
fixture with a quality oracle. Its load-bearing premise:

> tropical : filter modifies residues of modes already evaluated -> O(M), no new stage
>
> **Prediction under test:** the marginal cost of adding a filter ... approaches
> zero *as a fraction* in tropical as M grows.

The premise is cheap to test alone, and everything expensive in the fixture
rests on it. It is **false as implemented**, by two orders of magnitude.

## The measurement

`resonator(partials=M)` with and without a following modal stage, compiled
through `diffcli render-graph` with `TROPICAL_STAGE0=1`. The stage-0 split
does the isolation for free: `audio.ll` is the per-sample kernel, `coeff.ll`
the once-per-block one, so "does the filter cost anything per sample" is
literally "does `audio.ll`'s loop body grow".

```
    M   stage   audio  d_audio   /part   coeff  d_coeff         ns    d_ns%
    4       -     260                      116               34938
    4  filter    3439     3179   794.8     169       53     243230   596.2%
    4  reverb    9689     9429  2357.2     337      221    1153104  3200.5%
   16       -     332                      356              103770
   16  filter    8227     7895   493.4     481      125     763062   635.3%
   16  reverb   14417    14085   880.3     661      305    3599292  3368.5%
   64       -     620                     1316              442021
   64  filter   27379    26759   418.1    1729      413    3216188   627.6%
   64  reverb   33329    32709   511.1    1957      641   13236230  2894.5%
```

Both stages fit a linear model to three significant figures across a 16x range:

```
  filter   d_audio = 1607 + 393·M        (slope 393.0 both halves)
  reverb   d_audio = 7877 + 388·M        (slope 388.0 both halves)
  bare resonator                            6·M
```

So a filtered mode costs **~65x** what an unfiltered one costs, per sample, and
the marginal cost of the filter grows with M rather than approaching zero as a
fraction of it. `coeff.ll` barely moves (+53 / +125 / +413): the work is landing
in exactly the kernel the premise says it avoids.

## The mechanism: a partial-fraction expansion, every sample

Per source mode per sample, in the filtered audio loop body:

```
            M=4    M=16   M=64     per source mode
  fdiv       49      97    289          4
  fabs       32      80    272          4
  fmul      358     766   2398         34
  sqrt        2       2      2          0    (constant: filterPair, in the intercept)
```

**Four floating-point divides per source mode per sample** — about two complex
divisions, which is what composing against a conjugate pole PAIR by partial
fractions costs. At M=64 that is 256 divides per sample, and `fdiv` is the one
arm64 op that is not meaningfully pipelined, which is why the wall clock
(~630%) is worse than the instruction ratio alone predicts.

**None of it depends on τ.** Pole positions, residues, and the partial-fraction
denominators are all functions of `cutoff`/`resonance` and the source's own
modal parameters. It is coefficient math sitting in the audio kernel. The
premise is right about the ALGEBRA and wrong about the emitted code.

Why it is not hoisted: `cutoff`/`resonance` are `param:*` module slots
(`discipline := .glide`), so nothing proves them block-invariant, and
everything downstream of the slot read stays audio-rate. The constant `sqrt` in
the loop body is `filterPair`'s `unary .sqrt radicand` — its whole pole-pair
derivation, recomputed 44100 times a second for a value that changes at most
once per block.

## A hypothesis this refuted

`filter` reads `paramValue` directly and closes over the live slot reads,
declaring `controls := #[ModalControlRef.constant zero]`; `reverb` declares a
control and receives a frozen value (`build := fun frozenRt60 => …`,
`Playground/Decode.lean`). That looked like the whole story — filter bypasses a
freezing path that reverb uses.

Measuring reverb refuted it. Reverb uses the frozen path and is **worse**
(+9429 vs +3179 at M=4), and its per-mode slope is the same 388 vs 393. So
freezing `rt60` does not reach the composition against the source modes.
Whatever the freezing mechanism currently covers, it is not this. The two
stages differ only in their intercept — reverb derives 14 room modes per sample
where filter derives one pole pair — and share the per-mode term entirely.

## Why this blocks the fixture rather than feeding it

The fixture is designed to isolate the marginal cost of the filter as the
architectural claim. As things stand that measurement would report a large loss
to Faust — whose `fi.svf` filters the SUM at O(1) per sample — for a reason
that is not architectural. And the fixture has no way to tell the two apart: a
~65x implementation artifact and a genuine paradigm cost look identical in the
marginal-cost column.

Publishing that would be worse than not running it, in either direction. If it
came out against tropical it would indict the design for an emitter problem; if
some other axis came out favourable it would launder the artifact.

## What to do instead, in order

1. **Establish whether the composition is hoistable.** It has no τ in it, so in
   principle the whole `1607 + 393·M` moves to `coeff.ll` and the audio kernel
   goes back to ~6 instructions per mode. That is the real finding here, and it
   is worth far more than the benchmark: it is a ~65x on every filtered modal
   patch, not a number in a table.
2. **Decide the sweep tradeoff deliberately.** Hoisting means coefficients
   update once per block — 86 Hz at 512 frames / 44.1 kHz — so a fast cutoff
   sweep gets staircase-quantized. The current code pays full audio rate to get
   sample-accurate sweep. That is a real dial, and Faust does not have it: its
   biquad updates coefficients per sample but carries state belonging to the
   old ones. "tropical can choose; Faust cannot" is a better and more honest
   claim than any marginal-cost number.
3. **Then build the fixture**, with metric 3 rewritten. The handoff expects
   "tropical flat, Faust degrading with rate". If the composition is hoisted,
   tropical will NOT be flat — it will have its own artifact, a different one.
   Two systems, two failure modes, both measured against the oversampled
   reference.

## Reproduce

```sh
make build && make lean
benchmarks/modal_filter/preflight/marginal.py 4,16,64
```
