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
premise is right about the ALGEBRA and wrong about the emitted code. The
constant `sqrt` in the loop body is `filterPair`'s `unary .sqrt radicand` — its
whole pole-pair derivation, recomputed 44100 times a second for a value that
changes at most once per block.

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

## The slope is value-independent — by construction

Re-fit at a high cutoff (2500 Hz), high resonance (0.9), and long source decay
(12): **identical to the instruction** — slope 393.0, intercept 1607 in every
row (`robustness.py`). Obvious once seen: cutoff/resonance/decay are param
slots, so the emitted code shape cannot depend on their values. The slope is a
property of the emitted composition, not of any pole geometry. (Wall clock
636-666% across configs — noise.)

## LLVM does not rescue it either

The preflight counted the EMITTER's IR; the JIT then runs its O2 pipeline
(LICM included). Splitting the post-O2 module at the sample loop
(`posto2.py`, `loopsplit.py`): the preheader holds **one `fmul`**; the `sqrt`
and the whole divide army (152 fdiv at M=16, 536 at M=64) are inside the loop,
which also carries ~1400-5000 loads, mostly through pointers loaded from
`%arrays` that alias analysis cannot disambiguate from the ~250-900 in-loop
stores. The τ-free math is trapped at BOTH layers.

## The mechanism, named to the line

Two hypotheses died on the way to this one (the freezing-path story above, and
"param slots are opaque to staging" — refuted by the source: `paramRef` interns
at **`base := .s0`**, `Ir/Nodes.lean:369`, explicitly "the stage-0 coefficient
kernel's territory"). The staging ATTRIBUTES are correct: the entire
composition is s0-valued.

The failure is PLACEMENT. The modal linear stages lower through
`RoutedSumBegin`/`RoutedSumEnd` spans (the `rs_*` blocks in the emitted IR;
the routed terminal that serves orientation/direction), and
`Ir/Stage0.lean:546-560` (`placementFromStages`) masks every instruction
inside a routed span to s1 CATEGORICALLY:

```lean
    -- Static routed reductions are indivisible placement regions. ...
    let stageAt (i : Nat) : Stage :=
      if routedMask[i]! then .s1
```

No per-instruction hoist applies inside the span, and the whole-region move
(`tryRegion`) exists only for `ReduceBegin`/`ReduceEnd` units — that is the
path the bare resonator's bank takes to `coeff.ll` (the `banks-region-hoist`
gate), which is why the UNFILTERED source hoists beautifully and the filtered
one does not. An s0-valued computation is trapped by a deliberate v1
conservatism in placement — not by the algebra, not by the attributes.

## The price of the fix, measured without building it

If the composition ran at s0, the audio kernel would evaluate an
**(M+2)-mode bare bank** — the filter's conjugate pair joins the source's
modes with rescaled residues; that is the modal algebra's closure property
(the proven residue-composition ground). And because the emitted code shape is
value-independent (above), the bank's COST does not depend on what the
residues are — so `resonator(partials=M+2)` prices the hoisted kernel exactly,
no residue math required:

```
               fixture  audio-IR   ns/block   speedup
           filter(M=4)      3439     243854
       bare bank M+2=6       272      45458      5.4x
          filter(M=16)      8227     764292
      bare bank M+2=18       344     118584      6.4x
          filter(M=64)     27379    3104208
      bare bank M+2=66       632     422000      7.4x
```

Growing with M, as 1607 + 393·M over a ~6/mode bank predicts. Scope: this
bound is for the forward, direction-zero case measured here — and note the
filter node's direction is STATICALLY zero (`ModalControlRef.constant zero`,
`Playground/Decode.lean`), so for this node the routed orientation machinery
is inert by construction.

## Two shapes for the fix

1. **A `tryRegion` analog for routed spans**: a delimiter-matched routed
   region whose region-neutral stage is ≤ s0 moves to the coefficient stream
   as a unit, exactly as `banks-region-hoist` established for reduce regions.
   General (covers reverb and a swept phaser whose controls are s0), touches
   the placement pass and its availability rules.
2. **Degenerate the statically-forward filter to plain composition**: when a
   modal linear stage's direction is a compile-time zero, lower it through the
   ordinary bank path — no routed region exists to be trapped. Narrower (the
   filter node today, not reverb), but it is a lowering decision rather than a
   placement-pass change, and it cannot perturb the orientation machinery.

Either way the sweep tradeoff below is the semantic decision that comes with
it.

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

1. **Hoist the composition** — established above: it is s0 by attribute,
   trapped only by the routed-span placement mask, and worth a measured
   5.4-7.4x on this fixture (more at higher M). Two fix shapes above. This is
   worth far more than the benchmark: it is every filtered modal patch, not a
   number in a table.
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
benchmarks/modal_filter/preflight/marginal.py 4,16,64   # the headline table
benchmarks/modal_filter/preflight/robustness.py         # value-independence
benchmarks/modal_filter/preflight/posto2.py             # post-O2 loop bodies
benchmarks/modal_filter/preflight/loopsplit.py m16filter m64filter  # preheader split
```
