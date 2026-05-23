# Benchmark Results: input-slot refactor — depth vs flat

| Field           | Value                                       |
|-----------------|---------------------------------------------|
| Date            | 2026-05-23                                  |
| Branch          | `input-slot-refactor`                       |
| Captured at     | commit `7b5a09e` (post Phase 5)             |
| Hardware        | darwin/aarch64 (Apple Silicon, M-series)    |
| LLVM            | 22.1.1                                      |
| Opt level       | `OptimizationLevel::O2`                     |
| Reproduce       | `bun run tests/bench/depth_vs_flat.ts`      |

Benchmark numbers are temporal artifacts. When the bench script
changes shape or the engine's emit strategy materially shifts, this
file should be re-captured against the new commit hash. Treat it as
a snapshot, not a spec.

**Decision (at time of capture):** **NO-GO on inlineInstances
deletion** — keep both paths alive. Slot path is correct (Phase 4
gate is green for all currently-compiling cases) and competitive,
but introduces a 5-27% runtime overhead and a 18-59% JIT compile
overhead on genuinely-nested programs. It also exposes a real
correctness gap on `Bubble` / `BubbleCloud` (regs with non-literal
inits in sub-programs). The inline path is still pulling its
weight; defer deletion until the correctness gap closes and the
overhead numbers either improve or motivate a different path.

## What was measured

For each of 16 cases, we compiled and ran the same session graph
twice — once with `inlineNested:true` (legacy flat IR; the
`inlineInstances` strata pass splices every sub-instance into its
parent's body, producing one monolithic per-instance kernel), once
with `inlineNested:false` (M11 fractal slot path; every sub-instance
stays a kernel boundary with per-port WriteSlot / Slot reads
crossing it). We recorded TS pipeline time, LLVM JIT time, kernel
ns/sample, and wire-format plan size.

Methodology: cold disk cache, 32-frame warmup, 4096 frames × 256
samples measured, ns/sample = wall / total_samples. Independent
session per (case, mode) pair so register / slot init starts from
the same place in both runs.

## Headline numbers

| Case                       | Inst | Flat ns/s | Nested ns/s | Slowdown | Flat JIT | Nested JIT | Plan size |
|----------------------------|-----:|----------:|------------:|---------:|---------:|-----------:|----------:|
| Sin                        |    1 |       2.1 |         1.9 |   0.94×  |    13 ms |    0.1 ms* |    1.00×  |
| OnePole                    |    1 |       8.3 |         9.1 |   1.09×  |    11 ms |      10 ms |    1.31×  |
| Pow                        |    1 |       9.8 |         5.7 |   0.60×  |    34 ms |      22 ms |    0.73×  |
| SVF                        |    1 |       8.6 |         8.5 |   0.99×  |    11 ms |    0.1 ms* |    1.00×  |
| LadderFilter               |    1 |      38.3 |        41.6 |   1.12×  |    24 ms |      30 ms |    1.41×  |
| Phaser16                   |    1 |      48.2 |        49.4 |   1.06×  |    35 ms |      58 ms |    1.61×  |
| Bubble                     |    1 |      26.8 |       _ERR_ |     —    |    25 ms |      _ERR_ |     —     |
| BubbleCloud                |    1 |     212.6 |       _ERR_ |     —    |   743 ms |      _ERR_ |     —     |
| polyphony 8x Sin           |    8 |      18.6 |        18.7 |   1.01×  |    69 ms |    0.4 ms* |    1.00×  |
| polyphony 32x Sin          |   32 |      88.8 |        88.7 |   1.00×  |   536 ms |    1.5 ms* |    1.00×  |
| polyphony 8x SinOsc        |    8 |      36.7 |        39.4 |   1.07×  |    51 ms |      73 ms |    1.03×  |
| polyphony 8x OnePole       |    8 |      25.0 |        29.7 |   1.19×  |    25 ms |      38 ms |    1.33×  |
| polyphony 8x SVF           |    8 |      37.1 |        36.9 |   1.00×  |    60 ms |    0.6 ms* |    1.00×  |
| polyphony 8x LadderFilter  |    8 |     219.9 |       275.5 |   1.25×  |   351 ms |     442 ms |    1.41×  |
| polyphony 4x Phaser16      |    4 |     170.5 |       187.1 |   1.10×  |   579 ms |     485 ms |    1.61×  |
| polyphony 8x Phaser16      |    8 |     329.0 |       410.9 |   1.25×  |   763 ms |     904 ms |    1.61×  |

`*` Cache hit. When a program has no nested sub-instances (Sin, SVF,
polyphony-of-leaf-types), both modes produce byte-identical IR; the
nested-mode run hits the warm disk cache populated by the flat-mode
run that preceded it. The number reflects cache lookup, not actual
JIT work.

## What the numbers say

**1. Leaf cases are free.** Programs with no nested sub-instances
(Sin, SVF, polyphony of either) produce identical IR in both modes
and run identically. The slot path adds zero overhead when there's
nothing to slot.

**2. Genuine nested cases pay 5-25% runtime overhead.** The slot
path's WriteSlot / Slot round-trip at each kernel boundary doesn't
fully fold away under LLVM O2. Worst case observed:
`polyphony_8x_LadderFilter` at 1.25× and `polyphony_8x_Phaser16` at
1.25×. The cost scales with kernel-boundary density, not just
instance count — 8x Phaser16 (136 nested kernels) costs more than
32x Sin (32 monolithic kernels) per sample even though Sin's
instance count is higher.

**3. JIT compile overhead is larger but still small in absolute
terms.** Nested-mode IR is 30-60% larger (the per-child WriteSlots
add up), and LLVM takes ~20-50% longer to JIT it. For
`polyphony_8x_Phaser16` that's 904 ms vs 763 ms — both noticeable
on cold compile, neither blocking interactivity once the disk
cache is warm.

**4. Pow is the surprise outlier.** Pow compiled in nested mode is
0.60× the runtime cost of the flat compile, and 0.66× the JIT
cost. Pow is `exp(y * log(x))` — the flat compile produces a giant
chained expression that LLVM apparently handles WORSE than the
slot-broken nested version. Sample size of one; not load-bearing,
but worth remembering when designing the next layer of
optimization.

**5. `polyphony_4x_Phaser16` shows nested JIT FASTER (0.84×).**
For polyphonies of complex programs, nested mode appears to give
LLVM more uniform per-function work units to chew through, while
the flat compile produces 4 enormous functions back-to-back. The
crossover seems to be around 4-8 voices; at 8x Phaser16 the nested
mode is 1.18× slower again. Worth investigating if anyone wants
to chase polyphony-of-complex-programs perf.

**6. The realtime budget is uncontested either way.** Worst case
(`polyphony_8x_Phaser16` nested) sits at 1.81% of a 44.1kHz sample
period. Plenty of headroom for both modes. The slowdown ratios
matter for fitting MORE work into a single buffer, not for hitting
realtime at all.

## Bubble / BubbleCloud: the real-correctness gap

These two cases ERR'd in nested mode:

```
compileResolved: register init must lower to a literal value
```

Root cause: when `inlineInstances` is skipped (the slot path's
whole point), each child sub-instance keeps its own
`ResolvedProgram`, and that program's `RegDecl.init` may be an
ExprNode that the flat path implicitly reduced as a side effect of
splicing. `compileResolved`'s `regInit` helper accepts only
literal numbers / booleans / arrays / `zeros{N}` — not arbitrary
ExprNodes.

Bubble has a `reg env_smooth = 0` (literal, fine) but it inherits
nested children (SampleHold, TriggerRamp, Exp, EnvExpDecay, SVF),
one of which presumably has a non-literal init. Same shape recurs
in BubbleCloud (8x Bubble).

This is a real correctness gap in the slot path. It needs to be
fixed before any deletion-of-`inlineInstances` decision can land.
Possible approaches:

  1. Strengthen `compileResolved`'s `regInit` to evaluate constant
     ExprNodes (a small constant-folding step).
  2. Add a pre-emit strata pass that reduces register inits
     specifically (similar to how `extractSessionDelays` runs as a
     pre-emit pass).
  3. Make `inlineInstances` itself idempotent and per-program
     (not per-top-level), so sub-programs get their inits reduced
     during their own compile rather than via parent splicing.

Approach 1 is smallest. Approach 2 is most aligned with the strata-
pipeline shape. Approach 3 is the most invasive but might be the
right factoring long-term.

## Recommendation for the followup deletion PR

**Don't delete `inlineInstances` yet.** Three blockers:

  1. The Bubble / BubbleCloud correctness gap above. Until any
     stdlib program can compile under `inlineNested:false`, the
     slot path is not a drop-in replacement.

  2. The 5-25% runtime overhead is real but not catastrophic. If
     we can land the inputs-substitution rewrite WITHOUT this
     overhead — through LLVM passes, better slot fusion, or
     something else — the deletion case becomes much stronger.

  3. The slot path's larger IR (~1.5× plan size) inflates the disk
     cache footprint and the JSON-shipping overhead between TS and
     C++. Worth optimizing the wire format (per-child blocks could
     elide their `pre_input_instructions` wrapper for the empty
     case, already done) before committing.

**Keep both paths alive.** `inlineNested:true` remains the default;
`inlineNested:false` is the opt-in for the nested-aware tests
(`tests/equiv/nested_vs_inlined.test.ts`) and any future work that
needs per-kernel hot-swap, voice scopes, or per-instance JIT cache
keys (all of which want kernel boundaries to survive into the
emitted IR).

**Next milestones for slot-path adoption:**
  - Fix the Bubble / BubbleCloud register-init gap
  - Add depth-4 synthetic stress cases (constructed
    programmatically; currently the deepest real-stdlib case is
    depth-3 BubbleCloud)
  - Investigate why `polyphony_8x_OnePole` is the worst-scaling
    polyphony case (1.19× runtime + 1.52× JIT) despite OnePole
    being a relatively simple program
  - Measure the slot path under a `-O3` JIT (currently `O2`) to see
    if more aggressive load-store forwarding closes the gap
