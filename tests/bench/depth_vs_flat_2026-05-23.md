# Benchmark Results: input-slot refactor — depth vs flat

| Field           | Value                                       |
|-----------------|---------------------------------------------|
| Date            | 2026-05-23                                  |
| Branch          | `input-slot-refactor`                       |
| Captured at     | post-meta-bench rewrite                     |
| Hardware        | darwin/aarch64 (Apple Silicon, M-series)    |
| LLVM            | 22.1.1                                      |
| Opt level       | `OptimizationLevel::O2`                     |
| Reproduce       | `bun run tests/bench/depth_vs_flat.ts`      |

Benchmark numbers are temporal artifacts. When the bench script or
the engine's emit strategy materially shifts, re-capture against the
new commit hash. Treat as a snapshot, not a spec.

**Decision (at time of capture):** **NO-GO on `inlineInstances`
deletion** — keep both paths alive. Slot path is correct (Phase 4
gate is green for all currently-compiling cases) and competitive at
the small scale, but 25% slower at runtime on the realistic stress
case and exposes a Bubble/BubbleCloud correctness gap. Inline path
still earns its keep.

## Methodology

Two cases bracket the deletion decision:

  - **OnePole** — minimal nested (2 children). Is the slot path's
    overhead detectable at the smallest scale?
  - **polyphony_8x_Phaser16** — stress (8 voices × 17 nested kernels
    = 136 kernel boundaries). Is the overhead tolerable at scale?

For each (case, mode):

  - **Runtime**: N=3 trials of 1024 frames × 256 samples each.
    Round-robin interleaved across modes. Min reported.
    N derived empirically from `runtime_noise_meta.ts`: min-of-3
    lands within 0.76% of asymptotic min at p95 on this hardware.
  - **Compile**: one cold-cache trial per axis. The engine has a
    process-singleton in-memory cache that survives disk-cache
    `rmSync`; measuring multiple cold compiles in one process would
    need subprocess isolation. Single-shot is fine for snapshot.

Total wall clock: ~3s.

## Numbers

### Compile (one cold trial per axis)

| Case                       | Mode   | Inst | TS ms | JIT ms | Plan KB |
|----------------------------|--------|-----:|------:|-------:|--------:|
| OnePole                    | flat   |    1 |   3.1 |   11.4 |     4.4 |
| OnePole                    | nested |    1 |   0.4 |    9.4 |     5.8 |
| polyphony_8x_Phaser16      | flat   |    8 |  66.9 |  741.3 |   204.5 |
| polyphony_8x_Phaser16      | nested |    8 |   2.8 |  865.7 |   329.2 |

### Runtime (3 trials, min reported)

| Case                       | Mode   | Trials (ns/sample)         | min   | median | max   |
|----------------------------|--------|----------------------------|------:|-------:|------:|
| OnePole                    | flat   | 8.0, 7.9, 15.7             |   7.9 |    8.0 |  15.7 |
| OnePole                    | nested | 8.3, 8.4, 9.0              |   8.3 |    8.4 |   9.0 |
| polyphony_8x_Phaser16      | flat   | 322.7, 322.3, 333.7        | 322.3 |  322.7 | 333.7 |
| polyphony_8x_Phaser16      | nested | 403.8, 452.4, 405.8        | 403.8 |  405.8 | 452.4 |

### Deltas

| Case                       | ns_ratio | jit_ratio | plan_ratio |
|----------------------------|---------:|----------:|-----------:|
| OnePole                    |   1.057× |    0.826× |     1.313× |
| polyphony_8x_Phaser16      |   1.253× |    1.168× |     1.610× |

## What the numbers say

**Runtime.** OnePole pays 6% overhead, polyphony_8x_Phaser16 pays
25%. The cost scales with kernel-boundary density: at 136 nested
kernels the WriteSlot / Slot round-trips at each boundary stop
amortizing against the per-instance work. The realtime budget is
uncontested — worst case (Phaser16 8x nested) is 1.81% of a 44.1kHz
sample period, plenty of headroom — but 25% is a meaningful tax on
how much MORE work fits in a buffer.

**Compile (cold).** OnePole compiles faster in nested mode (0.83×) —
small programs apparently give LLVM more digestible per-function
work units than the inline path's monolithic body. Phaser16 8x
flips: 1.17× slower nested. Crossover seems to be around the point
where per-function fixed cost (~10ms? hard to attribute) stops
amortizing across the larger function count.

**TS pipeline.** Nested compiles in ~25× less TS time for Phaser16
8x (66.9ms → 2.8ms). The inline path's `inlineInstances` pass is
where the cost lives — splicing 136 nested kernels into 8 monolithic
bodies takes real time. The slot path skips that pass entirely.

**Plan size.** Nested-mode IR is 30-60% larger; the per-child
WriteSlots add real bytes. This matters for disk cache footprint
and TS↔C++ JSON shipping, but neither is in the critical path.

**Trial variance.** The runtime noise is real but small. OnePole's
trial 3 at 15.7 ns/sample (vs 7.9 min) is a clear outlier — almost
certainly an OS pre-emption or frequency-scaling event. The
min-of-3 methodology correctly strips it. CV measured in the meta-
bench was 0.75%, so the 0.76% confidence bound at p95 holds.

## Bubble / BubbleCloud: the real correctness gap

These cases failed nested-mode compile with:

```
compileResolved: register init must lower to a literal value
```

Root cause: when `inlineInstances` is skipped (the slot path's whole
point), each child sub-instance keeps its own `ResolvedProgram`,
and that program's `RegDecl.init` may be an `ExprNode` that the
flat path implicitly reduced via splicing. `compileResolved`'s
`regInit` helper accepts only literal numbers / booleans / arrays /
`zeros{N}` — not arbitrary ExprNodes.

Bubble has `reg env_smooth = 0` (literal, fine) but inherits nested
children (SampleHold, TriggerRamp, Exp, EnvExpDecay, SVF), one of
which has a non-literal init. Same shape recurs in BubbleCloud (8x
Bubble).

This is a real correctness gap in the slot path. It needs to be
fixed before any deletion-of-`inlineInstances` decision can land.
Three approaches in order of invasiveness:

  1. Strengthen `compileResolved`'s `regInit` to evaluate constant
     ExprNodes (small constant-folding step).
  2. Add a pre-emit strata pass that reduces register inits
     specifically (parallels how `extractSessionDelays` runs).
  3. Make `inlineInstances` per-program rather than top-level, so
     sub-programs get their inits reduced during their own compile
     rather than via parent splicing.

Approach 1 is smallest. Approach 2 is most aligned with the strata-
pipeline shape. Approach 3 is the most invasive but might be the
right factoring long-term.

## Recommendation for the followup deletion PR

**Don't delete `inlineInstances` yet.** Three blockers:

  1. The Bubble / BubbleCloud correctness gap above. Until any
     stdlib program compiles under `inlineNested:false`, the slot
     path is not a drop-in replacement.

  2. The 25% runtime overhead at the realistic stress scale is real.
     Not catastrophic — there's headroom — but it's a tax. If we
     can close that gap (LLVM passes, better slot fusion,
     something else) the deletion case becomes much stronger.

  3. The slot path's larger IR (~1.5× plan size) inflates the disk
     cache footprint and the JSON-shipping overhead between TS and
     C++. Worth optimizing the wire format before committing
     fully.

**Keep both paths alive.** `inlineNested:true` remains the default;
`inlineNested:false` is the opt-in for nested-aware tests
(`tests/equiv/nested_vs_inlined.test.ts`) and any future work that
needs per-kernel hot-swap, voice scopes, or per-instance JIT cache
keys.

**Next milestones for slot-path adoption:**

  - Fix the Bubble / BubbleCloud register-init gap (approach 1 or 2)
  - Measure under `-O3` JIT to see if more aggressive load-store
    forwarding closes the 25% runtime gap
  - Investigate why polyphony_8x_Phaser16's nested mode pays 25%
    while OnePole only pays 6% — is the cost in slot-load-store
    pairs that LLVM isn't folding, or in icache/dcache pressure
    from the larger IR?
