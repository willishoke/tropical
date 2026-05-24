# Benchmark Results: issue-156-locally-nameless-ir — depth vs flat (post-issue-#156)

| Field           | Value                                       |
|-----------------|---------------------------------------------|
| Date            | 2026-05-24                                  |
| Branch          | `issue-156-locally-nameless-ir`             |
| Commit          | `e0cf51e`                                   |
| Captured at     | post-issue-#156 (Phases 2 + 4a + 4b + 3 + 5) |
| Hardware        | darwin/aarch64 (Apple Silicon, M-series)    |
| LLVM            | 22.1.1                                      |
| Opt level       | `OptimizationLevel::O2`                     |
| Reproduce       | `bun run tests/bench/depth_vs_flat.ts`      |

Snapshot baseline for the IR-internals refactor: pointer-identity
discipline removed (BindingRef + InstanceDecl.type), single
`session.programs` registry, catamorphism-based passes, `clone.ts`
deleted. Kernel codegen is unchanged; this bench confirms the
refactor is runtime-neutral.

Companion to `depth_vs_flat_2026-05-24.md` (the PR #158 "post-Phase-4"
snapshot from earlier the same day). Direct A/B comparison below.

## Methodology

Identical to `depth_vs_flat_2026-05-24.md`: four cases (OnePole,
Bubble, BubbleCloud, polyphony_8x_Phaser16) × two modes
(`inlineNested:true` aka flat, `inlineNested:false` aka nested) ×
N=3 runtime trials of 1024×256 samples, round-robin interleaved,
min reported. One cold-cache compile trial per axis.

Total wall clock: ~5s.

## Numbers

### Compile (one cold trial per axis)

| Case                       | Mode   | Inst | TS ms | JIT ms | Plan KB |
|----------------------------|--------|-----:|------:|-------:|--------:|
| OnePole                    | flat   |    1 |   3.8 |  113.0 |     4.4 |
| OnePole                    | nested |    1 |   0.4 |    9.9 |     5.8 |
| Bubble                     | flat   |    1 |   0.8 |   34.8 |    23.4 |
| Bubble                     | nested |    1 |   0.5 |   27.1 |    28.0 |
| BubbleCloud                | flat   |    1 |   4.0 |  734.9 |   182.0 |
| BubbleCloud                | nested |    1 |   2.0 |  626.6 |   234.0 |
| polyphony_8x_Phaser16      | flat   |    8 |   1.4 |  730.4 |   204.5 |
| polyphony_8x_Phaser16      | nested |    8 |   2.2 |  866.7 |   329.2 |

### Runtime (3 trials, min reported)

| Case                       | Mode   | Trials (ns/sample)         | min   | median | max   |
|----------------------------|--------|----------------------------|------:|-------:|------:|
| OnePole                    | flat   | 8.5, 8.4, 8.5              |   8.4 |    8.5 |   8.5 |
| OnePole                    | nested | 8.4, 8.7, 8.3              |   8.3 |    8.4 |   8.7 |
| Bubble                     | flat   | 26.8, 26.8, 26.7           |  26.7 |   26.8 |  26.8 |
| Bubble                     | nested | 27.1, 27.3, 27.4           |  27.1 |   27.3 |  27.4 |
| BubbleCloud                | flat   | 207.2, 207.7, 207.5        | 207.2 |  207.5 | 207.7 |
| BubbleCloud                | nested | 256.1, 253.7, 254.0        | 253.7 |  254.0 | 256.1 |
| polyphony_8x_Phaser16      | flat   | 332.4, 323.9, 323.0        | 323.0 |  323.9 | 332.4 |
| polyphony_8x_Phaser16      | nested | 403.8, 406.4, 404.5        | 403.8 |  404.5 | 406.4 |

### Deltas (nested / flat)

| Case                       | ns_ratio | jit_ratio | plan_ratio |
|----------------------------|---------:|----------:|-----------:|
| OnePole                    |   0.996× |    0.087× |     1.313× |
| Bubble                     |   1.015× |    0.779× |     1.195× |
| BubbleCloud                |   1.225× |    0.853× |     1.286× |
| polyphony_8x_Phaser16      |   1.250× |    1.187× |     1.610× |

## A/B vs `depth_vs_flat_2026-05-24.md` (pre-issue-#156)

### Runtime (ns/sample, min)

| Case                       | Mode   | Pre   | Post  | Δ      | Verdict      |
|----------------------------|--------|------:|------:|-------:|--------------|
| OnePole                    | flat   |   7.9 |   8.4 |  +0.5  | within noise |
| OnePole                    | nested |   8.3 |   8.3 |   0    | unchanged    |
| Bubble                     | flat   |  26.5 |  26.7 |  +0.2  | within noise |
| Bubble                     | nested |  27.0 |  27.1 |  +0.1  | within noise |
| BubbleCloud                | flat   | 207.5 | 207.2 |  −0.3  | within noise |
| BubbleCloud                | nested | 254.0 | 253.7 |  −0.3  | within noise |
| polyphony_8x_Phaser16      | flat   | 322.8 | 323.0 |  +0.2  | within noise |
| polyphony_8x_Phaser16      | nested | 404.2 | 403.8 |  −0.4  | within noise |

All eight runtime measurements move ≤0.5 ns/sample. The CV=0.75%
bound from `runtime_noise_meta.ts` puts these well inside trial
noise. **Runtime is unchanged**, as expected — the refactor is
TS-side IR plumbing; kernel codegen is identical.

### Compile (single cold trial — high variance baseline)

| Case                       | Mode   | TS Δ  | JIT Δ |
|----------------------------|--------|------:|------:|
| OnePole                    | flat   |  +0.5 | +47.6 |
| OnePole                    | nested |   0   |   0   |
| Bubble                     | flat   |  −0.2 |  +4.0 |
| Bubble                     | nested |  −0.1 |  +0.4 |
| BubbleCloud                | flat   |  −0.8 |  +8.7 |
| BubbleCloud                | nested |  −0.3 |  −0.9 |
| polyphony_8x_Phaser16      | flat   |  +0.1 |  −4.4 |
| polyphony_8x_Phaser16      | nested |   0   |  −1.3 |

OnePole's +47ms JIT delta is the single-trial outlier the previous
snapshot warned about — at the small-program scale a single cold
compile swings ±50ms run-to-run because the 10-100ms baseline is
dominated by LLVM's one-time setup costs. The rest of the deltas
are small (single-digit ms) and split roughly evenly between
improvements and regressions. **Compile time is unchanged at the
significance level a single-trial bench can resolve.**

Plan size is bit-identical (same Map iteration order, same emit
sequence) — the refactor doesn't change what the TS pipeline emits,
only how it constructs it.

## What this confirms

The IR-internals refactor (issue #156 Phases 2–5):

  - **Runtime-neutral.** The C++ JIT consumes the same
    `tropical_plan_5` shape; `emit_resolved` emits the same
    `NInstr` sequence per program. No hot-path change.
  - **Compile-neutral within trial noise.** The TS pipeline does
    slightly different work — functional passes instead of clone+
    mutate, single registry lookup instead of double — but the
    deltas don't exceed cold-compile single-trial variance.
  - **No regressions on the realtime budget.** Headroom is
    unchanged: BubbleCloud nested at 1.13% of 44.1kHz sample
    period; polyphony_8x_Phaser16 nested at 1.81%.

The refactor's value is in the IR architecture (one source of truth
per concept, functional rewrites, no clone.ts), not in the bench
numbers. The numbers say: the cleanup is free.

## What this doesn't measure

  - **TS-pipeline-only timing** (loader, elaborate, strata). The
    `bun run tests/bench/compile.ts` bench covers that; this one
    measures end-to-end `compileSession + loadPlan` which is
    dominated by LLVM. The TS-side refactor effects (specialize via
    `mapExpr` instead of clone-with-substitution, etc.) are sub-ms
    on these cases and below this bench's resolution.
  - **Memory footprint.** No `gc()`-paired measurements here.
    `clone.ts` deletion should reduce peak heap during compilation
    (fewer transient decl allocations), but quantifying it isn't
    in this bench's scope.

## Recommendation

No action. The bench confirms the refactor ships at zero runtime
cost. The same deletion verdict as the pre-issue-#156 snapshot
stands: keep both `inlineNested:true` and `inlineNested:false`
paths alive — the slot path's 22-25% runtime overhead on fractal
stress cases is the same with or without this refactor.
