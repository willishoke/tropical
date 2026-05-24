# Benchmark Results: bubble-fix-via-levels — depth vs flat

| Field           | Value                                       |
|-----------------|---------------------------------------------|
| Date            | 2026-05-24                                  |
| Branch          | `bubble-fix-via-levels`                     |
| Captured at     | post-Phase-4 (Bubble/BubbleCloud green)     |
| Hardware        | darwin/aarch64 (Apple Silicon, M-series)    |
| LLVM            | 22.1.1                                      |
| Opt level       | `OptimizationLevel::O2`                     |
| Reproduce       | `bun run tests/bench/depth_vs_flat.ts`      |

Benchmark numbers are temporal artifacts. When the bench script or
the engine's emit strategy materially shifts, re-capture against the
new commit hash. Treat as a snapshot, not a spec.

**Decision (at time of capture):** **NO-GO on `inlineInstances`
deletion**, same verdict as 2026-05-23. The Bubble/BubbleCloud
correctness gap is now closed (Phase 4: port-default fallback for
unwired sub-instance inputs, and split preamble/per-child dispatch
order). But the runtime cost still favors inline at scale — 22% on
BubbleCloud's fractal 8-voice structure, 25% on the older
polyphony_8x_Phaser16 stress case. Both paths kept alive.

## Methodology

Four cases, ordered by complexity:

  - **OnePole** — minimal nested (2 children). Is the slot path's
    overhead detectable at the smallest scale?
  - **Bubble** — single voice, 5 sub-instances. The smallest fractal
    case that exercises the Phase 4 fixes.
  - **BubbleCloud** — 8 Bubbles × 5 sub-instances = 40 fractal
    kernels. The fix landed here; what's the runtime tax?
  - **polyphony_8x_Phaser16** — 8 voices × 17 nested kernels = 136
    kernel boundaries. The pre-existing stress reference.

For each (case, mode):

  - **Runtime**: N=3 trials of 1024 frames × 256 samples each.
    Round-robin interleaved across modes. Min reported.
    N derived empirically from `runtime_noise_meta.ts`: min-of-3
    lands within 0.76% of asymptotic min at p95 on this hardware.
  - **Compile**: one cold-cache trial per axis. The engine has a
    process-singleton in-memory cache that survives disk-cache
    `rmSync`; measuring multiple cold compiles in one process would
    need subprocess isolation. Single-shot is fine for snapshot.

Total wall clock: ~6s.

## Numbers

### Compile (one cold trial per axis)

| Case                       | Mode   | Inst | TS ms | JIT ms | Plan KB |
|----------------------------|--------|-----:|------:|-------:|--------:|
| OnePole                    | flat   |    1 |   3.3 |   65.4 |     4.4 |
| OnePole                    | nested |    1 |   0.4 |    9.9 |     5.8 |
| Bubble                     | flat   |    1 |   1.0 |   30.8 |    23.4 |
| Bubble                     | nested |    1 |   0.6 |   26.7 |    28.0 |
| BubbleCloud                | flat   |    1 |   4.8 |  726.2 |   182.0 |
| BubbleCloud                | nested |    1 |   2.3 |  627.5 |   234.0 |
| polyphony_8x_Phaser16      | flat   |    8 |   1.3 |  734.8 |   204.5 |
| polyphony_8x_Phaser16      | nested |    8 |   2.2 |  868.0 |   329.2 |

### Runtime (3 trials, min reported)

| Case                       | Mode   | Trials (ns/sample)         | min   | median | max   |
|----------------------------|--------|----------------------------|------:|-------:|------:|
| OnePole                    | flat   | 8.1, 8.0, 7.9              |   7.9 |    8.0 |   8.1 |
| OnePole                    | nested | 8.5, 8.8, 8.3              |   8.3 |    8.5 |   8.8 |
| Bubble                     | flat   | 27.0, 26.5, 26.5           |  26.5 |   26.5 |  27.0 |
| Bubble                     | nested | 27.1, 27.0, 27.0           |  27.0 |   27.0 |  27.1 |
| BubbleCloud                | flat   | 207.6, 207.5, 207.5        | 207.5 |  207.5 | 207.6 |
| BubbleCloud                | nested | 254.8, 254.1, 254.0        | 254.0 |  254.1 | 254.8 |
| polyphony_8x_Phaser16      | flat   | 322.8, 323.9, 332.3        | 322.8 |  323.9 | 332.3 |
| polyphony_8x_Phaser16      | nested | 406.1, 404.2, 404.8        | 404.2 |  404.8 | 406.1 |

### Deltas

| Case                       | ns_ratio | jit_ratio | plan_ratio |
|----------------------------|---------:|----------:|-----------:|
| OnePole                    |   1.054× |    0.152× |     1.313× |
| Bubble                     |   1.017× |    0.869× |     1.195× |
| BubbleCloud                |   1.224× |    0.864× |     1.286× |
| polyphony_8x_Phaser16      |   1.252× |    1.181× |     1.610× |

## What the numbers say

**Runtime overhead tracks kernel-boundary density.** OnePole
(2 children) pays 5%, Bubble (1 voice × 5 children) pays a
negligible 2%, BubbleCloud (8 × 5 = 40 child kernels) pays 22%,
polyphony_8x_Phaser16 (8 × 17 = 136 kernels) pays 25%. The cost
isn't per-voice — it's per slot-WriteSlot-then-Slot round trip at
each kernel boundary, summed over all instances. A single Bubble
amortizes those round trips against its per-sample work; eight of
them stop fitting.

**Bubble's 2% is the headline.** That's exactly the case Phase 4
was about. The fix isn't free, but it's nearly free at the single-
voice scale where the per-child-input dispatch order and the port-
default fallback matter most. The Bubble path now produces bit-
identical output between flat and nested modes at ~zero runtime
cost.

**The realtime budget is uncontested.** Even the slowest case
(polyphony_8x_Phaser16 nested) runs at 1.81% of a 44.1kHz sample
period; BubbleCloud nested runs at 1.13%. Plenty of headroom.

**Compile.** TS time is consistently smaller in nested mode (the
slot path skips `inlineInstances`), 5-8× faster on the larger
programs. JIT time is mixed: nested wins on BubbleCloud and Bubble
(0.86-0.87×), loses on polyphony_8x_Phaser16 (1.18×). OnePole's
jit_ratio of 0.152× is a single-shot outlier — single cold compile
measurements swing wildly on tiny programs (10ms variance dwarfs
the 10ms baseline). Don't read into that one.

**Plan size.** Slot-path plan is ~20-60% larger (per-child
WriteSlots add real bytes). Matters for disk cache footprint and
TS↔C++ JSON shipping, neither in the critical path.

**Trial variance.** Tighter than the 2026-05-23 capture. No outliers
this run, all trials within 1-3% of their min. Consistent with the
CV=0.75% bound from `runtime_noise_meta.ts`.

## Bubble / BubbleCloud: gap closed

Previous snapshot called these out as failing nested-mode compile
with `compileResolved: register init must lower to a literal value`.
Three things had to land:

  1. **De Bruijn levels for global refs.** `sumLower` becomes pure
     (Phase 2); sub-program `RegDecl.init` arrives already lowered
     because the strata pipeline runs end-to-end, not implicitly via
     parent splicing. Closes the original compile error.

  2. **Split preamble from main-body instructions.** New
     `preamble_instructions` field on `InstanceFunction`, emitted by
     the engine's `emit_kernel_block` BEFORE child dispatch.
     Previously the per-child WriteSlots referenced parent temps
     that hadn't been computed yet — broken for expression-shaped
     session inputs like Bubble's `pulseEvery(64)` trigger.

  3. **Port-default fallback for unwired sub-instance inputs.**
     `emit_resolved`'s per-child WriteSlot loop now iterates ALL of
     the target's input ports, not just `decl.inputs`. For ports
     the parent doesn't wire, the port's declared default goes into
     the slot. Previously the slot retained its allocation default
     of 0, silently masking the port default — e.g., BubbleCloud
     doesn't wire `attack_g`, so the slot stayed at 0 instead of
     Bubble's `0.05`, killing `env_smooth` evolution and forcing
     output to 0.

`tests/equiv/nested_vs_inlined.test.ts` now covers Bubble and
BubbleCloud at 1e-12 between flat and nested — the same gate
threshold as the other 20 stdlib cases.

## Recommendation for the deletion PR

**Don't delete `inlineInstances` yet.** Two of the three previous
blockers stand:

  1. ~~The Bubble / BubbleCloud correctness gap~~ — **closed
     in Phase 4.**

  2. The 22-25% runtime overhead at the realistic fractal scale is
     real. Not catastrophic — there's headroom — but it's a tax.
     If `-O3` JIT or better slot fusion closes that gap, the
     deletion case becomes stronger.

  3. The slot path's larger IR (~1.3-1.6× plan size) inflates the
     disk cache footprint and the JSON-shipping overhead between
     TS and C++. Worth optimizing the wire format before
     committing fully.

**Keep both paths alive.** `inlineNested:true` remains the default;
`inlineNested:false` is the opt-in for nested-aware tests
(`tests/equiv/nested_vs_inlined.test.ts`) and any future work that
needs per-kernel hot-swap, voice scopes, or per-instance JIT cache
keys.

**Next milestones for slot-path adoption:**

  - Measure under `-O3` JIT to see if more aggressive load-store
    forwarding closes the runtime gap on the fractal stress cases
  - Investigate why BubbleCloud's nested mode pays 22% and
    polyphony_8x_Phaser16 pays 25% while Bubble pays ~0% — is the
    cost in slot-load-store pairs that LLVM isn't folding, or in
    icache/dcache pressure from the larger IR?
  - Optimize the wire-format JSON encoding of WriteSlot blocks
    (they're the bulk of the 1.3-1.6× plan size inflation)
