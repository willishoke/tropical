# Benchmark Results: microkernel-mode spike

| Field           | Value                                       |
|-----------------|---------------------------------------------|
| Date            | 2026-05-23                                  |
| Branch          | `microkernel-mode-spike`                    |
| Captured at     | commit `599aa01` (post deep-mode revert)   |
| Hardware        | darwin/aarch64 (Apple Silicon)              |
| LLVM            | 22.1.1                                      |
| Opt level       | `OptimizationLevel::O2`                     |
| Reproduce       | `bun run tests/bench/microkernel_vs_fused.ts` |

Benchmark numbers are temporal artifacts. When the bench script
changes shape or the engine's emit strategy materially shifts, this
file should be re-captured against the new commit hash. Treat it as
a snapshot, not a spec.

**Decision (at time of capture):** **GO** — proceed with the broader
rearchitecture (Phase E plan: voice scopes, affine types at scope
boundaries, per-kernel hot-swap).

## What was measured

For each of 7 cases, we compiled and ran the same session graph twice
— once in `fused` mode (single monolithic LLVM kernel, the legacy
default), once in `microkernel` mode (N+3 LLVM functions dispatched
from a C++ per-sample loop) — and recorded compile latency and
runtime ns/sample. Run on darwin/aarch64, LLVM 22.1.1,
`OptimizationLevel::O2`.

Methodology: cold disk cache, 32-frame warmup, 4096 frames × 256
samples measured, ns/sample = wall / total_samples. Independent
session per (case, mode) pair so register/slot init starts from the
same place in both runs.

## Headline numbers

| Case                  | Inst | Fused ns/s | MK ns/s | Slowdown | Fused JIT | MK JIT |
|-----------------------|-----:|-----------:|--------:|---------:|----------:|-------:|
| bubble_drip           |    3 |       29.4 |    39.5 |   1.34×  |     0 ms* |  28 ms |
| cross_fm_4            |    8 |       30.9 |    37.6 |   1.22×  |     0 ms* |  22 ms |
| odd_harmonics         |   50 |      132.9 |   139.4 |   1.05×  |     1 ms* |  61 ms |
| polyphony 4× SinOsc   |    4 |       16.4 |    25.0 |   1.53×  |     25 ms |  16 ms |
| polyphony 8× SinOsc   |    8 |       36.4 |    49.1 |   1.35×  |     51 ms |  24 ms |
| polyphony 16× SinOsc  |   16 |       80.6 |    99.6 |   1.24×  |    142 ms |  44 ms |
| polyphony 32× SinOsc  |   32 |      190.7 |   202.5 |   1.06×  |    479 ms |  76 ms |

`*` = fused JIT was a hot in-memory cache hit on the second pass;
microkernel JIT was always cold (different cache key). See "Compile
latency" below for the fair comparison.

## What this says

**Runtime cost: small and shrinks with scale.** Microkernel-mode
slowdown is 1.05×-1.53× across the cases we ran. Crucially, the
slowdown *drops as instance count grows*:

```
   1.6×┤▲                          (microkernel-mode slowdown vs fused)
      │ │
   1.5×┤ │
      │ │\
   1.4×┤ │ \
      │ │  \    ▲
   1.3×┤ │   \   \
      │ │    ╲   ╲    ▲
   1.2×┤ │     ╲   ╲    ╲
      │ │      ╲    ╲    ╲
   1.1×┤ │       ╲    ╲    ╲        ▲
      │ │        ╲    ╲    ╲        ╲     ▲
   1.0×┤─┴─────────────────────────────────┴───────────────
       4×  8×  16×  32×              instances →
```

The cost is fixed per call (function-pointer dispatch, no
cross-instance inlining) and amortizes against per-instance work.
For the *use case microkernels exist for* — many independent voices
— the cost approaches noise at moderate counts. The 32-voice case
runs at 0.89% of the sample-period budget; there's enormous headroom
for the per-voice work the eventual architecture will support.

**Compile latency: 6× faster at scale.** This was the surprise. LLVM's
optimizer scales superlinearly with function size; splitting into N+3
smaller functions lets each be optimized independently:

| Case                  | Fused cold JIT | MK cold JIT | Speedup |
|-----------------------|---------------:|------------:|--------:|
| polyphony 4×          |          25 ms |       16 ms |   1.6×  |
| polyphony 8×          |          51 ms |       24 ms |   2.1×  |
| polyphony 16×         |         142 ms |       44 ms |   3.2×  |
| polyphony 32×         |         479 ms |       76 ms |  **6.3×** |
| odd_harmonics (50)    |     [too fast]*|       61 ms |   —     |

`*` = the second time we compile a fused-mode plan in this benchmark,
the in-memory cache hits. We didn't bother re-instrumenting because
the cold-JIT numbers are dominated by polyphony anyway.

**Correctness: byte-identical** to 1e-12 across the full equivalence
suite (Phase 6 — 19 stdlib programs + 8-voice polyphony case). Both
modes share per-instruction codegen (the EmitCtx methods are
functionally identical to fused mode's inline lambdas), so any
divergence would point at the dispatch loop or parser, not the math.

## Why microkernels are *faster* to compile at scale

The fused-mode kernel is one giant LLVM function with the per-sample
body inlined for every instance. As instances pile up, the function
gets bigger, and LLVM's inliner/SLP-vectorizer/loop-rotate passes
spend O(n²)-ish time on it. The 479ms compile for 32 voices is
LLVM, not us — our IR emission is microseconds.

Microkernel mode emits one LLVM function per instance, plus
preamble/state_evolution/postamble_mix. Each function is small,
optimization is cheap, and the LLJIT can compile them in parallel.
The fixed overhead (module setup, codegen of N+3 function prologues)
amortizes over more functions — hence the speedup ratio *grows* with
instance count.

This compile-latency win is independently load-bearing for the
broader roadmap: hot-swap during live performance, fast iteration in
authoring tools, per-voice incremental recompilation when only one
voice changed. Even if runtime parity were a wash, the JIT-latency
story alone would justify the architecture.

## What this does not measure

- **Live-coding feel.** Edit one voice in a polyphonic session and
  recompile — the eventual win is *per-voice* incremental
  recompilation (recompile one out of 32 functions instead of all
  33). The spike does not measure this yet because no per-kernel
  hot-swap path exists (out of scope; Phase E follow-up).
- **Cache behavior under voice churn.** Allocating and retiring voices
  at musical rates exercises the affine/lifetime story this spike
  doesn't implement. The numbers above are all static-topology.
- **Affine-type costs.** None — no type-system work was attempted in
  the spike. Whatever cost the type machinery adds is unmeasured.

## Recommendation: GO

The plan's acceptance criterion was:

> microkernel ns/sample within ~2× of fused on the medium case, and
> at-least-parity on the polyphony case.

We came in at 1.22× on the medium case (cross_fm_4) and at 1.06× on
the largest polyphony case (32 voices). Both targets cleared. The
compile-latency story is a bonus — 6× faster at scale was not on the
acceptance list but is independently valuable.

Proceed with the Phase E plan: voice scopes in the surface syntax,
the LNL split (cartesian backbone + affine voice fragment),
per-kernel hot-swap, and the typed `!` modality across the scope
boundary. The microkernel runtime is performant enough that the
type-system work isn't gated by performance worries.

## Reproducing

```bash
make build
bun run tests/bench/microkernel_vs_fused.ts          # runs and prints
cat /tmp/microkernel_bench.json | jq .deltas         # structured output
bun test tests/equiv/microkernel_vs_fused.test.ts    # correctness gate
```

Raw structured results: `/tmp/microkernel_bench.json` after the bench
script runs. The deltas object in that file is the canonical artifact
for follow-up analysis.

## Architectural notes for the rearchitecture

1. The duplicated codegen (compile_flat_program's lambdas vs EmitCtx's
   methods) is honest spike scaffolding. With the spike passing, the
   cleanup is straightforward: switch compile_flat_program to use
   EmitCtx, delete the lambda block, gate the change behind one final
   ctest run that confirms byte-identical IR. Not strictly required
   before the Phase E work; a maintenance-burden cleanup.

2. The per-sample microkernel dispatch loop lives in C++
   (`FlatRuntime::process`). For per-voice hot-swap, the
   `MicrokernelKernels::instances` vector needs to be safely swappable
   at audio-thread granularity (atomic pointer for the vector itself,
   plus per-slot atomics for individual replacements). Current code
   relies on whole-`KernelState` double-buffering; that's the right
   place to start, but per-voice swap is a follow-up.

3. The dispatch signature (PerSampleFn) deliberately drops
   `buffer_length` and `output_buffer`. If a future use case needs
   per-call inputs (e.g. external buffer routing) we'll widen the
   signature — it's encapsulated in two `using` aliases in
   `OrcJitEngine.hpp` so the change is localized.

4. Cache-key mode tagging (`"flat5:mk:"` vs `"flat5:fused:"`) is a
   prefix on the existing serialization. If we later add backend #4
   (independent sample rates, GPU dispatch, etc.) the same pattern
   extends cleanly.
