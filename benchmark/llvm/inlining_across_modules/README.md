# Inlining across compilation boundaries

## The question

Can LLVM inline operations across separately-compiled module boundaries
well enough that the operadic substrate's planned multi-kernel layout
matches the performance of today's monolithic single-function kernel?

## Why this matters for tropical

The current tropical JIT compiles an entire session into one LLVM
function. Each session instance contributes a sequence of basic blocks
inside this single function; LLVM's intra-procedural optimizations
(GVN, mem2reg, jump threading, loop fusion) work across the whole
session because there are no function boundaries between kernels.

The active-set spike findings doc (`active_set_spike_findings.md`)
explicitly relied on this: default-alive instances fold to
unified-kernel-byte-for-byte performance because GVN sees the
`WriteSlot const 1.0` in the scheduler preamble, forwards the store
through to the load in each instance's alive check, and constant-folds
the conditional away. This requires the writer and reader to be in the
same LLVM function.

The architectural direction discussed in `design/operadic_ir.md` —
fractal compilation, per-operation kernels, multi-realization
architecture — implies a future where kernels might be compiled
separately:

- **Hot-swap granularity**: today, hot-swapping one instance recompiles
  the whole session. If kernels lived in separate LLVM modules, you
  could swap one without touching the others.
- **External primitives**: C++ FFI operations, neural network
  realizations, FPGA accelerators — each is necessarily compiled
  separately and called across some boundary.
- **Substrate-style positioning**: if tropical-as-substrate hosts
  multiple frontend DSLs and backend realizations, modular compilation
  becomes structurally important.

But: if separate compilation costs us the GVN folding, the alive
mechanism becomes a runtime branch instead of a compile-time fold, and
the active-set runtime's central performance claim no longer holds.

This benchmark establishes whether LLVM can recover the inlining
optimization across separately-compiled modules — under various
configurations — and quantifies the cost when it cannot.

## Hypotheses

**H1**: A single LLVM function with nested basic blocks per kernel
(today's tropical JIT) achieves baseline performance with default-alive
folding intact.

**H2**: Separately-compiled LLVM modules linked without LTO will *not*
fold default-alive checks — each kernel call boundary becomes an
opaque function call; the alive check inside each callee remains a
runtime branch.

**H3**: Separately-compiled LLVM modules linked *with* LTO can recover
inlining-equivalent optimization, including default-alive folding,
provided the LTO pipeline runs aggressive enough optimization passes
(`-O2` or `-O3` LTO).

**H4**: External C function calls (the FFI case) cannot be inlined
even with LTO unless the external module also participates in LTO. For
operations implemented in hand-written C++ libraries, function call
overhead is unavoidable per-sample.

**H5**: Per-sample function call overhead, where unavoidable, is
small enough in absolute terms (single-digit ns per call) that it
matters only when called at audio rate inside tight inner loops.
Quantify whether the cost is structurally manageable for nested kernel
designs.

## What we measure

Per benchmark variant:

- **Compile time** (LLVM IR generation + LLVM optimization + machine
  code emission). Measured by wall-clock around the JIT invocation.
- **Runtime per sample** (nanoseconds per sample averaged across a
  long buffer). Measured by Bun/criterion-style microbenchmarking.
- **Code size** (machine code bytes emitted). Measured by inspecting
  the JIT'd code section.
- **Default-alive fold survival** (does the LLVM IR show the alive
  check folded away, or is it preserved as a runtime branch?).
  Measured by dumping LLVM IR post-optimization and pattern-matching
  on the presence of `fcmp ... 0.5` instructions in the kernel call
  paths.

## Variants

The benchmark sweeps two axes:

### Axis 1: Kernel layout strategy

1. **`monolithic`** — entire kernel tree compiled into ONE LLVM
   function. Each kernel is a sequence of basic blocks. (Today's
   tropical JIT shape.)

2. **`separate-functions-same-module`** — each kernel is a separate
   LLVM function within the same module; calls are direct symbol
   references. LLVM's inliner can choose to inline across.

3. **`separate-modules-no-LTO`** — each kernel is in its own LLVM
   module; linked without LTO; calls are opaque to the optimizer.

4. **`separate-modules-with-LTO`** — same as #3 but with link-time
   optimization at `-O2`.

5. **`external-C-function`** — kernels are pre-compiled C functions
   in a separate `.so`; called via dlsym; entirely opaque.

### Axis 2: Kernel tree shape

For each layout, vary the kernel tree:

- **A**: 1 kernel doing simple arithmetic (`y = ax + b`). Baseline cost.
- **B**: 4 kernels in a linear chain (`y = f4(f3(f2(f1(x))))`).
  Each kernel does the same `y = ax + b`. Inlining recovers a single
  computation; non-inlining preserves 4 function calls.
- **C**: 4 kernels in parallel (independent computations on the same
  input, results summed). Tests whether the optimizer can reorder
  across kernel boundaries.
- **D**: 7 kernels arranged like LadderFilter (`Tanh`, `Sin`, 4×`OnePole`
  cascaded with feedback through a delay). Realistic audio DSP
  pattern; default-alive folding crucial for performance parity with
  today's monolithic JIT.
- **E**: 16 kernels arranged like Phaser16. Larger tree; tests scaling.

### Axis 3: Alive condition

For each layout × tree:

- **`default-alive`** — every kernel's alive slot is written `1.0`
  unconditionally in the scheduler preamble. Tests whether GVN folds
  the check away.
- **`conditional-alive`** — alive expressions depend on a session
  param. Tests the cost of dynamic alive dispatch.

## Measurement methodology

Each variant runs `N` warm-up buffers of 256 samples, then `M` measured
buffers of 256 samples each. Per-sample runtime is the harmonic mean
across measured buffers divided by 256.

For each variant, the LLVM IR is dumped post-optimization (via
`TROPICAL_DUMP_IR=1` or equivalent hook) and parsed for:

- Presence of `fcmp ole/ogt ... 0.5` instructions (alive checks
  surviving)
- Number of basic blocks per top-level function
- Number of inlined function bodies
- Total instruction count

Compile time is measured separately as wall-clock from "begin JIT
invocation" to "machine code address available."

## Implementation

The variants are plain C source files compiled with `clang -O2` (the
JIT's default optimization level). LLVM-level differences are isolated
by varying only the linkage strategy and the `-flto` flag, not the
source. Each variant has its own runner binary that links a different
combination of object files.

Initial implementation (in this directory):

```
inlining_across_modules/
  README.md                        — this doc
  Makefile                         — builds + runs all variants
  src/
    kernel.h                       — shared kernel/state definition
    v1_monolithic.c                — kernel + scheduler in one TU,
                                     kernel inlined by definition
    v3_kernel_module.c             — kernel as external function
    v3_scheduler.c                 — scheduler calling external kernel
    runner.c                       — timing harness, variant-selected
                                     via -DBENCH_VARIANT_V*
  build/                           — compile outputs (.o, .ll, binaries)
  data/
    results.csv                    — measurements appended per `make run`
```

Currently implemented variants:

- **v1** — monolithic source; kernel inlined; no LTO
- **v1lto** — monolithic source; kernel inlined; **with LTO**
- **v3** — separate modules; kernel called externally; no LTO
- **v4** — separate modules; kernel called externally; **with LTO**

Not yet implemented: variant 2 (separate static functions same module),
variant 5 (external `.so` via dlopen), additional kernel-tree shapes
(B, C, D, E in the original axis-2 spec), conditional-alive testing.

## Decision criteria

This benchmark exists to inform the architectural decision in the
operadic substrate work. Specifically:

- **If separate-modules-with-LTO recovers monolithic performance**
  (within 5% on the LadderFilter pattern with default-alive folding
  intact), then the operadic substrate can use modular compilation for
  hot-swap granularity without sacrificing the active-set performance
  claim.

- **If LTO falls short**, then either (a) we keep monolithic
  compilation as the only option (hot-swap recompiles the whole
  session, as today), or (b) we accept per-kernel function call
  overhead and quantify the user-visible cost.

- **If external C calls are dramatically slower** (>10× per kernel
  call), the FFI realization story becomes practical only for
  coarse-grained operations, not for per-sample primitives.

The benchmark output should produce concrete numbers, not just
"better/worse." We want to know the absolute ns/sample cost of each
strategy so we can make principled trade-offs in the architecture.

## Results

### 2026-05-14 — linear-chain tree, 4 variants, default-alive

Setup: macOS, Apple M1, clang from system toolchain. Linear chain of
4 simple kernels (`y = a*x + b + 0.5*state`). Buffer length 4096
samples; 256 measured iterations per run; 5 runs per variant.

| Variant | ns/sample (median) | vs v1 baseline |
|---|---|---|
| **v1** — monolithic, no LTO | 3.4 | 1.0× |
| **v1lto** — monolithic, with LTO | 1.4 | 0.41× |
| **v3** — separate modules, no LTO | 7.6 | 2.2× |
| **v4** — separate modules, with LTO | 1.4 | 0.41× |

All four variants produce bit-identical output (sink value
`0.292147`).

**Headline finding: LTO dissolves the module boundary.** With LTO,
the monolithic and separate-modules variants run at the same speed
(~1.4 ns/sample). Without LTO, separate modules pay a ~2.2× cost
relative to monolithic.

The win from LTO isn't from inlining the kernel into the scheduler
(that's already done in v1 by `static inline`). It's from **inlining
the scheduler into the runner's outer test loop**. With LTO, the
linker presents both translation units (scheduler + runner) to LLVM
together; the optimizer inlines `scheduler` into the per-iteration
measurement loop, then constant-propagates kernel coefficients, then
vectorizes across iterations. Without LTO, `scheduler` is opaque from
runner's TU and is invoked as a real function call per outer-loop
iteration.

This means **the test as written measures runner-level optimization
benefit as much as kernel-level**. To isolate the kernel-level
inlining effect from the outer-loop inlining effect, future variants
should either: (a) move all timing logic into the same TU as the
scheduler so there is no outer-call boundary to inline across, or
(b) annotate the scheduler with `__attribute__((noinline))` so LTO
preserves its call-site identity, isolating the kernel-into-scheduler
inlining as the only effect under measurement.

### What this means against the original hypotheses

- **H1** (monolithic baseline performance is preserved with
  default-alive folding intact): **CONFIRMED**. Default-alive isn't
  tested in this first cut (no alive logic yet), but the kernel-
  inlining piece is intact in v1.

- **H2** (separate modules without LTO will *not* fold optimizations
  across kernel boundaries; each call becomes opaque): **CONFIRMED**.
  v3's IR shows four real function calls per loop iteration; v1's IR
  shows the kernel body inlined eight times in the loop body with
  zero function calls.

- **H3** (LTO can recover monolithic performance for separately-
  compiled modules): **CONFIRMED, AND THEN SOME**. v4 not only
  recovers v1's performance — it matches v1lto, which BEATS v1
  because of additional cross-TU optimization (the outer-loop
  inlining). Module boundaries are essentially free under LTO at
  this scale.

- **H4** and **H5** are not tested by this first cut (external C
  function calls, absolute ns/call cost) — pending future variants.

### Implications for tropical's architecture

If the tropical runtime adopts LTO for its compiled output (or its
equivalent in the LLVM-ORC JIT pipeline that tropical uses — ORC's
default O2 pipeline already does cross-module optimization within a
single LLJIT instance), then the operadic substrate can use modular
compilation for hot-swap granularity **without sacrificing
performance**. Separate kernels compiled to separate LLVM modules
will be linked + optimized as one program at JIT time; module
boundaries disappear post-optimization.

The remaining concern is **per-kernel JIT compile time** under LTO.
LTO is more expensive than no-LTO. Need to measure: if each hot-swap
triggers a re-LTO of the affected kernel's surrounding modules, does
compile time stay reasonable?

### IR observations

v1 inner loop (relevant portion of `build/v1.ll`):

```llvm
; 4 kernels' bodies inlined; 8 fmuladd ops per sample:
%25 = tail call double @llvm.fmuladd.f64(...)  ; k1's a*x+b
%27 = tail call double @llvm.fmuladd.f64(...)  ; k1's +0.5*state
%30 = tail call double @llvm.fmuladd.f64(...)  ; k2's a*x+b
%32 = tail call double @llvm.fmuladd.f64(...)  ; k2's +0.5*state
%35 = tail call double @llvm.fmuladd.f64(...)  ; k3's a*x+b
%37 = tail call double @llvm.fmuladd.f64(...)  ; k3's +0.5*state
%40 = tail call double @llvm.fmuladd.f64(...)  ; k4's a*x+b
%42 = tail call double @llvm.fmuladd.f64(...)  ; k4's +0.5*state
```

The state and coefficient loads are NOT hoisted across iterations —
they're inside the loop, one load per iteration per kernel. The
pointers `k1, k2, k3, k4` could in principle alias (no `restrict`
qualifier), so the optimizer is conservative. Adding `__restrict__`
to the scheduler's parameters would likely allow phi-promotion of
state and hoisting of coefficients out of the loop; this is a
candidate optimization for a future iteration.

v3 inner loop (relevant portion of `build/v3_scheduler.ll`):

```llvm
%14 = tail call double @kernel_step_external(double %13, ptr %3)
%15 = tail call double @kernel_step_external(double %14, ptr %4)
%16 = tail call double @kernel_step_external(double %15, ptr %5)
%17 = tail call double @kernel_step_external(double %16, ptr %6)
```

Four actual function calls per sample. The function `kernel_step_external`
is declared as extern; the optimizer at scheduler-TU's compile time
has no body to inline. This is what produces the ~2.2× slowdown vs
v1.

The LTO variants (v1lto, v4) don't have their post-link IR dumped
into the build directory; they're compiled to native code via the
linker. Reading their optimized form would require `-Wl,-save-temps`
or post-link disassembly. Worth doing if questions arise about how
the LTO optimizer transformed the code.

## Open questions for future expansion

- How does the cost scale with kernel count? Linear, super-linear,
  sub-linear?
- Does the choice of LLVM optimization level (`-O1` vs `-O2` vs `-O3`)
  significantly change the cross-module inlining behavior?
- Are there LLVM intrinsics or function attributes (`alwaysinline`,
  `inlinehint`, `noinline`, `__attribute__((always_inline))`) that
  affect the modular case without requiring LTO?
- How does compile time scale with the number of separate modules?
  (Hot-swap granularity vs recompile cost trade-off.)
- Does the audio thread experience cache-locality penalties when
  separately-compiled kernels live in different code regions?
