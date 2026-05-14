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

Each variant gets a small C++ harness that constructs the LLVM IR
directly (not via tropical's compiler) so we isolate the LLVM-level
optimization behavior from any tropical-specific transformation.

Structure:

```
benchmark/llvm/
  inlining_across_modules.md       — this doc
  bench/
    common.hpp                     — shared scaffolding
    bench_monolithic.cpp           — variant 1
    bench_separate_funcs.cpp       — variant 2
    bench_separate_modules.cpp     — variants 3 and 4
    bench_external_call.cpp        — variant 5
  data/
    YYYY-MM-DD-results.md          — measured results per run
```

Each `bench_*` produces a CSV row per (tree, alive) combination with
the measured metrics. The aggregated CSV across runs lives in
`data/`.

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

(To be filled in as benchmarks run. Append timestamped entries.)

### YYYY-MM-DD — pending

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
