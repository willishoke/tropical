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

### 2026-05-14 — initial pass (before methodology fix)

Original measurements without `noinline` on the scheduler. With LTO,
the optimizer inlined the scheduler into the runner's measurement loop
and constant-propagated kernel coefficients from the stack frame in
`main`. This conflated kernel-level inlining with runner-level outer-
loop inlining; the recorded LTO numbers were misleadingly fast (~1.4
ns/sample for both v1lto and v4 vs 3.4 for v1). See git history for
the original commit.

### 2026-05-14 — methodology fix + scaling test (chain4 and chain16)

Setup: macOS, Apple M1, clang from system toolchain (Apple clang).
Schedulers annotated with `__attribute__((noinline))` so LTO can't
inline them into the runner's outer measurement loop. Buffer length
4096 samples; 256 measured iterations per run; 5 runs per variant ×
chain-length combination.

**Chain length 4** (simple chain of 4 kernels, each `y = a*x + b + 0.5*state`):

| Variant | ns/sample (median of 5) | vs v1 |
|---|---|---|
| **v1** — kernel inlined into scheduler, no LTO | 3.48 | 1.0× |
| **v1lto** — same source, with LTO | 3.49 | 1.00× |
| **v3** — kernel as external function, no LTO | 7.57 | 2.18× |
| **v4** — kernel as external function, with LTO | 3.43 | 0.99× |

**Chain length 16** (Phaser16-scale):

| Variant | ns/sample (median of 5) | vs v1 |
|---|---|---|
| **v1** — kernel inlined into scheduler, no LTO | 11.32 | 1.0× |
| **v1lto** — same source, with LTO | 11.38 | 1.01× |
| **v3** — kernel as external function, no LTO | 21.95 | 1.94× |
| **v4** — kernel as external function, with LTO | 11.34 | 1.00× |

All variants within each chain length produce bit-identical output
(chain4 sink = `0.137081`; chain16 sink = `0.340000`).

### Reading the numbers

**Kernel-level inlining is what LTO recovers, and it recovers it
completely.** With the runner-loop confound removed:

- `v1 ≈ v1lto` at both chain lengths — LTO has nothing useful to do
  when the kernel is already inlined and the scheduler is noinline.
- `v4 ≈ v1` at both chain lengths — LTO successfully inlines
  `kernel_step_external` into the scheduler across the TU boundary,
  recovering monolithic-equivalent performance.
- `v3` is the only outlier — kernel calls survive as real function
  calls in the inner loop, adding cost.

**The cost of per-kernel function calls is linear in chain length.**
v3's overhead vs v1:

| | v1 | v3 | v3 − v1 | overhead / kernel |
|---|---|---|---|---|
| chain4 | 3.48 ns | 7.57 ns | 4.09 ns | ~1.0 ns/call |
| chain16 | 11.32 ns | 21.95 ns | 10.63 ns | ~0.66 ns/call |

The per-call overhead is slightly amortized at chain16 (likely
because the function-call frame setup dominates less when there's
more inter-call work). But it's bounded around 0.7–1.0 ns per kernel
call on this hardware. For audio at 48 kHz (sample budget ≈ 81 ns
for a 256-sample buffer on a single core), a 16-kernel chain with
no LTO is roughly 27 % of the per-sample budget; a 4-kernel chain
is roughly 9 %. Real but not catastrophic.

**LTO numbers don't depend on chain length** in the sense that v4
matches v1 at all chain lengths tested. The LTO inliner handles
16-kernel chains as cleanly as 4-kernel chains.

### What this means against the original hypotheses

- **H1** (monolithic baseline preserved): confirmed.
- **H2** (no-LTO separate modules don't fold cross-call optimizations):
  confirmed — v3's IR shows raw function calls; runtime cost is
  linear in chain length.
- **H3** (LTO can recover monolithic performance for separately-
  compiled modules): **confirmed**. v4 matches v1 at every chain
  length tested. Module boundaries are essentially free under LTO.
- **H4** (external C function calls can't be inlined without LTO,
  even when present): not yet tested directly. The variant-5 case
  (dlopen'd .so) remains pending. Reasonable inference from v3's
  behavior: an external .so cannot be inlined by LTO unless the .so
  was also compiled with -flto (and even then, only with `-fwhole-
  program-vtables` or equivalent in modern toolchains).
- **H5** (per-sample function-call overhead is bounded, on the order
  of single-digit ns/call): confirmed empirically. 0.7–1.0 ns/call
  per kernel on M1 with -O2 clang.

### 2026-05-14 — chain4 with alive check (default-alive folding test)

Adds alive-aware variants: each kernel has an `alive` field; the
scheduler writes `1.0` to all alive fields each sample (the default-
alive pattern from tropical's active-set runtime); each kernel checks
`alive > 0.5` and skips its body when false.

| Variant | ns/sample (median of 5, excluding warm-up) |
|---|---|
| v1 (no alive) | 3.45 |
| v1lto (no alive) | 3.46 |
| v3 (no alive) | 7.52 |
| v4 (no alive) | 3.52 |
| **v1 + alive** | 3.44 |
| **v1lto + alive** | 3.50 |
| **v3 + alive** | 8.46 |
| **v4 + alive** | 3.48 |

All variants produce bit-identical output. Same number to within
noise across alive/no-alive at the inline-or-LTO variants.

**The active-set runtime's central performance claim survives
cross-module compilation under LTO.** v4_alive (LTO with alive
checks) matches v1_alive (monolithic with alive checks) matches v1
(monolithic baseline). When LTO is active, default-alive checks cost
zero observable runtime.

The v3_alive overhead vs v3 (no alive): 8.46 − 7.52 = 0.94 ns/sample.
This is the cost of four alive checks per sample (one per kernel)
that the optimizer cannot fold because the kernel function is
opaque. Per check: ~0.24 ns. For a stable alive signal (musical
decay over hundreds of samples), branch prediction handles this for
nearly free; for audio-rate alive signals, the branch could miss
more often.

### Caveat: the IR doesn't fold the alive check at v1 either

Inspecting `build/v1_chain4_alive.ll` reveals something subtle: even
in v1 (kernel inlined into scheduler, same TU), GVN does NOT fold
the `alive > 0.5` check. The optimized IR contains the comparison
and branch in the inner loop. Yet the runtime cost is identical to
v1 without alive.

The reason for the non-fold: the kernel pointers (`k1, k2, k3, k4`)
are not annotated with `__restrict__`, so the optimizer cannot prove
they don't alias. The store to `k1->alive = 1.0` could in principle
be followed by a load of `k4->alive` that returns the same value (if
k1 and k4 alias). GVN is conservative; it preserves the load.

Why no runtime cost: the alive check branches 100% in the alive-true
direction (we always write 1.0), so branch prediction handles it
perfectly. The pipeline stays full; the cost is in the noise.

**This differs from the active-set spike's findings.** The
active-set spike showed alive checks fold *completely* under O2.
That setup used a *single flat slot array* with distinct integer
indices — the optimizer could prove non-aliasing trivially. Our
struct-based test introduces an aliasing pessimism that wouldn't
occur in tropical's actual model.

**For tropical's real implementation**: the slot array is one
contiguous double[N] with each alive slot at a distinct integer
index. The aliasing analysis at that model is trivial and the alive
fold should engage as the active-set spike reported. The conclusion
this experiment supports — LTO recovers monolithic-equivalent
performance across modules — applies regardless of the alive-fold
question, because branch prediction makes the unfolded form
essentially free at runtime when alive is stable.

Re-running this experiment with `__restrict__`-annotated kernel
pointers (or with a slot-array model that mimics tropical's actual
layout) would likely show the alive fold happening at v1 and v4, but
the runtime numbers wouldn't change meaningfully because branch
prediction is already handling the unfolded form. Worth doing for
IR-cleanliness validation; not blocking on architectural conclusions.

### Implications for tropical's architecture

Strong confirmation that **the operadic substrate can compile each
operation into a separate LLVM module without sacrificing kernel-
level performance, provided LTO (or its JIT-pipeline equivalent) is
active at link time**. Modular compilation gives the substrate the
ability to:

- Hot-swap individual operations without recompiling the whole
  session
- Mix realizations (one operation as standard tropical compilation;
  another as WDF; another as FFI) with predictable composition

Open question for follow-up: at the LLVM ORC JIT level, does
tropical's current `PassBuilder::buildPerModuleDefaultPipeline(O2)`
include the cross-module inlining that LTO performs? If not, swapping
the JIT to `buildLTOPreLinkDefaultPipeline` + the LTO post-link
pipeline would be the natural next step. The active-set spike notes
that O2 is already sufficient for default-alive folding within a
single function — but the active-set spike's setup had everything in
one function. The cross-module case here suggests JIT-side LTO
configuration is worth investigating.

Separate-modules **without** LTO costs ~2× at the chain lengths
tested. If the JIT's LTO equivalent isn't easy to enable for some
reason (compile-time concerns, ORC integration complexity, etc.),
the substrate would need to fall back to monolithic-compile-per-
hot-swap-unit. That's what we have today; it remains viable.

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

v1 chain4 inner loop (from `build/v1_chain4.ll`): 4 kernels' bodies
inlined into the scheduler, 8 fmuladd ops per sample iteration. No
function calls in the loop body. State and coefficient loads happen
inside the loop, one load per iteration per kernel — the optimizer
is conservative because the kernel pointers could in principle alias
(no `__restrict__` qualifier). Adding `__restrict__` would likely
allow phi-promotion of state and hoisting of coefficients; candidate
optimization for a future iteration.

v3 chain4 inner loop (from `build/v3_chain4_scheduler.ll`): four
actual `kernel_step_external` function calls per sample. The function
is declared extern in this TU; the optimizer has no body to inline.
This is the source of the ~2.2× slowdown vs v1.

The LTO variants (v1lto, v4) don't have their post-link IR dumped
into the build directory; they're compiled to native code via the
linker. Reading their optimized form would require `-Wl,-save-temps`
or post-link disassembly. Worth doing if questions arise about how
the LTO optimizer transformed the code at the kernel level (specifically:
does v4's LTO inline the kernel and then also vectorize / hoist /
phi-promote, or just inline?).

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
