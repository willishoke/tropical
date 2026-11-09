# The typed kernel — one pure function at the port

Status: **Landed through T3** (2026-10-28). Decided in conversation with
the owner; T0–T3 are built, T4 is folded into T3 (the scope reads output
channels), T5 is landed with T3 (the web runtime binds the port), T6 and T7
remain. The "What it states" sections below are the design; the
"As built" section records where the implementation settled. This document *replaces* §2, §3 and §4 of
`kernel-artifact-abi.md` and narrows §1, §5 and §6. It honours `cf-only.md`
without amendment: a kernel was always `f(τ, params)`, and this document
makes the port say so. `architecture.md` and `compatibility-matrix.md` are
amended by the migration section at the end, not by this text.

## The decision

A kernel is one typed pure function:

> **inputs : a record of named, shaped, typed values** (one of which is τ)
> **→ outputs : a record of named, shaped, typed values.**

Shapes are **static**: every extent is part of the kernel's type and is fixed
when the artifact is compiled. A different buffer length is a different
artifact, exactly as a different patch, mode or realization already is.

That is the whole port. There is no second kernel, no rate, no time axis
other than the τ input, no slot the kernel writes for the host to read back,
no argument the kernel ignores, and no state. Everything the current ABI
carries beyond this is a realization detail that moves below the port or is
deleted.

## Why one kernel and not a graph

The stage-0 split (`30dc01f`, 2026-07-08) separated the parameter-only work
`g(p)` from the per-sample work `f′(τ, g(p))` so the host could run `g` only
on parameter writes. It was bit-identical to the unsplit kernel (the goldens
byte-gated it) because it is currying, not a semantic change. The reasons it
was exposed at the port each turn out to be realization concerns:

- **Memoization across buffers.** Inside one call LLVM already hoists
  τ-independent work out of the sample loop, so `g` runs once per call. The
  split only lets `g` run *less* than once per call. For a K-mode bank that
  is K transcendental evaluations amortized over the buffer. A host that
  wants fewer makes bigger buffers.
- **Compile time.** The split's headline win was compile wall: one enormous
  function sent LLVM's backend superlinear. The emitter can still produce
  the hoisted preamble as a separate internal function or module behind
  one entry symbol. Same cure, invisible from outside.
- **Metal.** A GPU kernel is one thread per sample with no loop to hoist
  out of, so `g` must be a CPU pre-pass or a separate dispatch there. That
  is the Metal *adapter's* realization of one kernel, below the port.
- **Coherence.** The three-generation column protocol exists because `g`'s
  outputs land in storage the audio kernel reads later. With one kernel the
  kernel reads its input record at call entry; the rule collapses to "do not
  mutate inputs during a call", the ordinary contract of any pure function.

The absolute-time tile kernel (`d7d9e1d`, #239, 2026-08-19) is different in
kind: τ-dependent work evaluated on a coarser grid than τ, which is a second
time axis inside the kernel, admitted for one recognized phaser shape, filled
by a 652-line DSP-specific materializer inside the runtime, and disabled
unless two research flags are set. It is deleted, not reworked.

## The type

```
Kernel
  inputs  : [ (name, elem, shape, role) ]
  outputs : [ (name, elem, shape) ]
```

- `name` is the identity that survives a hot-swap. Controls project from one
  artifact to another by exact name, as slot names do today.
- `elem` is the element type. The JIT and wasm realizations use `f64` for
  values and `i64` for rails; Metal refines values to `f32` under the
  documented tolerance gate. τ's element type is the clock rail: `i64` in
  Q48.16, spelled `i64q16` (`fixed-carrier.md` is normative), so a
  fractional coordinate is representable and an integer-sample host writes
  `index << 16`.
- `shape` is a static extent list. A scalar is `[]`. The audio kernel's τ is
  `[N]`; its output is `[N, C]` frame-major, `output[s*C + c]`, as
  `stereo-output-contract.md` already states. Shapes are rate- and
  colour-agnostic; a video kernel's output is `[H, W, 3]` by the same rule.
- `role` annotates an input for the host's benefit only: `param` (a control
  the host writes under a discipline), `signal` (a buffer the host refreshes
  every call, such as live input), `table` (a constant the host seeds once).
  Roles never change the kernel's semantics; they tell a host what to do
  between calls.

The rate the plan was compiled for is a **constant of the artifact**
recorded in the manifest, not an input. Stage 0 already treats rate-only
expressions as foldable, so this is the existing behaviour made explicit.

### What the type excludes

- **No slot the kernel writes.** Inter-instance wires and coefficients are
  internal. The Metal "kernel-written port slots never write back" wart
  disappears because there is nothing to write back.
- **No second kernel.** `coefficientIr`, `coef:<n>` slots,
  `coeff_array_slots`, `tileIr`, `tile_array_slots`, `tile_interval_frames`
  and `phaser_time_staging` do not exist at the port.
- **No reserved arguments.** `%inputs`, `%registers`, `%temps` and
  `%param_ptrs` are gone. `register_count` is an internal scratch size the
  realization may use, not something a host allocates.
- **No `buffer_length`, no `start_sample_index`, no `sampleRate`.** N is in
  the type; τ is an input; rate is a constant.
- **No state**, as before. The kernel may be called for any τ array, in any
  order, and returns the same outputs for the same inputs.

### τ as an array

τ is passed, not derived. The audio host fills `τ[s] = (origin + s) << 16`,
a trivially vectorized fill. The scope fills whatever coordinates it wants,
including a backward scrub at a fractional speed. One kernel, two callers;
`render_window` stops being a second entry point and becomes a host that
constructs a τ array. The compiler specializes on the **static shape** `[N]`
(a known trip count), never on τ's values; there is no range fast path. If a
τ load is ever shown to cost something measurable, a specialization can be
added then, with the benchmark that justifies it.

## The artifact

```
KernelArtifact
  ir        : String   LLVM IR text defining one function, tropical_kernel
  msl       : String   Metal Shading Language, or "" (JIT-only)
  manifest  : String   the kernel type + the plan metadata the gates need
```

On disk: `kernel.ll`, `kernel.metal`, `manifest.json`. The artifact is
immutable once emitted.

### The LLVM entry point

One pointer per input, one pointer per output, in manifest order. Nothing
else.

```
define void @tropical_kernel(
  ptr noalias nocapture readonly %in_0,   ; inputs[0], elem/shape per manifest
  …
  ptr noalias nocapture readonly %in_k,
  ptr noalias nocapture          %out_0,  ; outputs[0]
  …
  ptr noalias nocapture          %out_m)
```

A scalar input is a pointer to one element so that every argument has the
same form and a host can bind by index without a per-kind calling
convention. The function performs no allocation, no I/O and no calls other
than the LLVM math intrinsics in `EmitLlvm.intrinsicDecls`. The wasm32
export has the same arguments as `i32` offsets into linear memory.

### The Metal entry point

One buffer binding per input and per output, in manifest order, plus the
thread position. No constants struct, because there are no constants left.

```
kernel void tropical_kernel(
    constant <elem>* in_0  [[buffer(0)]], …,
    device   <elem>* out_0 [[buffer(k+1)]], …,
    uint s [[thread_position_in_grid]])
```

Values cross f64→f32 as host-written data; τ stays i64. The grid extent is
N from the type. A Metal realization may run a CPU pre-pass for the
τ-independent part of the kernel and bind its result as an additional
*internal* buffer; that buffer is not in the manifest and a host never sees
it. The pre-pass runs in f64 on the CPU, which preserves the SNR the current
column crossing achieves.

### The manifest

```json
{
  "schema":  "tropical_kernel_1",
  "rate":    44100.0,
  "inputs":  [ { "name": "tau",          "elem": "i64q16", "shape": [512],  "role": "signal" },
               { "name": "param:freq",   "elem": "f64",    "shape": [],     "role": "param",
                 "default": 220.0, "discipline": "anchor", "companion": "param:freq.tau_base" },
               { "name": "in:mic",       "elem": "f64",    "shape": [512],  "role": "signal" },
               { "name": "table:wave",   "elem": "f64",    "shape": [2048], "role": "table",
                 "default": [ … ] } ],
  "outputs": [ { "name": "out",          "elem": "f64",    "shape": [512, 2] } ],
  "realization": { "mode": "fused", "register_count": 0 },
  "plan": { … tropical_plan_7 metadata, for the gates; a host that reads it is coupling to the compiler … }
}
```

`param_disciplines` becomes per-input annotation. `host-param-dispatch.md`
stays normative for the arithmetic; nothing in it changes, because the
disciplines were always host arithmetic on inputs between calls.

A host reads `schema`, `rate`, `inputs`, `outputs`. `realization` is for the
loader (which mode to compile). `plan` is for `tropicaltest` and is
explicitly not a contract.

## Hosts

A host owns an **input record**: one buffer per input, sized by the type,
seeded from `default` where present. Per call it:

1. writes τ (and any `signal` inputs it has fresh data for);
2. calls the kernel once;
3. reads the outputs.

Between calls it writes `param` inputs under their disciplines. It never
writes any buffer while a call is in flight; that is the entire coherence
rule. Two generations of the input record (the one the audio thread is
reading, the one the control thread is writing) and one atomic flip replace
the three-generation column protocol and the slot capture.

**Realtime input.** A `signal` input of shape `[N]` aligned with τ. The host
copies the device's current input buffer in before the call. Latency is one
buffer, chosen by the user through N. Nothing writes back; any feedback a
user builds by routing an output to the next call's input is an explicit,
buffer-delayed loop in the host, outside tropical, and the kernel never sees
a cycle. Under `cf-only.md`'s two time axes, a live input is a parameter of
array shape that changes on performance time; scrubbing τ holds the last
buffer constant exactly as it holds a knob.

**Hot-swap.** Unchanged in substance: the host keeps its origin coordinate,
projects controls into the new artifact's input record by name, and flips.
There is no register, array or slot transfer. Inputs the new type lacks are
dropped; inputs it adds take their defaults.

**Scope.** A separate artifact of the same plan, with its own N and its own
τ array. This is what "dual publication" has been in disguise; it stops
needing a special load entry point.

## What realizes the type (below the port)

- **Binding-time analysis** (`Ir/Stage0.lean`'s classifier) survives as an
  optimization with no name at the port: it feeds LICM on the JIT and the CPU
  pre-pass on Metal. Its typed-stage theorems
  (`proof-initiative-4-staging-refinement.md`) still hold; they now gate a
  realization, not a manifest field.
- **`bankSum`**, the block carrier and the modal math are untouched; they run
  at assemble time in Lean and emit instructions, not kernels.
- **The sample loop** is a compiler-emitted realization of map over `[N]`.
  Fused is the only realization a host must accept. Microkernel mode stays,
  if at all, as a JIT-internal equivalence oracle in `tropicaltest`, never
  as a host obligation.
- **The compile cache** keys on the plan *and the type* (N, C, rate), so the
  scope's artifact and the audio artifact are two cache entries of one plan.

## What is deleted

| today | fate |
|---|---|
| `tileIr`, `Ir/TileTime.lean`, `TROPICAL_PHASER_TIME_STAGING*` | deleted |
| `engine/metal/PhaserHigherOrderMaterializer.*`, `MetalRenderWorker`'s tile path | deleted |
| `tile_array_slots`, `tile_interval_frames`, `phaser_time_staging`, `metal_execution` | deleted from the manifest |
| `coefficientIr`, `coef:<n>` slots, `coeff_array_slots` | internal to the realization; not in the artifact |
| three-generation column buffering, slot capture | replaced by two input-record generations and one flip |
| `%inputs`, `%registers`, `%temps`, `%param_ptrs`, `buffer_length`, `start_sample_index`, `sampleRate` | gone from the signature |
| `TropicalKernelConsts`, buffers 0–3 fixed layout | replaced by per-input bindings |
| kernel-written module slots, `tropical_runtime_get_slot` on them | gone; outputs are the only thing a host reads |
| `tropical_runtime_render_window` | a host-side τ fill against the scope artifact |
| `load_ir_time_staged*` family (three entry points) | one `tropical_runtime_load(artifact)` |
| `compilation_mode = microkernel` as a host-visible mode | test-internal or deleted |
| `web/runtime/manifest.ts` fields | `inputs`/`outputs` |

## Migration

Goldens are the gate throughout. The JIT path's arithmetic does not change
(the stage-0 split was byte-identical going in, so folding it back behind one
entry point is byte-identical coming out); every step below must hold
`tests/golden/` byte-for-byte and `metal_vs_jit` at its current SNR.

| step | work | exit check |
|---|---|---|
| T0 | Delete the tile kernel, the materializer, the flags and the manifest fields. | `make validate` green; `PhaserHigherOrderMaterializer` gone; no `tile_` in `Plan.lean` |
| T1 | `Tropical.Kernel` type in core (inputs/outputs/elem/shape/role) and `tropical_kernel_1` manifest; emitted alongside plan 6, not yet consumed. | `tropicalc` writes `kernel.json` (the type, with the plan nested) beside `manifest.json`; the core-tier `typed-kernel` gate derives, shape-checks and round-trips it for every corpus patch |
| T2 | EmitLlvm emits the port with the coefficient plan as its prologue; EmitMsl binds `tau` with static frames and a literal rate. **Landed.** | goldens byte-identical; MSL goldens re-frozen once, deliberately |
| T3 | FlatRuntime loads one artifact over `tropical_kernel_1`, calls the argv thunk with tau and the param cells, keeps the Metal pre-pass. `load_ir*` family deleted. **Landed.** | goldens; `param_dispatch_conformance` green; TSAN suite green |
| T4 | τ as an input: the host fills it; `render_window` renders the observation kernel's output channels. **Landed with T3.** | scope output identical to the previous `render_window` for integer coordinates |
| T5 | Web runtime: `KernelManifest` is `tropical_kernel_1`; `wasm_vs_jit` and `web_plans_vs_jit` on the new ABI. **Landed with T3.** | both suites green |
| T6 | Realtime input: a `signal` role, a DAC input path, one patch that uses it. | a feedforward patch (delay-free FIR on mic) renders byte-identically between a recorded buffer and the live path |
| T7 | Retire `kernel-artifact-abi.md` §2–§4; update `architecture.md`, `compatibility-matrix.md`, `engine/CLAUDE.md`. | docs describe only what exists |

T0 and T1 are independent. T2 and T3 must land together or behind a flag,
since the signature and the loader change in lockstep. T6 is the only step
that adds behaviour and can wait.

## As built (T2/T3, 2026-10-28)

The three open questions were taken as recommended: τ is the Q32.32 rail
(`i64q32`), scalars bind as one-element buffers, `role` is a field of the
port. Where the implementation settled differently from the text above:

- **The thunk.** Beside `@tropical_kernel` the module exports
  `@tropical_kernel_argv(ptr argv)`, which unpacks a pointer table and calls
  the port. A C host cannot call a function of arbitrary arity; the thunk is
  the host's convenience, not part of the type. `compile_ir_text` renames
  every definition and looks the thunk up.
- **τ is the IR's time leaf.** The resolved IR's only time primitive is
  `ENode.tau`, the Q32.32 coordinate itself (Plan source `tau`, bound by the
  emitters to `tau[s]`); the integer sample index is the derived expression
  `tau >> 32`, spelled where a builder wants the integer grid (a glide's
  start tick, a seed) and nowhere in the emitters. Every clock is an
  expression over τ, and the one place a clock multiplied τ as an integer
  tick — `ScrubClock`'s `base + velocity · τ` and the playground master
  clock — now uses the exact split product `scaleClockQ`. So a fractional τ
  evaluates the closed form at that fractional instant, gated by
  `fractional-tau` in `tropicaltest`: `ScrubClock(½)` on the integer grid at
  τ = 2k+1 is byte-identical to the default clock at τ = k+½. The host-side τ
  fill with explicit coordinates is `tropical_runtime_render_tau`
  (`Runtime.renderTau` in Lean); `render_window` remains the integer
  progression over it. (Landed 2026-10-28, the day after T2/T3; an
  integer-grid render is byte-identical to before, the wire-plan and MSL text
  goldens were re-frozen once because the `tau << 32 >> 32` round trip is
  gone from the emitted plan.) Because this changes what plan source 0 means
  without changing any field, the plan schema is now `tropical_plan_7`: a
  `tropical_plan_6` document (source 0 the integer sample index) would
  decode to a different function, so both decoders refuse it with a
  recompile message (2026-10-29). The `fractional-tau` gate also drives
  `scaleClockQ` at dense fractional τ, where its low-word partial products
  are live.
- **Output channels are named.** The `out` port carries `channels`, one
  source name per channel, and a scope tap is an output channel of the
  observation artifact (the playground routes each tap to its own channel).
  `render_window` resolves names to channels and renders a wider window in
  chunks of the kernel's frame count.
- **Metal keeps a CPU pre-pass.** A Metal artifact carries `stage0Ir`, the
  coefficient plan as `@tropical_stage0(slots, arrays, array_sizes, temps)`
  over host storage, run at load and after every control write; the GPU
  kernel reads the slot vector and the hoisted coefficient columns it
  produces. The three-generation column protocol therefore survives for
  Metal only. The MSL binding is `output_buffer`(0), `slots`(1), `tau`(2),
  `coeff_columns`(3); a render tile is `Rgpu / frames` dispatches in one
  command buffer. Metal's own 31-binding limit is why scalars are not one
  binding each there; the port is unchanged.
- **Three generations of the control slots remain on the JIT path**, for
  the same reason the slot generations always had three: a control stream
  can publish twice inside one audio buffer. The coefficient generations and
  the slot capture of columns are gone from the JIT path.
- **Microkernel realization is deleted.** The plan's `compilation_mode`
  field survives as `fused` only; a host refuses anything else.
- **`get_slot` on a kernel-written slot** returns the host's seed on every
  backend now, uniformly; nothing reads such slots any more.
- **τ is Q48.16 (2026-11-01).** The rail moved from Q32.32 to Q48.16: 16
  fraction bits (2⁻¹⁶ sample, ≈ 0.3 ns at 48 kHz) buy a horizon of 2⁴⁷
  samples (93 years at 48 kHz) where Q32.32 went negative after 2³¹ (12.4 h).
  Q32.32 had broken non-integer-speed clocks, strike anchors and the derived
  `sampleIndex` past that point; `long-horizon` in `tropicaltest` pins all
  three. τ's element type is `i64q16`, so a manifest built for the old rail
  is refused, and `tropical_plan_7` was not yet released, so it keeps its
  name with source 0 now the Q48.16 coordinate. Speed stays a Q·.32 scale and
  phase Q0.32; audio goldens are byte-identical. The change also fixed a
  nested-output read typed `float` whatever its producer declared, which put
  wired clocks through f32 on Metal; the old τ+2⁴⁰ Metal gates had not seen
  it because `2⁴⁰ << 32` wraps to τ = 0.

## Embedded objects (spike, 2026-11-09)

The same port, compiled for a microcontroller with no FPU (Cortex-M0+,
`thumbv6m-none-eabi`) and handed over as a **relocatable object**, not a
linked image. A firmware, or a host test bench, links it with its own
linker script.

```
diffcli compile-object patch.json --out k.o --triple thumbv6m-none-eabi \
    --cpu cortex-m0plus --frames 16 --rate 48000 --out-elem i32 --out-format Q1.15
```

`compile-object` takes the IR `EmitLlvm` writes for the JIT and for
`compile-wasm`, runs the JIT's O2 pipeline against the target (without loop
unrolling or vectorization, so one loop iteration is one frame and flash
stays small) and stops at the object. The triple may be the host (default),
`thumbv6m-none-eabi`, or `wasm32-unknown-unknown` (a wasm object, not a
module). Beside the object it writes `k.o.manifest.json`: the
`tropical_kernel_1` manifest plus an `object` block with the triple, the
symbol, N, the per-port layout and format, and the numeric contract.

**Numeric contract.** IEEE binary64 for float values, two's-complement
wrapping i64 for ints, round-to-nearest-even, no FP contraction and no
fast-math. `EmitLlvm` writes no `fast`/`contract`/`afn` flags and no
`llvm.fmuladd`, and the toolchain keeps `AllowFPOpFusion = Standard`, so
no FMA is ever formed. Removing unrolling and vectorization changes no
value, because nothing in the IR may be reassociated.

**ABI and buffers.** The C calling convention of the triple (AAPCS on
ARM), one pointer per port in port order:

```
void tropical_kernel(const int64_t *tau,   /* N × i64, Q48.16, 8-byte aligned */
                     const double  *param, /* one f64 per param: slot (none in the LFO) */
                     int32_t       *out);  /* N × C i32 frame-major, 4-byte aligned */
```

Buffers are little-endian. The thumbv6m object has `.text` only (literal
pools sit inside it), with no `.data`, `.bss` or `.rodata` to initialize.
It also carries `.ARM.exidx` (16 bytes of `EXIDX_CANTUNWIND` entries,
which a linker script may discard), `.stack_sizes` (not loaded) and
`.ARM.attributes`: v6S-M, Thumb-1, soft float with no FP instructions
(`Tag_FP_arch` is absent).

**The kernel.** `FixedSinOscQ15` is `FixedSinOsc` for 32-bit integer
hardware. Its DDS phase on ℤ/2³² (`phaseQ32Lo`) is bit-identical to
`FixedSinOsc`'s and uses only 32-bit products: the whole-sample product is
wanted mod 2³², which is the low word that `MULS` computes, and the
fraction's product is split at bit 16 of the increment. The sine
(`fixedSinQ15Sig`) rounds the phase to 16 bits, folds it to a signed quarter
turn and evaluates an odd degree-7 polynomial in Horner form, with 16-bit
operands, 32-bit products and rounding shifts. The output is an int in
[-32767, 32767] that means `out / 32768` (Q1.15 in an i32; it also fits an
i16). Error, measured exhaustively over all 2¹⁶ rounded phases: 1.63 LSB
against `32768 sin(2π h/2¹⁶)`. Rounding the phase to 16 bits adds at most
π/2 LSB, so the bound against the float sine of the full 32-bit phase is
3.2 LSB (9.8e-5). The measured worst case over 65 536 frames is 2.67 LSB.

On the M0+ table (ALU and `MULS` 1 cycle, loads and stores 2, taken branch
2, `BL` 3) the hot loop is 102 instructions, 8 of them `MULS`, and takes
118–122 cycles per frame. At N = 16 a call costs at most 1 989 cycles, of
which 37 are prologue and epilogue. That is about 9% of a 1 333-cycle
sample budget at 64 MHz and 48 kHz. Flash is 524 bytes of `.text` (the
`_argv` thunk inlines a second copy, and a firmware that does not reference
it can drop it with `--gc-sections`). Stack is 32 bytes, and there is no
static RAM. `llvm-nm -u` lists nothing: no `__aeabi_*`, no `__muldi3`, no
soft-float helpers. The host object matches the JIT sample for sample over
65 536 frames, and at τ = 2⁴⁰ samples.
`tools/embedded_object.py` builds all three objects, reports these numbers
and gates the libcalls (`make validate` runs it with `--check-only`).

**Formats and cost.** Q15 × Q15 fits one 1-cycle `MULS` (16 × 16 → 32).
Q31 × Q31 needs the high word of a 64-bit product. ARMv6-M has no `UMULL`,
so that is an `__aeabi_lmul` call or about four `MULS` plus adds and
carries. That is why the sine is Q1.15. The phase stays Q0.32, because its
products are only wanted mod 2³².

**What the spike changed in the core.**

- `mulLo32` is a new op: `a · b` mod 2³², sign-extended onto the int rail.
  LLVM does not narrow `(a·b << 32) >> 32` over i64 `mul` on thumbv6m. It
  calls `__muldi3`, and the disassembly showed that before the op existed.
  The op is written as `trunc`/`mul i32`/`sext` (EmitMsl:
  `(long)(int)((uint)a * (uint)b)`).
- Int values cross module slots raw. Slots were f64 in `EmitLlvm`, so a
  wired clock went i64 → f64 → i64, which is a soft-float call pair per
  frame on a no-FPU target and loses bits past 2⁵³ rail units on any
  target. An int write now also lands in `%islots`, and an int read of a
  slot whose last emitted write was int loads that word. This is the rule
  EmitMsl already follows.
- The port functions are `nounwind`, so the object does not reference an
  EH personality routine.

**Gaps left open (smallest honest fix for each).**

1. *Params are f64 by contract.* A knob on a no-FPU target means soft-float
   `fmul`/`fdiv`/`fptosi` calls. LICM moves them out of the frame loop, but
   they stay in the object. The spike's LFO therefore fixes `freq` as a
   literal: changing it is a structural edit, which recompiles. Fix: a
   per-port element type for params (`i32` with a Q format, or `i64`) that
   the emitter loads as an int and that never enters an f64 slot. `Elem`
   already carries the vocabulary.
2. *The int rail is i64.* `mulLo32` keeps the products 32-bit, but each
   rounding shift after a product is still an i64 add-and-shift (about four
   extra cycles), and the clamp and the loop counter are 64-bit. That costs
   roughly 25 of the 120 cycles per frame. Fix: an `int32` scalar kind
   (ℤ/2³², the phase's own ring), or emitter range analysis that narrows
   values proven to fit 32 bits.
3. *The output's Q format is a flag, not a type.* `--out-elem i32` needs
   one sink input at gain 1 and stores the int output directly. `Q1.15`
   comes from `--out-format`, not from the program. Fix: give int output
   declarations a `fracBits` and carry it into `Kernel.Port`.
4. *f32* as a per-target float format on the LLVM path was not attempted.

**Clocks across nodes.** The kernel reads τ from its caller. Exact ratios
(phase-locked partials, the ℤ/2³² phase identity) hold only between
kernels evaluated on the same τ. Whether separate nodes share τ by
distributed sync or drift is the host's decision and is not settled here.
