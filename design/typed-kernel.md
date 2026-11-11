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
  values and `i64` for rails, or `f32` for values in an artifact compiled at
  binary32 (`--float f32`, "The float format per target" below); Metal
  refines values to `f32` under the documented tolerance gate. τ's element type is the clock rail: `i64` in
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
    --cpu cortex-m0plus --frames 16 --rate 48000
```

`compile-object` takes the IR `EmitLlvm` writes for the JIT and for
`compile-wasm`, runs the JIT's O2 pipeline against the target (without loop
unrolling or vectorization, so one loop iteration is one frame and flash
stays small) and stops at the object. The triple may be the host (default),
`thumbv6m-none-eabi`, or `wasm32-unknown-unknown` (a wasm object, not a
module). Beside the object it writes `k.o.manifest.json`: the
`tropical_kernel_1` manifest plus an `object` block with the triple, the
symbol, N, the per-port layout and format, and the numeric contract. Every
port's element type and fixed-point format come from the program (see
"Typed ports" below); no flag sets them.

**Numeric contract.** IEEE binary64 for float values (binary32 in an
artifact compiled at `--float f32`, see "The float format per target"), two's-complement
wrapping i64 for `int` and i32 for `int32` with the total integer semantics
below, round-to-nearest-even, no FP contraction and no fast-math. τ is read
as one 64-bit word per frame; the frame counter is 32-bit (N is a static
constant). `EmitLlvm` writes no `fast`/`contract`/`afn` flags and no
`llvm.fmuladd`, and the toolchain keeps `AllowFPOpFusion = Standard`, so
no FMA is ever formed. Removing unrolling and vectorization changes no
value, because nothing in the IR may be reassociated.

**ABI and buffers.** The C calling convention of the triple (AAPCS on
ARM), one pointer per port in port order:

```
void tropical_kernel(const int64_t *tau,   /* N × i64, Q48.16, 8-byte aligned */
                     const int32_t *freq,  /* one word per param, at its own type:
                                              here int32 Q16.16 (an f64 param is a
                                              const double *, an int one int64_t *) */
                     int32_t       *out);  /* N × C words frame-major, Q17.15, 4-byte aligned */
```

That is `SinOscQ15`'s port (below). The manifest's `object.ports` gives each
buffer's element, format, bytes and alignment.

Buffers are little-endian. On ELF each function has its own section
(`.text.tropical_kernel`, `.text.tropical_kernel_argv`), and the literal
pools sit inside them. There is no `.data`, `.bss` or `.rodata` to
initialize. The object also carries `.ARM.exidx.*` (8 bytes of
`EXIDX_CANTUNWIND` per function), `.stack_sizes` (metadata, not loaded) and
`.ARM.attributes`: v6S-M, Thumb-1, soft float with no FP instructions
(`Tag_FP_arch` is absent). A minimal placement, checked with `ld.lld
--gc-sections`:

```
SECTIONS {
  .text : { KEEP(*(.text.tropical_kernel)) *(.text*) *(.rodata*) } > FLASH
  .data : { *(.data*) } > SRAM AT > FLASH
  .bss  : { *(.bss*) *(COMMON) } > SRAM
  /DISCARD/ : { *(.ARM.exidx*) *(.stack_sizes) }
}
```

**The kernel.** `FixedSinOscQ15` is `FixedSinOsc` for 32-bit integer
hardware, written in `int32`. Its DDS phase on ℤ/2³² (`phaseQ32`) is the
same residue as `FixedSinOsc`'s and uses only 32-bit products: the
whole-sample product is wanted mod 2³², which is the low word that `MULS`
computes, and the fraction's product is split at bit 16 of the increment.
The sine (`fixedSinQ15Sig`) rounds the phase to 16 bits, folds it to a
signed quarter turn and evaluates an odd degree-7 polynomial in Horner form,
with 16-bit operands, 32-bit products and rounding shifts. Past the phase
rounding no intermediate exceeds 2²⁹ in magnitude (checked over all 2¹⁶
rounded phases), so nothing there wraps. The output is an `int32` in
[-32767, 32767] that means `out / 32768` (Q1.15 in an i32; it also fits an
i16). Error, measured exhaustively over all 2¹⁶ rounded phases: 1.63 LSB
against `32768 sin(2π h/2¹⁶)`. Rounding the phase to 16 bits adds at most
π/2 LSB, so the bound against the float sine of the full 32-bit phase is
3.2 LSB (9.8e-5). The measured worst case over 65 536 frames is 2.67 LSB.
`FixedSinOscQ15`'s output declares no format (its port is a plain `i32`
word, `Q32.0`), so its frozen renders stay what they were.

**The live-knob LFO.** `SinOscQ15` is the same oscillator with fixed-point
ports, so a host or a firmware drives it with no float anywhere: `freq` is
Hz as an `int32` Q16.16 word (`[-32768, 32768)` Hz in steps of 2⁻¹⁶ Hz),
`phase` a Q0.32 offset in cycles, `out` the Q1.15 sine in an `int32`
(`Q17.15`). A patch wires a typed param into `freq`:

```json
{"op":"paramDecl","name":"freq","type":"int32","frac_bits":16,"value":2}
```

The port is Hz rather than the phase increment itself. Hz is what a host
or a firmware has, and its meaning does not depend on the rate; the
conversion costs 17 cycles per call, about one cycle per frame at N = 16.
The rate stays a constant of the artifact. `hzQ16ToIncQ32` turns the word
`F` into the Q0.32 increment `F · 2¹⁶ / rate` without a division or a 64-bit
multiply. `M = round(2⁴⁸ / rate)` is folded at compile time and split as
`q·2³² + Mh·2¹⁶ + Ml`, and the increment is
`q·F + Fh·Mh + ⌊Fh·Ml/2¹⁶⌋ + ⌊Fl·Mh/2¹⁶⌋` with `F = Fh·2¹⁶ + Fl`. That is
three `MULS`, at most 3.25 increment LSBs from exact (3.6e-5 Hz at 48 kHz;
the knob's own step is 1.5e-5 Hz). It depends only on the param, so it is
stage 0 and runs once per call, before the frame loop. In the thumbv6m
object it is `0x06`–`0x20`: `ldr` of the word, `asrs`, two literal loads,
three `muls`, and adds and shifts, 17 cycles. Then `0x22`–`0x2a` store the
three loop invariants (`inc`, its high and low halves) to the stack, which
costs 8 cycles. The phase is `phaseQ32Of32`, the `int32` form of `phaseQ32`
with the increment signed, so a frequency is taken modulo the rate into
`[-rate/2, rate/2)` and a negative one runs the sine backward.

**Cycles.** `tools/embedded_object.py` counts on the M0+ table (ARM DDI
0484, zero wait states: ALU and `MULS` 1, loads and stores 2, `PUSH`/`POP`
1+N, `POP` with PC 3+N counting PC, `B` 2, `B<cond>` 2 taken and 1 not,
`BL` 3). `MULS` at one cycle assumes the M0+ is built with the single-cycle
multiplier. The small iterative multiplier is a build option that takes 32
cycles, which adds 31 cycles per `MULS` (279 per frame for `SinOscQ15`'s
nine). Each report states the assumption as `timing_model`. The script
enumerates every path once around the frame loop, data-dependent branches
included, and adds the path that enters the loop and the one that leaves
it. The enumeration is exhaustive or the script exits non-zero: a path that
revisits an instruction (an inner loop) has no bounded cost, and past the
path limit the set is not known to be complete, so in either case no range
is printed. The epilogue's cheapest and dearest paths both enter the call
range (`exit_cycles` and `exit_cycles_upper`). An ARMv6-M emulator on the same table measured two objects, and
both measurements fall inside the computed ranges:

| object (N = 16) | entry | frame | exit | call | measured |
|---|---|---|---|---|---|
| #252 LFO (i64 rail) | 20 | 114–118 | 10 | 1 854–1 918 | 1 854, worst block 1 902 |
| `FixedSinOscQ15`, `int32` | 17 | 86–90 | 10 | 1 403–1 467 | 1 403, worst block 1 451 |
| `FixedSinOscQ15`, 32-bit frame counter | 14 | 76–80 | 10 | 1 240–1 304 | |
| `SinOscQ15`, live `freq` | 39 | 78–82 | 10 | 1 297–1 361 | |

The cheapest path holds the sample inside the clamp. Holding it at -32767
costs one cycle more and at +32767 three more; the upper bound adds the
infeasible path through both clamp arms, which costs four. The old count
summed every instruction in the loop's address range, the clamp's skipped
arms and the trailing branch included. That made its lower bound four cycles
per frame too high.

`--wait-states W` estimates code in flash: each 32-bit fetch and each
literal-pool load costs W more cycles. The low end lets a fetch hide behind
the execution of the word before it, and the high end does not. This is an
estimate, not a bound. For the `int32` object at W = 2 it gives 2 152–3 010
cycles per call, and the emulator measured 2 245.

`SinOscQ15`'s frame loop is 67 instructions, 9 of them `MULS`: three for the
phase, because the increment is a variable now, and six for the sine. At
N = 16 a call takes at most 1 361 cycles, about 6% of a 1 333-cycle sample
budget at 64 MHz and 48 kHz. Flash is 220 bytes for `tropical_kernel`, stack
36 bytes, and there is no static RAM. The `_argv` thunk is 16 bytes and
calls the kernel. `llvm-nm -u` lists nothing: no `__aeabi_*`, no
`__muldi3`, no soft-float helpers, no division. The host object matches the
JIT sample for sample over 65 536 frames at 2, 1 234.5, -440.25, 23 999 and
0.0001 Hz, and again from τ = 2⁴⁰. The wasm32 object, linked with `wasm-ld`
and run in Bun, matches the host object. `make validate` and CI run
`tools/embedded_object.py --check-only` on both LFOs.

**Formats and cost.** Q15 × Q15 fits one 1-cycle `MULS` (16 × 16 → 32).
Q31 × Q31 needs the high word of a 64-bit product. ARMv6-M has no `UMULL`,
so that is an `__aeabi_lmul` call or about four `MULS` plus adds and
carries. That is why the sine is Q1.15. The phase stays Q0.32, because its
products are only wanted mod 2³².

**What the spike changed in the core.**

- The spike's `mulLo32` op (`a · b` mod 2³², sign-extended onto the i64
  rail) is retired; the `int32` kind replaced it. LLVM does not narrow
  `(a·b << 32) >> 32` over i64 `mul` on thumbv6m (it calls `__muldi3`), so a
  32-bit multiply has to be spelled as one, and `int32`'s `mul` is.
- Integer values cross module slots raw. Slots were f64 in `EmitLlvm`, so a
  wired clock went i64 → f64 → i64, which is a soft-float call pair per
  frame on a no-FPU target and loses bits past 2⁵³ rail units on any
  target. An integer write now also lands in a lane of its own width,
  `%islots` (i64) for `int` and `%i32slots` (i32) for `int32`, and an
  integer read of a slot whose last emitted write was an integer loads that
  word. This is the rule EmitMsl already follows (its slots are typed
  locals, `long` and `int`). The `i32` output sink stores an `int32`
  program output from its lane directly.
- The port functions are `nounwind`, so the object does not reference an
  EH personality routine.

**The `int32` kind (2026-11-10).** Integer widths are meaning; number
formats for reals are a target's choice. `int32` is a scalar kind next to
the 64-bit `int`: two's complement on ℤ/2³² on every target (`i32` in LLVM,
`int` in MSL). Add, sub, mul, neg and left shift wrap modulo 2³²; right
shift is arithmetic; division and remainder truncate toward zero.

**Total integer semantics.** Every integer op, on `int` and on `int32`, is
defined on every input in the IR, so no backend is ever handed undefined
behaviour (LLVM's `sdiv`/`srem` are immediate UB for a zero divisor and for
`MIN / -1`, `shl`/`ashr` past the width and an out-of-range `fptosi` are
poison, and the wasm instructions trap). The definitions are the standard
ones:

- `div`, `mod` (and `floorDiv` on integers) by zero give 0. `MIN / -1` is
  `MIN` (the quotient 2³¹ or 2⁶³ wrapped) and `MIN % -1` is 0.
- A shift amount is taken modulo the width, 32 or 64, as wasm and Java do:
  `x << 33` on `int32` is `x << 1`, and `x << -1` is `x << 31`.
- A float → integer conversion (`toInt`, `toInt32` of a `float`, and every
  implicit one, such as an array index or a dynamic bank count) saturates:
  truncate toward zero, clamp to the word, NaN → 0. This is Rust's `as` and
  wasm's `trunc_sat`.

`ConstFold.foldOp` is the definition: it folds every case, and
`ConstFoldLaws` states each integer route as an unconditional equation
(`foldOp_div_int32`, `foldOp_lshift_int32`, `foldOp_div_int32_min_neg_one`,
`truncSatToInt32_isI32`, …) with no `sorry` and no axiom. The reference
semantics is the same function: `Semantics.Scalar.algebra` evaluates a
Plan's ops with `foldOp` at each instruction's result sort (the Plan
semantics passes every constant's kind, instruction's result sort and slot
store's kind to the algebra, so it tells `int32` from `int`). The backends
implement it without UB. `EmitLlvm` divides by a safe divisor (1 where the
divisor is 0 or the quotient overflows) and then selects 0 for a zero
divisor, so `opt` cannot delete the guard. It masks a runtime shift amount,
reduces a literal one at emit time (so a constant shift's IR is unchanged),
and converts with `llvm.fptosi.sat`. `EmitMsl` does the same with a guarded
ternary quotient, masked or reduced shift amounts, `uint`/`ulong` ring ops
reinterpreted with `as_type` (signed overflow is undefined in MSL as in
C++), and an explicit NaN-and-range clamp before `int(f)`/`long(f)`. On
AArch64 the hardware already behaves this way (`sdiv` gives 0 for a zero
divisor, `fcvtzs` saturates), which is why the JIT rarely showed the bug;
LLVM's optimizer and wasm did.

There is no implicit crossing. An op never takes an `int32` operand
together with an `int` or `float` one, and a float-only op (`sqrt`,
`floor`, …) takes no `int32` (`PlanOp.kindMix?`). A value reaches an
instance input, an output port or a slot by widening only
(`ScalarKind.widensTo`: bool → int → float and bool → int32), so an `int32`
value never becomes an `int` or `float`, and nothing but `bool` becomes an
`int32`, without the cast spelled. The compiler checks this where it binds
an instance input and where it writes an output port, with an error naming
the cast; a wire lifted into a `__wire_N` instance keeps an `int32`
source's kind on its input. A plan that crosses anyway is refused before
any backend sees it: `FlatPlan.kindError?` checks every instruction in
execution order (each `reg` read at its producer's sort, each op's result
sort and operand kinds, stores and array elements by widening only, no
`int32` read out of an array, constants within their kind, known tags), and
`PlanDecode`, `EmitLlvm.emitKernel` and `EmitMsl.emitKernel` refuse a plan
that fails it. `flatPlanWellFormed` includes it.

The casts are the crossings: `toInt32` keeps an `int`'s low 32 bits or
converts a `float` saturating, `toInt` sign-extends, `toFloat` converts.
`bool` promotes to `int32` as it does to `int`. A bare literal, or a
literal-only subtree (`neg(3)`, `2 * 3`), beside an `int32` operand on
either side is `int32`, and its literals must lie in [-2³¹, 2³¹) (a bit
pattern above 2³¹ − 1 is spelled as its negative). Builders author with
`toInt32E` and `litI32`. In the patch bay an `int32` output connects only
to an `int32` input (and `bool` widens into one); any other connection
spells the cast in the wire. Arrays hold floats, so an `int32` array
element is refused. On Metal an `int32` bank reduction stays on the serial
loop at any width (the cooperative lane partials are float or long arenas).

`int32` sits inside `tropical_plan_7` as an additive change
(`compatibility-matrix.md`). The gates, all in `tropicaltest` unless named:

- `int32-model`: `Int32Zoo`'s word, which has every int32 op at runtime
  including zero and -1 divisors under `INT32_MIN`, shifts past the word
  and a saturating conversion, rendered by the JIT equals the reference
  semantics bit for bit. It runs across the input's 2³¹ wrap, and again
  through a 32-bit slot from `FixedSinOscQ15`.
- `int-edge`, the edge-value differential: `IntEdge64`, `IntEdge32` and
  `IntEdgeConv` run every `int` and `int32` op on MIN, MAX, -1, 0 and 1,
  runtime divisors 0 and -1, shift amounts -1/0/1/31/32/33/63/64/65, NaN,
  ±∞ and out-of-range floats, and a seeded random sample. Results are read
  out as exact 16-bit pieces. JIT and Metal (offline) equal the reference
  frame for frame, and every op's segment is checked to be non-zero.
  `wasm_vs_jit` holds WASM to the JIT bit for bit on the same programs, and
  `metal_vs_jit` holds Metal to it again.
- `int32-refusals` covers the builder spellings, the patch bay, and the
  real ingest path (`Session.compilePatch`, what `diffcli compile` and
  `load` run). An `int32` output into a `float` input refuses, bare or as
  `add(lfo.out, 1)`. So do `int32` and `float` into an `int` clock, `float`
  into an `int32` input, and an `int32` element in a lifted array. The same
  wires with the cast compile.
- `int32-well-kinded`: `PlanDecode` refuses hand-edited plans: an operand
  relabeled, a `reg` read at a foreign sort, an int32 constant
  of 3 000 000 000, an int32 store into a float slot, an int32 array read,
  and `MulLo32`.
- `wasm_vs_jit` compares WASM and JIT bit for bit on `FixedSinOscQ15`;
  `metal_vs_jit` requires `FixedSinOscQ15` bit-identical on the GPU at
  τ = 0 and τ = 2⁴⁰.

The rewrite onto `int32` left `FixedSinOscQ15`'s output bit-identical: three
renders frozen before it (`tests/golden/embedded/`) held, and so did renders
at τ = 10¹² samples. The total semantics changed none of them, and the
frozen renders still hold. The thumbv6m object is unchanged by the total
semantics: 75 hot-loop instructions, 86–90 cycles per frame (90–93 on the
old count, see "Cycles"), no libcalls. The LFO's shifts are all by literals,
which the emitter reduces at emit time.

**Typed ports (2026-11-10).** A param, and an integer port of a program,
carries its own number format. The vocabulary is scalar-level
(`Tropical.PortFormat`): a kind (`float`, `int32`, `int`) and, for an
integer, `fracBits`, so the word `w` means `w · 2^-fracBits`. Labels are Qm.n
with the sign bit in m, so m + n is the word width: Q16.16, Q0.32, Q17.15,
Q48.16. Inside the kernel an integer is an integer. No IR pass, op or
semantics reads a format.

- *Where it is declared.* A patch declares a param in its `paramDecl`:
  `"type": "int32"` or `"int"`, `"frac_bits"`, and `value`, the real value
  (`float` is the default and stays as before). A builder declares a
  program port's format on the port (`InputDecl.fracBits`,
  `OutputDecl.fracBits`, `AInputDecl.fracBits`). It reaches the patch bay as
  `frac_bits` in the port's `type_obj`. Playground knobs stay `float`.
- *Where it is checked.* A kind is the compiler's: an `int32` param binds
  an `int32` input only, by the same crossing check as any `int32` value. A
  format is checked where a value crosses a port without arithmetic: a wire
  that is exactly a param, or exactly another instance's output, into an
  input of the same integer kind must have the same `frac_bits`. A wire that
  computes (`rshift(param(f), 8)`) yields a plain word and binds any input
  of its kind, because arithmetic on words is how a patch converts between
  formats. Declarations are refused when a fraction has no integer kind, when
  more fraction bits than the word has are asked for, when the type is
  `bool`, and when a default falls outside the format's range.
- *Literals.* A bare literal crossing into a formatted input is the real
  value, exactly as `paramDecl.value` is: `{"freq": 440}` into `SinOscQ15`'s
  Q16.16 `freq` is 440 Hz, the word `440 · 2^16`. The rule is the
  bare-crossing rule above: it applies to a wire that is exactly a literal
  (`Session.bareLiteralAtPort`, in the session compile and in
  `export_program`). The arithmetic is exact decimal, and a literal whose
  word is not an integer (`0.00001` at Q16.16) or lies outside the format
  (`40000` at Q16.16, `0.5` at Q0.32) is refused at compile, as `0.5` is at
  an `int32` Q32.0 input. A literal inside arithmetic stays a word, like
  every value the IR computes with: `{"op":"add","args":[65536,0]}` is 1 Hz.
  An input with no fraction bits takes its literal as written, since there
  the word is the value. A program's own default is authored as a word
  (`litI32 65536`, which reaches the patch bay as `toInt32(65536)`), and an
  exported program's default that came from a bare literal is spelled the
  same way, so `save` → `load` never reads a word as a real value.
- *The plan.* `slot_types` lists each typed param slot and each output
  channel's source that carries an `int32` program output, with the format.
  It is omitted when empty. A typed param's `slot_defaults` entry is its
  word, `Kernel.encodeWord(value)`: `value · 2^fracBits` truncated toward
  zero and saturated, NaN → 0. That is the kernel's own float → integer
  conversion, and the C++ and web hosts use the same one.
  `FlatPlan.portError?` holds the rest. A typed param is read only at its
  kind and never written by the kernel. One output buffer has one element
  type. An integer channel has exactly one source at gain 1, since the f64
  mix and gain have no integer counterpart to pick silently, and that
  source is `int32`: an `int` slot has no output element type. An `int32`
  value reaches an output only through a listed source. That closes the
  last implicit `int32` crossing, the f64 sink. A typed param's default is
  an integer word within its kind's range, never a value to round or wrap.
  The session compile, `PlanDecode` and both emitters refuse a plan that
  fails it, and the hosts refuse such a manifest at load.
- *The kernel.* `EmitLlvm` loads a typed param as `i32` or `i64` straight
  into its slot's integer lane, so it never passes through f64. An `i32`
  output channel stores the word from its 32-bit lane.
- *The manifest.* A param input is `f64`, or `i32`/`i64` with `frac_bits`,
  and its `default` is then the word. The output is `f64`, or `i32` with
  `frac_bits`. τ carries `frac_bits: 16`. The object block's `ports` list
  each buffer's element, Q label, bytes and alignment.
- *Disciplines.* Only `raw` applies to an integer param. `glide`, `anchor`
  and `velocity` re-anchor in float arithmetic on f64 companions
  (`host-param-dispatch.md`). Nothing attaches one to a typed port today
  (patch-bay params carry no discipline, and playground knobs are `float`),
  and `portError?` refuses one whose base or companion is a typed slot
  rather than dropping it. The C++ dispatcher and the web `ParamWriter`
  refuse it too, over every slot the discipline touches: the base, the
  companions the manifest lists, and those it implies by name. An integer `anchor` is the natural next form: with the
  increment as the port, its correction `Δinc · now` is exact mod 2³². It
  waits for a producer.
- *Range.* A value is refused where it would saturate: a `paramDecl.value`
  at ingest and a `set_param` value alike must have its word
  `value · 2^fracBits` inside the word (`PortFormat.rangeError?`; the error
  names the range, e.g. `int32 Q16.16, [-32768, 32768)`). The engine refuses
  before the write, so its mirror, and so `save`, never holds a value `load`
  would refuse. The C++ dispatcher (`set_param` by name, reached from the
  control socket without the engine) and the web `ParamWriter` refuse it
  too. `FlatRuntime::set_slot`, the raw by-index writer, has no error
  channel and still saturates.
- *Hosts.* Both hosts take the real value and write the word.
  `FlatRuntime` keeps a typed param's word in its 8-byte slot cell (an
  `int32` sign-extended, so an `i32` load of the cell reads it). `set_param`
  encodes into the cell, `get_slot` decodes, and the kernel reads the cell
  with no conversion on the audio path. An `i32` output renders into its own
  buffer and is widened to `word · 2^-fracBits` for the f64 output, the
  scope and the DAC. The web `WasmKernel` writes the word into the param's
  cell (`Int32Array` or `BigInt64Array`) and widens an `i32` output the same
  way. Both readers refuse other elements by name, and a default word that
  does not fit its element. The web reader also refuses an `i64` default past
  ±(2⁵³ − 1): `JSON.parse` hands it a double, which may not be the word
  written (`INT64_MAX` reads as 2⁶³ and would wrap to `INT64_MIN`), so it is
  refused rather than seeded wrong. The native reader takes the word
  exactly, so the two never seed different words. Existing f64 manifests
  read and play exactly as before.
- *Metal.* Metal binds params through the f32 slot vector and its f64 CPU
  pre-pass, and neither holds a 32- or 64-bit word exactly, so `EmitMsl`
  refuses a typed param; typed params run on the JIT and wasm. An `int32`
  output on Metal writes `word · 2^-fracBits` as f32, which is exact for a
  word below 2²⁴, and equals the JIT port widened by the runtime.
- *Export.* `export_program` keeps an exposed input's and an exported
  output's format (`InputDecl.fracBits`, `OutputDecl.fracBits`, and
  `frac_bits` in the exported `ports`), so an exported `SinOscQ15` still
  plays at its Q1.15 peak and still binds a Q16.16 param.
- *Gates.* `typed-ports-jit`, `typed-ports-refusals` and
  `typed-ports-engine` (export, `save` → `load`, `set_param` range) in
  `tropicaltest` (`--ports-only`), the typed-param cases in `wasm_vs_jit`,
  the `SinOscQ15` literal cases in `metal_vs_jit`, the typed-param case in
  `test_module_process` (range, companions, default words), and
  `tests/web/runtime_typed_ports.test.ts` (the web reader, `encodeWord`,
  refusals). `FixedSinOscQ15`'s three frozen renders and the stdlib goldens
  are unchanged.

**Gaps left open (smallest honest fix for each).**

1. *Params are f64 by contract.* Closed by typed ports: a knob on a no-FPU
   target is an `int32` word the kernel loads as one (`SinOscQ15`, 39 + 78–82
   + 10 cycles per call).
2. *The int rail was i64.* Closed by the `int32` kind: the rounding shifts
   and the clamp are now 32-bit. The frame counter `%s` is now 32-bit too,
   which saves 10 cycles per frame on the M0+ (86 → 76): the add-with-carry
   pair goes, and the τ pointer becomes a post-increment `ldm`. τ itself
   stays a 64-bit read by type.
3. *The output's Q format is a flag, not a type.* Closed by typed ports: an
   `int32` output's `fracBits` is the program's, carried into `Kernel.Port`
   and the manifest. `--out-elem` and `--out-format` are gone.
4. *f32* as a per-target float format. Closed on the LLVM path by
   `--float f32` (below). Still open on Metal's slots. Metal binds params and the
   stage-0 coefficient slots through its f32 slot vector, so an `int32` word
   computed once per call crosses as an f32 and keeps 24 bits. With
   `SinOscQ15`'s `freq` from a float param through `toInt32(mul(f, 65536))`,
   the increment arrives as `f32(inc)`: measured against the JIT at 44.1 kHz
   (8 192 frames from τ = 0), -1 increment LSB at 440 Hz, -15 at 5 kHz and
   -61 at 20 kHz (an increment near 2³¹, where an f32 step is 128, so up to
   64 LSB), and the output drifts by up to 26 Q1.15 LSB in that window,
   growing with τ. A literal frequency folds to a constant and is exact
   (`metal_vs_jit`), and `EmitMsl` refuses a typed param outright. The fix
   is an integer lane in Metal's slot vector (or an `i32` slot type per
   target), which is this item's question.

## The float format per target (2026-11-11)

The `float` kind is realized at one IEEE format per artifact: binary64 by
default, binary32 when the artifact is compiled with `--float f32`
(`Kernel.FloatFormat`). It is a choice of realization, not a kind. The plan
is the same either way; no IR pass, op or semantics reads the format. Only
`EmitLlvm`, the kernel type and the hosts change. The integer kinds and τ's
Q48.16 rail are the same on every target.

```
diffcli compile-object patch.json --out k.o --triple thumbv7em-none-eabihf \
    --cpu cortex-m4 --frames 16 --rate 48000 --float f32
```

`--float f64|f32` is taken by `compile-object`, `compile-wasm`,
`compile-kernel`, `render-bytes` and `emit-ir`. Both hosts read either.

**What binary32 means in the kernel.**

- Every float value, slot, param cell and output is an LLVM `float`; a
  scratch or array cell keeps a float's bits in its low word. The float
  intrinsics are the `.f32` ones.
- A plan literal is the binary32 nearest the decimal as written, ties to
  even, rounded once from the exact decimal
  (`EmitCommon.binary32BitsOfDecimal?`), never from its binary64 rounding.
  The two differ for a decimal just above a binary32 tie that binary64
  rounds onto the tie. A literal past the binary32 range is refused, not
  made an infinity. A param's default is that same binary32, written in the
  manifest as its exact decimal (`Kernel.binary32Decimal`), so a host seeds
  the bits with no second rounding.
- An op on literals is emitted as an op, and LLVM folds it at binary32. The
  wire `mul(0.1, 9)` is `0x3F666667` (`f32(0.1) · 9`, rounded), while the
  literal `0.9` is `0x3F666666`. The only binary64 arithmetic before emission
  is a program builder's own Lean `Float` (`litF`, a coefficient table built
  with `Array.map`). Its result becomes a literal and is rounded once more,
  to binary32: `litF (0.1 * 9)` is `0x3F666666`, where the same product in
  the kernel is `0x3F666667`.
- `floor`, `ceil` and `round` are expanded inline (`roundingF32`), because
  FPv4-SP (the Cortex-M4F's FPU) has no rounding instruction and the
  intrinsics become `floorf`/`ceilf`/`roundf` calls there. For |x| < 2²³
  the sequence truncates through `fptosi.sat` to `i32` and back (exact),
  steps by one where `floor`/`ceil`/`round` need it, and puts back x's
  sign bit, so `floor(-0) = -0` and `ceil(-½) = round(-0.3) = -0`. NaN, ±∞
  and |x| ≥ 2²³ pass through. `ctest`'s `current_rounding_f32` runs the
  emitter's own text (`tests/fixtures/rounding_f32.ll`, held to the emitter
  by `tropicaltest`'s `f32-rounding-fixture`) through the JIT on all 2³²
  binary32 inputs against libm: identical bits for every non-NaN result, a
  NaN for every NaN. That covers 16 777 216 halfway inputs, 16 777 214
  subnormals, both zeros and both infinities, and takes about 5 s on four
  threads. The same sequence runs on every binary32 target, so wasm32, the
  host and the M4F compute the same bits.
- `ldexp` and `floatExponent` work in the binary32 exponent field (bias
  127 at bit 23). Each is exact for normal results, n ∈ [-126, 127].
- No FMA. `EmitLlvm` writes no `contract` flag and no `llvm.fmuladd`, and
  every toolchain keeps `AllowFPOpFusion = Standard` with no-infs,
  no-NaNs and no-signed-zeros off, so an `fmul` feeding an `fadd` stays two
  rounded ops (AArch64 `FMADD`, the M4F's `VFMA` and wasm's relaxed madd are
  never selected). `wasm_vs_jit` holds WASM to the JIT bit for bit at
  binary32 on `FixedSinOsc`, a `SoftClip` chain, the op zoo, `SpectralSaw`
  and a written float param.

**Params at binary32 are written raw.** `glide`, `anchor` and `velocity`
re-anchor on companions that hold sample counts and phases (`#t0`,
`tau_base`, `#phase`), which binary32 does not hold exactly past 2²⁴, so
`Kernel.ofPlan` refuses a non-raw discipline in a binary32 artifact by name.
The C++ dispatcher and the web `ParamWriter` refuse one on an `f32` cell too.
A host write rounds to the nearest binary32. A value past the binary32
range is refused rather than written as an infinity, by both hosts.

**Hosts.** The manifest declares every float port `f32` and records the
format as `realization.float`, written only when it is `f32`, so an `f64`
manifest is what it always was. The native `FlatRuntime` keeps an `f32`
param's bits in the low word of its 8-byte cell, renders an `f32` output
into its own buffer and widens it exactly to f64 for the scope and the DAC.
The web `WasmKernel` does the same with `Float32Array` views. Both readers
refuse an unknown `realization.float` by name, a float port declared at a
format other than the artifact's, and an `f32` default that is not a
binary32.

**Targets.** `compile-object` asks LLVM's own subtarget description what the
FPU realizes (`target_float_info`: binary32, binary64, the float ABI), so no
core list is kept here:

- `--float f32` on a target with no binary32 FPU (the M0+) is refused
  before any code is emitted: every float op would be a soft-float call. An
  integer program needs no float format there.
- After codegen the object's undefined symbols are read back. Any symbol
  at all is a routine the target could not do inline, since a port function
  calls nothing of its own. The compile is refused with each symbol and the
  op that asked for it, and no object is written. `--allow-libcalls` writes
  it anyway and lists the symbols in `object.libcalls`.
- Triples are normalized as LLVM's tools do. `thumbv7em-none-eabihf` is the
  hard-float procedure call standard, and the object carries
  `Tag_ABI_VFP_args = AAPCS VFP`. Unnormalized, LLVM read `eabihf` as an
  unknown OS and marked the object base AAPCS, which a hard-float firmware
  link refuses. Normalizing left both M0+ objects' code byte-identical; it
  adds `Tag_ABI_optimization_goals`.

**The M4F fixture.** `tests/fixtures/embedded_f32_softclip.json` is the
knob LFO (`SinOscQ15`, `int32` Q16.16 `freq`) widened with
`toFloat(out) · 2⁻¹⁵` into `SoftClip` with a float `drive` param. For
`thumbv7em-none-eabihf`/`cortex-m4` at binary32 and N = 16 it is 408 bytes
of `tropical_kernel` (the `_argv` thunk is 14 more), 40 bytes of stack, no
static RAM and no libcalls; the float work is `vcvt.f32.s32`, `vmul`,
`vadd`, `vdiv` and `vcmp`. `make validate` and CI run
`tools/embedded_object.py --core m4f --check-only` on it. The script reports
sizes, symbols, stack and build attributes for the M4F. It counts no
cycles, because its table is the M0+'s: for any other core the report says
so (`cycles_refused`), and `--wait-states` exits non-zero. No ARMv7E-M
emulator ran the object; its bits are argued, not measured: the same LLVM
module, FMA off, as the host and wasm32 objects, which agree with the JIT.

**What still needs a libcall on the M4F.** Sweeping the stdlib programs at
binary32 (each instantiated alone, inputs at their defaults):

| routine | op | programs |
|---|---|---|
| `__aeabi_l2f`, `__aeabi_ul2f` | `int` (64-bit) → binary32: a clock or phase computed on the `int` rail and then used as a float | `FlangeSin`, `MorphOsc`, `PluckedMorphOsc`, `ModalVoice`, `ReverseReverb`, `ReversibleComb`, `ThroughZeroFlanger`, and `FixedSinOsc` |
| `__aeabi_f2lz` | binary32 → `int` (64-bit): `toInt` of a float (the modal and comb families' indices and counts) | `ModalVoice`, `MorphOsc`, `PluckedMorphOsc`, `ReverseReverb`, `ReversibleComb`, `ThroughZeroFlanger` |
| `__aeabi_memclr4` (`memclr8` at f64, on the M0+ as well) | the emitter's slot-zeroing loop, which LLVM's loop-idiom pass turns into a `memset` on ARM when a zeroed slot is read (an unwired input at its default) | `FixedSin`, `Sin`, `SoftClip`, `Tanh`, `VCA` alone |

`FixedSinOscQ15` and `SinOscQ15` need none. The first two rows are the
programs' own arithmetic: an `int` crossing a float. A program that keeps
its phase in `int32` and casts with `toFloat` (as the fixture does) avoids
them. The third row is not about binary32 and predates it. A freestanding
`TargetLibraryInfo` in `compile_ir_to_object` would stop the idiom pass
from forming the call, and is left for a separate change.

**Clocks across nodes.** The kernel reads τ from its caller. Exact ratios
(phase-locked partials, the ℤ/2³² phase identity) hold only between
kernels evaluated on the same τ. Whether separate nodes share τ by
distributed sync or drift is the host's decision and is not settled here.
