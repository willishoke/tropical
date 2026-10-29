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
  Q32.32 (`fixed-carrier.md` is normative), so a fractional coordinate is
  representable and an integer-sample host writes `index << 32`.
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

τ is passed, not derived. The audio host fills `τ[s] = (origin + s) << 32`,
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
  "inputs":  [ { "name": "tau",          "elem": "i64q32", "shape": [512],  "role": "signal" },
               { "name": "param:freq",   "elem": "f64",    "shape": [],     "role": "param",
                 "default": 220.0, "discipline": "anchor", "companion": "param:freq.tau_base" },
               { "name": "in:mic",       "elem": "f64",    "shape": [512],  "role": "signal" },
               { "name": "table:wave",   "elem": "f64",    "shape": [2048], "role": "table",
                 "default": [ … ] } ],
  "outputs": [ { "name": "out",          "elem": "f64",    "shape": [512, 2] } ],
  "realization": { "mode": "fused", "register_count": 0 },
  "plan": { … tropical_plan_6 metadata, for the gates; a host that reads it is coupling to the compiler … }
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
  gone from the emitted plan.)
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
