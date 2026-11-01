# The kernel artifact — the port between the compiler and its hosts

Status: **Proposed** (hexagonal sprint, S10); §2–§4 are superseded by
`typed-kernel.md` (2026-10-28), which replaces this port with one typed pure
kernel. This document names the contract that exists in code today; it does
not change it. Where it
disagrees with the code, one of the two is wrong and the owner decides which.
It does not amend `cf-only.md`, `host-param-dispatch.md`, `architecture.md`
or `compatibility-matrix.md`; it points at them where they already govern.

## Why a single document

The compiler's product is a *kernel artifact*: kernel code plus a manifest.
The realtime audio host, the scopes, the web player, Reversible and future
consumers are adapters that load artifacts. Until now the shape of an
artifact was spread across comments in five places: `EmitLlvm.lean`,
`EmitMsl.lean` and `MetalKernel.mm` (kernel signatures), `OrcJitEngine.hpp`
(`NumericKernelFn`), `NumericProgramParser.hpp` and `web/runtime/manifest.ts`
(what a host reads from the manifest), and `tropical_c.h` (how a host loads).
A host author had to read all of them. This is the one place.

The producer is `Tropical.Artifact` (`lean/Tropical/Artifact.lean`); the
reference consumer is `FlatRuntime` through `Tropical.StagedLoad`. The
offline compiler `tropicalc` writes an artifact to disk with no runtime
linked, and `tools/check_artifact_parity.py` holds it byte-identical to what
the engine publishes.

## 1. The artifact

```
KernelArtifact
  ir            : String   LLVM IR text — the main kernel (always present)
  coefficientIr : String   LLVM IR text — the stage-0 coefficient kernel, or ""
  msl           : String   Metal Shading Language source, or "" (JIT-only)
  manifest      : String   tropical_plan_6 wire JSON of the *audio* plan
```

Empty strings mean *absent*. On disk (`tropicalc --out=<dir>`) these are
`audio.ll`, `coeff.ll`, `audio.metal` and `manifest.json`, the
same names `TROPICAL_STAGE0_DUMP` writes.

The manifest is the audio plan after the stage-0 split: the same schema a
plan always had, with its slot table extended by the `coef:<n>` slots the
main and coefficient kernels share. One manifest describes all the kernels
in the artifact.

An artifact is immutable once emitted. A host never edits it; a different
patch, mode or realization is a different artifact.

## 2. The kernel function

Every LLVM kernel in an artifact (main, coefficient) defines exactly
one function with this signature. In the IR text it is named
`tropical_kernel`; a host may rename it when it links (the JIT uses
`tropical_k_<md5(ir)>` so distinct kernels never collide).

```
define void @tropical_kernel(
  ptr %inputs,                 ; i64[]   external inputs (unused by production kernels; may be a zeroed region)
  ptr %registers,              ; i64[]   register_count cells (reserved; current kernels never store here)
  ptr %arrays,                 ; ptr[]   one pointer per array slot, into the array backing store
  ptr %array_sizes,            ; i64[]   element count of each array slot
  ptr %temps,                  ; i64[]   register_count scratch cells (current kernels keep temps in SSA)
  double %sampleRate,          ; the rate the plan was compiled for
  i64 %start_sample_index,     ; the sample index of this buffer's first sample
  ptr %param_ptrs,             ; reserved (always passed)
  ptr %output_buffer,          ; f64[]   buffer_length × C, frame-major: output[s*C + target]
  i64 %buffer_length,          ; samples to compute in this call
  ptr noalias nocapture %slots ; f64[]   slot_count module slots
)
```

The native type is `tropical_jit::NumericKernelFn` (`OrcJitEngine.hpp`);
the wasm32 export is the same eleven arguments with the two `i64`s as
`bigint`. The kernel loops `s` from 0 to `buffer_length − 1` and computes
sample `%current_idx = %start_sample_index + s`. It reads and writes only
the regions it is handed. It performs no allocation, no I/O and no calls
other than the LLVM math intrinsics listed in `EmitLlvm.intrinsicDecls`
(`sqrt`, `floor`, `ceil`, `round`, `fabs` on f64).

**Values and the clock rail.** Values are f64; integer and clock arithmetic
is i64 (the "rails"). The master clock is Q48.16 signed on i64: a wire's
`clock()` is τ itself (`fixed-carrier.md` is the normative Q-format
ledger). A kernel is a closed form `f(τ, params)`
(`cf-only.md`): it carries no state from one call to the next, so a host may
call it for any `start_sample_index`, in any order, and get the same
samples. That is what makes hot-swap click-free and scrubbing exact.

**Slots.** `slots[i]` is a module slot: a param input (`param:<name>`), an
inter-instance wire, or a coefficient (`coef:<n>`). The kernel reads
host-written slots and writes the slots it produces. A host seeds every slot
from `slot_defaults` and thereafter writes only param slots, through the
discipline the manifest declares (§5). Slot *names* are the identity that
survives a hot-swap: controls project from one artifact's slot layout to
another's by exact name.

**Output.** `output_buffer` is frame-major with `C = output_channel_count`
channels; each sink is one continuously produced channel
(`stereo-output-contract.md`). The device-boundary bound (`DEVICE_OUTPUT_BOUND`
/ `kDeviceOutputBound`) is a host policy applied at the DAC, not part of the
kernel.

## 3. The coefficient kernel

`coefficientIr`, when present, has the same signature and is called with
`buffer_length = 1`. It holds the plan's τ-independent instructions: the
host runs it once at load, before publishing, and again after every
control-plane slot write. Its outputs land in the `coef:<n>` slots the main
kernel reads, and in the *coefficient columns* — the array slots the
manifest lists as `coeff_array_slots` — which the main kernel's in-loop
`Index` reads.

A host must make one control transaction's results visible to the audio
kernel atomically: the scalar slots and every coefficient column it
produced are one generation. `FlatRuntime` does this with three
generation-buffered copies and one release-published generation word,
captured once per audio buffer. The guarantee a host owes is the
*coherence*, not the mechanism.

## 4. The Metal kernel

`msl`, when present, is the main kernel as a Metal compute kernel, one
thread per sample. Its binding ABI is byte-frozen and must match
`engine/metal/MetalKernel.mm`:

```
kernel void tropical_kernel(
    device float*                  output_buffer [[buffer(0)]],
    constant float*                slots         [[buffer(1)]],
    constant TropicalKernelConsts& k             [[buffer(2)]],
    constant float*                coeff_columns [[buffer(3)]],   // only when coeff_array_slots is non-empty
    uint s [[thread_position_in_grid]])

struct TropicalKernelConsts { ulong start_sample_index; float sample_rate; uint buffer_length; }  // 8+4+4, no padding
```

Values are f32 and clocks are the exact i64 rail; the f32 surface is
tolerance-gated against the f64 JIT (`tests/web/metal_vs_jit.test.ts`), so
a Metal host is *conformant within the documented tolerance*, not
bit-exact. Slots and coefficient columns cross to the GPU f64→f32 as
host-written data; the kernel writes only `output_buffer`. Compile with
`MTLMathModeSafe`.

## 5. The manifest

The manifest is the `tropical_plan_6` wire form (`Plan.lean`). It carries
the whole instruction graph, but **a host reads only the layout and control
fields below**. Everything else is compiler metadata, present so the plan
round-trips and so `tropicaltest` can gate it; a host that depends on it is
coupling to the compiler's internals.

| field | host use |
|---|---|
| `schema` | must be `"tropical_plan_6"`; anything else is refused (`compatibility-matrix.md`) |
| `config.sampleRate` | the `%sampleRate` argument |
| `slot_count`, `slot_names`, `slot_defaults` | size, identity and seed of `%slots` |
| `register_count` | size of `%registers` and `%temps` |
| `array_slot_count`, `array_slot_names`, `array_slot_sizes` | the array backing store and `%arrays` / `%array_sizes` |
| `output_channel_count` | `C`; omitted means 1 |
| `param_disciplines` | how to write each param slot (`host-param-dispatch.md`); omitted means every write is raw |
| `coeff_array_slots` | the coefficient columns (§3); omitted means none |
| `metal_execution` | `sample_threadgroups` dispatch shape and `threadgroup_scratch_bytes`, Metal only |
| `compilation_mode` | `fused` (the shipped realization) or `microkernel`; a host that cannot run the mode refuses the artifact |

The web player's `KernelManifest` (`web/runtime/manifest.ts`) is the
minimal statement of this table: `sampleRate`, `registerCount`,
`arraySlotSizes`, `slotCount`, `slotDefaults`, `outputChannelCount`. A host
that reads more than the table above should say why here.

`param_disciplines` is the one place the compiler *declares* and the host
*executes*: the compiler knows which slots are glide/anchor/velocity
companions, the host performs the re-anchoring arithmetic with its own
`now` and `SR`. `host-param-dispatch.md` is normative for the arithmetic.
(Sprint decision D1 is whether the Lean engine's copy of that arithmetic
stays the reference or becomes a conformance model; this document is
neutral on it.)

## 6. Loading

A host loads an artifact as one atomic publication: it compiles the
kernels, seeds the slots, runs the coefficient kernel once, and flips. The
previous kernel keeps playing until the flip, and a failed load leaves it
playing (`tropical_runtime_load_ir_time_staged*` return `false` and set
`tropical_last_error`). Each publication has a *generation*: a
`(programVersion, controlVersion)` pair constructed by the load itself,
which the compile handshake returns to the client so it can know exactly
which artifact and which control state it is hearing.

A host may publish **two artifacts at once** — the realtime one and a
JIT-only *observation* artifact for scopes and `render_window` — and the
controls project between their slot layouts by exact slot name.

Across a hot-swap the host carries only the sample index. Slot values come
from the new manifest's defaults and from the control layer re-applying its
current parameter values by name; there is no register or array transfer,
because there is nothing stateful to transfer (`architecture.md`,
"Fixed-topology compilation").

## 7. What is not part of the port

- The Lean types (`FlatPlan`, `KernelArtifact`, `SessionSt`), the session
  engine and MCP: these are the compiler's and its authoring adapter's.
- The instruction stream in the manifest: metadata, not a contract.
- The JIT's symbol naming, its IR cache (`~/.cache/tropical/kernels`), the
  three-generation slot buffering, the DAC output bound: host mechanisms
  behind the guarantees above.
- The wasm emitter (`tropical_wasm_compile_ir`, `libtropical_wasm`): a
  toolchain concern, in its own library that only `diffcli` links. The
  runtime carries no lld (`tools/check_runtime_deps.py`).

## Open questions for the owner

1. Should the reserved arguments (`%inputs`, `%registers`, `%param_ptrs`)
   stay in the signature, or be retired at the next deliberate ABI break?
   Retiring them changes `NumericKernelFn`, the wasm export and every host.
2. Does `compilation_mode = microkernel` remain a published realization, or
   is `fused` the only artifact shape a host must accept?
3. Which host is the conformance reference for a new host to be gated
   against (D2): the headless JIT alone, or also a pure-Lean oracle?
