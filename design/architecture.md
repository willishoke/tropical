Status: Current

# Tropical architecture

This document is the authority for the architecture on `main`. Historical
plans and migration records are useful context, but they do not override the
types and paths named here.

## Current architecture in five minutes

Tropical compiles a closed-form signal graph to one function
`f(τ, params)`. The production language has no register, delay, update, or
feedback constructor. A kernel can therefore evaluate any coordinate directly:
the previous sample is not an input.

There are three current construction paths:

```text
Lean arrow builders → assemble → Ir.Strata per-program lowering → registered types
                                                                        │
MCP mutations / program_2 JSON → typed SessionSt → synthetic resolved root
                                                                        │
                                                          assert session acyclic
                                                                        │
                                                                        ▼
                                                   partition + classify/hoist stage 0
                                                                        │
                                                              tropical_plan_5
                                                         ┌──────────────┼───────────┐
                                                         ▼              ▼           ▼
                                                    LLVM → JIT    LLVM → wasm32  MSL → Metal
```

The first path is authoring-time construction. The other two edit a session
whose instances refer to types that were already registered. The
`tropical_program_2` format is a patch bay, not a program-definition language:
an inline `programDecl` is rejected at ingest.

Compilation ends with a typed `Tropical.Plan.FlatPlan`. Lean emits LLVM IR for
the native and WebAssembly routes and MSL for Metal. The C++ runtime consumes
the emitted kernel plus the plan metadata; it does not interpret the plan
instruction stream to generate code.

At run time, parameter changes write slots. They do not lower or compile a new
graph. Structural selector changes and topology changes do. Publication is
clickless because a new closed-form kernel resumes at the current coordinate,
not because arbitrary kernel state is copied.

## Live parameters select complete universes

Every closed-form graph with live numeric controls follows one temporal law.
For the complete deferred graph `G`, the vector `p` of its live controls, and
the observation coordinate `t`:

```text
renderLive(G, p, t) = renderStatic(G, freeze(p, t), t)
```

`freeze(p, t)` samples every control under the terminal clock context through
that control's authored clock path, then uses the resulting single parameter
vector throughout the static rendering of the whole graph. It does not freeze
each node after an
upstream node has already rendered its history. Consequently, changing a live
control rewrites observations of earlier coordinates to the counterfactual
history selected by the new value. Restoring the same frozen vector restores
the same output at the same effective observation coordinate. Hold, seek, and
reverse are different routes to such coordinates; they do not add hidden
history to the result.

Modal effects preserve this law by remaining deferred until an actual signal
consumer or observation tap. A branch address changes only that branch's modal
response coordinate; it does not implicitly retime an independent control
source. Room-local sway follows the branch response clock, while its numeric
depth and rate come from the frozen terminal universe. An ordinary downstream
graph warp pushes a new terminal clock context before branch addresses or
controls are evaluated, so patch algebra retimes both normally. In particular,
an addressed response under context `κ` is `κ(address(κ, t))`, not an
after-the-fact transformation of an already resolved coordinate.

Rooms may use fixed structural topology, immutable tables, and bounded
capacities, but they do not contain parameter-history state or consult prior
control values. Generic closed-form glide sources are ordinary members of
`p`; a room consumes their value in the selected universe, not their previous
targets. In particular, the public room envelope for source age `|t-u|` is
`exp(-6.91 * |t-u| / rt60(t))`: RT60 names the nominal ratio-one decay of the
universe frozen at `t`, using the incumbent approximation to `ln(1000)`. It is
not the accumulated physical loss
`exp(-integral(u..t, 6.91 / rt60(τ), dτ))`.

A physically accumulated response would therefore be a separately named
stateful module in a sister runtime, not an implementation of this contract.
The generic lifting and canonical whole-graph commutation obligation are
formalized in
[`Semantics.ModalUniverse`](../lean/Tropical/Semantics/ModalUniverse.lean).
Production modal lowering must refine that canonical stage model; the theorem
is not, by itself, permission to optimize or collapse the production forest.

## Four different times

Keeping these phases separate prevents most architectural misunderstandings.

| Phase | What happens | Current representation |
|---|---|---|
| Authoring time | Lean combinators build an expression tree and declarations; `assemble` interns them into the graph. | [`EmitArrow.Sig`](../lean/Tropical/EmitArrow/Sig.lean), then [`Ir.ENode`/`ExprArena`](../lean/Tropical/Ir/Nodes.lean) |
| Compile time | Program registration performs direct `Ir.Strata` lowering. A structural session compile links those resolved types into a checked synthetic root, then partitions, classifies binding time, and emits backends. | [`Engine.Compile`](../lean/Tropical/Engine/Compile.lean), [`Ir.Strata`](../lean/Tropical/Ir/Strata.lean), [`Compile`](../lean/Tropical/Compile.lean), [`Ir.Stage0`](../lean/Tropical/Ir/Stage0.lean) |
| Control time | A host dispatches a parameter write according to the plan-carried discipline and writes one or more slots. | [`Plan.ParamDiscipline`](../lean/Tropical/Plan.lean), [`Engine.Audio`](../lean/Tropical/Engine/Audio.lean), [`tropical_socket`](../engine/c_api/tropical_socket.cpp) |
| Sample time | The selected backend evaluates the closed-form kernel at a coordinate and current slot values. | [`EmitLlvm`](../lean/Tropical/Ir/EmitLlvm.lean), [`EmitMsl`](../lean/Tropical/Ir/EmitMsl.lean), [`FlatRuntime`](../engine/runtime/FlatRuntime.hpp), [`WasmKernel`](../web/runtime/kernel.ts) |

Stage 0 is control-time computation, not persistent sample state. Its optional
coefficient kernel runs at load and after relevant slot writes, publishing
coefficient slots or whole coefficient-column generations for the sample-time
kernel.

## Front doors

### Lean arrow builders

[`Tropical.EmitArrow.Sig`](../lean/Tropical/EmitArrow/Sig.lean) is a fourteen
constructor authoring tree. `assemble` lowers it directly into the resolved
arena. The standard library is a chain of these builders in
[`Tropical.Stdlib`](../lean/Tropical/Stdlib.lean); there is no literate surface
parser or name-resolution elaborator between a builder and the trunk IR.

`Sig` is the fourteen-constructor authoring subset of the `ENode` executable
trunk. Other current producers can use additional trunk nodes such as `bool`
and `arraySet`; there is still one expression vocabulary rather than a rich
AST/core twin. `bankSum` is deliberate backend-visible data: it describes a
bounded ordered reduction, not a combinator waiting for a later “sum
lowering” pass.

### MCP session mutations

The Lean frontend owns the live [`SessionState`](../lean/Tropical/Session.lean).
MCP tools add registered instances and store wiring as
[`Tropical.WireExpr`](../lean/Tropical/WireExpr.lean). The decoder is the
language boundary: state operations and retired functional/combinator
spellings do not decode.

[`Engine.Compile.sessionToResolvedRoot`](../lean/Tropical/Engine/Compile.lean)
builds one synthetic resolved root directly from instance snapshots and typed
wires. It does not serialize a parsed program and does not invoke an
elaborator.

The playground is a product-specific authoring surface over the same waist.
[`Playground.compilePlanPure`](../lean/Tropical/Playground/Compile.lean)
decodes its graph, builds an arrow term, assembles a resolved program, and
continues through the same session-plan compiler.

### `tropical_program_2` patch JSON

[`Parse.Raise.normalizeProgramFile`](../lean/Tropical/Parse/Raise.lean)
validates and normalizes the outer document. Then
[`ProgramIO.Ingest`](../lean/Tropical/Engine/ProgramIO/Ingest.lean) loads
instances, params, wiring, and device outputs over registered program types.

This format cannot define a new DSP type. A `programDecl` fails with the
retirement message, and wire expressions pass through the closed
`WireExpr` decoder. `export_program` is the one live way to crystallize a
selected session subgraph into a registered type; it constructs resolved IR
directly in [`ProgramIO.Export`](../lean/Tropical/Engine/ProgramIO/Export.lean).

## The trunk and direct lowering

The trunk is [`Tropical.Ir.ENode`](../lean/Tropical/Ir/Nodes.lean) interned in
an `ExprArena`. Positional indices represent resolved inputs, params,
instances, and outputs. There is no rich AST and core-expression twin.

[`Tropical.Ir.Strata`](../lean/Tropical/Ir/Strata.lean) is a direct lowering,
not a progressive rich-to-core pipeline:

1. `assertAcyclic` checks the graph contract.
2. `inlineInstances` optionally removes nested instance boundaries. The
   per-program inline route enables it; the fractal session route may preserve
   boundaries for partitioning.
3. `identityElim` applies the categorical identity law.
4. `EArena.toResolved` copies only evaluator-reachable nodes into a fresh
   arena in the same vocabulary and checks the resolved shape.

The former specialization, sum-lowering, and array-lowering passes, and the
former separate core arena, are historical. They are not current compiler
stages.

## Session plan construction and staging

[`Tropical.Compile.compileSessionStaged`](../lean/Tropical/Compile.lean)
allocates typed module slots, partitions the synthetic root into nested
`InstanceFunction` blocks, adds runtime sources and device sinks, and returns a
`Tropical.Plan.FlatPlan` plus typed stage blocks.

[`Tropical.Ir.Stage0.hoistTyped`](../lean/Tropical/Ir/Stage0.lean) uses those
blocks to split τ-independent, slot-derived work from the audio kernel. If
nothing is hoisted there is only the audio plan. Otherwise the load carries:

- an audio LLVM kernel;
- a coefficient LLVM kernel, run once at load and after control writes;
- on Metal, an MSL audio kernel;
- one `tropical_plan_5` metadata manifest for the loaded artifacts.

Scalar coefficient slots tolerate the documented one-buffer race. Bank
coefficient columns use three generations, so the audio block observes one
whole published generation. For live Metal, the control thread narrows the
selected slot/column generation into an immutable render-epoch request; the
audio callback does not pack coefficient data.

## Invariant index

| Invariant | Created or checked at | Represented by | Downstream consumer |
|---|---|---|---|
| Acyclic graph | session compilation and direct program/export construction in [`Ir.Cycles`](../lean/Tropical/Ir/Cycles.lean), [`Tropical.Lowering`](../lean/Tropical/Lowering.lean), and [`ProgramIO.Export`](../lean/Tropical/Engine/ProgramIO/Export.lean) | topological order plus explicit cycle refusal | direct lowering, partition, and all emitters |
| Closed-form kernel | the authoring and wire vocabularies | absence of state/register/delay/update constructors in [`EmitArrow.Sig`](../lean/Tropical/EmitArrow/Sig.lean), [`Ir.ENode`](../lean/Tropical/Ir/Nodes.lean), and [`WireExpr`](../lean/Tropical/WireExpr.lean) | LLVM, wasm32, and Metal execution |
| Whole-graph parameter universe | canonical modal proof; production refinement is tracked as an open trust obligation | one terminal-context control freeze, with branch address isolated to response-coordinate resolution | random-access JIT/wasm/Metal rendering and observation taps |
| Typed wire | [`WireExpr` JSON decoder](../lean/Tropical/WireExpr.lean) and port validation in [`Engine.Wire`](../lean/Tropical/Engine/Wire.lean) | `Tropical.WireExpr` plus declared port positions/types | `sessionToResolvedRoot` and export construction |
| Source-expression meaning | direct [`Sig` denotation](../lean/Tropical/Semantics/Sig.lean), total [`ExprArena` denotation](../lean/Tropical/Semantics/Expr.lean), and [`lowerSigTree_preserves`](../lean/Tropical/Semantics/LowerSig.lean) | refusal-aware carrier-parametric equality across structural lowering | whole-program lowering and later refinement proofs |
| Bank order | authoring and emitter laws in [`EmitArrow.BankOrder`](../lean/Tropical/EmitArrow/BankOrder.lean) and [`Ir.EmitBankLaws`](../lean/Tropical/Ir/EmitBankLaws.lean) | `ENode.bankSum`, `ReduceBegin`/`ReduceEnd`, ordered tables | LLVM/JIT, wasm32, and MSL/Metal |
| Stage separation | stage attributes and [`Stage0.hoistTyped`](../lean/Tropical/Ir/Stage0.lean) | typed per-instruction stage blocks plus coefficient slots/columns | staged native and Metal loads |
| Host write discipline | playground decode/report and plan construction in [`Playground`](../lean/Tropical/Playground/) | `FlatPlan.paramDisciplines` and named companion slots | Lean host, C++ socket host, and other manifest hosts |

The formal and empirical status of these claims is tracked in the
[trusted-boundary ledger](trust-boundary.md). Backend agreement is evidence for
a refinement obligation; it is not by itself a proof of source semantics.

## Live-edit contract

| Edit | Compiler work | Publication behavior |
|---|---|---|
| Parameter value | No relower. The host applies `raw`, `glide`, `anchor`, or `velocity` discipline and writes slots; stage-0 work reruns when present. | JIT publishes the captured generation at a callback boundary. Metal prepares a new render epoch off the callback and activates it at its reported exact `E`. |
| Structural selector | Decode/lower/partition/emit again. Examples include a voice kind or a baked realization choice. | Publish a fresh kernel at the current coordinate. |
| Topology | Rebuild the synthetic root, lower, partition, emit, and load. | Publish a fresh kernel at the current coordinate. |
| Kernel publication | No semantic state migration. `FlatRuntime` carries only `sample_index`; fresh storage comes from the new manifest. Current parameter values are supplied by the control/session layer. | Atomic inactive-state build and flip; optional fade remains a device policy. |

There is no universal “every edit is sub-millisecond” contract. Parameter
writes and structural recompiles are different operations and must be measured
separately. Current dated measurements, cache conditions, machine details, and
percentiles live in the [performance baseline](../benchmarks/current_baseline/findings.md).

## Plan and retired-schema boundary

[`Tropical.Plan`](../lean/Tropical/Plan.lean) is the typed
`tropical_plan_5` producer contract. It carries:

- typed instruction and destination forms;
- nested instance functions;
- runtime sources and device sinks;
- module slots and defaults;
- stage-0 coefficient-column metadata;
- host parameter disciplines.

Lean's type has no legacy state initialization, state-register types/names, or
register update targets. The wire field `register_count` sizes the SSA temp
pool; `NOperand.reg` is a temp read, not persistent state.

The native C APIs and Lean `FlatPlan.ofWire` accept `tropical_plan_5` only.
Older or unknown schema tags and retired state/output carriers fail clearly at
the serialized-plan boundary; there is no compatibility lift. The C++ runtime
allocates no persistent state-register backing store. See the
[compatibility matrix](compatibility-matrix.md).

## Execution targets

| Target | Numeric regime | Product role | Correctness evidence |
|---|---|---|---|
| LLVM → ORC JIT | f64 value path plus i64 rails | Native reference, scopes/random-access windows, and portable native audio | frozen audio and plan goldens; native realization checks; trust-ledger obligations |
| LLVM → wasm32 | The same LLVM/f64 and i64 semantics, hosted in WebAssembly | Precompiled browser player | [`wasm_vs_jit`](../tests/web/wasm_vs_jit.test.ts) and precompiled-plan checks |
| MSL → Metal | f32 value path plus exact i64 clock rail; stage-0 coefficients originate in CPU f64 and narrow at upload | Heavy live modal audio on supported Apple builds; JIT remains dual-loaded for scopes/reference | [`metal_vs_jit`](../tests/web/metal_vs_jit.test.ts), MSL goldens, and Metal runtime tests |

Metal is selected with `TROPICAL_BACKEND=metal` on a Metal-enabled build.
The runtime dual-loads the JIT artifact: `process()` uses Metal, while
`render_window` retains the f64 reference. Hardware-specific latency,
reliability, and any completed or blocked soak rows are reported without
generalization in the
[Metal findings](../benchmarks/metal_live/findings.md).

### Live Metal epoch handoff

Live Metal submission is owned by one dedicated render worker, never by the
audio callback. `MetalKernel::render_tile` is a blocking worker primitive.
The worker renders into a fixed two-bank handoff with four preallocated tiles
per bank. Tile ownership moves only through:

```text
worker:   Free → Rendering → Ready
callback: Ready → Reading   → Free
```

The device callback quantum `Bdev` and GPU render quantum `Rgpu` are
independent. `Rgpu` must be a positive multiple of `Bdev`; the default is at
least 512 frames, rounded to that multiple. The callback consumes one exact
`Bdev` slice from a prepared tile. It does not submit or wait for Metal, take a
lock, allocate, reclaim worker state, inspect movable `KernelState` storage, or
pack slots/columns.

Each tile is tagged with an epoch id, monotonic device-frame start, independent
Tropical source-sample start, and frame count. A control transaction first
reserves a device activation boundary, computes its source-coordinate
`effective_sample_index`, materializes every companion slot at that exact
coordinate, renders a complete candidate window, and release-publishes one
activation descriptor. The callback makes one bounded descriptor read and
switches only at that boundary. Old audio is emitted strictly before `E`; the
new epoch begins at `E`. Clock jumps change the source coordinate without
rewinding the device frame; hot-swaps carry the source coordinate.

A missed candidate target is retargeted off the callback and all companion
math is recomputed at the replacement `E`. A candidate render failure refuses
activation and leaves the active epoch intact. Once active, a missing exact
tile, tag mismatch, or terminal Metal command failure produces whole-callback
silence and latches that epoch fault until a fresh explicit epoch activates.
The runtime never replays a tile, stretches a sample, waits, or falls back to
JIT on the live callback.

`TROPICAL_METAL_RENDER_TILE_FRAMES` is the qualification/test render-quantum
override. Runtime diagnostics expose device/render quanta, worker capacity,
published/acknowledged activation ids, dispatch failures, render starvation,
tag mismatches, activation retargets/failures, callback-thread provenance
violations, stage timestamps, activation-latency statistics, and worker
CPU/wall time. These measurements separate callback deadline, activation
latency, GPU tile duration, and render-ahead reserve; none may be substituted
for another.

## What the test layers establish

- Frozen audio goldens are the behavioral regression anchor.
- Stdlib plan/port goldens pin the builder-to-plan contract.
- Native realization checks compare supported JIT realization choices.
- wasm-vs-JIT checks backend agreement from the same plan.
- Metal-vs-JIT checks the documented f32 tolerance/SNR boundary and exact
  integer-clock behavior.
- Cycle and patch-bay refusal gates prove current front doors reject feedback
  and inline program definitions.
- The production non-emission gate inspects typed plans from representative
  builder, session, playground, and export paths before checking the schema
  tag on their wire form.

No differential can prove both sides correct. The
[trusted-boundary ledger](trust-boundary.md) names what is theorem-backed,
what is implementation-backed, and what remains empirical.

## Repository map

```text
lean/Tropical/
  EmitArrow/                 fourteen-constructor authoring tree and builders
  Ir/Nodes.lean              ResolvedProgram / ENode / ExprArena trunk
  Ir/Strata.lean             direct lowering
  Compile.lean               partition + tropical_plan_5 construction
  Ir/Stage0.lean             typed binding-time split
  Ir/EmitLlvm.lean           LLVM kernel emitter
  Ir/EmitMsl.lean            Metal kernel emitter
  Engine/                    session, MCP mutations, ingest/export, hot-swap
  Playground/                product graph decode/lower/report
engine/
  runtime/                   manifest host and exact-epoch tile handoff
  jit/                       textual LLVM → ORC and LLVM → wasm32 support
  metal/                     Metal kernel plus dedicated render worker
  c_api/                     stable native and socket boundary
web/
  runtime/                   precompiled WebAssembly player
tests/                       cross-backend and golden fixtures
```

For contributor commands and platform requirements, see
[`CLAUDE.md`](../CLAUDE.md). For retired-surface rejection evidence, see
the [compatibility matrix](compatibility-matrix.md).
