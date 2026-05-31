# tropical Architecture

Detailed reference for the internal architecture. The shape of the
document follows the shape of the compiler: each section is one IR
along the pipeline, named by the structure that gets discarded
between it and its predecessor.

The compiler does the same kind of work as a typical
programming-language compiler — successive passes that simplify the
program — and is designed around one rule: a pass never carries
forward structure the next pass would rather not see. Surface
syntax, names, type parameters, sum types, instance nesting, array
shapes, combinators all get retired at the right moment. By the end
of the chain, the IR is `tropical_plan_5`: per-instance instruction
streams over typed scalar slots plus a scheduler that drives them
per-sample. Two backends interpret that low-detail IR — the JIT and
WebAssembly — and equivalence test suites cross-check them
sample-for-sample.

## What tropical is, categorically

A single sentence to hang the system off:

> tropical's IR is a DAG-shaped operad — programs are typed
> signal-flow graphs with cycles broken explicitly by a single state
> primitive (`RegDecl`), and the compiler is a functor between this
> operad and the slot-operational operad consumed by the runtime.

The vocabulary doesn't have to be load-bearing day to day, but the
shape of the codebase matches it precisely:

- **Colors** = post-strata port types (`float`, `int`, `bool`,
  fixed-shape arrays of these).
- **Operations** = primitive ops (`add`, `mul`, `delay`, …) plus
  user-defined programs.
- **Composition** = wiring an output to an input.
- **Tensor (parallel composition)** = placing instances side by side
  in a `body.decls` list.
- **Encapsulation** = an instance's body can only see its own input
  ports; cross-program communication happens at the parent's level
  by writing into a child's input slot. This is structural, not a
  discipline: the IR types make boundary-crossing references
  inexpressible.

**The IR is acyclic by construction.** Source-level cycles that
don't pass through an explicit user register are rejected at the
elaborator (`CycleViolation`, with port-detailed Tier-2 errors and a
suggested explicit-delay fix). Session-level cycles in MCP-built
graphs are broken at the wire layer: every wire stored via
`setWireExpr` is wrapped in a unit delay, and a pre-emit pass hoists
those delays out of the wires into session-level slots updated in
the scheduler's `state_evolution` phase. Hand-written JSON patches
with cross-coupled instances must wrap their own back-edges in
`delay()` to break the session-level cycle —
`assertSessionAcyclic` (`compiler/ir/lowering/session_cycle_check.ts`)
runs as a defensive invariant at `compileSession`'s entry. Every
MCP wire gains exactly one sample of latency (~21 µs at 48 kHz),
matching VCV Rack's per-wire-delay mental model.

For the longer categorical story — the choice of trace functor (we
sit at "implicit cycle-breaking via explicit user delays"), the
pre-trace/post-trace operad split, multi-realization (WDF,
iterative fixed-point, FFI), and the continuous → discrete → finite-
precision semantic hierarchy — see the archived design conversation
at `design/archive/operadic_ir.md`. Most of it is the framing that
produced the current shape; some of it (multi-realization, slot
unification, fractal compilation) is infrastructure-complete but
not yet on the default path.

## Pipeline at a glance

```
.trop / tropical_program_2 / MCP edits
    │
    │   parse              drops layout, sugar, bounds annotations
    ▼
ParsedProgram
    │
    │   elaborate          drops names (every NameRef → decl object);
    │                      enforces acyclic-source invariant
    ▼
ResolvedProgram                              ◀── DAG-shaped graph IR
    │
    │   strata pipeline:
    │     assertAcyclic    confirms caller honored the contract
    │     specialize       drops type parameters
    │     sumLower         drops sum types
    │     inlineInstances  drops nesting
    │     arrayLower       drops shapes and combinators
    │     identityElim     categorical identity-law rewrite
    ▼
ResolvedProgram (post-strata)
    │      scalar-only · monomorphic · acyclic · non-nested · combinator-free
    │
    │   ┌─→ compileSession → tropical_plan_5  (JIT path)
    │   │       │
    │   │       │ liftWiresToInstances → extractSessionDelays
    │   │       │     → assertSessionAcyclic → compileSessionSlotted
    │   │       │     (per-instance compileResolved into
    │   │       │      instance_functions[] + scheduler_function)
    │   │       │
    │   │       │   ──── C API boundary (engine/c_api/tropical_c.h, koffi FFI) ────
    │   │       ▼
    │   │   NumericProgramParser → FlatProgram (multi-function)
    │   │   OrcJitEngine → LLVM IR (one kernel function whose body is the scheduler)
    │   │   FlatRuntime → buffer loop, double-buffered hot-swap
    │   │   TropicalDAC (RtAudio) → audio output
    │   │
    │   └─→ emit_wasm  → WebAssembly bytes
    │           │
    │           ▼
    │       WasmRuntime in AudioWorklet (web/worklet/runtime.ts)
```

Both backends consume the same `tropical_plan_5`: the session keeps
each instance compiled independently (one `InstanceFunction` per
instance, packed into the FlatPlan by the scheduler), and the kernel
is one LLVM function (or, in WASM, one `process` export) whose body
dispatches each instance in topological order per sample. The
equivalence tests pin the JIT and WASM outputs sample-for-sample to
each other.

Every pass below is presented as input IR, output IR, and the
structure dropped between them.

---

## 1. Surface and parse

**Input:** `.trop` source text or `tropical_program_2` JSON.
**Output:** `ParsedProgram` (`compiler/parse/nodes.ts`).
**Drops:** layout, comments, surface sugar.

### 1.1 The two front-ends

`.trop` source is the human-authored surface syntax used by the
stdlib and by hand-written patches. Four parsing layers produce a
`ParsedProgram`:

- `compiler/parse/lexer.ts` — token stream
- `compiler/parse/expressions.ts` — infix precedence, unary, calls,
  let/combinator lambdas (binders introduced on the fly via
  `BinderDecl`)
- `compiler/parse/statements.ts` — block bodies (decls + assigns)
- `compiler/parse/declarations.ts` — program signatures, port specs,
  type-param declarations, nested programs

The legacy/MCP entry is JSON. `compiler/parse/raise.ts` walks a
`tropical_program_2` JSON object and emits the same `ParsedProgram`
shape. Both front-ends converge on the same input to the elaborator.

### 1.2 Parse-time desugaring: bounds

`compiler/parse/lower_bounds.ts` runs immediately after parse, before
anything else sees the program. It rewrites:

- `signal[-1, 1]`, `unipolar[0, 1]`, `bipolar[-1, 1]`, `phase[0, 1]`,
  `freq[0, ∞)` — named bound aliases — to explicit `clamp` or
  `select` ops
- explicit `in [lo, hi]` annotations on input defaults — same
  treatment

Bounds are a *parse-time* concept. They never reach the elaborator
or the strata pipeline as a distinct construct.

### 1.3 What this pass deliberately doesn't do

`raise.ts` and the `.trop` parser perform **zero scope analysis**.
Every reference (`input("freq")`, `sin1.out`, `param("cutoff")`,
`reg("phase")`) emits a `NameRefNode` placeholder. The parser doesn't
know which declarations are in scope, doesn't validate that the name
exists, doesn't disambiguate between (say) an instance output and a
register read. Name resolution lives in exactly one pass: the
elaborator. This is the cleanest separation we've found between
syntactic and semantic concerns.

---

## 2. Elaborate

**Input:** `ParsedProgram`.
**Output:** `ResolvedProgram` (`compiler/ir/nodes.ts`).
**Drops:** names. Enforces the **acyclic-source** invariant.
**Keeps:** every other piece of surface structure (type params, sum
types, nesting, combinators, shapes).

`compiler/ir/elaborator.ts` is the unique site of name resolution.
A single top-down pass over the parsed program: each declaration is
constructed once when its parsed counterpart is encountered,
registered in the appropriate scope, and re-used by reference at
every site that names it.

The output is a graph IR. Decls (`InputDecl`, `OutputDecl`,
`RegDecl`, `ParamDecl`, `TypeParamDecl`, `InstanceDecl`,
`ProgramDecl`, `BinderDecl`, plus sum/struct/alias type defs) are
introduction sites. Refs (`InputRef`, `RegRef`, `ParamRef`,
`TypeParamRef`, `BindingRef`, `NestedOut`) are uses. Every ref
carries its `decl` field as a direct object pointer — `===`
identity, not a string lookup. `RegDecl` is the single state
primitive: surface `delay name = u init v` is sugar for
`reg name { init: v, update: u }`.

The elaborator also runs **strict cycle detection** at the source
level. Inter-instance cycles that don't pass through an explicit
user register throw `CycleViolation` here, with a port-detailed
Tier-2 error message and a suggested explicit-delay fix.
Acyclic-by-source is what makes everything downstream straight-
line: the strata pipeline never needs to break cycles, and
`assertAcyclic` at strata entry catches any caller that bypassed
the contract.

After this pass, no string lookups, no scope walks, no shadowing
disambiguation. Decl identity is the only substrate the rest of the
compiler operates on.

---

## 3. Strata pipeline

`compiler/ir/strata.ts` orchestrates six passes:

```typescript
export function strataPipeline(prog, typeArgs = new Map(), options = {}) {
  assertAcyclic(prog)
  const specialized = specializeProgram(prog, typeArgs)
  const summed     = sumLower(specialized)
  const inlined    = options.inlineNested === false ? summed : inlineInstances(summed)
  const arrayed    = arrayLower(inlined)
  return identityElim(arrayed)
}
```

Each pass is pure: returns a fresh `ResolvedProgram`, or — in the
no-op fast path — the input by identity. None of them mutate decls
on the input. **Acyclicity is the strataPipeline contract**; the
callers (`programTypeFromResolved`)
guarantee it, `assertAcyclic` confirms it.

### 3.1 assertAcyclic — boundary check

`compiler/ir/acyclic.ts`. Re-uses the SCC finder from
`compiler/ir/lowering/cycle_break.ts` (the shared cycle algorithm,
also available to future realizations that want their own
cycle-break policy — iterative, WDF, etc.). Throws
`AcyclicityViolation` if any non-trivial SCC survives into strata
input. In the standard path this never fires; the elaborator and
the session compiler have already ensured acyclicity.

### 3.2 specialize — drops type parameters

`compiler/ir/specialize.ts`. Takes a generic program and a map
`Map<TypeParamDecl, number>`; produces a fresh program with the
integers substituted in.

Substitution sites:
- `ShapeDim`s that are `TypeParamDecl`s become integers
- expression-position `TypeParamRef` nodes become numeric literals
  (with `ConstNode` wrapping when type information is needed)
- the root program's `typeParams` list is dropped

Each `(template, args)` pair produces a structurally fresh program;
`Delay<N=8>` and `Delay<N=44100>` give distinct `RegDecl` objects
with shapes `[8]` and `[44100]`. Sum/struct/alias type defs are
shared across the clone (preserves variant identity for match arms,
which `sum_lower` requires).

The session-emit cache (`session.specializationCache`) memoizes on
`(template, args)` keys. The pass itself doesn't consult the cache —
that's the loader's job.

### 3.3 sumLower — drops sum types

`compiler/ir/sum_lower.ts`. Decomposes every sum-typed `RegDecl`
into N+1 scalar `RegDecl`s — a discriminator slot (int) plus one
slot per `(variant, field)` pair across all variants — and lowers
`MatchExpr` to scalar select-chains and `TagExpr` to tag-literal
writes.

Constraints:
- a sum-typed reg's `init` MUST be a `TagExpr` (constant variant
  constructor); anything else is a structural error
- match-arm payload bindings are only supported when the scrutinee
  is a `RegRef` to a sum-typed reg

After this pass: no `tag` op, no `match` op, no sum-typed reg.
Decl identity is preserved end-to-end; all replacements are fresh
decls and refs are rewritten by identity.

`EnvExpDecay` and `TriggerRamp` in the stdlib are the canonical
examples — both define a two-variant sum that this pass decomposes
into a tag register plus a payload register.

### 3.4 inlineInstances — drops nesting

`compiler/ir/inline_instances.ts`. Splices each `InstanceDecl` into
its parent. After this pass:

- `body.decls` contains no `InstanceDecl`
- no expression contains a `NestedOut` ref

For each instance, depth-first bottom-up:

1. Specialize the instance's type with `instanceDecl.typeArgs` (so
   the inner already looks monomorphic by the time we touch it).
2. Recursively inline sub-instances inside the (specialized) inner.
3. Clone the inner with **input substitution**: every `InputRef`
   whose decl is in the inner's `ports.inputs` is replaced by the
   wired-in expression from `instanceDecl.inputs[port]`. Substituted
   expressions pass through by reference, preserving DAG sharing.
4. Lift cloned `RegDecl`s into the outer body, renamed
   `${instance.name}_${innerName}` and tagged
   `_liftedFrom: instance.name`. Lift cloned `next_update` assigns
   with their `target` rewritten. Lift `ProgramDecl`s and
   `ParamDecl`s as-is (no rename: `ParamDecl`s are session-scoped
   by name; `ProgramDecl`s are passive type bindings).
5. Record cloned `outputAssign` expressions in a substitution table
   keyed by the *template's* `OutputDecl` (matched by position to
   the cloned program's outputs). Replace every `NestedOut {
   instance, output }` ref in the outer's surviving expressions.

`_liftedFrom` is the post-strata replacement for the legacy
name-prefix parsing pattern. Identifying a decl's lineage is now an
object-field check, not a string regex.

The `inlineNested: false` option is the entry point for the
fractal-compilation path: sub-instances survive as kernel boundaries
and `partitionKernel` (the recursive partitioner in
`compile_session_slotted.ts`) emits an `InstanceFunction` for every
`InstanceDecl` at every level. Cross-kernel input wiring is handled
by slot-based parent→child writes: the parent's body emits
`WriteSlot` ops into a per-child `pre_child_instructions` block
(evaluated in the parent's scope, so every ref resolves), and the
child reads its inputs via slot reads in its own body. Both paths
(`inlineNested: true` flat, `inlineNested: false` fractal) are
verified sample-for-sample at 1e-12 by
`tests/equiv/nested_vs_inlined.test.ts`; the choice between them is
a runtime-cost tradeoff documented in
`tests/bench/depth_vs_flat_*.md`.

### 3.5 arrayLower — drops shapes and combinators

`compiler/ir/array_lower.ts`. Unrolls compile-time combinators and
lowers array ops to scalar primitives via static shape information.

After this pass:
- no `let`, `fold`, `scan`, `generate`, `iterate`, `chain`, `map2`,
  `zipWith` ops
- no `bindingRef` (every `BinderDecl` introduced by a combinator or
  let has been substituted away)
- every `zeros{count}` with literal `count` becomes an inline array
  `[0, 0, ..., 0]`

Survivors (the post-arrayLower form admits these):
- inline arrays as `ResolvedExpr[]` — element of `arrayPack`-style
  values
- `index(arr, i)` — left as-is (never constant-folded over inline
  literals)
- `arraySet(arr, i, v)` — left as-is; backs stateful arrays like
  `Delay`'s ring buffer

Substitution discipline: every `BindingRef.decl` is a pointer to a
`BinderDecl`. Substitution is by `Map<BinderDecl, ResolvedExpr>`,
which makes shadowing structurally impossible — shadowing would
require two different `BinderDecl`s with the same name, and that's
exactly what decl identity catches.

Each combinator iteration uses a fresh `WeakMap` memo. A memo is
valid only for one substitution map; reusing one across iterations
would conflate different `acc`/`elem` values.

### 3.6 identityElim — categorical identity-law rewrite

`compiler/ir/identity_elim.ts`. Eliminates `InstanceDecl`s whose
program body is the identity morphism (each output assigns the
corresponding input untouched). In the per-program path
`inlineInstances` already absorbs trivial sub-instances first, so
this pass rarely fires today; it carries weight in the session-
level path where lifted wire-programs sometimes lower to a
no-op kernel that the pass collapses cleanly.

### 3.7 Post-strata invariants

What you get from `strataPipeline`:

- **scalar-only** — no surviving array decls; arrays survive only as
  inline literals in expressions or as backing stores for stateful
  arrays
- **monomorphic** — no `TypeParamDecl`, no `TypeParamRef`
- **acyclic** — confirmed at strata entry; production code paths
  never produce cyclic IR
- **non-nested** — no `InstanceDecl`, no `NestedOut`
- **combinator-free** — no `let`, `fold`, `scan`, `generate`,
  `iterate`, `chain`, `map2`, `zipWith`
- **decl-identity-keyed** — refs hold decl objects, never strings

That's the smallest sub-IR sufficient for any per-sample evaluator.
The backends below operate on this image.

---

## 4. Sessions: materialize and compile

A session is the runtime view of a graph in flight: a registry of
typed `Instance`s, a wiring map keyed by `(instance, input)`, a
list of graph outputs wired into the synthetic `dac` instance, and
a parameter registry. The MCP server owns one long-lived session;
patches in `tropical_program_2` JSON are loaded into a session and
saved back out from one.

Two paths consume a session.

### 4.1 JIT path — `compileSession` (per-instance)

`compiler/ir/compile_session.ts` runs three pre-emit passes and
hands the result to `compileSessionSlotted`:

1. **`liftWiresToInstances`** (`compiler/ir/lift_wires.ts`) —
   anonymous-instance lift. Wires whose expressions contain array
   literals are extracted into anonymous `__wire_${i}` instances at
   session pre-compile time. The lifted programs go through the
   full strata pipeline so combinators lower correctly.

2. **`extractSessionDelays`**
   (`compiler/ir/lowering/extract_session_delays.ts`) — hoists every
   top-level `delay()`-wrapped wire into a fresh module slot. The
   wire is rewritten to a `sessionSlot` read; the source expression
   is recorded in `session.delaySlotRegistry` for
   `compileSessionSlotted` to emit as a `WriteSlot` in the
   scheduler's `state_evolution` phase. This is the structural
   mechanism that keeps the MCP-built IR acyclic — every wire becomes
   a slot-to-slot copy with one sample of latency.

3. **`assertSessionAcyclic`**
   (`compiler/ir/lowering/session_cycle_check.ts`) — defensive
   invariant on the post-extraction dep graph. Catches programmatic
   sessions that bypass both `setWireExpr`'s auto-wrap and explicit
   `delay()`.

Then `compileSessionSlotted` compiles each instance standalone
through `compileResolved` and packs the results into a
`tropical_plan_5` `FlatPlan` with one `InstanceFunction` per
instance plus one `SchedulerFunction`. The scheduler's four phases
sequence the per-sample work:

```
for each sample:
  preamble        (currently empty; reserved for future setup)
  for each instance, in topo order: body + writebacks
  state_evolution (WriteSlot per extracted delay)
  postamble       (DAC stitch — read graphOutput slots into mix temps)
  output mix
```

Branded indices throughout (`TempIdx`, `StateRegIdx`,
`ArraySlotIdx`, `ModuleSlotIdx`) make cross-namespace arithmetic a
compile error — the literal shape of a Phaser-era slot-mixing bug.
See `compiler/ir/slot_indices.ts`.

### 4.2 Fixed-topology compilation

Tropical compiles a session graph to native code with a fixed
topology for the lifetime of the kernel. Two compilation modes
selected by `FlatPlan.compilation_mode`:

- `'fused'` (default) — one monolithic LLVM function for the whole
  session per buffer. LLVM fuses across instances aggressively.
- `'microkernel'` — one LLVM function per top-level session instance,
  plus preamble / state_evolution / postamble_mix; the per-sample
  outer loop runs in C++ and dispatches them via function pointers.
  Trades cross-instance fusion for superlinear cold-compile speedup
  (LLVM optimizer scales worse on one huge function than on N
  smaller ones). Sample-for-sample equivalent to fused at 1e-12 per
  `tests/equiv/microkernel_vs_fused.test.ts`.

Topology changes (adding/removing instances, rewiring) trigger
hot-swap to a freshly compiled kernel with state transferred by
name; both modes use the same hot-swap mechanism. There is no
per-instance runtime gating — every instance runs every sample.
This is the shape of synthesis the language is good at;
dynamic-lifecycle semantics belong in a different language with a
different runtime.

---

## 5. Backends over the post-strata IR

The three sections below are not further compiler stages. They are
parallel interpretations of the same post-strata `ResolvedProgram`
into different targets. Param handles are the only thing that
differs between them: wiring expressions reference parameters by
name (`{op:'param', name}`); the materializer resolves names to
handles at compile time. For the JIT path the handle is a native
pointer (`tropical_param_t`); for the WASM path it's a SAB slot
index, stringified to keep the `tropical_plan_5` schema backend-
agnostic.

### 5.1 compileResolved → tropical_plan_5 → JIT

**`compiler/ir/compile_resolved.ts`** is the per-instance emit:

1. `buildSlotMaps(prog)` — assign integer slots to decl objects
   (`Map<RegDecl, number>`, etc.). Slot identity, not slot name, is
   what the JIT consumes.
2. `emitNumericProgram` (`compiler/ir/emit_resolved.ts`) — walk the
   lowered IR, emit a `PerInstancePlan` whose instructions follow
   the `tropical_plan_5` schema in `compiler/flat_plan.ts`.

`compile_session_slotted.ts` packs N `PerInstancePlan`s into
`instance_functions[]` of a `FlatPlan`, shifting indices by the
per-instance offsets, and builds the `SchedulerFunction` with the
DAC stitch postamble and the `state_evolution` `WriteSlot`s.

**Structural CSE.** `emit_resolved.ts` keys CSE on a bottom-up
structural id (interned via `${op}|${field=}|${child_id}` strings),
not node identity. This catches duplicates that strata's
clone-then-substitute introduces.

**Operand kinds.** `NOperand` discriminates: `const`, `input`, `reg`,
`array_reg`, `state_reg`, `param`, `rate`, `tick`. Terminals
(literals, sentinels, register reads) embed inline; non-terminal
expressions get a temp register.

**Array loops.** When `loop_count > 1` an instruction emits an
elementwise loop. `strides[i]` controls whether each argument
advances with the loop index (array stride = 1) or broadcasts
(stride = 0).

### 5.2 The JIT path

The plan crosses the C API boundary as JSON. On the C++ side:

```
NumericProgramParser::parse_plan5()    ← engine/runtime/NumericProgramParser.hpp
  └─ thin JSON deserializer; reads per-instance plans plus the
     scheduler into a FlatProgram (multi-function) struct. A
     backcompat lift exists for single-kernel plan_4 inputs (hand-
     crafted unit tests, legacy saved plans).
OrcJitEngine::compile_flat_program()   ← engine/jit/OrcJitEngine.{hpp,cpp}
  └─ singleton LLVM ORC.
     1. Build canonical cache key (MD5 of serialized program with
        param pointers replaced by ordinals).
     2. Check in-memory cache; then disk cache at
        ~/.cache/tropical/kernels/<build-id>/. Build-id derived
        from LC_UUID (macOS) / ELF build-id, so dylib rebuild
        auto-invalidates.
     3. Generate typed LLVM IR. One outer kernel function whose
        body, per sample, runs:
            preamble (currently empty)
            for each instance: body + writebacks
            state_evolution (delay-slot WriteSlots)
            postamble (DAC stitch reads)
            output mix
        Per-instruction operand resolution emits f64/i64/i1 with
        explicit coercion; array loops when loop_count > 1.
     4. LLJIT compile, look up symbol → NumericKernelFn.
FlatRuntime::load_plan()              ← engine/runtime/FlatRuntime.{hpp,cpp}
  └─ State init: stateInit values written into i64 backing store
     with type-aware bit-cast. Named state transfer: registers,
     arrays, and module slots copied by name from the outgoing
     kernel for click-free hot-swap. Atomic active-slot
     store-release publishes the new kernel.
FlatRuntime::process()                 (audio thread)
  └─ Acquire active state. Call kernel (single invocation processes
     the buffer). Advance sample_index. Apply 2048-sample smoothstep
     fade envelope.
TropicalDAC::audio_callback            ← engine/dac/TropicalDAC.hpp
  └─ Templated RtAudio driver. Copies mono output to all channels.
     Watcher thread polls 50 ms for device disconnect; recovers with
     500 ms backoff + fade-in. switch_device() is explicit live-switch.
```

**No transcendentals in the JIT.** `sin`, `cos`, `tanh`, `exp`,
`log`, `pow` are stdlib `.trop` programs (polynomial approximations
using arithmetic + `Ldexp` + `FloatExponent` — single-instruction
IEEE-754 bit ops for 2^n range reduction). They inline at strata
time. The kernel contains no libm calls and is deterministic across
platforms. Swap `stdlib/Sin.trop` to change the approximation.

**Adding an op.** A new operation that the JIT must support requires:
1. Add the variant to the `OpTag` enum in
   `engine/jit/OrcJitEngine.hpp`.
2. Add the tag-string mapping in
   `engine/runtime/NumericProgramParser.hpp::parse_op_tag`.
3. Add the LLVM IR emission case in
   `engine/jit/OrcJitEngine.cpp::compile_flat_program`.
The same op also needs to land in `WireFormatOp` (`compiler/expr.ts`),
the strata passes that traverse it, and the other two backends
(`emit_resolved.ts`, `emit_wasm.ts`).

### 5.3 emit_wasm — the WebAssembly backend

**`compiler/emit_wasm.ts`** + **`compiler/wasm_memory_layout.ts`**.
The second interpretation of post-strata `ResolvedProgram`, consuming
the same `tropical_plan_5` boundary type. The emitter produces a
standalone WASM module exporting a single
`process(buffer_length, start_sample_index)` function plus a shared
`memory`.

**Linear-memory layout** (`wasm_memory_layout.ts`):

```
inputs        f64[inputCount]                — set by host, kernel reads
registers     i64[registerCount]             — kernel state
temps         i64[registerCount]             — per-sample scratch
arrays        f64[arraySlotSizes...]         — array-typed register backing stores
param_table   f64[paramCount]                — host writes per-block
param_frame   f64[paramCount]                — host writes per-block (trigger snapshot)
output        f64[maxBlockSize]              — kernel writes mono audio out
```

8-byte aligned, contiguous from offset 0. Layout is shared with
`web/worklet/runtime.ts` so offsets stay in sync.

**WASM kernel structure** mirrors the LLVM kernel: same per-sample
sequencing (preamble / per-instance bodies + writebacks / state
evolution / postamble / output mix), implemented in WASM bytecode
over the linear memory layout above.

**Encoding.** `i64` cells store either an f64 bitcast, a signed
int, or a zero-extended bool; the per-instruction `result_type`
tells the codegen which load/store to use. f64 cells back arrays
and the output buffer.

**Param flow.** Plan `param.ptr` strings hold SAB slot indices
instead of native pointers (`tropical_param_t*`). The kernel emits
`f64.load (paramTableOffset + ptr*8)` for `param` operands. The
host populates these regions per-block from a `SharedArrayBuffer`
shared with `web/host/params.ts:WebParam`.

The runtime side — instantiation, hot-swap, fade envelope, param
snapshotting — lives in `web/worklet/runtime.ts` and mirrors
`FlatRuntime` in shape. See `web/CLAUDE.md` for the runtime details.

### 5.5 Equivalence

The pipeline is correct only if every pass and every backend agrees
with the per-sample semantics on the input. The cross-checking
suites:

- `tests/equiv/wasm_vs_jit.test.ts` — WASM emit and JIT agree
  sample-for-sample (the two backends, both off `tropical_plan_5`).
- `tests/equiv/microkernel_vs_fused.test.ts`,
  `tests/equiv/nested_vs_inlined.test.ts`,
  `tests/equiv/microkernel_deep.test.ts` — realization-variant
  differentials within the JIT (fused vs. per-instance microkernel,
  flat vs. nested), exercising the scheduler/slot layer the two
  backends share.
- `tests/equiv/migration_audio.test.ts` — byte-for-byte audio goldens.
- `tests/equiv/web_plans_vs_jit.test.ts` — every precompiled plan in
  `web/dist/patches/` matches the JIT output.

Any disagreement is a strata or backend bug; the suite localises
which.

---

## 6. ProgramType and Compiled

`compiler/program_types.ts`. Thin wrapper over a post-strata
`ResolvedProgram`. The wrapper is metadata; the IR is the value.
Free-function helpers (`inputNames`, `outputNames`,
`registerNames`, `inputPortTypes`, …) read off the resolved IR;
slot-derived fields cache lazily via `buildSlotMaps`.

`Instance` holds a `Compiled` plus an instance name, `baseTypeName`,
optional `typeArgs`, plus session-level `gateable` / `gateInput`
fields.

`session.ts:resolveProgramType` calls
`programTypeFromResolved(template, subst)` (full strata pipeline +
wrap), then `rename(key)` so the specialization cache key
(`Type<N=8>`) shows up as the program's name in serialized output.

---

## 7. Schema versions

Two distinct JSON schemas; do not confuse them.

| Schema | Produced by | Purpose |
|--------|-------------|---------|
| `tropical_program_2` | `compiler/program.ts`, `compiler/parse/raise.ts` | The high-detail input shape: a program with typed ports, a body block of decls/assigns, optionally generic in `type_params`. Authored by humans (in `.trop`) or by agents (over MCP). |
| `tropical_plan_5`    | `compiler/ir/compile_session_slotted.ts` (`compiler/flat_plan.ts` schema) | The low-detail output: per-instance instruction streams (`instance_functions[]`) plus a `scheduler_function` with preamble / state_evolution / postamble phases. The C++ JIT and the WASM emitter both consume this shape. The engine still accepts the older `tropical_plan_4` (single-kernel form) for hand-crafted unit tests; it's lifted into a one-instance plan_5 at parse time. |

Schema validation: `compiler/schema.ts` (Zod) for input;
`compiler/flat_plan.ts` (branded TypeScript types) for output.

Going from the first to the second without losing meaning is exactly
what the strata pipeline does.

### 7.1 tropical_program_2 sketch

```json
{
  "schema": "tropical_program_2",
  "name": "MyPatch",
  "body": {
    "op": "block",
    "decls": [
      { "op": "instance_decl", "name": "sin1", "program": "Sin",
        "inputs": { "x": { "op": "mul", "args": [6.283185307179586, { "op": "sample_index" }] } } },
      { "op": "instance_decl", "name": "amp", "program": "VCA",
        "inputs": {
          "audio": { "op": "ref", "instance": "sin1", "output": "out" },
          "cv": 0.5
        } }
    ],
    "assigns": []
  },
  "audio_outputs": [{ "instance": "amp", "output": "out" }]
}
```

### 7.2 tropical_plan_5 shape

```
{
  schema: "tropical_plan_5",
  config: { sample_rate, ... },
  instance_functions: [
    { name, register_count, state_init, register_names, register_types,
      array_slot_sizes, instructions, output_targets, register_targets, ... },
    ...
  ],
  scheduler_function: {
    preamble:        NInstr[],   // currently empty
    state_evolution: NInstr[],   // WriteSlot per extracted delay
    postamble:       NInstr[],   // DAC stitch reads
    outputs:         number[],   // indices into output mix
    ...
  }
}
```

The `tropical_plan_5` shape is the C-API contract. Anything the
backends need to know about the program — instance kernels, the
scheduler that drives them, names, types, slot counts, init state —
is in there; everything the compiler decided to forget along the
way is gone.

---

## 8. C API (`engine/c_api/tropical_c.h`)

The stable C interface between TypeScript (via koffi FFI) and C++.
All handles are opaque `void*`. Errors are thread-local, fetched via
`tropical_last_error()`; valid until the next call on the same thread.

### 8.1 ControlParam

- `tropical_param_new(init_value, time_const)` — smoothed parameter
  with one-pole lowpass; `time_const` is τ in seconds (e.g. `0.005`
  for ~5 ms ramp; `0.0` for no smoothing)
- `tropical_param_set` / `tropical_param_get` — atomic store / load
- `tropical_param_free`

`Param` is now the only control-parameter primitive. The legacy
`Trigger` type (fire-once via atomic exchange) was retired; on the
wire, `{op:'trigger', name}` refs are still accepted and aliased to
`{op:'param', name}` at materialization for backcompat.

### 8.2 FlatRuntime

- `tropical_runtime_new(buffer_length)`
- `tropical_runtime_load_plan(rt, json, len)` — parse + JIT + state
  init + atomic hot-swap. Returns false on compile failure;
  `tropical_last_error()` carries the message. Previous kernel keeps
  playing.
- `tropical_runtime_process(rt)` — process one buffer
- `tropical_runtime_output_buffer(rt)` → `const double*`
- `tropical_runtime_get_buffer_length(rt)`
- `tropical_runtime_begin_fade_in` / `_begin_fade_out` /
  `_is_fade_out_complete`
- `tropical_runtime_free`

### 8.3 DAC

- `tropical_dac_new_runtime(rt, sample_rate, channels)`
- `tropical_dac_start` / `_stop` / `_is_running`
- `tropical_dac_get_stats(dac, &stats)` — `callback_count`,
  `avg_callback_ms`, `max_callback_ms`, `underrun_count`,
  `overrun_count`
- `tropical_dac_reset_stats`
- `tropical_dac_is_reconnecting` — true while disconnect recovery is
  in progress
- `tropical_dac_get_active_device` / `_switch_device` — query / live
  device switching

### 8.4 Device enumeration (no DAC instance required)

- `tropical_audio_device_count()`
- `tropical_audio_get_device_ids(out, count)`
- `tropical_audio_get_device_info(id, &info)` — fills
  `tropical_device_info_t` with name, channels, default flag,
  preferred + supported sample rates
- `tropical_audio_default_output_device()`

---

## 9. FFI bridge (`compiler/runtime/`)

TypeScript wrappers over the C API via koffi.

- `bindings.ts` — raw koffi function declarations matching
  `tropical_c.h`. Loads `libtropical.dylib` from `build/` or
  `build-profile/`.
- `runtime.ts` — `Runtime` class wrapping `tropical_runtime_t`.
  FinalizationRegistry for GC-driven cleanup.
- `audio.ts` — `DAC` class wrapping `tropical_dac_t`. Static
  `listDevices()`.
- `param.ts` — `Param` wrapping `tropical_param_t`. Wiring
  references parameters by name; the materializer resolves the name
  to a `_h` handle and threads it into the plan via `paramHandles`.

---

## 10. MCP server (`mcp/server.ts`)

The primary agent interface. Runs on stdio, uses
`@modelcontextprotocol/sdk`. Maintains one long-lived `SessionState`.

23 tools, grouped by purpose. Every tool that mutates the signal
graph ultimately calls `wire()` → `applyFlatPlan(session, runtime)`,
which runs the full compile pipeline:

```
SessionState
  → compileSession (compiler/ir/compile_session.ts)
       → liftWiresToInstances → extractSessionDelays
         → assertSessionAcyclic → compileSessionSlotted
       → per-instance compileResolved → tropical_plan_5 JSON
  → JSON.stringify
  → runtime.loadPlan (NumericProgramParser → OrcJitEngine → FlatRuntime hot-swap)
```

Compile errors don't kill the session: they return a structured
error envelope (`ErrorEnvelope` in `mcp/server.ts`) and the previous
kernel keeps playing.

See [`mcp/CLAUDE.md`](../mcp/CLAUDE.md) for the full tool list,
SessionState integration, and error envelope shape.

---

## 11. Web backend (`web/`)

Browser co-implementation of the audio runtime. Same compiler
front-end, same strata pipeline, same `tropical_plan_5` boundary;
different emit target (WebAssembly vs. LLVM IR) and different param
handle representation (SAB slot index vs. native pointer).

```
web/
  build_patches.ts    Offline plan precompile: tropical_program_2 → web/dist/patches/*.plan.json
  build.ts            Full demo bundle: regen stdlib_bundled, precompile patches,
                      bundle worklet + main app, copy index.html
  bundle_stdlib.ts    Generates compiler/stdlib_bundled.ts from stdlib/*.trop
  dev.ts              Dev server with COOP/COEP headers (SAB requirement)
  host/               Main thread
    compiler.ts       compilePlan(FlatPlan) → LoadedPlan via emit_wasm
    context.ts        AudioContext + AudioWorkletNode wiring
    params.ts         ParamBank (SharedArrayBuffer), WebParam
  worklet/            Audio thread
    runtime.ts        WasmRuntime: dual-slot hot-swap, fade envelope, snapshotParams
    processor.ts     AudioWorkletProcessor delegate; postMessage protocol
  site/               Browser UI
    app.ts, index.html
```

The full TS pipeline runs offline at build time; the browser only
runs `emit_wasm` and the runtime. Patches in `web/dist/patches/` are
fetched, compiled to WASM, and posted into the worklet.

`WasmRuntime` mirrors `FlatRuntime`: dual-slot hot-swap, state
transfer by name, 2048-sample smoothstep fade. The WASM module
exports a single `process(blockSize, sampleIdx)` function and a
shared `memory` whose layout is fixed by `wasm_memory_layout.ts`.

See [`web/CLAUDE.md`](../web/CLAUDE.md) for the runtime protocol,
linear-memory layout, and equivalence gates.

---

## 12. Type system (`compiler/term.ts`, `compiler/array_wiring.ts`)

### 12.1 Port types

- `Float`, `Int`, `Bool` — scalar
- `ArrayType(element, shape)` — static-shape arrays
- `StructType(name)`, `SumType(name)` — named ADTs
- `product(factors)` — n-ary tensor product
- `Unit` — monoidal unit

### 12.2 Shape algebra

Numpy-style static broadcasting: shapes right-aligned, dimension
pairs must be equal or one must be 1. `broadcastShapes(a, b)`
returns the result or null. Row-major layout via `shapeStrides`,
`shapeSize`, `flattenIndex`.

### 12.3 Array wiring

`compiler/array_wiring.ts` validates connections between typed
ports. Scalar-to-array auto-broadcasts; array-to-scalar errors;
shape mismatches inside compatible broadcast rules insert
`broadcast_to` wrappers.

---

## 13. Build system

### CMake (`CMakeLists.txt`)

- Target `tropical_core` (shared library, output name `libtropical`)
- C++20
- LLVM ≥ 19 (FATAL on lower; uses `getOrInsertDeclaration`)
- Submodules: RtAudio (`lib/rtaudio`), nlohmann/json (`lib/json`)
- Default build type `RelWithDebInfo`
- Test target `test_module_process`
- Options: `TROPICAL_BUILD_PYTHON`, `TROPICAL_PROFILE`,
  `TROPICAL_LLVM_STATIC`, `TROPICAL_BUILD_TESTS`

### Makefile

- `make build` — configure + build C++ core
- `make profile` — build with profiling instrumentation
- `make mcp-ts` — build + launch MCP server on stdio via Bun
- `make validate` — `bun test` + `ctest` + stdlib audit
  (`scripts/validate_stdlib.ts`)
- `make clean` — remove build directories

### Bun (`package.json`, `tsconfig.json`)

- Runtime: Bun ≥ 1.3
- Key deps: `@modelcontextprotocol/sdk`, `koffi`, `zod`
- TypeScript: ES2022 / ESNext, strict, includes `compiler/` + `mcp/`,
  excludes `*.test.ts`

### MCP (`.mcp.json`)

```json
{ "mcpServers": { "tropical": { "command": "bun", "args": ["run", "mcp/server.ts"] } } }
```

### CI (`.github/workflows/`)

Three jobs: typecheck (`bunx tsc --noEmit`), build-and-test (LLVM
20, libasound2-dev, `bun test` + `ctest`), and YAML lint.

---

## 14. Testing

### 14.1 C++ tests (`engine/tests/test_module_process.cpp`)

Custom harness, no framework dependency. Exercises FlatRuntime C
API and JIT without an audio device. Tests build plan JSON strings
directly and assert on output buffer values. Covers sawtooth, clock
with array ratios, integer sequences, multi-instance fusion,
smoothed params, hot-swap state transfer, typed int/bool ops.

`cmake --build build -j4 && ctest --test-dir build`.

### 14.2 TS tests (`compiler/**/*.test.ts`, `tests/equiv/*.test.ts`)

Run via `bun test`. The load-bearing suites:

- `tests/equiv/wasm_vs_jit.test.ts`,
  `tests/equiv/web_plans_vs_jit.test.ts` — WASM emission equivalence
- `tests/equiv/microkernel_vs_fused.test.ts`,
  `tests/equiv/nested_vs_inlined.test.ts`,
  `tests/equiv/microkernel_deep.test.ts` — JIT realization-variant
  differentials; `tests/equiv/migration_audio.test.ts` — audio goldens
- `ir/*.test.ts` — strata pipeline unit tests (specialize,
  sum_lower, inline_instances, array_lower, identity_elim, slots,
  clone, acyclic, lowering/cycle_break, elaboration_diagnostics)
- `parse/*.test.ts` — lexer, parser, raise, round-trip,
  `stdlib_round_trip.test.ts` (every `.trop` print/re-parse)
- `apply_plan.test.ts` — plan application integration (requires
  `make build`)
- `compiler/wasm_runtime.test.ts`, `compiler/emit_wasm.test.ts` —
  WASM emission unit tests

### 14.3 Stdlib audit

`bun run scripts/validate_stdlib.ts` parses, elaborates, and lowers
every `stdlib/*.trop` and confirms post-strata invariants. Run by
`make validate`.

---

## 15. Key design decisions

### Acyclic by construction

Cycles in source code throw `CycleViolation` at elaborate-time;
cycles in session wiring are broken at the wire layer by
`setWireExpr`'s auto-wrap + `extractSessionDelays`'s hoist into the
scheduler. The compiler's strata pipeline asserts acyclic input at
its boundary and refuses to lower cyclic IR. This closes the
asymmetry where the JIT silently tolerated cycles via slot
back-edges while the interpreter rejected them, and makes "the
trace functor" a property of the realization layer (elaborator +
session compiler), not the compiler.

### Single-kernel fusion, fixed topology

The whole session compiles to one native kernel function. No
instance boundaries at runtime, no per-instance dispatch, no
interpreter on the audio thread. LLVM gets to optimize across the
full graph. Topology changes hot-swap a freshly compiled kernel
with state transferred by name; per-instance lifecycle semantics
belong in a different runtime.

### Pipeline of structure-dropping passes

The compiler is a chain of IR-to-IR passes where each pass drops a
specific kind of structure once it's been consumed. That shape is
what makes the layout of `compiler/ir/` predictable, makes it clear
where new passes belong, and makes sample-for-sample cross-backend
agreement the right correctness criterion. The shape also matches
the operadic reading sketched at the top of this document and in
`design/archive/operadic_ir.md` — the strata pipeline is a
composition of operad morphisms — but you can program against the
pipeline without that vocabulary.

### Decl identity instead of strings

Past `elaborate`, every reference is a TypeScript object pointer.
String-based scope walks would make later passes re-derive
information the elaborator already established. `_liftedFrom`
replaced what had been name-prefix string regex; this is the same
move, applied to a different kind of provenance. Branded indices
in the plan layer (`TempIdx`, `StateRegIdx`, `ArraySlotIdx`,
`ModuleSlotIdx`) extend the same discipline to integer slot
identifiers, making cross-namespace arithmetic a compile error.

### Hot-swap via double-buffered kernels

Live audio never stops for recompilation. The new kernel is built on
a background thread; named state transfers; a single atomic store
publishes it. At most one sample of stale state is read. Same shape
on both backends (FlatRuntime in C++, WasmRuntime in WASM).

### No interpreter fallback on the audio path

JIT failures are fatal at the runtime. Compile errors are caught
upstream (in `applyFlatPlan` / over MCP) and never reach the
runtime. Cross-backend agreement is guaranteed instead by the
`tests/equiv/wasm_vs_jit` equivalence gate (JIT vs. WASM), not by any
runtime fallback.

### Static shapes for arrays

All array shapes are known at compile time. That's what makes
`arrayLower` — full unroll of combinators and array ops — total. No
dynamic allocation on the audio thread.

### Transcendentals as programs

`sin`, `cos`, `tanh`, `exp`, `log`, `pow` are `.trop` files using
arithmetic + `Ldexp` + `FloatExponent`. They inline at strata time.
No libm dependency in the kernel; deterministic across platforms;
swap a file to change the math. See `stdlib/README.md`.

### Thread-safe control parameters

`ControlParam` uses atomic load/store with relaxed ordering.
Smoothed params apply one-pole lowpass per sample. Safe from any
thread; the audio thread never blocks.
