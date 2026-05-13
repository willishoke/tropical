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
of the chain, the IR is `tropical_plan_4`: a flat instruction stream
over typed scalar slots. Three backends interpret that low-detail IR
— JIT, pure-TS interpreter, WebAssembly — and equivalence test
suites cross-check them sample-for-sample.

For the higher-level framing — programs as a cartesian category of
typed signal-flow graphs with a guarded trace — see the root
`CLAUDE.md`. The shape of every pass below matches that picture; the
vocabulary doesn't have to be load-bearing in this document.

```
.trop / tropical_program_2 / MCP edits
    │
    │   parse              drops layout, sugar, bounds annotations
    ▼
ParsedProgram
    │
    │   elaborate          drops names (every NameRef → decl object)
    ▼
ResolvedProgram
    │
    │   strata pipeline:
    │     specialize       drops type parameters
    │     sumLower         drops sum types
    │     traceCycles      breaks cycles via the guarded trace (cycles → DelayDecls)
    │     inlineInstances  drops nesting
    │     arrayLower       drops shapes and combinators
    ▼
ResolvedProgram (post-strata) — scalar-only, monomorphic, acyclic, non-nested, combinator-free
    │
    │   ┌─→ compileResolved → tropical_plan_4
    │   │       │
    │   │       │   ──── C API boundary (engine/c_api/tropical_c.h, koffi FFI) ────
    │   │       │
    │   │       ▼
    │   │   FlatRuntime::load_plan → OrcJitEngine::compile_flat_program → native kernel
    │   │       │
    │   │       ▼
    │   │   audio callback (RtAudio / CoreAudio)
    │   │
    │   ├─→ interpret_resolved (pure-TS evaluator; oracle for tests)
    │   │
    │   └─→ emit_wasm → WebAssembly bytes
    │           │
    │           ▼
    │       WasmRuntime in AudioWorklet (web/worklet/runtime.ts)
```

Sessions (graphs in flight, edited over MCP) plug into the same
pipeline through one extra step that lifts the partially-typed
session graph into a top-level `ResolvedProgram`:

```
SessionState (instances + wiring + dac.out + params)
    │
    │   materializeSession (compiler/ir/materialize_session.ts)
    │     lift the session graph into the same ResolvedProgram shape
    │     the per-program path produces.
    ▼
ResolvedProgram (top-level synthetic)  →  strata pipeline  →  post-strata
```

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
**Drops:** names.
**Keeps:** every other piece of surface structure (type params, sum
types, nesting, cycles, combinators, shapes).

`compiler/ir/elaborator.ts` is the unique site of name resolution.
A single top-down pass over the parsed program: each declaration is
constructed once when its parsed counterpart is encountered,
registered in the appropriate scope, and re-used by reference at
every site that names it.

The output is a graph IR. Decls (`InputDecl`, `OutputDecl`,
`RegDecl`, `DelayDecl`, `ParamDecl`, `TypeParamDecl`,
`InstanceDecl`, `ProgramDecl`, `BinderDecl`, plus sum/struct/alias
type defs) are introduction sites. Refs (`InputRef`, `RegRef`,
`DelayRef`, `ParamRef`, `TypeParamRef`, `BindingRef`, `NestedOut`)
are uses. Every ref carries its `decl` field as a direct object
pointer — `===` identity, not a string lookup.

The IR admits cycles. A delay's `update` may transitively reference
its own register; an instance's input may reference a value that
depends on the same instance via feedback. Those structures are
present in the resolved IR; later strata break them when they need
to.

After this pass, no string lookups, no scope walks, no shadowing
disambiguation. Decl identity is the only substrate the rest of the
compiler operates on.

---

## 3. Strata pipeline

`compiler/ir/strata.ts` orchestrates five passes:

```typescript
export function strataPipeline(prog, typeArgs = new Map()) {
  const specialized = specializeProgram(prog, typeArgs)
  const summed     = sumLower(specialized)
  const cyclic     = traceCycles(summed)
  const inlined    = inlineInstances(cyclic)
  return arrayLower(inlined)
}
```

Each pass is pure: returns a fresh `ResolvedProgram`, or — in the
no-op fast path — the input by identity. None of them mutate decls
on the input.

### 3.1 specialize — drops type parameters

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

### 3.2 sumLower — drops sum types

`compiler/ir/sum_lower.ts`. Decomposes every sum-typed `DelayDecl`
into N+1 scalar `DelayDecl`s — a discriminator slot (int) plus one
slot per `(variant, field)` pair across all variants — and lowers
`MatchExpr` to scalar select-chains and `TagExpr` to tag-literal
writes.

Constraints:
- a sum-typed delay's `init` MUST be a `TagExpr` (constant variant
  constructor); anything else is a structural error
- match-arm payload bindings are only supported when the scrutinee
  is a `DelayRef` to a sum-typed delay

After this pass: no `tag` op, no `match` op, no sum-typed delay.
Decl identity is preserved end-to-end; all replacements are fresh
decls and refs are rewritten by identity.

`EnvExpDecay` and `TriggerRamp` in the stdlib are the canonical
examples — both define a two-variant sum that this pass decomposes
into a tag register plus a payload register.

### 3.3 traceCycles — the guarded trace

`compiler/ir/trace_cycles.ts`. Tarjan's SCC over the inter-instance
dependency graph. Each non-trivial SCC chooses a break target by
source order; for each output port of the break target referenced by
another cycle member, allocate a synthetic `DelayDecl` whose update
reads the original `NestedOut` and whose init is `0`. Rewrite the
offending `NestedOut`s to read the synthetic delay.

This is the implementation of the guarded trace operator from the
ideological framing in the root `CLAUDE.md`. Every back-edge is
forced through the unit-delay endomorphism, which is what makes the
resulting graph causal: there's no instantaneous feedback path. Once
this pass has run, the resulting `ResolvedProgram` is acyclic in the
strict sense — register/delay reads don't loop back.

Synthetic delays carry `_liftedFrom: 'synthetic'` provenance so
downstream passes (especially `applyGateableWraps` in
`materialize_session.ts`) can distinguish them from
inlined-from-an-instance decls.

For most stdlib programs this pass is the identity — feedback is
already broken explicitly via `delay`. The plumbing exists because
post-`inlineInstances` cycles can appear that weren't visible
pre-inlining.

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
4. Lift cloned `RegDecl`s and `DelayDecl`s into the outer body,
   renamed `${instance.name}_${innerName}` and tagged
   `_liftedFrom: instance.name`. Lift cloned `next_update` assigns
   with their `target` rewritten. Lift `ProgramDecl`s and
   `ParamDecl`s as-is (no rename: ParamDecls are session-scoped by
   name; ProgramDecls are passive type bindings).
5. Record cloned `outputAssign` expressions in a substitution table
   keyed by the *template's* `OutputDecl` (matched by position to
   the cloned program's outputs). Replace every `NestedOut {
   instance, output }` ref in the outer's surviving expressions.

`_liftedFrom` is the post-strata replacement for the legacy
name-prefix parsing pattern. Identifying a decl's lineage is now an
object-field check, not a string regex.

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

### 3.6 Post-strata invariants

What you get from `strataPipeline`:

- **scalar-only** — no surviving array decls; arrays survive only as
  inline literals in expressions or as backing stores for stateful
  arrays
- **monomorphic** — no `TypeParamDecl`, no `TypeParamRef`
- **acyclic** — every cycle is broken by a `DelayDecl`
- **non-nested** — no `InstanceDecl`, no `NestedOut`
- **combinator-free** — no `let`, `fold`, `scan`, `generate`,
  `iterate`, `chain`, `map2`, `zipWith`
- **decl-identity-keyed** — refs hold decl objects, never strings

That's the smallest sub-IR sufficient for any per-sample evaluator.
The three backends below operate on this image.

---

## 4. Materialize: sessions into the same shape as per-program

**Input:** `SessionState` (`compiler/session.ts`).
**Output:** `ResolvedProgram` (top-level synthetic).
**Drops:** the difference between "a session of instances + wiring"
and "a single program".

`compiler/ir/materialize_session.ts` lifts a partially-typed graph of
`ProgramInstance`s plus session-keyed wiring `ExprNode`s plus
`dac.out` graph_outputs into a synthetic top-level `ResolvedProgram`
whose body has:

- one `InstanceDecl` per session instance, type pre-specialized (the
  instance type was resolved either out of `session.resolvedRegistry`
  for non-generics or via `specializeProgram` against
  `session.genericTemplatesResolved` for generics)
- session wiring expressions translated `ExprNode → ResolvedExpr`
  with shared identity preserved via `ctx.exprMemo`
- `dac.out` `graphOutputs` materialized as `OutputAssign`s on
  fresh `OutputDecl`s named `${instance}.${output}`
- `ParamDecl`s synthesized lazily as `paramRef` translations
  encounter them; each `(name, kind)` pair gets one decl, reused at
  every reference
- session-level `delay()` ExprNodes extracted into synthetic
  `DelayDecl`s named `__sd${i}`

Once that synthetic program exists, the rest of the pipeline applies
uniformly: it goes through `strataPipeline` and lands in the same
post-strata form a per-program build does.

### 4.1 Gateable two-phase wrap

A session instance can be marked `gateable` with a gate
`ExprNode`. The materializer wraps that instance's outputs and own
state in `select(gate, raw, fallback)` so external observers see
zero (and state stalls) when the gate is false.

The wrap happens in two phases because of how strata's
`nestedOut` substitution works:

- **Pre-strata**: clone the instance type, append a synthetic
  `__gate__` input, and wrap every `outputAssign.expr` and own
  `regDecl`/`delayDecl` update with `select(gateRef, raw, fallback)`.
  Strata's input-substitution will splice the actual gate expression
  in at inline time, and the wrapped form captures into wherever
  `instance.out` is referenced.
- **Post-strata**: walk every lifted reg/delay (identified by
  `_liftedFrom === instance.name`) and apply the same select-wrap
  that wasn't possible pre-strata because those decls didn't exist
  yet.

To route gate expressions through strata's `nestedOut` inlining, the
materializer also injects a per-gate synthetic `outputDecl` named
`__gate__${instance}` carrying the gate expression. Strata inlines
the gate's `nestedOut` refs alongside everything else; post-strata
the materializer reads back the inlined gate, strips the synthetic
output, and uses the inlined form for the lifted-decl wraps.

### 4.2 Param handle threading

`materializeSessionForEmit` returns the synthetic `ResolvedProgram`
plus a `Map<string, ParamDecl>` keyed by name. `compile_session.ts`
uses that map to build a `Map<ParamDecl, {ptr: string}>` from the
session's param/trigger registries — the FFI handles get bound to
decl identity, not to names. `compileResolved` consumes that map and
emits `param` operands with the right `.ptr` field.

The same materializer feeds the pure-TS interpreter
(`interpret_resolved.ts`), which doesn't care about handles — it
keeps a `Map<ParamDecl, number>` of current values instead.

---

## 5. Backends over the post-strata IR

The three sections below are not further compiler stages. They are
parallel interpretations of the same post-strata `ResolvedProgram`
into different targets.

### 5.1 compileResolved → tropical_plan_4 → JIT

**`compiler/ir/compile_resolved.ts`** is the per-program emit:

1. `buildSlotMaps(prog)` — assign integer slots to decl objects
   (`Map<RegDecl, number>`, etc.). Slot identity, not slot name, is
   what the JIT consumes.
2. `emitNumericProgram` (`compiler/ir/emit_resolved.ts`) — walk the
   lowered IR, emit a `FlatProgram` matching the `tropical_plan_4`
   schema in `compiler/flat_plan.ts`.
3. Wrap the result in a `FlatPlan` JSON object, returned as the
   compile output.

**Structural CSE.** `emit_resolved.ts` keys CSE on a bottom-up
structural id (interned via `${op}|${field=}|${child_id}` strings),
not node identity. This catches duplicates that strata's
clone-then-substitute introduces (e.g. when the same expression is
referenced from multiple instance inputs after inlining).

**Operand kinds.** `NOperand` discriminates: `const`, `input`, `reg`,
`array_reg`, `state_reg`, `param`, `rate`, `tick`. Terminals
(literals, sentinels, register reads) embed inline; non-terminal
expressions get a temp register.

**Array loops.** When `loop_count > 1` an instruction emits an
elementwise loop. `strides[i]` controls whether each argument
advances with the loop index (array stride = 1) or broadcasts
(stride = 0).

**Gateable groups.** When `sourceTag` wraps survive into the
post-strata IR (less common; the materializer's two-phase wrap
usually folds them into `select` ops earlier), the emitter ships a
`groups` array carrying gate metadata so the JIT can short-circuit.

### 5.2 The JIT path

The plan crosses the C API boundary as JSON. On the C++ side:

```
NumericProgramParser::parse_plan4()    ← engine/runtime/NumericProgramParser.hpp
  └─ thin JSON deserializer; reads instructions into a FlatProgram struct.
     No expression walking — the IR is already a flat instruction stream.
OrcJitEngine::compile_flat_program()   ← engine/jit/OrcJitEngine.{hpp,cpp}
  └─ singleton LLVM ORC.
     1. Build canonical cache key (MD5 of serialized program with
        param pointers replaced by ordinals).
     2. Check in-memory cache; then disk cache at
        ~/.cache/tropical/kernels/<build-id>/. Build-id derived from
        LC_UUID (macOS) / ELF build-id, so dylib rebuild auto-invalidates.
     3. Generate typed LLVM IR. Outer sample loop iterates
        `buffer_length`; per-instruction operand resolution emits
        f64/i64/i1 with explicit coercion; array loops when
        loop_count > 1; output[s] = sum(mix targets).
     4. LLJIT compile, look up symbol → NumericKernelFn.
FlatRuntime::load_plan()              ← engine/runtime/FlatRuntime.{hpp,cpp}
  └─ State init: stateInit values written into i64 backing store with
     type-aware bit-cast. Named state transfer: registers and arrays
     copied by name from outgoing kernel for click-free hot-swap.
     Atomic active-slot store-release publishes the new kernel.
FlatRuntime::process()                 (audio thread)
  └─ Acquire active state. Snapshot trigger params (atomic exchange).
     Call kernel (single invocation processes the buffer). Advance
     sample_index. Apply 2048-sample smoothstep fade envelope.
TropicalDAC::audio_callback            ← engine/dac/TropicalDAC.hpp
  └─ Templated RtAudio driver. Copies mono output to all channels.
     Watcher thread polls 50ms for device disconnect; recovers with
     500ms backoff + fade-in. switch_device() is explicit live-switch.
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
(`emit_resolved.ts`, `interpret_resolved.ts`, `emit_wasm.ts`).

### 5.3 interpret_resolved — the oracle

**`compiler/interpret_resolved.ts`**. Pure-TS evaluator over
`ResolvedExpr` against a state map keyed by decl identity. No FFI,
no kernel compilation. Reaches every backend that rests on the same
post-strata IR; a JIT bug shows up here as a cross-backend divergence.

This is the independent oracle for `tests/equiv/jit_vs_interp.test.ts`.

### 5.4 emit_wasm — the WebAssembly backend

**`compiler/emit_wasm.ts`** + **`compiler/wasm_memory_layout.ts`**.
A third interpretation of post-strata `ResolvedProgram`, reusing the
same `tropical_plan_4` boundary type. The emitter produces a
standalone WASM module exporting a single `process(buffer_length,
start_sample_index)` function plus a shared `memory`.

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

**WASM kernel structure** mirrors the LLVM kernel:

```
for s in 0..buffer_length:
  sample_idx = start_sample_index + s
  (run all instructions, writing to temps[])
  register writeback: temps[register_targets[i]] → registers[i]
  output[s] = sum(temps[output_targets[outputs[i]]]) / 20
```

**Encoding.** `i64` cells store either an f64 bitcast, a signed int,
or a zero-extended bool; the per-instruction `result_type` tells the
codegen which load/store to use. f64 cells back arrays and the
output buffer.

**Param flow.** Plan `param.ptr` strings hold SAB slot indices
instead of native pointers (`tropical_param_t*`). The kernel emits
`f64.load (paramTableOffset + ptr*8)` for `param` operands and
`f64.load (paramFrameOffset + ptr*8)` for trigger snapshots. The host
populates these regions per-block from a `SharedArrayBuffer` shared
with `web/host/params.ts:WebParam`.

The runtime side — instantiation, hot-swap, fade envelope, param
snapshotting — lives in `web/worklet/runtime.ts` and mirrors
`FlatRuntime` in shape. See `web/CLAUDE.md` for the runtime details.

### 5.5 Equivalence

The pipeline is correct only if every pass and every backend agrees
with the per-sample semantics on the input. Four test suites
cross-check that:

- `tests/equiv/jit_vs_interp.test.ts` — JIT and `interpret_resolved`
  agree sample-for-sample on the same post-strata IR.
- `tests/equiv/jit_vs_interp_stdlib.test.ts` — same, expanded to
  every viable stdlib program.
- `tests/equiv/wasm_vs_jit.test.ts` — WASM emit and JIT agree
  sample-for-sample.
- `tests/equiv/web_plans_vs_jit.test.ts` — every precompiled plan in
  `web/dist/patches/` matches the JIT output.
- `tests/equiv/migration_audio.test.ts` — new pipeline matches
  pre-active-set goldens byte-for-byte.

Any disagreement is a strata, materialize, or backend bug; the suite
localises which.

---

## 6. ProgramType and ProgramInstance

`compiler/program_types.ts`. Thin wrapper over a post-strata
`ResolvedProgram`. The wrapper is metadata; the IR is the value.

```typescript
class ProgramType {
  readonly prog: ResolvedProgram

  get name():    string
  get inputNames():  string[]
  get outputNames(): string[]
  get inputPortTypes():    (PortType | undefined)[]
  get outputPortTypes():   (PortType | undefined)[]
  get registerNames():     string[]                 // lazy via buildSlotMaps
  get registerPortTypes(): (PortType | undefined)[] // lazy via buildSlotMaps
  get rawInputDefaults():  Record<string, ExprNode> // lazy
  rename(newName: string): void                     // for cache-key rebranding
  instantiateAs(name, opts?): ProgramInstance
}
```

There is no `_def` field, no slot-indexed copy, no upfront
flattening. When you need port metadata, you walk `prog.ports`; when
you need register slots, the lazy cache runs `buildSlotMaps` once
and memoizes.

`ProgramInstance` holds a `ProgramType` plus an instance name,
`baseTypeName`, optional `typeArgs`, plus session-level `gateable` /
`gateInput` fields. Getter accesses delegate to `type`.

`session.ts:resolveProgramType` calls
`programTypeFromResolved(template, subst)` (full strata pipeline +
wrap), then `type.rename(key)` so the specialization cache key
(`Type<N=8>`) shows up as the program's name in serialized output.

---

## 7. Schema versions

Two distinct JSON schemas; do not confuse them.

| Schema | Produced by | Purpose |
|--------|-------------|---------|
| `tropical_program_2` | `compiler/program.ts`, `compiler/parse/raise.ts` | The high-detail input shape: program with typed ports, body block of decls/assigns, optional `type_params`. Authored by humans (in `.trop`) or by agents (over MCP). |
| `tropical_plan_4` | `compiler/ir/compile_resolved.ts` (schema in `compiler/flat_plan.ts`) | The low-detail output: flat instruction stream over typed scalar slots. C++ JIT and WASM emitter both consume this shape. |

Schema validation: `compiler/schema.ts` (Zod) for input;
`compiler/flat_plan.ts` (TypeScript types) for output.

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

### 7.2 tropical_plan_4 sketch

```json
{
  "schema": "tropical_plan_4",
  "config": { "sample_rate": 44100.0 },
  "state_init": [0.0, /* ... */],
  "register_names": ["sin1_phase", /* ... */],
  "register_types": ["float", /* ... */],
  "array_slot_names": [],
  "instructions": [
    { "tag": "Mul", "dst": 0, "args": [{ "kind": "const", "val": 6.283185307179586 }, { "kind": "tick" }],
      "loop_count": 1, "strides": [], "result_type": "float" },
    /* ... */
  ],
  "outputs": [/* indices into output_targets */],
  "output_targets": [/* temp slot indices */],
  "register_count": 1,
  "register_targets": [/* ... */],
  "array_slot_sizes": [],
  "array_slot_count": 0
}
```

The `tropical_plan_4` shape is the C-API contract. Anything the
backends need to know about the program — names, types, slot
counts, instruction stream, init state — is in there; everything the
compiler decided to forget along the way is gone.

---

## 8. C API (`engine/c_api/tropical_c.h`)

The stable C interface between TypeScript (via koffi FFI) and C++.
All handles are opaque `void*`. Errors are thread-local, fetched via
`tropical_last_error()`; valid until the next call on the same thread.

### 8.1 ControlParam

- `tropical_param_new(init_value, time_const)` — smoothed parameter
  with one-pole lowpass; `time_const` is τ in seconds (e.g. `0.005`
  for ~5 ms ramp; `0.0` for no smoothing)
- `tropical_param_new_trigger()` — fire-once trigger; per-frame
  read+clear via atomic exchange
- `tropical_param_set` / `tropical_param_get` — atomic store / load
- `tropical_param_free`

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
- `param.ts` — `Param` (smoothed) and `Trigger` (fire-once)
  wrapping `tropical_param_t`. Wiring references them by name; the
  materializer resolves the name to a `_h` handle and threads it
  into the plan via `paramHandles`.

---

## 10. MCP server (`mcp/server.ts`)

The primary agent interface. Runs on stdio, uses
`@modelcontextprotocol/sdk`. Maintains one long-lived `SessionState`.

22 tools, grouped by purpose. Every tool that mutates the signal
graph ultimately calls `wire()` → `applyFlatPlan(session, runtime)`,
which runs the full compile pipeline:

```
SessionState
  → compileSession (compiler/ir/compile_session.ts)
       → materializeSessionForEmit (compiler/ir/materialize_session.ts)
       → strataPipeline
       → compileResolved → tropical_plan_4 JSON
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
front-end, same strata pipeline, same `tropical_plan_4` boundary;
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
    params.ts         ParamBank (SharedArrayBuffer), WebParam, WebTrigger
  worklet/            Audio thread
    runtime.ts        WasmRuntime: dual-slot hot-swap, fade envelope, snapshotParams
    processor.ts      AudioWorkletProcessor delegate; postMessage protocol
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

### CI (`.github/workflows/ci.yml`)

Two jobs: typecheck (`bunx tsc --noEmit`) and build-and-test (LLVM
20, libasound2-dev, `bun test` + `ctest`).

---

## 14. Testing

### 14.1 C++ tests (`engine/tests/test_module_process.cpp`)

Custom harness, no framework dependency. Exercises FlatRuntime C
API and JIT without an audio device. Tests build `tropical_plan_4`
JSON strings directly and assert on output buffer values. Covers
sawtooth, clock with array ratios, integer sequences, multi-instance
fusion, smoothed params, trigger params, hot-swap state transfer,
typed int/bool ops.

`cmake --build build -j4 && ctest --test-dir build`.

### 14.2 TS tests (`compiler/**/*.test.ts`, `tests/equiv/*.test.ts`)

Run via `bun test`. The load-bearing suites:

- `tests/equiv/jit_vs_interp.test.ts` — JIT vs. `interpret_resolved`
- `tests/equiv/wasm_vs_jit.test.ts`,
  `tests/equiv/web_plans_vs_jit.test.ts` — WASM emission equivalence
- `tests/equiv/migration_audio.test.ts` — new pipeline vs.
  pre-active-set goldens
- `ir/*.test.ts` — strata pipeline unit tests (specialize,
  sum_lower, trace_cycles, inline_instances, array_lower, slots,
  clone)
- `parse/*.test.ts` — lexer, parser, raise, round-trip,
  `stdlib_round_trip.test.ts` (every `.trop` print/re-parse)
- `apply_plan.test.ts` — plan application integration (requires
  `make build`)

### 14.3 Stdlib audit

`bun run scripts/validate_stdlib.ts` parses, elaborates, and lowers
every `stdlib/*.trop` and confirms post-strata invariants. Run by
`make validate`.

---

## 15. Key design decisions

### Single-kernel fusion

The whole patch compiles to one native function. No instance
boundaries at runtime, no per-instance dispatch, no interpreter on
the audio thread. LLVM gets to optimize across the full graph.

### Pipeline of structure-dropping passes

We resisted "layers" as a framing. The compiler is a chain of IR-to-IR
passes where each pass drops a specific kind of structure once it's
been consumed. That shape is what makes the layout of `compiler/ir/`
predictable, makes it clear where new passes belong, and makes
sample-for-sample cross-backend agreement the right correctness
criterion. The shape also matches a more theoretical reading
(cartesian category of typed signal-flow graphs with a guarded
trace) sketched in the root `CLAUDE.md`, but you can program against
the pipeline without that vocabulary.

### Decl identity instead of strings

Past `elaborate`, every reference is a TypeScript object pointer.
String-based scope walks would make later passes re-derive
information the elaborator already established. `_liftedFrom`
replaced what had been name-prefix string regex; this is the same
move, applied to a different kind of provenance.

### Hot-swap via double-buffered kernels

Live audio never stops for recompilation. The new kernel is built on
a background thread; named state transfers; a single atomic store
publishes it. At most one sample of stale state is read. Same shape
on both backends (FlatRuntime in C++, WasmRuntime in WASM).

### No interpreter fallback on the audio path

JIT failures are fatal at the runtime. Compile errors are caught
upstream (in `applyFlatPlan` / over MCP) and never reach the
runtime. `interpret_resolved` exists, but it's a test oracle, not a
fallback — it's what guarantees the JIT and the WASM emit aren't
silently disagreeing on the same IR.

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
Smoothed params apply one-pole lowpass per sample. Triggers fire
once via atomic exchange. Safe from any thread; the audio thread
never blocks.
