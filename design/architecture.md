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
of the chain, the IR is `tropical_plan_5`: a root instruction stream
(instances nested as `children`) over typed scalar slots plus output sinks
per-sample. Two backends interpret that low-detail IR — the JIT and
WebAssembly — and equivalence test suites cross-check them
sample-for-sample.

tropical is **closed-form-only**: every kernel is a pure function of a
time coordinate, `f(τ, params)`, with no per-sample state. The semantic
universe is the exp-poly ring — polynomials, exponentials, sinusoids of
the coordinate, and their products — so oscillators, FM/PM, additive,
modal resonance, and envelopes are all closed forms of τ rather than
accumulating state. See `design/cf-only.md` for the contract.

## What tropical is, categorically

A single sentence to hang the system off:

> tropical's IR is a DAG-shaped operad — programs are typed
> signal-flow graphs, acyclic by construction, and the compiler is a
> functor between this operad and the slot-operational operad consumed
> by the runtime.

The vocabulary doesn't have to be load-bearing day to day, but the
shape of the codebase matches it precisely:

- **Colors** = post-strata port types (`float`, `int`, `bool`,
  fixed-shape arrays of these).
- **Operations** = primitive ops (`add`, `mul`, `clamp`, …) plus
  user-defined programs.
- **Composition** = wiring an output to an input.
- **Tensor (parallel composition)** = placing instances side by side
  in a `body.decls` list.
- **Encapsulation** = an instance's body can only see its own input
  ports; cross-program communication happens at the parent's level
  by writing into a child's input slot. This is structural, not a
  discipline: the IR types make boundary-crossing references
  inexpressible.

**The IR is acyclic by construction — and there is no escape hatch.**
Cycles are rejected outright; nothing in the system breaks a cycle for
you. Source-level cycles throw `CycleViolation` at the elaborator, with
port-detailed Tier-2 errors. Session-level cycles in MCP-built graphs are
rejected too: `assertSessionAcyclic` (`lean/Tropical/Lowering.lean`) is a
plain "no cycles at all" rule at session-compile entry — there is no
per-wire unit delay, no delay-slot registry, and no register writeback to
absorb a back-edge. Feedback that needs genuine state (an IIR pole on live
or external input) is the ceded island: it belongs to a future stateful
sister runtime (`supertropical`), not here. See `design/cf-only.md`.

For the longer categorical story — the pre-trace/post-trace operad split,
multi-realization (WDF, iterative fixed-point, FFI), and the
continuous → discrete → finite-precision semantic hierarchy — see the
archived design conversation at `design/archive/operadic_ir.md`. That
material predates the closed-form-only commitment: it explores
cycle-breaking trace functors that the current language no longer offers
(cycles are simply rejected). Read it as the framing that produced the
acyclic-DAG shape, not as a description of current cycle handling.

## Pipeline at a glance

```
literate .md source / tropical_program_2 / MCP edits
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
    │   │       │ liftWiresToInstances → assertSessionAcyclic
    │   │       │     → compileSessionSlotted
    │   │       │     (buildSessionRoot → partitionKernel →
    │   │       │      instance_functions[] (root, nested children)
    │   │       │      + sinks[] (outputs) + sources[] (inputs))
    │   │       │
    │   │       │   ──── C API boundary (engine/c_api/tropical_c.h, Lean @[extern] FFI) ────
    │   │       ▼
    │   │   NumericProgramParser → FlatProgram (multi-function)
    │   │   OrcJitEngine → LLVM IR (one kernel: instances, then sinks)
    │   │   FlatRuntime → buffer loop, double-buffered hot-swap
    │   │   TropicalDAC (RtAudio) → audio output
    │   │
    │   └─→ compile-wasm (engine LLVM + lld, in-process) → .wasm + manifest
    │           │
    │           ▼
    │       WasmKernel in AudioWorklet (web/runtime/)
```

Both backends consume the same `tropical_plan_5`: by default the
session is serialized to a `ParsedProgram` and elaborated into a single
synthetic root program whose
`InstanceFunction` carries the session instances as nested `children`,
and the kernel is one LLVM function (or, in WASM, one `process` export)
whose body dispatches each instance per sample, then writes the output
sinks. The equivalence tests pin the JIT and WASM outputs
sample-for-sample to each other.

Every pass below is presented as input IR, output IR, and the
structure dropped between them.

---

## 1. Surface and parse

**Input:** literate `.md` source text or `tropical_program_2` JSON.
**Output:** `ParsedProgram` (`lean/Tropical/Parse/Nodes.lean`).
**Drops:** layout, comments, surface sugar.

### 1.1 The two front-ends

Literate `.md` source is the human-authored surface form used by the
stdlib and by hand-written patches. The surface parser
(`lean/Tropical/Parse/Surface/*`) produces a `ParsedProgram` in layers:

- `Surface/Lexer.lean` (+ `Surface/Cursor.lean`) — token stream
- `Surface/Expr.lean` — infix precedence, unary, calls,
  let/combinator lambdas (binders introduced on the fly via
  `BinderDecl`)
- `Surface/Statements.lean` — block bodies (decls + assigns)
- `Surface/Declarations.lean` — program signatures, port specs,
  type-param declarations, nested programs
- `Surface/Markdown.lean` — the literate-`.md` envelope around them

The legacy/MCP entry is JSON. `lean/Tropical/Parse/Raise.lean` walks a
`tropical_program_2` JSON object and emits the same `ParsedProgram`
shape. Both front-ends converge on the same input to the elaborator.

### 1.2 Parse-time desugaring: bounds

`lean/Tropical/Parse/Surface/Bounds.lean` runs immediately after parse,
before anything else sees the program. It rewrites:

- `signal[-1, 1]`, `unipolar[0, 1]`, `bipolar[-1, 1]`, `phase[0, 1]`,
  `freq[0, ∞)` — named bound aliases — to explicit `clamp` or
  `select` ops
- explicit `in [lo, hi]` annotations on input defaults — same
  treatment

Bounds are a *parse-time* concept. They never reach the elaborator
or the strata pipeline as a distinct construct.

### 1.3 What this pass deliberately doesn't do

`Raise.lean` and the surface parser perform **zero scope analysis**.
Every reference (`input("freq")`, `sin1.out`, `param("cutoff")`) emits a
`NameRefNode` placeholder. The parser doesn't know which declarations are
in scope, doesn't validate that the name exists, doesn't disambiguate
between (say) an instance output and a parameter read. Name resolution
lives in exactly one pass: the elaborator. This is the cleanest
separation we've found between syntactic and semantic concerns.

---

## 2. Elaborate

**Input:** `ParsedProgram`.
**Output:** `ResolvedProgram` (`lean/Tropical/Ir/Nodes.lean`).
**Drops:** names. Enforces the **acyclic-source** invariant.
**Keeps:** every other piece of surface structure (type params, sum
types, nesting, combinators, shapes).

`lean/Tropical/Ir/Elaborator.lean` is the unique site of name
resolution. A single top-down pass over the parsed program: each
declaration is constructed once when its parsed counterpart is
encountered, registered in the appropriate scope, and re-used by
reference at every site that names it.

The output is a graph IR. Decls (`InputDecl`, `OutputDecl`,
`ParamDecl`, `TypeParamDecl`, `InstanceDecl`, `ProgramDecl`,
`BinderDecl`, plus sum/struct/alias type defs) are introduction sites.
Refs (`InputRef`, `ParamRef`, `TypeParamRef`, `BindingRef`,
`NestedOut`) are uses. Every ref carries its decl as a branded integer
`idx` into a typed decl table — positional identity, not a string lookup
(the Lean image of the TS elaborator's `===` pointer identity). There is
no state-register decl: tropical is closed-form-only, so there is no
surface `reg`/`next` and no `delay` sugar to elaborate — the parser has
no production for them and rejects them outright (see `design/cf-only.md`).

The elaborator also runs **strict cycle detection** at the source
level. Inter-instance cycles throw `CycleViolation` here, with a
port-detailed Tier-2 error message — there is no explicit-register
escape hatch that would make a cycle legal; cycles are simply rejected.
Acyclic-by-source is what makes everything downstream straight-
line: the strata pipeline never needs to break cycles, and
`assertAcyclic` at strata entry catches any caller that bypassed
the contract.

After this pass, no string lookups, no scope walks, no shadowing
disambiguation. Decl identity is the only substrate the rest of the
compiler operates on.

---

## 3. Strata pipeline

`lean/Tropical/Ir/Strata.lean` orchestrates six passes:

```
strataPipeline prog typeArgs options =
  assertAcyclic prog
  let specialized = specializeProgram prog typeArgs
  let summed      = sumLower specialized
  let inlined     = if options.inlineNested == false then summed
                                                      else inlineInstances summed
  let arrayed     = arrayLower inlined
  identityElim arrayed
```

Each pass is a total function: returns a fresh `ResolvedProgram`, or —
in the no-op fast path — the input unchanged. None of them mutate the
input. **Acyclicity is the strataPipeline contract**; the callers
guarantee it, `assertAcyclic` confirms it.

### 3.1 assertAcyclic — boundary check

`lean/Tropical/Ir/Strata/Basic.lean`. Re-uses the SCC finder shared
with the session-level acyclicity check (the shared cycle algorithm).
Throws `AcyclicityViolation` if any non-trivial SCC survives into strata
input. In the standard path this never fires; the elaborator and the
session compiler have already rejected any cyclic graph.

### 3.2 specialize — drops type parameters

`lean/Tropical/Ir/Strata/Specialize.lean`. Takes a generic program and
a map keyed on `TypeParamDecl`; produces a fresh program with the
integers substituted in.

Substitution sites:
- `ShapeDim`s that are `TypeParamDecl`s become integers
- expression-position `TypeParamRef` nodes become numeric literals
  (with `ConstNode` wrapping when type information is needed)
- the root program's `typeParams` list is dropped

Each `(template, args)` pair produces a structurally fresh program;
a generic `Voice<N=8>` and `Voice<N=44100>` give distinct programs with
inline-array shapes `[8]` and `[44100]`. Sum/struct/alias type defs are
shared across the clone (preserves variant identity for match arms,
which `sum_lower` requires).

The session-emit cache (`session.specializationCache`) memoizes on
`(template, args)` keys. The pass itself doesn't consult the cache —
that's the loader's job.

### 3.3 sumLower — drops sum types

`lean/Tropical/Ir/Strata/SumLower.lean`. Lowers `MatchExpr` to scalar
select-chains and `TagExpr` to tag-literal writes, collapsing every
sum-typed value to a discriminator-plus-fields scalar bundle.

Constraint:
- match-arm payload bindings have no slot source to bind from —
  CF-only removed sum-typed registers (the only construct that could
  supply payload slots), so a match arm that binds a payload is a
  structural error.

After this pass: no `tag` op, no `match` op, no sum-typed value.
Decl identity is preserved end-to-end; replacements are by identity.

Sum types survive in the surface language (and in struct/alias type
defs) but, with no per-sample state, they appear only as transient
expression-level values that this pass folds into select-chains — there
is no longer a sum-typed *register* to decompose.

### 3.4 inlineInstances — drops nesting

`lean/Tropical/Ir/Strata/InlineInstances.lean`. Splices each
`InstanceDecl` into its parent. After this pass:

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
4. Lift `ProgramDecl`s and `ParamDecl`s as-is (no rename:
   `ParamDecl`s are session-scoped by name; `ProgramDecl`s are passive
   type bindings). There are no state registers to lift — the IR is
   stateless — so nothing is renamed or provenance-tagged.
5. Record cloned `outputAssign` expressions in a substitution table
   keyed by the *template's* `OutputDecl` (matched by position to
   the cloned program's outputs). Replace every `NestedOut {
   instance, output }` ref in the outer's surviving expressions.

The `inlineNested: false` option is the entry point for the
fractal-compilation path: sub-instances survive as kernel boundaries
and `partitionKernel` (the recursive partitioner in
`lean/Tropical/Compile.lean`) emits an `InstanceFunction` for every
`InstanceDecl` at every level. Cross-kernel input wiring is handled
by slot-based parent→child writes: the parent's body emits
`WriteSlot` ops into a per-child `pre_child_instructions` block
(evaluated in the parent's scope, so every ref resolves), and the
child reads its inputs via slot reads in its own body. Both paths
(`inlineNested: true` flat, `inlineNested: false` fractal) are
verified sample-for-sample at 1e-12 by the native mode-equivalence
check in `tropicaltest`; the choice between them is a runtime-cost
tradeoff (the ~25% flat-path win comes from IR-level slot removal,
not from anything LLVM could do — slots are observable hot-swap
state).

### 3.5 arrayLower — drops shapes and combinators

`lean/Tropical/Ir/Strata/ArrayLower.lean`. Unrolls compile-time
combinators and lowers array ops to scalar primitives via static shape
information.

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
- `arraySet(arr, i, v)` — left as-is; an immutable functional update
  on an inline array (the array itself is a value, not a backing store)

Substitution discipline: every `BindingRef.decl` is a pointer to a
`BinderDecl`. Substitution is by `Map<BinderDecl, ResolvedExpr>`,
which makes shadowing structurally impossible — shadowing would
require two different `BinderDecl`s with the same name, and that's
exactly what decl identity catches.

Each combinator iteration uses a fresh `WeakMap` memo. A memo is
valid only for one substitution map; reusing one across iterations
would conflate different `acc`/`elem` values.

### 3.6 identityElim — categorical identity-law rewrite

`lean/Tropical/Ir/Strata/IdentityElim.lean`. Eliminates `InstanceDecl`s whose
program body is the identity morphism (each output assigns the
corresponding input untouched). In the per-program path
`inlineInstances` already absorbs trivial sub-instances first, so
this pass rarely fires today; it carries weight in the session-
level path where lifted wire-programs sometimes lower to a
no-op kernel that the pass collapses cleanly.

### 3.7 Post-strata invariants

What you get from `strataPipeline`:

- **scalar-only** — no surviving array decls; arrays survive only as
  inline literals in expressions
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

### 4.1 JIT path — `compileSession`

`lean/Tropical/Compile.lean` (with the slot/lift machinery in
`lean/Tropical/Lowering.lean` and `lean/Tropical/Ir/WireProgram.lean`)
runs two pre-emit passes and hands the result to the slotted
session compiler:

1. **`liftWiresToInstances`** (`lean/Tropical/Ir/WireProgram.lean`) —
   anonymous-instance lift. Wires whose expressions contain array
   literals are extracted into anonymous `__wire_${i}` instances at
   session pre-compile time. The lifted programs go through the
   full strata pipeline so combinators lower correctly.

2. **`assertSessionAcyclic`** (`lean/Tropical/Lowering.lean`) —
   the session-acyclicity rule (Tarjan over the instance dep graph). It
   is a plain "no cycles at all" check: tropical is closed-form-only, so
   nothing breaks a session cycle for you — a cross-coupled MCP graph or
   hand-written JSON patch is rejected here, not silently delayed.

Then the slotted session compiler runs the root-program lowering:
`buildSessionRoot` serializes the whole session back to a `ParsedProgram`
and runs it through the SAME `elaborate` front
door the surface path uses, yielding one synthetic root `ResolvedProgram`:
instances become `InstanceDecl`s (their already-resolved types supplied
via the elaborator's `ExternalProgramResolver` hook — LINK, not
re-elaboration). That root is lowered through the *same* `partitionKernel`
the per-program fractal path uses (`ROOT_INSTANCE_PATH` naming
transparency, so children keep bare names). The result is a single
`InstanceFunction` (the root) whose `children` are the session instances.
Outputs are `sinks[]` (`emitSinks` reads each graphOutput slot directly);
slot-based session params are threaded as `paramSlots`. There are no
per-wire delays and no register writebacks — every instance is a pure
function of the time coordinate, so a wire is just a slot read with no
latency.

(The former per-instance lowering — N standalone `compileResolved` plans
stitched by a `SchedulerFunction` — existed only as the `root_vs_flat`
differential oracle. Once it had validated the root path it was retired,
taking the whole scheduler tier with it.)

The resulting `tropical_plan_5` `FlatPlan` sequences the per-sample work as
(the dispatch recurses into nested `children`):

```
for each sample:
  for each instance_function (recursively: preamble,
    per-child {pre_input, child}, body)
  for each sink: output[target] = gain · Σ slots[sink.inputs]
```

Branded indices throughout (`TempIdx`, `ArraySlotIdx`, `ModuleSlotIdx`)
make cross-namespace arithmetic a compile error — the literal shape of a
Phaser-era slot-mixing bug. See `lean/Tropical/Plan.lean` (and the
mirrored `web/wasm/slot_indices.ts` on the WASM path).

### 4.2 Fixed-topology compilation

Tropical compiles a session graph to native code with a fixed
topology for the lifetime of the kernel. Two compilation modes
selected by `FlatPlan.compilation_mode`:

- `'fused'` (default) — one monolithic LLVM function for the whole
  session per buffer. LLVM fuses across instances aggressively.
- `'microkernel'` — one LLVM function per top-level session instance,
  plus a `postamble_mix` (the sink mix); the per-sample outer loop runs
  in C++ and dispatches them via function pointers.
  Trades cross-instance fusion for superlinear cold-compile speedup
  (LLVM optimizer scales worse on one huge function than on N
  smaller ones). Sample-for-sample equivalent to fused at 1e-12 per
  the native mode-equivalence check in `tropicaltest`.

Topology changes (adding/removing instances, rewiring) trigger
hot-swap to a freshly compiled kernel; both modes use the same hot-swap
mechanism. There is no per-sample *state* to carry across the swap —
kernels are pure functions of the time coordinate, so only the sample
index continues; the fresh kernel zero-inits and the control plane
re-asserts param values against the live kernel via `set_slot`. There is
no per-instance runtime gating — every instance runs every sample.
This is the shape of synthesis the language is good at;
dynamic-lifecycle semantics belong in a different language with a
different runtime.

---

## 5. Backends over the post-strata IR

The three sections below are not further compiler stages. They are
parallel interpretations of the same post-strata `ResolvedProgram`
into different targets. Wiring expressions reference parameters by
name (`{op:'param', name}`). The **session** path resolves each to a
`param:name` **module slot** read — the control plane drives it via
`setSlot`, hot-swap transfers it by name (the root path threads it as
`paramSlots` so the root kernel's `ParamRef` lowers to the slot). The
standalone **per-program** path instead binds params to FFI handles:
a native pointer (`tropical_param_t`) on the JIT, a SAB slot index
(stringified to keep `tropical_plan_5` backend-agnostic) on the WASM
path.

### 5.1 compileResolved → tropical_plan_5 → JIT

**`lean/Tropical/Ir/CompileResolved.lean`** is the per-instance emit:

1. `buildSlotMaps(prog)` — assign integer slots to decls (keyed on
   decl idx). Slot identity, not slot name, is what the JIT
   consumes. (Slot allocation lives in `lean/Tropical/Compile.lean`.)
2. `emitNumericProgram` (`lean/Tropical/Ir/Emit.lean`) — walk the
   lowered Core IR, emit a `PerInstancePlan` whose instructions follow
   the `tropical_plan_5` schema in `lean/Tropical/Plan.lean`.

The slotted session compiler (`lean/Tropical/Compile.lean`) packs
`PerInstancePlan`s into `instance_functions[]` of a `FlatPlan`,
shifting indices by the per-instance offsets, and builds `sinks[]`
(`emitSinks`) from the session's graphOutputs. The result is a single
root `InstanceFunction` (the session instances nested as its
`children`); there are no session-level delays and no register
writebacks, so there is no scheduler tier.

**Structural CSE.** `Emit.lean` keys CSE on a bottom-up structural id
(arena `ExprId`s replicating the TS identity semantics), not node
identity. This catches duplicates that strata's clone-then-substitute
introduces.

**Operand kinds.** `NOperand` discriminates: `const`, `input`, `reg`,
`array_reg`, `session_array_reg`, `param`, `source`, `slot`. The `reg`
kind is the **SSA temp pool** — a computational intermediate within a
single sample, *not* per-sample state (CF-only has no state operand; the
shared name with the deleted state register is the codegen's one
unfortunate overload). The `source` kind carries an index into
`plan.sources[]` — the runtime-bound input family (canonical:
`[tick, rate]`); the engine switches on `sources[i].kind` to the
appropriate kernel arg. Terminals (literals, source reads) embed inline;
non-terminal expressions get a temp (a `reg` slot). (The former `tick`
and `rate` operand kinds survive only on the wire as plan_4 legacy
aliases; parser upgrades them to `source:0`/`source:1`.)

**Array loops.** When `loop_count > 1` an instruction emits an
elementwise loop. `strides[i]` controls whether each argument
advances with the loop index (array stride = 1) or broadcasts
(stride = 0).

### 5.2 The JIT path

The plan crosses the C API boundary as JSON. On the C++ side:

```
NumericProgramParser::parse_plan5()    ← engine/runtime/NumericProgramParser.hpp
  └─ thin JSON deserializer; reads the instance functions plus
     sinks[] into a FlatProgram (multi-function) struct. A
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
            for each instance_function (recursively: preamble,
              per-child {pre_input, child}, body)
            for each sink: output[target] = gain · Σ slots[inputs]
        Per-instruction operand resolution emits f64/i64/i1 with
        explicit coercion; array loops when loop_count > 1.
     4. LLJIT compile, look up symbol → NumericKernelFn.
FlatRuntime::load_plan()              ← engine/runtime/FlatRuntime.{hpp,cpp}
  └─ Build the fresh KernelState (SSA-temp scratch sized to
     register_count, module slots from defaults). Hot-swap carries
     no per-sample state across the flip — CF-only kernels are
     stateless, so `publish_state` continues only the sample index;
     the control plane re-asserts param values against the new kernel.
     Atomic active-slot store-release publishes it.
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
`log`, `pow` are stdlib programs (polynomial approximations
using arithmetic + `Ldexp` + `FloatExponent` — single-instruction
IEEE-754 bit ops for 2^n range reduction). They inline at strata
time. The kernel contains no libm calls and is deterministic across
platforms. Swap `stdlib/Sin.md` to change the approximation.

**Adding an op.** Codegen is Lean's `EmitLlvm` (textual LLVM IR); the
native ORC JIT and the browser's wasm32 module both consume that *same*
IR. So a new op lands in Lean only:
1. The wire-format op set (`lean/Tropical/Expr.lean`).
2. The strata passes that traverse it.
3. The IR emission case in `lean/Tropical/Ir/EmitLlvm.lean`.
The engine receives IR text and JITs it; `diffcli compile-wasm` lowers
the same IR to wasm32 in-process. No C++ change and no per-backend
emitter — the C++ plan compiler was retired in Phase 2.

### 5.3 The WebAssembly backend

The browser is a **precompiled-patch player**, not a second codegen.
There is one codegen (`EmitLlvm`); the wasm path lowers that *same* IR
to wasm32 **in-process** via the engine's LLVM + lld
(`tropical_jit::compile_ir_to_wasm`, gated by `TROPICAL_WASM_EMIT`,
exposed as `diffcli compile-wasm`). No subprocess, no `wasm-ld` on PATH,
no version skew — the wasm comes out of the exact LLVM the JIT uses, so
native≡wasm bit-exactness is structural (the IR carries no
fast-math/contract flags, so neither target forms an FMA).

Per patch the build ships two artifacts the browser fetches:
- `<slug>.wasm` — exports `tropical_kernel` (the 11-argument per-block
  kernel) + `__heap_base`; imports `env.memory` + `env.round`.
- `<slug>.manifest.json` — a `KernelManifest`: the subset of
  `tropical_plan_5` the runtime needs (sampleRate, SSA-temp/array/slot
  region sizing, slot defaults — there is no per-sample state to seed).
  This type is the producer↔consumer contract.

The browser runtime (`web/runtime/`, an extractable package depending
only on `KernelManifest`) owns one imported linear memory, places the
kernel's pointer-argument regions above `__heap_base`, sizes the SSA-temp
scratch region and seeds the module-slot / param defaults, and calls the
kernel per 128-sample block. No live recompile, no hot-swap, no
`SharedArrayBuffer` — those belong to the native instrument. A smoothstep
fade is the only envelope.

See `web/CLAUDE.md` for the memory layout, the worklet protocol, and the
contract.

### 5.5 Equivalence

The pipeline is correct only if every backend and every realization
variant agrees with the per-sample semantics on the input. The
correctness floor is the frozen audio **goldens** (re-baselined only
when a heard, confirmed change to the math demands it) plus the
developer's ear; the cross-checks anchor agreement around that floor:

- `tests/web/wasm_vs_jit.test.ts` — the wasm module (`diffcli
  compile-wasm` → `WasmKernel`) and the JIT (`diffcli render-bytes`) agree
  sample-for-sample. Same IR, two LLVM targets, so this guards FP-target
  divergence.
- `tests/web/web_plans_vs_jit.test.ts` — every shipped
  `web/dist/patches/<slug>.wasm` + manifest matches the JIT output.
- **native mode-equivalence** in `tropicaltest` (`lake exe
  tropicaltest`) — fused vs. microkernel (and the deep-nesting variant)
  at 1e-12, exercising the kernel/slot layer the two backends share.
- **audio goldens** in `tropicaltest` — byte-for-byte against the
  frozen `tests/golden/*.hash` reference output.

Any disagreement in a cross-check is a strata or backend bug, and the
runner localises which. Note what is *not* here: there is no
differential against a second compiler implementation. The TS oracle
that gated the Lean port was migration scaffolding — a differential
proves *agreement*, not correctness — and it was retired with the port
(see `design/lean_port.md`).

---

## 6. ProgramType and Compiled

A thin wrapper over a post-strata `ResolvedProgram` (the
`Compiled`/`ProgMeta` shape in `lean/Tropical/Session.lean`). The
wrapper is metadata; the IR is the value. Helpers (`inputNames`,
`outputNames`, …) read off the resolved IR; slot-derived fields are
computed via `buildSlotMaps`. (There is no `registerNames` helper —
the IR is stateless, so there are no state registers to name.)

A session `Instance` holds a `Compiled` plus an instance name,
`baseTypeName`, and optional `typeArgs`.

`resolveProgramType` (`lean/Tropical/Session.lean`, with the
specialize path in `lean/Tropical/TypeArgs.lean`) runs the full strata
pipeline + wrap, then renames so the specialization cache key
(`Type<N=8>`) shows up as the program's name in serialized output. The
engine caches the result per `Type<N=8>` key so a hit skips re-running
the pipeline.

---

## 7. Schema versions

Two distinct JSON schemas; do not confuse them.

| Schema | Produced by | Purpose |
|--------|-------------|---------|
| `tropical_program_2` | `lean/Tropical/Parse/Raise.lean` (and save/export reconstruction) | The high-detail input shape: a program with typed ports, a body block of decls/assigns, optionally generic in `type_params`. Authored by humans (in literate `.md`) or by agents (over MCP). |
| `tropical_plan_5`    | `lean/Tropical/Compile.lean` (`lean/Tropical/Plan.lean` schema) | The low-detail output: a root instruction stream (`instance_functions[]`, instances nested as `children`) plus `sinks[]` (device-bound outputs) and `sources[]` (runtime-bound inputs — canonical `[tick, rate]`). The engine consumes it as the codegen manifest (the kernel itself comes from Lean's IR); the web build derives a `.wasm` + a trimmed `KernelManifest` from it. The engine still accepts the older `tropical_plan_4` (single-kernel, top-level temp-mix) for hand-crafted unit tests; it's lifted into a one-instance plan_5 with the canonical sources at parse time. |

Schema validation is the Lean type layer: the typed parsed/plan ASTs
in `lean/Tropical/Parse/Nodes.lean` and `lean/Tropical/Plan.lean`, with
order-preserving JSON codecs (`Tropical/Parse/OrderedJson.lean`,
`Plan.lean`'s `toWirePlan` encoder). The web side consumes a trimmed
`KernelManifest` (`web/runtime/manifest.ts`), not a full plan mirror.

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
    { name, instance_name, register_count,  // register_count sizes the
                                            // SSA-temp scratch (not state)
      register_offset, array_slot_offset, instructions, children, ... },
    ...
  ],
  sinks: [
    { inputs: number[],  // output module-slot indices to sum
      gain:   number,    // output scale (was the engine's hardcoded ÷20)
      target: number },  // output channel/device (0 = default audio out)
    ...
  ],
  sources: [             // runtime-bound inputs — the dual of sinks.
    { kind: "tick" },    //   index 0: current sample index (i64)
    { kind: "rate" }     //   index 1: current sample rate (f64)
  ]
  // (omitted on the wire when equal to the canonical [tick, rate] pair;
  //  sink-less plan_4 / single-kernel fixtures instead carry top-level
  //  `output_targets`/`outputs` for the legacy temp-mix ÷20.)
}
```

The `tropical_plan_5` shape is the C-API contract. Anything the
backends need to know about the program — instance kernels, the output
sinks, the input sources, names, types, slot counts, slot/param
defaults — is in there; everything
the compiler decided to forget along the way is gone.

---

## 8. C API (`engine/c_api/tropical_c.h`)

The stable C interface between the Lean engine (via `@[extern]` FFI
over `lean/ffi/shim.c`) and C++. All handles are opaque `void*`. Errors
are thread-local, fetched via `tropical_last_error()`; valid until the
next call on the same thread.

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

## 9. FFI bridge (`lean/Tropical/Ffi.lean` + `lean/ffi/shim.c`)

Lean `@[extern]` bindings over the C API. `shim.c` (compiled against
`engine/c_api/tropical_c.h`) wraps each C entry point; `Ffi.lean`
exposes them as Lean externals returning external objects whose
finalizers call the matching `tropical_*_free`. Loads
`libtropical.dylib` from `build/`. This replaced the former
koffi-based TypeScript bridge.

- `Runtime` — wraps `tropical_runtime_t`; the DAC holds its Runtime
  alive as a Lean field so the audio thread never reads a freed handle.
- `DAC` — wraps `tropical_dac_t`, plus device enumeration.
- `Param` — wraps `tropical_param_t`. On the session path, wiring
  references parameters by name and the kernel reads `param:name`
  module slots driven via `setSlot` — no native pointer threads into
  the plan; the standalone per-program path is where the native
  `tropical_param_t` handle is used.

---

## 10. MCP server (`lean/` — one binary)

The primary agent interface. The MCP server is the native **Lean
frontend** (`lean/Main.lean`, built on the
[Turnstile](https://github.com/willishoke/turnstile) framework): it
validates each tool call against a typed schema and handles it in
process. There is no relay and no subprocess — the same binary that
serves MCP owns the long-lived `SessionState` (`lean/Tropical/
Engine.lean` + `Session.lean`), compiles plans, and drives the runtime
over FFI. One server, the full MCP surface — 22 tools plus resources
(program catalog, program-format doc, `lean/Tropical/Resources.lean`)
and the build-patch prompt. (The earlier `@modelcontextprotocol/sdk`
stdio server and, after it, the Turnstile-front-door-over-TS-relay
arrangement were both retired during the Lean port. The `feedback`
tool — which used to wrap a back-edge in a unit delay — is gone with
the rest of the per-sample-state machinery; cycles are now rejected.)

Every tool that mutates the signal graph ultimately runs `syncCompile`
(`lean/Tropical/Engine.lean`), which builds and loads its own plan:

```
SessionState
  → compileSession (lean/Tropical/Compile.lean)
       → liftWiresToInstances → assertSessionAcyclic
         → slotted session compile
       → compileResolved/emit → tropical_plan_5 JSON
  → runtime.loadPlan over FFI
       (NumericProgramParser → OrcJitEngine → FlatRuntime hot-swap)
```

Compile errors don't kill the session: they return a structured error
envelope (`Except`-shaped, mapped onto the `mcp/ERRORS.md` taxonomy)
and the previous kernel keeps playing.

See [`mcp/ERRORS.md`](../mcp/ERRORS.md) for the error envelope
taxonomy and [`mcp/CLAUDE.md`](../mcp/CLAUDE.md) for the tool list. The
two surviving files under `mcp/` are the behavioral test suites
(`errors.test.ts`, `wire_dac.test.ts`), run against the live Lean
engine via `TROPICAL_ENGINE_CMD`.

---

## 11. Web backend (`web/`)

Browser **precompiled-patch player**. Not a second codegen: there is one
codegen (`EmitLlvm`), and the browser consumes the *same* IR the JIT runs,
lowered to wasm32 in-process by the same LLVM (`diffcli compile-wasm`). Per
patch the build ships a `<slug>.wasm` + a trimmed `<slug>.manifest.json`
(a `KernelManifest`); the browser fetches both and instantiates — no
emitter, no compiler, no plan mirror in the loop.

```
web/
  runtime/            Extractable runtime package (→ @tropical/runtime-wasm)
    manifest.ts         KernelManifest — the producer↔consumer contract
    layout.ts           computeLayout: manifest → linear-memory regions
    kernel.ts           WasmKernel: instantiate + render(f64) + process(f32+fade)
  build_patches.ts    Per patch: diffcli compile-wasm → <slug>.wasm +
                      <slug>.manifest.json → web/dist/patches/ + index.json
  build.ts            Full demo bundle: build patches, bundle worklet + app
  dev.ts              Plain static server (no COOP/COEP — no SharedArrayBuffer)
  host/context.ts     startHost: AudioContext + worklet node; loadPatch / fade
  worklet/processor.ts AudioWorkletProcessor → WasmKernel; postMessage protocol
  site/               Browser UI: app.ts (patch picker, play/stop) + index.html
```

`web/runtime/` depends only on `KernelManifest` — nothing from the compiler
or `tropical_plan_5` — so it is ready to extract once a second consumer (an
Electron instrument, a hosted player) appears. The kernel imports
`env.memory` + `env.round` and exports `tropical_kernel` + `__heap_base`;
the host owns the linear memory and places the kernel's pointer-argument
regions above `__heap_base`. A smoothstep fade is the only envelope —
**player, not instrument**: no live recompile, no hot-swap, no
SharedArrayBuffer (those belong to the native Electron instrument).

See [`web/CLAUDE.md`](../web/CLAUDE.md) for the worklet protocol,
linear-memory layout, and equivalence gates.

---

## 12. Type system (`lean/Tropical/Ir/Nodes.lean`, `lean/Tropical/Wiring.lean`)

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

`lean/Tropical/Wiring.lean` validates connections between typed
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
- `make lean` — compile the FFI shim + build the Lean subtree (the
  production compiler + MCP server + harness, one binary)
- `make mcp-lean` — build C++ + Lean, then launch the MCP server
  (`lean/.lake/build/bin/frontend`)
- `make validate` — the full gate: the Lean golden/equiv runner
  (`tropicaltest`: frozen audio goldens + native fused≡microkernel
  mode-equiv), the web demo-plan precompile, the surviving bun suites
  (WASM≡JIT in `tests/web` + the MCP behavioral tests, both against
  `frontend --rpc` via `TROPICAL_ENGINE_CMD`), and `ctest`
- `make parse-all` — regenerate the committed `stdlib/parsed/` bridge
  from `stdlib/*.md` via the Lean surface parser (`diffcli parse-all`)
- `make clean` — remove build directories

(There are no `make diff-*` targets anymore — the differential harness
was migration scaffolding and was removed with the TS oracle. See
`design/lean_port.md`.)

### Bun (`package.json`)

Bun ≥ 1.3 survives only to run the WASM-side TypeScript and the
behavioral test suites; there is no TS compiler service. `koffi` is
gone (the runtime FFI is Lean `@[extern]`). The web build and the bun
test suites talk to the Lean engine.

### MCP (`.mcp.json`)

```json
{ "mcpServers": { "tropical": { "command": "sh",
  "args": ["-c", "make -s lean 1>&2 && exec lean/.lake/build/bin/frontend"] } } }
```

### CI (`.github/workflows/`)

`ci.yml` builds `libtropical` and the Lean frontend/harness, then runs
the `tropicaltest` golden + native-equiv runner, precompiles the web
demo plans, runs the behavioral + WASM≡JIT bun suites against the Lean
engine, and runs the C++ `ctest`s; a separate typecheck job runs
`bun run tsc --noEmit` over the surviving TS. `lint.yml` is the YAML
lint.

---

## 14. Testing

### 14.1 C++ tests (`engine/tests/test_module_process.cpp`)

Custom harness, no framework dependency. Exercises the FlatRuntime C
API and the JIT without an audio device, driving the `compile_ir_text
→ JIT → run` path directly: it feeds LLVM IR text plus a trimmed
manifest and asserts on output buffer values (a constant kernel, a
sample-index ramp). Stateless kernels mean there is no hot-swap state
transfer to test — the swap carries only the sample index.

`cmake --build build -j4 && ctest --test-dir build`.

### 14.2 Lean tests (the `tropicaltest` binary)

The compiler/pipeline test surface is Lean-internal — the strata,
elaborator, emit, and partition passes are tested in the Lean modules
themselves. The cross-cutting runner is `tropicaltest`
(`lean/Tropicaltest.lean`): it renders the corpus through the real
`libtropical` JIT (over `Tropical.Ffi`), hashes 16×256 samples, and
gates on two things — the **frozen audio goldens** (`tests/golden/*.hash`,
the correctness floor; `tropicaltest --write` re-baselines) and the
**native mode-equivalence** check (fused vs. microkernel at 1e-12).

### 14.3 Surviving bun suites (`bun test`, against the Lean engine)

Run with `TROPICAL_ENGINE_CMD="./lean/.lake/build/bin/frontend --rpc"`:

- `tests/web/wasm_vs_jit.test.ts`,
  `tests/web/web_plans_vs_jit.test.ts` — WASM emission equivalence
  (WASM emit vs. the JIT, both off `tropical_plan_5`)
- `mcp/errors.test.ts`, `mcp/wire_dac.test.ts` — MCP behavioral suites,
  executable spec for the tool surface, run unmodified against the Lean
  engine

There is no longer any `compiler/**/*.test.ts` or `tests/equiv/*` — the
TS compiler and its differential equiv suites were deleted with the
port. The stdlib round-trip / lower-and-check audit that used to be
`scripts/validate_stdlib.ts` is now part of `tropicaltest` and the
Lean parse∘print fixpoint tests; `make parse-all` regenerates the
`stdlib/parsed/` bridge it leans on.

---

## 15. Key design decisions

### Acyclic by construction — cycles rejected, not broken

Cycles in source code throw `CycleViolation` at elaborate-time; cycles
in session wiring are rejected by `assertSessionAcyclic` (a plain "no
cycles at all" rule). The compiler's strata pipeline asserts acyclic
input at its boundary and refuses to lower cyclic IR. tropical is
closed-form-only: there is no state primitive to break a cycle with, so
nothing breaks one for you — feedback that needs genuine per-sample
state (an IIR pole on live or external input) is the ceded island,
delegated to a future stateful sister runtime (`supertropical`). See
`design/cf-only.md`.

### Single-kernel fusion, fixed topology

The whole session compiles to one native kernel function. No
instance boundaries at runtime, no per-instance dispatch, no
interpreter on the audio thread. LLVM gets to optimize across the
full graph. Topology changes hot-swap a freshly compiled kernel; since
kernels are pure functions of the time coordinate, the swap carries no
per-sample state (only the sample index continues, and the control
plane re-asserts params). Per-instance lifecycle semantics belong in a
different runtime.

### Pipeline of structure-dropping passes

The compiler is a chain of IR-to-IR passes where each pass drops a
specific kind of structure once it's been consumed. That shape is
what makes the layout of `lean/Tropical/Ir/` predictable, makes it
clear where new passes belong, and makes sample-for-sample
cross-backend agreement the right correctness criterion. The shape
also matches
the operadic reading sketched at the top of this document and in
`design/archive/operadic_ir.md` — the strata pipeline is a
composition of operad morphisms — but you can program against the
pipeline without that vocabulary.

### Decl identity instead of strings

Past `elaborate`, every reference is a branded integer `idx` into a
typed decl table — positional identity, not a string (the Lean
completion of the de Bruijn move the TS elaborator had already started
with object pointers). String-based scope walks would make later passes
re-derive information the elaborator already established. Branded indices
in the plan layer (`TempIdx`, `ArraySlotIdx`, `ModuleSlotIdx`) extend
the same discipline to integer slot identifiers, making cross-namespace
arithmetic a compile error.

### Hot-swap via double-buffered kernels

Live audio never stops for recompilation. The new kernel is built on
a background thread and a single atomic store publishes it; only the
sample index continues across the flip, since closed-form kernels have
no per-sample state to carry, and the control plane re-asserts param
values against the new kernel. Hot-swap is a native-runtime concern
(FlatRuntime in C++); the WASM path is a precompiled-patch player and
does not hot-swap.

### No interpreter fallback on the audio path

JIT failures are fatal at the runtime. Compile errors are caught
upstream (in `syncCompile` / over MCP) and never reach the runtime.
Cross-backend agreement is anchored instead by the WASM≡JIT
equivalence gate (`tests/web/wasm_vs_jit`) plus native mode-equivalence,
not by any runtime fallback. (Note: the project deliberately ships no
reference *interpreter* either — a differential against one proves
agreement, not correctness; the correctness floor is frozen audio
goldens plus the developer's ear. See `design/lean_port.md`.)

### Static shapes for arrays

All array shapes are known at compile time. That's what makes
`arrayLower` — full unroll of combinators and array ops — total. No
dynamic allocation on the audio thread.

### Transcendentals as programs

`sin`, `cos`, `tanh`, `exp`, `log`, `pow` are stdlib programs using
arithmetic + `Ldexp` + `FloatExponent`. They inline at strata time.
No libm dependency in the kernel; deterministic across platforms;
swap a file to change the math. See `stdlib/README.md`.

### Thread-safe control parameters

`ControlParam` uses atomic load/store with relaxed ordering.
Smoothed params apply one-pole lowpass per sample. Safe from any
thread; the audio thread never blocks.
