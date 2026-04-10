# Testing Philosophy

An algebra for reasoning about what tests do, where they apply, and what's missing.

---

## 1. The Algebra

A test is a function `t : S -> {pass, fail}` where S is some subset of the system's state space. Everything else follows from how you compose and constrain that function.

**Scope.** The scope of a test is a contiguous interval `[stage_i, stage_j]` in the pipeline. A test with scope `[ExprNode, ExprNode]` observes one stage; a test with scope `[ProgramJSON, Audio]` observes the whole pipeline. Scopes form a lattice under subset ordering on pipeline intervals. A test with smaller scope fails closer to the fault.

**Oracle strength.** Every test's predicate falls on a spectrum:

| Level | Name | Criterion |
|-------|------|-----------|
| O0 | Liveness | Does not crash, does not hang |
| O1 | Range | Output is in some expected set (e.g. peak in (0, 100)) |
| O2 | Relational | Output satisfies a structural invariant (type preservation, idempotence, categorical law) |
| O3 | Differential | Output matches a reference implementation within epsilon |
| O4 | Exact | Output matches a known-good value to machine precision |

Most smoke tests are O0-O1. Most DSP correctness tests need O3-O4. The useful insight: you can have an O4 oracle for structure (exact instruction match) and only an O1 oracle for numerics (peak level) on the same test — oracle strength is per-assertion, not per-test.

**Observability.** When `t(s) = fail`, how many bits does the failure message carry? Low: "test failed." High: "at stage Flatten, module VCO1, expression `{op:'ref', module:'VCO1', output:'saw'}`, expected type `float` but got `float[4]`." Observability is orthogonal to scope and oracle strength.

**Covering.** A test suite covers a behavior region R if for every behavior b in R, there exists a test whose scope includes b and whose oracle is strong enough to distinguish correct from incorrect. Gaps in the covering are undertested behaviors. You can ask: is there a region covered by only O0? That's more honest than counting "unit tests."

**Composition.** Tests compose vertically (extending scope across more pipeline stages) and horizontally (parameterizing over more inputs). Property-based testing is horizontal composition — same oracle, wider input domain. End-to-end testing is vertical composition — more stages, same input. The ideal suite has tests at every oracle level for every pipeline scope, composed with enough horizontal breadth to cover the input space.

---

## 2. The Pipeline Lattice

Tropical's compilation pipeline is a linear chain. Each boundary between stages has a concrete type that serves as the testable observable.

```
Stage 0:  ProgramJSON            tropical_program_1 schema (stdlib or user-defined)
Stage 1:  loadModuleFromJSON     JSON -> ProgramDef via slottifyExpr (name -> slot)
Stage 2:  Session assembly       type/instance registries, wiring, input expressions
Stage 3:  ExprNode trees         recursive JSON union -- the universal IR
Stage 4:  Flattening             multi-module -> single expression set, ref resolution
Stage 5:  Combinator expansion   generate/fold/chain/scan/iterate -> scalar trees
Stage 6:  Array lowering         array ops -> scalar primitives (static unrolling)
Stage 7:  Instruction emission   ExprNode -> FlatProgram NInstr stream
Stage 8:  JSON serialization     tropical_plan_4 JSON string
          --- C API boundary (koffi FFI) ---
Stage 9:  Plan parsing           JSON -> FlatProgram C++ struct
Stage 10: JIT compilation        FlatProgram -> LLVM IR -> native kernel
Stage 11: FlatRuntime            kernel execution, double-buffered state management
Stage 12: Audio output           RtAudio callback
```

The **Term IR** path is a parallel branch (`ExprNode -> Term -> optimized Term`) used for type-checking and structural reasoning. It does not feed the audio path directly.

**Key property: stages 0-8 are all JSON-serializable.** This is what makes the interpreter strategy viable — an interpreter can consume any intermediate representation up through the plan and produce numerical output, bypassing the JIT entirely.

### Boundary types

| Boundary | Type | Serializable |
|----------|------|:---:|
| 0 -> 1 | `ProgramJSON` | yes |
| 1 -> 2 | `ProgramDef` (slot-indexed IR) | yes |
| 2 -> 3 | `ExprNode` trees per instance | yes |
| 3 -> 4 | flattened `ExprNode` set | yes |
| 4 -> 5 | `ExprNode` with combinators expanded | yes |
| 5 -> 6 | `ExprNode` with arrays lowered | yes |
| 6 -> 7 | `FlatProgram` (`NInstr[]`) | yes |
| 7 -> 8 | `tropical_plan_4` JSON string | by definition |
| 8 -> 9 | `FlatProgram` C++ struct | no |
| 9 -> 10 | `NumericKernelFn` (function pointer) | no |
| 10 -> 11 | `double[]` output buffer | yes |

---

## 3. Oracle Taxonomy per Stage

What the strongest available oracle is for each stage, what exists today, and where the gaps are.

| Stage | Transfer function | Strongest oracle | Current tests | Gap |
|-------|-------------------|------------------|---------------|-----|
| ProgramJSON schema | Zod parse + roundtrip | O4 | `program.test.ts` (14) | No fuzz of malformed JSON |
| loadModuleFromJSON | slottifyExpr (name -> slot) | O4 (structural) | None dedicated | **No tests for slottifyExpr** |
| ExprNode construction | expr.ts operations | O4 | `expr.test.ts` (14) | Only construction, no evaluation |
| Term IR | Categorical laws | O4 + O2 (PBT) | `term.test.ts` (45+, 200-500 PBT runs) | Excellent |
| Term optimization | Structural rewrites | O2 (PBT) | `optimizer.test.ts` (26, 300 PBT runs) | Excellent |
| Flattening | Inline + resolve refs | O4 | `flatten_wiring.test.ts` (12) | No nested call resolution tests |
| Combinator expansion | Unroll to scalars | O4 | `combinators.test.ts` (22) | Good |
| Array lowering | Static unroll | O4 | `lower_arrays.test.ts` (25) | Good |
| Instruction emission | emitNumericProgram() | O4 (structural) | **None** | **Critical: no emit_numeric.test.ts** |
| Plan serialization | JSON.stringify | O4 | `plan.test.ts` (20, plan_1) | plan_4 not directly tested |
| Plan parsing (C++) | JSON -> struct | O2 | `test_module_process.cpp` (implicit) | No dedicated parsing tests |
| JIT compilation | LLVM codegen | O3 (differential) | **No interpreter exists** | **Critical: no reference oracle** |
| FlatRuntime | Kernel execution | O3 (epsilon) | `test_module_process.cpp` (10), `apply_plan.test.ts` (11) | Only handwritten plans |
| Audio output | RtAudio callback | O0 | `mcp/patch.test.ts` (range check) | Inherently limited |

### Module type coverage

19 built-in module types exist as `stdlib/*.json`. Integration tests exercise only **VCO, VCA, Clock** (3/19) through the full pipeline with numerical output. All 19 load and flatten successfully, but no test verifies that the remaining 16 produce numerically correct audio.

Untested numerically: ADEnvelope, ADSREnvelope, Reverb, Phaser, Phaser16, Compressor, BassDrum, LadderFilter, BitCrusher, NoiseLFSR, TopoWaveguide, Delay8/16/512/4410/44100.

---

## 4. Coverage Map

Current test suite as a covering of (scope interval) x (oracle strength). Each cell: `*` = well-covered, `~` = partial, `.` = no coverage.

```
Scope (interval)                     O0  O1  O2  O3  O4
-------------------------------------------------------
[ProgramJSON schema]                  .   .   *   .   *
[loadModuleFromJSON / slottifyExpr]   ~   .   .   .   .
[ExprNode construction]               .   .   *   .   *
[Term IR + type checking]             .   .   *   .   *
[Term optimization]                   .   .   *   .   *
[Flattening]                          .   .   *   .   *
[Combinator expansion]                .   .   .   .   *
[Array lowering]                      .   .   .   .   *
[Instruction emission]                .   .   .   .   .   <- EMPTY
[Plan serialization]                  .   .   *   .   *
[Plan parsing (C++)]                  *   .   .   .   .
[JIT + Runtime]                       *   .   .   .   ~
[ProgramJSON -> FlatRuntime]          *   *   .   .   ~
[ProgramJSON -> Audio]                *   *   .   .   .
```

### Critical gaps, ranked

**1. Instruction emission (stage 7).** No `emit_numeric.test.ts` exists. This stage translates ExprNode trees into `FlatProgram` instruction streams. If it has a bug, tests downstream either catch it at O1 (range check on audio output) or miss it entirely. A structural oracle (O4) here would dramatically improve fault localization.

**2. No interpreter for differential testing.** No ExprNode interpreter exists. This would enable O3 testing across the entire pipeline: for any program, run the interpreter on the lowered ExprNode tree, run the JIT path, compare sample-by-sample. This single addition would convert every integration test from O1 to O3.

**3. slottifyExpr / loadModuleFromJSON untested.** These are the new functions (from the stdlib-as-JSON migration) that convert named references in ProgramJSON to slot-indexed ProgramDef. They're exercised indirectly through integration tests, but a bug in name resolution would be hard to diagnose without isolated tests.

**4. Only 3/19 modules tested numerically.** The stdlib JSON files are frozen artifacts. Their correctness was validated once during generation (against the now-deleted TypeScript factories). There is no ongoing regression oracle — if the flattener or emitter changes behavior, 16 modules could silently break.

---

## 5. The Interpreter Strategy

The single highest-impact addition to the test suite. A reference evaluator for ExprNode trees that serves as a universal differential oracle.

### What it is

A function `evalExprNode(node: ExprNode, env: Environment) -> number | boolean | number[]` that recursively evaluates an ExprNode tree to a concrete value. The environment provides: sample rate, sample index, register values, parameter values, input values.

### Why ExprNode is the right level

ExprNode is the stage where all module boundaries have been erased (post-flatten), all combinators expanded (post-lower_arrays), but the representation is still JSON. An interpreter at this level serves as oracle for everything from stage 6 through stage 11.

### Sketch

```typescript
function evalExprNode(node: ExprNode, env: Env): number | boolean | (number | boolean)[] {
  if (typeof node === 'number') return node
  if (typeof node === 'boolean') return node
  if (Array.isArray(node)) return node.map(n => evalExprNode(n, env))
  switch (node.op) {
    case 'add':    return eval(node.args[0]) + eval(node.args[1])
    case 'mul':    return eval(node.args[0]) * eval(node.args[1])
    case 'sin':    return Math.sin(eval(node.args[0]))
    case 'select': return eval(node.args[0]) ? eval(node.args[1]) : eval(node.args[2])
    case 'sample_rate':  return env.sampleRate
    case 'sample_index': return env.sampleIndex
    case 'state_reg':    return env.registers[node.slot]
    case 'input':        return env.inputs[node.slot]
    case 'param':        return env.params[node.ptr]
    // ... ~35 ops total, matching BINARY_TAG + UNARY_TAG in emit_numeric.ts
  }
}
```

The interpreter handles the post-lowering ExprNode dialect only — no `generate`, `fold`, `let`, `zeros`, `reshape`. Those are all expanded by `lowerArrayOps`. The ops it must support are exactly those in `BINARY_TAG`, `UNARY_TAG`, and the special cases in `emit_numeric.ts`: approximately 35 ops.

### Stateful evaluation

For modules with registers, the interpreter runs sample-by-sample: evaluate all output expressions, evaluate all register-target expressions, advance sample index, repeat. This mirrors `FlatRuntime::process()`.

### Use `Math.*`, not JIT transcendentals

The interpreter uses JavaScript's `Math.sin`, `Math.cos`, etc. The JIT uses inline minimax polynomial approximations. These diverge at ~1e-10. This is a feature: the O3 differential test with epsilon tolerance (1e-6) tests the algebraic structure of the computation while allowing implementation-level numerical variation. Disagreement beyond 1e-6 indicates a semantic bug.

### Where it lives

`compiler/interpret.ts`. Imports only `ExprNode` from `expr.ts`. No FFI, no C++ dependency. Pure TS, fully deterministic, runnable in `bun test`.

### What it enables

Every test in `apply_plan.test.ts` can be upgraded: instead of checking `peak(buf) > 0` (O1), check `|jit_output[i] - interp_output[i]| < 1e-6 for all i` (O3). Every stdlib module gets tested by loading the JSON, flattening, interpreting N samples, and comparing against JIT output. The interpreter is the ongoing regression oracle that replaces the deleted TypeScript module factories.

---

## 6. Concrete Next Steps

Ranked by coverage impact — which regions of behavior space each addition covers.

### Priority 1: Build the ExprNode interpreter

`compiler/interpret.ts` — ~200 lines of TS (35 ops, each 1-3 lines).

- **Scope**: enables O3 oracle for `[ExprNode, Audio]` — the widest interval
- **Dependency**: none
- **Impact**: converts every integration test from O1 to O3; makes module coverage trivial to expand; provides ongoing oracle for stdlib JSON correctness

### Priority 2: Stdlib golden-output tests

For each of the 19 `stdlib/*.json` files: load -> flatten -> JIT -> process N samples -> compare against interpreter output (O3) or snapshot as golden reference (O4).

- **Scope**: `[ProgramJSON, FlatRuntime]` per module type
- **Dependency**: interpreter (for O3), or standalone with snapshotted output (O4)
- **Impact**: covers 16 currently untested module types; catches regressions in flattener/emitter/JIT

### Priority 3: Create `emit_numeric.test.ts`

Feed ExprNode trees in, assert on emitted `NInstr[]`. Same pattern as `lower_arrays.test.ts`.

- **Scope**: `[ExprNode, FlatProgram]` — stage 7 only
- **Oracle**: O4 (exact structural match)
- **Start with**: arithmetic, register read/write, array pack/index, select, sample_rate/sample_index
- **Impact**: fills the completely empty row in the coverage grid

### Priority 4: `slottifyExpr` unit tests

Exercise name-to-slot conversion edge cases: unknown names, nested calls, delay refs, array registers.

- **Scope**: `[ProgramJSON, ProgramDef]` — stage 1 only
- **Oracle**: O4 (exact structural match on output ExprNode)
- **Impact**: tests the new code path from the stdlib-as-JSON migration

### Priority 5: Property-based tests for instruction emission

- **Scope**: `[ExprNode, FlatProgram]` — stage 7
- **Oracle**: O2 (relational invariants)
- **Properties**: output_targets.length matches expected output count; all register_targets are valid temp indices; instruction count bounded by ExprNode tree size
- **Pairs with**: interpreter differential tests (O2 + O3 together)

### Priority 6: Cross-boundary roundtrip tests

Emit a `FlatProgram` in TS, serialize to `tropical_plan_4` JSON, parse in C++ via `NumericProgramParser`, verify field-by-field.

- **Scope**: `[FlatProgram TS, FlatProgram C++]` — stages 7-9
- **Oracle**: O4 (structural match after JSON roundtrip)
- **Impact**: catches serialization/parsing mismatches at the FFI boundary

---

## 7. Determinism

Stages 0-9 (all TS + C++ parsing) are **fully deterministic** — pure functions from input to output. Same ExprNode in, same FlatProgram out. O4 oracles apply freely.

Stage 10 (JIT) is **functionally deterministic.** The kernel function is deterministic for a given FlatProgram. Inline transcendental approximations (sin, cos, exp, log, tanh) produce bit-identical results across runs on the same platform (no libm dependency). Cross-platform reproducibility is a non-goal (macOS/ARM64 only in practice).

Stage 11 (FlatRuntime) with **hot-swap** introduces observable non-determinism: the output of buffer N+1 depends on whether a hot-swap occurred between N and N+1. Tests exercising hot-swap must account for this (and `test_module_process.cpp` test 3 already does).

Stage 12 (Audio) is **inherently non-deterministic** — callback timing, device latency, buffer underruns. No O3/O4 oracle is possible. O0 (liveness) and O1 (range) are the strongest practical oracles.

---

## 8. Maintaining the Covering

Rules for keeping the coverage map current as the codebase evolves.

**New stdlib module.** Add JSON to `stdlib/`. Add golden-output test (load -> flatten -> process N samples). Verify interpreter match once interpreter exists.

**New ExprNode op.** Four tests required: (a) construction test in `expr.test.ts`, (b) emission test in `emit_numeric.test.ts`, (c) interpreter case in `interpret.ts`, (d) C++ OpTag test in `test_module_process.cpp` with handwritten `tropical_plan_4` JSON.

**Modifying flattener or emitter.** Run full stdlib golden-output suite. Any change to `flatten.ts`, `lower_arrays.ts`, or `emit_numeric.ts` can silently alter the behavior of all 19 modules.

**New pipeline stage.** Add at least one O2+ test in isolation, plus verify existing end-to-end tests still pass. Update this document's lattice diagram.
