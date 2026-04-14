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
| O2 | Relational | Output satisfies a structural invariant (type preservation, idempotence) |
| O3 | Differential | Output matches a reference implementation within epsilon |
| O4 | Exact | Output matches a known-good value to machine precision |

Most smoke tests are O0-O1. Most DSP correctness tests need O3-O4. The useful insight: you can have an O4 oracle for structure (exact instruction match) and only an O1 oracle for numerics (peak level) on the same test — oracle strength is per-assertion, not per-test.

**Observability.** When `t(s) = fail`, how many bits does the failure message carry? Low: "test failed." High: "at stage Flatten, instance VCO1, expression `{op:'ref', instance:'VCO1', output:'saw'}`, expected type `float` but got `float[4]`." Observability is orthogonal to scope and oracle strength.

**Covering.** A test suite covers a behavior region R if for every behavior b in R, there exists a test whose scope includes b and whose oracle is strong enough to distinguish correct from incorrect. Gaps in the covering are undertested behaviors. You can ask: is there a region covered by only O0? That's more honest than counting "unit tests."

**Composition.** Tests compose vertically (extending scope across more pipeline stages) and horizontally (parameterizing over more inputs). Property-based testing is horizontal composition — same oracle, wider input domain. End-to-end testing is vertical composition — more stages, same input. The ideal suite has tests at every oracle level for every pipeline scope, composed with enough horizontal breadth to cover the input space.

---

## 2. The Pipeline Lattice

Tropical's compilation pipeline is a linear chain. Each boundary between stages has a concrete type that serves as the testable observable.

```
Stage 0:  ProgramJSON            tropical_program_1 schema (stdlib or user-defined)
Stage 1:  loadProgramDef         JSON -> ProgramDef via slottifyExpr (name -> slot)
Stage 2:  Session assembly       type/instance registries, wiring, input expressions
Stage 3:  ExprNode trees         recursive JSON union -- the universal IR
Stage 4:  Flattening             multi-instance -> single expression set, ref resolution
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

**Key property: stages 0-8 are all JSON-serializable.** An interpreter at any of these stages could produce numerical output without the JIT.

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

## 3. Current Test Suite

**199 tests total** — 189 TS tests across 11 files (1152 expect() calls), 10 C++ tests.

### Test inventory

| File | Tests | Scope | Oracle | What it covers |
|------|------:|-------|--------|----------------|
| `program.test.ts` | 5 | [ProgramJSON] | O4 | Zod schema validation: minimal, graph, nested programs, rejects |
| `expr.test.ts` | 20 | [ExprNode] | O4 | Array construction, shape inference, operations, matmul, broadcasting, coercion |
| `compiler.test.ts` | 30 | [Session assembly] | O4 | portTypeFromString, exprDependencies, buildDependencyGraph, topologicalSort, tarjanSCC, extractInstanceInfo |
| `flatten_wiring.test.ts` | 11 | [Flattening] | O4 | Wiring type validation: scalar/array compatibility, broadcast insertion, shape mismatches |
| `array_wiring.test.ts` | 13 | [Wiring validation] | O4 | checkArrayConnection: scalar compat, auto-broadcast, struct types, 2D shapes |
| `combinators.test.ts` | 22 | [Combinator expansion] | O4 | let, generate, iterate, fold, scan, map2, zip_with, chain, binding passthrough |
| `lower_arrays.test.ts` | 22 | [Array lowering] | O4 | zeros, ones, fill, reshape, transpose, slice, reduce, broadcast_to, map, matmul, nested lowering |
| `bounds.test.ts` | 35 | [Flattening + bounds] | O4 | applyBounds, resolveBounds, resolveBaseType, loadProgramDef bounds, flattenSession clamp injection, audio safety clamp, type aliases |
| `emit_numeric.test.ts` | 15 | [Instruction emission] | O4 | Terminals, scalar binary/unary/ternary, type inference/promotion, arrays (Pack, stride patterns, size-1 unboxing, index, array_set), output/register targets, typed state registers, CSE memoization |
| `apply_plan.test.ts` | 12 | [ProgramJSON → FlatRuntime] | O1-O4 | Session wiring (connect, disconnect, switch output, batch update, rewire, arithmetic expressions), FlatRuntime execution (VCO+VCA, Clock, continuous output, hot-swap state preservation) |
| `render.test.ts` | 5 | [ProgramJSON → Audio samples] | O1-O3 | Sawtooth peak/RMS range, sine dominant frequency (FFT), hot-swap frequency change, WAV output, buffer-size determinism (bit-exact cross-config) |
| `test_module_process.cpp` | 10 | [Plan JSON → FlatRuntime] | O4 | Sawtooth oscillator, two-output mix, hot-swap state transfer, array literals, integer counter with modular wrap, select/conditional, multi-register clock, multiple outputs summed, typed int bitwise (LFSR), typed bool comparison + select |

### Test infrastructure

**Buffer backend** (`compiler/test_utils/audio.ts`) — Device-free audio rendering for integration tests. `renderFrames(runtime, nCalls)` drives `process()` synchronously and returns collected samples. Signal analysis: `peak()`, `rms()`, `dominantFrequency()` (Cooley-Tukey FFT), `magnitudeSpectrum()`. WAV output via `writeWav()`.

**C++ test harness** (`engine/tests/test_module_process.cpp`) — Custom `run_test()` / `ASSERT()` / `ASSERT_NEAR()` macros. Each test builds `tropical_plan_4` JSON by hand and exercises the C API directly.

### Module type coverage

19 built-in module types exist as `stdlib/*.json`. Integration tests exercise only **VCO, VCA, Clock** (3/19) through the full pipeline with numerical output. All 19 load and flatten successfully, but no test verifies that the remaining 16 produce numerically correct audio.

Untested numerically: ADEnvelope, ADSREnvelope, Reverb, Phaser, Phaser16, Compressor, BassDrum, LadderFilter, BitCrusher, NoiseLFSR, TopoWaveguide, Delay8/16/512/4410/44100.

---

## 4. Oracle Taxonomy per Stage

What the strongest available oracle is for each stage, what exists today, and where the gaps are.

| Stage | Transfer function | Strongest oracle | Current tests | Gap |
|-------|-------------------|------------------|---------------|-----|
| ProgramJSON schema | Zod parse + roundtrip | O4 | `program.test.ts` (5) | No fuzz of malformed JSON |
| loadProgramDef / slottifyExpr | Name -> slot conversion | O4 (structural) | Indirect via `bounds.test.ts` | **No isolated slottifyExpr tests** |
| ExprNode construction | expr.ts operations | O4 | `expr.test.ts` (20) | Only construction, no evaluation |
| Graph utilities | Dep graph, topo sort, SCC | O4 | `compiler.test.ts` (30) | Good |
| Wiring validation | Type compat + broadcast | O4 | `array_wiring.test.ts` (13), `flatten_wiring.test.ts` (11) | Good |
| Flattening | Inline + resolve refs | O4 | `flatten_wiring.test.ts` (11), `bounds.test.ts` (35) | No nested call resolution tests |
| Combinator expansion | Unroll to scalars | O4 | `combinators.test.ts` (22) | Good |
| Array lowering | Static unroll | O4 | `lower_arrays.test.ts` (22) | Good |
| Bounds enforcement | Clamp insertion | O4 | `bounds.test.ts` (35) | Thorough |
| Instruction emission | emitNumericProgram() | O4 (structural) | `emit_numeric.test.ts` (15) | Good structural coverage; no numerical oracle |
| Plan serialization | JSON.stringify | O4 | Implicit via apply_plan | No dedicated tests |
| Plan parsing (C++) | JSON -> struct | O2 | `test_module_process.cpp` (implicit) | No dedicated parsing tests |
| JIT compilation | LLVM codegen | O4 (handwritten) | `test_module_process.cpp` (10) | Only handwritten plans; no cross-boundary roundtrip |
| FlatRuntime | Kernel execution | O1-O4 | `apply_plan.test.ts` (12), `render.test.ts` (5), `test_module_process.cpp` (10) | Only 3/19 modules |
| Audio output | RtAudio callback | O0 | None | No automated audio device test |

---

## 5. Coverage Map

Current test suite as a covering of (scope interval) x (oracle strength). Each cell: `*` = well-covered, `~` = partial, `.` = no coverage.

```
Scope (interval)                     O0  O1  O2  O3  O4
-------------------------------------------------------
[ProgramJSON schema]                  .   .   *   .   *
[loadProgramDef / slottifyExpr]       ~   .   .   .   .
[ExprNode construction]               .   .   *   .   *
[Graph utilities]                     .   .   *   .   *
[Wiring validation]                   .   .   *   .   *
[Flattening]                          .   .   *   .   *
[Combinator expansion]                .   .   .   .   *
[Array lowering]                      .   .   .   .   *
[Bounds enforcement]                  .   .   *   .   *
[Instruction emission]                .   .   .   .   *
[Plan parsing (C++)]                  *   .   .   .   .
[JIT + Runtime (handwritten plans)]   .   .   .   .   *
[ProgramJSON -> FlatRuntime]          *   *   .   .   ~
[ProgramJSON -> Audio samples]        *   *   .   ~   ~
```

### Critical gaps, ranked

**1. No interpreter for differential testing.** No ExprNode interpreter exists. This would enable O3 testing across the entire pipeline: for any program, run the interpreter on the lowered ExprNode tree, run the JIT path, compare sample-by-sample. This single addition would convert every integration test from O1 to O3.

**2. Only 3/19 modules tested numerically.** The stdlib JSON files are frozen artifacts. Their correctness was validated once during generation. There is no ongoing regression oracle — if the flattener or emitter changes behavior, 16 modules could silently break.

**3. slottifyExpr untested in isolation.** `loadProgramDef` is exercised indirectly through `bounds.test.ts` and integration tests, but `slottifyExpr` itself (the pure name→slot tree walk) has no dedicated tests. A bug in name resolution would be hard to diagnose.

**4. No cross-boundary roundtrip tests.** The TS emitter and C++ parser are tested independently but never against each other. A serialization mismatch would only surface as a mysterious audio bug in integration tests.

**5. No plan serialization tests.** The `tropical_plan_4` JSON schema is implicit — defined by whatever `emit_numeric.ts` produces and `NumericProgramParser.hpp` accepts. No test verifies this contract directly.

---

## 6. The Interpreter Strategy

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

Every test in `apply_plan.test.ts` and `render.test.ts` can be upgraded: instead of checking `peak(buf) > 0` (O1), check `|jit_output[i] - interp_output[i]| < 1e-6 for all i` (O3). Every stdlib module gets tested by loading the JSON, flattening, interpreting N samples, and comparing against JIT output. The interpreter is the ongoing regression oracle for all 19 stdlib modules.

---

## 7. Concrete Next Steps

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

### Priority 3: `slottifyExpr` unit tests

Exercise name-to-slot conversion edge cases: unknown names, nested calls, delay refs, array registers.

- **Scope**: `[ProgramJSON, ProgramDef]` — stage 1 only
- **Oracle**: O4 (exact structural match on output ExprNode)
- **Impact**: tests a critical path in the stdlib-as-JSON migration

### Priority 4: Cross-boundary roundtrip tests

Emit a `FlatProgram` in TS, serialize to `tropical_plan_4` JSON, parse in C++ via `NumericProgramParser`, verify field-by-field.

- **Scope**: `[FlatProgram TS, FlatProgram C++]` — stages 7-9
- **Oracle**: O4 (structural match after JSON roundtrip)
- **Impact**: catches serialization/parsing mismatches at the FFI boundary

### Priority 5: Property-based tests for instruction emission

- **Scope**: `[ExprNode, FlatProgram]` — stage 7
- **Oracle**: O2 (relational invariants)
- **Properties**: output_targets.length matches expected output count; all register_targets are valid temp indices; instruction count bounded by ExprNode tree size
- **Pairs with**: interpreter differential tests (O2 + O3 together)

---

## 8. Determinism

Stages 0-9 (all TS + C++ parsing) are **fully deterministic** — pure functions from input to output. Same ExprNode in, same FlatProgram out. O4 oracles apply freely.

Stage 10 (JIT) is **functionally deterministic.** The kernel function is deterministic for a given FlatProgram. Inline transcendental approximations (sin, cos, exp, log, tanh) produce bit-identical results across runs on the same platform (no libm dependency). Cross-platform reproducibility is a non-goal (macOS/ARM64 only in practice).

Stage 11 (FlatRuntime) with **hot-swap** introduces observable non-determinism: the output of buffer N+1 depends on whether a hot-swap occurred between N and N+1. Tests exercising hot-swap must account for this (and both `test_module_process.cpp` test 3 and `render.test.ts` test 3 already do).

Stage 12 (Audio) is **inherently non-deterministic** — callback timing, device latency, buffer underruns. No O3/O4 oracle is possible. O0 (liveness) and O1 (range) are the strongest practical oracles.

**Buffer-size determinism**: `render.test.ts` verifies that the same program produces bit-identical output regardless of buffer size (32×16 vs 512×1). This holds because the JIT kernel is per-sample with state register updates between samples — no vectorization across samples.

---

## 9. Maintaining the Covering

Rules for keeping the coverage map current as the codebase evolves.

**New stdlib module.** Add JSON to `stdlib/`. Add golden-output test (load -> flatten -> process N samples). Verify interpreter match once interpreter exists.

**New ExprNode op.** Three tests required: (a) construction test in `expr.test.ts`, (b) emission test in `emit_numeric.test.ts`, (c) C++ OpTag test in `test_module_process.cpp` with handwritten `tropical_plan_4` JSON. Plus: interpreter case in `interpret.ts` once it exists.

**Modifying flattener or emitter.** Run full stdlib golden-output suite. Any change to `flatten.ts`, `lower_arrays.ts`, or `emit_numeric.ts` can silently alter the behavior of all 19 modules.

**New pipeline stage.** Add at least one O2+ test in isolation, plus verify existing end-to-end tests still pass. Update this document's lattice diagram.

**New bounded type or alias.** Add test cases in `bounds.test.ts` for both `resolveBounds` and `flattenSession` clamp injection paths.
