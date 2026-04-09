# compiler/

TypeScript layer. Handles module definition, expression construction, flattening, instruction emission, and the FFI bridge to C++. No audio processing happens here — this layer produces the `tropical_plan_4` JSON that the C++ engine JIT-compiles.

## Layout

```
expr.ts               ExprNode type, SignalExpr wrapper, all named operations, combinator constructors
module.ts             defineModule() DSL, delay(), feedback(), nested calls
module_library.ts     14 built-in module types + private sub-modules
program.ts            ProgramJSON (tropical_program_1) interface, conversions, load/save
flatten.ts            Patch → tropical_plan_4 (inline all modules, resolve refs)
lower_arrays.ts       Lower array ops + combinators to scalar primitives (static unrolling)
emit_numeric.ts       ExprNode trees → FlatProgram instruction stream
compiler.ts           Dependency graph, topological sort, term assembly
term.ts               Categorical IR (morphism, compose, tensor, trace, id)
type_check.ts         Type inference for terms, composition boundary validation
optimizer.ts          Term rewriting (identity elimination, flattening)
patch.ts              SessionState, patch load/save (tropical_patch_1), loadJSON schema detection
schema.ts             Zod validation schemas for patch/module/program JSON
apply_plan.ts         flattenPatch → JSON.stringify → runtime.loadPlan
array_wiring.ts       Typed port validation, auto-broadcast insertion
morphism_registry.ts  Named type coercion morphisms
plan.ts               ExecutionPlan (tropical_plan_1, intermediate/legacy)
bench_compile.ts      Compilation benchmarks

runtime/
  bindings.ts         koffi FFI declarations mirroring tropical_c.h
  runtime.ts          Runtime class (tropical_runtime_t wrapper, FinalizationRegistry)
  audio.ts            DAC class (tropical_dac_t wrapper, device listing)
  param.ts            Param (smoothed) and Trigger (fire-once), with .asExpr()
  audio_smoke.ts      Smoke test for audio output
```

## Expression system (`expr.ts`)

`ExprNode` is the universal IR — a recursive JSON-serializable union:

```typescript
type ExprNode = number | boolean | ExprNode[] | { op: string; ... }
```

`SignalExpr` wraps `ExprNode` with optional static shape metadata. All operations are free functions (no operator overloading in TS):

- **Arithmetic**: `add`, `sub`, `mul`, `div`, `mod`, `pow`, `floorDiv`
- **Comparison**: `lt`, `lte`, `gt`, `gte`, `eq`, `neq`
- **Bitwise**: `bitAnd`, `bitOr`, `bitXor`, `lshift`, `rshift`, `bitNot`
- **Math**: `neg`, `abs`, `sin`, `cos`, `exp`, `log`, `tanh`, `logicalNot`
- **Ternary**: `clamp`, `select`
- **Array**: `arrayPack`, `arraySet`, `index`, `zeros`, `ones`, `fill`, `reshape`, `transpose`, `slice`, `reduce`, `broadcastTo`, `mapArray`
- **Matrix**: `matrix`, `matmul` (supports arbitrary semirings via `mul_op`/`add_op`)
- **Combinators** (compile-time): `bindingExpr`, `let_`, `generate`, `repeat`, `iterate`, `fold`, `scan`, `map2`, `zipWith`, `chain`
- **Leaf nodes**: `sampleRate`, `sampleIndex`, `inputExpr`, `registerExpr`, `refExpr`, `nestedOutputExpr`, `delayValueExpr`, `paramExpr`, `triggerParamExpr`

## Module DSL (`module.ts`)

`defineModule()` builds expression trees symbolically — the `process` function doesn't compute values, it constructs a graph.

- Inputs and registers accessed via `SymbolMap.get()`, which returns `inputExpr(slot)` / `registerExpr(slot)` leaf nodes
- `delay(value, init)` — allocates a delay register, returns a `delayValueExpr` leaf
- `feedback(f, init)` — bundles register init with update morphism
- `ModuleType.call(...args)` — nested invocation inside `defineModule` bodies; creates `nestedOutputExpr` references resolved during flattening
- `definePureFunction(inputs, outputs, process)` — stateless module shorthand (no registers)

## Module library (`module_library.ts`)

14 built-in types registered by `loadBuiltins()`:

VCO (polyBLEP), Clock, ADEnvelope (polyBLAMP), ADSREnvelope (polyBLAMP), VCA, Reverb (Freeverb-style: 4 comb + 6 allpass), Phaser/Phaser16, Compressor, BassDrum, LadderFilter (4-pole Moog), BitCrusher, NoiseLFSR, TopoWaveguide (2D mesh), Delay variants (8/16/512/4410/44100).

Private sub-modules: `_wrap01`, `_polyBlep`, `_polyBlamp`, `_allpassStage`, `_defineCombFilter`.

## Compilation pipeline

### Flattening (`flatten.ts`)

The critical compilation step. Transforms a multi-module patch into a single flat instruction stream (`tropical_plan_4`).

1. **Input substitution** — replace `input(N)` with wiring expressions
2. **Reference resolution** — inline `ref(module, output)` by recursively substituting the referenced module's output expression
3. **Nested call resolution** — expand `nested_output(nodeId, outputId)` by inlining nested module expressions with offset register IDs
4. **Delay resolution** — convert `delay_value(nodeId)` to register reads
5. **Function inlining** — expand `call(function(body), args)` via input substitution
6. **Wiring type normalization** — validate compatibility, insert `broadcast_to` for shape mismatches

WeakMap memoization maintains DAG sharing and prevents exponential blowup.

### Array lowering and combinator expansion (`lower_arrays.ts`)

All shapes are static, so every operation fully unrolls:

- `zeros(shape)` → `ArrayPack` of zeros
- `reshape`, `transpose`, `slice` → reindexed `ArrayPack`
- `reduce(axis, op)` → unrolled fold
- `broadcast_to` → replicated elements
- `matmul(a, b)` → unrolled dot products (semiring lowering with `mul_op`/`add_op`)

Compile-time combinators expand via `substituteBindings` (replaces `{ op: 'binding', name }` nodes):
- `let` → sequential let\* evaluation + substitution
- `generate(n, i, body)` → inline array of body[i=0..n-1]
- `iterate(n, init, x, body)` → [init, f(init), f(f(init)), ...]
- `fold(arr, init, acc, elem, body)` → unrolled left fold to scalar
- `scan` → like fold but keeps intermediates as array
- `map2(arr, elem, body)` → substitute per element
- `zip_with(a, b, x, y, body)` → zip + substitute
- `chain(n, init, x, body)` → n serial applications

### Instruction emission (`emit_numeric.ts`)

Walks flattened ExprNode trees, emits a `FlatProgram`:

- `NInstr`: `tag` (op name → C++ `OpTag`), `dst` (temp slot), `args` (`NOperand[]`), `loop_count`, `strides`, `result_type`
- Operand kinds: `const`, `input`, `reg`, `array_reg`, `state_reg`, `param`, `rate`, `tick`
- Terminals (literals, inputs, registers) are embedded as operands, not separate instructions
- Output includes `register_count`, `array_slot_sizes`, `output_targets`, `register_targets`

### Term IR and type system (`compiler.ts`, `term.ts`, `type_check.ts`)

The compiler builds categorical terms from the dependency graph:

1. `buildDependencyGraph()` — extract module refs from input expressions
2. `tarjanSCC()` — cycle detection (feedback cycles are errors)
3. `topologicalSort()` — Kahn's algorithm with level grouping
4. `moduleToTerm()` — stateless → morphism, stateful → trace
5. `inferType()` — validate composition boundaries, tensor products, trace state alignment

Term constructors: `morphism`, `compose`, `tensor`, `trace`, `id`. Port types: `ScalarType`, `ArrayType`, `StructType`, `SumType`, `product`, `Unit`.

### Optimizer (`optimizer.ts`)

Structural rewrites iterated to fixed point:
- Identity elimination: `compose(id, f) → f`
- Compose flattening: right-associate nested compositions
- Tensor flattening: right-associate nested tensors

### Plan application (`apply_plan.ts`)

`applyFlatPlan(session, runtime)` ties it together: `flattenPatch()` → `JSON.stringify()` → `runtime.loadPlan()`. Called after any wiring mutation.

## FFI bridge (`runtime/`)

- `bindings.ts` — koffi function declarations matching `tropical_c.h`. Loads `libtropical.dylib` from `build/` or `build-profile/`.
- `runtime.ts` — `Runtime` class wrapping `tropical_runtime_t`. Uses FinalizationRegistry for GC-driven cleanup.
- `audio.ts` — `DAC` class wrapping `tropical_dac_t`. Static `listDevices()`.
- `param.ts` — `Param` (smoothed, one-pole lowpass) and `Trigger` (fire-once). `.asExpr()` returns a `SignalExpr` for wiring into expression trees.

## Tests

Run with `bun test`. Test files:

- `optimizer.test.ts` — term rewriting passes
- `flatten_wiring.test.ts` — flattening and wiring resolution
- `array_wiring.test.ts` — typed port validation, broadcast insertion
- `plan.test.ts` — execution plan generation and validation
- `compiler.test.ts` — dependency graph, topological sort, term assembly
- `expr.test.ts` — expression construction and evaluation
- `lower_arrays.test.ts` — array lowering to scalar primitives
- `apply_plan.test.ts` — plan application integration
- `term.test.ts` — term IR construction and type checking
- `combinators.test.ts` — compile-time combinator expansion (22 tests)
- `program.test.ts` — ProgramJSON schema conversions and Zod validation (14 tests)

## Adding a module type

1. Define in `module_library.ts` using `defineModule()` or `definePureFunction()`
2. Register in `loadBuiltins()`
3. No C++ changes needed unless you need a new expression op
