# compiler/

TypeScript layer. Handles program definition, expression construction, flattening, instruction emission, and the FFI bridge to C++. No audio processing happens here — this layer produces the `tropical_plan_4` JSON that the C++ engine JIT-compiles.

## Layout

```
expr.ts               ExprNode type, SignalExpr wrapper, all named operations
program_types.ts      ProgramDef IR, ProgramType, ProgramInstance (pure data types, no DSL)
program.ts            ProgramJSON (tropical_program_1) interface, conversions, stdlib loading
flatten.ts            Session → tropical_plan_4 (inline all instances, resolve refs)
lower_arrays.ts       Lower array ops + combinators to scalar primitives (static unrolling)
emit_numeric.ts       ExprNode trees → FlatProgram instruction stream
compiler.ts           Dependency graph, topological sort, SCC, port type conversion
term.ts               Port types (PortType, ScalarKind), shape algebra, type utilities
session.ts            SessionState, loadProgramDef (ProgramJSON → ProgramDef), loadJSON, prettyExpr
schema.ts             Zod validation schemas for ProgramJSON
apply_plan.ts         flattenSession → JSON.stringify → runtime.loadPlan
array_wiring.ts       Typed port validation, auto-broadcast insertion
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

- **Arithmetic**: `add`, `sub`, `mul`, `div`, `mod`, `floorDiv`, `ldexp`
- **Comparison**: `lt`, `lte`, `gt`, `gte`, `eq`, `neq`
- **Bitwise**: `bitAnd`, `bitOr`, `bitXor`, `lshift`, `rshift`, `bitNot`
- **Math**: `neg`, `abs`, `sqrt`, `float_exponent`, `logicalNot` (transcendentals — sin, cos, tanh, exp, log, pow — live in `stdlib/` as programs, not primitives)
- **Ternary**: `clamp`, `select`
- **Array**: `arrayPack`, `arraySet`, `index`, `zeros`, `ones`, `fill`, `reshape`, `transpose`, `slice`, `reduce`, `broadcastTo`, `mapArray`
- **Matrix**: `matrix`, `matmul` (supports arbitrary semirings via `mul_op`/`add_op`)
- **Leaf nodes**: `sampleRate`, `sampleIndex`, `inputExpr`, `registerExpr`, `refExpr`, `nestedOutputExpr`, `delayValueExpr`, `paramExpr`, `triggerParamExpr`

Compile-time combinators (`let`, `generate`, `iterate`, `fold`, `scan`, `map2`, `zip_with`, `chain`) are embedded directly in JSON as ExprNode ops and lowered by `lower_arrays.ts` — no TS wrapper functions needed.

## Program types (`program_types.ts`)

Pure data types — no DSL, no side effects:

- **`ProgramDef`** — the compiler's slot-indexed IR consumed by the flattener. Fields: `outputExprNodes`, `registerExprNodes`, `delayUpdateNodes`, `nestedCalls`, etc. Built from ProgramJSON by `loadProgramDef()` in `session.ts` via `slottifyExpr()` (pure name→slot tree walk).
- **`ProgramType`** — wraps a `ProgramDef`, registered in `SessionState.typeRegistry`
- **`ProgramInstance`** — named instance of a type, with port accessors

## Standard library (`stdlib/*.json`)

19 built-in types as ProgramJSON files, loaded by `loadStdlib()` in `program.ts`:

VCO (polyBLEP), Clock, ADEnvelope (polyBLAMP), ADSREnvelope (polyBLAMP), VCA, Reverb (Freeverb-style: 4 comb + 6 allpass), Phaser/Phaser16, Compressor, BassDrum, LadderFilter (4-pole Moog), BitCrusher, NoiseLFSR, TopoWaveguide (2D mesh), Delay variants (8/16/512/4410/44100).

Complex programs use inline `programs` for subprogram composition (e.g., VCO defines `_wrap01` and `_polyBlep` as nested programs).

## Compilation pipeline

### Flattening (`flatten.ts`)

The critical compilation step. Transforms a multi-instance session into a single flat instruction stream (`tropical_plan_4`).

1. **Input substitution** — replace `input(N)` with wiring expressions
2. **Reference resolution** — inline `ref(instance, output)` by recursively substituting the referenced instance's output expression
3. **Nested call resolution** — expand `nested_output(nodeId, outputId)` by inlining nested instance expressions with offset register IDs
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

### Graph utilities (`compiler.ts`)

Used by the flattener to determine execution order:

1. `buildDependencyGraph()` — extract instance refs from input expressions
2. `tarjanSCC()` — cycle detection (feedback cycles are errors)
3. `topologicalSort()` — Kahn's algorithm with level grouping
4. `extractInstanceInfo()` — convert ProgramDef to slot-indexed InstanceInfo
5. `portTypeFromString()` — parse type annotations to PortType

### Plan application (`apply_plan.ts`)

`applyFlatPlan(session, runtime)` ties it together: `flattenSession()` → `JSON.stringify()` → `runtime.loadPlan()`. Called after any wiring mutation.

## FFI bridge (`runtime/`)

- `bindings.ts` — koffi function declarations matching `tropical_c.h`. Loads `libtropical.dylib` from `build/` or `build-profile/`.
- `runtime.ts` — `Runtime` class wrapping `tropical_runtime_t`. Uses FinalizationRegistry for GC-driven cleanup.
- `audio.ts` — `DAC` class wrapping `tropical_dac_t`. Static `listDevices()`.
- `param.ts` — `Param` (smoothed, one-pole lowpass) and `Trigger` (fire-once). `.asExpr()` returns a `SignalExpr` for wiring into expression trees.

## Tests

Run with `bun test`. Test files:

- `flatten_wiring.test.ts` — flattening and wiring resolution
- `array_wiring.test.ts` — typed port validation, broadcast insertion
- `compiler.test.ts` — dependency graph, topological sort, SCC, port type conversion
- `expr.test.ts` — expression construction and evaluation
- `lower_arrays.test.ts` — array lowering to scalar primitives
- `apply_plan.test.ts` — plan application integration (requires native lib)
- `combinators.test.ts` — compile-time combinator expansion
- `program.test.ts` — ProgramJSON schema conversions and Zod validation
- `bounds.test.ts` — bounded type enforcement and clamp insertion
- `emit_numeric.test.ts` — instruction emission

## Adding a program type

1. Create a `stdlib/MyType.json` file using the `tropical_program_1` schema
2. The file is automatically loaded by `loadStdlib()` on startup
3. No C++ changes needed unless you need a new expression op
