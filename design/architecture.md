# tropical Architecture

Detailed reference for the internal architecture. Organized by functional unit.

---

## 1. Overview

tropical is a realtime audio synthesis system. An LLM or human author defines a *program* — a graph of DSP program instances wired together — and the system compiles the entire program into a single native kernel that runs per-sample in an audio callback. There is no interpreter fallback and no module boundaries at runtime.

The unified representation is `ProgramJSON` (`tropical_program_1`). A program with `process` is a leaf (direct computation); a program with `instances` and `audio_outputs` is a top-level graph; a program with `instances`, `inputs`, and `outputs` is a reusable composite. The legacy `tropical_patch_1` format is still accepted on load.

The system has three layers:

| Layer | Language | Directory | Role |
|-------|----------|-----------|------|
| **Compiler** | TypeScript | `compiler/` | DSL, module library, expression trees, combinators, flattening, instruction emission, type system |
| **Engine** | C++20 | `engine/` | Plan parsing, JIT compilation (LLVM ORC), per-sample kernel execution, audio output |
| **MCP Server** | TypeScript | `mcp/` | AI-native interface: tool-based program manipulation over stdio, backed by the compiler and engine |

The **C API** (`engine/c_api/tropical_c.h`) is the stable boundary between the TypeScript and C++ layers. TypeScript calls into C++ via koffi FFI.

### Data flow (high level)

```
Program definition (JSON / MCP tools / DSL)
    → Module instantiation (TS-only, no C++ calls)
    → Expression tree construction (ExprNode JSON trees)
    → Flattening (inline all modules into one expression set)
    → Combinator expansion (unroll generate/fold/chain/etc. to scalar trees)
    → Array lowering (expand static-shape array ops to scalar primitives)
    → Instruction emission (ExprNode → FlatProgram instruction stream)
    → JSON serialization (tropical_plan_4)
    ─── C API boundary ───
    → Plan parsing (JSON → FlatProgram struct)
    → JIT compilation (FlatProgram → LLVM IR → native kernel)
    → Per-sample execution in audio callback
    → Audio output (RtAudio / CoreAudio)
```

---

## 2. Expression System (`compiler/expr.ts`, `engine/expr/`)

Expressions are the universal representation: every signal computation in tropical is an expression tree.

### 2.1 TypeScript ExprNode

`ExprNode` is a recursive JSON-serializable union type:

```typescript
type ExprNode =
  | number              // literal float
  | boolean             // literal bool
  | ExprNode[]          // inline array
  | { op: string; ... } // named operation
```

`SignalExpr` is a thin wrapper around `ExprNode` that tracks optional static shape metadata and provides named construction functions (`add`, `mul`, `sin`, etc.). TypeScript has no operator overloading, so all operations are free functions.

Operations available:
- **Arithmetic**: `add`, `sub`, `mul`, `div`, `mod`, `pow`, `floorDiv`
- **Comparison**: `lt`, `lte`, `gt`, `gte`, `eq`, `neq`
- **Bitwise**: `bitAnd`, `bitOr`, `bitXor`, `lshift`, `rshift`, `bitNot`
- **Unary/math**: `neg`, `abs`, `sin`, `cos`, `exp`, `log`, `tanh`, `logicalNot`
- **Ternary**: `clamp`, `select`
- **Array**: `arrayPack`, `arraySet`, `index`, `zeros`, `ones`, `fill`, `reshape`, `transpose`, `slice`, `reduce`, `broadcastTo`, `mapArray`
- **Matrix**: `matrix`, `matmul`
- **Function**: `exprFunction`, `exprCall`
- **ADT**: `constructStruct`, `fieldAccess`, `constructVariant`, `matchVariant`
- **Combinators** (compile-time expansion): `bindingExpr`, `let_`, `generate`, `repeat`, `iterate`, `fold`, `scan`, `map2`, `zipWith`, `chain`
- **Leaf nodes**: `sampleRate`, `sampleIndex`, `inputExpr`, `registerExpr`, `refExpr`, `nestedOutputExpr`, `delayValueExpr`, `paramExpr`, `triggerParamExpr`

### 2.2 C++ Expression AST (`engine/expr/Expr.hpp`)

The C++ side has a parallel expression representation in `tropical_expr`:

- **`Value`** — tagged union: `Int` (i64), `Float` (f64), `Bool`, `Array` (vector of Values), `Matrix` (row-major flat vector with dimensions), `Struct`, `Sum`
- **`ExprSpec`** — tree node with `ExprKind` discriminator, `lhs`/`rhs`/`args` children, literal value, module name, slot IDs, control param pointer
- **`ExprKind`** enum covers ~40 node kinds including all arithmetic, comparison, bitwise, unary, array, ADT, and param operations

Supporting modules:
- **`ExprEval.hpp`** — Interpreter for `Value` operations (used during module definition, not at runtime)
- **`ExprRewrite.cpp`** — Constant folding, algebraic simplification (0*x→0, 1*x→x, x+0→x), dead ref replacement
- **`ExprStructural.cpp`** — Structural hashing, structural equality, pure function inlining with substitution

---

## 3. Module System (`compiler/module.ts`, `compiler/module_library.ts`)

### 3.1 Module Definition DSL

Modules are defined entirely in TypeScript using `defineModule()`:

```typescript
defineModule(
  name: string,
  inputs: PortSpec[],
  outputs: PortSpec[],
  regs: Record<string, RegInit>,
  process: (inputs: SymbolMap, regs: SymbolMap) => ProcessResult,
  sampleRate?: number,
  inputDefaults?: Record<string, ExprCoercible>,
)
```

The `process` function builds expression trees symbolically — it does not compute values. It reads inputs and registers via `SymbolMap.get()` (which returns `inputExpr(slotId)` / `registerExpr(slotId)` nodes) and returns `{ outputs, nextRegs }` maps of expression trees.

Key DSL features:
- **`delay(value, init)`** — One-sample delay. Allocates a delay node, captures the update expression, and returns a `delayValueExpr` leaf.
- **`feedback(f, init)`** — Helper that bundles a register init value with its update morphism.
- **`ModuleType.call(...args)`** — Nested module invocation. Used inside `defineModule` bodies to compose modules. Creates `nestedOutputExpr` references that are resolved during flattening.
- **`definePureFunction(inputs, outputs, process)`** — Stateless module shorthand. Wraps a `defineModule` with no registers.

### 3.2 ModuleType and ModuleInstance

- **`ModuleType`** holds the compiled `ModuleDef` (expression trees, port metadata, nested call info). It is a *type*, not an instance.
- **`ModuleInstance`** is a named instance of a type, carrying a reference to its `ModuleDef` plus an instance name. Instances are TS-only — no C API calls.

### 3.3 Built-in Module Library

`compiler/module_library.ts` defines all built-in module types:

| Type | Description | Inputs | Outputs | Registers |
|------|-------------|--------|---------|-----------|
| **VCO** | Band-limited oscillator (polyBLEP antialiasing) | freq, fm, fm_index | saw, tri, sin, sqr | phase, tri_state |
| **Clock** | Clock generator with ratio array | freq, ratios_in | output, ratios_out | (stateless via sampleIndex) |
| **ADEnvelope** | Attack-decay envelope (polyBLAMP) | gate, attack, decay | env | stage, phase, startLevel |
| **ADSREnvelope** | Full ADSR envelope (polyBLAMP) | gate, attack, decay, sustain, release | env | stage, phase, release_level |
| **VCA** | Voltage-controlled amplifier | audio, cv | out | (stateless) |
| **Reverb** | Freeverb-style (4 comb + 6 allpass) | input, mix, decay, damp | output | (nested module state) |
| **Phaser** / **Phaser16** | 4 / 16 stage allpass phaser | input, feedback, lfo_speed | output, lfo | fb |
| **Compressor** | Dynamic range compressor | input, sidechain, threshold, ratio, attack_ms, release_ms, makeup | output, gr | env, gr |
| **BassDrum** | Synthesized kick drum | gate, freq, punch, decay | output | stage, phase |
| **LadderFilter** | 4-pole Moog-style filter | input, cutoff, resonance | output | s1, s2, s3, s4 |
| **BitCrusher** | Bit depth and sample rate reduction | input, bits, downsample | output | hold, counter |
| **NoiseLFSR** | Linear feedback shift register noise | gate, rate | output | lfsr, counter |
| **TopoWaveguide** | 2D waveguide mesh (default 4x4) | strike, damp, tension | output | (matrix state via registers) |
| **Delay8/16/512/4410/44100** | Fixed-length delay lines | x | y | buf (array register) |

Private sub-modules used internally: `_wrap01`, `_polyBlep`, `_polyBlamp`, `_allpassStage`, `_defineCombFilter`.

---

## 4. Compilation Pipeline

### 4.1 Session State (`compiler/patch.ts`, `compiler/program.ts`)

`SessionState` is the mutable in-memory representation of a live session:

- `typeRegistry: Map<string, ModuleType>` — registered program types (builtins + user-defined)
- `instanceRegistry: Map<string, ModuleInstance>` — live program instances by name
- `inputExprNodes: Map<string, ExprNode>` — wiring expressions keyed as `"InstanceName:inputName"`
- `graphOutputs: Array<{ module, output }>` — which outputs are mixed to audio
- `params: Map<string, Param>` / `triggers: Map<string, Trigger>` — named control parameters
- `runtime: Runtime` — the FlatRuntime wrapper

Programs are loaded via `loadJSON()` which accepts both `tropical_program_1` and `tropical_patch_1`. Saved as `tropical_program_1` via `saveProgramFromSession()`.

`ProgramJSON` (`compiler/program.ts`) is the unified schema. Conversion functions bridge to/from the legacy `PatchJSON` format: `convertPatchToProgram`, `convertProgramToPatch`, `convertModuleDefToProgram`.

### 4.2 Compiler (`compiler/compiler.ts`)

The compiler converts a `CompilerInput` (modules + wiring + outputs) into a `CompiledPatch` containing a well-typed categorical `Term`.

Steps:
1. **Dependency graph** — Extract module references from input expressions via `exprDependencies()`, build adjacency map
2. **Cycle detection** — Tarjan's SCC algorithm. Feedback cycles are errors (auto-trace not yet implemented)
3. **Topological sort** — Kahn's algorithm with level grouping. Each level can execute in parallel
4. **Term assembly** — For each level: build wiring morphism, build module terms (stateless → morphism, stateful → trace), compose/tensor them
5. **Type checking** — Verify the assembled term via `inferType()`

### 4.3 Term Language (`compiler/term.ts`)

The categorical IR — a free monoidal category:

**Objects** (types):
- `ScalarType('float' | 'int' | 'bool')` — scalar types
- `ArrayType(element, shape)` — static-shape arrays
- `StructType(name)`, `SumType(name)` — algebraic data types
- `product(factors)` — tensor product of types
- `Unit` — monoidal unit

**Morphisms** (terms):
- `morphism(name, dom, cod, body)` — named function with expression body
- `compose(first, second)` — sequential composition (f ; g)
- `tensor(left, right)` — parallel execution (f x g)
- `trace(stateType, init, body)` — feedback with typed state
- `id(portType)` — identity morphism

### 4.4 Type Checking (`compiler/type_check.ts`)

Infers `{dom, cod}` for every term, validating composition boundaries (`cod(f) = dom(g)`), tensor products, and trace state alignment. Supports numpy-style shape broadcasting for array types.

### 4.5 Optimizer (`compiler/optimizer.ts`)

Structural rewrite passes on the term IR, iterated to fixed point:
1. **Identity elimination** — `compose(id, f) → f`, `tensor(f, id(Unit)) → f`
2. **Compose flattening** — right-associate nested compositions
3. **Tensor flattening** — right-associate nested tensors

### 4.6 Flattening (`compiler/flatten.ts`)

The critical compilation step: transforms a multi-module patch into a single flat instruction stream (tropical_plan_4).

Key operations:
1. **Input substitution** — Replace `input(N)` nodes with wiring expressions
2. **Reference resolution** — Inline `ref(module, output)` by recursively substituting the referenced module's output expression
3. **Nested call resolution** — Expand `nested_output(nodeId, outputId)` by inlining the nested module's expressions with offset register IDs
4. **Delay resolution** — Convert `delay_value(nodeId)` to register reads at computed offsets
5. **Function inlining** — Expand `call(function(body), args)` by substituting input nodes with arguments
6. **Wiring type normalization** — Validate type compatibility, insert `broadcast_to` wrappers for shape mismatches

All operations use WeakMap-based memoization to maintain DAG sharing and prevent exponential blowup.

### 4.7 Array Lowering and Combinator Expansion (`compiler/lower_arrays.ts`)

Lowers first-class array operations and compile-time combinators to scalar primitives. All shapes are static, so every expansion is fully unrolled.

**Array operations:**
- `zeros(shape)` → `ArrayPack` of zeros
- `reshape`, `transpose`, `slice` → reindexed `ArrayPack`
- `reduce(axis, op)` → unrolled fold
- `broadcast_to` → replicated elements
- `map(fn, arr)` → unrolled `call` per element
- `matmul(a, b, shape_a, shape_b)` → unrolled dot products via semiring lowering (supports arbitrary `mul_op`/`add_op` for generalized semirings)

**Compile-time combinators** (expand via `substituteBindings` — replaces `{ op: 'binding', name }` nodes with concrete values):
- `let(bind, body)` → sequential let\* evaluation, substitute into body
- `generate(n, i, body)` → `[body[i=0], body[i=1], ..., body[i=n-1]]` (inline array)
- `iterate(n, init, x, body)` → `[init, body[x=init], body[x=body[x=init]], ...]` (length n)
- `fold(arr, init, acc, elem, body)` → unrolled left fold threading accumulator, output: final scalar
- `scan(arr, init, acc, elem, body)` → like fold but collect each intermediate, output: inline array
- `map2(arr, elem, body)` → substitute per element, output: inline array
- `zip_with(a, b, x, y, body)` → zip elements, substitute both vars, output: inline array
- `chain(n, init, x, body)` → unroll n applications: `body[x=body[x=...body[x=init]...]]`, output: final value

Combinators are organized by intent to help LLMs narrow their search: **generative** (generate, repeat, iterate), **reductive** (fold, scan), **transformative** (map2, zip_with), **compositional** (chain, let).

### 4.8 Instruction Emission (`compiler/emit_numeric.ts`)

Walks the flattened ExprNode trees and emits a `FlatProgram` — a flat instruction stream for the C++ JIT.

Each instruction (`NInstr`) has:
- `tag` — operation name (maps to `OpTag` enum in C++)
- `dst` — destination temp register or array slot
- `args` — `NOperand[]` (typed: const, input, reg, array_reg, state_reg, param, rate, tick)
- `loop_count` — 1 for scalar, N for elementwise array loop
- `strides` — per-arg: 1 = iterate with loop index, 0 = broadcast
- `result_type` — `'float' | 'int' | 'bool'`

Terminals (literals, inputs, registers, params) are embedded as operands within instructions, not separate pseudo-ops.

The output `FlatProgram` also carries:
- `register_count` — temp register allocation
- `array_slot_sizes` — element count per array slot
- `output_targets` — which temps hold output values
- `register_targets` — which temps hold updated register values

### 4.9 Plan Application (`compiler/apply_plan.ts`)

`applyFlatPlan(session, runtime)` ties the pipeline together:
1. `flattenPatch(session)` → `FlatPlan` JSON
2. `JSON.stringify(plan)` → string
3. `runtime.loadPlan(json)` → C API call

This is called after any mutation to the session's wiring or outputs.

---

## 5. Engine: Plan Loading and JIT Compilation

### 5.1 Plan Schema: tropical_plan_4

The JSON plan sent to C++ has schema `tropical_plan_4`:

```json
{
  "schema": "tropical_plan_4",
  "config": { "sample_rate": 44100.0 },
  "state_init": [0.0, ...],
  "register_names": ["VCO1_phase", ...],
  "register_types": ["float", "int", ...],
  "array_slot_names": ["Delay1_buf", ...],
  "outputs": [0, 2, ...],
  "instructions": [ { "tag": "Add", "dst": 0, "args": [...], "loop_count": 1, "strides": [] }, ... ],
  "register_count": N,
  "array_slot_sizes": [512, ...],
  "output_targets": [3, 7, ...],
  "register_targets": [5, 8, ...]
}
```

### 5.2 NumericProgramParser (`engine/runtime/NumericProgramParser.hpp`)

Thin JSON deserializer: reads the `tropical_plan_4` JSON and produces a `tropical_jit::FlatProgram` struct. No expression tree walking — just reads the pre-compiled instruction stream.

### 5.3 OrcJitEngine (`engine/jit/OrcJitEngine.hpp`, `.cpp`)

Singleton LLVM ORC JIT engine. Compiles `FlatProgram` → native kernel.

**`compile_flat_program(program)`**:
1. Build canonical cache key (MD5 hash of serialized program, with param pointers replaced by ordinals)
2. Check in-memory cache → return if hit
3. Generate LLVM IR module:
   - Create function with signature: `(inputs, registers, arrays, array_sizes, temps, sample_rate, start_sample_index, param_ptrs, output_buffer, buffer_length) → void`
   - Emit outer sample loop (iterates `buffer_length` times)
   - For each instruction: resolve operands, emit typed LLVM IR
   - Write per-sample output to `output_buffer[sample_idx]` (sum of mix outputs)
   - Write back state registers after the loop
4. Add module to LLJIT, look up symbol → `NumericKernelFn`

**Type-directed code generation**: Every operand and instruction carries a scalar type (`Float`/`Int`/`Bool`). The JIT emits native `f64`/`i64`/`i1` operations with explicit coercion at type boundaries.

**Inline transcendentals**: `sin`, `cos`, `exp`, `log`, `tanh` are implemented as inline polynomial approximations (no libm calls), making kernels self-contained and deterministic across platforms.

**Kernel object cache**: Compiled object code is persisted to `~/.cache/tropical/kernels/<build-id>/`. Cache key is MD5 of the canonical program; the build-id subdirectory (derived from the binary's LC_UUID / ELF build-id) auto-invalidates when the dylib is rebuilt.

**Loop emission for arrays**: When `loop_count > 1`, the JIT emits an elementwise loop. `strides[i]` controls whether each argument advances with the loop index (array) or broadcasts (scalar).

### 5.4 FlatRuntime (`engine/runtime/FlatRuntime.hpp`, `.cpp`)

The execution container. Manages two `KernelState` slots for lock-free hot-swap.

**`KernelState`** holds:
- `kernel` — native function pointer (`NumericKernelFn`)
- `registers` — persistent state as `int64_t[]` (bit-cast for floats)
- `temps` — scratch registers
- `array_storage` / `array_ptrs` / `array_sizes` — array state
- `register_names` / `array_names` — for named state transfer on hot-swap
- `trigger_params` — list of trigger param pointers for per-frame snapshot
- `sample_rate`, `sample_index`

**Double-buffered hot-swap**: `load_plan()` compiles to the inactive slot, copies matching state from the active slot (by register/array name), then atomically swaps via `active_state_`. A mutex serializes concurrent builds. State is transferred *before* the atomic swap to eliminate the one-buffer pop that would occur if audio ran with zeroed registers.

**`process()`** (called from audio thread):
1. Load active state index (acquire)
2. Snapshot trigger params (atomic exchange → frame_value)
3. Call kernel (single invocation processes entire buffer)
4. Advance sample_index
5. Apply smoothstep fade envelope (fade-in / fade-out for click-free start/stop)

**Fade control**: Hermite smoothstep curve over 2048 samples (configurable). `begin_fade_in()` / `begin_fade_out()` set atomic counters decremented per sample in `process()`.

---

## 6. Audio Output (`engine/dac/TropicalDAC.hpp`)

`TropicalDACImpl<AudioSource>` is a templated DAC driver using RtAudio.

**AudioSource requirements**: `process()`, `outputBuffer`, `getBufferLength()`, `begin_fade_in()`, `begin_fade_out()`, `is_fade_out_complete()`. FlatRuntime satisfies this.

**Audio callback** (`fill_buffer`):
1. Call `source->process()`
2. Copy mono output to all channels
3. Track timing stats (avg/max callback duration, underrun/overrun counts)

**Device management**:
- Watcher thread polls every 50ms for device disconnection or default device changes
- On disconnect: abort stream, wait 500ms, reopen with fade-in
- `switch_device(id)`: explicit device switching with fade-in

**Stats**: `callback_count`, `avg_callback_ms`, `max_callback_ms`, `underrun_count` (driver-reported), `overrun_count` (callbacks exceeding time budget).

---

## 7. C API (`engine/c_api/tropical_c.h`)

Stable C interface between TypeScript (via koffi FFI) and C++. All handles are opaque `void*`.

**ControlParam API**:
- `tropical_param_new(init, time_const)` — smoothed parameter with one-pole lowpass
- `tropical_param_new_trigger()` — fire-once trigger parameter
- `tropical_param_set/get` — thread-safe atomic read/write

**FlatRuntime API**:
- `tropical_runtime_new(buffer_length)` → runtime handle
- `tropical_runtime_load_plan(rt, json, len)` — hot-swap plan load
- `tropical_runtime_process(rt)` — process one buffer
- `tropical_runtime_output_buffer(rt)` → `const double*`
- Fade control: `begin_fade_in`, `begin_fade_out`, `is_fade_out_complete`

**DAC API**:
- `tropical_dac_new_runtime(rt, sample_rate, channels)` — create DAC from runtime
- `tropical_dac_start/stop` — audio control
- `tropical_dac_get_stats/reset_stats` — callback statistics
- `tropical_dac_switch_device(dac, device_id)` — live device switching
- `tropical_dac_is_reconnecting` — disconnect detection

**Device enumeration** (no DAC instance required):
- `tropical_audio_device_count/get_device_ids/get_device_info/default_output_device`

**Error handling**: Thread-local error string via `tropical_last_error()`.

---

## 8. FFI Bridge (`compiler/runtime/`)

TypeScript classes wrapping the C API via koffi:

- **`bindings.ts`** — Raw koffi function declarations mirroring `tropical_c.h`. Loads `libtropical.dylib` from build directories.
- **`runtime.ts`** — `Runtime` class wrapping `tropical_runtime_t`. FinalizationRegistry for GC-driven cleanup.
- **`audio.ts`** — `DAC` class wrapping `tropical_dac_t`. Static `listDevices()` for enumeration.
- **`param.ts`** — `Param` (smoothed) and `Trigger` (fire-once) classes wrapping `tropical_param_t`. `.asExpr()` returns a `SignalExpr` for use in wiring expressions.

---

## 9. MCP Server (`mcp/server.ts`)

The primary agent interface. Runs on stdio, uses `@modelcontextprotocol/sdk`.

Maintains a `SessionState` and exposes 16 consolidated tools (plus 15 deprecated aliases for backward compatibility):

**Program management**: `define_program`, `add_instance`, `remove_instance`, `list_programs`, `list_instances`, `get_info`

**Wiring**: `wire` (set and/or remove input wiring in a single recompile — replaces the former `connect_modules`, `disconnect_modules`, `set_module_input`, `set_inputs_batch`), `list_wiring`

**Audio output**: `set_output` (declarative — replaces full output list, replaces former `add_graph_output` / `remove_graph_output`)

**Control parameters**: `set_param`, `list_params`

**Program I/O**: `load` (accepts `tropical_program_1` or `tropical_patch_1`), `save` (outputs `tropical_program_1`), `merge`

**Audio control**: `start_audio`, `stop_audio`, `audio_status`

Every mutation that affects the signal graph calls `wire()` → `applyFlatPlan(session, runtime)`, which re-flattens and hot-swaps the kernel. The `wire` tool batches multiple set/remove operations into a single recompile.

Old tool names (`define_module`, `instantiate_module`, `connect_modules`, `load_patch`, etc.) are preserved as deprecated aliases that delegate to the new handlers.

---

## 10. Program Format (`tropical_program_1`)

The unified JSON format for all DSP programs. Validated by Zod schemas in `compiler/schema.ts`. Defined in `compiler/program.ts`.

```json
{
  "schema": "tropical_program_1",
  "name": "MyPatch",
  "programs": {
    "Gain": {
      "schema": "tropical_program_1",
      "name": "Gain",
      "inputs": ["audio", "cv"],
      "outputs": ["out"],
      "process": { "outputs": { "out": { "op": "mul", "args": [{"op":"input","name":"audio"}, {"op":"input","name":"cv"}] } } }
    }
  },
  "instances": {
    "osc": { "program": "VCO", "inputs": { "freq": 440 } },
    "amp": { "program": "Gain", "inputs": {
      "audio": { "op": "ref", "module": "osc", "output": "sin" },
      "cv": 0.5
    }}
  },
  "audio_outputs": [{ "instance": "amp", "output": "out" }],
  "params": [{ "name": "freq", "value": 440, "time_const": 0.005 }]
}
```

Key fields: `schema`, `name`, `inputs`/`outputs` (leaf/composite), `process` (leaf body), `programs` (inline subprogram definitions), `instances` (instantiated subprograms with wiring in `inputs`), `audio_outputs` (graph output routing), `params`, `regs`, `delays`, `config`, `type_defs`.

### Legacy: `tropical_patch_1`

The old patch format is still accepted by `loadJSON()` and converted internally via `convertPatchToProgram()`. The `save` MCP tool outputs `tropical_program_1`. Existing patches in `patches/` remain valid without modification.
```

---

## 11. Type System (`compiler/term.ts`, `compiler/type_check.ts`, `compiler/morphism_registry.ts`)

### Port Types (objects of the category)
- `Float`, `Int`, `Bool` — scalar types
- `ArrayType(element, shape)` — static-shape arrays with numpy-style broadcasting
- `StructType(name)`, `SumType(name)` — named algebraic data types
- `product(factors)` — tensor product (n-ary)
- `Unit` — monoidal unit (empty product)

### Shape Algebra
Numpy-style static broadcasting: shapes are right-aligned, dimension pairs must be equal or one must be 1. `broadcastShapes(a, b)` returns the broadcasted shape or null. `shapeStrides`, `shapeSize`, `flattenIndex` support row-major layout.

### Array Wiring (`compiler/array_wiring.ts`)
Validates connections between typed ports. Scalar-to-array connections auto-broadcast. Array-to-scalar connections are errors. Shape mismatches within compatible broadcast rules insert `broadcast_to` wrappers.

### Morphism Registry (`compiler/morphism_registry.ts`)
Registry for named type coercion morphisms (e.g. equal-temperament realization from integer pitch classes to float frequencies). The compiler can auto-insert canonical morphisms at composition boundaries.

---

## 12. Build System

### CMake (`CMakeLists.txt`)
- **Target**: `tropical_core` (shared library, output name `libtropical`)
- **C++ standard**: C++20
- **Dependencies**: LLVM >= 15 (ORC JIT), RtAudio (submodule in `lib/rtaudio`), nlohmann/json (submodule in `lib/json`)
- **Build type**: Default `RelWithDebInfo`
- **Test target**: `test_module_process` — exercises C API + JIT without audio device
- **Options**: `TROPICAL_BUILD_PYTHON`, `TROPICAL_PROFILE`, `TROPICAL_LLVM_STATIC`, `TROPICAL_BUILD_TESTS`

### Makefile
- `make build` — configure + build C++ core
- `make mcp-ts` — build + launch MCP server on stdio via Bun
- `make profile` — build with profiling instrumentation
- `make clean` — remove build directories

### Node/Bun (`package.json`)
- Runtime: Bun
- Key dependencies: `@modelcontextprotocol/sdk`, `koffi` (FFI), `ink`/`react` (TUI), `zod` (schema validation)
- Script: `bun run mcp/server.ts`

### MCP Configuration (`.mcp.json`)
```json
{ "mcpServers": { "tropical": { "command": "bun", "args": ["run", "mcp/server.ts"] } } }
```

---

## 13. Testing

### C++ Tests (`engine/tests/test_module_process.cpp`)
Custom harness (no framework dependency). Exercises FlatRuntime C API and JIT code paths without an audio device. Tests build `tropical_plan_4` JSON strings directly and assert on output buffer values.

Test cases cover: sawtooth oscillator, clock with array ratios, integer sequence stepping, multi-module fusion, smoothed params, trigger params.

Run: `cmake --build build -j4 && ctest --test-dir build`

### TypeScript Tests (`compiler/*.test.ts`)
Unit tests for: optimizer, flatten wiring, array wiring, plan generation, compiler, expression emission, lower_arrays. Run via Bun test runner.

---

## 14. Key Design Decisions

### Single-kernel fusion
The entire patch compiles to one native function. No module boundaries, no per-module dispatch, no interpreter. This eliminates call overhead and enables LLVM to optimize across the entire signal graph.

### Expression trees as universal IR
ExprNode JSON trees are the interchange format between all layers. Module definitions produce them, flattening manipulates them, emission converts them to instructions, and the C++ JIT compiles them to native code.

### Hot-swap via double-buffered kernels
Live audio never stops for recompilation. The new kernel is compiled on a background thread, state is transferred by matching register/array names, and a single atomic store switches the active kernel. At most one sample of stale state is read (inaudible).

### No interpreter fallback
JIT failures are fatal. This simplifies the runtime (no dual code paths) and ensures all audio runs at native speed. The tradeoff is that plan compilation errors must be caught before reaching the engine.

### Static shapes for arrays
All array shapes are known at compile time. This enables complete loop unrolling during array lowering and typed LLVM IR emission. No dynamic allocation on the audio thread.

### Inline transcendentals
sin, cos, exp, log, tanh are polynomial approximations emitted as inline LLVM IR. No libm dependency in kernels. Deterministic across platforms. Sufficient precision for audio (~1e-7 to 1e-10 max error).

### Thread-safe control parameters
`ControlParam` uses atomic load/store (relaxed ordering). Smoothed params apply one-pole lowpass per sample. Triggers use atomic exchange for fire-once semantics. Both are safe to write from any thread.
