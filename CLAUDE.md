# tropical

Realtime audio synthesis with LLVM ORC JIT. The entire program — a graph of DSP program instances — compiles to a single native kernel that runs per-sample in an audio callback. No interpreter, no module boundaries at runtime.

## Build

```bash
make build          # C++ core, outputs build/libtropical.dylib
make mcp-ts         # build + launch MCP server on stdio (requires Bun)
make profile        # build with profiling instrumentation
make clean          # remove build directories
```

**Requirements:** CMake 3.20+, C++20, LLVM >= 15 (Homebrew: `/opt/homebrew/opt/llvm`), Bun (for MCP/TUI).

## Test

```bash
cmake --build build -j4 && ctest --test-dir build   # C++ tests (JIT + C API)
bun test                                              # TS compiler tests
bun test --exclude compiler/apply_plan.test.ts        # TS tests without native FFI
```

C++ tests (`engine/tests/test_module_process.cpp`) exercise the C API and JIT without an audio device. TS tests (`compiler/*.test.ts`) cover the optimizer, flattening, array wiring, type checking, expression emission, and more.

**Note:** `apply_plan.test.ts` loads `build/libtropical.dylib` via FFI. If the native lib isn't built, Bun will segfault (null dereference at load time — not a test failure, a crash). Run `make build` first, or use the `--exclude` form above to run only pure TS tests.

## Architecture

Three layers, one stable boundary:

```
compiler/             TS: expression system, JSON stdlib loading, combinators,
                      flattening, instruction emission, type system, term IR
  compiler/runtime/   FFI bridge to C++ (koffi bindings, Runtime, DAC, Param)
engine/               C++: plan parsing, LLVM JIT, per-sample execution, audio output
  engine/c_api/       Stable C API — the boundary between TS and C++
  engine/jit/         LLVM ORC JIT engine
  engine/runtime/     FlatRuntime (plan loading, kernel execution)
  engine/dac/         Audio output (RtAudio)
  engine/expr/        C++ expression AST, rewriting, evaluation
mcp/                  MCP server — primary agent interface over stdio
patches/              Example patches (tropical_program_1 JSON)
stdlib/               Built-in program types as ProgramJSON files (19 types)
```

### Data flow

```
Program (JSON / MCP tools)
  → ProgramJSON loading + type registration (TS-only, no C++ calls)
  → Expression tree construction (ExprNode)
  → Flattening (inline all modules → single expression set)
  → Combinator expansion (unroll generate/fold/chain/etc.)
  → Array lowering (static-shape array ops → scalar primitives)
  → Instruction emission (ExprNode → FlatProgram)
  → JSON serialization (tropical_plan_4)
  ─── C API boundary (tropical_c.h, koffi FFI) ───
  → NumericProgramParser (JSON → FlatProgram struct)
  → JIT compilation (FlatProgram → LLVM IR → native kernel)
  → FlatRuntime (per-sample execution, double-buffered hot-swap)
  → Audio output (RtAudio / CoreAudio)
```

### Schema versions

There are four distinct JSON schemas — don't confuse them:

| Schema | Produced by | Purpose |
|--------|-------------|---------|
| `tropical_program_1` | `compiler/program.ts` | **Primary.** Unified program format (instances, wiring, subprograms, outputs) |
| `tropical_patch_1` | `compiler/patch.ts` | Legacy patch format (modules, wiring, outputs). Still accepted on load. |
| `tropical_plan_1` | `compiler/plan.ts` | Intermediate execution plan (legacy, used by some tests) |
| `tropical_plan_4` | `compiler/flatten.ts` + `emit_numeric.ts` | Flat instruction stream sent to C++ JIT — the one that matters for audio |

## Compiler layer (`compiler/`)

The TypeScript layer handles everything from module definition through instruction emission. No audio processing happens here.

**Expression system** (`expr.ts`) — `ExprNode` is the universal IR: a recursive JSON union type (number, boolean, array, or `{op, ...}` object). `SignalExpr` wraps it with static shape tracking. All module I/O is defined as expression trees. Includes compile-time combinator constructors (`generate`, `fold`, `chain`, `map2`, etc.) for iteration and abstraction without manual unrolling.

**Program schema** (`program.ts`) — `ProgramJSON` (`tropical_program_1`) is the unified representation. A program with `process` = leaf, with `instances` + `audio_outputs` = graph, with `instances` + `inputs` + `outputs` = reusable composite. `loadStdlib()` loads `stdlib/*.json` into the type registry.

**Program types** (`module.ts`) — Pure data types: `ProgramDef` (slot-indexed IR for the flattener), `ModuleType`, `ModuleInstance`. No DSL — types are built from ProgramJSON by `loadProgramDef()` in `patch.ts`.

**Standard library** (`stdlib/*.json`) — 19 built-in program types as ProgramJSON files (VCO, Clock, ADEnvelope, Reverb, LadderFilter, etc.). Loaded by `loadStdlib()` in `program.ts`.

**Flattening** (`flatten.ts`) — The critical step. Inlines all module expressions, resolves inter-module references, expands nested calls, converts delays to register ops. Output is a `tropical_plan_4` JSON. Uses WeakMap memoization for DAG sharing.

**Array lowering** (`lower_arrays.ts`) — Lowers first-class array ops (zeros, reshape, transpose, slice, reduce, broadcast_to, map, matmul) and compile-time combinators (generate, iterate, fold, scan, map2, zip_with, chain, let) to scalar primitives via static unrolling. All shapes are compile-time constants.

**Instruction emission** (`emit_numeric.ts`) — Walks flattened ExprNode trees, emits `FlatProgram` instruction stream with typed operands (const, input, reg, array_reg, state_reg, param, rate, tick).

**Term IR** (`term.ts`, `compiler.ts`) — Categorical IR (free monoidal category): morphism, compose, tensor, trace, id. The compiler builds terms from the dependency graph via topological sort (Kahn's) and cycle detection (Tarjan's SCC). Type checked via `type_check.ts`. Optimized via `optimizer.ts` (identity elimination, compose/tensor flattening).

**FFI bridge** (`runtime/bindings.ts`, `runtime.ts`, `audio.ts`, `param.ts`) — koffi declarations mirroring `tropical_c.h`. `Runtime` wraps `tropical_runtime_t` with FinalizationRegistry. `Param`/`Trigger` provide `.asExpr()` for wiring control parameters into expression trees.

## Engine (`engine/`)

C++20, header-heavy (templates + inlining for audio-thread performance).

**C API** (`c_api/tropical_c.h`) — Stable boundary. Opaque handles for FlatRuntime, ControlParam, DAC. Thread-local error string via `tropical_last_error()`. All external access goes through here.

**Plan parsing** (`runtime/NumericProgramParser.hpp`) — Thin deserializer: reads `tropical_plan_4` JSON → `FlatProgram` struct. No expression tree walking — just reads the pre-compiled instruction stream.

**JIT** (`jit/OrcJitEngine.hpp/.cpp`) — Singleton LLVM ORC engine. `compile_flat_program()` generates typed LLVM IR (f64/i64/i1 with explicit coercion), emits inline polynomial approximations for transcendentals (sin, cos, exp, log, tanh — no libm), and compiles to native code. Kernel object cache at `~/.cache/tropical/kernels/<build-id>/` with auto-invalidation on dylib rebuild.

**FlatRuntime** (`runtime/FlatRuntime.hpp/.cpp`) — Double-buffered kernel hot-swap. `load_plan()` compiles to the inactive slot, transfers matching state by register/array name, atomically swaps. `process()` runs the kernel, snapshots trigger params, applies smoothstep fade envelope (2048-sample Hermite curve).

**Audio output** (`dac/TropicalDAC.hpp`) — Templated RtAudio driver. Device watcher thread (50ms poll), disconnect recovery with 500ms backoff, callback timing stats.

**Expression AST** (`expr/`) — C++ parallel to the TS expression system. `Value` (tagged union), `ExprSpec` (tree nodes), plus rewriting (constant folding, algebraic simplification) and structural hashing. Used during plan compilation, not at runtime.

## MCP server (`mcp/server.ts`)

Primary agent interface. Runs on stdio via `@modelcontextprotocol/sdk`. Maintains `SessionState` and exposes 16 consolidated tools for program management, wiring, audio control, and program I/O. Key tools: `define_program`, `add_instance`, `wire` (batched set/remove), `set_output`, `load`/`save`/`merge`. Old tool names (`define_module`, `connect_modules`, `load_patch`, etc.) are preserved as deprecated aliases. Every wiring mutation triggers `wire()` → `applyFlatPlan()` → full recompile and hot-swap.

## Conventions

- Commit messages: `type(scope): description` (e.g., `fix(jit):`, `feat(compiler):`, `refactor:`)
- Module types: PascalCase (`VCO`, `ADEnvelope`, `Clock`)
- Input/output names: lowercase (`freq`, `signal`, `out`, `saw`)
- C++ is header-heavy by design (templates, inlining for audio perf)
- JIT failures are fatal — no interpreter fallback
