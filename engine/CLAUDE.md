# src/

C++ core of tropical. C++17, header-heavy by design (templates + inlining for audio-thread performance).

## Layout

```
runtime/  FlatRuntime, NumericProgramBuilder, PlanParser, ExprCompiler
expr/     Expression AST, evaluation, structural ops, rewrite/optimization
graph/    GraphTypes.hpp only (core Value/ExprKind type aliases)
jit/      LLVM ORC JIT engine (OrcJitEngine.hpp/.cpp)
dac/      Audio output via RtAudio (TropicalDAC.hpp)
c_api/    Stable C API (tropical_c.h)
```

## C API boundary

All external access (TypeScript FFI, tests) goes through `c_api/tropical_c.h`. This exposes:
- **FlatRuntime** — `tropical_runtime_*` (create, load plan, process, output buffer, fade control)
- **ControlParam** — `tropical_param_*` (smoothed params and triggers for thread-safe control)
- **DAC** — `tropical_dac_*` (audio output backed by FlatRuntime)
- **Device enumeration** — `tropical_audio_*`

## Expression pipeline

1. **AST** (`expr/Expr.hpp`) — tagged union tree: literals, binary/unary ops, refs, arrays, delay nodes
2. **Structural ops** (`expr/ExprStructural.hpp/.cpp`) — reference collection, substitution, structural equality
3. **Rewrite** (`expr/ExprRewrite.cpp`) — constant folding, dead code elimination, algebraic simplification
4. **Eval** (`expr/ExprEval.hpp`) — interpreter (used during module definition, not at runtime)

## FlatRuntime compilation pipeline

When a plan is loaded via `tropical_runtime_load_plan()`:

1. **PlanParser** (`runtime/PlanParser.hpp`) — JSON → ExprSpec trees
2. **ExprCompiler** (`runtime/ExprCompiler.hpp`) — ExprSpec → CompiledProgram (instruction stream)
3. **NumericProgramBuilder** (`runtime/NumericProgramBuilder.hpp`) — CompiledProgram → NumericProgram (JIT IR)
4. **OrcJitEngine** (`jit/OrcJitEngine.cpp`) — NumericProgram → native kernel via LLVM ORC
5. **FlatRuntime** (`runtime/FlatRuntime.hpp`) — double-buffered kernel hot-swap, per-sample execution

JIT failures are fatal (no interpreter fallback at runtime).

## Adding a new module type

Module types are defined in TypeScript (`tui/src/module_library.ts`), not in C++. The C++ core is generic — it processes arbitrary expression trees. To add a module:

1. Define it in `tui/src/module_library.ts` using `defineModule()` or `definePureFunction()`
2. Register it in `loadBuiltins()` in the same file
3. No C++ changes needed unless you need a new expression node type

## Type system

`graph/GraphTypes.hpp` defines the core type aliases. Values are tagged unions supporting float, int, bool, array, and matrix types. The JIT pipeline infers static types to emit typed LLVM IR rather than falling back to dynamic dispatch.
