# egress

C++ realtime audio synthesis library with LLVM ORC JIT, a TypeScript TUI (Ink/React), and an MCP server for programmatic control.

## Build

```bash
make build          # C++ core with LLVM JIT, outputs build/libegress.dylib
make tui-ts         # build + launch full-screen TUI (requires Bun)
make mcp-ts         # build + launch MCP server on stdio
make debug          # build with tests enabled
```

**Requirements:** CMake 3.20+, C++17, LLVM >= 15 (Homebrew: `/opt/homebrew/opt/llvm`), Bun (for TUI/MCP).
Only tested on macOS (CoreAudio). Linux builds in CI without JIT.

## Test

```bash
cmake --build build -j4 && ctest --test-dir build
```

Tests run via `test_module_process` — exercises the C API and JIT code paths without an audio device. JIT is **off** in CI (no LLVM on runners); local testing with `make build` has it on.

## Architecture

```
src/
  runtime/    FlatRuntime (plan executor), NumericProgramBuilder, PlanParser, ExprCompiler
  expr/       Expression AST, evaluator, rewrite/optimization passes
  graph/      GraphTypes.hpp (core Value/ExprKind type aliases — the only surviving file)
  jit/        LLVM ORC JIT engine (OrcJitEngine)
  dac/        Audio output (RtAudio), templated for any AudioSource
  c_api/      Stable C API (egress_c.h) — all external access goes through here

tui/
  src/
    server.ts          MCP server (stdio transport)
    index.tsx          TUI entry point (Ink/React)
    bindings.ts        koffi FFI to libegress (param, runtime, DAC, device APIs)
    module_library.ts  Built-in module type definitions (VCO, Clock, etc.)
    module.ts          Module type DSL — builds expression trees, TS-only instantiation
    expr.ts            Expression builder functions (pure ExprNode trees, no C handles)
    patch.ts           Patch serialization/deserialization, session state
    flatten.ts         Flatten patch into egress_plan_2 (inline all module expressions)
    runtime.ts         FlatRuntime wrapper
```

**Data flow:** TypeScript DSL → expression trees (ExprNode JSON) → flatten → egress_plan_2 JSON → C++ FlatRuntime → JIT → audio callback

**Key boundary:** the C API in `src/c_api/egress_c.h` is the interface between TypeScript and C++. It exposes: FlatRuntime (plan loading, audio processing), ControlParam (smoothed/trigger parameters), DAC (audio output), and device enumeration. Module definitions and expression building happen entirely in TypeScript.

## Core concepts

- **FlatRuntime** receives a flat JSON plan (egress_plan_2), JIT-compiles all expressions into a single native kernel, and runs it per-sample. No module boundaries at runtime.
- **Module types** are defined in `tui/src/module_library.ts` using the DSL in `module.ts`. Each type specifies inputs, outputs, registers, and a process function that builds expression trees. Instantiation is TS-only — no C API calls.
- **Expressions** are symbolic JSON trees (ExprNode). They define module I/O behavior and get flattened + JIT-compiled to native code.
- **Flattening** (`flatten.ts`) inlines all module expression trees, resolves inter-module references, and produces a single egress_plan_2 with flat output/register expression arrays.
- **Single-sample boundary:** connected modules always read previous-sample outputs. No implicit multi-sample delay.

## JIT pipeline

Expression trees → PlanParser (JSON → ExprSpec) → ExprCompiler (ExprSpec → CompiledProgram) → NumericProgramBuilder (→ NumericProgram) → LLVM IR emission → ORC JIT compilation → native kernel.

JIT failures are **fatal** (no interpreter fallback). If a plan fails to compile, it throws.

## Patch format

Patches are JSON files in `patches/`. Schema version: `egress_patch_1`.

```json
{
  "schema": "egress_patch_1",
  "modules": [{"type": "VCO", "name": "VCO1"}, ...],
  "outputs": [{"module": "VCA1", "output": "out"}, ...],
  "input_exprs": [{"module": "VCA1", "input": "audio", "expr": {"op": "ref", "module": "VCO1", "output": "saw"}}, ...]
}
```

## MCP server

`make mcp-ts` starts the MCP server on stdio. It exposes tools to: define module types, instantiate modules, connect them, set parameters, control audio, save/load patches. This is the primary agent interface for building patches at runtime.

## Conventions

- Commit messages: `type(scope): description` (e.g., `fix(jit):`, `feat(tui):`, `refactor:`)
- Module types are PascalCase (`VCO`, `ADEnvelope`, `Clock`)
- Input/output names are lowercase (`freq`, `signal`, `out`, `saw`)
- C++ is header-heavy by design (templates, inlining for audio perf)
