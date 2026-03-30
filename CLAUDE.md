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
  expr/       Expression AST, evaluator, rewrite/optimization passes
  graph/      Graph runtime, Module class, JIT compilation pipeline
  jit/        LLVM ORC JIT engine
  dac/        Audio output (RtAudio)
  c_api/      Stable C API (egress_c.h) — all external access goes through here

tui/
  src/
    server.ts          MCP server (stdio transport)
    index.tsx          TUI entry point (Ink/React)
    bindings.ts        koffi FFI to libegress
    module_library.ts  Built-in module type definitions (VCO, Clock, etc.)
    module.ts          Module spec builder DSL
    expr.ts            Expression builder functions
    patch.ts           Patch serialization/deserialization
```

**Data flow:** TypeScript DSL → C API (via koffi FFI) → Graph → Module → Expr → JIT → audio callback

**Key boundary:** the C API in `src/c_api/egress_c.h` is the only interface between the TypeScript layer and the C++ core. All module definitions, graph operations, and audio control go through it.

## Core concepts

- **Graph** stores modules, per-input expression trees, output taps, and the output buffer.
- **Module** has named inputs, outputs, and optional registers. Inputs reset to defaults after each sample. Outputs clip to [-10.0, 10.0].
- **Expressions** are symbolic trees (leaves = literals or output refs). They define module I/O behavior and get JIT-compiled to native code.
- **Single-sample boundary:** connected modules always read previous-sample outputs. No implicit multi-sample delay.
- **Module types** are defined in `tui/src/module_library.ts` using the DSL in `module.ts`. Each type specifies inputs, outputs, registers, and a process function that builds expression trees.

## JIT pipeline

Expression trees → static type inference (float/int/bool) → LLVM IR emission → ORC JIT compilation → cached native kernel. Disk caching avoids recompilation across runs.

JIT failures are **fatal** (no interpreter fallback). If a module fails to compile, it throws.

## Large files

`ModuleNumericJitMethods.hpp` and `GraphRuntimeMethods.hpp` are 100K+ lines each (template-expanded JIT and runtime methods). Do not read them in full — search for specific functions instead.

## Patch format

Patches are JSON files in `patches/`. Schema version: `egress_patch_1`.

```json
{
  "schema": "egress_patch_1",
  "modules": [{"type": "VCO", "name": "VCO1"}, ...],
  "outputs": [{"module": "VCA1", "output": "out"}, ...],
  "connections": [{"target": "VCA1.signal", "expr": "VCO1.saw"}, ...]
}
```

## MCP server

`make mcp-ts` starts the MCP server on stdio. It exposes the full graph API as tools: instantiate modules, connect them, set parameters, control audio, save/load patches. This is the primary agent interface for building patches at runtime.

## Conventions

- Commit messages: `type(scope): description` (e.g., `fix(jit):`, `feat(tui):`, `refactor:`)
- Module types are PascalCase (`VCO`, `ADEnvelope`, `Clock`)
- Input/output names are lowercase (`freq`, `signal`, `out`, `saw`)
- C++ is header-heavy by design (templates, inlining for audio perf)
