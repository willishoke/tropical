# tropical

C++ realtime audio synthesis library with LLVM ORC JIT, a TypeScript TUI (Ink/React), and an MCP server for programmatic control.

## Build

```bash
make build          # C++ core with LLVM JIT, outputs build/libtropical.dylib
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
engine/       C++ execution: JIT (OrcJitEngine), DAC (RtAudio), FlatRuntime, C API
compiler/     TS language layer: expression DSL, module library, type system,
              compilation pipeline (flatten, plan, optimizer, type_check, term)
              compiler/runtime/  FFI bridge to engine (bindings, audio, param)
mcp/          MCP server — primary agent interface, AI-native delivery surface
```

**Data flow:** TypeScript DSL → expression trees (ExprNode JSON) → flatten → tropical_plan_2 JSON → C++ FlatRuntime → JIT → audio callback

**Key boundary:** the C API in `engine/c_api/tropical_c.h` is the interface between the compiler layer and C++. It exposes: FlatRuntime (plan loading, audio processing), ControlParam (smoothed/trigger parameters), DAC (audio output), and device enumeration. Module definitions and expression building happen entirely in TypeScript.

## Core concepts

- **FlatRuntime** receives a flat JSON plan (tropical_plan_2), JIT-compiles all expressions into a single native kernel, and runs it per-sample. No module boundaries at runtime.
- **Module types** are defined in `compiler/module_library.ts` using the DSL in `module.ts`. Each type specifies inputs, outputs, registers, and a process function that builds expression trees. Instantiation is TS-only — no C API calls.
- **Expressions** are symbolic JSON trees (ExprNode). They define module I/O behavior and get flattened + JIT-compiled to native code.
- **Flattening** (`compiler/flatten.ts`) inlines all module expression trees, resolves inter-module references, and produces a single tropical_plan_2 with flat output/register expression arrays.
- **Single-sample boundary:** connected modules always read previous-sample outputs. No implicit multi-sample delay.

## JIT pipeline

Expression trees → PlanParser (JSON → ExprSpec) → ExprCompiler (ExprSpec → CompiledProgram) → NumericProgramBuilder (→ NumericProgram) → LLVM IR emission → ORC JIT compilation → native kernel.

JIT failures are **fatal** (no interpreter fallback). If a plan fails to compile, it throws.

## Patch format

Patches are JSON files in `patches/`. Schema version: `tropical_patch_1`.

```json
{
  "schema": "tropical_patch_1",
  "modules": [{"type": "VCO", "name": "VCO1"}, ...],
  "outputs": [{"module": "VCA1", "output": "out"}, ...],
  "input_exprs": [{"module": "VCA1", "input": "audio", "expr": {"op": "ref", "module": "VCO1", "output": "saw"}}, ...]
}
```

## MCP server

`make mcp-ts` starts the MCP server on stdio. It exposes tools to: define module types, instantiate modules, connect them, set parameters, control audio, save/load patches. This is the primary agent interface for building patches at runtime.

## Signal levels and gain staging

All audio signals use a **±1 convention**. VCO outputs (saw, tri, sin, sqr) range from -1 to +1. The JIT output mixer sums graph outputs with **no automatic scaling** — what the patch produces is what hits the DAC.

**Gain staging is the patch's responsibility.** When summing N voices, scale by `1/N` or use VCAs to keep the mix within ±1. Exceeding ±1 will clip at the audio output. For example, 5 VCAs summed should be wrapped in `mul(sum, 0.2)`.

## Conventions

- Commit messages: `type(scope): description` (e.g., `fix(jit):`, `feat(tui):`, `refactor:`)
- Module types are PascalCase (`VCO`, `ADEnvelope`, `Clock`)
- Input/output names are lowercase (`freq`, `signal`, `out`, `saw`)
- C++ is header-heavy by design (templates, inlining for audio perf)
