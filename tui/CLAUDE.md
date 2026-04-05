# tui/

TypeScript layer for egress. Two entry points sharing the same bindings:

- `src/index.tsx` — full-screen TUI (Ink/React), launched via `make tui-ts`
- `src/server.ts` — MCP server on stdio, launched via `make mcp-ts`

## Runtime

Uses **Bun** (not Node). Install deps with `bun install`. Type-check with `bun run tsc --noEmit`.

## FFI bridge

`bindings.ts` uses **koffi** to load `libegress.dylib` and call the C API directly. It mirrors the function signatures in `src/c_api/egress_c.h`. The bindings cover:
- `egress_param_*` — smoothed/trigger parameter lifecycle
- `egress_runtime_*` — FlatRuntime (plan loading, processing, output buffer)
- `egress_dac_*` — audio output (backed by FlatRuntime)
- `egress_audio_*` — device enumeration

Library search order: `egress/` dir → project root → `build*/` subdirectories.

## Module definitions

Module types live in `module_library.ts`. Each type is built with the DSL in `module.ts`:

- `defineModule(name, inputs, outputs, registers, processFn)` — stateful module with registers
- `definePureFunction(inputs, outputs, processFn)` — stateless, can be inlined as nested calls

The process function receives `SymbolMap`s for inputs and registers and returns output expressions + next register values. These are symbolic expression trees (ExprNode), not runtime values. Module instantiation is TS-only — no C API calls.

## Expression DSL

`expr.ts` exports arithmetic builders (`add`, `sub`, `mul`, `div`, `mod`, `pow_`, etc.), comparisons (`lt`, `gt`, `gte`, `lte`), and special forms (`delay`, `sampleRate`, `sampleIndex`, `arrayPack`). Expressions are pure JSON-serializable trees (ExprNode) — no C handles.

## Patch serialization

`patch.ts` handles loading and saving patches as JSON. A `SessionState` holds the runtime handle, module instances, type registry, and output taps. `makeSession()` creates a fresh session with a FlatRuntime; `loadPatchFromJSON()` populates it from a patch file.

## Flattening pipeline

`flatten.ts` takes a SessionState and produces an `egress_plan_2` JSON plan where all module expression trees are inlined: input references are substituted with wiring expressions, inter-module refs are resolved by inlining the referenced module's output expression. The result is a single flat set of output_exprs and register_exprs with no module boundaries.

`apply_plan.ts` orchestrates: SessionState → flattenPatch() → JSON → runtime.loadPlan().
