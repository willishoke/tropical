# tui/

TypeScript layer for egress. Two entry points sharing the same bindings:

- `src/index.tsx` — full-screen TUI (Ink/React), launched via `make tui-ts`
- `src/server.ts` — MCP server on stdio, launched via `make mcp-ts`

## Runtime

Uses **Bun** (not Node). Install deps with `bun install`. Type-check with `bun run tsc --noEmit`.

## FFI bridge

`bindings.ts` uses **koffi** to load `libegress.dylib` and call the C API directly. It mirrors the function signatures in `src/c_api/egress_c.h`. When the C API changes, `bindings.ts` must be updated to match.

Library search order: `egress/` dir → project root → `build*/` subdirectories.

## Module definitions

Module types live in `module_library.ts`. Each type is built with the DSL in `module.ts`:

- `defineModule(name, inputs, outputs, registers, processFn)` — stateful module with registers
- `definePureFunction(inputs, outputs, processFn)` — stateless, can be inlined as nested calls

The process function receives `SymbolMap`s for inputs and registers and returns output expressions + next register values. These are symbolic expression trees, not runtime values.

## Expression DSL

`expr.ts` exports arithmetic builders (`add`, `sub`, `mul`, `div`, `mod`, `pow_`, etc.), comparisons (`lt`, `gt`, `gte`, `lte`), and special forms (`delay`, `sampleRate`, `sampleIndex`, `arrayPack`). Expressions are JSON-serializable trees sent to the C API.

## Patch serialization

`patch.ts` handles loading and saving patches as JSON. A `SessionState` holds the graph handle, module instances, type registry, and output taps. `makeSession()` creates a fresh session; `loadPatchFromJSON()` populates it from a patch file.
