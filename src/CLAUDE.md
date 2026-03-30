# src/

C++ core of egress. C++17, header-heavy by design (templates + inlining for audio-thread performance).

## Layout

```
c_api/    Stable C API — the ONLY external interface to the core
expr/     Expression AST, evaluation, structural ops, rewrite/optimization
graph/    Graph runtime, Module class, numeric JIT compilation
jit/      LLVM ORC JIT engine (OrcJitEngine.hpp/.cpp)
dac/      Audio output via RtAudio (EgressDAC.hpp)
```

## C API boundary

All external access (TypeScript FFI, Python ctypes, tests) goes through `c_api/egress_c.h`. This is the stable interface. Internal C++ classes are not exported.

## Expression pipeline

1. **AST** (`expr/Expr.hpp`) — tagged union tree: literals, binary/unary ops, refs, arrays, delay nodes
2. **Structural ops** (`expr/ExprStructural.hpp/.cpp`) — reference collection, substitution, structural equality
3. **Rewrite** (`expr/ExprRewrite.cpp`) — constant folding, dead code elimination, algebraic simplification
4. **Eval** (`expr/ExprEval.hpp`) — interpreter (used during module definition, not at runtime)

## Module compilation

When a module is instantiated, its expression trees go through:

1. Type inference — static float/int/bool types propagated through the expression tree
2. LLVM IR emission — expressions lowered to LLVM IR in `ModuleNumericJitMethods.hpp`
3. ORC JIT compilation — `OrcJitEngine.cpp` compiles IR to native code
4. Disk caching — compiled kernels cached by content hash to avoid recompilation

JIT failures are fatal (no interpreter fallback at runtime).

## Large files warning

These files are 100K+ lines of template-expanded methods. **Never read them in full.** Search for specific function names instead:

- `graph/ModuleNumericJitMethods.hpp` — JIT IR emission for all expression node types
- `graph/GraphRuntimeMethods.hpp` — fused graph kernel compilation and runtime dispatch

## Adding a new module type

Module types are defined in TypeScript (`tui/src/module_library.ts`), not in C++. The C++ core is generic — it processes arbitrary expression trees. To add a module:

1. Define it in `tui/src/module_library.ts` using `defineModule()` or `definePureFunction()`
2. Register it in `loadBuiltins()` in the same file
3. No C++ changes needed unless you need a new expression node type

## Type system

`GraphTypes.hpp` defines the core type aliases. Values are tagged unions supporting float, int, bool, array, and matrix types. The JIT pipeline infers static types to emit typed LLVM IR rather than falling back to dynamic dispatch.
