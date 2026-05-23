# tests/bench

Benchmarks for the compile pipeline and the audio backends. Runnable
via `bun run`, not `bun test` — these are scripts, not assertions.

| File              | What it measures |
|-------------------|------------------|
| `compile.ts`               | TS-pipeline-only cost. Times `compileSession` end-to-end on a fixed 13-module patch and per-module to find bottlenecks. |
| `jit_runtime.ts`           | End-to-end: TS pipeline → JSON.stringify → JIT loadPlan → `runtime.process` loop. Per-patch ns/sample and realtime ratio. Wipes the kernel cache for cold compiles. |
| `wasm_vs_jit.ts`           | Head-to-head: same FlatPlan compiled through both backends. Compile-time (TS, WASM emit/instantiate, JIT load) and runtime (ns/sample). |
| `microkernel_vs_fused.ts`  | Head-to-head: same session compiled in `fused` mode vs `microkernel` mode. Per-mode ns/sample, compile latency, and rt-ratio across small/medium/polyphony cases. |
| `corpus.ts`                | Shared list of patches that currently compile through `compileSession`. Other patches in `patches/` are blocked on known limitations (see file). |

| File              | What it captures |
|-------------------|------------------|
| `RESULTS.md`      | Frozen-in-time snapshot of bench numbers tied to a specific commit. Re-capture against a new commit when the engine's emit strategy materially shifts. |

## Quick reference

```bash
# Compile-time profile of a baked-in 13-module patch
bun run tests/bench/compile.ts

# End-to-end JIT timing on one or more patches
bun run tests/bench/jit_runtime.ts patches/cross_fm_4.json --frames=4096

# WASM vs JIT, multiple patches
bun run tests/bench/wasm_vs_jit.ts patches/bubble_cloud.json patches/cross_fm_4.json patches/cross_fm_evolved.json --frames=2048

# Microkernel vs fused, all built-in cases
bun run tests/bench/microkernel_vs_fused.ts
```

## What "ns/sample" means

The audio sample period at 44.1 kHz is ~22,676 ns. A kernel using
e.g. 400 ns/sample consumes ~1.8% of the realtime budget per voice;
the `rt%` column in the bench output reports this directly. Anything
approaching 100% will xrun.

## WASM runtime caveat

`wasm_vs_jit.ts` runs WASM through Bun's WebAssembly runtime
(JavaScriptCore's B3/OMG compilers). That's production-quality but
not the strongest WASM engine — V8 TurboFan and wasmtime+cranelift at
`OptLevel::Speed` will land in a similar range, often slightly
faster. Treat the WASM numbers as "browser-class" performance.
