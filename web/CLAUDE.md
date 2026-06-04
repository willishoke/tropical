# web/

Browser backend. A complete co-implementation of the audio runtime,
sitting alongside the C++ JIT. Same compiler front-end, same strata
pipeline, same `tropical_plan_5` boundary; the only thing that
differs is the emit target (WebAssembly vs. LLVM IR) and the param
handle representation (SAB slot index vs. native pointer).

The web backend is held to sample-for-sample equivalence with the
JIT. See the `wasm_*` test suites in `compiler/`.

## Layout

```
web/
  build_patches.ts       Offline plan precompilation: tropical_program_2 → web/dist/patches/*.plan.json
  build.ts               Full demo bundle: regen stdlib_bundled, precompile patches,
                         bundle worklet + main app, copy index.html → web/dist/
  bundle_stdlib.ts       Generates compiler/stdlib_bundled.ts from stdlib/*.md
  dev.ts                 Static dev server with COOP/COEP headers (SAB requirement)
  host/                  Main thread (UI, plan compile, param updates)
    compiler.ts          compilePlan(FlatPlan) → LoadedPlan via emit_wasm
    context.ts           AudioContext + AudioWorkletNode wiring
    params.ts            ParamBank (SharedArrayBuffer), WebParam
  worklet/               Audio thread (real-time render)
    runtime.ts           WasmRuntime: dual-slot hot-swap, fade envelope, snapshotParams
    processor.ts         AudioWorkletProcessor delegate; postMessage protocol
  site/                  Browser UI
    app.ts               Main-thread app: patch picker, play/stop
    index.html           Shell that loads app.js and worklet.js
```

## Where this fits in the bigger pipeline

The TS compiler pipeline (`parse → elaborate → strata`) runs offline,
on the build host. We cannot run it inside the browser bundle today
because its transitive imports pull in `koffi` (used by the native
runtime path). The output of that pipeline — `tropical_plan_5` JSON —
is the boundary that crosses into the browser:

```
build host (Bun + native)               browser
  parse/elaborate/strata     →           fetch /patches/<slug>.plan.json
  → compileSession                       → compilePlan(plan)
  → tropical_plan_5 JSON                    → emitWasm → WebAssembly bytes
  written to web/dist/patches/              → postMessage to AudioWorklet
                                            → WebAssembly.instantiate
                                            → WasmRuntime.process per block
```

Building runs the whole front-end on a chosen patch and produces a
plan; the browser only runs the WASM emitter and the runtime. The
WASM emitter (`compiler/emit_wasm.ts`) and the WASM runtime
(`web/worklet/runtime.ts`) are the two halves of the second backend
off post-strata `ResolvedProgram` (the other is `compileResolved` →
`tropical_plan_5` → JIT).

## Linear-memory layout

`compiler/wasm_memory_layout.ts` defines the byte regions of the WASM
module's exported `memory`. Layout is shared between the emitter and
the runtime so offsets stay in sync.

```
offset 0
  inputs        f64[inputCount]                — set by host, kernel reads
  registers     i64[registerCount]             — kernel state (float bitcast / int / bool)
  temps         i64[registerCount]             — per-sample scratch (same encoding)
  arrays        f64[arraySlotSizes...]         — array-typed register backing stores
  param_table   f64[paramCount]                — host writes per-block snapshot of Param.value
  param_frame   f64[paramCount]                — host writes per-block snapshot (legacy trigger.frame_value slot, retained for backcompat)
  output        f64[maxBlockSize]              — kernel writes mono audio out
```

All regions are 8-byte aligned. The encoding mirrors the native
engine: `i64` cells store either a float bitcast, a signed int, or a
zero-extended bool; the op's `result_type` (per `tropical_plan_5`
instruction) tells the codegen which load/store to emit.

## Param flow

`web/host/params.ts` maintains a `ParamBank` over a
`SharedArrayBuffer` (or plain `ArrayBuffer` if COOP/COEP isn't
configured). Two f64 slots per param: `[value, frame_value]`.

```
JS main thread:                 worklet (audio thread):           WASM kernel:
  WebParam.value = 440            WasmRuntime.snapshotParams         f64.load
  → bank.view[i*2] = 440          reads bank.view[i*2]               from param_table
                                  writes WASM mem at                  per sample
                                  param_table + i*8
```

The slot index is the param's handle. Wiring expressions (in
`tropical_program_2` and through MCP) reference parameters by **name**,
the session compiler turns the name into the `WebParam._h` (slot index),
and `emit_wasm.ts` stringifies that index to a `param.ptr` field in
the plan. The WASM kernel emits `f64.load (paramTableOffset + ptr*8)`
for `param` operands.

This is the same `param.ptr` shape the native plan uses, just
populated with a SAB index instead of a native `tropical_param_t*`.
The `tropical_plan_5` schema is backend-agnostic on this axis.

## Hot-swap

`WasmRuntime` (`worklet/runtime.ts`) holds two `Slot`s and an active
index. `loadPlan(plan)` instantiates the new WASM module, initializes
its `register` region from `state_init`, transfers matching state from
the outgoing slot (by register/array name, type-checked), atomically
flips `activeIdx`, and starts a 2048-sample smoothstep fade-in. Same
shape as `engine/runtime/FlatRuntime.cpp`'s state transfer.

Fade envelope is a Hermite smoothstep `t² · (3 − 2t)` over
`FADE_SAMPLES = 2048`, applied per sample to the f64 output as it's
copied into the f32 worklet output buffer.

## Worklet protocol

`worklet/processor.ts` runs in `AudioWorkletGlobalScope` and receives
messages:

| Message | Source | Effect |
|---------|--------|--------|
| `{type: 'init', paramsSab, maxParams}` | main thread, once at startup | Construct `WasmRuntime` with the SAB view |
| `{type: 'load', plan: LoadedPlan}` | main thread, on patch change | `compilePlan` output crossing the port; runtime instantiates and hot-swaps |
| `{type: 'fadeIn'}` / `{type: 'fadeOut'}` | main thread | Trigger fade envelope on the active slot |

The processor's `process()` is invoked by the browser every 128
samples; it delegates to `WasmRuntime.process()`. We post raw WASM
*bytes* (not a `WebAssembly.Module`) because Chrome silently drops
worklet messages containing pre-compiled `WebAssembly.Module`
objects; the worklet does the `WebAssembly.instantiate` itself.

## Stdlib in the browser

The browser cannot read from disk, so each `stdlib/*.md` code block is inlined into
`compiler/stdlib_bundled.ts` by `web/bundle_stdlib.ts` and consumed
by `loadStdlibFromSources()` in `compiler/stdlib_loader.ts`. The
`build.ts` step regenerates that file before bundling so it stays in
sync with `stdlib/`.

For the demo build the full TS pipeline doesn't actually run in the
browser — patches are precompiled at build time by
`web/build_patches.ts` and shipped as `tropical_plan_5` JSON. The
in-browser compile path exists (and the bundled stdlib feeds it) but
is not on the demo's hot path today.

## Equivalence gates

Three test suites lock the WASM backend to the JIT:

- `compiler/wasm_runtime.test.ts` — `WasmRuntime` in isolation (no
  AudioWorklet): block-driven render, fade envelope, state-transfer
  invariants.
- `tests/equiv/wasm_vs_jit.test.ts` — same `tropical_plan_5` through
  both backends; sample-for-sample agreement required.
- `tests/equiv/web_plans_vs_jit.test.ts` — every precompiled plan in
  `web/dist/patches/` matches the JIT output. Run after
  `bun web/build_patches.ts` and `make build`.

Any divergence is a bug in either `emit_wasm.ts`, the WASM runtime,
or the underlying strata pipeline (the latter shows up in
`tests/equiv/wasm_vs_jit`).

## Build / run

```bash
bun web/build_patches.ts    # → web/dist/patches/*.plan.json + index.json
bun web/build.ts            # full bundle: stdlib_bundled regen + patches + worklet + app + index.html
bun web/dev.ts              # static server with COOP/COEP for SAB
```

`bun web/dev.ts` is required for local development because
`SharedArrayBuffer` needs `Cross-Origin-Opener-Policy: same-origin` +
`Cross-Origin-Embedder-Policy: require-corp`. Without those headers
the runtime falls back to a plain `ArrayBuffer` (init-time param
snapshot only, no live updates across the worklet boundary).
