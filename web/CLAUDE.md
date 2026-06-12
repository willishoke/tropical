# web/

Browser backend. A complete co-implementation of the audio runtime,
sitting alongside the C++ JIT. Same `tropical_plan_5` boundary; the
only thing that differs is the emit target (WebAssembly vs. LLVM IR)
and the param handle representation (SAB slot index vs. native
pointer). The plan itself is produced by the Lean engine — the
browser never runs a compiler front-end.

The web backend is held to sample-for-sample equivalence with the
JIT. See the `tests/web/` suites.

## Layout

```
web/
  wasm/                  The WASM backend (no compiler/ dependency)
    emit_wasm.ts         tropical_plan_5 → WebAssembly bytes + linear-memory layout
    wasm_memory_layout.ts  byte regions of the WASM module's exported memory
    flat_plan.ts         tropical_plan_5 parse (parseWirePlan → FlatPlan)
    plan_types.ts        plan instruction / sink / source types
    slot_indices.ts      slot-index helpers shared by emitter and layout
  patches/               Source demo patches (tropical_program_2 JSON) + manifest.json
  build_patches.ts       Precompile each web/patches/*.json via the Lean engine
                         (diffcli compile --mode=fused) → web/dist/patches/*.plan.json
  build.ts               Full demo bundle: precompile patches, bundle worklet + main app,
                         copy index.html → web/dist/
  dev.ts                 Static dev server with COOP/COEP headers (SAB requirement)
  host/                  Main thread (UI, plan compile, param updates)
    compiler.ts          compilePlan(FlatPlan) → LoadedPlan via wasm/emit_wasm
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

The compiler is the Lean engine — the same engine code that serves
MCP (the `frontend` binary), here invoked through the `diffcli compile`
CLI offline on the build host. Its output — `tropical_plan_5` JSON — is
the boundary that crosses into the browser:

```
build host (Lean + Bun)                 browser
  diffcli compile <patch>    →           fetch /patches/<slug>.plan.json
    --mode=fused                         → compilePlan(plan)
  → tropical_plan_5 JSON                    → emitWasm → WebAssembly bytes
  written to web/dist/patches/              → postMessage to AudioWorklet
                                            → WebAssembly.instantiate
                                            → WasmRuntime.process per block
```

`web/build_patches.ts` shells `diffcli compile` over the source
patches in `web/patches/`; the browser only runs the WASM emitter and
the runtime. The WASM emitter (`web/wasm/emit_wasm.ts`) and the WASM
runtime (`web/worklet/runtime.ts`) are the two halves of the second
backend off `tropical_plan_5` (the other is the C++ JIT).

## Linear-memory layout

`web/wasm/wasm_memory_layout.ts` defines the byte regions of the WASM
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
and `web/wasm/emit_wasm.ts` stringifies that index to a `param.ptr`
field in the plan. The WASM kernel emits
`f64.load (paramTableOffset + ptr*8)` for `param` operands.

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

The browser never sees stdlib source. Patches are compiled to
`tropical_plan_5` ahead of time by the Lean engine (which resolves
every stdlib program type itself, from the committed
`stdlib/parsed/*.json` bridge), so the plan that ships already has
every composite inlined to scalar instructions. The browser fetches
the precompiled plan and runs only the WASM emitter + runtime — there
is no in-browser compile path and no bundled stdlib.

## Equivalence gates

Two test suites lock the WASM backend to the JIT (both in `tests/web/`,
both off `tropical_plan_5`; the native side is the Lean engine's JIT
via `diffcli render-bytes`):

- `tests/web/wasm_vs_jit.test.ts` — a hand-built patch is compiled by
  `diffcli compile --mode=fused`, then run through both backends (WASM
  side = `web/wasm/emit_wasm`, native side = `diffcli render-bytes`);
  sample-for-sample agreement required.
- `tests/web/web_plans_vs_jit.test.ts` — every precompiled plan in
  `web/dist/patches/` matches the JIT output. Run after
  `bun web/build_patches.ts` and `make lean`.

Any divergence is a bug in either `web/wasm/emit_wasm.ts`, the WASM
runtime, or the Lean compiler (the latter shows up in
`tests/web/wasm_vs_jit`).

## Build / run

```bash
bun web/build_patches.ts    # → web/dist/patches/*.plan.json + index.json (via Lean diffcli)
bun web/build.ts            # full bundle: patches + worklet + app + index.html
bun web/dev.ts              # static server with COOP/COEP for SAB
```

`bun web/dev.ts` is required for local development because
`SharedArrayBuffer` needs `Cross-Origin-Opener-Policy: same-origin` +
`Cross-Origin-Embedder-Policy: require-corp`. Without those headers
the runtime falls back to a plain `ArrayBuffer` (init-time param
snapshot only, no live updates across the worklet boundary).
