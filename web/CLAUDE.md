# web/

Browser backend — a **precompiled-patch player**. The browser fetches a
`.wasm` kernel + a small manifest and instantiates it; it runs no compiler
and no codegen. There is one codegen in the whole project (Lean's
`EmitLlvm`); the browser consumes the *same* LLVM IR the native JIT runs,
lowered to wasm32 in-process by the same LLVM (`diffcli compile-wasm`).

The web backend is held to sample-for-sample equivalence with the JIT. See
the `tests/web/` suites.

## The boundary: `.wasm` + manifest

Per patch, the build emits two artifacts the browser fetches:

- `<slug>.wasm` — a complete wasm32 module exporting `tropical_kernel` (the
  port: one pointer per input — `tau`, then the params — and one for the
  output, over a static 128-frame block) + `__heap_base`, importing
  `env.memory` and `env.round`. Produced by `diffcli compile-wasm --frames
  128` (engine LLVM + lld, in-process — no `wasm-ld` on PATH, no toolchain).
- `<slug>.manifest.json` — the kernel's type, `tropical_kernel_1`
  (`design/typed-kernel.md`): `tau:i64q16[128]`, one `f64` param per knob
  with its default, and `out:f64[128, C]`. Produced by `diffcli
  compile-kernel --frames 128`. The compiler's plan is nested inside for its
  own gates; the runtime never reads it.

## Layout

```
web/
  runtime/             The extractable runtime package (→ @tropical/runtime-wasm)
    manifest.ts          KernelManifest — the consumption contract
    layout.ts            computeLayout: KernelManifest → linear-memory regions
    kernel.ts            WasmKernel: instantiate + render(f64) + process(f32+fade)
    index.ts             public surface
  patches/             Source demo patches (tropical_program_2 JSON) + manifest.json
  build_patches.ts     Per patch: diffcli compile-wasm → <slug>.wasm + a trimmed
                       <slug>.manifest.json → web/dist/patches/ + index.json
  build.ts             Full demo bundle: build patches, bundle worklet + app, copy html
  dev.ts               Plain static server (no COOP/COEP — there's no SharedArrayBuffer)
  host/                Main thread
    context.ts           startHost: AudioContext + worklet node; loadPatch / fade
  worklet/             Audio thread
    processor.ts         AudioWorkletProcessor → WasmKernel; postMessage protocol
  site/                Browser UI
    app.ts               patch picker, play/stop, fetch .wasm + manifest
    index.html
```

`web/runtime/` depends **only** on `KernelManifest` (`tropical_kernel_1`) —
nothing from the compiler. It is a self-contained consumer of tropical's
artifact contract, ready to extract to its own package once a second consumer
(an Electron app, a hosted player) appears. The forcing function is the
dependency edge, not the repo boundary.

## Where this fits in the bigger pipeline

```
build host (Lean + LLVM + lld)              browser
  diffcli compile-wasm <patch>   →           fetch /patches/<slug>.wasm
  diffcli compile-kernel <patch>              fetch /patches/<slug>.manifest.json
    (Lean EmitLlvm → wasm32 TargetMachine     → WasmKernel.instantiate(wasm, manifest)
     → lld::wasm::link, in-process)           → kernel.process per 128-sample block
  → <slug>.wasm + <slug>.manifest.json
  written to web/dist/patches/                → AudioWorkletNode → speakers
```

`web/build_patches.ts` shells `diffcli` over the source patches; the browser
runs only the runtime package. Requires libtropical_wasm (the toolchain
library only `diffcli` links) built with `TROPICAL_WASM_EMIT` (`make build`
enables it; needs `brew install lld`).

## Player, not instrument

The demo plays precompiled patches: **no live recompile, no state-transfer
hot-swap, no SharedArrayBuffer params.** Those belong to the native
instrument (Electron + the ORC JIT), where topology edits trigger
recompile→hot-swap. `WasmKernel` keeps a `setParam` method (one cell per
param input) for a future host with live control, but the demo never
calls it — so the demo needs no COOP/COEP and deploys as plain static files. A
smoothstep fade (`beginFadeIn`/`beginFadeOut`) is the only envelope, to avoid
start/stop clicks.

## Linear-memory layout

The kernel takes one pointer per input and one for the output; everything
else lives inside the function. The host owns one imported
`WebAssembly.Memory` and places each region above the module's
`__heap_base` (`runtime/layout.ts`):

```
__heap_base
  tau       i64[frames]              — the Q48.16 clock of each frame
  params    f64 × (one cell per param input, port order)
  output    f64[frames * channels]   — frame-major
```

`WasmKernel.render(n)` fills tau and calls the kernel once per static block,
chunking a longer window over the same closed form, and returns a copy of
the raw f64 output (what the WASM≡JIT gate compares); `process(out, n)`
applies the fade and downcasts to the f32 worklet buffer.

## Worklet protocol

`worklet/processor.ts` runs in `AudioWorkletGlobalScope` and receives:

| Message | Effect |
|---------|--------|
| `{type:'load', wasm, manifest}` | queue; `process()` instantiates a `WasmKernel` and fades in |
| `{type:'fadeIn'}` / `{type:'fadeOut'}` | envelope control |

We post raw wasm *bytes* (not a `WebAssembly.Module`) because Chrome silently
drops worklet messages containing a Module; the worklet compiles them itself.

## Equivalence gates

Two suites in `tests/web/` lock the wasm path to the JIT (both compare the
kernel's f64 output, `WasmKernel.render` vs `diffcli render-bytes`):

- `wasm_vs_jit.test.ts` — current closed-form oscillator, nonlinear, and
  operation-coverage patches compiled by `diffcli compile-wasm` vs the JIT.
  The former array/writeback-shaped program is now a refusal case proving
  `programDecl` cannot reintroduce retired state. The executable cases are the
  *same IR* through two LLVM targets, so agreement mainly guards target-specific
  FP divergence.
- `web_plans_vs_jit.test.ts` — every shipped `web/dist/patches/<slug>.wasm` +
  manifest vs the JIT. If this passes, any browser bug is in the host/worklet
  glue, not codegen.
