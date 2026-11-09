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
  compile-kernel --frames 128` (`--rate R` or `--rate=R`, on both verbs,
  compiles for another rate; the default is 44.1 kHz; an unknown flag or a
  bad value is an error). The compiler's plan is nested inside for its own
  gates; the runtime never reads it.

## Layout

```
web/
  runtime/             The extractable runtime package (→ @tropical/runtime-wasm)
    manifest.ts          KernelManifest — the consumption contract
    layout.ts            computeLayout: KernelManifest → linear-memory regions
    kernel.ts            WasmKernel: instantiate + render/renderInto(f64) + seek + process(f32+fade)
    params.ts            ParamWriter: live param writes under raw/anchor disciplines
    index.ts             public surface
  patches/             Source demo patches (tropical_program_2 JSON or PatchGraph) + manifest.json
  build_patches.ts     Per patch: diffcli compile-wasm → <slug>.wasm + a trimmed
                       <slug>.manifest.json → web/dist/patches/ + index.json
  build.ts             Full demo bundle: build patches, bundle worklet + worker + app, copy html
  dev.ts               Plain static server (no COOP/COEP — there's no SharedArrayBuffer)
  host/                Main thread
    context.ts           startHost: AudioContext + worklet node + renderer Worker; loadPatch / fade
  worker/              Render-ahead thread
    render_ahead.ts      RenderAhead: the kernel, the splice, the lookahead (no Worker API; tested in virtual time)
    renderer.ts          the Worker shell around it
  worklet/             Audio thread
    processor.ts         AudioWorkletProcessor: copies queued chunks out; underrun concealment
  site/                Browser UI
    app.ts               patch picker, play/stop, fetch .wasm + manifest
    index.html
```

`build_patches.ts` takes `TROPICAL_DIFFCLI=<path>` to use a prebuilt `diffcli`
(e.g. from a worktree with no Lean build). A source patch is either a
`tropical_program_2` patch or a playground graph — a flat PatchGraph
(`{"nodes":[…],"out":…,"taps":[…]}`) or a v3 hierarchy document
(`{"version":3,"definitions":[…],"scene":…}`): `diffcli compile-wasm` /
`compile-kernel` detect the shape (`Playground.isGraphDocument`, or
`--graph`) and compile a graph
through `Playground.compilePlan` and the typed stage-0 split, as
`render-graph` does. Channel 0 is the graph's `out`; each named tap is a
further channel.

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
     → lld::wasm::link, in-process)             in the renderer Worker
  → <slug>.wasm + <slug>.manifest.json        → kernel.process, ~250 ms ahead
  written to web/dist/patches/                → chunks → AudioWorkletNode → speakers
```

`web/build_patches.ts` shells `diffcli` over the source patches; the browser
runs only the runtime package. Requires libtropical_wasm (the toolchain
library only `diffcli` links) built with `TROPICAL_WASM_EMIT` (`make build`
enables it; needs `brew install lld`).

## Player, not instrument

The demo plays precompiled patches: **no live recompile, no state-transfer
hot-swap, no SharedArrayBuffer params.** Those belong to the native
instrument (Electron + the ORC JIT), where topology edits trigger
recompile→hot-swap. A smoothstep fade (`beginFadeIn`/`beginFadeOut`) is the
only envelope, to avoid start/stop clicks.

**Knobs.** A patch's `manifest.json` entry may list `knobs` — each a
`param:` slot (named without the prefix) with a slider range. The page shows
a slider per knob; a move posts `{type:'param'}` to the renderer, which applies
it at the next splice (see below) through `ParamWriter`
(`runtime/params.ts`) under the discipline the kernel manifest declares
(`design/host-param-dispatch.md`): `raw` writes the value, `anchor` bumps the
`#phase` companion by `Δinc·now` so a re-pitch is continuous at any τ.
Companions and other disciplines are refused by name. A glide's discipline
sits on a base name with no port, so the kernel type cannot carry it; a
playable knob is therefore a `knob` node (raw, or anchor when it feeds a
freq inlet). `build_patches.ts` fails the build on a knob that names no
param or a discipline the player cannot write. Writes are messages, so the
demo still needs no COOP/COEP and deploys as plain static files.

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

## Render-ahead and underrun concealment

The kernel is a pure function of τ, so nothing forces it to run just in time
on the audio thread. A dedicated Worker (`worker/renderer.ts`) owns it and
keeps `lookaheadMs` (250) of output queued ahead of the worklet's playhead;
the worklet only copies chunks out, one render quantum at a time. The host
hands the two a MessageChannel, so chunks (transferred, never shared — still
no SharedArrayBuffer, no COOP/COEP) and playhead reports skip the main
thread. The AudioContext asks for `latencyHint: 'playback'`.

A control never edits the queued past. It lands at a *splice* frame
`marginMs` past the playhead (25 ms, or the context's `baseLatency` plus 5
if that is longer: the audio thread runs a device buffer of quanta back to
back); the epoch bumps and the renderer
re-renders the future from there. The worklet keeps its frames in a ring
indexed by frame; a chunk of a newer epoch overwrites the frames it covers,
and an older epoch's frames past it stay as a fallback — valid audio under
the previous knob values — until the renderer gets there. Knob latency is
the margin plus the device latency; stall tolerance is the lookahead. A knob
write at the splice is exactly an in-place write at that frame, so the
anchor discipline is continuous there. Anywhere else epochs meet they are
not continuous: a chunk that lands on frames already playing (a late
splice) crossfades from what was queued, and any other seam (the newest
epoch running into an older one, or the renderer resuming past older frames
after a stall) dips, fading out to the seam and back in.

Each knob move throws away what was rendered past its splice, so while
knobs move the renderer stays only as far ahead as keeps that re-rendering
within `controlBudget` (half a core) at a slider's ~60 events/s, from the
kernel's measured cost. Patches cheaper than ~3% of real time keep the full
lookahead; `spectral-filter-fm` (~4.5%) drops to about 180 ms. A patch swap
is a splice: the old patch fades out from the splice frame, and the new one
starts after it at τ = 0, fading in. Fades are functions of the frame
(`beginFadeIn(at)`), so a re-render repeats them exactly.

Underrun concealment: if the ring still runs dry, the worklet fades out
over a reserve of real frames (one quantum) instead of stepping to zero. It
resumes at the playhead, the wall-clock τ, not where playback stalled, and
fades back in. Native Metal does the same thing with tiles: its callback
copies prepared, epoch-tagged tiles (`design/trust-boundary.md`,
`METAL_CALLBACK_CONSUMES_PREPARED_EPOCHS`). Not built yet: when the
renderer cannot keep up at all, the measured cost is the hook for degrading
gracefully, by lowering a live cost knob such as a spectral family's
`partials`.

Nothing in the player reads a playback position today. A visual that wants
to sync to the audio must use the worklet's playhead (the played frame), not
the renderer's frame, which runs ahead by up to the lookahead.

| Message | Path | Effect |
|---------|------|--------|
| `{type:'load', wasm, manifest}` | host → renderer | instantiate; splice in after the current patch fades out |
| `{type:'param', name, value}` | host → renderer | write at the next splice |
| `{type:'fadeIn'}` / `{type:'fadeOut'}` | host → renderer | envelope from the next splice |
| `{type:'chunk', epoch, start, data}` | renderer → worklet | planar f32 frames from `start`; a newer epoch overwrites the frames it covers, an older one is dropped |
| `{type:'playhead', frame}` | worklet → renderer | the next frame the worklet will play, every 4 quanta |
| `{type:'mute'}` | host → worklet | a short local fade before the host closes the context |

We post raw wasm *bytes* and the renderer compiles them. (When the worklet
rendered, bytes were forced: Chrome silently drops worklet messages
containing a `WebAssembly.Module`.)

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
  glue, not codegen. The oracle renders every channel (patches by
  `render-bytes --all-channels`, graphs by `render-graph --all-channels`) and
  must carry exactly the artifact's channel count.
- `graph_wasm_vs_jit.test.ts` — small PatchGraphs (a swept flanger with taps; a
  modal bank whose coefficients hoist to the stage-0 prologue; signal-driven
  controls on every inlet) through `compile-wasm` vs `render-graph`, plus the
  unsplit (`TROPICAL_STAGE0=0`) bank kernel against the split one, the
  spectral kinds (`harmonic`, `envelope`: wasm ≡ JIT at τ = 0 and ten hours
  in, split ≡ unsplit bitwise, and harmonic → envelope ≡ the
  `SpectralLowpassSaw` fixture bitwise), a v3
  hierarchy document against its flat expansion, the τ-invariant-cone cost
  gate per inlet, and `--rate` (a 48 kHz kernel plays its pitch at 48 kHz).
- `render_ahead.test.ts` — the player's audio clock in virtual time
  (`playout_sim.ts`: a polyfilled `AudioWorkletProcessor`, a device FIFO, a
  stallable renderer; no device, no browser) on shipped kernels, old
  direct-render processor against render-ahead under one seeded stall
  schedule: zero-filled quanta, the largest step against the patch's own, a
  sustained overload resuming at the wall-clock τ, ahead ≡ in place sample for
  sample, a knob splice ≡ an in-place write at its frame, late splices, patch
  swaps, and each shipped patch's per-chunk render cost.
- `runtime_params.test.ts` — `ParamWriter`: an anchored re-pitch ten hours in
  is continuous and plays the new pitch (a raw one jumps); companions and
  unimplemented disciplines are refused.
- `graph_clock.test.ts` — closed-form clock properties of the playground
  generators and the signal-time `delay` at arbitrary τ (`seek`): exact
  phasor periodicity, a negative freq playing the saw and pulse backward bit
  for bit, a zero delay time is the source bitwise, 1/65536-frame tap
  resolution at frame ≈ 2^40, and anchored re-tunes.
- `graph_defaults.test.ts` — a knob a graph node omits from `params` seeds its
  vocabulary default.
- `cli_flags.test.ts` — `tropicalc` and `diffcli` share one flag reader and one
  graph detector: a v3 document compiles through both to the plan
  `render-graph` runs, `--graph` forces the graph door, `--rate N` ≡
  `--rate=N`, and an unknown flag or a bad value is an error.
