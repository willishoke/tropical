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
  output, over a static 128-frame block), `__heap_base` and
  `tropical_kernel_float_bits` (the module's float format), importing
  `env.memory` and `env.round`. Produced by `diffcli compile-wasm --frames
  128` (engine LLVM + lld, in-process — no `wasm-ld` on PATH, no toolchain).
- `<slug>.manifest.json` — the kernel's type, `tropical_kernel_1`
  (`design/typed-kernel.md`): `tau:i64q16[128]`, one scalar param per
  `param:` slot with its default and write discipline (`f64` unless it
  declares an integer word, `f32` in a `--float f32` artifact), and
  `out:f64[128, C]` (`i32` for an `int32` program output, `f32` at
  `--float f32`). Produced by `diffcli
  compile-kernel --frames 128` (`--rate R` or `--rate=R`, on both verbs,
  compiles for another rate; the default is 44.1 kHz; an unknown flag or a
  bad value is an error). The compiler's plan is nested inside for its own
  gates; the runtime reads only its `param_disciplines`, to refuse a
  binary32 artifact that pairs `f32` with a non-raw discipline.

## Layout

```
web/
  runtime/             The extractable runtime package (→ @tropical/runtime-wasm)
    manifest.ts          KernelManifest — the consumption contract
    layout.ts            computeLayout: KernelManifest → linear-memory regions
    kernel.ts            WasmKernel: instantiate + render/renderInto(f64) + renderAt(τ words) + seek + process(f32+fade) + setParam/getParam
    params.ts            ParamWriter: live param writes under raw/anchor disciplines
    index.ts             public surface
  patches/             Source demo patches (tropical_program_2 JSON or PatchGraph) + manifest.json
  build_patches.ts     Per patch: diffcli compile-wasm → <slug>.wasm, diffcli compile-kernel
                       → <slug>.manifest.json, into web/dist/patches/ + index.json
  build.ts             Full demo bundle: build patches, bundle worklet + worker + app, copy html
  dev.ts               Plain static server (no COOP/COEP — there's no SharedArrayBuffer)
  host/                Main thread
    context.ts           startHost: AudioContext + worklet node + renderer Worker; loadPatch / fade
  worker/              Render-ahead thread
    render_ahead.ts      RenderAhead: the kernel, the splice, the lookahead (no Worker API; tested in virtual time)
    renderer.ts          the Worker shell around it
    curve.ts             one cycle of an output channel via renderAt, for the page's curve display
  worklet/             Audio thread
    processor.ts         AudioWorkletProcessor: copies queued chunks out; underrun concealment
  site/                Browser UI
    app.ts               patch picker, play/stop, knobs, the curve canvas, fetch .wasm + manifest
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
further channel. A `tropical_program_2` patch makes a tap the same way: a
further `dac.out` assign with a `channel`.

**Audio channels and taps.** A tap is an observation channel, never sound.
The kernel type names its channels (`out.channels`, e.g.
`__root__.tap:cutoff`) but does not mark which are sound, so the patch's
`manifest.json` entry declares `audio`, the kernel channels that are sound
in host order, and `build_patches.ts` fails on a kernel with more than one
channel that does not declare them (a one-channel kernel is `[0]`). The
player sends only those to the worklet (`WasmKernel.process(outputs, n,
channels)`); one audio channel plays on every speaker. The safe default
holds below the build too: `RenderAhead.adopt` and `startHost`'s
`loadPatch` refuse a multichannel kernel without its list (a stale
`index.json`, another consumer of the runtime). Low-level `process` with no
list still plays every channel. A display may still
draw a tap: that is what taps are for. Carrying the distinction in the
kernel type (a role per output channel, set by the compiler, which knows
which assigns are taps) would remove the declaration; it is a core gap, not
built.

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
(`design/host-param-dispatch.md`): `raw` writes the value; `anchor` and
`anchor_clock` bump the `#phase` companion by the phase jump `Δinc·C` so a
re-pitch is continuous at any τ (`C` is the frame clock for `anchor`, the
master clock for `anchor_clock`, which playground knobs declare). Companions,
`glide` and `velocity` are refused by name at the write; an unknown
discipline or a bad `clock` is refused when the writer is built. A glide's discipline
sits on a base name with no port, so the kernel type cannot carry it; a
playable knob is therefore a `knob` node (raw, or anchor when it feeds a
freq inlet). A `tropical_program_2` patch's params are all raw: a
`paramDecl` cannot declare a discipline and the session compile attaches
none. So a program-2 knob is continuous only where a raw write is. A clock's
phase is `inc·τ`, so a raw write to a frequency re-evaluates the whole past
under the new frequency and moves the phase by `Δinc·τ`: at τ a minute in,
one slider step of `spline-voice`'s 2 Hz pulse rate moves the envelope by
about 0.64 of a cycle, to an effectively arbitrary point. The frequency
knobs of the spline patches (`pitch`, the pulse `rate`, the `lfo` rate) are
discontinuous this way; a control point, `width`, `depth` or `level` steps
the output by its own change at the current phase, which is smaller. These
knobs become continuous once a patch can declare an anchored frequency
(declared write disciplines for patch-bay params are in progress on
`feat/patch-disciplines`); the patches stay raw until then. `build_patches.ts` fails the build on a knob that names no
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
  params    one 8-byte cell per param input, port order (an f64, or an f32,
            i32 or i64 word at its low address)
  output    f64, f32 or i32 [frames * channels]   — frame-major
```

`WasmKernel.render(n)` fills tau and calls the kernel once per static block,
chunking a longer window over the same closed form, and returns a copy of
the output as f64, an `f32` or `i32` output widened exactly (what the
WASM≡JIT gate compares); `process(out, n)`
applies the fade and downcasts to the f32 worklet buffer.

`renderAt(taus)` evaluates the kernel at caller-chosen clock values: each
entry of the `BigInt64Array` is the port's own Q48.16 word, passed through
unchanged, one output frame per τ, in any order or spacing. A fractional
frame is exact to 2⁻¹⁶ of a frame and the value is `f(τ)` itself, not an
interpolation. At whole frames (`n << 16`) it is `render` bit for bit. It
does not move the frame counter or apply the fade. `tauOfFrame(x)` gives the
word for a frame `x` and refuses one that is not a multiple of 2⁻¹⁶ below
2³⁷ (build a larger word as a BigInt).

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

A visual that wants to sync to the audio must use the worklet's playhead
(the played frame), not the renderer's frame, which runs ahead by up to the
lookahead. The curve display (below) starts at the renderer's estimate of
the playhead: the last report advanced by the time since.

| Message | Path | Effect |
|---------|------|--------|
| `{type:'load', wasm, manifest, audio}` | host → renderer | instantiate; splice in after the current patch fades out; play only the `audio` channels (a multichannel kernel without them is refused) |
| `{type:'param', name, value}` | host → renderer | write at the next splice |
| `{type:'fadeIn'}` / `{type:'fadeOut'}` | host → renderer | envelope from the next splice |
| `{type:'display', displays, points, generation}` | host → renderer | draw one cycle of each listed channel; `[]` stops |
| `{type:'curve', generation, index, cycle, tau0, freq, data}` | renderer → host | display `index`'s points over clock cycle `cycle`, from clock word `tau0`; the host drops a curve of an older generation |
| `{type:'chunk', epoch, start, data}` | renderer → worklet | planar f32 frames from `start`; a newer epoch overwrites the frames it covers, an older one is dropped |
| `{type:'playhead', frame}` | worklet → renderer | the next frame the worklet will play, every 4 quanta |
| `{type:'mute'}` | host → worklet | a short local fade before the host closes the context |

We post raw wasm *bytes* and the renderer compiles them. (When the worklet
rendered, bytes were forced: Chrome silently drops worklet messages
containing a `WebAssembly.Module`.)

## The curve display

A patch's `manifest.json` entry may declare `displays`, each `{channel,
freq | freqParam, label}`: draw one cycle of output `channel` (usually a
tap) over a clock of frequency `freq`, fixed or the current value of a
param. The page sends the list to the renderer (`{type:'display'}`), which
evaluates each curve with `renderAt` at 512 τ values (`worker/curve.ts`)
under the newest spliced knob values, and posts the points back
(`{type:'curve'}`). The page only draws them. `build_patches.ts` fails the
build on a display whose channel or param the kernel does not have.

- **Which cycle.** The clock is read by the stdlib phasor law, the one the
  anchor discipline also uses: `inc = trunc(f·2³²/rate)`, phase
  `⌊inc·τ/2¹⁶⌋ mod 2³²` (a `ClockPhasor` on τ with offset 0). This is the
  host's copy of the law, so a clock with an offset or another law is drawn
  rotated or stretched: `build_patches.ts` refuses a `freqParam` with a
  non-raw discipline (an `anchor` param carries a `#phase` offset; an
  `anchor_clock` one runs on the master clock). In a binary32 artifact the
  kernel's increment can differ from the host's by one and the picture
  slowly drifts. The kernel-sourced fix is to tap the clock's phase as a
  channel and pick phase 0 from kernel output; not built. The curve is the cycle
  that contains the renderer's estimate of the playhead, from its phase 0:
  point `j` is the first clock word at or after phase `cycle + j/512`, so
  its phase is exact to within one word's advance (`inc/2¹⁶` phase units,
  about 2⁻²³ of a cycle at 220 Hz). Two redraws within one cycle are the
  same curve bit for bit, and a later cycle differs only by that grid
  offset, so the picture holds still under fixed knobs.
- **When.** The playhead estimate is the worklet's next frame, from its last
  report advanced by the time since. The knob values are the newest
  spliced, so the picture leads what is heard by the knob margin plus the
  device latency.
- **What a tap shows.** `spline-voice` taps the wave and the envelope, and
  `spline-tremolo` the wave at audio rate and the same wave at LFO rate;
  each curve depends only on its own clock, so one cycle of it is the shape
  itself. Drawing the audible `vca.out` instead would not be: a patch has
  one τ, so sweeping τ across a cycle also advances every other clock by
  the same span of time. Across one 220 Hz cycle of `spline-voice`'s output
  the envelope changes its peak by up to 29% at the defaults (85% at pitch
  55), and a short pulse can be shorter than the cycle. Drawing "the
  output's shape at τ₀, modulators frozen" needs the oscillator's phase as
  a kernel input separate from τ, which the port does not have. The
  smallest change that would give it: an optional second per-frame input
  on the port (say `phase:i64q16[N]`, defaulting to τ) that the compiler
  routes to the clocks a display names. Not built.
- **Cost.** A curve is 512 kernel frames. At most one round of curves runs
  per 33 ms; a knob write marks the curves stale and is coalesced into the
  next round; no round runs while the renderer is behind its lookahead, and
  none while a load is pending (so an old kernel is never drawn under a new
  patch); a failed load drops the new patch's displays, and each display
  list is a generation, so a curve still in flight from the previous list
  is dropped by the host. Two curves at 30 rounds a second are about 0.7× real time of
  kernel work, so their share of a core is 0.7× the patch's render cost
  (see the per-chunk cost table in `render_ahead.test.ts`, which also times
  one curve): negligible for the spline patches. It is not counted in
  `controlBudget`: the spline patches' curves cost about 40 µs each (about
  0.3% of a core for two at 30 rounds a second); a patch as heavy as
  `spectral-filter-fm` (about 560 µs a
  curve) declares no display.

## Known issues

- Kernels are compiled at 44.1 kHz and played at the AudioContext's rate
  with no resampling: on 48 kHz hardware everything plays 8.8% sharp and
  fast. The display's Hz are in kernel time (the manifest's `rate`).
- Taps outside the player. The spline patches are 3-channel programs (audio
  plus two taps) and `spectral-filter-fm` is 2-channel (audio plus a tap).
  Only the player knows which channels are sound, from `audio`. The native
  DAC (`engine/dac/TropicalDAC.hpp`, read, not run) maps plan channel `c` to
  device channel `c` (a mono plan upmixes): MCP `start_audio` opens 2
  channels by default, so `spectral-filter-fm` loaded there would play its
  cutoff tap on the right channel, and a 3-channel spline patch is refused
  when the stream opens ("requires 3 device output channels") or plays
  silence if swapped into a running stereo stream; with 3 or more device
  channels the taps go to channels 1 and 2. Same core gap as above: the
  kernel type does not carry audio-vs-tap.

## Equivalence gates

The `tests/web/` suites for the wasm path and the player. The first three
compare the kernel's output against the JIT (`WasmKernel.render` vs
`diffcli render-bytes` or `render-graph`):

- `wasm_vs_jit.test.ts` — current closed-form oscillator, nonlinear, and
  operation-coverage patches, with `int32`, typed-param and `--float f32`
  cases, compiled by `diffcli compile-wasm` vs the JIT.
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
  sample, a knob splice ≡ an in-place write at its frame (also a spline
  control point and a `PhaseClip` width on `spline-voice`), no tap channel
  reaching a host channel (`spectral-filter-fm`'s cutoff tap; both
  speakers play its one audio channel), late splices, patch swaps, and each
  shipped patch's per-chunk render and per-curve cost.
- `render_at.test.ts` — `renderAt` on a two-channel kernel: one period of
  two spline oscillators (Catmull-Rom, K = 12; B-spline, K = 5) at 1000
  fractional-frame τ values equals the f64 reference spline at each τ's
  exact integer phase within 1e-12, at τ = 0 and ten hours in; at whole
  frames it is `render` bit for bit (also on the shipped `spline-voice`).
  The display path: `cycleCurve` is `renderAt` on its τ grid bit for bit,
  starts at phase 0 of the cycle containing the frame, closes, redraws the
  same curve within a cycle, and follows a param write.
- `runtime_params.test.ts` — `ParamWriter`: an anchored re-pitch ten hours in
  is continuous and plays the new pitch (a raw one jumps); companions and
  unimplemented disciplines are refused.
- `runtime_typed_ports.test.ts` — the manifest reader and `ParamWriter` on
  `i32`/`i64` params and an `i32` output, and the refusals of what the
  reader cannot bind.
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
