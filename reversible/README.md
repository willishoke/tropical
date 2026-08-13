# reversible

The Swift/macOS control surface for Tropical. It authors downstream-only patch
graphs against the engine-owned vocabulary and drives the same served engine
used by other native clients. The master clock scrubs closed-form kernels
coherently because their output is a pure function `f(τ, params)`.

## Run

From the Tropical repository root:

```bash
reversible/scripts/run-dev
```

The app owns the Lean `frontend` binary and Unix-socket namespace. It uses
independent graph, control, scope, and telemetry JSON-RPC connections over the
served surface (`--serve`, with the same newline-delimited framing as `--rpc`),
while audio plays out of the host's device (RtAudio). The window is purely a
control surface. The server carries a control/data plane split: `set_param`,
`render_window`, and `playback_position` are answered synchronously in
C++ and never queue behind the Lean control thread — that is what keeps
knob writes and the Scope module's traces live through a long compile.

Reversible selects the Metal audio backend by default; the f64 JIT remains
dual-loaded for scopes and reference rendering. Set `TROPICAL_BACKEND=jit`
when launching `run-dev` to opt into the CPU reference path for diagnosis.

Ordinary `in` jacks accept multiple compatible sources directly. Signal
inlets sum them; modal inlets retain them as an ordered modal forest before
applying Reverb, Phaser, or another modal stage. Control, address, modulation,
and Scope-channel ports remain single-source. Legacy `Mix` and `Modal ∪`
nodes still load from saved documents, but the Add Module menu omits those
plumbing-only nodes because fan-in now lives on the receiving inlet.

Engine resolution order:

1. `Reversible.app/Contents/Resources/Tropical/frontend`
2. `TROPICAL_ENGINE_BIN` (an explicit developer/test override)
3. The owning checkout's `lean/.lake/build/bin/frontend` for a direct
   `swift run --package-path reversible Reversible` development launch

`reversible/scripts/run-dev` resolves both the package and engine from the
script's repository location. There is no implicit home-checkout fallback and
the launcher's current working directory is irrelevant.

Create a local ad-hoc-signed app bundle with:

```bash
reversible/scripts/bundle-app
```

The output is `build/Reversible.app`. The bundle contains the native app,
Tropical frontend, and runtime library.

The engine must serve `load_patch_graph`. Run the binary directly—never `lake exe`
(DYLD shadowing; see tropical/CLAUDE.md).
