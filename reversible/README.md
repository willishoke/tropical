# reversible

Private. The Swift/macOS control surface for tropical — a native port of
`playground/` (the Electron patch playground) from the `perf/dag-to-emit`
branch. Downstream-only patching that lowers through the session → arrow
slide; the master clock scrubs every voice, envelope, and reverb tail
coherently because kernels are closed-form `f(τ, params)` — no state,
so reverse is just receding τ.

## Run

From the Tropical repository root:

```bash
reversible/scripts/run-dev
```

The app owns the Lean `frontend` binary over its Unix-socket JSON-RPC
surface (`--serve`, same newline-delimited framing as `--rpc`) and plays
audio out of the host's device (RtAudio) — the window is purely a control
surface. The socket carries a control/data plane split: `set_param`,
`render_window`, and `playback_position` are answered synchronously in
C++ and never queue behind the Lean control thread — that is what keeps
knob writes and the Scope module's traces live through a long compile.

Engine resolution order:

1. `Reversible.app/Contents/Resources/Tropical/frontend`
2. `TROPICAL_ENGINE_BIN` (an explicit developer/test override)

`reversible/scripts/run-dev` resolves both the package and engine from the
script's repository location. There is no implicit home-checkout fallback and
the launcher's current working directory is irrelevant.

Create a local ad-hoc-signed app bundle with:

```bash
reversible/scripts/bundle-app
```

The output is `build/Reversible.app`. The bundle contains the native app,
Tropical frontend, and room-only grouped-carrier asset. It deliberately does
not package the authored-score wet cache.

The engine must be built from a branch that serves `load_patch_graph`
(`perf/dag-to-emit` or later). Run the binary directly—never `lake exe`
(DYLD shadowing; see tropical/CLAUDE.md).
