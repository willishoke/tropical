# reversible

Private. The Swift/macOS control surface for tropical — a native port of
`playground/` (the Electron patch playground) from the `perf/dag-to-emit`
branch. Downstream-only patching that lowers through the session → arrow
slide; the master clock scrubs every voice, envelope, and reverb tail
coherently because kernels are closed-form `f(τ, params)` — no state,
so reverse is just receding τ.

## Run

```bash
swift run
```

The app owns the Lean `frontend` binary over its Unix-socket JSON-RPC
surface (`--serve`, same newline-delimited framing as `--rpc`) and plays
audio out of the host's device (RtAudio) — the window is purely a control
surface. The socket carries a control/data plane split: `set_param`,
`render_window`, and `playback_position` are answered synchronously in
C++ and never queue behind the Lean control thread — that is what keeps
knob writes and the Scope module's traces live through a long compile.

Engine resolution order:
1. `TROPICAL_ENGINE_BIN` env var (path to the `frontend` binary)
2. `~/tropical/lean/.lake/build/bin/frontend`

The engine must be built from a branch that serves `load_patch_graph`
(`perf/dag-to-emit` or later). Run the binary directly — never `lake exe`
(DYLD shadowing; see tropical/CLAUDE.md).
