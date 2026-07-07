# tropical · scope

A dead-simple 2-channel oscilloscope TUI — the first *random-access* consumer of
the engine's kernels. Each frame it reads the audio playback position (the
master clock) and `render_window`s two voice slots over the window ending
there, drawing them as braille traces. Cadence is the app's own refresh;
content is the master position — so the trace stays **locked to audio** without
drifting, regardless of refresh jitter.

This works because the voices are `FixedSinOsc` — fully stateless, so the engine
can evaluate any sample-index window exactly and concurrently with the audio
thread (`render_window`), bit-identically to what's playing.

## Run

```
bun install
bun scope.tsx          # spawns its own engine + a 220/330 Hz two-voice session
```

Attach to an already-running `--serve` engine instead (e.g. one already playing):

```
TROPICAL_SOCK=/path/to.sock bun scope.tsx
```

Attached, the scope taps whatever patch the other TUI has live. An arrow patch
(the scrub's `load_patch_graph`) **publishes its own taps** — one per graph node,
each routed to a `render_window`-readable root output slot — so the scope
discovers them straight from `list_scope_taps` (no wiring). A session-model
patch has no pre-published taps, so the scope falls back to discovering its
instances and wiring each output to the reserved `scope` sink. Either way, `←/→`
on the source control cycles the taps. `tui/launch-demo.sh` runs both panes
against one engine.

`q` / Esc quits. With no audio device it shows a static window at sample 0
(`render_window` still works — only the master clock is frozen).

## Pieces

- `braille.ts`  — pure waveform → braille rendering (2×4 dots/char); unit-testable.
- `client.ts`   — JSON-RPC over the `--serve` Unix socket; `setupTwoVoice` session.
- `scope.tsx`   — the Ink UI.
- `test-frame.ts` — headless pipeline smoke test: `bun test-frame.ts`.
