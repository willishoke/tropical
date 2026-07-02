# tropical TUI — reversible τ-scrub + reverse reverb

A terminal control surface (TypeScript + [Ink](https://github.com/vadimdemedes/ink))
driving a live tropical patch over the engine's JSON-RPC surface. Its headline is
**global time-warp**: one `velocity` knob scrubs the master clock every generator
reads, so the whole closed-form patch runs forward, freezes, or reverses — and
because the voice is a *plucked* pluck (fast attack, slow decay) the reverse is
audibly a reverse (a slow swell into a hard cut), not an indistinguishable crisp
tone.

```
   plk ─► cmb ─► sfl ─► out         master clock: τ = tau_base + velocity·n
```

- `plk` — **PluckedMorphOsc**: a `MorphOsc` with a closed-form pluck envelope
  baked in. The dynamic content; reverses with the master clock.
- `cmb` — **one-sided resonant comb**: the voice read at a decaying series of
  clock offsets `k·delay`. Because the voice is plucked, each tap is a delayed
  *plucked* copy: an echo (`delay < 0`) or a **pre-echo** (`delay > 0`, reading
  the future — impossible on a stream). One-sided ⇒ time-asymmetric ⇒ the tail
  flips (echo ↔ pre-echo) when you reverse.
- `sfl` — **through-zero flange**: motion over the whole tail.

The graph is downstream-only, in the playground's node vocabulary
(`lean/Tropical/Playground.lean`). The slide (`normalize`) pushes each effect's
warp up onto the generators' clocks, so every tap is a genuine *re-clocking* of
the same closed-form voice — not a stored buffer. Nothing holds history.

**Why it can't be a black-box stream effect.** A pre-echo tap re-evaluates the
closed-form *plucked source* at a **future** τ; the reverse retraces identical τ
exactly. Both are things a streaming buffer cannot do.

## Live params — the arrowemit path

The old stateful primitives this demo was born on (`VelocityClock`,
`AnchoredPhase`, `Smooth`) were retired by the cf-only fork. The live-param
machinery moved into the EmitArrow patch-graph, so the TUI loads a **patch graph**
(`load_patch_graph`) and drives its knobs over four RPCs, each picked to stay
click-free with no per-sample state:

| RPC | knobs | behavior |
|-----|-------|----------|
| `set_param_velocity` | Time warp | the global τ scrub — re-bases `tau_base` so the master clock stays value-continuous across a velocity change (the `ScrubClock` host-split) |
| `set_param_freq`     | Voice pitch | phase-anchored freq change — a `#phase` slot absorbs the `Δf·τ` jump |
| `set_param_glide`    | Morph, Pluck rate, Tap, Tail decay, Flange depth | closed-form smoothstep ramp — `v0 + (v1−v0)·s²(3−2s)`, eased over 20 ms |
| `set_param`          | Flange rate | a raw slot write (steps at block rate) |

Every knob is a live `param:<id>.<knob>` module slot on the running kernel —
turning it drives the kernel with **no recompile**. The topology is fixed, so
nothing here ever relowers.

## Run

From this directory:

```bash
bun install      # first time only (ink + react)
bun start        # spawns the engine, loads the graph, starts audio
```

Requires the engine built (`make lean` from the repo root) and an audio device.

| key | action |
|-----|--------|
| ↑ / ↓ | select a parameter |
| ← / → | adjust (hold **⇧** for ×10 steps) |
| space | play / stop audio |
| r | reset all params to defaults |
| q | quit (stops audio, kills the engine) |

**Try:** slow the **Pluck rate** and set **Tap** positive (a pre-echo, reading
ahead), then drag **Time warp** down through 0 into the negative — the pluck
swells backward and the pre-echo tail flips to a trailing echo. Park near 0 to
freeze a pluck mid-bloom and crawl it with ←/→.

## How it works

- `client.ts` — spawns `frontend --serve`, speaks newline JSON-RPC
  (`{jsonrpc,id,method,params}` ⇄ `{result:{content:[{text:"<json>"}]}}` on the
  control plane; plain `result` on the C++ data plane).
- `patch.ts` — `buildGraph()` returns the fixed patch graph (`{nodes, out}`) in
  the playground node vocabulary; `PARAMS` describes the exposed knobs, each
  tagged with the RPC `mode` (`velocity` / `freq` / `glide` / `live`) that matches
  how `Playground.collectParams` allocates its slot.
- `app.tsx` — the Ink UI; each knob move fires the param's RPC, a lock-free slot
  write the audio thread picks up next buffer.

## Checks

```bash
bun run smoke.ts                    # loads the graph, verifies slots seed + every param drives (no audio)
TUI_NO_AUDIO=1 bun render-test.tsx  # headless mount + input test (no audio)
```
