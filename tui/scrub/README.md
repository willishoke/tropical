# tropical TUI — reversible τ-scrub + reverse reverb

A terminal control surface (TypeScript + [Ink](https://github.com/vadimdemedes/ink))
driving a live tropical patch over the engine's JSON-RPC surface. It's a
**transport over synthesis-time τ**: scrub one `velocity` and the whole sound
freezes, reverses, or varispeeds — coherently, because every voice is a
closed-form function of τ — and it carries a **reverse reverb** built the
closed-form way (no tape-flip sandwich).

```
  VelocityClock(velocity) ──► τ
       ├ AnchoredPhase(f0, τ)         ──► Φ   (pitch phase, cycles)
       └ AnchoredPhase(event_rate, τ) ──► Ψ   (event phase, cycles)
       │
       ├─ voice(Φ)·env(Ψ)             · dry event              ┐  each × a smooth
       ├─ voice(Φ+f0·D)·env(Ψ+er·D)·g · pre-echo 1             │  pluck env,
       │   …                                                   ├─ Σ gᵏ → SoftClip → dac
       └─ voice(Φ+f0·KD)·env(Ψ+er·KD)·gᴷ · pre-echo K          ┘

env(ψ) = 17.6·f·(1−f)⁶ (f=frac ψ): a smooth skewed pulse, zero at both ends of
the period so it's continuous across the wrap — no envelope step, no aliasing
comb, nothing to BLEP — yet asymmetric (fast rise, slow decay) so it still
reverses audibly.
```

**Why `AnchoredPhase`, not `f0·τ`.** Phase must be `∫f0`, not `f0·τ` — the
latter is only correct for constant pitch; a live pitch change jumps the phase
by `Δf0·τ`, an aliasing squelch that *grows without bound* as τ accumulates
(measured: 1% → 75% artifact energy over 6 s). `AnchoredPhase` re-bases the
phase at each rate change so it stays value-continuous (no jump) while remaining
closed-form in τ between changes — so pitch/event-rate parameterize freely with
**bounded** error, random access and exact reverse survive, and the offset reads
below still work (`phase(τ+kD) = phase(τ) + rate·kD`). `ModalVoice` is reused as
a phase reader (`f0=1`, `tau=Φ`).

**Why future taps = reverse reverb.** The sandwich trick (reverse → causal
reverb → reverse) convolves the signal with the *time-reversed* impulse
response — an anti-causal kernel reading the future, weighted by the reverb's
decay. A reverse feedback comb `y(τ)=x(τ)+g·y(τ+D)` unrolls (x closed-form) into
the geometric future-tap series above. So each event's reverb **swells into it
from before**. It can't be a black-box stream effect — it re-evaluates the
closed-form *source* at future τ, which is exactly what a stream can't do (hence
the sandwich existed).

- **Velocity is the scrub.** `1×` forward, `0` **freezes** (a hit hangs
  mid-bloom), negative runs **backward**, `>1` varispeeds.
- **Forward → reverse reverb** (swell into each hit). **Scrub negative → forward
  reverb** (the same taps trail the hit). One patch, both reverbs.
- **The envelope is a function of τ** — an asymmetric pluck — so it reverses
  *with* the scrub (forward pluck ↔ reverse-swell).

Uses `stdlib/reversible/` (`VelocityClock`, `ModalVoice`) + `SoftClip`, plus
two primitives added for this work: `AnchoredPhase` (the squelch fix) and
`Smooth` (snap-settling one-pole — de-zippers the raw stepped param slots).
(`BlepStep`, a reusable polyBLEP black box, also got added but the demo no
longer needs it — a continuous envelope has no edge to band-limit. `K = 10`
future taps in `patch.ts` — more taps = smoother bloom, slower load.)

## Run

From this directory:

```bash
bun install      # first time only (ink + react)
bun start        # spawns the engine, loads the patch, starts audio
```

Requires the engine built (`make lean` from the repo root) and an audio device.

| key | action |
|-----|--------|
| ↑ / ↓ | select a parameter |
| ← / → | adjust (hold **⇧** for ×10 steps) |
| space | play / stop audio |
| r | reset all params to defaults |
| q | quit (stops audio, kills the engine) |

**Try:** raise **Reverb amt** and **Tap spacing** so the bloom is obvious — each
hit should swell up from silence. Then on **Velocity**, hold **⇧←** to drag past
0: the bloom flips from *before* the hit to *after* it (reverse reverb → forward
reverb). Park near `0` to freeze a hit mid-bloom and crawl it with **←/→**.

## How it works

- `client.ts` — spawns `frontend --rpc`, speaks newline JSON-RPC
  (`{jsonrpc,id,method,params}` ⇄ `{result:{content:[{text:"<json>"}]}}`).
- `patch.ts` — the patch as one `tropical_program_2`: `paramDecl`s register the
  settable params; `instanceDecl`s wire the `K+1` `ModalVoice` taps (dry +
  future pre-echoes) with geometric weights and the τ-driven envelope. No
  feedback (the only state is `VelocityClock`'s τ accumulator), so no
  cycle-breaking delays needed.
- `app.tsx` — the Ink UI; each knob move fires `set_param`, a lock-free slot
  write the audio thread picks up next buffer.

## Checks

```bash
bun run smoke.ts                    # loads patch, verifies params + forward/reverse/freeze (no audio)
TUI_NO_AUDIO=1 bun render-test.tsx  # headless mount + reverse-transport test (no audio)
```
