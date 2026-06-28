# MorphOsc

A stateless oscillator that crossfades continuously between a sawtooth and a
sine, both derived from the **same** `FixedPhasor` so they stay phase-aligned.
`morph = 0` is a pure ramp saw (`2·phase − 1`), `morph = 1` is a pure sine, and
in between it's the linear blend `(1−morph)·saw + morph·sin`. Because every
stage is register-free (`FixedPhasor` + `Sin` + arithmetic), the whole voice
carries zero state and is random-access — `render_window` can evaluate any
sample-index window exactly, so a scope can show it morphing live, locked to
playback. The saw is the naive (non-band-limited) ramp; for a clean scope demo
that's the point — you see the shape, not an anti-aliased approximation.

## Signal flow

```mermaid
flowchart LR
  freq([freq]) --> PH
  morph([morph]) --> MIX
  subgraph internals
    PH["FixedPhasor"] -- phase --> SAW["2·phase − 1"]
    PH -- phase --> SIN["Sin(2π·phase)"]
    SAW --> MIX["(1−morph)·saw + morph·sin"]
    SIN --> MIX
  end
  MIX --> out([out])
```

## Internals

One `FixedPhasor` drives both shapes, so saw and sine share a phase and the
crossfade has no beating. The saw is `2·phase − 1` (a ramp in [−1, 1)); the sine
is the usual `Sin(2π·phase)`. `morph` (a unipolar control, typically driven by a
param or an LFO) linearly blends them: `out = (1−morph)·saw + morph·sin`.

No registers anywhere — drive `morph` from a `param` slot and a controller (or
the scope) can sweep it live while `render_window` reads the exact resulting
waveform.

## Source

```tropical
program MorphOsc(freq: freq = 220, morph: unipolar = 0, clk: clock = clock()) -> (out: float) {
  ph = ClockPhasor(clk: clk, freq: freq)
  sin = Sin(x: 6.283185307179586 * ph.phase)
  out = (1 - morph) * (2 * ph.phase - 1) + morph * sin.out
}
```
