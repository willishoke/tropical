# PluckedMorphOsc

A `MorphOsc` with a closed-form **pluck envelope** baked in — the dynamic content
the τ-scrub instrument needs. The envelope is a pure function of the clock, so
the whole voice reverses with the master clock: forward it's a fast-attack /
slow-decay pluck, backward it's a slow swell into a hard cut (the unmistakable
"reversed tape" cue). And because the envelope is *inside* the voice, any
downstream clock-warp (a delay/comb tap) reads a delayed *plucked* copy — so a
future tap is an audible pre-echo, not a silent bulk delay of a steady tone.

The event phase comes from a `ClockPhasor` at `event_rate`, so it's an exact
integer phasor (no `frac` precision drift as τ accumulates). The shape

```
env(f) = 17.6 · f · (1 − f)⁶,   f = frac(event_rate · τ)
```

is a smooth skewed pulse — zero at both ends of the period (continuous across the
wrap, so no click and nothing to band-limit), asymmetric (fast rise, slow decay)
so it reverses *audibly*. Peak ≈ 1 near `f ≈ 1/7`.

## Source

```tropical
program PluckedMorphOsc(freq: freq = 220, morph: unipolar = 0, clk: clock = clock(),
                        event_rate: freq = 1, phase: unipolar = 0) -> (out: float) {
  osc = MorphOsc(freq: freq, morph: morph, clk: clk, phase: phase)
  ev  = ClockPhasor(clk: clk, freq: event_rate)
  out = osc.out * (let { f: ev.phase } in
                   let { u: 1 - f } in
                   let { u2: u * u } in
                   17.6 * f * u2 * u2 * u2)
}
```
