# FMBell

DX7-style FM done the closed-form way — which is to say **phase modulation**
(what "FM synthesis" has always actually been): the carrier's *clock* is
displaced per sample by a modulator oscillator, `car(clk: θ + index·mod(θ))`.
True frequency modulation (modulating a `freq` input) cannot work statelessly —
phase is the *integral* of frequency, and a closed-form `inc·θ` phasor computes
`inc·θ`, not `∫`, so a swept `freq` blows up into noise. Displacing the clock
sidesteps that entirely: it adds `inc·index·mod` to the phase directly, exact
and drift-free, no accumulator.

Two things make it sing rather than sit still:

- an **inharmonic ratio** (`√2`) puts the sidebands off the harmonic grid — a
  bell/metallic spectrum instead of a static harmonic tone;
- a **breathing index** — a slow LFO swings the modulation depth, so the
  brightness evolves instead of holding a fixed timbre.

`index` is in samples of clock displacement (the FM index); `ratio` is the
modulator:carrier frequency ratio; `lfo` is the breath rate.

## Source

```tropical
program FMBell(carrier: freq = 220, ratio: freq = 1.4142135623730951, index: float = 90, lfo: freq = 0.3) -> (out: float) {
  breath = FixedSinOsc(freq: lfo)
  modu = FixedSinOsc(freq: carrier * ratio)
  car = FixedSinOsc(freq: carrier, clk: clock() + toInt(index * (1 + breath.sine) * modu.sine * 4294967296))
  out = car.sine
}
```
