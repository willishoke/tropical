# ClockPM

Phase modulation as **clock modulation**. A carrier oscillator is handed a clock
that is jittered, per sample, by a modulator oscillator: `car(clk: θ + depth ·
mod(θ))`. Displacing the clock by `δ` samples displaces the carrier's phase by
`inc · δ` — so adding an audio-rate signal to the clock *is* phase modulation,
and because `depth · mod` is a fixed-point sub-sample offset it is smooth, not
quantized to whole samples. The modulator is bipolar, so the clock swings
*through* the carrier's own time — through-zero PM — rather than only lagging
it.

Nothing here is a new primitive: PM, the flanger, varispeed, and reverse are all
the same move (hand an oscillator a transformed clock); PM is the case where the
transform is `+ an oscillator`. `depth` is in samples of clock displacement;
`ratio` sets the modulator frequency relative to the carrier.

## Source

```tropical
program ClockPM(carrier: freq = 220, ratio: freq = 1, depth: float = 40) -> (out: float) {
  modu = FixedSinOsc(freq: carrier * ratio)
  car = FixedSinOsc(freq: carrier, clk: clock() + toInt(depth * modu.sine * 4294967296))
  out = car.sine
}
```
