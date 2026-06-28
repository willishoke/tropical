# ClockPhasor

`FixedPhasor` rebuilt to read an explicit **fixed-point clock** instead of the
ambient sample index. The clock `clk` is a Q32.32 time coordinate — a signed
64-bit integer whose high 32 bits are the sample and whose low 32 bits are the
*sub-sample fraction*. At the root the clock is `clock()` (= `sampleIndex() <<
32`, zero fraction), and with that argument `ClockPhasor` is **bit-for-bit
identical** to `FixedPhasor`. The difference is everything you can do to the
argument: pass `clk` a reparameterized clock — `-clk` (reverse), `k*clk`
(varispeed), `clk - d` (a sub-sample delay, the flanger tap) — and the same
exact phasor sweeps backward, slower, or displaced, with no new machinery. The
sub-sample fraction is what makes fractional delays and smooth phase modulation
exact rather than quantized to whole samples.

This is the per-oscillator-clock substrate: an oscillator is a function of a
clock you hand it, and reparameterization is arithmetic on that clock.

## Internals

Phase is `((inc · θ_samples) mod 1) · 2³²`, where `θ_samples = clk / 2³²`. In
integer terms `phase = ((inc · clk) >> 32) mod 2³²` — but `inc · clk` is a
32×64-bit product (up to 96 bits), so it is split to stay inside the i64
temporaries:

- `thi = clk >> 32` (integer samples), `tlo = clk & (2³² − 1)` (fraction).
- `acc = inc · thi + ((inc · tlo) >> 32) + off`.

Both terms are summed and then reduced `& (2³² − 1)`. Because only the low 32
bits survive that mask, the split is exact and the right-shift's arithmetic /
logical distinction is irrelevant (the bits that differ are masked away). With
`clk = clock()` the fraction `tlo` is zero and `acc = inc · sampleIndex + off`
— exactly `FixedPhasor`. `inc` and `off` are computed as in `FixedPhasor`
(`inc = toInt(freq · 2³² / rate)`, `off = toInt(offset · 2³²)`); `offset`
remains the stateless continuity-correction hook for live `freq` changes.

## Source

```tropical
program ClockPhasor(clk: clock = 0, freq: freq = 440, offset: unipolar = 0) -> (phase: unipolar) {
  phase = let {
      inc: toInt(freq * 4294967296 / sampleRate());
      off: toInt(offset * 4294967296);
      thi: clk >> 32;
      tlo: clk & 4294967295
    } in let {
      acc: inc * thi + ((inc * tlo) >> 32) + off
    } in toFloat(acc & 4294967295) / 4294967296
}
```
