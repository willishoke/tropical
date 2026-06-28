# ClockReverseProbe

A witness that running an oscillator's clock backward runs the oscillator
backward. `FixedSinOsc(clk: -θ)` evaluates the sine at the negated time
coordinate; since `phase(-θ) = 1 - phase(θ)` exactly in fixed-point and
`sin(2π(1-p)) = -sin(2πp)`, the reversed sine is the negated forward sine. So
`forward + reverse` cancels to (near) zero at every sample — the residual is
only whatever asymmetry the `Sin` polynomial's range reduction leaves between
the two distinct phase inputs.

## Source

```tropical
program ClockReverseProbe(freq: freq = 220) -> (out: float) {
  fwd = FixedSinOsc(freq: freq, clk: clock())
  rev = FixedSinOsc(freq: freq, clk: 0 - clock())
  out = fwd.sine + rev.sine
}
```
