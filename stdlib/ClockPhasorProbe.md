# ClockPhasorProbe

The equivalence witness for the fixed-point clock substrate. It runs
`FixedPhasor` (ambient sample index) and `ClockPhasor(clk: clock())` (explicit
fixed-point clock) at the same frequency and outputs their **difference**. With
`clk = clock() = sampleIndex() << 32` the fraction is zero and the two phasors
compute the identical 32-bit phase word, so the difference is **exactly zero**
at every sample. A nonzero sample would mean the split-multiply or the clock
expansion diverged from the reference — the test asserts bit-exact silence.

## Source

```tropical
program ClockPhasorProbe(freq: freq = 440) -> (out: float) {
  ref = FixedPhasor(freq: freq)
  got = ClockPhasor(clk: clock(), freq: freq)
  out = ref.phase - got.phase
}
```
