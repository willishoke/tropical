# ClockChord

A major triad voiced from a **single** `ModalVoice` definition — the three
notes differ only in the *clock* each instance is handed. Scaling a clock scales
time, and scaling time scales pitch: `ModalVoice(clk: (3·θ)/2)` runs the voice
1.5× as fast, a perfect fifth above `ModalVoice(clk: θ)`. The whole modal stack
— all four incommensurate partials — transposes *together* under one clock
transform, which no single `freq` knob can do (you would have to retune four
ratios at once). This is the per-oscillator-clock idea at its plainest: an
oscillator is a function of a clock you give it, and a chord is the same
function evaluated on three clocks.

The ratios are exact integer rescalings of the fixed-point clock θ: `θ` (root),
`(5·θ)>>2` (5/4, major third), `(3·θ)>>1` (3/2, fifth). Because they are integer
arithmetic on the clock, the transposition is exact and drift-free — and
reverses with the rest of the patch when θ does.

## Source

```tropical
program ClockChord(f0: freq = 110) -> (out: float) {
  root  = ModalVoice(clk: clock(), f0: f0)
  third = ModalVoice(clk: (clock() * 5) >> 2, f0: f0)
  fifth = ModalVoice(clk: (clock() * 3) >> 1, f0: f0)
  out = 0.4 * root.out + 0.3 * third.out + 0.3 * fifth.out
}
```
