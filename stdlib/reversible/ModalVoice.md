# ModalVoice

A closed-form modal voice: a sum of undamped sinusoidal partials at
incommensurate frequencies, evaluated as a pure function of the Q32.32
integer clock. Because each partial is a `ClockPhasor → Sin` chain on the
shared `clk` and nothing here holds state, the whole voice is a pure function
of the clock — it can be evaluated at *any* clock value (forward, backward,
jumped) and it reverses exactly when the clock reverses. This is the "torus cousin": the partials are the modes,
their incommensurate ratios make the sum shimmer and never quite repeat.

The partials are *undamped* on purpose. Damping (`e^(-α·tau)`) is the
irreversible direction — read backward it grows without bound — so a
conservative, energy-preserving voice is the one that scrubs cleanly in both
directions and stays bounded. Striking and decay live on the stateful side
of the boundary; this voice is the reversible core.

## Signal flow

```mermaid
flowchart LR
  tau([tau]) --> S1["Sin f0"]
  tau --> S2["Sin 2.414·f0"]
  tau --> S3["Sin 4.236·f0"]
  tau --> S4["Sin 6.854·f0"]
  subgraph mix
    S1 --> SUM(("Σ weighted"))
    S2 --> SUM
    S3 --> SUM
    S4 --> SUM
  end
  SUM --> out([out])
```

## Internals

Four `Sin` partials at `f0 · {1, 1+√2, 2+√5, ...}` — irrational ratios so no
two partials are harmonically locked and the beat pattern never closes. Each
partial rides its own `ClockPhasor` on the shared integer clock: the phase is
reduced on the circle ℤ/2³² *before* it ever becomes a float, so large or
negative clocks carry full precision at any τ — which is exactly what makes
scrubbing (and negative time) work. (`Sin` still range-reduces its float
argument, but in this program it only ever sees the pre-reduced `2π·phase ∈
[0, 2π)`.)

The weights `0.4, 0.24, 0.16, 0.1` sum to `0.9 < 1`, keeping the voice below
clipping. There is no register anywhere in this program: it is purely
combinational in the clock.

## Source

```tropical
program ModalVoice(clk: clock = clock(), f0: freq = 110) -> (out: float) {
  p1 = ClockPhasor(clk: clk, freq: f0)
  p2 = ClockPhasor(clk: clk, freq: f0 * 2.414213562373095)
  p3 = ClockPhasor(clk: clk, freq: f0 * 4.23606797749979)
  p4 = ClockPhasor(clk: clk, freq: f0 * 6.854101966249685)
  s1 = Sin(x: 6.283185307179586 * p1.phase)
  s2 = Sin(x: 6.283185307179586 * p2.phase)
  s3 = Sin(x: 6.283185307179586 * p3.phase)
  s4 = Sin(x: 6.283185307179586 * p4.phase)
  out = 0.4 * s1.out + 0.24 * s2.out + 0.16 * s3.out + 0.1 * s4.out
}
```
