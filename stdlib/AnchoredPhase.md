# AnchoredPhase

Phase as a **re-anchored** closed form of τ — the fix for `phase = rate·τ`,
which is only correct when `rate` is constant (a live `rate` change jumps the
phase by `Δrate·τ`, a discontinuity that grows without bound as τ accumulates).

Within a segment where `rate` is constant, the phase is a pure closed form,
`phase = φ_a + rate·(τ − τ_a)` — so it stays random-access in τ and reverses
exactly (it's a formula, not an accumulation, so no drift). When `rate`
changes, the anchor `(φ_a, τ_a)` is re-based to *this instant* so the new
segment continues from exactly the current phase: value-continuous, no jump.
The only state is the anchor, updated **only on a rate change** (control rate),
not every sample — so it isn't a per-sample accumulator and keeps the
closed-form, jumpable property between changes. This is a "current-rate-locked
alternate universe": navigate τ freely under the current rate; changing the
rate forks a phase-continuous new universe.

A rate change is value-continuous but frequency-discontinuous (a slope corner,
BLAMP territory), i.e. just a pitch step — not the unbounded squelch.

## Source

```tropical
program AnchoredPhase(rate: float = 0, tau: float = 0) -> (phase: float) {
  reg anchorPhase = 0
  reg anchorTau = 0
  reg prevRate = 0
  phase = let {
      moved: rate != prevRate;
      effPhase: select(moved, anchorPhase + prevRate * (tau - anchorTau), anchorPhase);
      effTau: select(moved, tau, anchorTau)
    } in effPhase + rate * (tau - effTau)
  next anchorPhase = select(rate != prevRate, anchorPhase + prevRate * (tau - anchorTau), anchorPhase)
  next anchorTau = select(rate != prevRate, tau, anchorTau)
  next prevRate = rate
}
```
