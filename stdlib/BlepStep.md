# BlepStep

The reusable polyBLEP corrector — BLEP as a black box, *not* bolted into one
oscillator. Given a naive (aliased) signal, the phase position relative to a
discontinuity, the per-sample phase increment, and the jump size, it returns
the band-limited version: `naive + jump · polyBLEP(phase, inc)`.

The per-waveform part is only the **edge schedule** — where the jumps are and
how big — which each waveform computes from its own phase and feeds in. The
correction polynomial is universal, so a saw (one −1 jump/period), a square
(two ±1 jumps), hard sync (a jump of the current value), and a pluck envelope
(one +1 jump) all share this one program.

`phase` is the wrap-phase in `[0,1)`; `inc` is `d(phase)/d(sample)` — for a
τ-scrubbed source that is `rate · velocity / sampleRate`, so the correction
widens with scrub speed and vanishes when frozen. `abs(inc)` is used so it
behaves under reverse; the two branches correct the samples just-after and
just-before the edge (Välimäki 2-point polyBLEP).

## Source

```tropical
program BlepStep(naive: float = 0, phase: float = 0, inc: float = 0.0001, jump: float = 1) -> (out: float) {
  out = let {
      p: phase - floor(phase);
      h: clamp(abs(inc), 0.000001, 0.5);
      r0: (p - floor(p)) / h;
      r1: ((p - floor(p)) - 1) / h;
      blep: select(p < h, (r0 + r0) - r0 * r0 - 1,
            select(p > 1 - h, r1 * r1 + (r1 + r1) + 1, 0))
    } in naive + jump * blep
}
```
