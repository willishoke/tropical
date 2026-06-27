# PluckEnv

A stateless, reversible pluck envelope — the bounded-phase reformulation of a
triggered envelope. A triggered envelope (`EnvExpDecay` and friends) is
stateful twice over: a rising-edge detector that needs the *previous* sample
(`delay prev_trigger`), and an accumulator for "time since the last trigger"
(`reg`). `PluckEnv` has neither. The event becomes the **periodic wrap of an
event-phase** `Ψ`, "time since onset" becomes `frac(Ψ)`, and the envelope is a
bounded *shape* of that fraction:

```
env(Ψ) = 17.6 · f · (1 − f)⁶ ,   f = frac(Ψ)
```

As `Ψ` advances (an event-phase from `FixedPhasor` at the event rate), `f`
sweeps `0 → 1` each period and the shape plays out: a fast rise (peak near
`f = 1/7`), a slow decay, **zero at both ends** so it is continuous across the
wrap — no envelope step, no aliasing comb, nothing to band-limit — yet
**asymmetric**, so it reverses *audibly* (forward pluck ↔ reverse swell). Peak
≈ 1.0; the `17.6` normalizes it.

Because it is a pure function of `Ψ`, it is

- **stateless** — no register, passes `cfOnly`, fuses freely;
- **random-access / reversible** — equal `Ψ` ⟹ equal output, so a symmetric
  coordinate renders a bit-exact palindrome (the decay reverses into a swell);
- **periodic** — one pluck per event-phase cycle. This is the deliberate
  events↔periodicity trade: a one-shot becomes a *schedule*. For irregular
  rhythms, drive `Ψ` from a non-uniform but still closed-form event-phase.

The same reformulation covers the rest of the trigger family without state:
a **gate/pulse** is a narrow window of the fraction (`frac(Ψ) < ε`), a
**ramp / "time-since-event"** is `frac(Ψ)` itself, and a **sample-and-hold**
reads a closed-form source at `⌊Ψ⌋` (the index of the current period). All are
functions of the event-phase — the edge-detect register was never the event,
the phase wrap is.

`phase` is the event-phase in **cycles** (e.g. `FixedPhasor(freq: event_rate)`
for a periodic schedule; integer part counts events, fractional part is
position within the current one).

## Source

```tropical
program PluckEnv(phase: float = 0) -> (env: float) {
  env = let { f: phase - floor(phase) } in
    let { u: 1 - f } in
    let { u2: u * u } in 17.6 * f * (u2 * u2 * u2)
}
```
