# ScrubClock

The stateless replacement for `VelocityClock` — the live τ-scrub clock with
its accumulator moved to the host. `VelocityClock` integrates velocity in a
per-sample `reg` (`next t = t + velocity/sr`); `ScrubClock` computes the same
coordinate as a closed form of the sample index:

```
τ = tau_base + velocity · sampleIndex / sampleRate
```

No register — `τ` is **affine in the global sample index `n`**, so the kernel
keeps no memory. The accumulator becomes one host-held number, `tau_base`,
and the host re-bases it **only when velocity changes** (a control event), not
every sample:

- `velocity = 1`: forward time. `velocity = 0`: frozen (`τ = tau_base`,
  constant). `velocity = −1`: reverse. `velocity = 2`: double-speed varispeed.
- On a velocity change at sample `m`, the host sets
  `tau_base += (velocity_old − velocity_new) · m / sampleRate` so `τ` stays
  value-continuous across the change — exactly `FixedPhasor`'s stateless
  offset-continuity correction (`off += (inc_old − inc_new)·n`), applied to the
  clock instead of the phase. Between changes `tau_base` is fixed and the
  kernel read is pure affine.

This is fate-2 of the state migration: the velocity integration is genuinely
about live performance, so it belongs in the host transport, not the kernel.
The kernel is `f(τ, params)`; the host owns `tau_base` (one float) and the live
`velocity`. The instrument is a function from a number to a sound; the host
waves the tape in front of it.

The **reversible-scrub** path does not use this clock's host accumulator at
all: bit-exact reverse comes from supplying `τ` as a symmetric *coordinate*
(the `ReversibleProbe` triangle, or a host-computed scrub trajectory), so the
return trip retraces identical `τ` values rather than re-accumulating. This
clock is the *live forward/varispeed* driver; exact reverse is a coordinate the
consumer supplies. (See `design/cf-only.md`: the host having state is not the
kernel having state.)

## Source

```tropical
program ScrubClock(tau_base: float = 0, velocity: float = 1) -> (clk: clock) {
  clk = toInt(tau_base * sampleRate() * 4294967296)
      + toInt(velocity * 4294967296) * sampleIndex()
}
```
