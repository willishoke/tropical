# Smooth

A clean linear one-pole smoother — a leaky integrator with no saturation
(unlike `OnePole`, which `tanh`s its input/state). Use it to de-zipper a
stepped control value: feed a raw param in, get a glided value out. `rate` is
the per-sample coefficient (≈ 1 / time-constant-in-samples); smaller = slower
glide. At 48 kHz, `rate = 0.002` is a ~10 ms glide.

State is one register, so it `breaks_cycles` — the output is the previous
smoothed value, never the current input, so it never forms a same-sample loop.

## Source

It **snaps** to the target once within `eps`, so a settled value is *exactly*
the target (not an asymptote) — important when the smoothed value feeds a
rate whose constancy buys closed-form/random-access behavior downstream.

```tropical
program Smooth(x: float = 0, rate: float = 0.002, eps: float = 0.001) -> (out: float) breaks_cycles {
  reg s = 0
  out = s
  next s = select(abs(x - s) < eps, x, s + rate * (x - s))
}
```
