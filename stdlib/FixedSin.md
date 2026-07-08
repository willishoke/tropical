# FixedSin

A sine evaluated **entirely in fixed-point** — the datapath twin of the
integer phase carrier. `Sin` (float, radians) takes an already-reduced
argument and evaluates a float polynomial; `FixedSin` never leaves i64:
its argument is a **Q0.32 phase in cycles** (the raw output of
`FixedPhasor`/`ClockPhasor` *before* the `/2³²` float scale), its range
reduction is masking and shifts (exact, warp-transparent), and its
polynomial is Q2.30 integer arithmetic. The output is a **Q2.30 sample**
(`sin ∈ [−1, 1]` as `out/2³⁰`), scaled to float once at the boundary by
whoever mixes it.

Working in cycles rather than radians is what deletes the float range
reduction: one turn is exactly `2³²`, so "reduce to the nearest
half-turn" is `(phase + 2³⁰) >> 31` and the residual is already the
Q2.30 quarter-turn coordinate. There is no `round(x/π)` and no π
rounding error — the phase never existed in float.

## The algorithm

With `P` the Q0.32 phase (one cycle = `2³²`):

- **half-turn index** `n = (P + 2³⁰) >> 31` — the nearest multiple of a
  half turn (π), so the residual is a quarter-turn coordinate.
- **residual** `r = P − n·2³¹`, `r ∈ [−2³⁰, 2³⁰)` — this *is* the
  scaled argument `s = r/2³⁰ ∈ [−1, 1)` in Q2.30, with
  `sin(2π·P/2³²) = (−1)ⁿ · sin((π/2)·s)`.
- **parity sign** `sign = 1 − 2·(n & 1)`.
- **polynomial** `sin((π/2)s) = s · Σₖ aₖ (s²)ᵏ`, the degree-15 Taylor
  series of `sin((π/2)s)` — truncation ≈ 6.7e-12, far below the Q2.30
  quantum 2⁻³⁰. Coefficients land as `round(|aₖ|·2³⁰)`; the signs
  strictly alternate, so Horner is written **all-positive with
  subtractions** — every intermediate `accₖ` is non-negative, so every
  `>> 30` rescale (arithmetic shift = floor) sees a non-negative operand
  and truncates cleanly. The one signed shift is the final `(r·acc₀) >>
  30`, floor-rounded; the ≤1-ulp asymmetry at negative `r` is accepted
  and inside the error budget (max abs error ≈ 1e-8 ≈ −160 dB).

| k | aₖ = (−1)ᵏ(π/2)²ᵏ⁺¹/(2k+1)! | round(&#124;aₖ&#124;·2³⁰) |
|---|------------------------------|--------------------|
| 0 | 1.5707963267948966           | 1686629713 |
| 1 | −0.6459640975062463          | 693598668 |
| 2 | 0.07969262624616705          | 85569306 |
| 3 | −0.004681754135318688        | 5026995 |
| 4 | 0.00016044118478735983       | 172272 |
| 5 | −3.598843235212085e-06       | 3864 |
| 6 | 5.692172921967927e-08        | 61 |
| 7 | −6.688035109811468e-10       | 1 |

Every product of two Q2.30 values fits i64 (`|acc| ≤ π/2 < 2` ⇒
`acc·z < 2⁶¹`), and each multiply is 32×32→64-decomposable — the
property that lets an f32-native backend (Metal) evaluate this datapath
**byte-identically** to the JIT.

## Source

```tropical
program FixedSin(phase: int = 0) -> (out: int) {
  out = let {
      n: (phase + 1073741824) >> 31
    } in let {
      r: phase - (n << 31),
      sign: toInt(1) - toInt(2) * (n & 1)
    } in let {
      z: (r * r) >> 30
    } in let {
      acc6: 61 - (z >> 30)
    } in let {
      acc5: 3864 - ((acc6 * z) >> 30)
    } in let {
      acc4: 172272 - ((acc5 * z) >> 30)
    } in let {
      acc3: 5026995 - ((acc4 * z) >> 30)
    } in let {
      acc2: 85569306 - ((acc3 * z) >> 30)
    } in let {
      acc1: 693598668 - ((acc2 * z) >> 30)
    } in let {
      acc0: 1686629713 - ((acc1 * z) >> 30)
    } in sign * ((r * acc0) >> 30)
}
```

The `phase` contract is a **masked Q0.32 value in `[0, 2³²)`** — what
the phasor's `& 4294967295` produces. Cosine is the exact quarter-turn
shift of the same table: `FixedSin((phase + 1073741824) & 4294967295)`.
