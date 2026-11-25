# Rational clocks

Date: 2026-11-25. Code: `RatioClock` and `ClockMultDivQ15` in
`lean/Tropical/EmitArrow/Stdlib.lean`, the ring helpers in
`EmitArrow/Numerics.lean`. Model and proofs:
`lean/Tropical/Testing/ClockMultDivLaws.lean`. Gates:
`tropicaltest --clock-mult-div-only`, and the `RatioClock` cases in
`tests/web/wasm_vs_jit.test.ts`.

A rational clock runs at `n/d` of a master frequency. Two of them drift
against each other audibly and stay locked exactly: they share one master
position, so their phase relation is a fixed integer relation, forever.

Status: a draft. Its surface (the `freq` format, the nibble knobs, one ring
position shared with the spectral `ratios` column, where `ClockMultDivQ15`
lives) is open; see PR #262.

## `RatioClock`

| port | kind | meaning |
|---|---|---|
| `clk` | τ | the clock, Q48.16 |
| `freq` | `int32` Q16.16 | master frequency in Hz |
| `n`, `d` | `int32` | the ratio; only the low four bits are read, so 16 reads as 0 (the clock stops) and −1 reads as 15 |
| `offset` | `int32` Q0.32 | phase offset in cycles |
| `phase` (out) | `int` | the phase on the Q0.32 rail, in `[0, 2³²)` |

`phase` is the input every spline and `FixedSin` read, so a clock wires
into a spline bare: `{"phase": {"op": "ref", "instance": "clk1", "output":
"phase"}}`.

## Construction

- The master position is `Ψ ∈ ℤ/2⁶⁴`, `Ψ = ⌊I · τ / 2¹⁶⌋`. One wrap of the
  ring is the common period `Λ = lcm(1..15) = 360360` master cycles, so the increment is
  `I = ⌊F · K / 2¹⁶⌋` with `K = round(2⁶⁴ / (Λ · rate))`.
- The phase is the top word of `m·Ψ + offset·2³²` with `m = n · (Λ / d)`,
  an integer below 2²³. `Λ / d` is a table of 15 constants. The ring product
  divides nothing and drops nothing.
- Ψ depends on `clk` and `freq` only. Every `RatioClock` on the same clock
  and the same `freq` word computes the same Ψ, so separate instances are
  locked (see "The lock").
- `d = 0` or `n = 0` stops the clock at its offset.

## The lock

The lock holds on the full Q0.64 phases `Pᵢ = mᵢ·Ψ + oᵢ·2³²`:
`nⱼ·dᵢ·(Pᵢ − oᵢ·2³²) = nᵢ·dⱼ·(Pⱼ − oⱼ·2³²)` on ℤ/2⁶⁴, at every Ψ, for every
`freq`. The `phase` output is the top word of `Pᵢ`, so the observed 32-bit
words satisfy the relation only up to each word's floor.

The ratio must go in `n` and `d`. Two clocks share Ψ only if they read the
same `freq` word: a clock at `freq` 2F and one at `freq` F with `n = 2` are
not locked, because `I(2F)` and `2·I(F)` can differ by one LSB.

## What is proved and what is gated

Proved in `Testing/ClockMultDivLaws.lean` (no `sorry`, standard axioms):

- `multiplier_exact`: `d · (n · (Λ / d)) = n · Λ` for `d ∈ [1, 15]`, and
  `multiplier_lt`: the multiplier is below 2²³.
- `lock`: a generic identity on `BitVec 64` over free variables: if
  `dᵢ·mᵢ = nᵢ·Λ` and `dⱼ·mⱼ = nⱼ·Λ`, then
  `nⱼ·dᵢ·((mᵢ·Ψ + oᵢ) − oᵢ) = nᵢ·dⱼ·((mⱼ·Ψ + oⱼ) − oⱼ)`. It is not
  instantiated for the model's `phase` and `multiplier`; `multiplier_exact`
  supplies the hypothesis.
- `phase_recurs`: `m·(Ψ + 2⁶⁴) + o ≡ m·Ψ + o`. It is about Ψ only and
  says nothing about τ. `cycles_per_universal`: in one wrap of Ψ, clock
  `n/d` turns exactly `m` cycles.
- The limb identities (`mulWideU32_eq`, `mulLoU32By64_eq`, `mulLo64_eq`,
  `hiWord_of_limbs`, `mulTop_eq`) over natural numbers, assuming each limb
  product is exact, and `position_split`, the clock split of
  `ringPosition64`. They do not cover a signed `F` in `hzQ16ToInc64`; the
  gate does.

Gated against the JIT (`tropicaltest --clock-mult-div-only`, at the
Lean runtime's 44.1 kHz):

- The kernel's phase words equal the model over live writes, `freq` over the
  whole Q16.16 word, knob words outside 0..15, and τ out to ±2⁶².
- A spline read by a `RatioClock` equals, bit for bit, the same spline read
  at that phase word by a reference patch whose phase is a computing wire on
  the clock (`τ >> 16`, rendered at `τ = word·2¹⁶`). The phase words are
  observed through a gain-1 `VCA` on `toFloat(phase)`, which is exact in f64
  (below 2³²) and only there.
- At a master whose increment has `v₂(I) ≥ 18`, every channel repeats bit for
  bit after `2^(64 − v₂(I))` samples.
- WASM equals the JIT bit for bit (`tests/web/wasm_vs_jit.test.ts`).

At `--float f32` the constant `K` is computed on the float lane, and
model/JIT equality is not gated there. Computing `K` as an exact integer at
assemble time would remove the question.

## Recurrence

The lock is exact. Recurrence is a separate matter:

- **Exact recurrence.** The sampled position `Ψ(s) = I·s mod 2⁶⁴` always
  recurs exactly, with a period of `2^(64 − v₂(I))` samples, where `v₂` is
  the 2-adic valuation. For most masters that period is far beyond τ's
  range. The gate picks a master with `v₂(I) ≥ 18` only to bring the period
  inside it. Its masters (at 44.1 kHz) are word 147087 (2.244 Hz,
  `v₂ = 19`, a period of 2⁴⁵ samples, 25 years, 4969 wraps of Ψ) and word
  294174 (`v₂ = 20`, 2⁴⁴ samples, again 4969 wraps). The period is not one
  wrap of Ψ.
- **Near-recurrence.** After one common period, `Λ / f` seconds (100 hours
  at 1 Hz, 6 minutes at 1 kHz), Ψ is back at 0 to within half a sample's
  advance. Each clock is then within half a sample's phase advance of its
  offset: `(n/d)·f / (2·rate)` cycles, 4.6e-3 cycles for the master at
  440 Hz and 48 kHz. That is sub-sample.

## Precision

- **Increment LSB:** `Λ·rate/2⁶⁴` Hz of master. That is 9.4e-10 Hz at
  48 kHz and 8.6e-10 Hz at 44.1 kHz.
- **The realized master** is `I·Λ·rate/2⁶⁴`. It differs from `F` by at most
  `|F|·Λ·rate/2⁶⁵` (a relative error of 4.7e-10 at 48 kHz, from rounding
  `K`) plus one LSB (from the floor).
- **The error a user meets is the Q16.16 knob's own truncation,** up to
  2⁻¹⁶ Hz = 1.5e-5 Hz. It dominates both terms above.
- **Why 64 bits:** a 32-bit ring's LSB would be `Λ·48000/2³²` = 4.03 Hz.

## Realization

Every ring product is spelled in 32-bit multiplies (`mulWideU32`,
`mulLoU32By64`, `mulLo64`). An `int` `mul` is `__aeabi_lmul` on thumbv6m,
which `compile-object` refuses. The `Λ / d` table is read bit-sliced (a
shift, a mask and an or per bit), so there is no branch on Thumb-1. This is
an authoring-time choice: nothing below the waist expands an `i64` `mul`
for a core without `UMULL`.

## `ClockMultDivQ15`

Four `RatioClock` phases of one Ψ, with Q17.15 `gate` and `sine` outputs
and `int32` Q0.32 `phase` outputs. It is the Cortex-M0+ fixture
(`tests/fixtures/embedded_clock_mult_div.json`). It exists because an
embedded patch's outputs are `int32` channels and an `int` slot has no
output element type. A `phaseN` output is `int32`, so it reaches the spline's
`int` input through `toInt` (a computing wire); the spline masks the low
32 bits, so the sign extension does not matter.

## Not built

- Click-free retune, glide-to-lock, snap, and a held phase at `n = 0`.
  `freq`, `n` and `d` are integer typed slots, so they take only raw writes.
  A ring anchor needs a 64-bit Ψ-shift companion shared per master param. A
  per-clock Q0.32 offset would break the exact lock.
- Ratios with `d` above 15 (they need a larger `Λ`).
