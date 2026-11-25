# Rational clocks

Date: 2026-11-25. Code: `RatioClock` and `ClockMultDivQ15` in
`lean/Tropical/EmitArrow/Stdlib.lean`, the ring helpers in
`EmitArrow/Numerics.lean`. Model and proofs:
`lean/Tropical/Testing/ClockMultDivLaws.lean`. Gates:
`tropicaltest --clock-mult-div-only`, and the `RatioClock` cases in
`tests/web/wasm_vs_jit.test.ts`.

A rational clock runs at `n/d` of a master frequency. Two of them drift
against each other audibly and still recur exactly: they share one master
position, so their phase relation is a fixed integer relation, forever.

## `RatioClock`

| port | kind | meaning |
|---|---|---|
| `clk` | τ | the clock, Q48.16 |
| `freq` | `int32` Q16.16 | master frequency in Hz |
| `n`, `d` | `int32` | the ratio; the low four bits are read, so `n, d ∈ [0, 15]` |
| `offset` | `int32` Q0.32 | phase offset in cycles |
| `phase` (out) | `int` | the phase on the Q0.32 rail, in `[0, 2³²)` |

`phase` is the input every spline and `FixedSin` read, so a clock wires
into a spline bare: `{"phase": {"op": "ref", "instance": "clk1", "output":
"phase"}}`.

## Construction

- The master position is `Ψ ∈ ℤ/2⁶⁴`, `Ψ = ⌊I · τ / 2¹⁶⌋`. One wrap of the
  ring is `Λ = lcm(1..15) = 360360` master cycles, so the increment is
  `I = ⌊F · K / 2¹⁶⌋` with `K = round(2⁶⁴ / (Λ · rate))`.
- The phase is the top word of `m·Ψ + offset·2³²` with `m = n · (Λ / d)`,
  an integer below 2²³. `Λ / d` is a table of 15 constants. The ring product
  divides nothing and drops nothing.
- Ψ depends on `clk` and `freq` only. Every `RatioClock` on the same clock
  and master computes the same Ψ, so separate instances are locked.
- `d = 0` or `n = 0` stops the clock at its offset.

## What is proved and what is gated

Proved over the model (no `sorry`, standard axioms):

- `multiplier_exact`: `d · m = n · Λ`, so `m` is exact.
- `phase_recurs`, `cycles_per_universal`: one wrap of Ψ returns every
  clock to its phase, and clock `i` turns exactly `mᵢ` cycles in it.
- `lock`: `nⱼ·dᵢ·(Pᵢ − oᵢ) = nᵢ·dⱼ·(Pⱼ − oⱼ)` on ℤ/2⁶⁴ at every Ψ.
- The 32-bit limb identities (`mulWideU32_eq`, `mulLo64_eq`, …) and the
  clock split of `ringPosition64`.

Gated against the JIT: the kernel's phase words equal the model over live
writes, the full Q16.16 range and τ to ±2⁶². A spline read by a
`RatioClock` equals the same spline read at that phase word through an
`int` param, bit for bit. All channels repeat after one wrap. WASM equals
the JIT bit for bit.

## Recurrence in τ

The lock is exact for every `freq`. Recurrence in samples is exact when
`2⁶⁴ / I` is a whole number of samples, that is when the 2-adic valuation
of `I` is at least 18 at 48 kHz. Otherwise Ψ recurs in ring units but not
on a sample boundary. The realized master frequency differs from `F` by at
most `|F|·2⁻¹⁷` plus one increment LSB (9.4e-7 Hz at 48 kHz). The
universal cycle is `Λ / f` seconds: 100 hours at 1 Hz, 6 minutes at 1 kHz.

## Why 64 bits

A 32-bit ring would have an increment LSB of `Λ · 48000 / 2³² ≈ 4 kHz`.
With 64 bits it is 9.4e-7 Hz, finer than the Q16.16 knob's own step.

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

- Glide-to-lock, snap, and a held phase at `n = 0`. Each needs an integer
  anchor discipline at the param port: the host captures the phase
  difference and write time, and the kernel reads them as two more params.
- Ratios with `d` above 15 (they need a larger `Λ`).
