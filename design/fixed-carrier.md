# The fixed-point carrier (scope A) — Q-format ledger

Status: Phase A (modal island on the integer clock) + Phase B1 (the fixed-point
datapath sine) landed on `fixed-carrier`. This file is the numeric contract:
formats, rounding, coefficients, and the typing trap. The Metal consequence is
recorded in `metal-emitter-evaluation.md`.

## Why

Two independent motivations converged:

1. **Long-τ exactness.** The one production site that carried unbounded float
   time — the modal island's `ω·dSec` into `sinSig` — lost mantissa as τ grew,
   and lost it *differently* under warped (fractional) clocks, breaking
   time-translation exactness (the `modal-longtau` gate fails 824/1024 on the
   old body). The phase now reduces on ℤ/2³² (`relClockQ` + `modePhaseW` in
   `EmitArrow.lean`); precision is τ-independent.
2. **Metal.** Apple GPUs are f32-native; a float-datapath kernel can never be
   bit-exact across CPU/GPU. An i64 datapath whose multiplies are
   32×32→64-decomposable CAN be. `FixedSin` is that datapath for the sine.

## Q-format ledger

| quantity | format | storage | notes |
|---|---|---|---|
| clock | Q48.16 signed | i64 | τ itself (`ENode.tau`, the port's coordinate); the integer sample index is the derived `tau >> 16`. One definition: `clockFracBits` in `lean/Tropical/Ir/ClockFormat.lean` (C++ `engine/runtime/ClockFormat.hpp`, TS `CLOCK_FRAC_BITS`); the manifest names it as τ's element type `i64q16` |
| speed (scrub velocity) | Q·.32 signed | i64 | `toInt(velocity·2³²)`; `scaleClockQ` returns `⌊s·c/2³²⌋`, a clock of the same format as `c`, so speed resolution does not depend on the clock's |
| phase | Q0.32 (masked, cycles) | i64 | one turn = 2³²; `fixedPhase`/`modePhaseW`/ClockPhasor's pre-`toFloat` accumulator |
| half-turn index | int | i64 | `n = (P + 2³⁰) >> 31` |
| quarter-turn residual `s` | Q2.30 signed | i64 | `r = P − (n<<31)`; `r` IS `s = r/2³⁰ ∈ [−1, 1)` |
| `z = s²` | Q2.30 | i64 | `(r·r) >> 30`; product ≤ 2⁶⁰ |
| poly coefficients, Horner acc | Q2.30 | i64 | `round(|aₖ|·2³⁰)`, all-positive-with-subtractions (signs strictly alternate) |
| sample values (sin out) | Q2.30 signed | i64 | `sign · ((r·acc₀) >> 30)`; cos = sin at `(P + 2³⁰) & (2³²−1)` — exact quarter turn |
| N-way sums | i64 add | i64 | modular ⇒ associative + commutative — the law-6 (reassociation) property |
| boundary | `toFloat(x)/2^frac` | double | `fixedOutQ` — applied identically on both law sides |

**Rounding convention: floor.** Every `>> 30` rescale is `ashr`. The Horner is
written so all intermediate accs are non-negative (floor = truncate, clean);
the single signed floor-shift is the final `(r·acc₀) >> 30`, whose ≤1-ulp
asymmetry at negative `r` is inside budget. Do not add round-to-nearest bias
anywhere without re-freezing the corpus.

## Horizon

The clock's split trades sub-sample resolution for horizon. With `F` fraction
bits a coordinate resolves `2⁻ᶠ` of a sample and stays non-negative for
`2⁶³⁻ᶠ` samples:

| format | horizon at 48 kHz | at 96 kHz | resolution at 48 kHz |
|---|---|---|---|
| Q32.32 (2026-10-28 to 2026-11-01) | 12.4 h | 6.2 h | ≈ 5 fs |
| **Q48.16** | **93 years** | **46 years** | ≈ 0.3 ns |

Phase (Q0.32) and speed (Q·.32) keep their own 32 fraction bits, so neither
frequency nor speed resolution moved. The free-running phasors reduce on
ℤ/2³² and do not notice τ's sign bit (`inc · 2⁶⁴⁻ᶠ ≡ 0 mod 2³²`); what does
notice it is a non-integer speed (`v·τ` is not a homomorphism of the wrapped
ring), the strike-anchor landing `toInt(anchor·2ᶠ)` (out of range past the
horizon), and the derived `sampleIndex`. The `long-horizon` gate renders all
three past 2³¹ samples and at 2⁴⁰ through the host's τ fill; on Q32.32 each
leg failed at some origin.

Clocks that cross an instance boundary ride a module slot. The JIT and wasm
slot vector is f64, so a wired clock is exact while it has ≤ 53 significant
bits: every whole-sample clock to 2⁵³ samples, a clock with all 16 fraction
bits live to 2³⁷ samples (≈ 36 days at 44.1 kHz). A nested output is read at
its producer's declared type (`EmitSlots.nestedOutputTypes`), so on Metal a
wired clock stays a `long` instead of passing through f32.

**Overflow ledger:** `|acc| ≤ π/2·2³⁰ < 2³¹`, `z ≤ 2³⁰` ⇒ every product
< 2⁶¹; phase split-multiply `incr·tlo < 2³²·2¹⁶` for `incr < 2³²` (freq <
SR — always true for audio rates). No saturation anywhere; sums wrap modularly
by design. The phasor's whole-sample product `inc · (clk >> F)` wraps
modularly past `2⁶³ / inc` samples; only its low 32 bits survive the phase
mask, so the wrap is exact.

## Coefficients (degree-15 Taylor of sin((π/2)s) in z = s²)

`aₖ = (−1)ᵏ(π/2)^(2k+1)/(2k+1)!`, landed `round(|aₖ|·2³⁰)` (round-half-even,
50-digit π): `1686629713, 693598668, 85569306, 5026995, 172272, 3864, 61, 1`.
Truncation of the dropped degree-17 term ≈ 6.7e-12 ≪ 2⁻³⁰. Measured end-to-end
against the true sine at the exactly-known integer phase: **max abs error
3.2e-9** on the sin scale (`fixedsin-accuracy` gate, 4096 samples).

## The typing trap (litI)

`inferResultType`/`promoteTypes` float-promote any `add`/`mul`/`sub` whose
literal operand isn't in an int-EXPECTED context (bitwise ops and `toInt`
propagate int down; nothing else does). Worse, emit CSE keys on
`(ExprId, expected)`, so ONE clock expression can compile int in the phase
path and float in a `toFloat`/compare path of the same kernel — the float copy
sheds sub-sample bits past 2⁵³ rail units (2³⁷ samples, ≈ 36 days at 44.1 kHz,
on Q48.16; it was ≈ 47 s on Q32.32). Rule:
**every literal that meets a clock rides `litI` (EmitArrow) / `toInt(...)`
(surface)**. `relClockQ`/`modePhaseW` are robust because both operands are
forced int by their head ops.

## Gates

- `long-horizon` — scrub(½) on far coordinates ≡ the default clock at the
  half-sample instant, a modal bank struck at A+200 ≡ struck at 200, and
  `sampleIndex` (arrow and wire) exact, at B = 2³¹−256 and 2⁴⁰ samples
  through `Runtime.renderTau`; also pins `i64q16` to `clockFracBits`.

- `modal-longtau` — modal bank struck+read 2³⁰ samples later ≡ origin,
  byte-exact, on fractional (scrub-form) clocks. Discriminating: 824/1024
  differ on the pre-fix body.
- `fixedsin-accuracy` — Q2.30 sine vs true sine at the exact integer phase,
  < 2e-8 (measured 3.2e-9).
- `fixedsin-longtau` — fixed osc at τ+2³⁰+12345 samples ≡ origin osc shifted
  by `(inc·K) mod 2³²`, byte-exact (the modular identity the float carrier
  never had).
- `corpus-gate/FixedSin` — `stdlib/FixedSin.md` and `fixedSinCycSig` emit
  byte-identical plans (lockstep proof).

## Float residue (deliberate)

Stays float, with reasons:
- **Envelope `exp`** (`expSig`) — argument is the BOUNDED relative-clock
  seconds post-Phase-A; decay makes far-field precision irrelevant. On Metal
  this is the enumerated tolerance-gated residue.
- **Param landings** — `toIntE(x·2³²)` / `toIntE(x·2³⁰)`: one float multiply
  per live param, landing on the integer grid (ω quantizes to SR/2³² ≈ 1e-5 Hz
  — inaudible, documented at `modePhaseW`).
- **DAC scale** — one `toFloat`/divide at the boundary (`fixedOutQ`).

## Remaining (B2/B3, deferred if the day runs out)

- B2: `FixedSinOsc`/`MorphOsc` voices atop `FixedSin` (re-freeze voice
  goldens; ear check).
- B3: modal bank datapath in Q2.30 (env landed once per mode via
  `toIntE(env·2³⁰)`, integer mode sums ⇒ associative), per-law carrier table
  at the law block.
