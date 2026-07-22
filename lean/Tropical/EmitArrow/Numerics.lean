import Tropical.EmitArrow.Sig

/-!
# EmitArrow.Numerics — closed-form scalar kernels

The numeric datapaths every generator/envelope is built from, as pure `Sig`
functions: the exact Q2.30 fixed-point sine/cosine over a Q0.32 cycles phase,
the integer split-multiply phasor (`phasorPhaseSig`), and the float
polynomial transcendentals (`sinSig`, `cosSig`, `expSig`). `sin`/`cos`/`exp`
are pure polynomials — that op set never grows past `{+, ×, round, clamp,
ldexp}`. `logSig` (the gauge norm's inverse of `exp`) adds exactly two: the
`floatExponent` exponent-extract (option E's op) and one `div` (the `atanh`
argument) — coefficient-time only in its modal uses, so no per-sample cost.
-/

namespace Tropical.EmitArrow

open Tropical.Ir

-- ── The FIXED-POINT SINE: the sample datapath in i64 ─────────────────────────
-- `stdlib/FixedSin.md`'s algorithm as a Sig builder, kept in LOCKSTEP with the
-- .md (corpus-gated byte-identical). Cycles domain: the argument is the Q0.32
-- phase (one turn = 2³²) straight off the integer phasor — range reduction is
-- masking/shifts, exact and warp-transparent; there is no float π anywhere.
-- Output is a Q2.30 sample. Every multiply of two Q2.30 values fits i64 and is
-- 32×32→64-decomposable — what lets an f32-native backend (Metal) run this
-- datapath byte-identically to the JIT.

/-- `sin` over a MASKED Q0.32 cycles phase (`[0, 2³²)`), returning Q2.30.
    Half-turn index `n = (P + 2³⁰) >> 31`; residual `r = P − n·2³¹` IS the
    quarter-turn coordinate `s = r/2³⁰` in Q2.30; parity sign; degree-15
    Taylor of `sin((π/2)s)` in `z = s²`, Horner all-positive-with-subtractions
    (signs strictly alternate), so every `>> 30` rescale sees a non-negative
    operand. The final `(r·acc₀) >> 30` is the one signed floor-shift
    (≤1-ulp asymmetry at negative r, inside the ~1e-8 error budget). -/
def fixedSinCycSig (phaseQ : Sig) : Sig :=
  let n := rshift (add phaseQ (lit 1073741824)) (lit 31)
  let r := sub phaseQ (lshift n (lit 31))
  let sign := sub (litI 1) (mul (litI 2) (bitAnd n (lit 1)))
  let z := rshift (mul r r) (lit 30)
  let acc6 := sub (lit 61) (rshift z (lit 30))
  let acc5 := sub (lit 3864) (rshift (mul acc6 z) (lit 30))
  let acc4 := sub (lit 172272) (rshift (mul acc5 z) (lit 30))
  let acc3 := sub (lit 5026995) (rshift (mul acc4 z) (lit 30))
  let acc2 := sub (lit 85569306) (rshift (mul acc3 z) (lit 30))
  let acc1 := sub (lit 693598668) (rshift (mul acc2 z) (lit 30))
  let acc0 := sub (lit 1686629713) (rshift (mul acc1 z) (lit 30))
  mul sign (rshift (mul r acc0) (lit 30))

/-- `cos` as the exact quarter-turn shift of `fixedSinCycSig` — masked add,
    no second table, no float. -/
def fixedCosCycSig (phaseQ : Sig) : Sig :=
  fixedSinCycSig (bitAnd (add phaseQ (lit 1073741824)) (lit 4294967295))

/-- Scale a Q-fractional integer sample to float at the DAC boundary —
    `toFloat(x)/2^fracBits`, the generalization of `fixedOut` (Q0.32) to any
    Q format (the fixed sine is Q2.30 → `fixedOutQ 30`). Applied identically
    on both sides of any law, so byte-identical ints render byte-identical
    floats. -/
def fixedOutQ (fracBits : Nat) (x : Sig) : Sig :=
  div (toFloatE x) (lit (Int.pow 2 fracBits))

-- ── The BOOTSTRAP (part 1): the phasor + sine as pure Sig, over {+, ×, frac} ──
-- The generators (phasor, sine) were the last thing the arrow layer borrowed from
-- `.trop`. These pointwise pieces need no ArrowTerm; the term wrapper that lifts
-- them onto the clock leaf lives below `emitTerm`. Gated bit-exact vs `.trop`
-- `FixedSinOsc` by `bootstrap-sin`.

/-- `ClockPhasor`'s phase as pure arithmetic over a Q32.32 clock signal: the exact
    integer split-multiply `acc = inc·thi + (inc·tlo)>>32 + off`, masked to
    `[0,2³²)`, then `/2³²` to a unipolar float. `inc = ⌊freq·2³²/SR⌋`,
    `off = ⌊offset·2³²⌋`. Structurally `stdlib/ClockPhasor.md`, as a `Sig`. -/
def phasorPhaseSig (freqE offsetE clkSig : Sig) : Sig :=
  let inc := toIntE (div (mul freqE (lit 4294967296)) .sampleRate)
  let off := toIntE (mul offsetE (lit 4294967296))
  let thi := rshift clkSig (lit 32)
  let tlo := bitAnd clkSig (lit 4294967295)
  let acc := add (add (mul inc thi) (rshift (mul inc tlo) (lit 32))) off
  div (toFloatE (bitAnd acc (lit 4294967295))) (lit 4294967296)

/-- `stdlib/Sin`: range-reduce `r = x − round(x/π)·π`, sign from `n & 1`, a Horner
    polynomial in `r²` with the same minimax coefficients. Pointwise `Sig → Sig` —
    every transcendental is a polynomial, so it needs only `+`/`×` (and `round`
    for the range reduction). The `c0` term drops the fold's leading `0·r²` (a
    bit-exact no-op). -/
def sinSig (x : Sig) : Sig :=
  let n := roundE (mul x (lit 3183098861837907 16))
  let r := sub x (mul n (lit 3141592653589793 15))
  let sign := sub (lit 1) (mul (lit 2) (bitAnd n (lit 1)))
  let r2 := mul r r
  let poly :=
    add (lit 1) (mul (
    add (lit (-16666666666666666) 17) (mul (
    add (lit 8333333333333333 18) (mul (
    add (lit (-1984126984126984) 19) (mul (
    add (lit 27557319223985893 22) (mul (
        lit (-2505210838544172) 23) r2)) r2)) r2)) r2)) r2)
  mul sign (mul r poly)

/-- `stdlib/Exp`: `e^x` as `e^r · 2ⁿ`, `n = round(x·log₂e)`, `r = x − n·ln2`
    (ln2 split hi/lo for precision), `e^r ≈ 1 + r·(1 + r·p(r))` with the same
    6-term minimax `p`, then the exact `ldexp(exp_r, n)`. Pointwise `Sig → Sig`,
    only `{+, ×, round, clamp, ldexp}` — the decaying-envelope primitive the
    modal island needs, the exp twin of `sinSig`. -/
def expSig (x : Sig) : Sig :=
  let clamped := clampE x (lit (-87)) (lit 88)
  let n := roundE (mul clamped (lit 14426950408889634 16))
  let r := sub (sub clamped (mul n (lit 693145751953125 15)))
                            (mul n (lit 14286068203094173 22))
  let p :=
    add (lit 50000001201 11) (mul (
    add (lit 16666665459 11) (mul (
    add (lit 41665795894 12) (mul (
    add (lit 83334519073 13) (mul (
    add (lit 13981999507 13) (mul (lit 198756915 12) r)) r)) r)) r)) r)
  let exp_r := add (lit 1) (mul r (add (lit 1) (mul r p)))
  ldexpE exp_r n

/-- `ln x` for `x > 0` — `expSig`'s mirror, the second float transcendental the
    modal island needs (the self-measuring excitation gauge's `‖H‖^{−g}` wants a
    live `log`; `norm^{−g} = exp(−g·ln norm)`). Range-reduce by the IEEE exponent
    to the mantissa `m ∈ [√½, √2)` (`e = ⌊log₂ x⌋`, `m = x·2^{−e}`, halved when
    `m > √2` so the series stays centred), then `ln m = 2·atanh s`, `s = (m−1)/(m+1)`
    — a 5-term series (`|s| ≤ 0.172`, so the `s¹⁰` tail is < 1e-8). `ln x = e·ln2
    + ln m`. Two ops past the `{+,×,round,clamp,ldexp}` sine/exp set — `floatExponent`
    (the exponent extract) and `div` (the `atanh` argument); both already in the Sig
    vocabulary (`floatExponent` is option E's dynamic-`k` op). COEFFICIENT-TIME only
    in its modal uses (the gauge norm folds pole/residue slots, an s0 expression), so
    `floatExponent`'s f32-vs-f64 Metal split never runs per sample — the same
    discipline that keeps option E's dynamic `k` Metal-safe. Domain `x > 0` (like
    `Float.log`); the gauge floors its norm above 0 before calling. -/
def logSig (x : Sig) : Sig :=
  let e0 := Sig.unary .floatExponent x              -- ⌊log₂ x⌋ (float)
  let m0 := ldexpE x (neg e0)                       -- x·2^{−e0} ∈ [1, 2)
  let big := gt m0 (lit 14142135623730951 16)       -- m0 > √2 ⇒ halve, bump the exponent
  let m := selectE big (mul m0 (lit 5 1)) m0        -- m ∈ [√½, √2)
  let e := selectE big (add e0 (lit 1)) e0
  let s := div (sub m (lit 1)) (add m (lit 1))      -- |s| ≤ 0.172
  let s2 := mul s s
  let p :=                                          -- atanh(s)/s = 1 + s²/3 + s⁴/5 + s⁶/7 + s⁸/9
    add (lit 1) (mul s2 (
    add (lit 3333333333333333 16) (mul s2 (
    add (lit 2 1) (mul s2 (
    add (lit 14285714285714285 17) (mul s2 (
        lit 1111111111111111 16))))))))
  add (mul e (lit 6931471805599453 16)) (mul (mul (lit 2) s) p)

/-- 2π and π/2 as `Sig` literals; `cos x = sin(x + π/2)` reuses the gated `sinSig`. -/
def twoPiE : Sig := lit 6283185307179586 15
def halfPiE : Sig := lit 15707963267948966 16
def cosSig (x : Sig) : Sig := sinSig (add x halfPiE)

end Tropical.EmitArrow
