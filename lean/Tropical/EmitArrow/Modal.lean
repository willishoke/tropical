import Tropical.EmitArrow.Numerics
import Tropical.Ir.BanksFlag
import Tropical.EmitArrow.Term

/-!
# EmitArrow.Modal — the pole island: modal banks + the residue calculus

A modal bank is a gated sum of decaying sinusoids — the real part of
`Σ A·d^deg·e^{μd}` — evaluated as a pure function of the (already-warped)
clock: random-access, no state. `ModalMode` is the rectangular pole/amp
record; `bankFold` is the generic banked reduction over coefficient columns
(`Sig.bankSum` — O(1) plan size in mode count); the `*Table` lowerings are
its bit-identical banked twins. Composition (`voice ⋙ reverb`) is the
SYMBOLIC residue calculus (`residueComposeE`/`residueComposeEC`) over
`CplxE` — pure `+−×÷` on `Sig`, so poles and amps stay live param slots
through a room. The direction operator crossfades causal/anti-causal tails
without touching σ or ω.
-/

namespace Tropical.EmitArrow

open Tropical.Ir

/-- One mode in RECTANGULAR form: `d^deg · e^{−σd} · (c_re·cos(ωd) − c_im·sin(ωd))
    = Re(A · d^deg · e^{μd})` with pole `μ = −σ + iω` and complex amp `A = c_re +
    i·c_im`. Rectangular (not magnitude/phase) so the residue calculus can emit the
    coefficients as pure `+−×÷` expressions — no `sqrt`/`atan2` — which is what lets
    every field be a LIVE param slot, not a baked literal. Authored modes set
    `ω = 2π·f`, `c_re = amp`, `c_im = 0`. -/
structure ModalMode where
  sigma : Sig           -- σ = −Re μ  (decay)
  omega : Sig           -- ω = Im μ   (rad/s)
  cre   : Sig           -- Re A
  cim   : Sig := lit 0   -- Im A
  deg   : Nat := 0
  /-- WS-LP: the closed interval a LIVE `sigma` ranges over (the knob's span,
      supplied by the authoring site — e.g. the rt60 knob's display range mapped
      through σ = 6.91/rt60). The region classifier's input when `sigma` doesn't
      fold to a constant; the lifted kernel clamps the live σ to this interval,
      so the classification is sound by construction. `none` + live σ ⇒ the
      baked-pole fallback (bare bloom). -/
  sigmaRange : Option (Float × Float) := none

/-- Authored mode from (freq Hz, decay σ, real amp): `ω = 2π·f`, `c_re = amp`,
    `c_im = 0`. Any of `fHz`/`sigma`/`amp` may be a live `paramRef`. -/
def ModalMode.hz (fHz sigma amp : Sig) (deg : Nat := 0) : ModalMode :=
  { sigma, omega := mul twoPiE fHz, cre := amp, deg }

/-- `base^k` by repeated multiply (k is a mode's small polynomial order). -/
def powE (base : Sig) : Nat → Sig
  | 0 => lit 1
  | 1 => base
  | n + 1 => mul (powE base n) base

/-- Fold an authored-constant `Sig` back to its `Float` (the baked-pole read in
    `bloomCompose`, and the static-`k` amplitude read in `bankLandExp`). Partial:
    `none` on any live/unsupported node — a live pole is outside the baked-pole
    contract, not an error to paper over. (Hoisted above the landing sites so
    `bankLandExp` can consult it; public so the test witnesses can read baked
    `BloomPair` fields back as Floats.) -/
partial def sigConstF? : Sig → Option Float
  | .num n            => some n.toFloat
  | .unary .neg a     => (sigConstF? a).map (fun x => -x)
  | .unary .toFloat a => sigConstF? a
  | .binary .add a b  => do pure ((← sigConstF? a) + (← sigConstF? b))
  | .binary .sub a b  => do pure ((← sigConstF? a) - (← sigConstF? b))
  | .binary .mul a b  => do pure ((← sigConstF? a) * (← sigConstF? b))
  | .binary .div a b  => do
      let x ← sigConstF? a; let y ← sigConstF? b
      pure (if y == 0.0 then 0.0 else x / y)
  | _                 => none

-- ── Option E: the per-bank Q-landing exponent (the modal-datapath rail fix) ────
-- Every modal weight lands as Q4.28 (×2²⁸) and multiplies an exact Q2.30 rotator
-- in i64, so `w·env·2²⁸·2³⁰ = w·env·2⁵⁸` wraps against i64's 2⁶³ once the LANDED
-- magnitude `sup_d |env₂·A| > 2⁵ = 32` — reachable from the shipped vocabulary (a
-- resonant lowpass has `|A| ≈ Q = 0.55·80^res`, so the rail is a cap on filter Q
-- at 32; the top ~8% of the resonance knob wraps). The cure: land the weight at
-- `2^(28−k)` and shift back by `28−k`, with `k` chosen PER BANK from the
-- coefficient-time bound `maxAbs = maxₘ sup_d |env₂ₘ·Aₘ|`. For a deg-0 mode
-- `env₂ = e^{−σd} ≤ 1`, so the bound is the amplitude `|creₘ|+|cimₘ|`; a deg-`p`
-- mode's `env₂ = d^p·e^{−σd}` peaks at `(p/(σe))^p` (`bankLandExp` folds that
-- factor in — so the fix is sound for deg > 0 too, not only the deg-0 plain
-- vocabulary). The rendered VALUE is k-INVARIANT — the `>>(28−k)` undoes the
-- `·2^(28−k)` exactly and `(28−k)+30−(28−k)=30` leaves the accumulator at Q2.30,
-- so `fixedOutQ 30` is unchanged; only the quantization LSB moves. At `k=0` the
-- emitted ops are `·2²⁸` / `>>28` verbatim — byte-identical.
--
-- TWO caveats, both at the +198 dB extreme (maxAbs ≥ 2³³), practically inert but
-- stated for honesty. (1) THE k=28 CEILING: `k` clamps at 28 (a negative shift is
-- UB), so the per-mode guarantee holds only for `maxAbs < 2³³ = 32·2²⁸`; above it
-- `k` saturates and the product wraps. (2) THE ORTHOGONAL ACCUMULATOR RAIL: the
-- i64 sum `Σ oscQₘ` (each `>>(28−k)` to Q2.30) needs `Σₘ|Aₘ| < 2³³` — orders
-- looser than the per-mode `32·2^k` for any realistic bank, but NOT never: a
-- collected weight reaching 2³³ (a live pole swept within ~6e-11 rad/s of exact
-- coincidence, or absurd authored amps) binds it, and option E does not address
-- this rail. Both extremes are past any musical level; the device clamp (C = 256)
-- bounds their blast radius at the DAC.

/-- `⌊log₂|x|⌋` for a finite nonzero `x` — the IEEE-754 unbiased exponent, the
    exact Lean-Float mirror of the `floatExponent` op the DYNAMIC path emits (so a
    bank that happens to be all-const and one that is live agree on `k`). -/
private def floatExpZ (x : Float) : Int :=
  (((x.abs.toBits >>> 52) &&& 0x7ff).toNat : Int) - 1023

/-- `k = clamp(0, 28, ⌊log₂ maxAbs⌋ − 4)` as a `Nat` (the static path). Solves
    `maxAbs < 32·2^k` for `maxAbs < 2³³` (the k=28 ceiling: above it `k` saturates
    and the guarantee lapses — a +198 dB weight, practically inert). `maxAbs < 32
    ⇒ k = 0` (bit-identical). Non-finite (`+∞` from a non-decaying deg>0 mode) ⇒
    `k = 28` (max headroom); `≤ 0` ⇒ `k = 0` (a silent all-zero bank lands verbatim). -/
private def landK (maxAbs : Float) : Nat :=
  if maxAbs ≤ 0.0 then 0            -- silent/all-zero bank: land verbatim (k=0)
  else
    -- ∞/NaN reach here (¬(x ≤ 0)); floatExpZ reads their exponent field 2047 ⇒
    -- 1024 ⇒ clamp 28 (max headroom, matching the dynamic floatExponent path).
    let e := floatExpZ maxAbs - 4
    if e ≤ 0 then 0 else if e ≥ 28 then 28 else e.toNat

/-- The per-bank Q-landing exponent `k`. STATIC when every amp answers
    `sigConstF?` (⇒ `k` a compile-time `Nat`; `k=0` emits the landing literals
    verbatim, a byte-identical plan and a reused kernel-cache object). DYNAMIC
    otherwise (⇒ `k` an s0 `Sig`, hoisted to the coefficient kernel and crossing
    to the audio kernel as a `coef:` slot). -/
inductive LandExp where
  | static (k : Nat)
  | dynamic (kSig : Sig)

/-- `2^(28−k)` — the Q-landing scale multiplied before `toInt`. `lit 268435456`
    verbatim at `static 0`; `ldexp(1, 28−k)` (an exact power of two) when live. -/
def LandExp.scale : LandExp → Sig
  | .static k  => lit (Int.pow 2 (28 - k))
  | .dynamic k => ldexpE (lit 1) (sub (lit 28) k)

/-- `28−k` — the per-mode right shift (the operand of `rshift`). `lit 28` verbatim
    at `static 0`. Always in `[0,28]` by construction, so the emitted `ashr`/`>>`
    is never the out-of-range poison/UB LLVM and MSL leave undefined. -/
def LandExp.shift : LandExp → Sig
  | .static k  => lit (28 - k)
  | .dynamic k => sub (lit 28) k

/-- The envelope-peak factor `sup_{d≥0} d^p·e^{−σd} = (p/(σe))^p` for a mode of
    degree `p` (the polynomial-order lift): `1` for `p=0` (`e^{−σd} ≤ 1`), else the
    interior peak at `d = p/σ`. A non-decaying polynomial mode (`σ ≤ 0`, `p > 0`)
    has no finite sup — `+∞`, which lands `k=28` (max headroom; it cannot be made
    safe, and no residue calculus produces one). -/
private def envPeakF (deg : Nat) (sigma : Float) : Float :=
  if deg == 0 then 1.0
  else if sigma ≤ 0.0 then (1.0 / 0.0)
  else Float.pow (deg.toFloat / (sigma * 2.718281828459045)) deg.toFloat

/-- The `Sig` mirror of `envPeakF` for the DYNAMIC max fold: `(p/(σe))^p·(|cre|+
    |cim|)`. Reduces to exactly `|cre|+|cim|` at `deg=0`, so a deg-0 bank's `maxSig`
    is unchanged (byte-identical dynamic path). -/
private def modeWeightBoundSig (m : ModalMode) : Sig :=
  let amp := add (.unary .abs m.cre) (.unary .abs m.cim)
  if m.deg == 0 then amp
  else mul (powE (div (litF m.deg.toFloat) (mul m.sigma (litF 2.718281828459045))) m.deg) amp

/-- The per-bank landing exponent from a bank's mode weights. The guarded
    magnitude is `maxAbs = maxₘ sup_d |env₂ₘ·Aₘ| = maxₘ (p/(σe))^p·(|creₘ|+|cimₘ|)`,
    the deg-lifted L1 amp norm — invariant under the `modal-pair` 90° swap
    `(cre,cim)↦(cim,−cre)` (it just swaps the two amp terms; the env factor is
    amp-free), so the paired Re/Im banks and the amp-rotated oracle land at one
    `k`. The DYNAMIC `maxSig` folds the SAME per-mode subterms the columns are
    built from (never a `Sig.index` column read — `Stage0` pins `arrayReg`/
    `loopIdx` `.s1`, which would park the O(count) chain in the AUDIO kernel); its
    leaves are `num`/`paramRef`, so it stages to fold/s0 and rides the coefficient
    kernel. `floatExponent` is `⌊log₂⌋` and fails toward HEADROOM: `floatExponent 0
    = −1023 ⇒ k=0` (silent bank), `floatExponent NaN/∞ = 1024 ⇒ k=28` (max
    headroom), never k=0-toward-the-wrap. Both paths clamp `k∈[0,28]`. -/
def bankLandExp (modes : Array ModalMode) : LandExp := Id.run do
  let mut mx : Float := 0.0
  let mut allConst := true
  for m in modes do
    -- deg-0 (the whole plain vocabulary) never queries σ, so a live-σ deg-0 bank
    -- still takes the static path when its amps const-fold.
    if m.deg == 0 then
      match sigConstF? m.cre, sigConstF? m.cim with
      | some cr, some ci => let v := cr.abs + ci.abs; if v > mx then mx := v
      | _, _ => allConst := false
    else
      match sigConstF? m.cre, sigConstF? m.cim, sigConstF? m.sigma with
      | some cr, some ci, some sg =>
        let v := (cr.abs + ci.abs) * envPeakF m.deg sg
        if v > mx then mx := v
      | _, _, _ => allConst := false
  if allConst then
    return .static (landK mx)
  let maxSig := modes.foldl (fun acc m =>
    let a := modeWeightBoundSig m
    selectE (gt acc a) acc a) (lit 0)
  return .dynamic (clampE (sub (.unary .floatExponent maxSig) (lit 4)) (lit 0) (lit 28))

/-- The RELATIVE clock `clkRel = clk − anchor·2³²` as an EXACT i64 subtract.
    (A float-relative clock — `toFloat(clk)/2³² − anchor` — loses mantissa bits
    as the absolute clock grows, drifting with τ.) Subtracting on
    the integer clock first keeps time-translation exact at any τ; everything
    downstream (phase reduction, the bounded envelope coordinate, the causal gate)
    sees only the bounded relative value. -/
def relClockQ (clkInt anchorSamples : Sig) : Sig :=
  sub clkInt (toIntE (mul anchorSamples (lit 4294967296)))

/-- A mode's oscillator phase over the RELATIVE clock, integer-reduced (the
    FixedPhasor pattern): the frequency lands ONCE as `incr = ⌊(ω/2π)·2³²/SR⌋`
    (float math, so ω may be a live slot), then the exact split-multiply
    `incr·hi + (incr·lo)>>32` reduces `incr·clkRel` on the circle ℤ/2³² — the
    phase argument never leaves `[0, 2π)`, so its precision is τ-INDEPENDENT
    (a raw `ω·dSec` argument is unbounded — its mantissa starves as τ grows).
    ω quantizes to the SR/2³² grid (~1e-5 Hz at 44.1k) — inaudible.
    `rshift` is `ashr`, so `clkRel = (clkRel>>32)·2³² + (clkRel & (2³²−1))` holds
    exactly on NEGATIVE relative clocks (pre-strike, mirrored reads). Returns the
    phase in RADIANS `[0, 2π)` for the `sinSig`/`cosSig` polynomials. -/
def modePhaseQ (omega clkRel : Sig) : Sig :=
  let incr := toIntE (div (mul (div omega twoPiE) (lit 4294967296)) .sampleRate)
  let thi := rshift clkRel (lit 32)
  let tlo := bitAnd clkRel (lit 4294967295)
  bitAnd (add (mul incr thi) (rshift (mul incr tlo) (lit 32))) (lit 4294967295)

/-- `modePhaseQ` scaled to RADIANS `[0, 2π)` — for float-polynomial consumers
    (`sinSig`/`cosSig`). The fixed datapath consumes `modePhaseQ` directly. -/
def modePhaseW (omega clkRel : Sig) : Sig :=
  mul twoPiE (div (toFloatE (modePhaseQ omega clkRel)) (lit 4294967296))

/-- `modePhaseQ` with the quantized increment supplied DIRECTLY, skipping the
    `incr = ⌊(ω/2π)·2³²/SR⌋` step. The banked path stores `incr`'s pre-`toInt`
    float in a coefficient column and reads it per iteration; `toInt` in-loop then
    recovers the SAME i64 (f64 storage round-trips the value exactly, < 2^53), so
    `modePhaseQFromIncr (toIntE incrFloat) clkRel` is bit-identical to
    `modePhaseQ ω clkRel`. -/
def modePhaseQFromIncr (incr clkRel : Sig) : Sig :=
  let thi := rshift clkRel (lit 32)
  let tlo := bitAnd clkRel (lit 4294967295)
  bitAnd (add (mul incr thi) (rshift (mul incr tlo) (lit 32))) (lit 4294967295)

/-- A modal bank as a pure `Sig` over the (already-warped) clock: shift each pole
    to its time-since-strike on the INTEGER clock (`relClockQ`, exact at any τ),
    sum `d^deg·e^{−σd}·(c_re·cos φ − c_im·sin φ)` over the modes with each φ
    integer-reduced (`modePhaseW`), and GATE on `clkRel > 0` (causal: silent
    before the strike, a closed-form tail after — and, read through a reversing
    warp, the tail plays backward). The envelope coordinate `dSec` is float on
    the BOUNDED relative clock (`expSig` clamps; decay makes far-field precision
    irrelevant). Every sample is `f(clk)`: no state, random-access. Rides
    `arrUn … (.clk c)`, so warps reach it through the clock leaf. -/
def modalBankSig (modes : Array ModalMode) (clkInt : Sig) (anchorSamples : Sig) : Sig :=
  let clkRel := relClockQ clkInt anchorSamples
  let dSec := div (div (toFloatE clkRel) (lit 4294967296)) .sampleRate
  -- Q datapath: per mode, the slow scalars (envelope × residue weight) land ONCE
  -- as Q(4+k).(28−k) (`le`, option E — `k=0`/Q4.28 unless the bank's collected
  -- `|A|` crosses the i64 rail of 32; quantum 3.7e-9·2^k ≈ −168 dB at k=0 — a
  -- tail quieter than that truncates to true silence), the oscillator values are
  -- exact Q2.30 off the integer phase, and the mode SUM is i64 — modular, hence
  -- associative and commutative: reordering modes cannot move a bit, which float
  -- summation never gave us. One float scale at the boundary.
  let le := bankLandExp modes
  let bankQ := modes.foldl
    (fun acc m =>
      let phQ := modePhaseQ m.omega clkRel
      let env := expSig (neg (mul m.sigma dSec))
      let env2 := if m.deg == 0 then env else mul (powE dSec m.deg) env
      let wCre := toIntE (mul (mul env2 m.cre) le.scale)
      let wCim := toIntE (mul (mul env2 m.cim) le.scale)
      let oscQ := rshift (sub (mul wCre (fixedCosCycSig phQ))
                              (mul wCim (fixedSinCycSig phQ))) le.shift
      add acc oscQ)
    (litI 0)
  selectE (gt clkRel (lit 0)) (fixedOutQ 30 bankQ) (lit 0)

/-- The SYMBOLIC mode: a bank body's per-mode scalars as column reads at the
    loop index. A body written against `ModeSym` is the same lambda the unrolled
    fold applies to a concrete `ModalMode` — only where the scalars COME FROM
    changes (a column read at `loopIdx` instead of a baked subterm). `incr` is
    the pre-`toInt` float increment `(ω/2π)·2³²/SR` (exact < 2^53); `toInt`
    happens in the body (`modePhaseQFromIncr`). -/
structure ModeSym where
  incr  : Sig
  sigma : Sig
  cre   : Sig
  cim   : Sig

/-- A bank's coefficient columns: one `Sig.arr` per `ModalMode` field, one entry
    per mode — the SAME s0 subterms the unrolled path bakes per mode, destined
    for the stage-0 coefficient kernel. Built once per bank and shared by every
    fold over it (a direction bank runs TWO folds over one set of columns; the
    arena's hash-consing would dedupe rebuilt columns anyway, but sharing at the
    source keeps the intent visible). -/
structure BankCols where
  count : Nat
  /-- Optional LIVE effective count (trip-count-as-data): a `Sig` (typically a
      param-slot read) the reduction trips instead of the static `count`, which
      stays the CAPACITY (columns fill to capacity; the emitters clamp the live
      value to `[0, count]` at the loop head). `none` = the static bank. -/
  live? : Option Sig := none
  /-- The bank's binder id (`Sig.bankSum.idxId` / `Sig.loopIdx.id`). Ids need
      only be unique along a NESTING CHAIN, and `bankFold` banks never nest
      (tables materialize before their region, so two banks over one set of
      columns run sequentially) — 0 is correct for every Sig-level bank today.
      If nesting ever arrives here, thread a chain-unique allocator; the
      emitters fail loudly on an ancestor id collision. -/
  idxId : Nat := 0
  incr  : Sig
  sigma : Sig
  cre   : Sig
  cim   : Sig

def bankCols (modes : Array ModalMode) (live? : Option Sig := none) : BankCols where
  count := modes.size
  live? := live?
  incr  := Sig.arr (modes.map fun m =>
    div (mul (div m.omega twoPiE) (lit 4294967296)) .sampleRate)
  sigma := Sig.arr (modes.map (·.sigma))
  cre   := Sig.arr (modes.map (·.cre))
  cim   := Sig.arr (modes.map (·.cim))

/-- THE generic banked fold: `Σₖ body(mode k)` as one indexed reduction
    (`Sig.bankSum` → a `ReduceBegin` region — O(1) plan instructions in mode
    count). The body is an ordinary function of the symbolic mode; this is the
    only place a banked effect touches `loopIdx`/`bankSum`. Banking is a property
    of the fold, not of the effect: every banked lowering is `bankFold` applied
    to its own body, never a hand-built table twin. The loop visits modes in
    array order — the same order the unrolled `foldl` nests its adds — so for
    the i64 mode sum the render is BIT-IDENTICAL to the unroll. -/
def bankFold (cols : BankCols) (body : ModeSym → Sig) : Sig :=
  let k := Sig.loopIdx cols.idxId
  Sig.bankSum cols.count #[cols.incr, cols.sigma, cols.cre, cols.cim]
    (body { incr  := Sig.index cols.incr k
          , sigma := Sig.index cols.sigma k
          , cre   := Sig.index cols.cre k
          , cim   := Sig.index cols.cim k })
    cols.live?
    cols.idxId

/-- The BANKED lowering of a modal bank (banks-as-data): the SAME value
    as `modalBankSig`, but the mode sum is a `bankFold` indexed reduction over
    the coefficient columns instead of an unrolled fold — so the emitted plan is
    O(1) in mode count, not O(modes). Requires every mode `deg == 0` (the uniform
    datapath, true for resonator/reverb/residue banks); ragged banks route to the
    unrolled path. The body is EXACTLY `modalBankSig`'s op sequence over the
    symbolic mode (deg == 0, so `env2 = env`), and the mode sum is i64-modular
    (associative), so the render is BIT-IDENTICAL — the goldens gate the move.
    Drop-in for `modalBankSig`: same `(modes, clkInt, anchor)` signature, so it
    rides `modalBankTerm`'s `arrUn` (warps reach it through the already-warped
    clock leaf) unchanged. -/
def modalBankSigTable (modes : Array ModalMode) (clkInt anchorSamples : Sig)
    (live? : Option Sig := none) : Sig :=
  let clkRel := relClockQ clkInt anchorSamples
  let dSec := div (div (toFloatE clkRel) (lit 4294967296)) .sampleRate
  let le := bankLandExp modes                                    -- option E: per-bank Q exponent
  let bankQ := bankFold (bankCols modes live?) fun m =>
    let phQ  := modePhaseQFromIncr (toIntE m.incr) clkRel
    let env  := expSig (neg (mul m.sigma dSec))
    let wCre := toIntE (mul (mul env m.cre) le.scale)
    let wCim := toIntE (mul (mul env m.cim) le.scale)
    rshift (sub (mul wCre (fixedCosCycSig phQ))
                (mul wCim (fixedSinCycSig phQ))) le.shift
  selectE (gt clkRel (lit 0)) (fixedOutQ 30 bankQ) (lit 0)

/-- The ANALYTIC bank: the `(Re, Im)` pair of `Σ A·e^{iφ}·env` over ONE column
    set — the substrate for a phase TWIST (heterodyne) and for the
    divided-difference paired body, both of which need `Im` as well as the `Re`
    every existing lowering emits. Two `bankFold`s over one `BankCols` (the
    direction bank's pattern, `modalBankSigDirTable` — two sequential regions,
    `idxId` reuse is safe because they do not nest). The `re` fold is EXACTLY
    `modalBankSigTable`'s body (`wCre·cos − wCim·sin`); the `im` fold is
    `wCre·sin + wCim·cos`. Both gated `clkRel > 0` (causal). Identity the
    `modal-pair` gate leans on: `Im(A·e^{iφ}) = Re(−iA·e^{iφ})`, so the `im`
    component equals the `re` bank of the amp-rotated modes `(cre,cim)↦(cim,−cre)`
    — a bit-identical oracle, no new numerics. deg-0 (uniform) only. -/
def modalBankSigPairTable (modes : Array ModalMode) (clkInt anchorSamples : Sig)
    (live? : Option Sig := none) : Sig × Sig :=
  let clkRel := relClockQ clkInt anchorSamples
  let dSec := div (div (toFloatE clkRel) (lit 4294967296)) .sampleRate
  let cols := bankCols modes live?
  let le := bankLandExp modes                                    -- option E: shared by Re and Im
  let body (imag : Bool) : ModeSym → Sig := fun m =>
    let phQ  := modePhaseQFromIncr (toIntE m.incr) clkRel
    let env  := expSig (neg (mul m.sigma dSec))
    let wCre := toIntE (mul (mul env m.cre) le.scale)
    let wCim := toIntE (mul (mul env m.cim) le.scale)
    if imag then
      rshift (add (mul wCre (fixedSinCycSig phQ))
                  (mul wCim (fixedCosCycSig phQ))) le.shift
    else
      rshift (sub (mul wCre (fixedCosCycSig phQ))
                  (mul wCim (fixedSinCycSig phQ))) le.shift
  let gate := fun q => selectE (gt clkRel (lit 0)) (fixedOutQ 30 q) (lit 0)
  (gate (bankFold cols (body false)), gate (bankFold cols (body true)))

/-- True when a bank is eligible for the table lowering: every mode `deg == 0`
    (the uniform datapath). Ragged banks (mixed degree) must route to the
    unrolled `modalBankSig` (or, later, split into a banked deg-0 part ⊕ an
    unrolled remainder). -/
def bankIsUniform (modes : Array ModalMode) : Bool := modes.all (·.deg == 0)

/-- Complex arithmetic over `Sig` (real, imag) — the residue calculus done
    SYMBOLICALLY, so poles/coeffs can be live param slots. With literal operands
    every op constant-folds to the baked value; with a `paramRef` operand it stays
    a live expression the kernel re-evaluates when the slot moves. Only `+−×÷`. -/
abbrev CplxE := Sig × Sig
def caddE (a b : CplxE) : CplxE := (add a.1 b.1, add a.2 b.2)
def csubE (a b : CplxE) : CplxE := (sub a.1 b.1, sub a.2 b.2)
def cmulE (a b : CplxE) : CplxE := (sub (mul a.1 b.1) (mul a.2 b.2), add (mul a.1 b.2) (mul a.2 b.1))
def cdivE (a b : CplxE) : CplxE :=
  let d := add (mul b.1 b.1) (mul b.2 b.2)
  (div (add (mul a.1 b.1) (mul a.2 b.2)) d, div (sub (mul a.2 b.1) (mul a.1 b.2)) d)
def cnegE (a : CplxE) : CplxE := (neg a.1, neg a.2)

/-- The pole `μ = −σ + iω` and complex amp `A = c_re + i·c_im` of a mode, as `CplxE`. -/
def ModalMode.poleE (m : ModalMode) : CplxE := (neg m.sigma, m.omega)
def ModalMode.ampE (m : ModalMode) : CplxE := (m.cre, m.cim)
def modeOfE (pole amp : CplxE) (deg : Nat := 0) : ModalMode :=
  { sigma := neg pole.1, omega := pole.2, cre := amp.1, cim := amp.2, deg }

/-- The residue calculus, SYMBOLICALLY — `voice ⋙ reverb` with every pole/coeff an
    `Sig`. Same law as the build-time `residueCompose`: forced mode at each voice
    pole λ (amp `a·H(λ)`), ringing mode at each reverb pole ν (amp `−a·r/(λ−ν)`).
    No degeneracy branch — a build-time `|λ−ν|<tol` test is impossible on live
    poles; an exact coincidence is measure-zero for continuous knobs, and a NEAR
    coincidence gives a large finite coupling (the resonance you want). Validated
    against the Float `residueCompose` on literal poles (`symbolic-residue`). -/
def residueComposeE (voice reverb : Array ModalMode) : Array ModalMode :=
  voice.foldl (fun acc v =>
    let lam := v.poleE
    let a := v.ampE
    let Hlam := reverb.foldl (fun s r => caddE s (cdivE r.ampE (csubE lam r.poleE))) ((lit 0, lit 0) : CplxE)
    let acc := acc.push (modeOfE lam (cmulE a Hlam))
    reverb.foldl (fun acc r =>
      acc.push (modeOfE r.poleE (cnegE (cdivE (cmulE a r.ampE) (csubE lam r.poleE))))) acc) #[]

/-- `residueComposeE` COLLECTED to `m + n` modes. The uncollected form pushes one
    ringing mode per (λ, ν) pair — `m·n` modes all sitting at the same `n` reverb
    poles. Partial fractions over distinct simple poles needs only the pole UNION:
    each voice pole λ keeps its forced mode (amp `a·H(λ)`, unchanged), and each
    reverb pole ν keeps ONE ringing mode whose amp is the pair-amps summed over the
    voice, `−r·Σ_k a_k/(λ_k−ν)`. Same signal (the `residue-collected` gate pins it
    pointwise against the uncollected form), a factor `m` fewer transcendentals per
    sample. Amps stay `CplxE` expressions, so live poles (a resonator's `freq`
    slot) stay live through the composition — this is what keeps a voice's pitch
    knob working THROUGH a reverb. Same no-degeneracy-branch stance as
    `residueComposeE` (coincidence is measure-zero on live knobs; near-coincidence
    is a large finite coupling — the sympathetic resonance you want). An empty
    voice composes to the empty bank (silence), preserving the graceful-silence
    contract for a reverb with nothing patched into it. -/
def residueComposeEC (voice reverb : Array ModalMode) : Array ModalMode :=
  if voice.isEmpty then #[] else
  let forced := voice.map (fun v =>
    let Hlam := reverb.foldl (fun s r => caddE s (cdivE r.ampE (csubE v.poleE r.poleE)))
      ((lit 0, lit 0) : CplxE)
    modeOfE v.poleE (cmulE v.ampE Hlam))
  let ringing := reverb.map (fun r =>
    let coupling := voice.foldl (fun s v => caddE s (cdivE v.ampE (csubE v.poleE r.poleE)))
      ((lit 0, lit 0) : CplxE)
    modeOfE r.poleE (cnegE (cmulE r.ampE coupling)))
  forced ++ ringing

/-- The excitation-gauge adapter (§5): rescale every residue by the self-measured
    `1/‖H‖^g`, with `g` a LIVE scalar. `‖H‖` is the p=8 norm of the bank's OWN
    transfer function `H(iω) = Σᵢ Aᵢ/(iω − μᵢ)` sampled at its own pole frequencies
    (each resonance peaks near its pole, so `max_k|H(iωₖ)|` tracks `‖H‖∞`, and the
    smooth 8-norm — a summing fold, no `max` kink under a live sweep — approximates
    it). `g = 0` is the IDENTITY (`‖H‖⁰ = 1`, unity-DC — the committed strike gauge:
    strikes ping at a Q-independent level, `res` is ring time); `g = ½` the √Q trim;
    `g = 1` unity-peak (`H/‖H‖` has unit peak, so a tuned tone is level-invariant
    across a resonance sweep, pings fading as 1/Q). Self-measuring: the norm reads
    pole/residue data ALONE (nothing about `filterPair`/`res`), so it applies
    unchanged to any modal bank or composed segment — Modal ⇝ Modal. The scale is ONE
    real shared across the bank, so the render is EXACTLY `scale · (bare render)`
    (linear in residues) — hence `g = 0` is a VALUE no-op and any `g` is a pure
    re-level with no relower (a mid-ring sweep re-levels the ongoing tail, closed-
    form, no click). An empty bank stays empty.

    **s0 BY CONSTRUCTION** (`gaugeScale`): the norm is measured on the SETTLED poles
    (`settle` collapses a glide to its `#v1` target), so `logSig`'s `floatExponent`
    (f32≠f64 across backends) is coefficient-time and never runs per sample — the
    Metal divergence is unreachable, not documented against. A pole that won't settle
    (a genuine per-sample modulation, an LFO on a cutoff) makes `gaugeScale` `none`,
    and the bank DECLINES (identity) rather than emit an s1 norm. **One remaining note
    (taste-adjacent but a correctness caveat): deg 0.** `H` is the deg-0 form
    `Aᵢ/(iω−μᵢ)`; a deg-`p` mode's `Aᵢ·p!/(iω−μᵢ)^{p+1}` is under-weighted (exact for
    the shipped `filterPair`/`reverbRoom`/`resonatorBank`, all deg-0). -/
def gaugeScale (g : Sig) (modes : Array ModalMode) : Option Sig := do
  -- SETTLE every pole first — the whole scale then reads no clock (s0), so the
  -- `floatExponent` inside `logSig` is coefficient-time. `none` if any won't settle.
  let sm ← modes.mapM fun m => do
    let sg ← settle m.sigma; let om ← settle m.omega
    let cr ← settle m.cre;   let ci ← settle m.cim
    pure ({ m with sigma := sg, omega := om, cre := cr, cim := ci } : ModalMode)
  -- H(iωₖ) = Σᵢ Aᵢ/(σᵢ + i(ωₖ − ωᵢ))   (iωₖ − μᵢ, with μᵢ = −σᵢ + iωᵢ)
  let hAt := fun (wk : Sig) => sm.foldl (fun acc m =>
      caddE acc (cdivE m.ampE ((m.sigma, sub wk m.omega) : CplxE))) ((lit 0, lit 0) : CplxE)
  -- S = Σₖ |H(iωₖ)|⁸ = Σₖ (|H|²)⁴  (p = 8; even power ⇒ no sqrt)
  let S := sm.foldl (fun acc m =>
      let h := hAt m.omega
      let h2 := add (mul h.1 h.1) (mul h.2 h.2)
      add acc (mul (mul h2 h2) (mul h2 h2))) (lit 0)
  -- ‖H‖⁻ᵍ = S^{−g/8} = exp(−(g/8)·ln S). Floor S ∈ [1e−30, 1e30] (via `lit`, NOT
  -- `litF` — `litF 1e−30` rounds to 0, `litF 1e300` saturates to ≈1.8e7): the floor
  -- guards a silent bank (S→0 ⇒ logSig 0), the ceiling a non-finite one (S=∞ ⇒
  -- logSig ∞ = NaN), and 1e30 stays under f32-max so Metal never overflows the clamp.
  pure (expSig (mul (mul (neg g) (lit 125 3))
                    (logSig (clampE S (lit 1 30) (lit (10^30))))))

def normalizePeak (g : Sig) (modes : Array ModalMode) : Array ModalMode :=
  if modes.isEmpty then #[] else
  match gaugeScale g modes with
  | some scale => modes.map fun m => { m with cre := mul m.cre scale, cim := mul m.cim scale }
  | none => modes   -- un-settleable poles: decline to identity, never an s1 norm

/-- `residueComposeEC` with the Cauchy inner sums BANKED (WS-F). Each output mode's
    amp (`Hlam`/`coupling`) becomes a scalar `Sig.bankSum` over the SOURCE columns
    rather than a meta-unrolled fold — same value (per-term identical to `cdivE`,
    left-assoc = the fold's order, so BIT-IDENTICAL to `residueComposeEC`), but the
    coefficient-kernel FILL is O(m+n) reduce regions instead of O(m·n) unrolled ops
    when the poles are live: each inner sum is an all-s0 region that hoists to the
    coeff kernel (`Stage0.tryRegion` — scalar accumulator), and the column writes
    hoist as ordinary array fills. A compile-flatness win, orthogonal to
    `residueComposeDD`'s stability (the `1/Δ` still appears — this is the COLLECTED
    form). Inner binder id 1 (the audio bank is 0; the fills are sequential, emitted
    before the audio region). -/
def residueComposeBanked (voice reverb : Array ModalMode) : Array ModalMode :=
  if voice.isEmpty then #[] else
  let idH : Nat := 1
  -- Σₖ (rRe+i·rIm)/((pRe−nRe[k]) + i(pIm−nIm[k])), real or imag part, as a scalar
  -- bankSum; per term = `cdivE` (r·conj(d)/|d|²), left-associative.
  let cauchy := fun (pRe pIm nRe nIm rRe rIm : Sig) (count : Nat) (imag : Bool) =>
    Sig.bankSum count #[nRe, nIm, rRe, rIm]
      (let k := Sig.loopIdx idH
       let dRe := sub pRe (Sig.index nRe k)
       let dIm := sub pIm (Sig.index nIm k)
       let den := add (mul dRe dRe) (mul dIm dIm)
       let rkRe := Sig.index rRe k
       let rkIm := Sig.index rIm k
       if imag then div (sub (mul rkIm dRe) (mul rkRe dIm)) den
       else div (add (mul rkRe dRe) (mul rkIm dIm)) den)
      none idH
  let revNuRe := Sig.arr (reverb.map (·.poleE.1))
  let revNuIm := Sig.arr (reverb.map (·.poleE.2))
  let revRRe := Sig.arr (reverb.map (·.ampE.1))
  let revRIm := Sig.arr (reverb.map (·.ampE.2))
  let voNuRe := Sig.arr (voice.map (·.poleE.1))
  let voNuIm := Sig.arr (voice.map (·.poleE.2))
  let voARe := Sig.arr (voice.map (·.ampE.1))
  let voAIm := Sig.arr (voice.map (·.ampE.2))
  -- forced (over reverb): Hlam = Σᵣ r/(λ−ν), amp = a·Hlam
  let forced := voice.map (fun v =>
    let hRe := cauchy v.poleE.1 v.poleE.2 revNuRe revNuIm revRRe revRIm reverb.size false
    let hIm := cauchy v.poleE.1 v.poleE.2 revNuRe revNuIm revRRe revRIm reverb.size true
    modeOfE v.poleE (cmulE v.ampE (hRe, hIm)))
  -- ringing (over voice): with pole=ν the bankSum gives Σᵥ a/(ν−λ) = −coupling, so
  -- the EC amp `−r·coupling` becomes `r·(bankSum)` — no `cnegE`.
  let ringing := reverb.map (fun r =>
    let cRe := cauchy r.poleE.1 r.poleE.2 voNuRe voNuIm voARe voAIm voice.size false
    let cIm := cauchy r.poleE.1 r.poleE.2 voNuRe voNuIm voARe voAIm voice.size true
    modeOfE r.poleE (cmulE r.ampE (cRe, cIm)))
  forced ++ ringing

/-- `∫`: the antiderivative of a modal bank, exactly, as a build-time pole move.
    `∫ Σₖ Aₖ e^{μₖ d} dd = Σₖ (Aₖ/μₖ) e^{μₖ d} + C`, and choosing `C = −Σₖ Aₖ/μₖ`
    fixes the integral to 0 at the strike (`d=0`) — a DC atom (`μ=0`, so `e^{0·d}=1`)
    carrying that constant. So each mode's amp divides by its pole (`cdivE`) and one
    `μ=0` mode is appended. Pure `CplxE`, so poles/amps stay live. Stays deg-0
    (bankable). Requires deg-0, NONZERO poles — the modulator case: an LFO is a σ=0
    undamped mode with pole `iω≠0`, and integrating its bank IS the residue transform
    `a ↦ a/μ` behind "FM is PM of the integrated bank" (`demos/modal_vco.py` D3). A DC
    input mode (`μ=0`) would integrate to a `d·e^{0}` deg-1 ramp — out of scope for
    v1, and division by its zero pole is the caller's contract to avoid. -/
def integrateBank (modes : Array ModalMode) : Array ModalMode :=
  let integ := modes.map (fun m => modeOfE m.poleE (cdivE m.ampE m.poleE))
  let sumAmp := integ.foldl (fun s m => caddE s m.ampE) ((lit 0, lit 0) : CplxE)
  integ.push (modeOfE (lit 0, lit 0) (cnegE sumAmp))

/-- `Jₙ(b)` (Bessel, first kind) by trapezoid on the periodic integrand
    `cos(nθ − b·sin θ)` over `[0,π]` — spectral accuracy on the smooth periodic
    integrand. Build-time `Float`; the FM sideband weights of `besselFuse`. The
    index `nf` is the sideband number as a `Float` (so callers avoid `Int→Float`). -/
def besselJ (nf b : Float) (quad : Nat := 256) : Float := Id.run do
  let pi := 3.141592653589793
  let h := pi / quad.toFloat
  let mut acc : Float := 0.0
  for i in [0:quad + 1] do
    let th := h * i.toFloat
    let w := if i == 0 || i == quad then 0.5 else 1.0     -- trapezoid endpoints
    acc := acc + w * Float.cos (nf * th - b * Float.sin th)
  return acc * h / pi

/-- Static-index FM as a build-time pole move (Jacobi–Anger). Modulating a bank by
    `sin(ω_m d)` at index `b` sprouts, from each mode `(μ, A)`, a comb of sidebands
    at pole `μ + i·n·ω_m` with amp `A·Jₙ(b)`, `n ∈ [−N, N]` — so an FM'd voice is
    STILL a modal bank (it keeps poles, so it can feed the residue calculus). `ω_m`
    (rad/s) and `b` are baked in v1 (a change relowers); the carrier's poles/amps
    stay live (each sideband adds a baked offset to `ω` and scales `A` by the real
    `Jₙ(b)`). `N` is the sideband capacity — the tail `|n| > b` decays
    superexponentially, so `N ≈ ⌈b⌉ + few` is exact to machine precision
    (`demos/modal_fm.py` D4). deg-0. -/
def besselFuse (modes : Array ModalMode) (wm b : Float) (N : Nat) : Array ModalMode := Id.run do
  let mut out : Array ModalMode := #[]
  for m in modes do
    for i in [0:2 * N + 1] do
      let nf := i.toFloat - N.toFloat
      let jn := besselJ nf b
      out := out.push
        { sigma := m.sigma
        , omega := add m.omega (litF (nf * wm))
        , cre := mul m.cre (litF jn)
        , cim := mul m.cim (litF jn)
        , deg := m.deg }
  return out

/-- Affine pole reclock: the pole-space image of the affine clock warp `d ↦ a·d+b`.
    `e^{μ(a d + b)} = e^{μb}·e^{(μa)d}`, so the pole scales `μ ↦ μa` (`σ↦σa, ω↦ωa`)
    and the amp rotates `A ↦ A·e^{μb} = A·e^{−σb}(cos ωb + i sin ωb)`. Pure `CplxE`
    over `expSig`/`sinSig`/`cosSig`, so `a`,`b` may be live. AFFINE ONLY: a nonlinear
    warp does not preserve poles (the varispeed case is deferred, per
    `design/modal-island.local.md`); `b` in seconds, `a` dimensionless. deg-0 (the
    `(a d+b)^k` binomial for `deg>0` is out of scope for v1). -/
def reclockAffine (a b : Sig) (modes : Array ModalMode) : Array ModalMode :=
  modes.map fun m =>
    let envB := expSig (neg (mul m.sigma b))
    let eMub : CplxE := (mul envB (cosSig (mul m.omega b)), mul envB (sinSig (mul m.omega b)))
    let a' := cmulE m.ampE eMub
    { sigma := mul m.sigma a, omega := mul m.omega a, cre := a'.1, cim := a'.2, deg := m.deg }

-- ── The DIVIDED-DIFFERENCE paired-mode family (WS-B2) ─────────────────────────
-- Near-degenerate composition without the `1/Δ` cancellation: each (λ, ν) coupling
-- is ONE fused paired atom `Re(c·e^{νd}·d·cexpm1((λ−ν)d))`, `c = a·r` bounded. The
-- τ·e resonance at λ=ν is the smooth series limit of `cexpm1`, no branch. Its own
-- uniform family (all one shape). Fixed-point realization validated in
-- `demos/divdiff_qdatapath.py` (qA: never form `e^{λd}−e^{νd}`; compute
-- `(e^z−1)/z` with the `−1` exact and `|z|≥thr`).

/-- A fused divided-difference paired mode: two poles `λ, ν` (`CplxE`, pole form
    `(−σ, ω)`) and a BOUNDED coeff `c = a·r` (no `1/Δ`). -/
structure PairedMode where
  lam : Sig × Sig
  nu  : Sig × Sig
  c   : Sig × Sig

/-- `voice ⋙ reverb` as fused paired modes — one per (λ, ν), `c = a·r` via `cmulE`
    only (NO `cdivE`, so no `1/Δ` is ever formed at build time; the division is
    deferred to the render's stable `cexpm1`). Inherently `m·n` (collecting would
    reintroduce the `1/Δ` that `residueComposeEC` suffers) — the stability trade for
    the factor-m saving. The near-degenerate form: a live pole sweeping through a
    room pole stays finite and reproduces the resonance (`demos/modal_divdiff.py`). -/
def residueComposeDD (voice reverb : Array ModalMode) : Array PairedMode :=
  voice.flatMap fun v => reverb.map fun r =>
    { lam := v.poleE, nu := r.poleE, c := cmulE v.ampE r.ampE }

-- ── The EC/DD PARTITIONED compose (fork 3′ erasure, Phase 1) ──────────────────
-- The compiler owns the EC-vs-DD choice PER COUPLING: a (λ, ν) pair routes to
-- the fused paired atom when the collected `±c/Δ` representation would degrade,
-- and stays collected otherwise. Everything decides at compile time on
-- const-folded poles/amps; anything unmeasurable (live pole, live amp) stays
-- collected in v1 — Phase 2 classifies live poles over their declared interval.

/-- θ_acc — the accuracy lens (rad/s), frozen from `demos/ecdd_partition.py`
    (2026-07-23). The measured mechanism is the FREQUENCY GRID, not float64
    cancellation: rotator increments quantize ω at `2π·SR/2³² ≈ 6.5e-5 rad/s`,
    so the collected form renders a grid-quantized Δ — below one quantum the two
    increments coincide and the `±c/Δ` residues cancel to exact silence, and the
    mis-rendered beat keeps it decades over the gate floor up to ~0.05 rad/s.
    The paired body's series branch carries RAW Δ and holds ~2.4e-6 throughout.
    Frozen at the advantage crossing 4.6e-2 × 10 (generous toward DD — DD is
    accurate everywhere, so θ is a COST boundary, not a correctness boundary). -/
def ecddThetaAcc : Float := 0.4642

/-- The range lens: route when the collected ringing weight `|a·r|/|Δ|` exceeds
    the Q4.28 magnitude ceiling 8 with a 2× margin. Amp-dependent — `|Δ|` alone
    is not the criterion (cockpit D_p2: binding for `|a·r| ≳ 1.9`). -/
def ecddRailCeil : Float := 4.0

/-- The paired atom's own build-time range cap: per pair, `|Wc| ≤ |c|·sup_d
    |e^{νd}·d·cexpm1(Δd)| ≤ |c|·min(2/|Δ|, 1/(e·σ_min))` (the sup the
    remainder-handoff deferred until the DD wiring landed — this is that
    landing). A coupling whose bound exceeds the cap is NOT routed (stays
    collected — reject, the status-quo floor; the scale arm can come later if a
    real patch ever trips this). Cap 8 = the plain Q4.28 ceiling, 4× under the
    DD site's i64 rail of 32. -/
def ecddPairCap : Float := 8.0

/-- The σ INTERVAL a mode's damping ranges over: a const σ is its own point
    interval; a LIVE σ with a declared `sigmaRange` classifies over the knob
    span (WS-LP's build-time-over-the-declared-interval discipline); a live σ
    without a range is unclassifiable (`none` ⇒ the coupling stays collected). -/
def sigmaInterval? (m : ModalMode) : Option (Float × Float) :=
  match sigConstF? m.sigma with
  | some s => some (s, s)
  | none => m.sigmaRange

/-- A mode's pole with a LIVE σ CLAMPED to its declared interval (coefficient-
    time, the WS-LP kernel-clamp precedent) — what makes the paired route's
    build-time range cap sound by construction even if the host drives the knob
    out of its declared span. A const σ passes through untouched. -/
def clampedPoleE (m : ModalMode) : CplxE :=
  match sigConstF? m.sigma, m.sigmaRange with
  | none, some (lo, hi) => (neg (clampE m.sigma (litF lo) (litF hi)), m.omega)
  | _, _ => m.poleE

/-- The per-coupling routing predicate — compile-time only. `true` ⇒ the (v, r)
    coupling leaves the collected sums and becomes one `PairedMode`. σ may be
    LIVE with a declared range (Phase 2): the lenses evaluate at min |Δ| over
    the interval — a coupling whose interval DIPS under θ takes DD throughout
    the knob span, so no runtime select ever exists (D2; the WS-LP phase-3a
    pattern). ω and amps must const-fold, both modes deg-0; anything else stays
    collected. Fires on either lens (accuracy | range), gated by the paired
    range cap at the interval's worst point. -/
def couplingHot (v r : ModalMode) : Bool :=
  v.deg == 0 && r.deg == 0 &&
  (Id.run do
    let some (svLo, svHi) := sigmaInterval? v | return false
    let some (srLo, srHi) := sigmaInterval? r | return false
    let some wv := sigConstF? v.omega | return false
    let some wr := sigConstF? r.omega | return false
    let some ar := sigConstF? v.cre | return false
    let some ai := sigConstF? v.cim | return false
    let some rr := sigConstF? r.cre | return false
    let some ri := sigConstF? r.cim | return false
    -- min |Δ| over the σ interval(s): the span distance (0 when they overlap —
    -- the dipper), ω exact. Const σ degenerates to the point distance.
    let sepLo := if svLo < srLo then srLo else svLo
    let sepHi := if svHi < srHi then svHi else srHi
    let dSig := if sepLo > sepHi then sepLo - sepHi else 0.0
    let dAbs := Float.sqrt (dSig * dSig + (wv - wr) * (wv - wr))
    let cAbs := Float.sqrt (ar * ar + ai * ai) * Float.sqrt (rr * rr + ri * ri)
    let hot := dAbs < ecddThetaAcc || (dAbs > 0.0 && cAbs / dAbs > ecddRailCeil)
    -- the paired range cap: sup of the divided-difference kernel over the
    -- WHOLE interval — σ_min at the spans' low edge, |Δ| at its minimum
    -- (sound: the paired pole clamps to the interval, `clampedPoleE`)
    let smin := if svLo < srLo then svLo else srLo
    let sup1 := if dAbs > 0.0 then 2.0 / dAbs else 1e308
    let sup2 := if smin > 0.0 then 1.0 / (2.718281828459045 * smin) else 1e308
    let sup := if sup1 < sup2 then sup1 else sup2
    return hot && cAbs * sup < ecddPairCap)

/-- The PARTITIONED compose — the one `residueCompose` seam. Cold couplings take
    `residueComposeEC`'s collected shapes; hot couplings (per `couplingHot`)
    leave the Cauchy sums (BOTH halves — the forced amp's `r/(λ−ν)` term and the
    ringing amp's `a/(λ−ν)` term migrate together; the split is EXACT residue
    algebra, not an approximation) and land as fused `PairedMode` atoms.
    When NOTHING is hot the result is `residueComposeEC` VERBATIM (the same
    call), so the cold path is byte-identical to the pre-partition compiler —
    the erasure gate's discipline. -/
def residueComposePartitioned (voice reverb : Array ModalMode) :
    Array ModalMode × Array PairedMode := Id.run do
  if voice.isEmpty then return (#[], #[])
  let hotM := voice.map fun v => reverb.map fun r => couplingHot v r
  if hotM.all (·.all (!·)) then return (residueComposeEC voice reverb, #[])
  let isHot := fun (i q : Nat) => (hotM[i]!)[q]!
  let forced := voice.mapIdx fun i v =>
    let Hlam := reverb.zipIdx.foldl (init := ((lit 0, lit 0) : CplxE)) fun s (r, q) =>
      if isHot i q then s
      else caddE s (cdivE r.ampE (csubE v.poleE r.poleE))
    modeOfE v.poleE (cmulE v.ampE Hlam)
  let ringing := reverb.mapIdx fun q r =>
    let coupling := voice.zipIdx.foldl (init := ((lit 0, lit 0) : CplxE)) fun s (v, i) =>
      if isHot i q then s
      else caddE s (cdivE v.ampE (csubE v.poleE r.poleE))
    modeOfE r.poleE (cnegE (cmulE r.ampE coupling))
  let mut paired : Array PairedMode := #[]
  for (v, i) in voice.zipIdx do
    for (r, q) in reverb.zipIdx do
      if isHot i q then
        -- live-σ poles enter the paired atom CLAMPED to their declared
        -- interval, keeping the routing's range cap sound (Phase 2)
        paired := paired.push
          { lam := clampedPoleE v, nu := clampedPoleE r, c := cmulE v.ampE r.ampE }
  return (forced ++ ringing, paired)

/-- Multiply a `CplxE` by a real `Sig`. -/
def scaleRealE (s : Sig) (z : CplxE) : CplxE := (mul s z.1, mul s z.2)

/-- `(e^z − 1)/z` for a complex `z` as the Horner series `Σ_{k≥0} zᵏ/(k+1)!`, N=6
    terms — the STABLE branch for small `|z|`, where the `−1` in the direct form
    catastrophically cancels. Coeffs `1/(k+1)!`. Limit 1 at z=0 (the τ·e resonance).
    Validated: series↔direct discontinuity at the threshold is 2.5e-12. -/
def cexpm1SeriesE (z : CplxE) : CplxE :=
  let step := fun (ck : Float) (acc : CplxE) => caddE (litF ck, lit 0) (cmulE z acc)
  step 1.0 (step (1.0/2.0) (step (1.0/6.0) (step (1.0/24.0)
    (step (1.0/120.0) (step (1.0/720.0) (litF (1.0/5040.0), lit 0))))))

/-- The paired bank's coefficient columns — 7 (vs the single-pole `BankCols`'s 4):
    the ν rotator increment, the SIGNED difference (`ω_λ−ω_ν`) rotator increment,
    `σ_ν`, `σ_λ−σ_ν`, `ω_λ−ω_ν`, and the complex coeff. `poleE = (−σ, ω)`, so
    `σ_ν = −ν.1`, `σ_λ−σ_ν = ν.1−λ.1`, `ω_d = λ.2−ν.2`. -/
structure PairedBankCols where
  count : Nat
  live? : Option Sig := none
  idxId : Nat := 0
  incrNu   : Sig
  incrDiff : Sig
  sigmaNu  : Sig
  ds       : Sig
  wd       : Sig
  cre      : Sig
  cim      : Sig

structure PairedModeSym where
  incrNu   : Sig
  incrDiff : Sig
  sigmaNu  : Sig
  ds       : Sig
  wd       : Sig
  cre      : Sig
  cim      : Sig

def pairedBankCols (modes : Array PairedMode) (live? : Option Sig := none) : PairedBankCols where
  count := modes.size
  live? := live?
  incrNu   := Sig.arr (modes.map fun m => div (mul (div m.nu.2 twoPiE) (lit 4294967296)) .sampleRate)
  incrDiff := Sig.arr (modes.map fun m =>
    div (mul (div (sub m.lam.2 m.nu.2) twoPiE) (lit 4294967296)) .sampleRate)
  sigmaNu  := Sig.arr (modes.map fun m => neg m.nu.1)
  ds       := Sig.arr (modes.map fun m => sub m.nu.1 m.lam.1)
  wd       := Sig.arr (modes.map fun m => sub m.lam.2 m.nu.2)
  cre      := Sig.arr (modes.map fun m => m.c.1)
  cim      := Sig.arr (modes.map fun m => m.c.2)

def bankFoldPaired (cols : PairedBankCols) (body : PairedModeSym → Sig) : Sig :=
  let k := Sig.loopIdx cols.idxId
  Sig.bankSum cols.count
    #[cols.incrNu, cols.incrDiff, cols.sigmaNu, cols.ds, cols.wd, cols.cre, cols.cim]
    (body { incrNu := Sig.index cols.incrNu k, incrDiff := Sig.index cols.incrDiff k
          , sigmaNu := Sig.index cols.sigmaNu k, ds := Sig.index cols.ds k
          , wd := Sig.index cols.wd k, cre := Sig.index cols.cre k, cim := Sig.index cols.cim k })
    cols.live? cols.idxId

/-- The divided-difference paired-mode bank body (qA). Per mode: the ν rotator (exact
    integer phase) plus a SECOND integer-phase rotator at the signed difference
    frequency `ω_λ−ω_ν` for `e^z`; `cexpm1(z)` by a per-sample `selectE` between the
    direct `(e^z−1)/z` (`|z|²≥thr²=0.01`, guarded so the unused branch never divides
    by ~0) and the Horner series (small `|z|`); the complex weight
    `Wc = c·(e^{νd}·d·cexpm1)` lands in Q4.28 and combines with the ν oscillator in
    i64 — `modalBankSigTable`'s skeleton exactly, one float boundary scale.

    RANGE (WS-AA range lens). **Rail**: the same i64 landing as the plain sites —
    `|Wc|·2²⁸·2³⁰ < 2⁶³` ⇒ **per mode `|Wc| < 32`**. This is a FACTOR site: `Wc`
    carries the per-sample `d·cexpm1(Δd)` secular, which is UNBOUNDED by a
    coefficient-time `max|A|` (it needs the bake-time sup `min(2/|Δ|, 1/(e·σ_min))`),
    so the plain `bankLandExp` does NOT reach it. **Reachable max**: bounded at the
    ROUTING site, not here — the EC/DD partition (`couplingHot`) admits a coupling
    to this body only when the build-time sup `|c|·min(2/|Δ|, 1/(e·σ_min)) <
    ecddPairCap = 8` clears (4× under this rail; a coupling over the cap stays
    collected — reject, per the remainder-handoff's deferred item, now landed).
    Direct callers outside the partition (fixtures, the seam sweep) own their own
    amp discipline. `envDf`
    and `e^z` stay float (never landed); `z`'s imaginary part uses raw `ω_d·dSec`
    (the divisor needs only relative precision; the rotator phase is integer-reduced,
    consistent because `e^z` is periodic). -/
def modalBankSigTableDD (modes : Array PairedMode) (clkInt anchorSamples : Sig)
    (live? : Option Sig := none) : Sig :=
  let clkRel := relClockQ clkInt anchorSamples
  let dSec := div (div (toFloatE clkRel) (lit 4294967296)) .sampleRate
  let bankQ := bankFoldPaired (pairedBankCols modes live?) fun m =>
    let phQnu := modePhaseQFromIncr (toIntE m.incrNu) clkRel
    let phQdf := modePhaseQFromIncr (toIntE m.incrDiff) clkRel
    let cosDf := div (toFloatE (fixedCosCycSig phQdf)) (lit 1073741824)   -- Q2.30 → float
    let sinDf := div (toFloatE (fixedSinCycSig phQdf)) (lit 1073741824)
    let envNu := expSig (neg (mul m.sigmaNu dSec))
    let envDf := expSig (neg (mul m.ds dSec))
    let ez : CplxE := (mul envDf cosDf, mul envDf sinDf)                  -- e^z (float)
    let z : CplxE := (neg (mul m.ds dSec), mul m.wd dSec)                 -- z = (λ−ν)d
    let zsq := add (mul z.1 z.1) (mul z.2 z.2)
    let big := gt zsq (litF 0.01)                                        -- |z|² ≥ thr² (thr=0.1)
    let zsafe : CplxE := (selectE big z.1 (lit 1), selectE big z.2 (lit 0))
    let direct := cdivE (csubE ez (lit 1, lit 0)) zsafe                  -- (e^z−1)/z
    let series := cexpm1SeriesE z
    let cx : CplxE := (selectE big direct.1 series.1, selectE big direct.2 series.2)
    let wc := cmulE (m.cre, m.cim) (scaleRealE (mul envNu dSec) cx)      -- c·(e^{νd}·d·cexpm1)
    let wCre := toIntE (mul wc.1 (lit 268435456))
    let wCim := toIntE (mul wc.2 (lit 268435456))
    rshift (sub (mul wCre (fixedCosCycSig phQnu)) (mul wCim (fixedSinCycSig phQnu))) (lit 28)
  selectE (gt clkRel (lit 0)) (fixedOutQ 30 bankQ) (lit 0)


-- ── The bloom⋙reverb Γ-BRIDGE atom (WS-B3) ────────────────────────────────────
-- A pitch-bloomed mode feeding a reverb pole. The bloomed mode is the voice mode
-- on the bloom-warped clock, `Re(A·e^{μφ(d)})`, `φ(d) = d + B(1−e^{−gd})`;
-- composing with a reverb pole ν is the convolution `∫₀^d e^{ν(d−s)}e^{μφ(s)}ds`.
-- The Poisson-lattice expansion of `e^{−κe^{−gd}}` (κ = μB) is representation-
-- DEAD: its coefficients `(−κ)ⁿ/n!` are Taylor coefficients (unique — no
-- re-expansion escapes) cancelling from magnitude ~e^{|κ|}, and the shipped gong
-- runs |κ| = |μ|·B ≈ 19–178. Decay-spaced sidebands at one frequency are nearly
-- parallel atoms (Prony ill-conditioning); contrast `besselFuse`, whose
-- frequency-spaced sidebands Parseval bounds forever. The composition instead
-- has a closed form one RING EXTENSION up — the incomplete gamma `γ(a, κe^{−gd})`,
-- `a = (ν−μ)/g` — and after cancelling every power/log prefactor against the
-- carriers it is a branch-cut-free TWO-CARRIER atom (cockpit
-- `demos/modal_bloom_gamma.py`, ALL PASS: 3.7e-13 worst vs the independent
-- time-domain oracle, ≥11.8 digits over the (|a|,|κ|) box):
--
--   y(d) = Re[ c · ( K1·e^{νd} + K2(z(d))·e^{μφ(d)} ) ],   z(d) = κe^{−gd}
--   series side (|a+1| ≥ |z|):  K1 = C,       K2 = −M(1,a+1,z)/(ν−μ)
--   CF side     (|z| > |a+1|):  K1 = C − Γ★,  K2 = (Γ(a,z)·eᶻ·z^{−a})/g
--
-- C is the κ-side constant by the same rule at z = κ; the branches are bridged
-- EXACTLY by the d-constant Γ★ = Γ(a)·κ^{−a}·e^{κ}/g — the Γ(a) term the two
-- forms share (identity: `z^{−a}e^{z}e^{μφ(d)} = κ^{−a}e^{κ}e^{νd}`); its
-- e^{±π|Im a|/2} blowups cancel in the EXPONENT, so it is computed as
-- `exp(lgamma(a) − a·log κ + κ)/g` and is moderate. `z(d)` decays, so a CF pair
-- re-enters the series side at the baked `d_switch = ln(|κ|/|a+1|)/g` — a
-- per-sample `selectE`, the `cexpm1` discipline one floor up. κ→0 collapses to
-- the WS-B2 divided-difference atom: this is its κ-extension.

/-- Build-time complex `Float` — the `besselJ` tier: production BAKE-time
    numerics for the Γ-bridge constants, never per-sample. Local and minimal
    (no Mathlib). -/
structure CplxB where
  re : Float
  im : Float
deriving Inhabited, Repr

namespace CplxB
def add (a b : CplxB) : CplxB := ⟨a.re + b.re, a.im + b.im⟩
def sub (a b : CplxB) : CplxB := ⟨a.re - b.re, a.im - b.im⟩
def mul (a b : CplxB) : CplxB := ⟨a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re⟩
def neg (a : CplxB) : CplxB := ⟨-a.re, -a.im⟩
def scale (s : Float) (a : CplxB) : CplxB := ⟨s * a.re, s * a.im⟩
def normSq (a : CplxB) : Float := a.re * a.re + a.im * a.im
def abs (a : CplxB) : Float := Float.sqrt a.normSq
def div (a b : CplxB) : CplxB :=
  let d := b.normSq
  ⟨(a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d⟩
def exp (z : CplxB) : CplxB :=
  let e := Float.exp z.re
  ⟨e * Float.cos z.im, e * Float.sin z.im⟩
def log (z : CplxB) : CplxB := ⟨0.5 * Float.log z.normSq, Float.atan2 z.im z.re⟩
end CplxB

/-- A build-time complex constant as a `CplxE` literal pair. -/
def cplxLitE (x : CplxB) : CplxE := (litF x.re, litF x.im)

/-- The `CplxE` unit — the Kummer Horner's seed and the CF's numerator. -/
def cOneE : CplxE := (lit 1, lit 0)

/-- The fixed-depth Kummer `M(1, a+1, z)` Horner over the emitted reciprocals
    `invA[k] = 1/(a+k+1)` — THE series lane's expression, shared verbatim by the
    per-sample lane in `bloomComposedSig` (z live) and the live-pole lift's
    `mK = M(1,a+1,κ)` constant in `bloomCompose` (z = κ, a coefficient — s0), so
    the constant is computed by the SAME arithmetic the lane renders with. -/
def bloomM1E (invA : Array CplxE) (z : CplxE) : CplxE :=
  invA.foldr (fun ik h => caddE cOneE (cmulE (cmulE z ik) h)) cOneE

/-- The fixed-depth bottom-up continued fraction `CF(z) = Γ(a,z)eᶻz^{−a}` over
    the emitted constants `cfB` (b-terms, `z` added per level) and `cfN`
    (numerators) — THE CF lane's expression, shared verbatim by the per-sample
    lane in `bloomComposedSig` (z live) and the live-pole lift's `CF(κ)`
    constant in `bloomCompose` (z = κ, s0), so the constant is computed by the
    SAME arithmetic the lane renders with (WS-LP phase 2, `bloomM1E`'s twin).
    Requires `cfB.size = cfN.size + 1`. -/
def bloomCFE (cfB cfN : Array CplxE) (z : CplxE) : CplxE := Id.run do
  let kk := cfN.size
  let mut h : CplxE := caddE z cfB[kk]!
  for jr in [0:kk] do
    let j := kk - 1 - jr
    h := csubE (caddE z cfB[j]!) (cdivE cfN[j]! h)
  return cdivE cOneE h

/-- Is this `Sig` a pure s0 value — a function of knob slots and constants only
    (no `τ`/tick, no input, no bank machinery)? The live-pole lift's admission
    check: the lifted pair constants are correct (and Stage0-hoistable) only if
    the pole read is knob-invariant per sample. A GLIDED pole (`.sampleIndex`
    inside `glideExpr`) fails here — `settle` it first (the recorded WS-LP
    discipline; reverb rt60 is raw ⇒ s0, filter cutoff/resonance are glided).
    UNMEMOIZED structural walk (like `sigConstF?`): call it only on SHALLOW
    authored expressions (a pole read), never on combinator-built values whose
    subterms share by reference — their TREE is exponential in the DAG depth
    (a Horner like `bloomM1E`'s output would never return). -/
partial def sigIsS0 : Sig → Bool
  | .num _ | .paramRef _ | .sampleRate => true
  | .unary _ a => sigIsS0 a
  | .binary _ a b => sigIsS0 a && sigIsS0 b
  | .clamp a b c | .select a b c => sigIsS0 a && sigIsS0 b && sigIsS0 c
  | _ => false

/-- Complex log-gamma: Lanczos (g=7, n=9) on `Re z ≥ ½`, reflection below with
    `log sin(πz)` taken on the DOMINANT exponential (`s + log(1−e^{−2s})`, so
    large |Im z| never overflows). Build-time only (the Γ★ bridge). Gated at
    1.8e-15 against mpmath over the shipped a-range (cockpit D_bg5). -/
def lgammaB (z : CplxB) : CplxB :=
  let core : CplxB → CplxB := fun z =>
    let lanczos : Array Float := #[0.99999999999980993, 676.5203681218851,
      -1259.1392167224028, 771.32342877765313, -176.61502916214059,
      12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
      1.5056327351493116e-7]
    let zz := z.sub ⟨1, 0⟩
    let x := (Array.range 8).foldl
      (fun acc i => acc.add (CplxB.div ⟨lanczos[i+1]!, 0⟩ (zz.add ⟨(i+1).toFloat, 0⟩)))
      ⟨lanczos[0]!, 0⟩
    let t := zz.add ⟨7.5, 0⟩
    (((zz.add ⟨0.5, 0⟩).mul (CplxB.log t)).sub t).add
      ((CplxB.log x).add ⟨0.5 * Float.log (2.0 * 3.141592653589793), 0⟩)
  if z.re < 0.5 then
    let pi := 3.141592653589793
    -- s = ∓iπz picked so e^{s} is the dominant half of sin πz
    let s : CplxB := if z.im < 0 then ⟨-pi * z.im, pi * z.re⟩ else ⟨pi * z.im, -pi * z.re⟩
    let log2i : CplxB := if z.im < 0 then ⟨Float.log 2.0, pi / 2.0⟩
                         else ⟨Float.log 2.0, -pi / 2.0⟩
    let logsin := (s.add (CplxB.log (CplxB.sub ⟨1, 0⟩ (CplxB.exp (s.scale (-2.0)))))).sub log2i
    (CplxB.sub ⟨Float.log pi, 0⟩ logsin).sub (core (CplxB.sub ⟨1, 0⟩ z))
  else core z

-- ── The EMITTED complex transcendentals (WS-LP: the live Γ★ bridge) ────────────
-- The bloom crossing's bake-time constants are lifted from build-time `CplxB` to
-- emitted `CplxE` (Sig × Sig) so a LIVE pole survives the crossing. The one new
-- op under this is `atan2E` (Numerics); everything else is mechanical CplxE.

/-- Complex log at Sig level: `⟨½·log|z|², atan2(Im, Re)⟩` — `logSig` supplies the
    modulus half, `atan2E` the phase. The live twin of `CplxB.log`. -/
def clogE (z : CplxE) : CplxE :=
  (mul (lit 5 1) (logSig (add (mul z.1 z.1) (mul z.2 z.2))), atan2E z.2 z.1)

/-- Complex exp at Sig level: `e^{Re}·⟨cos Im, sin Im⟩`. The live twin of `CplxB.exp`. -/
def cexpE (z : CplxE) : CplxE :=
  let e := expSig z.1
  (mul e (cosSig z.2), mul e (sinSig z.2))

/-- Complex log-gamma at Sig level — the EMITTED twin of `lgammaB`. Same Lanczos
    (g=7, n=9) core, same reflection for `Re z < ½` on the dominant half of
    `log sin πz`; the build-time `if z.re<½` / `if z.im<0` branches become `selectE`s
    (the unselected core lane may go non-finite off its region — the select discards
    it, the bloom's established discipline). Both `clogE`/`cexpE` reachable. -/
def lgammaE (z : CplxE) : CplxE :=
  let lanczos : Array Float := #[0.99999999999980993, 676.5203681218851,
    -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
    1.5056327351493116e-7]
  let core : CplxE → CplxE := fun z =>
    let zz := csubE z (lit 1, lit 0)
    let x := (Array.range 8).foldl
      (fun acc i => caddE acc (cdivE ((litF lanczos[i+1]!, lit 0) : CplxE)
        (caddE zz ((litF (i+1).toFloat, lit 0) : CplxE))))
      ((litF lanczos[0]!, lit 0) : CplxE)
    let t := caddE zz ((litF 7.5, lit 0) : CplxE)
    caddE (csubE (cmulE (caddE zz ((lit 5 1, lit 0) : CplxE)) (clogE t)) t)
          (caddE (clogE x) ((litF (0.5 * Float.log (2.0 * 3.141592653589793)), lit 0) : CplxE))
  let pi := lit 3141592653589793 15
  let imNeg := gt (lit 0) z.2                                    -- Im z < 0
  let s : CplxE := (selectE imNeg (neg (mul pi z.2)) (mul pi z.2),
                    selectE imNeg (mul pi z.1) (neg (mul pi z.1)))
  let log2i : CplxE := ((litF (Float.log 2.0) : Sig),
                        selectE imNeg (lit 15707963267948966 16) (neg (lit 15707963267948966 16)))
  let logsin := csubE (caddE s (clogE (csubE ((lit 1, lit 0) : CplxE)
                  (cexpE (scaleRealE (lit (-2)) s))))) log2i
  let reflected := csubE (csubE ((litF (Float.log 3.141592653589793), lit 0) : CplxE) logsin)
                         (core (csubE ((lit 1, lit 0) : CplxE) z))
  let base := core z
  let useRefl := gt (lit 5 1) z.1                                -- Re z < ½
  (selectE useRefl reflected.1 base.1, selectE useRefl reflected.2 base.2)

/-- Γ★ = `exp(lgamma(a) − a·log κ + κ)/g` at Sig level — the live twin of
    `bloomGammaStar`, the bloom crossing's d-constant bridge between the two envelope
    branches (its `e^{±π|Im a|/2}` blowups cancel in the exponent). -/
def bloomGammaStarE (a kappa : CplxE) (g : Sig) : CplxE :=
  scaleRealE (div (lit 1) g)
    (cexpE (caddE (csubE (lgammaE a) (cmulE a (clogE kappa))) kappa))

/-- `M(1, a+1, z) = 1 + z/(a+1)(1 + z/(a+2)(…))` by forward recurrence —
    `(value, terms)`. Build-time (the κ-side constant + per-pair depth sizing);
    the per-sample twin is the fixed-depth Horner in `bloomComposedSig`. Stable
    for `|a+1| ≳ |z|` (imaginary-dominated a: terms bounded by `(|z|/|a|)ⁿ`). -/
def bloomM1 (a z : CplxB) (tol : Float := 1e-17) (cap : Nat := 4000) : CplxB × Nat := Id.run do
  let mut s : CplxB := ⟨1, 0⟩
  let mut t : CplxB := ⟨1, 0⟩
  for n in [1:cap] do
    t := (t.mul z).div (a.add ⟨n.toFloat, 0⟩)
    s := s.add t
    if t.abs ≤ tol * (max s.abs 1.0) then return (s, n)
  return (s, cap)

/-- `CF(z) = Γ(a,z)·eᶻ·z^{−a}` by modified Lentz on the standard continued
    fraction `1/(z+1−a− 1(1−a)/(z+3−a− 2(2−a)/(…)))` — `(value, depth)`.
    Build-time (the κ-side constant + per-pair depth sizing); the per-sample
    twin is the fixed-depth bottom-up fraction in `bloomComposedSig`. Stable
    for `|z| ≳ |a|` (the sweep-crossing region). -/
def bloomCF (a z : CplxB) (tol : Float := 1e-15) (cap : Nat := 4000) : CplxB × Nat := Id.run do
  let tiny := 1e-300
  let mut b : CplxB := (z.add ⟨1, 0⟩).sub a
  let mut c : CplxB := ⟨1.0 / tiny, 0⟩
  let mut d : CplxB := if b.normSq == 0.0 then ⟨tiny, 0⟩ else CplxB.div ⟨1, 0⟩ b
  let mut h : CplxB := d
  for i in [1:cap] do
    let an : CplxB := (⟨i.toFloat, 0⟩ : CplxB).mul (a.sub ⟨i.toFloat, 0⟩)  -- −i(i−a)
    b := b.add ⟨2, 0⟩
    d := (an.mul d).add b
    if d.abs < tiny then d := ⟨tiny, 0⟩
    c := b.add (an.div c)
    if c.abs < tiny then c := ⟨tiny, 0⟩
    d := CplxB.div ⟨1, 0⟩ d
    let delta := d.mul c
    h := h.mul delta
    if (delta.sub ⟨1, 0⟩).abs ≤ tol then return (h, i)
  return (h, cap)

/-- `Γ★ = Γ(a)·κ^{−a}·e^{κ}/g` — the d-constant bridge between the two envelope
    branches, computed in the EXPONENT (`exp(lgamma(a) − a·log κ + κ)/g`) where
    the `e^{±π|Im a|/2}` blowups of `Γ(a)` and `κ^{−a}` cancel, so the value is
    moderate. Build-time only. -/
def bloomGammaStar (a kappa : CplxB) (g : Float) : CplxB :=
  (CplxB.exp (((lgammaB a).sub (a.mul (CplxB.log kappa))).add kappa)).scale (1.0 / g)

-- ── The COINCIDENCE (`|a| < ½`) divided-difference constants (WS-A4, atom four) ──
-- The τ·e resonance — a room pole ON the settled partial (ν → μ, `a → 0`) — is a
-- REMOVABLE singularity: at `a = 0` the E1 numerator vanishes identically (0/0), so
-- the meaning is total there, it is only E1's `1/(ν−μ)` and the Γ★ bridge that
-- overflow. The fix is the a-divided-difference of the numerator, evaluated in the
-- SAME two-region split as the crossing: the CF branch (`bloomCF`) is already
-- coincidence-stable for large `z` (the Γ(a) pole cancels between `CF(z)` and
-- `CF(κ)` by the Γ★ identity), and for small `z` it degrades into the log region, so
-- there the series divided difference `Φ(a,z) = Σ dₙ zⁿ = (M_a(z)−eᶻ)/a` takes over
-- plus the τ·e secular `e^κ·(e^{νd}−e^{μd})/(ν−μ)` — a `residueComposeDD`-shaped
-- paired atom. Validated bit-clean over the whole (a,κ) box incl. a=0 exact and
-- lightly-damped long tails (`demos/modal_bloom_gamma.py`, `d_bg6`).

/-- Build-time complex `(eᶻ−1)/z` (the divided difference of `exp` at 0, limit 1),
    the stable branch for small `|z|` via the 7-term series `Σ zᵏ/(k+1)!`, direct
    otherwise. The `CplxB` twin of `cexpm1SeriesE`. -/
def cexpm1B (z : CplxB) : CplxB :=
  if z.normSq < 0.01 then
    let step := fun (ck : Float) (acc : CplxB) => (⟨ck, 0⟩ : CplxB).add (z.mul acc)
    step 1.0 (step (1.0/2.0) (step (1.0/6.0) (step (1.0/24.0)
      (step (1.0/120.0) (step (1.0/720.0) ⟨1.0/5040.0, 0⟩)))))
  else (CplxB.exp z).sub ⟨1, 0⟩ |>.div z

/-- The coincident series-DD coefficients `dₙ = (n·dₙ₋₁ − fₙ₋₁)/(n(a+n))`, `d₀ = 0`,
    `fₙ = 1/n!` — the a-divided-difference of `M(1,a+1,·)` against `eˣ`
    (`Φ(a,x) = Σ_{n≥1} dₙ xⁿ = (M_a(x)−eˣ)/a`). An EXACT rational recurrence: no
    `1/a` is ever formed (the singularity is removed analytically), so it is finite
    at `a = 0` (`dₙ → −Hₙ/n!`, `Hₙ` harmonic). The coincidence twin of `invA`;
    per-sample Horner over `z(d)` on the small-`z` (deep-tail) branch. -/
def bloomDCoef (aC : CplxB) (n : Nat) : Array CplxB := Id.run do
  let mut out : Array CplxB := #[]
  let mut dprev : CplxB := ⟨0, 0⟩   -- d₀
  let mut fprev : CplxB := ⟨1, 0⟩   -- f₀ = 1/0!
  for k in [1:n+1] do
    let kf := k.toFloat
    let dk := ((dprev.scale kf).sub fprev).div ((⟨kf, 0⟩ : CplxB).mul (aC.add ⟨kf, 0⟩))
    out := out.push dk
    dprev := dk
    fprev := fprev.scale (1.0 / kf)   -- fₖ = fₖ₋₁/k
  return out

/-- `Φ(a,κ)/g` — the d-CONSTANT of the coincident branch's `e^{νd}` carrier. TWO
    regimes, the same duality as the per-sample branches (the `Σ dₙκⁿ` sum and the
    closed form are float-representable on opposite sides):
    - `|κ| ≥ |a+1|` (shipped partials, |κ| up to ~178): the pole-FREE `lgamma(a+1)`
      bridge `Φ(a,κ) = e^κ·cexpm1(w)·(w/a) − CF(κ)`, `w = lgamma(a+1) − a·log κ`
      (identity `M_a(κ) = Γ(a+1)κ^{−a}e^κ − a·CF(κ)`; `w/a = lgamma(a+1)/a − log κ`,
      → `−γ` at `a → 0`, guarded). The `Σ dₙκⁿ` sum is D_bg3-dead here.
    - `|κ| < |a+1|` (subtle bloom / low partials): the bridge's two `κ^{−Re a}` terms
      CATASTROPHICALLY CANCEL to the tiny `Φ(a,κ) ≈ −κ/(a+1)` (relerr cliffs past
      100% by |κ|~0.01), so use the DIRECT sum `Σ dₙκⁿ` — machine-accurate here and
      float-SAFE (|κ| bounded), and `dCoef` is sized at `|a+1| ≥ |κ|`, enough terms.
    `cfK = CF(κ)` reuses the crossing branch's `bloomCF`. Build-time only. -/
def bloomPhiKappaOverG (aC kappa cfK : CplxB) (dCoef : Array CplxB) (g : Float) : CplxB :=
  if kappa.abs < (aC.add ⟨1, 0⟩).abs then
    (kappa.mul (dCoef.foldr (fun dk acc => dk.add (kappa.mul acc)) ⟨0, 0⟩)).scale (1.0 / g)
  else
    let euler : Float := 0.5772156649015329
    let laOverA : CplxB :=
      if aC.normSq < 1e-12 then ⟨-euler, 0⟩ else (lgammaB (aC.add ⟨1, 0⟩)).div aC
    let waOverA := laOverA.sub (CplxB.log kappa)
    let w := aC.mul waOverA
    ((((CplxB.exp kappa).mul (cexpm1B w)).mul waOverA).sub cfK).scale (1.0 / g)

-- ── The ROOM-CHAIN FOLD divided difference (WS-DDF) ───────────────────────────
-- A bloomed voice crossing a reverb CHAIN reassociates as (fold room1|>room2, then
-- cross the bloom once). The fold's collected residues carry a 1/Δ over the ROOM
-- poles (Δ = ν1−ν2); near-coincident rooms (two rooms sharing a mode region) drive
-- the residue `c/Δ` out of Q4.28 range as a VALUE (|c/Δ| > 8). MEASURED QUALIFIER
-- (the `ddfold` gate, arm A): at the BLOOMED site that does not yet break the
-- render — the bloom's per-sample K factor (≈ M(κ)/(gα) ≈ 1e-3 for rooms far from
-- the voice) divides the huge residue back down before anything lands, so the
-- collected fold stays finite to Δ ~ 1e-5 rad/s. The premise holds unqualified only
-- at the BARE-residue site (`modal_divdiff`'s D_dd2, where the residue lands with
-- no K factor in front of it). The fix below is the correct, modestly-tighter
-- form — and the transfer datum — not an urgent repair. The fix: the chain is
-- the DIVIDED DIFFERENCE of the bloom cross over the two room poles,
--   chain(d) = c·[Y0(μ,ν1;d) − Y0(μ,ν2;d)]/(ν1−ν2),
-- and since ν1≈ν2 ⟹ a1≈a2 it decomposes (cockpit `demos/modal_ddfold.py`, D_df2/3):
--   chain/c = e^{ν2 d}·[K1(a1)·d·cexpm1(Δd) + DDa(K1)/g] + DDa(K2)(z)/g·e^{μφ(d)}
-- reusing atom four's KIND of machinery one axis over — cexpm1 on the ν2 carrier,
-- an a-divided-difference on the Γ-bridge constants (SERIES side only: the rooms are
-- separated from the voice, |a|≫0, so `|a+1| ≥ |z|` throughout; no CF branch, no Γ★).

/-- The general-a a-divided-difference coefficients `Qₙ` (WS-DDF): `DDa(M(1,a+1,·))
    = Σ_{n≥1} Qₙ xⁿ`, the divided difference over two nearby a-values `a1, a2`
    (`aⱼ = (νⱼ−μ)/g` for a near-coincident ROOM pair). The stable recurrence never
    forms `M(a1)−M(a2)`:
      `Qₙ = Qₙ₋₁/(a2+n) − Pₙ₋₁/((a1+n)(a2+n))`, `Q₀ = 0`, `Pₙ = Pₙ₋₁/(a1+n)`, `P₀ = 1`.
    → `−Pₙ·Hₙ` (the a-derivative) at `a1 = a2`. The WS-DDF sibling of `bloomDCoef`
    (the a≈0, differ-against-`eˣ` instance); this is the general-a form for the
    room-chain FOLD's divided difference. Cockpit `demos/modal_ddfold.py` (D_df3,
    float64 vs mpmath ~8e-14). -/
def bloomFoldQCoef (a1 a2 : CplxB) (n : Nat) : Array CplxB := Id.run do
  let mut out : Array CplxB := #[]
  let mut pPrev : CplxB := ⟨1, 0⟩   -- P₀
  let mut qPrev : CplxB := ⟨0, 0⟩   -- Q₀
  for k in [1:n+1] do
    let kf := k.toFloat
    let a1k := a1.add ⟨kf, 0⟩
    let a2k := a2.add ⟨kf, 0⟩
    let qk := (qPrev.div a2k).sub (pPrev.div (a1k.mul a2k))
    out := out.push qk
    pPrev := pPrev.div a1k
    qPrev := qk
  return out

/-- `DDa(M(1,a+1,x)) = Σ_{n≥1} Qₙ xⁿ = x·Horner(Q)` over the baked `Qₙ`
    (`bloomFoldQCoef`) — build-time at `x = κ` (the `ddK1` constant), per-sample at
    `x = z(d)` in the realizer. -/
def bloomFoldDDaM (qcoef : Array CplxB) (x : CplxB) : CplxB :=
  x.mul (qcoef.foldr (fun q h => q.add (x.mul h)) ⟨0, 0⟩)

/-- One composed (voice μ, reverb ν) pair of the Γ-bridge atom. Every per-pair
    constant is a `CplxE`/`Sig` (WS-LP): the BAKED path (`besselFuse` parity — B,
    g, both pole sets fold to Floats; a change relowers) fills them with
    `cplxLitE`/`litF`, byte-identical to the old `CplxB` fields; the LIVE-σ path
    (a live-rt60 reverb pole) fills them with s0 expressions of the pole's knob
    slot, so Stage0 hoists them to the coefficient kernel and turning rt60 never
    relowers. B and g stay structural Floats (the bloom's shape, not a pole).
    The amp `c = a_voice·r_reverb` is live as always and enters linearly.
    `cfN.isEmpty` ⟺ series-only (the CF branch is not emitted at all);
    otherwise `dSwitch > 0` and the per-sample select bridges at it. -/
structure BloomPair where
  muSigma  : Sig
  muOmega  : Sig
  nuSigma  : Sig
  nuOmega  : Sig
  bloomB   : Float
  gRate    : Float
  c        : CplxE
  kappa    : CplxE
  k1Ser    : CplxE          -- C  (the κ-side constant)
  k1Cf     : CplxE          -- C − Γ★  (= −CF(κ)/g)
  fSer     : CplxE          -- −1/(ν−μ)  (the series envelope's factor)
  dSwitch  : Sig            -- 0 ⇒ series-only
  invA     : Array CplxE    -- 1/(a+k), k = 1..N (series Horner reciprocals)
  cfB      : Array CplxE    -- (2j+1)−a, j = 0..K (CF b-constants; z is added per sample)
  cfN      : Array CplxE    -- (j+1)((j+1)−a), j = 0..K−1 (CF numerators)
  -- coincidence (`|a| < ½`, WS-A4). `coincident = false` ⇒ the fields below are
  -- unused and the crossing/series-only per-sample body runs. When true: the CF
  -- branch (large z, `k1Cf`/`cfB`/`cfN`) bridges to the series-DD branch (small z)
  -- across `dSwitch`, and the τ·e secular rides a straight-μ carrier.
  coincident : Bool := false
  dCoef      : Array CplxE := #[]     -- dₙ, n = 1..N (the a-divided-difference Horner coeffs)
  k1SerDD    : CplxE := (lit 0, lit 0)  -- Φ(a,κ)/g (series-DD e^{νd} const, via the lgamma(a+1) bridge)
  eKappa     : CplxE := (lit 0, lit 0)  -- e^κ (the secular coeff c·e^κ)
  -- (μ−ν as the τ·e secular's z-coefficients: (Re, −Im) — a FIELD so the baked
  -- path emits the same single literals as before the CplxE lift.)
  secCoef    : CplxE := (lit 0, lit 0)
deriving Inhabited

/-- The bloom crossing's region for one composed (μ, ν) pair — a TOTAL partition
    of config space (`classifyBloomPair` is total, never `Option`). Dispatch is an
    exhaustive match, so a region without a handler is a compile error, and the
    depth-cap drop is a NAMED outcome (`excludedDepth`) the coverage gate tallies
    rather than a silent `continue`. This is the type the island's totality claim
    hangs off. The four served regions differ only in which per-pair lanes the
    realizer emits; the two axes are coincidence (`|a| < ½`, the pole on the
    settled partial, τ·e) and the CF/series boundary (`|κ|` vs `|a+1|`, whether the
    continued fraction is reached). -/
inductive SeamRegion where
  /-- `¬coincident ∧ |a+1| ≥ |κ|`: the E1 Kummer series alone (`M` Horner over
      `invA`). The CF is never reached — no CF lane emitted. -/
  | serOnly
  /-- `¬coincident ∧ |a+1| < |κ|`: the E1 series bridged to the continued fraction
      at `dSwitch > 0` — both lanes emitted, `selectE` picks per sample. -/
  | crossing
  /-- `|a| < ½ ∧ |κ| ≥ |a+1|` (τ·e, WS-A4): the CF (large z) bridged to the
      series-DD branch (small z) at `dSwitch > 0`, plus the τ·e secular. The E1
      series lane (`invA`/`k1Ser`, singular at a = 0) is NOT emitted. -/
  | coincidentCrossing
  /-- `|a| < ½ ∧ |κ| < |a+1|` (a subtle bloom, `dSwitch < 0`): the per-sample path
      starts on series-DD at d = 0 and never crosses, so the CF lane is dead (the
      `selectE` is const-true) and is NOT emitted — the tightest region, series-DD
      + secular only. -/
  | coincidentSubtle
  /-- Envelope depth over the 300 cap: a graceful drop (no lanes emitted). Counted,
      never silent — the coverage gate reports the excluded fraction. -/
  | excludedDepth
deriving Inhabited, DecidableEq

/-- Short label for the coverage gate's region histogram. -/
def SeamRegion.label : SeamRegion → String
  | .serOnly            => "serOnly"
  | .crossing           => "crossing"
  | .coincidentCrossing => "coincidentCrossing"
  | .coincidentSubtle   => "coincidentSubtle"
  | .excludedDepth      => "excludedDepth"

/-- The per-pair classification + baking data for one composed (μ, ν) pair — the
    executable form of the bloom atom's region partition (the sprint's epistemics
    fix: the boundary is an apparatus output, not a comment). TOTAL: `region` is
    always one of `SeamRegion`'s constructors (`excludedDepth` for a depth-cap
    drop, never a `none`). `bloomCompose` matches on `region` exhaustively to emit
    exactly that region's lanes; the seam-sweep harness and the coverage gate
    consult the SAME classifier, so "which region is this pair, and does the atom
    promise anything there" has one answer, in code. `nDepth`/`kDepth` size the
    coefficient arrays (`kDepth = 0` where no CF is reached). -/
structure BloomPairPlan where
  mu      : CplxB
  nu      : CplxB
  aC      : CplxB
  kappa   : CplxB
  region  : SeamRegion
  /-- `invA`/`dCoef` length / series depth (`nRaw + 8`). -/
  nDepth  : Nat
  /-- CF depth (`kRaw + 8`); 0 where no CF lane is emitted. -/
  kDepth  : Nat
deriving Inhabited

/-- Classify one composed (μ, ν) pair into its `SeamRegion` and size its depths —
    TOTAL (replaces the `Option`-with-flags `bloomPairPlan?`: the depth-cap drop is
    now the `excludedDepth` region, not a `none`). The control flow is preserved
    byte-for-byte from the old predicate — same `zBnd`, same `nRaw`/`kRaw` depth
    caps, same admission set — so the region label is the only new information; the
    two axes (coincidence `|a| < ½` and the CF boundary `|κ|` vs `|a+1|`) name the
    branches the old flags already encoded. -/
def classifyBloomPair (mu nu : CplxB) (B g : Float) : BloomPairPlan := Id.run do
  let aC : CplxB := (nu.sub mu).scale (1.0 / g)
  -- WS-A4 (atom four): the `|a| < ½` coincidence is served by the coincident
  -- divided difference; `|κ| < |a+1|` (a subtle bloom — see `bloomPhiKappaOverG`)
  -- is `dSwitch < 0`, where the per-sample path starts on the series-DD lane at
  -- d = 0 and the CF lane is dead.
  let coincident := aC.abs < 0.5
  let kappa := mu.scale B
  let aP1 := aC.add ⟨1, 0⟩
  let serOnly := !coincident && aP1.abs ≥ kappa.abs
  let excluded : BloomPairPlan :=
    { mu, nu, aC, kappa, region := .excludedDepth, nDepth := 0, kDepth := 0 }
  -- the worst z either per-sample branch evaluates: the branch boundary
  let zBnd := if serOnly then kappa else kappa.scale (aP1.abs / kappa.abs)
  let (_, nRaw) := bloomM1 aC zBnd
  if nRaw + 8 > 300 then return excluded
  if serOnly then
    return { mu, nu, aC, kappa, region := .serOnly, nDepth := nRaw + 8, kDepth := 0 }
  else
    let (_, kRaw) := bloomCF aC zBnd
    if kRaw + 8 > 300 then return excluded
    -- non-coincident here is always the CF-bridged crossing; coincident splits on
    -- the CF boundary (`dSwitch` sign = sign of `|κ| − |a+1|`): `|κ| ≥ |a+1|`
    -- reaches the CF (coincidentCrossing), else the CF lane is dead (subtle).
    let region : SeamRegion :=
      if !coincident then .crossing
      else if kappa.abs ≥ aP1.abs then .coincidentCrossing else .coincidentSubtle
    return { mu, nu, aC, kappa, region, nDepth := nRaw + 8, kDepth := kRaw + 8 }

/-- The bloom atom's admission predicate at the pair level: both envelope depths
    within the 300 cap — exactly the region `bloomCompose` keeps a pair (the
    classifier lands in a served region, not `excludedDepth`). TOTAL over the `|a|`
    axis since WS-A4: the `|a| < ½` coincidence is served by the divided-difference
    branch, and the ½ boundary is a scheme crossover, not an exclusion edge.
    Executable data the sweep probes at its edges, not an annotation. -/
def bloomAdmitsPair (mu nu : CplxB) (B g : Float) : Bool :=
  match (classifyBloomPair mu nu B g).region with
  | .excludedDepth => false
  | _ => true

/-- WS-LP: classify a live-σ pair over its whole σ interval — `some (region,
    nDepth, kDepth)` iff the pair sits in ONE non-coincident region at EVERY
    σ ∈ `[sigLo, sigHi]` (`serOnly` throughout, phase 1, or `crossing`
    throughout, phase 2), else `none` (the pair drops gracefully; a pair that
    CHANGES region across the interval is phase 3's union emit, and the
    coincident regions are with it). Only `Re a = (σ_μ − σ_ν)/g` moves with σ_ν
    (`Im a` and `κ = μ·B` are σ_ν-independent), so the region conditions reduce
    to interval extrema of `|a + c|` along a horizontal segment — the min at
    `Re a = −c` clamped to the segment, the max at an endpoint (convexity).
    Depths are sized at the interval's worst case (both endpoints + the closest
    approach to `−1`, where `|a+1|` bottoms out), with the baked classifier's
    `zBnd` per region (κ itself for serOnly; the branch-boundary `|z| = |a+1|`
    for crossing), same `+8` guard and `≤ 300` cap. -/
def classifyBloomPairLive (mu : CplxB) (nuOmega : Float) (sigLo sigHi : Float)
    (B g : Float) : Option (SeamRegion × Nat × Nat) := Id.run do
  let kappa := mu.scale B
  let imA := (nuOmega - mu.im) / g
  -- Re a = (−σ_ν − Re μ)/g, decreasing in σ_ν
  let reLo := (-sigHi - mu.re) / g
  let reHi := (-sigLo - mu.re) / g
  let absAt := fun (re c : Float) => Float.sqrt ((re + c) * (re + c) + imA * imA)
  let minAbsAt := fun (c : Float) =>
    absAt (max reLo (min reHi (-c))) c
  -- ¬coincident throughout: min |a| ≥ ½ (the τ·e coincidence stays baked-only)
  if minAbsAt 0.0 < 0.5 then return none
  let samples := #[reLo, reHi, max reLo (min reHi (-1.0))]
  if minAbsAt 1.0 ≥ kappa.abs then
    -- serOnly throughout: |a+1| never falls below |κ|
    let nRaw := samples.foldl (fun m re => max m (bloomM1 ⟨re, imA⟩ kappa).2) 0
    if nRaw + 8 > 300 then return none
    return some (.serOnly, nRaw + 8, 0)
  -- |a+1| dips below |κ| somewhere: crossing-throughout OR straddling (phase 3a
  -- — the union COLLAPSES onto the crossing lanes: on a serOnly-side config
  -- `dSwitch < 0`, the per-sample select sits on the series lane from d = 0,
  -- and the bridge identity makes its `Γ★ − CF(κ)/g` constant EQUAL serOnly's
  -- `mK/(ν−μ)`; f64-accurate over the whole interval because a straddling pair
  -- has `|Im a| < |κ|` by construction, which keeps the CF at z = κ inside its
  -- convergence territory — probed at ≤ 7.5e-13 vs mpmath incl. the |Im a| ≈
  -- |κ| edge (`demos/wslp_union_probe.py`; the DEEP-serOnly regime |Im a| ≫
  -- |κ| fails at rel ~50 there, which is why serOnly-throughout keeps its own
  -- arm). Only the depth sizing differs: a straddling pair's series lane runs
  -- from d = 0 on its serOnly side (|z| up to |κ|), so size at z = κ;
  -- crossing-throughout keeps the branch-boundary `zBnd` (phase 2, unchanged).
  let straddles := max (absAt reLo 1.0) (absAt reHi 1.0) ≥ kappa.abs
  let mut nRaw := 0
  let mut kRaw := 0
  for re in samples do
    let aC : CplxB := ⟨re, imA⟩
    let zBnd := kappa.scale ((aC.add ⟨1, 0⟩).abs / kappa.abs)
    nRaw := max nRaw (bloomM1 aC (if straddles then kappa else zBnd)).2
    kRaw := max kRaw (bloomCF aC zBnd).2
  if nRaw + 8 > 300 || kRaw + 8 > 300 then return none
  return some (.crossing, nRaw + 8, kRaw + 8)

/-- `bloomedVoice ⋙ reverb` as Γ-bridge pairs — the residue composition ACROSS a
    pitch-bloom warp (`B = β·scale/g` seconds of total clock advance, `g` the
    settle rate). Pole contract (WS-LP Phase 1): VOICE poles and reverb ω must
    fold to constants (`none` if not — the caller keeps such banks on
    `residueComposeE`'s unbloomed path); a reverb σ may be LIVE (the rt60 knob)
    when it is s0 (`sigIsS0` — raw slot reads, not glides) and carries a
    `sigmaRange` from the authoring site: the pair's constants are then built as
    s0 `CplxE` expressions of the live pole (Stage0 hoists them; rt60 turns with
    no relower), the live σ clamped to the classified interval so the region
    choice is sound by construction. A live pair must be `serOnly` over the
    WHOLE interval (`classifyBloomPairLiveSer`) — a pair that crosses a region
    boundary across the rt60 range drops gracefully (per pair, `continue`; the
    Phase 3 region-union emit removes this limit). Amps stay live as always.
    Admission drops (graceful exclusion, the documented v1 scope) are the
    `classifyBloomPair` `excludedDepth` region and are now depth-only: pairs whose
    envelope depth exceeds 300 (cockpit-measured shipped max ≈ 250 incl. the
    coincident CF side; the cap is headroom, not a tuning). The per-pair emit is
    REGION-INDEXED (WS-CL): the exhaustive `match plan.region` bakes exactly each
    region's lanes — the coincident regions no longer carry the dead E1 (`invA`)
    lane, and `coincidentSubtle` (`dSwitch < 0`) drops the whole CF lane the
    per-sample `selectE` always discarded. Pairs with `|a| < ½` (the pole ON the settled
    partial — the τ·e resonance) are SERVED since WS-A4 by the coincident
    divided-difference branch (CF unchanged for large z; `Φ(a,z)` series-DD + the
    `cexpm1` secular below `dSwitch`), exactly as WS-B2 hardened
    `residueComposeEC`'s coincidence. `B = 0` (or κ→0) degenerates every pair to
    series-only with `M ≡ 1` — the WS-B2 divided-difference atom, which the
    `modal-bloom-gamma` gate pins. -/
def bloomCompose (voice reverb : Array ModalMode) (B g : Float) :
    Option (Array BloomPair) := Id.run do
  let mut out : Array BloomPair := #[]
  for v in voice do
    let some vSig := sigConstF? v.sigma | return none
    let some vOm  := sigConstF? v.omega | return none
    for r in reverb do
      let some rOm  := sigConstF? r.omega | return none
      let mu : CplxB := ⟨-vSig, vOm⟩
      let c := cmulE v.ampE r.ampE
      match sigConstF? r.sigma with
      | none =>
        -- WS-LP Phase 1: a LIVE reverb σ (the rt60 pole). Admissible only when
        -- the pole read is s0 (a raw slot, not a glide — `settle` a glided pole
        -- first) and the authoring site declared its interval; otherwise the
        -- pre-WS-LP baked-pole fallback (`none` ⇒ the caller's bare bloom).
        let some (sLo, sHi) := r.sigmaRange | return none
        if !(sigIsS0 r.sigma) then return none
        match classifyBloomPairLive mu rOm sLo sHi B g with
        | none => continue   -- coincident / region-changing over the interval: graceful per-pair drop (Phase 3 widens)
        | some (region, nDepth, kDepth) =>
          -- the lift: every baked constant re-expressed as an s0 `CplxE` of the
          -- live pole. The clamp ENFORCES the classified interval in-kernel, so
          -- an out-of-range slot write saturates the crossing's σ instead of
          -- walking off the classified region. κ = μ·B is σ_ν-independent (baked).
          let sigC := clampE r.sigma (litF sLo) (litF sHi)
          let nuE : CplxE := (neg sigC, litF rOm)
          let dNuMu := csubE nuE (cplxLitE mu)
          let aE := scaleRealE (litF (1.0 / g)) dNuMu
          let invNuMuE := cdivE cOneE dNuMu
          let invA := (Array.range nDepth).map (fun k =>
            cdivE cOneE (caddE aE (litF (k + 1).toFloat, lit 0)))
          let kappaB := mu.scale B
          let kE := cplxLitE kappaB
          match region with
          | .serOnly =>
            out := out.push {
              muSigma := litF vSig, muOmega := litF vOm
              nuSigma := sigC, nuOmega := litF rOm
              bloomB := B, gRate := g, c, kappa := kE
              k1Ser := cmulE (bloomM1E invA kE) invNuMuE
              k1Cf := (lit 0, lit 0), fSer := cnegE invNuMuE
              dSwitch := lit 0, invA, cfB := #[], cfN := #[] }
          | .crossing =>
            -- WS-LP phase 2: the CF lane's constants as s0 `CplxE` of the live
            -- pole — `cfB`/`cfN` linear in `a`, `CF(κ)` by the SAME emitted
            -- fraction the lane renders with (`bloomCFE` at z = κ), the bridge
            -- constant by the emitted `Γ★` (`bloomGammaStarE`, phase 0), and
            -- `dSwitch = (ln|κ| − ½·ln|a+1|²)/g` via `logSig` (the `clogE`
            -- modulus form — no sqrt in the vocabulary needed).
            let cfB := (Array.range (kDepth + 1)).map (fun j =>
              ((sub (litF (2 * j + 1).toFloat) aE.1, neg aE.2) : CplxE))
            let cfN := (Array.range kDepth).map (fun j =>
              let jf := cplxLitE ⟨(j + 1).toFloat, 0⟩
              cmulE jf (csubE jf aE))
            let cfK := bloomCFE cfB cfN kE
            let cfOverG : CplxE := (div cfK.1 (litF g), div cfK.2 (litF g))
            let aP1 := caddE aE cOneE
            let dSwitch := div (sub (litF (Float.log kappaB.abs))
              (mul (lit 5 1) (logSig (add (mul aP1.1 aP1.1) (mul aP1.2 aP1.2))))) (litF g)
            out := out.push {
              muSigma := litF vSig, muOmega := litF vOm
              nuSigma := sigC, nuOmega := litF rOm
              bloomB := B, gRate := g, c, kappa := kE
              k1Ser := csubE (bloomGammaStarE aE kE (litF g)) cfOverG
              k1Cf := cnegE cfOverG, fSer := cnegE invNuMuE
              dSwitch, invA, cfB, cfN }
          | _ => continue   -- unreachable: the live classifier only emits serOnly/crossing
      | some rSig =>
        let nu : CplxB := ⟨-rSig, rOm⟩
        -- the shared classifier (WS-CL): `excludedDepth` ⇒ this pair is out of scope
        -- (a COUNTED graceful drop — the coverage gate tallies it, not a silent
        -- `continue`); every other region emits EXACTLY its own lanes (region-indexed
        -- — the coincident regions no longer bake the dead E1/CF lanes). `bloomCompose`
        -- and the seam-sweep harness consult THIS classifier, one answer in code.
        let plan := classifyBloomPair mu nu B g
        let aC := plan.aC
        let kappa := plan.kappa
        let aP1 := aC.add ⟨1, 0⟩
        match plan.region with
        | .excludedDepth => continue
        | .serOnly =>
          let invNuMu := CplxB.div ⟨1, 0⟩ (nu.sub mu)
          let invA := (Array.range plan.nDepth).map (fun k =>
            CplxB.div ⟨1, 0⟩ (aC.add ⟨(k + 1).toFloat, 0⟩))
          let (mK, _) := bloomM1 aC kappa
          out := out.push {
            muSigma := litF vSig, muOmega := litF vOm
            nuSigma := litF rSig, nuOmega := litF rOm
            bloomB := B, gRate := g, c, kappa := cplxLitE kappa
            k1Ser := cplxLitE (mK.mul invNuMu), k1Cf := (lit 0, lit 0)
            fSer := cplxLitE invNuMu.neg
            dSwitch := lit 0, invA := invA.map cplxLitE, cfB := #[], cfN := #[] }
        | .crossing =>
          let invNuMu := CplxB.div ⟨1, 0⟩ (nu.sub mu)
          let invA := (Array.range plan.nDepth).map (fun k =>
            CplxB.div ⟨1, 0⟩ (aC.add ⟨(k + 1).toFloat, 0⟩))
          let (cfK, _) := bloomCF aC kappa
          let cfB := (Array.range (plan.kDepth + 1)).map (fun j =>
            (⟨(2 * j + 1).toFloat - aC.re, -aC.im⟩ : CplxB))
          let cfN := (Array.range plan.kDepth).map (fun j =>
            let jf := (j + 1).toFloat
            (⟨jf, 0⟩ : CplxB).mul ((⟨jf, 0⟩ : CplxB).sub aC))
          let dSwitch := Float.log (kappa.abs / aP1.abs) / g
          let k1Cf := (cfK.scale (1.0 / g)).neg      -- −CF(κ)/g (CF-side e^{νd} const)
          let gs := bloomGammaStar aC kappa g
          out := out.push {
            muSigma := litF vSig, muOmega := litF vOm
            nuSigma := litF rSig, nuOmega := litF rOm
            bloomB := B, gRate := g, c, kappa := cplxLitE kappa
            k1Ser := cplxLitE (gs.sub (cfK.scale (1.0 / g))), k1Cf := cplxLitE k1Cf
            fSer := cplxLitE invNuMu.neg
            dSwitch := litF dSwitch, invA := invA.map cplxLitE
            cfB := cfB.map cplxLitE, cfN := cfN.map cplxLitE }
        | .coincidentCrossing =>
          -- WS-A4: the CF branch (large z, `k1Cf`/`cfB`/`cfN`), coincidence-stable,
          -- bridges below `dSwitch` to the series-DD branch (`dCoef`/`k1SerDD`) plus
          -- the τ·e secular `c·e^κ·(e^{νd}−e^{μd})/(ν−μ)`. The E1 series lane
          -- (`invA`/`k1Ser`/`fSer`, singular at a=0) is NOT emitted (region-indexed).
          let (cfK, _) := bloomCF aC kappa
          let cfB := (Array.range (plan.kDepth + 1)).map (fun j =>
            (⟨(2 * j + 1).toFloat - aC.re, -aC.im⟩ : CplxB))
          let cfN := (Array.range plan.kDepth).map (fun j =>
            let jf := (j + 1).toFloat
            (⟨jf, 0⟩ : CplxB).mul ((⟨jf, 0⟩ : CplxB).sub aC))
          let dSwitch := Float.log (kappa.abs / aP1.abs) / g
          let k1Cf := (cfK.scale (1.0 / g)).neg      -- −CF(κ)/g (CF-side e^{νd} const)
          let dCoef := bloomDCoef aC plan.nDepth
          out := out.push {
            muSigma := litF vSig, muOmega := litF vOm
            nuSigma := litF rSig, nuOmega := litF rOm
            bloomB := B, gRate := g, c, kappa := cplxLitE kappa
            k1Ser := (lit 0, lit 0), k1Cf := cplxLitE k1Cf, fSer := (lit 0, lit 0)
            dSwitch := litF dSwitch, invA := #[]
            cfB := cfB.map cplxLitE, cfN := cfN.map cplxLitE
            coincident := true
            dCoef := (dCoef.map cplxLitE)
            k1SerDD := cplxLitE (bloomPhiKappaOverG aC kappa cfK dCoef g)
            eKappa := cplxLitE (CplxB.exp kappa)
            secCoef := (litF (vSig - rSig), litF (rOm - vOm)) }
        | .coincidentSubtle =>
          -- WS-A4 subtle bloom (`dSwitch < 0`): the per-sample path is series-DD from
          -- d = 0, so the CF lane is DEAD (the `selectE` is const-true — LLVM `select`
          -- ignores the unselected operand). Region-indexed: emit series-DD + the τ·e
          -- secular ONLY — no CF lane (`k1Cf`/`cfB`/`cfN`, kept empty ⇒ `bloomComposedSig`
          -- routes to the subtle sub-branch), no `invA`. `bloomPhiKappaOverG`'s subtle
          -- branch reads only `dCoef`/κ (`cfK` unread), so a dummy `cfK` is bit-identical.
          let dCoef := bloomDCoef aC plan.nDepth
          out := out.push {
            muSigma := litF vSig, muOmega := litF vOm
            nuSigma := litF rSig, nuOmega := litF rOm
            bloomB := B, gRate := g, c, kappa := cplxLitE kappa
            k1Ser := (lit 0, lit 0), k1Cf := (lit 0, lit 0), fSer := (lit 0, lit 0)
            dSwitch := litF (Float.log (kappa.abs / aP1.abs) / g)
            invA := #[], cfB := #[], cfN := #[]
            coincident := true
            dCoef := (dCoef.map cplxLitE)
            k1SerDD := cplxLitE (bloomPhiKappaOverG aC kappa ⟨0, 0⟩ dCoef g)
            eKappa := cplxLitE (CplxB.exp kappa)
            secCoef := (litF (vSig - rSig), litF (rOm - vOm)) }
  return some out

/-- The bloom-composed pair bank as a pure `Sig` over the clock: per pair, TWO
    Q-rotator carriers — the reverb ring `e^{νd}` on the straight relative clock
    and the bloomed voice mode `e^{μφ(d)}` on the OFFSET clock (the same Q32.32
    offset add as `gongBloomWarp`; never a float round-trip of the absolute
    coordinate) — each weighted by its per-sample envelope (float, slowly
    varying — the DD's `envDf` stance): the fixed-depth series Horner, and for
    crossing pairs the fixed-depth bottom-up continued fraction with the
    branches bridged by the baked Γ★ constants and selected at `dSwitch`
    (`selectE`; the unselected lane may go non-finite off its region — the
    select picks the cockpit-validated branch, the DD's guarded-`cexpm1`
    stance). Weights land Q4.28, carriers are exact Q2.30, the pair sum is i64
    (the `modalBankSigTableDD` skeleton). Unrolled per pair — envelope depths
    are per-pair ragged, the non-uniform route. Causal gate on `clkRel > 0`.

    RANGE (WS-AA range lens). **Rail**: `|env·w| < 32` PER CARRIER, PER pair (2
    `land` calls for non-coincident pairs, 3 with the τ·e secular). A FACTOR site:
    each `w` carries the region-selected series-M / continued-fraction / Γ★ / τ·e
    factor, so the sup must be taken over d≥0 on the SELECTED lane only (a naive
    both-lane sup over-estimates by ~1e10 — the unselected series-M reaches ~5e10
    at z=κ where the selected CF weight is ~0.2). **Reachable max at every config
    that fires today: ≪ 32** (measured: SeamSweep gong-reverb 0.005; WS-A4
    coincident 0.27–0.92; a baked filterPair(Q44) stress config 2.48), so this
    site is **unprotected but not broken** at every config reachable today, and
    option E at k=0 is a byte-identical no-op for it. **Two reasons it is NOT landed
    in this pass, DEFERRED to the factor-site follow-on**: (a) it is UNREACHABLE from
    the surface — `bloomgong` (its only surface producer) is a WITHHELD kind, rejected
    by `Playground.checkServedKinds` (WS-LP made the live-rt60 CROSSING compile, but
    the surface stays sealed until this factor site carries the per-pair `k`), so
    only the tropicaltest gates and authored/loaded literal-pole programs emit it;
    (b) there is a reachable-by-baked-poles case option E CANNOT absorb — filtering a
    gong ON one of its own partials (room pole tuned to a voice partial: `Im a ≈ 0`)
    with `a` near a negative integer `−n` (`Re a ≈ −(σ_ν−σ_μ)/g`, so a resonance
    sweep walks `Re a` down the lattice −1,−2,…) AND large `|z|=|κ|`: there the
    fixed-depth float64 series-M Horner catastrophically cancels (the `1/(a+k+1)`
    factor at `k≈n−1` is near-singular — MEASURED rel error ~1e8, e.g. `|M_float|`
    ≈ 3.7e11 vs true ≈ 971 at `a=−0.98, |κ|=76.8`), so the LANDED weight can reach
    ~5e12–1e19, past the k≤28 ceiling of 32·2²⁸≈8.6e9. This is a CONDITIONING
    failure of the series REALIZATION, **not** a defect in the Γ★ bridge, which is
    EXACT there — the two lane-totals agree to 74+ digits at high precision (the
    identity `−M(z)/(ga) − CF(z)/g = −Γ(a)z^{−a}eᶻ/g`, verified
    `demos/bridge_verify.py`); the earlier "semantically invalid / lanes disagree by
    ~1e8" reading conflated the float64 Horner's own cancellation error with a lane
    disagreement. The near-integer configs are NOT incidentally depth-excluded
    (`zBnd ≈ |a+1|` is small, so `classifyBloomPair` admits them as `crossing`). Its
    correct remedy is a new `classifyBloomPair` admission guard rejecting a DISC
    around each negative integer (`|a+n| ≲ ε`), NOT the half-plane `Re a ≲ −1` (which
    excludes ~91% of the benign shipped register — measured), to a NEW `SeamRegion`
    (not `excludedDepth`, which the coverage gate's totality assertion relies on),
    landed ALONGSIDE the per-pair exponent — a correctness fix, not a landing-scale
    one. -/
def bloomComposedSig (pairs : Array BloomPair) (clkInt anchorSamples : Sig) : Sig :=
  let clkRel := relClockQ clkInt anchorSamples
  let dSec := div (div (toFloatE clkRel) (lit 4294967296)) .sampleRate
  let dPos := clampE dSec (lit 0) (lit 1000000)
  let one : CplxE := (lit 1, lit 0)
  let land := fun (env : Sig) (w : CplxE) (ph : Sig) =>
    rshift (sub (mul (toIntE (mul (mul env w.1) (lit 268435456))) (fixedCosCycSig ph))
                (mul (toIntE (mul (mul env w.2) (lit 268435456))) (fixedSinCycSig ph))) (lit 28)
  -- the per-sample continued fraction `CF(z) = Γ(a,z)eᶻz^{−a}` (`bloomCFE`,
  -- shared with the live-pole lift's CF(κ) constant), used by the crossing and
  -- coincident branches (both large-z sides).
  let cfEnv := fun (p : BloomPair) (z : CplxE) => bloomCFE p.cfB p.cfN z
  let bankQ := pairs.foldl (fun acc p =>
    let eg := expSig (neg (mul (litF p.gRate) dPos))
    let z : CplxE := (mul p.kappa.1 eg, mul p.kappa.2 eg)
    let off := mul (litF p.bloomB) (sub (lit 1) eg)
    let clkW := add clkRel (toIntE (mul (mul off .sampleRate) (lit 4294967296)))
    let phNu := modePhaseQ p.nuOmega clkRel
    let phMu := modePhaseQ p.muOmega clkW
    let envNu := expSig (neg (mul p.nuSigma dPos))
    let envMu := expSig (neg (mul p.muSigma (add dPos off)))
    if p.coincident then
      if p.cfN.isEmpty then
        -- WS-A4 / WS-CL: the subtle-bloom coincident pair (`dSwitch < 0`). The
        -- per-sample path is series-DD from d = 0 — the CF lane is dead (region-
        -- indexed: `cfN` is empty, so `cfEnv` is not even reached — it would index-
        -- panic). Series-DD + the τ·e secular, ALWAYS on. Bit-identical to the old
        -- coincident branch with the const-true `onSer` (the `selectE`s all picked
        -- the series-DD/secular arm; the LLVM `select` ignored the CF operand).
        let phiZ := cmulE z (p.dCoef.foldr (fun dk h => caddE dk (cmulE z h)) (lit 0, lit 0))
        let k2ser : CplxE := (div (neg phiZ.1) (litF p.gRate), div (neg phiZ.2) (litF p.gRate))
        let w1 := cmulE p.c p.k1SerDD
        let w2 := cmulE p.c k2ser
        let phMuS := modePhaseQ p.muOmega clkRel
        let envMuS := expSig (neg (mul p.muSigma dPos))
        let zsec : CplxE := (mul p.secCoef.1 dPos,     -- (ν−μ)d
                             mul p.secCoef.2 dPos)
        let zsq := add (mul zsec.1 zsec.1) (mul zsec.2 zsec.2)
        let big := gt zsq (litF 0.01)
        let zsafe : CplxE := (selectE big zsec.1 (lit 1), selectE big zsec.2 (lit 0))
        let ezr := expSig zsec.1
        let ez : CplxE := (mul ezr (cosSig zsec.2), mul ezr (sinSig zsec.2))
        let direct := cdivE (csubE ez one) zsafe
        let series := cexpm1SeriesE zsec
        let cxsec : CplxE := (selectE big direct.1 series.1, selectE big direct.2 series.2)
        let cek := cmulE p.c p.eKappa
        let wsec := cmulE cek (scaleRealE dPos cxsec)                   -- c·e^κ·d·cexpm1
        add acc (add (add (land envNu w1 phNu) (land envMu w2 phMu)) (land envMuS wsec phMuS))
      else
        -- WS-A4: the τ·e coincidence pair (`coincidentCrossing`). CF branch (large z,
        -- `onSer` false) bridges at `dSwitch` to the series-DD branch (small z) + the
        -- secular.
        let onSer := gt dPos p.dSwitch
        let cf := cfEnv p z
        let k2cf : CplxE := (div cf.1 (litF p.gRate), div cf.2 (litF p.gRate))   -- CF(z)/g
        -- series-DD: Φ(a,z) = z·Σ dₙ z^{n−1} (Horner over `dCoef`); k2 = −Φ(a,z)/g.
        let phiZ := cmulE z (p.dCoef.foldr (fun dk h => caddE dk (cmulE z h)) (lit 0, lit 0))
        let k2ser : CplxE := (div (neg phiZ.1) (litF p.gRate), div (neg phiZ.2) (litF p.gRate))
        let k1 : CplxE := (selectE onSer p.k1SerDD.1 p.k1Cf.1,
                           selectE onSer p.k1SerDD.2 p.k1Cf.2)
        let k2 : CplxE := (selectE onSer k2ser.1 k2cf.1, selectE onSer k2ser.2 k2cf.2)
        let w1 := cmulE p.c k1
        let w2 := cmulE p.c k2
        -- the τ·e secular `c·e^κ·e^{μd}·d·cexpm1((ν−μ)d)` on the STRAIGHT μ carrier
        -- (= `c·e^κ·(e^{νd}−e^{μd})/(ν−μ)`), gated OFF on the CF side.
        let phMuS := modePhaseQ p.muOmega clkRel
        let envMuS := expSig (neg (mul p.muSigma dPos))
        let zsec : CplxE := (mul p.secCoef.1 dPos,     -- (ν−μ)d
                             mul p.secCoef.2 dPos)
        let zsq := add (mul zsec.1 zsec.1) (mul zsec.2 zsec.2)
        let big := gt zsq (litF 0.01)
        let zsafe : CplxE := (selectE big zsec.1 (lit 1), selectE big zsec.2 (lit 0))
        let ezr := expSig zsec.1
        let ez : CplxE := (mul ezr (cosSig zsec.2), mul ezr (sinSig zsec.2))
        let direct := cdivE (csubE ez one) zsafe
        let series := cexpm1SeriesE zsec
        let cxsec : CplxE := (selectE big direct.1 series.1, selectE big direct.2 series.2)
        let cek := cmulE p.c p.eKappa
        let wsecFull := cmulE cek (scaleRealE dPos cxsec)                -- c·e^κ·d·cexpm1
        let wsec : CplxE := (selectE onSer wsecFull.1 (lit 0), selectE onSer wsecFull.2 (lit 0))
        add acc (add (add (land envNu w1 phNu) (land envMu w2 phMu)) (land envMuS wsec phMuS))
    else
      let mser := bloomM1E p.invA z
      let k2ser := cmulE mser p.fSer
      let (k1, k2) : CplxE × CplxE :=
        if p.cfN.isEmpty then (p.k1Ser, k2ser)
        else
          let cf := cfEnv p z
          let k2cf : CplxE := (div cf.1 (litF p.gRate), div cf.2 (litF p.gRate))
          let onSer := gt dPos p.dSwitch
          ((selectE onSer p.k1Ser.1 p.k1Cf.1,
            selectE onSer p.k1Ser.2 p.k1Cf.2),
           (selectE onSer k2ser.1 k2cf.1, selectE onSer k2ser.2 k2cf.2))
      let w1 := cmulE p.c k1
      let w2 := cmulE p.c k2
      add acc (add (land envNu w1 phNu) (land envMu w2 phMu)))
    (litI 0)
  selectE (gt clkRel (lit 0)) (fixedOutQ 30 bankQ) (lit 0)

/-- The bloom-composed pair bank as a TERM over the clock leaf — the realization
    of a bloomed source crossed against a (folded) room. Rides `arrUn … (.clk c)`
    like `modalBankTerm`, so master warps reach the two carriers through the
    already-warped clock. -/
def bloomComposedTerm (pairs : Array BloomPair) (anchor : Sig) (c : Clock) : ArrowTerm :=
  ArrowTerm.arrUn (fun clkSig => bloomComposedSig pairs clkSig anchor) (ArrowTerm.clk c)

/-- One composed (voice μ, near-coincident room PAIR (ν1, ν2)) of the WS-DDF fold
    atom — the divided difference of the bloom cross over the two room poles. All
    baked but the amp `c = a_voice·r1·r2` (`a_voice` live). Series side only (the
    rooms are separated from the voice, so `|a1+1| ≥ |κ|`; no CF branch, no Γ★). -/
structure BloomFoldPair where
  muSigma  : Float
  muOmega  : Float
  nu1Sigma : Float
  nu1Omega : Float
  nu2Sigma : Float
  nu2Omega : Float
  bloomB   : Float
  gRate    : Float
  c        : CplxE          -- a_voice · r1 · r2  (live amp)
  kappa    : CplxB
  k1a1     : CplxB          -- K1(a1) = M(1,a1+1,κ)/(g·a1)   (the ν2-carrier's baked coeff)
  ddK1g    : CplxB          -- DDa(K1)/g = DDa(M/a;κ)/g²      (the ν2-carrier's const term)
  invA1a2  : CplxB          -- 1/(a1·a2)   (per-sample DDa(K2) assembly)
  invA2    : CplxB          -- 1/a2
  invA     : Array CplxB    -- 1/(a1+k), k = 1..N   (M(a1;z) Horner, per-sample)
  qCoef    : Array CplxB    -- Qₙ, n = 1..N         (DDa(M;z) Horner, per-sample)
deriving Inhabited

/-- `bloomedVoice ⋙ (room1 ⋙ room2)` for a NEAR-COINCIDENT room pair — the WS-DDF
    fold atom. The chain is the divided difference of the bloom cross over the two
    room poles; each voice mode μ crosses the pair (ν1, ν2) into a `BloomFoldPair`
    carrying the `cexpm1`-carrier + a-DD constants (build-time; the a-DD is the
    general-a `bloomFoldQCoef` sibling of atom four's `bloomDCoef`). `r1`, `r2` are
    single baked room modes (the two filters folding); the voice bank's amps stay
    live. `none` if a live pole reaches the baked-pole contract; a voice mode too
    close to the rooms (`|a| < ½` or CF side `|a+1| < |κ|`) or over the 300 depth cap
    is skipped (graceful — that mode is out of the series-side v1 scope). -/
def bloomFoldCompose (voice : Array ModalMode) (r1 r2 : ModalMode) (B g : Float) :
    Option (Array BloomFoldPair) := Id.run do
  let some r1Sig := sigConstF? r1.sigma | return none
  let some r1Om  := sigConstF? r1.omega | return none
  let some r1Cre := sigConstF? r1.cre   | return none
  let some r1Cim := sigConstF? r1.cim   | return none
  let some r2Sig := sigConstF? r2.sigma | return none
  let some r2Om  := sigConstF? r2.omega | return none
  let some r2Cre := sigConstF? r2.cre   | return none
  let some r2Cim := sigConstF? r2.cim   | return none
  let nu1 : CplxB := ⟨-r1Sig, r1Om⟩
  let nu2 : CplxB := ⟨-r2Sig, r2Om⟩
  let r1r2 := (⟨r1Cre, r1Cim⟩ : CplxB).mul ⟨r2Cre, r2Cim⟩
  let mut out : Array BloomFoldPair := #[]
  for v in voice do
    let some vSig := sigConstF? v.sigma | return none
    let some vOm  := sigConstF? v.omega | return none
    let mu : CplxB := ⟨-vSig, vOm⟩
    let a1 := (nu1.sub mu).scale (1.0 / g)
    let a2 := (nu2.sub mu).scale (1.0 / g)
    let kappa := mu.scale B
    let aP1 := a1.add ⟨1, 0⟩
    -- series-side admission: rooms separated from the voice (no CF branch, no Γ★).
    if a1.abs < 0.5 || a2.abs < 0.5 || aP1.abs < kappa.abs then continue
    let (_, nRaw) := bloomM1 a1 kappa
    let nDepth := nRaw + 8    -- M(a1;z) depth; the a-DD's Hₙ factor is a small (~log) tail
    if nDepth > 300 then continue
    let qCoef := bloomFoldQCoef a1 a2 nDepth
    let invA := (Array.range nDepth).map (fun k => CplxB.div ⟨1, 0⟩ (a1.add ⟨(k + 1).toFloat, 0⟩))
    let (mK1, _) := bloomM1 a1 kappa                     -- M(a1;κ)
    let ddMκ := bloomFoldDDaM qCoef kappa                -- DDa(M;κ) = Σ κⁿ Qₙ
    let invA1a2 := CplxB.div ⟨1, 0⟩ (a1.mul a2)
    let invA2 := CplxB.div ⟨1, 0⟩ a2
    -- DDa(M/a;κ) = −M(a1;κ)/(a1 a2) + DDa(M;κ)/a2 ; k1a1 = M(a1;κ)/(g a1)
    let ddMaκ := (mK1.mul invA1a2).neg.add (ddMκ.mul invA2)
    out := out.push {
      muSigma := vSig, muOmega := vOm
      nu1Sigma := r1Sig, nu1Omega := r1Om, nu2Sigma := r2Sig, nu2Omega := r2Om
      bloomB := B, gRate := g
      c := cmulE v.ampE (cplxLitE r1r2)
      kappa
      k1a1 := mK1.div (a1.scale g)
      ddK1g := ddMaκ.scale (1.0 / (g * g))
      invA1a2, invA2, invA, qCoef }
  return some out

/-- The WS-DDF fold-atom bank as a pure `Sig` over the clock: per pair, TWO carriers
    (the ν2 ring `e^{ν2 d}` on the straight clock, the bloomed voice `e^{μφ(d)}` on
    the offset clock — as `bloomComposedSig`), weighted by the divided-difference
    form. The ν2-carrier weight is `K1(a1)·d·cexpm1(Δd) + ddK1/g` (the `cexpm1`
    secular over Δ = ν1−ν2, the `modalBankSigTableDD` shape); the voice-carrier
    weight is `DDa(K2)(z)/g = −DDa(M/a;z)/g²` per sample — `M(a1;z)` via the `invA`
    Horner, `DDa(M;z) = Σ zⁿ Qₙ` via the `qCoef` Horner. Weights land Q4.28,
    carriers exact Q2.30. Causal gate on `clkRel > 0`.

    RANGE (the WS-AA range lens' standing obligation, adopted 2026-07-20 —
    `design/seam-hardening-remainder-handoff.local.md` §2). **Rail**: the landing is
    `|w|·env·2²⁸` against a Q2.30 rotator in i64, so `|w|·env·2⁵⁸ < 2⁶³` ⇒ **per
    lane `|w|·env < 32`** — the shared modal-datapath rail
    (`design/modal-datapath-rail.local.md`; note `|w| > 32` is *necessary, not
    sufficient*: failure additionally needs differing per-mode wrap counts).
    **Reachable max**: NOT bounded by this atom's admission, which is a
    pole-DISTANCE test (`|a| ≥ ½`, `|a+1| ≥ |κ|`) — exactly the class the rail doc
    §(3) declares unsound. `ddK1g` carries the Kummer a-poles at `a₁ = −2, −3, …`,
    which admission does not exclude; an admitted config with `|a₁+3| = 0.002` lands
    ≈ 2.8e3, ~87× past the rail (in two equal-and-opposite lanes that cancel to
    ~1e-3 in the sum — a pure RANGE defect, not a math defect). Not reachable from
    any surface today: `bloomFoldCompose` has no caller outside the `ddfold` gate,
    and no Playground room bakes its poles into this shape. Routed to the option-E
    landing (remainder-handoff §1) per the role-split rule — no red witness lands
    before the fix. This predicate is `classifyBloomPair`'s shipped one transcribed
    a site over, so the exposure is INHERITED, not introduced; the atom-specific
    delta is that the divided difference is ~70× more exposed than its `k1a1`
    sibling at the same pole distance. -/
def bloomFoldComposedSig (pairs : Array BloomFoldPair) (clkInt anchorSamples : Sig) : Sig :=
  let clkRel := relClockQ clkInt anchorSamples
  let dSec := div (div (toFloatE clkRel) (lit 4294967296)) .sampleRate
  let dPos := clampE dSec (lit 0) (lit 1000000)
  let one : CplxE := (lit 1, lit 0)
  let land := fun (env : Sig) (w : CplxE) (ph : Sig) =>
    rshift (sub (mul (toIntE (mul (mul env w.1) (lit 268435456))) (fixedCosCycSig ph))
                (mul (toIntE (mul (mul env w.2) (lit 268435456))) (fixedSinCycSig ph))) (lit 28)
  let bankQ := pairs.foldl (fun acc p =>
    let eg := expSig (neg (mul (litF p.gRate) dPos))
    let z : CplxE := (mul (litF p.kappa.re) eg, mul (litF p.kappa.im) eg)
    let off := mul (litF p.bloomB) (sub (lit 1) eg)
    let clkW := add clkRel (toIntE (mul (mul off .sampleRate) (lit 4294967296)))
    let phNu2 := modePhaseQ (litF p.nu2Omega) clkRel
    let phMu := modePhaseQ (litF p.muOmega) clkW
    let envNu2 := expSig (neg (mul (litF p.nu2Sigma) dPos))
    let envMu := expSig (neg (mul (litF p.muSigma) (add dPos off)))
    -- ν2-carrier weight: K1(a1)·d·cexpm1(Δd) + ddK1/g. Δ = ν1−ν2 (pole difference).
    let w : CplxE := (mul (litF (p.nu2Sigma - p.nu1Sigma)) dPos,     -- Δ.re·d = (σ2−σ1)d
                      mul (litF (p.nu1Omega - p.nu2Omega)) dPos)      -- Δ.im·d = (ω1−ω2)d
    let wsq := add (mul w.1 w.1) (mul w.2 w.2)
    let big := gt wsq (litF 0.01)
    let wsafe : CplxE := (selectE big w.1 (lit 1), selectE big w.2 (lit 0))
    let ewr := expSig w.1
    let ew : CplxE := (mul ewr (cosSig w.2), mul ewr (sinSig w.2))
    let direct := cdivE (csubE ew one) wsafe
    let series := cexpm1SeriesE w
    let cxΔ : CplxE := (selectE big direct.1 series.1, selectE big direct.2 series.2)   -- cexpm1(Δd)
    let wNu := cmulE p.c (caddE (cmulE (cplxLitE p.k1a1) (scaleRealE dPos cxΔ)) (cplxLitE p.ddK1g))
    -- voice-carrier weight: DDa(K2)(z)/g = −DDa(M/a;z)/g², M(a1;z) & DDa(M;z) Horners.
    let mser := p.invA.foldr (fun ik h => caddE one (cmulE (cmulE z (cplxLitE ik)) h)) one   -- M(a1;z)
    let ddMz := cmulE z (p.qCoef.foldr (fun q h => caddE (cplxLitE q) (cmulE z h)) (lit 0, lit 0))
    let ddMaz := caddE (cnegE (cmulE mser (cplxLitE p.invA1a2))) (cmulE ddMz (cplxLitE p.invA2))
    let wMu := cmulE p.c (scaleRealE (litF (-1.0 / (p.gRate * p.gRate))) ddMaz)
    add acc (add (land envNu2 wNu phNu2) (land envMu wMu phMu)))
    (litI 0)
  selectE (gt clkRel (lit 0)) (fixedOutQ 30 bankQ) (lit 0)

/-- The WS-DDF fold-atom bank as a TERM over the clock leaf (rides `arrUn … (.clk c)`
    like `bloomComposedTerm`, so master warps reach the carriers). -/
def bloomFoldComposedTerm (pairs : Array BloomFoldPair) (anchor : Sig) (c : Clock) : ArrowTerm :=
  ArrowTerm.arrUn (fun clkSig => bloomFoldComposedSig pairs clkSig anchor) (ArrowTerm.clk c)

/-- The pitch-bloom clock warp for a BARE bloomed source (no room to cross): the
    offset `B·(1−e^{−g·d⁺})` added to the untouched integer clock — `B` already
    folds the register scale (`B = β·scale/g`). The scale-1 sibling of
    `gongBloomWarp` (which lives one layer up, in `Gong.lean`), defined here so
    `Patch.lowerInput` can realize a bloomed source's bare fallback without a
    circular import. `W(0)=0`, monotone, so the bank's own causal gate is
    untouched. -/
def bloomWarpClock (anchorSamples : Sig) (B g : Float) : Clock → Clock :=
  fun clk =>
    let clkRel := relClockQ clk anchorSamples
    let dSec := div (div (toFloatE clkRel) (lit 4294967296)) .sampleRate
    let dPos := clampE dSec (lit 0) (lit 1000000)
    let bloom := mul (litF B) (sub (lit 1) (expSig (neg (mul (litF g) dPos))))
    add clk (toIntE (mul (mul bloom .sampleRate) (lit 4294967296)))

-- ── The MODAL ISLAND (v1): a decaying-resonator bank as a term over the clock ──
-- The pole/modal island's emit path. A bank is a gated sum of decaying sinusoids
-- (`modalBankSig`) — the real part of Σ amp·e^{μd}. It needs NO new ArrowTerm
-- node: as a pure function of the warped clock it rides `arrUn … (.clk c)`, so
-- warps reclock it (affine = exact pole reclocking; nonlinear = the defined
-- varispeed case) and it is random-access by construction. The residue calculus
-- (voice ⋙ reverb) is a BUILD-TIME pass that fills the ModalMode array; the
-- runtime substrate is just this bank. Gated by `modal-bank`.

/-- Banks-as-data is the DEFAULT lowering: uniform (all deg-0) modal banks lower
    through the indexed reduction (`bankFold` — `modalBankSigTable` /
    `modalBankSigDirTable`) instead of unrolling. Sound because the banked
    render is bit-identical to the unroll (order-preserving loop, i64-modular
    sum — the `banks-as-data`/`banks-as-data-dir` gates pin it) and the
    coefficient columns are generation-buffered in FlatRuntime (no cross-column
    tear on live knob moves). `TROPICAL_BANKS_UNROLL` is the escape hatch back
    to the unrolled form (bisection ladder: the naive realization stays
    reachable). Reads the ONE shared flag (`Ir.banksEnabled`), read once at
    load, so the pure lowering may branch on it. -/
def banksTableEnabled : Bool := Tropical.Ir.banksEnabled

/-- A modal bank struck at `anchor` (samples) as a term over the clock leaf: no
    `gen`, no `.trop` instance — `{clk, +, ×, round, clamp, ldexp}` all the way
    down, and warp-reachable like any generator. Rides `arrUn … (.clk c)`, so
    warps reach the (banked or unrolled) body through the clock leaf identically. -/
def modalBankTerm (modes : Array ModalMode) (anchor : Sig) (c : Clock)
    (count? : Option Sig := none) : ArrowTerm :=
  -- A dynamic-count bank ALWAYS banks: a runtime count cannot be unrolled, so
  -- when `count?` is present the banked lowering fires regardless of
  -- TROPICAL_BANKS_UNROLL — the escape hatch governs only STATIC banks. (A
  -- non-uniform bank can't take the table lowering at all; it unrolls at
  -- capacity and the live count is dropped — graceful, and unreachable from
  -- the Playground, whose resonator banks are all deg-0.)
  let banked := bankIsUniform modes && (count?.isSome || banksTableEnabled)
  let lower := if banked
    then fun ms clk a => modalBankSigTable ms clk a count?
    else fun ms clk a => modalBankSig ms clk a
  ArrowTerm.arrUn (fun clkSig => lower modes clkSig anchor) (ArrowTerm.clk c)

/-- The PARTITIONED bank: the collected component through `modalBankTerm`'s
    exact lowering plus the paired component (`modalBankSigTableDD`) summed at
    the SAME clock leaf — one `arrUn`, so warps/scrub reach both bodies through
    the shared clock identically. `paired = #[]` falls through to
    `modalBankTerm` VERBATIM (the byte-identity discipline: a cold partition
    emits today's term). The two bank regions are sequential siblings, so the
    `idxId 0` reuse is safe (the `modalBankSigPairTable` precedent). -/
def modalBankTermPartitioned (plain : Array ModalMode) (paired : Array PairedMode)
    (anchor : Sig) (c : Clock) (count? : Option Sig := none) : ArrowTerm :=
  if paired.isEmpty then modalBankTerm plain anchor c count? else
  let banked := bankIsUniform plain && (count?.isSome || banksTableEnabled)
  let lowerP := if banked
    then fun ms clk a => modalBankSigTable ms clk a count?
    else fun ms clk a => modalBankSig ms clk a
  ArrowTerm.arrUn
    (fun clkSig => add (lowerP plain clkSig anchor)
                       (modalBankSigTableDD paired clkSig anchor))
    (ArrowTerm.clk c)

-- ── The DIRECTION operator: forward↔reverse crossfade ─────────────────────────
-- A bank's reading DIRECTION crossfades between the CAUSAL tail (energy at d>0, a
-- forward ring) and the ANTI-CAUSAL tail (energy at d<0, a pre-verb blooming INTO
-- the strike). Both keep the mode's own σ AND ω — only which side of the strike the
-- energy sits on changes. (An earlier version ROTATED the pole `μ↦e^{iθ}μ`; elegant,
-- but for audio modes `ω≫σ`, so any interior angle throws ω into the damping and the
-- tail decays in microseconds — silent. The crossfade never touches the decay, so it
-- stays audible across the whole knob. The rotation's cost was invisible because the
-- gate checked "bounded", not "audible".)

/-- A modal bank read with a forward↔reverse DIRECTION crossfade `dir ∈ [0,1]`.
    Per mode: the CAUSAL ring `e^{−σd}` gated `d>0`, and the ANTI-CAUSAL ring
    `e^{+σd} = e^{−σ|d|}` gated `d<0` reading the time-mirrored oscillator (`cos`
    even, `sin` odd), blended `(1−dir)·forward + dir·reverse`. σ and ω are untouched,
    so no setting can swing the frequency into the damping. `dir=0` reduces
    bit-for-bit to the forward bank (`modalBankSig`); `dir=1` is its exact time-mirror
    (reverse reverb). `dampScale?` bends the decay clock (sway) on both sides. Pure
    `f(clk)`: no state. -/
def modalBankSigDir (modes : Array ModalMode) (clkInt : Sig) (anchorSamples : Sig)
    (dir : Sig) (dampScale? : Option Sig := none) : Sig :=
  let clkRel := relClockQ clkInt anchorSamples
  let dSec := div (div (toFloatE clkRel) (lit 4294967296)) .sampleRate
  -- Q datapath, same ledger as `modalBankSig` (Q(4+k).(28−k) weight landings via
  -- option E's `le`, exact Q2.30 oscillators, i64 mode sums). The causal gates
  -- and the dir crossfade hoist OUT of the per-mode fold. `le` is derived from
  -- the AMP columns only (`bankLandExp` never reads env): the reverse arm's
  -- `envR = e^{+σd}` is > 1 for d > 0, but that landing is DISCARDED by the
  -- reverse `selectE` (active only for clkRel < 0, where envR ≤ 1) — its
  -- non-taken poison must not force k upward, and the audible reverse tail is the
  -- forward tail's time-mirror, equally |A|-bounded. So the same `le` serves both
  -- arms and `dir=0` stays bit-identical to `modalBankSig`.
  let le := bankLandExp modes
  let (fwdQ, revQ) := modes.foldl
    (fun (acc : Sig × Sig) m =>
      let phQ := modePhaseQ m.omega clkRel
      -- osc(−d) as the phase of the NEGATED relative clock (not cos-even/
      -- sin-odd), so `dir=1` is the forward tail's exact time-mirror — the
      -- fixed sine isn't bit-symmetric (one signed floor-shift), so the
      -- mirrored phase must be spelled out to match `fwd[2C−i]`. `modePhaseQ`
      -- is exact on the negated i64, so the rev side at `clkRel = −X`
      -- evaluates at bit-equal phase to the fwd side at `+X`.
      let phQN := modePhaseQ m.omega (neg clkRel)
      -- σ·d, optionally sway-bent (decay clock only, so pitch is untouched).
      let sd := mul m.sigma dSec
      let sd := match dampScale? with | none => sd | some s => mul sd s
      let envF := expSig (neg sd)
      let envF := if m.deg == 0 then envF else mul (powE dSec m.deg) envF
      let envR := expSig sd
      let envR := if m.deg == 0 then envR else mul (powE (neg dSec) m.deg) envR
      let wCreF := toIntE (mul (mul envF m.cre) le.scale)
      let wCimF := toIntE (mul (mul envF m.cim) le.scale)
      let wCreR := toIntE (mul (mul envR m.cre) le.scale)
      let wCimR := toIntE (mul (mul envR m.cim) le.scale)
      let fwdM := rshift (sub (mul wCreF (fixedCosCycSig phQ))
                              (mul wCimF (fixedSinCycSig phQ))) le.shift
      let revM := rshift (sub (mul wCreR (fixedCosCycSig phQN))
                              (mul wCimR (fixedSinCycSig phQN))) le.shift
      (add acc.1 fwdM, add acc.2 revM))
    (litI 0, litI 0)
  let fwd := selectE (gt clkRel (lit 0)) (fixedOutQ 30 fwdQ) (lit 0)
  let rev := selectE (gt (lit 0) clkRel) (fixedOutQ 30 revQ) (lit 0)
  add (mul (sub (lit 1) dir) fwd) (mul dir rev)

/-- The BANKED direction bank: `modalBankSigDir`'s value with BOTH mode sums as
    indexed reductions — the forward and reverse accumulators are two `bankFold`s
    over the SAME coefficient columns (a pair-valued fold is two scalar folds;
    each visits modes in array order, so each i64 sum agrees bit-for-bit with its
    unrolled half). This is NOT a hand table twin: both bodies are
    `modalBankSigDir`'s per-mode lambda split at the pair, written over the
    symbolic mode. Requires `deg == 0` uniformity (the `bankIsUniform` guard at
    dispatch), so the `powE` degree branches vanish, exactly as in
    `modalBankSigTable`. Same signature as `modalBankSigDir` — the dispatch in
    `modalBankTermDir` picks between them. -/
def modalBankSigDirTable (modes : Array ModalMode) (clkInt anchorSamples : Sig)
    (dir : Sig) (dampScale? : Option Sig := none) (live? : Option Sig := none) : Sig :=
  let clkRel := relClockQ clkInt anchorSamples
  let dSec := div (div (toFloatE clkRel) (lit 4294967296)) .sampleRate
  let cols := bankCols modes live?
  let le := bankLandExp modes                 -- option E: amp-only, shared fwd/rev (see modalBankSigDir)
  -- σ·d, optionally sway-bent (decay clock only, pitch untouched) — the same
  -- subterm both sides read, per symbolic mode.
  let sdOf := fun (m : ModeSym) =>
    let sd := mul m.sigma dSec
    match dampScale? with | none => sd | some s => mul sd s
  let fwdQ := bankFold cols fun m =>
    let phQ   := modePhaseQFromIncr (toIntE m.incr) clkRel
    let envF  := expSig (neg (sdOf m))
    let wCreF := toIntE (mul (mul envF m.cre) le.scale)
    let wCimF := toIntE (mul (mul envF m.cim) le.scale)
    rshift (sub (mul wCreF (fixedCosCycSig phQ))
                (mul wCimF (fixedSinCycSig phQ))) le.shift
  let revQ := bankFold cols fun m =>
    -- the mirrored phase spelled out on the NEGATED clock, as in the unrolled
    -- path (the fixed sine isn't bit-symmetric; see `modalBankSigDir`).
    let phQN  := modePhaseQFromIncr (toIntE m.incr) (neg clkRel)
    let envR  := expSig (sdOf m)
    let wCreR := toIntE (mul (mul envR m.cre) le.scale)
    let wCimR := toIntE (mul (mul envR m.cim) le.scale)
    rshift (sub (mul wCreR (fixedCosCycSig phQN))
                (mul wCimR (fixedSinCycSig phQN))) le.shift
  let fwd := selectE (gt clkRel (lit 0)) (fixedOutQ 30 fwdQ) (lit 0)
  let rev := selectE (gt (lit 0) clkRel) (fixedOutQ 30 revQ) (lit 0)
  add (mul (sub (lit 1) dir) fwd) (mul dir rev)

/-- The direction bank as a term over the clock leaf, with the LOWERING (unrolled
    or banked) supplied explicitly — shared by `modalBankTermDir` (which picks by
    flag) and the `banks-as-data-dir` equivalence gate (which builds both sides
    regardless of the flag). -/
def modalBankTermDirWith
    (lower : Array ModalMode → Sig → Sig → Sig → Option Sig → Sig)
    (modes : Array ModalMode) (anchor : Sig) (c : Clock)
    (dir : Sig) (damp? : Option (Sig × Sig) := none) : ArrowTerm :=
  ArrowTerm.arrUn
    (fun clkSig =>
      -- the sway LFO is a pure function of the SAME clock leaf the bank rides, so a
      -- master scrub reverses it coherently with the tail. Its phase is integer-
      -- reduced (`phasorPhaseSig`), never `rate·t` on unbounded float seconds.
      let dampScale? := damp?.map (fun (depth, rate) =>
        add (lit 1) (mul depth (sinSig (mul twoPiE (phasorPhaseSig rate (lit 0) clkSig)))))
      lower modes clkSig anchor dir dampScale?)
    (ArrowTerm.clk c)

/-- `modalBankTerm` with a DIRECTION crossfade. Rides the `.clk` leaf like
    `modalBankTerm`, so master warps still reach it. Dispatches to the banked
    lowering for uniform banks under the banks flag, like `modalBankTerm`. -/
def modalBankTermDir (modes : Array ModalMode) (anchor : Sig) (c : Clock)
    (dir : Sig) (damp? : Option (Sig × Sig) := none)
    (count? : Option Sig := none) : ArrowTerm :=
  -- Same dispatch rule as `modalBankTerm`: a dynamic count ALWAYS banks (a
  -- runtime count cannot be unrolled; TROPICAL_BANKS_UNROLL governs only
  -- static banks); non-uniform banks unroll at capacity, dropping the count.
  let lower : Array ModalMode → Sig → Sig → Sig → Option Sig → Sig :=
    if bankIsUniform modes && (count?.isSome || banksTableEnabled)
    then fun ms clk a d s? => modalBankSigDirTable ms clk a d s? count?
    else fun ms clk a d s? => modalBankSigDir ms clk a d s?
  modalBankTermDirWith lower modes anchor c dir damp?

/-- A bank's reading DIRECTION as data: the forward↔reverse crossfade `dir`
    (0 = forward tail, 1 = reverse/pre-verb) plus optional decay sway `(depth,
    rateHz)`. Attached to a `modalReverb` node, carried to `lowerInput`. -/
structure ModalDir where
  dir : Sig := lit 0
  damp : Option (Sig × Sig) := none

end Tropical.EmitArrow
