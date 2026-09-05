import Tropical.EmitArrow.Numerics
import Tropical.EmitArrow.Inspect
import Tropical.Ir.BanksFlag
import Tropical.EmitArrow.Term
import Tropical.Exact.Gamma

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
through a room. The low-level direction operator crossfades causal/anti-causal
tails without touching σ or ω. Public reverb fixes that room-kernel orientation
at forward before convolution; complete-output reversal remains a clock warp.
-/

namespace Tropical.EmitArrow

open Tropical.Ir
open Tropical.Exact (DyadicI CplxD CplxDI)

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
  cim   : Sig           -- Im A
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
def ModalMode.hz (fHz sigma amp : Sig) (deg : Nat := 0) : BuildM ModalMode := do
  let twoPi ← twoPiE
  let omega ← mul twoPi fHz
  let zero ← lit 0
  pure { sigma, omega, cre := amp, cim := zero, deg }

/-- `base^k` by repeated multiply (k is a mode's small polynomial order). -/
def powE (base : Sig) : Nat → BuildM Sig
  | 0 => lit 1
  | 1 => pure base
  | n + 1 => do
      let power ← powE base n
      mul power base

/- Fold an authored-constant `Sig` to a certified enclosure. This is what every
    DECISION site reads, because a decision made from a platform `libm`'s last
    bit can change the emitted program's shape (see `Tropical.Exact`), and since
    P3 it is the ONLY fold: `sigConstF?` below is its projection, so there is one
    implementation of "what does this authored constant mean" rather than a
    float one and an exact one that could disagree.

    Two things it does that the retired `Float` fold did not:

    * A DECIMAL literal is not dyadic, so `0.1` enters as a tight enclosure
      rather than pretending to be a point. That is where the authoring layer's
      exactness genuinely ends, and the carrier says so. (`JsonNumber.toFloat`,
      which the float fold used, also double-rounds — the conversion the linux
      CI miscompile was about.)
    * Division by zero POISONS instead of computing something. A caller that
      checks `ok` treats the mode as non-const and falls back to the live path,
      rather than folding a value that does not exist. -/
/- The same fold projected to its nearest `Float` — for the sites that BUILD a
    literal rather than decide with one (a baked pole read in `bloomCompose`, a
    clamp endpoint, a comb factor's ω), and for the test witnesses that read
    baked `BloomPair` fields back as doubles. Since P3 this is a projection of
    `sigConstD?` and not a second arithmetic: the value is computed exactly and
    rounded ONCE, where the retired version rounded at every node and could
    contract a multiply-add into an FMA at the host toolchain's discretion.

    It inherits `DyadicI.toFloat`'s reading of poison as `0.0`, so a `x/0` fold
    still comes back as the number zero here — the standing `sigConstF?` defect,
    preserved deliberately rather than fixed inside a carrier commit. Every
    DECISION site is already off this function and on `sigConstD?`, where the
    poison is visible. -/
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

/-- `⌊log₂|x|⌋` for a certified-positive enclosure — the exact mirror of the
    `floatExponent` op the DYNAMIC path emits (so a bank that happens to be
    all-const and one that is live agree on `k`).

    Read off the UPPER endpoint, which is the fail-safe direction: a larger
    `maxAbs` yields a larger `k`, hence a SMALLER landing scale and more
    headroom. The enclosure can only move `k` at all when the true magnitude
    sits within the enclosure width of a power of two, and then it moves it the
    safe way. -/
private def landExpZ (x : DyadicI) : Int := (DyadicI.abs x).hi.magBits - 1

/-- `k = clamp(0, 28, ⌊log₂ maxAbs⌋ − 4)` as a `Nat` (the static path). Solves
    `maxAbs < 32·2^k` for `maxAbs < 2³³` (the k=28 ceiling: above it `k` saturates
    and the guarantee lapses — a +198 dB weight, practically inert). `maxAbs < 32
    ⇒ k = 0` (bit-identical). A magnitude not certifiably above zero (a silent
    all-zero bank, or one whose enclosure cannot be separated from zero) lands
    verbatim at `k = 0`; an UNBOUNDED sup is handled by the caller, which lands
    `k = 28` (max headroom) without ever forming an infinity. -/
private def landK (maxAbs : DyadicI) : Nat :=
  if !DyadicI.certGt maxAbs DyadicI.zero then 0
  else
    let e := landExpZ maxAbs - 4
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
def LandExp.scale : LandExp → BuildM Sig
  | .static k  => lit (Int.pow 2 (28 - k))
  | .dynamic k => do
      let one ← lit 1
      let twentyEight ← lit 28
      let exponent ← sub twentyEight k
      ldexpE one exponent

/-- `28−k` — the per-mode right shift (the operand of `rshift`). `lit 28` verbatim
    at `static 0`. Always in `[0,28]` by construction, so the emitted `ashr`/`>>`
    is never the out-of-range poison/UB LLVM and MSL leave undefined. -/
def LandExp.shift : LandExp → BuildM Sig
  | .static k  => lit (28 - k)
  | .dynamic k => do
      let twentyEight ← lit 28
      sub twentyEight k

/-- The envelope-peak factor `sup_{d≥0} d^p·e^{−σd} = (p/(σe))^p` for a mode of
    degree `p` (the polynomial-order lift): `1` for `p=0` (`e^{−σd} ≤ 1`), else the
    interior peak at `d = p/σ`. `none` means NO FINITE SUP — a non-decaying
    polynomial mode (`σ ≤ 0`, `p > 0`), or a `σ` whose enclosure cannot be
    separated from zero. The caller lands `k = 28` there (max headroom; it cannot
    be made safe, and no residue calculus produces one).

    `none` in place of the old `+∞` is what removes a latent asymmetry: `+∞`
    times a zero amplitude is `NaN`, and `NaN > mx` is `false`, so the static
    fold silently SKIPPED such a mode and failed toward the wrap while the
    dynamic mirror (`floatExponent NaN = 1024`) failed toward headroom — the two
    paths disagreeing in opposite directions on the one case the rail exists
    for. The carrier has no infinities to multiply, so the caller decides
    explicitly: a silent mode contributes nothing at any envelope, and only a
    LOUD mode with an unbounded envelope forces the ceiling.

    The integer exponent is taken by exact repeated multiplication (`powNat`),
    not `exp(p·log b)` — closed form where the closed form exists. -/
private def envPeakD (deg : Nat) (sigma : DyadicI) : Option DyadicI :=
  if deg == 0 then some DyadicI.one
  else if !DyadicI.certGt sigma DyadicI.zero then none
  else
    let e := DyadicI.ofFloat 2.718281828459045
    let peak := DyadicI.powNat (DyadicI.div (DyadicI.ofNat deg) (DyadicI.mul sigma e)) deg
    if peak.ok then some peak else none

/-- The `Sig` mirror of `envPeakF` for the DYNAMIC max fold: `(p/(σe))^p·(|cre|+
    |cim|)`. Reduces to exactly `|cre|+|cim|` at `deg=0`, so a deg-0 bank's `maxSig`
    is unchanged (byte-identical dynamic path). -/
def modeWeightBoundSig (m : ModalMode) : BuildM Sig := do
  let absCre ← absE m.cre
  let absCim ← absE m.cim
  let amp ← add absCre absCim
  if m.deg == 0 then pure amp else
    let degree ← litF m.deg.toFloat
    let euler ← litF 2.718281828459045
    let denominator ← mul m.sigma euler
    let ratio ← div degree denominator
    let peak ← powE ratio m.deg
    mul peak amp

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
def bankLandExp (modes : Array ModalMode) : BuildM LandExp := do
  let builder ← get
  let constants := sigConstTable builder.exprs
  let mut mx : DyadicI := DyadicI.zero
  let mut unbounded := false
  let mut allConst := true
  for m in modes do
    -- deg-0 (the whole plain vocabulary) never queries σ, so a live-σ deg-0 bank
    -- still takes the static path when its amps const-fold.
    if m.deg == 0 then
      match sigConstDFrom? constants m.cre, sigConstDFrom? constants m.cim with
      | some cr, some ci =>
        let v := DyadicI.add (DyadicI.abs cr) (DyadicI.abs ci)
        if v.ok then mx := DyadicI.max mx v else allConst := false
      | _, _ => allConst := false
    else
      match sigConstDFrom? constants m.cre, sigConstDFrom? constants m.cim,
          sigConstDFrom? constants m.sigma with
      | some cr, some ci, some sg =>
        let amp := DyadicI.add (DyadicI.abs cr) (DyadicI.abs ci)
        if !amp.ok then allConst := false
        -- a SILENT mode contributes nothing at any envelope, so it must not
        -- force the ceiling — this is where the old `0 · ∞ = NaN` skip lived
        else if !DyadicI.certGt amp DyadicI.zero then pure ()
        else match envPeakD m.deg sg with
          | none => unbounded := true
          | some peak => mx := DyadicI.max mx (DyadicI.mul amp peak)
      | _, _, _ => allConst := false
  if allConst then
    return .static (if unbounded then 28 else landK mx)
  let zero ← lit 0
  let maxSig ← modes.foldlM (fun acc m => do
    let a ← modeWeightBoundSig m
    let greater ← gt acc a
    selectE greater acc a) zero
  let exponent ← floatExponentE maxSig
  let four ← lit 4
  let reduced ← sub exponent four
  let zero ← lit 0
  let twentyEight ← lit 28
  return .dynamic (← clampE reduced zero twentyEight)

/-- The RELATIVE clock `clkRel = clk − anchor·2³²` as an EXACT i64 subtract.
    (A float-relative clock — `toFloat(clk)/2³² − anchor` — loses mantissa bits
    as the absolute clock grows, drifting with τ.) Subtracting on
    the integer clock first keeps time-translation exact at any τ; everything
    downstream (phase reduction, the bounded envelope coordinate, the causal gate)
    sees only the bounded relative value. -/
def relClockQ (clkInt anchorSamples : Sig) : BuildM Sig := do
  let twoPow32 ← lit 4294967296
  let anchorFixed ← mul anchorSamples twoPow32
  let anchorInt ← toIntE anchorFixed
  sub clkInt anchorInt

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
def modePhaseQ (omega clkRel : Sig) : BuildM Sig := do
  let twoPi ← twoPiE
  let frequency ← div omega twoPi
  let twoPow32 ← lit 4294967296
  let scaled ← mul frequency twoPow32
  let sr ← sampleRate
  let incrementFloat ← div scaled sr
  let increment ← toIntE incrementFloat
  let thirtyTwo ← lit 32
  let thi ← rshift clkRel thirtyTwo
  let mask ← lit 4294967295
  let tlo ← bitAnd clkRel mask
  let highProduct ← mul increment thi
  let lowProduct ← mul increment tlo
  let lowShifted ← rshift lowProduct thirtyTwo
  let phase ← add highProduct lowShifted
  bitAnd phase mask

/-- `modePhaseQ` scaled to RADIANS `[0, 2π)` — for float-polynomial consumers
    (`sinSig`/`cosSig`). The fixed datapath consumes `modePhaseQ` directly. -/
def modePhaseW (omega clkRel : Sig) : BuildM Sig := do
  let twoPi ← twoPiE
  let phase ← modePhaseQ omega clkRel
  let phaseFloat ← toFloatE phase
  let twoPow32 ← lit 4294967296
  let cycles ← div phaseFloat twoPow32
  mul twoPi cycles

/-- `modePhaseQ` with the quantized increment supplied DIRECTLY, skipping the
    `incr = ⌊(ω/2π)·2³²/SR⌋` step. The banked path stores `incr`'s pre-`toInt`
    float in a coefficient column and reads it per iteration; `toInt` in-loop then
    recovers the SAME i64 (f64 storage round-trips the value exactly, < 2^53), so
    `modePhaseQFromIncr (toIntE incrFloat) clkRel` is bit-identical to
    `modePhaseQ ω clkRel`. -/
def modePhaseQFromIncr (incr clkRel : Sig) : BuildM Sig := do
  let thirtyTwo ← lit 32
  let thi ← rshift clkRel thirtyTwo
  let mask ← lit 4294967295
  let tlo ← bitAnd clkRel mask
  let highProduct ← mul incr thi
  let lowProduct ← mul incr tlo
  let lowShifted ← rshift lowProduct thirtyTwo
  let phase ← add highProduct lowShifted
  bitAnd phase mask

/-- The PERIODIC relative clock — `(τ − anchor) mod P` on the tick grid, exact
    and drift-free: the strike-train quotient computed the way every oscillator
    computes phase (the FixedPhasor reduction). The period lands ONCE as
    `incr = round(2³²/(P·SR))` (round, not truncate — a 1-ulp error in the
    float period must not cliff the increment), the split-multiply reduces
    `incr·clkRel` on ℤ/2³², and the masked phase re-expands to ticks. P
    quantizes to the 2³² grid exactly as ω does (the strike lattice shares the
    oscillators' grid story; a P whose sample count divides 2³² is EXACT — the
    gate's tight configs). The result lives in `[0, P·SR·2³²)` for ALL τ —
    before the anchor too (ashr/mask reduce negative clkRel correctly), so a
    bank reading this clock is an ETERNAL periodic train: the anchor is a
    phase reference, not a start (universe semantics — scrub anywhere, the
    train was always playing). Feed a bank as
    `modalBankSig ms (relClockQuot …) (lit 0)`: the body's own `relClockQ`
    subtracts nothing and every downstream read (dSec, phase, the causal
    gate) sees the quotient. Period in SECONDS (the physics unit, like σ/ω);
    the tick conversion reads the runtime sample rate. Requires P ≥ one
    sample (incr ≤ 2³² — the same headroom bound as an audio-rate ω). -/
def relClockQuot (clkInt anchorSamples : Sig) (pSec : Float) : BuildM Sig := do
  -- The range is (0, P] — one tick pre-subtracted before the reduction and
  -- re-added after — NOT [0, P): at the exact strike sample the quotient must
  -- read d = P (the stack as of one period ago, tails-without-fresh), because
  -- the fresh strike is the one the causal gate excludes at its own instant
  -- (`clkRel > 0`, matching an anchored bank sample-for-sample). A [0, P)
  -- read would zero the WHOLE tail stack for that one sample per bar — the
  -- gate misfiring on the eternal train's oldest coordinate. Off the strike
  -- samples the tick shift is a uniform ~2e-14 s lag — sub-quantization.
  let relative ← relClockQ clkInt anchorSamples
  let oneInt ← litI 1
  let clkRel ← sub relative oneInt
  let twoPow32 ← lit 4294967296
  let period ← litF pSec
  let sr ← sampleRate
  let periodSamples ← mul period sr
  let quotient ← div twoPow32 periodSamples
  let rounded ← roundE quotient
  let incr ← toIntE rounded
  let thirtyTwo ← lit 32
  let thi ← rshift clkRel thirtyTwo
  let mask ← lit 4294967295
  let tlo ← bitAnd clkRel mask
  let highProduct ← mul incr thi
  let lowProduct ← mul incr tlo
  let lowShifted ← rshift lowProduct thirtyTwo
  let phaseSum ← add highProduct lowShifted
  let phase ← bitAnd phaseSum mask
  let phaseFloat ← toFloatE phase
  let tickFloat ← mul phaseFloat periodSamples
  let tick ← toIntE tickFloat
  add tick oneInt

/-- A modal bank as a pure `Sig` over the (already-warped) clock: shift each pole
    to its time-since-strike on the INTEGER clock (`relClockQ`, exact at any τ),
    sum `d^deg·e^{−σd}·(c_re·cos φ − c_im·sin φ)` over the modes with each φ
    integer-reduced (`modePhaseW`), and GATE on `clkRel > 0` (causal: silent
    before the strike, a closed-form tail after — and, read through a reversing
    warp, the tail plays backward). The envelope coordinate `dSec` is float on
    the BOUNDED relative clock (`expSig` clamps; decay makes far-field precision
    irrelevant). Every sample is `f(clk)`: no state, random-access. Rides
    `arrUn … (.clk c)`, so warps reach it through the clock leaf. -/
def modalBankSig (modes : Array ModalMode) (clkInt : Sig)
    (anchorSamples : Sig) : BuildM Sig := do
  let clkRel ← relClockQ clkInt anchorSamples
  let clkFloat ← toFloatE clkRel
  let twoPow32 ← lit 4294967296
  let secondsTimesRate ← div clkFloat twoPow32
  let sr ← sampleRate
  let dSec ← div secondsTimesRate sr
  -- Q datapath: per mode, the slow scalars (envelope × residue weight) land ONCE
  -- as Q(4+k).(28−k) (`le`, option E — `k=0`/Q4.28 unless the bank's collected
  -- `|A|` crosses the i64 rail of 32; quantum 3.7e-9·2^k ≈ −168 dB at k=0 — a
  -- tail quieter than that truncates to true silence), the oscillator values are
  -- exact Q2.30 off the integer phase, and the mode SUM is i64 — modular, hence
  -- associative and commutative: reordering modes cannot move a bit, which float
  -- summation never gave us. One float scale at the boundary.
  let le ← bankLandExp modes
  let landingScale ← le.scale
  let landingShift ← le.shift
  let zeroInt ← litI 0
  let bankQ ← modes.foldlM (fun acc m => do
      let phQ ← modePhaseQ m.omega clkRel
      let sigmaTime ← mul m.sigma dSec
      let negative ← neg sigmaTime
      let env ← expSig negative
      let env2 ← if m.deg == 0 then pure env else do
        let power ← powE dSec m.deg
        mul power env
      let weightedCre ← mul env2 m.cre
      let landedCre ← mul weightedCre landingScale
      let wCre ← toIntE landedCre
      let weightedCim ← mul env2 m.cim
      let landedCim ← mul weightedCim landingScale
      let wCim ← toIntE landedCim
      let cos ← fixedCosCycSig phQ
      let cosProduct ← mul wCre cos
      let sin ← fixedSinCycSig phQ
      let sinProduct ← mul wCim sin
      let difference ← sub cosProduct sinProduct
      let oscQ ← rshift difference landingShift
      add acc oscQ) zeroInt
  let zero ← lit 0
  let afterStrike ← gt clkRel zero
  let output ← fixedOutQ 30 bankQ
  selectE afterStrike output zero

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

def bankCols (modes : Array ModalMode) (live? : Option Sig := none) : BuildM BankCols := do
  let twoPi ← twoPiE
  let twoPow32 ← lit 4294967296
  let sr ← sampleRate
  let increments ← modes.mapM fun m => do
    let frequency ← div m.omega twoPi
    let scaled ← mul frequency twoPow32
    div scaled sr
  let incr ← arr increments
  let sigma ← arr (modes.map (·.sigma))
  let cre ← arr (modes.map (·.cre))
  let cim ← arr (modes.map (·.cim))
  pure { count := modes.size, live?, incr, sigma, cre, cim }

/-- THE generic banked fold: `Σₖ body(mode k)` as one indexed reduction
    (`Sig.bankSum` → a `ReduceBegin` region — O(1) plan instructions in mode
    count). The body is an ordinary function of the symbolic mode; this is the
    only place a banked effect touches `loopIdx`/`bankSum`. Banking is a property
    of the fold, not of the effect: every banked lowering is `bankFold` applied
    to its own body, never a hand-built table twin. The loop visits modes in
    array order — the same order the unrolled `foldl` nests its adds — so for
    the i64 mode sum the render is BIT-IDENTICAL to the unroll. -/
def bankFold (cols : BankCols) (body : ModeSym → BuildM Sig) : BuildM Sig := do
  let k ← loopIdx cols.idxId
  let incr ← index cols.incr k
  let sigma ← index cols.sigma k
  let cre ← index cols.cre k
  let cim ← index cols.cim k
  let contribution ← body { incr, sigma, cre, cim }
  bankSum cols.count #[cols.incr, cols.sigma, cols.cre, cols.cim]
    contribution cols.live? cols.idxId

/-- `bankFold` with loop-invariant scalars threaded as 1-element coefficient
    columns (WS3b, fill-as-reduce). A banked body that references a
    region-external SCALAR compiles that scalar's support at the reference
    point — INSIDE the region — when the memo misses (the type-keyed memo), so
    a heavy invariant (the dynamic Q-landing chain over COMPOSED amplitudes,
    whose weights are Cauchy folds) is re-emitted into the audio loop and
    re-evaluated per iteration per sample: the measured O(modes²) shape.
    Tables have no such failure mode — `compileBankSum`'s stream law
    (`compileBankSum_stream`) materializes them ONCE before the region — so
    invariants ride the tables list as 1-element columns and the body reads
    them back by `index`. The loaded bits are the stored bits, so the value
    plane is untouched; only WHERE the invariant's support is emitted moves. -/
def bankFoldInv (cols : BankCols) (invariants : Array Sig)
    (body : ModeSym → Array Sig → BuildM Sig) : BuildM Sig := do
  let invTables ← invariants.mapM fun value => arr #[value]
  let zeroIdx ← lit 0
  let invReads ← invTables.mapM fun table => index table zeroIdx
  let k ← loopIdx cols.idxId
  let incr ← index cols.incr k
  let sigma ← index cols.sigma k
  let cre ← index cols.cre k
  let cim ← index cols.cim k
  let contribution ← body { incr, sigma, cre, cim } invReads
  bankSum cols.count (#[cols.incr, cols.sigma, cols.cre, cols.cim] ++ invTables)
    contribution cols.live? cols.idxId

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
    (live? : Option Sig := none) : BuildM Sig := do
  let clkRel ← relClockQ clkInt anchorSamples
  let clkFloat ← toFloatE clkRel
  let twoPow32 ← lit 4294967296
  let secondsTimesRate ← div clkFloat twoPow32
  let sr ← sampleRate
  let dSec ← div secondsTimesRate sr
  let le ← bankLandExp modes                                    -- option E: per-bank Q exponent
  let landingScale ← le.scale
  let landingShift ← le.shift
  let cols ← bankCols modes live?
  let mkBody := fun (scale shift : Sig) (m : ModeSym) => do
    let increment ← toIntE m.incr
    let phQ ← modePhaseQFromIncr increment clkRel
    let sigmaTime ← mul m.sigma dSec
    let negative ← neg sigmaTime
    let env ← expSig negative
    let weightedCre ← mul env m.cre
    let landedCre ← mul weightedCre scale
    let wCre ← toIntE landedCre
    let weightedCim ← mul env m.cim
    let landedCim ← mul weightedCim scale
    let wCim ← toIntE landedCim
    let cos ← fixedCosCycSig phQ
    let cosProduct ← mul wCre cos
    let sin ← fixedSinCycSig phQ
    let sinProduct ← mul wCim sin
    let difference ← sub cosProduct sinProduct
    rshift difference shift
  -- STATIC landing: fold literals, kept verbatim in the body (byte-identical
  -- plan, EmitMsl f64 emit-time folding — `modal-rail-identity` pins it).
  -- DYNAMIC landing: the chain is s0 but HEAVY once amplitudes are composed;
  -- thread it as invariant columns so it materializes before the region.
  let bankQ ← match le with
    | .static _ => bankFold cols (mkBody landingScale landingShift)
    | .dynamic _ =>
        bankFoldInv cols #[landingScale, landingShift]
          (fun m inv => mkBody inv[0]! inv[1]! m)
  let zero ← lit 0
  let afterStrike ← gt clkRel zero
  let output ← fixedOutQ 30 bankQ
  selectE afterStrike output zero

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
    (live? : Option Sig := none) : BuildM (Sig × Sig) := do
  let clkRel ← relClockQ clkInt anchorSamples
  let clkFloat ← toFloatE clkRel
  let twoPow32 ← lit 4294967296
  let secondsTimesRate ← div clkFloat twoPow32
  let sr ← sampleRate
  let dSec ← div secondsTimesRate sr
  let cols ← bankCols modes live?
  let le ← bankLandExp modes                                    -- option E: shared by Re and Im
  let landingScale ← le.scale
  let landingShift ← le.shift
  let body (imag : Bool) (scale shift : Sig) : ModeSym → BuildM Sig := fun m => do
    let increment ← toIntE m.incr
    let phQ ← modePhaseQFromIncr increment clkRel
    let sigmaTime ← mul m.sigma dSec
    let negative ← neg sigmaTime
    let env ← expSig negative
    let weightedCre ← mul env m.cre
    let landedCre ← mul weightedCre scale
    let wCre ← toIntE landedCre
    let weightedCim ← mul env m.cim
    let landedCim ← mul weightedCim scale
    let wCim ← toIntE landedCim
    if imag then
      let sin ← fixedSinCycSig phQ
      let left ← mul wCre sin
      let cos ← fixedCosCycSig phQ
      let right ← mul wCim cos
      let sum ← add left right
      rshift sum shift
    else
      let cos ← fixedCosCycSig phQ
      let left ← mul wCre cos
      let sin ← fixedSinCycSig phQ
      let right ← mul wCim sin
      let difference ← sub left right
      rshift difference shift
  -- Static landing stays verbatim; dynamic landing rides invariant columns
  -- (see `bankFoldInv`). The two folds intern the same 1-element tables, so
  -- one Pack serves both.
  let fold := fun (imag : Bool) => match le with
    | .static _ => bankFold cols (body imag landingScale landingShift)
    | .dynamic _ =>
        bankFoldInv cols #[landingScale, landingShift]
          (fun m inv => body imag inv[0]! inv[1]! m)
  let realQ ← fold false
  let imagQ ← fold true
  let gate := fun q => do
    let zero ← lit 0
    let afterStrike ← gt clkRel zero
    let output ← fixedOutQ 30 q
    selectE afterStrike output zero
  let real ← gate realQ
  let imag ← gate imagQ
  pure (real, imag)

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
def caddE (a b : CplxE) : BuildM CplxE := do
  let real ← add a.1 b.1
  let imag ← add a.2 b.2
  pure (real, imag)

def csubE (a b : CplxE) : BuildM CplxE := do
  let real ← sub a.1 b.1
  let imag ← sub a.2 b.2
  pure (real, imag)

def cmulE (a b : CplxE) : BuildM CplxE := do
  let rr ← mul a.1 b.1
  let ii ← mul a.2 b.2
  let real ← sub rr ii
  let ri ← mul a.1 b.2
  let ir ← mul a.2 b.1
  let imag ← add ri ir
  pure (real, imag)

def cdivE (a b : CplxE) : BuildM CplxE := do
  let br2 ← mul b.1 b.1
  let bi2 ← mul b.2 b.2
  let denominator ← add br2 bi2
  let arbr ← mul a.1 b.1
  let aibi ← mul a.2 b.2
  let realNumerator ← add arbr aibi
  let real ← div realNumerator denominator
  let aibr ← mul a.2 b.1
  let arbi ← mul a.1 b.2
  let imagNumerator ← sub aibr arbi
  let imag ← div imagNumerator denominator
  pure (real, imag)

def cnegE (a : CplxE) : BuildM CplxE := do
  let real ← neg a.1
  let imag ← neg a.2
  pure (real, imag)

/-- The pole `μ = −σ + iω` and complex amp `A = c_re + i·c_im` of a mode, as `CplxE`. -/
def ModalMode.poleE (m : ModalMode) : BuildM CplxE := do
  pure (← neg m.sigma, m.omega)
def ModalMode.ampE (m : ModalMode) : CplxE := (m.cre, m.cim)
def modeOfE (pole amp : CplxE) (deg : Nat := 0) : BuildM ModalMode := do
  let sigma ← neg pole.1
  pure { sigma, omega := pole.2, cre := amp.1, cim := amp.2, deg }

/-- The residue calculus, SYMBOLICALLY — `voice ⋙ reverb` with every pole/coeff an
    `Sig`. Same law as the build-time `residueCompose`: forced mode at each voice
    pole λ (amp `a·H(λ)`), ringing mode at each reverb pole ν (amp `−a·r/(λ−ν)`).
    No degeneracy branch — a build-time `|λ−ν|<tol` test is impossible on live
    poles; an exact coincidence is measure-zero for continuous knobs, and a NEAR
    coincidence gives a large finite coupling (the resonance you want). Validated
    against the Float `residueCompose` on literal poles (`symbolic-residue`). -/
def residueComposeE (voice reverb : Array ModalMode) : BuildM (Array ModalMode) := do
  voice.foldlM (fun acc v => do
    let lam ← v.poleE
    let a := v.ampE
    let zero ← lit 0
    let Hlam ← reverb.foldlM (fun sum r => do
      let pole ← r.poleE
      let difference ← csubE lam pole
      let quotient ← cdivE r.ampE difference
      caddE sum quotient) (zero, zero)
    let forcedAmp ← cmulE a Hlam
    let forced ← modeOfE lam forcedAmp
    let acc := acc.push forced
    reverb.foldlM (fun acc r => do
      let pole ← r.poleE
      let numerator ← cmulE a r.ampE
      let difference ← csubE lam pole
      let quotient ← cdivE numerator difference
      let ringingAmp ← cnegE quotient
      let mode ← modeOfE pole ringingAmp
      pure (acc.push mode)) acc) #[]

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
def residueComposeEC (voice reverb : Array ModalMode) : BuildM (Array ModalMode) := do
  if voice.isEmpty then pure #[] else
  let forced ← voice.mapM fun v => do
    let pole ← v.poleE
    let zero ← lit 0
    let Hlam ← reverb.foldlM (fun sum r => do
      let roomPole ← r.poleE
      let difference ← csubE pole roomPole
      let quotient ← cdivE r.ampE difference
      caddE sum quotient) (zero, zero)
    let amp ← cmulE v.ampE Hlam
    modeOfE pole amp
  let ringing ← reverb.mapM fun r => do
    let pole ← r.poleE
    let zero ← lit 0
    let coupling ← voice.foldlM (fun sum v => do
      let voicePole ← v.poleE
      let difference ← csubE voicePole pole
      let quotient ← cdivE v.ampE difference
      caddE sum quotient) (zero, zero)
    let product ← cmulE r.ampE coupling
    let amp ← cnegE product
    modeOfE pole amp
  pure (forced ++ ringing)

/-! Phase 3 needs the gauge's existing settling decision before the general
arena-rewrite proof surface moves in phase 4.  These two focused utilities walk
the frozen, child-descending arena in ID order: each source ID is classified or
rebuilt once, so shared subgraphs stay shared and malformed/unsupported shapes
fail closed. -/

private def readsSampleTable (arena : ExprArena) : Array Bool := Id.run do
  let mut reads : Array Bool := #[]
  for node in arena.nodes do
    let getRead := fun (id : ExprId) => (reads[id.idx]?).getD true
    let value := match node with
      | .sampleIndex => true
      | .binary _ a b => getRead a || getRead b
      | .unary _ a => getRead a
      | .clamp value lo hi => getRead value || getRead lo || getRead hi
      | .select condition then_ else_ =>
          getRead condition || getRead then_ || getRead else_
      | .index array index => getRead array || getRead index
      | .arr items => items.any getRead
      | .bankSum _ tables body _ _ => tables.any getRead || getRead body
      | .routedSum _ _ _ tables values count? _ =>
          tables.any getRead || values.any getRead || count?.any getRead
      | _ => false
    reads := reads.push value
  pure reads

private def isZeroNode (arena : ExprArena) (id : ExprId) : Bool :=
  match arena.deref id with
  | some (.num number) => number.mantissa == 0
  | _ => false

private def isOneNode (arena : ExprArena) (id : ExprId) : Bool :=
  match arena.deref id with
  | some (.num number) => number.mantissa == 1 && number.exponent == 0
  | _ => false

def settleSignals (roots : Array Sig) : BuildM (Option (Array Sig)) := do
  let source ← get
  let sourceReads := readsSampleTable source.exprs
  let mut mapped : Array Sig := #[]
  for node in source.exprs.nodes do
    let sourceId : Sig := ⟨mapped.size⟩
    let mappedAt := fun (id : ExprId) => (mapped[id.idx]?).getD sourceId
    let rebuilt ← match node with
      | .binary tag lhs rhs => binary tag (mappedAt lhs) (mappedAt rhs)
      | .unary tag arg => unary tag (mappedAt arg)
      | .clamp value lo hi =>
          -- Collapse ONLY the glide's own ramp fraction — `clamp(elapsed/duration,
          -- 0, 1)`, a division whose numerator reads the clock and whose
          -- denominator does not — to its saturated endpoint. The previous
          -- shape (any clock-reading clamp01) also collapsed OUTER clamp01s
          -- wrapped around glide VALUES (a direction weight, the gauge's g),
          -- forcing them to 1 regardless of their target — measured as the
          -- `modal-rail-dir` amplitude collapse.
          let isRampFraction :=
            isZeroNode source.exprs lo && isOneNode source.exprs hi &&
              match source.exprs.deref value with
              | some (.binary .div numerator denominator) =>
                  ((sourceReads[numerator.idx]?).getD true) &&
                    !((sourceReads[denominator.idx]?).getD true)
              | _ => false
          if isRampFraction then
            lit 1
          else
            clamp (mappedAt value) (mappedAt lo) (mappedAt hi)
      | .select condition then_ else_ =>
          select (mappedAt condition) (mappedAt then_) (mappedAt else_)
      | .arr items => arr (items.map mappedAt)
      | .index array idx => index (mappedAt array) (mappedAt idx)
      | .bankSum count tables body count? idxId =>
          bankSum count (tables.map mappedAt) (mappedAt body) count? idxId
      | .routedSum capacity outputCount routes tables values count? idxId =>
          routedSum capacity outputCount routes (tables.map mappedAt)
            (values.map mappedAt) (count?.map mappedAt) idxId
      | _ => pure sourceId
    mapped := mapped.push rebuilt
  let settled := roots.map fun root => (mapped[root.idx]?).getD root
  let result ← get
  let reads := readsSampleTable result.exprs
  if settled.any fun root => (reads[root.idx]?).getD true then
    pure none
  else
    pure (some settled)

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
def gaugeScale (g : Sig) (modes : Array ModalMode) : BuildM (Option Sig) := do
  -- SETTLE every pole first — the whole scale then reads no clock (s0), so the
  -- `floatExponent` inside `logSig` is coefficient-time. `none` if any won't settle.
  let roots := modes.flatMap fun mode => #[mode.sigma, mode.omega, mode.cre, mode.cim]
  let some settled ← settleSignals roots | pure none
  let sm := modes.mapIdx fun index mode =>
    ({ mode with
      sigma := settled[4 * index]!
      omega := settled[4 * index + 1]!
      cre := settled[4 * index + 2]!
      cim := settled[4 * index + 3]! } : ModalMode)
  -- H(iωₖ) = Σᵢ Aᵢ/(σᵢ + i(ωₖ − ωᵢ))   (iωₖ − μᵢ, with μᵢ = −σᵢ + iωᵢ)
  let hAt := fun (wk : Sig) => do
    let zero ← lit 0
    sm.foldlM (fun acc m => do
      let difference ← sub wk m.omega
      let quotient ← cdivE m.ampE (m.sigma, difference)
      caddE acc quotient) (zero, zero)
  -- S = Σₖ |H(iωₖ)|⁸ = Σₖ (|H|²)⁴  (p = 8; even power ⇒ no sqrt)
  let zero ← lit 0
  let S ← sm.foldlM (fun acc m => do
      let h ← hAt m.omega
      let real2 ← mul h.1 h.1
      let imag2 ← mul h.2 h.2
      let h2 ← add real2 imag2
      let h4a ← mul h2 h2
      let h4b ← mul h2 h2
      let h8 ← mul h4a h4b
      add acc h8) zero
  -- ‖H‖⁻ᵍ = S^{−g/8} = exp(−(g/8)·ln S). Floor S ∈ [1e−30, 1e30] (via `lit`, NOT
  -- `litF` — `litF 1e−30` rounds to 0, `litF 1e300` saturates to ≈1.8e7): the floor
  -- guards a silent bank (S→0 ⇒ logSig 0), the ceiling a non-finite one (S=∞ ⇒
  -- logSig ∞ = NaN), and 1e30 stays under f32-max so Metal never overflows the clamp.
  let negativeG ← neg g
  let oneEighth ← lit 125 3
  let exponentScale ← mul negativeG oneEighth
  let floor ← lit 1 30
  let ceiling ← lit (10^30)
  let bounded ← clampE S floor ceiling
  let logarithm ← logSig bounded
  let exponent ← mul exponentScale logarithm
  pure (some (← expSig exponent))

def normalizePeak (g : Sig) (modes : Array ModalMode) : BuildM (Array ModalMode) := do
  if modes.isEmpty then pure #[] else
  match ← gaugeScale g modes with
  | some scale => modes.mapM fun m => do
      let cre ← mul m.cre scale
      let cim ← mul m.cim scale
      pure { m with cre, cim }
  | none => pure modes   -- un-settleable poles: decline to identity, never an s1 norm

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
def residueComposeBanked (voice reverb : Array ModalMode) : BuildM (Array ModalMode) := do
  if voice.isEmpty then pure #[] else
  let idH : Nat := 1
  -- Σₖ (rRe+i·rIm)/((pRe−nRe[k]) + i(pIm−nIm[k])), real or imag part, as a scalar
  -- bankSum; per term = `cdivE` (r·conj(d)/|d|²), left-associative.
  let cauchy := fun (pRe pIm nRe nIm rRe rIm : Sig)
      (count : Nat) (imag : Bool) => do
    let k ← loopIdx idH
    let nReK ← index nRe k
    let dRe ← sub pRe nReK
    let nImK ← index nIm k
    let dIm ← sub pIm nImK
    let dRe2 ← mul dRe dRe
    let dIm2 ← mul dIm dIm
    let denominator ← add dRe2 dIm2
    let rkRe ← index rRe k
    let rkIm ← index rIm k
    let contribution ← if imag then do
      let left ← mul rkIm dRe
      let right ← mul rkRe dIm
      let numerator ← sub left right
      div numerator denominator
    else do
      let left ← mul rkRe dRe
      let right ← mul rkIm dIm
      let numerator ← add left right
      div numerator denominator
    bankSum count #[nRe, nIm, rRe, rIm] contribution none idH
  let reverbPoles ← reverb.mapM (·.poleE)
  let voicePoles ← voice.mapM (·.poleE)
  let revNuRe ← arr (reverbPoles.map (·.1))
  let revNuIm ← arr (reverbPoles.map (·.2))
  let revRRe ← arr (reverb.map (·.ampE.1))
  let revRIm ← arr (reverb.map (·.ampE.2))
  let voNuRe ← arr (voicePoles.map (·.1))
  let voNuIm ← arr (voicePoles.map (·.2))
  let voARe ← arr (voice.map (·.ampE.1))
  let voAIm ← arr (voice.map (·.ampE.2))
  -- forced (over reverb): Hlam = Σᵣ r/(λ−ν), amp = a·Hlam
  let forced ← voice.zip voicePoles |>.mapM fun (v, pole) => do
    let hRe ← cauchy pole.1 pole.2 revNuRe revNuIm revRRe revRIm reverb.size false
    let hIm ← cauchy pole.1 pole.2 revNuRe revNuIm revRRe revRIm reverb.size true
    let amp ← cmulE v.ampE (hRe, hIm)
    modeOfE pole amp
  -- ringing (over voice): with pole=ν the bankSum gives Σᵥ a/(ν−λ) = −coupling, so
  -- the EC amp `−r·coupling` becomes `r·(bankSum)` — no `cnegE`.
  let ringing ← reverb.zip reverbPoles |>.mapM fun (r, pole) => do
    let cRe ← cauchy pole.1 pole.2 voNuRe voNuIm voARe voAIm voice.size false
    let cIm ← cauchy pole.1 pole.2 voNuRe voNuIm voARe voAIm voice.size true
    let amp ← cmulE r.ampE (cRe, cIm)
    modeOfE pole amp
  pure (forced ++ ringing)

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
def integrateBank (modes : Array ModalMode) : BuildM (Array ModalMode) := do
  let integ ← modes.mapM fun m => do
    let pole ← m.poleE
    let amp ← cdivE m.ampE pole
    modeOfE pole amp
  let zero ← lit 0
  let sumAmp ← integ.foldlM (fun sum m => caddE sum m.ampE) (zero, zero)
  let constantAmp ← cnegE sumAmp
  let constant ← modeOfE (zero, zero) constantAmp
  pure (integ.push constant)

/-- The quadrature node table `(θᵢ, sin θᵢ)` for the default 256-panel trapezoid,
    certified once. A nullary `def`, so it is built at module init and every
    `besselJD` call is a table read — without it each call would pay 257 exact
    sines on top of its 257 exact cosines. -/
def besselQuadNodes : Array (DyadicI × DyadicI) := Id.run do
  let quad := 256
  let h := DyadicI.div Tropical.Exact.piI (DyadicI.ofNat quad)
  let mut out : Array (DyadicI × DyadicI) := #[]
  for i in [0:quad + 1] do
    let th := DyadicI.mul h (DyadicI.ofNat i)
    out := out.push (th, DyadicI.sin th)
  return out

/-- `Jₙ(b)` on the exact carrier — the SAME trapezoid, the same node count, no
    libm. Only the default `quad = 256` is tabulated; any other panel count
    recomputes its nodes. -/
def besselJD (nf b : DyadicI) (quad : Nat := 256) : DyadicI := Id.run do
  let h := DyadicI.div Tropical.Exact.piI (DyadicI.ofNat quad)
  let nodes := if quad == 256 then besselQuadNodes else Id.run do
    let mut o : Array (DyadicI × DyadicI) := #[]
    for i in [0:quad + 1] do
      let th := DyadicI.mul h (DyadicI.ofNat i)
      o := o.push (th, DyadicI.sin th)
    return o
  let mut acc : DyadicI := DyadicI.zero
  for i in [0:quad + 1] do
    let (th, sth) := nodes[i]!
    let term := DyadicI.cos (DyadicI.sub (DyadicI.mul nf th) (DyadicI.mul b sth))
    acc := DyadicI.add acc (if i == 0 || i == quad then term.shift (-1) else term)
  return DyadicI.div (DyadicI.mul acc h) Tropical.Exact.piI

/-- `Jₙ(b)` (Bessel, first kind) by trapezoid on the periodic integrand
    `cos(nθ − b·sin θ)` over `[0,π]` — spectral accuracy on the smooth periodic
    integrand; the FM sideband weights of `besselFuse`. The index `nf` is the
    sideband number as a `Float` (so callers avoid `Int→Float`).

    The `Float`-facing wrapper over `besselJD`: the signature is unchanged, so
    `besselFuse` and the two `modal-bessel` oracles keep compiling, while the
    257 cosines underneath stop being the platform's. -/
def besselJ (nf b : Float) (quad : Nat := 256) : Float :=
  DyadicI.toFloat (besselJD (DyadicI.ofFloat nf) (DyadicI.ofFloat b) quad)

/-- Static-index FM as a build-time pole move (Jacobi–Anger). Modulating a bank by
    `sin(ω_m d)` at index `b` sprouts, from each mode `(μ, A)`, a comb of sidebands
    at pole `μ + i·n·ω_m` with amp `A·Jₙ(b)`, `n ∈ [−N, N]` — so an FM'd voice is
    STILL a modal bank (it keeps poles, so it can feed the residue calculus). `ω_m`
    (rad/s) and `b` are baked in v1 (a change relowers); the carrier's poles/amps
    stay live (each sideband adds a baked offset to `ω` and scales `A` by the real
    `Jₙ(b)`). `N` is the sideband capacity — the tail `|n| > b` decays
    superexponentially, so `N ≈ ⌈b⌉ + few` is exact to machine precision
    (`demos/modal_fm.py` D4). deg-0. -/
def besselFuse (modes : Array ModalMode) (wm b : Float)
    (N : Nat) : BuildM (Array ModalMode) := do
  let mut out : Array ModalMode := #[]
  for m in modes do
    for i in [0:2 * N + 1] do
      let nf := i.toFloat - N.toFloat
      let jn := besselJ nf b
      let offset ← litF (nf * wm)
      let omega ← add m.omega offset
      let coefficient ← litF jn
      let cre ← mul m.cre coefficient
      let cim ← mul m.cim coefficient
      out := out.push
        { sigma := m.sigma
        , omega
        , cre
        , cim
        , deg := m.deg }
  pure out

/-- Affine pole reclock: the pole-space image of the affine clock warp `d ↦ a·d+b`.
    `e^{μ(a d + b)} = e^{μb}·e^{(μa)d}`, so the pole scales `μ ↦ μa` (`σ↦σa, ω↦ωa`)
    and the amp rotates `A ↦ A·e^{μb} = A·e^{−σb}(cos ωb + i sin ωb)`. Pure `CplxE`
    over `expSig`/`sinSig`/`cosSig`, so `a`,`b` may be live. AFFINE ONLY: a nonlinear
    warp does not preserve poles (the varispeed case is deferred, per
    `design/modal-island.local.md`); `b` in seconds, `a` dimensionless. deg-0 (the
    `(a d+b)^k` binomial for `deg>0` is out of scope for v1). -/
def reclockAffine (a b : Sig) (modes : Array ModalMode) : BuildM (Array ModalMode) :=
  modes.mapM fun m => do
    let sigmaOffset ← mul m.sigma b
    let negative ← neg sigmaOffset
    let envB ← expSig negative
    let omegaOffset ← mul m.omega b
    let cosine ← cosSig omegaOffset
    let real ← mul envB cosine
    let sine ← sinSig omegaOffset
    let imag ← mul envB sine
    let amp ← cmulE m.ampE (real, imag)
    let sigma ← mul m.sigma a
    let omega ← mul m.omega a
    pure { sigma, omega, cre := amp.1, cim := amp.2, deg := m.deg }

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
def residueComposeDD (voice reverb : Array ModalMode) : BuildM (Array PairedMode) := do
  let rows ← voice.mapM fun v => do
    let lam ← v.poleE
    reverb.mapM fun r => do
      let nu ← r.poleE
      let c ← cmulE v.ampE r.ampE
      pure { lam, nu, c }
  pure (rows.foldl (· ++ ·) #[])

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
    accurate everywhere, so θ is a COST boundary, not a correctness boundary).

    RATE PROVENANCE (the constant is a function of the datapath, not physics):
    frozen AT SR = 44100 with the 2³² rotator — the grid quantum `2π·SR/2³²`
    is LINEAR in SR and the advantage crossing tracks it, so the 44.1k–96k
    family moves the crossing by ≤ ~2.2×, well inside the ×10 margin. If the
    rotator width or the served rate family ever changes, re-freeze off the
    cockpit's D_p1c sweep (`demos/ecdd_partition.py`) rather than trusting
    the margin. -/
def ecddThetaAcc : Float := 0.4642

/-- The range lens: route when the collected ringing weight `|a·r|/|Δ|` exceeds
    the Q4.28 magnitude ceiling 8 with a 2× margin. Amp-dependent — `|Δ|` alone
    is not the criterion (cockpit D_p2: binding for `|a·r| ≳ 1.9`).

    SERVICE REGION (the cap interplay, stated — it is NOT the whole lens): the
    paired range cap admits only `|c|·min(2/|Δ|, 1/(e·σ_min)) < 8`, and the
    `2/|Δ|` arm of that min is EXACTLY the complement of this lens
    (`|c|·2/|Δ| < 8 ⟺ |c|/|Δ| < 4`). So a rail-fired coupling can only route
    through the DAMPING arm: `|Δ| < 2e·σ_min` (the damping bound is the binding
    sup) and `|c| < 8e·σ_min` (it clears). Well-damped heavy couplings route;
    a lightly-damped heavy coupling is `CouplingRoute.refused` — a STATED
    exclusion, never a silent fallback. -/
def ecddRailCeil : Float := 4.0

/-- The paired atom's own build-time range cap: per pair, `|Wc| ≤ |c|·sup_d
    |e^{νd}·d·cexpm1(Δd)| ≤ |c|·min(2/|Δ|, 1/(e·σ_min))` (the sup the
    remainder-handoff deferred until the DD wiring landed — this is that
    landing). A coupling whose bound exceeds the cap is NOT routed — it stays
    collected at the status-quo floor (wrong near coincidence), and the refusal
    is FIRST-CLASS (`CouplingRoute.refused`): the merged seam atom's admission
    excludes it rather than certifying the collected floor there. The scale arm
    is the recorded route out if a real patch ever lands in the refusal region.
    Cap 8 = the plain Q4.28 ceiling, 4× under the DD site's i64 rail of 32. -/
def ecddPairCap : Float := 8.0

/-! The same three lenses on the exact carrier — what the ROUTER actually
    compares against, since its verdict picks which body a coupling is emitted
    into. Each is the SAME double (a finite `Float` is a dyadic, so `ofFloat`
    moves nothing); the `Float` definitions above stay as the published,
    documented constants and as the seam sweep's reference. -/

def ecddThetaAccD : DyadicI := DyadicI.ofFloat ecddThetaAcc
def ecddRailCeilD : DyadicI := DyadicI.ofFloat ecddRailCeil
def ecddPairCapD  : DyadicI := DyadicI.ofFloat ecddPairCap

/-- The σ INTERVAL a mode's damping ranges over — what the ROUTER reads, since
    its verdict changes which body a coupling is emitted into. A const σ is its
    own point interval; a LIVE σ with a declared `sigmaRange` classifies over the
    knob span (WS-LP's build-time-over-the-declared-interval discipline); a live
    σ without a range is unclassifiable (`none` ⇒ the coupling stays collected). A declared
    `sigmaRange` is a pair of authored `Float`s and enters the carrier exactly
    (a finite double is a dyadic), so nothing about the declaration moves. -/
private def sigmaIntervalDFrom? (constants : Array (Option DyadicI))
    (m : ModalMode) : Option (DyadicI × DyadicI) :=
  match sigConstDFrom? constants m.sigma with
  | some s => some (s, s)
  | none => m.sigmaRange.map (fun (lo, hi) => (DyadicI.ofFloat lo, DyadicI.ofFloat hi))

def sigmaIntervalD? (m : ModalMode) : BuildM (Option (DyadicI × DyadicI)) := do
  let builder ← get
  pure (sigmaIntervalDFrom? (sigConstTable builder.exprs) m)

/-- A mode's pole with a LIVE σ CLAMPED to its declared interval (coefficient-
    time, the WS-LP kernel-clamp precedent) — what makes the paired route's
    build-time range cap sound by construction even if the host drives the knob
    out of its declared span. A const σ passes through untouched. Production
    unit modes arrive PRE-clamped (`clampSigmas` at graph ingestion), making
    this a value-identical second wrap there; it stays as defense in depth for
    direct callers (fixtures, the seam sweep) that own their own discipline. -/
private def clampedPoleEWith (constants : Array (Option DyadicI))
    (m : ModalMode) : BuildM CplxE := do
  match sigConstDFrom? constants m.sigma, m.sigmaRange with
  | none, some (lo, hi) =>
      let lo ← litF lo
      let hi ← litF hi
      let sigma ← clampE m.sigma lo hi
      pure (← neg sigma, m.omega)
  | _, _ => m.poleE

def clampedPoleE (m : ModalMode) : BuildM CplxE := do
  let builder ← get
  clampedPoleEWith (sigConstTable builder.exprs) m

/-- Clamp every LIVE σ with a declared interval to that interval, in-kernel —
    the UNIFORM-BANK extension of `clampedPoleE`, applied to unit modes at
    graph ingestion (`lowerModal`'s source/reverb arms). Two things it buys:
    (1) CONSISTENCY under knob overdrive — the collected modes and the paired
    atoms of one bank saturate TOGETHER when the host drives a knob out of its
    declared span, instead of the paired lanes clamping while the collected
    lanes follow the raw knob (an internally inconsistent bank); (2) COLD
    soundness — `couplingHot` decides cold at min |Δ| over the DECLARED
    interval, so an unclamped out-of-span drive could dip a cold coupling's
    |Δ| under θ_acc while it stays collected; the clamp makes the declared
    interval the enforced one. `sigmaRange` is kept on the mode (the
    classifiers still read it); the wrap is value-identity for any in-span
    drive, so in-range behavior is bit-identical. Const σ passes untouched
    (the cold byte-identity discipline). -/
def clampSigmas (modes : Array ModalMode) : BuildM (Array ModalMode) := do
  let builder ← get
  let constants := sigConstTable builder.exprs
  modes.mapM fun mode => do
    match sigConstDFrom? constants mode.sigma, mode.sigmaRange with
    | none, some (lo, hi) =>
        let lo ← litF lo
        let hi ← litF hi
        let sigma ← clampE mode.sigma lo hi
        pure { mode with sigma }
    | _, _ => pure mode

/-- Where a (v, r) coupling lands — the routing verdict with the paired range
    cap made a FIRST-CLASS outcome rather than a silent fallback, so the seam
    apparatus can state the refusal region as admission instead of certifying
    the collected floor over it. -/
inductive CouplingRoute where
  /-- Neither lens fires (or the coupling is unmeasurable at build time): the
      collected `±c/Δ` representation is accurate — stays in the Cauchy sums. -/
  | cold
  /-- A lens fires and the paired range cap clears: one fused `PairedMode`. -/
  | paired
  /-- A lens fires but the pair's landed sup exceeds `ecddPairCap`: stays
      collected at the status-quo floor (wrong near coincidence) — a STATED
      exclusion, out of the merged atom's certified region. Reachable only at
      extreme Q (`σ_min < |c|/(8e)` with a fired lens — rt60 of minutes at
      unit amps); the scale arm is the recorded route out. -/
  | refused
deriving DecidableEq

/-- The per-coupling routing verdict — compile-time only. σ may be LIVE with a
    declared range (Phase 2): the lenses evaluate at min |Δ| over the interval —
    a coupling whose interval DIPS under θ takes DD throughout the knob span, so
    no runtime select ever exists (D2; the WS-LP phase-3a pattern). ω and amps
    must const-fold, both modes deg-0; anything else is `cold`. A lens fires
    (accuracy | range) ⇒ `paired` if the paired range cap clears at the
    interval's worst point, else `refused`.

    BOTH AXES of the pole distance decide: `dAbs = √(dSig² + dw²)`, `dSig` the
    σ-span distance and `dw` the exact ω difference. Two modes at the SAME
    frequency but differently damped are SEPARATED, not coincident — which is
    what the σ axis is for, and what it did not do until the `dSig` repair
    (gate `ecdd-sigma-axis`). -/
private def classifyCouplingWith (constants : Array (Option DyadicI))
    (v r : ModalMode) : CouplingRoute :=
  if !(v.deg == 0 && r.deg == 0) then .cold else
  Id.run do
    let some (svLo, svHi) := sigmaIntervalDFrom? constants v | return .cold
    let some (srLo, srHi) := sigmaIntervalDFrom? constants r | return .cold
    let some wv := sigConstDFrom? constants v.omega | return .cold
    let some wr := sigConstDFrom? constants r.omega | return .cold
    let some ar := sigConstDFrom? constants v.cre | return .cold
    let some ai := sigConstDFrom? constants v.cim | return .cold
    let some rr := sigConstDFrom? constants r.cre | return .cold
    let some ri := sigConstDFrom? constants r.cim | return .cold
    -- min |Δ| over the σ interval(s): the SPAN DISTANCE between `[svLo,svHi]`
    -- and `[srLo,srHi]` — `max(0, max(svLo,srLo) − min(svHi,srHi))`, so zero
    -- when the spans overlap (the dipper) and the gap when they are separated —
    -- with ω exact. A const σ is its own point interval, so this degenerates to
    -- `|σ_v − σ_r|`. The two axes separate because `dw` does not depend on σ:
    -- minimizing `|Δ| = √((σ_r−σ_v)² + (ω_v−ω_r)²)` over the spans is exactly
    -- minimizing the σ term, so `dAbs` below IS min |Δ|.
    --
    -- THE THREE-ANSWER DISCIPLINE. `certGt` answers only on SEPARATED
    -- enclosures; an overlap — spans touching to within the enclosure width, or
    -- a σ whose `sigConstD?` fold poisoned — falls to `dSig := 0`, and that is
    -- the CONSERVATIVE side. Every consumer of `dAbs` wants a LOWER bound on
    -- min |Δ|: both lenses fire more readily as it shrinks, and the paired range
    -- cap's sup `2·|c|/|Δ|` only grows. Under-measuring spends plan SIZE;
    -- over-measuring spends ACCURACY, because DD is accurate everywhere while
    -- the collected `±c/Δ` floor is the one that is wrong near coincidence, and
    -- θ is a cost boundary (see `ecddThetaAcc`).
    --
    -- (Until 2026-07-24 `sepHi` took a `max` here, so `sepLo ≤ sepHi` held
    -- unconditionally, `dSig` was identically zero, and `dAbs` collapsed to
    -- `|ω_v − ω_r|` — any two modes sharing a frequency routed hot however
    -- differently damped. Repaired in its own commit, out of the carrier flip,
    -- so its differential has exactly one variable in it; gate
    -- `ecdd-sigma-axis` now pins the axis in both directions.)
    let sepLo := DyadicI.max svLo srLo
    let sepHi := DyadicI.min svHi srHi
    let dSig := if DyadicI.certGt sepLo sepHi then DyadicI.sub sepLo sepHi else DyadicI.zero
    let dw := DyadicI.sub wv wr
    let dAbs := DyadicI.sqrt (DyadicI.add (DyadicI.mul dSig dSig) (DyadicI.mul dw dw))
    let cAbs := DyadicI.mul (CplxDI.abs (CplxDI.mkI ar ai)) (CplxDI.abs (CplxDI.mkI rr ri))
    let dPos := DyadicI.certGt dAbs DyadicI.zero
    if !(DyadicI.certLt dAbs ecddThetaAccD
         || (dPos && DyadicI.certGt (DyadicI.div cAbs dAbs) ecddRailCeilD)) then
      return .cold
    -- the paired range cap: sup of the divided-difference kernel over the
    -- WHOLE interval — σ_min at the spans' low edge, |Δ| at its minimum
    -- (sound: the paired pole clamps to the interval, `clampedPoleE`).
    -- `min(sup₁, sup₂) < cap ⟺ either bound clears` — no infinity sentinels.
    let smin := DyadicI.min svLo srLo
    let capOk :=
      (dPos && DyadicI.certLt (DyadicI.mul cAbs (DyadicI.div (DyadicI.ofInt 2) dAbs))
                              ecddPairCapD)
      || (DyadicI.certGt smin DyadicI.zero
          && DyadicI.certLt (DyadicI.div cAbs (DyadicI.mul Tropical.Exact.DyadicI.eulerI smin))
                            ecddPairCapD)
    return if capOk then .paired else .refused

def classifyCoupling (v r : ModalMode) : BuildM CouplingRoute := do
  let builder ← get
  pure (classifyCouplingWith (sigConstTable builder.exprs) v r)

/-- `true` ⇒ the (v, r) coupling leaves the collected sums and becomes one
    `PairedMode` (`classifyCoupling = .paired`). -/
def couplingHot (v r : ModalMode) : BuildM Bool := do
  pure ((← classifyCoupling v r) == .paired)

/-- `true` ⇒ a lens fired but the paired range cap refused the coupling: it
    renders collected at the status-quo floor. The merged seam atom's admission
    predicate is the negation of this — the refusal is stated, not certified. -/
def couplingRefused (v r : ModalMode) : BuildM Bool := do
  pure ((← classifyCoupling v r) == .refused)

/-- The PARTITIONED compose — the one `residueCompose` seam. Cold couplings take
    `residueComposeEC`'s collected shapes; hot couplings (per `couplingHot`)
    leave the Cauchy sums (BOTH halves — the forced amp's `r/(λ−ν)` term and the
    ringing amp's `a/(λ−ν)` term migrate together; the split is EXACT residue
    algebra, not an approximation) and land as fused `PairedMode` atoms.
    When NOTHING is hot the result is `residueComposeEC` VERBATIM (the same
    call), so the cold path is byte-identical to the pre-partition compiler —
    the erasure gate's discipline. -/
def residueComposePartitioned (voice reverb : Array ModalMode) :
    BuildM (Array ModalMode × Array PairedMode) := do
  if voice.isEmpty then return (#[], #[])
  let builder ← get
  let constants := sigConstTable builder.exprs
  let hotM := voice.map fun v => reverb.map fun r =>
    classifyCouplingWith constants v r == .paired
  if hotM.all (·.all (!·)) then return (← residueComposeEC voice reverb, #[])
  let isHot := fun (i q : Nat) => (hotM[i]!)[q]!
  let forced ← voice.mapIdxM fun i v => do
    let pole ← v.poleE
    let zero ← lit 0
    let Hlam ← reverb.zipIdx.foldlM (init := (zero, zero)) fun sum (r, q) => do
      if isHot i q then pure sum else
      let roomPole ← r.poleE
      let difference ← csubE pole roomPole
      let quotient ← cdivE r.ampE difference
      caddE sum quotient
    let amp ← cmulE v.ampE Hlam
    modeOfE pole amp
  let ringing ← reverb.mapIdxM fun q r => do
    let pole ← r.poleE
    let zero ← lit 0
    let coupling ← voice.zipIdx.foldlM (init := (zero, zero)) fun sum (v, i) => do
      if isHot i q then pure sum else
      let voicePole ← v.poleE
      let difference ← csubE voicePole pole
      let quotient ← cdivE v.ampE difference
      caddE sum quotient
    let product ← cmulE r.ampE coupling
    let amp ← cnegE product
    modeOfE pole amp
  let mut paired : Array PairedMode := #[]
  for (v, i) in voice.zipIdx do
    for (r, q) in reverb.zipIdx do
      if isHot i q then
        -- live-σ poles enter the paired atom CLAMPED to their declared
        -- interval, keeping the routing's range cap sound (Phase 2)
        let lam ← clampedPoleEWith constants v
        let nu ← clampedPoleEWith constants r
        let c ← cmulE v.ampE r.ampE
        paired := paired.push { lam, nu, c }
  return (forced ++ ringing, paired)

/-- Multiply a `CplxE` by a real `Sig`. -/
def scaleRealE (s : Sig) (z : CplxE) : BuildM CplxE := do
  let real ← mul s z.1
  let imag ← mul s z.2
  pure (real, imag)

/-- `(e^z − 1)/z` for a complex `z` as the Horner series `Σ_{k≥0} zᵏ/(k+1)!`, N=6
    terms — the STABLE branch for small `|z|`, where the `−1` in the direct form
    catastrophically cancels. Coeffs `1/(k+1)!`. Limit 1 at z=0 (the τ·e resonance).
    Validated: series↔direct discontinuity at the threshold is 2.5e-12. -/
def cexpm1SeriesE (z : CplxE) : BuildM CplxE := do
  let base ← litF (1.0 / 5040.0)
  let zero ← lit 0
  #[1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0,
      1.0 / 2.0, 1.0].foldlM (fun acc coefficient => do
    let coefficient ← litF coefficient
    let product ← cmulE z acc
    caddE (coefficient, zero) product) (base, zero)

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

def pairedBankCols (modes : Array PairedMode)
    (live? : Option Sig := none) : BuildM PairedBankCols := do
  let twoPi ← twoPiE
  let twoPow32 ← lit 4294967296
  let sr ← sampleRate
  let incrNuValues ← modes.mapM fun m => do
    let frequency ← div m.nu.2 twoPi
    let scaled ← mul frequency twoPow32
    div scaled sr
  let incrDiffValues ← modes.mapM fun m => do
    let difference ← sub m.lam.2 m.nu.2
    let frequency ← div difference twoPi
    let scaled ← mul frequency twoPow32
    div scaled sr
  let sigmaNuValues ← modes.mapM fun m => neg m.nu.1
  let dsValues ← modes.mapM fun m => sub m.nu.1 m.lam.1
  let wdValues ← modes.mapM fun m => sub m.lam.2 m.nu.2
  let incrNu ← arr incrNuValues
  let incrDiff ← arr incrDiffValues
  let sigmaNu ← arr sigmaNuValues
  let ds ← arr dsValues
  let wd ← arr wdValues
  let cre ← arr (modes.map fun m => m.c.1)
  let cim ← arr (modes.map fun m => m.c.2)
  pure { count := modes.size, live?, incrNu, incrDiff, sigmaNu, ds, wd, cre, cim }

def bankFoldPaired (cols : PairedBankCols)
    (body : PairedModeSym → BuildM Sig) : BuildM Sig := do
  let k ← loopIdx cols.idxId
  let incrNu ← index cols.incrNu k
  let incrDiff ← index cols.incrDiff k
  let sigmaNu ← index cols.sigmaNu k
  let ds ← index cols.ds k
  let wd ← index cols.wd k
  let cre ← index cols.cre k
  let cim ← index cols.cim k
  let contribution ← body { incrNu, incrDiff, sigmaNu, ds, wd, cre, cim }
  bankSum cols.count
    #[cols.incrNu, cols.incrDiff, cols.sigmaNu, cols.ds, cols.wd, cols.cre, cols.cim]
    contribution cols.live? cols.idxId

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
    (live? : Option Sig := none) : BuildM Sig := do
  let clkRel ← relClockQ clkInt anchorSamples
  let clkFloat ← toFloatE clkRel
  let twoPow32 ← lit 4294967296
  let secondsTimesRate ← div clkFloat twoPow32
  let sr ← sampleRate
  let dSec ← div secondsTimesRate sr
  let cols ← pairedBankCols modes live?
  let bankQ ← bankFoldPaired cols fun m => do
    let incrNu ← toIntE m.incrNu
    let phQnu ← modePhaseQFromIncr incrNu clkRel
    let incrDiff ← toIntE m.incrDiff
    let phQdf ← modePhaseQFromIncr incrDiff clkRel
    let cosQ ← fixedCosCycSig phQdf
    let cosFloat ← toFloatE cosQ
    let q30 ← lit 1073741824
    let cosDf ← div cosFloat q30
    let sinQ ← fixedSinCycSig phQdf
    let sinFloat ← toFloatE sinQ
    let sinDf ← div sinFloat q30
    let sigmaTime ← mul m.sigmaNu dSec
    let negativeSigmaTime ← neg sigmaTime
    let envNu ← expSig negativeSigmaTime
    let differenceTime ← mul m.ds dSec
    let negativeDifferenceTime ← neg differenceTime
    let envDf ← expSig negativeDifferenceTime
    let ezReal ← mul envDf cosDf
    let ezImag ← mul envDf sinDf
    let zReal ← neg differenceTime
    let zImag ← mul m.wd dSec
    let z : CplxE := (zReal, zImag)
    let zReal2 ← mul zReal zReal
    let zImag2 ← mul zImag zImag
    let zsq ← add zReal2 zImag2
    let threshold ← litF 0.01
    let big ← gt zsq threshold
    let one ← lit 1
    let zero ← lit 0
    let safeReal ← selectE big zReal one
    let safeImag ← selectE big zImag zero
    let numerator ← csubE (ezReal, ezImag) (one, zero)
    let direct ← cdivE numerator (safeReal, safeImag)
    let series ← cexpm1SeriesE z
    let cxReal ← selectE big direct.1 series.1
    let cxImag ← selectE big direct.2 series.2
    let envTime ← mul envNu dSec
    let scaled ← scaleRealE envTime (cxReal, cxImag)
    let wc ← cmulE (m.cre, m.cim) scaled
    let q28 ← lit 268435456
    let landedCre ← mul wc.1 q28
    let wCre ← toIntE landedCre
    let landedCim ← mul wc.2 q28
    let wCim ← toIntE landedCim
    let carrierCos ← fixedCosCycSig phQnu
    let real ← mul wCre carrierCos
    let carrierSin ← fixedSinCycSig phQnu
    let imag ← mul wCim carrierSin
    let difference ← sub real imag
    let twentyEight ← lit 28
    rshift difference twentyEight
  let zero ← lit 0
  let afterStrike ← gt clkRel zero
  let output ← fixedOutQ 30 bankQ
  selectE afterStrike output zero



end Tropical.EmitArrow
