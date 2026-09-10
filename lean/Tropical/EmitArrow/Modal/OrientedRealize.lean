import Tropical.EmitArrow.Modal.Oriented
import Tropical.EmitArrow.Modal.Realize

/-!
# Realization of oriented modal banks

The algebraic carrier stays two-sided until the actual `Modal → Sig` seam.
This module is intentionally small: room composition remains in `Oriented`,
while this file owns the one terminal read and the whole-value gauge adapter.
-/

namespace Tropical.EmitArrow.Oriented

open Tropical.Ir
open Tropical.EmitArrow

/-- Terminal extension of the composable bank.  Divided-difference atoms are
    introduced only for the last same-side convolution, where they can be
    rendered stably without pretending they can already cross another modal
    stage.  The composable carrier remains `Bank`; this type makes the current
    closure boundary explicit. -/
structure TerminalBank where
  bank : Bank
  futurePaired : Array PairedMode := #[]
  pastPaired : Array PairedMode := #[]

def TerminalBank.ofBank (bank : Bank) : TerminalBank := { bank }

private def samePhysicalFrequency (left right : ModalMode) : BuildM Bool := do
  if left.omega == right.omega then return true
  match ← sigConstF? left.omega, ← sigConstF? right.omega with
  | some l, some r => pure (l == r)
  | _, _ => pure false

private def withUnitAmplitude (mode : ModalMode) : BuildM ModalMode := do
  let one ← lit 1
  let zero ← lit 0
  pure { mode with cre := one, cim := zero }

/-- Terminal DD routing is allowed to be more conservative than the fixed-point
    EC/DD carrier: equal physical frequencies and accuracy-lens-near physical
    poles must stay continuous even after a live direction weight has made the
    mode amplitude non-classifiable.  The terminal uses a float carrier below,
    so it does not inherit the old fixed Q4.28 amplitude cap that required
    compile-time residue bounds. -/
private def terminalHot (left right : ModalMode) : BuildM Bool := do
  if ← samePhysicalFrequency left right then return true
  if ← couplingHot left right then return true
  let unitLeft ← withUnitAmplitude left
  let unitRight ← withUnitAmplitude right
  couplingHot unitLeft unitRight

/-- Columns for one live Cauchy reduction.  Keeping physical pole coordinates
    here avoids reconstructing `omega` from the oscillator increment. -/
private structure CauchyCols where
  count : Nat
  poleRe : Sig
  poleIm : Sig
  ampRe : Sig
  ampIm : Sig

private structure CauchyModeSym where
  pole : CplxE
  amp : CplxE

private def cauchyCols (modes : Array ModalMode) : BuildM CauchyCols := do
  let poles ← modes.mapM (·.poleE)
  let amps := modes.map (·.ampE)
  let poleRe ← arr (poles.map (·.1))
  let poleIm ← arr (poles.map (·.2))
  let ampRe ← arr (amps.map (·.1))
  let ampIm ← arr (amps.map (·.2))
  pure { count := modes.size, poleRe, poleIm, ampRe, ampIm }

/-- Ordered complex reduction over a mode table. Binder 1 is reserved for
    these coefficient-side loops; the terminal oscillator bank uses binder 0.

    Emitted as TWO ordinary `bankSum` reductions (real, imaginary) over the
    same tables, not one two-output `routedSum`. The routed form shared the
    complex reciprocal between components, but `Stage0.placementFromStages`
    masks every routed span s1 categorically — so a Cauchy sum whose value is
    pure coefficient math (poles and amps from params, no τ anywhere) was
    re-evaluated every sample, and everything downstream of its image (the
    composed residues, hence the bank's coefficient columns) was pinned in the
    audio kernel with it. As `ReduceBegin`/`ReduceEnd` units the folds are
    whole-region hoistable by the EXISTING `tryRegion` (the
    `banks-region-hoist` precedent), the residue chain cascades to s0 behind
    them, and the mode tables hoist as banks-as-data columns.

    Value-identical to the routed form: both loops visit the tables in the
    same order, each item's component is the same expression the routed body
    produced, and the additive fold is the same left-to-right accumulation —
    so each component's sum is bit-identical. The reciprocal is evaluated once
    per component loop instead of once for both; at s0 that is once per
    control write, not once per sample. -/
private def cauchyFold (modes : Array ModalMode)
    (body : CauchyModeSym → BuildM CplxE) : BuildM CplxE := do
  if modes.isEmpty then return ← natE 0
  let cols ← cauchyCols modes
  let tableIndex ← loopIdx 1
  let poleRe ← index cols.poleRe tableIndex
  let poleIm ← index cols.poleIm tableIndex
  let ampRe ← index cols.ampRe tableIndex
  let ampIm ← index cols.ampIm tableIndex
  let value ← body {
    pole := (poleRe, poleIm)
    amp := (ampRe, ampIm) }
  let tables := #[cols.poleRe, cols.poleIm, cols.ampRe, cols.ampIm]
  let real ← bankSum cols.count tables value.1 none 1
  let imag ← bankSum cols.count tables value.2 none 1
  pure (real, imag)

private def differenceSum (pole : CplxE) (modes : Array ModalMode) : BuildM CplxE :=
  cauchyFold modes fun mode => do
    let denominator ← csubE pole mode.pole
    cdivE mode.amp denominator

private def physicalSum (pole : CplxE) (modes : Array ModalMode) : BuildM CplxE :=
  cauchyFold modes fun mode => do
    let denominator ← caddE pole mode.pole
    cdivE mode.amp denominator

private def terminalSameSide (left right : Array ModalMode) :
    BuildM (Array ModalMode × Array PairedMode) := do
  -- Keep a zero-residue row for the nonempty origin even when this same-side
  -- product is empty.  The terminal collector may still add an FP/PF residue
  -- to that pole below (notably a future-only source and a reversed room).
  if left.isEmpty then
    let modes ← right.mapM fun mode => do
      let pole ← mode.poleE
      let zero ← natE 0
      modeOfE pole zero
    return (modes, #[])
  if right.isEmpty then
    let modes ← left.mapM fun mode => do
      let pole ← mode.poleE
      let zero ← natE 0
      modeOfE pole zero
    return (modes, #[])
  let hot ← left.mapM fun l => right.mapM fun r => terminalHot l r
  let isHot := fun i j => (hot[i]!)[j]!
  let forced ← left.zipIdx.mapM fun (l, i) => do
    let cold := right.zipIdx.filterMap fun (r, j) =>
      if isHot i j then none else some r
    let pole ← l.poleE
    let transfer ← differenceSum pole cold
    let amplitude ← cmulE l.ampE transfer
    modeOfE pole amplitude
  let ringing ← right.zipIdx.mapM fun (r, j) => do
    let cold := left.zipIdx.filterMap fun (l, i) =>
      if isHot i j then none else some l
    let pole ← r.poleE
    let coupling ← differenceSum pole cold
    let amplitude ← cmulE r.ampE coupling
    modeOfE pole amplitude
  let mut paired := #[]
  for (l, i) in left.zipIdx do
    for (r, j) in right.zipIdx do
      if isHot i j then
        let lam ← clampedPoleE l
        let nu ← clampedPoleE r
        let c ← cmulE l.ampE r.ampE
        paired := paired.push {
          lam, nu, c }
  return (forced ++ ringing, paired)

/-- Float terminal realization of stable divided differences.  This avoids the
    fixed-amplitude admission needed by `modalBankSigTableDD`, which cannot
    certify residues produced by an earlier live room.  The paired rows remain
    data: one float reduction body serves every row in authored order instead
    of meta-unrolling the complex DD expression once per coupling. -/
private def pairedSig (pairs : Array PairedMode) (clkInt anchorSamples : Sig) : BuildM Sig := do
  if pairs.isEmpty then return ← lit 0
  let clkRel ← relClockQ clkInt anchorSamples
  let clkFloat ← toFloatE clkRel
  let twoPow32 ← lit 4294967296
  let secondsTimesRate ← div clkFloat twoPow32
  let sr ← sampleRate
  let dSec ← div secondsTimesRate sr
  let cols ← pairedBankCols pairs
  let value ← bankFoldPaired cols fun pair => do
    let deltaReal ← neg pair.ds
    let delta : CplxE := (deltaReal, pair.wd)
    let zReal ← mul delta.1 dSec
    let zImag ← mul delta.2 dSec
    let z : CplxE := (zReal, zImag)
    let zrealSq ← mul z.1 z.1
    let zimagSq ← mul z.2 z.2
    let zsq ← add zrealSq zimagSq
    let threshold ← litF 0.01
    let directLane ← gt zsq threshold
    let oneReal ← lit 1
    let zeroImag ← lit 0
    let zsafeReal ← selectE directLane z.1 oneReal
    let zsafeImag ← selectE directLane z.2 zeroImag
    let zsafe : CplxE := (zsafeReal, zsafeImag)
    let ezr ← expSig z.1
    let cosine ← cosSig z.2
    let ezReal ← mul ezr cosine
    let sine ← sinSig z.2
    let ezImag ← mul ezr sine
    let ez : CplxE := (ezReal, ezImag)
    let one ← natE 1
    let numerator ← csubE ez one
    let direct ← cdivE numerator zsafe
    let series ← cexpm1SeriesE z
    let cxReal ← selectE directLane direct.1 series.1
    let cxImag ← selectE directLane direct.2 series.2
    let cx : CplxE := (cxReal, cxImag)
    let scaledCx ← scaleRealE dSec cx
    let secular ← cmulE (pair.cre, pair.cim) scaledCx
    let sigmaTime ← mul pair.sigmaNu dSec
    let negativeSigmaTime ← neg sigmaTime
    let env ← expSig negativeSigmaTime
    -- `modePhaseQ` is a Q0.32 cycle word, not a radian-valued float.  Keep the
    -- same integer-reduced rotator used by the incumbent DD realization; feeding
    -- that word to `cosSig`/`sinSig` would rotate complex modes incorrectly while
    -- accidentally remaining invisible for the real-pole (omega=0) case.
    let increment ← toIntE pair.incrNu
    let phaseQ ← modePhaseQFromIncr increment clkRel
    let fixedCos ← fixedCosCycSig phaseQ
    let floatCos ← toFloatE fixedCos
    let scale ← lit 1073741824
    let carrierCos ← div floatCos scale
    let fixedSin ← fixedSinCycSig phaseQ
    let floatSin ← toFloatE fixedSin
    let carrierSin ← div floatSin scale
    let carrierReal ← mul env carrierCos
    let carrierImag ← mul env carrierSin
    let carrier : CplxE := (carrierReal, carrierImag)
    let product ← cmulE secular carrier
    pure product.1
  let zero ← lit 0
  let afterStrike ← gt clkRel zero
  selectE afterStrike value zero

/-- Settle a bank's coefficient plane to its control targets: a glide at rest
    or mid-ramp collapses to its `#v1` target (`settleSignals` — the semantics
    `gaugeScale` committed to first: settled knobs are coefficient-time).
    Every glide-disciplined knob is formally a function of the clock even at
    rest (the smoothstep saturates in VALUE but not in EXPRESSION), so without
    this the entire composed coefficient plane — pole derivations, Cauchy
    sums, residues — is s1 by attribute and re-evaluates every sample.
    Settled, it is s0 by construction and the placement pass hoists the whole
    composition (folds via `tryRegion`, mode tables as coefficient columns)
    into the coefficient kernel, leaving the audio kernel the bank itself.
    `none` when any coefficient carries genuine per-sample modulation (an LFO
    wired into a cutoff): the caller keeps the live expressions and the
    terminal evaluates per sample, exactly as before — the decline discipline,
    not a wrong answer. -/
def Bank.settled? (bank : Bank) : BuildM (Option Bank) := do
  let modeRoots := fun (modes : Array ModalMode) =>
    modes.flatMap fun m => #[m.sigma, m.omega, m.cre, m.cim]
  let roots := modeRoots bank.future ++ modeRoots bank.past
    ++ #[bank.atZero.1, bank.atZero.2]
  let some settled ← settleSignals roots | pure none
  let rebuild := fun (modes : Array ModalMode) (base : Nat) =>
    modes.mapIdx fun index m =>
      ({ m with
        sigma := settled[base + 4 * index]!
        omega := settled[base + 4 * index + 1]!
        cre := settled[base + 4 * index + 2]!
        cim := settled[base + 4 * index + 3]! } : ModalMode)
  let future := rebuild bank.future 0
  let past := rebuild bank.past (4 * bank.future.size)
  let atZeroBase := 4 * (bank.future.size + bank.past.size)
  pure (some { bank with
    future := future
    past := past
    atZero := (settled[atZeroBase]!, settled[atZeroBase + 1]!) })

/-- Read both strict half-axis banks and supply the continuous mixed-orientation
    convolution value at the strike itself. -/
def Bank.realizeSig (bank : Bank) (clkInt anchorSamples : Sig)
    (count? : Option Sig := none) : BuildM Sig := do
  -- The coefficient plane settles to its control targets before the terminal
  -- read; un-settleable coefficients keep the live per-sample path.
  let bank := (← bank.settled?).getD bank
  let twoPow32 ← lit 4294967296
  let anchorFixed ← mul anchorSamples twoPow32
  let anchorQ ← toIntE anchorFixed
  let two ← lit 2
  let twiceAnchor ← mul two anchorQ
  let mirroredClock ← sub twiceAnchor clkInt
  let futureBanked := bankIsUniform bank.future &&
    (count?.isSome || banksTableEnabled)
  let pastBanked := bankIsUniform bank.past && banksTableEnabled
  let future ← if futureBanked then
      modalBankSigTable bank.future clkInt anchorSamples count?
    else modalBankSig bank.future clkInt anchorSamples
  let past ← if pastBanked then
      modalBankSigTable bank.past mirroredClock anchorSamples none
    else modalBankSig bank.past mirroredClock anchorSamples
  let sides ← add future past
  let relativeClock ← relClockQ clkInt anchorSamples
  let zero ← lit 0
  let atStrike ← binary .eq relativeClock zero
  selectE atStrike bank.atZero.1 sides

/-- Stable terminal read for hot same-side couplings.  A past paired atom is
    the exact clock mirror of the existing causal divided-difference carrier. -/
def TerminalBank.realizeSig (terminal : TerminalBank)
    (clkInt anchorSamples : Sig) (count? : Option Sig := none) : BuildM Sig := do
  let future ← pairedSig terminal.futurePaired clkInt anchorSamples
  let twoPow32 ← lit 4294967296
  let anchorFixed ← mul anchorSamples twoPow32
  let anchorQ ← toIntE anchorFixed
  let two ← lit 2
  let twiceAnchor ← mul two anchorQ
  let mirroredClock ← sub twiceAnchor clkInt
  let past ← pairedSig terminal.pastPaired mirroredClock anchorSamples
  let bank ← terminal.bank.realizeSig clkInt anchorSamples count?
  let paired ← add future past
  add bank paired

/-- Add the exact mixed-orientation residue contribution to a mode already
    carrying its cold same-side residue.  For degree-zero `F(λ,a) * P(μ,b)`,
    both surviving pole residues receive `-a*b/(λ+μ)`.  Collecting those
    contributions on the existing poles avoids materializing two modes per
    Cartesian pair while preserving the bilateral partial fraction. -/
private def mixedResidue (original : ModalMode)
    (opposite : Array ModalMode) : BuildM CplxE := do
  let pole ← original.poleE
  let sum ← physicalSum pole opposite
  let product ← cmulE original.ampE sum
  cnegE product

private def addMixedResidue (sameSide : ModalMode)
    (contribution : CplxE) : BuildM ModalMode := do
  let cre ← add sameSide.cre contribution.1
  let cim ← add sameSide.cim contribution.2
  pure { sameSide with cre, cim }

/-- Compose the last room with EC/DD stability on both same-side arms.  The
    degree-zero mixed `FP`/`PF` pairs are collected onto their existing poles;
    this is the exact partial fraction, but keeps the terminal carrier linear
    rather than Cartesian in bank size.  This is intentionally terminal:
    paired atoms are not yet a generally composable modal source for a later
    room or gauge. -/
def Bank.convolveKernelTerminal (input : Bank) (room : Array ModalMode)
    (direction : Sig) : BuildM TerminalBank := do
  let kernel ← Bank.kernel room direction
  let degreeZero := (input.future ++ input.past ++ kernel.future ++ kernel.past).all
    fun mode => mode.deg == 0
  if !degreeZero then
    -- The float paired carrier below is presently degree-zero.  General
    -- exponential-polynomial inputs therefore stay on the exact algebraic
    -- path instead of silently losing their polynomial factors.  Its
    -- classifier recognizes structural coincidence; a generalized live DD
    -- carrier remains the named boundary for runtime-equal higher degrees.
    let bank ← input.convolveKernel room direction syntacticSameSideClassifier
    pure (TerminalBank.ofBank bank)
  else
    let (future, futurePaired) ← terminalSameSide input.future kernel.future
    let (past, pastPaired) ← terminalSameSide input.past kernel.past
    -- `terminalSameSide` retains authored input modes followed by authored
    -- kernel modes.  Add each opposite-axis Cauchy gain to that matching row;
    -- hot same-side pairs remain excluded there and occur once in the DD bank.
    let futureInput ← (future.extract 0 input.future.size).zip input.future
      |>.mapM fun (mode, original) => do
        let contribution ← mixedResidue original kernel.past
        let combined ← addMixedResidue mode contribution
        pure (combined, contribution)
    let futureKernel ← (future.extract input.future.size future.size).zip kernel.future
      |>.mapM fun (mode, original) => do
        let contribution ← mixedResidue original input.past
        let combined ← addMixedResidue mode contribution
        pure (combined, contribution)
    let pastInput ← (past.extract 0 input.past.size).zip input.past
      |>.mapM fun (mode, original) => do
        let contribution ← mixedResidue original kernel.future
        addMixedResidue mode contribution
    let pastKernel ← (past.extract input.past.size past.size).zip kernel.past
      |>.mapM fun (mode, original) => do
        let contribution ← mixedResidue original input.future
        addMixedResidue mode contribution
    -- Either origin side contains the exact mixed value at the strike.  Reuse
    -- the future-origin contribution nodes already feeding those residues;
    -- a separate Cartesian seam fold would duplicate the same live quotients.
    let zero ← natE 0
    let atZero ← (futureInput ++ futureKernel).foldlM (fun total row =>
      caddE total row.2) zero
    pure {
      bank := {
        future := (futureInput ++ futureKernel).map (fun row => row.1)
        past := pastInput ++ pastKernel
        atZero := atZero }
      futurePaired := futurePaired
      pastPaired := pastPaired }

/-- The bilateral transfer value at `s = i·omega`.  Future atoms contribute
    `A·p!/(s-λ)^(p+1)`; mirrored past atoms contribute
    `A·p!·(-1)^(p+1)/(s+λ)^(p+1)`. -/
def Bank.transferAt (bank : Bank) (omega : Sig) : BuildM CplxE := do
  let zeroReal ← lit 0
  let s : CplxE := (zeroReal, omega)
  let zero ← natE 0
  let future ← bank.future.foldlM (fun total mode => do
    let pole ← mode.poleE
    let difference ← csubE s pole
    let denominator ← cpowE difference (mode.deg + 1)
    let coefficient ← natE (factorial mode.deg)
    let numerator ← cmulE mode.ampE coefficient
    let quotient ← cdivE numerator denominator
    caddE total quotient) zero
  bank.past.foldlM (fun total mode => do
    let coefficient ← natE (factorial mode.deg)
    let rawNumerator ← cmulE mode.ampE coefficient
    let numerator ← if (mode.deg + 1) % 2 == 1 then cnegE rawNumerator else pure rawNumerator
    let pole ← mode.poleE
    let sum ← caddE s pole
    let denominator ← cpowE sum (mode.deg + 1)
    let quotient ← cdivE numerator denominator
    caddE total quotient) future

/-- One authored gauge scalar for the complete current modal value.  Both arms
    are sampled in one p=8 norm and receive one scalar; they are never normalized
    independently.  Unlike the older causal-only adapter this expression is the
    current static universe, so it deliberately does not settle live controls to
    a target value before measuring them. -/
def Bank.gaugeScale (g : Sig) (bank : Bank) : BuildM Sig := do
  let pastFrequencies ← bank.past.mapM fun mode => neg mode.omega
  let candidates := bank.future.map (fun mode => mode.omega) ++ pastFrequencies
  -- Probe identity belongs to the transfer function, not its partial-fraction
  -- spelling: splitting one atom into two half-amplitude atoms must not double
  -- its contribution to the p-norm's outer sample grid.
  let frequencies := candidates.foldl (fun probes frequency =>
    if probes.contains frequency then probes else probes.push frequency) #[]
  let zero ← lit 0
  let energy8 ← frequencies.foldlM (fun total omega => do
    let h ← bank.transferAt omega
    let realSquared ← mul h.1 h.1
    let imagSquared ← mul h.2 h.2
    let h2 ← add realSquared imagSquared
    let h4 ← mul h2 h2
    let h8 ← mul h4 h4
    add total h8) zero
  let lower ← lit 1 30
  let upper ← lit (10^30)
  let clamped ← clampE energy8 lower upper
  let logarithm ← logSig clamped
  let negativeG ← neg g
  let exponent ← lit 125 3
  let scaledG ← mul negativeG exponent
  let power ← mul scaledG logarithm
  expSig power

/-- Gauge a complete oriented modal value in place.  The exact strike seam is
    scaled with the same value as both exponential-polynomial arms. -/
def Bank.gauge (bank : Bank) (g : Sig) : BuildM Bank := do
  if bank.future.isEmpty && bank.past.isEmpty then return bank
  let scaleReal ← bank.gaugeScale g
  let zero ← lit 0
  let scale : CplxE := (scaleReal, zero)
  let future ← bank.future.mapM (scaleModeAmp scale)
  let past ← bank.past.mapM (scaleModeAmp scale)
  let atZero ← cmulE scale bank.atZero
  pure { future, past, atZero }

/-- Sway changes only this room kernel's physical damping before convolution.
    Pitch, the source prefix, and every other room remain untouched. -/
def swayKernel (modes : Array ModalMode) (sway rate responseClock : Sig) : BuildM (Array ModalMode) := do
  let zero ← lit 0
  let phase ← phasorPhaseSig rate zero responseClock
  let twoPi ← twoPiE
  let radians ← mul twoPi phase
  let sine ← sinSig radians
  let modulation ← mul sway sine
  let one ← lit 1
  let scale ← add one modulation
  modes.mapM fun mode => do
    let sigma ← mul mode.sigma scale
    pure { mode with sigma }

end Tropical.EmitArrow.Oriented
