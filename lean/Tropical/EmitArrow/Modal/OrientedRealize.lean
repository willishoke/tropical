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

private def samePhysicalFrequency (left right : ModalMode) : Bool :=
  left.omega == right.omega || match sigConstF? left.omega, sigConstF? right.omega with
    | some l, some r => l == r
    | _, _ => false

private def withUnitAmplitude (mode : ModalMode) : ModalMode :=
  { mode with cre := lit 1, cim := lit 0 }

/-- Terminal DD routing is allowed to be more conservative than the fixed-point
    EC/DD carrier: equal physical frequencies and accuracy-lens-near physical
    poles must stay continuous even after a live direction weight has made the
    mode amplitude non-classifiable.  The terminal uses a float carrier below,
    so it does not inherit the old fixed Q4.28 amplitude cap that required
    compile-time residue bounds. -/
private def terminalHot (left right : ModalMode) : Bool :=
  samePhysicalFrequency left right || couplingHot left right ||
    couplingHot (withUnitAmplitude left) (withUnitAmplitude right)

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

private def cauchyCols (modes : Array ModalMode) : CauchyCols where
  count := modes.size
  poleRe := Sig.arr (modes.map fun mode => mode.poleE.1)
  poleIm := Sig.arr (modes.map fun mode => mode.poleE.2)
  ampRe := Sig.arr (modes.map fun mode => mode.ampE.1)
  ampIm := Sig.arr (modes.map fun mode => mode.ampE.2)

/-- Ordered complex reduction over a mode table. Binder 1 is reserved for
    these coefficient-side loops; the terminal oscillator bank uses binder 0.
    Real and imaginary components are two routed outputs of ONE mapped body,
    so a complex reciprocal is evaluated once and both authored left folds see
    the same item order. -/
private def cauchyFold (modes : Array ModalMode)
    (body : CauchyModeSym → CplxE) : CplxE :=
  if modes.isEmpty then natE 0 else
  let cols := cauchyCols modes
  let index := Sig.loopIdx 1
  let value := body {
    pole := (Sig.index cols.poleRe index, Sig.index cols.poleIm index)
    amp := (Sig.index cols.ampRe index, Sig.index cols.ampIm index) }
  let tables := #[cols.poleRe, cols.poleIm, cols.ampRe, cols.ampIm]
  let routes := (Array.range cols.count).foldl
    (fun out _ => out.push (some 0) |>.push (some 1)) #[]
  let image := Sig.routedSum cols.count 2 routes tables #[value.1, value.2] none 1
  (Sig.index image (lit 0), Sig.index image (lit 1))

private def differenceSum (pole : CplxE) (modes : Array ModalMode) : CplxE :=
  cauchyFold modes fun mode => cdivE mode.amp (csubE pole mode.pole)

private def physicalSum (pole : CplxE) (modes : Array ModalMode) : CplxE :=
  cauchyFold modes fun mode => cdivE mode.amp (caddE pole mode.pole)

private def terminalSameSide (left right : Array ModalMode) :
    Array ModalMode × Array PairedMode := Id.run do
  -- Keep a zero-residue row for the nonempty origin even when this same-side
  -- product is empty.  The terminal collector may still add an FP/PF residue
  -- to that pole below (notably a future-only source and a reversed room).
  if left.isEmpty then
    return (right.map fun mode => modeOfE mode.poleE (natE 0), #[])
  if right.isEmpty then
    return (left.map fun mode => modeOfE mode.poleE (natE 0), #[])
  let hot := left.map fun l => right.map fun r => terminalHot l r
  let isHot := fun i j => (hot[i]!)[j]!
  let forced := left.mapIdx fun i l =>
    let cold := right.zipIdx.filterMap fun (r, j) =>
      if isHot i j then none else some r
    let transfer := differenceSum l.poleE cold
    modeOfE l.poleE (cmulE l.ampE transfer)
  let ringing := right.mapIdx fun j r =>
    let cold := left.zipIdx.filterMap fun (l, i) =>
      if isHot i j then none else some l
    let coupling := differenceSum r.poleE cold
    modeOfE r.poleE (cmulE r.ampE coupling)
  let mut paired := #[]
  for (l, i) in left.zipIdx do
    for (r, j) in right.zipIdx do
      if isHot i j then
        paired := paired.push {
          lam := clampedPoleE l
          nu := clampedPoleE r
          c := cmulE l.ampE r.ampE }
  return (forced ++ ringing, paired)

/-- Float terminal realization of stable divided differences.  This avoids the
    fixed-amplitude admission needed by `modalBankSigTableDD`, which cannot
    certify residues produced by an earlier live room.  The paired rows remain
    data: one float reduction body serves every row in authored order instead
    of meta-unrolling the complex DD expression once per coupling. -/
private def pairedSig (pairs : Array PairedMode) (clkInt anchorSamples : Sig) : Sig :=
  if pairs.isEmpty then lit 0 else
  let clkRel := relClockQ clkInt anchorSamples
  let dSec := div (div (toFloatE clkRel) (lit 4294967296)) .sampleRate
  let value := bankFoldPaired (pairedBankCols pairs) fun pair =>
    let delta : CplxE := (neg pair.ds, pair.wd)
    let z : CplxE := (mul delta.1 dSec, mul delta.2 dSec)
    let zsq := add (mul z.1 z.1) (mul z.2 z.2)
    let directLane := gt zsq (litF 0.01)
    let zsafe : CplxE :=
      (selectE directLane z.1 (lit 1), selectE directLane z.2 (lit 0))
    let ezr := expSig z.1
    let ez : CplxE := (mul ezr (cosSig z.2), mul ezr (sinSig z.2))
    let direct := cdivE (csubE ez (natE 1)) zsafe
    let series := cexpm1SeriesE z
    let cx : CplxE :=
      (selectE directLane direct.1 series.1,
       selectE directLane direct.2 series.2)
    let secular := cmulE (pair.cre, pair.cim) (scaleRealE dSec cx)
    let env := expSig (neg (mul pair.sigmaNu dSec))
    -- `modePhaseQ` is a Q0.32 cycle word, not a radian-valued float.  Keep the
    -- same integer-reduced rotator used by the incumbent DD realization; feeding
    -- that word to `cosSig`/`sinSig` would rotate complex modes incorrectly while
    -- accidentally remaining invisible for the real-pole (omega=0) case.
    let phaseQ := modePhaseQFromIncr (toIntE pair.incrNu) clkRel
    let carrierCos := div (toFloatE (fixedCosCycSig phaseQ)) (lit 1073741824)
    let carrierSin := div (toFloatE (fixedSinCycSig phaseQ)) (lit 1073741824)
    let carrier : CplxE := (mul env carrierCos, mul env carrierSin)
    (cmulE secular carrier).1
  selectE (gt clkRel (lit 0)) value (lit 0)

/-- Read both strict half-axis banks and supply the continuous mixed-orientation
    convolution value at the strike itself. -/
def Bank.realizeSig (bank : Bank) (clkInt anchorSamples : Sig)
    (count? : Option Sig := none) : Sig :=
  let anchorQ := toIntE (mul anchorSamples (lit 4294967296))
  let mirroredClock := sub (mul (lit 2) anchorQ) clkInt
  let futureBanked := bankIsUniform bank.future &&
    (count?.isSome || banksTableEnabled)
  let pastBanked := bankIsUniform bank.past && banksTableEnabled
  let future := if futureBanked then
      modalBankSigTable bank.future clkInt anchorSamples count?
    else modalBankSig bank.future clkInt anchorSamples
  let past := if pastBanked then
      modalBankSigTable bank.past mirroredClock anchorSamples none
    else modalBankSig bank.past mirroredClock anchorSamples
  let sides := add
    future
    past
  let atStrike := Sig.binary .eq (relClockQ clkInt anchorSamples) (lit 0)
  selectE atStrike bank.atZero.1 sides

/-- Stable terminal read for hot same-side couplings.  A past paired atom is
    the exact clock mirror of the existing causal divided-difference carrier. -/
def TerminalBank.realizeSig (terminal : TerminalBank)
    (clkInt anchorSamples : Sig) (count? : Option Sig := none) : Sig :=
  let future := pairedSig terminal.futurePaired clkInt anchorSamples
  let anchorQ := toIntE (mul anchorSamples (lit 4294967296))
  let mirroredClock := sub (mul (lit 2) anchorQ) clkInt
  let past := pairedSig terminal.pastPaired mirroredClock anchorSamples
  add (terminal.bank.realizeSig clkInt anchorSamples count?) (add future past)

/-- Add the exact mixed-orientation residue contribution to a mode already
    carrying its cold same-side residue.  For degree-zero `F(λ,a) * P(μ,b)`,
    both surviving pole residues receive `-a*b/(λ+μ)`.  Collecting those
    contributions on the existing poles avoids materializing two modes per
    Cartesian pair while preserving the bilateral partial fraction. -/
private def mixedResidue (original : ModalMode)
    (opposite : Array ModalMode) : CplxE :=
  cnegE (cmulE original.ampE (physicalSum original.poleE opposite))

private def addMixedResidue (sameSide : ModalMode)
    (contribution : CplxE) : ModalMode :=
  { sameSide with
    cre := add sameSide.cre contribution.1
    cim := add sameSide.cim contribution.2 }

/-- Compose the last room with EC/DD stability on both same-side arms.  The
    degree-zero mixed `FP`/`PF` pairs are collected onto their existing poles;
    this is the exact partial fraction, but keeps the terminal carrier linear
    rather than Cartesian in bank size.  This is intentionally terminal:
    paired atoms are not yet a generally composable modal source for a later
    room or gauge. -/
def Bank.convolveKernelTerminal (input : Bank) (room : Array ModalMode)
    (direction : Sig) : TerminalBank :=
  let kernel := Bank.kernel room direction
  let degreeZero := (input.future ++ input.past ++ kernel.future ++ kernel.past).all
    fun mode => mode.deg == 0
  if !degreeZero then
    -- The float paired carrier below is presently degree-zero.  General
    -- exponential-polynomial inputs therefore stay on the exact algebraic
    -- path instead of silently losing their polynomial factors.  Its
    -- classifier recognizes structural coincidence; a generalized live DD
    -- carrier remains the named boundary for runtime-equal higher degrees.
    TerminalBank.ofBank
      (input.convolveKernel room direction syntacticSameSideClassifier)
  else
    let (future, futurePaired) := terminalSameSide input.future kernel.future
    let (past, pastPaired) := terminalSameSide input.past kernel.past
    -- `terminalSameSide` retains authored input modes followed by authored
    -- kernel modes.  Add each opposite-axis Cauchy gain to that matching row;
    -- hot same-side pairs remain excluded there and occur once in the DD bank.
    let futureInput := (future.extract 0 input.future.size).zip input.future
      |>.map fun (mode, original) =>
        let contribution := mixedResidue original kernel.past
        (addMixedResidue mode contribution, contribution)
    let futureKernel := (future.extract input.future.size future.size).zip kernel.future
      |>.map fun (mode, original) =>
        let contribution := mixedResidue original input.past
        (addMixedResidue mode contribution, contribution)
    let pastInput := (past.extract 0 input.past.size).zip input.past
      |>.map fun (mode, original) =>
        addMixedResidue mode (mixedResidue original kernel.future)
    let pastKernel := (past.extract input.past.size past.size).zip kernel.past
      |>.map fun (mode, original) =>
        addMixedResidue mode (mixedResidue original input.future)
    -- Either origin side contains the exact mixed value at the strike.  Reuse
    -- the future-origin contribution nodes already feeding those residues;
    -- a separate Cartesian seam fold would duplicate the same live quotients.
    let atZero := (futureInput ++ futureKernel).foldl (fun total row =>
      caddE total row.2) (natE 0)
    { bank := {
        future := (futureInput ++ futureKernel).map (fun row => row.1)
        past := pastInput ++ pastKernel
        atZero }
      futurePaired
      pastPaired }

/-- The bilateral transfer value at `s = i·omega`.  Future atoms contribute
    `A·p!/(s-λ)^(p+1)`; mirrored past atoms contribute
    `A·p!·(-1)^(p+1)/(s+λ)^(p+1)`. -/
def Bank.transferAt (bank : Bank) (omega : Sig) : CplxE :=
  let s : CplxE := (lit 0, omega)
  let future := bank.future.foldl (fun total mode =>
    let denominator := cpowE (csubE s mode.poleE) (mode.deg + 1)
    caddE total (cdivE (cmulE mode.ampE (natE (factorial mode.deg))) denominator))
    (natE 0)
  bank.past.foldl (fun total mode =>
    let numerator := cmulE mode.ampE (natE (factorial mode.deg))
    let numerator := if (mode.deg + 1) % 2 == 1 then cnegE numerator else numerator
    let denominator := cpowE (caddE s mode.poleE) (mode.deg + 1)
    caddE total (cdivE numerator denominator)) future

/-- One authored gauge scalar for the complete current modal value.  Both arms
    are sampled in one p=8 norm and receive one scalar; they are never normalized
    independently.  Unlike the older causal-only adapter this expression is the
    current static universe, so it deliberately does not settle live controls to
    a target value before measuring them. -/
def Bank.gaugeScale (g : Sig) (bank : Bank) : Sig :=
  let candidates := bank.future.map (fun mode => mode.omega) ++
    bank.past.map (fun mode => neg mode.omega)
  -- Probe identity belongs to the transfer function, not its partial-fraction
  -- spelling: splitting one atom into two half-amplitude atoms must not double
  -- its contribution to the p-norm's outer sample grid.
  let frequencies := candidates.foldl (fun probes frequency =>
    if probes.contains frequency then probes else probes.push frequency) #[]
  let energy8 := frequencies.foldl (fun total omega =>
    let h := bank.transferAt omega
    let h2 := add (mul h.1 h.1) (mul h.2 h.2)
    add total (mul (mul h2 h2) (mul h2 h2))) (lit 0)
  expSig (mul (mul (neg g) (lit 125 3))
    (logSig (clampE energy8 (lit 1 30) (lit (10^30)))))

/-- Gauge a complete oriented modal value in place.  The exact strike seam is
    scaled with the same value as both exponential-polynomial arms. -/
def Bank.gauge (bank : Bank) (g : Sig) : Bank :=
  if bank.future.isEmpty && bank.past.isEmpty then bank else
  let scale : CplxE := (bank.gaugeScale g, lit 0)
  { future := bank.future.map (scaleModeAmp scale)
    past := bank.past.map (scaleModeAmp scale)
    atZero := cmulE scale bank.atZero }

/-- Sway changes only this room kernel's physical damping before convolution.
    Pitch, the source prefix, and every other room remain untouched. -/
def swayKernel (modes : Array ModalMode) (sway rate responseClock : Sig) : Array ModalMode :=
  let phase := phasorPhaseSig rate (lit 0) responseClock
  let scale := add (lit 1) (mul sway (sinSig (mul twoPiE phase)))
  modes.map fun mode => { mode with sigma := mul mode.sigma scale }

end Tropical.EmitArrow.Oriented
