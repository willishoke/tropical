import Tropical.EmitArrow.Modal.Live

/-!
# EmitArrow.Modal.BatchPlan

Measured live-beta capacity planner for the non-public timed-modal terminal
spike.  It does not change the production bloom classifier.  Existing pairs
that fit its 300-term contract retain that incumbent plan; only
beta-max depth exclusions are tested against a movable radial seam and a
guarded 384-term candidate capacity.

The movable result is deliberately evidence, not proof.  Radial series-depth
monotonicity is checked on fixed knots, sigma is checked on a fixed grid with
bounded adversarial refinement, and numerical equality of the series/CF lanes
remains a separate oracle obligation.
-/

namespace Tropical.EmitArrow

open Tropical.Exact (CplxD CplxDI DyadicI)

/-- Why the measured spike planner could not cover one live parameter box. -/
inductive TimedBloomBetaRefusal where
  | incumbent (reason : BloomExclusionReason)
  | nonMonotoneDepth
  | excludedDepthBox
  | depthBoxUnresolved
deriving Inhabited, DecidableEq

def TimedBloomBetaRefusal.label : TimedBloomBetaRefusal → String
  | .incumbent reason => reason.label
  | .nonMonotoneDepth => "nonMonotoneDepth"
  | .excludedDepthBox => "excludedDepthBox"
  | .depthBoxUnresolved => "depthBoxUnresolved"

/-- One pair's incumbent or moved-seam capacity result.  A zero radial
    numerator means the production classifier's existing seam is retained. -/
structure TimedBloomBetaPairPlan where
  incumbent       : Bool
  monotoneAudit   : Bool
  radialNumerator : Nat
  radialDenominator : Nat
  nDepth          : Nat
  kDepth          : Nat
  sigmaSamples    : Nat
  refinementRounds : Nat
deriving Inhabited, Repr

def TimedBloomBetaPairPlan.radialFraction (p : TimedBloomBetaPairPlan) : Float :=
  if p.incumbent || p.radialDenominator == 0 then 0.0
  else p.radialNumerator.toFloat / p.radialDenominator.toFloat

private structure TimedBloomDepthMeasure where
  nDepth : Nat
  kDepth : Nat
deriving Inhabited

private def TimedBloomDepthMeasure.worst (m : TimedBloomDepthMeasure) : Nat :=
  max m.nDepth m.kDepth

private def pushUniqueFloat (xs : Array Float) (x : Float) : Array Float :=
  if xs.contains x then xs else xs.push x

private def orderedSigmaSamples (mu : CplxB) (sigLo sigHi g : Float)
    (knots : Nat) (critical : Bool) (extra : Array Float := #[]) : Array Float := Id.run do
  let lo := min sigLo sigHi
  let hi := max sigLo sigHi
  let mut out : Array Float := #[]
  out := pushUniqueFloat out lo
  out := pushUniqueFloat out ((lo + hi) * 0.5)
  out := pushUniqueFloat out hi
  if knots > 0 then
    for q in [0:knots + 1] do
      out := pushUniqueFloat out (lo + (hi - lo) * q.toFloat / knots.toFloat)
  if critical then
    -- Re(a) = (-sigma - Re(mu))/g.  Sampling sigma = g*j - Re(mu)
    -- hits every negative-integer denominator line Re(a)=-j in the box.
    for j in [1:385] do
      let sigma := g * j.toFloat - mu.re
      if sigma >= lo && sigma <= hi then out := pushUniqueFloat out sigma
  for sigma in extra do
    if sigma >= lo && sigma <= hi then out := pushUniqueFloat out sigma
  return out.qsort (fun a b => decide (a < b))

private def cappedBloomDepth (depth : Nat) (rawLimit : Nat) : Nat :=
  if depth > rawLimit then rawLimit + 9 else depth + 8

private def timedBloomDepthAt (mu : CplxB) (nuOmega sigma g : Float)
    (kappaMax : CplxD) (radialNumerator radialDenominator rawLimit : Nat) :
    TimedBloomDepthMeasure :=
  let nu : CplxB := ⟨-sigma, nuOmega⟩
  let a := (nu.sub mu).scale (1.0 / g)
  -- m/2^16 is exactly representable as Float and enters the dyadic carrier
  -- exactly; the recurrence itself never sees a libm value.
  let t := Dyadic.ofFloat (radialNumerator.toFloat / radialDenominator.toFloat)
  let z := CplxD.scale t kappaMax
  let nRaw := bloomM1DepthD a.toPoint z bloomM1TolD (rawLimit + 1)
  let kRaw := bloomCFDepthD a.toPoint z bloomCFTolD (rawLimit + 1)
  { nDepth := cappedBloomDepth nRaw rawLimit
    kDepth := cappedBloomDepth kRaw rawLimit }

private def timedBloomWorstDepth (mu : CplxB) (nuOmega g : Float)
    (kappaMax : CplxD) (samples : Array Float)
    (radialNumerator radialDenominator rawLimit : Nat) : TimedBloomDepthMeasure :=
  samples.foldl (init := { nDepth := 0, kDepth := 0 }) fun acc sigma =>
    let d := timedBloomDepthAt mu nuOmega sigma g kappaMax
      radialNumerator radialDenominator rawLimit
    { nDepth := max acc.nDepth d.nDepth, kDepth := max acc.kDepth d.kDepth }

private def timedBloomSeriesMonotoneAudit (mu : CplxB) (nuOmega g : Float)
    (kappaMax : CplxD) (samples : Array Float) (q rawLimit : Nat) : Bool := Id.run do
  let radial := #[1, q / 8, 2 * q / 8, 3 * q / 8, 4 * q / 8,
    5 * q / 8, 6 * q / 8, 7 * q / 8, q]
  for sigma in samples do
    let mut first := true
    let mut prevN := 0
    for m in radial do
      let d := timedBloomDepthAt mu nuOmega sigma g kappaMax m q rawLimit
      if !first && d.nDepth < prevN then return false
      first := false
      prevN := d.nDepth
  return true

private def chooseTimedBloomRadial (mu : CplxB) (nuOmega g : Float)
    (kappaMax : CplxD) (samples : Array Float) (q rawLimit : Nat) :
    Option (Nat × TimedBloomDepthMeasure) := Id.run do
  let atLo := timedBloomWorstDepth mu nuOmega g kappaMax samples 1 q rawLimit
  let atHi := timedBloomWorstDepth mu nuOmega g kappaMax samples q q rawLimit
  if atLo.nDepth > rawLimit + 8 then return none
  if atHi.nDepth <= rawLimit + 8 then
    return some (q, { atHi with kDepth := 0 })
  -- Stay as far toward kappaMax as the series capacity permits.  Unlike a
  -- depth-minimax search, this cannot walk the CF arbitrarily toward z=0 merely
  -- because Lentz's stopping counter happens to become small there.
  let mut lo := 1
  let mut hi := q
  for _ in [0:16] do
    if hi - lo > 1 then
      let mid := (lo + hi) / 2
      let d := timedBloomWorstDepth mu nuOmega g kappaMax samples mid q rawLimit
      if d.nDepth <= rawLimit + 8 then lo := mid else hi := mid
  let best := timedBloomWorstDepth mu nuOmega g kappaMax samples lo q rawLimit
  if best.kDepth <= rawLimit + 8 then return some (lo, best) else return none

private def worstTimedBloomSigma (mu : CplxB) (nuOmega g : Float)
    (kappaMax : CplxD) (samples : Array Float) (m q rawLimit : Nat) :
    Float × TimedBloomDepthMeasure := Id.run do
  let mut worstSigma := samples[0]!
  let mut worst := timedBloomDepthAt mu nuOmega worstSigma g kappaMax m q rawLimit
  for sigma in samples do
    let d := timedBloomDepthAt mu nuOmega sigma g kappaMax m q rawLimit
    if d.worst > worst.worst then
      worstSigma := sigma
      worst := d
  return (worstSigma, worst)

/-- Plan one source pole×room carrier over a live beta/sigma box.  The public
    defaults are intentionally the spike's measured hypothesis: capacity 384,
    exact Q16 radial search, 33 sigma knots, and at most four refinements.

    This function is not connected to Patch lowering or vocabulary admission.
    An `.ok` moved plan means only that deterministic capped stopping counts fit
    on the measured grid under the sampled monotonicity guard. -/
def planTimedBloomBetaPair (mu : CplxB) (nuOmega sigLo sigHi betaMax scale g : Float)
    (capacity : Nat := 384) (sigmaKnots : Nat := 32) :
    Except TimedBloomBetaRefusal TimedBloomBetaPairPlan := Id.run do
  if g <= 0.0 || betaMax < 0.0 || scale < 0.0 || capacity < 9 then
    return .error (.incumbent .coefficientMaterialization)
  let Bmax := betaMax * scale / g
  match classifyBloomPairLiveChecked mu nuOmega sigLo sigHi Bmax g with
  | .ok plan =>
    return .ok {
      incumbent := true
      monotoneAudit := true
      radialNumerator := 0
      radialDenominator := 0
      nDepth := plan.nDepth
      kDepth := plan.kDepth
      sigmaSamples := 0
      refinementRounds := 0 }
  | .error reason =>
    if reason != .excludedDepth then return .error (.incumbent reason)
  let q : Nat := 65536
  let rawLimit := capacity - 8
  let kappaMax := (mu.scale Bmax).toPoint
  let mut extra : Array Float := #[]
  for round in [0:5] do
    let searchSamples := orderedSigmaSamples mu sigLo sigHi g 0 false extra
    let monotone := timedBloomSeriesMonotoneAudit mu nuOmega g kappaMax searchSamples q rawLimit
    if !monotone then return .error .nonMonotoneDepth
    let some (m, _) := chooseTimedBloomRadial mu nuOmega g kappaMax searchSamples q rawLimit
      | return .error .excludedDepthBox
    let verifySamples := orderedSigmaSamples mu sigLo sigHi g sigmaKnots true extra
    let (worstSigma, measured0) := worstTimedBloomSigma mu nuOmega g kappaMax
      verifySamples m q rawLimit
    -- At rho=kappaMax every beta in the box begins on or below the seam, so the
    -- CF lane is unreachable and need not consume capacity.
    let measured := if m == q then { measured0 with kDepth := 0 } else measured0
    if measured.worst <= capacity then
      return .ok {
        incumbent := false
        monotoneAudit := monotone
        radialNumerator := m
        radialDenominator := q
        nDepth := measured.nDepth
        kDepth := measured.kDepth
        sigmaSamples := verifySamples.size
        refinementRounds := round }
    if round == 4 then return .error .depthBoxUnresolved
    extra := pushUniqueFloat extra worstSigma
  return .error .depthBoxUnresolved

/-- Materialize one fixed-beta/fixed-sigma non-coincident pair at a measured
    moved seam.  This is an oracle instrument for the terminal spike, not a
    production composer.  The seam radius is fixed from the beta-max box; when
    the current beta begins below it, the κ-side constant is computed from the
    convergent series directly instead of evaluating CF(κ) outside its qualified
    lane.  Topology/depth remain the box plan's fixed values. -/
def materializeTimedBloomBetaPair? (mu nu : CplxB) (c : CplxE)
    (beta betaMax scale g : Float) (plan : TimedBloomBetaPairPlan) : Option BloomPair := do
  if plan.incumbent || plan.radialDenominator == 0 || g <= 0.0 then none else
  let B := beta * scale / g
  let Bmax := betaMax * scale / g
  let aC := (nu.sub mu).scale (1.0 / g)
  let aD := aC.toExact
  let kappa := mu.scale B
  let kD := kappa.toExact
  let kMaxD := (mu.scale Bmax).toExact
  let tI := DyadicI.div (DyadicI.ofNat plan.radialNumerator)
    (DyadicI.ofNat plan.radialDenominator)
  let rhoD := CplxDI.scale tI kMaxD
  let absK := CplxDI.abs kD
  let absRho := CplxDI.abs rhoD
  if !absRho.ok || DyadicI.straddlesZero absRho then none else
  let kZero := absK.ok && absK.lo.isZero && absK.hi.isZero
  let crossing := plan.kDepth > 0 && !kZero && DyadicI.certGt absK absRho
  let gD := DyadicI.ofFloat g
  let invNuMuD := CplxDI.div CplxDI.one (CplxDI.sub nu.toExact mu.toExact)
  let invAD := (Array.range plan.nDepth).map fun k =>
    CplxDI.div CplxDI.one (CplxDI.add aD (CplxDI.ofNat (k + 1)))
  let cfBD := (Array.range (plan.kDepth + 1)).map fun j =>
    CplxDI.mkI (DyadicI.sub (DyadicI.ofNat (2 * j + 1)) aD.re) (DyadicI.neg aD.im)
  let cfND := (Array.range plan.kDepth).map fun j =>
    let jf := CplxDI.ofNat (j + 1)
    CplxDI.mul jf (CplxDI.sub jf aD)
  let k1 := if crossing then
      let (cfKp, _) := bloomCFPointD aC.toPoint kappa.toPoint bloomCFTolD
      let cfOverG := CplxDI.scale (DyadicI.inv gD) cfKp.asPointI
      (CplxDI.sub (bloomGammaStarD aD kD gD) cfOverG, CplxDI.neg cfOverG)
    else
      let nK := if kZero then 0 else
        bloomM1DepthD aC.toPoint kappa.toPoint bloomM1TolD (plan.nDepth + 1)
      let mK := if kZero then CplxDI.one else bloomM1D aD kD nK
      (CplxDI.mul mK invNuMuD, CplxDI.zero)
  let dSwitchD := if kZero || plan.kDepth == 0 then DyadicI.ofFloat (-1.0e30)
    else DyadicI.div (DyadicI.log (DyadicI.div absK absRho)) gD
  let kappaE ← cplxLitD? kD
  let k1SerE ← cplxLitD? k1.1
  let k1CfE ← cplxLitD? k1.2
  let fSerE ← cplxLitD? (CplxDI.neg invNuMuD)
  let invAE ← invAD.mapM cplxLitD?
  let cfBE ← cfBD.mapM cplxLitD?
  let cfNE ← cfND.mapM cplxLitD?
  pure {
    muSigma := litF (-mu.re), muOmega := litF mu.im
    nuSigma := litF (-nu.re), nuOmega := litF nu.im
    bloomB := B, gRate := g, c, kappa := kappaE
    k1Ser := k1SerE, k1Cf := k1CfE, fSer := fSerE
    dSwitch := litF (DyadicI.toFloat dSwitchD)
    invA := invAE, cfB := cfBE, cfN := cfNE }

end Tropical.EmitArrow
