import Tropical.EmitArrow.Modal.Bloom

/-!
# EmitArrow.Modal.Live

Live-pole bloom classification, lifted Γ-bridge composition, and room-chain folds.
-/

namespace Tropical.EmitArrow

open Tropical.Ir
open Tropical.Exact (DyadicI CplxD CplxDI)

/-- Why one requested voice×room bloom crossing could not be materialized.  The
    reason is data, not a silent `continue`: callers may inspect every refused
    pair and the Patch lowerer turns any nonempty set into an explicit error. -/
inductive BloomExclusionReason where
  | voiceSigmaNotConstant
  | voiceOmegaNotConstant
  | roomOmegaNotConstant
  | liveSigmaRangeMissing
  | liveSigmaNotS0
  | liveRegionUnsupported
  | excludedConditioning
  | excludedDepth
  | coefficientMaterialization
deriving Inhabited, DecidableEq

def BloomExclusionReason.label : BloomExclusionReason → String
  | .voiceSigmaNotConstant      => "voiceSigmaNotConstant"
  | .voiceOmegaNotConstant      => "voiceOmegaNotConstant"
  | .roomOmegaNotConstant       => "roomOmegaNotConstant"
  | .liveSigmaRangeMissing      => "liveSigmaRangeMissing"
  | .liveSigmaNotS0             => "liveSigmaNotS0"
  | .liveRegionUnsupported      => "liveRegionUnsupported"
  | .excludedConditioning       => "excludedConditioning"
  | .excludedDepth              => "excludedDepth"
  | .coefficientMaterialization => "coefficientMaterialization"

structure BloomPairExclusion where
  voiceIndex : Option Nat
  roomIndex  : Option Nat
  reason     : BloomExclusionReason
deriving Inhabited

structure BloomComposition where
  expectedPairs : Nat
  pairs          : Array BloomPair
  exclusions     : Array BloomPairExclusion
deriving Inhabited

def BloomComposition.isComplete (c : BloomComposition) : Bool :=
  c.exclusions.isEmpty && c.pairs.size == c.expectedPairs

/-- Compatibility lens for direct atom tests. Production lowering inspects the
    typed exclusions and reports them; this helper never turns an exclusion
    into a partial bank. -/
def BloomComposition.toOption (c : BloomComposition) : Option (Array BloomPair) :=
  if c.isComplete then some c.pairs else none

def BloomComposition.refusalSummary (c : BloomComposition) : String :=
  String.intercalate ", " (c.exclusions.toList.map fun x =>
    match x.voiceIndex, x.roomIndex with
    | some vi, some ri => s!"voice[{vi}]×room[{ri}]:{x.reason.label}"
    | _, _ => s!"global:{x.reason.label}")

/-- The two regions the LIVE classifier can emit — a two-constructor sum, so
    `bloomCompose`'s match is exhaustive BY TYPE rather than by an
    "unreachable" comment (the coincident regions, conditioning stop-line, and
    depth exclusion are not representable here: the checked classifier returns
    a typed refusal and the compatibility lens returns `none`). The baked
    classifier keeps the full six-region `SeamRegion`. -/
inductive LiveRegion where
  | serOnly
  | crossing

structure LiveBloomPairPlan where
  region : LiveRegion
  nDepth : Nat
  kDepth : Nat

/-- WS-LP: classify a live-σ pair over its whole σ interval. A supported interval
    is series-only throughout or uses the crossing lanes (including a safe
    series/crossing straddle); conditioning, depth, and coincident-region cases
    return typed refusals. Only `Re a = (σ_μ − σ_ν)/g` moves with σ_ν
    (`Im a` and `κ = μ·B` are σ_ν-independent), so the region conditions reduce
    to interval extrema of `|a + c|` along a horizontal segment — the min at
    `Re a = −c` clamped to the segment, the max at an endpoint (convexity).
    Depths are sized at the interval's worst case (both endpoints + the closest
    approach to `−1`, where `|a+1|` bottoms out), with the baked classifier's
    `zBnd` per region (κ itself for serOnly; the branch-boundary `|z| = |a+1|`
    for crossing), same `+8` guard and `≤ 300` cap. -/
def classifyBloomPairLiveChecked (mu : CplxB) (nuOmega : Float) (sigLo sigHi : Float)
    (B g : Float) : Except BloomExclusionReason LiveBloomPairPlan := Id.run do
  let kappa := mu.scale B
  let imA := (nuOmega - mu.im) / g
  -- Re a = (−σ_ν − Re μ)/g, decreasing in σ_ν
  let reLo := (-sigHi - mu.re) / g
  let reHi := (-sigLo - mu.re) / g
  -- the region conditions, on the exact carrier: this predicate decides whether
  -- a live-σ pair is emitted at all and with which lanes, so it may not depend
  -- on a platform's rounding. `absKappa`/`absAt`/`minAbsAt` are the same
  -- expressions, evaluated as enclosures; each threshold is taken in the
  -- conservative direction. Failure to certify support becomes a typed refusal.
  let imAD := DyadicI.ofFloat imA
  let reLoD := DyadicI.ofFloat reLo
  let reHiD := DyadicI.ofFloat reHi
  let absKappaD := CplxDI.abs kappa.toExact
  let absAtD := fun (re : DyadicI) (c : DyadicI) =>
    DyadicI.sqrt (DyadicI.add (DyadicI.square (DyadicI.add re c)) (DyadicI.square imAD))
  let minAbsAtD := fun (c : DyadicI) =>
    absAtD (DyadicI.max reLoD (DyadicI.min reHiD (DyadicI.neg c))) c
  if bloomExcludedConditioningLive reLo reHi imA kappa then
    return .error .excludedConditioning
  -- ¬coincident throughout: min |a| ≥ ½ (the τ·e coincidence stays baked-only)
  if !DyadicI.certGt (minAbsAtD DyadicI.zero) (DyadicI.ofFloat 0.5) then
    return .error .liveRegionUnsupported
  if absKappaD.ok && absKappaD.lo.isZero && absKappaD.hi.isZero then
    return .ok { region := .serOnly, nDepth := 0, kDepth := 0 }
  let samples := #[reLo, reHi, max reLo (min reHi (-1.0))]
  if !DyadicI.certLt (minAbsAtD DyadicI.one) absKappaD then
    -- serOnly throughout: |a+1| never falls below |κ|
    let nRaw := samples.foldl (fun m re =>
      max m (bloomM1DepthD (CplxD.ofFloats re imA) kappa.toPoint bloomM1TolD)) 0
    if nRaw + 8 > bloomDepthCap then return .error .excludedDepth
    return .ok { region := .serOnly, nDepth := nRaw + 8, kDepth := 0 }
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
  let straddles :=
    !DyadicI.certLt (DyadicI.max (absAtD reLoD DyadicI.one) (absAtD reHiD DyadicI.one))
                    absKappaD
  let mut nRaw := 0
  let mut kRaw := 0
  let kP := kappa.toPoint
  for re in samples do
    -- same discipline as the baked classifier: `zBnd` sizes an emitted array, so
    -- it is computed on the point carrier, not in f64
    let aP := CplxD.ofFloats re imA
    let ratio := (Dyadic.divRel? .down Tropical.Exact.workingPrec
                    (CplxD.abs (CplxD.add aP CplxD.one)) (CplxD.abs kP)).getD 1
    let zBnd := CplxD.scale ratio kP
    nRaw := max nRaw (bloomM1DepthD aP (if straddles then kP else zBnd) bloomM1TolD)
    kRaw := max kRaw (bloomCFDepthD aP zBnd bloomCFTolD)
  if nRaw + 8 > bloomDepthCap || kRaw + 8 > bloomDepthCap then return .error .excludedDepth
  return .ok { region := .crossing, nDepth := nRaw + 8, kDepth := kRaw + 8 }

/-- Compatibility lens for existing direct callers. New production lowering uses
    `classifyBloomPairLiveChecked` so the refusal reason remains available. -/
def classifyBloomPairLive (mu : CplxB) (nuOmega : Float) (sigLo sigHi : Float)
    (B g : Float) : Option LiveBloomPairPlan :=
  (classifyBloomPairLiveChecked mu nuOmega sigLo sigHi B g).toOption

/-- `bloomedVoice ⋙ reverb` as Γ-bridge pairs — the residue composition ACROSS a
    pitch-bloom warp (`B = β·scale/g` seconds of total clock advance, `g` the
    settle rate). Pole contract (WS-LP Phase 1): VOICE poles and reverb ω must
    fold to constants (`none` if not — the caller keeps such banks on
    `residueComposeE`'s unbloomed path); a reverb σ may be LIVE (the rt60 knob)
    when it is s0 (`sigIsS0` — raw slot reads, not glides) and carries a
    `sigmaRange` from the authoring site: the pair's constants are then built as
    s0 `CplxE` expressions of the live pole (Stage0 hoists them; rt60 turns with
    no relower), the live σ clamped to the classified interval so the region
    choice is sound by construction. `classifyBloomPairLiveChecked` admits a
    serOnly, crossing-throughout, or safe straddling interval; conditioning,
    depth, coincidence, and pole-contract failures are typed reasons that make
    the outer composition all-or-nothing. Amps stay live as always. The shared
    `bloomDepthCap` is headroom, not a tuning. The per-pair emit is
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
private def cplxLitsD? (values : Array CplxDI) : BuildM (Option (Array CplxE)) := do
  let mut result : Array CplxE := #[]
  for value in values do
    match ← cplxLitD? value with
    | none => return none
    | some literal => result := result.push literal
  pure (some result)

private def bloomComposePairs? (voice reverb : Array ModalMode) (B g : Float) :
    BuildM (Option (Array BloomPair)) := do
  let mut out : Array BloomPair := #[]
  for v in voice do
    let some vSig ← sigConstF? v.sigma | return none
    let some vOm ← sigConstF? v.omega | return none
    for r in reverb do
      let some rOm ← sigConstF? r.omega | return none
      let mu : CplxB := ⟨-vSig, vOm⟩
      let c ← cmulE v.ampE r.ampE
      match ← sigConstF? r.sigma with
      | none =>
        -- WS-LP Phase 1: a LIVE reverb σ (the rt60 pole). Admissible only when
        -- the pole read is s0 (a raw slot, not a glide — `settle` a glided pole
        -- first) and the authoring site declared its interval; the checked
        -- preflight reports either violation before this private materializer runs.
        let some (sLo, sHi) := r.sigmaRange | return none
        if !(← sigIsS0 r.sigma) then return none
        match classifyBloomPairLiveChecked mu rOm sLo sHi B g with
        | .error _ => continue
        | .ok plan =>
          -- the lift: every baked constant re-expressed as an s0 `CplxE` of the
          -- live pole. The clamp ENFORCES the classified interval in-kernel, so
          -- an out-of-range slot write saturates the crossing's σ instead of
          -- walking off the classified region. κ = μ·B is σ_ν-independent (baked).
          let sLo ← litF sLo
          let sHi ← litF sHi
          let sigC ← clampE r.sigma sLo sHi
          let rOmega ← litF rOm
          let nuReal ← neg sigC
          let muE ← cplxLitE mu
          let dNuMu ← csubE (nuReal, rOmega) muE
          let invG ← litF (1.0 / g)
          let aE ← scaleRealE invG dNuMu
          let one ← cOneE
          let invNuMuE ← cdivE one dNuMu
          let zero ← lit 0
          let invA ← (Array.range plan.nDepth).mapM fun k => do
            let index ← litF (k + 1).toFloat
            let denominator ← caddE aE (index, zero)
            cdivE one denominator
          let kappaB := mu.scale B
          let kE ← cplxLitE kappaB
          let muSigma ← litF vSig
          let muOmega ← litF vOm
          match plan.region with
          | .serOnly =>
            let m1 ← bloomM1E invA kE
            let k1Ser ← cmulE m1 invNuMuE
            let fSer ← cnegE invNuMuE
            out := out.push {
              muSigma, muOmega, nuSigma := sigC, nuOmega := rOmega
              bloomB := B, gRate := g, c, kappa := kE
              k1Ser, k1Cf := (zero, zero), fSer
              dSwitch := zero, invA, cfB := #[], cfN := #[] }
          | .crossing =>
            -- WS-LP phase 2: the CF lane's constants as s0 `CplxE` of the live
            -- pole — `cfB`/`cfN` linear in `a`, `CF(κ)` by the SAME emitted
            -- fraction the lane renders with (`bloomCFE` at z = κ), the bridge
            -- constant by the emitted `Γ★` (`bloomGammaStarE`, phase 0), and
            -- `dSwitch = (ln|κ| − ½·ln|a+1|²)/g` via `logSig` (the `clogE`
            -- modulus form — no sqrt in the vocabulary needed).
            let cfB ← (Array.range (plan.kDepth + 1)).mapM fun j => do
              let odd ← litF (2 * j + 1).toFloat
              let real ← sub odd aE.1
              let imag ← neg aE.2
              pure (real, imag)
            let cfN ← (Array.range plan.kDepth).mapM fun j => do
              let jf ← cplxLitE ⟨(j + 1).toFloat, 0⟩
              let difference ← csubE jf aE
              cmulE jf difference
            let cfK ← bloomCFE cfB cfN kE
            let gSig ← litF g
            let cfReal ← div cfK.1 gSig
            let cfImag ← div cfK.2 gSig
            let cfOverG : CplxE := (cfReal, cfImag)
            let aP1 ← caddE aE one
            -- `ln|κ|` is BAKED (κ is σ_ν-independent), so its half moves to the
            -- carrier; only the `|a+1|` half is live and keeps its emitted
            -- `logSig` modulus form (no sqrt in the vocabulary).
            let logKappa ← litF (DyadicI.toFloat
              (DyadicI.log (CplxDI.abs kappaB.toExact)))
            let real2 ← mul aP1.1 aP1.1
            let imag2 ← mul aP1.2 aP1.2
            let norm2 ← add real2 imag2
            let logNorm ← logSig norm2
            let half ← lit 5 1
            let halfLogNorm ← mul half logNorm
            let switchNumerator ← sub logKappa halfLogNorm
            let dSwitch ← div switchNumerator gSig
            let gammaStar ← bloomGammaStarE aE kE gSig
            let k1Ser ← csubE gammaStar cfOverG
            let k1Cf ← cnegE cfOverG
            let fSer ← cnegE invNuMuE
            out := out.push {
              muSigma, muOmega, nuSigma := sigC, nuOmega := rOmega
              bloomB := B, gRate := g, c, kappa := kE
              k1Ser, k1Cf, fSer
              dSwitch, invA, cfB, cfN }
      | some rSig =>
        let nu : CplxB := ⟨-rSig, rOm⟩
        -- The checked preflight has already rejected every excluded region for
        -- the whole composition. This private materializer therefore emits only
        -- supported region-indexed lanes; its defensive `continue` arms cannot
        -- create a production partial bank.
        let plan := classifyBloomPair mu nu B g
        let aC := plan.aC
        let kappa := plan.kappa
        -- P2, THE VALUES FLIP. The floats above are the DEPTH LOOPS' inputs and
        -- stay float on purpose: moving them would move `nDepth`/`kDepth` and
        -- therefore the emitted array SIZES, which is a structural change and
        -- does not belong inside a value differential. Everything DOWNSTREAM of
        -- them is exact from here on. The lifts are exact (a finite double is a
        -- dyadic), so nothing is lost crossing.
        let aD := aC.toExact
        let kD := kappa.toExact
        let kAbs := CplxDI.abs kD
        let kZero := kAbs.ok && kAbs.lo.isZero && kAbs.hi.isZero
        let gD := DyadicI.ofFloat g
        let invNuMuD := CplxDI.div CplxDI.one (CplxDI.sub nu.toExact mu.toExact)
        let aP1D := CplxDI.add aD CplxDI.one
        -- `ln(|κ|/|a+1|)/g` — the per-sample series↔CF bridge point
        let dSwitchD := if kZero then DyadicI.zero else DyadicI.div
          (DyadicI.log (DyadicI.div (CplxDI.abs kD) (CplxDI.abs aP1D))) gD
        let invAD := (Array.range plan.nDepth).map (fun k =>
          CplxDI.div CplxDI.one (CplxDI.add aD (CplxDI.ofNat (k + 1))))
        let cfBD := (Array.range (plan.kDepth + 1)).map (fun j =>
          CplxDI.mkI (DyadicI.sub (DyadicI.ofNat (2 * j + 1)) aD.re) (DyadicI.neg aD.im))
        let cfND := (Array.range plan.kDepth).map (fun j =>
          let jf := CplxDI.ofNat (j + 1)
          CplxDI.mul jf (CplxDI.sub jf aD))
        match plan.region with
        | .excludedConditioning => continue
        | .excludedDepth => continue
        | .serOnly =>
          -- M(κ)'s own term count is a REPRODUCIBILITY question (point carrier);
          -- the sum it sizes is an ACCURACY question (enclosure). Note this count
          -- is not `plan.nDepth`: that one sizes the emitted `invA` at `zBnd`,
          -- this one converges the κ-side constant.
          let nK := if kZero then 0 else bloomM1DepthD aC.toPoint kappa.toPoint bloomM1TolD
          let mK := if kZero then CplxDI.one else bloomM1D aD kD nK
          let some k1SerE ← cplxLitD? (CplxDI.mul mK invNuMuD) | return none
          let some kappaE ← cplxLitD? kD | return none
          let some fSerE ← cplxLitD? (CplxDI.neg invNuMuD) | return none
          let some invAE ← cplxLitsD? invAD | return none
          let muSigma ← litF vSig
          let muOmega ← litF vOm
          let nuSigma ← litF rSig
          let nuOmega ← litF rOm
          let zero ← lit 0
          out := out.push {
            muSigma, muOmega, nuSigma, nuOmega
            bloomB := B, gRate := g, c, kappa := kappaE
            k1Ser := k1SerE, k1Cf := (zero, zero)
            fSer := fSerE
            dSwitch := zero, invA := invAE, cfB := #[], cfN := #[] }
        | .crossing =>
          -- CF(κ) runs on the POINT carrier: modified Lentz self-corrects, and an
          -- enclosure cannot follow that (it widens ~2 bits per iteration and
          -- poisons around i≈100, well short of the shipped CF depths). So the
          -- CF-side constants carry NO certificate — `asPointI` ASSERTS the
          -- computed value, it does not bound the true one. Everything else in
          -- this arm keeps its enclosure; only the CF is laundered, on purpose
          -- and by name.
          let (cfKp, _) := bloomCFPointD aC.toPoint kappa.toPoint bloomCFTolD
          let cfOverG := CplxDI.scale (DyadicI.inv gD) cfKp.asPointI
          let gs := bloomGammaStarD aD kD gD
          let some kappaE ← cplxLitD? kD | return none
          let some k1SerE ← cplxLitD? (CplxDI.sub gs cfOverG) | return none
          let some k1CfE ← cplxLitD? (CplxDI.neg cfOverG) | return none
          let some fSerE ← cplxLitD? (CplxDI.neg invNuMuD) | return none
          let some invAE ← cplxLitsD? invAD | return none
          let some cfBE ← cplxLitsD? cfBD | return none
          let some cfNE ← cplxLitsD? cfND | return none
          let muSigma ← litF vSig
          let muOmega ← litF vOm
          let nuSigma ← litF rSig
          let nuOmega ← litF rOm
          let dSwitch ← litF (DyadicI.toFloat dSwitchD)
          out := out.push {
            muSigma, muOmega, nuSigma, nuOmega
            bloomB := B, gRate := g, c, kappa := kappaE
            k1Ser := k1SerE, k1Cf := k1CfE
            fSer := fSerE
            dSwitch, invA := invAE
            cfB := cfBE, cfN := cfNE }
        | .coincidentCrossing =>
          -- WS-A4: the CF branch (large z, `k1Cf`/`cfB`/`cfN`), coincidence-stable,
          -- bridges below `dSwitch` to the series-DD branch (`dCoef`/`k1SerDD`) plus
          -- the τ·e secular `c·e^κ·(e^{νd}−e^{μd})/(ν−μ)`. The E1 series lane
          -- (`invA`/`k1Ser`/`fSer`, singular at a=0) is NOT emitted (region-indexed).
          let (cfKp, _) := bloomCFPointD aC.toPoint kappa.toPoint bloomCFTolD
          let cfK := cfKp.asPointI
          let cfOverG := CplxDI.scale (DyadicI.inv gD) cfK
          let dCoef := bloomDCoefD aD plan.nDepth
          let some kappaE ← cplxLitD? kD | return none
          let some k1CfE ← cplxLitD? (CplxDI.neg cfOverG) | return none
          let some cfBE ← cplxLitsD? cfBD | return none
          let some cfNE ← cplxLitsD? cfND | return none
          let some dCoefE ← cplxLitsD? dCoef | return none
          let some k1SerDDE ← cplxLitD? (bloomPhiKappaOverGD aD kD cfK dCoef gD)
            | return none
          let some eKappaE ← cplxLitD? (CplxDI.exp kD) | return none
          let muSigma ← litF vSig
          let muOmega ← litF vOm
          let nuSigma ← litF rSig
          let nuOmega ← litF rOm
          let zero ← lit 0
          let dSwitch ← litF (DyadicI.toFloat dSwitchD)
          let secReal ← litF (vSig - rSig)
          let secImag ← litF (rOm - vOm)
          out := out.push {
            muSigma, muOmega, nuSigma, nuOmega
            bloomB := B, gRate := g, c, kappa := kappaE
            k1Ser := (zero, zero), k1Cf := k1CfE, fSer := (zero, zero)
            dSwitch, invA := #[]
            cfB := cfBE, cfN := cfNE
            coincident := true
            dCoef := dCoefE
            k1SerDD := k1SerDDE
            eKappa := eKappaE
            -- `secCoef` stays FLOAT deliberately: `vSig − rSig` is a single IEEE
            -- operation on two already-exact inputs, hence exactly rounded, so
            -- `litF` of the carrier's answer would be bit-identical. Flipping it
            -- would buy nothing — single-op float arithmetic on exact inputs is
            -- not a libm dependency, and that is what keeps P2 from sprawling.
            secCoef := (secReal, secImag) }
        | .coincidentSubtle =>
          -- WS-A4 subtle bloom (`dSwitch < 0`): the per-sample path is series-DD from
          -- d = 0, so the CF lane is DEAD (the `selectE` is const-true — LLVM `select`
          -- ignores the unselected operand). Region-indexed: emit series-DD + the τ·e
          -- secular ONLY — no CF lane (`k1Cf`/`cfB`/`cfN`, kept empty ⇒ `bloomComposedSig`
          -- routes to the subtle sub-branch), no `invA`. `bloomPhiKappaOverGD`'s subtle
          -- branch reads only `dCoef`/κ (`cfK` unread), so a dummy `cfK` is bit-identical.
          let dCoef := bloomDCoefD aD plan.nDepth
          let some kappaE ← cplxLitD? kD | return none
          let some dCoefE ← cplxLitsD? dCoef | return none
          let some k1SerDDE ←
            cplxLitD? (bloomPhiKappaOverGD aD kD CplxDI.zero dCoef gD) | return none
          let some eKappaE ← cplxLitD? (CplxDI.exp kD) | return none
          let muSigma ← litF vSig
          let muOmega ← litF vOm
          let nuSigma ← litF rSig
          let nuOmega ← litF rOm
          let zero ← lit 0
          let dSwitch ← litF (DyadicI.toFloat dSwitchD)
          let secReal ← litF (vSig - rSig)
          let secImag ← litF (rOm - vOm)
          out := out.push {
            muSigma, muOmega, nuSigma, nuOmega
            bloomB := B, gRate := g, c, kappa := kappaE
            k1Ser := (zero, zero), k1Cf := (zero, zero), fSer := (zero, zero)
            dSwitch
            invA := #[], cfB := #[], cfN := #[]
            coincident := true
            dCoef := dCoefE
            k1SerDD := k1SerDDE
            eKappa := eKappaE
            secCoef := (secReal, secImag) }
  pure (some out)

/-- Classify the complete requested Cartesian product before materializing any
    coefficient arrays.  This is the stop-line inventory: every unsupported or
    excluded pair has stable indices and a named reason. -/
private def bloomCompositionExclusions (voice reverb : Array ModalMode) (B g : Float) :
    BuildM (Array BloomPairExclusion) := do
  let mut out : Array BloomPairExclusion := #[]
  for vi in [0:voice.size] do
    let some v := voice[vi]? | continue
    let some vSig ← sigConstF? v.sigma
      | for ri in [0:reverb.size] do
          out := out.push { voiceIndex := some vi, roomIndex := some ri,
                            reason := .voiceSigmaNotConstant }
        continue
    let some vOm ← sigConstF? v.omega
      | for ri in [0:reverb.size] do
          out := out.push { voiceIndex := some vi, roomIndex := some ri,
                            reason := .voiceOmegaNotConstant }
        continue
    let mu : CplxB := ⟨-vSig, vOm⟩
    for ri in [0:reverb.size] do
      let some r := reverb[ri]? | continue
      let some rOm ← sigConstF? r.omega
        | out := out.push { voiceIndex := some vi, roomIndex := some ri,
                            reason := .roomOmegaNotConstant }
          continue
      match ← sigConstF? r.sigma with
      | none =>
        let some (sLo, sHi) := r.sigmaRange
          | out := out.push { voiceIndex := some vi, roomIndex := some ri,
                              reason := .liveSigmaRangeMissing }
            continue
        if !(← sigIsS0 r.sigma) then
          out := out.push { voiceIndex := some vi, roomIndex := some ri,
                            reason := .liveSigmaNotS0 }
          continue
        match classifyBloomPairLiveChecked mu rOm sLo sHi B g with
        | .ok _ => pure ()
        | .error reason =>
          out := out.push { voiceIndex := some vi, roomIndex := some ri, reason }
      | some rSig =>
        let nu : CplxB := ⟨-rSig, rOm⟩
        match (classifyBloomPair mu nu B g).region with
        | .excludedConditioning =>
          out := out.push { voiceIndex := some vi, roomIndex := some ri,
                            reason := .excludedConditioning }
        | .excludedDepth =>
          out := out.push { voiceIndex := some vi, roomIndex := some ri,
                            reason := .excludedDepth }
        | _ => pure ()
  return out

/-- Typed, all-or-nothing bloom composition.  `pairs` is populated only when
    every requested voice×room pair is supported; otherwise the complete
    exclusion inventory is returned and no partial room response exists for a
    caller to realize accidentally. -/
def bloomComposeChecked (voice reverb : Array ModalMode) (B g : Float) : BuildM BloomComposition := do
  let expectedPairs := voice.size * reverb.size
  let exclusions ← bloomCompositionExclusions voice reverb B g
  if !exclusions.isEmpty then
    pure { expectedPairs, pairs := #[], exclusions }
  else
    match ← bloomComposePairs? voice reverb B g with
    | some pairs =>
      if pairs.size == expectedPairs then
        pure { expectedPairs, pairs, exclusions := #[] }
      else
        pure { expectedPairs, pairs := #[], exclusions := #[{
            voiceIndex := none, roomIndex := none,
            reason := .coefficientMaterialization }] }
    | none =>
      pure { expectedPairs, pairs := #[], exclusions := #[{
          voiceIndex := none, roomIndex := none,
          reason := .coefficientMaterialization }] }

/-- Compatibility lens preserving the historical `Option (Array BloomPair)`
    API. New production lowering uses `bloomComposeChecked` and reports reasons. -/
def bloomCompose (voice reverb : Array ModalMode) (B g : Float) : BuildM (Option (Array BloomPair)) := do
  pure (← bloomComposeChecked voice reverb B g).toOption

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

    RANGE (WS-AA range lens). The selected two or three `env·w` factors still
    land at fixed Q4.28. A per-sample exponent is not an acceptable patch: these
    factors depend on `dPos`, and moving `floatExponent`/`ldexp` into every audio
    sample and pair would violate the coefficient-time landing invariant, make
    backend precision affect exponent choices, and scale badly for a 672-pair
    room. A correct selected-lane supremum remains an explicit architecture gate.

    Landing also cannot repair an ill-conditioned value. The measured
    `a=-0.98, |κ|≈72.2` Horner witness is wrong by >1e8 and can exceed the
    `k≤28` ceiling. `classifyBloomPair` therefore names and refuses the
    CF-crossing intersection with the exact-binary radius-`1/32` discs around
    `-1,…,-300`; the live classifier applies the conservative interval analogue.
    `BloomComposition` makes any such or depth/contract exclusion explicit, and
    Patch lowering refuses the requested room crossing rather than rendering a
    bare bloom or partial room. The fixed factor landing is not a proof for
    arbitrary authored tables outside this measured admission contract, which is
    why the public `bloomgong` surface remains withheld. -/
def bloomComposedSig (pairs : Array BloomPair) (clkInt anchorSamples : Sig) : BuildM Sig := do
  let clkRel ← relClockQ clkInt anchorSamples
  let clkFloat ← toFloatE clkRel
  let twoPow32 ← lit 4294967296
  let secondsTimesRate ← div clkFloat twoPow32
  let sr ← sampleRate
  let dSec ← div secondsTimesRate sr
  let zero ← lit 0
  let million ← lit 1000000
  let dPos ← clampE dSec zero million
  let oneReal ← lit 1
  let zeroImag ← lit 0
  let one : CplxE := (oneReal, zeroImag)
  let land := fun (env : Sig) (w : CplxE) (ph : Sig) => do
    let realWeight ← mul env w.1
    let scale ← lit 268435456
    let realScaled ← mul realWeight scale
    let realInt ← toIntE realScaled
    let cosine ← fixedCosCycSig ph
    let realPart ← mul realInt cosine
    let imagWeight ← mul env w.2
    let imagScaled ← mul imagWeight scale
    let imagInt ← toIntE imagScaled
    let sine ← fixedSinCycSig ph
    let imagPart ← mul imagInt sine
    let difference ← sub realPart imagPart
    let shift ← lit 28
    rshift difference shift
  -- the per-sample continued fraction `CF(z) = Γ(a,z)eᶻz^{−a}` (`bloomCFE`,
  -- shared with the live-pole lift's CF(κ) constant), used by the crossing and
  -- coincident branches (both large-z sides).
  let zeroInt ← litI 0
  let bankQ ← pairs.foldlM (fun acc p => do
    let gRate ← litF p.gRate
    let gd ← mul gRate dPos
    let negGd ← neg gd
    let eg ← expSig negGd
    let zReal ← mul p.kappa.1 eg
    let zImag ← mul p.kappa.2 eg
    let z : CplxE := (zReal, zImag)
    let bloomB ← litF p.bloomB
    let oneMinusEg ← sub oneReal eg
    let off ← mul bloomB oneMinusEg
    let offSamples ← mul off sr
    let offFixed ← mul offSamples twoPow32
    let offInt ← toIntE offFixed
    let clkW ← add clkRel offInt
    let phNu ← modePhaseQ p.nuOmega clkRel
    let phMu ← modePhaseQ p.muOmega clkW
    let nuTime ← mul p.nuSigma dPos
    let envNu ← neg nuTime >>= expSig
    let dWarped ← add dPos off
    let muTime ← mul p.muSigma dWarped
    let envMu ← neg muTime >>= expSig
    if p.coincident then
      if p.cfN.isEmpty then
        -- WS-A4 / WS-CL: the subtle-bloom coincident pair (`dSwitch < 0`). The
        -- per-sample path is series-DD from d = 0 — the CF lane is dead (region-
        -- indexed: `cfN` is empty, so `cfEnv` is not even reached — it would index-
        -- panic). Series-DD + the τ·e secular, ALWAYS on. Bit-identical to the old
        -- coincident branch with the const-true `onSer` (the `selectE`s all picked
        -- the series-DD/secular arm; the LLVM `select` ignored the CF operand).
        let zeroC : CplxE := (zero, zeroImag)
        let horner ← p.dCoef.foldrM (fun dk h => do
          let zh ← cmulE z h
          caddE dk zh) zeroC
        let phiZ ← cmulE z horner
        let negPhiReal ← neg phiZ.1
        let k2Real ← div negPhiReal gRate
        let negPhiImag ← neg phiZ.2
        let k2Imag ← div negPhiImag gRate
        let k2ser : CplxE := (k2Real, k2Imag)
        let w1 ← cmulE p.c p.k1SerDD
        let w2 ← cmulE p.c k2ser
        let phMuS ← modePhaseQ p.muOmega clkRel
        let straightMuTime ← mul p.muSigma dPos
        let envMuS ← neg straightMuTime >>= expSig
        let zsecReal ← mul p.secCoef.1 dPos
        let zsecImag ← mul p.secCoef.2 dPos
        let zsec : CplxE := (zsecReal, zsecImag)
        let zrealSq ← mul zsec.1 zsec.1
        let zimagSq ← mul zsec.2 zsec.2
        let zsq ← add zrealSq zimagSq
        let threshold ← litF 0.01
        let big ← gt zsq threshold
        let zsafeReal ← selectE big zsec.1 oneReal
        let zsafeImag ← selectE big zsec.2 zeroImag
        let zsafe : CplxE := (zsafeReal, zsafeImag)
        let ezr ← expSig zsec.1
        let cosZ ← cosSig zsec.2
        let ezReal ← mul ezr cosZ
        let sinZ ← sinSig zsec.2
        let ezImag ← mul ezr sinZ
        let ez : CplxE := (ezReal, ezImag)
        let numerator ← csubE ez one
        let direct ← cdivE numerator zsafe
        let series ← cexpm1SeriesE zsec
        let cxsecReal ← selectE big direct.1 series.1
        let cxsecImag ← selectE big direct.2 series.2
        let cxsec : CplxE := (cxsecReal, cxsecImag)
        let cek ← cmulE p.c p.eKappa
        let secular ← scaleRealE dPos cxsec
        let wsec ← cmulE cek secular
        let laneNu ← land envNu w1 phNu
        let laneMu ← land envMu w2 phMu
        let lanes ← add laneNu laneMu
        let laneSec ← land envMuS wsec phMuS
        let allLanes ← add lanes laneSec
        add acc allLanes
      else
        -- WS-A4: the τ·e coincidence pair (`coincidentCrossing`). CF branch (large z,
        -- `onSer` false) bridges at `dSwitch` to the series-DD branch (small z) + the
        -- secular.
        let onSer ← gt dPos p.dSwitch
        let cf ← bloomCFE p.cfB p.cfN z
        let k2cfReal ← div cf.1 gRate
        let k2cfImag ← div cf.2 gRate
        let k2cf : CplxE := (k2cfReal, k2cfImag)
        -- series-DD: Φ(a,z) = z·Σ dₙ z^{n−1} (Horner over `dCoef`); k2 = −Φ(a,z)/g.
        let zeroC : CplxE := (zero, zeroImag)
        let horner ← p.dCoef.foldrM (fun dk h => do
          let zh ← cmulE z h
          caddE dk zh) zeroC
        let phiZ ← cmulE z horner
        let negPhiReal ← neg phiZ.1
        let k2serReal ← div negPhiReal gRate
        let negPhiImag ← neg phiZ.2
        let k2serImag ← div negPhiImag gRate
        let k2ser : CplxE := (k2serReal, k2serImag)
        let k1Real ← selectE onSer p.k1SerDD.1 p.k1Cf.1
        let k1Imag ← selectE onSer p.k1SerDD.2 p.k1Cf.2
        let k1 : CplxE := (k1Real, k1Imag)
        let k2Real ← selectE onSer k2ser.1 k2cf.1
        let k2Imag ← selectE onSer k2ser.2 k2cf.2
        let k2 : CplxE := (k2Real, k2Imag)
        let w1 ← cmulE p.c k1
        let w2 ← cmulE p.c k2
        -- the τ·e secular `c·e^κ·e^{μd}·d·cexpm1((ν−μ)d)` on the STRAIGHT μ carrier
        -- (= `c·e^κ·(e^{νd}−e^{μd})/(ν−μ)`), gated OFF on the CF side.
        let phMuS ← modePhaseQ p.muOmega clkRel
        let straightMuTime ← mul p.muSigma dPos
        let envMuS ← neg straightMuTime >>= expSig
        let zsecReal ← mul p.secCoef.1 dPos
        let zsecImag ← mul p.secCoef.2 dPos
        let zsec : CplxE := (zsecReal, zsecImag)
        let zrealSq ← mul zsec.1 zsec.1
        let zimagSq ← mul zsec.2 zsec.2
        let zsq ← add zrealSq zimagSq
        let threshold ← litF 0.01
        let big ← gt zsq threshold
        let zsafeReal ← selectE big zsec.1 oneReal
        let zsafeImag ← selectE big zsec.2 zeroImag
        let zsafe : CplxE := (zsafeReal, zsafeImag)
        let ezr ← expSig zsec.1
        let cosZ ← cosSig zsec.2
        let ezReal ← mul ezr cosZ
        let sinZ ← sinSig zsec.2
        let ezImag ← mul ezr sinZ
        let ez : CplxE := (ezReal, ezImag)
        let numerator ← csubE ez one
        let direct ← cdivE numerator zsafe
        let series ← cexpm1SeriesE zsec
        let cxsecReal ← selectE big direct.1 series.1
        let cxsecImag ← selectE big direct.2 series.2
        let cxsec : CplxE := (cxsecReal, cxsecImag)
        let cek ← cmulE p.c p.eKappa
        let secular ← scaleRealE dPos cxsec
        let wsecFull ← cmulE cek secular
        let wsecReal ← selectE onSer wsecFull.1 zero
        let wsecImag ← selectE onSer wsecFull.2 zeroImag
        let wsec : CplxE := (wsecReal, wsecImag)
        let laneNu ← land envNu w1 phNu
        let laneMu ← land envMu w2 phMu
        let lanes ← add laneNu laneMu
        let laneSec ← land envMuS wsec phMuS
        let allLanes ← add lanes laneSec
        add acc allLanes
    else
      let mser ← bloomM1E p.invA z
      let k2ser ← cmulE mser p.fSer
      let (k1, k2) ← if p.cfN.isEmpty then pure (p.k1Ser, k2ser) else do
        let cf ← bloomCFE p.cfB p.cfN z
        let k2cfReal ← div cf.1 gRate
        let k2cfImag ← div cf.2 gRate
        let k2cf : CplxE := (k2cfReal, k2cfImag)
        let onSer ← gt dPos p.dSwitch
        let k1Real ← selectE onSer p.k1Ser.1 p.k1Cf.1
        let k1Imag ← selectE onSer p.k1Ser.2 p.k1Cf.2
        let k2Real ← selectE onSer k2ser.1 k2cf.1
        let k2Imag ← selectE onSer k2ser.2 k2cf.2
        pure ((k1Real, k1Imag), (k2Real, k2Imag))
      let w1 ← cmulE p.c k1
      let w2 ← cmulE p.c k2
      let laneNu ← land envNu w1 phNu
      let laneMu ← land envMu w2 phMu
      let lanes ← add laneNu laneMu
      add acc lanes) zeroInt
  let afterStrike ← gt clkRel zero
  let output ← fixedOutQ 30 bankQ
  selectE afterStrike output zero

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
    BuildM (Option (Array BloomFoldPair)) := do
  let some r1Sig ← sigConstF? r1.sigma | return none
  let some r1Om  ← sigConstF? r1.omega | return none
  let some r1Cre ← sigConstF? r1.cre   | return none
  let some r1Cim ← sigConstF? r1.cim   | return none
  let some r2Sig ← sigConstF? r2.sigma | return none
  let some r2Om  ← sigConstF? r2.omega | return none
  let some r2Cre ← sigConstF? r2.cre   | return none
  let some r2Cim ← sigConstF? r2.cim   | return none
  let nu1 : CplxB := ⟨-r1Sig, r1Om⟩
  let nu2 : CplxB := ⟨-r2Sig, r2Om⟩
  let r1r2 := (⟨r1Cre, r1Cim⟩ : CplxB).mul ⟨r2Cre, r2Cim⟩
  let mut out : Array BloomFoldPair := #[]
  for v in voice do
    let some vSig ← sigConstF? v.sigma | return none
    let some vOm  ← sigConstF? v.omega | return none
    let mu : CplxB := ⟨-vSig, vOm⟩
    let a1 := (nu1.sub mu).scale (1.0 / g)
    let a2 := (nu2.sub mu).scale (1.0 / g)
    let kappa := mu.scale B
    let a1D := a1.toExact
    let a2D := a2.toExact
    let kD := kappa.toExact
    let gD := DyadicI.ofFloat g
    -- Series-side admission on the EXACT carrier: this predicate decides whether
    -- a voice mode is emitted AT ALL, and `nRaw` SIZES the emitted `invA`/`qCoef`
    -- arrays — both structural, so neither may come from a platform's libm.
    -- (P1 flipped `classifyBloomPair`'s twin of this predicate and missed this
    -- site, because `bloomFoldCompose` has no caller outside the `ddfold` gate.)
    -- Each threshold is taken in the direction that DROPS a mode it cannot
    -- certify — the graceful-exclusion rule `classifyBloomPairLive` follows.
    let half := DyadicI.ofFloat 0.5
    if !DyadicI.certGt (CplxDI.abs a1D) half then continue
    if !DyadicI.certGt (CplxDI.abs a2D) half then continue
    if !DyadicI.certGt (CplxDI.abs (CplxDI.add a1D CplxDI.one)) (CplxDI.abs kD) then continue
    let nRaw := bloomM1DepthD a1.toPoint kappa.toPoint bloomM1TolD
    let nDepth := nRaw + 8    -- M(a1;z) depth; the a-DD's Hₙ factor is a small (~log) tail
    if nDepth > bloomDepthCap then continue
    -- the constants, exact and then projected — `CplxB ⟨toFloat re, toFloat im⟩`
    -- emits bit-identically to `cplxLitD?`, since `cplxLitE` is `litF` of exactly
    -- those two doubles; a poisoned constant drops the mode rather than emitting
    -- a fabricated zero.
    let toB := fun (x : CplxDI) => if x.ok then some (⟨x.re.toFloat, x.im.toFloat⟩ : CplxB) else none
    let qCoefD := bloomFoldQCoefD a1D a2D nDepth
    let invAD := (Array.range nDepth).map (fun k =>
      CplxDI.div CplxDI.one (CplxDI.add a1D (CplxDI.ofNat (k + 1))))
    let mK1D := bloomM1D a1D kD (bloomM1DepthD a1.toPoint kappa.toPoint bloomM1TolD)
    let ddMκD := bloomFoldDDaMD qCoefD kD          -- DDa(M;κ) = Σ κⁿ Qₙ
    let invA1a2D := CplxDI.div CplxDI.one (CplxDI.mul a1D a2D)
    let invA2D := CplxDI.div CplxDI.one a2D
    -- DDa(M/a;κ) = −M(a1;κ)/(a1 a2) + DDa(M;κ)/a2 ; k1a1 = M(a1;κ)/(g a1)
    let ddMaκD := CplxDI.add (CplxDI.neg (CplxDI.mul mK1D invA1a2D)) (CplxDI.mul ddMκD invA2D)
    let some qCoef := qCoefD.mapM toB | continue
    let some invA := invAD.mapM toB | continue
    let some k1a1 := toB (CplxDI.div mK1D (CplxDI.scale gD a1D)) | continue
    let some ddK1g := toB (CplxDI.scale (DyadicI.inv (DyadicI.mul gD gD)) ddMaκD) | continue
    let some invA1a2 := toB invA1a2D | continue
    let some invA2 := toB invA2D | continue
    let roomAmp ← cplxLitE r1r2
    let c ← cmulE v.ampE roomAmp
    out := out.push {
      muSigma := vSig, muOmega := vOm
      nu1Sigma := r1Sig, nu1Omega := r1Om, nu2Sigma := r2Sig, nu2Omega := r2Om
      bloomB := B, gRate := g
      c
      kappa
      k1a1, ddK1g, invA1a2, invA2, invA, qCoef }
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
def bloomFoldComposedSig (pairs : Array BloomFoldPair) (clkInt anchorSamples : Sig) : BuildM Sig := do
  let clkRel ← relClockQ clkInt anchorSamples
  let clkFloat ← toFloatE clkRel
  let twoPow32 ← lit 4294967296
  let secondsTimesRate ← div clkFloat twoPow32
  let sr ← sampleRate
  let dSec ← div secondsTimesRate sr
  let zero ← lit 0
  let million ← lit 1000000
  let dPos ← clampE dSec zero million
  let oneReal ← lit 1
  let zeroImag ← lit 0
  let one : CplxE := (oneReal, zeroImag)
  let land := fun (env : Sig) (w : CplxE) (ph : Sig) => do
    let realWeight ← mul env w.1
    let scale ← lit 268435456
    let realScaled ← mul realWeight scale
    let realInt ← toIntE realScaled
    let cosine ← fixedCosCycSig ph
    let realPart ← mul realInt cosine
    let imagWeight ← mul env w.2
    let imagScaled ← mul imagWeight scale
    let imagInt ← toIntE imagScaled
    let sine ← fixedSinCycSig ph
    let imagPart ← mul imagInt sine
    let difference ← sub realPart imagPart
    let shift ← lit 28
    rshift difference shift
  let zeroInt ← litI 0
  let bankQ ← pairs.foldlM (fun acc p => do
    let gRate ← litF p.gRate
    let gd ← mul gRate dPos
    let negGd ← neg gd
    let eg ← expSig negGd
    let kappaReal ← litF p.kappa.re
    let zReal ← mul kappaReal eg
    let kappaImag ← litF p.kappa.im
    let zImag ← mul kappaImag eg
    let z : CplxE := (zReal, zImag)
    let bloomB ← litF p.bloomB
    let oneMinusEg ← sub oneReal eg
    let off ← mul bloomB oneMinusEg
    let offSamples ← mul off sr
    let offFixed ← mul offSamples twoPow32
    let offInt ← toIntE offFixed
    let clkW ← add clkRel offInt
    let nu2Omega ← litF p.nu2Omega
    let phNu2 ← modePhaseQ nu2Omega clkRel
    let muOmega ← litF p.muOmega
    let phMu ← modePhaseQ muOmega clkW
    let nu2Sigma ← litF p.nu2Sigma
    let nu2Time ← mul nu2Sigma dPos
    let negNu2Time ← neg nu2Time
    let envNu2 ← expSig negNu2Time
    let muSigma ← litF p.muSigma
    let dWarped ← add dPos off
    let muTime ← mul muSigma dWarped
    let negMuTime ← neg muTime
    let envMu ← expSig negMuTime
    -- ν2-carrier weight: K1(a1)·d·cexpm1(Δd) + ddK1/g. Δ = ν1−ν2 (pole difference).
    let deltaReal ← litF (p.nu2Sigma - p.nu1Sigma)
    let wReal ← mul deltaReal dPos
    let deltaImag ← litF (p.nu1Omega - p.nu2Omega)
    let wImag ← mul deltaImag dPos
    let w : CplxE := (wReal, wImag)
    let wrealSq ← mul w.1 w.1
    let wimagSq ← mul w.2 w.2
    let wsq ← add wrealSq wimagSq
    let threshold ← litF 0.01
    let big ← gt wsq threshold
    let wsafeReal ← selectE big w.1 oneReal
    let wsafeImag ← selectE big w.2 zeroImag
    let wsafe : CplxE := (wsafeReal, wsafeImag)
    let ewr ← expSig w.1
    let cosW ← cosSig w.2
    let ewReal ← mul ewr cosW
    let sinW ← sinSig w.2
    let ewImag ← mul ewr sinW
    let ew : CplxE := (ewReal, ewImag)
    let numerator ← csubE ew one
    let direct ← cdivE numerator wsafe
    let series ← cexpm1SeriesE w
    let deltaQuotReal ← selectE big direct.1 series.1
    let deltaQuotImag ← selectE big direct.2 series.2
    let cxDelta : CplxE := (deltaQuotReal, deltaQuotImag)
    let k1a1 ← cplxLitE p.k1a1
    let scaledDelta ← scaleRealE dPos cxDelta
    let secular ← cmulE k1a1 scaledDelta
    let ddK1g ← cplxLitE p.ddK1g
    let nuWeight ← caddE secular ddK1g
    let wNu ← cmulE p.c nuWeight
    -- voice-carrier weight: DDa(K2)(z)/g = −DDa(M/a;z)/g², M(a1;z) & DDa(M;z) Horners.
    let mser ← p.invA.foldrM (fun ik h => do
      let ikE ← cplxLitE ik
      let zik ← cmulE z ikE
      let product ← cmulE zik h
      caddE one product) one
    let zeroC : CplxE := (zero, zeroImag)
    let ddMHorner ← p.qCoef.foldrM (fun q h => do
      let qE ← cplxLitE q
      let zh ← cmulE z h
      caddE qE zh) zeroC
    let ddMz ← cmulE z ddMHorner
    let invA1a2 ← cplxLitE p.invA1a2
    let mOverA ← cmulE mser invA1a2
    let negMOverA ← cnegE mOverA
    let invA2 ← cplxLitE p.invA2
    let ddMOverA2 ← cmulE ddMz invA2
    let ddMaz ← caddE negMOverA ddMOverA2
    let inverseGSquared ← litF (-1.0 / (p.gRate * p.gRate))
    let scaledDd ← scaleRealE inverseGSquared ddMaz
    let wMu ← cmulE p.c scaledDd
    let nuLane ← land envNu2 wNu phNu2
    let muLane ← land envMu wMu phMu
    let lanes ← add nuLane muLane
    add acc lanes) zeroInt
  let afterStrike ← gt clkRel zero
  let output ← fixedOutQ 30 bankQ
  selectE afterStrike output zero

/-- The WS-DDF fold-atom bank as a TERM over the clock leaf (rides `arrUn … (.clk c)`
    like `bloomComposedTerm`, so master warps reach the carriers). -/
def bloomFoldComposedTerm (pairs : Array BloomFoldPair) (anchor : Sig) (c : Clock) : ArrowTerm :=
  ArrowTerm.arrUn (fun clkSig => bloomFoldComposedSig pairs clkSig anchor) (ArrowTerm.clk c)

/-- The pitch-bloom clock warp for a BARE bloomed source (no room requested): the
    offset `B·(1−e^{−g·d⁺})` added to the untouched integer clock — `B` already
    folds the register scale (`B = β·scale/g`). The scale-1 sibling of
    `gongBloomWarp` (which lives one layer up, in `Gong.lean`), defined here so
    `Patch.lowerInput` can realize the explicitly bare source without a
    circular import. `W(0)=0`, monotone, so the bank's own causal gate is
    untouched. -/
def bloomWarpClock (anchorSamples : Sig) (B g : Float) : Clock → BuildM Clock :=
  fun clk => do
    let clkRel ← relClockQ clk anchorSamples
    let clkFloat ← toFloatE clkRel
    let twoPow32 ← lit 4294967296
    let secondsTimesRate ← div clkFloat twoPow32
    let sr ← sampleRate
    let dSec ← div secondsTimesRate sr
    let zero ← lit 0
    let million ← lit 1000000
    let dPos ← clampE dSec zero million
    let gSig ← litF g
    let gd ← mul gSig dPos
    let negGd ← neg gd
    let decay ← expSig negGd
    let one ← lit 1
    let settled ← sub one decay
    let bloomB ← litF B
    let bloom ← mul bloomB settled
    let bloomSamples ← mul bloom sr
    let bloomFixed ← mul bloomSamples twoPow32
    let bloomInt ← toIntE bloomFixed
    add clk bloomInt


end Tropical.EmitArrow
