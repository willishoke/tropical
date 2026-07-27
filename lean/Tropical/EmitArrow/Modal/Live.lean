import Tropical.EmitArrow.Modal.Bloom

/-!
# EmitArrow.Modal.Live

Live-pole bloom classification, lifted Γ-bridge composition, and room-chain folds.
-/

namespace Tropical.EmitArrow

open Tropical.Ir
open Tropical.Exact (DyadicI CplxD CplxDI)

/-- The two regions the LIVE classifier can emit — a two-constructor sum, so
    `bloomCompose`'s match is exhaustive BY TYPE rather than by an
    "unreachable" comment (the coincident regions and the depth exclusion are
    not representable here: they return `none` and stay baked-only). The
    baked classifier keeps the full five-region `SeamRegion`. -/
inductive LiveRegion where
  | serOnly
  | crossing

structure LiveBloomPairPlan where
  region : LiveRegion
  nDepth : Nat
  kDepth : Nat

/-- WS-LP: classify a live-σ pair over its whole σ interval — `some plan` iff
    the pair sits in ONE non-coincident region at EVERY
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
    (B g : Float) : Option LiveBloomPairPlan := Id.run do
  let kappa := mu.scale B
  let imA := (nuOmega - mu.im) / g
  -- Re a = (−σ_ν − Re μ)/g, decreasing in σ_ν
  let reLo := (-sigHi - mu.re) / g
  let reHi := (-sigLo - mu.re) / g
  -- the region conditions, on the exact carrier: this predicate decides whether
  -- a live-σ pair is emitted at all and with which lanes, so it may not depend
  -- on a platform's rounding. `absKappa`/`absAt`/`minAbsAt` are the same
  -- expressions, evaluated as enclosures; each threshold is taken in the
  -- direction that DROPS a pair it cannot certify (graceful exclusion is the
  -- house rule for a legal-but-unservable config, never a warning).
  let imAD := DyadicI.ofFloat imA
  let reLoD := DyadicI.ofFloat reLo
  let reHiD := DyadicI.ofFloat reHi
  let absKappaD := CplxDI.abs kappa.toExact
  let absAtD := fun (re : DyadicI) (c : DyadicI) =>
    DyadicI.sqrt (DyadicI.add (DyadicI.square (DyadicI.add re c)) (DyadicI.square imAD))
  let minAbsAtD := fun (c : DyadicI) =>
    absAtD (DyadicI.max reLoD (DyadicI.min reHiD (DyadicI.neg c))) c
  -- ¬coincident throughout: min |a| ≥ ½ (the τ·e coincidence stays baked-only)
  if !DyadicI.certGt (minAbsAtD DyadicI.zero) (DyadicI.ofFloat 0.5) then return none
  let samples := #[reLo, reHi, max reLo (min reHi (-1.0))]
  if !DyadicI.certLt (minAbsAtD DyadicI.one) absKappaD then
    -- serOnly throughout: |a+1| never falls below |κ|
    let nRaw := samples.foldl (fun m re =>
      max m (bloomM1DepthD (CplxD.ofFloats re imA) kappa.toPoint bloomM1TolD)) 0
    if nRaw + 8 > 300 then return none
    return some { region := .serOnly, nDepth := nRaw + 8, kDepth := 0 }
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
  if nRaw + 8 > 300 || kRaw + 8 > 300 then return none
  return some { region := .crossing, nDepth := nRaw + 8, kDepth := kRaw + 8 }

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
        | some plan =>
          -- the lift: every baked constant re-expressed as an s0 `CplxE` of the
          -- live pole. The clamp ENFORCES the classified interval in-kernel, so
          -- an out-of-range slot write saturates the crossing's σ instead of
          -- walking off the classified region. κ = μ·B is σ_ν-independent (baked).
          let sigC := clampE r.sigma (litF sLo) (litF sHi)
          let nuE : CplxE := (neg sigC, litF rOm)
          let dNuMu := csubE nuE (cplxLitE mu)
          let aE := scaleRealE (litF (1.0 / g)) dNuMu
          let invNuMuE := cdivE cOneE dNuMu
          let invA := (Array.range plan.nDepth).map (fun k =>
            cdivE cOneE (caddE aE (litF (k + 1).toFloat, lit 0)))
          let kappaB := mu.scale B
          let kE := cplxLitE kappaB
          match plan.region with
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
            let cfB := (Array.range (plan.kDepth + 1)).map (fun j =>
              ((sub (litF (2 * j + 1).toFloat) aE.1, neg aE.2) : CplxE))
            let cfN := (Array.range plan.kDepth).map (fun j =>
              let jf := cplxLitE ⟨(j + 1).toFloat, 0⟩
              cmulE jf (csubE jf aE))
            let cfK := bloomCFE cfB cfN kE
            let cfOverG : CplxE := (div cfK.1 (litF g), div cfK.2 (litF g))
            let aP1 := caddE aE cOneE
            -- `ln|κ|` is BAKED (κ is σ_ν-independent), so its half moves to the
            -- carrier; only the `|a+1|` half is live and keeps its emitted
            -- `logSig` modulus form (no sqrt in the vocabulary).
            let dSwitch := div (sub (litF (DyadicI.toFloat
                (DyadicI.log (CplxDI.abs kappaB.toExact))))
              (mul (lit 5 1) (logSig (add (mul aP1.1 aP1.1) (mul aP1.2 aP1.2))))) (litF g)
            out := out.push {
              muSigma := litF vSig, muOmega := litF vOm
              nuSigma := sigC, nuOmega := litF rOm
              bloomB := B, gRate := g, c, kappa := kE
              k1Ser := csubE (bloomGammaStarE aE kE (litF g)) cfOverG
              k1Cf := cnegE cfOverG, fSer := cnegE invNuMuE
              dSwitch, invA, cfB, cfN }
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
        -- P2, THE VALUES FLIP. The floats above are the DEPTH LOOPS' inputs and
        -- stay float on purpose: moving them would move `nDepth`/`kDepth` and
        -- therefore the emitted array SIZES, which is a structural change and
        -- does not belong inside a value differential. Everything DOWNSTREAM of
        -- them is exact from here on. The lifts are exact (a finite double is a
        -- dyadic), so nothing is lost crossing.
        let aD := aC.toExact
        let kD := kappa.toExact
        let gD := DyadicI.ofFloat g
        let invNuMuD := CplxDI.div CplxDI.one (CplxDI.sub nu.toExact mu.toExact)
        let aP1D := CplxDI.add aD CplxDI.one
        -- `ln(|κ|/|a+1|)/g` — the per-sample series↔CF bridge point
        let dSwitchD := DyadicI.div
          (DyadicI.log (DyadicI.div (CplxDI.abs kD) (CplxDI.abs aP1D))) gD
        let invAD := (Array.range plan.nDepth).map (fun k =>
          CplxDI.div CplxDI.one (CplxDI.add aD (CplxDI.ofNat (k + 1))))
        let cfBD := (Array.range (plan.kDepth + 1)).map (fun j =>
          CplxDI.mkI (DyadicI.sub (DyadicI.ofNat (2 * j + 1)) aD.re) (DyadicI.neg aD.im))
        let cfND := (Array.range plan.kDepth).map (fun j =>
          let jf := CplxDI.ofNat (j + 1)
          CplxDI.mul jf (CplxDI.sub jf aD))
        match plan.region with
        | .excludedDepth => continue
        | .serOnly =>
          -- M(κ)'s own term count is a REPRODUCIBILITY question (point carrier);
          -- the sum it sizes is an ACCURACY question (enclosure). Note this count
          -- is not `plan.nDepth`: that one sizes the emitted `invA` at `zBnd`,
          -- this one converges the κ-side constant.
          let nK := bloomM1DepthD aC.toPoint kappa.toPoint bloomM1TolD
          let mK := bloomM1D aD kD nK
          let some k1SerE := cplxLitD? (CplxDI.mul mK invNuMuD) | return none
          let some kappaE := cplxLitD? kD | return none
          let some fSerE  := cplxLitD? (CplxDI.neg invNuMuD) | return none
          let some invAE  := invAD.mapM cplxLitD? | return none
          out := out.push {
            muSigma := litF vSig, muOmega := litF vOm
            nuSigma := litF rSig, nuOmega := litF rOm
            bloomB := B, gRate := g, c, kappa := kappaE
            k1Ser := k1SerE, k1Cf := (lit 0, lit 0)
            fSer := fSerE
            dSwitch := lit 0, invA := invAE, cfB := #[], cfN := #[] }
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
          let some kappaE := cplxLitD? kD | return none
          let some k1SerE := cplxLitD? (CplxDI.sub gs cfOverG) | return none
          let some k1CfE  := cplxLitD? (CplxDI.neg cfOverG) | return none
          let some fSerE  := cplxLitD? (CplxDI.neg invNuMuD) | return none
          let some invAE  := invAD.mapM cplxLitD? | return none
          let some cfBE   := cfBD.mapM cplxLitD? | return none
          let some cfNE   := cfND.mapM cplxLitD? | return none
          out := out.push {
            muSigma := litF vSig, muOmega := litF vOm
            nuSigma := litF rSig, nuOmega := litF rOm
            bloomB := B, gRate := g, c, kappa := kappaE
            k1Ser := k1SerE, k1Cf := k1CfE
            fSer := fSerE
            dSwitch := litF (DyadicI.toFloat dSwitchD), invA := invAE
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
          let some kappaE   := cplxLitD? kD | return none
          let some k1CfE    := cplxLitD? (CplxDI.neg cfOverG) | return none
          let some cfBE     := cfBD.mapM cplxLitD? | return none
          let some cfNE     := cfND.mapM cplxLitD? | return none
          let some dCoefE   := dCoef.mapM cplxLitD? | return none
          let some k1SerDDE := cplxLitD? (bloomPhiKappaOverGD aD kD cfK dCoef gD) | return none
          let some eKappaE  := cplxLitD? (CplxDI.exp kD) | return none
          out := out.push {
            muSigma := litF vSig, muOmega := litF vOm
            nuSigma := litF rSig, nuOmega := litF rOm
            bloomB := B, gRate := g, c, kappa := kappaE
            k1Ser := (lit 0, lit 0), k1Cf := k1CfE, fSer := (lit 0, lit 0)
            dSwitch := litF (DyadicI.toFloat dSwitchD), invA := #[]
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
            secCoef := (litF (vSig - rSig), litF (rOm - vOm)) }
        | .coincidentSubtle =>
          -- WS-A4 subtle bloom (`dSwitch < 0`): the per-sample path is series-DD from
          -- d = 0, so the CF lane is DEAD (the `selectE` is const-true — LLVM `select`
          -- ignores the unselected operand). Region-indexed: emit series-DD + the τ·e
          -- secular ONLY — no CF lane (`k1Cf`/`cfB`/`cfN`, kept empty ⇒ `bloomComposedSig`
          -- routes to the subtle sub-branch), no `invA`. `bloomPhiKappaOverGD`'s subtle
          -- branch reads only `dCoef`/κ (`cfK` unread), so a dummy `cfK` is bit-identical.
          let dCoef := bloomDCoefD aD plan.nDepth
          let some kappaE   := cplxLitD? kD | return none
          let some dCoefE   := dCoef.mapM cplxLitD? | return none
          let some k1SerDDE :=
            cplxLitD? (bloomPhiKappaOverGD aD kD CplxDI.zero dCoef gD) | return none
          let some eKappaE  := cplxLitD? (CplxDI.exp kD) | return none
          out := out.push {
            muSigma := litF vSig, muOmega := litF vOm
            nuSigma := litF rSig, nuOmega := litF rOm
            bloomB := B, gRate := g, c, kappa := kappaE
            k1Ser := (lit 0, lit 0), k1Cf := (lit 0, lit 0), fSer := (lit 0, lit 0)
            dSwitch := litF (DyadicI.toFloat dSwitchD)
            invA := #[], cfB := #[], cfN := #[]
            coincident := true
            dCoef := dCoefE
            k1SerDD := k1SerDDE
            eKappa := eKappaE
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
    if nDepth > 300 then continue
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
    out := out.push {
      muSigma := vSig, muOmega := vOm
      nu1Sigma := r1Sig, nu1Omega := r1Om, nu2Sigma := r2Sig, nu2Omega := r2Om
      bloomB := B, gRate := g
      c := cmulE v.ampE (cplxLitE r1r2)
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


end Tropical.EmitArrow
