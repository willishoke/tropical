import Tropical.EmitArrow.Modal.RoomProfile
import Tropical.EmitArrow.Modal.Residue

/-!
# Graph-specialized grouped-room reference

This is the deliberately internal Plan-6 M5 reference.  It specializes one
source-independent `RoomProfile` against one arbitrary admitted modal island,
materializing forward and cyclic-future prefix tables as coefficient arrays.
It is an executable semantic oracle and cost probe, not a public node, asset
format, accepted acoustic profile, or production batching decision.
-/

namespace Tropical.EmitArrow

open Tropical.Ir
open Tropical.Exact (DyadicI CplxDI)

inductive RoomExclusionKind where
  | invalidProfileIdentity
  | unsupportedEvaluator
  | invalidSampleRate
  | sampleRateMismatch
  | invalidPoleDomain
  | branchCapacity
  | groupCapacity
  | invalidCarrier
  | sourceCapacity
  | generatedScalarCapacity
  | unsupportedDegree
  | nonconstantPole
  | nonfinitePole
  | poleOutOfDomain
  | prefixComputation
deriving Repr, DecidableEq

/-- One indexed reason the complete specialization request was refused. -/
structure RoomExclusion where
  kind : RoomExclusionKind
  row? : Option Nat := none
  group? : Option Nat := none
  detail : String
deriving Repr, DecidableEq

/-- Admission is all-or-nothing.  Every independently detectable exclusion is
    reported, and no partially generated specialization escapes. -/
structure RoomRefusal where
  exclusions : Array RoomExclusion
deriving Repr, DecidableEq

def RoomExclusion.summary (x : RoomExclusion) : String :=
  let where_ := match x.row?, x.group? with
    | some r, some g => s!"row {r}, group {g}: "
    | some r, none => s!"row {r}: "
    | none, some g => s!"group {g}: "
    | none, none => ""
  s!"{where_}{x.detail}"

def RoomRefusal.summary (x : RoomRefusal) : String :=
  String.intercalate "; " (x.exclusions.toList.map RoomExclusion.summary)

/-- Runtime metadata derived from a room-only carrier group. -/
structure RoomReferenceGroup where
  id : String
  period : Nat
  prefixOffset : Nat
  logRadius : Sig

/-- Source-dependent cache material, named separately from the profile that
    produced it.  Pole coordinates/residues and generated prefixes live here,
    never in `RoomProfile`. -/
structure RoomReferenceSpecialization where
  private mk ::
  profileId : String
  profileVersion : Nat
  evaluatorVersion : String
  sampleRate : Nat
  modes : Array ModalMode
  groups : Array RoomReferenceGroup
  prefixStride : Nat
  forwardPrefix : Array CplxE
  reversePrefix : Array CplxE
  generatedScalarCount : Nat

private def exclusion (kind : RoomExclusionKind) (detail : String)
    (row? group? : Option Nat := none) : RoomExclusion :=
  { kind, row?, group?, detail }

private def jsonD (n : Lean.JsonNumber) : DyadicI := DyadicI.ofJsonNumber n

/-- The evaluator emits exact, twelve-place decimal literals and refuses any
    generated coefficient outside this fixed envelope.  The bound is part of
    the v1 evaluator contract: prefix growth can never reach `litF`'s lossy
    UInt64 path or hand an unbounded coefficient to the runtime. -/
private def roomPrefixMagnitudeLimit : DyadicI := DyadicI.ofNat 1000000

private def withinPrefixEnvelope (x : DyadicI) : Bool :=
  x.ok && Dyadic.ble (DyadicI.abs x).hi roomPrefixMagnitudeLimit.lo

/-- Deterministic exact-carrier emission.  The mantissa is computed in
    unbounded `Int` arithmetic, after the fixed evaluator envelope is checked;
    no intermediate `Float` or `UInt64` can saturate. -/
private def emittedD? (x : DyadicI) : BuildM (Option Sig) := do
  if withinPrefixEnvelope x then
    let mantissa := Dyadic.toDecimalMantissa (DyadicI.mid x) 12
    -- `JsonNumber`'s textual form for zero with a decimal exponent is not a
    -- valid JSON numeral (`0.e-…`); canonical zero has no exponent.
    let value ← if mantissa == 0 then lit 0 else lit mantissa 12
    pure (some value)
  else pure none

private def containsI (lo hi x : DyadicI) : Bool :=
  lo.ok && hi.ok && x.ok && Dyadic.ble lo.lo x.lo && Dyadic.ble x.hi hi.hi

private def validDomain (d : RoomPoleDomain) : Bool :=
  let f0 := jsonD d.minFrequencyHz
  let f1 := jsonD d.maxFrequencyHz
  let s0 := jsonD d.minSigma
  let s1 := jsonD d.maxSigma
  f0.ok && f1.ok && s0.ok && s1.ok &&
    !DyadicI.certLt f0 DyadicI.zero && DyadicI.certLt f0 f1 &&
    DyadicI.certGt s0 DyadicI.zero && DyadicI.certLt s0 s1

private inductive SourcePoleRead where
  | nonconstant
  | nonfinite
  | value (sigma omega : DyadicI)

private def sourcePoleRead (m : ModalMode) : BuildM SourcePoleRead := do
  match ← sigConstD? m.sigma, ← sigConstD? m.omega with
  | none, _ | _, none => pure .nonconstant
  | some sigma, some omega =>
    if sigma.ok && omega.ok then pure (.value sigma omega) else pure .nonfinite

private def sourcePole? (m : ModalMode) : BuildM (Option (DyadicI × DyadicI)) := do
  match ← sourcePoleRead m with
  | .value sigma omega => pure (some (sigma, omega))
  | _ => pure none

private def collectAdmissionExclusions (profile : RoomProfile)
    (sampleRate : Nat) (modes : Array ModalMode) : BuildM (Array RoomExclusion) := do
  let mut out : Array RoomExclusion := #[]
  if profile.id.isEmpty || profile.profileVersion == 0 then
    out := out.push (exclusion .invalidProfileIdentity
      "profile id must be nonempty and profileVersion must be positive")
  if profile.evaluatorVersion != groupedRoomReferenceEvaluatorVersion then
    out := out.push (exclusion .unsupportedEvaluator
      s!"unsupported evaluator '{profile.evaluatorVersion}'")
  if profile.sampleRate == 0 then
    out := out.push (exclusion .invalidSampleRate "profile sample rate must be positive")
  if sampleRate != profile.sampleRate then
    out := out.push (exclusion .sampleRateMismatch
      s!"profile requires {profile.sampleRate} Hz; specialization requested {sampleRate} Hz")
  if !validDomain profile.admission.poles then
    out := out.push (exclusion .invalidPoleDomain
      "pole bounds must be finite ordered ranges with sigma > 0 and frequency >= 0")
  if profile.admission.maxBranches < 1 then
    out := out.push (exclusion .branchCapacity
      "the one-island reference requires maxBranches >= 1")
  if profile.groups.isEmpty then
    out := out.push (exclusion .invalidCarrier "profile must contain at least one carrier group")
  if profile.groups.size > profile.admission.maxCarrierGroups then
    out := out.push (exclusion .groupCapacity
      s!"{profile.groups.size} groups exceed capacity {profile.admission.maxCarrierGroups}")

  let mut seenIds : Array String := #[]
  let mut prefixStride := 0
  for (group, gi) in profile.groups.zipIdx do
    let radius := jsonD group.radius
    if group.id.isEmpty || seenIds.contains group.id then
      out := out.push (exclusion .invalidCarrier
        "carrier group ids must be nonempty and unique" none (some gi))
    else
      seenIds := seenIds.push group.id
    if group.period == 0 || group.period != group.carrier.size then
      out := out.push (exclusion .invalidCarrier
        s!"period {group.period} does not match carrier length {group.carrier.size}"
        none (some gi))
    if group.period > profile.admission.maxPeriod then
      out := out.push (exclusion .invalidCarrier
        s!"period {group.period} exceeds capacity {profile.admission.maxPeriod}"
        none (some gi))
    if !radius.ok || !DyadicI.certGt radius DyadicI.zero ||
        !DyadicI.certLt radius DyadicI.one then
      out := out.push (exclusion .invalidCarrier
        "radius must be finite and strictly inside (0, 1)" none (some gi))
    for sample in group.carrier do
      if !(jsonD sample).ok then
        out := out.push (exclusion .invalidCarrier
          "carrier samples must be finite decimals" none (some gi))
    prefixStride := prefixStride + group.period

  if modes.size > profile.admission.maxSourceRows then
    out := out.push (exclusion .sourceCapacity
      s!"{modes.size} source rows exceed capacity {profile.admission.maxSourceRows}")
  let scalarCount := 4 * modes.size * prefixStride
  if scalarCount > profile.admission.maxGeneratedScalars || scalarCount > 4294967295 then
    out := out.push (exclusion .generatedScalarCapacity
      s!"{scalarCount} generated scalars exceed the admitted/Plan-6 limit")

  let fLo := jsonD profile.admission.poles.minFrequencyHz
  let fHi := jsonD profile.admission.poles.maxFrequencyHz
  let sLo := jsonD profile.admission.poles.minSigma
  let sHi := jsonD profile.admission.poles.maxSigma
  -- Admission uses the same authored decimal constant as `ModalMode.hz`.
  -- That makes the documented inclusive Hz boundary attainable exactly.
  let twoPi := jsonD ⟨6283185307179586, 15⟩
  for (mode, row) in modes.zipIdx do
    if mode.deg != 0 then
      out := out.push (exclusion .unsupportedDegree
        s!"degree {mode.deg} is unsupported; the reference admits degree zero"
        (some row))
    match ← sourcePoleRead mode with
    | .nonconstant =>
      out := out.push (exclusion .nonconstantPole
        "sigma and omega must be build-time constants" (some row))
    | .nonfinite =>
      out := out.push (exclusion .nonfinitePole
        "sigma and omega must be finite" (some row))
    | .value sigma omega =>
      let omegaAbs := DyadicI.abs omega
      let omegaLo := DyadicI.mul fLo twoPi
      let omegaHi := DyadicI.mul fHi twoPi
      if !containsI sLo sHi sigma || !containsI omegaLo omegaHi omegaAbs then
        out := out.push (exclusion .poleOutOfDomain
          s!"source pole is outside profile '{profile.id}' admission" (some row))
  return out

private def exactPolePerSample (sigma omega : DyadicI) (sampleRate : Nat) : CplxDI :=
  let rate := DyadicI.ofNat sampleRate
  CplxDI.exp ⟨DyadicI.neg (DyadicI.div sigma rate), DyadicI.div omega rate⟩

private def emittedCplx? (z : CplxDI) : BuildM (Option CplxE) := do
  if !z.ok then return none
  let some re ← emittedD? z.re | return none
  let some im ← emittedD? z.im | return none
  pure (some (re, im))

/-- Prefix law computed entirely in the exact bake carrier:
    `A[r] = Σ_{k=0}^r carrier[k]·(radius/pole)^k` and
    `B[r] = Σ_{s=0}^{P-1} carrier[(r+s)%P]·(pole·radius)^s`. -/
private def specializePrefixPair (mode : ModalMode) (group : RoomCarrierGroup)
    (sampleRate : Nat) : BuildM (Option (Array CplxE × Array CplxE)) := do
  let some (sigma, omega) ← sourcePole? mode | return none
  let radius := jsonD group.radius
  let pole := exactPolePerSample sigma omega sampleRate
  let ratio := CplxDI.div (CplxDI.ofI radius) pole
  let futureStep := CplxDI.scale radius pole
  if !ratio.ok || !futureStep.ok then return none
  let forwardD : Array CplxDI := Id.run do
    let mut out : Array CplxDI := #[]
    let mut acc := CplxDI.zero
    let mut power := CplxDI.one
    for k in [0:group.period] do
      let c := jsonD group.carrier[k]!
      acc := CplxDI.add acc (CplxDI.scale c power)
      out := out.push acc
      power := CplxDI.mul power ratio
    return out
  let reverseD : Array CplxDI := Id.run do
    let mut out : Array CplxDI := #[]
    for r in [0:group.period] do
      let mut acc := CplxDI.zero
      let mut power := CplxDI.one
      for s in [0:group.period] do
        let c := jsonD group.carrier[(r + s) % group.period]!
        acc := CplxDI.add acc (CplxDI.scale c power)
        power := CplxDI.mul power futureStep
      out := out.push acc
    return out
  let mut forward : Array CplxE := #[]
  for value in forwardD do
    let some emitted ← emittedCplx? value | return none
    forward := forward.push emitted
  let mut reverse : Array CplxE := #[]
  for value in reverseD do
    let some emitted ← emittedCplx? value | return none
    reverse := reverse.push emitted
  pure (some (forward, reverse))

/-- Specialize one complete modal island against a source-independent profile.
    Prefixes depend on `(sigma, omega)` only; `cre`, `cim`, and the later anchor
    remain live graph data. -/
def specializeGroupedRoomReference (profile : RoomProfile) (sampleRate : Nat)
    (modes : Array ModalMode) : BuildM (Except RoomRefusal RoomReferenceSpecialization) := do
  let exclusions ← collectAdmissionExclusions profile sampleRate modes
  if !exclusions.isEmpty then return .error { exclusions }
  let mut groups : Array RoomReferenceGroup := #[]
  let mut prefixStride := 0
  for group in profile.groups do
    let radius := jsonD group.radius
    let logRadius := DyadicI.log radius
    let logRadiusE ← emittedD? logRadius
    if !logRadius.ok || logRadiusE.isNone then
      return .error { exclusions := #[exclusion .prefixComputation
        "the radius logarithm failed or exceeded the v1 coefficient envelope"
        none (some groups.size)] }
    groups := groups.push {
      id := group.id
      period := group.period
      prefixOffset := prefixStride
      logRadius := logRadiusE.get! }
    prefixStride := prefixStride + group.period

  let mut forwardPrefix : Array CplxE := #[]
  let mut reversePrefix : Array CplxE := #[]
  for (mode, row) in modes.zipIdx do
    for (group, gi) in profile.groups.zipIdx do
      match ← specializePrefixPair mode group sampleRate with
      | none =>
        return .error { exclusions := #[exclusion .prefixComputation
          "exact prefix generation failed" (some row) (some gi)] }
      | some (forward, reverse) =>
        forwardPrefix := forwardPrefix ++ forward
        reversePrefix := reversePrefix ++ reverse
  let generatedScalarCount := 2 * (forwardPrefix.size + reversePrefix.size)
  pure (.ok {
    profileId := profile.id
    profileVersion := profile.profileVersion
    evaluatorVersion := profile.evaluatorVersion
    sampleRate
    modes
    groups
    prefixStride
    forwardPrefix
    reversePrefix
    generatedScalarCount })

private def floorE (x : Sig) : BuildM Sig := unary .floor x
private def ceilE (x : Sig) : BuildM Sig := unary .ceil x
private def sqrtE (x : Sig) : BuildM Sig := unary .sqrt x
private def ltE (a b : Sig) : BuildM Sig := binary .lt a b
private def gteE (a b : Sig) : BuildM Sig := binary .gte a b
private def floorDivE (a b : Sig) : BuildM Sig := binary .floorDiv a b
private def modE (a b : Sig) : BuildM Sig := binary .mod a b

/-- Longer local polynomials keep the executable reference close to its direct
    analytic oracle without changing the incumbent modal-bank kernels. -/
private def roomSinSig (x : Sig) : BuildM Sig := do
  let invPi ← lit 3183098861837907 16
  let turns ← mul x invPi
  let n ← roundE turns
  let pi ← lit 3141592653589793 15
  let nPi ← mul n pi
  let r ← sub x nPi
  let one ← lit 1
  let two ← lit 2
  let oddMask ← lit 1
  let parity ← bitAnd n oddMask
  let doubledParity ← mul two parity
  let sign ← sub one doubledParity
  let z ← mul r r
  let coefficients : Array (Int × Nat) := #[
    (281145725434552076, 32), (-764716373181981648, 30),
    (160590438368216146, 27), (-250521083854417188, 25),
    (275573192239858907, 23), (-198412698412698413, 21),
    (833333333333333333, 20), (-166666666666666667, 18)]
  let first ← lit coefficients[0]!.1 coefficients[0]!.2
  let p ← (coefficients.extract 1 coefficients.size).foldlM (fun acc coefficient => do
    let product ← mul z acc
    let value ← lit coefficient.1 coefficient.2
    add value product) first
  let zp ← mul z p
  let polynomial ← add one zp
  let sine ← mul r polynomial
  mul sign sine

private def roomCosSig (x : Sig) : BuildM Sig := do
  let halfPi ← halfPiE
  let shifted ← add x halfPi
  roomSinSig shifted

private def roomExpSig (x : Sig) : BuildM Sig := do
  let lo ← lit (-87)
  let hi ← lit 88
  let clamped ← clampE x lo hi
  let invLn2 ← lit 14426950408889634 16
  let scaled ← mul clamped invLn2
  let n ← roundE scaled
  let ln2Hi ← lit 693145751953125 15
  let highPart ← mul n ln2Hi
  let remainder ← sub clamped highPart
  let ln2Lo ← lit 14286068203094173 22
  let lowPart ← mul n ln2Lo
  let r ← sub remainder lowPart
  let coefficients : Array (Int × Nat) := #[
    (208767569878680989, 26), (250521083854417188, 25),
    (275573192239858907, 24), (275573192239858907, 23),
    (248015873015873016, 22), (198412698412698413, 21),
    (138888888888888889, 20), (833333333333333333, 20),
    (416666666666666667, 19), (166666666666666667, 18), (5, 1)]
  let first ← lit coefficients[0]!.1 coefficients[0]!.2
  let p ← (coefficients.extract 1 coefficients.size).foldlM (fun acc coefficient => do
    let product ← mul r acc
    let value ← lit coefficient.1 coefficient.2
    add value product) first
  let rp ← mul r p
  let one ← lit 1
  let inner ← add one rp
  let rInner ← mul r inner
  let mantissa ← add one rInner
  ldexpE mantissa n

private def cscaleE (s : Sig) (z : CplxE) : BuildM CplxE := do
  let real ← mul s z.1
  let imag ← mul s z.2
  pure (real, imag)

private def cselectE (c : Sig) (a b : CplxE) : BuildM CplxE := do
  let real ← selectE c a.1 b.1
  let imag ← selectE c a.2 b.2
  pure (real, imag)

/-- `exp((-sigma+i*omega)t/Fs)`, with `t` in samples.  The caller supplies a
    clock already made relative in integer Q32.32, so far absolute seeks do not
    spend phase or envelope mantissa bits. -/
private def roomPoleExpSamples (bakedRate sigma omega t : Sig) : BuildM CplxE := do
  let sec ← div t bakedRate
  let sigmaTime ← mul sigma sec
  let negative ← neg sigmaTime
  let env ← roomExpSig negative
  let phase ← mul omega sec
  let cosine ← roomCosSig phase
  let real ← mul env cosine
  let sine ← roomSinSig phase
  let imag ← mul env sine
  pure (real, imag)

private def radiusPow (logRadius exponent : Sig) : BuildM Sig := do
  let power ← mul logRadius exponent
  roomExpSig power

private def complexTableRead (reTable imTable sourceIdx groupOffset residue
    sourceStride : Sig) : BuildM CplxE := do
  let sourceBase ← mul sourceIdx sourceStride
  let groupBase ← add sourceBase groupOffset
  let i ← add groupBase residue
  let real ← index reTable i
  let imag ← index imTable i
  pure (real, imag)

private def groupedRoomReferencePair (sourceIdx groupIdx : Sig)
    (bakedRate sigma omega cre cim position u : Sig)
    (sourceStride periods logRadii offsets forwardRe forwardIm reverseRe reverseIm : Sig) : BuildM Sig := do
  let periodValue ← index periods groupIdx
  let periodI ← toIntE periodValue
  let periodF ← toFloatE periodI
  let logRadius ← index logRadii groupIdx
  let offsetValue ← index offsets groupIdx
  let groupOffset ← toIntE offsetValue
  let radiusP ← radiusPow logRadius periodF
  let ep ← roomPoleExpSamples bakedRate sigma omega periodF
  let zP ← cscaleE radiusP ep
  let oneReal ← lit 1
  let zeroImag ← lit 0
  let one : CplxE := (oneReal, zeroImag)
  let invEp ← cdivE one ep
  let g ← cscaleE radiusP invEp

  let causal ← gteE u zeroImag
  let flooredU ← floorE u
  let causalM ← selectE causal flooredU zeroImag
  let m ← toIntE causalM
  let qI ← floorDivE m periodI
  let rI ← modE m periodI
  let q ← toFloatE qI
  let aR ← complexTableRead forwardRe forwardIm sourceIdx groupOffset rI sourceStride
  let oneInt ← litI 1
  let lastIndex ← sub periodI oneInt
  let aLast ← complexTableRead forwardRe forwardIm sourceIdx groupOffset lastIndex sourceStride
  let eu ← roomPoleExpSamples bakedRate sigma omega u
  let pq ← mul periodF q
  let radiusQ ← radiusPow logRadius pq
  let negativePq ← neg pq
  let poleQ ← roomPoleExpSamples bakedRate sigma omega negativePq
  let gq ← cscaleE radiusQ poleQ
  let euGq ← cmulE eu gq
  let causalDenomReal ← sub oneReal g.1
  let causalDenomImag ← neg g.2
  let causalDenom : CplxE := (causalDenomReal, causalDenomImag)
  let denomRealSquared ← mul causalDenom.1 causalDenom.1
  let denomImagSquared ← mul causalDenom.2 causalDenom.2
  let denomSquared ← add denomRealSquared denomImagSquared
  let denomMagnitude ← sqrtE denomSquared
  let nearThreshold ← lit 1 10
  let nearOne ← ltE denomMagnitude nearThreshold
  let qEu ← cscaleE q eu
  let directNumerator ← csubE eu euGq
  let directSum ← cdivE directNumerator causalDenom
  let sQ ← cselectE nearOne qEu directSum
  let lastSQ ← cmulE aLast sQ
  let residueTerm ← cmulE aR euGq
  let forwardNormal ← caddE lastSQ residueTerm
  let lastLimit ← cmulE aLast qEu
  let residueLimit ← cmulE aR eu
  let forwardLimit ← caddE lastLimit residueLimit
  let forwardResponse ← cselectE nearOne forwardLimit forwardNormal
  let forwardCre ← mul cre forwardResponse.1
  let forwardCim ← mul cim forwardResponse.2
  let forwardDifference ← sub forwardCre forwardCim
  let forwardReal ← selectE causal forwardDifference zeroImag

  let negativeU ← neg u
  let d0 ← ceilE negativeU
  let positiveD ← gt d0 zeroImag
  let selectedD ← selectE positiveD d0 zeroImag
  let d ← toIntE selectedD
  let dF ← toFloatE d
  let residueD ← modE d periodI
  let b ← complexTableRead reverseRe reverseIm sourceIdx groupOffset residueD sourceStride
  let reverseDenomReal ← sub oneReal zP.1
  let reverseDenomImag ← neg zP.2
  let reverseDenom : CplxE := (reverseDenomReal, reverseDenomImag)
  let reverseTime ← add u dF
  let reversePhase ← roomPoleExpSamples bakedRate sigma omega reverseTime
  let reverseProduct ← cmulE reversePhase b
  let radiusD ← radiusPow logRadius dF
  let reverseNumerator ← cscaleE radiusD reverseProduct
  let reverseResponse ← cdivE reverseNumerator reverseDenom
  let reverseCre ← mul cre reverseResponse.1
  let reverseCim ← mul cim reverseResponse.2
  let reverseReal ← sub reverseCre reverseCim

  let negativeOne ← lit (-1)
  let p ← clampE position negativeOne oneReal
  let onePlusP ← add oneReal p
  let half ← lit 5 1
  let forwardPower ← mul half onePlusP
  let forwardGain ← sqrtE forwardPower
  let oneMinusP ← sub oneReal p
  let reversePower ← mul half oneMinusP
  let reverseGain ← sqrtE reversePower
  let forwardOutput ← mul forwardGain forwardReal
  let reverseOutput ← mul reverseGain reverseReal
  add forwardOutput reverseOutput

/-- Evaluate one graph specialization.  The room's prefix tables are Plan-6
    coefficient arrays; source/group trip counts affect data, not expression
    size. -/
def groupedRoomReferenceSig (specialization : RoomReferenceSpecialization)
    (clk anchor position : Sig) : BuildM Sig := do
  if specialization.modes.isEmpty || specialization.groups.isEmpty then return ← lit 0
  let bakedRate ← lit specialization.sampleRate
  let sigma ← arr (specialization.modes.map (·.sigma))
  let omega ← arr (specialization.modes.map (·.omega))
  let cre ← arr (specialization.modes.map (·.cre))
  let cim ← arr (specialization.modes.map (·.cim))
  let periodValues ← specialization.groups.mapM fun group => lit group.period
  let periods ← arr periodValues
  let logRadii ← arr (specialization.groups.map (·.logRadius))
  let offsetValues ← specialization.groups.mapM fun group => lit group.prefixOffset
  let offsets ← arr offsetValues
  let forwardRe ← arr (specialization.forwardPrefix.map (·.1))
  let forwardIm ← arr (specialization.forwardPrefix.map (·.2))
  let reverseRe ← arr (specialization.reversePrefix.map (·.1))
  let reverseIm ← arr (specialization.reversePrefix.map (·.2))
  let sourceStride ← lit specialization.prefixStride
  let clkRel ← relClockQ clk anchor
  let clkFloat ← toFloatE clkRel
  let twoPow32 ← lit 4294967296
  let u ← div clkFloat twoPow32
  let sourceIdx ← loopIdx 4300
  let groupIdx ← loopIdx 4301
  let columns := #[sigma, omega, cre, cim, periods, logRadii, offsets,
    forwardRe, forwardIm, reverseRe, reverseIm]
  let sigmaValue ← index sigma sourceIdx
  let omegaValue ← index omega sourceIdx
  let creValue ← index cre sourceIdx
  let cimValue ← index cim sourceIdx
  let pair ← groupedRoomReferencePair sourceIdx groupIdx
      bakedRate sigmaValue omegaValue creValue cimValue
      position u sourceStride periods logRadii offsets
      forwardRe forwardIm reverseRe reverseIm
  let inner ← bankSum specialization.groups.size columns pair none 4301
  let room ← bankSum specialization.modes.size columns inner none 4300
  -- Prefixes and pole evolution are baked against one rate.  A plan with a
  -- different runtime rate fails closed instead of combining two time bases.
  let runtimeRate ← sampleRate
  let sameRate ← binary .eq runtimeRate bakedRate
  let zero ← lit 0
  selectE sameRate room zero

/-- Arrow terminal over the caller's clock rail.  POSITION is live; room amount
    remains an external gain. -/
def groupedRoomReferenceTerm (specialization : RoomReferenceSpecialization)
    (anchor position : Sig) (clk : Clock) : ArrowTerm :=
  .arrUn (fun c => groupedRoomReferenceSig specialization c anchor position) (.clk clk)

end Tropical.EmitArrow
