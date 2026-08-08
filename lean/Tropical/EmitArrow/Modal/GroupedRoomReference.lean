import Tropical.EmitArrow.Modal.RoomProfile
import Tropical.EmitArrow.Modal.Residue

/-!
# Graph-specialized grouped-room reference

This is the deliberately internal Plan-5 M5 reference.  It specializes one
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
private def emittedD? (x : DyadicI) : Option Sig :=
  if withinPrefixEnvelope x then
    let mantissa := Dyadic.toDecimalMantissa (DyadicI.mid x) 12
    -- `JsonNumber`'s textual form for zero with a decimal exponent is not a
    -- valid JSON numeral (`0.e-…`); canonical zero has no exponent.
    some (if mantissa == 0 then lit 0 else lit mantissa 12)
  else none

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

private def sourcePoleRead (m : ModalMode) : SourcePoleRead :=
  match sigConstD? m.sigma, sigConstD? m.omega with
  | none, _ | _, none => .nonconstant
  | some sigma, some omega =>
    if sigma.ok && omega.ok then .value sigma omega else .nonfinite

private def sourcePole? (m : ModalMode) : Option (DyadicI × DyadicI) :=
  match sourcePoleRead m with
  | .value sigma omega => some (sigma, omega)
  | _ => none

private def collectAdmissionExclusions (profile : RoomProfile)
    (sampleRate : Nat) (modes : Array ModalMode) : Array RoomExclusion := Id.run do
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
      s!"{scalarCount} generated scalars exceed the admitted/Plan-5 limit")

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
    match sourcePoleRead mode with
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

private def emittedCplx? (z : CplxDI) : Option CplxE :=
  if !z.ok then none else do
  let re ← emittedD? z.re
  let im ← emittedD? z.im
  pure (re, im)

/-- Prefix law computed entirely in the exact bake carrier:
    `A[r] = Σ_{k=0}^r carrier[k]·(radius/pole)^k` and
    `B[r] = Σ_{s=0}^{P-1} carrier[(r+s)%P]·(pole·radius)^s`. -/
private def specializePrefixPair (mode : ModalMode) (group : RoomCarrierGroup)
    (sampleRate : Nat) : Option (Array CplxE × Array CplxE) := do
  let (sigma, omega) ← sourcePole? mode
  let radius := jsonD group.radius
  let pole := exactPolePerSample sigma omega sampleRate
  let ratio := CplxDI.div (CplxDI.ofI radius) pole
  let futureStep := CplxDI.scale radius pole
  if !ratio.ok || !futureStep.ok then none else
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
  let forward ← forwardD.mapM emittedCplx?
  let reverse ← reverseD.mapM emittedCplx?
  pure (forward, reverse)

/-- Specialize one complete modal island against a source-independent profile.
    Prefixes depend on `(sigma, omega)` only; `cre`, `cim`, and the later anchor
    remain live graph data. -/
def specializeGroupedRoomReference (profile : RoomProfile) (sampleRate : Nat)
    (modes : Array ModalMode) : Except RoomRefusal RoomReferenceSpecialization := do
  let exclusions := collectAdmissionExclusions profile sampleRate modes
  if !exclusions.isEmpty then throw { exclusions }
  let mut groups : Array RoomReferenceGroup := #[]
  let mut prefixStride := 0
  for group in profile.groups do
    let radius := jsonD group.radius
    let logRadius := DyadicI.log radius
    let logRadiusE := emittedD? logRadius
    if !logRadius.ok || logRadiusE.isNone then
      throw { exclusions := #[exclusion .prefixComputation
        "log(radius) failed or exceeded the v1 coefficient envelope"
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
      match specializePrefixPair mode group sampleRate with
      | none =>
        throw { exclusions := #[exclusion .prefixComputation
          "exact prefix generation failed" (some row) (some gi)] }
      | some (forward, reverse) =>
        forwardPrefix := forwardPrefix ++ forward
        reversePrefix := reversePrefix ++ reverse
  let generatedScalarCount := 2 * (forwardPrefix.size + reversePrefix.size)
  pure {
    profileId := profile.id
    profileVersion := profile.profileVersion
    evaluatorVersion := profile.evaluatorVersion
    sampleRate
    modes
    groups
    prefixStride
    forwardPrefix
    reversePrefix
    generatedScalarCount }

private def floorE (x : Sig) : Sig := .unary .floor x
private def ceilE (x : Sig) : Sig := .unary .ceil x
private def sqrtE (x : Sig) : Sig := .unary .sqrt x
private def ltE (a b : Sig) : Sig := .binary .lt a b
private def gteE (a b : Sig) : Sig := .binary .gte a b
private def floorDivE (a b : Sig) : Sig := .binary .floorDiv a b
private def modE (a b : Sig) : Sig := .binary .mod a b

/-- Longer local polynomials keep the executable reference close to its direct
    analytic oracle without changing the incumbent modal-bank kernels. -/
private def roomSinSig (x : Sig) : Sig :=
  let n := roundE (mul x (lit 3183098861837907 16))
  let r := sub x (mul n (lit 3141592653589793 15))
  let sign := sub (lit 1) (mul (lit 2) (bitAnd n (lit 1)))
  let z := mul r r
  let p :=
    add (lit (-166666666666666667) 18) (mul z (
    add (lit 833333333333333333 20) (mul z (
    add (lit (-198412698412698413) 21) (mul z (
    add (lit 275573192239858907 23) (mul z (
    add (lit (-250521083854417188) 25) (mul z (
    add (lit 160590438368216146 27) (mul z (
    add (lit (-764716373181981648) 30) (mul z
        (lit 281145725434552076 32))))))))))))))
  mul sign (mul r (add (lit 1) (mul z p)))

private def roomCosSig (x : Sig) : Sig := roomSinSig (add x halfPiE)

private def roomExpSig (x : Sig) : Sig :=
  let clamped := clampE x (lit (-87)) (lit 88)
  let n := roundE (mul clamped (lit 14426950408889634 16))
  let r := sub (sub clamped (mul n (lit 693145751953125 15)))
                            (mul n (lit 14286068203094173 22))
  let p :=
    add (lit 5 1) (mul r (
    add (lit 166666666666666667 18) (mul r (
    add (lit 416666666666666667 19) (mul r (
    add (lit 833333333333333333 20) (mul r (
    add (lit 138888888888888889 20) (mul r (
    add (lit 198412698412698413 21) (mul r (
    add (lit 248015873015873016 22) (mul r (
    add (lit 275573192239858907 23) (mul r (
    add (lit 275573192239858907 24) (mul r (
    add (lit 250521083854417188 25) (mul r
        (lit 208767569878680989 26))))))))))))))))))))
  ldexpE (add (lit 1) (mul r (add (lit 1) (mul r p)))) n

private def cscaleE (s : Sig) (z : CplxE) : CplxE :=
  (mul s z.1, mul s z.2)

private def cselectE (c : Sig) (a b : CplxE) : CplxE :=
  (selectE c a.1 b.1, selectE c a.2 b.2)

/-- `exp((-sigma+i*omega)t/Fs)`, with `t` in samples.  The caller supplies a
    clock already made relative in integer Q32.32, so far absolute seeks do not
    spend phase or envelope mantissa bits. -/
private def roomPoleExpSamples (bakedRate sigma omega t : Sig) : CplxE :=
  let sec := div t bakedRate
  let env := roomExpSig (neg (mul sigma sec))
  let phase := mul omega sec
  (mul env (roomCosSig phase), mul env (roomSinSig phase))

private def radiusPow (logRadius exponent : Sig) : Sig :=
  roomExpSig (mul logRadius exponent)

private def complexTableRead (reTable imTable sourceIdx groupOffset residue
    sourceStride : Sig) : CplxE :=
  let i := add (add (mul sourceIdx sourceStride) groupOffset) residue
  (Sig.index reTable i, Sig.index imTable i)

private def groupedRoomReferencePair (sourceIdx groupIdx : Sig)
    (bakedRate sigma omega cre cim position u : Sig)
    (sourceStride periods logRadii offsets forwardRe forwardIm reverseRe reverseIm : Sig) : Sig :=
  let periodI := toIntE (Sig.index periods groupIdx)
  let periodF := toFloatE periodI
  let logRadius := Sig.index logRadii groupIdx
  let groupOffset := toIntE (Sig.index offsets groupIdx)
  let radiusP := radiusPow logRadius periodF
  let ep := roomPoleExpSamples bakedRate sigma omega periodF
  let zP := cscaleE radiusP ep
  let invEp := cdivE (lit 1, lit 0) ep
  let g := cscaleE radiusP invEp

  let causal := gteE u (lit 0)
  let m := toIntE (selectE causal (floorE u) (lit 0))
  let qI := floorDivE m periodI
  let rI := modE m periodI
  let q := toFloatE qI
  let aR := complexTableRead forwardRe forwardIm sourceIdx groupOffset rI sourceStride
  let aLast := complexTableRead forwardRe forwardIm sourceIdx groupOffset
    (sub periodI (litI 1)) sourceStride
  let eu := roomPoleExpSamples bakedRate sigma omega u
  let pq := mul periodF q
  let gq := cscaleE (radiusPow logRadius pq)
    (roomPoleExpSamples bakedRate sigma omega (neg pq))
  let euGq := cmulE eu gq
  let causalDenom : CplxE := (sub (lit 1) g.1, neg g.2)
  let nearOne := ltE
    (sqrtE (add (mul causalDenom.1 causalDenom.1)
      (mul causalDenom.2 causalDenom.2)))
    (lit 1 10)
  let sQ := cselectE nearOne (cscaleE q eu)
    (cdivE (csubE eu euGq) causalDenom)
  let forwardNormal := caddE (cmulE aLast sQ) (cmulE aR euGq)
  let forwardLimit := caddE (cmulE aLast (cscaleE q eu)) (cmulE aR eu)
  let forwardResponse := cselectE nearOne forwardLimit forwardNormal
  let forwardReal := selectE causal
    (sub (mul cre forwardResponse.1) (mul cim forwardResponse.2)) (lit 0)

  let d0 := ceilE (neg u)
  let d := toIntE (selectE (gt d0 (lit 0)) d0 (lit 0))
  let dF := toFloatE d
  let b := complexTableRead reverseRe reverseIm sourceIdx groupOffset
    (modE d periodI) sourceStride
  let reverseDenom : CplxE := (sub (lit 1) zP.1, neg zP.2)
  let reversePhase := roomPoleExpSamples bakedRate sigma omega (add u dF)
  let reverseNumerator := cscaleE (radiusPow logRadius dF)
    (cmulE reversePhase b)
  let reverseResponse := cdivE reverseNumerator reverseDenom
  let reverseReal := sub (mul cre reverseResponse.1) (mul cim reverseResponse.2)

  let p := clampE position (lit (-1)) (lit 1)
  let forwardGain := sqrtE (mul (lit 5 1) (add (lit 1) p))
  let reverseGain := sqrtE (mul (lit 5 1) (sub (lit 1) p))
  add (mul forwardGain forwardReal) (mul reverseGain reverseReal)

/-- Evaluate one graph specialization.  The room's prefix tables are Plan-5
    coefficient arrays; source/group trip counts affect data, not expression
    size. -/
def groupedRoomReferenceSig (specialization : RoomReferenceSpecialization)
    (clk anchor position : Sig) : Sig :=
  if specialization.modes.isEmpty || specialization.groups.isEmpty then lit 0 else
  let bakedRate := lit specialization.sampleRate
  let sigma := Sig.arr (specialization.modes.map (·.sigma))
  let omega := Sig.arr (specialization.modes.map (·.omega))
  let cre := Sig.arr (specialization.modes.map (·.cre))
  let cim := Sig.arr (specialization.modes.map (·.cim))
  let periods := Sig.arr (specialization.groups.map fun g => lit g.period)
  let logRadii := Sig.arr (specialization.groups.map (·.logRadius))
  let offsets := Sig.arr (specialization.groups.map fun g => lit g.prefixOffset)
  let forwardRe := Sig.arr (specialization.forwardPrefix.map (·.1))
  let forwardIm := Sig.arr (specialization.forwardPrefix.map (·.2))
  let reverseRe := Sig.arr (specialization.reversePrefix.map (·.1))
  let reverseIm := Sig.arr (specialization.reversePrefix.map (·.2))
  let sourceStride := lit specialization.prefixStride
  let clkRel := relClockQ clk anchor
  let u := div (toFloatE clkRel) (lit 4294967296)
  let sourceIdx := Sig.loopIdx 4300
  let groupIdx := Sig.loopIdx 4301
  let columns := #[sigma, omega, cre, cim, periods, logRadii, offsets,
    forwardRe, forwardIm, reverseRe, reverseIm]
  let inner := Sig.bankSum specialization.groups.size columns
    (groupedRoomReferencePair sourceIdx groupIdx
      bakedRate
      (Sig.index sigma sourceIdx) (Sig.index omega sourceIdx)
      (Sig.index cre sourceIdx) (Sig.index cim sourceIdx)
      position u sourceStride periods logRadii offsets
      forwardRe forwardIm reverseRe reverseIm)
    none 4301
  let room := Sig.bankSum specialization.modes.size columns inner none 4300
  -- Prefixes and pole evolution are baked against one rate.  A plan with a
  -- different runtime rate fails closed instead of combining two time bases.
  selectE (.binary .eq .sampleRate bakedRate) room (lit 0)

/-- Arrow terminal over the caller's clock rail.  POSITION is live; room amount
    remains an external gain. -/
def groupedRoomReferenceTerm (specialization : RoomReferenceSpecialization)
    (anchor position : Sig) (clk : Clock) : ArrowTerm :=
  .arrUn (fun c => groupedRoomReferenceSig specialization c anchor position) (.clk clk)

end Tropical.EmitArrow
