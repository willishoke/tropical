import Tropical.EmitArrow.Modal.Residue
import Tropical.Plan

/-!
# GroupedRoom — the frozen Clouds room-position terminal

The production profile is deliberately closed: twelve source poles, twelve
groups, one native-rate immutable prefix payload.  The tables enter a patch as
two root array inputs and are bound to Plan-6 immutable storage after session
slot allocation.  No prefix data is represented as `Sig.arr`, copied into the
coefficient epoch, or rebuilt in the audio callback.
-/

namespace Tropical.EmitArrow

open Tropical.Ir
open Tropical.Exact (DyadicI)

def groupedRoomProfile : String := "clouds-current-radii-mono-v1"
def groupedRoomAssetPath : String :=
  "playground/assets/grouped-room/clouds-current-radii-mono-v1-44100.tgrm"
def groupedRoomAssetBytes : Nat := 2717376
def groupedRoomAssetSha256 : String :=
  "838019933ddc885cb519ae0ba40233ee5d3e95cce4e8951ca566c8f1f5f65986"
def groupedRoomSampleRate : Nat := 44100
def groupedRoomPrefixElementCount : Nat := 339600
def groupedRoomForwardByteOffset : Nat := 576
def groupedRoomReverseByteOffset : Nat := 1358976

def groupedRoomForwardInputName : String := "__groupedroom_forward_prefix"
def groupedRoomReverseInputName : String := "__groupedroom_reverse_prefix"

def groupedRoomCacheProfile : String := "clouds-current-radii-mono-v1-scene-cache"
def groupedRoomCacheAssetPath : String :=
  "playground/assets/grouped-room/clouds-current-radii-mono-v1-scene-44100.f32le"
def groupedRoomCacheAssetBytes : Nat := 5644800
def groupedRoomCacheAssetSha256 : String :=
  "22b534e561aa1fef8aa4535ff321ee5df90c9cfb2743c274fc44d183216a615e"
def groupedRoomCacheElementCount : Nat := 705600
def groupedRoomCacheForwardInputName : String := "__groupedroom_scene_forward"
def groupedRoomCacheReverseInputName : String := "__groupedroom_scene_reverse"

/-- The only large values in the grouped-room expression are external array
    inputs.  They are intentionally default-less: failure to bind the Plan-6
    asset is a compile/load error, never a silent zero-room fallback. -/
def groupedRoomInputDecls : Array AInputDecl := #[
  { name := groupedRoomForwardInputName
    type? := some (.array .float #[(groupedRoomPrefixElementCount : Nat)]) },
  { name := groupedRoomReverseInputName
    type? := some (.array .float #[(groupedRoomPrefixElementCount : Nat)]) }
]

/-- The release-Mac reserve fallback is exactly two fixed-scene mono bases.
    It remains Plan-6 immutable data; only the graph specialization changes. -/
def groupedRoomCacheInputDecls : Array AInputDecl := #[
  { name := groupedRoomCacheForwardInputName
    type? := some (.array .float #[(groupedRoomCacheElementCount : Nat)]) },
  { name := groupedRoomCacheReverseInputName
    type? := some (.array .float #[(groupedRoomCacheElementCount : Nat)]) }
]

def groupedRoomForwardTable : Sig := .inputRef ⟨0⟩
def groupedRoomReverseTable : Sig := .inputRef ⟨1⟩

def groupedRoomCacheForwardTable : Sig := .inputRef ⟨0⟩
def groupedRoomCacheReverseTable : Sig := .inputRef ⟨1⟩

private def floorE (x : Sig) : Sig := .unary .floor x
private def ceilE (x : Sig) : Sig := .unary .ceil x
private def sqrtE (x : Sig) : Sig := .unary .sqrt x
private def ltE (a b : Sig) : Sig := .binary .lt a b
private def gteE (a b : Sig) : Sig := .binary .gte a b
private def floorDivE (a b : Sig) : Sig := .binary .floorDiv a b
private def modE (a b : Sig) : Sig := .binary .mod a b

/-- Grouped-room-only high-accuracy transcendental kernels.  The incumbent
    modal polynomials intentionally retain their frozen audio bytes; this
    terminal needs the tighter asset-oracle gate, so it carries longer series
    without changing any existing voice. -/
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

private def roomCosSig (x : Sig) : Sig :=
  roomSinSig (add x halfPiE)

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

/-- Exact-rational phase reduction for this fixed-coordinate profile.  Reducing
    `(frequency * floor(t)) mod Fs` in i64 before returning to float keeps the
    Metal argument bounded without quantizing the source frequency; the
    fractional remainder stays analytic. -/
private def roomPhaseSamples (frequency t : Sig) : Sig :=
  let whole := toIntE (floorE t)
  let fraction := sub t (toFloatE whole)
  let freqI := toIntE frequency
  let rateI := toIntE (.sampleRate : Sig)
  let integralRemainder := modE (mul freqI whole) rateI
  let cycles := div
    (add (toFloatE integralRemainder) (mul frequency fraction)) .sampleRate
  mul twoPiE cycles

/-- `exp((-σ+i2πf)t/Fs)`, where `t` is in samples. -/
private def poleExpSamples (sigma frequency t : Sig) : CplxE :=
  let sec := div t .sampleRate
  let env := roomExpSig (neg (mul sigma sec))
  let phase := roomPhaseSamples frequency t
  (mul env (roomCosSig phase), mul env (roomSinSig phase))

private def radiusPow (logRadius exponent : Sig) : Sig :=
  roomExpSig (mul logRadius exponent)

private def complexTableRead (table sourceIdx groupOffset residue : Sig) : CplxE :=
  let scalarBase := add (add (mul sourceIdx (litI 14150)) groupOffset) residue
  let floatBase := mul (litI 2) scalarBase
  (Sig.index table floatBase, Sig.index table (add floatBase (litI 1)))

private def groupedRoomPair (sourceIdx groupIdx : Sig)
    (sigma frequency cre cim position u : Sig)
    (periods logRadii offsets forward reverse : Sig) : Sig :=
  let periodI := toIntE (Sig.index periods groupIdx)
  let periodF := toFloatE periodI
  let logRadius := Sig.index logRadii groupIdx
  let groupOffset := toIntE (Sig.index offsets groupIdx)
  let radiusP := radiusPow logRadius periodF

  -- Share the expensive pole evaluations across both arms.  From
  -- Z = r^P exp(eP), the causal ratio is G = r^P / exp(eP); one evaluation at
  -- +P therefore supplies both denominators.
  let ep := poleExpSamples sigma frequency periodF
  let zP := cscaleE radiusP ep
  let invEp := cdivE (lit 1, lit 0) ep
  let g := cscaleE radiusP invEp

  -- Causal: discrete quotient/remainder comes from floor(u), while the analytic
  -- exponential retains the original fractional coordinate.  Unsupported
  -- negative coordinates use m=0 for safe table addressing, then gate to zero.
  let causal := gteE u (lit 0)
  let m := toIntE (selectE causal (floorE u) (lit 0))
  let qI := floorDivE m periodI
  let rI := modE m periodI
  let q := toFloatE qI
  let aR := complexTableRead forward sourceIdx groupOffset rI
  let aLast := complexTableRead forward sourceIdx groupOffset (sub periodI (litI 1))
  -- A_R S(q+1) + (A_last-A_R) S(q)
  --   = A_last S(q) + A_R exp(eu) G^q.
  -- This exact rearrangement replaces two complete geometric-block
  -- evaluations with one shared exp(eu) and one G^q evaluation.
  let eu := poleExpSamples sigma frequency u
  let pq := mul periodF q
  let gq := cscaleE (radiusPow logRadius pq)
    (poleExpSamples sigma frequency (neg pq))
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

  -- Reverse: d=max(ceil(-u),0), preserving real u in exp(e(u+d)).
  let d0 := ceilE (neg u)
  let d := toIntE (selectE (gt d0 (lit 0)) d0 (lit 0))
  let dF := toFloatE d
  let b := complexTableRead reverse sourceIdx groupOffset (modE d periodI)
  let reverseDenom : CplxE := (sub (lit 1) zP.1, neg zP.2)
  let reversePhase := poleExpSamples sigma frequency (add u dF)
  let reverseNumerator := cscaleE (radiusPow logRadius dF) (cmulE reversePhase b)
  let reverseResponse := cdivE reverseNumerator reverseDenom
  let reverseReal := sub (mul cre reverseResponse.1) (mul cim reverseResponse.2)

  let p := clampE position (lit (-1)) (lit 1)
  let forwardGain := sqrtE (mul (lit 5 1) (add (lit 1) p))
  let reverseGain := sqrtE (mul (lit 5 1) (sub (lit 1) p))
  add (mul forwardGain forwardReal) (mul reverseGain reverseReal)

private def groupedRoomPeriods : Sig := Sig.arr #[
  lit 1116, lit 1188, lit 1277, lit 1356, lit 1422, lit 1491,
  lit 1557, lit 1617, lit 1112, lit 882, lit 682, lit 450]

private def groupedRoomLogRadii : Sig := Sig.arr #[
  lit (-391596104250758995) 22, lit (-391596104250758995) 22,
  lit (-391596104250758995) 22, lit (-391596104250758995) 22,
  lit (-391596104250758995) 22, lit (-391596104250758995) 22,
  lit (-391596104250758995) 22, lit (-391596104250758995) 22,
  lit (-642212780902766032) 21, lit (-808786720949492303) 21,
  lit (-104782085305192129) 20, lit (-158780215155733453) 20]

private def groupedRoomOffsets : Sig := Sig.arr #[
  lit 0, lit 1116, lit 2304, lit 3581, lit 4937, lit 6359,
  lit 7850, lit 9407, lit 11024, lit 12136, lit 13018, lit 13700]

/-- One nested 12×12 reduction in frozen source/group order.  Forward and
    reverse are combined inside the pair body, so POSITION does not duplicate
    either reduction topology. -/
def groupedRoomSig (modes : Array ModalMode) (clk anchor position : Sig) : Sig :=
  let sigma := Sig.arr (modes.map (·.sigma))
  let frequency := Sig.arr #[
    lit 211, lit 211, lit 433, lit 433, lit 887, lit 887,
    lit 1511, lit 1511, lit 2837, lit 2837, lit 5081, lit 5081]
  let cre := Sig.arr (modes.map (·.cre))
  let cim := Sig.arr (modes.map (·.cim))
  let periods := groupedRoomPeriods
  let logRadii := groupedRoomLogRadii
  let offsets := groupedRoomOffsets
  let forward := groupedRoomForwardTable
  let reverse := groupedRoomReverseTable
  let u := sub (div (toFloatE clk) (lit 4294967296)) anchor
  let sourceIdx := Sig.loopIdx 4100
  let groupIdx := Sig.loopIdx 4101
  let inner := Sig.bankSum 12 #[periods, logRadii, offsets, forward, reverse]
    (groupedRoomPair sourceIdx groupIdx
      (Sig.index sigma sourceIdx) (Sig.index frequency sourceIdx)
      (Sig.index cre sourceIdx) (Sig.index cim sourceIdx)
      position u periods logRadii offsets forward reverse)
    none 4101
  Sig.bankSum 12 #[sigma, frequency, cre, cim, periods, logRadii, offsets, forward, reverse]
    inner none 4100

/-- The terminal stays on the clock rail, so master seek/scrub and downstream
    stateless warps retain the ordinary Arrow semantics. -/
def groupedRoomTerm (modes : Array ModalMode) (anchor position : Sig)
    (clk : Clock) : ArrowTerm :=
  .arrUn (fun c => groupedRoomSig modes c anchor position) (.clk clk)

/-- Cyclic linear read for the exact 16-second scene basis.  FLOW may land at
    fractional or negative scene coordinates; wrapping preserves the demo's
    fixed loop and interpolation avoids integer-address stepping. -/
private def groupedRoomCacheRead (table coordinate : Sig) : Sig :=
  let count := lit groupedRoomCacheElementCount
  let wrapped := sub coordinate (mul count (floorE (div coordinate count)))
  let baseF := floorE wrapped
  let base := toIntE baseF
  let next := selectE (gteE base (litI (groupedRoomCacheElementCount - 1)))
    (litI 0) (add base (litI 1))
  let fraction := sub wrapped baseF
  let a := Sig.index table base
  let b := Sig.index table next
  add a (mul fraction (sub b a))

def groupedRoomCacheSig (clk position : Sig) : Sig :=
  let coordinate := div (toFloatE clk) (lit 4294967296)
  let forward := groupedRoomCacheRead groupedRoomCacheForwardTable coordinate
  let reverse := groupedRoomCacheRead groupedRoomCacheReverseTable coordinate
  let p := clampE position (lit (-1)) (lit 1)
  let forwardGain := sqrtE (mul (lit 5 1) (add (lit 1) p))
  let reverseGain := sqrtE (mul (lit 5 1) (sub (lit 1) p))
  add (mul forwardGain forward) (mul reverseGain reverse)

def groupedRoomCacheTerm (position : Sig) (clk : Clock) : ArrowTerm :=
  .arrUn (fun c => groupedRoomCacheSig c position) (.clk clk)

private def expectedOmega (frequency : Nat) : Sig :=
  let twoPi := DyadicI.ofJsonNumber ⟨6283185307179586, 15⟩
  litF (DyadicI.toFloat (DyadicI.mul twoPi (DyadicI.ofNat frequency)))

private def expectedSigma (mantissa : Int) (exponent : Nat) : Sig :=
  litF (DyadicI.toFloat (DyadicI.ofJsonNumber ⟨mantissa, exponent⟩))

private def sameConst (a b : Sig) : Bool :=
  match sigConstF? a, sigConstF? b with
  | some x, some y => x == y
  | _, _ => false

/-- Refuse a profile/table mismatch at compile time.  Amplitudes and phases are
    intentionally free per hit; pole coordinates and order are not. -/
def validateGroupedRoomModes (modes : Array ModalMode) : Except String Unit := do
  let expected : Array (Nat × Int × Nat) := #[
    (211, 75, 1), (211, 955, 1), (433, 865, 2), (433, 10035, 2),
    (887, 98, 1), (887, 1052, 1), (1511, 1095, 2), (1511, 11005, 2),
    (2837, 121, 1), (2837, 1149, 1), (5081, 1325, 2), (5081, 11975, 2)]
  if modes.size != expected.size then
    throw s!"groupedroom profile '{groupedRoomProfile}' requires exactly 12 source poles; got {modes.size}"
  for i in [0:expected.size] do
    let some m := modes[i]?
      | throw s!"groupedroom profile '{groupedRoomProfile}' source pole {i} is missing"
    let (frequency, sigmaMantissa, sigmaExponent) := expected[i]!
    if m.deg != 0 || !sameConst m.sigma (expectedSigma sigmaMantissa sigmaExponent)
        || !sameConst m.omega (expectedOmega frequency) then
      throw s!"groupedroom profile '{groupedRoomProfile}' source pole {i} does not match the frozen (frequency, sigma) coordinate; regenerate the asset for a changed instrument"

/-- Resolve the two package inputs to their allocated slots and attach the one
    immutable asset descriptor. -/
def bindGroupedRoomAsset (plan : Tropical.Plan.FlatPlan) :
    Except String Tropical.Plan.FlatPlan := do
  let forwardName := s!"__root__.{groupedRoomForwardInputName}"
  let reverseName := s!"__root__.{groupedRoomReverseInputName}"
  let some forwardSlot := plan.arraySlotNames.idxOf? forwardName
    | throw s!"groupedroom: immutable input slot '{forwardName}' was not allocated"
  let some reverseSlot := plan.arraySlotNames.idxOf? reverseName
    | throw s!"groupedroom: immutable input slot '{reverseName}' was not allocated"
  if plan.arraySlotSizes[forwardSlot]? != some groupedRoomPrefixElementCount
      || plan.arraySlotSizes[reverseSlot]? != some groupedRoomPrefixElementCount then
    throw "groupedroom: immutable prefix slot size does not match the frozen profile"
  let asset : Tropical.Plan.ImmutableAsset := {
    path := groupedRoomAssetPath
    byteCount := groupedRoomAssetBytes
    sha256 := groupedRoomAssetSha256
    sampleRate := groupedRoomSampleRate
    arrays := #[
      { slot := forwardSlot, byteOffset := groupedRoomForwardByteOffset,
        elementCount := groupedRoomPrefixElementCount },
      { slot := reverseSlot, byteOffset := groupedRoomReverseByteOffset,
        elementCount := groupedRoomPrefixElementCount }] }
  pure { plan with immutableAssets := plan.immutableAssets.push asset }

def bindGroupedRoomCacheAsset (plan : Tropical.Plan.FlatPlan) :
    Except String Tropical.Plan.FlatPlan := do
  let forwardName := s!"__root__.{groupedRoomCacheForwardInputName}"
  let reverseName := s!"__root__.{groupedRoomCacheReverseInputName}"
  let some forwardSlot := plan.arraySlotNames.idxOf? forwardName
    | throw s!"groupedroomcache: immutable input slot '{forwardName}' was not allocated"
  let some reverseSlot := plan.arraySlotNames.idxOf? reverseName
    | throw s!"groupedroomcache: immutable input slot '{reverseName}' was not allocated"
  if plan.arraySlotSizes[forwardSlot]? != some groupedRoomCacheElementCount
      || plan.arraySlotSizes[reverseSlot]? != some groupedRoomCacheElementCount then
    throw "groupedroomcache: immutable scene slot size does not match the frozen profile"
  let asset : Tropical.Plan.ImmutableAsset := {
    path := groupedRoomCacheAssetPath
    byteCount := groupedRoomCacheAssetBytes
    sha256 := groupedRoomCacheAssetSha256
    sampleRate := groupedRoomSampleRate
    arrays := #[
      { slot := forwardSlot, byteOffset := 0,
        elementCount := groupedRoomCacheElementCount },
      { slot := reverseSlot, byteOffset := groupedRoomCacheElementCount * 4,
        elementCount := groupedRoomCacheElementCount }] }
  pure { plan with immutableAssets := plan.immutableAssets.push asset }

end Tropical.EmitArrow
