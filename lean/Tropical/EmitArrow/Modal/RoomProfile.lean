import Tropical.Exact.Cplx

/-!
# Source-independent grouped-room profiles

This module is deliberately data-only.  A `RoomProfile` describes periodic
room carriers and the finite source/capacity envelope in which the internal
Plan-6 reference may specialize them.  Source poles, residues, strike anchors,
generated prefix tables, and rendered samples do not belong here.

This is not an asset ABI or a public patch node.  It is the smallest executable
M5 contract needed to prove that one unchanged room description can accept
arbitrary admitted modal coordinates before a packaged format is chosen.
-/

namespace Tropical.EmitArrow

/-- Stable decimal carrier input.  Keeping authored room data as `JsonNumber`
    lets admission and prefix generation enter the exact interval carrier
    without a platform `Float` decision. -/
structure RoomCarrierGroup where
  id : String
  period : Nat
  radius : Lean.JsonNumber
  carrier : Array Lean.JsonNumber
deriving Repr, DecidableEq

/-- Inclusive source-pole domain for a source-generic room specialization.
    Frequency is stated in Hz at the profile boundary; modal rows continue to
    carry their native `omega` in rad/s. -/
structure RoomPoleDomain where
  minFrequencyHz : Lean.JsonNumber
  maxFrequencyHz : Lean.JsonNumber
  minSigma : Lean.JsonNumber
  maxSigma : Lean.JsonNumber
deriving Repr, DecidableEq

/-- Whole-request resource bounds.  M5's reference consumes one timed island,
    but `maxBranches` is recorded now so a future forest evaluator cannot
    silently reinterpret a one-branch profile as an unbounded batch contract. -/
structure RoomAdmission where
  poles : RoomPoleDomain
  maxBranches : Nat
  maxSourceRows : Nat
  maxCarrierGroups : Nat
  maxPeriod : Nat
  maxGeneratedScalars : Nat
deriving Repr, DecidableEq

/-- The only live room control admitted by the M5 reference. -/
inductive RoomPositionLaw where
  | equalPowerForwardReverse
deriving Repr, DecidableEq

/-- Carrier values retain their authored scale.  Room/send amount stays an
    external live gain rather than becoming source data in the profile. -/
inductive RoomNormalization where
  | preserveCarrierScale
deriving Repr, DecidableEq

/-- A source-independent room identity plus its admission/capacity contract.
    There is intentionally no source row, anchor, timeline, prefix, wet score,
    capture, or rendered-content field. -/
structure RoomProfile where
  id : String
  profileVersion : Nat
  evaluatorVersion : String
  sampleRate : Nat
  groups : Array RoomCarrierGroup
  positionLaw : RoomPositionLaw := .equalPowerForwardReverse
  normalization : RoomNormalization := .preserveCarrierScale
  admission : RoomAdmission
deriving Repr, DecidableEq

/-- Evaluator identity pinned by the first graph-specialized reference. -/
def groupedRoomReferenceEvaluatorVersion : String :=
  "grouped-prefix-reference-v1"

end Tropical.EmitArrow
