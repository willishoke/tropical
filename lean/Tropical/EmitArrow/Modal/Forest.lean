import Tropical.EmitArrow.Modal.Kernel

/-!
# EmitArrow.Modal.Forest

Deferred modal compiler values and their authored-order stage spine.
-/

namespace Tropical.EmitArrow

/-- A room kernel is either already expressed as modal modes (internal
    fixtures) or constructed from one live control after all controls
    have been frozen at the terminal observation coordinate. -/
inductive ModalRoomKernel where
  | fixed (modes : Array ModalMode)
  | controlled (control : ModalControlRef) (build : Sig → Array ModalMode)

/-- One ordinary room operator.  Direction belongs to this kernel, never to the
    source prefix or the complete branch.  Sway likewise belongs to the room's
    decay law and therefore travels with the stage that authored it. -/
structure OrdinaryRoomStage where
  kernel : ModalRoomKernel
  direction : ModalControlRef := ModalControlRef.constant (lit 0)
  sway? : Option (ModalControlRef × ModalControlRef) := none
  /-- Product-facing temporal direction needs a perceptual output correction:
      its exact anti-causal arm lives mostly before a normal playback strike.
      Low-level oriented kernels keep their unscaled mathematical meaning. -/
  normalizeDirectionLevel : Bool := false

/-- Modal→Modal operators in authored order.  Ordinary rooms retain their
    bloom-bridge metadata; every other linear construction enters through the
    generic factor-preserving stage.  Gauges remain separate because they are
    nonlinear in the complete current modal value. -/
inductive ModalStage where
  | ordinaryRoom (room : OrdinaryRoomStage)
  | linear (stage : ModalLinearStage)
  | gauge (control : ModalControlRef)

/-- The deferred source representation.  Rooms and gauges live exclusively on
    `ModalBranch.stages`, so heterogeneous ordering is never encoded by mutating
    or prematurely folding this payload. -/
inductive ModalSource where
  | plain (voice : Array ModalMode)
  | bloomed (voice : Array ModalMode) (bloomB gRate : Float)

/-- One independently timed branch of a lowered modal island. The bank is the
    spectral payload; the remaining fields describe how and where only this
    branch is realized. Keeping that metadata per branch is what prevents a
    modal fan-in from borrowing its first input's strike anchor or clock. -/
structure ModalBranch where
  source : ModalSource
  stages : Array ModalStage := #[]
  strikeAnchor : Sig
  realizationClock : Clock
  addressNode? : Option String
  modeCount? : Option Sig

/-- The compiler value carried by a modal edge. Authored order is semantic:
    effects map over it and a Modal→Sig seam realizes and sums it left-to-right. -/
abbrev ModalForest := Array ModalBranch

/-- Control dependencies in their terminal binding order. -/
def ModalRoomKernel.controls : ModalRoomKernel → Array ModalControlRef
  | .fixed _ => #[]
  | .controlled control _ => #[control]

def OrdinaryRoomStage.controls (room : OrdinaryRoomStage) : Array ModalControlRef :=
  room.kernel.controls ++ #[room.direction] ++
    (room.sway?.map (fun (sway, rate) => #[sway, rate])).getD #[]

def ModalStage.controls : ModalStage → Array ModalControlRef
  | .ordinaryRoom room => room.controls
  | .linear stage => stage.controls
  | .gauge control => #[control]

def ModalBranch.controls (branch : ModalBranch) : Array ModalControlRef :=
  branch.stages.foldl (fun controls stage => controls ++ stage.controls) #[]

/-- The COLLECTED chain fold, arrival order — today's exact expressions (the
    byte-identity comparator, and the fallback for the surfaces that need a
    plain `Array ModalMode`: gauge, mix, direction). -/
def foldRoomsEC (voice : Array ModalMode) (rooms : Array (Array ModalMode)) :
    Array ModalMode :=
  rooms.foldl residueComposeEC voice

/-- The PARTITIONED chain fold: cold rooms fold first in arrival order
    (collected, verbatim), rooms carrying a hot coupling fold LAST, and only
    the final fold partitions — so `PairedMode`s form once and never
    re-compose. When no room is hot this is `foldRoomsEC` VERBATIM.

    THE DETECTION INVARIANT (stated precisely, not hedged): heat is decided
    twice — the sort HEURISTIC here reads source-UNIT poles/amps; the
    partition re-decides per coupling on the COMPOSED amps, but only at the
    final fold. The two predicates cohere exactly this far:
    · θ_acc (accuracy-lens) heat is a pole-distance test — amp-independent —
      and composition never moves poles, so the sort CANNOT miss it; every
      near-coincident coupling reaches the final fold and partitions.
    · Rail-lens heat is amp-dependent, and composed ringing amps carry `1/Δ`
      factors from earlier folds — so a coupling can be rail-cold on unit
      amps yet rail-hot on composed amps. If it forms BEFORE the final fold
      it folds collected, silently, at the status-quo floor. That is: **amp-
      dependent heat is only detected at the final fold.** The exposure is
      narrow today — the rail lens's routed service region is itself cap-
      bounded to well-damped couplings (`ecddRailCeil` doc) — but it WIDENS
      if the cap or lens is ever retuned; re-examine this invariant then.
    A multi-hot residual (a hot coupling forming before the final fold — a
    deliberate triple/multi-unison) likewise stays collected at the
    status-quo floor, served gracefully, not silently wrong. -/
def foldRoomsPartitioned (voice : Array ModalMode)
    (rooms : Array (Array ModalMode)) :
    Array ModalMode × Array PairedMode :=
  if rooms.isEmpty then (voice, #[]) else
  let units := #[voice] ++ rooms
  let roomHot := fun (j : Nat) (room : Array ModalMode) =>
    room.any fun q => units.zipIdx.any fun (u, i) =>
      i != j + 1 && u.any fun p => couplingHot p q || couplingHot q p
  if (rooms.zipIdx.all fun (r, j) => !roomHot j r) then
    (foldRoomsEC voice rooms, #[])
  else
    let cold := rooms.zipIdx.filterMap fun (r, j) =>
      if roomHot j r then none else some r
    let hot := rooms.zipIdx.filterMap fun (r, j) =>
      if roomHot j r then some r else none
    let ordered := cold ++ hot
    let front := ordered.pop
    let last := ordered.back!
    residueComposePartitioned (foldRoomsEC voice front) last

end Tropical.EmitArrow
