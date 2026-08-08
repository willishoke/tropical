import Tropical.EmitArrow.Modal.Realize

/-!
# EmitArrow.Modal.Forest

Deferred modal compiler values and their source-local room-folding helpers.
-/

namespace Tropical.EmitArrow

/-- The lowered modal value: a PLAIN source with its room chain DEFERRED
    (accumulated, folded at realization — the EC/DD partition site) or a
    BLOOMED source kept modal — its voice modes and the FOLDED room chain held
    SEPARATE, so the pitch bloom is crossed ONCE (`bloomCompose`) at
    realization, AFTER the rooms fold. Rooms fold by the residue calculus
    reinterpreted as filter∘filter (the two-hats reading: a bank is both a
    struck source `Σaᵢe^{λᵢd}` and a transfer function `Σaᵢ/(s−λᵢ)`, same
    data), so gong⋙reverb⋙reverb crosses each nonlinearity exactly once.

    Why the plain arm defers (fork 3′): rooms COMMUTE (filter∘filter, the
    bit-exact registered EC-commute law), so the fold order is free — and the
    partition wants hot rooms (couplings near coincidence, `couplingHot`) to
    fold LAST, where their couplings become `PairedMode` atoms that never need
    to re-compose. A cold chain folds in arrival order through
    `residueComposeEC` — byte-identical to the eager fold this replaced. -/
inductive ModalBank where
  | plain (voice : Array ModalMode) (rooms : Array (Array ModalMode))
  | bloomed (voice room : Array ModalMode) (bloomB gRate : Float)

/-- One independently timed branch of a lowered modal island. The bank is the
    spectral payload; the remaining fields describe how and where only this
    branch is realized. Keeping that metadata per branch is what prevents a
    modal fan-in from borrowing its first input's strike anchor or clock. -/
structure ModalBranch where
  bank : ModalBank
  strikeAnchor : Sig
  realizationClock : Clock
  addressNode? : Option String
  direction? : Option ModalDir
  modeCount? : Option Sig

/-- The compiler value carried by a modal edge. Authored order is semantic:
    effects map over it and a Modal→Sig seam realizes and sums it left-to-right. -/
abbrev ModalForest := Array ModalBranch

/-- Structural equality for the metadata that permits a legacy pole union.
    The bank is deliberately excluded: compatible plain banks are concatenated
    after their deferred rooms fold. -/
def ModalBranch.sameRealizationMetadata (a b : ModalBranch) : Bool :=
  a.strikeAnchor == b.strikeAnchor &&
  a.realizationClock == b.realizationClock &&
  a.addressNode? == b.addressNode? &&
  a.direction? == b.direction? &&
  a.modeCount? == b.modeCount?

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
