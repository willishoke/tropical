import Tropical.EmitArrow.Modal.Block
import Tropical.EmitArrow.Modal.Forest
import Tropical.Testing.ArrowFixtures

/-!
# Block-carrier algebra gates (slice Phase 1)

Symbolic-layer witnesses for `EmitArrow.Block`: constant `Sig` coefficients
are folded back from the frozen arena and compared against the two exact
oracles the algebra must reduce to.

* SINGLETON LAW — when no cluster forms, the block coefficients ARE the
  collected residues (`foldRoomsEC`), up to reassociation (product of sums
  versus sum of products — the same value in exact arithmetic, so a relative
  tolerance at the f64 floor, never byte identity).
* CONFLUENCE LAW — an identity-coincident voice/room pole forms one size-2
  block whose Newton coefficients are exactly the `Oriented` coincident
  algebra's degree-1 and degree-0 amplitudes at that pole (`exp[z,z] = d·e^{zd}`).
* TOTALITY — a reversed room yields past rows; an over-cap cluster splits into
  value classes at the collected floor; spellings of one value are one node.

No realizer participates: Phase 2 owns the render.
-/

namespace Tropical.Tropicaltest.Block

open Tropical
open Tropical.EmitArrow
open Tropical.EmitArrow.Block
open Tropical.Ir

private def passGate (label detail : String) : IO Bool := do
  IO.println s!"  PASS  {label}  {detail}"
  pure true

private def failGate (label detail : String) : IO Bool := do
  IO.println s!"  FAIL  {label}  {detail}"
  pure false

private def foldCplx (constants : Array (Option Tropical.Exact.DyadicI))
    (value : CplxE) : Option (Float × Float) := do
  let re ← (sigConstDFrom? constants value.1).map Tropical.Exact.DyadicI.toFloat
  let im ← (sigConstDFrom? constants value.2).map Tropical.Exact.DyadicI.toFloat
  pure (re, im)

private def cabs (z : Float × Float) : Float := Float.sqrt (z.1 * z.1 + z.2 * z.2)
private def cdist (a b : Float × Float) : Float := cabs (a.1 - b.1, a.2 - b.2)

/-- A literal mode from (σ, ω, Re A, Im A) as decimal mantissa/exponent pairs. -/
private def litMode (sigma omega cre cim : Int × Nat) : BuildM ModalMode := do
  pure { sigma := ← lit sigma.1 sigma.2, omega := ← lit omega.1 omega.2,
         cre := ← lit cre.1 cre.2, cim := ← lit cim.1 cim.2 }

private def properOf (modes : Array ModalMode) : BuildM ModalKernelExpr := do
  pure (.proper (.oriented modes (← lit 0)))

/-- The three separated banks of the singleton law. -/
private def voice3 : BuildM (Array ModalMode) := do
  pure #[← litMode (3, 0) (6280, 1) (8, 1) (1, 1),
         ← litMode (11, 0) (14100, 1) (5, 1) (-2, 1),
         ← litMode (7, 0) (-3300, 1) (6, 1) (0, 0)]
private def room3a : BuildM (Array ModalMode) := do
  pure #[← litMode (9, 0) (9424, 1) (7, 1) (0, 0),
         ← litMode (21, 0) (-20100, 1) (4, 1) (3, 1),
         ← litMode (15, 0) (2500, 1) (9, 1) (0, 0)]
private def room3b : BuildM (Array ModalMode) := do
  pure #[← litMode (13, 0) (31400, 1) (6, 1) (-1, 1),
         ← litMode (5, 0) (-7700, 1) (3, 1) (0, 0),
         ← litMode (17, 0) (18800, 1) (8, 1) (2, 1)]

/-- SINGLETON LAW: `voice ⋙ r₁ ⋙ r₂` with every pole separated — nine singleton
    rows whose coefficients match the collected fold's residues at the same pole. -/
private def singletonLaw : IO (Except String Float) := do
  match Tropical.Testing.ArrowFixtures.freezeBuild {} do
      let voice ← voice3
      let r1 ← room3a
      let r2 ← room3b
      let collected ← foldRoomsEC voice #[r1, r2]
      let spine : ModalKernelExpr :=
        .cascade #[← properOf voice, ← properOf r1, ← properOf r2]
      let rows ← decompose spine
      pure (collected, rows) with
  | .error e => pure (.error s!"build: {e}")
  | .ok (arena, (collected, rows)) =>
    let constants := sigConstTable arena
    do
      if rows.size != 9 then return .error s!"expected 9 singleton rows, got {rows.size}"
      if rows.any (·.nodes.size != 1) then return .error "a cluster formed among separated poles"
      let mut worst := 0.0
      for row in rows do
        let some node := foldCplx constants row.nodes[0]! | return .error "node did not fold"
        let some coeff := foldCplx constants row.coeffs[0]! | return .error "coefficient did not fold"
        -- the collected mode at the same pole
        let mut matched := false
        for m in collected do
          let some pole := foldCplx constants (m.sigma, m.omega) | continue
          let poleC : Float × Float := (-pole.1, pole.2)
          if cdist poleC node < 1e-9 then
            let some amp := foldCplx constants m.ampE | return .error "collected amp did not fold"
            let rel := cdist amp coeff / max (cabs amp) 1e-300
            worst := max worst rel
            matched := true
        if !matched then return .error "a block row's pole has no collected twin"
      pure (.ok worst)

/-- CONFLUENCE LAW: the voice's first pole is the room's pole by expression
    identity (the same literals intern to the same `Sig`s). The block terminal
    must form one size-2 row `{z, z}` whose Newton coefficients equal the
    `Oriented` coincident algebra's (deg-1, deg-0) amplitudes at `z`. -/
private def confluenceLaw : IO (Except String Float) := do
  match Tropical.Testing.ArrowFixtures.freezeBuild {} do
      let shared ← litMode (9, 0) (9424, 1) (7, 1) (0, 0)
      let other ← litMode (4, 0) (25100, 1) (5, 1) (1, 1)
      let voice : Array ModalMode := #[{ shared with cre := ← lit 6 1, cim := ← lit 3 1 }, other]
      let room : Array ModalMode := #[shared]
      let zero ← lit 0
      let bank ← (← Oriented.Bank.ofFuture voice).convolveKernel room zero
        Oriented.syntacticSameSideClassifier
      let spine : ModalKernelExpr := .cascade #[← properOf voice, ← properOf room]
      let rows ← decompose spine
      pure (bank, rows) with
  | .error e => pure (.error s!"build: {e}")
  | .ok (arena, (bank, rows)) =>
    let constants := sigConstTable arena
    do
      let some pair := rows.find? (·.nodes.size == 2) | return .error "no size-2 block formed"
      if rows.size != 2 then return .error s!"expected 2 rows (pair + singleton), got {rows.size}"
      let some z := foldCplx constants pair.nodes[0]! | return .error "node did not fold"
      let some n1 := foldCplx constants pair.coeffs[0]! | return .error "n1 did not fold"
      let some n2 := foldCplx constants pair.coeffs[1]! | return .error "n2 did not fold"
      -- the oriented bank's deg-1 and deg-0 amplitudes at z
      let mut a1 : Float × Float := (0, 0)
      let mut a0 : Float × Float := (0, 0)
      for m in bank.future do
        let some sigma := (sigConstDFrom? constants m.sigma).map Tropical.Exact.DyadicI.toFloat | continue
        let some omega := (sigConstDFrom? constants m.omega).map Tropical.Exact.DyadicI.toFloat | continue
        if cdist (-sigma, omega) z < 1e-9 then
          let some amp := foldCplx constants m.ampE | return .error "oriented amp did not fold"
          if m.deg == 1 then a1 := (a1.1 + amp.1, a1.2 + amp.2)
          else if m.deg == 0 then a0 := (a0.1 + amp.1, a0.2 + amp.2)
          else return .error s!"unexpected degree {m.deg}"
      let rel1 := cdist a1 n1 / max (cabs a1) 1e-300
      let rel0 := cdist a0 n2 / max (cabs a0) 1e-300
      pure (.ok (max rel1 rel0))

/-- TOTALITY: a reversed room is admitted and yields past rows; four
    near-equal distinct-valued poles exceed the body cap and split into four
    singleton rows (the collected floor); three spellings of ONE value are one
    confluent row of multiplicity three (degree, never a divided difference). -/
private def totality : IO (Except String Unit) := do
  match Tropical.Testing.ArrowFixtures.freezeBuild {} do
      let voice ← voice3
      let room ← room3a
      let reversed : ModalKernelExpr := .cascade #[← properOf voice,
        .proper (.oriented room (← lit 1))]
      let bilateral ← decompose reversed
      let comb : Array ModalMode := #[
        ← litMode (9, 0) (9424, 1) (7, 1) (0, 0),
        ← litMode (9, 0) (94241, 2) (7, 1) (0, 0),
        ← litMode (9, 0) (94242, 2) (7, 1) (0, 0),
        ← litMode (9, 0) (94243, 2) (7, 1) (0, 0)]
      let dense ← decompose (.cascade #[← properOf voice, ← properOf comb])
      let spelled : Array ModalMode := #[
        ← litMode (9, 0) (9424, 1) (7, 1) (0, 0),
        ← litMode (90, 1) (94240, 2) (5, 1) (0, 0),
        ← litMode (900, 2) (942400, 3) (3, 1) (0, 0)]
      let confluent ← decompose (.cascade #[← properOf voice, ← properOf spelled])
      pure (bilateral, dense, confluent) with
  | .error e => pure (.error s!"build: {e}")
  | .ok (_, (bilateral, dense, confluent)) =>
    let past := bilateral.filter (·.orientation == .past)
    let future := bilateral.filter (·.orientation == .future)
    if !(past.size == 3 && future.size == 3) then
      return .error s!"reversed room: expected 3 future + 3 past rows, got {future.size} + {past.size}"
    let combRows := dense.filter (·.nodes.size == 1)
    if !(dense.size == 7 && combRows.size == 7) then
      return .error s!"dense comb: expected 7 singleton rows (3 voice + 4 split), got {dense.size}"
    let some triple := confluent.find? (·.nodes.size == 3) | return .error "spelled triple: no size-3 row"
    if !triple.confluent then return .error "spelled triple: not marked confluent"
    pure (.ok ())

private def showResult (x : Except String Float) : String :=
  match x with
  | .ok v => s!"ok {v}"
  | .error e => e

private def showUnit (x : Except String Unit) : String :=
  match x with
  | .ok () => "ok"
  | .error e => e

def runBlockAlgebra : IO Bool := do
  let singleton ← singletonLaw
  let confluence ← confluenceLaw
  let refused ← totality
  match singleton, confluence, refused with
  | .ok s, .ok c, .ok () =>
    if s < 1e-12 && c < 1e-12 then
      passGate "block-algebra"
        s!"singleton law rel {s} (block coefficients = collected residues up to reassociation); confluence law rel {c} (size-2 block = Oriented deg-1/deg-0 amps); reversed room admitted as past rows; over-cap comb splits to the collected floor; three spellings of one value are one confluent row"
    else
      failGate "block-algebra" s!"off the law: singleton rel {s}, confluence rel {c}"
  | s, c, r =>
    failGate "block-algebra"
      s!"singleton: {showResult s}; confluence: {showResult c}; refusals: {showUnit r}"

end Tropical.Tropicaltest.Block
