import Tropical.Semantics.Strata

/-!
# Strata-exit semantic fixtures

One compact arena covers shared roots, every strata-specific expression family,
nested registry conversion, first-use deduplication, and unreachable expression
and program entries.  The examples exercise the executable witness and the
public refinement theorem without choosing a particular value algebra.
-/

namespace Tropical.Testing.StrataSemantics

open Tropical.Ir
open Tropical.Ir.Core
open Tropical.Ir.Strata
open Tropical.Semantics

private structure ExprFixture where
  arena : ExprArena
  tile : ExprId
  bank : ExprId
  routed : ExprId

private def buildExprFixture : ExprFixture :=
  let action : StateM ExprArena (ExprId × ExprId × ExprId) := do
    let one ← eintern (.num ⟨1, 0⟩)
    let tileIndex ← eintern .tileSampleIndex
    let tilePhase ← eintern .tilePhase
    let tile ← eintern (.tileArray #[tileIndex, tilePhase])
    let loop ← eintern (.loopIdx 7)
    let bank ← eintern (.bankSum 2 #[tile] loop none 7)
    let routed ← eintern
      (.routedSum 2 2 #[some 0, some 1] #[tile] #[one, bank] none 9)
    -- Deliberately unreachable from every program root.
    let _ ← eintern (.num ⟨99, 0⟩)
    pure (tile, bank, routed)
  let (roots, arena) := action.run {}
  { arena, tile := roots.1, bank := roots.2.1, routed := roots.2.2 }

private def exprFixture : ExprFixture := buildExprFixture

private def childProgram : Program :=
  { name := "Child"
    inputs := #[{ name := "x" }]
    outputs := #[{ name := "out" }]
    assigns := #[{ target := .port ⟨0⟩, expr := ⟨0⟩ }] }

private def unreachableProgram : Program :=
  { name := "Dead"
    outputs := #[{ name := "unused" }]
    assigns := #[{ target := .port ⟨0⟩, expr := ⟨7⟩ }] }

private def rootProgram : Program :=
  { name := "Root"
    outputs := #[{ name := "left" }, { name := "right" }]
    decls := #[
      .inst "first" "Child" #[{ port := ⟨0⟩, value := exprFixture.bank }],
      .inst "second" "Child" #[{ port := ⟨0⟩, value := exprFixture.tile }]]
    assigns := #[
      { target := .port ⟨0⟩, expr := exprFixture.routed },
      { target := .port ⟨1⟩, expr := exprFixture.bank }]
    -- `Dead` is present in storage but no instance references it.
    registry := #[("Child", ⟨0⟩), ("Dead", ⟨1⟩)] }

private def fixtureArena : Arena :=
  { programs := #[childProgram, unreachableProgram, rootProgram]
    exprs := exprFixture.arena }

private def fixtureRoot : ProgramIdx := ⟨2⟩

private theorem fixtureExprWf : ArenaWellFormed fixtureArena.exprs :=
  semanticWfCheck_sound (by native_decide)

private theorem fixtureProgramsWf : progPoolWf fixtureArena.programs = true := by
  native_decide

private def fixtureCheck : Bool :=
    match EArena.toResolved fixtureArena fixtureRoot with
    | Except.error _ => false
    | Except.ok (dst, core) =>
      dst.nodes.size + 1 == fixtureArena.exprs.nodes.size &&
      core.registry.map (·.1) == #["Child"] &&
      dst.nodes.any (fun node => match node with | .tileArray .. => true | _ => false) &&
      dst.nodes.any (fun node => match node with | .tileSampleIndex => true | _ => false) &&
      dst.nodes.any (fun node => match node with | .tilePhase => true | _ => false) &&
      dst.nodes.any (fun node => match node with | .bankSum .. => true | _ => false) &&
      dst.nodes.any (fun node => match node with | .routedSum .. => true | _ => false)

example : fixtureCheck = true := by
  native_decide

example (alg : Algebra α) (invocation : ProgramInputs α)
    (dst : ExprArena) (core : CoreProgram)
    (hresult : EArena.toResolved fixtureArena fixtureRoot = Except.ok (dst, core)) :
    denoteProgram alg fixtureArena fixtureExprWf fixtureProgramsWf
        fixtureRoot invocation =
      denoteCoreProgram alg dst (toResolved_destination_wellFormed hresult)
        core invocation := by
  exact toResolved_preserves_denotation fixtureArena fixtureRoot dst core
    fixtureExprWf fixtureProgramsWf
    (toResolved_destination_wellFormed hresult) hresult alg invocation

end Tropical.Testing.StrataSemantics
