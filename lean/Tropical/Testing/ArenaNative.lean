import Tropical.EmitArrow.Sig
import Tropical.EmitArrow.ArenaSig
import Tropical.Ir.Strata
import Tropical.Ir.CompileResolved
import Tropical.Testing.PlanWire

/-!
# Arena-native authoring phase-1 fixture

This module keeps a deliberately small old/new corpus while both authoring
representations coexist.  It compares the production plan wire exactly (not
just by hash), pins unique-node counts, checks the authored and reachable
arenas, and exercises an instance input, nested output, explicit sharing,
`bankSum`, and `routedSum` ordering.
-/

namespace Tropical.Testing.ArenaNative

open Tropical.Ir

private abbrev NativeBody := Tropical.EmitArrow.ArenaNative.ProgramBody

structure FixtureCorpus where
  arena : Arena
  leaf : ProgramIdx
  root : ProgramIdx

structure Phase1Evidence where
  programCount : Nat
  authoredUniqueNodes : Nat
  leafReachableNodes : Nat
  rootReachableNodes : Nat
  leafWireBytes : Nat
  rootWireBytes : Nat
deriving Repr

private def output : Array OutputDecl :=
  #[{ name := "out", type? := some (.scalar .float) }]

/-! The frozen recursive-authoring baseline.  Expressions and declaration
    order intentionally mirror the arena-native fixture below. -/
private def buildLegacyCorpus : Except String FixtureCorpus := do
  let (arena, leaf) := Tropical.EmitArrow.assemble {} "Phase1Leaf" #[] output
    #[] #[(.port ⟨0⟩, Tropical.EmitArrow.lit 7)] #[]

  let identityInputs : Array Tropical.EmitArrow.AInputDecl := #[{
    name := "x"
    type? := some (.scalar .float)
    defaultSig := some (Tropical.EmitArrow.lit 0)
  }]
  let (arena, identity) := Tropical.EmitArrow.assemble arena "Phase1Identity"
    identityInputs output #[]
    #[(.port ⟨0⟩, .inputRef ⟨0⟩)] #[]

  let two := Tropical.EmitArrow.lit 2
  let three := Tropical.EmitArrow.lit 3
  let shared := Tropical.EmitArrow.add two three
  let table : Tropical.EmitArrow.Sig := .arr #[two, three]
  let loop : Tropical.EmitArrow.Sig := .loopIdx 7
  let indexed : Tropical.EmitArrow.Sig := .index table loop
  let body := Tropical.EmitArrow.mul indexed shared
  let bank : Tropical.EmitArrow.Sig := .bankSum 2 #[table] body none 7
  let instanceDecl : Tropical.EmitArrow.AInst := {
    name := "identity"
    programName := "Phase1Identity"
    inputs := #[{ port := ⟨0⟩, value := shared }]
  }
  let nested : Tropical.EmitArrow.Sig := .nestedOut ⟨0⟩ ⟨0⟩
  let result := Tropical.EmitArrow.add nested bank
  let (arena, root) := Tropical.EmitArrow.assemble arena "Phase1Root" #[]
    output #[instanceDecl] #[(.port ⟨0⟩, result)]
    #[("Phase1Identity", identity)]
  pure { arena, leaf, root }

/-! The ID-native vertical slice.  `shared` is bound once and reused; the
    second equal `add` must observe an intern hit.  The routed reduction and
    the `99 + 99` expression are deliberately dead so the production exit can
    demonstrate reachable-only copying. -/
private def buildNativeCorpus : Except String FixtureCorpus := do
  let (arena, leaf) ← Tropical.EmitArrow.ArenaNative.assemble {}
      "Phase1Leaf" output #[] do
    let seven ← Tropical.EmitArrow.ArenaNative.lit 7
    pure ({ assigns := #[(.port ⟨0⟩, seven)] } : NativeBody)

  let (arena, identity) ← Tropical.EmitArrow.ArenaNative.assemble arena
      "Phase1Identity" output #[] do
    let zero ← Tropical.EmitArrow.ArenaNative.lit 0
    let input ← Tropical.EmitArrow.ArenaNative.inputRef ⟨0⟩
    pure ({
      inputs := #[{
        name := "x"
        type? := some (.scalar .float)
        defaultSig := some zero
      }]
      assigns := #[(.port ⟨0⟩, input)]
    } : NativeBody)

  let (arena, root) ← Tropical.EmitArrow.ArenaNative.assemble arena
      "Phase1Root" output #[("Phase1Identity", identity)] do
    let two ← Tropical.EmitArrow.ArenaNative.lit 2
    let three ← Tropical.EmitArrow.ArenaNative.lit 3
    let shared ← Tropical.EmitArrow.ArenaNative.add two three
    let equalAgain ← Tropical.EmitArrow.ArenaNative.add two three
    unless shared == equalAgain do
      throw "ArenaNative phase-1 fixture: equal construction missed eintern"

    let loop ← Tropical.EmitArrow.ArenaNative.loopIdx 7
    let table ← Tropical.EmitArrow.ArenaNative.arr #[two, three]
    let indexed ← Tropical.EmitArrow.ArenaNative.index table loop
    let contribution ← Tropical.EmitArrow.ArenaNative.mul indexed shared
    let bank ← Tropical.EmitArrow.ArenaNative.bankSum
      2 #[table] contribution none 7

    -- Constructor/order coverage for the routed sibling.  It is dead at the
    -- root on purpose; the copied reachable graph must omit it.
    let routed ← Tropical.EmitArrow.ArenaNative.routedSum
      2 2 #[some 1, none] #[table] #[shared, indexed] none 7
    let builder ← get
    unless builder.exprs.deref routed == some
        (.routedSum 2 2 #[some 1, none] #[table]
          #[shared, indexed] none 7) do
      throw "ArenaNative phase-1 fixture: routedSum changed authored order"

    let ninetyNine ← Tropical.EmitArrow.ArenaNative.lit 99
    let _dead ← Tropical.EmitArrow.ArenaNative.add ninetyNine ninetyNine

    let instanceIdx ← Tropical.EmitArrow.ArenaNative.inst
      "identity" "Phase1Identity" #[{ port := ⟨0⟩, value := shared }]
    let nested ← Tropical.EmitArrow.ArenaNative.nestedOut instanceIdx ⟨0⟩
    let result ← Tropical.EmitArrow.ArenaNative.add nested bank
    pure ({ assigns := #[(.port ⟨0⟩, result)] } : NativeBody)

  pure { arena, leaf, root }

private def resolvedWire (corpus : FixtureCorpus) (idx : ProgramIdx) :
    Except String (ExprArena × String) := do
  let (exprs, core) ← (Tropical.Ir.Strata.runResolved
    { inlineNested := true } corpus.arena idx).mapError (·.message)
  let plan ← Tropical.Ir.CompileResolved.compileResolved core exprs
  let wire ← plan.toWire
  pure (exprs, wire.compress)

private def failureIsAtomic : Bool :=
  let original : Arena := {}
  let failed := Tropical.EmitArrow.ArenaNative.assemble original
    "Phase1Refusal" output #[] do
      let _ ← Tropical.EmitArrow.ArenaNative.lit 1
      throw "phase-1 refusal"
  match failed with
  | .error message =>
      message == "phase-1 refusal" &&
        original.programs.isEmpty && original.exprs.nodes.isEmpty
  | .ok _ => false

/-- Pure acceptance evidence for the phase-1 vertical slice.  Exact node
    counts make explicit sharing and reachability observable; exact wire
    equality is stronger than a fixture-local golden hash. -/
def phase1Evidence : Except String Phase1Evidence := do
  let legacy ← buildLegacyCorpus
  let native ← buildNativeCorpus

  unless native.arena.programs.size == 3 do
    throw s!"ArenaNative phase-1 fixture: expected 3 programs, got {native.arena.programs.size}"
  unless native.arena.exprs.nodes.size == 16 do
    throw s!"ArenaNative phase-1 fixture: expected 16 authored unique nodes, got {native.arena.exprs.nodes.size}"
  unless native.arena.exprs.wf do
    throw "ArenaNative phase-1 fixture: authored expression arena is not child-descending"
  unless failureIsAtomic do
    throw "ArenaNative phase-1 fixture: failed build changed publication behavior"

  let (legacyLeafExprs, legacyLeafWire) ← resolvedWire legacy legacy.leaf
  let (nativeLeafExprs, nativeLeafWire) ← resolvedWire native native.leaf
  unless nativeLeafExprs.nodes.size == 1 do
    throw s!"ArenaNative phase-1 fixture: expected 1 reachable leaf node, got {nativeLeafExprs.nodes.size}"
  unless nativeLeafExprs.wf do
    throw "ArenaNative phase-1 fixture: reachable leaf arena is not child-descending"
  unless nativeLeafExprs.nodes == legacyLeafExprs.nodes &&
      nativeLeafWire == legacyLeafWire do
    throw "ArenaNative phase-1 fixture: leaf program differs from recursive baseline"

  let (legacyRootExprs, legacyRootWire) ← resolvedWire legacy legacy.root
  let (nativeRootExprs, nativeRootWire) ← resolvedWire native native.root
  unless nativeRootExprs.nodes.size == 9 do
    throw s!"ArenaNative phase-1 fixture: expected 9 reachable root nodes, got {nativeRootExprs.nodes.size}"
  unless nativeRootExprs.nodes.size < native.arena.exprs.nodes.size do
    throw "ArenaNative phase-1 fixture: dead authored nodes survived reachability GC"
  unless nativeRootExprs.wf do
    throw "ArenaNative phase-1 fixture: reachable root arena is not child-descending"
  unless nativeRootExprs.nodes == legacyRootExprs.nodes &&
      nativeRootWire == legacyRootWire do
    throw "ArenaNative phase-1 fixture: root program differs from recursive baseline"

  pure {
    programCount := native.arena.programs.size
    authoredUniqueNodes := native.arena.exprs.nodes.size
    leafReachableNodes := nativeLeafExprs.nodes.size
    rootReachableNodes := nativeRootExprs.nodes.size
    leafWireBytes := nativeLeafWire.length
    rootWireBytes := nativeRootWire.length
  }

def phase1VerticalSlicePasses : Bool := phase1Evidence.isOk

def runPhase1Gate : IO Bool := do
  match phase1Evidence with
  | .ok evidence =>
    IO.println s!"  PASS  arena-native-phase1  {repr evidence}"
    pure true
  | .error error =>
    IO.println s!"  FAIL  arena-native-phase1  {error}"
    pure false

end Tropical.Testing.ArenaNative
