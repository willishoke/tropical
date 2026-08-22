import Tropical.EmitArrow.ClockAlgebra
import Tropical.Ir.ConstFoldLaws
import Tropical.Ir.Emit
import Tropical.Semantics.Plan

/-!
# Clock-rail to Plan capstones

The frontend clock denotation is mathematical `Int`; production Plan integer
operations use the signed two's-complement i64 image.  This file names that
image, pins source-operand lookup, and lifts the arena-native clock laws across
the image.  `CompileResultRefinesScalar` is the deliberately small shared
relation used at the compiler/Plan seam.  No statement here identifies the
reference Plan interpreter with LLVM, MSL, a driver, or hardware.
-/

namespace Tropical.EmitArrow

open Tropical.Ir
open Tropical.Ir.Emit
open Tropical.Plan
open Tropical.Semantics

/-- The documented signed i64 image of a mathematical clock integer. -/
def i64Image (clock : Int) : Int := wrap64 clock

theorem i64Image_isI64 (clock : Int) : IsI64 (i64Image clock) :=
  wrap64_isI64 clock

theorem i64Image_idem (clock : Int) : i64Image (i64Image clock) = i64Image clock :=
  wrap64_idem clock

theorem i64Image_add (a b : Int) :
    i64Image (i64Image a + i64Image b) = i64Image (a + b) :=
  wrap64_add a b

theorem i64Image_sub (a b : Int) :
    i64Image (i64Image a - i64Image b) = i64Image (a - b) :=
  wrap64_sub a b

theorem i64Image_mul (a b : Int) :
    i64Image (i64Image a * i64Image b) = i64Image (a * b) :=
  wrap64_mul a b

theorem i64Image_neg (a : Int) : i64Image (-i64Image a) = i64Image (-a) :=
  wrap64_neg a

/-- A successful scalar compilation related to its observable value in the
    final Plan state.  Keeping the compiler run in the relation rules out
    postulating an unrelated operand, while the evaluation field uses the
    shared Plan semantics rather than a second interpreter. -/
structure CompileResultRefinesScalar
    (arena : ExprArena) (hw : arena.wf = true) (root : ExprId)
    (expected : Option Tropical.Plan.ScalarType) (emitState emitState' : EmitSt)
    (result : CompileResult) (alg : Algebra α) (inputs : PlanInputs α)
    (planState : PlanState α) (value : α) where
  compile : (compileNode arena hw root expected).run emitState =
    .ok (result, emitState')
  operand : NOperand
  scalarResult : result = .scalar operand .int
  evaluates : evalOperand alg inputs planState operand = .ok (.scalar value)

/-- The exact source-to-Plan seam for an arena-native clock rail.  Constructor
    closure establishes `CompileResultRefinesScalar`; this capstone exposes the
    final observable equation in the explicit signed-i64 image. -/
theorem compileClockRail_refines
    {arena : ExprArena} {root : ExprId} (rail : OnClockRail arena root)
    (env : ClockEnv) (hw : arena.wf = true)
    (emitState emitState' : EmitSt) (result : CompileResult)
    (alg : Algebra Int) (inputs : PlanInputs Int) (planState : PlanState Int)
    (hrefines : CompileResultRefinesScalar arena hw root (some .int)
      emitState emitState' result alg inputs planState
      (i64Image (denoteClock rail env))) :
    ∃ operand, result = .scalar operand .int ∧
      evalOperand alg inputs planState operand =
        .ok (.scalar (i64Image (denoteClock rail env))) := by
  exact ⟨hrefines.operand, hrefines.scalarResult, hrefines.evaluates⟩

/-- The canonical `.tick` Plan operand index is populated by the named tick
    source.  `evalOperand_source_of_getElem` turns this image fact into operand
    evaluation at the public Plan seam. -/
theorem declaredTick_source
    (inputs : PlanInputs α) (image : PlanSourceImage α) :
    (inputs.withDeclaredSources defaultSources image).sources[sourceTick]? =
      some image.tick := by
  simp [PlanInputs.withDeclaredSources, defaultSources, PlanSourceImage.value,
    sourceTick]

/-- Tile materialization keeps absolute tile tick at declared source index 3.
    This is source correspondence only; it makes no interpolated-Metal claim. -/
theorem declaredTileTick_source
    (inputs : PlanInputs α) (image : PlanSourceImage α) :
    (inputs.withDeclaredSources tileSources image).sources[3]? =
      some image.tileTick := by
  simp [PlanInputs.withDeclaredSources, tileSources, PlanSourceImage.value]

theorem warp_inv_i64 {arena : ExprArena} {c delta sum root : ExprId}
    (clockRail : OnClockRail arena c) (deltaRail : OnClockRail arena delta)
    (sumNode : arena.deref sum = some (.binary .add c delta))
    (rootNode : arena.deref root = some (.binary .sub sum delta))
    (env : ClockEnv) :
    i64Image (denoteClock (.sub rootNode (.add sumNode clockRail deltaRail)
      deltaRail) env) = i64Image (denoteClock clockRail env) :=
  congrArg i64Image (warp_inv clockRail deltaRail sumNode rootNode env)

theorem warp_inv'_i64 {arena : ExprArena} {c delta difference root : ExprId}
    (clockRail : OnClockRail arena c) (deltaRail : OnClockRail arena delta)
    (differenceNode : arena.deref difference = some (.binary .sub c delta))
    (rootNode : arena.deref root = some (.binary .add difference delta))
    (env : ClockEnv) :
    i64Image (denoteClock
      (.add rootNode (.sub differenceNode clockRail deltaRail) deltaRail) env) =
      i64Image (denoteClock clockRail env) :=
  congrArg i64Image
    (warp_inv' clockRail deltaRail differenceNode rootNode env)

theorem warp_assoc_i64 {arena : ExprArena}
    {c delta1 delta2 leftInner leftRoot rightInner rightRoot : ExprId}
    (clockRail : OnClockRail arena c)
    (delta1Rail : OnClockRail arena delta1)
    (delta2Rail : OnClockRail arena delta2)
    (leftInnerNode : arena.deref leftInner = some (.binary .sub c delta1))
    (leftRootNode : arena.deref leftRoot = some (.binary .sub leftInner delta2))
    (rightInnerNode : arena.deref rightInner = some (.binary .add delta1 delta2))
    (rightRootNode : arena.deref rightRoot = some (.binary .sub c rightInner))
    (env : ClockEnv) :
    i64Image (denoteClock
      (.sub leftRootNode (.sub leftInnerNode clockRail delta1Rail) delta2Rail) env) =
    i64Image (denoteClock
      (.sub rightRootNode clockRail (.add rightInnerNode delta1Rail delta2Rail)) env) :=
  congrArg i64Image (warp_assoc clockRail delta1Rail delta2Rail leftInnerNode
    leftRootNode rightInnerNode rightRootNode env)

theorem rev_involution_i64 {arena : ExprArena} {c inner root : ExprId}
    (clockRail : OnClockRail arena c)
    (innerNode : arena.deref inner = some (.unary .neg c))
    (rootNode : arena.deref root = some (.unary .neg inner))
    (env : ClockEnv) :
    i64Image (denoteClock (.neg rootNode (.neg innerNode clockRail)) env) =
      i64Image (denoteClock clockRail env) :=
  congrArg i64Image (rev_involution clockRail innerNode rootNode env)

theorem rev_swap_i64 {arena : ExprArena}
    {c delta leftInner leftRoot rightInner rightRoot : ExprId}
    (clockRail : OnClockRail arena c) (deltaRail : OnClockRail arena delta)
    (leftInnerNode : arena.deref leftInner = some (.binary .sub c delta))
    (leftRootNode : arena.deref leftRoot = some (.unary .neg leftInner))
    (rightInnerNode : arena.deref rightInner = some (.unary .neg c))
    (rightRootNode : arena.deref rightRoot = some (.binary .add rightInner delta))
    (env : ClockEnv) :
    i64Image (denoteClock (.neg leftRootNode
      (.sub leftInnerNode clockRail deltaRail)) env) =
    i64Image (denoteClock (.add rightRootNode
      (.neg rightInnerNode clockRail) deltaRail) env) :=
  congrArg i64Image (rev_swap clockRail deltaRail leftInnerNode leftRootNode
    rightInnerNode rightRootNode env)

end Tropical.EmitArrow
