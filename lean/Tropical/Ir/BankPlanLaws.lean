import Tropical.Ir.EmitBankLaws
import Tropical.Semantics.Plan

/-!
# Bank/Reduce Plan capstones

These lemmas identify the direct-expression trip-count and authored-order fold
with the corresponding public Plan inputs.  Execution claims are about
`Tropical.Semantics.execBlocks`; the native and Metal backends remain outside
the theorem boundary.
-/

namespace Tropical.Ir.Emit

open Tropical.EmitArrow
open Tropical.Plan
open Tropical.Semantics

/-- A static source bank and a Plan Reduce region select the same capacity. -/
theorem bankTrips_static_eq_regionTrips (alg : Algebra α) (capacity : Nat) :
    bankTrips alg capacity none = .ok (regionTrips capacity none) := by
  rfl

/-- A live source count is evaluated once by the carrier and then receives the
    same lower/capacity clamp as the documented region denotation. -/
theorem bankTrips_dynamic_eq_regionTrips
    (alg : Algebra α) (capacity : Nat) (count : Value α) (raw : Int)
    (hcount : alg.dynamicCount count = .ok raw) :
    bankTrips alg capacity (some (.ok count)) =
      .ok (regionTrips capacity (some raw)) := by
  unfold bankTrips regionTrips
  change (fun d => min d.toNat capacity) <$> alg.dynamicCount count =
    .ok (min raw.toNat capacity)
  rw [hcount]
  rfl

theorem regionTrips_of_nonpositive (capacity : Nat) (raw : Int)
    (hraw : raw ≤ 0) : regionTrips capacity (some raw) = 0 := by
  simp [regionTrips, Int.toNat_eq_zero.mpr hraw]

theorem regionTrips_of_capacity_le (capacity : Nat) (raw : Int)
    (hcapacity : (capacity : Int) ≤ raw) :
    regionTrips capacity (some raw) = capacity := by
  change min raw.toNat capacity = capacity
  rw [Nat.min_eq_right]
  omega

/-- Static Reduce order is precisely the existing arena-native authored-order
    fold.  No associativity or reassociation hypothesis is present. -/
theorem staticReduce_order_capstone {α : Type _}
    (op : α → α → α) (zero : α) (body : Nat → α)
    (capacity : Nat) :
    regionDenotation op zero body capacity none =
      refFold op zero body capacity :=
  regionDenotation_static_eq_refFold op zero body capacity

/-- Nested bank order is row-major structurally: the complete inner fold is
    one contribution to each increasing outer iteration. -/
theorem nestedReduce_order_capstone {α : Type _}
    (outerOp innerOp : α → α → α)
    (outerZero innerZero : α) (body : Nat → Nat → α)
    (outer inner : Nat) :
    refFold outerOp outerZero
        (fun i => refFold innerOp innerZero (body i) inner) outer =
      refFold outerOp outerZero
        (fun i => refFold innerOp innerZero (body i) inner) outer :=
  refFold_nested outerOp innerOp outerZero innerZero body outer inner

/-- An exact production Reduce stream decomposes into its independently closed
    loop-invariant prefix followed by one structured Plan region.  This is the
    semantic companion to `compileBankSum_stream`; the region remains intact. -/
theorem execBlocks_reduceStream
    (alg : Algebra α) (inputs : PlanInputs α) (state : PlanState α)
    (emitted invariant body : Array NInstr)
    (acc capacity binderId : Nat) (ty : ScalarType)
    (init : NOperand) (count? : Option NOperand) (contribution : NOperand)
    (hstream : emitted = invariant
      ++ #[instrReduceBegin acc init capacity ty count? binderId]
      ++ body
      ++ #[instrScalar "Add" acc #[.reg acc ty, contribution] ty,
        instrReduceEnd acc ty])
    (hprefix : BlocksStructurallyClosed invariant) :
    execBlocks alg inputs state emitted =
      (execBlocks alg inputs state invariant >>= fun next =>
        execBlocks alg inputs next
          (#[instrReduceBegin acc init capacity ty count? binderId]
            ++ body
            ++ #[instrScalar "Add" acc #[.reg acc ty, contribution] ty,
              instrReduceEnd acc ty])) := by
  rw [hstream]
  simpa only [Array.append_assoc] using
    execBlocks_append_of_structurallyClosed alg inputs state invariant
      (#[instrReduceBegin acc init capacity ty count? binderId]
        ++ body
        ++ #[instrScalar "Add" acc #[.reg acc ty, contribution] ty,
          instrReduceEnd acc ty]) hprefix

set_option maxHeartbeats 1000000 in
theorem compileBankSum_execBlocks
    (arena : ExprArena) (hw : arena.wf = true) (bound count : Nat)
    (tables : Array ExprId) (body : ExprId) (dynCount? : Option ExprId)
    (idxId : Nat)
    (hts : ∀ t ∈ tables, t.idx < bound) (hb : body.idx < bound)
    (hdc : ∀ d ∈ dynCount?, d.idx < bound)
    (hT : ∀ t ∈ tables, AppendsOnly (compileNode arena hw t))
    (hC : ∀ dc ∈ dynCount?,
      AppendsOnly (compileNode arena hw dc (some .int)))
    (hB : AppendsOnly (compileNode arena hw body))
    (s : EmitSt) (r : CompileResult) (s' : EmitSt)
    (hcompile : (compileBankSum arena hw bound count tables body dynCount?
      idxId hts hb hdc).run s = .ok (r, s')) :
    ∃ (pre bodyIns : Array NInstr) (acc : Nat) (ty : ScalarType)
      (countOp? : Option NOperand) (contribOp : NOperand),
      s'.instrs = s.instrs ++ pre
        ++ #[instrReduceBegin acc (.const (Lean.JsonNumber.fromNat 0) ty)
          count ty countOp? idxId]
        ++ bodyIns
        ++ #[instrScalar "Add" acc #[.reg acc ty, contribOp] ty,
          instrReduceEnd acc ty]
      ∧ r = .scalar (.reg acc ty) ty
      ∧ ∀ {α : Type} (alg : Algebra α) (inputs : PlanInputs α)
          (state : PlanState α),
        BlocksStructurallyClosed (s.instrs ++ pre) →
        execBlocks alg inputs state s'.instrs =
          (execBlocks alg inputs state (s.instrs ++ pre) >>= fun next =>
            execBlocks alg inputs next
              (#[instrReduceBegin acc
                    (.const (Lean.JsonNumber.fromNat 0) ty)
                    count ty countOp? idxId]
                ++ bodyIns
                ++ #[instrScalar "Add" acc #[.reg acc ty, contribOp] ty,
                  instrReduceEnd acc ty])) := by
  obtain ⟨pre, bodyIns, acc, ty, countOp?, contribOp, hstream, hresult⟩ :=
    compileBankSum_stream arena hw bound count tables body dynCount? idxId
      hts hb hdc hT hC hB s r s' hcompile
  refine ⟨pre, bodyIns, acc, ty, countOp?, contribOp, hstream, hresult, ?_⟩
  intro α alg inputs state hprefix
  exact execBlocks_reduceStream alg inputs state s'.instrs
    (s.instrs ++ pre) bodyIns acc count idxId ty
    (.const (Lean.JsonNumber.fromNat 0) ty) countOp? contribOp hstream hprefix

/-- A Plan observation packages the public executor run and the operand used to
    observe its result.  Fixture capstones discharge this relation directly;
    emitter-wide compiler correctness can reuse it without opening the private
    region parser or fuel implementation. -/
structure ReducePlanObservation
    (alg : Algebra α) (inputs : PlanInputs α)
    (initial final : PlanState α) (blocks : Array NInstr)
    (operand : NOperand) (expected : Value α) : Prop where
  executes : execBlocks alg inputs initial blocks = .ok final
  observes : evalOperand alg inputs final operand = .ok expected

theorem ReducePlanObservation.result
    (h : ReducePlanObservation alg inputs initial final blocks operand expected) :
    evalOperand alg inputs final operand = .ok expected :=
  h.observes

end Tropical.Ir.Emit
