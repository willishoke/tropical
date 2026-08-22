import Tropical.Ir.EmitBankLaws
import Tropical.Semantics.Plan

/-!
# Routed-sum lowering and Plan capstones

The source semantics maps each active item once and applies routes in authored
`(item, emit)` order. This module first pins the production emitter's exact
delimiter stream; semantic theorems below consume the shared Plan interpreter
and make no claim about LLVM or Metal execution.
-/

namespace Tropical.Ir.Emit

open Tropical.Plan
open Tropical.Semantics

theorem AppendsOnly.validateRoutedBodyEffects (instrs : Array NInstr) :
    AppendsOnly (validateRoutedBodyEffects instrs) := by
  unfold Tropical.Ir.Emit.validateRoutedBodyEffects
  split <;> simp [AppendsOnly.throw, AppendsOnly.pure]

theorem AppendsOnly.finishRoutedMappedValue (r : CompileResult) :
    AppendsOnly (Tropical.Ir.Emit.finishRoutedMappedValue r) := by
  unfold Tropical.Ir.Emit.finishRoutedMappedValue
  cases r with
  | scalar op ty =>
      cases ty <;> simp [AppendsOnly.throw, AppendsOnly.pure]
  | array op size ty =>
      simp [AppendsOnly.throw]

theorem AppendsOnly.routedMappedValue
    (arena : ExprArena) (hw : arena.wf = true) (value : ExprId)
    (h : AppendsOnly (compileNode arena hw value (some .float))) :
    AppendsOnly (compileNode arena hw value (some .float) >>=
      Tropical.Ir.Emit.finishRoutedMappedValue) :=
  AppendsOnly.bind h (fun r => AppendsOnly.finishRoutedMappedValue r)

theorem run_validateRoutedConfig
    (capacity outputCount : Nat) (routes : Array (Option Nat))
    (values : Array ExprId) (s : EmitSt)
    (hDepth : s.routedDepth = 0)
    (hCapacity : capacity ≠ 0) (hOutputs : outputCount ≠ 0)
    (hFanout : values.isEmpty = false)
    (hRouteCount : routes.size = capacity * values.size)
    (hTargets : routes.findSome? (routedInvalidTarget? outputCount) = none) :
    (validateRoutedConfig s.routedDepth capacity outputCount routes values).run s =
      .ok (PUnit.unit, s) := by
  unfold Tropical.Ir.Emit.validateRoutedConfig
  simp [hDepth, hCapacity, hOutputs, hFanout, hRouteCount]
  cases hfind : routes.findSome? (routedInvalidTarget? outputCount) with
  | none => change Except.ok _ = Except.ok _; rfl
  | some target => rw [hTargets] at hfind; cases hfind

theorem validateRoutedBodyEffects_success
    (instrs : Array NInstr) (s s' : EmitSt) (u : Unit)
    (h : (validateRoutedBodyEffects instrs).run s = .ok (u, s')) : s' = s := by
  unfold Tropical.Ir.Emit.validateRoutedBodyEffects at h
  split at h
  · change Except.error _ = Except.ok _ at h
    cases h
  · change Except.ok (PUnit.unit, s) = Except.ok (u, s') at h
    cases h
    rfl

set_option maxHeartbeats 1000000 in
theorem compileRoutedSum_stream
    (arena : ExprArena) (hw : arena.wf = true)
    (bound capacity outputCount : Nat) (routes : Array (Option Nat))
    (tables values : Array ExprId) (dynCount? : Option ExprId) (idxId : Nat)
    (hts : ∀ t ∈ tables, t.idx < bound)
    (hvs : ∀ v ∈ values, v.idx < bound)
    (hdc : ∀ d ∈ dynCount?, d.idx < bound)
    (hT : ∀ t ∈ tables, AppendsOnly (compileNode arena hw t))
    (hC : ∀ dc ∈ dynCount?,
      AppendsOnly (compileNode arena hw dc (some .int)))
    (hV : ∀ v ∈ values,
      AppendsOnly (compileNode arena hw v (some .float)))
    (s : EmitSt) (r : CompileResult) (s' : EmitSt)
    (hDepth : s.routedDepth = 0)
    (hCapacity : capacity ≠ 0) (hOutputs : outputCount ≠ 0)
    (hFanout : values.isEmpty = false)
    (hRouteCount : routes.size = capacity * values.size)
    (hTargets : routes.findSome? (routedInvalidTarget? outputCount) = none)
    (h : (compileRoutedSum arena hw bound capacity outputCount routes tables
      values dynCount? idxId hts hvs hdc).run s = .ok (r, s')) :
    ∃ (pre bodyIns : Array NInstr) (dst : Nat)
      (countOp? : Option NOperand) (mappedOps : Array NOperand),
      s'.instrs = s.instrs ++ pre
        ++ #[instrRoutedSumBegin dst capacity outputCount routes countOp? idxId]
        ++ bodyIns
        ++ #[instrRoutedSumYield dst mappedOps, instrRoutedSumEnd dst]
      ∧ r = .array (.arrayReg dst) outputCount .float := by
  rw [compileRoutedSum.eq_def] at h
  simp only [bind] at h
  rw [step_get] at h
  obtain ⟨_, s0, hConfig, h⟩ := step_ok h
  rw [run_validateRoutedConfig capacity outputCount routes values s hDepth
    hCapacity hOutputs hFanout hRouteCount hTargets] at hConfig
  cases hConfig
  obtain ⟨_, s1, hFor, h⟩ := step_ok h
  obtain ⟨dT, hdT⟩ := AppendsOnly.arrayForM _ _
    (fun x _ => AppendsOnly.discard (hT x.val x.property)) s _ s1 hFor
  cases dynCount? with
  | none =>
      rw [step_pure, step_allocArraySlot, step_get, step_get, step_modify] at h
      obtain ⟨mapped, sBody, hMapped, h⟩ := step_ok h
      obtain ⟨dV, hdV⟩ := AppendsOnly.arrayMapM _ _
        (fun x hx => AppendsOnly.routedMappedValue arena hw x.val
          (hV x.val x.property)) _ mapped sBody hMapped
      rw [step_modify, step_get, step_get] at h
      obtain ⟨u, sChecked, hChecked, h⟩ := step_ok h
      have hsChecked := validateRoutedBodyEffects_success _ _ _ u hChecked
      subst sChecked
      rw [step_modify, step_emit, step_emit, step_modify, run_pure'] at h
      simp only [Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨hr, hs⟩ := h
      subst hr
      subst hs
      refine ⟨dT, dV, s1.nextArraySlot, none, mapped, ?_, rfl⟩
      rw [hdV, hdT, insertIdx!_at_prefix]
      simp only [Array.push_eq_append, Array.append_assoc]
      rfl
  | some dc =>
      obtain ⟨cres, s2, hCount, h⟩ := step_ok h
      obtain ⟨dC, hdC⟩ := hC dc rfl s1 cres s2 hCount
      cases hArray : cres.isArray with
      | true =>
          rw [if_pos (by rw [hArray])] at h
          rw [step_throw] at h
          cases h
      | false =>
          rw [if_neg (by rw [hArray]; exact Bool.false_ne_true)] at h
          rw [step_pure, step_pure, step_allocArraySlot, step_get, step_get,
            step_modify] at h
          obtain ⟨mapped, sBody, hMapped, h⟩ := step_ok h
          obtain ⟨dV, hdV⟩ := AppendsOnly.arrayMapM _ _
            (fun x hx => AppendsOnly.routedMappedValue arena hw x.val
              (hV x.val x.property)) _ mapped sBody hMapped
          rw [step_modify, step_get, step_get] at h
          obtain ⟨u, sChecked, hChecked, h⟩ := step_ok h
          have hsChecked := validateRoutedBodyEffects_success _ _ _ u hChecked
          subst sChecked
          rw [step_modify, step_emit, step_emit, step_modify, run_pure'] at h
          simp only [Except.ok.injEq, Prod.mk.injEq] at h
          obtain ⟨hr, hs⟩ := h
          subst hr
          subst hs
          refine ⟨dT ++ dC, dV, s2.nextArraySlot, some cres.op, mapped, ?_, rfl⟩
          rw [hdV, hdC, hdT, insertIdx!_at_prefix]
          simp only [Array.push_eq_append, Array.append_assoc]
          rfl

end Tropical.Ir.Emit
