import Tropical.Ir.Emit
import Tropical.EmitArrow.BankOrder

/-!
# WS-D: the emit half of the bank order-preservation theorem (slice 3c)

`compileBankSum` (`Ir/Emit.lean`) lowers `ENode.bankSum` to a
`ReduceBegin`/body/`ReduceEnd` region. The frontend half
(`EmitArrow/BankOrder.lean`) proves the production bank denotation is
the authored increasing-index fold; this file proves the emit half: **the
instruction stream `compileBankSum` produces has exactly the
region shape** — everything the sub-compiles emitted for the tables and
the optional runtime count (loop-invariant, BEFORE the region), then
`ReduceBegin` spliced IMMEDIATELY before the body's instructions, then the
body, then the `Add`-accumulate and `ReduceEnd`, and nothing else. In
particular the splice-at-recorded-size logic (the part nested banks lean
on — "the outer's insertion point never moves") is verified:
`ReduceBegin` lands at the exact boundary between loop-invariant and
per-iteration instructions (`insertIdx!_at_prefix`).

The single hypothesis `AppendsOnly (compileNode …)` states the
sub-compiler only appends instructions — the emitter-wide discipline
(`emit` pushes; the one non-append in the whole emitter is
`compileBankSum`'s own splice, which inserts at an index ≥ its entry
stream length and so preserves every enclosing region's prefix — exactly
what this theorem shows, one nesting level at a time).

What remains between this theorem and "banked audio ≡ unrolled audio" is
ONE sentence about the backends — `REDUCE_REGION_EXECUTES_IN_ARRAY_ORDER`
(`Ir/Emit.lean`), discharged by inspection per backend.
-/

namespace Tropical.Ir.Emit

open Tropical.Plan

/-- The emitter contract the region-shape theorem consumes: a compile
    action only APPENDS to the instruction stream (it never edits or
    reorders what was already emitted). -/
def AppendsOnly {α : Type} (act : EmitM α) : Prop :=
  ∀ (s : EmitSt) (r : α) (s' : EmitSt),
    act.run s = .ok (r, s') → ∃ d, s'.instrs = s.instrs ++ d

theorem AppendsOnly.pure {α} (a : α) : AppendsOnly (pure a : EmitM α) := by
  intro s r s' h
  rw [StateT.run_pure] at h
  cases h
  exact ⟨#[], by simp⟩

theorem AppendsOnly.bind {α β} {x : EmitM α} {f : α → EmitM β}
    (hx : AppendsOnly x) (hf : ∀ a, AppendsOnly (f a)) :
    AppendsOnly (x >>= f) := by
  intro s r s' h
  rw [StateT.run_bind] at h
  match hx1 : x.run s with
  | .error e => rw [hx1] at h; cases h
  | .ok (a, s₁) =>
    rw [hx1] at h
    obtain ⟨d₁, hd₁⟩ := hx s a s₁ hx1
    obtain ⟨d₂, hd₂⟩ := hf a s₁ r s' (by simpa [bind, Except.bind] using h)
    exact ⟨d₁ ++ d₂, by rw [hd₂, hd₁, Array.append_assoc]⟩

theorem AppendsOnly.discard {α} {x : EmitM α} (hx : AppendsOnly x) :
    AppendsOnly (Functor.discard x) := by
  intro s r s' h
  have hrun : (Functor.discard x).run s
      = (x.run s).map (fun (p : α × EmitSt) => (PUnit.unit, p.2)) := by
    simp [Functor.discard, Functor.mapConst, Functor.map, StateT.map, StateT.run]
  rw [hrun] at h
  match hx1 : x.run s with
  | .error e => rw [hx1] at h; cases h
  | .ok (a, s₁) =>
    rw [hx1] at h
    cases h
    exact hx s a _ hx1

private theorem AppendsOnly.listFoldlM {α} (l : List α) (f : PUnit → α → EmitM PUnit)
    (hf : ∀ x ∈ l, ∀ u, AppendsOnly (f u x)) :
    ∀ (init : PUnit), AppendsOnly (List.foldlM f init l) := by
  induction l with
  | nil => intro init; rw [List.foldlM_nil]; exact .pure _
  | cons a l ih =>
    intro init
    rw [List.foldlM_cons]
    exact .bind (hf a (by simp) init)
      (fun u => ih (fun x hx => hf x (by simp [hx])) u)

theorem AppendsOnly.arrayForM {α} (xs : Array α) (f : α → EmitM PUnit)
    (hf : ∀ x ∈ xs, AppendsOnly (f x)) : AppendsOnly (Array.forM f xs) := by
  show AppendsOnly (Array.foldlM (fun _ => f) PUnit.unit xs 0 xs.size)
  rw [← Array.foldlM_toList]
  exact AppendsOnly.listFoldlM xs.toList (fun _ => f)
    (fun x hx _ => hf x (by simpa using hx)) PUnit.unit

private theorem List.insertIdx_length_append {α} (l₁ l₂ : List α) (x : α) :
    (l₁ ++ l₂).insertIdx l₁.length x = l₁ ++ x :: l₂ := by
  induction l₁ with
  | nil => simp [List.insertIdx]
  | cons a l ih => simpa [List.insertIdx_succ_cons] using ih

/-- Inserting at the recorded boundary of a grown stream lands exactly at
    the boundary: the splice arithmetic `compileBankSum` relies on. -/
theorem insertIdx!_at_prefix {α} (xs d : Array α) (x : α) :
    (xs ++ d).insertIdx! xs.size x = xs ++ #[x] ++ d := by
  unfold Array.insertIdx!
  split
  · apply Array.ext'
    rw [Array.toList_insertIdx]
    simpa using List.insertIdx_length_append xs.toList d.toList x
  · rename_i h
    exact absurd (by simp : xs.size ≤ (xs ++ d).size) h

-- ─────────────────────────────────────────────────────────────
-- Concrete run/step lemmas (StateT over Except, definitional)
-- ─────────────────────────────────────────────────────────────

theorem run_allocReg (s : EmitSt) :
    (allocReg).run s = .ok (s.nextReg, { s with nextReg := s.nextReg + 1 }) := rfl

theorem run_emit (i : NInstr) (s : EmitSt) :
    (emit i).run s = .ok (PUnit.unit,
      { s with instrs := s.instrs.push i,
               instrStages := s.instrStages.push s.curStage }) := rfl

/-- Peel one `StateT.bind` off a successful run. -/
theorem step_ok {α β} {x : EmitM α} {f : α → EmitM β} {s : EmitSt} {r : β} {s' : EmitSt}
    (h : (StateT.bind x f).run s = .ok (r, s')) :
    ∃ a s₁, x.run s = .ok (a, s₁) ∧ (f a).run s₁ = .ok (r, s') := by
  match hx : x.run s with
  | .error e =>
    exfalso
    have : (StateT.bind x f).run s = .error e := by
      show (x s >>= _) = _
      show (x.run s >>= _) = _
      rw [hx]; rfl
    rw [this] at h; cases h
  | .ok (a, s₁) =>
    refine ⟨a, s₁, rfl, ?_⟩
    have : (StateT.bind x f).run s = (f a).run s₁ := by
      show (x s >>= _) = _
      show (x.run s >>= _) = _
      rw [hx]; rfl
    rw [this] at h; exact h

theorem step_pure {α β} (a : α) (f : α → EmitM β) (s : EmitSt) :
    (StateT.bind (pure a) f).run s = (f a).run s := rfl
theorem step_get {β} (f : EmitSt → EmitM β) (s : EmitSt) :
    (StateT.bind get f).run s = (f s).run s := rfl
theorem step_allocReg {β} (f : Nat → EmitM β) (s : EmitSt) :
    (StateT.bind allocReg f).run s
      = (f s.nextReg).run { s with nextReg := s.nextReg + 1 } := rfl
theorem step_modify {β} (g : EmitSt → EmitSt) (f : PUnit → EmitM β) (s : EmitSt) :
    (StateT.bind (modify g) f).run s = (f PUnit.unit).run (g s) := rfl
theorem step_emit {β} (i : NInstr) (f : PUnit → EmitM β) (s : EmitSt) :
    (StateT.bind (emit i) f).run s = (f PUnit.unit).run
      { s with instrs := s.instrs.push i,
               instrStages := s.instrStages.push s.curStage } := rfl
theorem step_throw {α β} (e : String) (f : α → EmitM β) (s : EmitSt) :
    (StateT.bind (throw e) f).run s = .error e := rfl
theorem run_pure' {α} (a : α) (s : EmitSt) : (pure a : EmitM α).run s = .ok (a, s) := rfl

set_option maxHeartbeats 1000000 in
theorem compileBankSum_stream
    (arena : ExprArena) (hw : arena.wf = true) (bound count : Nat)
    (tables : Array ExprId) (body : ExprId) (dynCount? : Option ExprId) (idxId : Nat)
    (hts : ∀ t ∈ tables, t.idx < bound) (hb : body.idx < bound)
    (hdc : ∀ d ∈ dynCount?, d.idx < bound)
    (hT : ∀ t ∈ tables, AppendsOnly (compileNode arena hw t))
    (hC : ∀ dc ∈ dynCount?, AppendsOnly (compileNode arena hw dc (some .int)))
    (hB : AppendsOnly (compileNode arena hw body))
    (s : EmitSt) (r : CompileResult) (s' : EmitSt)
    (h : (compileBankSum arena hw bound count tables body dynCount? idxId hts hb hdc).run s
          = .ok (r, s')) :
    ∃ (pre bodyIns : Array NInstr) (acc : Nat) (ty : Tropical.Parse.ScalarKind)
      (countOp? : Option NOperand) (contribOp : NOperand),
      s'.instrs = s.instrs ++ pre
        ++ #[instrReduceBegin acc (.const (Lean.JsonNumber.fromNat 0) ty) count ty countOp? idxId]
        ++ bodyIns
        ++ #[instrScalar "Add" acc #[.reg acc ty, contribOp] ty, instrReduceEnd acc ty]
      ∧ r = .scalar (.reg acc ty) ty := by
  rw [compileBankSum.eq_def] at h
  cases dynCount? with
  | none =>
    simp only [bind] at h
    -- 1. tables + count (the loop-invariant prefix)
    obtain ⟨_, s1, hFor, h⟩ := step_ok h
    obtain ⟨dT, hdT⟩ := AppendsOnly.arrayForM _ _
      (fun x _ => AppendsOnly.discard (hT x.val x.property)) s _ s1 hFor
    rw [step_pure, step_allocReg, step_get, step_get] at h
    -- 2. the body
    obtain ⟨contrib, s6, hbody, h⟩ := step_ok h
    obtain ⟨dB, hdB⟩ := hB _ contrib s6 hbody
    simp only at hdB
    -- 3. the tail: array-valued body dies on throw; scalar body emits the region
    cases hIA : contrib.isArray with
    | true => rw [if_pos (by rw [hIA])] at h; rw [step_throw] at h; cases h
    | false =>
      rw [if_neg (by rw [hIA]; exact Bool.false_ne_true)] at h
      rw [step_pure, step_modify, step_emit, step_emit, step_modify, run_pure'] at h
      simp only [Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨hr, hs⟩ := h
      subst hr; subst hs
      refine ⟨dT, dB, s1.nextReg,
        (if (contrib.scalarType == Parse.ScalarKind.bool) = true
          then Parse.ScalarKind.int else contrib.scalarType),
        none, contrib.op, ?_, rfl⟩
      rw [hdB, hdT, insertIdx!_at_prefix]
      simp only [Array.push_eq_append, Array.append_assoc]
      rfl
  | some dc =>
    simp only [bind] at h
    obtain ⟨_, s1, hFor, h⟩ := step_ok h
    obtain ⟨dT, hdT⟩ := AppendsOnly.arrayForM _ _
      (fun x _ => AppendsOnly.discard (hT x.val x.property)) s _ s1 hFor
    -- the runtime effective count compiles with the tables (loop-invariant)
    obtain ⟨cres, s2, hcnt, h⟩ := step_ok h
    obtain ⟨dC, hdC⟩ := hC dc rfl s1 cres s2 hcnt
    cases hCA : cres.isArray with
    | true => rw [if_pos (by rw [hCA])] at h; rw [step_throw] at h; cases h
    | false =>
      rw [if_neg (by rw [hCA]; exact Bool.false_ne_true)] at h
      rw [step_pure, step_pure, step_allocReg, step_get, step_get] at h
      obtain ⟨contrib, s6, hbody, h⟩ := step_ok h
      obtain ⟨dB, hdB⟩ := hB _ contrib s6 hbody
      simp only at hdB
      cases hIA : contrib.isArray with
      | true => rw [if_pos (by rw [hIA])] at h; rw [step_throw] at h; cases h
      | false =>
        rw [if_neg (by rw [hIA]; exact Bool.false_ne_true)] at h
        rw [step_pure, step_modify, step_emit, step_emit, step_modify, run_pure'] at h
        simp only [Except.ok.injEq, Prod.mk.injEq] at h
        obtain ⟨hr, hs⟩ := h
        subst hr; subst hs
        refine ⟨dT ++ dC, dB, s2.nextReg,
          (if (contrib.scalarType == Parse.ScalarKind.bool) = true
            then Parse.ScalarKind.int else contrib.scalarType),
          some cres.op, contrib.op, ?_, rfl⟩
        rw [hdB, hdC, hdT, insertIdx!_at_prefix]
        simp only [Array.push_eq_append, Array.append_assoc]
        rfl


-- ─────────────────────────────────────────────────────────────
-- WS-B + WS-D: the region's assumed semantics, and THE one assumption
-- ─────────────────────────────────────────────────────────────

/-- The number of iterations a region actually runs (WS-B,
    trip-count-as-data): the static capacity, or the runtime effective
    count CLAMPED to `[0, capacity]` at the loop head. The clamp is part
    of the semantics, not a side condition — see `regionTrips_le`. -/
def regionTrips (capacity : Nat) (dyn? : Option Int) : Nat :=
  match dyn? with
  | none => capacity
  | some d => min d.toNat capacity

/-- **WS-B.** A bank whose runtime count exceeds capacity provably runs a
    PREFIX — it can never read past the tables (which fill to capacity). -/
theorem regionTrips_le (capacity : Nat) (dyn? : Option Int) :
    regionTrips capacity dyn? ≤ capacity := by
  cases dyn? with
  | none => exact Nat.le_refl _
  | some d => exact Nat.min_le_right _ _

/-- **REDUCE_REGION_EXECUTES_IN_ARRAY_ORDER** — the single trusted
    assumption left under the bank order-preservation theorem (WS-D), in
    one sentence:

    > `ReduceBegin acc init n [dyn]` … `ReduceEnd acc` executes its body
    > for `i = 0, 1, …, trips−1` in INCREASING order, where
    > `trips = min (max dyn 0) n` (`n` when no dyn operand rides the
    > instruction), accumulating `acc := op acc body(i)` — no
    > reassociation, no vectorization across `i`.

    This function is that sentence's Lean transcription: the region's
    denotation is `refFold` over the clamped prefix, in array order. The
    assumption itself — that each backend's ReduceBegin/ReduceEnd
    execution implements THIS function — is not a Lean proposition (the
    execution is C++/LLVM/MSL); it is discharged by inspection, one
    sentence per backend:

    - `EmitLlvm.lean` `"ReduceBegin"`/`"ReduceEnd"` (the JIT and the wasm
      emitter share this emitter): `icmp slt/sgt + select` clamp to
      `[0, loopCount]` before the loop blocks open; `rd_cond` tests
      `i < bound`, the body stores `acc`, `rd` steps `i + 1` — a single
      scalar chain in increasing order.
    - `EmitMsl.lean` `"ReduceBegin"`/`"ReduceEnd"` (Metal):
      `min(max(dyn, 0L), loopCount)` bound before the loop;
      `for (long rd = 0; rd < bound; ++rd)` with a scalar accumulator
      local — same shape, same order.

    Everything else — that the two builder branches are the same tree
    (`denoteExpr_staticBank_order`), that the fold is the array-order
    fold (`refFold`), that nesting composes row-major (`refFold_nested`),
    that the emitted stream has exactly
    the region shape with the tables hoisted before it
    (`compileBankSum_stream`), and that an over-capacity live count runs
    a prefix (`regionTrips_le`) — is a theorem. -/
def regionDenotation {α : Type _} (op : α → α → α) (init : α)
    (body : Nat → α) (capacity : Nat) (dyn? : Option Int) : α :=
  Tropical.EmitArrow.refFold op init body (regionTrips capacity dyn?)

/-- The arena-native capstone: under the named backend assumption encoded by
    `regionDenotation`, a static emitted region performs exactly the same
    increasing-index left fold as the direct `ENode.bankSum` denotation. -/
theorem regionDenotation_static_eq_refFold {α : Type _}
    (op : α → α → α) (zero : α) (body : Nat → α) (capacity : Nat) :
    regionDenotation op zero body capacity none =
      Tropical.EmitArrow.refFold op zero body capacity := rfl

end Tropical.Ir.Emit
