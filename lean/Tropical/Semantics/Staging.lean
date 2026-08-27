import Tropical.Ir.Staging
import Tropical.Semantics.Expr

/-!
# Semantic laws for binding-time staging

This module keeps the proof-facing staging contract separate from the
executable classifier.  In particular, `Stage.le` is a Boolean decision
procedure, so the lattice laws below state both its order theory and the
least-upper-bound property implemented by `Stage.join`.
-/

namespace Tropical.Semantics.Staging

open Tropical.Ir
open Tropical.Ir.Staging

/-- Strict lexicographic order used by nested-output dependency arrays. -/
def NestedDependencyLt (a b : Nat × Nat) : Prop :=
  a.1 < b.1 ∨ (a.1 = b.1 ∧ a.2 < b.2)

private def NestedDependencyLe (a b : Nat × Nat) : Prop :=
  a.1 < b.1 ∨ (a.1 = b.1 ∧ a.2 ≤ b.2)

private instance (a b : Nat × Nat) : Decidable (NestedDependencyLe a b) := by
  unfold NestedDependencyLe
  infer_instance

local instance : Ord (Nat × Nat) := nestedDependencyOrd

/-- The representation invariant promised by `StageSig`: dependency arrays
    are strictly ascending and contain no duplicates.  `Nodup` is stated
    explicitly even though it follows mathematically from strict order, so
    downstream audits need not rely on that derived fact. -/
structure StageSigShape (sig : StageSig) : Prop where
  inputsAscending : sig.inputs.toList.Pairwise (· < ·)
  inputsDeduplicated : sig.inputs.toList.Nodup
  nestedAscending : sig.nested.toList.Pairwise NestedDependencyLt
  nestedDeduplicated : sig.nested.toList.Nodup

/-- Every stored signature has the representation invariant.  This is
    intentionally separate from `ArenaWellFormed`: arena termination/dedup
    soundness and staging classification are distinct proof obligations. -/
def SignaturesShaped (arena : ExprArena) : Prop :=
  ∀ sig ∈ arena.sigs, StageSigShape sig

/-- Each stored signature was computed from exactly the preceding signature
    prefix and its same-index node.  This excludes hand-built arenas whose
    signature array merely has the right length. -/
def SignaturesGenerated (arena : ExprArena) : Prop :=
  ∀ (index : Nat) (hIndex : index < arena.nodes.size),
    arena.sigs[index]? = some
      (enodeSig (arena.sigs.extract 0 index) arena.nodes[index])

/-- Canonical staging invariant: aligned semantic arena, generated
    signatures, and the strict/deduplicated dependency representation. -/
structure SignaturesSound (arena : ExprArena) : Prop where
  arenaWellFormed : ArenaWellFormed arena
  generated : SignaturesGenerated arena
  shaped : SignaturesShaped arena

theorem empty_signaturesShaped : SignaturesShaped ({} : ExprArena) := by
  intro sig h
  simp at h

theorem empty_signaturesGenerated : SignaturesGenerated ({} : ExprArena) := by
  intro index hIndex
  simp at hIndex

theorem empty_signaturesSound : SignaturesSound ({} : ExprArena) :=
  { arenaWellFormed := Tropical.Semantics.emptyArena_wellFormed
    generated := empty_signaturesGenerated
    shaped := empty_signaturesShaped }

theorem signaturesShaped_push {arena : ExprArena} {sig : StageSig}
    (hArena : SignaturesShaped arena) (hSig : StageSigShape sig) :
    SignaturesShaped { arena with sigs := arena.sigs.push sig } := by
  intro query hQuery
  rw [Array.mem_push] at hQuery
  rcases hQuery with hOld | rfl
  · exact hArena query hOld
  · exact hSig

/-- Qualified interning preserves the signature-shape invariant.  The
    qualification is local to the newly computed signature; hash-cons hits
    retain the already-shaped arena unchanged. -/
theorem eintern_preserves_signaturesShaped {arena : ExprArena} {node : ENode}
    (hArena : SignaturesShaped arena)
    (hNode : StageSigShape (enodeSig arena.sigs node)) :
    SignaturesShaped (eintern node arena).2 := by
  rw [Tropical.Semantics.eintern_run]
  split
  · exact hArena
  · exact signaturesShaped_push hArena hNode

theorem stageSigShape_fold :
    StageSigShape ({ base := .fold } : StageSig) := by
  constructor <;> simp

theorem stageSigShape_s1 :
    StageSigShape ({ base := .s1 } : StageSig) := by
  constructor <;> simp

theorem stageSigShape_input (inputIdx : Nat) :
    StageSigShape ({ base := .fold, inputs := #[inputIdx] } : StageSig) := by
  constructor <;> simp

theorem stageSigShape_nested (instanceIdx outputIdx : Nat) :
    StageSigShape
      ({ base := .fold, nested := #[(instanceIdx, outputIdx)] } : StageSig) := by
  constructor <;> simp

private theorem eraseDups_sublist [BEq α] (xs : List α) :
    List.Sublist xs.eraseDups xs := by
  cases xs with
  | nil => simp
  | cons head tail =>
    rw [List.eraseDups_cons]
    apply List.Sublist.cons₂
    exact (eraseDups_sublist (tail.filter fun value => !value == head)).trans
      List.filter_sublist
termination_by xs.length
decreasing_by
  exact Nat.lt_succ_of_le (List.length_filter_le _ _)

private theorem eraseDups_nodup [BEq α] [LawfulBEq α] (xs : List α) :
    xs.eraseDups.Nodup := by
  cases xs with
  | nil => simp
  | cons head tail =>
    rw [List.eraseDups_cons]
    apply List.nodup_cons.mpr
    constructor
    · intro hMember
      have hFiltered : head ∈ tail.filter (fun value => !value == head) :=
        List.mem_eraseDups.mp hMember
      simp at hFiltered
    · exact eraseDups_nodup (tail.filter fun value => !value == head)
termination_by xs.length
decreasing_by
  exact Nat.lt_succ_of_le (List.length_filter_le _ _)

private theorem nat_pairwise_lt_of_le_nodup (xs : List Nat)
    (hle : xs.Pairwise (· ≤ ·)) (hnodup : xs.Nodup) :
    xs.Pairwise (· < ·) := by
  cases xs with
  | nil => simp
  | cons head tail =>
    rw [List.pairwise_cons] at hle ⊢
    rw [List.nodup_cons] at hnodup
    constructor
    · intro value hValue
      have hne : head ≠ value := by
        intro heq
        exact hnodup.1 (heq ▸ hValue)
      exact Nat.lt_of_le_of_ne (hle.1 value hValue) hne
    · exact nat_pairwise_lt_of_le_nodup tail hle.2 hnodup.2
termination_by xs.length

private theorem nat_compare_not_gt (lhs rhs : Nat) :
    (compare lhs rhs != Ordering.gt) = decide (lhs ≤ rhs) := by
  cases hcmp : compare lhs rhs with
  | lt =>
    have hlt := Nat.compare_eq_lt.mp hcmp
    change true = decide (lhs ≤ rhs)
    simp [Nat.le_of_lt hlt]
  | eq =>
    have heq := Nat.compare_eq_eq.mp hcmp
    change true = decide (lhs ≤ rhs)
    simp [heq]
  | gt =>
    have hgt := Nat.compare_eq_gt.mp hcmp
    change false = decide (lhs ≤ rhs)
    simp [Nat.not_le.mpr hgt]

private theorem mergeAsc_nat_shape (a b : Array Nat) :
    let merged := mergeAsc a b
    merged.toList.Pairwise (· < ·) ∧ merged.toList.Nodup := by
  let sorted := (a.toList ++ b.toList).mergeSort fun lhs rhs =>
    compare lhs rhs != Ordering.gt
  have hcompare (lhs rhs : Nat) :
      (compare lhs rhs != Ordering.gt) = decide (lhs ≤ rhs) :=
    nat_compare_not_gt lhs rhs
  have htrans : ∀ lhs middle rhs : Nat,
      (compare lhs middle != Ordering.gt) = true →
      (compare middle rhs != Ordering.gt) = true →
      (compare lhs rhs != Ordering.gt) = true := by
    intro lhs middle rhs hl hm
    simp only [hcompare, decide_eq_true_eq] at hl hm ⊢
    omega
  have htotal : ∀ lhs rhs : Nat,
      (compare lhs rhs != Ordering.gt) ||
        (compare rhs lhs != Ordering.gt) := by
    intro lhs rhs
    simp only [hcompare, decide_eq_true_eq, Bool.or_eq_true]
    omega
  have hsorted : sorted.Pairwise (· ≤ ·) := by
    have h := List.pairwise_mergeSort htrans htotal (a.toList ++ b.toList)
    simpa only [sorted, hcompare, decide_eq_true_eq] using h
  have hsub : List.Sublist sorted.eraseDups sorted := eraseDups_sublist sorted
  have hle : sorted.eraseDups.Pairwise (· ≤ ·) := hsorted.sublist hsub
  have hnodup : sorted.eraseDups.Nodup := eraseDups_nodup sorted
  change sorted.eraseDups.Pairwise (· < ·) ∧ sorted.eraseDups.Nodup
  exact ⟨nat_pairwise_lt_of_le_nodup sorted.eraseDups hle hnodup, hnodup⟩

private theorem nested_pairwise_lt_of_le_nodup (xs : List (Nat × Nat))
    (hle : xs.Pairwise NestedDependencyLe) (hnodup : xs.Nodup) :
    xs.Pairwise NestedDependencyLt := by
  cases xs with
  | nil => simp
  | cons head tail =>
    rw [List.pairwise_cons] at hle ⊢
    rw [List.nodup_cons] at hnodup
    constructor
    · intro value hValue
      have hne : head ≠ value := by
        intro heq
        exact hnodup.1 (heq ▸ hValue)
      rcases hle.1 value hValue with hFirst | ⟨hFirst, hSecond⟩
      · exact Or.inl hFirst
      · right
        refine ⟨hFirst, Nat.lt_of_le_of_ne hSecond ?_⟩
        intro hSecondEq
        apply hne
        apply Prod.ext <;> assumption
    · exact nested_pairwise_lt_of_le_nodup tail hle.2 hnodup.2
termination_by xs.length

private theorem mergeAsc_nested_shape (a b : Array (Nat × Nat)) :
    let merged := mergeAsc a b
    merged.toList.Pairwise NestedDependencyLt ∧ merged.toList.Nodup := by
  let sorted := (a.toList ++ b.toList).mergeSort fun lhs rhs =>
    compare lhs rhs != Ordering.gt
  have hcompare (lhs rhs : Nat × Nat) :
      (compare lhs rhs != Ordering.gt) =
        decide (NestedDependencyLe lhs rhs) := by
    rcases lhs with ⟨lhsFirst, lhsSecond⟩
    rcases rhs with ⟨rhsFirst, rhsSecond⟩
    change
      ((match compare lhsFirst rhsFirst with
        | .eq => compare lhsSecond rhsSecond
        | ordering => ordering) != Ordering.gt) = _
    cases hFirst : compare lhsFirst rhsFirst with
    | lt =>
      have hlt := Nat.compare_eq_lt.mp hFirst
      change true = decide (NestedDependencyLe
        (lhsFirst, lhsSecond) (rhsFirst, rhsSecond))
      simp [NestedDependencyLe, hlt]
    | eq =>
      have heq := Nat.compare_eq_eq.mp hFirst
      subst rhsFirst
      change (compare lhsSecond rhsSecond != Ordering.gt) =
        decide (NestedDependencyLe
          (lhsFirst, lhsSecond) (lhsFirst, rhsSecond))
      simp [NestedDependencyLe, nat_compare_not_gt]
    | gt =>
      have hgt := Nat.compare_eq_gt.mp hFirst
      change false = decide (NestedDependencyLe
        (lhsFirst, lhsSecond) (rhsFirst, rhsSecond))
      simp [NestedDependencyLe]
      omega
  have htrans : ∀ lhs middle rhs : Nat × Nat,
      (compare lhs middle != Ordering.gt) = true →
      (compare middle rhs != Ordering.gt) = true →
      (compare lhs rhs != Ordering.gt) = true := by
    intro lhs middle rhs hl hm
    simp only [hcompare, decide_eq_true_eq] at hl hm ⊢
    rcases hl with hl | ⟨hl1, hl2⟩ <;>
      rcases hm with hm | ⟨hm1, hm2⟩
    · left; omega
    · left; omega
    · left; omega
    · right; constructor <;> omega
  have htotal : ∀ lhs rhs : Nat × Nat,
      (compare lhs rhs != Ordering.gt) ||
        (compare rhs lhs != Ordering.gt) := by
    intro lhs rhs
    simp only [hcompare, decide_eq_true_eq, Bool.or_eq_true]
    unfold NestedDependencyLe
    omega
  have hsorted : sorted.Pairwise NestedDependencyLe := by
    have h := List.pairwise_mergeSort htrans htotal (a.toList ++ b.toList)
    simpa only [sorted, hcompare, decide_eq_true_eq] using h
  have hsub : List.Sublist sorted.eraseDups sorted := eraseDups_sublist sorted
  have hle : sorted.eraseDups.Pairwise NestedDependencyLe := hsorted.sublist hsub
  have hnodup : sorted.eraseDups.Nodup := eraseDups_nodup sorted
  change sorted.eraseDups.Pairwise NestedDependencyLt ∧ sorted.eraseDups.Nodup
  exact ⟨nested_pairwise_lt_of_le_nodup sorted.eraseDups hle hnodup, hnodup⟩

/-- Signature joins normalize both dependency unions to the promised strict,
    duplicate-free representation. -/
theorem StageSigShape.join {a b : StageSig}
    (_ha : StageSigShape a) (_hb : StageSigShape b) :
    StageSigShape (a.join b) := by
  let hInputs := mergeAsc_nat_shape a.inputs b.inputs
  let hNested := mergeAsc_nested_shape a.nested b.nested
  exact {
    inputsAscending := hInputs.1
    inputsDeduplicated := hInputs.2
    nestedAscending := hNested.1
    nestedDeduplicated := hNested.2 }

private theorem signatureAt_shape {sigs : Array StageSig}
    (hSigs : ∀ sig ∈ sigs, StageSigShape sig) (id : ExprId) :
    StageSigShape (sigs[id.idx]?.getD { base := .s1 }) := by
  cases hGet : sigs[id.idx]? with
  | none => simpa [hGet] using stageSigShape_s1
  | some sig =>
    have hMem : sig ∈ sigs := by
      exact Array.mem_of_getElem? hGet
    simpa [hGet] using hSigs sig hMem

private theorem foldl_signatures_shape {sigs : Array StageSig}
    (hSigs : ∀ sig ∈ sigs, StageSigShape sig) (items : List ExprId)
    (initial : StageSig) (hInitial : StageSigShape initial) :
    StageSigShape (items.foldl
      (fun accumulated id =>
        accumulated.join (sigs[id.idx]?.getD { base := .s1 })) initial) := by
  induction items generalizing initial with
  | nil => exact hInitial
  | cons head tail ih =>
    simp only [List.foldl_cons]
    exact ih (initial := initial.join (sigs[head.idx]?.getD { base := .s1 }))
      (hInitial.join (signatureAt_shape hSigs head))

/-- Every node constructor preserves the dependency-array representation
    invariant when the already-interned child signature prefix has it. -/
theorem enodeSig_shape {sigs : Array StageSig}
    (hSigs : ∀ sig ∈ sigs, StageSigShape sig) (node : ENode) :
    StageSigShape (enodeSig sigs node) := by
  cases node with
  | num | sampleRate | loopIdx =>
    exact stageSigShape_fold
  | bool =>
    exact stageSigShape_fold
  | sampleIndex | tileSampleIndex | tilePhase =>
    exact stageSigShape_s1
  | paramRef =>
    exact {
      inputsAscending := by simp [enodeSig]
      inputsDeduplicated := by simp [enodeSig]
      nestedAscending := by simp [enodeSig]
      nestedDeduplicated := by simp [enodeSig] }
  | inputRef inputIdx =>
    simpa [enodeSig] using stageSigShape_input inputIdx.idx
  | nestedOut instanceIdx outputIdx =>
    simpa [enodeSig] using stageSigShape_nested instanceIdx.idx outputIdx.idx
  | arr items | tileArray items =>
    simp only [enodeSig]
    rw [← Array.foldl_toList]
    exact foldl_signatures_shape hSigs items.toList _ stageSigShape_fold
  | binary tag lhs rhs =>
    simp only [enodeSig]
    exact (signatureAt_shape hSigs lhs).join (signatureAt_shape hSigs rhs)
  | unary tag arg =>
    simpa only [enodeSig] using signatureAt_shape hSigs arg
  | clamp value lo hi | select value lo hi | arraySet value lo hi =>
    simp only [enodeSig]
    exact ((signatureAt_shape hSigs value).join
      (signatureAt_shape hSigs lo)).join (signatureAt_shape hSigs hi)
  | index array index =>
    simp only [enodeSig]
    exact (signatureAt_shape hSigs array).join
      (signatureAt_shape hSigs index)
  | bankSum capacity tables body dynCount binderId =>
    simp only [enodeSig]
    have hTables : StageSigShape
        (tables.foldl (fun accumulated id =>
          accumulated.join (sigs[id.idx]?.getD { base := .s1 }))
          { base := .fold }) := by
      rw [← Array.foldl_toList]
      exact foldl_signatures_shape hSigs tables.toList _ stageSigShape_fold
    have hBody := hTables.join (signatureAt_shape hSigs body)
    cases dynCount with
    | none => exact hBody
    | some count => exact hBody.join (signatureAt_shape hSigs count)
  | routedSum capacity outputCount routes tables values dynCount binderId =>
    simp only [enodeSig]
    have hItems : StageSigShape
        ((tables ++ values).foldl (fun accumulated id =>
          accumulated.join (sigs[id.idx]?.getD { base := .s1 }))
          { base := .fold }) := by
      rw [← Array.foldl_toList]
      exact foldl_signatures_shape hSigs (tables ++ values).toList _
        stageSigShape_fold
    cases dynCount with
    | none => exact hItems
    | some count => exact hItems.join (signatureAt_shape hSigs count)

theorem eintern_preserves_signaturesShaped_auto {arena : ExprArena}
    {node : ENode} (hArena : SignaturesShaped arena) :
    SignaturesShaped (eintern node arena).2 :=
  eintern_preserves_signaturesShaped hArena (enodeSig_shape hArena node)

theorem signaturesGenerated_push {arena : ExprArena} {node : ENode}
    (hGenerated : SignaturesGenerated arena)
    (hAligned : arena.sigs.size = arena.nodes.size) :
    SignaturesGenerated {
      arena with
      nodes := arena.nodes.push node
      sigs := arena.sigs.push (enodeSig arena.sigs node) } := by
  intro index hIndex
  simp only [Array.size_push] at hIndex
  by_cases hOld : index < arena.nodes.size
  · have hSigOld : index < arena.sigs.size := by
      simpa [hAligned] using hOld
    have hStop : index ≤ arena.sigs.size := by
      simpa [hAligned] using Nat.le_of_lt hOld
    rw [Array.getElem?_push_lt hSigOld,
      Array.extract_push_of_le hStop, Array.getElem_push_lt hOld]
    have hAt := hGenerated index hOld
    rw [Array.getElem?_eq_getElem hSigOld] at hAt
    exact hAt
  · have hNew : index = arena.nodes.size := by omega
    subst index
    simp [hAligned.symm]
    have hGet : (arena.nodes.push node)[arena.sigs.size] = node := by
      simpa [hAligned] using
        (Array.getElem_push_eq (xs := arena.nodes) (x := node))
    rw [hGet]

theorem eintern_preserves_signaturesSound {arena : ExprArena} {node : ENode}
    (hSound : SignaturesSound arena)
    (hChildren : ChildrenInPrefix arena node) :
    SignaturesSound (eintern node arena).2 := by
  have hWellFormed :=
    (Tropical.Semantics.eintern_preserves hSound.arenaWellFormed hChildren).1
  rw [Tropical.Semantics.eintern_run] at hWellFormed ⊢
  split at *
  · exact hSound
  · exact {
      arenaWellFormed := hWellFormed
      generated := signaturesGenerated_push hSound.generated
        hSound.arenaWellFormed.signaturesAligned
      shaped := signaturesShaped_push hSound.shaped
        (enodeSig_shape hSound.shaped node) }

theorem stage_le_refl (stage : Stage) : stage.le stage = true := by
  cases stage <;> rfl

private theorem stage_beq_s1 (stage : Stage) :
    (stage == Stage.s1) = true ↔ stage = Stage.s1 := by
  cases stage with
  | fold =>
    constructor <;> intro h <;> cases h
  | s0 =>
    constructor <;> intro h <;> cases h
  | s1 =>
    constructor <;> intro <;> rfl

theorem stage_le_trans {a b c : Stage}
    (hab : a.le b = true) (hbc : b.le c = true) : a.le c = true := by
  cases a <;> cases b <;> cases c <;>
    simp_all [Stage.le, stage_beq_s1]

theorem stage_le_antisymm {a b : Stage}
    (hab : a.le b = true) (hba : b.le a = true) : a = b := by
  cases a <;> cases b <;> simp_all [Stage.le, stage_beq_s1]

theorem stage_join_assoc (a b c : Stage) :
    (a.join b).join c = a.join (b.join c) := by
  cases a <;> cases b <;> cases c <;> rfl

theorem stage_join_comm (a b : Stage) : a.join b = b.join a := by
  cases a <;> cases b <;> rfl

theorem stage_join_idem (a : Stage) : a.join a = a := by
  cases a <;> rfl

theorem stage_le_join_left (a b : Stage) : a.le (a.join b) = true := by
  cases a <;> cases b <;> rfl

theorem stage_le_join_right (a b : Stage) : b.le (a.join b) = true := by
  cases a <;> cases b <;> rfl

theorem stage_join_least {a b upper : Stage}
    (ha : a.le upper = true) (hb : b.le upper = true) :
    (a.join b).le upper = true := by
  cases a <;> cases b <;> cases upper <;> simp_all [Stage.le, Stage.join]

private theorem foldl_join_bounded (items : List β) (value : β → Stage)
    (initial upper : Stage)
    (hfinal : (items.foldl (fun stage item => stage.join (value item))
      initial).le upper = true) :
    initial.le upper = true ∧ ∀ item ∈ items, (value item).le upper = true := by
  induction items generalizing initial with
  | nil =>
    exact ⟨hfinal, by simp⟩
  | cons head tail ih =>
    simp only [List.foldl_cons] at hfinal
    have hrest := ih (initial := initial.join (value head)) hfinal
    have hHeadBound :=
      stage_le_trans (stage_le_join_right initial (value head)) hrest.1
    constructor
    · exact stage_le_trans (stage_le_join_left initial (value head)) hrest.1
    · intro item hItem
      simp only [List.mem_cons] at hItem
      rcases hItem with rfl | hTail
      · exact hHeadBound
      · exact hrest.2 item hTail

/-- Signature containment: intrinsic stage is earlier and both symbolic
    dependency sets are included. -/
structure StageSigLe (a b : StageSig) : Prop where
  base : a.base.le b.base = true
  inputs : ∀ inputIdx ∈ a.inputs, inputIdx ∈ b.inputs
  nested : ∀ dependency ∈ a.nested, dependency ∈ b.nested

theorem StageSigLe.refl (sig : StageSig) : StageSigLe sig sig :=
  { base := stage_le_refl sig.base
    inputs := fun _ h => h
    nested := fun _ h => h }

theorem mem_mergeAsc [Ord α] [BEq α] [LawfulBEq α]
    {value : α} {a b : Array α} :
    value ∈ mergeAsc a b ↔ value ∈ a ∨ value ∈ b := by
  simp [mergeAsc]

theorem StageSigLe.joinLeft (a b : StageSig) : StageSigLe a (a.join b) :=
  { base := stage_le_join_left a.base b.base
    inputs := by
      intro inputIdx hInput
      exact mem_mergeAsc.mpr (Or.inl hInput)
    nested := by
      intro dependency hDependency
      exact mem_mergeAsc.mpr (Or.inl hDependency) }

theorem StageSigLe.joinRight (a b : StageSig) : StageSigLe b (a.join b) :=
  { base := stage_le_join_right a.base b.base
    inputs := by
      intro inputIdx hInput
      exact mem_mergeAsc.mpr (Or.inr hInput)
    nested := by
      intro dependency hDependency
      exact mem_mergeAsc.mpr (Or.inr hDependency) }

theorem StageSigLe.trans {a b c : StageSig}
    (hab : StageSigLe a b) (hbc : StageSigLe b c) : StageSigLe a c :=
  { base := stage_le_trans hab.base hbc.base
    inputs := fun inputIdx hInput => hbc.inputs inputIdx (hab.inputs inputIdx hInput)
    nested := fun dependency hDependency =>
      hbc.nested dependency (hab.nested dependency hDependency) }

private theorem foldl_join_initial (items : List ExprId)
    (sigs : Array StageSig) (initial : StageSig) :
    StageSigLe initial
      (items.foldl (fun accumulated id =>
        accumulated.join (sigs[id.idx]?.getD { base := .s1 })) initial) := by
  induction items generalizing initial with
  | nil => exact StageSigLe.refl initial
  | cons head tail ih =>
    simp only [List.foldl_cons]
    exact (StageSigLe.joinLeft initial _).trans
      (ih (initial := initial.join (sigs[head.idx]?.getD { base := .s1 })))

private theorem foldl_join_contains (items : List ExprId)
    (sigs : Array StageSig) (initial : StageSig) {child : ExprId}
    (hChild : child ∈ items) :
    StageSigLe (sigs[child.idx]?.getD { base := .s1 })
      (items.foldl (fun accumulated id =>
        accumulated.join (sigs[id.idx]?.getD { base := .s1 })) initial) := by
  induction items generalizing initial with
  | nil => simp at hChild
  | cons head tail ih =>
    simp only [List.mem_cons] at hChild
    simp only [List.foldl_cons]
    rcases hChild with rfl | hTail
    · exact (StageSigLe.joinRight initial _).trans
        (foldl_join_initial tail sigs _)
    · exact (ih (initial := initial.join
        (sigs[head.idx]?.getD { base := .s1 })) hTail)

/-- Every semantic child signature is contained in the signature computed for
    its parent node. -/
theorem enodeSig_contains_child (sigs : Array StageSig) (node : ENode)
    {child : ExprId} (hChild : child ∈ node.children) :
    StageSigLe (sigs[child.idx]?.getD { base := .s1 })
      (enodeSig sigs node) := by
  cases node with
  | num | bool | inputRef | paramRef | nestedOut | sampleRate
  | sampleIndex | tileSampleIndex | tilePhase | loopIdx =>
    simp [ENode.children] at hChild
  | arr items | tileArray items =>
    simp only [enodeSig]
    rw [← Array.foldl_toList]
    apply foldl_join_contains items.toList sigs _
    simpa [ENode.children] using hChild
  | binary tag lhs rhs =>
    have hBinary : child = lhs ∨ child = rhs := by
      simpa [ENode.children] using hChild
    simp only [enodeSig]
    rcases hBinary with hLeft | hRight
    · subst child; exact StageSigLe.joinLeft _ _
    · subst child; exact StageSigLe.joinRight _ _
  | unary tag arg =>
    have hUnary : child = arg := by simpa [ENode.children] using hChild
    subst child
    change StageSigLe (sigs[arg.idx]?.getD { base := .s1 })
      (sigs[arg.idx]?.getD { base := .s1 })
    exact StageSigLe.refl _
  | clamp value lo hi | select value lo hi | arraySet value lo hi =>
    have hTernary : child = value ∨ child = lo ∨ child = hi := by
      simpa [ENode.children] using hChild
    simp only [enodeSig]
    rcases hTernary with hValue | hLo | hHi
    · subst child
      exact (StageSigLe.joinLeft _ _).trans (StageSigLe.joinLeft _ _)
    · subst child
      exact (StageSigLe.joinRight _ _).trans (StageSigLe.joinLeft _ _)
    · subst child
      exact StageSigLe.joinRight _ _
  | index array index =>
    have hIndexNode : child = array ∨ child = index := by
      simpa [ENode.children] using hChild
    simp only [enodeSig]
    rcases hIndexNode with hArray | hIndex
    · subst child; exact StageSigLe.joinLeft _ _
    · subst child; exact StageSigLe.joinRight _ _
  | bankSum capacity tables body dynCount binderId =>
    simp only [enodeSig]
    have hBank : (child ∈ tables ∨ child = body) ∨
        dynCount = some child := by
      simpa [ENode.children] using hChild
    rw [← Array.foldl_toList]
    rcases hBank with (hTable | hBody) | hDyn
    · have hTableLe := foldl_join_contains tables.toList sigs
          ({ base := .fold } : StageSig)
          (by simpa using hTable)
      cases dynCount with
      | none => exact hTableLe.trans (StageSigLe.joinLeft _ _)
      | some count =>
        exact (hTableLe.trans (StageSigLe.joinLeft _ _)).trans
          (StageSigLe.joinLeft _ _)
    · subst child
      cases dynCount with
      | none => exact StageSigLe.joinRight _ _
      | some count =>
        exact (StageSigLe.joinRight _ _).trans (StageSigLe.joinLeft _ _)
    · cases dynCount with
      | none => simp at hDyn
      | some count =>
        simp at hDyn
        subst count
        exact StageSigLe.joinRight _ _
  | routedSum capacity outputCount routes tables values dynCount binderId =>
    simp only [enodeSig]
    have hRouted : child ∈ tables ∨ child ∈ values ∨
        dynCount = some child := by
      simpa [ENode.children] using hChild
    rw [← Array.foldl_toList]
    rcases hRouted with hTable | hValue | hDyn
    · have hItemLe := foldl_join_contains (tables ++ values).toList sigs
          ({ base := .fold } : StageSig)
          (by simp; exact Or.inl hTable)
      cases dynCount with
      | none => exact hItemLe
      | some count => exact hItemLe.trans (StageSigLe.joinLeft _ _)
    · have hItemLe := foldl_join_contains (tables ++ values).toList sigs
          ({ base := .fold } : StageSig)
          (by simp; exact Or.inr hValue)
      cases dynCount with
      | none => exact hItemLe
      | some count => exact hItemLe.trans (StageSigLe.joinLeft _ _)
    · cases dynCount with
      | none => simp at hDyn
      | some count =>
        simp at hDyn
        subst count
        exact StageSigLe.joinRight _ _

/-- Resolve one nested-output dependency.  Missing children and outputs are
    maximally dynamic. -/
def childStage (ctx : StageCtx) (instanceIdx outputIdx : Nat) : Stage :=
  resolveNested ctx (instanceIdx, outputIdx)

private theorem foldl_join_le_of_bounded (items : List β)
    (value : β → Stage) (initial upper : Stage)
    (hInitial : initial.le upper = true)
    (hItems : ∀ item ∈ items, (value item).le upper = true) :
    (items.foldl (fun stage item => stage.join (value item)) initial).le
      upper = true := by
  induction items generalizing initial with
  | nil => exact hInitial
  | cons head tail ih =>
    simp only [List.foldl_cons]
    apply ih
    · exact stage_join_least hInitial (hItems head (by simp))
    · intro item hItem
      exact hItems item (by simp [hItem])

theorem resolve_base_le (ctx : StageCtx) (sig : StageSig) :
    sig.base.le (resolve ctx sig) = true := by
  unfold resolve
  rw [← Array.foldl_toList, ← Array.foldl_toList]
  let afterInputs := sig.inputs.toList.foldl (fun stage inputIdx =>
    stage.join (ctx.inputStages[inputIdx]?.getD .s1)) sig.base
  let final := sig.nested.toList.foldl (fun stage dependency =>
    stage.join (resolveNested ctx dependency)) afterInputs
  have hNested := foldl_join_bounded sig.nested.toList
    (resolveNested ctx) afterInputs final (stage_le_refl final)
  have hInputs := foldl_join_bounded sig.inputs.toList
    (fun inputIdx => ctx.inputStages[inputIdx]?.getD .s1)
    sig.base final (by simpa [afterInputs] using hNested.1)
  simpa [afterInputs, final] using hInputs.1

theorem resolve_input_le (ctx : StageCtx) (sig : StageSig) {inputIdx : Nat}
    (hInput : inputIdx ∈ sig.inputs) :
    (ctx.inputStages[inputIdx]?.getD .s1).le (resolve ctx sig) = true := by
  unfold resolve
  rw [← Array.foldl_toList, ← Array.foldl_toList]
  let afterInputs := sig.inputs.toList.foldl (fun stage inputIdx =>
    stage.join (ctx.inputStages[inputIdx]?.getD .s1)) sig.base
  let final := sig.nested.toList.foldl (fun stage dependency =>
    stage.join (resolveNested ctx dependency)) afterInputs
  have hNested := foldl_join_bounded sig.nested.toList
    (resolveNested ctx) afterInputs final (stage_le_refl final)
  have hInputs := foldl_join_bounded sig.inputs.toList
    (fun inputIdx => ctx.inputStages[inputIdx]?.getD .s1)
    sig.base final (by simpa [afterInputs] using hNested.1)
  simpa [afterInputs, final] using
    hInputs.2 inputIdx (by simpa using hInput)

theorem resolve_nested_le (ctx : StageCtx) (sig : StageSig)
    {dependency : Nat × Nat} (hDependency : dependency ∈ sig.nested) :
    (resolveNested ctx dependency).le (resolve ctx sig) = true := by
  unfold resolve
  rw [← Array.foldl_toList, ← Array.foldl_toList]
  let afterInputs := sig.inputs.toList.foldl (fun stage inputIdx =>
    stage.join (ctx.inputStages[inputIdx]?.getD .s1)) sig.base
  let final := sig.nested.toList.foldl (fun stage dependency =>
    stage.join (resolveNested ctx dependency)) afterInputs
  have hNested := foldl_join_bounded sig.nested.toList
    (resolveNested ctx) afterInputs final (stage_le_refl final)
  simpa [afterInputs, final] using
    hNested.2 dependency (by simpa using hDependency)

theorem StageSigLe.resolve_mono {a b : StageSig} (h : StageSigLe a b)
    (ctx : StageCtx) : (resolve ctx a).le (resolve ctx b) = true := by
  unfold resolve
  rw [← Array.foldl_toList, ← Array.foldl_toList,
    ← Array.foldl_toList, ← Array.foldl_toList]
  let upper :=
    b.nested.toList.foldl (fun stage dependency =>
      stage.join (resolveNested ctx dependency))
      (b.inputs.toList.foldl (fun stage inputIdx =>
        stage.join (ctx.inputStages[inputIdx]?.getD .s1)) b.base)
  have hBase : a.base.le upper = true :=
    stage_le_trans h.base (by
      simpa [upper, resolve, ← Array.foldl_toList] using resolve_base_le ctx b)
  have hInputs : ∀ inputIdx ∈ a.inputs.toList,
      (ctx.inputStages[inputIdx]?.getD .s1).le upper = true := by
    intro inputIdx hInput
    simpa [upper, resolve, ← Array.foldl_toList] using
      resolve_input_le ctx b (h.inputs inputIdx (by simpa using hInput))
  have hAfterInputs := foldl_join_le_of_bounded a.inputs.toList
    (fun inputIdx => ctx.inputStages[inputIdx]?.getD .s1)
    a.base upper hBase hInputs
  apply foldl_join_le_of_bounded a.nested.toList (resolveNested ctx) _ upper
    hAfterInputs
  intro dependency hDependency
  simpa [upper, resolve, ← Array.foldl_toList] using
    resolve_nested_le ctx b (h.nested dependency (by simpa using hDependency))

/-- Semantic environment agreement through one binding time.  Rate and open
    loop binders are structural/fold inputs.  Control parameters become fixed
    at `s0`; the ordinary and tile sample clocks independently become fixed at
    `s1`.  Symbolic input and nested-output dependencies agree exactly when
    their resolved stage is no later than the requested boundary.  Agreement
    is phrased through the production lookup operations, preserving refusal
    behavior as well as successful values. -/
def EnvAgreesThrough (stage : Stage) (ctx : StageCtx)
    (a b : SigEnv α) : Prop :=
  a.sampleRate = b.sampleRate ∧
  a.loops = b.loops ∧
  (Stage.s0.le stage = true →
    ∀ i, lookupValue "paramRef" a.params i =
      lookupValue "paramRef" b.params i) ∧
  (Stage.s1.le stage = true → a.sampleIndex = b.sampleIndex) ∧
  (Stage.s1.le stage = true → a.tileSampleIndex = b.tileSampleIndex) ∧
  (∀ i, (ctx.inputStages[i]?.getD Stage.s1).le stage = true →
    lookupValue "inputRef" a.inputs i =
      lookupValue "inputRef" b.inputs i) ∧
  (∀ instanceIdx outputIdx,
    (childStage ctx instanceIdx outputIdx).le stage = true →
    lookupNested a instanceIdx outputIdx =
      lookupNested b instanceIdx outputIdx)

theorem EnvAgreesThrough.sampleRate {stage : Stage} {ctx : StageCtx}
    {a b : SigEnv α} (h : EnvAgreesThrough stage ctx a b) :
    a.sampleRate = b.sampleRate := h.1

theorem EnvAgreesThrough.loops {stage : Stage} {ctx : StageCtx}
    {a b : SigEnv α} (h : EnvAgreesThrough stage ctx a b) :
    a.loops = b.loops := h.2.1

theorem EnvAgreesThrough.params {stage : Stage} {ctx : StageCtx}
    {a b : SigEnv α} (h : EnvAgreesThrough stage ctx a b)
    (hstage : Stage.s0.le stage = true) (i : Nat) :
    lookupValue "paramRef" a.params i = lookupValue "paramRef" b.params i :=
  h.2.2.1 hstage i

theorem EnvAgreesThrough.sampleIndex {stage : Stage} {ctx : StageCtx}
    {a b : SigEnv α} (h : EnvAgreesThrough stage ctx a b)
    (hstage : Stage.s1.le stage = true) : a.sampleIndex = b.sampleIndex :=
  h.2.2.2.1 hstage

theorem EnvAgreesThrough.tileSampleIndex {stage : Stage} {ctx : StageCtx}
    {a b : SigEnv α} (h : EnvAgreesThrough stage ctx a b)
    (hstage : Stage.s1.le stage = true) :
    a.tileSampleIndex = b.tileSampleIndex :=
  h.2.2.2.2.1 hstage

theorem EnvAgreesThrough.input {stage : Stage} {ctx : StageCtx}
    {a b : SigEnv α} (h : EnvAgreesThrough stage ctx a b) (i : Nat)
    (hstage : (ctx.inputStages[i]?.getD Stage.s1).le stage = true) :
    lookupValue "inputRef" a.inputs i = lookupValue "inputRef" b.inputs i :=
  h.2.2.2.2.2.1 i hstage

theorem EnvAgreesThrough.nested {stage : Stage} {ctx : StageCtx}
    {a b : SigEnv α} (h : EnvAgreesThrough stage ctx a b)
    (instanceIdx outputIdx : Nat)
    (hstage : (childStage ctx instanceIdx outputIdx).le stage = true) :
    lookupNested a instanceIdx outputIdx = lookupNested b instanceIdx outputIdx :=
  h.2.2.2.2.2.2 instanceIdx outputIdx hstage

theorem EnvAgreesThrough.bindLoop {stage : Stage} {ctx : StageCtx}
    {a b : SigEnv α} (h : EnvAgreesThrough stage ctx a b)
    (binderId : Nat) (value : Value α) :
    EnvAgreesThrough stage ctx (a.bindLoop binderId value)
      (b.bindLoop binderId value) := by
  refine ⟨h.sampleRate, ?_, h.2.2.1, h.2.2.2.1,
    h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2⟩
  funext query
  simp only [SigEnv.bindLoop]
  split
  · rfl
  · exact congrFun h.loops query

/-- Pointwise order on staging contexts.  Missing entries remain maximally
    dynamic, so extending a context with earlier-stage bindings can only lower
    the resolved stage. -/
def StageCtxLe (a b : StageCtx) : Prop :=
  (∀ (i : Nat), (a.inputStages[i]?.getD Stage.s1).le
      (b.inputStages[i]?.getD Stage.s1) = true) ∧
  (∀ (instanceIdx outputIdx : Nat),
    (childStage a instanceIdx outputIdx).le
      (childStage b instanceIdx outputIdx) = true)

theorem resolve_le_s1 (ctx : StageCtx) (sig : StageSig) :
    (resolve ctx sig).le .s1 = true := by
  cases h : resolve ctx sig <;> rfl

theorem stageOf_le_s1 (arena : ExprArena) (ctx : StageCtx) (id : ExprId) :
    (stageOf arena ctx id).le .s1 = true := by
  cases h : stageOf arena ctx id <;> rfl

/-- A dangling expression id is classified maximally dynamically. -/
theorem stageOf_dangling {arena : ExprArena} {ctx : StageCtx} {id : ExprId}
    (h : arena.sig? id = none) : stageOf arena ctx id = .s1 := by
  simp [stageOf, h]

/-- Any successful signature lookup resolves exactly that stored signature. -/
theorem stageOf_resolves {arena : ExprArena} {ctx : StageCtx} {id : ExprId}
    {sig : StageSig} (h : arena.sig? id = some sig) :
    stageOf arena ctx id = resolve ctx sig := by
  simp [stageOf, h]

theorem child_stage_le_of_parent {arena : ExprArena}
    (hSound : SignaturesSound arena) (ctx : StageCtx)
    {id child : ExprId} {node : ENode}
    (hDeref : arena.deref id = some node)
    (hChild : child ∈ node.children) {stage : Stage}
    (hStage : (stageOf arena ctx id).le stage = true) :
    (stageOf arena ctx child).le stage = true := by
  have hIdBound := Tropical.Semantics.deref_index_lt hDeref
  have hChildLt := hSound.arenaWellFormed.childrenDescend
    hDeref child hChild
  have hGenerated := hSound.generated id.idx hIdBound
  have hNode : arena.nodes[id.idx] = node := by
    simpa [ExprArena.deref, Array.getElem?_eq_getElem, hIdBound] using hDeref
  let sigPrefix := arena.sigs.extract 0 id.idx
  have hPrefixChild : sigPrefix[child.idx]? = arena.sigs[child.idx]? := by
    rw [Array.getElem?_extract_of_lt]
    · simp
    · simp [sigPrefix, hSound.arenaWellFormed.signaturesAligned]
      omega
  have hContains := enodeSig_contains_child sigPrefix node hChild
  rw [hPrefixChild] at hContains
  have hResolveChild := hContains.resolve_mono ctx
  have hParentResolve :
      (resolve ctx (enodeSig sigPrefix node)).le stage = true := by
    rw [stageOf_resolves hGenerated] at hStage
    simpa [sigPrefix, hNode] using hStage
  have hChildSigBound : child.idx < arena.sigs.size := by
    rw [hSound.arenaWellFormed.signaturesAligned]
    exact Nat.lt_trans hChildLt hIdBound
  have hChildSig : arena.sigs[child.idx]? =
      some arena.sigs[child.idx] := by
    simpa [Array.getElem?_eq_getElem, hChildSigBound]
  rw [hChildSig] at hResolveChild
  rw [stageOf_resolves hChildSig]
  exact stage_le_trans hResolveChild hParentResolve

/-- Intern-time stage classification is semantically conservative for every
    generated expression DAG.  The theorem compares production denotations,
    including refusal results, under the exact binding-time agreement rules. -/
theorem stageSig_sound {arena : ExprArena} (hSound : SignaturesSound arena)
    (alg : Algebra α) (ctx : StageCtx) (a b : SigEnv α)
    {id : ExprId} {node : ENode} (hDeref : arena.deref id = some node)
    {stage : Stage} (hStage : (stageOf arena ctx id).le stage = true)
    (hEnv : EnvAgreesThrough stage ctx a b) :
    denoteExpr alg a arena hSound.arenaWellFormed id =
      denoteExpr alg b arena hSound.arenaWellFormed id := by
  have hChildEq (child : ExprId) (hChild : child ∈ node.children) :
      denoteExpr alg a arena hSound.arenaWellFormed child =
        denoteExpr alg b arena hSound.arenaWellFormed child := by
    have hChildStage := child_stage_le_of_parent hSound ctx hDeref hChild hStage
    obtain ⟨childNode, hChildDeref⟩ :=
      Tropical.Semantics.deref_of_index_lt
        (Nat.lt_trans
          (hSound.arenaWellFormed.childrenDescend hDeref child hChild)
          (Tropical.Semantics.deref_index_lt hDeref))
    exact stageSig_sound hSound alg ctx a b hChildDeref hChildStage hEnv
  have hIdBound := Tropical.Semantics.deref_index_lt hDeref
  have hGenerated := hSound.generated id.idx hIdBound
  have hNode : arena.nodes[id.idx] = node := by
    simpa [ExprArena.deref, Array.getElem?_eq_getElem, hIdBound] using hDeref
  let parentSig := enodeSig (arena.sigs.extract 0 id.idx) node
  have hParentResolve : (resolve ctx parentSig).le stage = true := by
    rw [stageOf_resolves hGenerated] at hStage
    simpa [parentSig, hNode] using hStage
  rw [denoteExpr_of_deref alg a arena hSound.arenaWellFormed hDeref,
    denoteExpr_of_deref alg b arena hSound.arenaWellFormed hDeref]
  cases node with
  | num | bool => rfl
  | sampleRate => exact congrArg Except.ok hEnv.sampleRate
  | sampleIndex =>
    exact congrArg Except.ok (hEnv.sampleIndex
      (stage_le_trans (resolve_base_le ctx parentSig) hParentResolve))
  | tileSampleIndex =>
    exact congrArg Except.ok (hEnv.tileSampleIndex
      (stage_le_trans (resolve_base_le ctx parentSig) hParentResolve))
  | tilePhase => rfl
  | loopIdx binderId =>
    simp only [denoteNode]
    rw [hEnv.loops]
  | paramRef paramIdx =>
    exact hEnv.params
      (stage_le_trans (resolve_base_le ctx parentSig) hParentResolve)
      paramIdx.idx
  | inputRef inputIdx =>
    exact hEnv.input inputIdx.idx
      (stage_le_trans
        (resolve_input_le ctx parentSig (by simp [parentSig, enodeSig]))
        hParentResolve)
  | nestedOut instanceIdx outputIdx =>
    exact hEnv.nested instanceIdx.idx outputIdx.idx
      (stage_le_trans
        (resolve_nested_le ctx parentSig (by simp [parentSig, enodeSig]))
        hParentResolve)
  | arr items | tileArray items =>
    simp only [denoteNode]
    have hItems :
        items.attach.map
            (fun item => denoteExpr alg a arena hSound.arenaWellFormed item.1) =
          items.attach.map
            (fun item => denoteExpr alg b arena hSound.arenaWellFormed item.1) := by
      apply Array.ext
      · simp
      · intro i hiA hiB
        simp only [Array.getElem_map, Array.getElem_attach]
        apply hChildEq
        simp [ENode.children]
    rw [hItems]
  | binary tag lhs rhs =>
    simp only [denoteNode]
    rw [hChildEq lhs (by simp [ENode.children]),
      hChildEq rhs (by simp [ENode.children])]
  | unary tag arg =>
    simp only [denoteNode]
    rw [hChildEq arg (by simp [ENode.children])]
  | clamp value lo hi =>
    simp only [denoteNode]
    rw [hChildEq value (by simp [ENode.children]),
      hChildEq lo (by simp [ENode.children]),
      hChildEq hi (by simp [ENode.children])]
  | select cond then_ else_ =>
    simp only [denoteNode]
    rw [hChildEq cond (by simp [ENode.children]),
      hChildEq then_ (by simp [ENode.children]),
      hChildEq else_ (by simp [ENode.children])]
  | arraySet array index value =>
    rfl
  | index array index =>
    simp only [denoteNode]
    rw [hChildEq array (by simp [ENode.children]),
      hChildEq index (by simp [ENode.children])]
  | bankSum capacity tables body dynCount binderId =>
    simp only [denoteNode]
    have hBodyChild : body ∈
        (ENode.bankSum capacity tables body dynCount binderId).children := by
      simp [ENode.children]
    have hTableArray :
        tables.attach.map
            (fun item => denoteExpr alg a arena hSound.arenaWellFormed item.1) =
          tables.attach.map
            (fun item => denoteExpr alg b arena hSound.arenaWellFormed item.1) := by
      apply Array.ext
      · simp
      · intro i hiA hiB
        simp only [Array.getElem_map, Array.getElem_attach]
        apply hChildEq
        simp [ENode.children]
    have hBody (loopValue : Value α) :
        denoteExpr alg (a.bindLoop binderId loopValue)
            arena hSound.arenaWellFormed body =
          denoteExpr alg (b.bindLoop binderId loopValue)
            arena hSound.arenaWellFormed body := by
      have hBodyStage := child_stage_le_of_parent hSound ctx hDeref
        hBodyChild hStage
      obtain ⟨bodyNode, hBodyDeref⟩ :=
        Tropical.Semantics.deref_of_index_lt
          (Nat.lt_trans
            (hSound.arenaWellFormed.childrenDescend hDeref body
              hBodyChild) hIdBound)
      exact stageSig_sound hSound alg ctx
        (a.bindLoop binderId loopValue) (b.bindLoop binderId loopValue)
        hBodyDeref hBodyStage (hEnv.bindLoop binderId loopValue)
    rw [hTableArray]
    cases dynCount with
    | none => simp only [hBody]
    | some count =>
      simp only [hChildEq count (by simp [ENode.children]), hBody]
  | routedSum capacity outputCount routes tables values dynCount binderId =>
    simp only [denoteNode]
    have hTableArray :
        tables.attach.map
            (fun item => denoteExpr alg a arena hSound.arenaWellFormed item.1) =
          tables.attach.map
            (fun item => denoteExpr alg b arena hSound.arenaWellFormed item.1) := by
      apply Array.ext
      · simp
      · intro i hiA hiB
        simp only [Array.getElem_map, Array.getElem_attach]
        apply hChildEq
        simp [ENode.children]
    have hValueArray (loopValue : Value α) :
        values.attach.map
            (fun item => denoteExpr alg (a.bindLoop binderId loopValue)
              arena hSound.arenaWellFormed item.1) =
          values.attach.map
            (fun item => denoteExpr alg (b.bindLoop binderId loopValue)
              arena hSound.arenaWellFormed item.1) := by
      apply Array.ext
      · simp
      · intro i hiA hiB
        simp only [Array.getElem_map, Array.getElem_attach]
        have hi : i < values.size := by simpa using hiA
        let value := values[i]
        have hValueMem : value ∈ values := Array.getElem_mem hi
        have hValueChild : value ∈
            (ENode.routedSum capacity outputCount routes tables values dynCount
              binderId).children := by
          simp only [ENode.children, Array.mem_append]
          exact Or.inl (Or.inr hValueMem)
        have hValueStage := child_stage_le_of_parent hSound ctx hDeref
          hValueChild hStage
        obtain ⟨valueNode, hValueDeref⟩ :=
          Tropical.Semantics.deref_of_index_lt
            (Nat.lt_trans
              (hSound.arenaWellFormed.childrenDescend hDeref value
                hValueChild) hIdBound)
        exact stageSig_sound hSound alg ctx
          (a.bindLoop binderId loopValue) (b.bindLoop binderId loopValue)
          hValueDeref hValueStage (hEnv.bindLoop binderId loopValue)
    rw [hTableArray]
    cases dynCount with
    | none => simp only [hValueArray]
    | some count =>
      simp only [hChildEq count (by simp [ENode.children]), hValueArray]
termination_by id.idx
decreasing_by
  all_goals
    apply hSound.arenaWellFormed.childrenDescend hDeref
    simp_all [ENode.children]

theorem stageSig_sound_sampleRate (alg : Algebra α) (ctx : StageCtx)
    (a b : SigEnv α) (arena : ExprArena) (hArena : ArenaWellFormed arena)
    {id : ExprId} (hDeref : arena.deref id = some .sampleRate)
    (henv : EnvAgreesThrough stage ctx a b) :
    denoteExpr alg a arena hArena id = denoteExpr alg b arena hArena id := by
  rw [denoteExpr_of_deref alg a arena hArena hDeref,
    denoteExpr_of_deref alg b arena hArena hDeref]
  exact congrArg Except.ok henv.sampleRate

theorem stageSig_sound_paramRef (alg : Algebra α) (ctx : StageCtx)
    (a b : SigEnv α) (arena : ExprArena) (hArena : ArenaWellFormed arena)
    {id : ExprId} {paramIdx : Nat}
    (hDeref : arena.deref id = some (.paramRef ⟨paramIdx⟩))
    (hstage : Stage.s0.le stage = true)
    (henv : EnvAgreesThrough stage ctx a b) :
    denoteExpr alg a arena hArena id = denoteExpr alg b arena hArena id := by
  rw [denoteExpr_of_deref alg a arena hArena hDeref,
    denoteExpr_of_deref alg b arena hArena hDeref]
  exact henv.params hstage paramIdx

theorem stageSig_sound_inputRef (alg : Algebra α) (ctx : StageCtx)
    (a b : SigEnv α) (arena : ExprArena) (hArena : ArenaWellFormed arena)
    {id : ExprId} {inputIdx : Nat}
    (hDeref : arena.deref id = some (.inputRef ⟨inputIdx⟩))
    (hstage : (ctx.inputStages[inputIdx]?.getD Stage.s1).le stage = true)
    (henv : EnvAgreesThrough stage ctx a b) :
    denoteExpr alg a arena hArena id = denoteExpr alg b arena hArena id := by
  rw [denoteExpr_of_deref alg a arena hArena hDeref,
    denoteExpr_of_deref alg b arena hArena hDeref]
  exact henv.input inputIdx hstage

theorem stageSig_sound_nestedOut (alg : Algebra α) (ctx : StageCtx)
    (a b : SigEnv α) (arena : ExprArena) (hArena : ArenaWellFormed arena)
    {id : ExprId} {instanceIdx outputIdx : Nat}
    (hDeref : arena.deref id =
      some (.nestedOut ⟨instanceIdx⟩ ⟨outputIdx⟩))
    (hstage : (childStage ctx instanceIdx outputIdx).le stage = true)
    (henv : EnvAgreesThrough stage ctx a b) :
    denoteExpr alg a arena hArena id = denoteExpr alg b arena hArena id := by
  rw [denoteExpr_of_deref alg a arena hArena hDeref,
    denoteExpr_of_deref alg b arena hArena hDeref]
  exact henv.nested instanceIdx outputIdx hstage

theorem stageSig_sound_sampleIndex (alg : Algebra α) (ctx : StageCtx)
    (a b : SigEnv α) (arena : ExprArena) (hArena : ArenaWellFormed arena)
    {id : ExprId} (hDeref : arena.deref id = some .sampleIndex)
    (hstage : Stage.s1.le stage = true)
    (henv : EnvAgreesThrough stage ctx a b) :
    denoteExpr alg a arena hArena id = denoteExpr alg b arena hArena id := by
  rw [denoteExpr_of_deref alg a arena hArena hDeref,
    denoteExpr_of_deref alg b arena hArena hDeref]
  exact congrArg Except.ok (henv.sampleIndex hstage)

theorem stageSig_sound_tileSampleIndex (alg : Algebra α) (ctx : StageCtx)
    (a b : SigEnv α) (arena : ExprArena) (hArena : ArenaWellFormed arena)
    {id : ExprId} (hDeref : arena.deref id = some .tileSampleIndex)
    (hstage : Stage.s1.le stage = true)
    (henv : EnvAgreesThrough stage ctx a b) :
    denoteExpr alg a arena hArena id = denoteExpr alg b arena hArena id := by
  rw [denoteExpr_of_deref alg a arena hArena hDeref,
    denoteExpr_of_deref alg b arena hArena hDeref]
  exact congrArg Except.ok (henv.tileSampleIndex hstage)

theorem stageSig_sound_tilePhase (alg : Algebra α) (ctx : StageCtx)
    (a b : SigEnv α) (arena : ExprArena) (hArena : ArenaWellFormed arena)
    {id : ExprId} (hDeref : arena.deref id = some .tilePhase)
    (_henv : EnvAgreesThrough stage ctx a b) :
    denoteExpr alg a arena hArena id = denoteExpr alg b arena hArena id := by
  rw [denoteExpr_of_deref alg a arena hArena hDeref,
    denoteExpr_of_deref alg b arena hArena hDeref]
  rfl

/-- The absolute tile clock reads its independent semantic environment rail.
    Ordinary exact/JIT callers may bind it equal to `sampleIndex`, while a
    materializer may supply an endpoint coordinate without changing the audio
    carrier clock. -/
theorem denoteExpr_tileSampleIndex (alg : Algebra α) (env : SigEnv α)
    (arena : ExprArena) (hArena : ArenaWellFormed arena) {id : ExprId}
    (hDeref : arena.deref id = some .tileSampleIndex) :
    denoteExpr alg env arena hArena id = .ok env.tileSampleIndex := by
  rw [denoteExpr_of_deref alg env arena hArena hDeref]
  rfl

/-- Direct/JIT semantics observes the exact left endpoint: `tilePhase` is the
    literal zero.  This deliberately says nothing about nonzero interpolated
    Metal lanes. -/
theorem denoteExpr_tilePhase_zero (alg : Algebra α) (env : SigEnv α)
    (arena : ExprArena) (hArena : ArenaWellFormed arena) {id : ExprId}
    (hDeref : arena.deref id = some .tilePhase) :
    denoteExpr alg env arena hArena id = alg.literal (0 : Nat) := by
  rw [denoteExpr_of_deref alg env arena hArena hDeref]
  rfl

/-- Marking `tilePhase` as per-sample is conservative: its direct expression
    denotation is nevertheless the constant left endpoint. -/
theorem tilePhase_signature_is_conservative :
    (enodeSig #[] .tilePhase).base = .s1 := by
  rfl

end Tropical.Semantics.Staging
