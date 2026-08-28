import Tropical.Semantics.Program
import Tropical.Semantics.WellFormed
import Tropical.Ir.Strata.EArena

/-!
# Strata-exit refinement

The executable reachability copy retains and validates its complete ID-renaming
memo.  This module turns that checked witness into the canonical semantic
relations and proves expression denotation preservation independently of ID
identity or destination sharing.
-/

namespace Tropical.Semantics

open Tropical.Ir
open Tropical.Ir.Core
open Tropical.Ir.Strata

/-- Every completed memo entry relates exactly dereferenced constructors after
    renaming all expression children. -/
def ExprCopyMapRel (src dst : ExprArena) (memo : ExprCopyMemo) : Prop :=
  ∀ ⦃srcId dstId⦄, memo[srcId.idx]? = some dstId →
    ∃ srcNode dstNode,
      src.deref srcId = some srcNode ∧
      remapENode? memo srcNode = some dstNode ∧
      dst.deref dstId = some dstNode

/-- Proof-visible correspondence between one source root and its copied root. -/
def ExprCopyRel (src dst : ExprArena) (srcId dstId : ExprId) : Prop :=
  ∃ memo, memo[srcId.idx]? = some dstId ∧ ExprCopyMapRel src dst memo

theorem semanticWfCheck_sound {arena : ExprArena}
    (hcheck : semanticWfCheck arena = true) : ArenaWellFormed arena := by
  simp only [semanticWfCheck, Bool.and_eq_true] at hcheck
  obtain ⟨⟨hDescend, hSigs⟩, hDedup⟩ := hcheck
  constructor
  · exact childrenDescend_of_wf hDescend
  · intro node id hGet
    have hMem : (node, id) ∈ arena.dedup.toList :=
      Std.HashMap.mem_toList_iff_getElem?_eq_some.mpr hGet
    have hEntry := List.all_eq_true.mp hDedup (node, id) hMem
    exact eq_of_beq hEntry
  · exact eq_of_beq hSigs

theorem checkExprCopyMemo_sound {src dst : ExprArena} {memo : ExprCopyMemo}
    (hcheck : checkExprCopyMemo src dst memo = true) :
    ExprCopyMapRel src dst memo := by
  intro srcId dstId hGet
  have hMem : (srcId.idx, dstId) ∈ memo.toList :=
    Std.HashMap.mem_toList_iff_getElem?_eq_some.mpr hGet
  have hEntry := List.all_eq_true.mp hcheck (srcId.idx, dstId) hMem
  split at hEntry
  next hNone => simp at hEntry
  next srcNode hSrc =>
    split at hEntry
    next hNone => simp at hEntry
    next dstNode hRemap =>
      exact ⟨srcNode, dstNode, hSrc, hRemap, eq_of_beq hEntry⟩

/-- Recursive correspondence of the evaluator-reachable program spine.  The
    registry array is related in first-use order, while unreachable source
    pool entries are intentionally absent. -/
inductive ProgramCopyRel (ea : EArena)
    (hPrograms : progPoolWf ea.programs = true) (memo : ExprCopyMemo) :
    ProgramIdx → CoreProgram → Prop where
  | node {root : ProgramIdx} {source : Program} {core : CoreProgram}
      {children : Array (String × {t : ProgramIdx // t.idx < root.idx})}
      (sourceAt : ea.programs[root.idx]? = some source)
      (viewCopied : remapProgramCopyView? memo
        (Tropical.Ir.Strata.Program.copyView source) =
          some (Tropical.Ir.Strata.CoreProgram.copyView core))
      (childrenCollected :
        referencedPrograms ea hPrograms root source sourceAt = .ok children)
      (registryKeys : children.map (·.1) = core.registry.map (·.1))
      (instancePresent : ∀ {name typeKey inputs},
        BodyDecl.inst name typeKey inputs ∈ source.decls →
        ∃ sourceChild coreChild,
          source.registryGet? typeKey = some sourceChild ∧
          core.registryGet? typeKey = some coreChild)
      (instanceCopied : ∀ {name typeKey inputs sourceChild coreChild},
        BodyDecl.inst name typeKey inputs ∈ source.decls →
        source.registryGet? typeKey = some sourceChild →
        core.registryGet? typeKey = some coreChild →
        ProgramCopyRel ea hPrograms memo sourceChild coreChild) :
      ProgramCopyRel ea hPrograms memo root core

theorem checkProgramCopy_sound {ea : EArena}
    (hPrograms : progPoolWf ea.programs = true) {memo : ExprCopyMemo}
    {root : ProgramIdx} {core : CoreProgram}
    (hcheck : checkProgramCopy ea hPrograms memo root core = true) :
    ProgramCopyRel ea hPrograms memo root core := by
  rw [checkProgramCopy] at hcheck
  split at hcheck
  next hp => simp at hcheck
  next source hp =>
    split at hcheck
    next hview => simp at hcheck
    next copiedView hview =>
      split at hcheck
      next error hchildren => simp at hcheck
      next children hchildren =>
        simp only [Bool.and_eq_true] at hcheck
        obtain ⟨⟨hViewEq, hKeysEq⟩, hChildren⟩ := hcheck
        refine .node hp ?_ hchildren (eq_of_beq hKeysEq) ?_ ?_
        · exact hview.trans (congrArg some (eq_of_beq hViewEq))
        · intro name typeKey inputs hMem
          have hChildCheck := Array.all_eq_true'.mp hChildren
            (.inst name typeKey inputs) hMem
          simp only at hChildCheck
          split at hChildCheck
          next hs => simp at hChildCheck
          next sourceChild hs =>
            split at hChildCheck
            next hr => simp at hChildCheck
            next coreChild hr => exact ⟨sourceChild, coreChild, hs, hr⟩
        · intro name typeKey inputs sourceChild coreChild hMem hs hr
          have hChildCheck := Array.all_eq_true'.mp hChildren
            (.inst name typeKey inputs) hMem
          simp only at hChildCheck
          rw [hs, hr] at hChildCheck
          exact checkProgramCopy_sound hPrograms hChildCheck
termination_by root.idx
decreasing_by
  apply progPool_registry_lt hPrograms (by assumption) hs

private inductive ExprIdsCopyRel (memo : ExprCopyMemo) :
    List ExprId → List ExprId → Prop where
  | nil : ExprIdsCopyRel memo [] []
  | cons : memo[srcId.idx]? = some dstId →
      ExprIdsCopyRel memo src dst →
      ExprIdsCopyRel memo (srcId :: src) (dstId :: dst)

private theorem remapExprIdList?_eq_some_iff
    (memo : ExprCopyMemo) (src dst : List ExprId) :
    remapExprIdList? memo src = some dst ↔ ExprIdsCopyRel memo src dst := by
  induction src generalizing dst with
  | nil =>
    cases dst with
    | nil => exact ⟨fun _ => .nil, fun _ => rfl⟩
    | cons head tail =>
      constructor
      · intro h
        simp [remapExprIdList?] at h
      · intro h
        cases h
  | cons head tail ih =>
    constructor
    · intro h
      cases hHead : memo[head.idx]? with
      | none => simp [remapExprIdList?, hHead] at h
      | some mappedHead =>
        cases hTail : remapExprIdList? memo tail with
        | none => simp [remapExprIdList?, hHead, hTail] at h
        | some mappedTail =>
          simp [remapExprIdList?, hHead, hTail] at h
          subst dst
          exact .cons hHead ((ih mappedTail).mp hTail)
    · intro h
      cases h with
      | cons hHead hTail =>
        rw [remapExprIdList?, hHead, (ih _).mpr hTail]

private theorem remapExprIds?_eq_some_iff
    (memo : ExprCopyMemo) (src dst : Array ExprId) :
    remapExprIds? memo src = some dst ↔
      ExprIdsCopyRel memo src.toList dst.toList := by
  constructor
  · intro h
    cases hMapped : remapExprIdList? memo src.toList with
    | none => simp [remapExprIds?, hMapped] at h
    | some mapped =>
      simp [remapExprIds?, hMapped] at h
      subst dst
      simpa using
        (remapExprIdList?_eq_some_iff memo src.toList mapped).mp hMapped
  · intro h
    have hMapped :=
      (remapExprIdList?_eq_some_iff memo src.toList dst.toList).mpr h
    simp [remapExprIds?, hMapped]

private theorem ExprIdsCopyRel.length {memo : ExprCopyMemo} {src dst}
    (h : ExprIdsCopyRel memo src dst) : src.length = dst.length := by
  induction h with
  | nil => rfl
  | cons _ _ ih => simp [ih]

private theorem ExprIdsCopyRel.get {memo : ExprCopyMemo} {src dst}
    (h : ExprIdsCopyRel memo src dst) (index : Nat)
    (hSrc : index < src.length) (hDst : index < dst.length) :
    memo[src[index].idx]? = some dst[index] := by
  induction h generalizing index with
  | nil => simp at hSrc
  | cons hHead hTail ih =>
    cases index with
    | zero => simpa using hHead
    | succ index =>
      simp only [List.length_cons, Nat.succ_lt_succ_iff] at hSrc hDst
      simpa using ih index hSrc hDst

private theorem denoteExpr_copy_map_eq {src dst : ExprArena}
    {memo : ExprCopyMemo} (hMap : ExprCopyMapRel src dst memo)
    (alg : Algebra α) (env : SigEnv α)
    (hSrcArena : ArenaWellFormed src) (hDstArena : ArenaWellFormed dst)
    {srcId dstId : ExprId} (hMemo : memo[srcId.idx]? = some dstId) :
    denoteExpr alg env src hSrcArena srcId =
      denoteExpr alg env dst hDstArena dstId := by
  obtain ⟨srcNode, dstNode, hSrcDeref, hRemap, hDstDeref⟩ := hMap hMemo
  rw [denoteExpr_of_deref alg env src hSrcArena hSrcDeref,
    denoteExpr_of_deref alg env dst hDstArena hDstDeref]
  have copyArray (childEnv : SigEnv α) (srcIds dstIds : Array ExprId)
      (hIds : ExprIdsCopyRel memo srcIds.toList dstIds.toList)
      (hChildren : ∀ child ∈ srcIds, child ∈ srcNode.children) :
      srcIds.attach.map
          (fun item => denoteExpr alg childEnv src hSrcArena item.1) =
        dstIds.attach.map
          (fun item => denoteExpr alg childEnv dst hDstArena item.1) := by
    apply Array.ext
    · simpa using hIds.length
    · intro index hSrcIndex hDstIndex
      simp only [Array.getElem_map, Array.getElem_attach]
      have hs : index < srcIds.size := by simpa using hSrcIndex
      have hd : index < dstIds.size := by simpa using hDstIndex
      apply denoteExpr_copy_map_eq hMap alg childEnv hSrcArena hDstArena
      apply hIds.get index <;> simpa using ‹_›
  cases srcNode with
  | num value =>
    simp [remapENode?] at hRemap
    subst dstNode
    rfl
  | bool value =>
    simp [remapENode?] at hRemap
    subst dstNode
    rfl
  | arr items =>
    cases hItems : remapExprIds? memo items with
    | none => simp [remapENode?, hItems] at hRemap
    | some copied =>
      simp [remapENode?, hItems] at hRemap
      subst dstNode
      simp only [denoteNode]
      rw [copyArray env items copied
        ((remapExprIds?_eq_some_iff memo items copied).mp hItems)
        (by simp [ENode.children])]
  | tileArray items =>
    cases hItems : remapExprIds? memo items with
    | none => simp [remapENode?, hItems] at hRemap
    | some copied =>
      simp [remapENode?, hItems] at hRemap
      subst dstNode
      simp only [denoteNode]
      rw [copyArray env items copied
        ((remapExprIds?_eq_some_iff memo items copied).mp hItems)
        (by simp [ENode.children])]
  | binary tag lhs rhs =>
    cases hLhs : memo[lhs.idx]? with
    | none => simp [remapENode?, hLhs] at hRemap
    | some copiedLhs =>
      cases hRhs : memo[rhs.idx]? with
      | none => simp [remapENode?, hLhs, hRhs] at hRemap
      | some copiedRhs =>
        simp [remapENode?, hLhs, hRhs] at hRemap
        subst dstNode
        simp only [denoteNode]
        rw [denoteExpr_copy_map_eq hMap alg env hSrcArena hDstArena hLhs,
          denoteExpr_copy_map_eq hMap alg env hSrcArena hDstArena hRhs]
  | unary tag arg =>
    cases hArg : memo[arg.idx]? with
    | none => simp [remapENode?, hArg] at hRemap
    | some copiedArg =>
      simp [remapENode?, hArg] at hRemap
      subst dstNode
      simp only [denoteNode]
      rw [denoteExpr_copy_map_eq hMap alg env hSrcArena hDstArena hArg]
  | clamp value lo hi =>
    cases hValue : memo[value.idx]? with
    | none => simp [remapENode?, hValue] at hRemap
    | some copiedValue =>
      cases hLo : memo[lo.idx]? with
      | none => simp [remapENode?, hValue, hLo] at hRemap
      | some copiedLo =>
        cases hHi : memo[hi.idx]? with
        | none => simp [remapENode?, hValue, hLo, hHi] at hRemap
        | some copiedHi =>
          simp [remapENode?, hValue, hLo, hHi] at hRemap
          subst dstNode
          simp only [denoteNode]
          rw [denoteExpr_copy_map_eq hMap alg env hSrcArena hDstArena hValue,
            denoteExpr_copy_map_eq hMap alg env hSrcArena hDstArena hLo,
            denoteExpr_copy_map_eq hMap alg env hSrcArena hDstArena hHi]
  | select cond then_ else_ =>
    cases hCond : memo[cond.idx]? with
    | none => simp [remapENode?, hCond] at hRemap
    | some copiedCond =>
      cases hThen : memo[then_.idx]? with
      | none => simp [remapENode?, hCond, hThen] at hRemap
      | some copiedThen =>
        cases hElse : memo[else_.idx]? with
        | none => simp [remapENode?, hCond, hThen, hElse] at hRemap
        | some copiedElse =>
          simp [remapENode?, hCond, hThen, hElse] at hRemap
          subst dstNode
          simp only [denoteNode]
          rw [denoteExpr_copy_map_eq hMap alg env hSrcArena hDstArena hCond,
            denoteExpr_copy_map_eq hMap alg env hSrcArena hDstArena hThen,
            denoteExpr_copy_map_eq hMap alg env hSrcArena hDstArena hElse]
  | arraySet array index value =>
    cases hArray : memo[array.idx]? <;>
      cases hIndex : memo[index.idx]? <;>
      cases hValue : memo[value.idx]? <;>
      simp [remapENode?, hArray, hIndex, hValue] at hRemap
    subst dstNode
    rfl
  | index array index =>
    cases hArray : memo[array.idx]? with
    | none => simp [remapENode?, hArray] at hRemap
    | some copiedArray =>
      cases hIndex : memo[index.idx]? with
      | none => simp [remapENode?, hArray, hIndex] at hRemap
      | some copiedIndex =>
        simp [remapENode?, hArray, hIndex] at hRemap
        subst dstNode
        simp only [denoteNode]
        rw [denoteExpr_copy_map_eq hMap alg env hSrcArena hDstArena hArray,
          denoteExpr_copy_map_eq hMap alg env hSrcArena hDstArena hIndex]
  | inputRef index | paramRef index | sampleRate | sampleIndex
  | tileSampleIndex | tilePhase | loopIdx index =>
    simp [remapENode?] at hRemap
    subst dstNode
    rfl
  | nestedOut instanceIdx outputIdx =>
    simp [remapENode?] at hRemap
    subst dstNode
    rfl
  | bankSum capacity tables body dynCount? binderId =>
    cases hTables : remapExprIds? memo tables with
    | none => simp [remapENode?, hTables] at hRemap
    | some copiedTables =>
      cases hBody : memo[body.idx]? with
      | none => simp [remapENode?, hTables, hBody] at hRemap
      | some copiedBody =>
        cases dynCount? with
        | none =>
          simp [remapENode?, hTables, hBody] at hRemap
          subst dstNode
          simp only [denoteNode]
          rw [copyArray env tables copiedTables
            ((remapExprIds?_eq_some_iff memo tables copiedTables).mp hTables)
            (by
              intro child hChild
              simp [ENode.children, hChild])]
          have hBodyEq (loopValue : Value α) :=
            denoteExpr_copy_map_eq hMap alg
              (env.bindLoop binderId loopValue) hSrcArena hDstArena hBody
          simp only [hBodyEq]
        | some count =>
          cases hCount : memo[count.idx]? with
          | none => simp [remapENode?, hTables, hBody, hCount] at hRemap
          | some copiedCount =>
            simp [remapENode?, hTables, hBody, hCount] at hRemap
            subst dstNode
            simp only [denoteNode]
            rw [copyArray env tables copiedTables
              ((remapExprIds?_eq_some_iff memo tables copiedTables).mp hTables)
              (by
                intro child hChild
                simp [ENode.children, hChild])]
            rw [denoteExpr_copy_map_eq hMap alg env hSrcArena hDstArena hCount]
            have hBodyEq (loopValue : Value α) :=
              denoteExpr_copy_map_eq hMap alg
                (env.bindLoop binderId loopValue) hSrcArena hDstArena hBody
            simp only [hBodyEq]
  | routedSum capacity outputCount routes tables values dynCount? binderId =>
    cases hTables : remapExprIds? memo tables with
    | none => simp [remapENode?, hTables] at hRemap
    | some copiedTables =>
      cases hValues : remapExprIds? memo values with
      | none => simp [remapENode?, hTables, hValues] at hRemap
      | some copiedValues =>
        have hTableCopies :=
          (remapExprIds?_eq_some_iff memo tables copiedTables).mp hTables
        have hValueCopies :=
          (remapExprIds?_eq_some_iff memo values copiedValues).mp hValues
        have hValueSize : values.size = copiedValues.size := by
          simpa using hValueCopies.length
        have hMappedValues :
            (fun loopValue => values.attach.map fun item =>
              denoteExpr alg (env.bindLoop binderId loopValue)
                src hSrcArena item.1) =
            (fun loopValue => copiedValues.attach.map fun item =>
              denoteExpr alg (env.bindLoop binderId loopValue)
                dst hDstArena item.1) := by
          funext loopValue
          exact copyArray (env.bindLoop binderId loopValue)
            values copiedValues hValueCopies (by
              intro child hChild
              simp [ENode.children, hChild])
        cases dynCount? with
        | none =>
          simp [remapENode?, hTables, hValues] at hRemap
          subst dstNode
          simp only [denoteNode]
          rw [copyArray env tables copiedTables hTableCopies
            (by
              intro child hChild
              simp [ENode.children, hChild])]
          rw [hValueSize, hMappedValues]
        | some count =>
          cases hCount : memo[count.idx]? with
          | none => simp [remapENode?, hTables, hValues, hCount] at hRemap
          | some copiedCount =>
            simp [remapENode?, hTables, hValues, hCount] at hRemap
            subst dstNode
            simp only [denoteNode]
            rw [copyArray env tables copiedTables hTableCopies
              (by
                intro child hChild
                simp [ENode.children, hChild]),
              denoteExpr_copy_map_eq hMap alg env hSrcArena hDstArena hCount]
            rw [hValueSize, hMappedValues]
termination_by srcId.idx
decreasing_by
  all_goals
    apply hSrcArena.childrenDescend hSrcDeref
    simp_all [ENode.children]

/-- A constructor-generic copy preserves successful values and refusals for
    every algebra and environment.  Reduction bodies are instantiated at the
    same bound loop environment on both sides. -/
theorem denoteExpr_copy_eq {src dst : ExprArena} {srcId dstId : ExprId}
    (hcopy : ExprCopyRel src dst srcId dstId)
    (alg : Algebra α) (env : SigEnv α)
    (hSrc : ArenaWellFormed src) (hDst : ArenaWellFormed dst) :
    denoteExpr alg env src hSrc srcId = denoteExpr alg env dst hDst dstId := by
  obtain ⟨memo, hMemo, hMap⟩ := hcopy
  exact denoteExpr_copy_map_eq hMap alg env hSrc hDst hMemo

/-- A checked reachable program copy preserves the complete recursive program
    observation, including refusal behavior and nested instance outputs. -/
theorem ProgramCopyRel.denote_eq {ea : EArena}
    {hPrograms : progPoolWf ea.programs = true} {memo : ExprCopyMemo}
    {root : ProgramIdx} {core : CoreProgram}
    (hcopy : ProgramCopyRel ea hPrograms memo root core)
    (hMap : ExprCopyMapRel ea.exprs dst memo)
    (alg : Algebra α) (hSource : ArenaWellFormed ea.exprs)
    (hDest : ArenaWellFormed dst) (invocation : ProgramInputs α) :
    denoteProgram alg ea hSource hPrograms root invocation =
      denoteCoreProgram alg dst hDest core invocation := by
  cases hcopy with
  | node sourceAt viewCopied childrenCollected registryKeys
      instancePresent instanceCopied =>
    let bridge : SigEnv α → ExprId → Result α := fun env sourceId =>
      match memo[sourceId.idx]? with
      | none => denoteExpr alg env ea.exprs hSource sourceId
      | some destId => denoteExpr alg env dst hDest destId
    have hBridge :
        (fun env id => denoteExpr alg env ea.exprs hSource id) = bridge := by
      funext env sourceId
      simp only [bridge]
      split
      next => rfl
      next destId hMemo =>
        exact denoteExpr_copy_map_eq hMap alg env hSource hDest hMemo
    rw [denoteProgram, sourceAt, denoteCoreProgram, hBridge]
    apply denoteProgramModel_rel alg bridge
      (fun env id => denoteExpr alg env dst hDest id)
      (remapProgramCopyView?_sound memo viewCopied) invocation
    · intro env sourceId destId hMemo
      simp [bridge, hMemo]
    · intro typeKey childInputs hUse
      obtain ⟨name, modelInputs, hModelMem⟩ := hUse
      obtain ⟨sourceInputs, hSourceMem⟩ :=
        sourceProgramModel_inst_mem hModelMem
      obtain ⟨sourceChild, coreChild, hs, hc⟩ :=
        instancePresent hSourceMem
      rw [hs, hc]
      exact ProgramCopyRel.denote_eq
        (instanceCopied hSourceMem hs hc) hMap alg hSource hDest childInputs
termination_by root.idx
decreasing_by
  have hlt := progPool_registry_lt hPrograms sourceAt hs
  subst_vars
  exact hlt

theorem toResolved_witness_of_eq {ea : EArena} {root : ProgramIdx}
    {dst : ExprArena} {core : CoreProgram}
    (hresult : ea.toResolved root = .ok (dst, core)) :
    ∃ result : ResolvedCopy ea root,
      ea.toResolvedWithWitness root = .ok result ∧
      result.exprs = dst ∧ result.program = core := by
  unfold EArena.toResolved at hresult
  cases hw : ea.toResolvedWithWitness root with
  | error error =>
    rw [hw] at hresult
    contradiction
  | ok result =>
    rw [hw] at hresult
    have hpair : (result.exprs, result.program) = (dst, core) := by
      exact Except.ok.inj hresult
    have hDst := congrArg Prod.fst hpair
    have hCore := congrArg Prod.snd hpair
    refine ⟨result, rfl, hDst, ?_⟩
    exact hCore

/-- Public strata-exit capstone: every successful reachable conversion
    preserves the source program's recursive denotation for arbitrary
    supported algebras and invocations. -/
theorem toResolved_preserves_denotation (ea : EArena) (root : ProgramIdx)
    (dst : ExprArena) (core : CoreProgram)
    (hSource : ArenaWellFormed ea.exprs)
    (hPrograms : progPoolWf ea.programs = true)
    (hDest : ArenaWellFormed dst)
    (hresult : ea.toResolved root = .ok (dst, core))
    (alg : Algebra α) (invocation : ProgramInputs α) :
    denoteProgram alg ea hSource hPrograms root invocation =
      denoteCoreProgram alg dst hDest core invocation := by
  obtain ⟨result, hw, hDst, hCore⟩ := toResolved_witness_of_eq hresult
  subst dst
  subst core
  have hDestChecked := semanticWfCheck_sound result.destinationChecked
  have hMap := checkExprCopyMemo_sound result.expressionsChecked
  have hProgram := checkProgramCopy_sound hPrograms result.programChecked
  exact ProgramCopyRel.denote_eq hProgram hMap alg hSource hDestChecked invocation

/-- Destination well-formedness is an immediate structural corollary of a
    successful public conversion. -/
theorem toResolved_destination_wellFormed {ea : EArena} {root : ProgramIdx}
    {dst : ExprArena} {core : CoreProgram}
    (hresult : ea.toResolved root = .ok (dst, core)) :
    ArenaWellFormed dst := by
  obtain ⟨result, hw, hDst, hCore⟩ := toResolved_witness_of_eq hresult
  subst dst
  exact semanticWfCheck_sound result.destinationChecked

/-- The returned registry keys are exactly the converter's deduplicated
    instance references, in authored first-use order. -/
theorem toResolved_registry_firstUse {ea : EArena} {root : ProgramIdx}
    {dst : ExprArena} {core : CoreProgram}
    (hPrograms : progPoolWf ea.programs = true)
    (hresult : ea.toResolved root = .ok (dst, core)) :
    ∃ source, ∃ sourceAt : ea.programs[root.idx]? = some source, ∃ children,
      referencedPrograms ea hPrograms root source sourceAt = .ok children ∧
      children.map (·.1) = core.registry.map (·.1) := by
  obtain ⟨result, hw, hDst, hCore⟩ := toResolved_witness_of_eq hresult
  subst dst
  subst core
  have hProgram := checkProgramCopy_sound hPrograms result.programChecked
  cases hProgram with
  | node sourceAt viewCopied childrenCollected registryKeys
      instancePresent instanceCopied =>
    exact ⟨_, sourceAt, _, childrenCollected, registryKeys⟩

/-- Relational GC-inertness: two source arenas that copy the same reachable
    program and expression spine to one core have identical observations,
    regardless of all source data outside those relations. -/
theorem ProgramCopyRel.unreachable_inert
    {left right : EArena} {leftPrograms : progPoolWf left.programs = true}
    {rightPrograms : progPoolWf right.programs = true}
    {leftMemo rightMemo : ExprCopyMemo} {leftRoot rightRoot : ProgramIdx}
    {core : CoreProgram} {dst : ExprArena}
    (leftCopy : ProgramCopyRel left leftPrograms leftMemo leftRoot core)
    (rightCopy : ProgramCopyRel right rightPrograms rightMemo rightRoot core)
    (leftMap : ExprCopyMapRel left.exprs dst leftMemo)
    (rightMap : ExprCopyMapRel right.exprs dst rightMemo)
    (alg : Algebra α) (leftWf : ArenaWellFormed left.exprs)
    (rightWf : ArenaWellFormed right.exprs) (dstWf : ArenaWellFormed dst)
    (invocation : ProgramInputs α) :
    denoteProgram alg left leftWf leftPrograms leftRoot invocation =
      denoteProgram alg right rightWf rightPrograms rightRoot invocation := by
  calc
    _ = denoteCoreProgram alg dst dstWf core invocation :=
      leftCopy.denote_eq leftMap alg leftWf dstWf invocation
    _ = _ :=
      (rightCopy.denote_eq rightMap alg rightWf dstWf invocation).symm

end Tropical.Semantics
