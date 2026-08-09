import Tropical.Semantics.Expr

/-!
# Denotational preservation for production `lowerSigTree`

`LowersTo` records every child state transition and exact final intern for the
all-constructor production lowering.  `LowersTo.preserves` proves that any such
trace maintains the arena invariants and carrier-parametric denotation.
`lowerSigTree_preserves` combines that result with the checked trace generated
by the production structural reference.
-/

namespace Tropical.Semantics

open Tropical.Ir
open Tropical.EmitArrow

mutual
  /-- One production `Sig` lowers through a sequence of child lowerings and one
      exact `eintern` step to a production arena/id result. -/
  inductive LowersTo : Sig → ExprArena → ExprId → ExprArena → Prop where
    | num (hIntern : (eintern (.num number)).run arena = (resultId, arena')) :
        LowersTo (.num number) arena resultId arena'
    | binary
        (hLeft : LowersTo lhs arena lhsId afterLeft)
        (hRight : LowersTo rhs afterLeft rhsId afterRight)
        (hIntern :
          (eintern (.binary tag lhsId rhsId)).run afterRight = (resultId, arena')) :
        LowersTo (.binary tag lhs rhs) arena resultId arena'
    | unary
        (hArg : LowersTo arg arena argId afterArg)
        (hIntern :
          (eintern (.unary tag argId)).run afterArg = (resultId, arena')) :
        LowersTo (.unary tag arg) arena resultId arena'
    | clamp
        (hValue : LowersTo value arena valueId afterValue)
        (hLo : LowersTo lo afterValue loId afterLo)
        (hHi : LowersTo hi afterLo hiId afterHi)
        (hIntern :
          (eintern (.clamp valueId loId hiId)).run afterHi = (resultId, arena')) :
        LowersTo (.clamp value lo hi) arena resultId arena'
    | select
        (hCond : LowersTo condition arena condId afterCond)
        (hThen : LowersTo then_ afterCond thenId afterThen)
        (hElse : LowersTo else_ afterThen elseId afterElse)
        (hIntern :
          (eintern (.select condId thenId elseId)).run afterElse =
            (resultId, arena')) :
        LowersTo (.select condition then_ else_) arena resultId arena'
    | inputRef
        (hIntern :
          (eintern (.inputRef index)).run arena = (resultId, arena')) :
        LowersTo (.inputRef index) arena resultId arena'
    | paramRef
        (hIntern :
          (eintern (.paramRef index)).run arena = (resultId, arena')) :
        LowersTo (.paramRef index) arena resultId arena'
    | nestedOut
        (hIntern :
          (eintern (.nestedOut instanceIdx outputIdx)).run arena =
            (resultId, arena')) :
        LowersTo (.nestedOut instanceIdx outputIdx) arena resultId arena'
    | sampleRate
        (hIntern : (eintern .sampleRate).run arena = (resultId, arena')) :
        LowersTo .sampleRate arena resultId arena'
    | sampleIndex
        (hIntern : (eintern .sampleIndex).run arena = (resultId, arena')) :
        LowersTo .sampleIndex arena resultId arena'
    | arr
        (hItems : LowersMany items.toList arena #[] itemIds afterItems)
        (hIntern :
          (eintern (.arr itemIds)).run afterItems = (resultId, arena')) :
        LowersTo (.arr items) arena resultId arena'
    | index
        (hArray : LowersTo array arena arrayId afterArray)
        (hIndex : LowersTo index afterArray indexId afterIndex)
        (hIntern :
          (eintern (.index arrayId indexId)).run afterIndex =
            (resultId, arena')) :
        LowersTo (.index array index) arena resultId arena'
    | loopIdx
        (hIntern :
          (eintern (.loopIdx binderId)).run arena = (resultId, arena')) :
        LowersTo (.loopIdx binderId) arena resultId arena'
    | bankSumNone
        (hTables : LowersMany tables.toList arena #[] tableIds afterTables)
        (hBody : LowersTo body afterTables bodyId afterBody)
        (hIntern :
          (eintern (.bankSum capacity tableIds bodyId none binderId)).run
            afterBody = (resultId, arena')) :
        LowersTo (.bankSum capacity tables body none binderId)
          arena resultId arena'
    | bankSumSome
        (hTables : LowersMany tables.toList arena #[] tableIds afterTables)
        (hBody : LowersTo body afterTables bodyId afterBody)
        (hCount : LowersTo count afterBody countId afterCount)
        (hIntern :
          (eintern (.bankSum capacity tableIds bodyId (some countId) binderId)).run
            afterCount = (resultId, arena')) :
        LowersTo (.bankSum capacity tables body (some count) binderId)
          arena resultId arena'
    | routedSumNone
        (hTables : LowersMany tables.toList arena #[] tableIds afterTables)
        (hValues : LowersMany values.toList afterTables #[] valueIds afterValues)
        (hIntern :
          (eintern (.routedSum capacity outputCount routes tableIds valueIds
            none binderId)).run afterValues = (resultId, arena')) :
        LowersTo (.routedSum capacity outputCount routes tables values none binderId)
          arena resultId arena'
    | routedSumSome
        (hTables : LowersMany tables.toList arena #[] tableIds afterTables)
        (hValues : LowersMany values.toList afterTables #[] valueIds afterValues)
        (hCount : LowersTo count afterValues countId afterCount)
        (hIntern :
          (eintern (.routedSum capacity outputCount routes tableIds valueIds
            (some countId) binderId)).run afterCount = (resultId, arena')) :
        LowersTo (.routedSum capacity outputCount routes tables values
          (some count) binderId) arena resultId arena'

  /-- Ordered trace for `Array.mapM lowerSigTree`, expressed over the list view
      used by Lean's lawful-monad lemma.  `acc` makes the push order explicit. -/
  inductive LowersMany :
      List Sig → ExprArena → Array ExprId → Array ExprId → ExprArena → Prop where
    | nil : LowersMany [] arena acc acc arena
    | cons
        (hHead : LowersTo head arena headId afterHead)
        (hTail :
          LowersMany tail afterHead (acc.push headId) result finalArena) :
        LowersMany (head :: tail) arena acc result finalArena
end

private theorem lower_attach_mapM (items : Array Sig) :
    items.attach.mapM
      (fun item => match item with | ⟨item, _⟩ => lowerSigTree item)
      = items.mapM lowerSigTree := by
  change items.attach.mapM (fun item => lowerSigTree item.1)
    = items.mapM lowerSigTree
  simpa only [Array.unattach_attach] using Array.mapM_subtype
    (xs := items.attach)
    (f := fun item : { item // item ∈ items } => lowerSigTree item.1)
    (g := lowerSigTree) (fun _ _ => rfl)

private theorem lowerMany_produces
    (items : List Sig)
    (ih : ∀ item ∈ items, ∀ arena,
      let (id, arena') := (lowerSigTree item).run arena
      LowersTo item arena id arena')
    (arena : ExprArena) (acc : Array ExprId) :
    LowersMany items arena acc
      (List.foldlM
        (fun ids item => ids.push <$> lowerSigTree item) acc items arena).1
      (List.foldlM
        (fun ids item => ids.push <$> lowerSigTree item) acc items arena).2 := by
  induction items generalizing arena acc with
  | nil =>
    simp only [List.foldlM_nil]
    exact .nil
  | cons head tail tailIH =>
    rw [List.foldlM_cons]
    simp only [bind, StateT.bind, Functor.map, StateT.map, pure]
    generalize hRun : lowerSigTree head arena = result
    obtain ⟨headId, afterHead⟩ := result
    have hHead := ih head (by simp) arena
    change LowersTo head arena (lowerSigTree head arena).1
      (lowerSigTree head arena).2 at hHead
    rw [hRun] at hHead
    have hTail : LowersMany tail afterHead (acc.push headId)
        (List.foldlM
          (fun ids item => ids.push <$> lowerSigTree item)
          (acc.push headId) tail afterHead).1
        (List.foldlM
          (fun ids item => ids.push <$> lowerSigTree item)
          (acc.push headId) tail afterHead).2 := tailIH
      (fun item hmem nextArena =>
        ih item (by simp [hmem]) nextArena)
      afterHead (acc.push headId)
    exact LowersMany.cons hHead hTail

/-- The production structural reference realizes `LowersTo` for all fourteen
    constructors, from every initial arena. -/
theorem lowerSigTree_lowersTo (sig : Sig) (arena : ExprArena) :
    LowersTo sig arena (lowerSigTree sig arena).1
      (lowerSigTree sig arena).2 := by
  induction sig using lowerSigTree.induct generalizing arena with
  | case1 number =>
    rw [lowerSigTree.eq_1]
    exact .num rfl
  | case2 tag lhs rhs ihLeft ihRight =>
    rw [lowerSigTree.eq_2]
    simp only [StateT.bind, bind]
    generalize hLeftRun : lowerSigTree lhs arena = leftResult
    obtain ⟨lhsId, afterLeft⟩ := leftResult
    simp only
    generalize hRightRun : lowerSigTree rhs afterLeft = rightResult
    obtain ⟨rhsId, afterRight⟩ := rightResult
    simp only
    exact .binary
      (by simpa [hLeftRun] using ihLeft arena)
      (by simpa [hRightRun] using ihRight afterLeft)
      rfl
  | case3 tag arg ihArg =>
    rw [lowerSigTree.eq_3]
    simp only [StateT.bind, bind]
    generalize hArgRun : lowerSigTree arg arena = argResult
    obtain ⟨argId, afterArg⟩ := argResult
    simp only
    exact .unary (by simpa [hArgRun] using ihArg arena) rfl
  | case4 value lo hi ihValue ihLo ihHi =>
    rw [lowerSigTree.eq_4]
    simp only [StateT.bind, bind]
    generalize hValueRun : lowerSigTree value arena = valueResult
    obtain ⟨valueId, afterValue⟩ := valueResult
    simp only
    generalize hLoRun : lowerSigTree lo afterValue = loResult
    obtain ⟨loId, afterLo⟩ := loResult
    simp only
    generalize hHiRun : lowerSigTree hi afterLo = hiResult
    obtain ⟨hiId, afterHi⟩ := hiResult
    simp only
    exact .clamp
      (by simpa [hValueRun] using ihValue arena)
      (by simpa [hLoRun] using ihLo afterValue)
      (by simpa [hHiRun] using ihHi afterLo)
      rfl
  | case5 cond then_ else_ ihCond ihThen ihElse =>
    rw [lowerSigTree.eq_5]
    simp only [StateT.bind, bind]
    generalize hCondRun : lowerSigTree cond arena = condResult
    obtain ⟨condId, afterCond⟩ := condResult
    simp only
    generalize hThenRun : lowerSigTree then_ afterCond = thenResult
    obtain ⟨thenId, afterThen⟩ := thenResult
    simp only
    generalize hElseRun : lowerSigTree else_ afterThen = elseResult
    obtain ⟨elseId, afterElse⟩ := elseResult
    simp only
    exact .select
      (by simpa [hCondRun] using ihCond arena)
      (by simpa [hThenRun] using ihThen afterCond)
      (by simpa [hElseRun] using ihElse afterThen)
      rfl
  | case6 index =>
    rw [lowerSigTree.eq_6]
    exact .inputRef rfl
  | case7 index =>
    rw [lowerSigTree.eq_7]
    exact .paramRef rfl
  | case8 instanceIdx outputIdx =>
    rw [lowerSigTree.eq_8]
    exact .nestedOut rfl
  | case9 =>
    rw [lowerSigTree.eq_9]
    exact .sampleRate rfl
  | case10 =>
    rw [lowerSigTree.eq_10]
    exact .sampleIndex rfl
  | case11 items ihItems =>
    rw [lowerSigTree.eq_11, lower_attach_mapM]
    rw [Array.mapM_eq_foldlM, ← Array.foldlM_toList]
    simp only [StateT.bind, bind]
    generalize hItemsRun :
      List.foldlM
        (fun ids item => ids.push <$> lowerSigTree item) #[] items.toList arena
        = itemsResult
    obtain ⟨itemIds, afterItems⟩ := itemsResult
    simp only
    exact .arr
      (by
        have hMany := lowerMany_produces items.toList
          (fun item hmem nextArena =>
            ihItems item (by simpa using hmem) nextArena)
          arena #[]
        change LowersMany items.toList arena #[]
          (List.foldlM
            (fun ids item => ids.push <$> lowerSigTree item) #[] items.toList arena).1
          (List.foldlM
            (fun ids item => ids.push <$> lowerSigTree item) #[] items.toList arena).2
          at hMany
        rw [hItemsRun] at hMany
        exact hMany)
      rfl
  | case12 array index ihArray ihIndex =>
    rw [lowerSigTree.eq_12]
    simp only [StateT.bind, bind]
    generalize hArrayRun : lowerSigTree array arena = arrayResult
    obtain ⟨arrayId, afterArray⟩ := arrayResult
    simp only
    generalize hIndexRun : lowerSigTree index afterArray = indexResult
    obtain ⟨indexId, afterIndex⟩ := indexResult
    simp only
    exact .index
      (by simpa [hArrayRun] using ihArray arena)
      (by simpa [hIndexRun] using ihIndex afterArray)
      rfl
  | case13 binderId =>
    rw [lowerSigTree.eq_13]
    exact .loopIdx rfl
  | case14 capacity tables body dynCount? binderId ihTables ihBody ihCount =>
    rw [lowerSigTree.eq_14, lower_attach_mapM]
    rw [Array.mapM_eq_foldlM, ← Array.foldlM_toList]
    simp only [StateT.bind, bind, pure]
    generalize hTablesRun :
      List.foldlM
        (fun ids table => ids.push <$> lowerSigTree table) #[] tables.toList arena
        = tablesResult
    obtain ⟨tableIds, afterTables⟩ := tablesResult
    simp only
    generalize hBodyRun : lowerSigTree body afterTables = bodyResult
    obtain ⟨bodyId, afterBody⟩ := bodyResult
    simp only
    cases dynCount? with
    | none =>
      simp only
      exact .bankSumNone
        (by
          have hMany := lowerMany_produces tables.toList
            (fun table hmem nextArena =>
              ihTables table (by simpa using hmem) nextArena)
            arena #[]
          change LowersMany tables.toList arena #[]
            (List.foldlM
              (fun ids table => ids.push <$> lowerSigTree table)
              #[] tables.toList arena).1
            (List.foldlM
              (fun ids table => ids.push <$> lowerSigTree table)
              #[] tables.toList arena).2
            at hMany
          rw [hTablesRun] at hMany
          exact hMany)
        (by simpa [hBodyRun] using ihBody afterTables)
        rfl
    | some count =>
      simp only [StateT.bind, StateT.pure, bind, pure]
      generalize hCountRun : lowerSigTree count afterBody = countResult
      obtain ⟨countId, afterCount⟩ := countResult
      simp only
      exact .bankSumSome
        (by
          have hMany := lowerMany_produces tables.toList
            (fun table hmem nextArena =>
              ihTables table (by simpa using hmem) nextArena)
            arena #[]
          change LowersMany tables.toList arena #[]
            (List.foldlM
              (fun ids table => ids.push <$> lowerSigTree table)
              #[] tables.toList arena).1
            (List.foldlM
              (fun ids table => ids.push <$> lowerSigTree table)
              #[] tables.toList arena).2
            at hMany
          rw [hTablesRun] at hMany
          exact hMany)
        (by simpa [hBodyRun] using ihBody afterTables)
        (by simpa [hCountRun] using ihCount tableIds afterBody)
        rfl
  | case15 capacity outputCount routes tables values dynCount? binderId
      ihTables ihValues ihCount =>
    rw [lowerSigTree.eq_15, lower_attach_mapM, lower_attach_mapM]
    rw [Array.mapM_eq_foldlM, ← Array.foldlM_toList]
    simp only [StateT.bind, bind, pure]
    generalize hTablesRun :
      List.foldlM
        (fun ids table => ids.push <$> lowerSigTree table) #[] tables.toList arena
        = tablesResult
    obtain ⟨tableIds, afterTables⟩ := tablesResult
    simp only
    rw [Array.mapM_eq_foldlM, ← Array.foldlM_toList]
    generalize hValuesRun :
      List.foldlM
        (fun ids value => ids.push <$> lowerSigTree value) #[] values.toList afterTables
        = valuesResult
    obtain ⟨valueIds, afterValues⟩ := valuesResult
    have hTablesTrace : LowersMany tables.toList arena #[] tableIds afterTables := by
      have hMany := lowerMany_produces tables.toList
        (fun table hmem nextArena =>
          ihTables table (by simpa using hmem) nextArena)
        arena #[]
      change LowersMany tables.toList arena #[]
        (List.foldlM
          (fun ids table => ids.push <$> lowerSigTree table)
          #[] tables.toList arena).1
        (List.foldlM
          (fun ids table => ids.push <$> lowerSigTree table)
          #[] tables.toList arena).2 at hMany
      simpa [hTablesRun] using hMany
    have hValuesTrace :
        LowersMany values.toList afterTables #[] valueIds afterValues := by
      have hMany := lowerMany_produces values.toList
        (fun value hmem nextArena =>
          ihValues value (by simpa using hmem) nextArena)
        afterTables #[]
      change LowersMany values.toList afterTables #[]
        (List.foldlM
          (fun ids value => ids.push <$> lowerSigTree value)
          #[] values.toList afterTables).1
        (List.foldlM
          (fun ids value => ids.push <$> lowerSigTree value)
          #[] values.toList afterTables).2 at hMany
      simpa [hValuesRun] using hMany
    cases dynCount? with
    | none =>
      simp only [StateT.bind, StateT.pure, bind, pure]
      exact .routedSumNone hTablesTrace hValuesTrace rfl
    | some count =>
      simp only [StateT.bind, StateT.pure, bind, pure]
      generalize hCountRun : lowerSigTree count afterValues = countResult
      obtain ⟨countId, afterCount⟩ := countResult
      exact .routedSumSome hTablesTrace hValuesTrace
        (by simpa [hCountRun] using ihCount tableIds valueIds afterValues) rfl

private def LowersToSpec {sig before rootId after}
    (_h : LowersTo sig before rootId after) : Prop :=
  ∀ {α : Type} (alg : Algebra α) (env : SigEnv α)
      (_hBefore : ArenaWellFormed before),
    ∃ hAfter : ArenaWellFormed after,
      Extends before after ∧
        DenotesAt alg env after hAfter sig rootId

private def LowersManySpec {items before acc result after}
    (_h : LowersMany items before acc result after) : Prop :=
  ∀ {α : Type} (alg : Algebra α) (env : SigEnv α)
      (hBefore : ArenaWellFormed before)
      (pre : List Sig),
    DenotesMany alg env before hBefore pre acc.toList →
      ∃ hAfter : ArenaWellFormed after,
        Extends before after ∧
          DenotesMany alg env after hAfter
            (pre ++ items) result.toList

private theorem intern_denotes {α : Type} {arena after : ExprArena}
    {node : ENode} {rootId : ExprId} {sig : Sig}
    (alg : Algebra α) (env : SigEnv α)
    (hArena : ArenaWellFormed arena)
    (hChildren : ChildrenInPrefix arena node)
    (hIntern : eintern node arena = (rootId, after))
    (hValue : ∀ (hAfter : ArenaWellFormed after),
      Extends arena after →
        denoteNode alg env node (fun childEnv child _ =>
          denoteExpr alg childEnv after hAfter child) =
          denoteSig alg env sig) :
    ∃ hAfter : ArenaWellFormed after,
      Extends arena after ∧
        DenotesAt alg env after hAfter sig rootId := by
  have hSpec := eintern_preserves hArena hChildren
  rw [hIntern] at hSpec
  obtain ⟨hAfter, hExtends, hDeref⟩ := hSpec
  refine ⟨hAfter, hExtends, node, hDeref, ?_⟩
  rw [denoteExpr_of_deref alg env after hAfter hDeref]
  exact hValue hAfter hExtends

/-- Every relational lowering trace preserves the carrier-parametric
    denotation while maintaining the production arena invariants. -/
theorem LowersTo.preserves {sig before rootId after}
    (h : LowersTo sig before rootId after) : LowersToSpec h := by
  apply LowersTo.rec
    (motive_1 := fun _ _ _ _ h => LowersToSpec h)
    (motive_2 := fun _ _ _ _ _ h => LowersManySpec h)
  case num =>
    intro number arena resultId arena' hIntern α alg env hArena
    apply intern_denotes (sig := .num number) alg env hArena
      (by simp [ChildrenInPrefix, ENode.children]) hIntern
    intro hAfter hExtends
    simp [denoteNode, denoteSig]
  case binary =>
    intro lhs arena lhsId afterLeft rhs rhsId afterRight tag resultId arena'
      hLeft hRight hIntern ihLeft ihRight α alg env hArena
    obtain ⟨hLeftArena, hExtLeft, hDenLeft⟩ :=
      ihLeft alg env hArena
    obtain ⟨hRightArena, hExtRight, hDenRight⟩ :=
      ihRight alg env hLeftArena
    have hDenLeft' := hDenLeft.extends hRightArena hExtRight
    obtain ⟨hFinal, hExtFinal, hDenFinal⟩ :=
      intern_denotes (sig := .binary tag lhs rhs) alg env hRightArena
        (by
          intro child hMem
          simp [ENode.children] at hMem
          rcases hMem with rfl | rfl
          · obtain ⟨_, hDeref, _⟩ := hDenLeft'
            exact deref_index_lt hDeref
          · obtain ⟨_, hDeref, _⟩ := hDenRight
            exact deref_index_lt hDeref)
        hIntern
        (by
          intro hAfter hExtends
          obtain ⟨_, _, hLeftValue⟩ :=
            hDenLeft'.extends hAfter hExtends
          obtain ⟨_, _, hRightValue⟩ :=
            hDenRight.extends hAfter hExtends
          simp only [denoteNode]
          rw [denoteSig, hLeftValue, hRightValue])
    exact ⟨hFinal, (hExtLeft.trans hExtRight).trans hExtFinal, hDenFinal⟩
  case unary =>
    intro arg arena argId afterArg tag resultId arena'
      hArg hIntern ihArg α alg env hArena
    obtain ⟨hArgArena, hExtArg, hDenArg⟩ := ihArg alg env hArena
    obtain ⟨hFinal, hExtFinal, hDenFinal⟩ :=
      intern_denotes (sig := .unary tag arg) alg env hArgArena
        (by
          intro child hMem
          simp [ENode.children] at hMem
          subst child
          obtain ⟨_, hDeref, _⟩ := hDenArg
          exact deref_index_lt hDeref)
        hIntern
        (by
          intro hAfter hExtends
          obtain ⟨_, _, hArgValue⟩ :=
            hDenArg.extends hAfter hExtends
          simp only [denoteNode]
          rw [denoteSig, hArgValue]
          rfl)
    exact ⟨hFinal, hExtArg.trans hExtFinal, hDenFinal⟩
  case clamp =>
    intro value arena valueId afterValue lo loId afterLo hi hiId afterHi
      resultId arena' hValue hLo hHi hIntern ihValue ihLo ihHi
      α alg env hArena
    obtain ⟨hValueArena, hExtValue, hDenValue⟩ :=
      ihValue alg env hArena
    obtain ⟨hLoArena, hExtLo, hDenLo⟩ :=
      ihLo alg env hValueArena
    obtain ⟨hHiArena, hExtHi, hDenHi⟩ :=
      ihHi alg env hLoArena
    have hDenValue' :=
      (hDenValue.extends hLoArena hExtLo).extends hHiArena hExtHi
    have hDenLo' := hDenLo.extends hHiArena hExtHi
    obtain ⟨hFinal, hExtFinal, hDenFinal⟩ :=
      intern_denotes (sig := .clamp value lo hi) alg env hHiArena
        (by
          intro child hMem
          simp [ENode.children] at hMem
          rcases hMem with rfl | rfl | rfl
          · obtain ⟨_, hDeref, _⟩ := hDenValue'
            exact deref_index_lt hDeref
          · obtain ⟨_, hDeref, _⟩ := hDenLo'
            exact deref_index_lt hDeref
          · obtain ⟨_, hDeref, _⟩ := hDenHi
            exact deref_index_lt hDeref)
        hIntern
        (by
          intro hAfter hExtends
          obtain ⟨_, _, hValueEq⟩ :=
            hDenValue'.extends hAfter hExtends
          obtain ⟨_, _, hLoEq⟩ := hDenLo'.extends hAfter hExtends
          obtain ⟨_, _, hHiEq⟩ := hDenHi.extends hAfter hExtends
          simp only [denoteNode]
          rw [denoteSig, hValueEq, hLoEq, hHiEq])
    exact ⟨hFinal,
      ((hExtValue.trans hExtLo).trans hExtHi).trans hExtFinal,
      hDenFinal⟩
  case select =>
    intro condition arena condId afterCond then_ thenId afterThen else_
      elseId afterElse resultId arena' hCond hThen hElse hIntern
      ihCond ihThen ihElse α alg env hArena
    obtain ⟨hCondArena, hExtCond, hDenCond⟩ :=
      ihCond alg env hArena
    obtain ⟨hThenArena, hExtThen, hDenThen⟩ :=
      ihThen alg env hCondArena
    obtain ⟨hElseArena, hExtElse, hDenElse⟩ :=
      ihElse alg env hThenArena
    have hDenCond' :=
      (hDenCond.extends hThenArena hExtThen).extends hElseArena hExtElse
    have hDenThen' := hDenThen.extends hElseArena hExtElse
    obtain ⟨hFinal, hExtFinal, hDenFinal⟩ :=
      intern_denotes (sig := .select condition then_ else_) alg env hElseArena
        (by
          intro child hMem
          simp [ENode.children] at hMem
          rcases hMem with rfl | rfl | rfl
          · obtain ⟨_, hDeref, _⟩ := hDenCond'
            exact deref_index_lt hDeref
          · obtain ⟨_, hDeref, _⟩ := hDenThen'
            exact deref_index_lt hDeref
          · obtain ⟨_, hDeref, _⟩ := hDenElse
            exact deref_index_lt hDeref)
        hIntern
        (by
          intro hAfter hExtends
          obtain ⟨_, _, hCondEq⟩ :=
            hDenCond'.extends hAfter hExtends
          obtain ⟨_, _, hThenEq⟩ :=
            hDenThen'.extends hAfter hExtends
          obtain ⟨_, _, hElseEq⟩ :=
            hDenElse.extends hAfter hExtends
          simp only [denoteNode]
          rw [denoteSig, hCondEq, hThenEq, hElseEq])
    exact ⟨hFinal,
      ((hExtCond.trans hExtThen).trans hExtElse).trans hExtFinal,
      hDenFinal⟩
  case inputRef =>
    intro index arena resultId arena' hIntern α alg env hArena
    apply intern_denotes (sig := .inputRef index) alg env hArena
      (by simp [ChildrenInPrefix, ENode.children]) hIntern
    intro hAfter hExtends
    simp [denoteNode, denoteSig]
  case paramRef =>
    intro index arena resultId arena' hIntern α alg env hArena
    apply intern_denotes (sig := .paramRef index) alg env hArena
      (by simp [ChildrenInPrefix, ENode.children]) hIntern
    intro hAfter hExtends
    simp [denoteNode, denoteSig]
  case nestedOut =>
    intro instanceIdx outputIdx arena resultId arena' hIntern
      α alg env hArena
    apply intern_denotes (sig := .nestedOut instanceIdx outputIdx)
      alg env hArena
      (by simp [ChildrenInPrefix, ENode.children]) hIntern
    intro hAfter hExtends
    simp [denoteNode, denoteSig]
  case sampleRate =>
    intro arena resultId arena' hIntern α alg env hArena
    apply intern_denotes (sig := .sampleRate) alg env hArena
      (by simp [ChildrenInPrefix, ENode.children]) hIntern
    intro hAfter hExtends
    simp [denoteNode, denoteSig]
  case sampleIndex =>
    intro arena resultId arena' hIntern α alg env hArena
    apply intern_denotes (sig := .sampleIndex) alg env hArena
      (by simp [ChildrenInPrefix, ENode.children]) hIntern
    intro hAfter hExtends
    simp [denoteNode, denoteSig]
  case arr =>
    intro arena itemIds afterItems resultId arena' items hItems hIntern
      ihItems α alg env hArena
    obtain ⟨hItemsArena, hExtItems, hDenItems⟩ :=
      ihItems alg env hArena [] (by exact .nil)
    simp only [List.nil_append] at hDenItems
    obtain ⟨hFinal, hExtFinal, hDenFinal⟩ :=
      intern_denotes (sig := .arr items) alg env hItemsArena
        (by
          intro child hMem
          obtain ⟨_, hDeref⟩ := hDenItems.deref_of_mem
            (by simpa [ENode.children] using hMem)
          exact deref_index_lt hDeref)
        hIntern
        (by
          intro hAfter hExtends
          have hItemsFinal := hDenItems.extends hAfter hExtends
          have hMaps := hItemsFinal.array_map_eq
          rw [denoteNode, denoteSig, attach_map_value, hMaps]
          rfl)
    exact ⟨hFinal, hExtItems.trans hExtFinal, hDenFinal⟩
  case index =>
    intro array arena arrayId afterArray index indexId afterIndex
      resultId arena' hArray hIndex hIntern ihArray ihIndex
      α alg env hArena
    obtain ⟨hArrayArena, hExtArray, hDenArray⟩ :=
      ihArray alg env hArena
    obtain ⟨hIndexArena, hExtIndex, hDenIndex⟩ :=
      ihIndex alg env hArrayArena
    have hDenArray' := hDenArray.extends hIndexArena hExtIndex
    obtain ⟨hFinal, hExtFinal, hDenFinal⟩ :=
      intern_denotes (sig := .index array index) alg env hIndexArena
        (by
          intro child hMem
          simp [ENode.children] at hMem
          rcases hMem with rfl | rfl
          · obtain ⟨_, hDeref, _⟩ := hDenArray'
            exact deref_index_lt hDeref
          · obtain ⟨_, hDeref, _⟩ := hDenIndex
            exact deref_index_lt hDeref)
        hIntern
        (by
          intro hAfter hExtends
          obtain ⟨_, _, hArrayEq⟩ :=
            hDenArray'.extends hAfter hExtends
          obtain ⟨_, _, hIndexEq⟩ :=
            hDenIndex.extends hAfter hExtends
          simp only [denoteNode]
          rw [denoteSig, hArrayEq, hIndexEq]
          rfl)
    exact ⟨hFinal, (hExtArray.trans hExtIndex).trans hExtFinal, hDenFinal⟩
  case loopIdx =>
    intro binderId arena resultId arena' hIntern α alg env hArena
    apply intern_denotes (sig := .loopIdx binderId) alg env hArena
      (by simp [ChildrenInPrefix, ENode.children]) hIntern
    intro hAfter hExtends
    rw [denoteNode, denoteSig]
    rfl
  case bankSumNone =>
    intro arena tableIds afterTables body bodyId afterBody capacity binderId
      resultId arena' tables hTables hBody hIntern ihTables ihBody
      α alg env hArena
    obtain ⟨hTablesArena, hExtTables, hDenTables⟩ :=
      ihTables alg env hArena [] (by exact .nil)
    simp only [List.nil_append] at hDenTables
    obtain ⟨hBodyArena, hExtBody, hDenBody⟩ :=
      ihBody alg env hTablesArena
    have hDenTables' := hDenTables.extends hBodyArena hExtBody
    obtain ⟨hFinal, hExtFinal, hDenFinal⟩ :=
      intern_denotes
        (sig := .bankSum capacity tables body none binderId)
        alg env hBodyArena
        (by
          intro child hMem
          simp [ENode.children] at hMem
          rcases hMem with hTable | rfl
          · obtain ⟨_, hDeref⟩ :=
              hDenTables'.deref_of_mem (by simpa using hTable)
            exact deref_index_lt hDeref
          · obtain ⟨_, hDeref, _⟩ := hDenBody
            exact deref_index_lt hDeref)
        hIntern
        (by
          intro hAfter hExtends
          have hTablesFinal := hDenTables'.extends hAfter hExtends
          have hTableMaps := hTablesFinal.array_map_eq
          have hBodyValue (loopValue : Value α) :
              denoteExpr alg (env.bindLoop binderId loopValue)
                  afterBody hBodyArena bodyId =
                denoteSig alg (env.bindLoop binderId loopValue) body := by
            obtain ⟨hBoundArena, _, hBoundBody⟩ :=
              ihBody alg (env.bindLoop binderId loopValue) hTablesArena
            obtain ⟨_, _, hBoundValue⟩ := hBoundBody
            exact hBoundValue
          have hBodyFinal (loopValue : Value α) :
              denoteExpr alg (env.bindLoop binderId loopValue)
                  arena' hAfter bodyId =
                denoteSig alg (env.bindLoop binderId loopValue) body := by
            obtain ⟨bodyNode, hBodyDeref, _⟩ := hDenBody
            exact
              (denoteExpr_extends hBodyArena hAfter hExtends alg
                (env.bindLoop binderId loopValue) hBodyDeref).symm.trans
                (hBodyValue loopValue)
          simp only [denoteNode, attach_map_value]
          rw [denoteSig, hTableMaps]
          simp only [hBodyFinal]
          rfl)
    exact ⟨hFinal,
      (hExtTables.trans hExtBody).trans hExtFinal, hDenFinal⟩
  case bankSumSome =>
    intro arena tableIds afterTables body bodyId afterBody count countId
      afterCount capacity binderId resultId arena' tables hTables hBody
      hCount hIntern ihTables ihBody ihCount α alg env hArena
    obtain ⟨hTablesArena, hExtTables, hDenTables⟩ :=
      ihTables alg env hArena [] (by exact .nil)
    simp only [List.nil_append] at hDenTables
    obtain ⟨hBodyArena, hExtBody, hDenBody⟩ :=
      ihBody alg env hTablesArena
    obtain ⟨hCountArena, hExtCount, hDenCount⟩ :=
      ihCount alg env hBodyArena
    have hDenTables' :=
      (hDenTables.extends hBodyArena hExtBody).extends hCountArena hExtCount
    have hDenBody' := hDenBody.extends hCountArena hExtCount
    obtain ⟨hFinal, hExtFinal, hDenFinal⟩ :=
      intern_denotes
        (sig := .bankSum capacity tables body (some count) binderId)
        alg env hCountArena
        (by
          intro child hMem
          simp [ENode.children] at hMem
          rcases hMem with (hTable | rfl) | rfl
          · obtain ⟨_, hDeref⟩ :=
              hDenTables'.deref_of_mem (by simpa using hTable)
            exact deref_index_lt hDeref
          · obtain ⟨_, hDeref, _⟩ := hDenBody'
            exact deref_index_lt hDeref
          · obtain ⟨_, hDeref, _⟩ := hDenCount
            exact deref_index_lt hDeref)
        hIntern
        (by
          intro hAfter hExtends
          have hTablesFinal := hDenTables'.extends hAfter hExtends
          have hTableMaps := hTablesFinal.array_map_eq
          obtain ⟨_, _, hCountValue⟩ :=
            hDenCount.extends hAfter hExtends
          have hBodyFinal (loopValue : Value α) :
              denoteExpr alg (env.bindLoop binderId loopValue)
                  arena' hAfter bodyId =
                denoteSig alg (env.bindLoop binderId loopValue) body := by
            obtain ⟨hBoundArena, hExtBound, hBoundBody⟩ :=
              ihBody alg (env.bindLoop binderId loopValue) hTablesArena
            have hBoundBody' := hBoundBody.extends hCountArena hExtCount
            obtain ⟨bodyNode, hBodyDeref, hBodyValue⟩ := hBoundBody'
            exact
              (denoteExpr_extends hCountArena hAfter hExtends alg
                (env.bindLoop binderId loopValue) hBodyDeref).symm.trans
                hBodyValue
          simp only [denoteNode, attach_map_value]
          rw [denoteSig, hTableMaps, hCountValue]
          simp only [hBodyFinal]
          rfl)
    exact ⟨hFinal,
      ((hExtTables.trans hExtBody).trans hExtCount).trans hExtFinal,
      hDenFinal⟩
  case routedSumNone =>
    intro arena tableIds afterTables valueIds afterValues capacity outputCount
      routes binderId resultId arena' tables values hTables hValues hIntern
      ihTables ihValues α alg env hArena
    obtain ⟨hTablesArena, hExtTables, hDenTables⟩ :=
      ihTables alg env hArena [] (by exact .nil)
    simp only [List.nil_append] at hDenTables
    obtain ⟨hValuesArena, hExtValues, hDenValues⟩ :=
      ihValues alg env hTablesArena [] (by exact .nil)
    simp only [List.nil_append] at hDenValues
    have hFanout : valueIds.size = values.size := by
      simpa using hDenValues.length_eq
    have hDenTables' := hDenTables.extends hValuesArena hExtValues
    obtain ⟨hFinal, hExtFinal, hDenFinal⟩ :=
      intern_denotes
        (sig := .routedSum capacity outputCount routes tables values none binderId)
        alg env hValuesArena
        (by
          intro child hMem
          simp [ENode.children] at hMem
          rcases hMem with hTable | hValue
          · obtain ⟨_, hDeref⟩ :=
              hDenTables'.deref_of_mem (by simpa using hTable)
            exact deref_index_lt hDeref
          · obtain ⟨_, hDeref⟩ :=
              hDenValues.deref_of_mem (by simpa using hValue)
            exact deref_index_lt hDeref)
        hIntern
        (by
          intro hAfter hExtends
          have hTablesFinal := hDenTables'.extends hAfter hExtends
          have hTableMaps := hTablesFinal.array_map_eq
          have hValueMaps (loopValue : Value α) :
              valueIds.map
                  (denoteExpr alg (env.bindLoop binderId loopValue)
                    arena' hAfter) =
                values.map
                  (denoteSig alg (env.bindLoop binderId loopValue)) := by
            obtain ⟨hBoundArena, _, hBoundValues⟩ :=
              ihValues alg (env.bindLoop binderId loopValue)
                hTablesArena [] (by exact .nil)
            simp only [List.nil_append] at hBoundValues
            exact (hBoundValues.extends hAfter hExtends).array_map_eq
          simp only [denoteNode, attach_map_value]
          rw [denoteSig, hTableMaps]
          simp only [hValueMaps]
          rw [hFanout])
    exact ⟨hFinal,
      (hExtTables.trans hExtValues).trans hExtFinal, hDenFinal⟩
  case routedSumSome =>
    intro arena tableIds afterTables valueIds afterValues count countId
      afterCount capacity outputCount routes binderId resultId arena' tables
      values hTables hValues hCount hIntern ihTables ihValues ihCount
      α alg env hArena
    obtain ⟨hTablesArena, hExtTables, hDenTables⟩ :=
      ihTables alg env hArena [] (by exact .nil)
    simp only [List.nil_append] at hDenTables
    obtain ⟨hValuesArena, hExtValues, hDenValues⟩ :=
      ihValues alg env hTablesArena [] (by exact .nil)
    simp only [List.nil_append] at hDenValues
    have hFanout : valueIds.size = values.size := by
      simpa using hDenValues.length_eq
    obtain ⟨hCountArena, hExtCount, hDenCount⟩ :=
      ihCount alg env hValuesArena
    have hDenTables' :=
      (hDenTables.extends hValuesArena hExtValues).extends hCountArena hExtCount
    have hDenValues' := hDenValues.extends hCountArena hExtCount
    obtain ⟨hFinal, hExtFinal, hDenFinal⟩ :=
      intern_denotes
        (sig := .routedSum capacity outputCount routes tables values
          (some count) binderId)
        alg env hCountArena
        (by
          intro child hMem
          have hCases :
              child ∈ tableIds ∨ child ∈ valueIds ∨ child = countId := by
            simpa [ENode.children] using hMem
          rcases hCases with hTable | hValue | rfl
          · obtain ⟨_, hDeref⟩ :=
              hDenTables'.deref_of_mem (by simpa using hTable)
            exact deref_index_lt hDeref
          · obtain ⟨_, hDeref⟩ :=
              hDenValues'.deref_of_mem (by simpa using hValue)
            exact deref_index_lt hDeref
          · obtain ⟨_, hDeref, _⟩ := hDenCount
            exact deref_index_lt hDeref)
        hIntern
        (by
          intro hAfter hExtends
          have hTablesFinal := hDenTables'.extends hAfter hExtends
          have hTableMaps := hTablesFinal.array_map_eq
          obtain ⟨_, _, hCountValue⟩ := hDenCount.extends hAfter hExtends
          have hValueMaps (loopValue : Value α) :
              valueIds.map
                  (denoteExpr alg (env.bindLoop binderId loopValue)
                    arena' hAfter) =
                values.map
                  (denoteSig alg (env.bindLoop binderId loopValue)) := by
            obtain ⟨hBoundArena, hExtBound, hBoundValues⟩ :=
              ihValues alg (env.bindLoop binderId loopValue)
                hTablesArena [] (by exact .nil)
            simp only [List.nil_append] at hBoundValues
            have hBoundValues' :=
              hBoundValues.extends hCountArena hExtCount
            exact (hBoundValues'.extends hAfter hExtends).array_map_eq
          simp only [denoteNode, attach_map_value]
          rw [denoteSig, hTableMaps, hCountValue]
          simp only [hValueMaps]
          rw [hFanout])
    exact ⟨hFinal,
      ((hExtTables.trans hExtValues).trans hExtCount).trans hExtFinal,
      hDenFinal⟩
  case nil =>
    intro arena acc α alg env hArena pre hPre
    refine ⟨hArena, Extends.refl arena, ?_⟩
    simpa using hPre
  case cons =>
    intro head arena headId afterHead tail result finalArena acc
      hHead hTail ihHead ihTail α alg env hArena pre hPre
    obtain ⟨hHeadArena, hExtHead, hDenHead⟩ :=
      ihHead alg env hArena
    have hPre' := (hPre.extends hHeadArena hExtHead).snoc hDenHead
    obtain ⟨hFinal, hExtTail, hDenTail⟩ :=
      ihTail alg env hHeadArena (pre ++ [head])
        (by simpa using hPre')
    refine ⟨hFinal, hExtHead.trans hExtTail, ?_⟩
    simpa [List.append_assoc] using hDenTail

/-- The production structural lowering preserves denotation for every `Sig`
    constructor and every well-formed initial arena. -/
theorem lowerSigTree_preserves (sig : Sig) (arena : ExprArena)
    (hArena : ArenaWellFormed arena) (alg : Algebra α) (env : SigEnv α) :
    let result := lowerSigTree sig arena
    ∃ hResult : ArenaWellFormed result.2,
      Extends arena result.2 ∧
        denoteExpr alg env result.2 hResult result.1 =
          denoteSig alg env sig := by
  have hTrace := lowerSigTree_lowersTo sig arena
  obtain ⟨hResult, hExtends, node, hDeref, hValue⟩ :=
    hTrace.preserves alg env hArena
  exact ⟨hResult, hExtends, hValue⟩

end Tropical.Semantics
