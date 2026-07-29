import Tropical.Semantics.Arena

/-!
# Checked relational lowering for the production `lowerSigTree`

This is fallback 1 from the semantic-spine handoff.  `LowersTo` is an
all-constructor operational relation over the production `Sig`, `ENode`,
`ExprArena`, and `eintern`.  It exposes every child state transition and the
exact final intern, including ordered arrays, optional dynamic counts, and bank
binder ids.  The capstone theorem proves that the production structural
reference produces this relation.

The relation is intentionally not advertised as the requested denotational
preservation theorem.  Completing that theorem requires preservation of
`ArenaWellFormed.dedupSound` across `eintern`, which production does not yet
make provable (see `Arena.lean` and the semantics map).
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

/-- The checked fallback capstone: the production structural reference realizes
    `LowersTo` for all fourteen constructors, from every initial arena. -/
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

end Tropical.Semantics
