import Tropical.EmitArrow.BuilderLaws
import Tropical.Semantics.Expr

/-!
# Exact semantics of `EmitArrow.shiftSampleIndex`

The builder certificate proves that the frozen DAG rebuild remains owned and
well formed.  This module adds the semantic ingredient: an explicit relation
between every old node and its recursively remapped image.  The relation is
deliberately independent of hash-cons allocation, so sharing and deduplication
are allowed while the child mapping remains exact.
-/

namespace Tropical.Semantics

open Tropical.Ir
open Tropical.EmitArrow

/-- Remap every child edge of a non-`sampleIndex` node.  The `sampleIndex`
    case is represented separately by `ShiftCopy`, because it maps directly
    to the already-built shifted clock rather than to a rebuilt node. -/
def remapShiftedNode (mapId : ExprId → ExprId) : ENode → ENode
  | .num n => .num n
  | .bool b => .bool b
  | .arr items => .arr (items.map mapId)
  | .tileArray items => .tileArray (items.map mapId)
  | .binary tag lhs rhs => .binary tag (mapId lhs) (mapId rhs)
  | .unary tag arg => .unary tag (mapId arg)
  | .clamp value lo hi => .clamp (mapId value) (mapId lo) (mapId hi)
  | .select cond then_ else_ =>
      .select (mapId cond) (mapId then_) (mapId else_)
  | .arraySet array index value =>
      .arraySet (mapId array) (mapId index) (mapId value)
  | .index array index => .index (mapId array) (mapId index)
  | .inputRef idx => .inputRef idx
  | .paramRef idx => .paramRef idx
  | .nestedOut instanceIdx outputIdx => .nestedOut instanceIdx outputIdx
  | .sampleRate => .sampleRate
  | .sampleIndex => .sampleIndex
  | .tileSampleIndex => .tileSampleIndex
  | .tilePhase => .tilePhase
  | .loopIdx binderId => .loopIdx binderId
  | .bankSum capacity tables body dynCount? binderId =>
      .bankSum capacity (tables.map mapId) (mapId body)
        (dynCount?.map mapId) binderId
  | .routedSum capacity outputCount routes tables values dynCount? binderId =>
      .routedSum capacity outputCount routes (tables.map mapId)
        (values.map mapId) (dynCount?.map mapId) binderId

/-- A hash-cons-insensitive recursive copy of a frozen arena.  Every ordinary
    node dereferences to the same constructor with mapped child IDs;
    `sampleIndex` alone maps to `shiftedTick`. -/
def ShiftCopy (before after : ExprArena) (mapId : ExprId → ExprId)
    (shiftedTick : ExprId) : Prop :=
  ∀ id node, before.deref id = some node →
    match node with
    | .sampleIndex => mapId id = shiftedTick
    | node => after.deref (mapId id) = some (remapShiftedNode mapId node)

/-- The portion of a frozen arena already rebuilt into `mapped`. -/
def PrefixShiftCopy (before after : ExprArena) (mapped : Array ExprId)
    (shiftedTick : ExprId) : Prop :=
  ∀ id node, before.deref id = some node → id.idx < mapped.size →
    match node with
    | .sampleIndex => shiftedId mapped id = shiftedTick
    | node => after.deref (shiftedId mapped id) =
        some (remapShiftedNode (shiftedId mapped) node)

/-- Array-prefix preservation for the mapping accumulated by the rebuild. -/
structure MappingExtends (before after : Array ExprId) : Prop where
  size_le : before.size ≤ after.size
  getElem?_eq : ∀ i, i < before.size → after[i]? = before[i]?

theorem MappingExtends.refl (mapped : Array ExprId) :
    MappingExtends mapped mapped :=
  ⟨Nat.le_refl _, fun _ _ => rfl⟩

theorem MappingExtends.trans {a b c : Array ExprId}
    (hab : MappingExtends a b) (hbc : MappingExtends b c) :
    MappingExtends a c :=
  ⟨Nat.le_trans hab.size_le hbc.size_le, fun i hi => by
    rw [hbc.getElem?_eq i (Nat.lt_of_lt_of_le hi hab.size_le),
      hab.getElem?_eq i hi]⟩

theorem MappingExtends.push (mapped : Array ExprId) (id : ExprId) :
    MappingExtends mapped (mapped.push id) :=
  ⟨by simp, fun i hi => by
    simpa [Array.getElem?_eq_getElem hi] using
      (Array.getElem?_push_lt (xs := mapped) (x := id) hi)⟩

theorem shiftedId_of_lt {mapped : Array ExprId} {id : ExprId}
    (hi : id.idx < mapped.size) :
    shiftedId mapped id = mapped[id.idx] := by
  simp [shiftedId, Array.getElem?_eq_getElem hi]

theorem shiftedId_eq_of_mappingExtends {before after : Array ExprId}
    (hExtends : MappingExtends before after) {id : ExprId}
    (hi : id.idx < before.size) :
    shiftedId after id = shiftedId before id := by
  unfold shiftedId
  rw [hExtends.getElem?_eq id.idx hi]

/-- Remapping a node only observes the images of that node's children. -/
theorem remapShiftedNode_congr {f g : ExprId → ExprId} {node : ENode}
    (h : ∀ child ∈ node.children, f child = g child) :
    remapShiftedNode f node = remapShiftedNode g node := by
  have hMap (items : Array ExprId)
      (hItems : ∀ child ∈ items, f child = g child) :
      items.map f = items.map g := by
    apply Array.ext
    · simp
    · intro i hiF hiG
      simp only [Array.getElem_map]
      have hi : i < items.size := by simpa using hiF
      exact hItems items[i] (by simp)
  cases node with
  | num | bool | inputRef | paramRef | nestedOut | sampleRate
  | sampleIndex | tileSampleIndex | tilePhase | loopIdx => rfl
  | arr items =>
    simp only [remapShiftedNode]
    rw [hMap items (by intro child hChild; exact h child (by
      simpa [ENode.children] using hChild))]
  | tileArray items =>
    simp only [remapShiftedNode]
    rw [hMap items (by intro child hChild; exact h child (by
      simpa [ENode.children] using hChild))]
  | binary tag lhs rhs =>
    simp only [remapShiftedNode]
    rw [h lhs (by simp [ENode.children]), h rhs (by simp [ENode.children])]
  | unary tag arg =>
    simp only [remapShiftedNode]
    rw [h arg (by simp [ENode.children])]
  | clamp value lo hi =>
    simp only [remapShiftedNode]
    rw [h value (by simp [ENode.children]), h lo (by simp [ENode.children]),
      h hi (by simp [ENode.children])]
  | select cond then_ else_ =>
    simp only [remapShiftedNode]
    rw [h cond (by simp [ENode.children]), h then_ (by simp [ENode.children]),
      h else_ (by simp [ENode.children])]
  | arraySet array index value =>
    simp only [remapShiftedNode]
    rw [h array (by simp [ENode.children]), h index (by simp [ENode.children]),
      h value (by simp [ENode.children])]
  | index array index =>
    simp only [remapShiftedNode]
    rw [h array (by simp [ENode.children]), h index (by simp [ENode.children])]
  | bankSum capacity tables body dynCount? binderId =>
    simp only [remapShiftedNode]
    rw [hMap tables (by intro child hChild; exact h child (by
      simp [ENode.children, hChild])), h body (by simp [ENode.children])]
    cases dynCount? with
    | none => rfl
    | some count => simp [h count (by simp [ENode.children])]
  | routedSum capacity outputCount routes tables values dynCount? binderId =>
    simp only [remapShiftedNode]
    rw [hMap tables (by intro child hChild; exact h child (by
      simp [ENode.children, hChild])),
      hMap values (by intro child hChild; exact h child (by
        simp [ENode.children, hChild]))]
    cases dynCount? with
    | none => rfl
    | some count => simp [h count (by simp [ENode.children])]

theorem remapShiftedNode_children (mapId : ExprId → ExprId) (node : ENode) :
    (remapShiftedNode mapId node).children = node.children.map mapId := by
  have hOption (item : Option ExprId) :
      (item.map mapId).toArray = item.toArray.map mapId := by
    cases item <;> simp
  cases node <;> simp [remapShiftedNode, ENode.children,
    Array.map_append, Array.map_push, hOption]

theorem internSig_deref_of_run {builder after : Builder} {node : ENode}
    {id : ExprId} (hBuilder : BuilderWellFormed builder)
    (hChildren : ENodeChildrenIn builder.exprs node)
    (hRun : (internSig node).run builder = .ok (id, after)) :
    after.exprs.deref id = some node := by
  let result := (eintern node).run builder.exprs
  have hIntern := eintern_preserves hBuilder.arena
    (enodeChildrenIn_iff_childrenInPrefix.mp hChildren)
  change ArenaWellFormed result.2 ∧ Extends builder.exprs result.2 ∧
    result.2.deref result.1 = some node at hIntern
  rw [internSig_run] at hRun
  change Except.ok (result.1,
    { builder with exprs := result.2 }) = Except.ok (id, after) at hRun
  have hPair := Except.ok.inj hRun
  have hId : result.1 = id := congrArg Prod.fst hPair
  have hAfter : { builder with exprs := result.2 } = after :=
    congrArg Prod.snd hPair
  subst id
  subst after
  exact hIntern.2.2

/-- One successful recursive rebuild returns the requested remapped node;
    `sampleIndex` is the sole case that returns the existing shifted clock. -/
theorem rebuildShiftedNode_image {original current after : Builder}
    {mapped : Array ExprId} {shiftedTick id : ExprId} {node : ENode}
    (hCurrent : BuilderWellFormed current)
    (hExtends : BuilderExtends original current)
    (hMapped : SigsIn current mapped)
    (hTick : SigIn current shiftedTick)
    (hChildren : ENodeChildrenIn original.exprs node)
    (hRun : (rebuildShiftedNode mapped shiftedTick node).run current =
      .ok (id, after)) :
    BuilderWellFormed after ∧ BuilderExtends current after ∧
      match node with
      | .sampleIndex => id = shiftedTick
      | node => after.exprs.deref id =
          some (remapShiftedNode (shiftedId mapped) node) := by
  have hPres := rebuildShiftedNode_preserves hCurrent hExtends hMapped
    hTick hChildren
  rw [hRun] at hPres
  refine ⟨hPres.1, hPres.2.1, ?_⟩
  have hRemapped : ENodeChildrenIn current.exprs
      (remapShiftedNode (shiftedId mapped) node) := by
    intro child hChild
    rw [remapShiftedNode_children] at hChild
    obtain ⟨oldChild, hOldChild, rfl⟩ := Array.mem_map.mp hChild
    exact shiftedId_in hExtends hMapped (hChildren oldChild hOldChild)
  cases node <;> simp only [rebuildShiftedNode, remapShiftedNode] at hRun ⊢
  case sampleIndex =>
    simpa using (congrArg Prod.fst (Except.ok.inj hRun)).symm
  all_goals
    exact internSig_deref_of_run hCurrent hRemapped hRun

/-- Extending both the target arena and the accumulated mapping preserves every
    already-certified image. -/
theorem PrefixShiftCopy.transport {before : ExprArena}
    {current after : Builder} {mapped result : Array ExprId}
    {shiftedTick : ExprId} (hBefore : ArenaWellFormed before)
    (hPrefix : PrefixShiftCopy before current.exprs mapped shiftedTick)
    (hBuilderExtends : BuilderExtends current after)
    (hMappingExtends : MappingExtends mapped result) :
    ∀ id node, before.deref id = some node → id.idx < mapped.size →
      match node with
      | .sampleIndex => shiftedId result id = shiftedTick
      | node => after.exprs.deref (shiftedId result id) =
          some (remapShiftedNode (shiftedId result) node) := by
  intro id node hDeref hi
  have hOld := hPrefix id node hDeref hi
  have hId := shiftedId_eq_of_mappingExtends hMappingExtends hi
  have hNonSample (hne : node ≠ .sampleIndex) :
      after.exprs.deref (shiftedId result id) =
        some (remapShiftedNode (shiftedId result) node) := by
    have hOld' : current.exprs.deref (shiftedId mapped id) =
        some (remapShiftedNode (shiftedId mapped) node) := by
      simpa [hne] using hOld
    have hNode := remapShiftedNode_congr
      (node := node) (f := shiftedId result) (g := shiftedId mapped) (by
        intro child hChild
        exact shiftedId_eq_of_mappingExtends hMappingExtends
          (Nat.lt_trans (hBefore.childrenDescend hDeref child hChild) hi))
    calc
      after.exprs.deref (shiftedId result id) =
          after.exprs.deref (shiftedId mapped id) := by rw [hId]
      _ = some (remapShiftedNode (shiftedId mapped) node) :=
        hBuilderExtends.exprs hOld'
      _ = some (remapShiftedNode (shiftedId result) node) :=
        congrArg some hNode.symm
  cases node with
  | sampleIndex => simpa [hId] using hOld
  | num | bool | arr | tileArray | binary | unary | clamp | select | arraySet
  | index | inputRef | paramRef | nestedOut | sampleRate | tileSampleIndex
  | tilePhase | loopIdx | bankSum | routedSum =>
    exact hNonSample (by simp)

theorem deref_of_nodes_split {arena : ExprArena} {done rest : List ENode}
    {node : ENode} (hSplit : arena.nodes.toList = done ++ node :: rest) :
    arena.deref ⟨done.length⟩ = some node := by
  rw [ExprArena.deref]
  have hGet := congrArg (fun items : List ENode => items[done.length]?) hSplit
  simpa using hGet

/-- Add the just-rebuilt head to an already-certified mapping prefix. -/
theorem PrefixShiftCopy.push {before : ExprArena}
    {current after : Builder} {mapped : Array ExprId}
    {shiftedTick newId : ExprId} {node : ENode}
    (hBefore : ArenaWellFormed before)
    (hPrefix : PrefixShiftCopy before current.exprs mapped shiftedTick)
    (hExtends : BuilderExtends current after)
    (hHeadDeref : before.deref ⟨mapped.size⟩ = some node)
    (hHeadSample : node = .sampleIndex → newId = shiftedTick)
    (hHeadOther : node ≠ .sampleIndex → after.exprs.deref newId =
      some (remapShiftedNode (shiftedId mapped) node)) :
    PrefixShiftCopy before after.exprs (mapped.push newId) shiftedTick := by
  have hOld := hPrefix.transport hBefore hExtends
    (MappingExtends.push mapped newId)
  intro id actual hDeref hi
  by_cases hlt : id.idx < mapped.size
  · exact hOld id actual hDeref hlt
  have heq : id.idx = mapped.size := by
    simpa using (Nat.eq_of_lt_succ_of_not_lt (by simpa using hi) hlt)
  have hId : id = ⟨mapped.size⟩ := by
    cases id
    simp_all
  subst id
  have hNode : actual = node :=
    Option.some.inj (hDeref.symm.trans hHeadDeref)
  subst actual
  have hNewId : shiftedId (mapped.push newId) ⟨mapped.size⟩ = newId := by
    simp [shiftedId]
  have hRemap :
      remapShiftedNode (shiftedId (mapped.push newId)) node =
        remapShiftedNode (shiftedId mapped) node :=
    remapShiftedNode_congr (by
      intro child hChild
      exact shiftedId_eq_of_mappingExtends (MappingExtends.push mapped newId)
        (hBefore.childrenDescend hHeadDeref child hChild))
  have hNonSample (hne : node ≠ .sampleIndex) :
      after.exprs.deref (shiftedId (mapped.push newId) ⟨mapped.size⟩) =
        some (remapShiftedNode (shiftedId (mapped.push newId)) node) := by
    calc
      after.exprs.deref (shiftedId (mapped.push newId) ⟨mapped.size⟩) =
          after.exprs.deref newId := by rw [hNewId]
      _ = some (remapShiftedNode (shiftedId mapped) node) := hHeadOther hne
      _ = some (remapShiftedNode (shiftedId (mapped.push newId)) node) :=
        congrArg some hRemap.symm
  cases node with
  | sampleIndex => simpa [hNewId] using hHeadSample rfl
  | num | bool | arr | tileArray | binary | unary | clamp | select | arraySet
  | index | inputRef | paramRef | nestedOut | sampleRate | tileSampleIndex
  | tilePhase | loopIdx | bankSum | routedSum =>
    exact hNonSample (by simp)

/-- The recursive production walk preserves its starting mapping, certifies
    every newly processed frozen node, and returns the expected mapping size. -/
theorem rebuildShiftedNodes_prefix {original current : Builder}
    {nodes done : List ENode} {shiftedTick : ExprId}
    {mapped : Array ExprId}
    (hOriginal : BuilderWellFormed original)
    (hCurrent : BuilderWellFormed current)
    (hExtends : BuilderExtends original current)
    (hTick : SigIn current shiftedTick)
    (hMapped : SigsIn current mapped)
    (hSplit : original.exprs.nodes.toList = done ++ nodes)
    (hSize : mapped.size = done.length)
    (hPrefix : PrefixShiftCopy original.exprs current.exprs mapped shiftedTick) :
    match (rebuildShiftedNodes nodes shiftedTick mapped).run current with
    | .error _ => True
    | .ok (result, after) =>
      BuilderWellFormed after ∧ BuilderExtends current after ∧
      SigsIn after result ∧ MappingExtends mapped result ∧
      PrefixShiftCopy original.exprs after.exprs result shiftedTick ∧
      result.size = mapped.size + nodes.length := by
  induction nodes generalizing done current mapped with
  | nil =>
    exact ⟨hCurrent, BuilderExtends.refl current, hMapped,
      MappingExtends.refl mapped, hPrefix, by simp⟩
  | cons node rest ih =>
    rw [show rebuildShiftedNodes (node :: rest) shiftedTick mapped =
      rebuildShiftedNode mapped shiftedTick node >>= fun id =>
        rebuildShiftedNodes rest shiftedTick (mapped.push id) from rfl]
    rw [StateT.run_bind]
    generalize hRunHead :
      (rebuildShiftedNode mapped shiftedTick node).run current = runHead
    cases runHead with
    | error message => trivial
    | ok pair =>
      rcases pair with ⟨newId, middle⟩
      have hHeadDeref : original.exprs.deref ⟨mapped.size⟩ = some node := by
        rw [hSize]
        exact deref_of_nodes_split hSplit
      have hChildren : ENodeChildrenIn original.exprs node := by
        intro child hChild
        exact Nat.lt_trans
          (hOriginal.arena.childrenDescend hHeadDeref child hChild)
          (deref_index_lt hHeadDeref)
      have hHeadPres := rebuildShiftedNode_preserves hCurrent hExtends hMapped
        hTick hChildren
      rw [hRunHead] at hHeadPres
      have hHead := rebuildShiftedNode_image hCurrent hExtends hMapped
        hTick hChildren hRunHead
      have hOriginalMiddle := BuilderExtends.trans hExtends hHead.2.1
      have hTickMiddle := hHead.2.1.sigIn hTick
      have hMappedMiddle : SigsIn middle (mapped.push newId) := by
        intro id hId
        rw [Array.mem_push] at hId
        rcases hId with hOld | rfl
        · exact hHead.2.1.sigIn (hMapped id hOld)
        · exact hHeadPres.2.2
      have hHeadSample : node = .sampleIndex → newId = shiftedTick := by
        intro hNode
        subst node
        simpa only using hHead.2.2
      have hHeadOther : node ≠ .sampleIndex → middle.exprs.deref newId =
          some (remapShiftedNode (shiftedId mapped) node) := by
        intro hNode
        cases node <;> try { exact (hNode rfl).elim } <;>
          simpa only using hHead.2.2
      have hPrefixMiddle : PrefixShiftCopy original.exprs middle.exprs
          (mapped.push newId) shiftedTick :=
        hPrefix.push hOriginal.arena hHead.2.1 hHeadDeref hHeadSample
          hHeadOther
      have hRestSplit : original.exprs.nodes.toList =
          (done ++ [node]) ++ rest := by
        simpa [List.append_assoc] using hSplit
      have hRestSize : (mapped.push newId).size =
          (done ++ [node]).length := by simp [hSize]
      have hTail := ih hHead.1 hOriginalMiddle hTickMiddle
        hMappedMiddle hRestSplit hRestSize hPrefixMiddle
      change match
          (rebuildShiftedNodes rest shiftedTick (mapped.push newId)).run middle with
        | .error _ => True
        | .ok (result, after) =>
          BuilderWellFormed after ∧ BuilderExtends current after ∧
          SigsIn after result ∧ MappingExtends mapped result ∧
          PrefixShiftCopy original.exprs after.exprs result shiftedTick ∧
          result.size = mapped.size + (node :: rest).length
      generalize hRunTail :
        (rebuildShiftedNodes rest shiftedTick (mapped.push newId)).run middle =
          runTail at hTail ⊢
      cases runTail with
      | error message => trivial
      | ok pair =>
        rcases pair with ⟨result, after⟩
        refine ⟨hTail.1, BuilderExtends.trans hHead.2.1 hTail.2.1,
          hTail.2.2.1,
          MappingExtends.trans (MappingExtends.push mapped newId)
            hTail.2.2.2.1,
          hTail.2.2.2.2.1, ?_⟩
        simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using
          hTail.2.2.2.2.2

/-- A complete frozen-arena rebuild establishes the recursive copy relation. -/
theorem rebuildShiftedNodes_shiftCopy {original current : Builder}
    {shiftedTick : ExprId}
    (hOriginal : BuilderWellFormed original)
    (hCurrent : BuilderWellFormed current)
    (hExtends : BuilderExtends original current)
    (hTick : SigIn current shiftedTick) :
    match (rebuildShiftedNodes original.exprs.nodes.toList shiftedTick).run
        current with
    | .error _ => True
    | .ok (mapped, after) =>
      BuilderWellFormed after ∧ BuilderExtends current after ∧
      SigsIn after mapped ∧
      ShiftCopy original.exprs after.exprs (shiftedId mapped) shiftedTick ∧
      mapped.size = original.exprs.nodes.size := by
  have hResult := rebuildShiftedNodes_prefix hOriginal hCurrent hExtends hTick
    (mapped := #[]) (done := []) (nodes := original.exprs.nodes.toList)
    (by simp [SigsIn]) (by simp) (by simp) (by
      intro id node hDeref hi
      simp at hi)
  generalize hRun :
    (rebuildShiftedNodes original.exprs.nodes.toList shiftedTick).run current =
      result at hResult ⊢
  cases result with
  | error message => trivial
  | ok pair =>
    rcases pair with ⟨mapped, after⟩
    refine ⟨hResult.1, hResult.2.1, hResult.2.2.1, ?_, ?_⟩
    · intro id node hDeref
      exact hResult.2.2.2.2.1 id node hDeref (by
        rw [hResult.2.2.2.2.2]
        simpa using deref_index_lt hDeref)
    · simpa using hResult.2.2.2.2.2

/-- Environments used on the two sides differ only at the sample coordinate.
    Keeping loop bindings in the relation is what makes the theorem apply
    recursively inside ordinary banks and routed banks. -/
structure AgreesOutsideSampleIndex (source shifted : SigEnv α) : Prop where
  inputs : source.inputs = shifted.inputs
  params : source.params = shifted.params
  nestedOutputs : source.nestedOutputs = shifted.nestedOutputs
  sampleRate : source.sampleRate = shifted.sampleRate
  tileSampleIndex : source.tileSampleIndex = shifted.tileSampleIndex
  loops : source.loops = shifted.loops

theorem AgreesOutsideSampleIndex.bindLoop
    {source shifted : SigEnv α}
    (h : AgreesOutsideSampleIndex source shifted) (binderId : Nat)
    (value : Value α) :
    AgreesOutsideSampleIndex (source.bindLoop binderId value)
      (shifted.bindLoop binderId value) := by
  constructor <;> simp [SigEnv.bindLoop, h.inputs, h.params,
    h.nestedOutputs, h.sampleRate, h.tileSampleIndex, h.loops]

/-- The semantic substitution theorem for an explicit recursive copy.

`hTick` is the precise, carrier-parametric statement of `t + frames`: the
replacement clock evaluates to the source environment's sample coordinate.
For the concrete `buildShiftedTick`, a companion theorem below discharges it
from the algebra's literal/conversion/addition behavior. -/
theorem denoteExpr_shiftCopy {before after : ExprArena}
    (hBefore : ArenaWellFormed before) (hAfter : ArenaWellFormed after)
    {mapId : ExprId → ExprId} {shiftedTick : ExprId}
    (hCopy : ShiftCopy before after mapId shiftedTick)
    (alg : Algebra α) {source shifted : SigEnv α}
    (hEnv : AgreesOutsideSampleIndex source shifted)
    (hTick : ∀ tickEnv, tickEnv.tileSampleIndex = shifted.tileSampleIndex →
      denoteExpr alg tickEnv after hAfter shiftedTick =
        .ok source.sampleIndex) {root : ExprId} {node : ENode}
    (hRoot : before.deref root = some node) :
    denoteExpr alg source before hBefore root =
      denoteExpr alg shifted after hAfter (mapId root) := by
  have hChild {child : ExprId} (hMem : child ∈ node.children)
      {childSource childShifted : SigEnv α}
      (hChildEnv : AgreesOutsideSampleIndex childSource childShifted)
      (hChildTick : ∀ tickEnv,
        tickEnv.tileSampleIndex = childShifted.tileSampleIndex →
        denoteExpr alg tickEnv after hAfter shiftedTick =
          .ok childSource.sampleIndex) :
      denoteExpr alg childSource before hBefore child =
        denoteExpr alg childShifted after hAfter (mapId child) := by
    have hLt : child.idx < root.idx :=
      hBefore.childrenDescend hRoot child hMem
    obtain ⟨childNode, hChildRoot⟩ := deref_of_index_lt
      (Nat.lt_trans hLt (deref_index_lt hRoot))
    exact denoteExpr_shiftCopy hBefore hAfter hCopy alg hChildEnv hChildTick
      hChildRoot
  rw [denoteExpr_of_deref alg source before hBefore hRoot]
  have hImage := hCopy root node hRoot
  cases node with
  | sampleIndex =>
    simpa [denoteNode, hImage] using (hTick shifted rfl).symm
  | num n =>
    rw [denoteExpr_of_deref alg shifted after hAfter hImage]
    rfl
  | bool b =>
    rw [denoteExpr_of_deref alg shifted after hAfter hImage]
    rfl
  | inputRef inputIdx =>
    rw [denoteExpr_of_deref alg shifted after hAfter hImage]
    simp [denoteNode, remapShiftedNode, hEnv.inputs]
  | paramRef paramIdx =>
    rw [denoteExpr_of_deref alg shifted after hAfter hImage]
    simp [denoteNode, remapShiftedNode, hEnv.params]
  | nestedOut instanceIdx outputIdx =>
    rw [denoteExpr_of_deref alg shifted after hAfter hImage]
    simp [denoteNode, remapShiftedNode, lookupNested, hEnv.nestedOutputs]
  | sampleRate =>
    rw [denoteExpr_of_deref alg shifted after hAfter hImage]
    simp [denoteNode, remapShiftedNode, hEnv.sampleRate]
  | tileSampleIndex =>
    rw [denoteExpr_of_deref alg shifted after hAfter hImage]
    exact congrArg Except.ok hEnv.tileSampleIndex
  | tilePhase =>
    rw [denoteExpr_of_deref alg shifted after hAfter hImage]
    rfl
  | loopIdx binderId =>
    rw [denoteExpr_of_deref alg shifted after hAfter hImage]
    simp [denoteNode, remapShiftedNode, hEnv.loops]
  | arr items =>
    rw [denoteExpr_of_deref alg shifted after hAfter hImage]
    simp only [denoteNode, remapShiftedNode]
    congr 2
    apply Array.ext
    · simp
    · intro i hiSource hiShifted
      simp only [Array.getElem_map, Array.getElem_attach]
      exact hChild (by simp [ENode.children]) hEnv hTick
  | tileArray items =>
    rw [denoteExpr_of_deref alg shifted after hAfter hImage]
    simp only [denoteNode, remapShiftedNode]
    congr 2
    apply Array.ext
    · simp
    · intro i hiSource hiShifted
      simp only [Array.getElem_map, Array.getElem_attach]
      exact hChild (by simp [ENode.children]) hEnv hTick
  | binary tag lhs rhs =>
    rw [denoteExpr_of_deref alg shifted after hAfter hImage]
    simp only [denoteNode, remapShiftedNode]
    rw [hChild (by simp [ENode.children]) hEnv hTick,
      hChild (by simp [ENode.children]) hEnv hTick]
  | unary tag arg =>
    rw [denoteExpr_of_deref alg shifted after hAfter hImage]
    simp only [denoteNode, remapShiftedNode]
    rw [hChild (by simp [ENode.children]) hEnv hTick]
  | clamp value lo hi =>
    rw [denoteExpr_of_deref alg shifted after hAfter hImage]
    simp only [denoteNode, remapShiftedNode]
    rw [hChild (by simp [ENode.children]) hEnv hTick,
      hChild (by simp [ENode.children]) hEnv hTick,
      hChild (by simp [ENode.children]) hEnv hTick]
  | select cond then_ else_ =>
    rw [denoteExpr_of_deref alg shifted after hAfter hImage]
    simp only [denoteNode, remapShiftedNode]
    rw [hChild (by simp [ENode.children]) hEnv hTick,
      hChild (by simp [ENode.children]) hEnv hTick,
      hChild (by simp [ENode.children]) hEnv hTick]
  | arraySet array index value =>
    rw [denoteExpr_of_deref alg shifted after hAfter hImage]
    rfl
  | index array index =>
    rw [denoteExpr_of_deref alg shifted after hAfter hImage]
    simp only [denoteNode, remapShiftedNode]
    rw [hChild (by simp [ENode.children]) hEnv hTick,
      hChild (by simp [ENode.children]) hEnv hTick]
  | bankSum capacity tables body dynCount? binderId =>
    rw [denoteExpr_of_deref alg shifted after hAfter hImage]
    simp only [denoteNode, remapShiftedNode]
    have hTables :
        tables.attach.map
            (fun item => denoteExpr alg source before hBefore item.1) =
          (tables.map mapId).attach.map
            (fun item => denoteExpr alg shifted after hAfter item.1) := by
      apply Array.ext
      · simp
      · intro i hiSource hiShifted
        simp only [Array.getElem_map, Array.getElem_attach]
        exact hChild (by simp [ENode.children]) hEnv hTick
    have hBody (loopValue : Value α) :
        denoteExpr alg (source.bindLoop binderId loopValue)
            before hBefore body =
          denoteExpr alg (shifted.bindLoop binderId loopValue)
            after hAfter (mapId body) :=
      hChild (by simp [ENode.children])
        (hEnv.bindLoop binderId loopValue) (by
          simpa [SigEnv.bindLoop] using hTick)
    rw [hTables]
    cases dynCount? with
    | none => simp [hBody]
    | some count =>
      have hCount := hChild (child := count)
        (by simp [ENode.children]) hEnv hTick
      simp only [Option.map_some]
      rw [hCount]
      simp only [hBody]
  | routedSum capacity outputCount routes tables values dynCount? binderId =>
    rw [denoteExpr_of_deref alg shifted after hAfter hImage]
    simp only [denoteNode, remapShiftedNode, Array.size_map]
    have hTables :
        tables.attach.map
            (fun item => denoteExpr alg source before hBefore item.1) =
          (tables.map mapId).attach.map
            (fun item => denoteExpr alg shifted after hAfter item.1) := by
      apply Array.ext
      · simp
      · intro i hiSource hiShifted
        simp only [Array.getElem_map, Array.getElem_attach]
        exact hChild (by simp [ENode.children]) hEnv hTick
    have hValues (loopValue : Value α) :
        values.attach.map
            (fun item => denoteExpr alg
              (source.bindLoop binderId loopValue)
              before hBefore item.1) =
          (values.map mapId).attach.map
            (fun item => denoteExpr alg
              (shifted.bindLoop binderId loopValue)
              after hAfter item.1) := by
      apply Array.ext
      · simp
      · intro i hiSource hiShifted
        simp only [Array.getElem_map, Array.getElem_attach]
        exact hChild (by simp [ENode.children])
          (hEnv.bindLoop binderId loopValue) (by
            simpa [SigEnv.bindLoop] using hTick)
    rw [hTables]
    cases dynCount? with
    | none => simp [hValues]
    | some count =>
      have hCount := hChild (child := count)
        (by simp [ENode.children]) hEnv hTick
      simp only [Option.map_some]
      rw [hCount]
      simp only [hValues]
termination_by root.idx
decreasing_by
  exact hBefore.childrenDescend hRoot child hMem

end Tropical.Semantics
