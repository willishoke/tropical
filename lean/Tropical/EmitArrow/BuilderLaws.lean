import Tropical.Semantics.WellFormed

/-!
# Qualified laws for the ID-native builder

The public `BuildM` is an ordinary state transformer and therefore admits
unchecked `set`/`modify`.  The contracts below certify the production smart
constructors, with explicit ownership hypotheses for every input `Sig`.
-/

namespace Tropical.EmitArrow

open Tropical.Ir
open Tropical.Semantics

/-- A successful signal-producing action preserves builder invariants and
    returns an addressable signal.  Failure is observationally harmless. -/
def ProducesSig (action : BuildM Sig) : Prop :=
  ∀ builder, BuilderWellFormed builder →
    match action.run builder with
    | .error _ => True
    | .ok (id, after) =>
      BuilderWellFormed after ∧ BuilderExtends builder after ∧ SigIn after id

/-- Postcondition for one concrete signal-building run. -/
def SigBuildResultWellFormed (before : Builder)
    (result : Except String (Sig × Builder)) : Prop :=
  match result with
  | .error _ => True
  | .ok (id, after) =>
    BuilderWellFormed after ∧ BuilderExtends before after ∧ SigIn after id

/-- Every signal in an authored array belongs to the active builder. -/
def SigsIn (builder : Builder) (ids : Array Sig) : Prop :=
  ∀ id ∈ ids, SigIn builder id

/-- An optional signal, when present, belongs to the active builder. -/
def OptionalSigIn (builder : Builder) (id? : Option Sig) : Prop :=
  ∀ id ∈ id?, SigIn builder id

theorem pure_preserves_builderWF {α : Type} (value : α) :
    PreservesBuilderWF (pure value : BuildM α) := by
  intro builder hBuilder
  exact ⟨hBuilder, BuilderExtends.refl builder⟩

/-- Generic sequencing rule used by the multi-intern production helpers. -/
theorem bind_preserves {α β : Type} (first : BuildM α) (next : α → BuildM β)
    (validFirst : Builder → α → Prop) (validFinal : Builder → β → Prop)
    (hFirst : ∀ builder, BuilderWellFormed builder →
      match first.run builder with
      | .error _ => True
      | .ok (value, after) => BuilderWellFormed after ∧
        BuilderExtends builder after ∧ validFirst after value)
    (hNext : ∀ builder value, BuilderWellFormed builder →
      validFirst builder value →
      match (next value).run builder with
      | .error _ => True
      | .ok (result, after) => BuilderWellFormed after ∧
        BuilderExtends builder after ∧ validFinal after result) :
    ∀ builder, BuilderWellFormed builder →
      match (first >>= next).run builder with
      | .error _ => True
      | .ok (result, after) => BuilderWellFormed after ∧
        BuilderExtends builder after ∧ validFinal after result := by
  intro builder hBuilder
  rw [StateT.run_bind]
  generalize hRunFirst : first.run builder = runFirst
  cases runFirst with
  | error message => trivial
  | ok pair =>
    rcases pair with ⟨value, middle⟩
    change match (next value).run middle with
      | .error _ => True
      | .ok (result, after) => BuilderWellFormed after ∧
        BuilderExtends builder after ∧ validFinal after result
    have hFirstResult := hFirst builder hBuilder
    rw [hRunFirst] at hFirstResult
    have hNextResult := hNext middle value hFirstResult.1 hFirstResult.2.2
    generalize hRunNext : (next value).run middle = runNext at hNextResult ⊢
    cases runNext with
    | error message => trivial
    | ok pair =>
      rcases pair with ⟨result, after⟩
      exact ⟨hNextResult.1,
        BuilderExtends.trans hFirstResult.2.1 hNextResult.2.1,
        hNextResult.2.2⟩

/-- State-specialized sequencing, allowing the first postcondition to retain
    facts about the particular input builder. -/
theorem bind_preserves_at {α β : Type} {builder : Builder}
    (first : BuildM α) (next : α → BuildM β)
    (validFirst : Builder → α → Prop) (validFinal : Builder → β → Prop)
    (hFirst : match first.run builder with
      | .error _ => True
      | .ok (value, after) => BuilderWellFormed after ∧
        BuilderExtends builder after ∧ validFirst after value)
    (hNext : ∀ middle value, BuilderWellFormed middle →
      BuilderExtends builder middle → validFirst middle value →
      match (next value).run middle with
      | .error _ => True
      | .ok (result, after) => BuilderWellFormed after ∧
        BuilderExtends middle after ∧ validFinal after result) :
    match (first >>= next).run builder with
    | .error _ => True
    | .ok (result, after) => BuilderWellFormed after ∧
      BuilderExtends builder after ∧ validFinal after result := by
  rw [StateT.run_bind]
  generalize hRunFirst : first.run builder = runFirst at hFirst ⊢
  cases runFirst with
  | error message => trivial
  | ok pair =>
    rcases pair with ⟨value, middle⟩
    change match (next value).run middle with
      | .error _ => True
      | .ok (result, after) => BuilderWellFormed after ∧
        BuilderExtends builder after ∧ validFinal after result
    have hNextResult := hNext middle value hFirst.1 hFirst.2.1 hFirst.2.2
    generalize hRunNext : (next value).run middle = runNext at hNextResult ⊢
    cases runNext with
    | error message => trivial
    | ok pair =>
      rcases pair with ⟨result, after⟩
      exact ⟨hNextResult.1,
        BuilderExtends.trans hFirst.2.1 hNextResult.2.1,
        hNextResult.2.2⟩

theorem internSig_run (builder : Builder) (node : ENode) :
    (internSig node).run builder =
      .ok (((eintern node).run builder.exprs).1,
        { builder with exprs := ((eintern node).run builder.exprs).2 }) := by
  rfl

/-- One qualified intern step preserves the builder and returns an owned id. -/
theorem internSig_preserves_of_children {builder : Builder} {node : ENode}
    (hBuilder : BuilderWellFormed builder)
    (hChildren : ENodeChildrenIn builder.exprs node) :
    match (internSig node).run builder with
    | .error _ => True
    | .ok (id, after) =>
      BuilderWellFormed after ∧ BuilderExtends builder after ∧ SigIn after id := by
  rw [internSig_run]
  let result := (eintern node).run builder.exprs
  have hIntern := eintern_preserves hBuilder.arena
    (enodeChildrenIn_iff_childrenInPrefix.mp hChildren)
  change ArenaWellFormed result.2 ∧ Extends builder.exprs result.2 ∧
    result.2.deref result.1 = some node at hIntern
  have hOldSig : ∀ ⦃id⦄, SigIn builder id →
      ExprIdIn result.2 id := by
    intro id hid
    change ExprIdIn builder.exprs id at hid
    rw [exprIdIn_iff_deref] at hid ⊢
    obtain ⟨oldNode, hOldNode⟩ := hid
    exact ⟨oldNode, hIntern.2.1 hOldNode⟩
  refine ⟨⟨hIntern.1, ?_⟩, ⟨hIntern.2.1, ?_⟩, ?_⟩
  · intro decl hDecl input hInput
    change ExprIdIn result.2 input.value
    exact hOldSig (hBuilder.decls decl hDecl input hInput)
  · intro i decl hDecl
    simpa using hDecl
  · change ExprIdIn result.2 result.1
    exact deref_index_lt hIntern.2.2

/-- Common qualified-intern contract. -/
theorem internSig_preserves (node : ENode)
    (hChildren : ∀ builder, BuilderWellFormed builder →
      ENodeChildrenIn builder.exprs node) :
    ProducesSig (internSig node) := by
  intro builder hBuilder
  exact internSig_preserves_of_children hBuilder (hChildren builder hBuilder)

private theorem childrenIn_arr {builder : Builder} {items : Array Sig}
    (h : SigsIn builder items) :
    ENodeChildrenIn builder.exprs (.arr items) := by
  intro child hChild
  exact h child hChild

private theorem childrenIn_bankSum {builder : Builder} {count idxId : Nat}
    {tables : Array Sig} {body : Sig} {dynCount? : Option Sig}
    (hTables : SigsIn builder tables) (hBody : SigIn builder body)
    (hDyn : OptionalSigIn builder dynCount?) :
    ENodeChildrenIn builder.exprs (.bankSum count tables body dynCount? idxId) := by
  intro child hChild
  simp only [ENode.children, Array.mem_append, Array.mem_push] at hChild
  rcases hChild with (hChild | rfl) | hChild
  · exact hTables child hChild
  · exact hBody
  · cases dynCount? with
    | none => simp at hChild
    | some dyn =>
      have hEq : child = dyn := by simpa using hChild
      subst child
      exact hDyn dyn (by simp)

private theorem childrenIn_routedSum {builder : Builder}
    {capacity outputCount idxId : Nat} {routes : Array (Option Nat)}
    {tables values : Array Sig} {dynCount? : Option Sig}
    (hTables : SigsIn builder tables) (hValues : SigsIn builder values)
    (hDyn : OptionalSigIn builder dynCount?) :
    ENodeChildrenIn builder.exprs
      (.routedSum capacity outputCount routes tables values dynCount? idxId) := by
  intro child hChild
  simp only [ENode.children, Array.mem_append] at hChild
  rcases hChild with (hChild | hChild) | hChild
  · exact hTables child hChild
  · exact hValues child hChild
  · cases dynCount? with
    | none => simp at hChild
    | some dyn =>
      have hEq : child = dyn := by simpa using hChild
      subst child
      exact hDyn dyn (by simp)

-- Leaf constructors ----------------------------------------------------------

theorem num_preserves (n : Lean.JsonNumber) : ProducesSig (num n) := by
  simpa [num] using internSig_preserves (.num n) (by
    intro builder hBuilder
    simp [ENodeChildrenIn, ENode.children])

theorem lit_preserves (mantissa : Int) (exponent : Nat := 0) :
    ProducesSig (lit mantissa exponent) := by
  simpa [lit] using num_preserves (⟨mantissa, exponent⟩ : Lean.JsonNumber)

theorem inputRef_preserves (idx : InputIdx) : ProducesSig (inputRef idx) := by
  simpa [inputRef] using internSig_preserves (.inputRef idx) (by
    intro builder hBuilder
    simp [ENodeChildrenIn, ENode.children])

theorem paramRef_preserves (idx : ParamIdx) : ProducesSig (paramRef idx) := by
  simpa [paramRef] using internSig_preserves (.paramRef idx) (by
    intro builder hBuilder
    simp [ENodeChildrenIn, ENode.children])

theorem nestedOut_preserves (instance_ : InstanceIdx) (output : OutputIdx) :
    ProducesSig (nestedOut instance_ output) := by
  simpa [nestedOut] using internSig_preserves (.nestedOut instance_ output) (by
    intro builder hBuilder
    simp [ENodeChildrenIn, ENode.children])

theorem sampleRate_preserves : ProducesSig sampleRate := by
  simpa [sampleRate] using internSig_preserves .sampleRate (by
    intro builder hBuilder
    simp [ENodeChildrenIn, ENode.children])

theorem sampleIndex_preserves : ProducesSig sampleIndex := by
  simpa [sampleIndex] using internSig_preserves .sampleIndex (by
    intro builder hBuilder
    simp [ENodeChildrenIn, ENode.children])

theorem loopIdx_preserves (idxId : Nat) : ProducesSig (loopIdx idxId) := by
  simpa [loopIdx] using internSig_preserves (.loopIdx idxId) (by
    intro builder hBuilder
    simp [ENodeChildrenIn, ENode.children])

-- Constructors with owned children ------------------------------------------

theorem binary_preserves {builder : Builder} (hBuilder : BuilderWellFormed builder)
    (tag : BinaryOpTag) {lhs rhs : Sig}
    (hLhs : SigIn builder lhs) (hRhs : SigIn builder rhs) :
    match (binary tag lhs rhs).run builder with
    | .error _ => True
    | .ok (id, after) =>
      BuilderWellFormed after ∧ BuilderExtends builder after ∧ SigIn after id := by
  simpa [binary] using internSig_preserves_of_children (node := .binary tag lhs rhs)
    hBuilder (by simpa [ENodeChildrenIn, ENode.children, SigIn] using And.intro hLhs hRhs)

theorem unary_preserves {builder : Builder} (hBuilder : BuilderWellFormed builder)
    (tag : UnaryOpTag) {arg : Sig} (hArg : SigIn builder arg) :
    match (unary tag arg).run builder with
    | .error _ => True
    | .ok (id, after) =>
      BuilderWellFormed after ∧ BuilderExtends builder after ∧ SigIn after id := by
  simpa [unary] using internSig_preserves_of_children (node := .unary tag arg)
    hBuilder (by simpa [ENodeChildrenIn, ENode.children, SigIn] using hArg)

theorem clamp_preserves {builder : Builder} (hBuilder : BuilderWellFormed builder)
    {value lo hi : Sig} (hValue : SigIn builder value)
    (hLo : SigIn builder lo) (hHi : SigIn builder hi) :
    match (clamp value lo hi).run builder with
    | .error _ => True
    | .ok (id, after) =>
      BuilderWellFormed after ∧ BuilderExtends builder after ∧ SigIn after id := by
  simpa [clamp] using internSig_preserves_of_children
    (node := .clamp value lo hi) hBuilder
    (by simpa [ENodeChildrenIn, ENode.children, SigIn] using And.intro hValue (And.intro hLo hHi))

theorem select_preserves {builder : Builder} (hBuilder : BuilderWellFormed builder)
    {cond then_ else_ : Sig} (hCond : SigIn builder cond)
    (hThen : SigIn builder then_) (hElse : SigIn builder else_) :
    match (select cond then_ else_).run builder with
    | .error _ => True
    | .ok (id, after) =>
      BuilderWellFormed after ∧ BuilderExtends builder after ∧ SigIn after id := by
  simpa [select] using internSig_preserves_of_children
    (node := .select cond then_ else_) hBuilder
    (by simpa [ENodeChildrenIn, ENode.children, SigIn] using And.intro hCond (And.intro hThen hElse))

theorem arr_preserves {builder : Builder} (hBuilder : BuilderWellFormed builder)
    {items : Array Sig} (hItems : SigsIn builder items) :
    match (arr items).run builder with
    | .error _ => True
    | .ok (id, after) =>
      BuilderWellFormed after ∧ BuilderExtends builder after ∧ SigIn after id := by
  simpa [arr] using internSig_preserves_of_children
    (node := .arr items) hBuilder (childrenIn_arr hItems)

theorem index_preserves {builder : Builder} (hBuilder : BuilderWellFormed builder)
    {array index_ : Sig} (hArray : SigIn builder array)
    (hIndex : SigIn builder index_) :
    match (index array index_).run builder with
    | .error _ => True
    | .ok (id, after) =>
      BuilderWellFormed after ∧ BuilderExtends builder after ∧ SigIn after id := by
  simpa [index] using internSig_preserves_of_children
    (node := .index array index_) hBuilder
    (by simpa [ENodeChildrenIn, ENode.children, SigIn] using And.intro hArray hIndex)

theorem bankSum_preserves {builder : Builder} (hBuilder : BuilderWellFormed builder)
    (count : Nat) {tables : Array Sig} {body : Sig} {dynCount? : Option Sig}
    (idxId : Nat := 0) (hTables : SigsIn builder tables)
    (hBody : SigIn builder body) (hDyn : OptionalSigIn builder dynCount?) :
    match (bankSum count tables body dynCount? idxId).run builder with
    | .error _ => True
    | .ok (id, after) =>
      BuilderWellFormed after ∧ BuilderExtends builder after ∧ SigIn after id := by
  simpa [bankSum] using internSig_preserves_of_children
    (node := .bankSum count tables body dynCount? idxId) hBuilder
    (childrenIn_bankSum hTables hBody hDyn)

theorem routedSum_preserves {builder : Builder}
    (hBuilder : BuilderWellFormed builder) (capacity outputCount : Nat)
    (routes : Array (Option Nat)) {tables values : Array Sig}
    {dynCount? : Option Sig} (idxId : Nat := 0)
    (hTables : SigsIn builder tables) (hValues : SigsIn builder values)
    (hDyn : OptionalSigIn builder dynCount?) :
    match (routedSum capacity outputCount routes tables values dynCount? idxId).run builder with
    | .error _ => True
    | .ok (id, after) =>
      BuilderWellFormed after ∧ BuilderExtends builder after ∧ SigIn after id := by
  simpa [routedSum] using internSig_preserves_of_children
    (node := .routedSum capacity outputCount routes tables values dynCount? idxId)
    hBuilder (childrenIn_routedSum hTables hValues hDyn)

-- Scalar vocabulary aliases and short sequences ------------------------------

theorem mul_preserves {builder : Builder} (hBuilder : BuilderWellFormed builder)
    {a b : Sig} (ha : SigIn builder a) (hb : SigIn builder b) :
    SigBuildResultWellFormed builder ((mul a b).run builder) :=
  by simpa [mul] using binary_preserves hBuilder .mul ha hb

theorem add_preserves {builder : Builder} (hBuilder : BuilderWellFormed builder)
    {a b : Sig} (ha : SigIn builder a) (hb : SigIn builder b) :
    SigBuildResultWellFormed builder ((add a b).run builder) :=
  by simpa [add] using binary_preserves hBuilder .add ha hb

theorem sub_preserves {builder : Builder} (hBuilder : BuilderWellFormed builder)
    {a b : Sig} (ha : SigIn builder a) (hb : SigIn builder b) :
    SigBuildResultWellFormed builder ((sub a b).run builder) :=
  by simpa [sub] using binary_preserves hBuilder .sub ha hb

theorem div_preserves {builder : Builder} (hBuilder : BuilderWellFormed builder)
    {a b : Sig} (ha : SigIn builder a) (hb : SigIn builder b) :
    SigBuildResultWellFormed builder ((div a b).run builder) :=
  by simpa [div] using binary_preserves hBuilder .div ha hb

theorem bitAnd_preserves {builder : Builder} (hBuilder : BuilderWellFormed builder)
    {a b : Sig} (ha : SigIn builder a) (hb : SigIn builder b) :
    SigBuildResultWellFormed builder ((bitAnd a b).run builder) :=
  by simpa [bitAnd] using binary_preserves hBuilder .bitAnd ha hb

theorem bitOr_preserves {builder : Builder} (hBuilder : BuilderWellFormed builder)
    {a b : Sig} (ha : SigIn builder a) (hb : SigIn builder b) :
    SigBuildResultWellFormed builder ((bitOr a b).run builder) :=
  by simpa [bitOr] using binary_preserves hBuilder .bitOr ha hb

theorem rshift_preserves {builder : Builder} (hBuilder : BuilderWellFormed builder)
    {a b : Sig} (ha : SigIn builder a) (hb : SigIn builder b) :
    SigBuildResultWellFormed builder ((rshift a b).run builder) :=
  by simpa [rshift] using binary_preserves hBuilder .rshift ha hb

theorem lshift_preserves {builder : Builder} (hBuilder : BuilderWellFormed builder)
    {a b : Sig} (ha : SigIn builder a) (hb : SigIn builder b) :
    SigBuildResultWellFormed builder ((lshift a b).run builder) :=
  by simpa [lshift] using binary_preserves hBuilder .lshift ha hb

theorem gt_preserves {builder : Builder} (hBuilder : BuilderWellFormed builder)
    {a b : Sig} (ha : SigIn builder a) (hb : SigIn builder b) :
    SigBuildResultWellFormed builder ((gt a b).run builder) :=
  by simpa [gt] using binary_preserves hBuilder .gt ha hb

theorem ldexpE_preserves {builder : Builder}
    (hBuilder : BuilderWellFormed builder) {mantissa exponent : Sig}
    (hm : SigIn builder mantissa) (he : SigIn builder exponent) :
    SigBuildResultWellFormed builder ((ldexpE mantissa exponent).run builder) :=
  by simpa [ldexpE] using binary_preserves hBuilder .ldexp hm he

theorem toIntE_preserves {builder : Builder} (hBuilder : BuilderWellFormed builder)
    {a : Sig} (ha : SigIn builder a) :
    SigBuildResultWellFormed builder ((toIntE a).run builder) :=
  by simpa [toIntE] using unary_preserves hBuilder .toInt ha

theorem neg_preserves {builder : Builder} (hBuilder : BuilderWellFormed builder)
    {a : Sig} (ha : SigIn builder a) :
    SigBuildResultWellFormed builder ((neg a).run builder) :=
  by simpa [neg] using unary_preserves hBuilder .neg ha

theorem absE_preserves {builder : Builder} (hBuilder : BuilderWellFormed builder)
    {a : Sig} (ha : SigIn builder a) :
    SigBuildResultWellFormed builder ((absE a).run builder) :=
  by simpa [absE] using unary_preserves hBuilder .abs ha

theorem floatExponentE_preserves {builder : Builder}
    (hBuilder : BuilderWellFormed builder) {a : Sig} (ha : SigIn builder a) :
    SigBuildResultWellFormed builder ((floatExponentE a).run builder) :=
  by simpa [floatExponentE] using unary_preserves hBuilder .floatExponent ha

theorem roundE_preserves {builder : Builder} (hBuilder : BuilderWellFormed builder)
    {a : Sig} (ha : SigIn builder a) :
    SigBuildResultWellFormed builder ((roundE a).run builder) :=
  by simpa [roundE] using unary_preserves hBuilder .round ha

theorem toFloatE_preserves {builder : Builder}
    (hBuilder : BuilderWellFormed builder) {a : Sig} (ha : SigIn builder a) :
    SigBuildResultWellFormed builder ((toFloatE a).run builder) :=
  by simpa [toFloatE] using unary_preserves hBuilder .toFloat ha

theorem clampE_preserves {builder : Builder} (hBuilder : BuilderWellFormed builder)
    {value lo hi : Sig} (hv : SigIn builder value) (hl : SigIn builder lo)
    (hh : SigIn builder hi) :
    SigBuildResultWellFormed builder ((clampE value lo hi).run builder) :=
  by simpa [clampE] using clamp_preserves hBuilder hv hl hh

theorem selectE_preserves {builder : Builder} (hBuilder : BuilderWellFormed builder)
    {cond then_ else_ : Sig} (hc : SigIn builder cond)
    (ht : SigIn builder then_) (he : SigIn builder else_) :
    SigBuildResultWellFormed builder ((selectE cond then_ else_).run builder) :=
  by simpa [selectE] using select_preserves hBuilder hc ht he

theorem litI_preserves (mantissa : Int) : ProducesSig (litI mantissa) := by
  unfold ProducesSig
  simpa [litI] using bind_preserves (lit mantissa) (unary .toInt)
    (fun builder id => SigIn builder id)
    (fun builder id => SigIn builder id)
    (lit_preserves mantissa)
    (fun builder value hBuilder hValue =>
      unary_preserves hBuilder .toInt hValue)

theorem sumLeftTail_preserves {builder : Builder}
    (hBuilder : BuilderWellFormed builder) {items : List Sig} {acc : Sig}
    (hItems : ∀ item ∈ items, SigIn builder item)
    (hAcc : SigIn builder acc) :
    SigBuildResultWellFormed builder ((sumLeftTail items acc).run builder) := by
  induction items generalizing builder acc with
  | nil => exact ⟨hBuilder, BuilderExtends.refl builder, hAcc⟩
  | cons item rest ih =>
    rw [show sumLeftTail (item :: rest) acc =
      add acc item >>= fun next => sumLeftTail rest next from rfl]
    rw [StateT.run_bind]
    have hAdd := add_preserves hBuilder hAcc (hItems item (by simp))
    generalize hRunAdd : (add acc item).run builder = runAdd at hAdd ⊢
    cases runAdd with
    | error message => trivial
    | ok pair =>
      rcases pair with ⟨next, middle⟩
      change SigBuildResultWellFormed builder ((sumLeftTail rest next).run middle)
      have hRestItems : ∀ query ∈ rest, SigIn middle query := by
        intro query hQuery
        exact hAdd.2.1.sigIn (hItems query (by simp [hQuery]))
      have hTail := ih hAdd.1 hRestItems hAdd.2.2
      generalize hRunTail : (sumLeftTail rest next).run middle = runTail
        at hTail ⊢
      cases runTail with
      | error message => trivial
      | ok pair =>
        rcases pair with ⟨result, after⟩
        exact ⟨hTail.1, BuilderExtends.trans hAdd.2.1 hTail.2.1,
          hTail.2.2⟩

theorem sumLeft_preserves {builder : Builder} (hBuilder : BuilderWellFormed builder)
    {items : Array Sig} (hItems : SigsIn builder items) :
    SigBuildResultWellFormed builder ((sumLeft items).run builder) := by
  unfold sumLeft
  generalize hList : items.toList = list
  cases list with
  | nil => exact (lit_preserves 0) builder hBuilder
  | cons first rest =>
    apply sumLeftTail_preserves hBuilder
    · intro item hItem
      apply hItems item
      have : item ∈ items.toList := by rw [hList]; simp [hItem]
      simpa using this
    · apply hItems first
      have : first ∈ items.toList := by rw [hList]; simp
      simpa using this

theorem litF_preserves (value : Float) : ProducesSig (litF value) := by
  simp only [litF]
  split <;> split
  · exact lit_preserves 0
  · exact lit_preserves _ 12
  · exact lit_preserves 0
  · exact lit_preserves _ 12

-- Declaration construction --------------------------------------------------

/-- Every signal captured by a proposed instance declaration is owned. -/
def AInstWellFormed (builder : Builder) (decl : AInst) : Prop :=
  ∀ input ∈ decl.inputs, SigIn builder input.value

theorem declareInst_run (builder : Builder) (decl : AInst) :
    (declareInst decl).run builder =
      .ok (⟨builder.decls.size⟩,
        { builder with decls := builder.decls.push decl }) := by
  rfl

theorem declareInst_preserves {builder : Builder} (hBuilder : BuilderWellFormed builder)
    {decl : AInst} (hDecl : AInstWellFormed builder decl) :
    match (declareInst decl).run builder with
    | .error _ => True
    | .ok (idx, after) => BuilderWellFormed after ∧
      BuilderExtends builder after ∧ idx.idx < after.decls.size := by
  rw [declareInst_run]
  refine ⟨⟨hBuilder.arena, ?_⟩, ?_, by simp⟩
  · intro query hQuery input hInput
    rw [Array.mem_push] at hQuery
    rcases hQuery with hOld | rfl
    · exact hBuilder.decls query hOld input hInput
    · exact hDecl input hInput
  · constructor
    · exact Extends.refl builder.exprs
    · intro i oldDecl hOld
      have hi := (Array.getElem?_eq_some_iff.mp hOld).1
      rw [Array.getElem?_push_lt hi]
      simpa [Array.getElem?_eq_getElem hi] using hOld

theorem inst_preserves {builder : Builder} (hBuilder : BuilderWellFormed builder)
    (name programName : String) (inputs : Array AInput := #[])
    (hInputs : ∀ input ∈ inputs, SigIn builder input.value) :
    match (inst name programName inputs).run builder with
    | .error _ => True
    | .ok (idx, after) => BuilderWellFormed after ∧
      BuilderExtends builder after ∧ idx.idx < after.decls.size := by
  simpa [inst] using declareInst_preserves (decl := { name, programName, inputs })
    hBuilder (by exact hInputs)

-- Certified assembly ---------------------------------------------------------

/-- Expression-root obligations contributed directly by an ordinary body and
    by caller-supplied declarations. Builder-created declarations are already
    covered by `BuilderWellFormed`. -/
def ProgramBodyWellFormed (builder : Builder) (body : ProgramBody)
    (extraDecls : Array BodyDecl := #[]) : Prop :=
  (∀ input ∈ body.inputs, ∀ id ∈ input.defaultSig, SigIn builder id) ∧
  (∀ assign ∈ body.assigns, SigIn builder assign.2) ∧
  (∀ decl ∈ extraDecls, match decl with
    | .inst _ _ inputs => ∀ input ∈ inputs, SigIn builder input.value
    | _ => True)

/-- The analogous root obligations for a complete body. -/
def CompleteProgramBodyWellFormed (builder : Builder)
    (body : CompleteProgramBody) (extraDecls : Array BodyDecl := #[]) : Prop :=
  (∀ input ∈ body.inputs, ∀ id ∈ input.defaultSig, SigIn builder id) ∧
  (∀ assign ∈ body.assigns, SigIn builder assign.2) ∧
  (∀ decl ∈ extraDecls, match decl with
    | .inst _ _ inputs => ∀ input ∈ inputs, SigIn builder input.value
    | _ => True)

/-- Pure program value published by ordinary assembly after a successful run. -/
def assembledProgram (name : String) (outputs : Array OutputDecl)
    (registry : Array (String × ProgramIdx)) (builder : Builder)
    (body : ProgramBody) (extraDecls : Array BodyDecl := #[]) : Program :=
  { name
    inputs := body.inputs.map fun decl =>
      { name := decl.name, type? := decl.type?, default? := decl.defaultSig }
    outputs
    decls := (builder.decls.map fun decl =>
      .inst decl.name decl.programName (decl.inputs.map fun input =>
        { port := input.port, value := input.value })) ++ extraDecls
    assigns := body.assigns.map fun (target, expr) => { target, expr }
    registry }

/-- Pure program value published by complete assembly after a successful run. -/
def assembledCompleteProgram (name : String)
    (registry : Array (String × ProgramIdx)) (builder : Builder)
    (body : CompleteProgramBody) (extraDecls : Array BodyDecl := #[]) : Program :=
  { name
    inputs := body.inputs.map fun decl =>
      { name := decl.name, type? := decl.type?, default? := decl.defaultSig }
    outputs := body.outputs
    decls := (builder.decls.map fun decl =>
      .inst decl.name decl.programName (decl.inputs.map fun input =>
        { port := input.port, value := input.value })) ++ extraDecls
    assigns := body.assigns.map fun (target, expr) => { target, expr }
    registry }

theorem assembledProgram_exprRefs {builder : Builder}
    (hBuilder : BuilderWellFormed builder) {name : String}
    {outputs : Array OutputDecl} {registry : Array (String × ProgramIdx)}
    {body : ProgramBody} {extraDecls : Array BodyDecl}
    (hBody : ProgramBodyWellFormed builder body extraDecls) :
    ProgramExprRefsIn builder.exprs
      (assembledProgram name outputs registry builder body extraDecls) := by
  unfold ProgramExprRefsIn assembledProgram
  constructor
  · intro input hInput id hId
    obtain ⟨source, hSource, rfl⟩ := Array.mem_map.mp hInput
    exact hBody.1 source hSource id hId
  constructor
  · intro decl hDecl
    rw [Array.mem_append] at hDecl
    rcases hDecl with hGenerated | hExtra
    · obtain ⟨source, hSource, rfl⟩ := Array.mem_map.mp hGenerated
      intro input hInput
      obtain ⟨authored, hAuthored, rfl⟩ := Array.mem_map.mp hInput
      exact hBuilder.decls source hSource authored hAuthored
    · cases decl with
      | param name value => trivial
      | prog name target => trivial
      | inst name typeKey inputs => exact hBody.2.2 (.inst name typeKey inputs) hExtra
  · intro assign hAssign
    obtain ⟨source, hSource, rfl⟩ := Array.mem_map.mp hAssign
    exact hBody.2.1 source hSource

theorem assembledCompleteProgram_exprRefs {builder : Builder}
    (hBuilder : BuilderWellFormed builder) {name : String}
    {registry : Array (String × ProgramIdx)} {body : CompleteProgramBody}
    {extraDecls : Array BodyDecl}
    (hBody : CompleteProgramBodyWellFormed builder body extraDecls) :
    ProgramExprRefsIn builder.exprs
      (assembledCompleteProgram name registry builder body extraDecls) := by
  unfold ProgramExprRefsIn assembledCompleteProgram
  constructor
  · intro input hInput id hId
    obtain ⟨source, hSource, rfl⟩ := Array.mem_map.mp hInput
    exact hBody.1 source hSource id hId
  constructor
  · intro decl hDecl
    rw [Array.mem_append] at hDecl
    rcases hDecl with hGenerated | hExtra
    · obtain ⟨source, hSource, rfl⟩ := Array.mem_map.mp hGenerated
      intro input hInput
      obtain ⟨authored, hAuthored, rfl⟩ := Array.mem_map.mp hInput
      exact hBuilder.decls source hSource authored hAuthored
    · cases decl with
      | param name value => trivial
      | prog name target => trivial
      | inst name typeKey inputs =>
        exact hBody.2.2 (.inst name typeKey inputs) hExtra
  · intro assign hAssign
    obtain ⟨source, hSource, rfl⟩ := Array.mem_map.mp hAssign
    exact hBody.2.1 source hSource

theorem programWellFormed_push {arena : ExprArena} {programs : Array Program}
    {program : Program} (hArena : ArenaWellFormed arena)
    (hPool : progPoolWf (programs.push program) = true)
    (hExprs : ProgramExprRefsIn arena program)
    (hIndices : ProgramIndicesWellFormed arena (programs.push program) program)
    (hRegistry : ProgramRegistryWellFormed (programs.push program)
      programs.size program) :
    ProgramWellFormed arena (programs.push program) programs.size := by
  have hLookup : (programs.push program)[programs.size]? = some program := by simp
  constructor
  · exact ⟨program, hLookup⟩
  · exact hArena
  · exact hPool
  · intro query hQuery
    have hEq : query = program := by
      rw [hLookup] at hQuery
      exact Option.some.inj hQuery.symm
    subst query
    exact hExprs
  · intro query hQuery
    have hEq : query = program := by
      rw [hLookup] at hQuery
      exact Option.some.inj hQuery.symm
    subst query
    exact hIndices
  · intro query hQuery
    have hEq : query = program := by
      rw [hLookup] at hQuery
      exact Option.some.inj hQuery.symm
    subst query
    exact hRegistry

theorem assemble_run_of_build {arena : Arena} {name : String}
    {outputs : Array OutputDecl} {registry : Array (String × ProgramIdx)}
    {build : BuildM ProgramBody} {extraDecls : Array BodyDecl}
    {body : ProgramBody} {final : Builder}
    (hRun : build.run { exprs := arena.exprs } = .ok (body, final)) :
    assemble arena name outputs registry build extraDecls =
      .ok ({ arena with
        programs := arena.programs.push
          (assembledProgram name outputs registry final body extraDecls)
        exprs := final.exprs }, ⟨arena.programs.size⟩) := by
  simp only [assemble, hRun]
  rfl

/-- Certified ordinary assembly. Pool/index/registry clauses remain explicit
    because production `assemble` does not run those validators. -/
theorem assemble_of_certified_build {arena : Arena} {name : String}
    {outputs : Array OutputDecl} {registry : Array (String × ProgramIdx)}
    {build : BuildM ProgramBody} {extraDecls : Array BodyDecl}
    {body : ProgramBody} {final : Builder}
    (hArena : ArenaWellFormed arena.exprs)
    (hRun : build.run { exprs := arena.exprs } = .ok (body, final))
    (hBuild : PreservesBuilderWF build)
    (hBody : ProgramBodyWellFormed final body extraDecls)
    (hPool : progPoolWf (arena.programs.push
      (assembledProgram name outputs registry final body extraDecls)) = true)
    (hIndices : ProgramIndicesWellFormed final.exprs
      (arena.programs.push (assembledProgram name outputs registry final body extraDecls))
      (assembledProgram name outputs registry final body extraDecls))
    (hRegistry : ProgramRegistryWellFormed
      (arena.programs.push (assembledProgram name outputs registry final body extraDecls))
      arena.programs.size
      (assembledProgram name outputs registry final body extraDecls)) :
    match assemble arena name outputs registry build extraDecls with
    | .error _ => False
    | .ok (resultArena, resultRoot) =>
      ProgramWellFormed resultArena.exprs resultArena.programs resultRoot.idx := by
  rw [assemble_run_of_build hRun]
  have hInitial : BuilderWellFormed ({ exprs := arena.exprs } : Builder) := by
    exact ⟨hArena, by simp [BuilderDeclsWellFormed]⟩
  have hFinal := hBuild { exprs := arena.exprs } hInitial
  rw [hRun] at hFinal
  simpa using programWellFormed_push hFinal.1.arena hPool
    (assembledProgram_exprRefs hFinal.1 hBody) hIndices hRegistry

theorem assembleComplete_run_of_build {α : Type} {arena : Arena} {name : String}
    {registry : Array (String × ProgramIdx)}
    {build : BuildM (CompleteProgramBody × α)} {extraDecls : Array BodyDecl}
    {body : CompleteProgramBody} {value : α} {final : Builder}
    (hRun : build.run { exprs := arena.exprs } = .ok ((body, value), final)) :
    assembleCompleteWithResult arena name registry build extraDecls =
      .ok ({ arena with
        programs := arena.programs.push
          (assembledCompleteProgram name registry final body extraDecls)
        exprs := final.exprs }, ⟨arena.programs.size⟩, value) := by
  simp only [assembleCompleteWithResult, hRun]
  rfl

/-- Certified complete assembly, with the same explicit unvalidated clauses. -/
theorem assembleComplete_of_certified_build {α : Type} {arena : Arena}
    {name : String} {registry : Array (String × ProgramIdx)}
    {build : BuildM (CompleteProgramBody × α)} {extraDecls : Array BodyDecl}
    {body : CompleteProgramBody} {value : α} {final : Builder}
    (hArena : ArenaWellFormed arena.exprs)
    (hRun : build.run { exprs := arena.exprs } = .ok ((body, value), final))
    (hBuild : PreservesBuilderWF build)
    (hBody : CompleteProgramBodyWellFormed final body extraDecls)
    (hPool : progPoolWf (arena.programs.push
      (assembledCompleteProgram name registry final body extraDecls)) = true)
    (hIndices : ProgramIndicesWellFormed final.exprs
      (arena.programs.push (assembledCompleteProgram name registry final body extraDecls))
      (assembledCompleteProgram name registry final body extraDecls))
    (hRegistry : ProgramRegistryWellFormed
      (arena.programs.push (assembledCompleteProgram name registry final body extraDecls))
      arena.programs.size
      (assembledCompleteProgram name registry final body extraDecls)) :
    match assembleCompleteWithResult arena name registry build extraDecls with
    | .error _ => False
    | .ok (resultArena, resultRoot, _) =>
      ProgramWellFormed resultArena.exprs resultArena.programs resultRoot.idx := by
  rw [assembleComplete_run_of_build hRun]
  have hInitial : BuilderWellFormed ({ exprs := arena.exprs } : Builder) := by
    exact ⟨hArena, by simp [BuilderDeclsWellFormed]⟩
  have hFinal := hBuild { exprs := arena.exprs } hInitial
  rw [hRun] at hFinal
  simpa using programWellFormed_push hFinal.1.arena hPool
    (assembledCompleteProgram_exprRefs hFinal.1 hBody) hIndices hRegistry

/-- Observable failure atomicity: failure publishes no arena/program pair. -/
theorem assemble_failure_no_result {arena : Arena} {name : String}
    {outputs : Array OutputDecl} {registry : Array (String × ProgramIdx)}
    {build : BuildM ProgramBody} {extraDecls : Array BodyDecl} {message : String}
    (hFailure : assemble arena name outputs registry build extraDecls = .error message) :
    ¬ ∃ result, assemble arena name outputs registry build extraDecls = .ok result := by
  rintro ⟨result, hResult⟩
  rw [hFailure] at hResult
  contradiction

theorem assembleComplete_failure_no_result {α : Type} {arena : Arena}
    {name : String} {registry : Array (String × ProgramIdx)}
    {build : BuildM (CompleteProgramBody × α)} {extraDecls : Array BodyDecl}
    {message : String}
    (hFailure : assembleCompleteWithResult arena name registry build extraDecls =
      .error message) :
    ¬ ∃ result,
      assembleCompleteWithResult arena name registry build extraDecls = .ok result := by
  rintro ⟨result, hResult⟩
  rw [hFailure] at hResult
  contradiction

end Tropical.EmitArrow
