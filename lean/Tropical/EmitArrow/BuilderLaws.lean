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

private theorem childrenIn_tileArray {builder : Builder} {items : Array Sig}
    (h : SigsIn builder items) :
    ENodeChildrenIn builder.exprs (.tileArray items) := by
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

theorem tilePhase_preserves : ProducesSig tilePhase := by
  simpa [tilePhase] using internSig_preserves .tilePhase (by
    intro builder hBuilder
    simp [ENodeChildrenIn, ENode.children])

theorem tileSampleIndex_preserves : ProducesSig tileSampleIndex := by
  simpa [tileSampleIndex] using internSig_preserves .tileSampleIndex (by
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

theorem tileArray_preserves {builder : Builder}
    (hBuilder : BuilderWellFormed builder) {items : Array Sig}
    (hItems : SigsIn builder items) :
    match (tileArray items).run builder with
    | .error _ => True
    | .ok (id, after) =>
      BuilderWellFormed after ∧ BuilderExtends builder after ∧ SigIn after id := by
  simpa [tileArray] using internSig_preserves_of_children
    (node := .tileArray items) hBuilder (childrenIn_tileArray hItems)

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

end Tropical.EmitArrow
