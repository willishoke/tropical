import Tropical.Semantics.Arena

/-!
# Denotation of the production expression DAG

`denoteExpr` interprets an `ExprId` in a frozen, child-descending production
arena.  Its recursion follows the exact `ENode` edges and therefore terminates
by decreasing node index.  The two `ENode` constructors outside the `Sig`
vocabulary remain explicit refusals.
-/

namespace Tropical.Semantics

open Tropical.Ir

/-- Interpret one already-dereferenced node, using `recur` for its proven
    children.  The membership argument exposes exactly the decrease needed by
    the total arena evaluator. -/
def denoteNode (alg : Algebra α) (env : SigEnv α) (node : ENode)
    (recur : (childEnv : SigEnv α) → (child : ExprId) →
      child ∈ node.children → Result α) : Result α :=
  match node with
  | .num number => alg.literal number
  | .bool _ => refusal "bool" "boolean nodes are outside the Sig semantics"
  | .arr items =>
    match sequence
        (items.attach.map fun ⟨child, hMem⟩ =>
          recur env child (by simpa [ENode.children] using hMem)) with
    | .error error => .error error
    | .ok values => .ok (.array values)
  | .binary tag lhs rhs =>
    applyBinary alg tag
      (recur env lhs (by simp [ENode.children]))
      (recur env rhs (by simp [ENode.children]))
  | .unary tag arg =>
    match recur env arg (by simp [ENode.children]) with
    | .error error => .error error
    | .ok value => alg.unary tag value
  | .clamp value lo hi =>
    applyTernary alg.clamp
      (recur env value (by simp [ENode.children]))
      (recur env lo (by simp [ENode.children]))
      (recur env hi (by simp [ENode.children]))
  | .select cond then_ else_ =>
    applyTernary alg.select
      (recur env cond (by simp [ENode.children]))
      (recur env then_ (by simp [ENode.children]))
      (recur env else_ (by simp [ENode.children]))
  | .arraySet _ _ _ =>
    refusal "arraySet" "array mutation is outside the Sig semantics"
  | .index array index =>
    match recur env array (by simp [ENode.children]) with
    | .error error => .error error
    | .ok arrayValue =>
      match recur env index (by simp [ENode.children]) with
      | .error error => .error error
      | .ok indexValue => alg.index arrayValue indexValue
  | .inputRef idx => lookupValue "inputRef" env.inputs idx.idx
  | .paramRef idx => lookupValue "paramRef" env.params idx.idx
  | .nestedOut instanceIdx outputIdx =>
    lookupNested env instanceIdx.idx outputIdx.idx
  | .sampleRate => .ok env.sampleRate
  | .sampleIndex => .ok env.sampleIndex
  | .loopIdx binderId =>
    match env.loops binderId with
    | some value => .ok value
    | none => refusal "loopIdx" s!"binder {binderId} is not open"
  | .bankSum capacity tables body dynCount? binderId =>
    match sequence
        (tables.attach.map fun ⟨table, hMem⟩ =>
          recur env table (by simp [ENode.children, hMem])) with
    | .error error => .error error
    | .ok _ =>
      let dynResult := match dynCount? with
        | none => none
        | some count => some (recur env count (by simp [ENode.children]))
      match bankTrips alg capacity dynResult with
      | .error error => .error error
      | .ok trips =>
        match alg.zero with
        | .error error => .error error
        | .ok zero =>
          let step : Value α → Nat → Outcome (Value α) := fun acc index => do
            let loopValue ← alg.loopIndex index
            let contribution ←
              recur (env.bindLoop binderId loopValue) body
                (by simp [ENode.children])
            alg.binary .add acc contribution
          (List.foldlM step zero (List.range trips) : Outcome (Value α))
  | .routedSum capacity outputCount routes tables values dynCount? binderId =>
    denoteRoutedSum alg capacity outputCount values.size routes
      (tables.attach.map fun ⟨table, hMem⟩ =>
        recur env table (by simp [ENode.children, hMem]))
      (fun loopValue => values.attach.map fun ⟨value, hMem⟩ =>
        recur (env.bindLoop binderId loopValue) value
          (by simp [ENode.children, hMem]))
      (match dynCount? with
        | none => none
        | some count => some (recur env count (by simp [ENode.children])))

/-- Denotation of a production expression DAG rooted at `id`. -/
def denoteExpr (alg : Algebra α) (env : SigEnv α) (arena : ExprArena)
    (hArena : ArenaWellFormed arena) (id : ExprId) : Result α :=
  match _hd : arena.deref id with
  | none => refusal "expr" s!"dangling ExprId {id.idx}"
  | some node =>
    denoteNode alg env node fun childEnv child _ =>
      denoteExpr alg childEnv arena hArena child
termination_by id.idx
decreasing_by
  all_goals
    exact hArena.childrenDescend ‹arena.deref _ = some _› _ ‹_ ∈ node.children›

/-- Unfold `denoteExpr` after a successful dereference without exposing its
    dependent match to downstream proofs. -/
theorem denoteExpr_of_deref (alg : Algebra α) (env : SigEnv α)
    (arena : ExprArena) (hArena : ArenaWellFormed arena)
    {id : ExprId} {node : ENode} (hDeref : arena.deref id = some node) :
    denoteExpr alg env arena hArena id =
      denoteNode alg env node (fun childEnv child _ =>
        denoteExpr alg childEnv arena hArena child) := by
  rw [denoteExpr]
  split
  next hNone => simp_all
  next actual hActual =>
    have : actual = node := Option.some.inj (hActual.symm.trans hDeref)
    subst actual
    rfl

/-- Appending nodes does not change the meaning of any addressable old root. -/
theorem denoteExpr_extends {before after : ExprArena}
    (hBefore : ArenaWellFormed before) (hAfter : ArenaWellFormed after)
    (hExtends : Extends before after) (alg : Algebra α) (env : SigEnv α)
    {id : ExprId} {node : ENode} (hDeref : before.deref id = some node) :
    denoteExpr alg env before hBefore id =
      denoteExpr alg env after hAfter id := by
  have hChild (child : ExprId) (hMem : child ∈ node.children) :
      denoteExpr alg env before hBefore child =
        denoteExpr alg env after hAfter child := by
    have hChildLt := hBefore.childrenDescend hDeref child hMem
    obtain ⟨childNode, hChildDeref⟩ :=
      deref_of_index_lt
        (Nat.lt_trans hChildLt (deref_index_lt hDeref))
    exact denoteExpr_extends hBefore hAfter hExtends alg env hChildDeref
  rw [denoteExpr_of_deref alg env before hBefore hDeref,
    denoteExpr_of_deref alg env after hAfter (hExtends hDeref)]
  cases node with
  | num | bool | inputRef | paramRef | nestedOut | sampleRate
  | sampleIndex | loopIdx => simp [denoteNode]
  | arr items =>
    simp only [denoteNode]
    have hItems :
        items.attach.map
            (fun item => denoteExpr alg env before hBefore item.1) =
          items.attach.map
            (fun item => denoteExpr alg env after hAfter item.1) := by
      apply Array.ext
      · simp
      · intro i hiBefore hiAfter
        simp only [Array.getElem_map, Array.getElem_attach]
        apply hChild
        simp [ENode.children]
    rw [hItems]
  | binary tag lhs rhs =>
    simp only [denoteNode]
    rw [hChild lhs (by simp [ENode.children]),
      hChild rhs (by simp [ENode.children])]
  | unary tag arg =>
    simp only [denoteNode]
    rw [hChild arg (by simp [ENode.children])]
  | clamp value lo hi =>
    simp only [denoteNode]
    rw [hChild value (by simp [ENode.children]),
      hChild lo (by simp [ENode.children]),
      hChild hi (by simp [ENode.children])]
  | select cond then_ else_ =>
    simp only [denoteNode]
    rw [hChild cond (by simp [ENode.children]),
      hChild then_ (by simp [ENode.children]),
      hChild else_ (by simp [ENode.children])]
  | arraySet array index value =>
    simp [denoteNode]
  | index array index =>
    simp only [denoteNode]
    rw [hChild array (by simp [ENode.children]),
      hChild index (by simp [ENode.children])]
  | bankSum capacity tables body dynCount? binderId =>
    simp only [denoteNode]
    have hTableArray :
        tables.attach.map
            (fun item => denoteExpr alg env before hBefore item.1) =
          tables.attach.map
            (fun item => denoteExpr alg env after hAfter item.1) := by
      apply Array.ext
      · simp
      · intro i hiBefore hiAfter
        simp only [Array.getElem_map, Array.getElem_attach]
        apply hChild
        simp [ENode.children]
    have hBody (loopValue : Value α) :
        denoteExpr alg (env.bindLoop binderId loopValue)
            before hBefore body =
          denoteExpr alg (env.bindLoop binderId loopValue)
            after hAfter body := by
      obtain ⟨bodyNode, hBodyDeref⟩ :=
        deref_of_index_lt
          (Nat.lt_trans
            (hBefore.childrenDescend hDeref body
              (by simp [ENode.children]))
            (deref_index_lt hDeref))
      exact denoteExpr_extends hBefore hAfter hExtends alg
        (env.bindLoop binderId loopValue) hBodyDeref
    rw [hTableArray]
    cases dynCount? with
    | none =>
      simp only [hBody]
    | some count =>
      simp only [hChild count (by simp [ENode.children]), hBody]
  | routedSum capacity outputCount routes tables values dynCount? binderId =>
    simp only [denoteNode]
    have hTableArray :
        tables.attach.map
            (fun item => denoteExpr alg env before hBefore item.1) =
          tables.attach.map
            (fun item => denoteExpr alg env after hAfter item.1) := by
      apply Array.ext
      · simp
      · intro i hiBefore hiAfter
        simp only [Array.getElem_map, Array.getElem_attach]
        apply hChild
        simp [ENode.children]
    have hValueArray (loopValue : Value α) :
        values.attach.map
            (fun item => denoteExpr alg (env.bindLoop binderId loopValue)
              before hBefore item.1) =
          values.attach.map
            (fun item => denoteExpr alg (env.bindLoop binderId loopValue)
              after hAfter item.1) := by
      apply Array.ext
      · simp
      · intro i hiBefore hiAfter
        simp only [Array.getElem_map, Array.getElem_attach]
        have hi : i < values.size := by simpa using hiBefore
        let value := values[i]
        have hMem : value ∈ values := Array.getElem_mem hi
        obtain ⟨valueNode, hValueDeref⟩ :=
          deref_of_index_lt
            (Nat.lt_trans
              (hBefore.childrenDescend hDeref value
                (by simp [ENode.children, hMem]))
              (deref_index_lt hDeref))
        exact denoteExpr_extends hBefore hAfter hExtends alg
          (env.bindLoop binderId loopValue) hValueDeref
    rw [hTableArray]
    cases dynCount? with
    | none => simp only [hValueArray]
    | some count =>
      simp only [hChild count (by simp [ENode.children]), hValueArray]
termination_by id.idx
decreasing_by
  all_goals
    apply hBefore.childrenDescend hDeref
    simp_all [ENode.children]

theorem attach_map_value (xs : Array β) (f : β → γ) :
    xs.attach.map (fun item => f item.1) = xs.map f := by
  apply Array.ext
  · simp
  · intro i hLeft hRight
    simp only [Array.getElem_map, Array.getElem_attach]

end Tropical.Semantics
