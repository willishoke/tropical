import Std.Data.HashMap
import Tropical.Ir.Nodes
import Tropical.Ir.CoreArena

/-!
# ExprArena — hash-consed (DAG) form of the full resolved `Expr`

The native-DAG representation for the whole strata pipeline (issue #190). Where
`CoreArena`/`CNode` cover the 14-constructor post-strata subset (emit's
boundary), `ExprArena`/`ENode` cover *all* of `Expr` — combinators (`fold`,
`generate`, …), `letIn`, `tag`/`match` and their binders — so the arena can be
live from the elaborator through `inlineInstances` (where the bloat would
otherwise form) and on to emit.

A node is **flat**: its children are `ExprId`s, so it is O(1) to hash and
compare, and interning at construction makes two equal subtrees one node. When
`inlineInstances` substitutes a wired expression at many use sites, it stores the
same `ExprId` — the modulated-clock duplication never materializes.

Binders carry their `BinderIdx`, so two otherwise-identical combinators with
*different* binder indices stay distinct (alpha-correct); within one program,
identical subtrees carry identical binder indices and so share.
-/

namespace Tropical.Ir

open Lean (JsonNumber)

/-- A `LetBinder` with an id-valued body. -/
structure ELetBinder where
  binder : Binder
  value : ExprId
deriving BEq, Repr, Inhabited

/-- A `TagPayload` with an id-valued field. -/
structure ETagPayload where
  field : Nat
  value : ExprId
deriving BEq, Repr, Inhabited

/-- A `MatchArm` with an id-valued body. -/
structure EMatchArm where
  variant : Nat
  binders : Array Binder
  body : ExprId
deriving BEq, Repr, Inhabited

/-- A resolved expression node with children referenced by `ExprId`.
    Mirrors `Expr`'s constructors; flat (no inlined subtrees). -/
inductive ENode where
  | num (n : JsonNumber)
  | bool (b : Bool)
  | arr (items : Array ExprId)
  | binary (tag : BinaryOpTag) (lhs rhs : ExprId)
  | unary (tag : UnaryOpTag) (arg : ExprId)
  | clamp (value lo hi : ExprId)
  | select (cond then_ else_ : ExprId)
  | arraySet (arr idx value : ExprId)
  | index (arr idx : ExprId)
  | zeros (count : ExprId)
  | inputRef (idx : InputIdx)
  | paramRef (idx : ParamIdx)
  | typeParamRef (idx : TypeParamIdx)
  | bindingRef (idx : BinderIdx)
  | nestedOut (instance_ : InstanceIdx) (output : OutputIdx)
  | sampleRate
  | sampleIndex
  | fold (over init : ExprId) (acc elem : Binder) (body : ExprId)
  | scan (over init : ExprId) (acc elem : Binder) (body : ExprId)
  | generate (count : ExprId) (iter : Binder) (body : ExprId)
  | iterate (count init : ExprId) (iter : Binder) (body : ExprId)
  | chain (count init : ExprId) (iter : Binder) (body : ExprId)
  | map2 (over : ExprId) (elem : Binder) (body : ExprId)
  | zipWith (a b : ExprId) (x y : Binder) (body : ExprId)
  | letIn (binders : Array ELetBinder) (body : ExprId)
  | tag (def_ : TypeDefIdx) (variant : Nat) (payload : Array ETagPayload)
  | match_ (def_ : TypeDefIdx) (scrutinee : ExprId) (arms : Array EMatchArm)
deriving BEq, Repr, Inhabited

/-- O(1) structural hash — children are ids (no subtree recursion). Op tags and
    binders fold through hashable components (`.wire`, `.idx`, names). -/
def enodeHash : ENode → UInt64
  | .num n          => mixHash 1 (hash n)
  | .bool b         => mixHash 2 (hash b)
  | .arr items      => mixHash 3 (hash (items.map (·.idx)))
  | .binary t a b   => mixHash (mixHash (mixHash 4 (hash t.wire)) (hash a.idx)) (hash b.idx)
  | .unary t a      => mixHash (mixHash 5 (hash t.wire)) (hash a.idx)
  | .clamp a b c    => mixHash (mixHash (mixHash 6 (hash a.idx)) (hash b.idx)) (hash c.idx)
  | .select a b c   => mixHash (mixHash (mixHash 7 (hash a.idx)) (hash b.idx)) (hash c.idx)
  | .arraySet a b c => mixHash (mixHash (mixHash 8 (hash a.idx)) (hash b.idx)) (hash c.idx)
  | .index a b      => mixHash (mixHash 9 (hash a.idx)) (hash b.idx)
  | .zeros c        => mixHash 10 (hash c.idx)
  | .inputRef i     => mixHash 11 (hash i.idx)
  | .paramRef i     => mixHash 12 (hash i.idx)
  | .typeParamRef i => mixHash 13 (hash i.idx)
  | .bindingRef i   => mixHash 14 (hash i.idx)
  | .nestedOut i o  => mixHash (mixHash 15 (hash i.idx)) (hash o.idx)
  | .sampleRate     => 16
  | .sampleIndex    => 17
  | .fold o i a e b => mixHash (mixHash (mixHash (mixHash (mixHash 18 (hash o.idx)) (hash i.idx)) (hash a.idx.idx)) (hash e.idx.idx)) (hash b.idx)
  | .scan o i a e b => mixHash (mixHash (mixHash (mixHash (mixHash 19 (hash o.idx)) (hash i.idx)) (hash a.idx.idx)) (hash e.idx.idx)) (hash b.idx)
  | .generate c i b => mixHash (mixHash (mixHash 20 (hash c.idx)) (hash i.idx.idx)) (hash b.idx)
  | .iterate c i it b => mixHash (mixHash (mixHash (mixHash 21 (hash c.idx)) (hash i.idx)) (hash it.idx.idx)) (hash b.idx)
  | .chain c i it b => mixHash (mixHash (mixHash (mixHash 22 (hash c.idx)) (hash i.idx)) (hash it.idx.idx)) (hash b.idx)
  | .map2 o e b     => mixHash (mixHash (mixHash 23 (hash o.idx)) (hash e.idx.idx)) (hash b.idx)
  | .zipWith a b x y bd => mixHash (mixHash (mixHash (mixHash (mixHash 24 (hash a.idx)) (hash b.idx)) (hash x.idx.idx)) (hash y.idx.idx)) (hash bd.idx)
  | .letIn bs b     => mixHash (mixHash 25 (hash (bs.map (fun lb => (lb.binder.idx.idx, lb.value.idx))))) (hash b.idx)
  | .tag d v p      => mixHash (mixHash (mixHash 26 (hash d.idx)) (hash v)) (hash (p.map (fun tp => (tp.field, tp.value.idx))))
  | .match_ d s arms => mixHash (mixHash (mixHash 27 (hash d.idx)) (hash s.idx)) (hash (arms.map (fun a => (a.variant, a.body.idx))))

instance : Hashable ENode := ⟨enodeHash⟩

/-- Interned full-Expr node store. -/
structure ExprArena where
  nodes : Array ENode := #[]
  dedup : Std.HashMap ENode ExprId := {}
deriving Inhabited

abbrev EArenaM := StateM ExprArena

def eintern (n : ENode) : EArenaM ExprId := do
  let a ← get
  match a.dedup.get? n with
  | some id => pure id
  | none =>
    let id : ExprId := ⟨a.nodes.size⟩
    set { a with nodes := a.nodes.push n, dedup := a.dedup.insert n id }
    pure id

def ExprArena.deref (a : ExprArena) (id : ExprId) : Option ENode :=
  a.nodes[id.idx]?

/-- Lower a tree `Expr` into the arena, interning bottom-up. -/
partial def toExprArena : Expr → EArenaM ExprId
  | .num n          => eintern (.num n)
  | .bool b         => eintern (.bool b)
  | .arr items      => do eintern (.arr (← items.mapM toExprArena))
  | .binary t a b   => do eintern (.binary t (← toExprArena a) (← toExprArena b))
  | .unary t a      => do eintern (.unary t (← toExprArena a))
  | .clamp a b c    => do eintern (.clamp (← toExprArena a) (← toExprArena b) (← toExprArena c))
  | .select a b c   => do eintern (.select (← toExprArena a) (← toExprArena b) (← toExprArena c))
  | .arraySet a b c => do eintern (.arraySet (← toExprArena a) (← toExprArena b) (← toExprArena c))
  | .index a b      => do eintern (.index (← toExprArena a) (← toExprArena b))
  | .zeros c        => do eintern (.zeros (← toExprArena c))
  | .inputRef i     => eintern (.inputRef i)
  | .paramRef i     => eintern (.paramRef i)
  | .typeParamRef i => eintern (.typeParamRef i)
  | .bindingRef i   => eintern (.bindingRef i)
  | .nestedOut i o  => eintern (.nestedOut i o)
  | .sampleRate     => eintern .sampleRate
  | .sampleIndex    => eintern .sampleIndex
  | .fold o i a e b => do eintern (.fold (← toExprArena o) (← toExprArena i) a e (← toExprArena b))
  | .scan o i a e b => do eintern (.scan (← toExprArena o) (← toExprArena i) a e (← toExprArena b))
  | .generate c it b => do eintern (.generate (← toExprArena c) it (← toExprArena b))
  | .iterate c i it b => do eintern (.iterate (← toExprArena c) (← toExprArena i) it (← toExprArena b))
  | .chain c i it b => do eintern (.chain (← toExprArena c) (← toExprArena i) it (← toExprArena b))
  | .map2 o e b     => do eintern (.map2 (← toExprArena o) e (← toExprArena b))
  | .zipWith a b x y bd => do eintern (.zipWith (← toExprArena a) (← toExprArena b) x y (← toExprArena bd))
  | .letIn bs b     => do
    let bs' ← bs.mapM fun lb => match lb with
      | .mk binder value => do pure ({ binder, value := ← toExprArena value } : ELetBinder)
    eintern (.letIn bs' (← toExprArena b))
  | .tag d v p      => do
    let p' ← p.mapM fun tp => match tp with
      | .mk field value => do pure ({ field, value := ← toExprArena value } : ETagPayload)
    eintern (.tag d v p')
  | .match_ d s arms => do
    let arms' ← arms.mapM fun arm => match arm with
      | .mk variant binders body => do pure ({ variant, binders, body := ← toExprArena body } : EMatchArm)
    eintern (.match_ d (← toExprArena s) arms')

/-- Rebuild a tree `Expr` from an id (round-trip witness for tests). -/
partial def ExprArena.toExpr (a : ExprArena) (id : ExprId) : Expr :=
  match a.deref id with
  | none => .num 0
  | some n => match n with
    | .num x          => .num x
    | .bool b         => .bool b
    | .arr items      => .arr (items.map a.toExpr)
    | .binary t l r   => .binary t (a.toExpr l) (a.toExpr r)
    | .unary t x      => .unary t (a.toExpr x)
    | .clamp x y z    => .clamp (a.toExpr x) (a.toExpr y) (a.toExpr z)
    | .select x y z   => .select (a.toExpr x) (a.toExpr y) (a.toExpr z)
    | .arraySet x y z => .arraySet (a.toExpr x) (a.toExpr y) (a.toExpr z)
    | .index x y      => .index (a.toExpr x) (a.toExpr y)
    | .zeros c        => .zeros (a.toExpr c)
    | .inputRef i     => .inputRef i
    | .paramRef i     => .paramRef i
    | .typeParamRef i => .typeParamRef i
    | .bindingRef i   => .bindingRef i
    | .nestedOut i o  => .nestedOut i o
    | .sampleRate     => .sampleRate
    | .sampleIndex    => .sampleIndex
    | .fold o i ac e b => .fold (a.toExpr o) (a.toExpr i) ac e (a.toExpr b)
    | .scan o i ac e b => .scan (a.toExpr o) (a.toExpr i) ac e (a.toExpr b)
    | .generate c it b => .generate (a.toExpr c) it (a.toExpr b)
    | .iterate c i it b => .iterate (a.toExpr c) (a.toExpr i) it (a.toExpr b)
    | .chain c i it b => .chain (a.toExpr c) (a.toExpr i) it (a.toExpr b)
    | .map2 o e b     => .map2 (a.toExpr o) e (a.toExpr b)
    | .zipWith x y u v b => .zipWith (a.toExpr x) (a.toExpr y) u v (a.toExpr b)
    | .letIn bs b     => .letIn (bs.map fun lb => .mk lb.binder (a.toExpr lb.value)) (a.toExpr b)
    | .tag d v p      => .tag d v (p.map fun tp => .mk tp.field (a.toExpr tp.value))
    | .match_ d s arms => .match_ d (a.toExpr s) (arms.map fun arm => .mk arm.variant arm.binders (a.toExpr arm.body))

-- (The post-strata DAG → `CoreArena` conversion is `EArena.toResolved` in
-- `Strata/EArena.lean` — a reachable-only walk from the root program. A
-- whole-arena fold like an earlier `toCoreArena` cannot be used here: `nodes`
-- is append-only and still holds intermediate combinator nodes that later
-- passes lowered away, which such a fold would falsely reject as "survived".)

-- ─────────────────────────────────────────────────────────────
-- EProgram — `Program` with id-valued expressions
-- ─────────────────────────────────────────────────────────────

/-- `InputDecl` with an id-valued default. -/
structure EInputDecl where
  name : String
  type? : Option PortType := none
  default? : Option ExprId := none
deriving Repr, Inhabited

/-- `InstanceInput` with an id-valued wire. -/
structure EInstanceInput where
  port : InputIdx
  value : ExprId
deriving Repr, Inhabited

/-- `BodyDecl` with id-valued instance inputs. -/
inductive EBodyDecl where
  | param (name : String) (value? : Option JsonNumber)
  | inst (name : String) (typeKey : String)
      (typeArgs : Array InstanceTypeArg) (inputs : Array EInstanceInput)
  | prog (name : String) (program : ProgramIdx)
deriving Repr, Inhabited

/-- `OutputAssign` with an id-valued expression. -/
structure EOutputAssign where
  target : OutputTarget
  expr : ExprId
deriving Repr, Inhabited

/-- `Program` with id-valued expressions: the strata pipeline's working form.
    `registry` still references sibling programs by `ProgramIdx` into the
    (tree) `Arena`; a pass converts each on demand with `toEProgram`. -/
structure EProgram where
  name : String
  typeParams : Array TypeParamPoolIdx := #[]
  inputs : Array EInputDecl := #[]
  outputs : Array OutputDecl := #[]
  typeDefs : Array TypeDefIdx := #[]
  decls : Array EBodyDecl := #[]
  assigns : Array EOutputAssign := #[]
  binderCount : Nat := 0
  registry : Array (String × ProgramIdx) := #[]
deriving Repr, Inhabited

/-- Lower a tree `Program` into id form, interning every expression. -/
def toEProgram (p : Program) : EArenaM EProgram := do
  let inputs ← p.inputs.mapM fun d => do
    pure ({ name := d.name, type? := d.type?,
            default? := ← d.default?.mapM toExprArena } : EInputDecl)
  let decls ← p.decls.mapM fun d => do
    match d with
    | .param n v => pure (.param n v)
    | .prog n pi => pure (.prog n pi)
    | .inst n k ta ins =>
      let ins' ← ins.mapM fun i => do
        pure ({ port := i.port, value := ← toExprArena i.value } : EInstanceInput)
      pure (.inst n k ta ins')
  let assigns ← p.assigns.mapM fun a => do
    pure ({ target := a.target, expr := ← toExprArena a.expr } : EOutputAssign)
  pure { name := p.name, typeParams := p.typeParams, inputs, outputs := p.outputs,
         typeDefs := p.typeDefs, decls, assigns, binderCount := p.binderCount,
         registry := p.registry }

end Tropical.Ir
