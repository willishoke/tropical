import Tropical.Ir.Nodes

/-!
# Core — the post-strata program shape (Phase 5 stage 6)

"The smallest sub-IR sufficient for any per-sample evaluator," made a
type instead of prose. `CoreProgram` is the post-strata program: scalar,
monomorphic, acyclic, combinator-free. Its **expression leaves are
`ExprId`s** into a shared `CoreArena` (Phase B) — the tree twin `CoreExpr`
is gone; there is one post-strata expression representation, the hash-consed
DAG.

Nesting is NOT dropped: the fractal session path keeps InstanceDecls
as kernel boundaries (the flat path simply has none). Lifted
ProgramDecls and never-instantiated registry entries are inert type
bindings no evaluator reaches — they pass through as names.

The executable downcasts that produce a `CoreProgram` — `EArena.toResolved`
(strata exit) and `checkResolvedArena` (elaborated-tree boundary) — validate
the post-strata invariant list (rejecting any strata-dropped constructor that
survived) as they intern each leaf. They follow exactly the evaluator-reachable
graph: body exprs, then recursively the program behind each instance's typeKey.
The port-type resolvers below are shared by both.
-/

namespace Tropical.Ir.Core

open Lean (JsonNumber)
open Tropical.Ir

-- ─────────────────────────────────────────────────────────────
-- Resolved port types (Phase 6 enrichment)
--
-- Emit/partition need port shapes (scalar counts, array sizes,
-- element kinds) and reg scalar types. The TS emit reads these off
-- the IR's PortType objects with arena-free field access (`alias.base`
-- etc.); the downcast resolves the pool references once so Core stays
-- a self-contained tree.
-- ─────────────────────────────────────────────────────────────

/-- Array element: scalar kind, or an alias carrying its base. The two
    survive distinctly because TS consumers disagree on the collapse —
    `inputPortTypes` uses `element.base`, `expandPortToSlots` uses
    `'float'` for alias elements. -/
inductive CoreElem where
  | scalar (k : ScalarKind)
  | aliased (base : ScalarKind)
deriving Repr, Inhabited

/-- A shape dimension post-strata: a literal, or an unresolved type
    param (specialize didn't run — emit reproduces the TS error). -/
inductive CoreShapeDim where
  | lit (n : JsonNumber)
  | unresolved
deriving Repr, Inhabited

inductive CorePortType where
  | scalar (k : ScalarKind)
  /-- Alias port. `base` is the alias's underlying scalar; slot
      expansion treats the port as one opaque float slot, type
      inference uses the base. -/
  | alias (base : ScalarKind)
  | array (element : CoreElem) (shape : Array CoreShapeDim)
deriving Repr, Inhabited

structure CoreInstanceInput where
  port : InputIdx
  value : ExprId
deriving Repr, Inhabited

inductive CoreBodyDecl where
  | param (name : String) (value? : Option JsonNumber)
  /-- Fractal kernel boundary; `typeKey` resolves in the enclosing
      `CoreProgram`'s registry. typeArgs survive as inert metadata on
      the fractal path (targets are resolved pre-strata in production). -/
  | inst (name : String) (typeKey : String)
      (typeArgs : Array InstanceTypeArg) (inputs : Array CoreInstanceInput)
  /-- Inert lifted type binding; no evaluator reaches it. -/
  | progDecl (name : String)
deriving Repr, Inhabited

structure CoreInputDecl where
  name : String
  type? : Option CorePortType := none
  default? : Option ExprId := none
deriving Repr, Inhabited

structure CoreOutputDecl where
  name : String
  type? : Option CorePortType := none
deriving Repr, Inhabited

structure CoreOutputAssign where
  target : OutputTarget
  expr : ExprId
deriving Repr, Inhabited

/-- A post-strata program, materialized as a tree. `registry` holds
    only the instance-referenced (evaluator-reachable) entries, keyed
    by typeKey in first-use order. -/
inductive CoreProgram where
  | mk (name : String)
       (inputs : Array CoreInputDecl)
       (outputs : Array CoreOutputDecl)
       (decls : Array CoreBodyDecl)
       (assigns : Array CoreOutputAssign)
       (registry : Array (String × CoreProgram))
deriving Inhabited

def CoreProgram.name : CoreProgram → String
  | .mk n .. => n

def CoreProgram.inputs : CoreProgram → Array CoreInputDecl
  | .mk _ i .. => i

def CoreProgram.outputs : CoreProgram → Array CoreOutputDecl
  | .mk _ _ o .. => o

def CoreProgram.decls : CoreProgram → Array CoreBodyDecl
  | .mk _ _ _ d .. => d

def CoreProgram.assigns : CoreProgram → Array CoreOutputAssign
  | .mk _ _ _ _ a _ => a

def CoreProgram.registry : CoreProgram → Array (String × CoreProgram)
  | .mk _ _ _ _ _ r => r

/-- Instance-type lookup (TS `getInstanceType` against the registry). -/
def CoreProgram.registryGet? (p : CoreProgram) (key : String) : Option CoreProgram :=
  (p.registry.find? (·.1 == key)).map (·.2)

/-- Projected param table (positions are `ParamIdx`). -/
def CoreProgram.params (p : CoreProgram) : Array CoreBodyDecl :=
  p.decls.filter fun d => match d with | .param .. => true | _ => false

/-- Projected instance table (positions are `InstanceIdx`). -/
def CoreProgram.instances (p : CoreProgram) : Array CoreBodyDecl :=
  p.decls.filter fun d => match d with | .inst .. => true | _ => false

-- ─────────────────────────────────────────────────────────────
-- Port-type resolution (shared with the DAG downcast)
--
-- The expression downcast (`Expr`/`CoreExpr` → `ExprId`) moved to
-- `EArena.toResolved` (Phase B: the strata DAG is threaded straight to
-- emit instead of flattened to a tree and re-interned). These helpers —
-- the pool-reference resolution for port types — stay here (they need
-- only the identity pools, not the arena) and are public so the
-- id-form downcast can reuse them.
-- ─────────────────────────────────────────────────────────────

/-- Resolve a pool-referencing element type to its self-contained Core
    form. Non-alias typeDefs in element position are unrepresentable in
    well-formed post-strata IR; collapse to float (the TS access path
    would read `undefined.base`-adjacent shapes as float downstream). -/
def resolveElem (arena : Arena) : ScalarOrAlias → CoreElem
  | .scalar k => .scalar k
  | .alias td =>
    match arena.typeDef? td with
    | some (.alias _ base) => .aliased base
    | _ => .aliased .float

def resolvePortType (arena : Arena) : PortType → CorePortType
  | .scalar k => .scalar k
  | .alias td =>
    match arena.typeDef? td with
    | some (.alias _ base) => .alias base
    | _ => .alias .float
  | .array element shape =>
    .array (resolveElem arena element) (shape.map fun
      | .lit n => CoreShapeDim.lit n
      | .typeParam _ => CoreShapeDim.unresolved)

def resolveOptPortType (arena : Arena) :
    Option PortType → Option CorePortType
  | some t => some (resolvePortType arena t)
  | none => none

end Tropical.Ir.Core
