import Tropical.Ir.Nodes

/-!
# Core — the post-strata program shape

"The smallest sub-IR sufficient for any per-sample evaluator," made a
type instead of prose. `CoreProgram` is the post-strata program: scalar,
monomorphic, acyclic, combinator-free. Its **expression leaves are
`ExprId`s** into a shared `ExprArena` (Phase B) — the tree twin `CoreExpr`
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
-- Port types
--
-- Emit/partition need port shapes (scalar counts, array sizes,
-- element kinds). With user type defs and generics retired, the
-- resolved `PortType` is already self-contained (builtin scalar kinds,
-- literal dims) — Core uses it directly; there is no separate
-- post-strata port-type universe.
-- ─────────────────────────────────────────────────────────────

/-- Post-strata port type = the resolved port type. -/
abbrev CorePortType := PortType

structure CoreInstanceInput where
  port : InputIdx
  value : ExprId
deriving Repr, Inhabited

inductive CoreBodyDecl where
  | param (name : String) (value? : Option JsonNumber)
  /-- Fractal kernel boundary; `typeKey` resolves in the enclosing
      `CoreProgram`'s registry. -/
  | inst (name : String) (typeKey : String) (inputs : Array CoreInstanceInput)
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

end Tropical.Ir.Core
