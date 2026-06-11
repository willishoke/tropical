import Tropical.Ir.Nodes

/-!
# Core — the post-strata sub-IR as a type (Phase 5 stage 6)

"The smallest sub-IR sufficient for any per-sample evaluator," made a
type instead of prose. A `CoreExpr` is an `Expr` with the strata-dropped
constructors gone by construction:

  - no `typeParamRef`           (monomorphic — specialize)
  - no `tag` / `match`          (sum-free — sumLower)
  - no `fold`/`scan`/`generate`/`iterate`/`chain`/`map2`/`zipWith`/
    `letIn`/`bindingRef`/`zeros` (combinator-free — arrayLower)

Nesting is NOT dropped: the fractal session path keeps InstanceDecls
as kernel boundaries (the flat path simply has none). Lifted
ProgramDecls and never-instantiated registry entries are inert type
bindings no evaluator reaches — they pass through as names, unchecked.
`check` follows exactly the evaluator-reachable graph: body exprs,
then recursively the program behind each instance's typeKey.

`check` is the executable downcast — the formalization of the
post-strata invariant list, run as an assertion after the full Lean
pipeline, and the spec any later typed-boundary refactor of the passes
is checked against. Phase 6's emit/partition consume `CoreProgram`
with total matches — no impossible-case panics.

Harness scope note: on the differential `strata-file` nested path,
instance targets are raw-elaborated (production strata-processes each
instance type at registration), so the harness asserts `check` on the
inline path only; the production seam asserts it per registered
program.
-/

namespace Tropical.Ir.Core

open Lean (JsonNumber)
open Tropical.Ir

inductive CoreExpr where
  | num (n : JsonNumber)
  | bool (b : Bool)
  | arr (items : Array CoreExpr)
  | binary (tag : BinaryOpTag) (lhs rhs : CoreExpr)
  | unary (tag : UnaryOpTag) (arg : CoreExpr)
  | clamp (value lo hi : CoreExpr)
  | select (cond then_ else_ : CoreExpr)
  | arraySet (arr idx value : CoreExpr)
  | index (arr idx : CoreExpr)
  | inputRef (idx : InputIdx)
  | regRef (idx : RegIdx)
  | paramRef (idx : ParamIdx)
  | nestedOut (instance_ : InstanceIdx) (output : OutputIdx)
  | sampleRate
  | sampleIndex
deriving Repr, Inhabited

structure CoreInstanceInput where
  port : InputIdx
  value : CoreExpr
deriving Repr, Inhabited

inductive CoreBodyDecl where
  | reg (name : String) (init : CoreExpr) (update? : Option CoreExpr)
      (type? : Option ScalarOrAlias) (liftedFrom? : Option String)
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
  default? : Option CoreExpr := none
deriving Repr, Inhabited

structure CoreOutputAssign where
  target : OutputTarget
  expr : CoreExpr
deriving Repr, Inhabited

/-- A post-strata program, materialized as a tree. `registry` holds
    only the instance-referenced (evaluator-reachable) entries, keyed
    by typeKey in first-use order. -/
inductive CoreProgram where
  | mk (name : String)
       (inputs : Array CoreInputDecl)
       (outputNames : Array String)
       (decls : Array CoreBodyDecl)
       (assigns : Array CoreOutputAssign)
       (registry : Array (String × CoreProgram))
deriving Inhabited

def CoreProgram.name : CoreProgram → String
  | .mk n .. => n

def CoreProgram.decls : CoreProgram → Array CoreBodyDecl
  | .mk _ _ _ d .. => d

def CoreProgram.assigns : CoreProgram → Array CoreOutputAssign
  | .mk _ _ _ _ a _ => a

def CoreProgram.registry : CoreProgram → Array (String × CoreProgram)
  | .mk _ _ _ _ _ r => r

-- ─────────────────────────────────────────────────────────────
-- check — the executable downcast
-- ─────────────────────────────────────────────────────────────

private def fail {α} (prog : String) (what : String) : Except String α :=
  .error s!"core check ('{prog}'): {what} survived strata"

partial def checkExpr (progName : String) : Expr → Except String CoreExpr
  | .num n => return .num n
  | .bool b => return .bool b
  | .arr items => return .arr (← items.mapM (checkExpr progName))
  | .binary tag a b =>
    return .binary tag (← checkExpr progName a) (← checkExpr progName b)
  | .unary tag a => return .unary tag (← checkExpr progName a)
  | .clamp a b c =>
    return .clamp (← checkExpr progName a) (← checkExpr progName b) (← checkExpr progName c)
  | .select a b c =>
    return .select (← checkExpr progName a) (← checkExpr progName b) (← checkExpr progName c)
  | .arraySet a b c =>
    return .arraySet (← checkExpr progName a) (← checkExpr progName b) (← checkExpr progName c)
  | .index a b => return .index (← checkExpr progName a) (← checkExpr progName b)
  | .inputRef i => return .inputRef i
  | .regRef i => return .regRef i
  | .paramRef i => return .paramRef i
  | .nestedOut i o => return .nestedOut i o
  | .sampleRate => return .sampleRate
  | .sampleIndex => return .sampleIndex
  | .typeParamRef _ => fail progName "a typeParamRef (specialize)"
  | .bindingRef _ => fail progName "a bindingRef (arrayLower)"
  | .tag .. => fail progName "a tag (sumLower)"
  | .match_ .. => fail progName "a match (sumLower)"
  | .zeros _ => fail progName "a zeros (arrayLower)"
  | .fold .. => fail progName "a fold (arrayLower)"
  | .scan .. => fail progName "a scan (arrayLower)"
  | .generate .. => fail progName "a generate (arrayLower)"
  | .iterate .. => fail progName "an iterate (arrayLower)"
  | .chain .. => fail progName "a chain (arrayLower)"
  | .map2 .. => fail progName "a map2 (arrayLower)"
  | .zipWith .. => fail progName "a zipWith (arrayLower)"
  | .letIn .. => fail progName "a let (arrayLower)"

private def checkOptExpr (progName : String) :
    Option Expr → Except String (Option CoreExpr)
  | some e => some <$> checkExpr progName e
  | none => pure none

partial def check (arena : Arena) (rootIdx : ProgramIdx) :
    Except String CoreProgram := do
  let some prog := arena.program? rootIdx
    | .error s!"core check: program pool index {rootIdx.idx} out of range"
  unless prog.typeParams.isEmpty do
    fail prog.name s!"{prog.typeParams.size} typeParam decl(s) (specialize)"
  let decls ← prog.decls.mapM fun d => do
    match d with
    | .reg name init update? type? liftedFrom? =>
      -- Post-sumLower a reg's type is a scalar or an alias to one;
      -- sum-typed regs were decomposed into scalar slots.
      if let some (.alias td) := type? then
        if let some (.sum tdName _) := arena.typeDef? td then
          fail prog.name s!"reg '{name}' typed by sum '{tdName}' (sumLower)"
      pure (CoreBodyDecl.reg name (← checkExpr prog.name init)
        (← checkOptExpr prog.name update?) type? liftedFrom?)
    | .param name value? => pure (CoreBodyDecl.param name value?)
    | .inst name typeKey tArgs inputs =>
      pure (CoreBodyDecl.inst name typeKey tArgs
        (← inputs.mapM fun i => do
          pure { port := i.port, value := ← checkExpr prog.name i.value : CoreInstanceInput }))
    | .prog name _ => pure (CoreBodyDecl.progDecl name)
  let assigns ← prog.assigns.mapM fun a => do
    pure { target := a.target, expr := ← checkExpr prog.name a.expr : CoreOutputAssign }
  let inputs ← prog.inputs.mapM fun i => do
    pure { name := i.name, default? := ← checkOptExpr prog.name i.default? : CoreInputDecl }
  -- Registry: follow only instance-referenced entries (the
  -- evaluator-reachable graph), recursively.
  let mut registry : Array (String × CoreProgram) := #[]
  for d in prog.decls do
    if let .inst name typeKey _ _ := d then
      unless registry.any (·.1 == typeKey) do
        let some tIdx := prog.registryGet? typeKey
          | Except.error s!"core check ('{prog.name}'): instance '{name}' typeKey '{typeKey}' missing from registry"
        registry := registry.push (typeKey, ← check arena tIdx)
  return .mk prog.name inputs (prog.outputs.map (·.name)) decls assigns registry

end Tropical.Ir.Core
