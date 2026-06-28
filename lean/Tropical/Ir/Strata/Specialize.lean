import Tropical.Ir.Nodes
import Tropical.Ir.Recursion
import Tropical.Ir.Strata.Basic
import Tropical.Ir.Strata.EArena

/-!
# specialize — port of compiler/ir/specialize.ts (Phase 5 pass 1)

Type-param substitution on the resolved IR. Produces a fresh `Program`
per (template, type-args) pair, substituting:
  - `typeParamRef` (in expression position) → integer literal
  - `ShapeDim.typeParam` → integer literal

The fresh program's `typeParams` is emptied; nested registry programs,
typeDefs, and `programDecl` targets are shared (pool indices — the
Lean image of TS sharing-by-reference). The no-op fast path (empty
substitution) returns the input pool index unchanged, mirroring the TS
identity short-circuit.

TS receives type args as a `Map<TypeParamDecl, number>` keyed by decl
identity, built by-name in the harness (`argsByName`); here the
by-name resolution and validation live together. The validation order
is TS's exactly — unknown names first (over the args in JSON field
order), then per declared param (in `typeParams` order): explicit arg
→ integer check, else default, else missing-arg. Error strings are
byte-exact comparable outputs.
-/

namespace Tropical.Ir.Strata.Specialize

open Lean (JsonNumber)
open Tropical.Ir

/-- JS `Number.isInteger` over the parsed double. -/
private def isIntegerValued (n : JsonNumber) : Bool :=
  let f := n.toFloat
  f == f.floor

/-- One supplied type-arg, identity-resolved. `poolIdx? = none` marks a
    name that matched no declared param (the TS harness mints a fresh
    decl there, keeping the not-declared error reachable). -/
structure ArgEntry where
  poolIdx? : Option TypeParamPoolIdx
  name : String
  value : JsonNumber
deriving Repr, Inhabited

private def declaredNames (declared : Array (TypeParamPoolIdx × TypeParamDecl)) : String :=
  let s := ", ".intercalate (declared.toList.map (·.2.name))
  if s.isEmpty then "(none)" else s

/-- Port of `buildSubst`: validate identity-keyed args against the
    program's declared type params and fill defaults. The result is in
    `typeParams` order and total over it. -/
private def buildSubst (progName : String)
    (declared : Array (TypeParamPoolIdx × TypeParamDecl))
    (typeArgs : Array ArgEntry) :
    Except Error (Array (TypeParamPoolIdx × JsonNumber)) := do
  for arg in typeArgs do
    unless arg.poolIdx?.any (fun p => declared.any (·.1 == p)) do
      throw ⟨s!"specializeProgram('{progName}'): type-arg '{arg.name}' is not a declared " ++
        s!"type-param (have: {declaredNames declared})"⟩
  let mut subst : Array (TypeParamPoolIdx × JsonNumber) := #[]
  for (poolIdx, tp) in declared do
    match typeArgs.find? (·.poolIdx? == some poolIdx) with
    | some arg =>
      unless isIntegerValued arg.value do
        throw ⟨s!"specializeProgram('{progName}'): type-arg '{tp.name}' must be an integer, got {arg.value}"⟩
      subst := subst.push (poolIdx, arg.value)
    | none =>
      match tp.default? with
      | some d => subst := subst.push (poolIdx, d)
      | none =>
        throw ⟨s!"specializeProgram('{progName}'): missing required type-arg '{tp.name}' (no default)"⟩
  return subst

/-- Identity-keyed entry (the TS `specializeProgram` proper). Used by
    the CLI adapter below and by inlineInstances' `specializeInner`. -/
def runCore (arena : Arena) (rootIdx : ProgramIdx)
    (typeArgs : Array ArgEntry) :
    Except Error (Arena × ProgramIdx) := do
  let some prog := arena.program? rootIdx
    | throw ⟨s!"specializeProgram: program pool index {rootIdx.idx} out of range"⟩
  let declared ← prog.typeParams.mapM fun i =>
    match arena.typeParam? i with
    | some tp => pure (i, tp)
    | none => throw
        (⟨s!"specializeProgram('{prog.name}'): typeParam pool index {i.idx} out of range"⟩ : Error)
  let subst ← buildSubst prog.name declared typeArgs
  -- Short-circuit: a non-generic program with no args to substitute is
  -- already "specialized" — same pool index, arena untouched.
  if subst.isEmpty then
    return (arena, rootIdx)

  -- `subst` is total over `typeParams` in declaration order, so the
  -- positional (TypeParamIdx) mirror is a direct projection.
  let byIdx : Array JsonNumber := subst.map (·.2)
  let hooks : MapHooks := {
    expr := fun e => match e with
      | .typeParamRef i => (byIdx[i.idx]?).map .num
      | _ => none
  }
  let rw := mapExpr hooks
  let shapeDim : ShapeDim → ShapeDim := fun d => match d with
    | .typeParam p =>
      match subst.find? (·.1 == p) with
      | some (_, v) => .lit v
      | none => d
    | d => d
  let portType := mapPortType shapeDim

  let mapDecl : BodyDecl → BodyDecl := fun
    -- Session-scoped by name; preserved as-is.
    | .param name value? => .param name value?
    | .inst name typeKey tArgs inputs =>
      .inst name typeKey tArgs (inputs.map fun i => { i with value := rw i.value })
    -- Nested program decls have their own typeParams scope; shared.
    | .prog name p => .prog name p

  let fresh : Program := { prog with
    typeParams := #[]   -- emptied: every reference has been substituted
    inputs := prog.inputs.map fun i =>
      { i with type? := i.type?.map portType, default? := i.default?.map rw }
    outputs := prog.outputs.map fun o => { o with type? := o.type?.map portType }
    decls := prog.decls.map mapDecl
    assigns := prog.assigns.map fun a => { a with expr := rw a.expr } }
  return ({ arena with programs := arena.programs.push fresh },
          ⟨arena.programs.size⟩)

/-- By-name adapter (the TS harness's `argsByName` + specializeProgram):
    each name binds the FIRST declared param with that name; unmatched
    names stay unresolved so the not-declared error fires in
    `buildSubst`. -/
def run (arena : Arena) (rootIdx : ProgramIdx)
    (typeArgs : Array (String × JsonNumber)) :
    Except Error (Arena × ProgramIdx) := do
  let some prog := arena.program? rootIdx
    | throw ⟨s!"specializeProgram: program pool index {rootIdx.idx} out of range"⟩
  let declared ← prog.typeParams.mapM fun i =>
    match arena.typeParam? i with
    | some tp => pure (i, tp)
    | none => throw
        (⟨s!"specializeProgram('{prog.name}'): typeParam pool index {i.idx} out of range"⟩ : Error)
  let args := typeArgs.map fun (name, value) =>
    { poolIdx? := (declared.find? (·.2.name == name)).map (·.1), name, value : ArgEntry }
  runCore arena rootIdx args

-- ─────────────────────────────────────────────────────────────
-- Id-form (#190 native-DAG) — mirrors runCore/run on EArena
-- ─────────────────────────────────────────────────────────────

/-- Id-form `runCore`: type-param substitution over the shared DAG. -/
def runCoreE (rootIdx : ProgramIdx) (typeArgs : Array ArgEntry) : PassM ProgramIdx := do
  let prog ← getEProgram rootIdx "specializeProgram"
  let declared ← prog.typeParams.mapM fun i => do
    match ← typeParamP? i with
    | some tp => pure (i, tp)
    | none => failP s!"specializeProgram('{prog.name}'): typeParam pool index {i.idx} out of range"
  let subst ← liftE (buildSubst prog.name declared typeArgs)
  if subst.isEmpty then
    return rootIdx
  let byIdx : Array JsonNumber := subst.map (·.2)
  let hooks : MapHooksId := {
    node := fun e => match e with
      | .typeParamRef i => match byIdx[i.idx]? with
        | some v => do pure (some (← einternP (.num v)))
        | none => pure none
      | _ => pure none }
  let rw := mapExprId hooks
  let shapeDim : ShapeDim → ShapeDim := fun d => match d with
    | .typeParam p =>
      match subst.find? (·.1 == p) with
      | some (_, v) => .lit v
      | none => d
    | d => d
  let portType := mapPortType shapeDim
  let inputs ← prog.inputs.mapM fun i => do
    pure ({ name := i.name, type? := i.type?.map portType,
            default? := ← i.default?.mapM rw } : EInputDecl)
  let outputs := prog.outputs.map fun o => { o with type? := o.type?.map portType }
  let decls ← prog.decls.mapM fun d => do
    match d with
    | .param name value? => pure (EBodyDecl.param name value?)
    | .inst name typeKey tArgs inputs =>
      pure (EBodyDecl.inst name typeKey tArgs
        (← inputs.mapM fun i => do
          pure ({ port := i.port, value := ← rw i.value } : EInstanceInput)))
    | .prog name p => pure (EBodyDecl.prog name p)
  let assigns ← prog.assigns.mapM fun a => do
    pure ({ target := a.target, expr := ← rw a.expr } : EOutputAssign)
  pushEProgram { prog with typeParams := #[], inputs, outputs, decls, assigns }

/-- Id-form by-name adapter (the harness `argsByName` + specializeProgram). -/
def runE (rootIdx : ProgramIdx) (typeArgs : Array (String × JsonNumber)) : PassM ProgramIdx := do
  let prog ← getEProgram rootIdx "specializeProgram"
  let declared ← prog.typeParams.mapM fun i => do
    match ← typeParamP? i with
    | some tp => pure (i, tp)
    | none => failP s!"specializeProgram('{prog.name}'): typeParam pool index {i.idx} out of range"
  let args := typeArgs.map fun (name, value) =>
    { poolIdx? := (declared.find? (·.2.name == name)).map (·.1), name, value : ArgEntry }
  runCoreE rootIdx args

end Tropical.Ir.Strata.Specialize
