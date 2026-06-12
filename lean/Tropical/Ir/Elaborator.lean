import Tropical.Ir.Nodes

/-!
# Elaborator — line-faithful port of compiler/ir/elaborator.ts

`ParsedProgram → ResolvedProgram` over the arena. Single top-down pass:
each declaration is constructed once, registered in scope, and re-used
by (typed-index) reference at every site that names it.

Faithfulness notes — every observable TS behavior is replicated:

- **Decl reordering.** `registerBodyDecls` pushes ALL nested
  programDecls first (source order), then non-program decls in source
  order: resolved `body.decls` ≠ parsed order.
- **Two-phase decl elaboration.** Shells are registered first (forward
  refs work), expressions resolved in a second pass iterating in
  pairing-insertion order (= non-program source order). A parsed
  `delayDecl` shell registers `update := 0` so a later `next x = e`
  throws `duplicate update for reg '...'`.
- **Sequential let.** Each binder is minted, its value resolved in a
  scope WITHOUT it, THEN pushed — subsequent entries and the body see
  it. Restore-on-exit shadowing falls out of the persistent binder list.
- **Binder minting order** (`BinderIdx` values): let entries in record
  order; fold/scan mint acc then elem; zipWith x then y; match arms
  mint binders in variant.payload declaration order (not the pattern's
  written order), arms in source order, the scrutinee resolved AFTER
  all arms (matching the TS object-literal evaluation order).
- **Transitive registry merge.** `registerInstanceDecl` inserts the
  target under its program name, then the target's own registry entries
  in their order, skipping keys already present. Insertion order is
  observable through the codec.
- **Error strings** byte-match TS `ElaborationError` messages, and the
  cycle error reproduces `throwOnCycles` + `formatCycleDiagnostic` +
  the `CycleViolation` prefix exactly (including the typographic
  apostrophe in the suggested fix).
- **findInstanceCycles** (compiler/ir/lowering/cycle_break.ts): same
  dep-edge collection order, Tarjan visit order, SCC member order
  (stack-pop order — the SCC root is last), and nontrivial filter
  (size > 1 or self-loop).

Documented divergence: TS `resolveNestedOut` tolerates a raw *numeric*
`output` (positional). No ParsedProgram producer can emit one — the
surface parser always wraps the port in a `nameRef`
(compiler/parse/expressions.ts), `raise.ts` stringifies via
`nameRef(String(node.output))`, and `session_to_parsed.ts` emits
`nameRef` — and the typed Parse AST (stage 2) carries only the name, so
this port implements the NameRef branch only.
-/

namespace Tropical.Ir

open Lean (JsonNumber)
open Tropical.Parse (ParsedExpr)

-- ─────────────────────────────────────────────────────────────
-- Errors
-- ─────────────────────────────────────────────────────────────

/-- Port of `ElaborationError` (plain message) and `CycleViolation`
    (message prebuilt by `throwOnCycles`). -/
inductive ElabError where
  | elaboration (msg : String)
  | cycle (msg : String)
deriving Repr

def ElabError.message : ElabError → String
  | .elaboration m => m
  | .cycle m => m

-- ─────────────────────────────────────────────────────────────
-- Monad + scope
-- ─────────────────────────────────────────────────────────────

/-- Port of `ExternalProgramResolver`: already-resolved programs by
    name, as arena indices. -/
abbrev Resolver := String → Option ProgramIdx

structure ElabSt where
  arena : Arena := {}
  /-- Next-fresh BinderIdx for the program being elaborated. Saved and
      restored around nested-program elaboration (each program has its
      own binder namespace starting at 0). -/
  binderCount : Nat := 0

abbrev ElabM := StateT ElabSt (Except ElabError)

private def throwElab {α} (msg : String) : ElabM α :=
  throw (.elaboration msg)

/-- The chain-visible scope families: type defs, type params, variant
    registry, nested program types. (Value decls never leak to nested
    programs — `lookupValueRef` is local-only in TS.) -/
structure Frame where
  typeParams : Array (String × TypeParamPoolIdx) := #[]
  typeDefs : Array (String × TypeDefIdx) := #[]
  /-- variant name → (parent sum pool idx, variant position). -/
  variants : Array (String × TypeDefIdx × Nat) := #[]
  programs : Array (String × ProgramIdx) := #[]

structure Scope where
  frame : Frame
  parents : List Frame
  resolver : Option Resolver := none
  /-- Position = InputIdx. -/
  inputNames : Array String := #[]
  /-- name → RegIdx, in registration order. -/
  regs : Array (String × Nat) := #[]
  params : Array (String × Nat) := #[]
  /-- name → (InstanceIdx, typeKey). -/
  instances : Array (String × Nat × String) := #[]
  registry : Array (String × ProgramIdx) := #[]
  /-- Innermost-first; a later push shadows. -/
  binders : List (String × Binder) := []

private def Scope.frames (s : Scope) : List Frame :=
  s.frame :: s.parents

private def lookupTypeDefChain (s : Scope) (name : String) : Option TypeDefIdx :=
  s.frames.findSome? fun f => (f.typeDefs.find? (·.1 == name)).map (·.2)

private def lookupTypeParamChain (s : Scope) (name : String) : Option TypeParamPoolIdx :=
  s.frames.findSome? fun f => (f.typeParams.find? (·.1 == name)).map (·.2)

private def lookupVariantChain (s : Scope) (name : String) : Option (TypeDefIdx × Nat) :=
  s.frames.findSome? fun f => (f.variants.find? (·.1 == name)).map (·.2)

private def lookupProgramChain (s : Scope) (name : String) : Option ProgramIdx :=
  s.frames.findSome? fun f => (f.programs.find? (·.1 == name)).map (·.2)

private def getProgram (i : ProgramIdx) : ElabM Program := do
  match (← get).arena.program? i with
  | some p => pure p
  | none => throwElab s!"internal: arena program index {i.idx} out of range"

private def getTypeDef (i : TypeDefIdx) : ElabM TypeDef := do
  match (← get).arena.typeDef? i with
  | some td => pure td
  | none => throwElab s!"internal: arena typeDef index {i.idx} out of range"

private def getTypeParamName (i : TypeParamPoolIdx) : ElabM String := do
  match (← get).arena.typeParam? i with
  | some tp => pure tp.name
  | none => throwElab s!"internal: arena typeParam index {i.idx} out of range"

/-- Allocate a fresh binder with a unique-per-program idx (port of
    `mintBinder`). The caller pushes it into scope. -/
private def mintBinder (name : String) : ElabM Binder := do
  let n := (← get).binderCount
  modify fun st => { st with binderCount := n + 1 }
  pure { name, idx := ⟨n⟩ }

private def pushBinder (s : Scope) (b : Binder) : Scope :=
  { s with binders := (b.name, b) :: s.binders }

private def jnum0 : Expr := .num (JsonNumber.fromInt 0)

-- ─────────────────────────────────────────────────────────────
-- Builtin tables (elaborator.ts constants, verbatim)
-- ─────────────────────────────────────────────────────────────

/-- BUILTIN_TYPE_TO_SCALAR (includes `phase`). -/
private def builtinTypeToScalar? : String → Option ScalarKind
  | "float" => some .float
  | "int" => some .int
  | "bool" => some .bool
  | "signal" | "freq" | "unipolar" | "bipolar" | "phase" => some .float
  | _ => none

private def nullaryCalls : List String :=
  ["sample_rate", "sample_index", "sampleRate", "sampleIndex"]

private def unaryCall? : String → Option UnaryOpTag
  | "sqrt" => some .sqrt
  | "abs" => some .abs
  | "neg" => some .neg
  | "floor" => some .floor
  | "ceil" => some .ceil
  | "round" => some .round
  | "not" => some .not
  | "bit_not" | "bitNot" => some .bitNot
  | "to_int" | "toInt" => some .toInt
  | "to_bool" | "toBool" => some .toBool
  | "to_float" | "toFloat" => some .toFloat
  | "float_exponent" | "floatExponent" => some .floatExponent
  | _ => none

private def binaryCall? : String → Option BinaryOpTag
  | "floor_div" | "floorDiv" => some .floorDiv
  | "ldexp" => some .ldexp
  | _ => none

-- ─────────────────────────────────────────────────────────────
-- Type defs + port types
-- ─────────────────────────────────────────────────────────────

private def resolveStructField (f : Tropical.Parse.StructField) : StructField :=
  { name := f.name, type := .scalar f.scalarType }

private def resolveTypeDef : Tropical.Parse.TypeDef → ElabM TypeDef
  | .struct name fields =>
    pure (.struct name (fields.map resolveStructField))
  | .sum name variants =>
    pure (.sum name (variants.map fun v =>
      { name := v.name, payload := v.payload.map resolveStructField }))
  | .alias name base =>
    match Tropical.Parse.ScalarKind.ofWire? base with
    | some k => pure (.alias name k)
    | none =>
      throwElab s!"alias '{name}' base must be a scalar kind (float/int/bool); got '{base}'"

/-- Port of `resolveElement` — array element name must be a scalar kind
    or alias. -/
private def resolveElement (scope : Scope) (name : String) : ElabM ScalarOrAlias := do
  match builtinTypeToScalar? name with
  | some k => pure (.scalar k)
  | none =>
    match lookupTypeDefChain scope name with
    | some tdIdx =>
      match ← getTypeDef tdIdx with
      | .alias .. => pure (.alias tdIdx)
      | td => throwElab s!"port type '{name}' must be a scalar kind or alias; got {td.opName}"
    | none => throwElab s!"unknown type name '{name}'"

private def resolveShapeDim (scope : Scope) : Tropical.Parse.ShapeDim → ElabM ShapeDim
  | .lit n => pure (.lit n)
  | .ref name =>
    match lookupTypeParamChain scope name with
    | some tp => pure (.typeParam tp)
    | none =>
      throwElab s!"array shape dim '{name}' is not a declared type-param of any enclosing program"

private def resolvePortType (scope : Scope) : Tropical.Parse.PortTypeDecl → ElabM PortType
  | .scalar name => do
    match builtinTypeToScalar? name with
    | some k => pure (.scalar k)
    | none =>
      match lookupTypeDefChain scope name with
      | some tdIdx =>
        match ← getTypeDef tdIdx with
        | .alias .. => pure (.alias tdIdx)
        | _ => throwElab s!"unknown port type '{name}'"
      | none => throwElab s!"unknown port type '{name}'"
  | .array element shape => do
    let elem ← resolveElement scope element
    let mut dims : Array ShapeDim := #[]
    for d in shape do
      dims := dims.push (← resolveShapeDim scope d)
    pure (.array elem dims)

-- ─────────────────────────────────────────────────────────────
-- Expressions
-- ─────────────────────────────────────────────────────────────

/-- Port of `lookupValueRef` — local scope only, fixed category order. -/
private def lookupValueRef (s : Scope) (name : String) : Option Expr :=
  match s.binders.find? (·.1 == name) with
  | some (_, b) => some (.bindingRef b.idx)
  | none =>
  match s.regs.find? (·.1 == name) with
  | some (_, i) => some (.regRef ⟨i⟩)
  | none =>
  match s.params.find? (·.1 == name) with
  | some (_, i) => some (.paramRef ⟨i⟩)
  | none =>
  match s.inputNames.idxOf? name with
  | some i => some (.inputRef ⟨i⟩)
  | none =>
  match s.frame.typeParams.findIdx? (·.1 == name) with
  | some i => some (.typeParamRef ⟨i⟩)
  | none => none

private def getSum (defIdx : TypeDefIdx) : ElabM (String × Array SumVariant) := do
  match ← getTypeDef defIdx with
  | .sum name variants => pure (name, variants)
  | td => throwElab s!"internal: variant registry points at non-sum '{td.name}'"

mutual

partial def resolveExpr (scope : Scope) : ParsedExpr → ElabM Expr
  | .num n => pure (.num n)
  | .bool b => pure (.bool b)
  | .arr items => do
    let mut out : Array Expr := #[]
    for e in items do
      out := out.push (← resolveExpr scope e)
    pure (.arr out)
  | .binary tag lhs rhs => do
    pure (.binary (BinaryOpTag.ofParse tag) (← resolveExpr scope lhs) (← resolveExpr scope rhs))
  | .unary tag arg => do
    pure (.unary (UnaryOpTag.ofParse tag) (← resolveExpr scope arg))
  | .nameRef name =>
    match lookupValueRef scope name with
    | some e => pure e
    | none => throwElab s!"unknown name '{name}'"
  | .binding name =>
    match scope.binders.find? (·.1 == name) with
    | some (_, b) => pure (.bindingRef b.idx)
    | none =>
      throwElab s!"binding '{name}' is not in scope (parser said it was bound — likely a parser bug)"
  | .nestedOut refName outputName => resolveNestedOut scope refName outputName
  | .index arr idx => do
    pure (.index (← resolveExpr scope arr) (← resolveExpr scope idx))
  | .call callee args => resolveCall scope callee args
  | .tag variant payload => resolveTag scope variant payload
  | .match_ scrutinee arms => resolveMatch scope scrutinee arms
  | .letIn bind body => resolveLet scope bind body
  | .fold over init accVar elemVar body => do
    let acc ← mintBinder accVar
    let elem ← mintBinder elemVar
    let bodyR ← resolveExpr (pushBinder (pushBinder scope acc) elem) body
    pure (.fold (← resolveExpr scope over) (← resolveExpr scope init) acc elem bodyR)
  | .scan over init accVar elemVar body => do
    let acc ← mintBinder accVar
    let elem ← mintBinder elemVar
    let bodyR ← resolveExpr (pushBinder (pushBinder scope acc) elem) body
    pure (.scan (← resolveExpr scope over) (← resolveExpr scope init) acc elem bodyR)
  | .generate count var body => do
    let iter ← mintBinder var
    let bodyR ← resolveExpr (pushBinder scope iter) body
    pure (.generate (← resolveExpr scope count) iter bodyR)
  | .iterate count var init body => do
    let iter ← mintBinder var
    let bodyR ← resolveExpr (pushBinder scope iter) body
    pure (.iterate (← resolveExpr scope count) (← resolveExpr scope init) iter bodyR)
  | .chain count var init body => do
    let iter ← mintBinder var
    let bodyR ← resolveExpr (pushBinder scope iter) body
    pure (.chain (← resolveExpr scope count) (← resolveExpr scope init) iter bodyR)
  | .map2 over elemVar body => do
    let elem ← mintBinder elemVar
    let bodyR ← resolveExpr (pushBinder scope elem) body
    pure (.map2 (← resolveExpr scope over) elem bodyR)
  | .zipWith a b xVar yVar body => do
    let x ← mintBinder xVar
    let y ← mintBinder yVar
    let bodyR ← resolveExpr (pushBinder (pushBinder scope x) y) body
    pure (.zipWith (← resolveExpr scope a) (← resolveExpr scope b) x y bodyR)

partial def resolveNestedOut (scope : Scope) (refName outputName : String) : ElabM Expr := do
  let some (_, instIdx, typeKey) := scope.instances.find? (·.1 == refName)
    | throwElab s!"instance '{refName}' is not declared in this scope"
  let some targetIdx := (scope.registry.find? (·.1 == typeKey)).map (·.2)
    | throwElab s!"internal: instance '{refName}' typeKey '{typeKey}' not in scope registry"
  let target ← getProgram targetIdx
  match target.outputs.findIdx? (·.name == outputName) with
  | some o => pure (.nestedOut ⟨instIdx⟩ ⟨o⟩)
  | none =>
    let portList := String.intercalate ", " (target.outputs.map (·.name)).toList
    throwElab s!"instance '{refName}': program '{target.name}' has no output '{outputName}' (have: {portList})"

partial def resolveCall (scope : Scope) (callee : ParsedExpr)
    (args : Array ParsedExpr) : ElabM Expr := do
  let .nameRef fname := callee
    | throwElab "unsupported call form: callee must be an identifier (no first-class function values yet)"
  if nullaryCalls.contains fname then
    if args.size != 0 then
      throwElab s!"'{fname}()' takes no arguments"
    if fname == "sample_rate" || fname == "sampleRate" then
      pure .sampleRate
    else
      pure .sampleIndex
  else if let some tag := unaryCall? fname then
    if args.size != 1 then
      throwElab s!"'{fname}' takes 1 argument; got {args.size}"
    pure (.unary tag (← resolveExpr scope args[0]!))
  else if let some tag := binaryCall? fname then
    if args.size != 2 then
      throwElab s!"'{fname}' takes 2 arguments; got {args.size}"
    pure (.binary tag (← resolveExpr scope args[0]!) (← resolveExpr scope args[1]!))
  else if fname == "zeros" then
    if args.size != 1 then
      throwElab s!"'zeros' takes 1 argument (count); got {args.size}"
    pure (.zeros (← resolveExpr scope args[0]!))
  else if fname == "arraySet" || fname == "array_set" then
    if args.size != 3 then
      throwElab s!"'{fname}' takes 3 arguments (arr, idx, value); got {args.size}"
    pure (.arraySet (← resolveExpr scope args[0]!) (← resolveExpr scope args[1]!)
                    (← resolveExpr scope args[2]!))
  else if fname == "clamp" then
    if args.size != 3 then
      throwElab s!"'clamp' takes 3 arguments (value, lo, hi); got {args.size}"
    pure (.clamp (← resolveExpr scope args[0]!) (← resolveExpr scope args[1]!)
                 (← resolveExpr scope args[2]!))
  else if fname == "select" then
    if args.size != 3 then
      throwElab s!"'select' takes 3 arguments (cond, then, else); got {args.size}"
    pure (.select (← resolveExpr scope args[0]!) (← resolveExpr scope args[1]!)
                  (← resolveExpr scope args[2]!))
  else
    throwElab <|
      s!"unknown function '{fname}'. The resolved IR has no escape hatch for unknown calls — " ++
      "add the builtin to the elaborator's registry, or use an instance declaration if it's a program type."

partial def resolveTag (scope : Scope) (variantName : String)
    (payload : Option (Array Tropical.Parse.TagPayloadEntry)) : ElabM Expr := do
  let some (defIdx, varPos) := lookupVariantChain scope variantName
    | throwElab s!"tag construction: unknown variant '{variantName}'"
  let (_, variants) ← getSum defIdx
  let variant := variants[varPos]!
  -- `supplied`: Map semantics — set overwrites in place, insertion
  -- order observable through the extras error. Every payload value is
  -- resolved (side effects included) before field validation, like TS.
  let mut supplied : Array (String × Expr) := #[]
  for entry in payload.getD #[] do
    let v ← resolveExpr scope entry.value
    match supplied.findIdx? (·.1 == entry.field) with
    | some i => supplied := supplied.set! i (entry.field, v)
    | none => supplied := supplied.push (entry.field, v)
  let mut out : Array TagPayload := #[]
  let mut consumed : Array String := #[]
  for h : fi in [0:variant.payload.size] do
    let field := variant.payload[fi]
    match supplied.find? (·.1 == field.name) with
    | none => throwElab s!"tag '{variantName}': missing payload field '{field.name}'"
    | some (_, v) =>
      out := out.push (.mk fi v)
      consumed := consumed.push field.name
  let extras := supplied.filter fun (n, _) => !consumed.contains n
  if !extras.isEmpty then
    let names := String.intercalate ", " (extras.map (·.1)).toList
    throwElab s!"tag '{variantName}': unknown payload field(s): {names}"
  pure (.tag defIdx varPos out)

partial def resolveMatch (scope : Scope) (scrutinee : ParsedExpr)
    (arms : Array Tropical.Parse.MatchArm) : ElabM Expr := do
  if arms.isEmpty then
    throwElab "match expression has no arms"
  let firstName := arms[0]!.variant
  let some (defIdx, _) := lookupVariantChain scope firstName
    | throwElab s!"match: unknown variant '{firstName}' in first arm"
  let (sumName, variants) ← getSum defIdx
  let mut seen : Array Nat := #[]
  let mut outArms : Array MatchArm := #[]
  for a in arms do
    let some varPos := variants.findIdx? (·.name == a.variant)
      | throwElab s!"match: variant '{a.variant}' is not a member of sum type '{sumName}'"
    let variant := variants[varPos]!
    if seen.contains varPos then
      throwElab s!"match: duplicate arm for variant '{variant.name}'"
    seen := seen.push varPos
    if a.binds.size != variant.payload.size then
      throwElab s!"match arm '{variant.name}': expected {variant.payload.size} binder(s) (one per payload field), got {a.binds.size}"
    let mut bindByField : Array (String × String) := #[]
    for (field, bind) in a.binds do
      if bindByField.any (·.1 == field) then
        throwElab s!"match arm '{variant.name}': duplicate pattern field '{field}'"
      bindByField := bindByField.push (field, bind)
    -- Binders minted in variant.payload declaration order.
    let mut binders : Array Binder := #[]
    let mut consumed : Array String := #[]
    for field in variant.payload do
      match bindByField.find? (·.1 == field.name) with
      | none =>
        throwElab s!"match arm '{variant.name}': missing pattern binding for payload field '{field.name}'"
      | some (_, bindName) =>
        binders := binders.push (← mintBinder bindName)
        consumed := consumed.push field.name
    let extras := bindByField.filter fun (n, _) => !consumed.contains n
    if !extras.isEmpty then
      let names := String.intercalate ", " (extras.map (·.1)).toList
      throwElab s!"match arm '{variant.name}': unknown pattern field(s): {names}"
    let armScope := binders.foldl pushBinder scope
    let body ← resolveExpr armScope a.body
    outArms := outArms.push (.mk varPos binders body)
  -- Exhaustiveness — before scrutinee resolution (TS object-literal
  -- field order: the return's `scrutinee: resolveExpr(...)` runs last).
  for h : vi in [0:variants.size] do
    if !seen.contains vi then
      throwElab s!"match on '{sumName}' is non-exhaustive: missing variant '{variants[vi].name}'"
  pure (.match_ defIdx (← resolveExpr scope scrutinee) outArms)

partial def resolveLet (scope : Scope) (bind : Array (String × ParsedExpr))
    (body : ParsedExpr) : ElabM Expr := do
  -- Sequential let* semantics: mint, resolve value WITHOUT the binder,
  -- then push it for subsequent entries and the body.
  let mut s := scope
  let mut binders : Array LetBinder := #[]
  for (name, valueExpr) in bind do
    let binder ← mintBinder name
    let value ← resolveExpr s valueExpr
    binders := binders.push (.mk binder value)
    s := pushBinder s binder
  pure (.letIn binders (← resolveExpr s body))

end

-- ─────────────────────────────────────────────────────────────
-- Cycle detection (port of cycle_break.ts findInstanceCycles)
-- ─────────────────────────────────────────────────────────────

/-- Dep-edge collection (collectNestedOutInstances): exact traversal
    order; first-add wins position (Set insertion parity). -/
private partial def collectNestedOutDeps (numInstances : Nat) (acc : Array Nat) :
    Expr → Array Nat
  | .num _ | .bool _ => acc
  | .arr items => items.foldl (collectNestedOutDeps numInstances) acc
  | .nestedOut inst _ =>
    if inst.idx < numInstances && !acc.contains inst.idx then acc.push inst.idx else acc
  | .match_ _ scrutinee arms =>
    arms.foldl (fun a arm => collectNestedOutDeps numInstances a arm.body)
      (collectNestedOutDeps numInstances acc scrutinee)
  | .fold over init _ _ body | .scan over init _ _ body =>
    collectNestedOutDeps numInstances
      (collectNestedOutDeps numInstances
        (collectNestedOutDeps numInstances acc over) init) body
  | .generate count _ body =>
    collectNestedOutDeps numInstances (collectNestedOutDeps numInstances acc count) body
  | .iterate count init _ body | .chain count init _ body =>
    collectNestedOutDeps numInstances
      (collectNestedOutDeps numInstances
        (collectNestedOutDeps numInstances acc count) init) body
  | .map2 over _ body =>
    collectNestedOutDeps numInstances (collectNestedOutDeps numInstances acc over) body
  | .zipWith a b _ _ body =>
    collectNestedOutDeps numInstances
      (collectNestedOutDeps numInstances
        (collectNestedOutDeps numInstances acc a) b) body
  | .letIn binders body =>
    collectNestedOutDeps numInstances
      (binders.foldl (fun a b => collectNestedOutDeps numInstances a b.value) acc) body
  | .tag _ _ payload =>
    payload.foldl (fun a p => collectNestedOutDeps numInstances a p.value) acc
  | .zeros count => collectNestedOutDeps numInstances acc count
  | .binary _ lhs rhs =>
    collectNestedOutDeps numInstances (collectNestedOutDeps numInstances acc lhs) rhs
  | .unary _ arg => collectNestedOutDeps numInstances acc arg
  | .clamp a b c | .select a b c | .arraySet a b c =>
    collectNestedOutDeps numInstances
      (collectNestedOutDeps numInstances
        (collectNestedOutDeps numInstances acc a) b) c
  | .index a b =>
    collectNestedOutDeps numInstances (collectNestedOutDeps numInstances acc a) b
  | .inputRef _ | .regRef _ | .paramRef _ | .typeParamRef _ | .bindingRef _
  | .sampleRate | .sampleIndex => acc

private structure TarjanSt where
  indexOf : Array (Option Nat)
  lowlink : Array Nat
  onStack : Array Bool
  stack : Array Nat := #[]
  sccs : Array (Array Nat) := #[]
  next : Nat := 0

/-- Tarjan's SCC, recursion + orders matching cycle_break.ts: nodes
    visited in instance order, successors in dep insertion order, SCC
    members in stack-pop order (the SCC root is last). -/
private partial def strongConnect (deps : Array (Array Nat)) (v : Nat)
    (st0 : TarjanSt) : TarjanSt := Id.run do
  let mut st := st0
  st := { st with
    indexOf := st.indexOf.set! v (some st.next)
    lowlink := st.lowlink.set! v st.next
    next := st.next + 1
    stack := st.stack.push v
    onStack := st.onStack.set! v true }
  for w in deps[v]! do
    if st.indexOf[w]!.isNone then
      st := strongConnect deps w st
      st := { st with lowlink := st.lowlink.set! v (Nat.min st.lowlink[v]! st.lowlink[w]!) }
    else if st.onStack[w]! then
      st := { st with lowlink := st.lowlink.set! v (Nat.min st.lowlink[v]! (st.indexOf[w]!.getD 0)) }
  if some st.lowlink[v]! == st.indexOf[v]! then
    let mut scc : Array Nat := #[]
    let mut go := true
    while go do
      match st.stack.back? with
      | none => go := false
      | some w =>
        st := { st with stack := st.stack.pop, onStack := st.onStack.set! w false }
        scc := scc.push w
        if w == v then go := false
    st := { st with sccs := st.sccs.push scc }
  return st

/-- Port of `findInstanceCycles`: non-trivial SCCs of the inter-instance
    dep graph, as instance-name lists in SCC member order. -/
def findInstanceCycles (prog : Program) : Array (Array String) := Id.run do
  let insts : Array (String × Array InstanceInput) := prog.decls.filterMap fun d =>
    match d with
    | .inst name _ _ inputs => some (name, inputs)
    | _ => none
  if insts.isEmpty then return #[]
  let n := insts.size
  let deps : Array (Array Nat) := insts.map fun (_, inputs) =>
    inputs.foldl (fun acc w => collectNestedOutDeps n acc w.value) #[]
  let mut st : TarjanSt := {
    indexOf := Array.replicate n none
    lowlink := Array.replicate n 0
    onStack := Array.replicate n false }
  for v in [0:n] do
    if st.indexOf[v]!.isNone then
      st := strongConnect deps v st
  let nontrivial := st.sccs.filter fun scc =>
    scc.size > 1 || (scc.size == 1 && (deps[scc[0]!]!).contains scc[0]!)
  return nontrivial.map (·.map fun i => insts[i]!.1)

/-- Port of `throwOnCycles` + `formatCycleDiagnostic` + the
    `CycleViolation` message — byte-exact, including the suggested-fix
    snippet's straight quote + typographic apostrophe pairing. -/
private def throwOnCycles (prog : Program) : ElabM Unit := do
  let cycles := findInstanceCycles prog
  if cycles.isEmpty then return
  let diagnostics := cycles.map fun scc =>
    let target := scc[0]!
    let memberPath := String.intercalate " → " scc.toList
    let suggestedFix :=
      s!"Suggested fix: insert a 'delay' statement on one of '{target}'’s " ++
      "output ports to break the cycle explicitly. " ++
      s!"Example: 'delay {target}_out_delayed = {target}.<port> init 0' " ++
      s!"and route cycle members from {target}_out_delayed instead."
    s!"tropical: cycle in program '{prog.name}' without a user register\n" ++
    s!"  Instances in cycle: {memberPath}\n" ++
    s!"  {suggestedFix}"
  throw (.cycle ("tropical: strict cycle policy violated:\n" ++
    String.intercalate "\n\n" diagnostics.toList))

-- ─────────────────────────────────────────────────────────────
-- Instance args (second pass)
-- ─────────────────────────────────────────────────────────────

private def resolveInstanceArgs (scope : Scope) (instName : String) (typeKey : String)
    (typeArgs : Option (Array (String × JsonNumber)))
    (inputs : Option (Array (String × ParsedExpr))) :
    ElabM (Array InstanceTypeArg × Array InstanceInput) := do
  let some targetIdx := (scope.registry.find? (·.1 == typeKey)).map (·.2)
    | throwElab s!"internal: instance '{instName}' typeKey '{typeKey}' not in scope registry"
  let target ← getProgram targetIdx
  -- Type args: param NameRef → position in the target's typeParams.
  let mut paramNames : Array String := #[]
  for tp in target.typeParams do
    paramNames := paramNames.push (← getTypeParamName tp)
  let mut tas : Array InstanceTypeArg := #[]
  for (paramName, value) in typeArgs.getD #[] do
    let some pos := paramNames.idxOf? paramName
      | let expected := if paramNames.isEmpty then "(none)"
          else String.intercalate ", " paramNames.toList
        throwElab s!"instance '{instName}': type-arg '{paramName}' is not a declared type-param of '{target.name}' (have: {expected})"
    if tas.any (·.param.idx == pos) then
      throwElab s!"instance '{instName}': duplicate type-arg '{paramName}'"
    tas := tas.push { param := ⟨pos⟩, value }
  -- Inputs: port NameRef → position in the target's ports.inputs.
  let mut ins : Array InstanceInput := #[]
  for (portName, valueExpr) in inputs.getD #[] do
    let some pos := target.inputs.findIdx? (·.name == portName)
      | let expected := if target.inputs.isEmpty then "(none)"
          else String.intercalate ", " (target.inputs.map (·.name)).toList
        throwElab s!"instance '{instName}': input '{portName}' is not a declared port of '{target.name}' (have: {expected})"
    if ins.any (·.port.idx == pos) then
      throwElab s!"instance '{instName}': duplicate input '{portName}'"
    let value ← resolveExpr scope valueExpr
    ins := ins.push { port := ⟨pos⟩, value }
  pure (tas, ins)

-- ─────────────────────────────────────────────────────────────
-- Program elaboration
-- ─────────────────────────────────────────────────────────────

private def assocSet (a : Array (String × ProgramIdx)) (k : String) (v : ProgramIdx) :
    Array (String × ProgramIdx) :=
  match a.findIdx? (·.1 == k) with
  | some i => a.set! i (k, v)
  | none => a.push (k, v)

partial def elaborateProgram (p : Tropical.Parse.Program) (parents : List Frame)
    (resolver : Option Resolver) : ElabM ProgramIdx := do
  let savedBinderCount := (← get).binderCount
  modify fun st => { st with binderCount := 0 }
  let mut frame : Frame := {}

  -- 1. Type defs from ports.type_defs — in scope before port walks.
  for td in (p.ports.bind (·.typeDefs)).getD #[] do
    let resolved ← resolveTypeDef td
    if frame.typeDefs.any (·.1 == resolved.name) then
      throwElab s!"duplicate type def '{resolved.name}'"
    let tdIdx : TypeDefIdx := ⟨(← get).arena.typeDefs.size⟩
    modify fun st => { st with arena :=
      { st.arena with typeDefs := st.arena.typeDefs.push resolved } }
    frame := { frame with typeDefs := frame.typeDefs.push (resolved.name, tdIdx) }
    if let .sum _ variants := resolved then
      for h : vi in [0:variants.size] do
        let v := variants[vi]
        if frame.variants.any (·.1 == v.name) then
          throwElab s!"variant '{v.name}' is declared in multiple sum types — variant names must be unique"
        frame := { frame with variants := frame.variants.push (v.name, tdIdx, vi) }

  -- 2. Type params.
  for (name, spec) in p.typeParams.getD #[] do
    let poolIdx : TypeParamPoolIdx := ⟨(← get).arena.typeParams.size⟩
    modify fun st => { st with arena :=
      { st.arena with typeParams := st.arena.typeParams.push { name, default? := spec.default? } } }
    frame := { frame with typeParams := frame.typeParams.push (name, poolIdx) }

  -- 3. Input + output ports. Defaults are resolved incrementally —
  --    a default sees earlier inputs and the type params, nothing else.
  let mut inputs : Array InputDecl := #[]
  for portSpec in (p.ports.bind (·.inputs)).getD #[] do
    let scope : Scope := { frame, parents, resolver, inputNames := inputs.map (·.name) }
    let decl ← match portSpec with
      | .bare name => pure { name : InputDecl }
      | .spec s => do
        let type? ← match s.type? with
          | none => pure none
          | some t => pure (some (← resolvePortType scope t))
        let default? ← match s.default? with
          | none => pure none
          | some d => pure (some (← resolveExpr scope d))
        pure { name := s.name, type?, default? : InputDecl }
    if inputs.any (·.name == decl.name) then
      throwElab s!"duplicate input port '{decl.name}'"
    inputs := inputs.push decl
  let mut outputs : Array OutputDecl := #[]
  for portSpec in (p.ports.bind (·.outputs)).getD #[] do
    let scope : Scope := { frame, parents, resolver, inputNames := inputs.map (·.name) }
    let decl ← match portSpec with
      | .bare name => pure { name : OutputDecl }
      | .spec s => do
        let type? ← match s.type? with
          | none => pure none
          | some t => pure (some (← resolvePortType scope t))
        pure { name := s.name, type? : OutputDecl }
    if outputs.any (·.name == decl.name) then
      throwElab s!"duplicate output port '{decl.name}'"
    outputs := outputs.push decl

  -- 4. Register body decls. Pre-pass: ALL nested programDecls first
  --    (source order) — sibling instances may reference them.
  let mut decls : Array BodyDecl := #[]
  let mut pairing : Array (Nat × Tropical.Parse.BodyDecl) := #[]
  let mut regsTbl : Array (String × Nat) := #[]
  let mut regCells : Array (String × Nat) := #[]
  let mut paramsTbl : Array (String × Nat) := #[]
  let mut instTbl : Array (String × Nat × String) := #[]
  let mut registry : Array (String × ProgramIdx) := #[]
  for d in p.body.decls do
    if let .prog name inner := d then
      let innerIdx ← elaborateProgram inner (frame :: parents) resolver
      if frame.programs.any (·.1 == name) then
        throwElab s!"duplicate nested program '{name}'"
      frame := { frame with programs := frame.programs.push (name, innerIdx) }
      decls := decls.push (.prog name innerIdx)
  -- Then the rest, in source order: decl shells with placeholder
  -- expressions; per-kind de Bruijn levels recorded as we go.
  let mut regCount := 0
  let mut paramCount := 0
  let mut instCount := 0
  for d in p.body.decls do
    match d with
    | .prog .. => pure ()  -- already handled
    | .reg name _ type? => do
      if regsTbl.any (·.1 == name) then
        throwElab s!"duplicate reg '{name}'"
      let rType? : Option ScalarOrAlias ← match type? with
        | none => pure none
        | some tname =>
          match builtinTypeToScalar? tname with
          | some k => pure (some (.scalar k))
          | none =>
            let scope : Scope := { frame, parents, resolver }
            match lookupTypeDefChain scope tname with
            | some tdIdx =>
              match ← getTypeDef tdIdx with
              | .alias .. => pure (some (.alias tdIdx))
              | _ => throwElab s!"reg '{name}': unknown type '{tname}'"
            | none => throwElab s!"reg '{name}': unknown type '{tname}'"
      let cell := decls.size
      decls := decls.push (.reg name jnum0 none rType? none)
      pairing := pairing.push (cell, d)
      regsTbl := regsTbl.push (name, regCount)
      regCells := regCells.push (name, cell)
      regCount := regCount + 1
    | .delay name _ _ _ => do
      if regsTbl.any (·.1 == name) then
        throwElab s!"duplicate reg/delay '{name}'"
      -- `update := 0` placeholder: the delay form commits to having an
      -- update, so a later `next` on it detects the conflict.
      let cell := decls.size
      decls := decls.push (.reg name jnum0 (some jnum0) none none)
      pairing := pairing.push (cell, d)
      regsTbl := regsTbl.push (name, regCount)
      regCells := regCells.push (name, cell)
      regCount := regCount + 1
    | .param name value? => do
      if paramsTbl.any (·.1 == name) then
        throwElab s!"duplicate param '{name}'"
      let cell := decls.size
      decls := decls.push (.param name value?)
      pairing := pairing.push (cell, d)
      paramsTbl := paramsTbl.push (name, paramCount)
      paramCount := paramCount + 1
    | .inst name progName _ _ => do
      if instTbl.any (·.1 == name) then
        throwElab s!"duplicate instance '{name}'"
      let scope : Scope := { frame, parents, resolver }
      let targetIdx? : Option ProgramIdx :=
        match lookupProgramChain scope progName with
        | some t => some t
        | none => resolver.bind (· progName)
      let some targetIdx := targetIdx?
        | throwElab <|
            s!"instance '{name}': program type '{progName}' is not a nested program in scope " ++
            "and no external resolver provided it. Pass an ExternalProgramResolver to elaborate() " ++
            "to resolve cross-program references (e.g. stdlib types)."
      let target ← getProgram targetIdx
      let tk := target.name
      registry := assocSet registry tk targetIdx
      -- Transitive merge: the target's own registry entries in their
      -- order, skipping keys already present.
      for (k, v) in target.registry do
        if !registry.any (·.1 == k) then
          registry := registry.push (k, v)
      let cell := decls.size
      decls := decls.push (.inst name tk #[] #[])
      pairing := pairing.push (cell, d)
      instTbl := instTbl.push (name, instCount, tk)
      instCount := instCount + 1

  -- 5. Resolve expressions inside body decls, in pairing order.
  let scope : Scope := {
    frame, parents, resolver
    inputNames := inputs.map (·.name)
    regs := regsTbl, params := paramsTbl, instances := instTbl, registry }
  for (cell, parsed) in pairing do
    match parsed, decls[cell]! with
    | .reg _ initParsed _, .reg name _ update? type? lf => do
      let init ← resolveExpr scope initParsed
      decls := decls.set! cell (.reg name init update? type? lf)
    | .delay _ updateParsed initParsed _, .reg name _ _ type? lf => do
      let update ← resolveExpr scope updateParsed
      let init ← resolveExpr scope initParsed
      decls := decls.set! cell (.reg name init (some update) type? lf)
    | .param .., _ => pure ()  -- no expressions on paramDecl
    | .inst _ _ typeArgs instInputs, .inst name tk _ _ => do
      let (tas, ins) ← resolveInstanceArgs scope name tk typeArgs instInputs
      decls := decls.set! cell (.inst name tk tas ins)
    | parsedD, resolvedD =>
      throwElab s!"internal: paired {parsedDeclOp parsedD} with {resolvedDeclOp resolvedD}"

  -- 6. Body assigns. `next x = e` folds into the reg's update field.
  let mut assigns : Array OutputAssign := #[]
  for a in p.body.assigns do
    match a with
    | .output name exprParsed => do
      let target ← if name == "dac.out" then
          pure OutputTarget.dac
        else
          match outputs.findIdx? (·.name == name) with
          | some i => pure (OutputTarget.port ⟨i⟩)
          | none => throwElab s!"outputAssign references unknown output port '{name}'"
      assigns := assigns.push { target, expr := ← resolveExpr scope exprParsed }
    | .next _ name exprParsed => do
      let some (_, cell) := regCells.find? (·.1 == name)
        | throwElab s!"next-update target '{name}' is not a declared reg or delay"
      match decls[cell]! with
      | .reg rname init update? type? lf => do
        if update?.isSome then
          throwElab s!"duplicate update for reg '{name}' (already set by decl-side update or earlier next-update)"
        let update ← resolveExpr scope exprParsed
        decls := decls.set! cell (.reg rname init (some update) type? lf)
      | _ => throwElab s!"internal: next-update cell for '{name}' is not a reg"

  -- Build the program (mkProgram), validating registry coverage
  -- (validateProgramRegistry — same error text) before the cycle check.
  for d in decls do
    if let .inst iname tk _ _ := d then
      if !registry.any (·.1 == tk) then
        let keys := String.intercalate ", " (registry.map (·.1)).toList
        let keysShown := if keys.isEmpty then "(empty)" else keys
        throwElab <|
          s!"validateProgramRegistry: instance '{iname}' typeKey '{tk}' " ++
          s!"is not in the supplied program registry " ++
          s!"(keys present: {keysShown}). " ++
          "Construction site must add the target program to the registry before mkProgram/withDeclTables."
  let prog : Program := {
    name := p.name
    typeParams := frame.typeParams.map (·.2)
    inputs, outputs
    typeDefs := frame.typeDefs.map (·.2)
    decls, assigns
    binderCount := (← get).binderCount
    registry }

  -- Strict cycle policy (Phase 4b): inter-instance cycles that don't
  -- pass through an explicit user register throw CycleViolation.
  throwOnCycles prog

  let idx : ProgramIdx := ⟨(← get).arena.programs.size⟩
  modify fun st => { st with
    arena := { st.arena with programs := st.arena.programs.push prog }
    binderCount := savedBinderCount }
  pure idx
where
  parsedDeclOp : Tropical.Parse.BodyDecl → String
    | .reg .. => "regDecl" | .delay .. => "delayDecl" | .param .. => "paramDecl"
    | .inst .. => "instanceDecl" | .prog .. => "programDecl"
  resolvedDeclOp : BodyDecl → String
    | .reg .. => "regDecl" | .param .. => "paramDecl"
    | .inst .. => "instanceDecl" | .prog .. => "programDecl"

-- ─────────────────────────────────────────────────────────────
-- Public entry
-- ─────────────────────────────────────────────────────────────

/-- Elaborate into an existing arena (the chain form: stdlib programs
    elaborated in dependency order share one arena, the resolver maps
    names to earlier results). -/
def elaborateInto (arena : Arena) (p : Tropical.Parse.Program)
    (resolver : Option Resolver := none) : Except ElabError (Arena × ProgramIdx) := do
  let (idx, st) ← (elaborateProgram p [] resolver).run { arena }
  pure (st.arena, idx)

/-- Port of `elaborate(prog, resolveExternalProgram?)`. -/
def elaborate (p : Tropical.Parse.Program) (resolver : Option Resolver := none) :
    Except ElabError (Arena × ProgramIdx) :=
  elaborateInto {} p resolver

end Tropical.Ir
