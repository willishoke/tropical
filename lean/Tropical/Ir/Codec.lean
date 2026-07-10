import Tropical.Ir.Nodes
import Tropical.Parse.OrderedJson

/-!
# Resolved⇄JSON codec — port of compiler/ir/resolved_codec.ts

Schema `tropical_resolved_1`: three identity pools (typeParams,
typeDefs, programs) plus a root program index.

**Decode** rebuilds an `Arena` keeping the wire pool order as the arena
order, in a single forward pass — the encoder's topological guarantees
(alias deps before dependents, programs in post-order DFS from the
root) mean every reference points strictly earlier in its pool, and a
forward reference is a decode error.

**Encode** performs the same canonical reordering the TS encoder does:
pool ids are assigned on *first reference* during a deterministic
traversal — for each program: registry entries (in insertion order,
recursing into targets), then `typeParams`, `inputs`, `outputs`,
`typeDefs`, `decls` (nested `programDecl` targets recurse here),
`assigns`; the program itself is pushed *after* its referents
(post-order). TypeDefs push their alias field dependencies before
themselves. Encoding a TS-encoded wire after decode therefore
reproduces the pool order exactly.

Optional fields are absent, never null. Numbers are `JsonNumber`
(decimal text preserved); the differential comparators compare numbers
by IEEE-754 bit pattern after re-parsing.
-/

namespace Tropical.Ir.Codec

open Lean (Json JsonNumber)
open Tropical.Parse (JsonV)

def schemaTag : String := "tropical_resolved_1"

-- ─────────────────────────────────────────────────────────────
-- Encode
-- ─────────────────────────────────────────────────────────────

namespace Encode

structure St where
  typeParamPool : Array Json := #[]
  typeDefPool : Array Json := #[]
  programPool : Array Json := #[]
  /-- arena index → assigned pool id (none = not yet pooled). -/
  typeParamIds : Array (Option Nat)
  typeDefIds : Array (Option Nat)
  programIds : Array (Option Nat)

abbrev EncM := StateT St (Except String)

private def encErr {α} (msg : String) : EncM α :=
  throw s!"encodeResolved: {msg}"

private def optField (key : String) : Option Json → List (String × Json)
  | some v => [(key, v)]
  | none => []

private def scalarJson (k : ScalarKind) : Json := Json.str k.wire

def typeParamId (arena : Arena) (i : TypeParamPoolIdx) : EncM Nat := do
  let st ← get
  match st.typeParamIds[i.idx]? with
  | none => encErr s!"typeParam pool index {i.idx} out of range"
  | some (some id) => pure id
  | some none =>
    let some tp := arena.typeParam? i
      | encErr s!"typeParam pool index {i.idx} out of range"
    let entry := Json.mkObj <|
      [("name", Json.str tp.name)]
      ++ optField "default" (tp.default?.map Json.num)
    let id := (← get).typeParamPool.size
    modify fun st => { st with
      typeParamPool := st.typeParamPool.push entry
      typeParamIds := st.typeParamIds.set! i.idx (some id) }
    pure id

mutual

/-- Pool a typeDef, encoding alias field deps strictly before pushing
    self (the decoder's single-forward-pass contract). -/
partial def typeDefId (arena : Arena) (i : TypeDefIdx) : EncM Nat := do
  let st ← get
  match st.typeDefIds[i.idx]? with
  | none => encErr s!"typeDef pool index {i.idx} out of range"
  | some (some id) => pure id
  | some none =>
    let some td := arena.typeDef? i
      | encErr s!"typeDef pool index {i.idx} out of range"
    let entry : Json ← match td with
      | .alias name base =>
        pure <| Json.mkObj [("kind", Json.str "alias"), ("name", Json.str name),
                            ("base", scalarJson base)]
      | .struct name fields => do
        let mut fs : Array Json := #[]
        for f in fields do
          fs := fs.push (← encField arena f)
        pure <| Json.mkObj [("kind", Json.str "struct"), ("name", Json.str name),
                            ("fields", Json.arr fs)]
      | .sum name variants => do
        let mut vs : Array Json := #[]
        for v in variants do
          let mut fs : Array Json := #[]
          for f in v.payload do
            fs := fs.push (← encField arena f)
          vs := vs.push <| Json.mkObj [("name", Json.str v.name), ("payload", Json.arr fs)]
        pure <| Json.mkObj [("kind", Json.str "sum"), ("name", Json.str name),
                            ("variants", Json.arr vs)]
    let id := (← get).typeDefPool.size
    modify fun st => { st with
      typeDefPool := st.typeDefPool.push entry
      typeDefIds := st.typeDefIds.set! i.idx (some id) }
    pure id

partial def encField (arena : Arena) (f : StructField) : EncM Json := do
  pure <| Json.mkObj [("name", Json.str f.name), ("type", ← encScalarOrAlias arena f.type)]

partial def encScalarOrAlias (arena : Arena) : ScalarOrAlias → EncM Json
  | .scalar k => pure (scalarJson k)
  | .alias td => do pure <| Json.mkObj [("alias", Lean.toJson (← typeDefId arena td))]

end

def encShapeDim (arena : Arena) : ShapeDim → EncM Json
  | .lit n => pure (Json.num n)
  | .typeParam tp => do
    pure <| Json.mkObj [("typeParam", Lean.toJson (← typeParamId arena tp))]

def encPortType (arena : Arena) : PortType → EncM Json
  | .scalar k =>
    pure <| Json.mkObj [("kind", Json.str "scalar"), ("scalar", scalarJson k)]
  | .alias td => do
    pure <| Json.mkObj [("kind", Json.str "alias"), ("alias", Lean.toJson (← typeDefId arena td))]
  | .array element shape => do
    let mut dims : Array Json := #[]
    for d in shape do
      dims := dims.push (← encShapeDim arena d)
    pure <| Json.mkObj [("kind", Json.str "array"),
                        ("element", ← encScalarOrAlias arena element),
                        ("shape", Json.arr dims)]

def encBinder (b : Binder) : Json :=
  Json.mkObj [("name", Json.str b.name), ("idx", Lean.toJson b.idx.idx)]

/-- Encode a resolved expression to `tropical_resolved_1` JSON, derefing the
    id-form through `arena.exprs`. Byte-identical output to the former tree
    encoder — the arena is an implementation detail of how the graph is stored. -/
partial def encExpr (arena : Arena) (id : ExprId) : EncM Json := do
  let some node := arena.exprs.deref id
    | encErr s!"encExpr: dangling ExprId {id.idx}"
  match node with
  | .num n => pure (Json.num n)
  | .bool b => pure (Json.bool b)
  | .arr items => do
    let mut out : Array Json := #[]
    for e in items do
      out := out.push (← encExpr arena e)
    pure (Json.arr out)
  | .binary tag lhs rhs => do
    pure <| Json.mkObj [("op", Json.str tag.wire),
                        ("args", Json.arr #[← encExpr arena lhs, ← encExpr arena rhs])]
  | .unary tag arg => do
    pure <| Json.mkObj [("op", Json.str tag.wire),
                        ("args", Json.arr #[← encExpr arena arg])]
  | .clamp a b c => do
    pure <| Json.mkObj [("op", Json.str "clamp"),
      ("args", Json.arr #[← encExpr arena a, ← encExpr arena b, ← encExpr arena c])]
  | .select a b c => do
    pure <| Json.mkObj [("op", Json.str "select"),
      ("args", Json.arr #[← encExpr arena a, ← encExpr arena b, ← encExpr arena c])]
  | .arraySet a b c => do
    pure <| Json.mkObj [("op", Json.str "arraySet"),
      ("args", Json.arr #[← encExpr arena a, ← encExpr arena b, ← encExpr arena c])]
  | .index a b => do
    pure <| Json.mkObj [("op", Json.str "index"),
      ("args", Json.arr #[← encExpr arena a, ← encExpr arena b])]
  | .zeros count => do
    pure <| Json.mkObj [("op", Json.str "zeros"), ("count", ← encExpr arena count)]
  | .inputRef i => pure <| Json.mkObj [("op", Json.str "inputRef"), ("idx", Lean.toJson i.idx)]
  | .paramRef i => pure <| Json.mkObj [("op", Json.str "paramRef"), ("idx", Lean.toJson i.idx)]
  | .typeParamRef i => pure <| Json.mkObj [("op", Json.str "typeParamRef"), ("idx", Lean.toJson i.idx)]
  | .bindingRef i => pure <| Json.mkObj [("op", Json.str "bindingRef"), ("idx", Lean.toJson i.idx)]
  | .nestedOut inst out =>
    pure <| Json.mkObj [("op", Json.str "nestedOut"),
                        ("instance", Lean.toJson inst.idx), ("output", Lean.toJson out.idx)]
  | .sampleRate => pure <| Json.mkObj [("op", Json.str "sampleRate")]
  | .sampleIndex => pure <| Json.mkObj [("op", Json.str "sampleIndex")]
  | .loopIdx => pure <| Json.mkObj [("op", Json.str "loopIdx")]
  | .bankSum count tables body dynCount? => do
    let mut ts : Array Json := #[]
    for t in tables do
      ts := ts.push (← encExpr arena t)
    -- `dyn_count` is optional on the wire (absent = static bank), so pre-existing
    -- serialized programs decode unchanged.
    pure <| Json.mkObj <| [("op", Json.str "bankSum"), ("count", Lean.toJson count),
                        ("tables", Json.arr ts), ("body", ← encExpr arena body)]
                        ++ optField "dyn_count" (← dynCount?.mapM (encExpr arena))
  | .fold over init acc elem body => do
    pure <| Json.mkObj [("op", Json.str "fold"),
      ("over", ← encExpr arena over), ("init", ← encExpr arena init),
      ("acc", encBinder acc), ("elem", encBinder elem), ("body", ← encExpr arena body)]
  | .scan over init acc elem body => do
    pure <| Json.mkObj [("op", Json.str "scan"),
      ("over", ← encExpr arena over), ("init", ← encExpr arena init),
      ("acc", encBinder acc), ("elem", encBinder elem), ("body", ← encExpr arena body)]
  | .generate count iter body => do
    pure <| Json.mkObj [("op", Json.str "generate"), ("count", ← encExpr arena count),
      ("iter", encBinder iter), ("body", ← encExpr arena body)]
  | .iterate count init iter body => do
    pure <| Json.mkObj [("op", Json.str "iterate"),
      ("count", ← encExpr arena count), ("init", ← encExpr arena init),
      ("iter", encBinder iter), ("body", ← encExpr arena body)]
  | .chain count init iter body => do
    pure <| Json.mkObj [("op", Json.str "chain"),
      ("count", ← encExpr arena count), ("init", ← encExpr arena init),
      ("iter", encBinder iter), ("body", ← encExpr arena body)]
  | .map2 over elem body => do
    pure <| Json.mkObj [("op", Json.str "map2"), ("over", ← encExpr arena over),
      ("elem", encBinder elem), ("body", ← encExpr arena body)]
  | .zipWith a b x y body => do
    pure <| Json.mkObj [("op", Json.str "zipWith"),
      ("a", ← encExpr arena a), ("b", ← encExpr arena b),
      ("x", encBinder x), ("y", encBinder y), ("body", ← encExpr arena body)]
  | .letIn binders body => do
    let mut bs : Array Json := #[]
    for b in binders do
      bs := bs.push <| Json.mkObj [("binder", encBinder b.binder),
                                   ("value", ← encExpr arena b.value)]
    pure <| Json.mkObj [("op", Json.str "let"), ("binders", Json.arr bs),
                        ("in", ← encExpr arena body)]
  | .tag def_ variant payload => do
    let defId ← typeDefId arena def_
    let mut ps : Array Json := #[]
    for p in payload do
      ps := ps.push <| Json.mkObj [("field", Lean.toJson p.field),
                                   ("value", ← encExpr arena p.value)]
    pure <| Json.mkObj [("op", Json.str "tag"), ("def", Lean.toJson defId),
                        ("variant", Lean.toJson variant), ("payload", Json.arr ps)]
  | .match_ def_ scrutinee arms => do
    let defId ← typeDefId arena def_
    let scrut ← encExpr arena scrutinee
    let mut as_ : Array Json := #[]
    for arm in arms do
      as_ := as_.push <| Json.mkObj [
        ("variant", Lean.toJson arm.variant),
        ("binders", Json.arr (arm.binders.map encBinder)),
        ("body", ← encExpr arena arm.body)]
    pure <| Json.mkObj [("op", Json.str "match"), ("def", Lean.toJson defId),
                        ("scrutinee", scrut), ("arms", Json.arr as_)]

mutual

/-- Pool a program: registry targets first (insertion order), then the
    program's own fields in TS field order, then push self (post-order
    DFS — referenced programs strictly before referencing programs). -/
partial def programId (arena : Arena) (i : ProgramIdx) : EncM Nat := do
  let st ← get
  match st.programIds[i.idx]? with
  | none => encErr s!"program pool index {i.idx} out of range"
  | some (some id) => pure id
  | some none =>
    let some p := arena.program? i
      | encErr s!"program pool index {i.idx} out of range"
    let encoded ← encProgram arena p
    let id := (← get).programPool.size
    modify fun st => { st with
      programPool := st.programPool.push encoded
      programIds := st.programIds.set! i.idx (some id) }
    pure id

partial def encProgram (arena : Arena) (p : Program) : EncM Json := do
  -- Registry first — matches the TS encoder's statement order, which
  -- determines pool-id assignment for everything reachable.
  let mut registry : Array Json := #[]
  for (key, target) in p.registry do
    registry := registry.push <| Json.arr #[Json.str key, Lean.toJson (← programId arena target)]
  let mut typeParams : Array Json := #[]
  for tp in p.typeParams do
    typeParams := typeParams.push (Lean.toJson (← typeParamId arena tp))
  let mut inputs : Array Json := #[]
  for d in p.inputs do
    let mut fields : List (String × Json) := [("name", Json.str d.name)]
    match d.type? with
    | some t => fields := fields ++ [("type", ← encPortType arena t)]
    | none => pure ()
    match d.default? with
    | some e => fields := fields ++ [("default", ← encExpr arena e)]
    | none => pure ()
    inputs := inputs.push (Json.mkObj fields)
  let mut outputs : Array Json := #[]
  for d in p.outputs do
    let mut fields : List (String × Json) := [("name", Json.str d.name)]
    match d.type? with
    | some t => fields := fields ++ [("type", ← encPortType arena t)]
    | none => pure ()
    outputs := outputs.push (Json.mkObj fields)
  let mut typeDefs : Array Json := #[]
  for td in p.typeDefs do
    typeDefs := typeDefs.push (Lean.toJson (← typeDefId arena td))
  let mut decls : Array Json := #[]
  for d in p.decls do
    decls := decls.push (← encBodyDecl arena d)
  let mut assigns : Array Json := #[]
  for a in p.assigns do
    let target : Json := match a.target with
      | .port i => Lean.toJson i.idx
      | .dac => Json.mkObj [("kind", Json.str "dac")]
    assigns := assigns.push <| Json.mkObj [("target", target), ("expr", ← encExpr arena a.expr)]
  pure <| Json.mkObj [
    ("name", Json.str p.name),
    ("typeParams", Json.arr typeParams),
    ("inputs", Json.arr inputs),
    ("outputs", Json.arr outputs),
    ("typeDefs", Json.arr typeDefs),
    ("decls", Json.arr decls),
    ("assigns", Json.arr assigns),
    ("binderCount", Lean.toJson p.binderCount),
    ("registry", Json.arr registry)]

partial def encBodyDecl (arena : Arena) : BodyDecl → EncM Json
  | .param name value? =>
    pure <| Json.mkObj <|
      [("op", Json.str "paramDecl"), ("name", Json.str name)]
      ++ optField "value" (value?.map Json.num)
  | .inst name typeKey typeArgs inputs => do
    let mut tas : Array Json := #[]
    for a in typeArgs do
      tas := tas.push <| Json.mkObj [("param", Lean.toJson a.param.idx), ("value", Json.num a.value)]
    let mut ins : Array Json := #[]
    for w in inputs do
      ins := ins.push <| Json.mkObj [("port", Lean.toJson w.port.idx),
                                     ("value", ← encExpr arena w.value)]
    pure <| Json.mkObj [
      ("op", Json.str "instanceDecl"), ("name", Json.str name),
      ("typeKey", Json.str typeKey),
      ("typeArgs", Json.arr tas), ("inputs", Json.arr ins)]
  | .prog name program => do
    pure <| Json.mkObj [("op", Json.str "programDecl"), ("name", Json.str name),
                        ("program", Lean.toJson (← programId arena program))]

end

end Encode

/-- Encode an arena + root into `tropical_resolved_1` wire JSON,
    performing the TS encoder's canonical pool reordering. -/
def encodeResolved (arena : Arena) (root : ProgramIdx) : Except String Json := do
  let init : Encode.St := {
    typeParamIds := Array.replicate arena.typeParams.size none
    typeDefIds := Array.replicate arena.typeDefs.size none
    programIds := Array.replicate arena.programs.size none }
  let (rootId, st) ← (Encode.programId arena root).run init
  pure <| Json.mkObj [
    ("schema", Json.str schemaTag),
    ("typeParamPool", Json.arr st.typeParamPool),
    ("typeDefPool", Json.arr st.typeDefPool),
    ("programPool", Json.arr st.programPool),
    ("root", Lean.toJson rootId)]

-- ─────────────────────────────────────────────────────────────
-- Decode (wire pool order becomes the arena order)
-- ─────────────────────────────────────────────────────────────

namespace Decode

private def err {α} (ctx msg : String) : Except String α :=
  .error s!"decodeResolved: {ctx}: {msg}"

private def reqField (ctx : String) (j : JsonV) (k : String) : Except String JsonV :=
  match j.getField? k with
  | some v => pure v
  | none => err ctx s!"missing field '{k}'"

private def reqStr (ctx : String) (j : JsonV) (k : String) : Except String String := do
  match ← reqField ctx j k with
  | .str s => pure s
  | _ => err ctx s!"field '{k}' must be a string"

private def reqArr (ctx : String) (j : JsonV) (k : String) : Except String (Array JsonV) := do
  match ← reqField ctx j k with
  | .arr a => pure a
  | _ => err ctx s!"field '{k}' must be an array"

private def reqNat (ctx : String) (j : JsonV) (k : String) : Except String Nat := do
  match ← reqField ctx j k with
  | .num n =>
    if n.exponent == 0 && n.mantissa ≥ 0 then pure n.mantissa.toNat
    else err ctx s!"field '{k}' must be a non-negative integer"
  | _ => err ctx s!"field '{k}' must be a number"

private def optNum (ctx : String) (j : JsonV) (k : String) :
    Except String (Option Lean.JsonNumber) :=
  match j.getField? k with
  | none => pure none
  | some (.num n) => pure (some n)
  | some _ => err ctx s!"field '{k}' must be a number"

private def scalarKind (ctx : String) (s : String) : Except String ScalarKind :=
  match Tropical.Parse.ScalarKind.ofWire? s with
  | some k => pure k
  | none => err ctx s!"unknown scalar kind '{s}'"

/-- `ScalarKind | {alias: n}`. Wire pool indices are relative to the
    decode batch; `tdBase` offsets them into the (possibly pre-seeded)
    arena pool. The alias index must point strictly earlier in the wire
    pool (forward-pass contract); range is validated against the count
    decoded so far — entries below `tdBase` belong to earlier decodes
    and are unreachable from this batch's (non-negative) wire indices. -/
private def scalarOrAlias (ctx : String) (j : JsonV) (typeDefs : Array TypeDef)
    (tdBase : Nat) : Except String ScalarOrAlias := do
  match j with
  | .str s => pure (.scalar (← scalarKind ctx s))
  | .obj _ =>
    let idx ← reqNat ctx j "alias"
    match typeDefs[tdBase + idx]? with
    | some (.alias ..) => pure (.alias ⟨tdBase + idx⟩)
    | some td => err ctx s!"typeDef '{td.name}' is not an alias"
    | none => err ctx s!"typeDef pool index {idx} is out of range (decoded so far: {typeDefs.size - tdBase})"
  | _ => err ctx "expected a scalar kind or {alias} record"

private def decodeTypeDef (i : Nat) (j : JsonV) (typeDefs : Array TypeDef)
    (tdBase : Nat) : Except String TypeDef := do
  let ctx := s!"typeDefPool[{i}]"
  let kind ← reqStr ctx j "kind"
  let name ← reqStr ctx j "name"
  let field (fctx : String) (f : JsonV) : Except String StructField := do
    pure { name := ← reqStr fctx f "name"
           type := ← scalarOrAlias s!"{fctx}.type" (← reqField fctx f "type") typeDefs tdBase }
  match kind with
  | "alias" => pure (.alias name (← scalarKind s!"{ctx}.base" (← reqStr ctx j "base")))
  | "struct" => do
    let mut fields : Array StructField := #[]
    for h : k in [0:(← reqArr ctx j "fields").size] do
      let fs ← reqArr ctx j "fields"
      fields := fields.push (← field s!"{ctx}.fields[{k}]" fs[k]!)
    pure (.struct name fields)
  | "sum" => do
    let vs ← reqArr ctx j "variants"
    let mut variants : Array SumVariant := #[]
    for h : k in [0:vs.size] do
      let v := vs[k]
      let vctx := s!"{ctx}.variants[{k}]"
      let vName ← reqStr vctx v "name"
      let ps ← reqArr vctx v "payload"
      let mut payload : Array StructField := #[]
      for h2 : m in [0:ps.size] do
        payload := payload.push (← field s!"{vctx}.payload[{m}]" ps[m])
      variants := variants.push { name := vName, payload }
    pure (.sum name variants)
  | other => err ctx s!"unknown typeDef kind '{other}'"

private def binder (ctx : String) (j : JsonV) : Except String Binder := do
  pure { name := ← reqStr ctx j "name", idx := ⟨← reqNat ctx j "idx"⟩ }

/-- The decode monad interns each node into the shared expression DAG as it is
    parsed, so a decoded `Program` is id-valued like the elaborator's output. -/
private abbrev DecM := StateT ExprArena (Except String)

private def internD (n : ENode) : DecM ExprId := fun a => .ok ((eintern n).run a)

/-- Parse + intern a `tropical_resolved_1` expression, returning its arena id.
    (Byte-round-trips with `encExpr` above.) The `req*`/`err`/`binder` helpers
    return `Except String` and auto-lift into `DecM`. -/
private partial def expr (ctx : String) (j : JsonV) (tdBase tdCount : Nat) :
    DecM ExprId := do
  match j with
  | .num n => internD (.num n)
  | .bool b => internD (.bool b)
  | .arr items => do
    let mut out : Array ExprId := #[]
    for h : i in [0:items.size] do
      out := out.push (← expr s!"{ctx}[{i}]" items[i] tdBase tdCount)
    internD (.arr out)
  | .obj _ =>
    let some op := j.opOf? | err ctx "expression object missing string 'op'"
    let args (n : Nat) : Except String (Array JsonV) := do
      let a ← reqArr ctx j "args"
      if a.size != n then err ctx s!"'{op}' expects {n} args, got {a.size}"
      else pure a
    let sub (k : String) : DecM ExprId := do
      expr s!"{ctx}.{k}" (← reqField ctx j k) tdBase tdCount
    let defIdx : Except String TypeDefIdx := do
      let d ← reqNat ctx j "def"
      if d < tdCount then pure ⟨tdBase + d⟩
      else err ctx s!"typeDef pool index {d} is out of range"
    if let some tag := BinaryOpTag.ofWire? op then
      let a ← args 2
      internD (.binary tag (← expr s!"{ctx}.args[0]" a[0]! tdBase tdCount)
                           (← expr s!"{ctx}.args[1]" a[1]! tdBase tdCount))
    else if let some tag := UnaryOpTag.ofWire? op then
      let a ← args 1
      internD (.unary tag (← expr s!"{ctx}.args[0]" a[0]! tdBase tdCount))
    else match op with
    | "clamp" => do
      let a ← args 3
      internD (.clamp (← expr s!"{ctx}.args[0]" a[0]! tdBase tdCount)
                      (← expr s!"{ctx}.args[1]" a[1]! tdBase tdCount)
                      (← expr s!"{ctx}.args[2]" a[2]! tdBase tdCount))
    | "select" => do
      let a ← args 3
      internD (.select (← expr s!"{ctx}.args[0]" a[0]! tdBase tdCount)
                       (← expr s!"{ctx}.args[1]" a[1]! tdBase tdCount)
                       (← expr s!"{ctx}.args[2]" a[2]! tdBase tdCount))
    | "arraySet" => do
      let a ← args 3
      internD (.arraySet (← expr s!"{ctx}.args[0]" a[0]! tdBase tdCount)
                         (← expr s!"{ctx}.args[1]" a[1]! tdBase tdCount)
                         (← expr s!"{ctx}.args[2]" a[2]! tdBase tdCount))
    | "index" => do
      let a ← args 2
      internD (.index (← expr s!"{ctx}.args[0]" a[0]! tdBase tdCount)
                      (← expr s!"{ctx}.args[1]" a[1]! tdBase tdCount))
    | "zeros" => internD (.zeros (← sub "count"))
    | "inputRef" => internD (.inputRef ⟨← reqNat ctx j "idx"⟩)
    | "paramRef" => internD (.paramRef ⟨← reqNat ctx j "idx"⟩)
    | "typeParamRef" => internD (.typeParamRef ⟨← reqNat ctx j "idx"⟩)
    | "bindingRef" => internD (.bindingRef ⟨← reqNat ctx j "idx"⟩)
    | "nestedOut" =>
      internD (.nestedOut ⟨← reqNat ctx j "instance"⟩ ⟨← reqNat ctx j "output"⟩)
    | "sampleRate" => internD .sampleRate
    | "sampleIndex" => internD .sampleIndex
    | "loopIdx" => internD .loopIdx
    | "bankSum" => do
      let ts ← reqArr ctx j "tables"
      let mut tables : Array ExprId := #[]
      for h : i in [0:ts.size] do
        tables := tables.push (← expr s!"{ctx}.tables[{i}]" ts[i] tdBase tdCount)
      -- optional runtime effective count (trip-count-as-data); absent = static.
      let dc? ← match j.getField? "dyn_count" with
        | some dj => some <$> expr s!"{ctx}.dyn_count" dj tdBase tdCount
        | none => pure none
      internD (.bankSum (← reqNat ctx j "count") tables (← sub "body") dc?)
    | "fold" => do
      internD (.fold (← sub "over") (← sub "init")
        (← binder s!"{ctx}.acc" (← reqField ctx j "acc"))
        (← binder s!"{ctx}.elem" (← reqField ctx j "elem")) (← sub "body"))
    | "scan" => do
      internD (.scan (← sub "over") (← sub "init")
        (← binder s!"{ctx}.acc" (← reqField ctx j "acc"))
        (← binder s!"{ctx}.elem" (← reqField ctx j "elem")) (← sub "body"))
    | "generate" => do
      internD (.generate (← sub "count")
        (← binder s!"{ctx}.iter" (← reqField ctx j "iter")) (← sub "body"))
    | "iterate" => do
      internD (.iterate (← sub "count") (← sub "init")
        (← binder s!"{ctx}.iter" (← reqField ctx j "iter")) (← sub "body"))
    | "chain" => do
      internD (.chain (← sub "count") (← sub "init")
        (← binder s!"{ctx}.iter" (← reqField ctx j "iter")) (← sub "body"))
    | "map2" => do
      internD (.map2 (← sub "over")
        (← binder s!"{ctx}.elem" (← reqField ctx j "elem")) (← sub "body"))
    | "zipWith" => do
      internD (.zipWith (← sub "a") (← sub "b")
        (← binder s!"{ctx}.x" (← reqField ctx j "x"))
        (← binder s!"{ctx}.y" (← reqField ctx j "y")) (← sub "body"))
    | "let" => do
      let bs ← reqArr ctx j "binders"
      let mut binders : Array ELetBinder := #[]
      for h : i in [0:bs.size] do
        let b := bs[i]
        let bctx := s!"{ctx}.binders[{i}]"
        binders := binders.push {
          binder := ← binder s!"{bctx}.binder" (← reqField bctx b "binder"),
          value := ← expr s!"{bctx}.value" (← reqField bctx b "value") tdBase tdCount }
      internD (.letIn binders (← sub "in"))
    | "tag" => do
      let d ← defIdx
      let variant ← reqNat ctx j "variant"
      let ps ← reqArr ctx j "payload"
      let mut payload : Array ETagPayload := #[]
      for h : i in [0:ps.size] do
        let p := ps[i]
        let pctx := s!"{ctx}.payload[{i}]"
        payload := payload.push {
          field := ← reqNat pctx p "field",
          value := ← expr s!"{pctx}.value" (← reqField pctx p "value") tdBase tdCount }
      internD (.tag d variant payload)
    | "match" => do
      let d ← defIdx
      let scrutinee ← sub "scrutinee"
      let as_ ← reqArr ctx j "arms"
      let mut arms : Array EMatchArm := #[]
      for h : i in [0:as_.size] do
        let a := as_[i]
        let actx := s!"{ctx}.arms[{i}]"
        let bs ← reqArr actx a "binders"
        let mut binders : Array Binder := #[]
        for h2 : k in [0:bs.size] do
          binders := binders.push (← binder s!"{actx}.binders[{k}]" bs[k])
        arms := arms.push {
          variant := ← reqNat actx a "variant", binders,
          body := ← expr s!"{actx}.body" (← reqField actx a "body") tdBase tdCount }
      internD (.match_ d scrutinee arms)
    | other => err ctx s!"unknown expression op '{other}'"
  | _ => err ctx "expected an expression value"

private def decodePortType (ctx : String) (j : JsonV) (typeDefs : Array TypeDef)
    (tdBase tpBase tpCount : Nat) : Except String PortType := do
  let kind ← reqStr ctx j "kind"
  match kind with
  | "scalar" => pure (.scalar (← scalarKind s!"{ctx}.scalar" (← reqStr ctx j "scalar")))
  | "alias" => do
    let idx ← reqNat ctx j "alias"
    match typeDefs[tdBase + idx]? with
    | some (.alias ..) => pure (.alias ⟨tdBase + idx⟩)
    | some td => err ctx s!"typeDef '{td.name}' is not an alias"
    | none => err ctx s!"typeDef pool index {idx} out of range"
  | "array" => do
    let element ← scalarOrAlias s!"{ctx}.element" (← reqField ctx j "element") typeDefs tdBase
    let dims ← reqArr ctx j "shape"
    let mut shape : Array ShapeDim := #[]
    for h : i in [0:dims.size] do
      match dims[i] with
      | .num n => shape := shape.push (.lit n)
      | d@(.obj _) =>
        let tp ← reqNat s!"{ctx}.shape[{i}]" d "typeParam"
        if tp < tpCount then shape := shape.push (.typeParam ⟨tpBase + tp⟩)
        else err s!"{ctx}.shape[{i}]" s!"typeParam pool index {tp} out of range"
      | _ => err s!"{ctx}.shape[{i}]" "expected a number or {typeParam} record"
    pure (.array element shape)
  | other => err ctx s!"unknown port-type kind '{other}'"

private def decodeProgram (i : Nat) (j : JsonV) (typeDefs : Array TypeDef)
    (tdBase tpBase tpCount pBase programsSoFar : Nat) : DecM Program := do
  let ctx := s!"programPool[{i}]"
  let tdCount := typeDefs.size - tdBase
  let name ← reqStr ctx j "name"
  let progRef (rctx : String) (n : Nat) : Except String ProgramIdx :=
    if n < programsSoFar then pure ⟨pBase + n⟩
    else err rctx s!"program pool index {n} is out of range (decoded so far: {programsSoFar}); forward references violate the post-order pool contract"

  let mut typeParams : Array TypeParamPoolIdx := #[]
  for h : k in [0:(← reqArr ctx j "typeParams").size] do
    let tps ← reqArr ctx j "typeParams"
    match tps[k]! with
    | .num n =>
      let idx := n.mantissa.toNat
      if n.exponent == 0 && idx < tpCount then typeParams := typeParams.push ⟨tpBase + idx⟩
      else err s!"{ctx}.typeParams[{k}]" s!"typeParam pool index out of range"
    | _ => err s!"{ctx}.typeParams[{k}]" "expected a pool index"

  let mut inputs : Array InputDecl := #[]
  for h : k in [0:(← reqArr ctx j "inputs").size] do
    let ds ← reqArr ctx j "inputs"
    let d := ds[k]!
    let dctx := s!"{ctx}.inputs[{k}]"
    let type? ← match d.getField? "type" with
      | none => pure none
      | some t => pure (some (← decodePortType s!"{dctx}.type" t typeDefs tdBase tpBase tpCount))
    let default? ← match d.getField? "default" with
      | none => pure none
      | some e => pure (some (← expr s!"{dctx}.default" e tdBase tdCount))
    inputs := inputs.push { name := ← reqStr dctx d "name", type?, default? }

  let mut outputs : Array OutputDecl := #[]
  for h : k in [0:(← reqArr ctx j "outputs").size] do
    let ds ← reqArr ctx j "outputs"
    let d := ds[k]!
    let dctx := s!"{ctx}.outputs[{k}]"
    let type? ← match d.getField? "type" with
      | none => pure none
      | some t => pure (some (← decodePortType s!"{dctx}.type" t typeDefs tdBase tpBase tpCount))
    outputs := outputs.push { name := ← reqStr dctx d "name", type? }

  let mut typeDefRefs : Array TypeDefIdx := #[]
  for h : k in [0:(← reqArr ctx j "typeDefs").size] do
    let tds ← reqArr ctx j "typeDefs"
    match tds[k]! with
    | .num n =>
      let idx := n.mantissa.toNat
      if n.exponent == 0 && idx < tdCount then typeDefRefs := typeDefRefs.push ⟨tdBase + idx⟩
      else err s!"{ctx}.typeDefs[{k}]" "typeDef pool index out of range"
    | _ => err s!"{ctx}.typeDefs[{k}]" "expected a pool index"

  let mut decls : Array BodyDecl := #[]
  for h : k in [0:(← reqArr ctx j "decls").size] do
    let ds ← reqArr ctx j "decls"
    let d := ds[k]!
    let dctx := s!"{ctx}.decls[{k}]"
    let op ← reqStr dctx d "op"
    let dname ← reqStr dctx d "name"
    match op with
    | "paramDecl" =>
      decls := decls.push (.param dname (← optNum dctx d "value"))
    | "instanceDecl" => do
      let typeKey ← reqStr dctx d "typeKey"
      let tas ← reqArr dctx d "typeArgs"
      let mut typeArgs : Array InstanceTypeArg := #[]
      for h2 : m in [0:tas.size] do
        let a := tas[m]
        let actx := s!"{dctx}.typeArgs[{m}]"
        let value ← match a.getField? "value" with
          | some (.num n) => pure n
          | _ => err actx "field 'value' must be a number"
        typeArgs := typeArgs.push { param := ⟨← reqNat actx a "param"⟩, value }
      let ins ← reqArr dctx d "inputs"
      let mut instInputs : Array InstanceInput := #[]
      for h2 : m in [0:ins.size] do
        let w := ins[m]
        let wctx := s!"{dctx}.inputs[{m}]"
        instInputs := instInputs.push {
          port := ⟨← reqNat wctx w "port"⟩
          value := ← expr s!"{wctx}.value" (← reqField wctx w "value") tdBase tdCount }
      decls := decls.push (.inst dname typeKey typeArgs instInputs)
    | "programDecl" =>
      decls := decls.push (.prog dname (← progRef s!"{dctx}.program" (← reqNat dctx d "program")))
    | other => err dctx s!"unknown body-decl op '{other}'"

  let mut assigns : Array OutputAssign := #[]
  for h : k in [0:(← reqArr ctx j "assigns").size] do
    let as_ ← reqArr ctx j "assigns"
    let a := as_[k]!
    let actx := s!"{ctx}.assigns[{k}]"
    let target ← match a.getField? "target" with
      | some (.num n) =>
        if n.exponent == 0 && n.mantissa ≥ 0 then pure (OutputTarget.port ⟨n.mantissa.toNat⟩)
        else err actx "target must be a non-negative integer or {kind:'dac'}"
      | some t@(.obj _) =>
        if t.getStr? "kind" == some "dac" then pure OutputTarget.dac
        else err actx "unknown target kind"
      | _ => err actx "missing or malformed 'target'"
    assigns := assigns.push { target, expr := ← expr s!"{actx}.expr" (← reqField actx a "expr") tdBase tdCount }

  let mut registry : Array (String × ProgramIdx) := #[]
  for h : k in [0:(← reqArr ctx j "registry").size] do
    let rs ← reqArr ctx j "registry"
    match rs[k]! with
    | .arr pair =>
      if pair.size != 2 then err s!"{ctx}.registry[{k}]" "expected a [key, poolIdx] pair"
      else
        let key ← match pair[0]! with
          | .str s => pure s
          | _ => err s!"{ctx}.registry[{k}][0]" "expected a string key"
        let idx ← match pair[1]! with
          | .num n =>
            if n.exponent == 0 && n.mantissa ≥ 0 then pure n.mantissa.toNat
            else err s!"{ctx}.registry[{k}][1]" "expected a pool index"
          | _ => err s!"{ctx}.registry[{k}][1]" "expected a pool index"
        registry := registry.push (key, ← progRef s!"{ctx}.registry[{k}]" idx)
    | _ => err s!"{ctx}.registry[{k}]" "expected a [key, poolIdx] pair"

  pure { name, typeParams, inputs, outputs, typeDefs := typeDefRefs,
         decls, assigns, binderCount := ← reqNat ctx j "binderCount", registry }

end Decode

/-- Decode `tropical_resolved_1` wire JSON, **appending** into an
    existing arena. The wire's pool indices are batch-relative; the
    lower-index invariant makes the rebase a mechanical shift by the
    base pool sizes. Each call appends a self-contained copy — wire
    indices cannot reach entries below the bases, and nothing already
    in the arena is touched. -/
def decodeResolvedInto (arena : Arena) (j : JsonV) :
    Except String (Arena × ProgramIdx) := do
  let schema ← match j.getField? "schema" with
    | some (.str s) => pure s
    | _ => .error "decodeResolved: missing 'schema'"
  if schema != schemaTag then
    .error s!"decodeResolved: unsupported schema '{schema}' (expected '{schemaTag}')"

  let pool (k : String) : Except String (Array JsonV) :=
    match j.getField? k with
    | some (.arr a) => pure a
    | _ => .error s!"decodeResolved: {k}: expected an array"

  let tpBase := arena.typeParams.size
  let tdBase := arena.typeDefs.size
  let pBase  := arena.programs.size

  -- 1. Type params — leaves.
  let mut typeParams : Array TypeParamDecl := arena.typeParams
  for i in [0:(← pool "typeParamPool").size] do
    let tps ← pool "typeParamPool"
    let r := tps[i]!
    let ctx := s!"typeParamPool[{i}]"
    let name ← match r.getField? "name" with
      | some (.str s) => pure s
      | _ => .error s!"decodeResolved: {ctx}: missing string 'name'"
    let default? ← match r.getField? "default" with
      | none => pure none
      | some (.num n) => pure (some n)
      | some _ => .error s!"decodeResolved: {ctx}: 'default' must be a number"
    typeParams := typeParams.push { name, default? }
  let tpCount := typeParams.size - tpBase

  -- 2. Type defs — single forward pass.
  let mut typeDefs : Array TypeDef := arena.typeDefs
  for i in [0:(← pool "typeDefPool").size] do
    let tds ← pool "typeDefPool"
    typeDefs := typeDefs.push (← Decode.decodeTypeDef i tds[i]! typeDefs tdBase)

  -- 3. Programs — single forward pass, interning expressions into the shared
  -- DAG (seeded from the existing arena's `exprs`).
  let mut programs : Array Program := arena.programs
  let mut exprs : ExprArena := arena.exprs
  for i in [0:(← pool "programPool").size] do
    let ps ← pool "programPool"
    let (p, exprs') ← (Decode.decodeProgram i ps[i]! typeDefs tdBase tpBase tpCount
           pBase (programs.size - pBase)).run exprs
    programs := programs.push p
    exprs := exprs'

  let root ← match j.getField? "root" with
    | some (.num n) =>
      if n.exponent == 0 && n.mantissa.toNat < programs.size - pBase then
        pure (ProgramIdx.mk (pBase + n.mantissa.toNat))
      else .error "decodeResolved: root: program pool index out of range"
    | _ => .error "decodeResolved: missing 'root'"

  pure ({ typeParams, typeDefs, programs, exprs }, root)

/-- Decode `tropical_resolved_1` wire JSON into a fresh arena + root,
    keeping the wire pool order as the arena order. -/
def decodeResolved (j : JsonV) : Except String (Arena × ProgramIdx) :=
  decodeResolvedInto {} j

end Tropical.Ir.Codec
