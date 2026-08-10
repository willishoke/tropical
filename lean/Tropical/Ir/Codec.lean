import Tropical.Ir.Nodes
import Tropical.Parse.OrderedJson

/-!
# Resolved⇄JSON codec

Schema `tropical_resolved_1`: the program pool plus a root program
index. (The typeParam/typeDef pools left with generics and the
sum-type lowering — the wire carries only trunk structure.)

**Decode** rebuilds an `Arena` keeping the wire pool order as the arena
order, in a single forward pass — the encoder's topological guarantee
(programs in post-order DFS from the root) means every reference points
strictly earlier in the pool, and a forward reference is a decode error.

**Encode** assigns pool ids on *first reference* during a deterministic
traversal — for each program: registry entries (in insertion order,
recursing into targets), then `inputs`, `outputs`, `decls` (nested
`programDecl` targets recurse here), `assigns`; the program itself is
pushed *after* its referents (post-order).

Optional fields are absent, never null. Numbers are `JsonNumber`
(decimal text preserved).
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
  programPool : Array Json := #[]
  /-- arena index → assigned pool id (none = not yet pooled). -/
  programIds : Array (Option Nat)

abbrev EncM := StateT St (Except String)

private def encErr {α} (msg : String) : EncM α :=
  throw s!"encodeResolved: {msg}"

private def optField (key : String) : Option Json → List (String × Json)
  | some v => [(key, v)]
  | none => []

private def scalarJson (k : ScalarKind) : Json := Json.str k.wire

def encPortType : PortType → Json
  | .scalar k =>
    Json.mkObj [("kind", Json.str "scalar"), ("scalar", scalarJson k)]
  | .array element shape =>
    Json.mkObj [("kind", Json.str "array"), ("element", scalarJson element),
                ("shape", Json.arr (shape.map Json.num))]

/-- Encode a resolved expression to `tropical_resolved_1` JSON, derefing the
    id-form through `arena.exprs`. Total by descent on `id.idx` (`hw` is
    checked once by `encodeResolved`). -/
def encExpr (arena : Arena) (hw : arena.exprs.wf = true) (id : ExprId) : EncM Json := do
  match _hd : arena.exprs.deref id with
  | none => encErr s!"encExpr: dangling ExprId {id.idx}"
  | some (.num n) => pure (Json.num n)
  | some (.bool b) => pure (Json.bool b)
  | some (.arr items) => do
    let out ← items.attach.mapM fun ⟨e, _⟩ => encExpr arena hw e
    pure (Json.arr out)
  | some (.binary tag lhs rhs) => do
    pure <| Json.mkObj [("op", Json.str tag.wire),
                        ("args", Json.arr #[← encExpr arena hw lhs, ← encExpr arena hw rhs])]
  | some (.unary tag arg) => do
    pure <| Json.mkObj [("op", Json.str tag.wire),
                        ("args", Json.arr #[← encExpr arena hw arg])]
  | some (.clamp a b c) => do
    pure <| Json.mkObj [("op", Json.str "clamp"),
      ("args", Json.arr #[← encExpr arena hw a, ← encExpr arena hw b, ← encExpr arena hw c])]
  | some (.select a b c) => do
    pure <| Json.mkObj [("op", Json.str "select"),
      ("args", Json.arr #[← encExpr arena hw a, ← encExpr arena hw b, ← encExpr arena hw c])]
  | some (.arraySet a b c) => do
    pure <| Json.mkObj [("op", Json.str "arraySet"),
      ("args", Json.arr #[← encExpr arena hw a, ← encExpr arena hw b, ← encExpr arena hw c])]
  | some (.index a b) => do
    pure <| Json.mkObj [("op", Json.str "index"),
      ("args", Json.arr #[← encExpr arena hw a, ← encExpr arena hw b])]
  | some (.inputRef i) => pure <| Json.mkObj [("op", Json.str "inputRef"), ("idx", Lean.toJson i.idx)]
  | some (.paramRef i) => pure <| Json.mkObj [("op", Json.str "paramRef"), ("idx", Lean.toJson i.idx)]
  | some (.nestedOut inst out) =>
    pure <| Json.mkObj [("op", Json.str "nestedOut"),
                        ("instance", Lean.toJson inst.idx), ("output", Lean.toJson out.idx)]
  | some .sampleRate => pure <| Json.mkObj [("op", Json.str "sampleRate")]
  | some .sampleIndex => pure <| Json.mkObj [("op", Json.str "sampleIndex")]
  -- `id` omitted when 0 so pre-nesting programs serialize byte-identically.
  | some (.loopIdx id) =>
    let idField : Option Json := if id == 0 then none else some (Lean.toJson id)
    pure <| Json.mkObj <| [("op", Json.str "loopIdx")] ++ optField "id" idField
  | some (.bankSum count tables body dynCount? idxId) => do
    let ts ← tables.attach.mapM fun ⟨t, _⟩ => encExpr arena hw t
    -- `dyn_count`/`idx_id` are optional on the wire (absent = static bank /
    -- binder id 0), so pre-existing serialized programs decode unchanged.
    let dynField : Option Json ← match _hdo : dynCount? with
      | none => pure none
      | some d => pure (some (← encExpr arena hw d))
    let idField : Option Json := if idxId == 0 then none else some (Lean.toJson idxId)
    pure <| Json.mkObj <| [("op", Json.str "bankSum"), ("count", Lean.toJson count),
                        ("tables", Json.arr ts), ("body", ← encExpr arena hw body)]
                        ++ optField "dyn_count" dynField
                        ++ optField "idx_id" idField
  | some (.routedSum capacity outputCount routes tables values dynCount? idxId) => do
    let ts ← tables.attach.mapM fun ⟨t, _⟩ => encExpr arena hw t
    let vs ← values.attach.mapM fun ⟨v, _⟩ => encExpr arena hw v
    let routeJson := routes.map fun route => match route with
      | none => Json.null
      | some output => Lean.toJson output
    let dynField : Option Json ← match _hdo : dynCount? with
      | none => pure none
      | some d => pure (some (← encExpr arena hw d))
    let idField : Option Json := if idxId == 0 then none else some (Lean.toJson idxId)
    pure <| Json.mkObj <|
      [("op", Json.str "routedSum"), ("capacity", Lean.toJson capacity),
       ("output_count", Lean.toJson outputCount), ("routes", Json.arr routeJson),
       ("tables", Json.arr ts), ("values", Json.arr vs)]
      ++ optField "dyn_count" dynField ++ optField "idx_id" idField
termination_by id.idx
decreasing_by
  all_goals
    apply Tropical.Ir.ExprArena.forall_children_lt hw ‹_ = some _›
    simp_all [ENode.children] <;>
      first
        | exact Or.inl ⟨_, by assumption, rfl⟩
        | exact Or.inr ⟨_, by assumption, rfl⟩
        | exact ⟨_, by assumption, rfl⟩

/- TOTAL: the recursion runs through the PROGRAM pool via registry
   indices and nested programDecl links; `hwp` (checked once by
   `encodeResolved`) says every pool edge points strictly down, so the
   DFS descends on the pool index. The measure is lexicographic
   `(idx, phase)`: programId at phase 2 resolves its program and hands
   the children bound (`progPool_children_lt`) to encProgram at phase 1,
   whose registry follows and per-decl encodes (phase 0) re-enter
   programId at a strictly smaller idx. -/
mutual

/-- Pool a program: registry targets first (insertion order), then the
    program's own fields in field order, then push self (post-order
    DFS — referenced programs strictly before referencing programs). -/
def programId (arena : Arena) (hw : arena.exprs.wf = true)
    (hwp : progPoolWf arena.programs = true) (i : ProgramIdx) : EncM Nat := do
  let st ← get
  match st.programIds[i.idx]? with
  | none => encErr s!"program pool index {i.idx} out of range"
  | some (some id) => pure id
  | some none =>
    match hp : arena.programs[i.idx]? with
    | none => encErr s!"program pool index {i.idx} out of range"
    | some p => do
      let encoded ← encProgram arena hw hwp i.idx p (progPool_children_lt hwp hp)
      let id := (← get).programPool.size
      modify fun st => { st with
        programPool := st.programPool.push encoded
        programIds := st.programIds.set! i.idx (some id) }
      pure id
termination_by (i.idx, 2)

def encProgram (arena : Arena) (hw : arena.exprs.wf = true)
    (hwp : progPoolWf arena.programs = true) (bound : Nat) (p : Program)
    (hch : ∀ c ∈ p.progChildren, c.idx < bound) : EncM Json := do
  -- Registry first — pool-id assignment order for everything reachable.
  -- Collect the targets with their decrease facts, then pool in order.
  let mut regTargets : Array (String × {t : ProgramIdx // t.idx < bound}) := #[]
  for kt in p.registry.attach do
    regTargets := regTargets.push (kt.1.1,
      ⟨kt.1.2, hch kt.1.2 (Array.mem_append.mpr (Or.inl
        (Array.mem_map.mpr ⟨kt.1, kt.2, rfl⟩)))⟩)
  let registry ← regTargets.mapM
    fun (kt : String × {t : ProgramIdx // t.idx < bound}) => do
      have hlt : kt.2.1.idx < bound := kt.2.2
      pure <| Json.arr #[Json.str kt.1, Lean.toJson (← programId arena hw hwp kt.2.1)]
  let mut inputs : Array Json := #[]
  for d in p.inputs do
    let mut fields : List (String × Json) := [("name", Json.str d.name)]
    match d.type? with
    | some t => fields := fields ++ [("type", encPortType t)]
    | none => pure ()
    match d.default? with
    | some e => fields := fields ++ [("default", ← encExpr arena hw e)]
    | none => pure ()
    inputs := inputs.push (Json.mkObj fields)
  let mut outputs : Array Json := #[]
  for d in p.outputs do
    let mut fields : List (String × Json) := [("name", Json.str d.name)]
    match d.type? with
    | some t => fields := fields ++ [("type", encPortType t)]
    | none => pure ()
    outputs := outputs.push (Json.mkObj fields)
  let decls ← p.decls.attach.mapM fun dm =>
    encBodyDecl arena hw hwp bound dm.1 fun name t hdt =>
      hch t (Array.mem_append.mpr (Or.inr
        (Array.mem_filterMap.mpr ⟨dm.1, dm.2, by simp [hdt]⟩)))
  let mut assigns : Array Json := #[]
  for a in p.assigns do
    let target : Json := match a.target with
      | .port i => Lean.toJson i.idx
      | .dac => Json.mkObj [("kind", Json.str "dac")]
    assigns := assigns.push <| Json.mkObj [("target", target), ("expr", ← encExpr arena hw a.expr)]
  pure <| Json.mkObj [
    ("name", Json.str p.name),
    ("inputs", Json.arr inputs),
    ("outputs", Json.arr outputs),
    ("decls", Json.arr decls),
    ("assigns", Json.arr assigns),
    ("registry", Json.arr registry)]
termination_by (bound, 1)

def encBodyDecl (arena : Arena) (hw : arena.exprs.wf = true)
    (hwp : progPoolWf arena.programs = true) (bound : Nat) (d : BodyDecl)
    (hpd : ∀ name t, d = .prog name t → t.idx < bound) : EncM Json := do
  match _hn : d with
  | .param name value? =>
    pure <| Json.mkObj <|
      [("op", Json.str "paramDecl"), ("name", Json.str name)]
      ++ optField "value" (value?.map Json.num)
  | .inst name typeKey inputs => do
    let mut ins : Array Json := #[]
    for w in inputs do
      ins := ins.push <| Json.mkObj [("port", Lean.toJson w.port.idx),
                                     ("value", ← encExpr arena hw w.value)]
    pure <| Json.mkObj [
      ("op", Json.str "instanceDecl"), ("name", Json.str name),
      ("typeKey", Json.str typeKey), ("inputs", Json.arr ins)]
  | .prog name program => do
    have _hlt : program.idx < bound := hpd name program rfl
    pure <| Json.mkObj [("op", Json.str "programDecl"), ("name", Json.str name),
                        ("program", Lean.toJson (← programId arena hw hwp program))]
termination_by (bound, 0)

end

end Encode

/-- Encode an arena + root into `tropical_resolved_1` wire JSON,
    performing the canonical pool reordering. -/
def encodeResolved (arena : Arena) (root : ProgramIdx) : Except String Json := do
  -- Two O(edges) sweeps buy the termination measures: the expression
  -- arena's child-descending ids (`encExpr`) and the program pool's
  -- (`programId`'s DFS) — both hold by construction.
  if hw : arena.exprs.wf then
    if hwp : progPoolWf arena.programs then
      let init : Encode.St := {
        programIds := Array.replicate arena.programs.size none }
      let (rootId, st) ← (Encode.programId arena hw hwp root).run init
      pure <| Json.mkObj [
        ("schema", Json.str schemaTag),
        ("programPool", Json.arr st.programPool),
        ("root", Lean.toJson rootId)]
    else
      throw "encodeResolved: program pool is not child-descending (internal construction-order bug)"
  else
    throw "encodeResolved: arena is not child-descending (internal interning-order bug)"

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

/-- The decode monad interns each node into the shared expression DAG as it is
    parsed, so a decoded `Program` is id-valued like the elaborator's output. -/
private abbrev DecM := StateT ExprArena (Except String)

private def internD (n : ENode) : DecM ExprId := fun a => .ok ((eintern n).run a)

/- Descending accessors: same behavior and error strings as `reqField` /
   `reqArr`, but the result carries the `sizeOf` bound `expr`'s
   termination measure descends through (the Parse-layer pattern). -/

private def reqFieldD (ctx : String) (j : JsonV) (k : String) :
    Except String {v : JsonV // sizeOf v < sizeOf j} :=
  match hf : j.getField? k with
  | some v => pure ⟨v, Tropical.Parse.JsonV.sizeOf_lt_of_getField hf⟩
  | none => err ctx s!"missing field '{k}'"

private def reqArrD (ctx : String) (j : JsonV) (k : String) :
    Except String {items : Array JsonV // sizeOf items < sizeOf j} := do
  match ← reqFieldD ctx j k with
  | ⟨.arr items, h⟩ => pure ⟨items, by simp at h; omega⟩
  | _ => err ctx s!"field '{k}' must be an array"

/-- Parse + intern a `tropical_resolved_1` expression, returning its arena id.
    (Byte-round-trips with `encExpr` above.) The `req*`/`err` helpers return
    `Except String` and auto-lift into `DecM`. Total by descent on
    `sizeOf j`. -/
private def expr (ctx : String) (j : JsonV) : DecM ExprId := do
  match _hj : j with
  | .num n => internD (.num n)
  | .bool b => internD (.bool b)
  | .arr items => do
    let out ← items.attach.zipIdx.mapM fun (⟨x, _⟩, i) =>
      expr s!"{ctx}[{i}]" x
    internD (.arr out)
  | .obj _ =>
    let some op := j.opOf? | err ctx "expression object missing string 'op'"
    let args1 : Except String {v : JsonV // sizeOf v < sizeOf j} := do
      let ⟨a, ha⟩ ← reqArrD ctx j "args"
      if hn : a.size = 1 then
        pure ⟨a[0], Nat.lt_trans (Tropical.Parse.sizeOf_lt_of_getElem (by omega)) ha⟩
      else err ctx s!"'{op}' expects 1 args, got {a.size}"
    let args2 : Except String
        ({v : JsonV // sizeOf v < sizeOf j} × {v : JsonV // sizeOf v < sizeOf j}) := do
      let ⟨a, ha⟩ ← reqArrD ctx j "args"
      if hn : a.size = 2 then
        pure (⟨a[0], Nat.lt_trans (Tropical.Parse.sizeOf_lt_of_getElem (by omega)) ha⟩,
              ⟨a[1], Nat.lt_trans (Tropical.Parse.sizeOf_lt_of_getElem (by omega)) ha⟩)
      else err ctx s!"'{op}' expects 2 args, got {a.size}"
    let args3 : Except String
        ({v : JsonV // sizeOf v < sizeOf j} × {v : JsonV // sizeOf v < sizeOf j}
          × {v : JsonV // sizeOf v < sizeOf j}) := do
      let ⟨a, ha⟩ ← reqArrD ctx j "args"
      if hn : a.size = 3 then
        pure (⟨a[0], Nat.lt_trans (Tropical.Parse.sizeOf_lt_of_getElem (by omega)) ha⟩,
              ⟨a[1], Nat.lt_trans (Tropical.Parse.sizeOf_lt_of_getElem (by omega)) ha⟩,
              ⟨a[2], Nat.lt_trans (Tropical.Parse.sizeOf_lt_of_getElem (by omega)) ha⟩)
      else err ctx s!"'{op}' expects 3 args, got {a.size}"
    if let some tag := BinaryOpTag.ofWire? op then
      let (⟨a0, _⟩, ⟨a1, _⟩) ← args2
      internD (.binary tag (← expr s!"{ctx}.args[0]" a0) (← expr s!"{ctx}.args[1]" a1))
    else if let some tag := UnaryOpTag.ofWire? op then
      let ⟨a0, _⟩ ← args1
      internD (.unary tag (← expr s!"{ctx}.args[0]" a0))
    else match op with
    | "clamp" => do
      let (⟨a0, _⟩, ⟨a1, _⟩, ⟨a2, _⟩) ← args3
      internD (.clamp (← expr s!"{ctx}.args[0]" a0) (← expr s!"{ctx}.args[1]" a1)
                      (← expr s!"{ctx}.args[2]" a2))
    | "select" => do
      let (⟨a0, _⟩, ⟨a1, _⟩, ⟨a2, _⟩) ← args3
      internD (.select (← expr s!"{ctx}.args[0]" a0) (← expr s!"{ctx}.args[1]" a1)
                       (← expr s!"{ctx}.args[2]" a2))
    | "arraySet" => do
      let (⟨a0, _⟩, ⟨a1, _⟩, ⟨a2, _⟩) ← args3
      internD (.arraySet (← expr s!"{ctx}.args[0]" a0) (← expr s!"{ctx}.args[1]" a1)
                         (← expr s!"{ctx}.args[2]" a2))
    | "index" => do
      let (⟨a0, _⟩, ⟨a1, _⟩) ← args2
      internD (.index (← expr s!"{ctx}.args[0]" a0) (← expr s!"{ctx}.args[1]" a1))
    | "inputRef" => internD (.inputRef ⟨← reqNat ctx j "idx"⟩)
    | "paramRef" => internD (.paramRef ⟨← reqNat ctx j "idx"⟩)
    | "nestedOut" =>
      internD (.nestedOut ⟨← reqNat ctx j "instance"⟩ ⟨← reqNat ctx j "output"⟩)
    | "sampleRate" => internD .sampleRate
    | "sampleIndex" => internD .sampleIndex
    -- optional binder id (nested banks); absent = 0, the pre-nesting form.
    | "loopIdx" => do
      let id ← match j.getField? "id" with
        | some _ => reqNat ctx j "id"
        | none => pure 0
      internD (.loopIdx id)
    | "bankSum" => do
      let ⟨ts, _⟩ ← reqArrD ctx j "tables"
      let tables ← ts.attach.zipIdx.mapM fun (⟨t, _⟩, i) =>
        expr s!"{ctx}.tables[{i}]" t
      -- optional runtime effective count (trip-count-as-data); absent = static.
      let dc? ← match hdc : j.getField? "dyn_count" with
        | some dj => some <$> expr s!"{ctx}.dyn_count" dj
        | none => pure none
      -- optional binder id (nested banks); absent = 0.
      let idxId ← match j.getField? "idx_id" with
        | some _ => reqNat ctx j "idx_id"
        | none => pure 0
      let count ← reqNat ctx j "count"
      let ⟨bodyJ, _⟩ ← reqFieldD ctx j "body"
      internD (.bankSum count tables (← expr s!"{ctx}.body" bodyJ) dc? idxId)
    | "routedSum" => do
      let ⟨ts, _⟩ ← reqArrD ctx j "tables"
      let tables ← ts.attach.zipIdx.mapM fun (⟨t, _⟩, i) =>
        expr s!"{ctx}.tables[{i}]" t
      let ⟨vs, _⟩ ← reqArrD ctx j "values"
      let values ← vs.attach.zipIdx.mapM fun (⟨v, _⟩, i) =>
        expr s!"{ctx}.values[{i}]" v
      let ⟨routeValues, _⟩ ← reqArrD ctx j "routes"
      let routes ← routeValues.attach.zipIdx.mapM fun (⟨route, _⟩, i) =>
        match route with
        | .null => pure none
        | .num n =>
          if n.exponent == 0 && n.mantissa ≥ 0 then
            pure (some n.mantissa.toNat)
          else err s!"{ctx}.routes[{i}]" "expected a natural number or null"
        | _ => err s!"{ctx}.routes[{i}]" "expected a natural number or null"
      let dc? ← match hdc : j.getField? "dyn_count" with
        | some dj => some <$> expr s!"{ctx}.dyn_count" dj
        | none => pure none
      let idxId ← match j.getField? "idx_id" with
        | some _ => reqNat ctx j "idx_id"
        | none => pure 0
      internD (.routedSum (← reqNat ctx j "capacity")
        (← reqNat ctx j "output_count") routes tables values dc? idxId)
    | other => err ctx s!"unknown expression op '{other}'"
  | _ => err ctx "expected an expression value"
termination_by sizeOf j
decreasing_by
  all_goals first
    | omega
    | (simp_all <;> omega)
    | (have := Array.sizeOf_lt_of_mem ‹_ ∈ items›; simp_all <;> omega)
    | (have := Array.sizeOf_lt_of_mem ‹_ ∈ ts›; simp_all <;> omega)
    | (have := Array.sizeOf_lt_of_mem ‹_ ∈ vs›; simp_all <;> omega)
    | (have := Tropical.Parse.JsonV.sizeOf_lt_of_getField (by assumption); simp_all <;> omega)

private def decodePortType (ctx : String) (j : JsonV) : Except String PortType := do
  let kind ← reqStr ctx j "kind"
  match kind with
  | "scalar" => pure (.scalar (← scalarKind s!"{ctx}.scalar" (← reqStr ctx j "scalar")))
  | "array" => do
    let element ← match ← reqField ctx j "element" with
      | .str s => scalarKind s!"{ctx}.element" s
      | _ => err s!"{ctx}.element" "expected a scalar kind"
    let dims ← reqArr ctx j "shape"
    let mut shape : Array JsonNumber := #[]
    for h : i in [0:dims.size] do
      match dims[i] with
      | .num n => shape := shape.push n
      | _ => err s!"{ctx}.shape[{i}]" "expected a number (type-param dims are retired)"
    pure (.array element shape)
  | other => err ctx s!"unknown port-type kind '{other}'"

private def decodeProgram (i : Nat) (j : JsonV)
    (pBase programsSoFar : Nat) : DecM Program := do
  let ctx := s!"programPool[{i}]"
  let name ← reqStr ctx j "name"
  let progRef (rctx : String) (n : Nat) : Except String ProgramIdx :=
    if n < programsSoFar then pure ⟨pBase + n⟩
    else err rctx s!"program pool index {n} is out of range (decoded so far: {programsSoFar}); forward references violate the post-order pool contract"

  let mut inputs : Array InputDecl := #[]
  for h : k in [0:(← reqArr ctx j "inputs").size] do
    let ds ← reqArr ctx j "inputs"
    let d := ds[k]!
    let dctx := s!"{ctx}.inputs[{k}]"
    let type? ← match d.getField? "type" with
      | none => pure none
      | some t => pure (some (← decodePortType s!"{dctx}.type" t))
    let default? ← match d.getField? "default" with
      | none => pure none
      | some e => pure (some (← expr s!"{dctx}.default" e))
    inputs := inputs.push { name := ← reqStr dctx d "name", type?, default? }

  let mut outputs : Array OutputDecl := #[]
  for h : k in [0:(← reqArr ctx j "outputs").size] do
    let ds ← reqArr ctx j "outputs"
    let d := ds[k]!
    let dctx := s!"{ctx}.outputs[{k}]"
    let type? ← match d.getField? "type" with
      | none => pure none
      | some t => pure (some (← decodePortType s!"{dctx}.type" t))
    outputs := outputs.push { name := ← reqStr dctx d "name", type? }

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
      let ins ← reqArr dctx d "inputs"
      let mut instInputs : Array InstanceInput := #[]
      for h2 : m in [0:ins.size] do
        let w := ins[m]
        let wctx := s!"{dctx}.inputs[{m}]"
        instInputs := instInputs.push {
          port := ⟨← reqNat wctx w "port"⟩
          value := ← expr s!"{wctx}.value" (← reqField wctx w "value") }
      decls := decls.push (.inst dname typeKey instInputs)
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
    assigns := assigns.push { target, expr := ← expr s!"{actx}.expr" (← reqField actx a "expr") }

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

  pure { name, inputs, outputs, decls, assigns, registry }

end Decode

/-- Decode `tropical_resolved_1` wire JSON, **appending** into an
    existing arena. The wire's pool indices are batch-relative; the
    lower-index invariant makes the rebase a mechanical shift by the
    base pool size. Each call appends a self-contained copy — wire
    indices cannot reach entries below the base, and nothing already
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

  let pBase := arena.programs.size

  -- Programs — single forward pass, interning expressions into the shared
  -- DAG (seeded from the existing arena's `exprs`).
  let mut programs : Array Program := arena.programs
  let mut exprs : ExprArena := arena.exprs
  for i in [0:(← pool "programPool").size] do
    let ps ← pool "programPool"
    let (p, exprs') ← (Decode.decodeProgram i ps[i]!
           pBase (programs.size - pBase)).run exprs
    programs := programs.push p
    exprs := exprs'

  let root ← match j.getField? "root" with
    | some (.num n) =>
      if n.exponent == 0 && n.mantissa.toNat < programs.size - pBase then
        pure (ProgramIdx.mk (pBase + n.mantissa.toNat))
      else .error "decodeResolved: root: program pool index out of range"
    | _ => .error "decodeResolved: missing 'root'"

  pure ({ programs, exprs }, root)

/-- Decode `tropical_resolved_1` wire JSON into a fresh arena + root,
    keeping the wire pool order as the arena order. -/
def decodeResolved (j : JsonV) : Except String (Arena × ProgramIdx) :=
  decodeResolvedInto {} j

end Tropical.Ir.Codec
