import Lean.Data.Json
import Tropical.Errors
import Tropical.Expr
import Tropical.Wiring
import Tropical.Session

/-!
# Session lowering — Phase 3 of the Lean port

Ports of the TS session→root lowering passes, run per compile over the
engine's mirror (pure — the durable mirror is never mutated here):

- slot allocation (`allocateParamSlot` + `allocateOutputSlots` over the
  catalog port types, canonical order: params → instance outputs →
  extraction delay slots)
- `extractSessionDelays` — hoist every `delay()` in a wire to a module
  slot (scalar) or ioArraySlot (array source), rewriting the wire to a
  `sessionSlot` / `sessionArraySlot` read
- `assertSessionAcyclic` — Tarjan tripwire over the post-extraction
  instance dep graph (exact TS message)
- `computeInstanceTopoOrder` — Kahn with sorted ties
- `sessionToParsed` — the session serialized as a `ParsedProgram`
  (compiler/parse/nodes.ts shapes, the free JSON seam): delay decls,
  instance decls in topo order, param decls

Wire *lifting* (array literals → anonymous `__wire_N` instances) stays
in the compiler service — it needs strata. `needsWireLift` (the pure
detection) lives here so the engine knows when to request it.
-/

namespace Tropical.Lowering

open Lean (Json toJson)
open Tropical.Expr (getField? getStrField? opOf?)
open Tropical.Wiring (parsePortType? PortType ScalarKind)

-- ── Lift detection (compiler/ir/lift_wires.ts needsWireLift) ─────────────────

partial def needsWireLift (expr : Json) : Bool :=
  match expr with
  | .arr _ => true   -- bare array literal
  | .obj _ =>
    let op := opOf? expr
    if op == some "array" || op == some "arrayLiteral" then true
    else
      let inArr (k : String) : Bool :=
        match getField? expr k with
        | some (.arr items) => items.any needsWireLift
        | _ => false
      inArr "args" || inArr "items"
  | _ => false

-- ── Slot allocation (compiler/session.ts expandPortToSlots discipline) ──────

structure Alloc where
  slotCount   : Nat := 0
  paramSlots  : Array (String × Nat) := #[]
  outputSlots : Array (String × Nat) := #[]
  outputMeta  : Array (String × Json) := #[]
  ioCount     : Nat := 0
  ioSizes     : Array Nat := #[]
  ioNames     : Array String := #[]

def Alloc.metaFor? (a : Alloc) (key : String) : Option Json :=
  (a.outputMeta.find? (·.1 == key)).map (·.2)

private def defaultPortType : Json :=
  Json.mkObj [("kind", Json.str "scalar"), ("scalar", Json.str "float")]

/-- One output port's allocation (port of `expandPortToSlots` +
    `allocateOutputSlots`): scalar/alias → one module slot named by the
    port key; array → one ioArraySlot of size ∏shape. -/
private def allocOutputPort (a : Alloc) (key : String) (typeObj : Option Json) : Alloc :=
  let tObj := typeObj.getD defaultPortType
  match parsePortType? tObj with
  | some (.array elem shape) =>
    let size := shape.foldl (· * ·) 1
    let pmeta := Json.mkObj [
      ("scalarSlotNames", Json.arr #[]),
      ("scalarTypes", Json.arr #[]),
      ("arraySlot", toJson a.ioCount),
      ("arraySize", toJson size),
      ("arrayElemType", Json.str (elem.kind.getD .float).wire),
      ("portType", tObj)]
    { a with
      ioCount := a.ioCount + 1
      ioSizes := a.ioSizes.push size
      ioNames := a.ioNames.push key
      outputMeta := a.outputMeta.push (key, pmeta) }
  | other =>
    -- scalar / alias / unparseable: one float-ish module slot.
    let scalarType := match other with
      | some (.scalar k) => k.wire
      | _ => "float"   -- alias → 'float' (expandPortToSlots), fallback likewise
    let pmeta := Json.mkObj [
      ("scalarSlotNames", Json.arr #[Json.str key]),
      ("scalarTypes", Json.arr #[Json.str scalarType]),
      ("portType", tObj)]
    { a with
      slotCount := a.slotCount + 1
      outputSlots := a.outputSlots.push (key, a.slotCount)
      outputMeta := a.outputMeta.push (key, pmeta) }

/-- Canonical allocation: params (registry order), then every instance's
    output ports (instance order). Extraction delay slots continue after. -/
def allocate (params : Array String) (instances : Array (String × InstanceInfo)) : Alloc := Id.run do
  let mut a : Alloc := {}
  for p in params do
    a := { a with slotCount := a.slotCount + 1, paramSlots := a.paramSlots.push (p, a.slotCount) }
  for (name, info) in instances do
    for port in info.progMeta.outputs do
      a := allocOutputPort a s!"{name}.{port.name}" port.typeObj
  return a

-- ── Delay extraction (ir/lowering/extract_session_delays.ts) ────────────────

structure DelayEntry where
  slotName   : String
  init       : Json
  sourceExpr : Json := Json.null
  isArray    : Bool := false
  slotIdx    : Option Nat := none
  arraySlot  : Option Nat := none
  arraySize  : Option Nat := none

def DelayEntry.toJson (e : DelayEntry) : Json :=
  Json.mkObj <|
    [("slotName", Json.str e.slotName), ("init", e.init),
     ("sourceExpr", e.sourceExpr), ("scalarType", Json.str "float")]
    ++ (match e.slotIdx with | some i => [("slotIdx", Lean.toJson i)] | none => [])
    ++ (if e.isArray then [("isArray", Json.bool true)] else [])
    ++ (match e.arraySlot with | some i => [("arraySlot", Lean.toJson i)] | none => [])
    ++ (match e.arraySize with | some i => [("arraySize", Lean.toJson i)] | none => [])

structure ExtractSt where
  alloc   : Alloc
  entries : Array DelayEntry := #[]

private def isDelayNode (j : Json) : Bool :=
  opOf? j == some "delay" &&
  (match getField? j "args" with
   | some (.arr args) => args.size == 1
   | _ => false)

/-- `{op:'ref', instance, output}` against an array-typed output →
    element count (port of `inferArraySourceShape`). -/
private def inferArraySourceShape (alloc : Alloc) (expr : Json) : Option Nat := do
  guard (opOf? expr == some "ref")
  let inst ← getStrField? expr "instance"
  let output ← getStrField? expr "output"
  let pmeta ← alloc.metaFor? s!"{inst}.{output}"
  let size ← getField? pmeta "arraySize"
  match size with
  | .num n => some n.toFloat.toUInt64.toNat
  | _ => none

/-- Walk a wire, hoisting every `delay()` to a slot. Outer-first: the
    entry is reserved before recursing into the source so nested chains
    accumulate one sample per level in registry order. -/
partial def rewriteAndCollect (expr : Json) (context : String) (st : ExtractSt) : Json × ExtractSt :=
  match expr with
  | .arr items => Id.run do
    let mut st := st
    let mut out : Array Json := #[]
    for h : i in [0:items.size] do
      let (e', st') := rewriteAndCollect items[i] s!"{context}[{i}]" st
      out := out.push e'
      st := st'
    return (.arr out, st)
  | .obj m =>
    if isDelayNode expr then
      let args := match getField? expr "args" with
        | some (.arr a) => a
        | _ => #[]
      let init := (getField? expr "init").getD (toJson (0 : Nat))
      let id := (getStrField? expr "id").getD
        s!"__autodelay:{context}#{st.entries.size}"
      match inferArraySourceShape st.alloc args[0]! with
      | some size =>
        let arraySlot := st.alloc.ioCount
        let alloc := { st.alloc with
          ioCount := st.alloc.ioCount + 1
          ioSizes := st.alloc.ioSizes.push size
          ioNames := st.alloc.ioNames.push id }
        let entryIdx := st.entries.size
        let entry : DelayEntry :=
          { slotName := id, init, isArray := true,
            arraySlot := some arraySlot, arraySize := some size }
        let st := { alloc, entries := st.entries.push entry }
        let (src, st) := rewriteAndCollect args[0]! s!"{context}.delay" st
        let st := { st with entries := st.entries.set! entryIdx { entry with sourceExpr := src } }
        (Json.mkObj [("op", Json.str "sessionArraySlot"),
                     ("index", toJson arraySlot), ("size", toJson size)], st)
      | none =>
        let slotIdx := st.alloc.slotCount
        let alloc := { st.alloc with slotCount := st.alloc.slotCount + 1 }
        let entryIdx := st.entries.size
        let entry : DelayEntry := { slotName := id, init, slotIdx := some slotIdx }
        let st := { alloc, entries := st.entries.push entry }
        let (src, st) := rewriteAndCollect args[0]! s!"{context}.delay" st
        let st := { st with entries := st.entries.set! entryIdx { entry with sourceExpr := src } }
        (Json.mkObj [("op", Json.str "sessionSlot"), ("index", toJson slotIdx)], st)
    else Id.run do
      -- Generic recurse over every expression-valued field.
      let mut st := st
      let mut fields : List (String × Json) := []
      for (k, v) in m.toArray do
        if k == "op" then
          fields := fields ++ [(k, v)]
        else
          match v with
          | .arr items =>
            let mut out : Array Json := #[]
            for h : i in [0:items.size] do
              match items[i] with
              | .obj _ | .arr _ | .num _ | .bool _ =>
                let (e', st') := rewriteAndCollect items[i] s!"{context}.{k}[{i}]" st
                out := out.push e'
                st := st'
              | e => out := out.push e
            fields := fields ++ [(k, .arr out)]
          | .obj _ =>
            let (v', st') := rewriteAndCollect v s!"{context}.{k}" st
            fields := fields ++ [(k, v')]
            st := st'
          | _ => fields := fields ++ [(k, v)]
      return (Json.mkObj fields, st)
  | _ => (expr, st)

/-- Extract every delay across all wires (wire-map order). Returns the
    rewritten wires, the registry, and the advanced allocation. -/
def extractSessionDelays (wires : Array Wire) (alloc : Alloc) :
    Array Wire × Array DelayEntry × Alloc := Id.run do
  let mut st : ExtractSt := { alloc }
  let mut out : Array Wire := #[]
  for w in wires do
    let (e', st') := rewriteAndCollect w.expr w.key st
    out := out.push { w with expr := e' }
    st := st'
  return (out, st.entries, st.alloc)

-- ── Dep graph, cycle tripwire, topo order ────────────────────────────────────

private partial def collectInstanceRefs (expr : Json) (acc : Array String) : Array String :=
  match expr with
  | .arr items => items.foldl (fun a e => collectInstanceRefs e a) acc
  | .obj m =>
    if opOf? expr == some "ref" then
      match getStrField? expr "instance" with
      | some i => if acc.contains i then acc else acc.push i
      | none => acc
    else
      m.toArray.foldl (fun a (kv : String × Json) =>
        match kv.2 with
        | .obj _ | .arr _ => collectInstanceRefs kv.2 a
        | _ => a) acc
  | _ => acc

/-- consumer → producers, in instance order; producer sets keep
    discovery order (TS Set-insertion parity). -/
private def buildDeps (instances : Array (String × InstanceInfo)) (wires : Array Wire)
    (dropSelfEdges : Bool) : Array (String × Array String) := Id.run do
  let mut deps : Array (String × Array String) := instances.map fun (n, _) => (n, #[])
  for w in wires do
    match deps.findIdx? (·.1 == w.instName) with
    | none => pure ()
    | some i =>
      let (n, producers) := deps[i]!
      let mut ps := producers
      for r in collectInstanceRefs w.expr #[] do
        if !ps.contains r && !(dropSelfEdges && r == n) then
          ps := ps.push r
      deps := deps.set! i (n, ps)
  return deps

/-- Tarjan SCC (port of compiler/compiler.ts tarjanSCC; same visit and
    neighbor order). -/
private partial def tarjanSCC (deps : Array (String × Array String)) : Array (Array String) := Id.run do
  let mut indices : Std.HashMap String Nat := {}
  let mut lowlinks : Std.HashMap String Nat := {}
  let mut onStack : Std.HashMap String Bool := {}
  let mut stack : Array String := #[]
  let mut sccs : Array (Array String) := #[]
  let mut nextIdx := 0

  let neighbors := fun (v : String) =>
    ((deps.find? (·.1 == v)).map (·.2)).getD #[]

  -- Explicit-stack DFS replicating the recursive algorithm.
  let rec visit (v : String) (st : Std.HashMap String Nat × Std.HashMap String Nat ×
      Std.HashMap String Bool × Array String × Array (Array String) × Nat) :
      Std.HashMap String Nat × Std.HashMap String Nat ×
      Std.HashMap String Bool × Array String × Array (Array String) × Nat := Id.run do
    let (indices0, lowlinks0, onStack0, stack0, sccs0, idx0) := st
    let mut indices := indices0.insert v idx0
    let mut lowlinks := lowlinks0.insert v idx0
    let mut onStack := onStack0.insert v true
    let mut stack := stack0.push v
    let mut sccs := sccs0
    let mut idx := idx0 + 1
    for w in neighbors v do
      if !indices.contains w then
        let (i', l', o', s', c', x') := visit w (indices, lowlinks, onStack, stack, sccs, idx)
        indices := i'; lowlinks := l'; onStack := o'; stack := s'; sccs := c'; idx := x'
        let lv := (lowlinks.get? v).getD 0
        let lw := (lowlinks.get? w).getD 0
        lowlinks := lowlinks.insert v (Nat.min lv lw)
      else if (onStack.get? w).getD false then
        let lv := (lowlinks.get? v).getD 0
        let iw := (indices.get? w).getD 0
        lowlinks := lowlinks.insert v (Nat.min lv iw)
    if (lowlinks.get? v).getD 0 == (indices.get? v).getD 0 then
      let mut scc : Array String := #[]
      let mut go := true
      while go do
        match stack.back? with
        | none => go := false
        | some w =>
          stack := stack.pop
          onStack := onStack.insert w false
          scc := scc.push w
          if w == v then go := false
      sccs := sccs.push scc
    return (indices, lowlinks, onStack, stack, sccs, idx)

  let mut st := (indices, lowlinks, onStack, stack, sccs, nextIdx)
  for (v, _) in deps do
    let (i, _, _, _, _, _) := st
    if !i.contains v then
      st := visit v st
  let (_, _, _, _, out, _) := st
  return out

/-- Port of `assertSessionAcyclic` — exact TS message. -/
def assertSessionAcyclic (instances : Array (String × InstanceInfo))
    (wires : Array Wire) : EngineM Unit := do
  let deps := buildDeps instances wires (dropSelfEdges := false)
  let sccs := tarjanSCC deps
  let selfEdge := fun (n : String) =>
    (((deps.find? (·.1 == n)).map (·.2)).getD #[]).contains n
  let nontrivial := sccs.filter fun scc =>
    scc.size > 1 || (scc.size == 1 && selfEdge scc[0]!)
  if nontrivial.isEmpty then
    pure ()
  else
    let lines := nontrivial.map fun c =>
      s!"  - {String.intercalate " → " c.toList} → {c[0]!}"
    let msg := String.intercalate "\n" <|
      ["compileSession: session graph contains inter-instance cycles that don't pass through a session-level delay():"]
      ++ lines.toList
      ++ [""]
      ++ ["MCP wire helpers auto-wrap every wire in a unit delay, which breaks cycles at the session level. If you're building a SessionState programmatically (test fixture, legacy ingest), either route wires through `setWireExpr` from compiler/session.ts or explicitly wrap each back-edge in {op:'delay', args:[...], init:0}."]
    internalError msg

/-- Kahn with lexicographically sorted ready-queues (port of
    computeInstanceTopoOrder; incomplete orders append the remainder in
    instance order). -/
def computeInstanceTopoOrder (instances : Array (String × InstanceInfo))
    (wires : Array Wire) : Array String := Id.run do
  let deps := buildDeps instances wires (dropSelfEdges := true)
  let mut inDegree : Array (String × Nat) := deps.map fun (n, ps) => (n, ps.size)
  let mut consumers : Array (String × Array String) := deps.map fun (n, _) => (n, #[])
  for (consumer, producers) in deps do
    for p in producers do
      match consumers.findIdx? (·.1 == p) with
      | some i =>
        let (n, cs) := consumers[i]!
        if !cs.contains consumer then consumers := consumers.set! i (n, cs.push consumer)
      | none => pure ()

  let mut order : Array String := #[]
  let mut queue := ((inDegree.filter (·.2 == 0)).map (·.1)).qsort (· < ·)
  while !queue.isEmpty do
    order := order ++ queue
    let mut next : Array String := #[]
    for node in queue do
      let cs := ((consumers.find? (·.1 == node)).map (·.2)).getD #[]
      for c in cs do
        match inDegree.findIdx? (·.1 == c) with
        | some i =>
          let (n, d) := inDegree[i]!
          let d' := d - 1
          inDegree := inDegree.set! i (n, d')
          if d' == 0 then next := next.push c
        | none => pure ()
    queue := next.qsort (· < ·)

  if order.size == deps.size then return order
  let mut out := order
  for (n, _) in instances do
    if !out.contains n then out := out.push n
  return out

-- ── Session → ParsedProgram (ir/session_to_parsed.ts) ───────────────────────

private def builtinCallOps : List String :=
  ["select", "clamp", "round", "ldexp", "floorDiv",
   "sqrt", "abs", "floatExponent", "arraySet",
   "floor", "ceil", "toInt", "toBool", "toFloat"]

private def builtinNullaryOps : List String := ["sampleRate", "sampleIndex"]

private def binaryOps : List String :=
  ["add", "sub", "mul", "div", "mod",
   "lt", "lte", "gt", "gte", "eq", "neq",
   "and", "or",
   "bitAnd", "bitOr", "bitXor", "lshift", "rshift"]

private def unaryOps : List String := ["neg", "not", "bitNot"]

private def nameRef (n : String) : Json :=
  Json.mkObj [("op", Json.str "nameRef"), ("name", Json.str n)]

/-- index→slotName lookups for the two hoisted-delay namespaces. -/
private structure SlotNames where
  scalar : Array (Nat × String)
  array  : Array (Nat × String)

/-- Port of `wireExprToParsed` — exact closed op set and error strings.
    Collected param names accumulate in the state. -/
private partial def wireExprToParsed (expr : Json) (slots : SlotNames)
    (params : Array String) : EngineM (Json × Array String) := do
  match expr with
  | .num _ | .bool _ => pure (expr, params)
  | .arr items => do
    let mut out : Array Json := #[]
    let mut ps := params
    for e in items do
      let (e', ps') ← wireExprToParsed e slots ps
      out := out.push e'
      ps := ps'
    pure (.arr out, ps)
  | .obj _ =>
    let some op := opOf? expr
      | internalError s!"session_to_parsed: wire node missing op: {expr.compress}"
    let args := match getField? expr "args" with
      | some (.arr a) => a
      | _ => #[]
    let mapArgs (ps : Array String) : EngineM (Array Json × Array String) := do
      let mut out : Array Json := #[]
      let mut acc := ps
      for a in args do
        let (a', acc') ← wireExprToParsed a slots acc
        out := out.push a'
        acc := acc'
      pure (out, acc)
    if op == "ref" then
      pure (Json.mkObj [
        ("op", Json.str "nestedOut"),
        ("ref", nameRef ((getStrField? expr "instance").getD "")),
        ("output", nameRef ((getStrField? expr "output").getD ""))], params)
    else if op == "sessionSlot" then
      let idx := match getField? expr "index" with
        | some (.num n) => n.toFloat.toUInt64.toNat
        | _ => 0
      match slots.scalar.find? (·.1 == idx) with
      | some (_, name) => pure (nameRef name, params)
      | none => internalError s!"session_to_parsed: sessionSlot index {idx} has no delay decl"
    else if op == "sessionArraySlot" then
      let idx := match getField? expr "index" with
        | some (.num n) => n.toFloat.toUInt64.toNat
        | _ => 0
      match slots.array.find? (·.1 == idx) with
      | some (_, name) => pure (nameRef name, params)
      | none => internalError s!"session_to_parsed: sessionArraySlot index {idx} has no array delay decl"
    else if op == "array" then
      match getField? expr "items" with
      | some (.arr items) => do
        let mut out : Array Json := #[]
        let mut ps := params
        for e in items do
          let (e', ps') ← wireExprToParsed e slots ps
          out := out.push e'
          ps := ps'
        pure (.arr out, ps)
      | _ => internalError s!"session_to_parsed: out-of-slice wire op '{op}'"
    else if op == "param" || op == "trigger" then
      let name := (getStrField? expr "name").getD ""
      pure (nameRef name, if params.contains name then params else params.push name)
    else if op == "delay" then
      internalError "session_to_parsed: unextracted delay() in wire — run extractSessionDelays first"
    else if builtinNullaryOps.contains op then
      pure (Json.mkObj [("op", Json.str "call"), ("callee", nameRef op), ("args", Json.arr #[])], params)
    else if builtinCallOps.contains op then
      let (out, ps) ← mapArgs params
      pure (Json.mkObj [("op", Json.str "call"), ("callee", nameRef op), ("args", Json.arr out)], ps)
    else if binaryOps.contains op || unaryOps.contains op then
      let (out, ps) ← mapArgs params
      pure (Json.mkObj [("op", Json.str op), ("args", Json.arr out)], ps)
    else
      internalError s!"session_to_parsed: out-of-slice wire op '{op}'"
  | _ => internalError s!"session_to_parsed: invalid wire value {expr.compress}"

/-- Port of `sessionToParsedProgram`: delay decls, instance decls in topo
    order (inputs sorted by port name), param decls (sorted). -/
def sessionToParsed (instances : Array (String × InstanceInfo))
    (wiresPost : Array Wire) (entries : Array DelayEntry) : EngineM Json := do
  let slots : SlotNames := {
    scalar := entries.filterMap fun e =>
      if e.isArray then none else e.slotIdx.map fun i => (i, e.slotName)
    array := entries.filterMap fun e =>
      if e.isArray then e.arraySlot.map fun i => (i, e.slotName) else none }

  let mut decls : Array Json := #[]
  let mut params : Array String := #[]

  -- 1. Per-wire delays → delay decls.
  for e in entries do
    let (update, ps) ← wireExprToParsed e.sourceExpr slots params
    params := ps
    let init : Json :=
      if e.isArray then
        .arr (Array.replicate (e.arraySize.getD 0) e.init)
      else e.init
    decls := decls.push <| Json.mkObj [
      ("op", Json.str "delayDecl"), ("name", Json.str e.slotName),
      ("update", update), ("init", init)]

  -- 2. Instances in topo order, inputs sorted by port name.
  let order := computeInstanceTopoOrder instances wiresPost
  for name in order do
    let some info := (instances.find? (·.1 == name)).map (·.2) | continue
    let wires := (wiresPost.filter (·.instName == name)).qsort
      (fun a b => a.portName < b.portName)
    let mut inputs : Array Json := #[]
    for w in wires do
      let (value, ps) ← wireExprToParsed w.expr slots params
      params := ps
      inputs := inputs.push <| Json.mkObj [("port", nameRef w.portName), ("value", value)]
    decls := decls.push <| Json.mkObj <|
      [("op", Json.str "instanceDecl"), ("name", Json.str name),
       ("program", nameRef info.progMeta.programName)]
      ++ (if inputs.isEmpty then [] else [("inputs", Json.arr inputs)])

  -- 3. Param decls (sorted names).
  for pname in params.qsort (· < ·) do
    decls := decls.push <| Json.mkObj [("op", Json.str "paramDecl"), ("name", Json.str pname)]

  pure <| Json.mkObj [
    ("op", Json.str "program"), ("name", Json.str "__session__"),
    ("body", Json.mkObj [("op", Json.str "block"),
                         ("decls", Json.arr decls), ("assigns", Json.arr #[])])]

end Tropical.Lowering
