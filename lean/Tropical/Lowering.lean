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
  catalog port types, canonical order: params → instance outputs)
- `assertSessionAcyclic` — Tarjan tripwire over the instance dep graph
  (CF-only: inter-instance cycles are rejected outright)
- `computeInstanceTopoOrder` — Kahn with sorted ties

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

/-- CF-only inter-instance cycle tripwire. -/
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
      ["compileSession: CF-only — inter-instance cycles are not allowed:"]
      ++ lines.toList
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

end Tropical.Lowering
