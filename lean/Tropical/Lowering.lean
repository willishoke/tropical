import Lean.Data.Json
import Tropical.Errors
import Tropical.Expr
import Tropical.Wiring
import Tropical.Session
import Tropical.Ir.Cycles

/-!
# Session lowering — Phase 3 of the Lean port

Ports of the TS session→root lowering passes, run per compile over the
engine's mirror (pure — the durable mirror is never mutated here):

- slot allocation (`allocateParamSlot` + `allocateOutputSlots` over the
  catalog port types, canonical order: params → instance outputs)
- `assertSessionAcyclic` — cycle tripwire over the instance dep graph,
  via the shared total detector `Ir.findCycle`
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

/-- CF-only inter-instance cycle tripwire (the shared total detector,
    `Ir.findCycle`, over the wire dep graph). Producer names outside
    the instance set (dangling refs) cannot sit on a cycle and are
    dropped at interning. -/
def assertSessionAcyclic (instances : Array (String × InstanceInfo))
    (wires : Array Wire) : EngineM Unit := do
  let deps := buildDeps instances wires (dropSelfEdges := false)
  let depsIdx : Array (Array Nat) := deps.map fun (_, ps) =>
    ps.filterMap fun p => deps.findIdx? (·.1 == p)
  if let some cyc := Tropical.Ir.findCycle depsIdx then
    -- `deps` edges point consumer → producer; reverse for flow order.
    let names := cyc.reverse.map fun i => ((deps[i]?).map (·.1)).getD "?"
    internalError <| String.intercalate "\n"
      ["compileSession: CF-only — inter-instance cycles are not allowed:",
       s!"  - {Tropical.Ir.renderLoop names}"]

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
