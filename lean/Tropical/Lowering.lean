import Lean.Data.Json
import Tropical.Errors
import Tropical.Expr
import Tropical.Wiring
import Tropical.Session
import Tropical.Ir.Cycles

/-!
# Session-to-root lowering

These deterministic passes run inside the Lean frontend for every structural
session compile (the durable `SessionSt` is never mutated here):

- slot allocation (`allocateParamSlot` + `allocateOutputSlots` over the
  catalog port types, canonical order: params → instance outputs)
- `assertSessionAcyclic` — cycle tripwire over the instance dep graph,
  via the shared total detector `Ir.findCycle`
  (CF-only: inter-instance cycles are rejected outright)
- `computeInstanceTopoOrder` — Kahn with sorted ties

Wire lifting (array literals → anonymous `__wire_N` instances) is handled by
`Engine.Compile.liftIfNeeded` before this module builds the synthetic root.
Lift detection is `WireExpr.needsLift` on the session's typed wire values.
-/

namespace Tropical.Lowering

open Lean (Json toJson)
open Tropical.Expr (getField? getStrField? opOf?)
open Tropical.Wiring (parsePortType? PortType ScalarKind)

-- ── Slot allocation ──────────────────────────────────────────────────────────

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
    output ports (instance order). -/
def allocate (params : Array String) (instances : Array (String × InstanceInfo)) : Alloc := Id.run do
  let mut a : Alloc := {}
  for p in params do
    a := { a with slotCount := a.slotCount + 1, paramSlots := a.paramSlots.push (p, a.slotCount) }
  for (name, info) in instances do
    for port in info.progMeta.outputs do
      a := allocOutputPort a s!"{name}.{port.name}" port.typeObj
  return a

-- ── Dep graph, cycle tripwire, topo order ────────────────────────────────────

/-- consumer → producers, in instance order; producer sets keep discovery
    order. -/
private def buildDeps (instances : Array (String × InstanceInfo)) (wires : Array Wire)
    (dropSelfEdges : Bool) : Array (String × Array String) := Id.run do
  let mut deps : Array (String × Array String) := instances.map fun (n, _) => (n, #[])
  for w in wires do
    match deps.findIdx? (·.1 == w.instName) with
    | none => pure ()
    | some i =>
      let (n, producers) := deps[i]!
      let mut ps := producers
      for r in w.expr.deps do
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

/-- Kahn with lexicographically sorted ready queues; incomplete orders append
    the remainder in instance order. -/
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
