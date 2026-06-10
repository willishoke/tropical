import Std.Data.HashMap
import Lean.Data.Json
import Tropical.Expr

/-!
# Session state — the authoritative graph topology

Phase 1 of the Lean port: the session lives here, not in TypeScript.
What Lean owns: program metadata mirror (port names/types/defaults,
fed by the compiler service's catalog entries), instances (ordered),
wires (ordered, canonical pre-extraction form), graph outputs, the
param mirror, and name counters.

The compiler service receives this state as a snapshot on every
compile (`sync`) and rebuilds its own TS session from it — Lean never
stores compile-internal forms (`sessionSlot` rewrites, slot indices,
delay registries). Wires keep the authored `delay()` shape end to end.

Ordering discipline: TS `Map` iterates in insertion order, and
re-setting an existing key keeps its original position. `instances`
and `wires` are ordered arrays with update-in-place to match —
`list_instances`, `list_wiring`, and snapshot order all depend on it.
-/

namespace Tropical

open Lean (Json)
open Tropical.Expr (wrapInUnitDelay)

/-- One port's metadata from a catalog entry. -/
structure PortInfo where
  name    : String
  typeStr : Option String := none   -- display string ("float", "float[8]")
  typeObj : Option Json := none     -- structured PortType (get_info echo, wire checks)
  default : Option Json := none     -- raw ExprNode default (inputs only)
deriving Inhabited

/-- A registered program's metadata (concrete or generic template). -/
structure ProgMeta where
  programName : String
  generic     : Bool
  typeParams  : Option Json := none   -- {name: {type:'int', default?}} for generics
  inputs      : Array PortInfo := #[]
  outputs     : Array PortInfo := #[]
  registers   : Array PortInfo := #[]
deriving Inhabited

private def parsePort (j : Json) : PortInfo :=
  let nonNull (k : String) : Option Json :=
    match Tropical.Expr.getField? j k with
    | some .null => none
    | v => v
  { name    := (Tropical.Expr.getStrField? j "name").getD ""
    typeStr := match nonNull "type" with | some (.str s) => some s | _ => none
    typeObj := nonNull "type_obj"
    default := nonNull "default" }

def ProgMeta.fromEntry (j : Json) : ProgMeta :=
  let ports (k : String) : Array PortInfo :=
    match Tropical.Expr.getField? j k with
    | some (.arr a) => a.map parsePort
    | _ => #[]
  let typeParams := match Tropical.Expr.getField? j "type_params" with
    | some .null | none => none
    | some tp => some tp
  { programName := (Tropical.Expr.getStrField? j "program_name").getD ""
    generic     := match Tropical.Expr.getField? j "generic" with
                   | some (.bool b) => b | _ => false
    typeParams
    inputs      := ports "inputs"
    outputs     := ports "outputs"
    registers   := ports "registers" }

def ProgMeta.inputNames (m : ProgMeta) : Array String := m.inputs.map (·.name)
def ProgMeta.outputNames (m : ProgMeta) : Array String := m.outputs.map (·.name)

/-- A live instance: base type name, type args (generics), and the
    (possibly specialized) port metadata. -/
structure InstanceInfo where
  baseTypeName : String
  typeArgs     : Option Json := none
  progMeta     : ProgMeta
deriving Inhabited

/-- A stored wire. Always the canonical authored form — MCP-set wires
    carry the auto-delay wrap; load/merge-adopted wires are whatever the
    state dump reconstructed. -/
structure Wire where
  instName : String
  portName : String
  expr     : Json
deriving Inhabited

def Wire.key (w : Wire) : String := s!"{w.instName}:{w.portName}"

structure SessionSt where
  /-- Program registration order — mirrors the service's `session.programs`
      map order (stdlib, then session definitions). -/
  catalogOrder : Array String := #[]
  programs     : Std.HashMap String ProgMeta := {}
  instances    : Array (String × InstanceInfo) := #[]
  wires        : Array Wire := #[]
  graphOutputs : Array (String × String) := #[]   -- (instance, output)
  /-- Param mirror: name → last-known value (raw Json to preserve lexical
      number forms). The service owns the live Param handles. -/
  params       : Array (String × Json) := #[]
  nameCounters : Std.HashMap String Nat := {}

namespace SessionSt

def addProgram (st : SessionSt) (m : ProgMeta) : SessionSt :=
  { st with
    programs := st.programs.insert m.programName m
    catalogOrder := if st.catalogOrder.contains m.programName
                    then st.catalogOrder
                    else st.catalogOrder.push m.programName }

def findInstance? (st : SessionSt) (name : String) : Option InstanceInfo :=
  (st.instances.find? (·.1 == name)).map (·.2)

def instanceNames (st : SessionSt) : Array String := st.instances.map (·.1)

def addInstance (st : SessionSt) (name : String) (info : InstanceInfo) : SessionSt :=
  { st with instances := st.instances.push (name, info) }

def removeInstance (st : SessionSt) (name : String) : SessionSt :=
  { st with instances := st.instances.filter (·.1 != name) }

/-- TS-Map semantics: replacing an existing key keeps its position. -/
def setWireRaw (st : SessionSt) (instName portName : String) (expr : Json) : SessionSt :=
  match st.wires.findIdx? (fun w => w.instName == instName && w.portName == portName) with
  | some i => { st with wires := st.wires.set! i { instName, portName, expr } }
  | none   => { st with wires := st.wires.push { instName, portName, expr } }

def findWire? (st : SessionSt) (instName portName : String) : Option Wire :=
  st.wires.find? (fun w => w.instName == instName && w.portName == portName)

/-- The MCP wire-storage convention: wrap in a session-level unit delay
    (port of `setWireExpr` in compiler/session.ts). -/
def setWire (st : SessionSt) (instName portName : String) (rawExpr : Json)
    (init : Json := Lean.toJson (0 : Nat)) (id : Option String := none) : SessionSt :=
  let key := s!"{instName}:{portName}"
  let delayId := id.getD s!"__autodelay:{key}"
  st.setWireRaw instName portName (wrapInUnitDelay rawExpr init delayId)

def removeWire (st : SessionSt) (instName portName : String) : SessionSt :=
  { st with wires := st.wires.filter (fun w => !(w.instName == instName && w.portName == portName)) }

def nextName (st : SessionSt) (prefix' : String) : SessionSt × String :=
  let count := (st.nameCounters.get? prefix').getD 0 + 1
  ({ st with nameCounters := st.nameCounters.insert prefix' count },
   s!"{prefix'}{count}")

def setParamValue (st : SessionSt) (name : String) (value : Json) : SessionSt :=
  match st.params.findIdx? (·.1 == name) with
  | some i => { st with params := st.params.set! i (name, value) }
  | none   => { st with params := st.params.push (name, value) }

end SessionSt

end Tropical
