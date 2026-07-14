import Std.Data.HashMap
import Lean.Data.Json
import Tropical.Expr
import Tropical.Ir.Codec
import Tropical.Plan

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

/-- A live instance: base type name, type args (generics), the
    (possibly specialized) port metadata, and a snapshot of the typed
    store's program for this instance's type — captured at
    add/replicate/lift/load-adoption time. The snapshot is the parity
    image of TS instances holding their `compiled.prog`: the compile
    root resolver links against the instance's snapshot, never a late
    name lookup (redefinition would diverge otherwise). -/
structure InstanceInfo where
  baseTypeName : String
  typeArgs     : Option Json := none
  progMeta     : ProgMeta
  resolvedIdx  : Option Tropical.Ir.ProgramIdx := none
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
  /-- Scope taps: (tapName, srcInstance, srcOutput). Opt-in observation
      points — like `graphOutputs` (dac) but routed to a render_window-
      readable slot instead of the audio output. The keep-set the collapse
      honors: params ∪ dac ∪ scope-taps stay materialized. -/
  scopeTaps    : Array (String × String × String) := #[]
  /-- Param mirror: name → last-known value (raw Json to preserve lexical
      number forms). The service owns the live Param handles. -/
  params       : Array (String × Json) := #[]
  /-- Host-contract dispatch table for the LOADED plan (name → discipline),
      seeded from the plan's own `param_disciplines` at load — the unified
      `set_param` dispatches from this, so no client ever chooses a verb.
      Empty (a session-model patch, an old plan) ⇒ every write is raw. -/
  paramDisciplines : Array Tropical.Plan.ParamDiscipline := #[]
  nameCounters : Std.HashMap String Nat := {}
  /-- The typed store (Phase 4 stage 4a): every adopted catalog entry's
      post-strata resolved IR, appended via `decodeResolvedInto`. The
      arena only ever grows; indices stay valid for the session's life. -/
  arena        : Tropical.Ir.Arena := {}
  /-- Current name → store index, keyed by the stored program's `name`
      (the decoded `prog.name`). Redefinition overwrites; instances are
      insulated by their `resolvedIdx` snapshots. -/
  resolvedByName : Std.HashMap String Tropical.Ir.ProgramIdx := {}
  /-- The registration-path mirror of TS `session.programs` (Phase 4
      stage 4b): for every *registered* program name, the arena index of
      the form the TS map holds — post-strata for concrete programs
      (adopted from the service entry), the engine's own raw elaborated
      template for generics. `define_program` elaboration resolves
      cross-program references through this map, exactly like the TS
      elaborator's `externalResolver` over `session.programs`. Distinct
      from `resolvedByName`, which is keyed by *decoded* prog names and
      also collects specializations adopted via `resolve_type`. -/
  templateByName : Std.HashMap String Tropical.Ir.ProgramIdx := {}

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

/-- Decode a catalog entry's `resolved` field (when present and
    non-null) into the typed store, recording the program under its own
    decoded `name`. Returns the adopted store index — `none` for
    generic templates, which ship `resolved: null`. The bridge from
    `Lean.Json` to the codec's ordered `JsonV` is a re-parse of the
    compressed string (lossless: `JsonNumber` decimals survive, and the
    resolved wire has no order-sensitive objects). -/
def adoptResolved (st : SessionSt) (entry : Json) :
    Except String (SessionSt × Option Tropical.Ir.ProgramIdx) :=
  match Tropical.Expr.getField? entry "resolved" with
  | none | some .null => .ok (st, none)
  | some r => do
    let jv ← (Tropical.Parse.JsonV.parse r.compress).mapError
      (s!"entry.resolved: JSON re-parse failed: {·}")
    let (arena, idx) ← Tropical.Ir.Codec.decodeResolvedInto st.arena jv
    let some p := arena.program? idx
      | .error "entry.resolved: decodeResolvedInto returned an out-of-range root"
    .ok ({ st with arena, resolvedByName := st.resolvedByName.insert p.name idx }, some idx)

end SessionSt

end Tropical
