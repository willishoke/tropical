import Std.Data.HashMap
import Lean.Data.Json
import Tropical.Expr
import Tropical.WireExpr
import Tropical.Ir.Codec
import Tropical.Plan

/-!
# Session state — the authoritative graph topology

The Lean frontend owns the complete live session: registered program metadata
and resolved roots, ordered instances and typed wires, graph outputs, control
values and disciplines, and name counters. `Engine.Compile.syncCompile`
constructs a synthetic resolved root from this state and emits/loads the
kernel directly; there is no compiler-service or TypeScript-session mirror.

Wires are already in the closed `WireExpr` grammar. Delay/state spellings are
refused during decoding rather than preserved for a later pass.

`instances` and `wires` remain ordered arrays, and replacing a wire keeps its
position. Tool output and deterministic session construction depend on that
order.
-/

namespace Tropical

open Lean (Json)

/-- One port's metadata from a catalog entry. -/
structure PortInfo where
  name    : String
  typeStr : Option String := none   -- display string ("float", "float[8]")
  typeObj : Option Json := none     -- structured PortType (get_info echo, wire checks)
  default : Option WireExpr := none -- typed ExprNode default (inputs only)
deriving Inhabited

/-- A registered program's current port metadata. `registers` is retained only
    while old catalog-shaped inputs are decoded; production builders leave it
    empty and current kernels have no persistent registers. -/
structure ProgMeta where
  programName : String
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
    -- Entry defaults come from `literalDefault`, which only emits the
    -- compilable literal-class forms — decode cannot fail on them; a
    -- malformed default from a hand-written entry degrades to none.
    default := (nonNull "default").bind fun j => (WireExpr.ofJson j).toOption }

def ProgMeta.fromEntry (j : Json) : ProgMeta :=
  let ports (k : String) : Array PortInfo :=
    match Tropical.Expr.getField? j k with
    | some (.arr a) => a.map parsePort
    | _ => #[]
  { programName := (Tropical.Expr.getStrField? j "program_name").getD ""
    inputs      := ports "inputs"
    outputs     := ports "outputs"
    registers   := ports "registers" }

def ProgMeta.inputNames (m : ProgMeta) : Array String := m.inputs.map (·.name)
def ProgMeta.outputNames (m : ProgMeta) : Array String := m.outputs.map (·.name)

/-- A live instance: its registered type name, current port metadata, and the
    resolved-root snapshot captured when the instance is created or adopted.
    The root builder links that snapshot rather than performing a late name
    lookup. `typeArgs` remains in the tool/session shape so obsolete generic
    requests can be rejected explicitly; current programs are concrete. -/
structure InstanceInfo where
  baseTypeName : String
  typeArgs     : Option Json := none
  progMeta     : ProgMeta
  resolvedIdx  : Option Tropical.Ir.ProgramIdx := none
deriving Inhabited

/-- A stored wire. The expression is TYPED — every ingest path decodes
    through `WireExpr` (the refusal site), so the store cannot hold a
    spelling the grammar lacks. -/
structure Wire where
  instName : String
  portName : String
  expr     : WireExpr
deriving Inhabited

def Wire.key (w : Wire) : String := s!"{w.instName}:{w.portName}"

structure ScopeTap where
  name : String
  sourceInstance : String
  sourceOutput : String
deriving Inhabited

def ScopeTap.slot (tap : ScopeTap) : String :=
  s!"{tap.sourceInstance}.{tap.sourceOutput}"

/-- One source routed to an independently addressable DAC channel. Multiple
    routes may target the same channel; compilation folds them into one sink
    in authored source order. -/
structure GraphOutput where
  channel : Nat := 0
  sourceInstance : String
  sourceOutput : String
deriving Inhabited, Repr

structure SessionSt where
  /-- Program registration order (stdlib, then exported session types). -/
  catalogOrder : Array String := #[]
  programs     : Std.HashMap String ProgMeta := {}
  instances    : Array (String × InstanceInfo) := #[]
  wires        : Array Wire := #[]
  graphOutputs : Array GraphOutput := #[]
  /-- Opt-in observation points — like `graphOutputs` (dac) but routed to a
      render_window-readable slot instead of the audio output. The keep-set
      the collapse honors: params ∪ dac ∪ scope-taps stay materialized. -/
  scopeTaps    : Array ScopeTap := #[]
  /-- Param mirror: name → last-known value (raw Json preserves lexical number
      forms). The loaded runtime owns the live module slots. -/
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
  /-- Registered program name → resolved arena index. The historical field
      name remains internal, but every current entry is concrete: JSON
      load/merge/export and stdlib boot resolve registered instance types
      through this map. Distinct from `resolvedByName`, which is keyed by the
      decoded program's own name. -/
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

/-- Replacing an existing key keeps its deterministic position. -/
def setWireRaw (st : SessionSt) (instName portName : String) (expr : WireExpr) : SessionSt :=
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
    decoded `name`. Returns `none` when an entry carries no resolved program;
    current registered production entries do carry one. The bridge from
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
