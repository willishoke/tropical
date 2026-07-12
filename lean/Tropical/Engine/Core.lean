import Std.Data.HashMap
import Tropical.Errors
import Tropical.Ffi
import Tropical.Expr
import Tropical.Wiring
import Tropical.Session
import Tropical.Lowering
import Tropical.Parse.Nodes
import Tropical.Parse.Raise
import Tropical.Ir.Elaborator
import Tropical.Ir.Strata
import Tropical.Ir.Core
import Tropical.Ir.WireProgram
import Tropical.Ir.EmitLlvm
import Tropical.Ir.EmitMsl
import Tropical.StagedLoad
import Tropical.TypeArgs
import Tropical.Compile
import Tropical.Entries
import Tropical.Playground

/-!
# Engine.Core — session env, tool-arg access, and lookup vocabulary

`Env` (the runtime handle: session store, native runtime, DAC, params) and
`EngineM`, plus the shared helpers every tool handler draws on: typed
tool-argument access over `Json`, the instance/port lookup helpers with their
ported failure shapes, and `adaptInputExpr` (input type-check + auto-broadcast).
The helpers exposed here are the engine's shared vocabulary — non-`private`
because every downstream handler module consumes them.
-/

namespace Tropical.Engine

open Lean (Json toJson)
open Tropical.Expr (getField? getStrField? opOf? validateExpr exprDependencies prettyExpr)
open Tropical.Wiring (parsePortType? checkArrayConnection PortType)

structure Env where
  state   : IO.Ref SessionSt
  /-- The native FlatRuntime this engine owns: plans load here; the DAC
      reads from it; param slots are driven on it. -/
  runtime : Ffi.Runtime
  dac     : IO.Ref (Option Ffi.Dac)
  /-- `TROPICAL_BACKEND=metal` at boot: every session compile dual-loads
      (LLVM IR → JIT + MSL → GPU pipeline) and audio dispatches on Metal.
      The JIT keeps serving render_window/the scope either way. -/
  metalBackend : Bool := false
  /-- `TROPICAL_ARROW` at boot: build the resolved session root directly via
      `sessionToResolvedRoot` (the arrow path) instead of the legacy
      `sessionToParsed → reparse → elaborate` round-trip. Read once here rather
      than per graph mutation; the two paths are plan-byte-identical (gated by
      `runSessionViaArrowEquiv`). -/
  arrowRoot : Bool := false

-- Reserved audio-output boundary leaf.
def dacName : String := "dac"
def dacOut : String := "out"
def scopeName : String := "scope"

-- ── Json access helpers ──────────────────────────────────────────────────────

/-- Field access with `??` semantics: absent and `null` are both `none`. -/
def arg? (args : Json) (k : String) : Option Json :=
  match getField? args k with
  | some .null => none
  | v => v

def argStr? (args : Json) (k : String) : Option String :=
  match arg? args k with
  | some (.str s) => some s
  | _ => none

def argArr (args : Json) (k : String) : Array Json :=
  match arg? args k with
  | some (.arr a) => a
  | _ => #[]

def jsonNull : Json := Json.null

/-- Render a Json value the way a TS template literal renders it
    (`${x}`): bare strings unquoted, everything else as JSON. -/
def tsInterp : Json → String
  | .str s => s
  | j => j.compress

-- ── Lookup helpers (ported failure shapes) ───────────────────────────────────

def requireInstance (st : SessionSt) (name : String) (param : String) : EngineM InstanceInfo :=
  match st.findInstance? name with
  | some info => pure info
  | none => throwEnum .unknownInstance param (Json.str name) st.instanceNames

/-- Output name-or-index → index. Numbers (and digit strings) pass
    through unchecked, mirroring TS. -/
private def resolveOutputIdx (pm : ProgMeta) (nameOrIdx : Json) : EngineM Nat :=
  match nameOrIdx with
  | .num n => pure n.toFloat.toUInt64.toNat
  | .str s =>
    if s.toList.all Char.isDigit && !s.isEmpty then
      pure s.toNat!
    else
      match pm.outputNames.idxOf? s with
      | some i => pure i
      | none => throwEnum .unknownOutput "output" (Json.str s) pm.outputNames
  | j => throwEnum .unknownOutput "output" j pm.outputNames

def resolveInputIdx (pm : ProgMeta) (nameOrIdx : Json) : EngineM Nat :=
  match nameOrIdx with
  | .num n => pure n.toFloat.toUInt64.toNat
  | .str s =>
    if s.toList.all Char.isDigit && !s.isEmpty then
      pure s.toNat!
    else
      match pm.inputNames.idxOf? s with
      | some i => pure i
      | none => throwEnum .unknownInput "input" (Json.str s) pm.inputNames
  | j => throwEnum .unknownInput "input" j pm.inputNames

def resolveOutputName (pm : ProgMeta) (nameOrIdx : Json) : EngineM String := do
  let idx ← resolveOutputIdx pm nameOrIdx
  pure <| (pm.outputNames[idx]?).getD (toString idx)

def resolveInputName (pm : ProgMeta) (nameOrIdx : Json) : EngineM String := do
  let idx ← resolveInputIdx pm nameOrIdx
  pure <| (pm.inputNames[idx]?).getD (toString idx)

def inputTypeObj (pm : ProgMeta) (idx : Nat) : Option Json :=
  (pm.inputs[idx]?).bind (·.typeObj)

-- ── adaptInputExpr (type-check + auto-broadcast) ─────────────────────────────

private def scalarTypeJson (k : String) : Json :=
  Json.mkObj [("kind", Json.str "scalar"), ("scalar", Json.str k)]

/-- Infer a source port type. Returns both the parsed form (for the
    connection check) and the structured PortType Json (echoed verbatim
    as the `got` field of `type_mismatch` envelopes — TS passes the
    PortType object itself). -/
private def srcTypeOf (st : SessionSt) (node : Json) : Option (PortType × Json) :=
  match node with
  | .num _  => some (.scalar .float, scalarTypeJson "float")
  | .bool _ => some (.scalar .bool, scalarTypeJson "bool")
  | .arr items =>
    some (.array { display := "float", kind := some .float } #[items.size],
          Json.mkObj [("kind", Json.str "array"), ("element", Json.str "float"),
                      ("shape", Json.arr #[Lean.toJson items.size])])
  | .obj _ =>
    if opOf? node == some "ref" then do
      let instName ← getStrField? node "instance"
      let info ← st.findInstance? instName
      let outIdx ← match getField? node "output" with
        | some (.num n) => some n.toFloat.toUInt64.toNat
        | some (.str s) => info.progMeta.outputNames.idxOf? s
        | _ => none
      let port ← info.progMeta.outputs[outIdx]?
      let typeObj ← port.typeObj
      let parsed ← parsePortType? typeObj
      pure (parsed, typeObj)
    else none
  | _ => none

def adaptInputExpr (st : SessionSt) (node : Json) (dstTypeObj : Option Json)
    (instanceName inputName : String) : EngineM Json := do
  match srcTypeOf st node with
  | none => pure node
  | some (srcType, srcTypeJson) =>
    let dstType := dstTypeObj.bind parsePortType?
    let check := checkArrayConnection (some srcType) dstType node
    if !check.compatible then
      throwPredicate .typeMismatch "expr" node "type_compatible"
        dstTypeObj (some srcTypeJson)
        (some s!"Type mismatch on '{instanceName}'.{inputName}: {check.error.getD ""}")
    pure (check.broadcastExpr.getD node)

end Tropical.Engine
