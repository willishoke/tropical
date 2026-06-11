import Lean.Data.Json
import Tropical.Parse.Nodes

/-!
# Plan layer — `tropical_plan_5` as a type (Phase 6 stage 6a)

Port of `compiler/flat_plan.ts` plus the instruction/operand types from
`compiler/ir/emit_resolved.ts`. Two layers, exactly as in TS:

- **Internal**: `NOperand` / `DstSlot` / `NInstr` / `PerInstancePlan` /
  `InstanceFunction` / `SinkSpec` / `SourceSpec` / `FlatPlan`. The TS
  branded index newtypes (`TempIdx`, `StateRegIdx`, `ArraySlotIdx`,
  `ModuleSlotIdx`) erase to `Nat` here — the namespace discipline they
  enforced at TS compile time is carried by the constructor positions
  of the discriminated unions (an `NOperand.reg` slot IS a temp index;
  there is no bare-number arithmetic site left to confuse).

- **Wire**: `toWire` encoders producing the plain-JSON shape the C++
  engine parses. Key-omission rules are mirrored from `toWirePlan` /
  `toWireInstanceFn`: `compilation_mode` omitted when `fused`, `sinks`
  omitted when empty, `sources` omitted when canonical `[tick, rate]`,
  `preamble_instructions` / `pre_input_instructions` / `children`
  omitted when empty. (Structural diff is key-order-insensitive, but
  key *presence* matters to it.)

The legacy plan_4 carriers (`output_targets` / `outputs` temp-mix,
`rate`/`tick` operand kinds) are wire-format backcompat the TS parser
upgrades on read; the Lean side only ever *produces* plans, so they
have no representation here.

Numbers that are data (const vals, state init, slot defaults, sink
gain) are `Lean.JsonNumber` so decimal text re-parses to the identical
double on the TS/engine side.

A `DstSlot.sessionArray` reaching the wire encoder is the same
invariant violation it is in TS (`dstSlotToWire` throws): remap is the
only place that collapses sessionArray → array, and the encoder fails
loudly rather than emitting a phantom slot index. Operands have no
such check in TS (they serialize by blind cast), so `sessionArrayReg`
encodes verbatim — divergence-visible, exactly as the TS path behaves.
-/

namespace Tropical.Plan

open Lean (Json JsonNumber toJson)

abbrev ScalarType := Tropical.Parse.ScalarKind

private def scalarJson (t : ScalarType) : Json := Json.str t.wire

-- ─────────────────────────────────────────────────────────────
-- Operands and instructions
-- ─────────────────────────────────────────────────────────────

inductive NOperand where
  | const (val : JsonNumber) (scalarType : ScalarType)
  | input (slot : Nat) (scalarType : ScalarType)
  /-- A temp read (`kind: 'reg'` on the wire). -/
  | reg (slot : Nat) (scalarType : ScalarType)
  | arrayReg (slot : Nat)
  /-- Pre-remap-only: session-absolute array slot. Remap collapses to
      `arrayReg` (passthrough slot value) before the wire. -/
  | sessionArrayReg (slot : Nat)
  | stateReg (slot : Nat) (scalarType : ScalarType)
  | param (ptr : String) (scalarType : ScalarType)
  | source (index : Nat) (scalarType : ScalarType)
  | slot (index : Nat) (scalarType : ScalarType)
deriving Repr, Inhabited

/-- Canonical source indices (sessions always emit `[tick, rate]`). -/
def sourceTick : Nat := 0
def sourceRate : Nat := 1

def opTick : NOperand := .source sourceTick .int
def opRate : NOperand := .source sourceRate .float

def NOperand.toWire : NOperand → Json
  | .const v t => Json.mkObj [("kind", Json.str "const"), ("val", Json.num v),
      ("scalar_type", scalarJson t)]
  | .input s t => Json.mkObj [("kind", Json.str "input"), ("slot", toJson s),
      ("scalar_type", scalarJson t)]
  | .reg s t => Json.mkObj [("kind", Json.str "reg"), ("slot", toJson s),
      ("scalar_type", scalarJson t)]
  | .arrayReg s => Json.mkObj [("kind", Json.str "array_reg"), ("slot", toJson s)]
  | .sessionArrayReg s => Json.mkObj [("kind", Json.str "session_array_reg"),
      ("slot", toJson s)]
  | .stateReg s t => Json.mkObj [("kind", Json.str "state_reg"), ("slot", toJson s),
      ("scalar_type", scalarJson t)]
  | .param p t => Json.mkObj [("kind", Json.str "param"), ("ptr", Json.str p),
      ("scalar_type", scalarJson t)]
  | .source i t => Json.mkObj [("kind", Json.str "source"), ("index", toJson i),
      ("scalar_type", scalarJson t)]
  | .slot i t => Json.mkObj [("kind", Json.str "slot"), ("index", toJson i),
      ("scalar_type", scalarJson t)]

/-- Discriminated writeback namespace. -/
inductive DstSlot where
  | temp (slot : Nat)
  | array (slot : Nat)
  /-- Pre-remap-only; never reaches the wire. -/
  | sessionArray (slot : Nat)
  | moduleSlot (index : Nat)
deriving Repr, Inhabited

private def sessionArrayLeak : String :=
  "dstSlotToWire: 'sessionArray' dst leaked to wire format. " ++
  "remapInstancePlan must convert it to 'array' before serialization."

def DstSlot.wireKind : DstSlot → Except String String
  | .temp _ => .ok "temp"
  | .array _ => .ok "array"
  | .moduleSlot _ => .ok "moduleSlot"
  | .sessionArray _ => .error sessionArrayLeak

def DstSlot.wireIdx : DstSlot → Except String Nat
  | .temp s => .ok s
  | .array s => .ok s
  | .moduleSlot i => .ok i
  | .sessionArray _ => .error sessionArrayLeak

structure NInstr where
  tag : String
  dst : DstSlot
  args : Array NOperand
  loopCount : Nat := 1
  strides : Array Nat := #[]
  resultType : ScalarType
deriving Repr, Inhabited

def NInstr.toWire (i : NInstr) : Except String Json := do
  return Json.mkObj [
    ("tag", Json.str i.tag),
    ("dst", toJson (← i.dst.wireIdx)),
    ("dst_kind", Json.str (← i.dst.wireKind)),
    ("args", Json.arr (i.args.map (·.toWire))),
    ("loop_count", toJson i.loopCount),
    ("strides", toJson i.strides),
    ("result_type", scalarJson i.resultType)]

private def instrsToWire (instrs : Array NInstr) : Except String Json := do
  return Json.arr (← instrs.mapM (·.toWire))

-- ─── Instruction constructors (ports of the TS typed constructors) ──────────

def instrScalar (tag : String) (dst : Nat) (args : Array NOperand)
    (resultType : ScalarType) : NInstr :=
  { tag, dst := .temp dst, args, resultType }

def instrArray (tag : String) (dst : Nat) (args : Array NOperand)
    (loopCount : Nat) (strides : Array Nat) (resultType : ScalarType) : NInstr :=
  { tag, dst := .array dst, args, loopCount, strides, resultType }

def instrSessionArray (tag : String) (dst : Nat) (args : Array NOperand)
    (loopCount : Nat) (strides : Array Nat) (resultType : ScalarType) : NInstr :=
  { tag, dst := .sessionArray dst, args, loopCount, strides, resultType }

def instrPack (dst : Nat) (args : Array NOperand) : NInstr :=
  { tag := "Pack", dst := .array dst, args, resultType := .float }

def instrSetElement (dst : Nat) (args : Array NOperand) : NInstr :=
  { tag := "SetElement", dst := .array dst, args, resultType := .float }

def instrSessionSetElement (dst : Nat) (args : Array NOperand) : NInstr :=
  { tag := "SetElement", dst := .sessionArray dst, args, resultType := .float }

def instrIndex (dst : Nat) (args : Array NOperand) (resultType : ScalarType) : NInstr :=
  { tag := "Index", dst := .temp dst, args, resultType }

def instrWriteSlot (dst : Nat) (value : NOperand)
    (scalarType : ScalarType := .float) : NInstr :=
  { tag := "WriteSlot", dst := .moduleSlot dst, args := #[value],
    resultType := scalarType }

-- ─────────────────────────────────────────────────────────────
-- Register targets
-- ─────────────────────────────────────────────────────────────

inductive RegTarget where
  | temp (slot : Nat)
  | arrayManaged
deriving Repr, Inhabited

/-- Wire format: `-1` for arrayManaged, raw temp index otherwise. -/
def RegTarget.toWire : RegTarget → Json
  | .temp s => toJson s
  | .arrayManaged => toJson (-1 : Int)

-- ─────────────────────────────────────────────────────────────
-- PerInstancePlan — output of compileResolved
-- ─────────────────────────────────────────────────────────────

/-- State-init value: scalar (number/bool) or an inline array backing
    store for an array-typed reg. The TS type lies (`(number|boolean)[]`
    after a cast) — arrays flow through verbatim. -/
inductive StateInit where
  | num (n : JsonNumber)
  | bool (b : Bool)
  | arr (items : Array JsonNumber)
deriving Repr, Inhabited

def StateInit.toWire : StateInit → Json
  | .num n => Json.num n
  | .bool b => Json.bool b
  | .arr items => Json.arr (items.map Json.num)

def StateInit.isArr : StateInit → Bool
  | .arr _ => true | _ => false

structure PerInstancePlan where
  registerCount : Nat
  arraySlotCount : Nat
  arraySlotSizes : Array Nat
  instructions : Array NInstr
  perChildPreInput : Array (Array NInstr)
  /-- Per-output-port temp indices (local; the session compiler shifts). -/
  outputTargets : Array Nat
  registerTargets : Array RegTarget
  stateInit : Array StateInit
  registerNames : Array String
  registerTypes : Array ScalarType
  arraySlotNames : Array String
deriving Repr, Inhabited

/-- Wire encoding for the diff-emit gate (the TS side serializes the
    same shape from `emit_cmd.ts`; `PerInstancePlan` has no production
    wire format — this exists so per-program emit is comparable before
    the partitioner lands). -/
def PerInstancePlan.toWire (p : PerInstancePlan) : Except String Json := do
  return Json.mkObj [
    ("register_count", toJson p.registerCount),
    ("array_slot_count", toJson p.arraySlotCount),
    ("array_slot_sizes", toJson p.arraySlotSizes),
    ("instructions", ← instrsToWire p.instructions),
    ("per_child_pre_input", Json.arr (← p.perChildPreInput.mapM instrsToWire)),
    ("output_targets", toJson p.outputTargets),
    ("register_targets", Json.arr (p.registerTargets.map (·.toWire))),
    ("state_init", Json.arr (p.stateInit.map (·.toWire))),
    ("register_names", toJson p.registerNames),
    ("register_types", Json.arr (p.registerTypes.map scalarJson)),
    ("array_slot_names", toJson p.arraySlotNames)]

-- ─────────────────────────────────────────────────────────────
-- InstanceFunction — a per-instance slice inside a FlatPlan
-- ─────────────────────────────────────────────────────────────

inductive InstanceFunction where
  | mk (name : String)
       (instanceName : String)
       (preambleInstructions : Array NInstr)
       (instructions : Array NInstr)
       (preInputInstructions : Array NInstr)
       (registerOffset : Nat)
       (stateRegOffset : Nat)
       (arraySlotOffset : Nat)
       (registerCount : Nat)
       (registerTargets : Array RegTarget)
       (children : Array InstanceFunction)
deriving Inhabited

namespace InstanceFunction

def name : InstanceFunction → String
  | .mk n .. => n

def instanceName : InstanceFunction → String
  | .mk _ i .. => i

def preambleInstructions : InstanceFunction → Array NInstr
  | .mk _ _ p .. => p

def instructions : InstanceFunction → Array NInstr
  | .mk _ _ _ i .. => i

def preInputInstructions : InstanceFunction → Array NInstr
  | .mk _ _ _ _ p .. => p

def registerOffset : InstanceFunction → Nat
  | .mk _ _ _ _ _ r .. => r

def stateRegOffset : InstanceFunction → Nat
  | .mk _ _ _ _ _ _ s .. => s

def arraySlotOffset : InstanceFunction → Nat
  | .mk _ _ _ _ _ _ _ a .. => a

def registerCount : InstanceFunction → Nat
  | .mk _ _ _ _ _ _ _ _ c .. => c

def registerTargets : InstanceFunction → Array RegTarget
  | .mk _ _ _ _ _ _ _ _ _ t _ => t

def children : InstanceFunction → Array InstanceFunction
  | .mk _ _ _ _ _ _ _ _ _ _ c => c

/-- Replace the pre-input block (the parent attaches each per-child
    block after compiling its own body). -/
def withPreInput (f : InstanceFunction) (block : Array NInstr) : InstanceFunction :=
  match f with
  | .mk n i pre instrs _ ro so ao rc rt ch => .mk n i pre instrs block ro so ao rc rt ch

/-- Mirrors `toWireInstanceFn`: preamble/pre_input/children omitted
    when empty so legacy JSON consumers see the bytes they expect. -/
partial def toWire (f : InstanceFunction) : Except String Json := do
  let base := #[
    ("name", Json.str f.name),
    ("instance_name", Json.str f.instanceName),
    ("instructions", ← instrsToWire f.instructions),
    ("register_offset", toJson f.registerOffset),
    ("state_reg_offset", toJson f.stateRegOffset),
    ("array_slot_offset", toJson f.arraySlotOffset),
    ("register_count", toJson f.registerCount),
    ("register_targets", Json.arr (f.registerTargets.map RegTarget.toWire))]
  let base ← if f.preambleInstructions.isEmpty then pure base
    else do pure <| base.push ("preamble_instructions", ← instrsToWire f.preambleInstructions)
  let base ← if f.preInputInstructions.isEmpty then pure base
    else do pure <| base.push ("pre_input_instructions", ← instrsToWire f.preInputInstructions)
  let base ← if f.children.isEmpty then pure base
    else do
      let kids ← f.children.mapM toWire
      pure <| base.push ("children", Json.arr kids)
  return Json.mkObj base.toList

end InstanceFunction

-- ─────────────────────────────────────────────────────────────
-- Sinks and sources
-- ─────────────────────────────────────────────────────────────

structure SinkSpec where
  /-- Output module-slot indices summed into this sink. -/
  inputs : Array Nat
  gain : JsonNumber
  target : Nat
deriving Repr, Inhabited

/-- Default sink gain — 1/20, the v1 headroom scale as data. -/
def defaultSinkGain : JsonNumber := ⟨5, 2⟩  -- 5 × 10⁻² = 0.05

def SinkSpec.toWire (s : SinkSpec) : Json :=
  Json.mkObj [("inputs", toJson s.inputs), ("gain", Json.num s.gain),
    ("target", toJson s.target)]

inductive SourceKind where
  | tick | rate
deriving BEq, Repr, Inhabited

def SourceKind.wire : SourceKind → String
  | .tick => "tick" | .rate => "rate"

/-- Canonical source ordering: `[tick, rate]`. -/
def defaultSources : Array SourceKind := #[.tick, .rate]

def isDefaultSources (s : Array SourceKind) : Bool :=
  s == defaultSources

-- ─────────────────────────────────────────────────────────────
-- CompilationMode
-- ─────────────────────────────────────────────────────────────

inductive CompilationMode where
  | fused | microkernel | microkernelDeep
deriving BEq, Repr, Inhabited

def CompilationMode.wire : CompilationMode → String
  | .fused => "fused"
  | .microkernel => "microkernel"
  | .microkernelDeep => "microkernel-deep"

def CompilationMode.ofWire? : String → Option CompilationMode
  | "fused" => some .fused
  | "microkernel" => some .microkernel
  | "microkernel-deep" => some .microkernelDeep
  | _ => none

-- ─────────────────────────────────────────────────────────────
-- FlatPlan — the runnable plan
-- ─────────────────────────────────────────────────────────────

structure FlatPlan where
  sampleRate : JsonNumber := (44100 : Nat)
  compilationMode : CompilationMode := .fused
  stateInit : Array StateInit
  registerNames : Array String
  registerTypes : Array ScalarType
  arraySlotNames : Array String
  registerCount : Nat
  arraySlotCount : Nat
  arraySlotSizes : Array Nat
  instanceFunctions : Array InstanceFunction
  sinks : Array SinkSpec
  sources : Array SourceKind := defaultSources
  slotCount : Nat
  slotNames : Array String
  /-- Numbers in TS; delay-slot inits and param values land here
      verbatim (raw Json so lexical number forms survive). -/
  slotDefaults : Array Json
deriving Inhabited

/-- Mirrors `toWirePlan`'s omission rules. -/
def FlatPlan.toWire (p : FlatPlan) : Except String Json := do
  let fields := #[("schema", Json.str "tropical_plan_5"),
    ("config", Json.mkObj [("sampleRate", Json.num p.sampleRate)])]
  let fields := if p.compilationMode == .fused then fields
    else fields.push ("compilation_mode", Json.str p.compilationMode.wire)
  let fields := fields
    ++ #[("state_init", Json.arr (p.stateInit.map (·.toWire))),
      ("register_names", toJson p.registerNames),
      ("register_types", Json.arr (p.registerTypes.map scalarJson)),
      ("array_slot_names", toJson p.arraySlotNames),
      ("register_count", toJson p.registerCount),
      ("array_slot_count", toJson p.arraySlotCount),
      ("array_slot_sizes", toJson p.arraySlotSizes),
      ("slot_count", toJson p.slotCount),
      ("slot_names", toJson p.slotNames),
      ("slot_defaults", Json.arr p.slotDefaults),
      ("instance_functions", Json.arr (← p.instanceFunctions.mapM (·.toWire)))]
  let fields := if p.sinks.isEmpty then fields
    else fields.push ("sinks", Json.arr (p.sinks.map (·.toWire)))
  let fields := if isDefaultSources p.sources then fields
    else fields.push ("sources", Json.arr (p.sources.map fun s =>
      Json.mkObj [("kind", Json.str s.wire)]))
  return Json.mkObj fields.toList

end Tropical.Plan
