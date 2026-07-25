import Lean.Data.Json
import Tropical.Parse.Nodes
import Tropical.Ir.Core

/-!
# Plan layer — `tropical_plan_5` as a type

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
  /-- A temp read (`kind: 'reg'` on the wire). This is the SSA temp pool,
      not per-sample state — CF-only has no state operands. -/
  | reg (slot : Nat) (scalarType : ScalarType)
  | arrayReg (slot : Nat)
  /-- Pre-remap-only: session-absolute array slot. Remap collapses to
      `arrayReg` (passthrough slot value) before the wire. -/
  | sessionArrayReg (slot : Nat)
  | param (ptr : String) (scalarType : ScalarType)
  | source (index : Nat) (scalarType : ScalarType)
  | slot (index : Nat) (scalarType : ScalarType)
  /-- The iteration index (i64) of the enclosing `ReduceBegin`/`ReduceEnd`
      region whose binder id is `id` (nested regions: the emitters resolve the
      id against the stack of open regions, innermost first). Meaningless
      outside a region with this id (emitters reject it there). `id = 0` is
      the pre-nesting default and is OMITTED on the wire, so single-region
      plans serialize byte-identically to the id-less form. -/
  | loopIdx (id : Nat := 0)
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
  | .param p t => Json.mkObj [("kind", Json.str "param"), ("ptr", Json.str p),
      ("scalar_type", scalarJson t)]
  | .source i t => Json.mkObj [("kind", Json.str "source"), ("index", toJson i),
      ("scalar_type", scalarJson t)]
  | .slot i t => Json.mkObj [("kind", Json.str "slot"), ("index", toJson i),
      ("scalar_type", scalarJson t)]
  | .loopIdx id => Json.mkObj <|
      [("kind", Json.str "loop_idx")]
      ++ (if id == 0 then [] else [("id", toJson id)])
      ++ [("scalar_type", scalarJson .int)]

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

/-- The scalar-op signature Σ: the arithmetic / logical / bitwise / cast
    operations a plan instruction's `tag` names when it computes a scalar into a
    temp or slot. Structural tags (`Pack`, `Index`, `ReduceBegin`, `WriteSlot`,
    …) are NOT here — they route through their own emit paths, not `emitOp`.

    This is the one object every backend interprets as a TOTAL algebra: the type
    inference (`Emit.resultType`), the LLVM emitter, the MSL emitter, and the
    MSL constant folder each match on it exhaustively, so extending the signature
    forces every interpretation to define the new symbol (a partial `| _ => fail`
    would not be an algebra). `name`/`ofString?` are the (de)serialization to the
    plan wire tag; `ofString?` is the single point where an unknown scalar op is
    rejected. -/
inductive PlanOp where
  | add | sub | mul | div | mod | floorDiv
  | less | lessEq | greater | greaterEq | equal | notEqual
  | and | or
  | bitAnd | bitOr | bitXor | lshift | rshift | ldexp
  | neg | abs | sqrt | floor | ceil | round
  | not | bitNot | floatExponent | toInt | toBool | toFloat
  | select | clamp
deriving BEq, Repr, Inhabited

/-- The plan wire tag for a scalar op (Capitalized; the `tropical_plan_5`
    instruction `tag`). -/
def PlanOp.name : PlanOp → String
  | .add => "Add" | .sub => "Sub" | .mul => "Mul" | .div => "Div"
  | .mod => "Mod" | .floorDiv => "FloorDiv"
  | .less => "Less" | .lessEq => "LessEq" | .greater => "Greater"
  | .greaterEq => "GreaterEq" | .equal => "Equal" | .notEqual => "NotEqual"
  | .and => "And" | .or => "Or"
  | .bitAnd => "BitAnd" | .bitOr => "BitOr" | .bitXor => "BitXor"
  | .lshift => "LShift" | .rshift => "RShift" | .ldexp => "Ldexp"
  | .neg => "Neg" | .abs => "Abs" | .sqrt => "Sqrt"
  | .floor => "Floor" | .ceil => "Ceil" | .round => "Round"
  | .not => "Not" | .bitNot => "BitNot" | .floatExponent => "FloatExponent"
  | .toInt => "ToInt" | .toBool => "ToBool" | .toFloat => "ToFloat"
  | .select => "Select" | .clamp => "Clamp"

/-- Parse a plan wire tag into the scalar-op signature — the single validation
    point for scalar ops (an unrecognized tag is not in Σ). -/
def PlanOp.ofString? : String → Option PlanOp
  | "Add" => some .add | "Sub" => some .sub | "Mul" => some .mul
  | "Div" => some .div | "Mod" => some .mod | "FloorDiv" => some .floorDiv
  | "Less" => some .less | "LessEq" => some .lessEq | "Greater" => some .greater
  | "GreaterEq" => some .greaterEq | "Equal" => some .equal | "NotEqual" => some .notEqual
  | "And" => some .and | "Or" => some .or
  | "BitAnd" => some .bitAnd | "BitOr" => some .bitOr | "BitXor" => some .bitXor
  | "LShift" => some .lshift | "RShift" => some .rshift | "Ldexp" => some .ldexp
  | "Neg" => some .neg | "Abs" => some .abs | "Sqrt" => some .sqrt
  | "Floor" => some .floor | "Ceil" => some .ceil | "Round" => some .round
  | "Not" => some .not | "BitNot" => some .bitNot | "FloatExponent" => some .floatExponent
  | "ToInt" => some .toInt | "ToBool" => some .toBool | "ToFloat" => some .toFloat
  | "Select" => some .select | "Clamp" => some .clamp
  | _ => none

/-- Classification of the signature (properties of an op symbol, so a total
    predicate with a default arm is fine — unlike the emit algebras, which must
    be exhaustive). -/
def PlanOp.isBitwise : PlanOp → Bool
  | .bitAnd | .bitOr | .bitXor | .lshift | .rshift | .bitNot => true
  | _ => false

def PlanOp.isComparison : PlanOp → Bool
  | .less | .lessEq | .greater | .greaterEq | .equal | .notEqual
  | .not | .and | .or => true
  | _ => false

def PlanOp.isTranscendental : PlanOp → Bool
  | .sqrt | .floor | .ceil | .round | .ldexp | .floatExponent => true
  | _ => false

def promoteTypes (a b : ScalarType) : ScalarType :=
  if a == .float || b == .float then .float
  else if a == .int || b == .int then .int
  else .bool

/-- The type-inference algebra over the signature: the result sort of an op from
    its argument sorts. Casts fix the sort; bitwise → int, comparison → bool,
    transcendental → float; `select`/`clamp` read the arm/value sort; the rest
    promote across args (head as seed — `promoteTypes` is idempotent). -/
def PlanOp.resultType : PlanOp → Array ScalarType → ScalarType
  | .toInt,   _ => .int
  | .toBool,  _ => .bool
  | .toFloat, _ => .float
  | op, argTypes =>
    if op.isBitwise then .int
    else if op.isComparison then .bool
    else if op.isTranscendental then .float
    else match op with
      | .select => promoteTypes (argTypes[1]?.getD .float) (argTypes[2]?.getD .float)
      | .clamp => argTypes[0]?.getD .float
      | _ => if argTypes.isEmpty then .float else argTypes.foldl promoteTypes argTypes[0]!

structure NInstr where
  tag : String
  dst : DstSlot
  args : Array NOperand
  loopCount : Nat := 1
  strides : Array Nat := #[]
  resultType : ScalarType
  /-- `ReduceBegin` only: the binder id its body's `loopIdx id` operands refer
      to (nested banks). Rides the wire as `loop_id`, OMITTED when 0 so
      single-region plans stay byte-identical; decoding an absent field yields
      0, and a plan with one open region resolves id-0 `loopIdx` against it. -/
  loopId : Nat := 0
deriving Repr, Inhabited

def NInstr.toWire (i : NInstr) : Except String Json := do
  return Json.mkObj <| [
    ("tag", Json.str i.tag),
    ("dst", toJson (← i.dst.wireIdx)),
    ("dst_kind", Json.str (← i.dst.wireKind)),
    ("args", Json.arr (i.args.map (·.toWire))),
    ("loop_count", toJson i.loopCount)]
    ++ (if i.loopId == 0 then [] else [("loop_id", toJson i.loopId)])
    ++ [
    ("strides", toJson i.strides),
    ("result_type", scalarJson i.resultType)]

/-- Wire-encode an instruction stream. Not `private`: the harness's
    per-program plan encoding (`Testing/PlanWire`) shares it. -/
def instrsToWire (instrs : Array NInstr) : Except String Json := do
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

/-- Open an indexed-reduction region: `dst` is the accumulator temp
    (seeded from `init`), `loopCount` the trip count. The instructions
    up to the matching `ReduceEnd` run once per iteration; within them
    `.loopIdx` is the iteration index and reads/writes of the
    accumulator temp see/update the running value. Loop-body temps do
    not escape the region (post-region reads fall back to the
    zero-initialized scratch, the emitters' usual graceful rule).

    `count?` (trip-count-as-data): an optional RUNTIME effective count as a
    second arg — `loopCount` stays the static capacity; the emitters resolve
    `args[1]` once before the loop and trip `clamp(args[1], 0, loopCount)`
    iterations. Absent = the static path, byte-identical emission.

    `loopId` (nested banks): the region's binder id — body `loopIdx id`
    operands resolve against it through the stack of open regions. 0 (the
    default) is the single-region form and is omitted on the wire. -/
def instrReduceBegin (accTemp : Nat) (init : NOperand) (loopCount : Nat)
    (resultType : ScalarType) (count? : Option NOperand := none)
    (loopId : Nat := 0) : NInstr :=
  { tag := "ReduceBegin", dst := .temp accTemp,
    args := match count? with | none => #[init] | some c => #[init, c],
    loopCount, resultType, loopId }

/-- Close the innermost reduction region opened on `accTemp`. -/
def instrReduceEnd (accTemp : Nat) (resultType : ScalarType) : NInstr :=
  { tag := "ReduceEnd", dst := .temp accTemp, args := #[], resultType }

-- ─────────────────────────────────────────────────────────────
-- PerInstancePlan — output of compileResolved
-- ─────────────────────────────────────────────────────────────

structure PerInstancePlan where
  registerCount : Nat
  arraySlotCount : Nat
  arraySlotSizes : Array Nat
  instructions : Array NInstr
  perChildPreInput : Array (Array NInstr)
  /-- Per-output-port temp indices (local; the session compiler shifts). -/
  outputTargets : Array Nat
  arraySlotNames : Array String
  /-- Staging metadata, parallel to `instructions` / `perChildPreInput`:
      the binding-time stage of the node each instruction was emitted
      for (typed stage-0 refactor Phase 1). Never serialized. -/
  instrStages : Array (Option Tropical.Ir.Stage) := #[]
  perChildPreInputStages : Array (Array (Option Tropical.Ir.Stage)) := #[]
deriving Repr, Inhabited

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
       (arraySlotOffset : Nat)
       (registerCount : Nat)
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

def arraySlotOffset : InstanceFunction → Nat
  | .mk _ _ _ _ _ _ a .. => a

def registerCount : InstanceFunction → Nat
  | .mk _ _ _ _ _ _ _ c .. => c

def children : InstanceFunction → Array InstanceFunction
  | .mk _ _ _ _ _ _ _ _ c => c

/-- Replace the pre-input block (the parent attaches each per-child
    block after compiling its own body). -/
def withPreInput (f : InstanceFunction) (block : Array NInstr) : InstanceFunction :=
  match f with
  | .mk n i pre instrs _ ro ao rc ch => .mk n i pre instrs block ro ao rc ch

/-- Mirrors `toWireInstanceFn`: preamble/pre_input/children omitted
    when empty so legacy JSON consumers see the bytes they expect. -/
partial def toWire (f : InstanceFunction) : Except String Json := do
  let base := #[
    ("name", Json.str f.name),
    ("instance_name", Json.str f.instanceName),
    ("instructions", ← instrsToWire f.instructions),
    ("register_offset", toJson f.registerOffset),
    ("array_slot_offset", toJson f.arraySlotOffset),
    ("register_count", toJson f.registerCount)]
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

/-- Default sink gain — UNITY. The device sink is a pure summer: it mixes its
    input slots and hands the total to the channel untouched. Amplitude is a
    frontend concern (the playground carries a live `master.gain` VCA in its
    graph; other frontends own their own level), so the backend invents no
    headroom scale of its own. (Was 1/20 — the v1 headroom hack, now gone.) -/
def defaultSinkGain : JsonNumber := ⟨1, 0⟩  -- 1 × 10⁰ = 1.0

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

theorem CompilationMode.ofWire_wire (m : CompilationMode) :
    CompilationMode.ofWire? m.wire = some m := by
  cases m <;> rfl

-- ─────────────────────────────────────────────────────────────
-- FlatPlan — the runnable plan
-- ─────────────────────────────────────────────────────────────

/-- Per-param write discipline — HOST-CONTRACT data. A runtime host (the C++
    FlatRuntime, a Swift/Metal host, the wasm player) reads this from the
    manifest and dispatches param writes itself, so no client ever chooses
    semantics (see design/host-param-dispatch.md for the normative
    re-anchoring math each discipline requires). `name` is the base param
    (slot `param:<name>`); `companions` are the discipline's implementation
    slots (`#v0/#v1/#t0` for glide, `#phase` for anchor, the `tau_base`
    sibling for velocity). -/
structure ParamDiscipline where
  name : String
  discipline : String
  glideDurSec : Option JsonNumber := none
  companions : Array String := #[]
deriving Repr, Inhabited

def ParamDiscipline.toWire (d : ParamDiscipline) : Json :=
  let fields := #[("name", Json.str d.name), ("discipline", Json.str d.discipline)]
  let fields := match d.glideDurSec with
    | some n => fields.push ("glide_dur_sec", Json.num n)
    | none => fields
  let fields := if d.companions.isEmpty then fields
    else fields.push ("companions", toJson d.companions)
  Json.mkObj fields.toList

structure FlatPlan where
  sampleRate : JsonNumber := (44100 : Nat)
  compilationMode : CompilationMode := .fused
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
  /-- Host-contract dispatch table (empty for plans with no live params;
      omitted from the wire when empty, so old plans and old parsers are
      both untouched). -/
  paramDisciplines : Array ParamDiscipline := #[]
  /-- Array slot indices FILLED by the stage-0 coefficient kernel (banks-as-data
      coefficient columns). The runtime double-buffers exactly these — the coeff
      kernel writes a back generation and flips one atomic word so the audio
      kernel reads a whole, consistent generation of columns (no cross-column
      tear on a live knob move). Empty ⇒ no double-buffering (omitted from wire). -/
  coeffArraySlots : Array Nat := #[]
deriving Inhabited

/-- Mirrors `toWirePlan`'s omission rules. -/
def FlatPlan.toWire (p : FlatPlan) : Except String Json := do
  let fields := #[("schema", Json.str "tropical_plan_5"),
    ("config", Json.mkObj [("sampleRate", Json.num p.sampleRate)])]
  let fields := if p.compilationMode == .fused then fields
    else fields.push ("compilation_mode", Json.str p.compilationMode.wire)
  let fields := fields
    ++ #[("array_slot_names", toJson p.arraySlotNames),
      ("register_count", toJson p.registerCount),
      ("array_slot_count", toJson p.arraySlotCount),
      ("array_slot_sizes", toJson p.arraySlotSizes),
      ("slot_count", toJson p.slotCount),
      ("slot_names", toJson p.slotNames),
      ("slot_defaults", Json.arr p.slotDefaults),
      ("instance_functions", Json.arr (← p.instanceFunctions.mapM (·.toWire)))]
  let fields := if p.paramDisciplines.isEmpty then fields
    else fields.push ("param_disciplines", Json.arr (p.paramDisciplines.map (·.toWire)))
  let fields := if p.coeffArraySlots.isEmpty then fields
    else fields.push ("coeff_array_slots", toJson p.coeffArraySlots)
  let fields := if p.sinks.isEmpty then fields
    else fields.push ("sinks", Json.arr (p.sinks.map (·.toWire)))
  let fields := if isDefaultSources p.sources then fields
    else fields.push ("sources", Json.arr (p.sources.map fun s =>
      Json.mkObj [("kind", Json.str s.wire)]))
  return Json.mkObj fields.toList

end Tropical.Plan
