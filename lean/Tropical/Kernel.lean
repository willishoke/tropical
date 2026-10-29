import Tropical.Plan
import Tropical.PlanDecode

/-!
# Kernel — the typed kernel at the port

`design/typed-kernel.md` (2026-10-28): a kernel is one pure function from a
record of named, static-shaped, typed inputs to a record of outputs. This
module is the type and its wire form, `tropical_kernel_1`. The type is
*derived* from a `FlatPlan`: `EmitLlvm`/`EmitMsl` emit the port it names,
`Tropical.Artifact` ships it as the artifact's manifest, and the loaders
(`FlatRuntime`, the web runtime) read it.

What the derivation says:

* `tau` is the first input: the clock rail (`i64` in Q32.32, the
  `fixed-carrier.md` ledger), shape `[frames]`, role `signal`. Shapes are
  static, so `frames` is a parameter of the type — a different buffer
  length is a different kernel.
* Every `param:<name>` slot is a scalar `f64` input with role `param`, its
  seed from `slot_defaults` and its write discipline from
  `param_disciplines` (`host-param-dispatch.md` stays normative for the
  arithmetic). Wire slots and `coef:<n>` slots are internal and absent.
* `out` is the one output: `f64`, shape `[frames, channels]`, frame-major
  (`stereo-output-contract.md`).
* The rate is a constant of the kernel, not an input.

Element types and roles carry their wire spellings so a host reads one
table. Nothing here is audio-specific beyond the derivation's choice of
names: a video kernel's output is `[H, W, 3]` by the same rule.
-/

namespace Tropical.Kernel

open Lean (Json JsonNumber toJson)
open Tropical.Plan (FlatPlan ParamDiscipline CompilationMode)

/-- The wire schema this module reads and writes. -/
def schema : String := "tropical_kernel_1"

/-- Element types. `f64` values and `i64` rails are the JIT/wasm
    realization; Metal refines values to `f32` under its tolerance gate.
    `i64q32` is the signed Q32.32 clock rail. -/
inductive Elem where
  | f64 | i64 | i64q32
deriving BEq, Repr, Inhabited

def Elem.wire : Elem → String
  | .f64 => "f64" | .i64 => "i64" | .i64q32 => "i64q32"

def Elem.ofWire? : String → Option Elem
  | "f64" => some .f64 | "i64" => some .i64 | "i64q32" => some .i64q32
  | _ => none

theorem Elem.ofWire_wire (e : Elem) : Elem.ofWire? e.wire = some e := by
  cases e <;> rfl

/-- What a host does with an input between calls. Never semantic: `param`
    is written under a discipline, `signal` is refreshed every call, `table`
    is seeded once. -/
inductive Role where
  | param | signal | table
deriving BEq, Repr, Inhabited

def Role.wire : Role → String
  | .param => "param" | .signal => "signal" | .table => "table"

def Role.ofWire? : String → Option Role
  | "param" => some .param | "signal" => some .signal | "table" => some .table
  | _ => none

theorem Role.ofWire_wire (r : Role) : Role.ofWire? r.wire = some r := by
  cases r <;> rfl

/-- The host-facing annotation of a `param` input: how to write it. -/
structure Discipline where
  discipline : String
  glideDurSec : Option JsonNumber := none
  companions : Array String := #[]
deriving Repr, Inhabited

/-- One named, shaped, typed value at the port. `default` is the seed a host
    writes before the first call; a `JsonNumber` keeps the plan's lexical
    number form, as `slot_defaults` does. -/
structure Port where
  name : String
  elem : Elem
  shape : Array Nat
  role : Option Role := none
  default : Option JsonNumber := none
  discipline : Option Discipline := none
  /-- For an output of shape `[frames, C]`: the name of each channel's
      source (the slot its sink mixes first), `""` for a silent channel.
      The scope resolves a tap name to a channel through this. -/
  channels : Array String := #[]
deriving Inhabited

/-- The realization a loader compiles; not part of the type's meaning. -/
structure Realization where
  mode : CompilationMode := .fused
  registerCount : Nat := 0
deriving Repr, Inhabited

/-- The kernel type plus the plan it was derived from. `plan` is nested in
    the wire form (as `tropical_plan_6`) for the gates; a host that reads it
    is coupling to the compiler's internals. -/
structure KernelType where
  rate : JsonNumber
  inputs : Array Port
  outputs : Array Port
  realization : Realization := {}
  plan : FlatPlan
deriving Inhabited

/-- A one-line description of a port for gate messages. -/
def Port.describe (p : Port) : String :=
  let role := match p.role with | some r => r.wire | none => "-"
  s!"{p.name}:{p.elem.wire}{p.shape} role={role}"

/-- The conventional names of the two ports every audio kernel has. -/
def tauName : String := "tau"
def outName : String := "out"

/-- The slot-name prefix that marks a host-written control. -/
def paramPrefix : String := "param:"

/-- The slot indices of a plan's param inputs, in slot order — the order the
    port's input pointers follow after `tau`. Shared by the type derivation
    and the emitters so both agree by construction. -/
def paramSlotIndices (plan : FlatPlan) : Array Nat := Id.run do
  let mut out := #[]
  for i in [0:plan.slotNames.size] do
    if plan.slotNames[i]!.startsWith paramPrefix then out := out.push i
  return out

/-- The source name of each output channel: the first slot its sink mixes. -/
def channelNames (plan : FlatPlan) : Array String := Id.run do
  let mut names := Array.replicate plan.outputChannelCount ""
  for sink in plan.sinks do
    if sink.target < names.size then
      if let some slot := sink.inputs[0]? then
        names := names.set! sink.target (plan.slotNames[slot]?.getD "")
  return names

private def disciplineOf (plan : FlatPlan) (base : String) : Option Discipline :=
  (plan.paramDisciplines.find? (·.name == base)).map fun d =>
    { discipline := d.discipline, glideDurSec := d.glideDurSec,
      companions := d.companions }

/-- Derive the kernel type of a plan compiled for `frames` samples per call. -/
def ofPlan (plan : FlatPlan) (frames : Nat) : Except String KernelType := do
  let tau : Port :=
    { name := tauName, elem := .i64q32, shape := #[frames], role := some .signal }
  let mut inputs : Array Port := #[tau]
  for i in paramSlotIndices plan do
    let slotName := plan.slotNames[i]!
    let base := (slotName.drop paramPrefix.length).toString
    let default ← match plan.slotDefaults[i]? with
      | none => pure none
      | some (.num n) => pure (some n)
      | some other =>
        throw s!"Kernel: slot '{slotName}' default is not a number: {other.compress}"
    inputs := inputs.push {
      name := slotName, elem := .f64, shape := #[], role := some .param,
      default, discipline := disciplineOf plan base }
  let out : Port :=
    { name := outName, elem := .f64, shape := #[frames, plan.outputChannelCount],
      channels := channelNames plan }
  return {
    rate := plan.sampleRate
    inputs
    outputs := #[out]
    realization := { mode := plan.compilationMode, registerCount := plan.registerCount }
    plan }

-- ── Wire form ────────────────────────────────────────────────────────────

def Discipline.toWire (d : Discipline) : Json :=
  let fields := #[("discipline", Json.str d.discipline)]
  let fields := match d.glideDurSec with
    | some n => fields.push ("glide_dur_sec", Json.num n)
    | none => fields
  let fields := if d.companions.isEmpty then fields
    else fields.push ("companions", toJson d.companions)
  Json.mkObj fields.toList

def Port.toWire (p : Port) : Json :=
  let fields := #[("name", Json.str p.name), ("elem", Json.str p.elem.wire),
    ("shape", toJson p.shape)]
  let fields := match p.role with
    | some r => fields.push ("role", Json.str r.wire)
    | none => fields
  let fields := match p.default with
    | some d => fields.push ("default", Json.num d)
    | none => fields
  let fields := match p.discipline with
    | some d => fields.push ("discipline", d.toWire)
    | none => fields
  let fields := if p.channels.isEmpty then fields
    else fields.push ("channels", toJson p.channels)
  Json.mkObj fields.toList

def Realization.toWire (r : Realization) : Json :=
  Json.mkObj [("mode", Json.str r.mode.wire),
    ("register_count", toJson r.registerCount)]

def KernelType.toWire (k : KernelType) : Except String Json := do
  let plan ← k.plan.toWire
  return Json.mkObj [
    ("schema", Json.str schema),
    ("rate", Json.num k.rate),
    ("inputs", Json.arr (k.inputs.map Port.toWire)),
    ("outputs", Json.arr (k.outputs.map Port.toWire)),
    ("realization", k.realization.toWire),
    ("plan", plan)]

private def reqStr (ctx : String) (j : Json) (k : String) : Except String String :=
  match j.getObjVal? k with
  | .ok (.str s) => .ok s
  | _ => .error s!"{ctx}: missing string field '{k}'"

private def reqNum (ctx : String) (j : Json) (k : String) : Except String JsonNumber :=
  match j.getObjVal? k with
  | .ok (.num n) => .ok n
  | _ => .error s!"{ctx}: missing number field '{k}'"

private def optNat (j : Json) (k : String) (dflt : Nat) : Nat :=
  match j.getObjVal? k with
  | .ok (.num n) => if n.exponent == 0 && n.mantissa ≥ 0 then n.mantissa.toNat else dflt
  | _ => dflt

private def natArr (ctx : String) (j : Json) (k : String) : Except String (Array Nat) :=
  match j.getObjVal? k with
  | .ok (.arr xs) => xs.mapM fun x => match x with
      | .num n => if n.exponent == 0 && n.mantissa ≥ 0 then .ok n.mantissa.toNat
          else .error s!"{ctx}: '{k}' entry is not a natural number"
      | _ => .error s!"{ctx}: '{k}' entry is not a number"
  | _ => .error s!"{ctx}: missing array field '{k}'"

private def strArr (j : Json) (k : String) : Array String :=
  match j.getObjVal? k with
  | .ok (.arr xs) => xs.filterMap fun x => match x with | .str s => some s | _ => none
  | _ => #[]

def Discipline.ofWire (j : Json) : Except String Discipline := do
  let discipline ← reqStr "Kernel.Discipline" j "discipline"
  let glideDurSec := match j.getObjVal? "glide_dur_sec" with
    | .ok (.num n) => some n
    | _ => none
  return { discipline, glideDurSec, companions := strArr j "companions" }

def Port.ofWire (j : Json) : Except String Port := do
  let name ← reqStr "Kernel.Port" j "name"
  let elemStr ← reqStr "Kernel.Port" j "elem"
  let some elem := Elem.ofWire? elemStr
    | throw s!"Kernel.Port '{name}': unknown elem '{elemStr}'"
  let shape ← natArr s!"Kernel.Port '{name}'" j "shape"
  let role ← match j.getObjVal? "role" with
    | .ok (.str r) =>
      match Role.ofWire? r with
      | some role => pure (some role)
      | none => throw s!"Kernel.Port '{name}': unknown role '{r}'"
    | _ => pure none
  let default ← match j.getObjVal? "default" with
    | .ok (.num d) => pure (some d)
    | .ok other => throw s!"Kernel.Port '{name}': default is not a number: {other.compress}"
    | _ => pure none
  let discipline ← match j.getObjVal? "discipline" with
    | .ok d => some <$> Discipline.ofWire d
    | _ => pure none
  return { name, elem, shape, role, default, discipline, channels := strArr j "channels" }

def Realization.ofWire (j : Json) : Except String Realization := do
  let modeStr ← reqStr "Kernel.Realization" j "mode"
  let some mode := CompilationMode.ofWire? modeStr
    | throw s!"Kernel.Realization: unknown mode '{modeStr}'"
  return { mode, registerCount := optNat j "register_count" 0 }

def KernelType.ofWire (j : Json) : Except String KernelType := do
  let found ← reqStr "Kernel" j "schema"
  if found != schema then
    throw s!"Kernel: schema '{found}' is not '{schema}'"
  let rate ← reqNum "Kernel" j "rate"
  let ports (k : String) : Except String (Array Port) :=
    match j.getObjVal? k with
    | .ok (.arr xs) => xs.mapM Port.ofWire
    | _ => .error s!"Kernel: missing array field '{k}'"
  let inputs ← ports "inputs"
  let outputs ← ports "outputs"
  let realization ← match j.getObjVal? "realization" with
    | .ok r => Realization.ofWire r
    | _ => pure {}
  let plan ← match j.getObjVal? "plan" with
    | .ok p => FlatPlan.ofWire p
    | _ => throw "Kernel: missing field 'plan'"
  return { rate, inputs, outputs, realization, plan }

/-- The `tropical_kernel_1` manifest text for a plan. -/
def manifest (plan : FlatPlan) (frames : Nat) : Except String String := do
  let k ← ofPlan plan frames
  return (← k.toWire).compress

end Tropical.Kernel
