import Tropical.Plan
import Tropical.PlanDecode
import Tropical.Ir.ConstFold
import Tropical.Ir.EmitCommon

/-!
# Kernel — the typed kernel at the port

`design/typed-kernel.md` (2026-10-28): a kernel is one pure function from a
record of named, static-shaped, typed inputs to a record of outputs. This
module is the type and its wire form, `tropical_kernel_1`. The type is
*derived* from a `FlatPlan`: `EmitLlvm`/`EmitMsl` emit the port it names,
`Tropical.Artifact` ships it as the artifact's manifest, and the loaders
(`FlatRuntime`, the web runtime) read it.

What the derivation says:

* `tau` is the first input: the clock rail (`i64` in Q48.16, the
  `fixed-carrier.md` ledger and `Ir/ClockFormat.lean`), shape `[frames]`, role `signal`. Shapes are
  static, so `frames` is a parameter of the type — a different buffer
  length is a different kernel.
* Every `param:<name>` slot is a scalar input with role `param`, its seed
  from `slot_defaults` and its write discipline from `param_disciplines`
  (`host-param-dispatch.md` stays normative for the arithmetic). It is `f64`
  unless the plan's `slot_types` declares it an integer word: then `i32`
  (`int32`) or `i64` (`int`) with `frac_bits`, the word's fixed-point format
  (`Tropical.PortFormat`). Wire slots and `coef:<n>` slots are internal and
  absent.
* `out` is the one output, shape `[frames, channels]`, frame-major
  (`stereo-output-contract.md`): `f64`, or `i32` with `frac_bits` when the
  channels carry an `int32` program output (`slot_types` lists their
  sources; `FlatPlan.portError?` keeps one element type per buffer).
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

/-- Element types. `f64` and `f32` are the `float` kind's two realizations
    (`FloatFormat`): every target realizes `float` as `f64` unless the
    artifact is compiled at `f32`, and Metal refines values to f32 under its
    tolerance gate below a port that stays `f64`. `i64q16` is the signed
    Q48.16 clock rail: `i64` with `clockFracBits` = 16 fraction bits
    (`Ir/ClockFormat.lean`; the `long-horizon` gate pins the two together).
    `i32` and `i64` are integer words: an `int32` param or output, an `int`
    param. Their fixed-point format is the port's `frac_bits`, declared by
    the program. -/
inductive Elem where
  | f64 | i64 | i64q16 | i32 | f32
deriving BEq, Repr, Inhabited

def Elem.wire : Elem → String
  | .f64 => "f64" | .i64 => "i64" | .i64q16 => "i64q16" | .i32 => "i32"
  | .f32 => "f32"

def Elem.ofWire? : String → Option Elem
  | "f64" => some .f64 | "i64" => some .i64 | "i64q16" => some .i64q16
  | "i32" => some .i32 | "f32" => some .f32
  | _ => none

/-- Bytes per element (= its alignment in every buffer layout). -/
def Elem.bytes : Elem → Nat
  | .f64 | .i64 | .i64q16 => 8
  | .i32 | .f32 => 4

/-- The IEEE 754 format a target realizes the `float` kind in: binary64
    (`f64`, the default) or binary32 (`f32`). It is a realization of `float`,
    chosen when an artifact is compiled, not a kind: no IR pass, op or
    semantics reads it, and the plan is the same either way. `EmitLlvm` emits
    the value lane at this width, and the kernel type declares every float
    port at it. The integer kinds and τ's rail do not change. -/
inductive FloatFormat where
  | f64 | f32
deriving BEq, Repr, Inhabited

def FloatFormat.wire : FloatFormat → String
  | .f64 => "f64" | .f32 => "f32"

def FloatFormat.ofWire? : String → Option FloatFormat
  | "f64" => some .f64 | "f32" => some .f32
  | _ => none

/-- The port element of a float value at this format. -/
def FloatFormat.elem : FloatFormat → Elem
  | .f64 => .f64 | .f32 => .f32

/-- IEEE 754's name for the format, as the object manifest states it. -/
def FloatFormat.ieeeName : FloatFormat → String
  | .f64 => "ieee754-binary64" | .f32 => "ieee754-binary32"

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
  /-- For an integer port: its fraction bits, so word `w` means
      `w · 2^-fracBits` (`Tropical.PortFormat`). `none` on float ports. τ
      carries 16, the rail's (`i64q16`). -/
  fracBits : Option Nat := none
deriving Inhabited

/-- The realization a loader compiles; not part of the type's meaning.
    `float` is the format the artifact's value lane was emitted at; the
    float ports already say it (`f64`/`f32`), and it is written only when it
    is not the default, so an `f64` manifest is what it always was. -/
structure Realization where
  mode : CompilationMode := .fused
  registerCount : Nat := 0
  float : FloatFormat := .f64
deriving Repr, Inhabited

/-- The kernel type plus the plan it was derived from. `plan` is nested in
    the wire form (as `tropical_plan_7`) for the gates; a host that reads it
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

/-- The element type of a port of format `f`, with `float` realized at
    `float`. -/
def Elem.ofFormat (f : Tropical.PortFormat) (float : FloatFormat := .f64) : Elem :=
  match f.kind with
  | .int32 => .i32
  | .int => .i64
  | _ => float.elem

/-- The word a host writes for value `x` at an integer port of format `f`:
    `x · 2^fracBits`, truncated toward zero and saturated to the word, NaN →
    0. This is the kernel's own float → integer conversion
    (`ConstFold.truncSatToInt32`/`truncSatToInt`, `llvm.fptosi.sat`), so a
    host and the compiler encode a knob value the same way; the scaling by a
    power of two is exact in binary64. -/
def encodeWord (f : Tropical.PortFormat) (x : Float) : Int :=
  let scaled := x * (2.0 : Float) ^ f.fracBits.toFloat
  match f.kind with
  | .int32 => Tropical.Ir.truncSatToInt32 scaled
  | _ => Tropical.Ir.truncSatToInt scaled

/-- The output's format (`FlatPlan.outputFormat`). -/
def outputFormat (plan : FlatPlan) : Tropical.PortFormat := plan.outputFormat

/-- The exact decimal value of a binary32 bit pattern (finite), as a JSON
    number: `sig · 2^e` is `sig · 5^-e / 10^-e`, so every binary32 has a
    finite decimal expansion, and a reader that parses it to binary64 gets
    the binary32 value exactly (binary64 holds every binary32). -/
def binary32Decimal (bits : UInt32) : JsonNumber := Id.run do
  let neg := bits >>> 31 == 1
  let e := ((bits >>> 23) &&& 0xFF).toNat
  let m := (bits &&& 0x7FFFFF).toNat
  let (sig, exp2) : Nat × Int := if e == 0 then (m, -149) else (m + 0x800000, (e : Int) - 150)
  let (mant, dexp) : Nat × Nat :=
    if exp2 ≥ 0 then (sig * 2 ^ exp2.toNat, 0)
    else (sig * 5 ^ (-exp2).toNat, (-exp2).toNat)
  let mut mant := mant
  let mut dexp := dexp
  while dexp > 0 && mant % 10 == 0 do
    mant := mant / 10
    dexp := dexp - 1
  return ⟨if neg then -(mant : Int) else mant, dexp⟩

/-- A float param's seed in a binary32 artifact: the binary32 nearest the
    plan's decimal default (`binary32BitsOfDecimal?`, the same rounding the
    kernel's literals get), written exactly, so every host seeds the same
    bits with no second rounding. -/
private def binary32Default (slotName : String) (n : JsonNumber) : Except String JsonNumber :=
  match Tropical.Ir.EmitCommon.binary32BitsOfDecimal? n with
  | some bits => .ok (binary32Decimal bits)
  | none => .error s!"Kernel: param '{slotName}' default {n} is outside the binary32 range"

/-- Why a plan cannot be realized at binary32 for its write disciplines, or
    `none`. A binary32 artifact's params are written raw: glide, anchor and
    velocity re-anchor on companions that hold sample counts and phases
    (`#t0`, `tau_base`, `#phase`), which binary32 does not hold exactly past
    2^24. The plan's discipline table is walked directly, as
    `FlatPlan.portError?` walks it: a glided knob has no port of its own (its
    ports are the `#v0`/`#v1`/`#t0` companions), so a walk over the param
    ports alone would miss it. -/
def binary32DisciplineError? (plan : FlatPlan) : Option String := Id.run do
  for d in plan.paramDisciplines do
    if d.discipline != "raw" then
      let slots := String.intercalate ", " ((#[d.name] ++ d.companions).toList.map (s!"param:{·}"))
      return some s!"param '{d.name}' has the '{d.discipline}' discipline (slots {slots}); a binary32 artifact's params are written raw (its companions are sample counts and phases, which binary32 does not hold exactly)"
  return none

/-- Derive the kernel type of a plan compiled for `frames` samples per call,
    its `float` values realized at `float`. -/
def ofPlan (plan : FlatPlan) (frames : Nat) (float : FloatFormat := .f64) :
    Except String KernelType := do
  if let some e := plan.portError? then throw s!"Kernel: {e}"
  if float == .f32 then
    if let some e := binary32DisciplineError? plan then throw s!"Kernel: {e}"
  let tau : Port :=
    { name := tauName, elem := .i64q16, shape := #[frames], role := some .signal,
      fracBits := some 16 }
  let mut inputs : Array Port := #[tau]
  for i in paramSlotIndices plan do
    let slotName := plan.slotNames[i]!
    let base := (slotName.drop paramPrefix.length).toString
    let default ← match plan.slotDefaults[i]? with
      | none => pure none
      | some (.num n) => pure (some n)
      | some other =>
        throw s!"Kernel: slot '{slotName}' default is not a number: {other.compress}"
    let fmt := (plan.slotType? i).getD {}
    let discipline := disciplineOf plan base
    let default ← match float, fmt.kind.isInteger, default with
      | .f32, false, some n => some <$> binary32Default slotName n
      | _, _, d => pure d
    inputs := inputs.push {
      name := slotName, elem := Elem.ofFormat fmt float, shape := #[], role := some .param,
      default, discipline,
      fracBits := if fmt.kind.isInteger then some fmt.fracBits else none }
  let outFmt := outputFormat plan
  let out : Port :=
    { name := outName, elem := Elem.ofFormat outFmt float,
      shape := #[frames, plan.outputChannelCount],
      channels := channelNames plan,
      fracBits := if outFmt.kind.isInteger then some outFmt.fracBits else none }
  return {
    rate := plan.sampleRate
    inputs
    outputs := #[out]
    realization := { mode := plan.compilationMode, registerCount := plan.registerCount, float }
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
  let fields := match p.fracBits with
    | some n => fields.push ("frac_bits", toJson n)
    | none => fields
  Json.mkObj fields.toList

def Realization.toWire (r : Realization) : Json :=
  let fields := [("mode", Json.str r.mode.wire), ("register_count", toJson r.registerCount)]
  Json.mkObj (if r.float == .f64 then fields else fields ++ [("float", Json.str r.float.wire)])

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
  let fracBits := match j.getObjVal? "frac_bits" with
    | .ok (.num n) => if n.exponent == 0 && n.mantissa ≥ 0 then some n.mantissa.toNat else none
    | _ => none
  return { name, elem, shape, role, default, discipline, channels := strArr j "channels",
           fracBits }

def Realization.ofWire (j : Json) : Except String Realization := do
  let modeStr ← reqStr "Kernel.Realization" j "mode"
  let some mode := CompilationMode.ofWire? modeStr
    | throw s!"Kernel.Realization: unknown mode '{modeStr}'"
  let float ← match j.getObjVal? "float" with
    | .ok (.str f) =>
      match FloatFormat.ofWire? f with
      | some fmt => pure fmt
      | none => throw s!"Kernel.Realization: unknown float format '{f}'"
    | .ok _ => throw "Kernel.Realization: 'float' must be a string"
    | .error _ => pure .f64
  return { mode, registerCount := optNat j "register_count" 0, float }

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
def manifest (plan : FlatPlan) (frames : Nat) (float : FloatFormat := .f64) :
    Except String String := do
  let k ← ofPlan plan frames float
  return (← k.toWire).compress

end Tropical.Kernel
