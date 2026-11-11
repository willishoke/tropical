import Tropical.Tropicaltest.Int32
import Tropical.Kernel
import Tropical.Ir.EmitMsl

/-!
# Typed ports: params and outputs that carry their own number format

A param declares its kind (`float`, `int32`, `int`) and, for an integer, its
fraction bits (`paramDecl` `type`/`frac_bits`); a program's integer ports
declare theirs (`InputDecl`/`OutputDecl` `fracBits`). The plan lists the typed
port-facing slots (`slot_types`), the kernel loads a typed param as its word
and stores an `int32` output as its word, and the manifest says which. The
gates:

* **jit** — `SinOscQ15` with a live `int32` Q16.16 `freq` param: the plan's
  `slot_types`, the param's word default, the manifest (param `i32`/16, out
  `i32`/15, and a lossless wire round trip), the JIT render against the
  reference semantics from sample 0 and from 2⁴⁰, a `set_param` write against
  a fresh compile at the written value, and an `int` param through
  `FixedSin.phase` against the literal it stands for.
* **refusals** — mismatched kinds and formats through the real ingest path
  (`Session.compilePatch`), malformed declarations, hand-edited plans that
  `PlanDecode` must refuse (a discipline on an integer param, a typed slot
  read at another kind, an `int32` sink with its `slot_types` entry removed,
  mixed output formats), and Metal's refusal of a typed param.
-/

namespace Tropical.Tropicaltest.TypedPorts

open Tropical
open Tropical.Plan

private def patchText (name : String) (decls : List Lean.Json) (dac : Lean.Json) : String :=
  (Lean.Json.mkObj [("schema", "tropical_program_2"), ("name", Lean.Json.str name),
    ("body", Lean.Json.mkObj [("op", "block"), ("decls", Lean.Json.arr decls.toArray),
      ("assigns", Lean.Json.arr #[Lean.Json.mkObj [("op", "outputAssign"),
        ("name", "dac.out"), ("expr", dac)]])])]).compress

private def instDecl (name program : String) (inputs : List (String × Lean.Json)) : Lean.Json :=
  Lean.Json.mkObj [("op", "instanceDecl"), ("name", Lean.Json.str name),
    ("program", Lean.Json.str program), ("inputs", Lean.Json.mkObj inputs)]

private def paramDecl (name : String) (value : Lean.Json)
    (type? : Option String := none) (fracBits? : Option Nat := none) : Lean.Json :=
  Lean.Json.mkObj <| [("op", Lean.Json.str "paramDecl"), ("name", Lean.Json.str name)] ++
    (match type? with | some t => [("type", Lean.Json.str t)] | none => []) ++
    (match fracBits? with | some n => [("frac_bits", Lean.toJson n)] | none => []) ++
    [("value", value)]

private def ref (inst out : String) : Lean.Json :=
  Lean.Json.mkObj [("op", "ref"), ("instance", Lean.Json.str inst), ("output", Lean.Json.str out)]

private def param (name : String) : Lean.Json :=
  Lean.Json.mkObj [("op", "param"), ("name", Lean.Json.str name)]

private def opJ (op : String) (args : List Lean.Json) : Lean.Json :=
  Lean.Json.mkObj [("op", Lean.Json.str op), ("args", Lean.Json.arr args.toArray)]

private def num (x : Float) : Lean.Json :=
  match Lean.JsonNumber.fromFloat? x with
  | .inr n => Lean.Json.num n
  | .inl _ => Lean.Json.null

/-- The knob LFO: `SinOscQ15` with `freq` an `int32` Q16.16 param. -/
def knobPatch (hz : Float) : String :=
  patchText "knob_lfo"
    [paramDecl "freq" (num hz) "int32" (some 16),
     instDecl "lfo" "SinOscQ15" [("freq", param "freq")]]
    (ref "lfo" "out")

private def compile (text : String) : Except String (FlatPlan × Array (Array (Option Ir.Stage))) := do
  let st ← Session.bootRegistry.mapError (·.toJson.compress)
  match (Session.compilePatch text).runOn st with
  | (.ok r, _) => .ok r
  | (.error f, _) => .error f.toJson.compress

private def render (plan : FlatPlan) (blocks : Array (Array (Option Ir.Stage)))
    (start n : Nat) (writes : List (String × Float) := []) : IO (Array Float) := do
  let rt ← Ffi.Runtime.new (UInt32.ofNat n)
  StagedLoad.loadTyped rt plan blocks
  for (name, v) in writes do
    match ← rt.setParam name v with
    | some _ => pure ()
    | none => throw <| IO.userError s!"set_param {name}: unknown"
  rt.setSampleIndex start.toUInt64
  rt.process
  pure (decodeF64LE (← rt.outputBytes))

private def firstMismatch (got want : Array Float) : Option Nat := Id.run do
  if got.size != want.size then return some (min got.size want.size)
  for k in [0:got.size] do
    if got[k]!.toBits != want[k]!.toBits then return some k
  return none

private def slotIndexOf (plan : FlatPlan) (name : String) : Option Nat :=
  plan.slotNames.findIdx? (· == name)

def runTypedPortsJit : IO Bool := do
  let mut bad : Array String := #[]
  let n := 512
  match compile (knobPatch 2.0), compile (knobPatch 1234.5) with
  | .error e, _ | _, .error e => return ← failGate "typed-ports-jit" s!"compile: {e}"
  | .ok (plan, blocks), .ok (plan2, blocks2) =>
    -- the plan: the param slot holds a word, both typed slots are listed
    let some fi := slotIndexOf plan "param:freq"
      | return ← failGate "typed-ports-jit" "no param:freq slot"
    if plan.slotType? fi != some { kind := .int32, fracBits := 16 } then
      bad := bad.push s!"slot_types: param:freq is {repr (plan.slotType? fi)}"
    if plan.slotDefaults[fi]? != some (Lean.toJson (131072 : Nat)) then
      bad := bad.push s!"param:freq default is {(plan.slotDefaults[fi]?.map (·.compress)).getD "?"}, wanted the word 131072"
    if plan.outputFormat != { kind := .int32, fracBits := 15 } then
      bad := bad.push s!"output format is {plan.outputFormat.describe}"
    -- the manifest
    match Kernel.ofPlan plan 16 with
    | .error e => bad := bad.push s!"manifest: {e}"
    | .ok k =>
      let p? := k.inputs.find? (·.name == "param:freq")
      match p? with
      | some p =>
        unless p.elem == .i32 && p.fracBits == some 16 && p.role == some .param
            && (p.default.map (·.toString)) == some "131072" do
          bad := bad.push s!"manifest param: {p.describe} frac={p.fracBits} default={(p.default.map (·.toString)).getD "-"}"
      | none => bad := bad.push "manifest: no param:freq input"
      match k.outputs[0]? with
      | some o =>
        unless o.elem == .i32 && o.fracBits == some 15 do
          bad := bad.push s!"manifest output: {o.describe} frac={o.fracBits}"
      | none => bad := bad.push "manifest: no output"
      match k.toWire with
      | .error e => bad := bad.push s!"manifest wire: {e}"
      | .ok j =>
        match Kernel.KernelType.ofWire j with
        | .error e => bad := bad.push s!"manifest decode: {e}"
        | .ok k' =>
          let sig (k : Kernel.KernelType) := (k.inputs ++ k.outputs).map fun p =>
            (p.name, p.elem.wire, p.fracBits, (p.default.map (·.toString)))
          if sig k != sig k' then bad := bad.push "manifest: the wire round trip changed a port"
          if k'.plan.slotTypes != plan.slotTypes then
            bad := bad.push "manifest: the nested plan lost slot_types"
    -- JIT ≡ the reference semantics, scaled by the output's 2⁻¹⁵
    for start in [0, 1099511627776] do
      let got ← render plan blocks start n
      match Int32.refRender plan start n with
      | .error e => bad := bad.push s!"reference: {e}"
      | .ok words =>
        let want := words.map (· * (2.0 : Float) ^ (-15.0 : Float))
        if let some k := firstMismatch got want then
          bad := bad.push s!"from {start}: frame {k}: JIT {got[k]?} ≠ reference {want[k]?}"
        if !(got.any (· != 0.0)) then bad := bad.push s!"from {start}: silent"
    -- the observation path (render_window / the scope) widens the same words
    let rt ← Ffi.Runtime.new (UInt32.ofNat n)
    StagedLoad.loadTyped rt plan blocks
    let observed := decodeF64LE (← rt.renderTau
      ((Array.range n).map fun k => (UInt64.ofNat (4096 + k)) <<< 16) #[0])
    let processed ← render plan blocks 4096 n
    if let some k := firstMismatch observed processed then
      bad := bad.push s!"renderTau: frame {k}: {observed[k]?} ≠ process {processed[k]?}"
    -- set_param on the word: a live write equals a fresh compile at that value
    let written ← render plan blocks 4096 n [("freq", 1234.5)]
    let fresh ← render plan2 blocks2 4096 n
    if let some k := firstMismatch written fresh then
      bad := bad.push s!"set_param 1234.5: frame {k}: {written[k]?} ≠ fresh compile {fresh[k]?}"
    -- an `int` param (frac 0) through `FixedSin.phase`, against its literal
    let intPatch (wire : Lean.Json) (decls : List Lean.Json) :=
      patchText "int_param" (decls ++ [instDecl "s" "FixedSin" [("phase", wire)]]) (ref "s" "out")
    match compile (intPatch (param "ph") [paramDecl "ph" (Lean.toJson (1073741824 : Nat)) "int"]),
          compile (intPatch (Lean.toJson (1073741824 : Nat)) []),
          compile (intPatch (Lean.toJson (3221225472 : Nat)) []) with
    | .ok (pi, bi), .ok (pl, bl), .ok (pl3, bl3) =>
      match (slotIndexOf pi "param:ph").bind pi.slotType? with
      | some f => if f.kind != .int then bad := bad.push s!"int param listed as {f.describe}"
      | none => bad := bad.push "int param not in slot_types"
      let a ← render pi bi 0 64
      let b ← render pl bl 0 64
      if let some k := firstMismatch a b then
        bad := bad.push s!"int param: frame {k}: {a[k]?} ≠ literal {b[k]?}"
      let c ← render pi bi 0 64 [("ph", 3221225472.0)]
      let d ← render pl3 bl3 0 64
      if let some k := firstMismatch c d then
        bad := bad.push s!"int param set_param: frame {k}: {c[k]?} ≠ literal {d[k]?}"
    | .error e, _, _ | _, .error e, _ | _, _, .error e => bad := bad.push s!"int param: {e}"
    if bad.isEmpty then
      passGate "typed-ports-jit" s!"SinOscQ15 with an int32 Q16.16 param: slot_types, word default, manifest (i32/16 in, i32/15 out, wire round trip), JIT ≡ reference from 0 and 2⁴⁰ ({n} frames each), renderTau ≡ process, set_param ≡ fresh compile; an int param ≡ its literal, written and seeded"
    else failGate "typed-ports-jit" (String.intercalate "; " bad.toList)

/-- Ingest-path cases: each `(label, patch, needle?)` refuses with `needle`,
    or compiles when `none`. -/
private def ingestCases : List (String × String × Option String) :=
  let lfo (freq : Lean.Json) := instDecl "lfo" "SinOscQ15" [("freq", freq)]
  let out := ref "lfo" "out"
  [ ("int32 Q8.24 param → Q16.16 input (bare)",
      patchText "f1" [paramDecl "f" (num 2.0) "int32" (some 24), lfo (param "f")] out,
      some "Format mismatch: input 'lfo.freq' is int32 Q16.16 but param 'f' is int32 Q8.24"),
    ("int32 Q8.24 param, shifted by 8, → Q16.16 input",
      patchText "ok1" [paramDecl "f" (num 2.0) "int32" (some 24),
        lfo (opJ "rshift" [param "f", Lean.toJson (8 : Nat)])] out, none),
    ("float param → int32 input",
      patchText "f2" [paramDecl "f" (num 2.0), lfo (param "f")] out,
      some "input 'freq' of instance 'lfo' is int32 but its value is float"),
    ("int32 param → float input",
      patchText "f3" [paramDecl "f" (num 2.0) "int32" (some 16),
        instDecl "osc" "FixedSinOsc" [("freq", param "f")]] (ref "osc" "sine"),
      some "input 'freq' of instance 'osc' is float but its value is int32"),
    ("int param → int32 input",
      patchText "f4" [paramDecl "f" (num 2.0) "int" (some 16), lfo (param "f")] out,
      some "input 'freq' of instance 'lfo' is int32 but its value is int"),
    ("Q17.15 output → Q0.32 phase input (bare)",
      patchText "f5" [lfo (Lean.toJson (65536 : Nat)),
        instDecl "b" "SinOscQ15" [("phase", ref "lfo" "out")]] (ref "b" "out"),
      some "Format mismatch: input 'b.phase' is int32 Q0.32 but output 'lfo.out' is int32 Q17.15"),
    ("frac_bits on a float param",
      patchText "f6" [paramDecl "f" (num 2.0) "float" (some 4), lfo (Lean.toJson (65536 : Nat))] out,
      some "fraction bits need an integer kind"),
    ("33 fraction bits on int32",
      patchText "f7" [paramDecl "f" (num 0.25) "int32" (some 33), lfo (Lean.toJson (65536 : Nat))] out,
      some "int32 has 32 bits; 33 fraction bits do not fit"),
    ("a default outside the format's range",
      patchText "f8" [paramDecl "f" (num 40000.0) "int32" (some 16), lfo (param "f")] out,
      some "value 40000 is outside the range of int32 Q16.16"),
    ("bool param",
      patchText "f9" [paramDecl "f" (num 1.0) "bool", lfo (Lean.toJson (65536 : Nat))] out,
      some "type 'bool' is not a param type"),
    ("int32 output and float output on one output buffer",
      (Lean.Json.mkObj [("schema", "tropical_program_2"), ("name", "f10"),
        ("body", Lean.Json.mkObj [("op", "block"), ("decls", Lean.Json.arr #[
            lfo (Lean.toJson (65536 : Nat)), instDecl "osc" "FixedSinOsc" []]),
          ("assigns", Lean.Json.arr #[
            Lean.Json.mkObj [("op", "outputAssign"), ("name", "dac.out"), ("expr", out)],
            Lean.Json.mkObj [("op", "outputAssign"), ("name", "dac.out"),
              ("expr", ref "osc" "sine"), ("channel", Lean.toJson (1 : Nat))]])])]).compress,
      some "one output buffer has one element type")]

def runTypedPortsRefusals : IO Bool := do
  let mut bad : Array String := #[]
  for (label, text, needle?) in ingestCases do
    match compile text, needle? with
    | .ok _, none => pure ()
    | .ok _, some needle => bad := bad.push s!"'{label}': compiled, expected '{needle}'"
    | .error e, none => bad := bad.push s!"'{label}': refused: {e}"
    | .error e, some needle =>
      unless (e.splitOn needle).length > 1 do
        bad := bad.push s!"'{label}': refused with {e}, expected '{needle}'"
  -- hand-edited plans PlanDecode must refuse
  match compile (knobPatch 2.0), compile (patchText "old" [instDecl "lfo" "FixedSinOscQ15" [("freq", Lean.toJson (2 : Nat))]] (ref "lfo" "out")) with
  | .error e, _ | _, .error e => bad := bad.push s!"compile: {e}"
  | .ok (plan, _), .ok (old, _) =>
    let fi := (slotIndexOf plan "param:freq").getD 0
    let reads (p : FlatPlan) (f : NOperand → NOperand) : FlatPlan :=
      let rw (i : NInstr) : NInstr := { i with args := i.args.map f }
      let rec go (inst : InstanceFunction) : InstanceFunction :=
        match inst with
        | .mk name iname pre instrs preIn regOff arrOff regCount children =>
          .mk name iname (pre.map rw) (instrs.map rw) (preIn.map rw) regOff arrOff regCount
            (children.attach.map fun ⟨c, _⟩ => go c)
      { p with instanceFunctions := p.instanceFunctions.map go }
    let edits : List (String × String × FlatPlan) := [
      ("anchor discipline on an int32 param", "is not defined on the int32 Q16.16 slot",
        { plan with paramDisciplines :=
            #[({ name := "freq", discipline := "anchor", companions := #["freq#phase"] } :
              ParamDiscipline)] }),
      ("int32 param read as bool", "is int32 but is read as bool",
        reads plan (fun o => match o with
          | .slot i .int32 => if i == fi then .slot i .bool else .slot i .int32
          | o => o)),
      ("int32 sink source missing from slot_types", "is int32 but slot_types does not list it",
        { old with slotTypes := #[] }),
      ("param listed as float", "a typed param is int32 or int",
        { plan with slotTypes := plan.slotTypes.map fun t =>
            if t.slot == fi then { t with format := {} } else t }),
      ("output listed as Q0.40", "fraction bits do not fit",
        { old with slotTypes := old.slotTypes.map fun t =>
            { t with format := { t.format with fracBits := 40 } } })]
    for (label, needle, edited) in edits do
      match edited.toWire with
      | .error e => bad := bad.push s!"'{label}': encode: {e}"
      | .ok j =>
        match FlatPlan.ofWire j with
        | .ok _ => bad := bad.push s!"'{label}': PlanDecode accepted it"
        | .error e =>
          unless (e.splitOn needle).length > 1 do
            bad := bad.push s!"'{label}': refused with '{e}', expected '{needle}'"
    -- the unedited plans decode
    for (label, p) in [("knob", plan), ("old", old)] do
      match p.toWire >>= FlatPlan.ofWire with
      | .error e => bad := bad.push s!"{label}: decode refused: {e}"
      | .ok p' => if p'.slotTypes != p.slotTypes then bad := bad.push s!"{label}: slot_types lost"
    -- Metal refuses a typed param, and keeps an int32 output's word exact
    match Ir.EmitMsl.emitKernel plan 16 with
    | .ok _ => bad := bad.push "EmitMsl accepted a typed param"
    | .error e =>
      unless (e.splitOn "Metal binds params as f32 slots").length > 1 do
        bad := bad.push s!"EmitMsl refused with '{e}'"
    if let .error e := Ir.EmitMsl.emitKernel old 16 then
      bad := bad.push s!"EmitMsl refused the fixed-frequency LFO: {e}"
  if bad.isEmpty then
    passGate "typed-ports-refusals" s!"{ingestCases.length} ingest cases (format and kind mismatches refused, a shifted wire compiles, bad declarations refused), 5 hand-edited plans refused by PlanDecode, Metal refuses a typed param"
  else failGate "typed-ports-refusals" (String.intercalate "; " bad.toList)

end Tropical.Tropicaltest.TypedPorts
