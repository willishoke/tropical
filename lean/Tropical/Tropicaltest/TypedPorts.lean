import Tropical.Tropicaltest.Int32
import Tropical.Kernel
import Tropical.Ir.EmitMsl
import Tropical.NativeBoot
import Tropical.Testing.EngineMirror

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
  mixed output formats, an out-of-range or fractional param default, an
  `int` channel source), and Metal's refusal of a typed param. Bare literals
  into fixed-point inputs are real values: in range and exact, or refused.
* **engine** — through the session engine: `export_program` keeps the port
  formats (an exported `SinOscQ15` plays at its Q1.15 peak, takes a bare
  literal and a Q16.16 param, and its exposed default is the literal's
  word); `save` → `load` round-trips typed params after `set_param`; and an
  out-of-range `set_param` is refused with the range, leaving the mirror and
  `save` loadable.
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
    -- a bare literal into a fixed-point input is its real value: `440` into
    -- the Q16.16 `freq` is the param at 440; `0.25` into the Q0.32 `phase` is
    -- the word 2³⁰. Inside arithmetic a literal stays a word: `65536 + 0`
    -- is 1 Hz.
    let lfoWith (inputs : List (String × Lean.Json)) (decls : List Lean.Json := []) :=
      patchText "lit_lfo" (decls ++ [instDecl "lfo" "SinOscQ15" inputs]) (ref "lfo" "out")
    let litCases : List (String × String × String) := [
      ("literal 440 Hz ≡ param 440", lfoWith [("freq", Lean.toJson (440 : Nat))], knobPatch 440.0),
      ("literal 0.5 Hz ≡ param 0.5", lfoWith [("freq", num 0.5)], knobPatch 0.5),
      ("add(65536, 0) is the word: 1 Hz",
        lfoWith [("freq", opJ "add" [Lean.toJson (65536 : Nat), Lean.toJson (0 : Nat)])], knobPatch 1.0),
      ("phase literal 0.25 ≡ the word 2³⁰ through arithmetic",
        lfoWith [("freq", Lean.toJson (1000 : Nat)), ("phase", num 0.25)],
        lfoWith [("freq", Lean.toJson (1000 : Nat)),
          ("phase", opJ "add" [Lean.toJson (1073741824 : Nat), Lean.toJson (0 : Nat)])])]
    for (label, a, b) in litCases do
      match compile a, compile b with
      | .ok (pa, ba), .ok (pb, bb) =>
        let x ← render pa ba 0 n
        let y ← render pb bb 0 n
        if let some k := firstMismatch x y then
          bad := bad.push s!"{label}: frame {k}: {x[k]?} ≠ {y[k]?}"
        if !(x.any (· != 0.0)) then bad := bad.push s!"{label}: silent"
      | .error e, _ | _, .error e => bad := bad.push s!"{label}: {e}"
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
      passGate "typed-ports-jit" s!"SinOscQ15 with an int32 Q16.16 param: slot_types, word default, manifest (i32/16 in, i32/15 out, wire round trip), JIT ≡ reference from 0 and 2⁴⁰ ({n} frames each), renderTau ≡ process, set_param ≡ fresh compile; bare literals at Q16.16/Q0.32 ≡ their real values, a literal in arithmetic a word; an int param ≡ its literal, written and seeded"
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
      patchText "f5" [lfo (Lean.toJson (1 : Nat)),
        instDecl "b" "SinOscQ15" [("phase", ref "lfo" "out")]] (ref "b" "out"),
      some "Format mismatch: input 'b.phase' is int32 Q0.32 but output 'lfo.out' is int32 Q17.15"),
    ("frac_bits on a float param",
      patchText "f6" [paramDecl "f" (num 2.0) "float" (some 4), lfo (Lean.toJson (1 : Nat))] out,
      some "fraction bits need an integer kind"),
    ("33 fraction bits on int32",
      patchText "f7" [paramDecl "f" (num 0.25) "int32" (some 33), lfo (Lean.toJson (1 : Nat))] out,
      some "int32 has 32 bits; 33 fraction bits do not fit"),
    ("a default outside the format's range",
      patchText "f8" [paramDecl "f" (num 40000.0) "int32" (some 16), lfo (param "f")] out,
      some "value 40000 is outside the range of int32 Q16.16"),
    ("bool param",
      patchText "f9" [paramDecl "f" (num 1.0) "bool", lfo (Lean.toJson (1 : Nat))] out,
      some "type 'bool' is not a param type"),
    ("a bare literal past the Q16.16 range",
      patchText "f11" [lfo (Lean.toJson (40000 : Nat))] out,
      some "input 'lfo.freq': literal 40000 is outside the range of int32 Q16.16, [-32768, 32768)"),
    ("a bare literal finer than 2^-16",
      patchText "f12" [lfo (num 0.00001)] out,
      some "input 'lfo.freq': literal 0.00001 is not exactly representable in int32 Q16.16"),
    ("a bare literal past the Q0.32 range",
      patchText "f13" [instDecl "lfo" "SinOscQ15" [("phase", num 0.5)]] out,
      some "input 'lfo.phase': literal 0.5 is outside the range of int32 Q0.32, [-0.5, 0.5)"),
    ("a bare literal 0.5 Hz at Q16.16",
      patchText "ok2" [lfo (num 0.5)] out, none),
    ("int32 output and float output on one output buffer",
      (Lean.Json.mkObj [("schema", "tropical_program_2"), ("name", "f10"),
        ("body", Lean.Json.mkObj [("op", "block"), ("decls", Lean.Json.arr #[
            lfo (Lean.toJson (1 : Nat)), instDecl "osc" "FixedSinOsc" []]),
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
  let intParamPatch := patchText "int_param"
    [paramDecl "ph" (Lean.toJson (1073741824 : Nat)) "int",
     instDecl "s" "FixedSin" [("phase", param "ph")]] (ref "s" "out")
  match compile (knobPatch 2.0), compile (patchText "old" [instDecl "lfo" "FixedSinOscQ15" [("freq", Lean.toJson (2 : Nat))]] (ref "lfo" "out")),
        compile intParamPatch with
  | .error e, _, _ | _, .error e, _ | _, _, .error e => bad := bad.push s!"compile: {e}"
  | .ok (plan, _), .ok (old, _), .ok (intPlan, _) =>
    let fi := (slotIndexOf plan "param:freq").getD 0
    let phi := (slotIndexOf intPlan "param:ph").getD 0
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
            { t with format := { t.format with fracBits := 40 } } }),
      ("int32 param default past the word", "is outside the int32 word",
        { plan with slotDefaults := plan.slotDefaults.set! fi (Lean.toJson (10000000000 : Nat)) }),
      ("int32 param default not an integer", "is not an integer word",
        { plan with slotDefaults := plan.slotDefaults.set! fi (num 131072.5) }),
      ("an int param read straight into the output", "a typed output is int32",
        { intPlan with sinks := intPlan.sinks.map fun sk => { sk with inputs := #[phi] } })]
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
    passGate "typed-ports-refusals" s!"{ingestCases.length} ingest cases (format and kind mismatches refused, a shifted wire compiles, bad declarations and unrepresentable bare literals refused), 8 hand-edited plans refused by PlanDecode, Metal refuses a typed param"
  else failGate "typed-ports-refusals" (String.intercalate "; " bad.toList)

/-- Run an engine action; its failure as JSON text. -/
private def runE {α : Type} (act : Tropical.EngineM α) : IO (Except String α) := do
  match ← act.run with
  | .ok a => pure (.ok a)
  | .error f => pure (.error f.toJson.compress)

/-- Compile the engine's current session (the mirror) and render `n` frames
    from sample 0 through a fresh runtime. -/
private def renderSession (env : Tropical.Engine.Env) (n : Nat) :
    IO (Except String (FlatPlan × Array Float)) := do
  match ← runE (Tropical.Engine.compileMirrorStaged env .fused) with
  | .error e => pure (.error e)
  | .ok m => pure (.ok (m.plan, ← render m.plan m.stageBlocks 0 n))

private def loadJ (program : Lean.Json) : Lean.Json :=
  Lean.Json.mkObj [("program", program)]

private def parseJ (text : String) : Lean.Json :=
  (Lean.Json.parse text).toOption.getD Lean.Json.null

private def fieldNat? (j : Lean.Json) (k : String) : Option Nat :=
  match j.getObjValD k with
  | .num n => if n.exponent == 0 && n.mantissa ≥ 0 then some n.mantissa.toNat else none
  | _ => none

def runTypedPortsEngine : IO Bool := do
  let mut bad : Array String := #[]
  let n := 512
  let env ← Tropical.NativeBoot.boot
  -- the reference: SinOscQ15 at a bare 500 Hz
  let direct := parseJ (patchText "direct"
    [instDecl "lfo" "SinOscQ15" [("freq", Lean.toJson (500 : Nat))]] (ref "lfo" "out"))
  let want ← match ← runE (Tropical.Engine.handleLoad env (loadJ direct)) with
    | .error e => bad := bad.push s!"load direct: {e}"; pure #[]
    | .ok _ =>
      match ← renderSession env n with
      | .error e => bad := bad.push s!"render direct: {e}"; pure #[]
      | .ok (_, xs) => pure xs
  -- export it with `f` exposed (its wire, the bare 500, becomes the default)
  match ← runE (Tropical.Engine.handleExportProgram env (Lean.Json.mkObj [
      ("name", "TypedLfoExport"),
      ("outputs", Lean.Json.mkObj [("out", Lean.Json.mkObj [("instance", "lfo"), ("output", "out")])]),
      ("inputs", Lean.Json.mkObj [("f", "lfo:freq")])])) with
  | .error e => bad := bad.push s!"export: {e}"
  | .ok r =>
    let ports := (r.getObjValD "program").getObjValD "ports"
    let in0 := ((ports.getObjValD "inputs").getArrVal? 0).toOption.getD Lean.Json.null
    let out0 := ((ports.getObjValD "outputs").getArrVal? 0).toOption.getD Lean.Json.null
    unless fieldNat? in0 "frac_bits" == some 16 && fieldNat? out0 "frac_bits" == some 15 do
      bad := bad.push s!"export ports lost frac_bits: {ports.compress}"
    -- the exported program JSON normalizes with its formats intact
    match Tropical.Parse.JsonV.parse (r.getObjValD "program").compress
        >>= Tropical.Parse.Raise.normalizeProgramFile with
    | .error e => bad := bad.push s!"normalize exported program: {e}"
    | .ok node =>
      unless (node.toJson.compress.splitOn "\"frac_bits\":16").length > 1 do
        bad := bad.push "normalize dropped the exported input's frac_bits"
  let uses : List (String × Lean.Json) := [
    ("a bare literal into the exported input",
      parseJ (patchText "use1" [instDecl "w" "TypedLfoExport" [("f", Lean.toJson (500 : Nat))]] (ref "w" "out"))),
    ("a Q16.16 param into the exported input",
      parseJ (patchText "use2" [paramDecl "freq" (Lean.toJson (500 : Nat)) "int32" (some 16),
        instDecl "w" "TypedLfoExport" [("f", param "freq")]] (ref "w" "out"))),
    ("the exported input's default (the literal's word)",
      parseJ (patchText "use3" [instDecl "w" "TypedLfoExport" []] (ref "w" "out")))]
  for (label, patch) in uses do
    match ← runE (Tropical.Engine.handleLoad env (loadJ patch)) with
    | .error e => bad := bad.push s!"{label}: load: {e}"
    | .ok _ =>
      match ← renderSession env n with
      | .error e => bad := bad.push s!"{label}: {e}"
      | .ok (plan, got) =>
        if plan.outputFormat != { kind := .int32, fracBits := 15 } then
          bad := bad.push s!"{label}: output format {plan.outputFormat.describe}"
        let peak := got.foldl (fun m x => max m x.abs) 0.0
        unless peak > 0.99 && peak ≤ 1.0 do bad := bad.push s!"{label}: peak {peak}"
        if let some k := firstMismatch got want then
          bad := bad.push s!"{label}: frame {k}: {got[k]?} ≠ direct {want[k]?}"
        -- and it survives save → load (a default's word is not rescaled)
        match ← runE (do
            let saved ← Tropical.Engine.handleSave env
            Tropical.Engine.handleLoad env (loadJ (saved.getObjValD "program"))) with
        | .error e => bad := bad.push s!"{label}: save → load: {e}"
        | .ok _ =>
          match ← renderSession env n with
          | .error e => bad := bad.push s!"{label}: after save → load: {e}"
          | .ok (_, again) =>
            if let some k := firstMismatch again want then
              bad := bad.push s!"{label}: after save → load: frame {k}: {again[k]?} ≠ direct {want[k]?}"
  -- save → load with typed params, after set_param
  let knob := parseJ (patchText "knob"
    [paramDecl "freq" (num 2.0) "int32" (some 16),
     paramDecl "ph" (num 0.25) "int32" (some 32),
     instDecl "lfo" "SinOscQ15" [("freq", param "freq"), ("phase", param "ph")]] (ref "lfo" "out"))
  let setP (name : String) (v : Float) :=
    Tropical.Engine.handleSetParam env (Lean.Json.mkObj [("name", Lean.Json.str name), ("value", num v)])
  let freshAt := parseJ (patchText "knob_at"
    [paramDecl "freq" (num 1234.5) "int32" (some 16),
     paramDecl "ph" (num (-0.125)) "int32" (some 32),
     instDecl "lfo" "SinOscQ15" [("freq", param "freq"), ("phase", param "ph")]] (ref "lfo" "out"))
  let fresh ← match ← runE (Tropical.Engine.handleLoad env (loadJ freshAt)) with
    | .error e => bad := bad.push s!"load knob_at: {e}"; pure #[]
    | .ok _ =>
      match ← renderSession env n with
      | .error e => bad := bad.push s!"render knob_at: {e}"; pure #[]
      | .ok (_, xs) => pure xs
  match ← runE (do
      let _ ← Tropical.Engine.handleLoad env (loadJ knob)
      let _ ← setP "freq" 1234.5
      let _ ← setP "ph" (-0.125)
      Tropical.Engine.handleSave env) with
  | .error e => bad := bad.push s!"set_param/save: {e}"
  | .ok saved =>
    let program := saved.getObjValD "program"
    for needle in ["\"frac_bits\":16", "\"frac_bits\":32", "\"value\":1234.5", "\"value\":-0.125"] do
      unless (program.compress.splitOn needle).length > 1 do
        bad := bad.push s!"save: no {needle} in {program.compress}"
    match ← runE (Tropical.Engine.handleLoad env (loadJ program)) with
    | .error e => bad := bad.push s!"reload saved: {e}"
    | .ok _ =>
      match ← renderSession env n with
      | .error e => bad := bad.push s!"render saved: {e}"
      | .ok (plan, got) =>
        let word (nm : String) := (slotIndexOf plan s!"param:{nm}").bind (plan.slotDefaults[·]?)
        unless word "freq" == some (Lean.toJson (80904192 : Int))
            && word "ph" == some (Lean.toJson (-536870912 : Int)) do
          bad := bad.push s!"saved words: freq {(word "freq").map (·.compress)}, ph {(word "ph").map (·.compress)}"
        if let some k := firstMismatch got fresh then
          bad := bad.push s!"saved → loaded: frame {k}: {got[k]?} ≠ fresh {fresh[k]?}"
    -- an out-of-range set_param is refused with the range, and changes nothing
    for (nm, v, needle) in [("freq", 100000.0, "value 100000 is outside the range of int32 Q16.16, [-32768, 32768)"),
                            ("ph", 0.5, "value 0.5 is outside the range of int32 Q0.32, [-0.5, 0.5)")] do
      match ← runE (setP nm v) with
      | .ok _ => bad := bad.push s!"set_param {nm} {v}: accepted"
      | .error e =>
        unless (e.splitOn needle).length > 1 && (e.splitOn "invalid_value").length > 1 do
          bad := bad.push s!"set_param {nm} {v}: refused with {e}, expected '{needle}'"
    match ← runE (do
        let ps ← Tropical.Engine.handleListParams env
        let saved ← Tropical.Engine.handleSave env
        let _ ← Tropical.Engine.handleLoad env (loadJ (saved.getObjValD "program"))
        pure ps) with
    | .error e => bad := bad.push s!"after a refused set_param: {e}"
    | .ok ps =>
      unless (ps.compress.splitOn "\"value\":1234.5").length > 1 do
        bad := bad.push s!"the mirror changed on a refused set_param: {ps.compress}"
  if bad.isEmpty then
    passGate "typed-ports-engine" s!"export_program keeps Q16.16/Q17.15 (an exported SinOscQ15 ≡ the direct one through a bare literal, a Q16.16 param and its default; peak < 1), save → load round-trips two typed params after set_param, an out-of-range set_param is refused with its range and leaves save loadable"
  else failGate "typed-ports-engine" (String.intercalate "; " bad.toList)

end Tropical.Tropicaltest.TypedPorts
