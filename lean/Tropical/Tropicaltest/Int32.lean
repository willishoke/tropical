import Tropical.Tropicaltest.ArrowLaws
import Tropical.Testing.ArrowFixtures
import Tropical.Semantics.PlanWellFormed
import Tropical.Semantics.Scalar
import Tropical.Session
import Tropical.Wiring

/-!
# The integer kinds: `int32`, and the total integer semantics

`int32` is two's complement on ℤ/2³² on every target, and every integer op of
both kinds is total (`ConstFold`, `Semantics.Scalar`). The gates:

* **model** — `Int32Zoo`'s word (every int32 op, seeded from the sample
  index's low word so nothing folds, with runtime zero and -1 divisors under
  `INT32_MIN`, runtime shift amounts past the word, and a saturating
  conversion) rendered by the JIT equals the reference semantics — the plan
  executed by `Semantics.Scalar.algebra` — bit for bit, from sample 0 and from
  just below 2³¹ (so the input word itself wraps); and again with the word's
  input wired from `FixedSinOscQ15`'s int32 output through a module slot.
* **edge** — the edge-value differential: `IntEdge64`, `IntEdge32` and
  `IntEdgeConv` (every `int` and `int32` op on MIN, MAX, -1, 0, 1, runtime
  divisors 0 and -1, shift amounts -1/0/1/31/32/33/63/64/65, NaN, ±∞ and
  out-of-range floats into the conversions, and a seeded random sample)
  rendered by the JIT, and by Metal offline when this build has it, equal the
  reference semantics frame for frame. WASM is held to the JIT bit for bit on
  the same programs by `tests/web/wasm_vs_jit.test.ts` (and Metal again by
  `metal_vs_jit.test.ts`).
* **refusals** — an op mixing `int32` with `int` or `float`, an int32
  literal outside `[-2³¹, 2³¹)` or with a fraction, and `litI32` out of
  range are compile errors; the patch bay connects `int32` only to `int32`
  (and widens `bool` into it); and through the real ingest path
  (`Session.compilePatch`, what `diffcli compile` and `load` run) an `int32`
  value reaching an `int` or `float` input — bare or inside an expression —
  or a `float` value reaching an `int32` input is refused, while the same
  wires with the cast spelled compile.
* **well-kinded plans** — `flatPlanWellFormed` accepts compiled plans, and
  `PlanDecode` refuses hand-edited ones: an int32 operand relabeled `int`, a
  `reg` read at a sort its producer did not write, an int32 constant out of
  range, an int32 `WriteSlot` into a float slot, an int32 array read, and a
  retired or unknown tag (`MulLo32`).
-/

namespace Tropical.Tropicaltest.Int32

open Tropical
open Tropical.Plan
open Tropical.EmitArrow
open Tropical.Ir (Arena ProgramIdx CVal)

abbrev ScalarKind := Tropical.Parse.ScalarKind

private def renderFrom (plan : FlatPlan) (start n : Nat) : IO (Array Float) := do
  let rt ← Tropical.Ffi.Runtime.new (UInt32.ofNat n)
  Tropical.StagedLoad.load rt plan
  rt.setSampleIndex start.toUInt64
  rt.process
  pure (decodeF64LE (← rt.outputBytes))

/-- The Metal render of `n` frames from `start`, offline (no device is
    opened), or `none` when this build or machine has no Metal backend. -/
private def renderMetalFrom (plan : FlatPlan) (start n : Nat) : IO (Option (Array Float)) := do
  let buffer := 256
  try
    let rt ← Tropical.Ffi.Runtime.new (UInt32.ofNat buffer)
    Tropical.StagedLoad.loadMsl rt plan
    if start != 0 then rt.setSampleIndex start.toUInt64
    let mut out : Array Float := #[]
    for _ in [0:(n + buffer - 1) / buffer] do
      rt.processOffline
      out := out ++ decodeF64LE (← rt.outputBytes)
    pure (some (out.extract 0 n))
  catch _ => pure none

/-- The reference render: the plan executed by the reference semantics
    (`Semantics.observeFlatPlan` over `Semantics.Scalar.algebra`), channel 0
    of the sink image, for frames `start … start + n - 1`. The clock source is
    τ = sample · 2¹⁶ (Q48.16), slots start at the plan's defaults, arrays at 0. -/
def refRender (plan : FlatPlan) (start n : Nat) : Except String (Array Float) := do
  let slots : Array (Tropical.Semantics.Value CVal) := plan.slotDefaults.map fun j =>
    match j with
    | .num v => .scalar (.f v.toFloat)
    | _ => .scalar (.f 0.0)
  let arrays : Array (Array (Tropical.Semantics.Value CVal)) :=
    plan.arraySlotSizes.map fun size => Array.replicate size (.scalar (.f 0.0))
  let mut out := #[]
  for k in [0:n] do
    let frame : Int := start + k
    let sources : Array (Tropical.Semantics.Value CVal) := plan.sources.map fun
      | .tau => .scalar (.i (frame * 65536))
      | .rate => .scalar (.f plan.sampleRate.toFloat)
    let inputs : Tropical.Semantics.PlanInputs CVal := { sources, initialSlots := slots, initialArrays := arrays }
    match Tropical.Semantics.observeFlatPlan Tropical.Semantics.Scalar.algebra inputs plan with
    | .error e => throw s!"frame {frame}: {e.operation}: {e.detail}"
    | .ok obs =>
      let ch0 : Option (Tropical.Semantics.Value CVal) := obs.sinks[0]?
      match ch0 with
      | some (.scalar (.f x)) => out := out.push x
      | some (.scalar (.i v)) | some (.scalar (.i32 v)) => out := out.push (Float.ofInt v)
      | some (.scalar (.b v)) => out := out.push (if v then 1.0 else 0.0)
      | _ => throw s!"frame {frame}: channel 0 is not a scalar"
  pure out

/-- The first frame (from `from`) where `got` and `want` differ bitwise. -/
private def firstMismatch (got want : Array Float) (skip : Nat → Bool := fun _ => false) :
    Option Nat := Id.run do
  for k in [0:min got.size want.size] do
    if !skip k && got[k]!.toBits != want[k]!.toBits then return some k
  return none

private def carrier (name : String) (out : BuildM Sig) : Except String (Arena × ProgramIdx) :=
  assemble {} name #[{ name := "out", type? := some (.scalar .float) }] #[] do
    pure { assigns := #[(.port ⟨0⟩, ← out)] }

private def word : BuildM Sig := do toInt32E (← sampleIndex)

/-- A root over registered instances: `FixedSinOscQ15` at 1234.567 Hz, and,
    when `wired`, `Int32Zoo` with `x` wired from the LFO's int32 output (a
    module-slot crossing at int32); `out` is the zoo's output, or the LFO's
    word as a float. -/
private def wiredRoot (wired : Bool) : IO (Except String Tropical.Plan.FlatPlan) := do
  match ← arrowElabStdlib with
  | .error e => pure (.error e)
  | .ok (arena, resolved) => pure <| buildAndFinish do
    let (arena, zooIdx) ← buildInt32Zoo arena
    let resolved := resolved.push ("Int32Zoo", zooIdx)
    let registry ← buildRegistry arena resolved #["FixedSinOscQ15", "Int32Zoo"]
    assemble arena (if wired then "WiredZoo" else "LfoWord")
        #[{ name := "out", type? := some (.scalar .float) }] registry do
      let lfo ← inst "lfo" "FixedSinOscQ15" #[{ port := ⟨0⟩, value := ← lit 1234567 3 }]
      let y ← nestedOut lfo ⟨0⟩
      if wired then
        let zoo ← inst "zoo" "Int32Zoo" #[{ port := ⟨0⟩, value := y }]
        pure { assigns := #[(.port ⟨0⟩, ← nestedOut zoo ⟨0⟩)] }
      else
        pure { assigns := #[(.port ⟨0⟩, ← toFloatE y)] }

/-- JIT ≡ reference over `n` frames from `start`; the failure, if any. -/
private def jitVsRef (label : String) (plan : FlatPlan) (start n : Nat) : IO (Option String) := do
  let got ← renderFrom plan start n
  match refRender plan start n with
  | .error e => pure (some s!"{label}: reference: {e}")
  | .ok want =>
    if got.size != n || want.size != n then
      pure (some s!"{label}: rendered {got.size}/{want.size} frames, wanted {n}")
    else match firstMismatch got want with
      | some k => pure (some s!"{label}: frame {start + k}: JIT {got[k]!} ≠ reference {want[k]!}")
      | none => pure none

def runInt32Model : IO Bool := do
  let n := 2048
  let mut bad : Array String := #[]
  -- (a) every op, seeded from the sample index's low word
  match buildAndFinish (carrier "Int32ZooRoot" (do
      div (← toFloatE (← int32ZooWord (← word))) (← lit 2147483648))) with
  | .error e => bad := bad.push s!"compile: {e}"
  | .ok plan =>
    for start in [0, 2147483648 - 1024] do
      if let some e ← jitVsRef s!"zoo from {start}" plan start n then bad := bad.push e
  -- (b) the same word with `x` crossing a module slot from FixedSinOscQ15
  match ← wiredRoot false, ← wiredRoot true with
  | .error e, _ | _, .error e => bad := bad.push s!"wired compile: {e}"
  | .ok lfoPlan, .ok zooPlan =>
    let ys ← renderFrom lfoPlan 0 n
    let distinct := (ys.foldl (fun acc y => if acc.contains y then acc else acc.push y) #[]).size
    if distinct < 64 then bad := bad.push s!"wired: only {distinct} distinct LFO words"
    if let some e ← jitVsRef "wired zoo" zooPlan 0 n then bad := bad.push e
  if bad.isEmpty then
    passGate "int32-model" s!"JIT ≡ the reference semantics (Semantics.Scalar) bit for bit: {2 * n} frames of Int32Zoo's word across the input's 2³¹ wrap (runtime zero/-1 divisors under INT32_MIN, shifts past the word, saturation included), {n} with x wired from FixedSinOscQ15 through an int32 slot"
  else failGate "int32-model" (String.intercalate "; " bad.toList)

/-- The edge-value differential. -/
def runIntEdge : IO Bool := do
  let mut bad : Array String := #[]
  let mut notes : Array String := #[]
  let mut metalRan := false
  let specs : Array (String × (Arena → Except String (Arena × ProgramIdx)) × Nat
      × Array (String × ScalarKind)) := #[
    ("IntEdge64", buildIntEdge .int, intEdgeFramesPerOp .int,
      (intEdgeOps .int).map fun (n, k, _) => (n, k)),
    ("IntEdge32", buildIntEdge .int32, intEdgeFramesPerOp .int32,
      (intEdgeOps .int32).map fun (n, k, _) => (n, k)),
    ("IntEdgeConv", buildIntEdgeConv, intEdgeConvFramesPerOp, intEdgeConvOps)]
  for (name, build, fpo, ops) in specs do
    let n := fpo * ops.size
    match buildAndFinish (build {}) with
    | .error e => bad := bad.push s!"{name}: compile: {e}"
    | .ok plan =>
      let got ← renderFrom plan 0 n
      match refRender plan 0 n with
      | .error e => bad := bad.push s!"{name}: reference: {e}"
      | .ok want =>
        let opAt (k : Nat) : String × ScalarKind := ops[k / fpo]!
        -- not vacuous: every op's segment carries a nonzero frame
        for o in [0:ops.size] do
          unless (List.range fpo).any (fun j => want[o * fpo + j]! != 0.0) do
            bad := bad.push s!"{name}: op {ops[o]!.1} reads out all zeros (a vacuous segment)"
        match firstMismatch got want with
        | some k =>
          bad := bad.push s!"{name}: JIT ≠ reference at frame {k} ({(opAt k).1}, pair {(k % fpo) / 4}, piece {k % 4}): {got[k]!} vs {want[k]!}"
        | none => notes := notes.push s!"{name} {n}"
        -- Metal: f32 values, so a float RESULT (`toFloat`) is outside the
        -- exact comparison; every integer result is exact.
        match ← renderMetalFrom plan 0 n with
        | none => pure ()
        | some gpu =>
          metalRan := true
          match firstMismatch gpu want (fun k => (opAt k).2 == .float) with
          | some k =>
            bad := bad.push s!"{name}: Metal ≠ reference at frame {k} ({(opAt k).1}, pair {(k % fpo) / 4}, piece {k % 4}): {gpu[k]!} vs {want[k]!}"
          | none => pure ()
  let metalNote := if metalRan then "; Metal (offline) ≡ reference on every integer result"
    else "; Metal unavailable in this build (skipped)"
  if bad.isEmpty then
    passGate "int-edge" s!"JIT ≡ reference semantics on every int and int32 op at the edges (frames: {String.intercalate ", " notes.toList}){metalNote}"
  else failGate "int-edge" (String.intercalate "; " bad.toList)

private def expectRefusal (label needle : String)
    (built : Except String (Arena × ProgramIdx)) : Option String :=
  match buildAndFinish built with
  | .ok _ => some s!"{label}: compiled, expected a refusal"
  | .error e =>
    if (e.splitOn needle).length > 1 then none
    else some s!"{label}: refused with '{e}', expected '{needle}'"

/-- The stdlib registry plus `Int32Zoo`, as `diffcli compile --fixtures`
    sees it. -/
private def fixtureRegistry : Except String Tropical.SessionSt := do
  let st ← Tropical.Session.bootRegistry.mapError (·.toJson.compress)
  let act : Tropical.Session.SessionM Unit := do
    let s ← get
    match buildInt32Zoo s.arena with
    | .error e => Tropical.internalError e
    | .ok (arena', idx) => discard <| Tropical.Session.registerResolved "Int32Zoo" arena' idx
  match act.runOn st with
  | (.ok (), st') => .ok st'
  | (.error f, _) => .error f.toJson.compress

/-- A `tropical_program_2` patch: `decls` then `dac.out = v.out`. -/
private def patchText (name : String) (decls : List Lean.Json) (dac : Lean.Json) : String :=
  (Lean.Json.mkObj [("schema", "tropical_program_2"), ("name", Lean.Json.str name),
    ("body", Lean.Json.mkObj [("op", "block"), ("decls", Lean.Json.arr decls.toArray),
      ("assigns", Lean.Json.arr #[Lean.Json.mkObj [("op", "outputAssign"),
        ("name", "dac.out"), ("expr", dac)]])])]).compress

private def instDecl (name program : String) (inputs : List (String × Lean.Json)) : Lean.Json :=
  Lean.Json.mkObj [("op", "instanceDecl"), ("name", Lean.Json.str name),
    ("program", Lean.Json.str program), ("inputs", Lean.Json.mkObj inputs)]

private def ref (inst out : String) : Lean.Json :=
  Lean.Json.mkObj [("op", "ref"), ("instance", Lean.Json.str inst), ("output", Lean.Json.str out)]

private def opJ (op : String) (args : List Lean.Json) : Lean.Json :=
  Lean.Json.mkObj [("op", Lean.Json.str op), ("args", Lean.Json.arr args.toArray)]

/-- The ingest-path crossings: each `(label, patch, needle?)` must refuse
    with `needle` (or compile when `none`). -/
private def ingestCases : List (String × String × Option String) :=
  let lfo := instDecl "lfo" "FixedSinOscQ15" [("freq", (100 : Nat))]
  let vcaOut := ref "v" "out"
  let vca (audio : Lean.Json) := instDecl "v" "VCA" [("audio", audio), ("cv", (1 : Nat))]
  [ ("int32 out → float in (bare ref)",
      patchText "x1" [lfo, vca (ref "lfo" "out")] vcaOut,
      some "input 'audio' of instance 'v' is float but its value is int32"),
    ("add(int32 out, 1) → float in",
      patchText "x2" [lfo, vca (opJ "add" [ref "lfo" "out", (1 : Nat)])] vcaOut,
      some "input 'audio' of instance 'v' is float but its value is int32"),
    ("float out → int in (FixedSinOscQ15.clk)",
      patchText "x3" [instDecl "osc" "FixedSinOsc" [("freq", (100 : Nat))],
        instDecl "lfo" "FixedSinOscQ15" [("freq", (100 : Nat)), ("clk", ref "osc" "sine")]]
        (ref "lfo" "out"),
      some "input 'clk' of instance 'lfo' is int but its value is float"),
    ("int32 out → int in (FixedSinOscQ15.clk)",
      patchText "x3b" [instDecl "a" "FixedSinOscQ15" [("freq", (100 : Nat))],
        instDecl "lfo" "FixedSinOscQ15" [("freq", (100 : Nat)), ("clk", ref "a" "out")]]
        (ref "lfo" "out"),
      some "input 'clk' of instance 'lfo' is int but its value is int32"),
    ("float out → int32 in (Int32Zoo.x)",
      patchText "x4" [instDecl "osc" "FixedSinOsc" [("freq", (100 : Nat))],
        vca (ref "osc" "sine"), instDecl "zoo" "Int32Zoo" [("x", ref "v" "out")]]
        (ref "zoo" "out"),
      some "input 'x' of instance 'zoo' is int32 but its value is float"),
    ("toFloat(int32 out) → float in",
      patchText "ok1" [lfo, vca (opJ "toFloat" [ref "lfo" "out"])] vcaOut, none),
    ("toFloat(add(int32 out, 1)) → float in",
      patchText "ok2" [lfo, vca (opJ "toFloat" [opJ "add" [ref "lfo" "out", (1 : Nat)]])] vcaOut, none),
    ("add(neg(3), int32 out) adopts int32, then toFloat",
      patchText "ok3" [lfo, vca (opJ "toFloat" [opJ "add" [opJ "neg" [(3 : Nat)], ref "lfo" "out"]])]
        vcaOut, none),
    ("toInt(int32 out) → int in",
      patchText "ok4" [instDecl "a" "FixedSinOscQ15" [("freq", (100 : Nat))],
        instDecl "lfo" "FixedSinOscQ15" [("freq", (100 : Nat)), ("clk", opJ "toInt" [ref "a" "out"])]]
        (ref "lfo" "out"), none),
    ("toInt32(float out) → int32 in",
      patchText "ok5" [instDecl "osc" "FixedSinOsc" [("freq", (100 : Nat))],
        vca (ref "osc" "sine"), instDecl "zoo" "Int32Zoo" [("x", opJ "toInt32" [ref "v" "out"])]]
        (ref "zoo" "out"), none),
    ("int32 out → dac (bare)",
      patchText "ok6" [lfo] (ref "lfo" "out"), none),
    -- a lifted wire (`__wire_N`) keeps an int32 source's kind on its input
    ("index([1, 2, 3], int32 out) → float in (lifted wire)",
      patchText "ok7" [lfo, vca (opJ "index" [Lean.Json.arr #[(1 : Nat), (2 : Nat), (3 : Nat)],
        ref "lfo" "out"])] vcaOut, none),
    ("[int32 out, 1.0] (lifted wire) → array element",
      patchText "x5" [lfo, vca (Lean.Json.arr #[ref "lfo" "out", (1 : Nat)])] vcaOut,
      some "an int32 array element")]

def runInt32Refusals : IO Bool := do
  let mut bad : Array String := #[]
  let cases : List (String × String × Except String (Arena × ProgramIdx)) := [
    ("int32 + int", "mixes int32 with int",
      carrier "MixInt" (do toFloatE (← add (← word) (← sampleIndex)))),
    ("int32 < float", "Less mixes int32 with float",
      carrier "MixFloat" (do
        selectE (← binary .lt (← word) (← sampleRate)) (← lit 1) (← lit 0))),
    ("int32 literal 2³¹", "outside int32's range",
      carrier "BigLit" (do toFloatE (← add (← word) (← lit 2147483648)))),
    ("int32 literal 0.5", "cannot narrow to int32",
      carrier "FracLit" (do toFloatE (← add (← word) (← lit 5 1)))),
    ("sqrt of an int32", "Sqrt computes on float and takes no int32",
      carrier "SqrtWord" (do unary .sqrt (← word))),
    ("litI32 2³¹", "outside int32's range",
      carrier "BigLitI32" (do toFloatE (← add (← word) (← litI32 2147483648)))),
    ("int32 assigned to a float output", "output port 0 is float but its value is int32",
      carrier "BareWordOut" word)]
  for (label, needle, built) in cases do
    if let some msg := expectRefusal label needle built then bad := bad.push msg
  -- The literal rule's positive half: a bare literal or a literal-only
  -- subtree on either side of an int32 operand is int32, and a cast is the
  -- only crossing.
  if let .error e := buildAndFinish (carrier "LitBothSides" (do
      toFloatE (← sub (← lit 7) (← mul (← word) (← lit (-3)))))) then
    bad := bad.push s!"literal adoption: {e}"
  if let .error e := buildAndFinish (carrier "LitTreeLeft" (do
      toFloatE (← add (← neg (← lit 3)) (← word)))) then
    bad := bad.push s!"literal-subtree adoption (left): {e}"
  if let .error e := buildAndFinish (carrier "LitTreeRight" (do
      toFloatE (← sub (← word) (← mul (← lit 2) (← neg (← lit 3)))))) then
    bad := bad.push s!"literal-subtree adoption (right): {e}"
  -- The patch bay: int32 meets only int32; bool widens into it.
  let node : WireExpr := .clock
  let conn (src dst : ScalarKind) : Bool :=
    (Tropical.Wiring.checkArrayConnection (some (.scalar src)) (some (.scalar dst)) node).compatible
  let expected : List (ScalarKind × ScalarKind × Bool) := [
    (.int32, .int32, true), (.bool, .int32, true), (.int32, .float, false),
    (.int32, .int, false), (.int, .int32, false), (.float, .int32, false),
    (.int, .float, true)]
  for (src, dst, ok) in expected do
    if conn src dst != ok then
      bad := bad.push s!"wiring {src.wire} → {dst.wire}: compatible = {conn src dst}, expected {ok}"
  -- The ingest path: instance-input binding refuses every implicit crossing.
  match fixtureRegistry with
  | .error e => bad := bad.push s!"fixture registry: {e}"
  | .ok registry =>
    for (label, text, needle?) in ingestCases do
      match (Tropical.Session.compilePatch text).runOn registry, needle? with
      | (.ok _, _), none => pure ()
      | (.ok _, _), some needle => bad := bad.push s!"ingest '{label}': compiled, expected '{needle}'"
      | (.error f, _), none => bad := bad.push s!"ingest '{label}': refused: {f.toJson.compress}"
      | (.error f, _), some needle =>
        let msg := f.toJson.compress
        unless (msg.splitOn needle).length > 1 do
          bad := bad.push s!"ingest '{label}': refused with {msg}, expected '{needle}'"
  if bad.isEmpty then
    passGate "int32-refusals" s!"{cases.length} mixed/out-of-range spellings refused, literals and literal-only subtrees adopt int32 from either side, patch bay int32 ↔ int32 only, {ingestCases.length} ingest-path crossings refused or (with the cast) compiled"
  else failGate "int32-refusals" (String.intercalate "; " bad.toList)

/-- Apply `f` to the first instruction (execution order) it changes. -/
private partial def editFirst (f : NInstr → Option NInstr) (inst : InstanceFunction) :
    InstanceFunction × Bool :=
  match inst with
  | .mk name iname pre instrs preIn regOff arrOff regCount children =>
    let fix (xs : Array NInstr) (done : Bool) : Array NInstr × Bool := Id.run do
      let mut out := #[]
      let mut done := done
      for i in xs do
        if !done then
          match f i with
          | some i' => out := out.push i'; done := true
          | none => out := out.push i
        else out := out.push i
      pure (out, done)
    let (pre', d0) := fix pre false
    let (children', d1) := children.foldl (fun (acc, d) c =>
      if d then (acc.push c, d) else
        let (c', d') := editFirst f c
        (acc.push c', d')) (#[], d0)
    let (instrs', d2) := fix instrs d1
    (.mk name iname pre' instrs' preIn regOff arrOff regCount children', d2)

private def editPlan (plan : FlatPlan) (f : NInstr → Option NInstr) : FlatPlan × Bool :=
  let (fns, changed) := plan.instanceFunctions.foldl (fun (acc, d) fn =>
    if d then (acc.push fn, d) else
      let (fn', d') := editFirst f fn
      (acc.push fn', d')) (#[], false)
  ({ plan with instanceFunctions := fns }, changed)

/-- Serialize, then decode: what a hand-edited plan file meets. -/
private def roundTrip (plan : FlatPlan) : Except String FlatPlan := do
  FlatPlan.ofWire (← plan.toWire)

def runInt32WellKinded : IO Bool := do
  let mut bad : Array String := #[]
  match buildAndFinish (buildInt32Zoo {}), buildAndFinish (buildIntEdge .int {}) with
  | .error e, _ | _, .error e => return ← failGate "int32-well-kinded" s!"compile: {e}"
  | .ok zoo, .ok edge =>
    -- (A carrier with arrays carries no array slot names, which the
    -- structural check wants; the kind check is the one asked of it here.)
    unless Tropical.Semantics.flatPlanWellFormed zoo do
      bad := bad.push "Int32Zoo: compiled plan is not well formed"
    for (label, plan) in [("Int32Zoo", zoo), ("IntEdge64", edge)] do
      if let some e := plan.kindError? then bad := bad.push s!"{label}: ill-kinded: {e}"
      if let .error e := roundTrip plan then bad := bad.push s!"{label}: decode refused: {e}"
    -- hand edits, each of which `PlanDecode` must refuse
    let edits : List (String × String × FlatPlan × (NInstr → Option NInstr)) := [
      ("int32 operand relabeled int", "is read as int but was produced as int32", zoo, fun i =>
        if i.tag == "Add" && i.resultType == .int32 then
          match i.args[0]? with
          | some (NOperand.reg s .int32) => some { i with args := i.args.set! 0 (.reg s .int) }
          | _ => none
        else none),
      ("reg read at another sort than its producer's", "but was produced as", zoo, fun i =>
        if i.tag == "ToFloat" then
          match i.args[0]? with
          | some (NOperand.reg s .int32) => some { i with args := #[.reg s .int], resultType := .float }
          | _ => none
        else none),
      ("int32 constant 3000000000", "outside [-2147483648, 2147483647]", zoo, fun i =>
        match i.args.findIdx? (fun | .const _ .int32 => true | _ => false) with
        | some j => some { i with args := i.args.set! j (.const (3000000000 : Nat) .int32) }
        | none => none),
      ("int32 WriteSlot into a float slot", "cannot be stored as float", zoo, fun i =>
        if i.tag == "Add" && i.resultType == .int32 then
          match i.dst with
          | .temp t => some { tag := "WriteSlot", dst := .moduleSlot 0, args := #[.reg t .int32],
                              resultType := .float }
          | _ => none
        else none),
      ("int32 read out of an array", "an array element is float", edge, fun i =>
        if i.tag == "Index" then some { i with resultType := .int32 } else none),
      ("retired tag MulLo32", "unknown instruction tag 'MulLo32'", zoo, fun i =>
        if i.tag == "Mul" && i.resultType == .int32 then some { i with tag := "MulLo32" } else none)]
    for (label, needle, base, f) in edits do
      let (edited, changed) := editPlan base f
      if !changed then bad := bad.push s!"'{label}': no instruction to edit"
      else
        if edited.kindError?.isNone then
          bad := bad.push s!"'{label}': the kind check accepted it"
        if Tropical.Semantics.flatPlanWellFormed edited then
          bad := bad.push s!"'{label}': flatPlanWellFormed accepted it"
        match roundTrip edited with
        | .ok _ => bad := bad.push s!"'{label}': PlanDecode accepted it"
        | .error e =>
          unless (e.splitOn needle).length > 1 do
            bad := bad.push s!"'{label}': refused with '{e}', expected '{needle}'"
    if bad.isEmpty then
      passGate "int32-well-kinded" s!"compiled plans are well formed and decode; PlanDecode and flatPlanWellFormed refuse {edits.length} hand edits (int32 relabeled int, a reg read at a foreign sort, an int32 constant out of range, an int32 store into a float slot, an int32 array read, MulLo32)"
    else failGate "int32-well-kinded" (String.intercalate "; " bad.toList)

end Tropical.Tropicaltest.Int32
