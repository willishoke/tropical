import Tropical.Tropicaltest.ArrowLaws
import Tropical.Testing.ArrowFixtures
import Tropical.Semantics.PlanWellFormed
import Tropical.Wiring

/-!
# The int32 scalar kind

`int32` is two's complement on ℤ/2³² on every target. Three gates:

* **model** — `int32ZooWord` (every int32 op, seeded from the sample index's
  low word so nothing folds) rendered by the JIT equals `int32ZooModel`, the
  same word computed on `Int` with the constant folder's `wrap32`, bit for
  bit, from sample 0 and from just below 2³¹ (so the input word itself
  wraps); and again with the word's input wired from `FixedSinOscQ15`'s
  int32 output through a module slot.
* **refusals** — an op mixing `int32` with `int` or `float`, an int32
  literal outside `[-2³¹, 2³¹)` or with a fraction, and `litI32` out of
  range are compile errors; the patch bay connects `int32` only to `int32`
  (and widens `bool` into it).
* **well-kinded plans** — `flatPlanWellFormed` accepts the compiled
  `Int32Zoo` plan and refuses the same plan with one int32 operand relabeled
  `int`.
-/

namespace Tropical.Tropicaltest.Int32

open Tropical
open Tropical.Plan
open Tropical.EmitArrow
open Tropical.Ir (Arena ProgramIdx)

abbrev ScalarKind := Tropical.Parse.ScalarKind

private def renderFrom (plan : FlatPlan) (start n : Nat) : IO (Array Float) := do
  let rt ← Tropical.Ffi.Runtime.new (UInt32.ofNat n)
  Tropical.StagedLoad.load rt plan
  rt.setSampleIndex start.toUInt64
  rt.process
  pure (decodeF64LE (← rt.outputBytes))

private def modelOf (x : Int) : Float :=
  Float.ofInt (int32ZooModel (Tropical.Ir.wrap32 x)) / 2147483648.0

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

def runInt32Model : IO Bool := do
  let n := 2048
  let mut bad : Array String := #[]
  -- (a) every op, seeded from the sample index's low word
  match buildAndFinish (carrier "Int32ZooRoot" (do
      div (← toFloatE (← int32ZooWord (← word))) (← lit 2147483648))) with
  | .error e => bad := bad.push s!"compile: {e}"
  | .ok plan =>
    for start in [0, 2147483648 - 1024] do
      let got ← renderFrom plan start n
      if got.size != n then
        bad := bad.push s!"start {start}: rendered {got.size} samples, wanted {n}"
      else
        for k in [0:n] do
          let want := modelOf (Int.ofNat (start + k))
          if got[k]!.toBits != want.toBits then
            bad := bad.push s!"sample {start + k}: JIT {got[k]!} ≠ model {want}"
            break
  -- (b) the same word with `x` crossing a module slot from FixedSinOscQ15
  match ← wiredRoot false, ← wiredRoot true with
  | .error e, _ | _, .error e => bad := bad.push s!"wired compile: {e}"
  | .ok lfoPlan, .ok zooPlan =>
    let ys ← renderFrom lfoPlan 0 n
    let zs ← renderFrom zooPlan 0 n
    let distinct := (ys.foldl (fun acc y => if acc.contains y then acc else acc.push y) #[]).size
    if ys.size != n || zs.size != n || distinct < 64 then
      bad := bad.push s!"wired: {ys.size}/{zs.size} samples, {distinct} distinct LFO words"
    else
      for k in [0:n] do
        let y := ys[k]!
        let want := modelOf (if y < 0 then -(Int.ofNat (-y).toUInt64.toNat) else Int.ofNat y.toUInt64.toNat)
        if zs[k]!.toBits != want.toBits then
          bad := bad.push s!"wired sample {k}: JIT {zs[k]!} ≠ model {want} (x = {y})"
          break
  if bad.isEmpty then
    passGate "int32-model" s!"JIT ≡ the Int model (wrap32) bit for bit: {2 * n} samples across the input word's 2³¹ wrap, {n} with x wired from FixedSinOscQ15 through an int32 slot"
  else failGate "int32-model" (String.intercalate "; " bad.toList)

private def expectRefusal (label needle : String)
    (built : Except String (Arena × ProgramIdx)) : Option String :=
  match buildAndFinish built with
  | .ok _ => some s!"{label}: compiled, expected a refusal"
  | .error e =>
    if (e.splitOn needle).length > 1 then none
    else some s!"{label}: refused with '{e}', expected '{needle}'"

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
      carrier "BigLitI32" (do toFloatE (← add (← word) (← litI32 2147483648))))]
  for (label, needle, built) in cases do
    if let some msg := expectRefusal label needle built then bad := bad.push msg
  -- The literal rule's positive half: a bare literal on either side of an
  -- int32 operand is an int32 constant, and a cast is the only crossing.
  if let .error e := buildAndFinish (carrier "LitBothSides" (do
      toFloatE (← sub (← lit 7) (← mul (← word) (← lit (-3)))))) then
    bad := bad.push s!"literal adoption: {e}"
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
  if bad.isEmpty then
    passGate "int32-refusals" s!"{cases.length} mixed/out-of-range spellings refused, literals adopt int32 from either side, patch bay int32 ↔ int32 only"
  else failGate "int32-refusals" (String.intercalate "; " bad.toList)

/-- Relabel the first int32 operand of the first int32 `Add` as `int`. -/
private partial def relabel (inst : InstanceFunction) : InstanceFunction × Bool :=
  match inst with
  | .mk name iname pre instrs preIn regOff arrOff regCount children =>
    let fix (xs : Array NInstr) : Array NInstr × Bool := Id.run do
      let mut out := #[]
      let mut done := false
      for i in xs do
        if !done && i.tag == "Add" && i.resultType == .int32 then
          match i.args[0]? with
          | some (NOperand.reg s .int32) =>
            out := out.push { i with args := i.args.set! 0 (.reg s .int) }
            done := true
          | _ => out := out.push i
        else out := out.push i
      pure (out, done)
    let (instrs', d1) := fix instrs
    let (children', d2) := children.foldl (fun (acc, d) c =>
      if d then (acc.push c, d) else
        let (c', d') := relabel c
        (acc.push c', d')) (#[], d1)
    (.mk name iname pre instrs' preIn regOff arrOff regCount children', d1 || d2)

def runInt32WellKinded : IO Bool := do
  match buildAndFinish (buildInt32Zoo {}) with
  | .error e => failGate "int32-well-kinded" s!"compile: {e}"
  | .ok plan =>
    let good := Tropical.Semantics.flatPlanWellFormed plan
    let (fns, changed) := plan.instanceFunctions.foldl (fun (acc, d) f =>
      if d then (acc.push f, d) else
        let (f', d') := relabel f
        (acc.push f', d')) (#[], false)
    let mixed := { plan with instanceFunctions := fns }
    let bad := Tropical.Semantics.flatPlanWellFormed mixed
    if good && changed && !bad then
      passGate "int32-well-kinded" "Int32Zoo's plan is well formed; the same plan with one int32 operand relabeled int is not"
    else
      failGate "int32-well-kinded" s!"well formed {good}, relabeled {changed}, mixed accepted {bad}"

end Tropical.Tropicaltest.Int32
