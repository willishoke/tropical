import Lean.Data.Json

/-!
# Structural JSON comparison

The Lean half of the differential harness's comparator (the TS half is
`scripts/diff/structural.ts`). Two pipelines may serialize the same
plan with different key order or float formatting, so JSON is never
compared as text: objects compare by key set, arrays element-wise, and
numbers as IEEE-754 doubles by bit pattern (`Float.toBits`), which
distinguishes +0/−0 and is immune to formatting.

Canonical float *emission* (the other half of the float story) lands
with `Tropical.Plan` in Phase 2 — the C++ NumericProgramParser parses
doubles, so emission only has to round-trip, not match TS textually.
-/

namespace Tropical.Json

open Lean (Json JsonNumber)

structure DiffEntry where
  path : String
  a : Json
  b : Json
deriving Repr

instance : ToString DiffEntry where
  toString d := s!"{d.path}\n  a: {d.a.compress}\n  b: {d.b.compress}"

/-- Bit-identical as IEEE-754 doubles. -/
private def numEq (x y : JsonNumber) : Bool :=
  x.toFloat.toBits == y.toFloat.toBits

private def maxDiffs : Nat := 20

/-- Collect up to `maxDiffs` structural divergences between two JSON values. -/
partial def diff (a b : Json) (path : String := "$")
    (acc : Array DiffEntry := #[]) : Array DiffEntry :=
  if acc.size ≥ maxDiffs then acc else
  match a, b with
  | .num x, .num y =>
    if numEq x y then acc else acc.push ⟨path, a, b⟩
  | .arr xs, .arr ys =>
    if xs.size ≠ ys.size then
      acc.push ⟨s!"{path}.length", Lean.toJson xs.size, Lean.toJson ys.size⟩
    else Id.run do
      let mut acc := acc
      for i in [0:xs.size] do
        acc := diff xs[i]! ys[i]! s!"{path}[{i}]" acc
      pure acc
  | .obj ma, .obj mb => Id.run do
    let keysA := ma.toList.map (·.1)
    let keysB := mb.toList.map (·.1)
    let keys  := (keysA ++ keysB).eraseDups.mergeSort (· ≤ ·)
    let mut acc := acc
    for k in keys do
      if acc.size ≥ maxDiffs then return acc
      match a.getObjVal? k, b.getObjVal? k with
      | .ok va, .ok vb    => acc := diff va vb s!"{path}.{k}" acc
      | .ok va, .error _  => acc := acc.push ⟨s!"{path}.{k}", va, Json.str "<absent>"⟩
      | .error _, .ok vb  => acc := acc.push ⟨s!"{path}.{k}", Json.str "<absent>", vb⟩
      | .error _, .error _ => pure ()
    pure acc
  | _, _ =>
    -- null / bool / string / mixed kinds: compressed text is canonical here
    if a.compress == b.compress then acc else acc.push ⟨path, a, b⟩

/-- True when the two values are structurally identical. -/
def eqv (a b : Json) : Bool := (diff a b).isEmpty

end Tropical.Json
