import Tropical.Semantics.Algebra

/-!
# Shared environment and reduction semantics

This module contains the carrier/environment operations shared by the direct
production `ExprArena` denotation and the temporarily retained recursive
reference denotation.  It has no dependency on recursive `EmitArrow.Sig`.
-/

namespace Tropical.Semantics

open Tropical.Ir

structure SigEnv (α : Type) where
  inputs : Array (Value α) := #[]
  params : Array (Value α) := #[]
  nestedOutputs : Array (Array (Value α)) := #[]
  sampleRate : Value α
  sampleIndex : Value α
  /-- Absolute tile/materializer coordinate. Ordinary exact/JIT evaluation
      supplies the same value as `sampleIndex`; staged materialization may
      supply an independent endpoint coordinate. -/
  tileSampleIndex : Value α
  /-- Unique bank binder id to its current value. `bindLoop` shadows the same
      id, although well-formed production terms forbid ancestor collisions. -/
  loops : Nat → Option (Value α) := fun _ => none

def SigEnv.bindLoop (env : SigEnv α) (id : Nat) (value : Value α) : SigEnv α :=
  { env with loops := fun query => if query = id then some value else env.loops query }

/-- Environment-level well-formedness is intentionally small: the three
    runtime rails are scalar. -/
def EnvWellFormed (env : SigEnv α) : Prop :=
  (∃ value, env.sampleRate = .scalar value) ∧
  (∃ value, env.sampleIndex = .scalar value) ∧
  (∃ value, env.tileSampleIndex = .scalar value)

/-- Interpret one flat environment reference with the production refusal text. -/
def lookupValue (operation : String) (xs : Array (Value α)) (idx : Nat) :
    Result α :=
  match xs[idx]? with
  | some value => .ok value
  | none => refusal operation s!"index {idx} is out of bounds (size {xs.size})"

/-- Interpret one nested-output reference with the production refusal text. -/
def lookupNested (env : SigEnv α) (instanceIdx outputIdx : Nat) :
    Result α :=
  match env.nestedOutputs[instanceIdx]? with
  | none =>
    refusal "nestedOut"
      s!"instance index {instanceIdx} is out of bounds (size {env.nestedOutputs.size})"
  | some outputs =>
    lookupValue "nestedOut.output" outputs outputIdx

/-- The number of iterations a bank executes. This is the exact clamp shape
    used by production `Tropical.Ir.regionTrips`. -/
def bankTrips (alg : Algebra α) (capacity : Nat) (dynCount? : Option (Result α)) :
    Except Refusal Nat := do
  match dynCount? with
  | none => pure capacity
  | some result =>
    let value ← result
    return min (← alg.dynamicCount value).toNat capacity

/-- The shared routed-fold denotation. `mapped loopValue` evaluates every emit
    position exactly once for one item; route application follows emit order. -/
def denoteRoutedSum (alg : Algebra α) (capacity outputCount fanout : Nat)
    (routes : Array (Option Nat)) (tables : Array (Result α))
    (mapped : Value α → Array (Result α))
    (dynCount? : Option (Result α)) : Result α := (do
  if capacity == 0 then
    throw { operation := "routedSum", detail := "capacity must be nonzero" }
  if outputCount == 0 then
    throw { operation := "routedSum", detail := "output count must be nonzero" }
  if fanout == 0 then
    throw { operation := "routedSum", detail := "fanout must be nonzero" }
  if routes.size != capacity * fanout then
    throw { operation := "routedSum", detail :=
      s!"route count {routes.size} does not equal capacity×fanout {capacity * fanout}" }
  if let some output := routes.findSome? fun route => match route with
      | some output => if output >= outputCount then some output else none
      | none => none then
    throw { operation := "routedSum", detail :=
      s!"route target {output} is out of bounds (output count {outputCount})" }
  discard <| sequence tables
  let trips ← bankTrips alg capacity dynCount?
  let zero ← alg.zero
  let step : Array (Value α) → Nat → Outcome (Array (Value α)) :=
    fun acc item => do
      let loopValue ← alg.loopIndex item
      let values := mapped loopValue
      if values.size != fanout then
        throw { operation := "routedSum", detail :=
          s!"mapped fanout {values.size} changed from structural fanout {fanout}" }
      let mappedValues ← sequence values
      let emitStep : Array (Value α) → Nat →
          Outcome (Array (Value α)) := fun current emit => do
        match routes[item * fanout + emit]! with
        | none => pure current
        | some output =>
          let next ← alg.binary .add current[output]! mappedValues[emit]!
          pure (current.set! output next)
      List.foldlM emitStep acc (List.range fanout)
  let result ← List.foldlM step
    (Array.replicate outputCount zero) (List.range trips)
  pure (.array result) : Outcome (Value α))

end Tropical.Semantics
