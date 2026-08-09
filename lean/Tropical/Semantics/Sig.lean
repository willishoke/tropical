import Tropical.Semantics.Algebra
import Tropical.EmitArrow.Sig

/-!
# Denotation of the production authoring syntax

This module interprets `Tropical.EmitArrow.Sig` directly.  Missing references,
unbound loop ids, invalid indexing, failed primitive operations, and invalid
dynamic counts remain explicit `Except` failures.  Banks use unique binder ids
and a scalar left fold over indices `0 .. trips-1`; no associativity or
commutativity premise appears.
-/

namespace Tropical.Semantics

open Tropical.Ir
open Tropical.EmitArrow

structure SigEnv (α : Type) where
  inputs : Array (Value α) := #[]
  params : Array (Value α) := #[]
  nestedOutputs : Array (Array (Value α)) := #[]
  sampleRate : Value α
  sampleIndex : Value α
  /-- Unique bank binder id to its current value.  `bindLoop` shadows the same
      id, although well-formed production terms forbid ancestor collisions. -/
  loops : Nat → Option (Value α) := fun _ => none

def SigEnv.bindLoop (env : SigEnv α) (id : Nat) (value : Value α) : SigEnv α :=
  { env with loops := fun query => if query = id then some value else env.loops query }

/-- Environment-level well-formedness is intentionally small: the two runtime
    rails are scalar.  Reference and binder bounds belong to `SigWellFormed`
    because they depend on the term being interpreted. -/
def EnvWellFormed (env : SigEnv α) : Prop :=
  (∃ value, env.sampleRate = .scalar value) ∧
  (∃ value, env.sampleIndex = .scalar value)

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

/-- Static well-formedness for a `Sig` under an environment and a stack of open
    bank binder ids.  Bounds are propositions; failed primitive operations
    remain dynamic refusals in the denotation. -/
def SigWellFormed (env : SigEnv α) (openBinders : List Nat := []) : Sig → Prop
  | .num _ => True
  | .binary _ lhs rhs =>
    SigWellFormed env openBinders lhs ∧ SigWellFormed env openBinders rhs
  | .unary _ arg => SigWellFormed env openBinders arg
  | .clamp value lo hi =>
    SigWellFormed env openBinders value ∧
      SigWellFormed env openBinders lo ∧ SigWellFormed env openBinders hi
  | .select cond then_ else_ =>
    SigWellFormed env openBinders cond ∧
      SigWellFormed env openBinders then_ ∧ SigWellFormed env openBinders else_
  | .inputRef idx => idx.idx < env.inputs.size
  | .paramRef idx => idx.idx < env.params.size
  | .nestedOut instanceIdx outputIdx =>
    ∃ outputs, env.nestedOutputs[instanceIdx.idx]? = some outputs ∧
      outputIdx.idx < outputs.size
  | .sampleRate | .sampleIndex => True
  | .arr items => ∀ item ∈ items, SigWellFormed env openBinders item
  | .index array index =>
    SigWellFormed env openBinders array ∧ SigWellFormed env openBinders index
  | .loopIdx id => id ∈ openBinders
  | .bankSum _ tables body dynCount? idxId =>
    idxId ∉ openBinders ∧
      (∀ table ∈ tables, SigWellFormed env openBinders table) ∧
      SigWellFormed env (idxId :: openBinders) body ∧
      (∀ count ∈ dynCount?, SigWellFormed env openBinders count)
  | .routedSum capacity outputCount routes tables values dynCount? idxId =>
    capacity > 0 ∧ outputCount > 0 ∧ values.size > 0 ∧
      routes.size = capacity * values.size ∧
      (∀ route ∈ routes, ∀ output ∈ route, output < outputCount) ∧
      idxId ∉ openBinders ∧
      (∀ table ∈ tables, SigWellFormed env openBinders table) ∧
      (∀ value ∈ values, SigWellFormed env (idxId :: openBinders) value) ∧
      (∀ count ∈ dynCount?, SigWellFormed env openBinders count)

/-- The number of iterations a bank executes.  This is the exact clamp shape
    used by production `Tropical.Ir.regionTrips`: conversion to `Int` is owned
    by the algebra, `Int.toNat` clamps the signed value below at zero, and
    `Nat.min` clamps it above at the static capacity. -/
def bankTrips (alg : Algebra α) (capacity : Nat) (dynCount? : Option (Result α)) :
    Except Refusal Nat := do
  match dynCount? with
  | none => pure capacity
  | some result =>
    let value ← result
    return min (← alg.dynamicCount value).toNat capacity

/-- The shared routed-fold denotation used by both `Sig` and `ENode`.
    `mapped loopValue` evaluates every emit position exactly once for one item;
    route application then follows emit order. -/
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

/-- Denotation of every current production `Sig` constructor.  Array literals
    and bank tables are evaluated in source order.  A bank accumulator is a
    left fold in increasing index order, with the binder installed by id. -/
def denoteSig (alg : Algebra α) (env : SigEnv α) :
    (sig : Sig) → Except Refusal (Value α)
  | .num number => alg.literal number
  | .binary tag lhs rhs =>
    applyBinary alg tag (denoteSig alg env lhs) (denoteSig alg env rhs)
  | .unary tag arg =>
    match denoteSig alg env arg with
    | .error error => .error error
    | .ok value => alg.unary tag value
  | .clamp value lo hi =>
    applyTernary alg.clamp
      (denoteSig alg env value) (denoteSig alg env lo) (denoteSig alg env hi)
  | .select cond then_ else_ =>
    applyTernary alg.select
      (denoteSig alg env cond) (denoteSig alg env then_) (denoteSig alg env else_)
  | .inputRef idx => lookupValue "inputRef" env.inputs idx.idx
  | .paramRef idx => lookupValue "paramRef" env.params idx.idx
  | .nestedOut instanceIdx outputIdx =>
    lookupNested env instanceIdx.idx outputIdx.idx
  | .sampleRate => .ok env.sampleRate
  | .sampleIndex => .ok env.sampleIndex
  | .arr items =>
    match sequence (items.map (denoteSig alg env)) with
    | .error error => .error error
    | .ok values => .ok (.array values)
  | .index array index =>
    match denoteSig alg env array with
    | .error error => .error error
    | .ok arrayValue =>
      match denoteSig alg env index with
      | .error error => .error error
      | .ok indexValue => alg.index arrayValue indexValue
  | .loopIdx id =>
    match env.loops id with
    | some value => .ok value
    | none => refusal "loopIdx" s!"binder {id} is not open"
  | .bankSum capacity tables body dynCount? idxId =>
    -- Tables are explicit eager operands in the production node.  Their values
    -- are read by the body's indexed subexpressions, but refusals here must not
    -- disappear merely because the resulting array is otherwise unused.
    match sequence (tables.map (denoteSig alg env)) with
    | .error error => .error error
    | .ok _ =>
      let dynResult := match dynCount? with
        | none => none
        | some count => some (denoteSig alg env count)
      match bankTrips alg capacity dynResult with
      | .error error => .error error
      | .ok trips =>
        match alg.zero with
        | .error error => .error error
        | .ok zero =>
          let step : Value α → Nat → Outcome (Value α) := fun acc index => do
            let loopValue ← alg.loopIndex index
            let contribution ← denoteSig alg (env.bindLoop idxId loopValue) body
            alg.binary .add acc contribution
          (List.foldlM step zero (List.range trips) : Outcome (Value α))
  | .routedSum capacity outputCount routes tables values dynCount? idxId =>
    denoteRoutedSum alg capacity outputCount values.size routes
      (tables.map (denoteSig alg env))
      (fun loopValue =>
        values.map (denoteSig alg (env.bindLoop idxId loopValue)))
      (match dynCount? with
        | none => none
        | some count => some (denoteSig alg env count))
termination_by sig => sizeOf sig
decreasing_by
  all_goals first
    | (have := Array.sizeOf_lt_of_mem ‹_ ∈ _›; simp_all; omega)
    | (simp_all; omega)

end Tropical.Semantics
