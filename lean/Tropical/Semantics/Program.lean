import Tropical.Semantics.Expr
import Tropical.Ir.Core

/-!
# Evaluator-reachable program semantics

`Program` and `CoreProgram` differ in representation, not in the structure an
evaluator observes.  Both normalize to `ProgramModel`; registry lookup remains
outside the model because unreachable registry entries are deliberately inert.

Evaluation is authored-order and refusal-preserving: inputs (supplied value or
default), parameters, instances, then output assignments.  An instance input
expression is evaluated in the parent environment containing only already
evaluated child outputs, matching `nestedOut`'s positional contract.
-/

namespace Tropical.Semantics

open Lean (JsonNumber)
open Tropical.Ir
open Tropical.Ir.Core

/-- External values for one program invocation.  `none` means that the input's
    expression default must be evaluated.  Runtime rails are inherited by
    every recursively evaluated child. -/
structure ProgramInputs (α : Type) where
  values : Array (Option (Value α)) := #[]
  sampleRate : Value α
  sampleIndex : Value α

structure ModelInput (ρ : Type) where
  name : String
  default? : Option ρ := none

structure ModelInstanceInput (ρ : Type) where
  port : InputIdx
  value : ρ

inductive ModelDecl (ρ : Type) where
  | param (name : String) (value? : Option JsonNumber)
  | inst (name typeKey : String) (inputs : Array (ModelInstanceInput ρ))
  | progDecl (name : String)

structure ModelAssign (ρ : Type) where
  target : OutputTarget
  expr : ρ

/-- Common evaluator-facing shape.  Registry entries are intentionally absent:
    the recursive denotations below resolve only keys named by `.inst`. -/
structure ProgramModel (ρ : Type) where
  name : String
  inputs : Array (ModelInput ρ)
  outputCount : Nat
  decls : Array (ModelDecl ρ)
  assigns : Array (ModelAssign ρ)

def sourceProgramModel (program : Program) : ProgramModel ExprId :=
  { name := program.name
    inputs := program.inputs.map fun input =>
      { name := input.name, default? := input.default? }
    outputCount := program.outputs.size
    decls := program.decls.map fun
      | .param name value? => .param name value?
      | .inst name typeKey inputs => .inst name typeKey <|
          inputs.map fun input => { port := input.port, value := input.value }
      | .prog name _ => .progDecl name
    assigns := program.assigns.map fun assign =>
      { target := assign.target, expr := assign.expr } }

def coreProgramModel (program : CoreProgram) : ProgramModel ExprId :=
  { name := program.name
    inputs := program.inputs.map fun input =>
      { name := input.name, default? := input.default? }
    outputCount := program.outputs.size
    decls := program.decls.map fun
      | .param name value? => .param name value?
      | .inst name typeKey inputs => .inst name typeKey <|
          inputs.map fun input => { port := input.port, value := input.value }
      | .progDecl name => .progDecl name
    assigns := program.assigns.map fun assign =>
      { target := assign.target, expr := assign.expr } }

structure InstanceObservation (α : Type) where
  name : String
  typeKey : String
  outputs : Array (Value α)
deriving Repr

/-- Observable result of one reachable program evaluation.  `assigns` retains
    authored order (including DAC targets); `portOutputs` is the last-write
    image used to interpret a parent's `nestedOut`. -/
structure ProgramObservation (α : Type) where
  name : String
  inputs : Array (Value α)
  params : Array (Value α)
  instances : Array (InstanceObservation α)
  assigns : Array (OutputTarget × Value α)
  portOutputs : Array (Option (Value α))
deriving Repr

private def programRefusal (operation detail : String) : Outcome β :=
  .error { operation, detail }

private def baseEnv (invocation : ProgramInputs α) : SigEnv α :=
  { sampleRate := invocation.sampleRate
    sampleIndex := invocation.sampleIndex }

private def resolveInputs (eval : SigEnv α → ρ → Result α)
    (model : ProgramModel ρ) (invocation : ProgramInputs α) :
    Outcome (Array (Value α)) :=
  List.foldlM (fun values index => do
      let supplied := invocation.values[index]?.join
      let some input := model.inputs[index]?
        | programRefusal "program.input"
            s!"input declaration {index} disappeared during evaluation"
      let value ← match supplied with
        | some value => pure value
        | none =>
          match input.default? with
          | some root => eval { (baseEnv invocation) with inputs := values } root
          | none => programRefusal "program.input" <|
              s!"input {index} ('{input.name}') has no supplied value or default"
      pure (values.push value))
    #[] (List.range model.inputs.size)

private def resolveParams (alg : Algebra α) (model : ProgramModel ρ) :
    Outcome (Array (Value α)) :=
  model.decls.foldlM (fun values decl =>
    match decl with
    | .param name (some value) => values.push <$> alg.literal value
    | .param name none =>
        programRefusal "program.param" s!"parameter '{name}' has no value"
    | .inst .. | .progDecl .. => pure values) #[]

private def sparseInstanceInputs (eval : SigEnv α → ρ → Result α)
    (env : SigEnv α) (inputs : Array (ModelInstanceInput ρ)) :
    Outcome (Array (Option (Value α))) := do
  let size := inputs.foldl (fun bound input => max bound (input.port.idx + 1)) 0
  inputs.foldlM (fun values input => do
      pure (values.set! input.port.idx (some (← eval env input.value))))
    (Array.replicate size none)

private def completeChildOutputs (instanceName : String)
    (outputs : Array (Option (Value α))) : Outcome (Array (Value α)) :=
  outputs.foldlM (fun values output =>
    match output with
    | some value => pure (values.push value)
    | none => programRefusal "program.instance.output"
        s!"instance '{instanceName}' left a referenced output unassigned") #[]

private def resolveInstances (eval : SigEnv α → ρ → Result α)
    (model : ProgramModel ρ) (invocation : ProgramInputs α)
    (inputValues paramValues : Array (Value α))
    (child : String → ProgramInputs α → Outcome (ProgramObservation α)) :
    Outcome (Array (InstanceObservation α) × Array (Array (Value α))) :=
  model.decls.foldlM (fun state decl => do
      match decl with
      | .inst name typeKey instanceInputs =>
        let env : SigEnv α :=
          { (baseEnv invocation) with
            inputs := inputValues
            params := paramValues
            nestedOutputs := state.2 }
        let supplied ← sparseInstanceInputs eval env instanceInputs
        let observation ← child typeKey
          { values := supplied
            sampleRate := invocation.sampleRate
            sampleIndex := invocation.sampleIndex }
        let outputs ← completeChildOutputs name observation.portOutputs
        pure (state.1.push { name, typeKey, outputs }, state.2.push outputs)
      | .param .. | .progDecl .. => pure state)
    (#[], #[])

private def resolveAssigns (eval : SigEnv α → ρ → Result α)
    (model : ProgramModel ρ) (env : SigEnv α) :
    Outcome (Array (OutputTarget × Value α) × Array (Option (Value α))) :=
  model.assigns.foldlM (fun state assign => do
      let value ← eval env assign.expr
      let portOutputs ← match assign.target with
        | .dac => pure state.2
        | .port output =>
          if output.idx < model.outputCount then
            pure (state.2.set! output.idx (some value))
          else
            programRefusal "program.output"
              s!"output target {output.idx} is out of bounds (size {model.outputCount})"
      pure (state.1.push (assign.target, value), portOutputs))
    (#[], Array.replicate model.outputCount none)

/-- Evaluate one representation-neutral program node.  `child` is invoked only
    for keys named by instance declarations, making unreachable registry data
    semantically invisible. -/
def denoteProgramModel (alg : Algebra α) (eval : SigEnv α → ρ → Result α)
    (model : ProgramModel ρ) (invocation : ProgramInputs α)
    (child : String → ProgramInputs α → Outcome (ProgramObservation α)) :
    Outcome (ProgramObservation α) := do
  let inputValues ← resolveInputs eval model invocation
  let paramValues ← resolveParams alg model
  let (instances, nestedOutputs) ←
    resolveInstances eval model invocation inputValues paramValues child
  let env : SigEnv α :=
    { (baseEnv invocation) with
      inputs := inputValues
      params := paramValues
      nestedOutputs }
  let (assigns, portOutputs) ← resolveAssigns eval model env
  pure (ProgramObservation.mk model.name inputValues paramValues
    instances assigns portOutputs)

/-- Denotation of a pooled source program.  Pool recursion follows only
    instance registry hits and terminates by `progPoolWf`. -/
def denoteProgram (alg : Algebra α) (arena : Arena)
    (hExpr : ArenaWellFormed arena.exprs)
    (hPrograms : progPoolWf arena.programs = true)
    (root : ProgramIdx) (invocation : ProgramInputs α) :
    Outcome (ProgramObservation α) :=
  match hp : arena.programs[root.idx]? with
  | none => programRefusal "program" s!"pool index {root.idx} is out of bounds"
  | some program =>
    denoteProgramModel alg
      (fun env id => denoteExpr alg env arena.exprs hExpr id)
      (sourceProgramModel program) invocation fun key childInputs =>
        match hr : program.registryGet? key with
        | none => programRefusal "program.registry"
            s!"instance type key '{key}' is missing in program '{program.name}'"
        | some child => denoteProgram alg arena hExpr hPrograms child childInputs
termination_by root.idx
decreasing_by exact progPool_registry_lt hPrograms hp hr

/-- Denotation of a recursive post-strata core program. -/
def denoteCoreProgram (alg : Algebra α) (arena : ExprArena)
    (hExpr : ArenaWellFormed arena) (program : CoreProgram)
    (invocation : ProgramInputs α) : Outcome (ProgramObservation α) :=
  denoteProgramModel alg
    (fun env id => denoteExpr alg env arena hExpr id)
    (coreProgramModel program) invocation fun key childInputs =>
      match hr : program.registryGet? key with
      | none => programRefusal "program.registry"
          s!"instance type key '{key}' is missing in program '{program.name}'"
      | some child => denoteCoreProgram alg arena hExpr child childInputs
termination_by sizeOf program
decreasing_by exact CoreProgram.sizeOf_lt_of_registryGet? hr

end Tropical.Semantics
