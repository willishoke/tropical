import Tropical.Semantics.Expr
import Tropical.Ir.Core
import Tropical.Ir.Strata.EArena

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
open Tropical.Ir.Strata

/-- External values for one program invocation.  `none` means that the input's
    expression default must be evaluated.  Runtime rails are inherited by
    every recursively evaluated child. -/
structure ProgramInputs (α : Type) where
  values : Array (Option (Value α)) := #[]
  sampleRate : Value α
  sampleIndex : Value α
  tileSampleIndex : Value α

abbrev ModelInput := ProgramCopyInput
abbrev ModelInstanceInput := ProgramCopyInstanceInput
abbrev ModelDecl := ProgramCopyDecl
abbrev ModelAssign := ProgramCopyAssign

/-- Common evaluator-facing shape.  Registry entries are intentionally absent:
    the recursive denotations below resolve only keys named by `.inst`. -/
abbrev ProgramModel := ProgramCopyView

def sourceProgramModel (program : Program) : ProgramModel ExprId :=
  Tropical.Ir.Strata.Program.copyView program

def coreProgramModel (program : CoreProgram) : ProgramModel ExprId :=
  Tropical.Ir.Strata.CoreProgram.copyView program

theorem sourceProgramModel_inst_mem {program : Program} {name typeKey inputs}
    (h : ProgramCopyDecl.inst name typeKey inputs ∈
      (sourceProgramModel program).decls.toList) :
    ∃ sourceInputs, BodyDecl.inst name typeKey sourceInputs ∈ program.decls := by
  simp only [sourceProgramModel, Tropical.Ir.Strata.Program.copyView,
    Array.toList_map, List.mem_map] at h
  obtain ⟨decl, hMem, hEq⟩ := h
  cases decl with
  | param sourceName value? => cases hEq
  | prog sourceName program => cases hEq
  | inst sourceName sourceTypeKey sourceInputs =>
    simp only at hEq
    cases hEq
    exact ⟨sourceInputs, by simpa using hMem⟩

inductive ListRel (rel : ρ → σ → Prop) : List ρ → List σ → Prop where
  | nil : ListRel rel [] []
  | cons : rel source dest → ListRel rel sources dests →
      ListRel rel (source :: sources) (dest :: dests)

inductive ModelInputRel (rel : ρ → σ → Prop) :
    ModelInput ρ → ModelInput σ → Prop where
  | withoutDefault (name : String) :
      ModelInputRel rel ⟨name, none⟩ ⟨name, none⟩
  | withDefault (name : String) (h : rel source dest) :
      ModelInputRel rel ⟨name, some source⟩ ⟨name, some dest⟩

structure ModelInstanceInputRel (rel : ρ → σ → Prop)
    (source : ModelInstanceInput ρ) (dest : ModelInstanceInput σ) : Prop where
  port : source.port = dest.port
  value : rel source.value dest.value

inductive ModelDeclRel (rel : ρ → σ → Prop) :
    ModelDecl ρ → ModelDecl σ → Prop where
  | param (name : String) (value? : Option JsonNumber) :
      ModelDeclRel rel (.param name value?) (.param name value?)
  | inst (name typeKey : String) (sourceInputs destInputs)
      (inputs : ListRel (ModelInstanceInputRel rel)
        sourceInputs.toList destInputs.toList) :
      ModelDeclRel rel (.inst name typeKey sourceInputs)
        (.inst name typeKey destInputs)
  | progDecl (name : String) :
      ModelDeclRel rel (.progDecl name) (.progDecl name)

structure ModelAssignRel (rel : ρ → σ → Prop)
    (source : ModelAssign ρ) (dest : ModelAssign σ) : Prop where
  target : source.target = dest.target
  expr : rel source.expr dest.expr

/-- Pointwise relation on the common evaluator-facing program shape. -/
structure ProgramModelRel (rel : ρ → σ → Prop)
    (source : ProgramModel ρ) (dest : ProgramModel σ) : Prop where
  name : source.name = dest.name
  outputCount : source.outputCount = dest.outputCount
  inputs : ListRel (ModelInputRel rel)
    source.inputs.toList dest.inputs.toList
  decls : ListRel (ModelDeclRel rel)
    source.decls.toList dest.decls.toList
  assigns : ListRel (ModelAssignRel rel)
    source.assigns.toList dest.assigns.toList

private theorem list_mapM_rel {map : ρ → Option σ} {rel : ρ → σ → Prop}
    (hMap : ∀ source dest, map source = some dest → rel source dest)
    {source : List ρ} {dest : List σ} (h : source.mapM map = some dest) :
    ListRel rel source dest := by
  induction source generalizing dest with
  | nil =>
    simp at h
    subst dest
    exact .nil
  | cons head tail ih =>
    cases hHead : map head with
    | none => simp [hHead] at h
    | some mappedHead =>
      cases hTail : tail.mapM map with
      | none => simp [hHead, hTail] at h
      | some mappedTail =>
        simp [hHead, hTail] at h
        subst dest
        exact .cons (hMap head mappedHead hHead) (ih hTail)

private theorem array_mapM_rel {map : ρ → Option σ} {rel : ρ → σ → Prop}
    (hMap : ∀ source dest, map source = some dest → rel source dest)
    {source : Array ρ} {dest : Array σ} (h : source.mapM map = some dest) :
    ListRel rel source.toList dest.toList := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hList : source.toList.mapM map with
  | none => simp [hList] at h
  | some mapped =>
    simp [hList] at h
    subst dest
    simpa using list_mapM_rel hMap hList

private theorem remapProgramCopyInput?_sound (memo : ExprCopyMemo)
    {source dest} (h : remapProgramCopyInput? memo source = some dest) :
    ModelInputRel (fun source dest => memo[source.idx]? = some dest) source dest := by
  cases source with
  | mk name default? =>
    cases default? with
    | none =>
      simp [remapProgramCopyInput?] at h
      subst dest
      exact .withoutDefault name
    | some root =>
      cases hRoot : memo[root.idx]? with
      | none => simp [remapProgramCopyInput?, remapExprId?, hRoot] at h
      | some mapped =>
        simp [remapProgramCopyInput?, remapExprId?, hRoot] at h
        subst dest
        exact .withDefault name hRoot

private theorem remapProgramCopyInstanceInput?_sound (memo : ExprCopyMemo)
    {source dest}
    (h : remapProgramCopyInstanceInput? memo source = some dest) :
    ModelInstanceInputRel (fun source dest => memo[source.idx]? = some dest)
      source dest := by
  cases source with
  | mk port value =>
    cases hValue : memo[value.idx]? with
    | none =>
      simp [remapProgramCopyInstanceInput?, remapExprId?, hValue] at h
    | some mapped =>
      simp [remapProgramCopyInstanceInput?, remapExprId?, hValue] at h
      subst dest
      exact ⟨rfl, hValue⟩

private theorem remapProgramCopyDecl?_sound (memo : ExprCopyMemo)
    {source dest} (h : remapProgramCopyDecl? memo source = some dest) :
    ModelDeclRel (fun source dest => memo[source.idx]? = some dest) source dest := by
  cases source with
  | param name value? =>
    simp [remapProgramCopyDecl?] at h
    subst dest
    exact .param name value?
  | progDecl name =>
    simp [remapProgramCopyDecl?] at h
    subst dest
    exact .progDecl name
  | inst name typeKey inputs =>
    cases hInputs : inputs.mapM (remapProgramCopyInstanceInput? memo) with
    | none => simp [remapProgramCopyDecl?, hInputs] at h
    | some mappedInputs =>
      simp [remapProgramCopyDecl?, hInputs] at h
      subst dest
      exact .inst name typeKey inputs mappedInputs <|
        array_mapM_rel (fun _ _ => remapProgramCopyInstanceInput?_sound memo) hInputs

private theorem remapProgramCopyAssign?_sound (memo : ExprCopyMemo)
    {source dest} (h : remapProgramCopyAssign? memo source = some dest) :
    ModelAssignRel (fun source dest => memo[source.idx]? = some dest)
      source dest := by
  cases source with
  | mk target expr =>
    cases hExpr : memo[expr.idx]? with
    | none => simp [remapProgramCopyAssign?, remapExprId?, hExpr] at h
    | some mapped =>
      simp [remapProgramCopyAssign?, remapExprId?, hExpr] at h
      subst dest
      exact ⟨rfl, hExpr⟩

theorem remapProgramCopyView?_sound (memo : ExprCopyMemo) {source dest}
    (h : remapProgramCopyView? memo source = some dest) :
    ProgramModelRel (fun source dest => memo[source.idx]? = some dest)
      source dest := by
  cases source with
  | mk name inputs outputCount decls assigns =>
    cases hInputs : inputs.mapM (remapProgramCopyInput? memo) with
    | none => simp [remapProgramCopyView?, hInputs] at h
    | some mappedInputs =>
      cases hDecls : decls.mapM (remapProgramCopyDecl? memo) with
      | none => simp [remapProgramCopyView?, hInputs, hDecls] at h
      | some mappedDecls =>
        cases hAssigns : assigns.mapM (remapProgramCopyAssign? memo) with
        | none => simp [remapProgramCopyView?, hInputs, hDecls, hAssigns] at h
        | some mappedAssigns =>
          simp [remapProgramCopyView?, hInputs, hDecls, hAssigns] at h
          subst dest
          exact
            { name := rfl
              outputCount := rfl
              inputs := array_mapM_rel
                (fun _ _ => remapProgramCopyInput?_sound memo) hInputs
              decls := array_mapM_rel
                (fun _ _ => remapProgramCopyDecl?_sound memo) hDecls
              assigns := array_mapM_rel
                (fun _ _ => remapProgramCopyAssign?_sound memo) hAssigns }

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
    sampleIndex := invocation.sampleIndex
    tileSampleIndex := invocation.tileSampleIndex }

private def resolveInputsGo (eval : SigEnv α → ρ → Result α)
    (invocation : ProgramInputs α) :
    Nat → List (ModelInput ρ) → Array (Value α) →
      Outcome (Array (Value α))
  | _, [], values => pure values
  | index, input :: rest, values => do
      let supplied := invocation.values[index]?.join
      let value ← match supplied with
        | some value => pure value
        | none =>
          match input.default? with
          | some root => eval { (baseEnv invocation) with inputs := values } root
          | none => programRefusal "program.input" <|
              s!"input {index} ('{input.name}') has no supplied value or default"
      resolveInputsGo eval invocation (index + 1) rest (values.push value)

private def resolveInputs (eval : SigEnv α → ρ → Result α)
    (model : ProgramModel ρ) (invocation : ProgramInputs α) :
    Outcome (Array (Value α)) :=
  resolveInputsGo eval invocation 0 model.inputs.toList #[]

private def resolveParamsGo (alg : Algebra α) :
    List (ModelDecl ρ) → Array (Value α) → Outcome (Array (Value α))
  | [], values => pure values
  | decl :: rest, values => do
    let values ← match decl with
      | .param name (some value) => values.push <$> alg.literal value
      | .param name none =>
          programRefusal "program.param" s!"parameter '{name}' has no value"
      | .inst .. | .progDecl .. => pure values
    resolveParamsGo alg rest values

private def resolveParams (alg : Algebra α) (model : ProgramModel ρ) :
    Outcome (Array (Value α)) :=
  resolveParamsGo alg model.decls.toList #[]

private def sparseInstanceInputSize (inputs : List (ModelInstanceInput ρ)) : Nat :=
  inputs.foldl (fun bound input => max bound (input.port.idx + 1)) 0

private def sparseInstanceInputsGo (eval : SigEnv α → ρ → Result α)
    (env : SigEnv α) : List (ModelInstanceInput ρ) →
      Array (Option (Value α)) → Outcome (Array (Option (Value α)))
  | [], values => pure values
  | input :: rest, values => do
      let value ← eval env input.value
      sparseInstanceInputsGo eval env rest
        (values.set! input.port.idx (some value))

private def sparseInstanceInputs (eval : SigEnv α → ρ → Result α)
    (env : SigEnv α) (inputs : Array (ModelInstanceInput ρ)) :
    Outcome (Array (Option (Value α))) := do
  sparseInstanceInputsGo eval env inputs.toList
    (Array.replicate (sparseInstanceInputSize inputs.toList) none)

private def completeChildOutputs (instanceName : String)
    (outputs : Array (Option (Value α))) : Outcome (Array (Value α)) :=
  outputs.foldlM (fun values output =>
    match output with
    | some value => pure (values.push value)
    | none => programRefusal "program.instance.output"
        s!"instance '{instanceName}' left a referenced output unassigned") #[]

private def resolveInstancesGo (eval : SigEnv α → ρ → Result α)
    (invocation : ProgramInputs α)
    (inputValues paramValues : Array (Value α))
    (child : String → ProgramInputs α → Outcome (ProgramObservation α)) :
    List (ModelDecl ρ) →
    (Array (InstanceObservation α) × Array (Array (Value α))) →
      Outcome (Array (InstanceObservation α) × Array (Array (Value α)))
  | [], state => pure state
  | decl :: rest, state =>
      match decl with
      | .inst name typeKey instanceInputs =>
        let env : SigEnv α :=
          { (baseEnv invocation) with
            inputs := inputValues
            params := paramValues
            nestedOutputs := state.2 }
        match sparseInstanceInputs eval env instanceInputs with
        | .error refusal => Except.error refusal
        | .ok supplied =>
          match child typeKey
              { values := supplied
                sampleRate := invocation.sampleRate
                sampleIndex := invocation.sampleIndex
                tileSampleIndex := invocation.tileSampleIndex } with
          | .error refusal => Except.error refusal
          | .ok observation =>
            match completeChildOutputs name observation.portOutputs with
            | .error refusal => Except.error refusal
            | .ok outputs =>
              resolveInstancesGo eval invocation inputValues paramValues child rest
                (state.1.push { name, typeKey, outputs }, state.2.push outputs)
      | .param .. | .progDecl .. =>
          resolveInstancesGo eval invocation inputValues paramValues child rest state

private def resolveInstances (eval : SigEnv α → ρ → Result α)
    (model : ProgramModel ρ) (invocation : ProgramInputs α)
    (inputValues paramValues : Array (Value α))
    (child : String → ProgramInputs α → Outcome (ProgramObservation α)) :
    Outcome (Array (InstanceObservation α) × Array (Array (Value α))) :=
  resolveInstancesGo eval invocation inputValues paramValues child
    model.decls.toList (#[], #[])

private def resolveAssignsGo (eval : SigEnv α → ρ → Result α)
    (outputCount : Nat) (env : SigEnv α) : List (ModelAssign ρ) →
    (Array (OutputTarget × Value α) × Array (Option (Value α))) →
      Outcome (Array (OutputTarget × Value α) × Array (Option (Value α)))
  | [], state => pure state
  | assign :: rest, state =>
      match eval env assign.expr with
      | .error refusal => .error refusal
      | .ok value =>
        match assign.target with
        | .dac => resolveAssignsGo eval outputCount env rest
            (state.1.push (assign.target, value), state.2)
        | .port output =>
          if output.idx < outputCount then
            resolveAssignsGo eval outputCount env rest
              (state.1.push (assign.target, value),
                state.2.set! output.idx (some value))
          else
            programRefusal "program.output"
              s!"output target {output.idx} is out of bounds (size {outputCount})"

private def resolveAssigns (eval : SigEnv α → ρ → Result α)
    (model : ProgramModel ρ) (env : SigEnv α) :
    Outcome (Array (OutputTarget × Value α) × Array (Option (Value α))) :=
  resolveAssignsGo eval model.outputCount env model.assigns.toList
    (#[], Array.replicate model.outputCount none)

private theorem resolveInputsGo_rel
    {rel : ρ → σ → Prop}
    {evalSource : SigEnv α → ρ → Result α}
    {evalDest : SigEnv α → σ → Result α}
    (hEval : ∀ env source dest, rel source dest →
      evalSource env source = evalDest env dest)
    (invocation : ProgramInputs α) {source dest}
    (hInputs : ListRel (ModelInputRel rel) source dest)
    (index : Nat) (values : Array (Value α)) :
    resolveInputsGo evalSource invocation index source values =
      resolveInputsGo evalDest invocation index dest values := by
  induction hInputs generalizing index values with
  | nil => rfl
  | @cons sourceHead destHead sourceTail destTail hInput _ ih =>
    cases hInput with
    | withoutDefault name =>
      simp only [resolveInputsGo]
      split <;> simp [ih]
    | withDefault name hRoot =>
      simp only [resolveInputsGo]
      split
      next => simp [ih]
      next =>
        rw [hEval _ _ _ hRoot]
        generalize evalDest _ _ = result
        cases result <;> simp [ih]

private theorem resolveParamsGo_rel {rel : ρ → σ → Prop}
    (alg : Algebra α) {source dest}
    (hDecls : ListRel (ModelDeclRel rel) source dest)
    (values : Array (Value α)) :
    resolveParamsGo alg source values = resolveParamsGo alg dest values := by
  induction hDecls generalizing values with
  | nil => rfl
  | @cons sourceHead destHead sourceTail destTail hDecl _ ih =>
    cases hDecl with
    | param name value? =>
      cases value? with
      | none => rfl
      | some value =>
        generalize alg.literal value = result
        cases result <;> simp [resolveParamsGo, ih]
    | inst => simp [resolveParamsGo, ih]
    | progDecl => simp [resolveParamsGo, ih]

private theorem sparseInstanceInputSize_rel {rel : ρ → σ → Prop}
    {source dest}
    (hInputs : ListRel (ModelInstanceInputRel rel) source dest) :
    sparseInstanceInputSize source = sparseInstanceInputSize dest := by
  have go : ∀ initial,
      source.foldl (fun bound input => max bound (input.port.idx + 1)) initial =
        dest.foldl (fun bound input => max bound (input.port.idx + 1)) initial := by
    intro initial
    induction hInputs generalizing initial with
    | nil => rfl
    | cons hInput _ ih =>
      cases hInput with
      | mk hPort hValue =>
        rw [List.foldl_cons, List.foldl_cons, hPort]
        exact ih _
  exact go 0

private theorem sparseInstanceInputsGo_rel
    {rel : ρ → σ → Prop}
    {evalSource : SigEnv α → ρ → Result α}
    {evalDest : SigEnv α → σ → Result α}
    (hEval : ∀ env source dest, rel source dest →
      evalSource env source = evalDest env dest)
    (env : SigEnv α) {source dest}
    (hInputs : ListRel (ModelInstanceInputRel rel) source dest)
    (values : Array (Option (Value α))) :
    sparseInstanceInputsGo evalSource env source values =
      sparseInstanceInputsGo evalDest env dest values := by
  induction hInputs generalizing values with
  | nil => rfl
  | @cons sourceHead destHead sourceTail destTail hInput _ ih =>
    cases hInput with
    | mk hPort hValue =>
      simp only [sparseInstanceInputsGo]
      rw [hEval _ _ _ hValue, hPort]
      generalize evalDest env destHead.value = result
      cases result with
      | error refusal => rfl
      | ok value => exact ih _

private theorem sparseInstanceInputs_rel
    {rel : ρ → σ → Prop}
    {evalSource : SigEnv α → ρ → Result α}
    {evalDest : SigEnv α → σ → Result α}
    (hEval : ∀ env source dest, rel source dest →
      evalSource env source = evalDest env dest)
    (env : SigEnv α) {source dest : Array (ModelInstanceInput _)}
    (hInputs : ListRel (ModelInstanceInputRel rel)
      source.toList dest.toList) :
    sparseInstanceInputs evalSource env source =
      sparseInstanceInputs evalDest env dest := by
  unfold sparseInstanceInputs
  rw [sparseInstanceInputSize_rel hInputs]
  exact sparseInstanceInputsGo_rel hEval env hInputs _

private theorem resolveInstancesGo_rel
    {rel : ρ → σ → Prop}
    {evalSource : SigEnv α → ρ → Result α}
    {evalDest : SigEnv α → σ → Result α}
    (hEval : ∀ env source dest, rel source dest →
      evalSource env source = evalDest env dest)
    (invocation : ProgramInputs α)
    (inputValues paramValues : Array (Value α))
    {childSource childDest :
      String → ProgramInputs α → Outcome (ProgramObservation α)}
    {source dest}
    (hChild : ∀ key inputs,
      (∃ name instanceInputs, ProgramCopyDecl.inst name key instanceInputs ∈ source) →
      childSource key inputs = childDest key inputs)
    (hDecls : ListRel (ModelDeclRel rel) source dest)
    (state : Array (InstanceObservation α) × Array (Array (Value α))) :
    resolveInstancesGo evalSource invocation inputValues paramValues
        childSource source state =
      resolveInstancesGo evalDest invocation inputValues paramValues
        childDest dest state := by
  induction hDecls generalizing state with
  | nil => rfl
  | @cons sourceHead destHead sourceTail destTail hDecl _ ih =>
    have hChildTail : ∀ key inputs,
        (∃ name instanceInputs,
          ProgramCopyDecl.inst name key instanceInputs ∈ sourceTail) →
        childSource key inputs = childDest key inputs := by
      intro key childInputs hUse
      apply hChild key childInputs
      obtain ⟨name, instanceInputs, hMem⟩ := hUse
      exact ⟨name, instanceInputs, by simp [hMem]⟩
    cases hDecl with
    | param name value? =>
      simp only [resolveInstancesGo]
      exact ih hChildTail state
    | progDecl name =>
      simp only [resolveInstancesGo]
      exact ih hChildTail state
    | inst name typeKey sourceInputs destInputs hInputs =>
      have hChildKey : childSource typeKey = childDest typeKey := by
        funext childInputs
        apply hChild typeKey childInputs
        exact ⟨name, sourceInputs, by simp⟩
      simp only [resolveInstancesGo]
      rw [sparseInstanceInputs_rel hEval _ hInputs]
      rw [hChildKey]
      cases hSupplied : sparseInstanceInputs evalDest _ destInputs with
      | error refusal =>
        change (Except.error refusal : Outcome _) = Except.error refusal
        rfl
      | ok supplied =>
        change
          (match childDest typeKey
              { values := supplied
                sampleRate := invocation.sampleRate
                sampleIndex := invocation.sampleIndex
                tileSampleIndex := invocation.tileSampleIndex } with
          | .error refusal => Except.error refusal
          | .ok observation =>
            match completeChildOutputs name observation.portOutputs with
            | .error refusal => Except.error refusal
            | .ok outputs =>
              resolveInstancesGo evalSource invocation inputValues paramValues
                childSource sourceTail
                (state.1.push { name, typeKey, outputs }, state.2.push outputs)) =
          (match childDest typeKey
              { values := supplied
                sampleRate := invocation.sampleRate
                sampleIndex := invocation.sampleIndex
                tileSampleIndex := invocation.tileSampleIndex } with
          | .error refusal => Except.error refusal
          | .ok observation =>
            match completeChildOutputs name observation.portOutputs with
            | .error refusal => Except.error refusal
            | .ok outputs =>
              resolveInstancesGo evalDest invocation inputValues paramValues
                childDest destTail
                (state.1.push { name, typeKey, outputs }, state.2.push outputs))
        cases hChildResult : childDest typeKey
            { values := supplied
              sampleRate := invocation.sampleRate
              sampleIndex := invocation.sampleIndex
              tileSampleIndex := invocation.tileSampleIndex } with
        | error refusal =>
          change (Except.error refusal : Outcome _) = Except.error refusal
          rfl
        | ok observation =>
          change
            (match completeChildOutputs name observation.portOutputs with
            | .error refusal => Except.error refusal
            | .ok outputs =>
              resolveInstancesGo evalSource invocation inputValues paramValues
                childSource sourceTail
                (state.1.push { name, typeKey, outputs }, state.2.push outputs)) =
            (match completeChildOutputs name observation.portOutputs with
            | .error refusal => Except.error refusal
            | .ok outputs =>
              resolveInstancesGo evalDest invocation inputValues paramValues
                childDest destTail
                (state.1.push { name, typeKey, outputs }, state.2.push outputs))
          cases hOutputs : completeChildOutputs name observation.portOutputs with
          | error refusal =>
            change (Except.error refusal : Outcome _) = Except.error refusal
            rfl
          | ok outputs =>
            change resolveInstancesGo evalSource invocation inputValues paramValues
              childSource sourceTail
                (state.1.push { name, typeKey, outputs }, state.2.push outputs) =
              resolveInstancesGo evalDest invocation inputValues paramValues
                childDest destTail
                  (state.1.push { name, typeKey, outputs }, state.2.push outputs)
            exact ih hChildTail _

private theorem resolveAssignsGo_rel
    {rel : ρ → σ → Prop}
    {evalSource : SigEnv α → ρ → Result α}
    {evalDest : SigEnv α → σ → Result α}
    (hEval : ∀ env source dest, rel source dest →
      evalSource env source = evalDest env dest)
    (outputCount : Nat) (env : SigEnv α) {source dest}
    (hAssigns : ListRel (ModelAssignRel rel) source dest)
    (state : Array (OutputTarget × Value α) × Array (Option (Value α))) :
    resolveAssignsGo evalSource outputCount env source state =
      resolveAssignsGo evalDest outputCount env dest state := by
  induction hAssigns generalizing state with
  | nil => rfl
  | @cons sourceHead destHead sourceTail destTail hAssign _ ih =>
    cases hAssign with
    | mk hTarget hExpr =>
      cases sourceHead with
      | mk sourceTarget sourceExpr =>
        cases destHead with
        | mk destTarget destExpr =>
          simp only at hTarget hExpr ⊢
          subst destTarget
          simp only [resolveAssignsGo]
          rw [hEval _ _ _ hExpr]
          cases hResult : evalDest env destExpr with
          | error refusal =>
            change (Except.error refusal : Outcome _) = Except.error refusal
            rfl
          | ok value =>
            change
              (match sourceTarget with
              | .dac => resolveAssignsGo evalSource outputCount env sourceTail
                  (state.1.push (sourceTarget, value), state.2)
              | .port output =>
                if output.idx < outputCount then
                  resolveAssignsGo evalSource outputCount env sourceTail
                    (state.1.push (sourceTarget, value),
                      state.2.set! output.idx (some value))
                else programRefusal "program.output"
                  s!"output target {output.idx} is out of bounds (size {outputCount})") =
              (match sourceTarget with
              | .dac => resolveAssignsGo evalDest outputCount env destTail
                  (state.1.push (sourceTarget, value), state.2)
              | .port output =>
                if output.idx < outputCount then
                  resolveAssignsGo evalDest outputCount env destTail
                    (state.1.push (sourceTarget, value),
                      state.2.set! output.idx (some value))
                else programRefusal "program.output"
                  s!"output target {output.idx} is out of bounds (size {outputCount})")
            cases sourceTarget with
            | dac =>
              change resolveAssignsGo evalSource outputCount env sourceTail
                (state.1.push (.dac, value), state.2) =
                resolveAssignsGo evalDest outputCount env destTail
                  (state.1.push (.dac, value), state.2)
              exact ih _
            | port output =>
              change (if output.idx < outputCount then
                resolveAssignsGo evalSource outputCount env sourceTail
                  (state.1.push (.port output, value),
                    state.2.set! output.idx (some value))
                else programRefusal "program.output"
                  s!"output target {output.idx} is out of bounds (size {outputCount})") =
              (if output.idx < outputCount then
                resolveAssignsGo evalDest outputCount env destTail
                  (state.1.push (.port output, value),
                    state.2.set! output.idx (some value))
                else programRefusal "program.output"
                  s!"output target {output.idx} is out of bounds (size {outputCount})")
              split
              next hlt => exact ih _
              next hnot => rfl

/-- Evaluate one representation-neutral program node.  `child` is invoked only
    for keys named by instance declarations, making unreachable registry data
    semantically invisible. -/
def denoteProgramModel (alg : Algebra α) (eval : SigEnv α → ρ → Result α)
    (model : ProgramModel ρ) (invocation : ProgramInputs α)
    (child : String → ProgramInputs α → Outcome (ProgramObservation α)) :
    Outcome (ProgramObservation α) :=
  match resolveInputs eval model invocation with
  | .error refusal => .error refusal
  | .ok inputValues =>
    match resolveParams alg model with
    | .error refusal => .error refusal
    | .ok paramValues =>
      match resolveInstances eval model invocation inputValues paramValues child with
      | .error refusal => .error refusal
      | .ok (instances, nestedOutputs) =>
        let env : SigEnv α :=
          { (baseEnv invocation) with
            inputs := inputValues
            params := paramValues
            nestedOutputs }
        match resolveAssigns eval model env with
        | .error refusal => .error refusal
        | .ok (assigns, portOutputs) =>
          pure (ProgramObservation.mk model.name inputValues paramValues
            instances assigns portOutputs)

/-- Representation parametricity of the evaluator-facing program model.
    Refusals and successful observations are preserved pointwise. -/
theorem denoteProgramModel_rel {rel : ρ → σ → Prop}
    (alg : Algebra α)
    (evalSource : SigEnv α → ρ → Result α)
    (evalDest : SigEnv α → σ → Result α)
    {source : ProgramModel ρ} {dest : ProgramModel σ}
    (hModel : ProgramModelRel rel source dest)
    (invocation : ProgramInputs α)
    (childSource childDest :
      String → ProgramInputs α → Outcome (ProgramObservation α))
    (hEval : ∀ env source dest, rel source dest →
      evalSource env source = evalDest env dest)
    (hChild : ∀ key inputs,
      (∃ name instanceInputs,
        ProgramCopyDecl.inst name key instanceInputs ∈ source.decls.toList) →
      childSource key inputs = childDest key inputs) :
    denoteProgramModel alg evalSource source invocation childSource =
      denoteProgramModel alg evalDest dest invocation childDest := by
  cases hModel with
  | mk hName hOutputCount hInputs hDecls hAssigns =>
    unfold denoteProgramModel resolveInputs resolveParams resolveInstances resolveAssigns
    rw [resolveInputsGo_rel hEval invocation hInputs]
    cases hInputValues : resolveInputsGo evalDest invocation 0 dest.inputs.toList #[] with
    | error refusal =>
      dsimp only
    | ok inputValues =>
      dsimp only
      change
        (match resolveParamsGo alg source.decls.toList #[] with
        | .error refusal => Except.error refusal
        | .ok paramValues => _) = _
      rw [resolveParamsGo_rel alg hDecls]
      cases hParamValues : resolveParamsGo alg dest.decls.toList #[] with
      | error refusal =>
        dsimp only
      | ok paramValues =>
        dsimp only
        rw [resolveInstancesGo_rel hEval invocation inputValues paramValues
          hChild hDecls (#[], #[])]
        cases hInstances : resolveInstancesGo evalDest invocation inputValues paramValues
            childDest dest.decls.toList (#[], #[]) with
        | error refusal =>
          dsimp only
        | ok result =>
          dsimp only
          obtain ⟨instances, nestedOutputs⟩ := result
          let env : SigEnv α :=
            { (baseEnv invocation) with
              inputs := inputValues
              params := paramValues
              nestedOutputs }
          rw [hOutputCount]
          rw [resolveAssignsGo_rel hEval dest.outputCount env hAssigns
            (#[], Array.replicate dest.outputCount none)]
          cases hAssignValues : resolveAssignsGo evalDest dest.outputCount env
              dest.assigns.toList (#[], Array.replicate dest.outputCount none) with
          | error refusal =>
            dsimp only
          | ok result =>
            dsimp only
            obtain ⟨assigns, portOutputs⟩ := result
            rw [hName]

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
