import Tropical.EmitArrow.ArenaModal.Oriented

/-!
# Factor-preserving modal kernels

This is the compiler meaning of feed-forward modal topology.  Direct paths are
represented explicitly as convolution identities; they are not confused with
`Oriented.Bank.atZero`, which is only the value of a two-sided expansion at one
coordinate.  Parallel and cascade retain authored order and remain factored
until a terminal chooses a realization.
-/

namespace Tropical.EmitArrow.ArenaNative

open Tropical.Ir

/-- A live modal control retains its authored fallback and, when patched, the
    signal node that supersedes it.  Binding happens once the terminal response
    coordinate is known. -/
structure ModalControlRef where
  fallback : ArrowTerm
  signalNode? : Option String := none

def ModalControlRef.constant (value : Sig) : ModalControlRef :=
  { fallback := .konst value }

/-- Proper (no-Dirac) kernel atoms currently understood by the oriented bank
    oracle.  `oriented` is the general room/filter atom.  `causalTail` records
    the structural first-order tail needed to recover an all-pass product
    without inspecting or comparing floating expressions. -/
inductive ModalProperKernel where
  | oriented (modes : Array ModalMode) (direction : Sig)
  | causalTail (tail : ModalMode)

/-- A resolved current-universe linear modal kernel.  It is compiler-private
    authoring structure, not an executable-trunk or backend opcode. -/
inductive ModalKernelExpr where
  | identity
  | proper (kernel : ModalProperKernel)
  | scale (value : Sig) (kernel : ModalKernelExpr)
  | parallel (kernels : Array ModalKernelExpr)
  | cascade (kernels : Array ModalKernelExpr)
  | blend (mix : Sig) (dry wet : ModalKernelExpr)

instance : Inhabited ModalKernelExpr := ⟨.identity⟩

/-- One deferred linear stage.  `build` sees one frozen terminal universe and
    returns only generic kernel topology; effect names do not survive here. -/
structure ModalLinearStage where
  controls : Array ModalControlRef := #[]
  build : Sig → Array Sig → BuildM ModalKernelExpr

private def ModalProperKernel.applyGeneric (kernel : ModalProperKernel)
    (input : Oriented.Bank) : BuildM Oriented.Bank :=
  match kernel with
  | .oriented modes direction =>
      input.convolveKernel modes direction Oriented.syntacticSameSideClassifier
  | .causalTail tail =>
      do
        let zero ← lit 0
        input.convolveKernel #[tail] zero Oriented.syntacticSameSideClassifier

/-- Literal reference interpretation.  This may expand parallel products and
    is therefore reserved for small or oracle shapes; production terminals
    inspect the retained factors first. -/
def ModalKernelExpr.applyGeneric (input : Oriented.Bank) : ModalKernelExpr → BuildM Oriented.Bank
  | .identity => pure input
  | .proper kernel => kernel.applyGeneric input
  | .scale value kernel => do
      let bank ← kernel.applyGeneric input
      let zero ← lit 0
      bank.scale (value, zero)
  | .parallel kernels => do
      let zeroBank ← Oriented.Bank.ofFuture #[]
      kernels.attach.foldlM (fun total kernel => do
        let branch ← kernel.1.applyGeneric input
        total.add branch) zeroBank
  | .cascade kernels => do
      kernels.attach.foldlM (fun value kernel =>
        kernel.1.applyGeneric value) input
  | .blend mix dry wet => do
      let dryBank ← dry.applyGeneric input
      let wetBank ← wet.applyGeneric input
      Oriented.Bank.blend dryBank wetBank mix
termination_by kernel => sizeOf kernel
decreasing_by
  all_goals first
    | decreasing_tactic
    | (have := Array.sizeOf_lt_of_mem kernel.2; simp_all; omega)

/-- One exact first-order all-pass section: the convolution identity in
    parallel with its causal exponential tail. -/
def ModalKernelExpr.allpassSection (tail : ModalMode) : ModalKernelExpr :=
  .parallel #[.identity, .proper (.causalTail tail)]

/-- A retained all-pass cascade followed by one dry/wet blend. -/
def ModalKernelExpr.dryWetAllpassCascade (tails : Array ModalMode)
    (mix : Sig) : ModalKernelExpr :=
  .blend mix .identity (.cascade (tails.map ModalKernelExpr.allpassSection))

/-- The exact topology of one first-order identity-plus-tail section. -/
def ModalKernelExpr.allpassTail? : ModalKernelExpr → Option ModalMode
  | .parallel kernels => match kernels.toList with
      | [.identity, .proper (.causalTail tail)] => some tail
      | _ => none
  | _ => none

/-- Extract the retained section rows of an all-pass cascade. -/
def ModalKernelExpr.allpassCascadeTails? : ModalKernelExpr → Option (Array ModalMode)
  | .cascade kernels => kernels.mapM ModalKernelExpr.allpassTail?
  | _ => none

/-- Extract a dry/wet all-pass product by generic structure. -/
def ModalKernelExpr.dryWetAllpassCascadeShape? :
    ModalKernelExpr → Option (Array ModalMode × Sig)
  | .blend mix .identity wet => do
      let tails ← wet.allpassCascadeTails?
      if tails.isEmpty then none else some (tails, mix)
  | _ => none

/-- Extract one ordinary oriented proper kernel.  Filters use the same case as
    rooms; the terminal cares about structure, not the product node name. -/
def ModalKernelExpr.orientedShape? : ModalKernelExpr → Option (Array ModalMode × Sig)
  | .proper (.oriented modes direction) => some (modes, direction)
  | _ => none

/-- Compose several deferred linear stages into one stage without evaluating
    any control early.  Their control bundles remain in authored stage order. -/
def ModalLinearStage.cascade (stages : Array ModalLinearStage) : ModalLinearStage where
  controls := stages.foldl (fun out stage => out ++ stage.controls) #[]
  build := fun response values => do
    let (_, kernels) ← stages.foldlM (fun (state : Nat × Array ModalKernelExpr) stage => do
      let cursor := state.1
      let next := cursor + stage.controls.size
      let kernel ← stage.build response (values.extract cursor next)
      pure (next, state.2.push kernel)) (0, #[])
    pure (.cascade kernels)

/-- The topology `x + proper(x)` represented as one retained kernel factor. -/
def ModalLinearStage.withDirect (stage : ModalLinearStage) : ModalLinearStage where
  controls := stage.controls
  build := fun response values => do
    let kernel ← stage.build response values
    pure (.parallel #[.identity, kernel])

/-- Scale the complete input value by one deferred control. -/
def ModalLinearStage.scale (control : ModalControlRef)
    (complement : Bool := false) : ModalLinearStage where
  controls := #[control]
  build := fun _ values => do
    let value ← match values[0]? with
      | some value => pure value
      | none => lit 0
    let value ← if complement then do
      let one ← lit 1
      sub one value
    else pure value
    pure (.scale value .identity)

/-- Put an independently authored wet stage chain beside its untouched input. -/
def ModalLinearStage.dryWet (stages : Array ModalLinearStage)
    (mix : ModalControlRef) : ModalLinearStage :=
  let wet := ModalLinearStage.cascade stages
  { controls := wet.controls.push mix
    build := fun response values => do
      let wetCount := wet.controls.size
      let wetKernel ← wet.build response (values.extract 0 wetCount)
      let mixValue ← match values[wetCount]? with
        | some value => pure value
        | none => lit 0
      pure (.blend mixValue .identity wetKernel) }

end Tropical.EmitArrow.ArenaNative
