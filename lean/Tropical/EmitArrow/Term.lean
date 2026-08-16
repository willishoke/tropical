import Tropical.EmitArrow.Sig

/-!
# EmitArrow.Term — ID-native voices and reified arrows

This is the phase-2 ID-valued combinator surface.  `ArrowTerm` remains an
inspectable syntax, but every opaque function that constructs expression
nodes is a `BuildM` action.  Emission sequences those actions left-to-right in
the same authored order as the recursive implementation.
-/

namespace Tropical.EmitArrow

open Tropical.Ir

structure Voice where
  programName : String
  wire : Clock → BuildM (Array AInput)
  output : OutputIdx := ⟨0⟩
  phaseAnchor : Option (InputIdx × (Clock → BuildM Sig)) := none

def Builder.osc (v : Voice) (name : String) (clkE : Clock) : BuildM Sig := do
  let inputs ← v.wire clkE
  let instanceIdx ← declareInst { name, programName := v.programName, inputs }
  nestedOut instanceIdx v.output

/-- A cartesian morphism over already-built signal IDs. -/
abbrev Mor := Array Sig → BuildM (Array Sig)

def idMor : Mor := fun xs => pure xs

def seq (f g : Mor) : Mor := fun xs => do
  let ys ← f xs
  g ys

def fan (f g : Mor) : Mor := fun xs => do
  let ys ← f xs
  let zs ← g xs
  pure (ys ++ zs)

def par (m : Nat) (f g : Mor) : Mor := fun xs => do
  let ys ← f (xs.extract 0 m)
  let zs ← g (xs.extract m xs.size)
  pure (ys ++ zs)

def first (m : Nat) (f : Mor) : Mor := par m f idMor

def second (m : Nat) (g : Mor) : Mor := par m idMor g

def dup : Mor := fun xs => pure (xs ++ xs)

def exl (m : Nat) : Mor := fun xs => pure (xs.extract 0 m)

def exr (m : Nat) : Mor := fun xs => pure (xs.extract m xs.size)

/-- Lift ordered ID rewiring or pointwise node construction into a morphism. -/
def arrMor (f : Array Sig → BuildM (Array Sig)) : Mor := f

def instMor (name programName : String) (portOrder : Array InputIdx)
    (numOut : Nat) : Mor := fun args => do
  let inputs : Array AInput :=
    (portOrder.zip args).map (fun (port, value) => { port, value })
  let instanceIdx ← declareInst { name, programName, inputs }
  (Array.range numOut).mapM fun output => nestedOut instanceIdx ⟨output⟩

def clockLit : BuildM Clock := do
  let sample ← sampleIndex
  let shift ← lit 32
  lshift sample shift

inductive ArrowTerm where
  | gen (v : Voice) (name : String) (clk : Clock)
  | warp (φ : Clock → BuildM Clock) (t : ArrowTerm)
  | scale (weight : Sig) (t : ArrowTerm)
  | arrUn (f : Sig → BuildM Sig) (t : ArrowTerm)
  | arrN (f : Array Sig → BuildM Sig) (ts : Array ArrowTerm)
  | sum (ts : Array ArrowTerm)
  | swarp (mw : Clock → Sig → BuildM Clock) (modulator : ArrowTerm)
      (t : ArrowTerm)
  | konst (signal : Sig)
  | prod (x y : ArrowTerm)
  | clk (clock : Clock)
  | pmGen (v : Voice) (name : String) (baseClk : Clock) (depth : Sig)
      (modulator : ArrowTerm)

instance : Inhabited ArrowTerm := ⟨.sum #[]⟩

/-- Pure structural normalization; expression construction remains in
    `emitTermC`, so normalization does not perturb builder effect order. -/
def normalize : ArrowTerm → ArrowTerm
  | .gen v name clk => .gen v name clk
  | .scale weight t => .scale weight (normalize t)
  | .arrUn f t => .arrUn f (normalize t)
  | .arrN f ts => .arrN f (ts.attach.map fun ⟨t, _⟩ => normalize t)
  | .sum ts => .sum (ts.attach.map fun ⟨t, _⟩ => normalize t)
  | .warp φ t => .warp φ (normalize t)
  | .swarp mw modulator t =>
      .swarp mw (normalize modulator) (normalize t)
  | .konst signal => .konst signal
  | .prod x y => .prod (normalize x) (normalize y)
  | .clk clock => .clk clock
  | .pmGen v name baseClk depth modulator =>
      .pmGen v name baseClk depth (normalize modulator)

private def anchoredVoice (v : Voice) (baseClk warpedClk : Clock)
    (phaseOffset? : Option Sig := none) : BuildM Voice := do
  match v.phaseAnchor with
  | some (port, correction) =>
      let shift ← sub baseClk warpedClk
      let anchorCorrection ← correction shift
      let correction ← match phaseOffset? with
        | none => pure anchorCorrection
        | some phaseOffset => add anchorCorrection phaseOffset
      pure { v with wire := fun clock => do
        let inputs ← v.wire clock
        inputs.mapM fun input => do
          if input.port.idx == port.idx then
            let value ← add input.value correction
            pure { input with value }
          else
            pure input }
  | none => pure v

/-- Emit under an enclosing clock transform.  Children, modulators, instance
    declarations, and left-associated sums are sequenced in authored order. -/
def emitTermC (cmod : Clock → BuildM Clock) : ArrowTerm → BuildM Sig
  | .gen v name clk => do
      let warpedClk ← cmod clk
      let v ← anchoredVoice v clk warpedClk
      let builder ← get
      Builder.osc v s!"{name}{builder.decls.size}" warpedClk
  | .scale weight t => do
      let signal ← emitTermC cmod t
      mul weight signal
  | .arrUn f t => do
      let signal ← emitTermC cmod t
      f signal
  | .arrN f ts => do
      let signals ← ts.attach.mapM fun ⟨t, _⟩ => emitTermC cmod t
      f signals
  | .warp φ t =>
      emitTermC (fun clock => do
        let warped ← φ clock
        cmod warped) t
  | .swarp mw modulator t =>
      emitTermC (fun clock => do
        let modSignal ← emitTermC cmod modulator
        let warped ← mw clock modSignal
        cmod warped) t
  | .konst signal => pure signal
  | .prod x y => do
      let sx ← emitTermC cmod x
      let sy ← emitTermC cmod y
      mul sx sy
  | .clk clock => cmod clock
  | .pmGen v name baseClk depth modulator => do
      let modSignal ← emitTermC cmod modulator
      let warpedClk ← cmod baseClk
      let phaseOffset ← mul depth modSignal
      let v ← anchoredVoice v baseClk warpedClk (some phaseOffset)
      let builder ← get
      Builder.osc v s!"{name}{builder.decls.size}" warpedClk
  | .sum ts => do
      let signals ← ts.attach.mapM fun ⟨t, _⟩ => emitTermC cmod t
      match signals[0]? with
      | none => lit 0
      | some first =>
          (signals.extract 1 signals.size).foldlM add first
termination_by term => sizeOf term
decreasing_by
  all_goals first
    | decreasing_tactic
    | (have := Array.sizeOf_lt_of_mem ‹_ ∈ _›; simp_all; omega)

def emitTerm (term : ArrowTerm) : BuildM Sig := emitTermC pure term

def flangeEffectWith (back forward : Clock → BuildM Clock)
    (signal : ArrowTerm) : BuildM ArrowTerm := do
  let half ← lit 5 1
  let quarter ← lit 25 2
  pure (.sum #[
    .scale half signal,
    .scale quarter (.warp back signal),
    .scale quarter (.warp forward signal)])

def sweptDelta (seconds modulator : Sig) : BuildM Sig := do
  let secondsModulated ← mul seconds modulator
  let sr ← sampleRate
  let samples ← mul secondsModulated sr
  let twoPow32 ← lit 4294967296
  let fixedSamples ← mul samples twoPow32
  toIntE fixedSamples

def sflangeBack (seconds : Sig) : Clock → Sig → BuildM Clock :=
  fun clock modulator => do
    let delta ← sweptDelta seconds modulator
    sub clock delta

def sflangeFwd (seconds : Sig) : Clock → Sig → BuildM Clock :=
  fun clock modulator => do
    let delta ← sweptDelta seconds modulator
    add clock delta

def sweptFlangeEffect (back forward : Clock → Sig → BuildM Clock)
    (modulator signal : ArrowTerm) : BuildM ArrowTerm := do
  let half ← lit 5 1
  let quarter ← lit 25 2
  pure (.sum #[
    .scale half signal,
    .scale quarter (.swarp back modulator signal),
    .scale quarter (.swarp forward modulator signal)])

end Tropical.EmitArrow
