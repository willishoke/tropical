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
  /-- The phase-offset input, if the program has one: where `pmGen` adds its
      phase modulation. (It once also carried a warped-tap phase correction
      against the voice's default frequency; see `anchoredVoice`.) -/
  phasePort : Option InputIdx := none

/-- Two instance declarations compute the same closed form: same program, same
    input expressions port for port (expression IDs are hash-consed, so equal
    IDs are equal expressions — the clock included). -/
private def sameInstance (programName : String) (inputs : Array AInput)
    (decl : AInst) : Bool :=
  decl.programName == programName && decl.inputs.size == inputs.size &&
    (decl.inputs.zip inputs).all fun (a, b) => a.port == b.port && a.value == b.value

/-- Instantiate a voice at a clock. Hash-consed on `(closed form, clock)`: if an
    instance of the same program with the same input expressions already exists
    in this build, its output is reused. Every program is a pure function of its
    inputs (there is no state primitive), so one evaluation serves every reader —
    a fan-out of the same source at the same clock (a dry path read by a mix and
    by a tap, a modulator re-emitted under each generator it warps) costs one
    evaluation, not one per wire. -/
def Builder.osc (v : Voice) (name : String) (clkE : Clock) : BuildM Sig := do
  let inputs ← v.wire clkE
  let builder ← get
  match builder.decls.findIdx? (sameInstance v.programName inputs) with
  | some i => nestedOut ⟨i⟩ v.output
  | none =>
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

/-- The identity clock: τ itself, the fixed-point coordinate the port is called
    with. A fractional τ is a fractional clock; nothing here rounds. -/
def clockLit : BuildM Clock := tau

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
  /-- A CONTROL-DRIVEN generator: `controls` are signal terms (a knob is a
      `konst`; any wire is its lowered term) evaluated under the SAME clock
      context as the generator itself, so a warp downstream slides the whole
      cone — controls included — up to the generator. `mk` builds the voice
      from the control signals; `post` maps its raw output (e.g. a phasor's
      phase) to the generator's signal, given the same controls. -/
  | genC (mk : Array Sig → Voice) (post : Array Sig → Sig → BuildM Sig)
      (name : String) (clk : Clock) (controls : Array ArrowTerm)

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
  | .genC mk post name clk controls =>
      .genC mk post name clk (controls.attach.map fun ⟨t, _⟩ => normalize t)

/-- Add a phase offset (PM) to a voice's phase input.

    A warped generator is NOT phase-corrected: it reads its source's closed
    form at the warped clock with the live parameters, exactly — the tap of a
    delay is the source at `τ − d`, notch positions follow the live pitch, and
    a reversed source keeps its pitch. (The former correction added
    `(pitch − pitch₀)·shift` against the AUTHORED default `pitch₀`, which kept
    a warped tap from jumping on a re-tune by pretending the source had stayed
    at `pitch₀` over the shift: wrong after the first re-tune — notch positions
    tracked the old pitch, and a time-varying shift such as `reverse` heard a
    pitch error `(pitch − pitch₀)·shift′`.) The host's anchor keeps the source
    continuous on the ambient clock; a tap at shift `s` steps by `Δinc·s` at a
    re-tune, which is the source's own history being rewritten. -/
private def anchoredVoice (v : Voice) (phaseOffset? : Option Sig := none) :
    BuildM Voice := do
  match v.phasePort, phaseOffset? with
  | some port, some phaseOffset =>
      pure { v with wire := fun clock => do
        let inputs ← v.wire clock
        inputs.mapM fun input => do
          if input.port.idx == port.idx then
            let value ← add input.value phaseOffset
            pure { input with value }
          else
            pure input }
  | _, _ => pure v

/-- Emit under an enclosing clock transform.  Children, modulators, instance
    declarations, and left-associated sums are sequenced in authored order. -/
def emitTermC (cmod : Clock → BuildM Clock) : ArrowTerm → BuildM Sig
  | .gen v name clk => do
      let warpedClk ← cmod clk
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
      let v ← anchoredVoice v (some phaseOffset)
      let builder ← get
      Builder.osc v s!"{name}{builder.decls.size}" warpedClk
  | .genC mk post name clk controls => do
      let signals ← controls.attach.mapM fun ⟨t, _⟩ => emitTermC cmod t
      let warpedClk ← cmod clk
      let builder ← get
      let raw ← Builder.osc (mk signals) s!"{name}{builder.decls.size}" warpedClk
      post signals raw
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
  let clockScale ← clockOneLit
  let fixedSamples ← mul samples clockScale
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
