import Tropical.EmitArrow.Sig
import Tropical.EmitArrow.Spectral

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
      against the voice's default frequency; see `Voice.withPhaseOffset`.) -/
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
  /-- A PARTIAL FAMILY (`EmitArrow.Spectral`): partials `k = 1..K` of one
      uniform fundamental, phases `k·φ` from one phasor, weights `a_k`. The
      algebra sees it as a bundle — an envelope distributes down to it — and it lowers
      to one `bankSum` region. -/
  | family (f : PartialFamily)
  /-- A COMPLEX SPECTRAL ENVELOPE over `t`: each partial of `t` is multiplied
      by `G(f_k, τ)` at its own frequency (amplitude × |G|, phase + arg G).
      `controls` are signal terms evaluated in the envelope's clock context,
      like `genC`'s. `t` must normalize to families through `sum` and `scale`;
      a term whose partials the algebra cannot see (an opaque generator, a
      product, a warp — the warped partial frequency needs θ′) is refused at
      emission. -/
  | envelope (law : EnvelopeLaw) (controls : Array ArrowTerm) (t : ArrowTerm)

instance : Inhabited ArrowTerm := ⟨.sum #[]⟩

/-- Distribute an envelope down to the families it multiplies. The envelope is
    linear (`G·(x + y) = G·x + G·y`) and commutes with scaling by any signal
    (`G` multiplies a partial's amplitude, as `scale` does), so it moves through
    `sum` and `scale`; it stops above a family (the normal form) and above
    anything else, where emission refuses it. -/
def envelopeInto (law : EnvelopeLaw) (controls : Array ArrowTerm) :
    ArrowTerm → ArrowTerm
  | .sum ts => .sum (ts.attach.map fun ⟨t, _⟩ => envelopeInto law controls t)
  | .scale weight t => .scale weight (envelopeInto law controls t)
  | t => .envelope law controls t

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
  | .family f => .family f
  | .envelope law controls t =>
      envelopeInto law (controls.attach.map fun ⟨c, _⟩ => normalize c) (normalize t)

/-- The voice with `offset` added to its phase input: the PM law, for `pmGen`
    and for a control-driven generator (`genC`) whose modulator is one of its
    controls. `offset` is built when the voice is wired, inside the build. A
    voice without a phase port is returned unchanged.

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
def Voice.withPhaseOffset (v : Voice) (offset : BuildM Sig) : Voice :=
  match v.phasePort with
  | some port => { v with wire := fun clock => do
      let inputs ← v.wire clock
      let phaseOffset ← offset
      inputs.mapM fun input => do
        if input.port.idx == port.idx then
          let value ← add input.value phaseOffset
          pure { input with value }
        else
          pure input }
  | none => v

/-- A spectral envelope whose controls have been emitted, waiting for the
    families below it. -/
abbrev PendingEnvelope := EnvelopeLaw × Array Sig

/-- The refusal for an envelope over a term whose partials the algebra cannot
    see. -/
def envelopeRefusal (what : String) : String :=
  s!"spectral envelope over {what}: an envelope applies to partial families, through sum and scale; this term's partials are not visible to the algebra"

/-- Emit under an enclosing clock transform, with the spectral envelopes that
    apply to this subterm.  Children, modulators, instance declarations, and
    left-associated sums are sequenced in authored order. An envelope's controls
    are emitted where the envelope stands; the envelope then rides down through
    `sum` and `scale` to the families it multiplies. Below a pending envelope a
    warp is refused (the warped partial's frequency needs θ′), as is any term
    that is not a family, a sum, a scale or another envelope. -/
def emitTermE (cmod : Clock → BuildM Clock) (envs : Array PendingEnvelope) :
    ArrowTerm → BuildM Sig
  | .gen v name clk => do
      unless envs.isEmpty do throw (envelopeRefusal s!"the generator '{name}'")
      let warpedClk ← cmod clk
      let builder ← get
      Builder.osc v s!"{name}{builder.decls.size}" warpedClk
  | .scale weight t => do
      let signal ← emitTermE cmod envs t
      mul weight signal
  | .arrUn f t => do
      unless envs.isEmpty do throw (envelopeRefusal "a pointwise map")
      let signal ← emitTermE cmod #[] t
      f signal
  | .arrN f ts => do
      unless envs.isEmpty do throw (envelopeRefusal "a pointwise map")
      let signals ← ts.attach.mapM fun ⟨t, _⟩ => emitTermE cmod #[] t
      f signals
  | .warp φ t => do
      unless envs.isEmpty do throw (envelopeRefusal "a warped term")
      emitTermE (fun clock => do
        let warped ← φ clock
        cmod warped) #[] t
  | .swarp mw modulator t => do
      unless envs.isEmpty do throw (envelopeRefusal "a warped term")
      emitTermE (fun clock => do
        let modSignal ← emitTermE cmod #[] modulator
        let warped ← mw clock modSignal
        cmod warped) #[] t
  | .konst signal => do
      unless envs.isEmpty do throw (envelopeRefusal "a constant")
      pure signal
  | .prod x y => do
      unless envs.isEmpty do throw (envelopeRefusal "a product")
      let sx ← emitTermE cmod #[] x
      let sy ← emitTermE cmod #[] y
      mul sx sy
  | .clk clock => do
      unless envs.isEmpty do throw (envelopeRefusal "a clock")
      cmod clock
  | .pmGen v name baseClk depth modulator => do
      unless envs.isEmpty do throw (envelopeRefusal s!"the generator '{name}'")
      let modSignal ← emitTermE cmod #[] modulator
      let warpedClk ← cmod baseClk
      let phaseOffset ← mul depth modSignal
      let v := v.withPhaseOffset (pure phaseOffset)
      let builder ← get
      Builder.osc v s!"{name}{builder.decls.size}" warpedClk
  | .genC mk post name clk controls => do
      unless envs.isEmpty do throw (envelopeRefusal s!"the generator '{name}'")
      let signals ← controls.attach.mapM fun ⟨t, _⟩ => emitTermE cmod #[] t
      let warpedClk ← cmod clk
      let builder ← get
      let raw ← Builder.osc (mk signals) s!"{name}{builder.decls.size}" warpedClk
      post signals raw
  | .family f => do
      let warpedClk ← cmod f.clk
      familySig f warpedClk envs
  | .envelope law controls t => do
      let signals ← controls.attach.mapM fun ⟨c, _⟩ => emitTermE cmod #[] c
      emitTermE cmod (envs.push (law, signals)) t
  | .sum ts => do
      let signals ← ts.attach.mapM fun ⟨t, _⟩ => emitTermE cmod envs t
      match signals[0]? with
      | none => lit 0
      | some first =>
          (signals.extract 1 signals.size).foldlM add first
termination_by term => sizeOf term
decreasing_by
  all_goals first
    | decreasing_tactic
    | (have := Array.sizeOf_lt_of_mem ‹_ ∈ _›; simp_all; omega)

/-- Emit under an enclosing clock transform (no envelope pending). -/
def emitTermC (cmod : Clock → BuildM Clock) (term : ArrowTerm) : BuildM Sig :=
  emitTermE cmod #[] term

def emitTerm (term : ArrowTerm) : BuildM Sig := emitTermC pure term

def flangeEffectWith (back forward : Clock → BuildM Clock)
    (signal : ArrowTerm) : BuildM ArrowTerm := do
  let half ← lit 5 1
  let quarter ← lit 25 2
  pure (.sum #[
    .scale half signal,
    .scale quarter (.warp back signal),
    .scale quarter (.warp forward signal)])

/-- An SVF-shaped complex spectral envelope (`svfEnvelopeLaw`) over `t`, with
    cutoff (Hz) and Q as signal terms: a filter SHAPE applied per partial, not
    filter dynamics. -/
def svfEnvelope (response : SvfResponse) (cutoff q : ArrowTerm) (t : ArrowTerm) :
    ArrowTerm :=
  .envelope (svfEnvelopeLaw response) #[cutoff, q] t

end Tropical.EmitArrow
