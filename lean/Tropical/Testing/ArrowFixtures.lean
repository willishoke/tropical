import Tropical.EmitArrow

/-!
# ArrowFixtures — EmitArrow gate carriers (test support, not production)

The device-under-test builders for the EmitArrow byte-gates and audio
equivalence gates: the `build*` carrier programs, the warp-law clock pairs,
the `WarpBankProgram` specs, the demo patch graphs, and the build-time
`Float` residue calculus with its moment validator. Consumed ONLY by
`tropicaltest` (the golden/equivalence runner) and the `diffcli` emit-gate
verbs — nothing on the production compile path imports this module.

Everything here stays in `namespace Tropical.EmitArrow` so gate call sites
read identically to the production combinators they exercise.
-/

namespace Tropical.EmitArrow

open Lean (JsonNumber)
open Tropical.Ir

/-! The warp-bank program signature, as port references. The signature is the
    same shape for every voice — only the input NAMES/defaults differ (which is
    a declaration concern, not a wiring one), so the value-level refs are shared:
    `clk : int` (input 0), `pitch : float` (input 1), `offset : float` (input 2). -/

/-- The clock input (`clk : int`), as a value. -/
def clkIn : Clock := .inputRef ⟨0⟩
/-- The pitch input (`freq`/`f0` : float) — input 1 regardless of its name. -/
def pitchIn : Sig := .inputRef ⟨1⟩
/-- The flange-offset input (`depth`/`delta` : float) — input 2. -/
def offsetIn : Sig := .inputRef ⟨2⟩

/-- δ = `toInt(offset · sampleRate · 2³²)` — the flange offset, `offset` seconds
    expressed in Q32.32 samples. A function of the clock/params (`offsetIn` is
    input 2), so warping by it is a lawful arrow. Shared across voices: both
    targets carry the identical offset expression (only its input *name* —
    `depth` vs `delta` — differs, and names aren't referenced here). -/
def deltaSamples : Sig := toIntE (mul (mul offsetIn .sampleRate) (lit 4294967296))

/-- The `FixedSinOsc` voice — pitch at port 0, clock at port 1 (source order). -/
def fixedSinOscVoice : Voice :=
  { programName := "FixedSinOsc"
    wire := fun clkE => #[ ⟨⟨0⟩, pitchIn⟩, ⟨⟨1⟩, clkE⟩ ] }

/-- The `ModalVoice` voice — clock at port 0, pitch at port 1 (source order). -/
def modalVoice : Voice :=
  { programName := "ModalVoice"
    wire := fun clkE => #[ ⟨⟨0⟩, clkE⟩, ⟨⟨1⟩, pitchIn⟩ ] }

-- ─────────────────────────────────────────────────────────────
-- `warpBank` — the voice-generic flanger combinator
-- ─────────────────────────────────────────────────────────────

/-- One tap of a warp bank: a named voice instance at a warped clock, scaled by
    `weight` in the final sum. -/
structure Tap where
  name : String
  warp : Clock → Clock
  weight : Sig

/-- The voice-generic warp bank as one arrow expression:
    `(warp φ₀ &&& warp φ₁ &&& …) >>> voiceₓn >>> weightedSum`.
    The shared `clkIn` fanned `n` ways is the diagonal `&&&`; each tap is one
    `voice` instance at its warped clock; the weighted sum is the collapsing
    `arr`. Generic over the voice — the SAME combinator builds `FlangeSin` and
    `ReversibleComb`, differing only in the `Voice` and `Tap`s supplied.

    The sum is left-associated (`((w₀·t₀ + w₁·t₁) + w₂·t₂)`) to match the
    source programs' `add(add(_, _), _)` nesting exactly. -/
def warpBank (v : Voice) (taps : Array Tap) : Builder × Sig := Id.run do
  let mut b : Builder := {}
  let mut summands : Array Sig := #[]
  for tap in taps do
    let (sig, b') := b.osc v tap.name (tap.warp clkIn)
    b := b'
    summands := summands.push (mul tap.weight sig)
  let out := match summands[0]? with
    | none => lit 0
    | some s0 => (summands.extract 1 summands.size).foldl add s0
  (b, out)

/-- The flanger taps shared by both targets: dry (`id`, 0.5) plus the two
    delayed taps (`−δ` / `+δ`, 0.25 each). `δ = deltaSamples` references the
    offset input by index, so the same taps serve every voice. -/
def flangerTaps : Array Tap := #[
  { name := "dry",   warp := fun c => c,                 weight := lit 5 1 },
  { name := "past",  warp := fun c => sub c deltaSamples, weight := lit 25 2 },
  { name := "ahead", warp := fun c => add c deltaSamples, weight := lit 25 2 } ]

-- ─────────────────────────────────────────────────────────────
-- M3 — assemble the resolved `Program` and push it into the arena
-- ─────────────────────────────────────────────────────────────

/-- `clk : int`, default `clock() = sampleIndex << 32`. Shared by every voice. -/
def clkInputDecl : AInputDecl :=
  { name := "clk", type? := some (.scalar .int),
    defaultSig := some (.binary .lshift .sampleIndex (lit 32)) }

/-- A pitch input (`freq`/`f0` : float), default `select(hz > 0, hz, 0)` — the
    elaborated form of the source's `select(220>0, 220, 0)` / `…110…`. -/
def pitchInputDecl (name : String) (hz : Int) : AInputDecl :=
  { name, type? := some (.scalar .float),
    defaultSig := some (.select (.binary .gt (lit hz) (lit 0)) (lit hz) (lit 0)) }

/-- An offset input (`depth`/`delta` : float), default `0.0007`. -/
def offsetInputDecl (name : String) : AInputDecl :=
  { name, type? := some (.scalar .float), defaultSig := some (lit 7 4) }

/-- A full warp-bank program: the program name, the voice it is generic over,
    its three input declarations (`clk`, pitch, offset), and the taps. Two
    instantiations of this record — `flangeSinSpec` and `reversibleCombSpec` —
    are the whole of the per-target difference. -/
structure WarpBankProgram where
  name : String
  voice : Voice
  inputs : Array AInputDecl
  taps : Array Tap

/-- Build a `WarpBankProgram`'s `Program` into `arena`, returning its
    `ProgramIdx`. `resolved` is the name→idx map `elabChain` produces; the only
    program linked against is `spec.voice.programName` — its registry (and the
    transitive merge of the voice's own entries) mirrors the elaborator's
    `registerInstanceDecl`. Insertion order is codec-observable. -/
def buildWarpBank (spec : WarpBankProgram) (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : Except String (Arena × ProgramIdx) := do
  let some vIdx := (resolved.find? (·.1 == spec.voice.programName)).map (·.2)
    | .error s!"EmitArrow: voice '{spec.voice.programName}' not found in the \
        elaborated stdlib chain"
  let some vProg := arena.program? vIdx
    | .error s!"EmitArrow: voice '{spec.voice.programName}' program index out of range"
  -- Transitive registry merge (mirrors `registerInstanceDecl`): the voice under
  -- its program name, then the voice's own registry entries in order, skipping
  -- keys already present.
  let mut registry : Array (String × ProgramIdx) := #[(vProg.name, vIdx)]
  for (k, v) in vProg.registry do
    if !registry.any (·.1 == k) then registry := registry.push (k, v)
  let (b, outExpr) := warpBank spec.voice spec.taps
  pure (assemble arena spec.name spec.inputs
    #[{ name := "out", type? := some (.scalar .float) }]
    b.decls #[(.port ⟨0⟩, outExpr)] registry)

-- ─────────────────────────────────────────────────────────────
-- The two instantiations — one combinator, two voices, two programs
-- ─────────────────────────────────────────────────────────────

/-- `FlangeSin` over the `FixedSinOsc` voice (the slice-1 regression gate). -/
def flangeSinSpec : WarpBankProgram :=
  { name := "FlangeSin", voice := fixedSinOscVoice
    inputs := #[clkInputDecl, pitchInputDecl "freq" 220, offsetInputDecl "depth"]
    taps := flangerTaps }

/-- `ReversibleComb` over the `ModalVoice` voice (the slice-2 gate). -/
def reversibleCombSpec : WarpBankProgram :=
  { name := "ReversibleComb", voice := modalVoice
    inputs := #[clkInputDecl, pitchInputDecl "f0" 110, offsetInputDecl "delta"]
    taps := flangerTaps }

-- The canonical `FlangeSin` / `ReversibleComb` / `FixedSin` / `FixedSinOsc` /
-- `MorphOsc` builders were promoted to `Tropical.Stdlib` (production). The
-- `flangeSinSpec`/`reversibleCombSpec`/`warpBank`/`morphOscMor` machinery stays
-- here for the slide/graph/law/carrier gates that still exercise it.

-- ─────────────────────────────────────────────────────────────
-- C1 — build the FOUNDATIONAL VOICE directly: `FixedSinOsc` from scratch
-- ─────────────────────────────────────────────────────────────

/-! The flanger family above *sources* `FixedSinOsc` as a `Voice` instance and
    leans on strata's `inlineInstances` to flatten it. The cutover wants
    EmitArrow to build the voice ITSELF — no instance boundary, the per-program
    path's one flat DAG. `buildFixedSinOsc` does exactly that: it reconstructs
    the post-strata (scalar, inlined) `FixedSinOsc` body — the `FixedPhasor`
    fixed-point phase (integer split-multiply on the Q32.32 clock) composed with
    the `Sin` polynomial (Payne–Hanek reduction + degree-11 Horner) — entirely
    from the smart constructors, then emits it.

    Two things make this byte-identical to `diffcli emit-stdlib FixedSinOsc`:

    * The post-strata form is a single inlined `Sig` tree (instances inlined,
      the `Sin` `fold` unrolled, the `let`s flattened). EmitArrow's Lean-level
      value reuse (`phase`, `x`, `n`, `r`, `r2` bound once and shared) yields the
      STRUCTURALLY-identical tree — the shared subterms appear duplicated exactly
      as the unrolled post-strata tree does — so `compileResolved`'s
      value-numbering produces the identical instruction stream.
    * The literals carry the same `JsonNumber` mantissa/exponent the surface
      parser produced (e.g. `6.283185307179586 = 6283185307179586·10⁻¹⁵`,
      `−2.505210838544172e-8 = −2505210838544172·10⁻²³`).



    Notes on the op set this exercises beyond the flanger: `div`, `bitAnd`,
    `rshift`, `lshift`, `round`, `toFloat`, and the `clamp` the elaborator emits
    for the `unipolar` bound (`ClockPhasor.offset` ⇒ `clamp _ 0 1`, and the
    `phase` output bound likewise). All plain scalar ops — no richer types. -/

-- `buildFixedSin` / `buildFixedSinOsc` promoted to `Tropical.Stdlib`.

-- ─────────────────────────────────────────────────────────────
-- C1 — a real MULTI-PORT program from the cartesian combinators: `MorphOsc`
-- ─────────────────────────────────────────────────────────────

/-! `buildFixedSinOsc` proved the combinators can emit a SISO generator's body
    from scratch (absorbing `inlineInstances` for one voice). `MorphOsc` is the
    DATA-axis step up — a genuine multi-port composition, built point-free from
    the products surface above:

      `ClockPhasor ⋙ (saw &&& Sin) ⋙ crossfade`

    It exercises everything `warpBank` did not: a real multi-INPUT instance
    (`ClockPhasor(clk, freq)` via the named-port bridge), the `ph.phase` diagonal
    fanned into *heterogeneous* consumers (a saw shaper AND a `Sin` instance —
    `&&&`), genuine `⋙` composition between two DIFFERENT sub-programs
    (ClockPhasor's phase feeds Sin's `x`), and the crossfade product
    `(1−morph)·saw + morph·sin`. The byte-gate (`runEmitCorpusGate "MorphOsc"`)
    asserts this reproduces the hand-written `stdlib/MorphOsc.md` exactly. -/


/-- ClockPhasor's input ports MorphOsc fills: `clk` (port 0), `freq` (port 1),
    `offset` (port 2 — the phase-anchor hook, wired from MorphOsc's `phase`). -/
def clockPhasorPorts : Array InputIdx := #[⟨0⟩, ⟨1⟩, ⟨2⟩]

/-- The phasor morphism `[clk, freq, offset] ⇝ [phase]` — the named-port bridge
    over `ClockPhasor`. -/
def phasorMor : Mor := instMor "ph" "ClockPhasor" clockPhasorPorts 1

/-- The saw shaper `[phase] ⇝ [2·phase − 1]` (a naive ramp; pure `arr`). -/
def sawMor : Mor := arrMor (fun w => #[sub (mul (lit 2) w[0]!) (lit 1)])

/-- The sine path `[phase] ⇝ [Sin(2π·phase).out]` — scale-by-2π (`arr`) ⋙ the
    `FixedSin` bridge. The `⋙` here is the cross-program inline: the phase is
    re-landed as its exact Q0.32 integer (lossless — P < 2³² ≪ 2⁵³), `FixedSin`'s
    Q2.30 body is inlined with no surviving instance boundary, and the sample
    scales to float once on the way out. -/
def sinMor : Mor :=
  seq (arrMor (fun w => #[toIntE (mul w[0]! (lit 4294967296))]))
      (seq (instMor "sin" "FixedSin" #[⟨0⟩] 1)
           (arrMor (fun w => #[div (toFloatE w[0]!) (lit 1073741824)])))

/-- The crossfade product `[a, b, mix] ⇝ [(1−mix)·a + mix·b]` (pure `arr`) —
    `CrossFade`'s body, inlined. -/
def crossfadeMor : Mor :=
  arrMor (fun w => #[add (mul (sub (lit 1) w[2]!) w[0]!) (mul w[2]! w[1]!)])

/-- `MorphOsc` as one cartesian pipeline over inputs `[freq, morph, clk, phase]`:
    route to `[clk, freq, phase, morph]`, run the phasor on the first three
    (`clk, freq, offset`) while `morph` rides along (`first 3`), fan the phase into
    saw and sine while `morph` rides along (`first 1 (saw &&& sin)`), then
    crossfade. The whole body is `ClockPhasor ⋙ (saw &&& Sin) ⋙ crossfade`; `morph`
    is threaded through the products, never recomputed. -/
def morphOscMor : Mor :=
  seq (arrMor (fun w => #[w[2]!, w[0]!, w[3]!, w[1]!]))   -- [freq,morph,clk,phase] → [clk,freq,phase,morph]
    (seq (first 3 phasorMor)                              -- → [phase, morph]
      (seq (first 1 (fan sawMor sinMor))                  -- → [saw, sin, morph]
           crossfadeMor))                                  -- → [out]

-- `buildMorphOsc` promoted to `Tropical.Stdlib` (the `morphOscMor` pipeline
-- stays here — `buildMorphOscLit` still uses it).

/-- An input-free `MorphOsc` carrier (literal `freqHz`, literal `morph`,
    closed-form `clk = sampleIndex << 32`) for the standard-rep differential —
    same combinator pipeline as `buildMorphOsc`, but renderable directly as a
    session root (no input ports to bind), like the warp-law carriers. -/
def buildMorphOscLit (name : String) (freqHz : Int) (morph : Sig)
    (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let registry ← buildRegistry arena resolved #["ClockPhasor", "FixedSin"]
  -- `clk = sampleIndex << 32` inline (the `clkInputDecl` default; `clockLit` is
  -- defined below in the warp-law section, so spell it out here).
  let (outs, b) := morphOscMor #[lit freqHz, morph, .binary .lshift .sampleIndex (lit 32), lit 0] {}
  .ok (assemble arena name #[]
    #[{ name := "out", type? := some (.scalar .float) }]
    b.decls #[(.port ⟨0⟩, outs[0]!)] registry)

-- ─────────────────────────────────────────────────────────────
-- M4 (slice 3) — the warp ARROW LAWS as audio goldens
-- ─────────────────────────────────────────────────────────────

/-! Slices 1-2 certified that one `warpBank` combinator reproduces two
    hand-written stdlib programs byte-for-byte. Slice 3 certifies the warp
    ALGEBRA: the arrow laws of `warp` hold byte-exactly **in rendered audio**.
    Warps are integer add/sub on the Q32.32 fixed-point clock; integer add/sub
    is exact and associative, so the two algebraically-equal sides of a law feed
    the oscillator a *bit-identical* int64 clock and render *bit-identical* audio
    — even though the emitted plans differ (there is no algebraic tree
    normalization; `(clk+δ)−δ` keeps its extra add+sub instructions).

    The carrier is a single closed-form oscillator. To keep both sides of a law
    self-contained (no input ports to bind), the clock and pitch are LITERAL
    closed forms rather than input refs: `clk = sampleIndex << 32` inline, pitch
    a constant. Composing warps needs no new combinator — they are just nested
    clock expressions (`sub (add clk δ) δ`, `neg (neg clk)`, …). -/

/-- A δ literal: `toInt(seconds · sampleRate · 2³²)` with `seconds` a concrete
    decimal (`mantissa · 10^(-exponent)`), no input ref — a closed-form Q32.32
    integer sample count, identical across the two sides of every law. -/
def deltaLit (mantissa : Int) (exponent : Nat) : Sig :=
  toIntE (mul (mul (lit mantissa exponent) .sampleRate) (lit 4294967296))

/-- δ₁ = 0.0007 s, as Q32.32 samples. -/
def delta1 : Sig := deltaLit 7 4
/-- δ₂ = 0.0011 s, as Q32.32 samples. -/
def delta2 : Sig := deltaLit 11 4

/-- The `FixedSinOsc` voice with a LITERAL pitch (220 Hz) in place of the
    `pitchIn` input ref — so the warp-law carrier is fully closed-form (no input
    ports for the session-root lowering to bind to). Clock still at port 1. -/
def litPitchSinOscVoice : Voice :=
  { programName := "FixedSinOsc"
    wire := fun clkE => #[ ⟨⟨0⟩, lit 220⟩, ⟨⟨1⟩, clkE⟩ ] }

/-- A single-oscillator carrier clocked at `clkE`: one `FixedSinOsc` voice
    (literal pitch) whose clock is the closed-form expression `clkE`, output =
    its signal. No input ports. The two algebraically-equal `clkE`s of a warp
    law render bit-identical audio though their plans differ.

    Reuses `Builder.osc` (the voice references the elaborated `FixedSinOsc` body;
    strata inlines it) and mirrors `buildWarpBank`'s registry merge. -/
def buildClockCarrier (name : String) (clkE : Clock) (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : Except String (Arena × ProgramIdx) := do
  let v := litPitchSinOscVoice
  let some vIdx := (resolved.find? (·.1 == v.programName)).map (·.2)
    | .error s!"EmitArrow: voice '{v.programName}' not found in the elaborated stdlib chain"
  let some vProg := arena.program? vIdx
    | .error s!"EmitArrow: voice '{v.programName}' program index out of range"
  let mut registry : Array (String × ProgramIdx) := #[(vProg.name, vIdx)]
  for (k, vi) in vProg.registry do
    if !registry.any (·.1 == k) then registry := registry.push (k, vi)
  let (sig, b) := ({} : Builder).osc v "voice" clkE
  pure (assemble arena name #[] #[{ name := "out", type? := some (.scalar .float) }]
    b.decls #[(.port ⟨0⟩, sig)] registry)

/-- Literal-pitch `FixedSinOsc` at 12 kHz — a tone high enough that a small
    lowpass FIR visibly attenuates (for the convolution stress test). -/
def litPitch12kVoice : Voice :=
  { programName := "FixedSinOsc"
    wire := fun clkE => #[ ⟨⟨0⟩, lit 12000⟩, ⟨⟨1⟩, clkE⟩ ] }

/-- A weighted multi-tap carrier over the closed-form `clockLit`: each `Tap` is a
    clock-warp + weight, fanned from the one clock and summed left-assoc. With
    integer-sample-delay warps this IS an FIR convolution. Closed-form (literal
    pitch, no input ports), so it renders directly as a session root — same shape
    as `buildClockCarrier`, generalized from one tap to a bank. -/
def buildTapCarrier (name : String) (v : Voice) (taps : Array Tap)
    (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let some vIdx := (resolved.find? (·.1 == v.programName)).map (·.2)
    | .error s!"EmitArrow: voice '{v.programName}' not found in the elaborated stdlib chain"
  let some vProg := arena.program? vIdx
    | .error s!"EmitArrow: voice '{v.programName}' program index out of range"
  let mut registry : Array (String × ProgramIdx) := #[(vProg.name, vIdx)]
  for (k, vi) in vProg.registry do
    if !registry.any (·.1 == k) then registry := registry.push (k, vi)
  let mut b : Builder := {}
  let mut summands : Array Sig := #[]
  for tap in taps do
    let (sig, b') := b.osc v tap.name (tap.warp clockLit)
    b := b'
    summands := summands.push (mul tap.weight sig)
  let out := match summands[0]? with
    | none => lit 0
    | some s0 => (summands.extract 1 summands.size).foldl add s0
  pure (assemble arena name #[] #[{ name := "out", type? := some (.scalar .float) }]
    b.decls #[(.port ⟨0⟩, out)] registry)

/-- Literal-pitch `FixedSinOsc` at an arbitrary `hz` (pitch at port 0, clock at
    port 1) — closed form, no input refs. -/
def litPitchVoice (hz : Int) : Voice :=
  { programName := "FixedSinOsc"
    wire := fun clkE => #[ ⟨⟨0⟩, lit hz⟩, ⟨⟨1⟩, clkE⟩ ] }

/-- An FM/PM carrier: a `modHz` modulator oscillator (reading the UNwarped
    `clockLit` — the pinning that keeps the warp a closed form in τ) drives a
    *fractional, sub-sample* clock warp `φ(τ) = clk − ⌊depth·mod(τ)·2³²⌋`, fed to
    a `carHz` carrier. `depth` is in samples; the `·2³²` keeps the warp in Q32.32
    so `mod`'s fractional value lands in the clock's sub-sample bits. This is the
    `sub clk (m clk)` modulated warp — a genuinely nonlinear reparametrization,
    not an affine shift. Closed form, no input ports. -/
def buildFmCarrier (name : String) (carHz modHz depthSamples : Int)
    (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let some vIdx := (resolved.find? (·.1 == "FixedSinOsc")).map (·.2)
    | .error "EmitArrow: voice 'FixedSinOsc' not found in the elaborated stdlib chain"
  let some vProg := arena.program? vIdx
    | .error "EmitArrow: voice 'FixedSinOsc' program index out of range"
  let mut registry : Array (String × ProgramIdx) := #[(vProg.name, vIdx)]
  for (k, vi) in vProg.registry do
    if !registry.any (·.1 == k) then registry := registry.push (k, vi)
  let (modSig, b0) := ({} : Builder).osc (litPitchVoice modHz) "mod" clockLit
  let warpedClk : Clock :=
    sub clockLit (toIntE (mul (mul (lit depthSamples) modSig) (lit 4294967296)))
  let (carSig, b) := b0.osc (litPitchVoice carHz) "car" warpedClk
  pure (assemble arena name #[] #[{ name := "out", type? := some (.scalar .float) }]
    b.decls #[(.port ⟨0⟩, carSig)] registry)

/-- Two-level phase modulation (operator FM, DX-style): a `mod2Hz` oscillator
    (reading the ambient clock) warps the `modHz` modulator's clock; that
    modulator's output in turn warps the `carHz` carrier's clock. The modulator
    is itself a warped oscillator, so this tests whether the warp/substitution
    composes through NESTING. Closed form, no input ports. -/
def buildPmPmCarrier (name : String) (carHz modHz mod2Hz depth1 depth2 : Int)
    (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let some vIdx := (resolved.find? (·.1 == "FixedSinOsc")).map (·.2)
    | .error "EmitArrow: voice 'FixedSinOsc' not found in the elaborated stdlib chain"
  let some vProg := arena.program? vIdx
    | .error "EmitArrow: voice 'FixedSinOsc' program index out of range"
  let mut registry : Array (String × ProgramIdx) := #[(vProg.name, vIdx)]
  for (k, vi) in vProg.registry do
    if !registry.any (·.1 == k) then registry := registry.push (k, vi)
  -- innermost: mod2 at the ambient clock; warps the modulator's clock
  let (mod2Sig, b0) := ({} : Builder).osc (litPitchVoice mod2Hz) "mod2" clockLit
  let modClk : Clock :=
    sub clockLit (toIntE (mul (mul (lit depth2) mod2Sig) (lit 4294967296)))
  -- modulator at the mod2-warped clock; its output warps the carrier's clock
  let (modSig, b1) := b0.osc (litPitchVoice modHz) "mod" modClk
  let carClk : Clock :=
    sub clockLit (toIntE (mul (mul (lit depth1) modSig) (lit 4294967296)))
  let (carSig, b) := b1.osc (litPitchVoice carHz) "car" carClk
  pure (assemble arena name #[] #[{ name := "out", type? := some (.scalar .float) }]
    b.decls #[(.port ⟨0⟩, carSig)] registry)

-- Law 1 — INVERSE / CANCELLATION:  warp(back δ) ⋙ warp(fwd δ) = id
-- Both `(clk+δ)−δ` and `(clk−δ)+δ` cancel to `clk` in exact int64; we build
-- the prose form `(clk+δ)−δ` (fwd inner, back outer).
/-- LHS clock `(clk+δ)−δ` — the oscillator after a forward then inverse warp. -/
def invLawLhsClock : Clock := sub (add clockLit delta1) delta1
/-- RHS clock `clk` — the identity. -/
def invLawRhsClock : Clock := clockLit

-- Law 2 — ADDITIVE DELAY / FUNCTORIALITY:
--   warp(back δ₁) ⋙ warp(back δ₂) = warp(back (δ₁+δ₂))
/-- LHS clock `(clk−δ₁)−δ₂` — two delays composed. -/
def addLawLhsClock : Clock := sub (sub clockLit delta1) delta2
/-- RHS clock `clk−(δ₁+δ₂)` — one delay by the summed offset. -/
def addLawRhsClock : Clock := sub clockLit (add delta1 delta2)

-- ─────────────────────────────────────────────────────────────
-- M5 (slice 4) — the cartesian diagonal / fan-out law
-- ─────────────────────────────────────────────────────────────

/-! Slice 3 certified the warp ALGEBRA (compose, cancel) in audio. Slice 4
    certifies the **cartesian diagonal** (`Δ : A ⇝ A × A`, `Δ = id &&& id`): the
    fan-out of one source into two differently-warped flangers is denotationally
    a pure `let` — equal to two independent (source + flanger) pairs. Two carriers,
    same `out`, same denotation:

    * `DiagonalShared` — ONE dry oscillator at `clk`, *fanned* into two flangers
      (offsets δ₁, δ₂); its `nestedOut` is referenced by BOTH flanger sums.
      5 osc instances: `dry@clk · past@clk−δ₁ · ahead@clk+δ₁ · past@clk−δ₂ · ahead@clk+δ₂`.
    * `DiagonalIndependent` — TWO dry oscillators (both at `clk`, same literal
      pitch), each its own flanger. 6 osc instances — the dry tap is declared
      twice.

    `out = flanger(δ₁) + flanger(δ₂)` with the IDENTICAL weighted-sum tree on
    both sides (`add(add(0.5·dry, 0.25·past), 0.25·ahead)`, the source/warpBank
    left-assoc convention). The ONLY structural difference is the duplicated dry
    instance — so the rendered audio is bit-identical (the two dry oscillators are
    the same closed form fed the same clock), and the COST question is whether
    within-program strata CSE collapses the duplicate so both forms reach the same
    minimal DAG. Like slice 3 the carriers are input-free (literal pitch, closed-
    form `clockLit`), so the session-root lowering binds no inputs.

    NB (session boundary): this CSE is a *within-program* property. Across
    SEPARATE top-level session instances cross-instance CSE does NOT happen (an
    earlier spike found two identical oscs cost ~2×) — so the diagonal's
    *efficiency* holds inside one program, not across a session graph. We do not
    build a multi-instance session here; the within-program pair is the slice. -/

/-- The flanger weighted sum, left-associated to match the source/`warpBank`
    convention exactly: `add(add(0.5·dry, 0.25·past), 0.25·ahead)`. Shared by both
    diagonal forms so their output expression trees are identical up to the dry
    instance index. -/
def flangerSum (dry past ahead : Sig) : Sig :=
  add (add (mul (lit 5 1) dry) (mul (lit 25 2) past)) (mul (lit 25 2) ahead)

/-- One INDEPENDENT flanger over voice `v` at base clock `baseClk`, offset `d`:
    three fresh osc instances (`dry`/`past`/`ahead`, suffixed `tag`), weighted-
    summed. The dry tap is its own instance — the duplicated source. -/
def Builder.flanger (b : Builder) (v : Voice) (baseClk : Clock) (d : Sig)
    (tag : String) : Builder × Sig :=
  let (dry,   b) := b.osc v ("dry" ++ tag)   baseClk
  let (past,  b) := b.osc v ("past" ++ tag)  (sub baseClk d)
  let (ahead, b) := b.osc v ("ahead" ++ tag) (add baseClk d)
  (b, flangerSum dry past ahead)

/-- One flanger over voice `v` at `baseClk`, offset `d`, sharing a pre-built dry
    signal `dry` (the fanned source). Only the two delayed taps are fresh — the
    `&&&` diagonal on the shared source. Same `flangerSum` tree as `Builder.flanger`. -/
def Builder.flangerSharedDry (b : Builder) (v : Voice) (baseClk : Clock) (d : Sig)
    (dry : Sig) (tag : String) : Builder × Sig :=
  let (past,  b) := b.osc v ("past" ++ tag)  (sub baseClk d)
  let (ahead, b) := b.osc v ("ahead" ++ tag) (add baseClk d)
  (b, flangerSum dry past ahead)

/-- Push an input-free voice program (`decls` + one `out = expr` assign) into
    `arena`, merging the `litPitchSinOscVoice` registry like `buildClockCarrier`.
    Shared by every input-free EmitArrow carrier (clock carrier + diagonals). -/
def buildVoiceProgram (name : String) (decls : Array AInst) (out : Sig)
    (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let v := litPitchSinOscVoice
  let some vIdx := (resolved.find? (·.1 == v.programName)).map (·.2)
    | .error s!"EmitArrow: voice '{v.programName}' not found in the elaborated stdlib chain"
  let some vProg := arena.program? vIdx
    | .error s!"EmitArrow: voice '{v.programName}' program index out of range"
  let mut registry : Array (String × ProgramIdx) := #[(vProg.name, vIdx)]
  for (k, vi) in vProg.registry do
    if !registry.any (·.1 == k) then registry := registry.push (k, vi)
  pure (assemble arena name #[] #[{ name := "out", type? := some (.scalar .float) }]
    decls #[(.port ⟨0⟩, out)] registry)

/-- SHARED diagonal: one dry source fanned into two flangers (δ₁, δ₂). 5 osc
    instances; the dry `nestedOut` feeds both flanger sums. -/
def buildSharedDiagonal (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  let v := litPitchSinOscVoice
  let (dry, b) := ({} : Builder).osc v "dry" clockLit
  let (b, f1) := b.flangerSharedDry v clockLit delta1 dry "1"
  let (b, f2) := b.flangerSharedDry v clockLit delta2 dry "2"
  buildVoiceProgram "DiagonalShared" b.decls (add f1 f2) arena resolved

/-- INDEPENDENT diagonal: two dry sources (same literal pitch + clock), each its
    own flanger (δ₁, δ₂). 6 osc instances — the dry tap declared twice. Same
    denotation as `buildSharedDiagonal`. -/
def buildIndependentDiagonal (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  let v := litPitchSinOscVoice
  let (b, f1) := ({} : Builder).flanger v clockLit delta1 "1"
  let (b, f2) := b.flanger v clockLit delta2 "2"
  buildVoiceProgram "DiagonalIndependent" b.decls (add f1 f2) arena resolved

-- ─────────────────────────────────────────────────────────────
-- M6 (slice 5) — REVERSE as warp(neg): the moat operation, certified as laws
-- ─────────────────────────────────────────────────────────────

/-! Slice 5 certifies REVERSE — the moat — as warp arrow laws. Reverse precomposes
    the voice's timebase with negation: `reverse = warp(neg)`, the voice reading
    `s(−clk)`. Three laws, all single-literal-pitch input-free carriers:

    * **Involution** `warp(neg) ⋙ warp(neg) = id`: the voice clocked at `−(−clk)`
      vs at `clk`. Byte-IDENTICAL — `−(−x) = x` exactly on the int64 clock (the
      combinator-level twin of the `ClockReverseProbe` golden).
    * **Reverse-swaps-delay** `warp(neg) ⋙ warp(back δ) = warp(fwd δ) ⋙ warp(neg)`:
      both reduce to the voice at `−clk+δ` — reversal turns a delay into an advance.
      Byte-IDENTICAL (pure int64 clock arithmetic, exact); plans differ.
    * **Reverse commutes with the symmetric flanger** `warp(neg) ⋙ flanger ≡
      flanger ⋙ warp(neg)` — denotationally equal, but under reverse the ±δ taps
      SWAP tree slot (see below), so the float value-sum REASSOCIATES. -/

-- Law 1 — INVOLUTION:  warp(neg) ⋙ warp(neg) = id
/-- LHS clock `−(−clk)` — the voice after reverse then reverse. -/
def revInvolutionLhsClock : Clock := neg (neg clockLit)
/-- RHS clock `clk` — the identity. -/
def revInvolutionRhsClock : Clock := clockLit

-- Law 2 — REVERSE-SWAPS-DELAY:  warp(neg) ⋙ warp(back δ) = warp(fwd δ) ⋙ warp(neg)
-- Per the `⋙` convention (left operand = OUTER warp, as in slices 3-4): the LHS
-- applies `neg` outer over `back δ`, the RHS applies `fwd δ` outer over `neg`.
/-- LHS clock `−(clk−δ) = −clk+δ` — reverse of a delayed clock. -/
def revSwapLhsClock : Clock := neg (sub clockLit delta1)
/-- RHS clock `(−clk)+δ = −clk+δ` — advance of a reversed clock. The delay became
    an advance; both denote `−clk+δ` in exact int64, so byte-identical. -/
def revSwapRhsClock : Clock := add (neg clockLit) delta1

-- Law 3 — REVERSE COMMUTES WITH THE SYMMETRIC FLANGER:
--   warp(neg) ⋙ flanger ≡ flanger ⋙ warp(neg)
-- Both build the SAME left-assoc `flangerSum` tree `add(add(0.5·dry,0.25·past),
-- 0.25·ahead)` over the SAME tap SET {−clk, −clk−δ, −clk+δ}. The difference is
-- WHICH physical tap lands in the `past` vs `ahead` slot, driven by whether `neg`
-- is the OUTER or INNER warp (the `⋙` convention again):
--
--   LHS = warp(neg) ⋙ flanger  (neg OUTER):  tap clock = neg(tapφ(clk))
--     dry   = neg(clk)            = −clk
--     past  = neg(clk−δ)          = −clk+δ      ← +δ tap in the `past` slot
--     ahead = neg(clk+δ)          = −clk−δ      ← −δ tap in the `ahead` slot
--   RHS = flanger ⋙ warp(neg)  (neg INNER):  tap clock = tapφ(neg(clk))
--     dry   = neg(clk)            = −clk
--     past  = neg(clk)−δ          = −clk−δ      ← −δ tap in the `past` slot
--     ahead = neg(clk)+δ          = −clk+δ      ← +δ tap in the `ahead` slot
--
-- So the −δ and +δ summands swap tree position: `(A + B) + C` vs `(A + C) + B`.
-- Float add is NOT associative, so the rendered audio is denotationally equal
-- (max|Δ| at the ULP scale) but NOT byte-identical. With a FIXED-POINT value
-- carrier (integer add is associative AND commutative) it would be byte-exact;
-- the float value sum is the only non-exact link — the (fixed-point) clock side
-- is exact (laws 1-2 prove it). THIS IS THE FINDING.

/-- LHS `warp(neg) ⋙ flanger`: `neg` is the OUTER warp, so `neg(clk−δ)=−clk+δ`
    and `neg(clk+δ)=−clk−δ` — the ±δ taps land swapped vs the RHS. Same
    `flangerSum` tree, input-free literal-pitch carrier. -/
def buildReverseThenFlanger (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  let v := litPitchSinOscVoice
  let tapClk (φ : Clock → Clock) : Clock := neg (φ clockLit)
  let (dry,   b) := ({} : Builder).osc v "dry"   (tapClk (fun c => c))
  let (past,  b) := b.osc v "past"  (tapClk (fun c => sub c delta1))
  let (ahead, b) := b.osc v "ahead" (tapClk (fun c => add c delta1))
  buildVoiceProgram "ReverseThenFlanger" b.decls (flangerSum dry past ahead) arena resolved

/-- RHS `flanger ⋙ warp(neg)`: `neg` is the INNER warp, so `neg(clk)−δ=−clk−δ`
    and `neg(clk)+δ=−clk+δ` — the ±δ taps keep the source flanger's slot order.
    Same `flangerSum` tree as the LHS; only the per-tap clock differs. -/
def buildFlangerThenReverse (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  let v := litPitchSinOscVoice
  let negClk : Clock := neg clockLit
  let tapClk (φ : Clock → Clock) : Clock := φ negClk
  let (dry,   b) := ({} : Builder).osc v "dry"   (tapClk (fun c => c))
  let (past,  b) := b.osc v "past"  (tapClk (fun c => sub c delta1))
  let (ahead, b) := b.osc v "ahead" (tapClk (fun c => add c delta1))
  buildVoiceProgram "FlangerThenReverse" b.decls (flangerSum dry past ahead) arena resolved

-- ─────────────────────────────────────────────────────────────
-- M7 (slice 6) — a FIXED-POINT VALUE carrier: the float reassociation
-- failure (slice 5, law 3) becomes BYTE-IDENTICAL
-- ─────────────────────────────────────────────────────────────

/-! Slice 5 found that `reverse ⋙ flanger ≡ flanger ⋙ reverse` is denotationally
    true but NOT byte-identical in FLOAT (1271/4096 samples differ): under reverse
    the ±δ taps SWAP tree slot, so the float weighted-sum reassociates
    `(A+B)+C` vs `(A+C)+B`, and float add is not associative. Slice 6 proves the
    fix — the SAME warp combinators instantiated at a FIXED-POINT value carrier,
    where the mix is INTEGER arithmetic (associative AND commutative), make that
    same law BYTE-IDENTICAL.

    The "fixed-point value" is the existing `int` carrier reinterpreted as a
    Q-fractional saw, mixed with the existing INTEGER IR ops. No type-system /
    `ScalarKind` change, no fixed-point `Sin`, no touching the engine's float
    carrier or the frozen float goldens — just integer `Sig` over the same
    Q32.32 clock the warps already speak, with a single `toFloat` scale at the
    DAC boundary.

    * SOURCE: a raw Q0.32 saw phasor INTEGER value — ClockPhasor's split-multiply
      but stopping BEFORE the `/2³²` toFloat (`fixedPhase`). Integer-only.
    * MIX: the SAME 0.5/0.25/0.25 weights as the float `flangerSum`, but as
      integer right-shifts `((dry>>1)+(past>>2))+(ahead>>2)` (`fixedFlangerSum`).
      Integer add is associative, so the past/ahead slot swap is invisible.
    * OUTPUT: `toFloat(mix)/2³²` at the boundary (`fixedOut`) — the SAME map on
      both law sides, so byte-identical ints render byte-identical floats. -/

/-- `⌊freq·2³²/SR⌋` as a LITERAL int (freq = 220 Hz, SR = 44100):
    `⌊220·4294967296/44100⌋ = 21426140`. Precomputed so the source needs no
    float division — it is integer end to end. -/
def fixedFreqInc : Sig := lit 21426140

/-- A fixed-point Q0.32 saw phasor as a PURE INTEGER function of the clock —
    ClockPhasor's split-multiply (`acc = inc·thi + (inc·tlo)>>32`, masked to
    `[0,2³²)`) but stopping BEFORE the `/2³²` toFloat, so the value stays a raw
    Q0.32 integer the flanger can mix with integer arithmetic. `thi = clk>>32`
    (whole samples), `tlo = clk & (2³²−1)` (the sub-sample fraction). Integer ops
    ONLY: `.rshift`, `.bitAnd`, `.add`, `.mul`. The `& (2³²−1)` reductions make
    the arithmetic/logical-shift distinction irrelevant (masked away), exactly as
    in ClockPhasor. -/
def fixedPhase (clk : Clock) : Sig :=
  let thi := .binary .rshift clk (lit 32)
  let tlo := .binary .bitAnd clk (lit 4294967295)
  let acc := add (mul fixedFreqInc thi)
                 (.binary .rshift (mul fixedFreqInc tlo) (lit 32))
  .binary .bitAnd acc (lit 4294967295)

/-- The fixed-point flanger weighted sum: the SAME 0.5/0.25/0.25 weights as the
    float `flangerSum`, but as INTEGER right-shifts on the Q0.32 source values —
    `((dry>>1) + (past>>2)) + (ahead>>2)`. Integer add is associative AND
    commutative (i64 wraparound is modular), so the past/ahead slot swap that
    `reverse` induces leaves the sum BIT-IDENTICAL — the float reassociation
    `(A+B)+C ≠ (A+C)+B` (slice 5, 1271/4096) cannot occur. Left-assoc to mirror
    `flangerSum`'s tree exactly. -/
def fixedFlangerSum (dry past ahead : Sig) : Sig :=
  add (add (.binary .rshift dry (lit 1)) (.binary .rshift past (lit 2)))
      (.binary .rshift ahead (lit 2))

/-- Scale the final fixed-point mix to the float output at the DAC boundary — a
    single `toFloat · /2³²` map, applied IDENTICALLY on both law sides, so
    byte-identical integer mixes render byte-identical floats. The Q0.32 mix is
    in `[0,2³²)`, so the float output is a unipolar saw in `[0,1)`. -/
def fixedOut (mix : Sig) : Sig :=
  .binary .div (.unary .toFloat mix) (lit 4294967296)

/-- Push an input-free, INSTANCE-FREE expression program into `arena` (no voice,
    no registry — the fixed-point carrier is pure integer arithmetic on
    `sampleIndex`, referencing no stdlib program). Returns its `ProgramIdx`. -/
def buildExprCarrier (name : String) (out : Sig) (arena : Arena) : Arena × ProgramIdx :=
  assemble arena name #[] #[{ name := "out", type? := some (.scalar .float) }]
    #[] #[(.port ⟨0⟩, out)] #[]

/-- A single fixed-point source carrier `out = fixedOut(fixedPhase(clkE))` — the
    fixed-point analog of `buildClockCarrier`, for the single-source warp laws
    (involution, reverse-swaps-delay, additive) over the INTEGER source. The two
    algebraically-equal clocks of a law feed `fixedPhase` a bit-identical int64
    clock, so the rendered audio is byte-identical (the clock side is exact). -/
def buildFixedSourceCarrier (name : String) (clkE : Clock) (arena : Arena) :
    Arena × ProgramIdx :=
  buildExprCarrier name (fixedOut (fixedPhase clkE)) arena

-- ── The RESIDUE CALCULUS (build-time): voice ⋙ reverb as one modal bank ────────
-- Composing a voice (poles λ, amps a) with a modal reverb (poles ν, residues r,
-- impulse response Σ r e^{νt}) IS a convolution — and the convolution of a sum of
-- exponentials with a sum of exponentials is again a sum of exponentials, its
-- coefficients pure residues. So `voice ⋙ reverb` is a BUILD-TIME complex
-- computation that produces a ModalMode array; no runtime convolution, no state.

/-- Build-time complex arithmetic for the residue calculus. Local and minimal
    (no Mathlib): poles `μ = −σ + iω`, mode amplitudes `A`. -/
structure Cplx where
  re : Float
  im : Float
deriving Inhabited

namespace Cplx
def ofReal (x : Float) : Cplx := ⟨x, 0.0⟩
def add (a b : Cplx) : Cplx := ⟨a.re + b.re, a.im + b.im⟩
def sub (a b : Cplx) : Cplx := ⟨a.re - b.re, a.im - b.im⟩
def mul (a b : Cplx) : Cplx := ⟨a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re⟩
def neg (a : Cplx) : Cplx := ⟨-a.re, -a.im⟩
def normSq (a : Cplx) : Float := a.re * a.re + a.im * a.im
def div (a b : Cplx) : Cplx :=
  let d := b.normSq
  ⟨(a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d⟩
def abs (a : Cplx) : Float := Float.sqrt a.normSq
def arg (a : Cplx) : Float := Float.atan2 a.im a.re
def powNat (a : Cplx) : Nat → Cplx
  | 0 => ⟨1.0, 0.0⟩
  | n + 1 => (powNat a n).mul a
end Cplx

/-- A composed mode: pole μ = −σ+iω, complex amp A, polynomial order `deg`
    (deg=1 is the `A·d·e^{μd}` double pole a coincident voice/reverb pole makes). -/
structure CMode where
  pole : Cplx
  amp  : Cplx
  deg  : Nat := 0

/-- The residue calculus: `voice ⋙ reverb` mode by mode. Each voice pole λ (amp a)
    contributes a FORCED mode at λ with amp `a·H(λ)`, `H(λ)=Σ_q r_q/(λ−ν_q)` (over
    NON-coincident poles), and per reverb pole ν_q a RINGING mode at ν_q with amp
    `−a·r_q/(λ−ν_q)`. When λ = ν_q (sympathetic resonance — a voice partial sitting
    exactly on a room mode) that pole is DEGENERATE: instead of a 1/0 it yields a
    `τ·e^{μd}` DOUBLE POLE (deg 1, amp `a·r_q`) — the resonance blow-up, from the
    algebra. Exactly the convolution of the two exponential sums; the deg-0
    couplings still sum to zero (continuous onset). -/
def residueCompose (voice reverb : Array (Cplx × Cplx)) : Array CMode :=
  let tol := 1e-6
  voice.foldl (fun acc pa =>
    let lam := pa.1
    let a := pa.2
    let Hlam := reverb.foldl (fun s nr =>
      if (lam.sub nr.1).normSq < tol then s
      else s.add (nr.2.div (lam.sub nr.1))) (Cplx.ofReal 0.0)
    let acc := acc.push { pole := lam, amp := a.mul Hlam }
    reverb.foldl (fun acc nr =>
      if (lam.sub nr.1).normSq < tol then
        acc.push { pole := nr.1, amp := a.mul nr.2, deg := 1 }
      else
        acc.push { pole := nr.1, amp := ((a.mul nr.2).div (lam.sub nr.1)).neg }) acc) #[]

/-- A composed mode → `ModalMode` (rectangular): `μ = −σ+iω`, `A = c_re+i·c_im`.
    No `sqrt`/`atan2` — straight `litF` of the pole and coefficient parts. -/
def cmodeToModalMode (m : CMode) : ModalMode :=
  { sigma := litF (-m.pole.re)
    omega := litF m.pole.im
    cre := litF m.amp.re
    cim := litF m.amp.im
    deg := m.deg }

/-- A mode's contribution to the convolution's k-th derivative at 0: a term
    `A·d^p·e^{μd}` has `y⁽ᵏ⁾(0) = A·(k!/(k−p)!)·μ^{k−p}` (0 for k<p) — the deg-0
    `A·μᵏ` and the deg-1 `A·k·μ^{k−1}` in one formula. -/
def cmodeMoment (m : CMode) (k : Nat) : Cplx :=
  if k < m.deg then ⟨0.0, 0.0⟩
  else
    let ff := (List.range m.deg).foldl (fun acc j => acc * (k - j).toFloat) 1.0
    (m.amp.mul (m.pole.powNat (k - m.deg))).mul (Cplx.ofReal ff)

/-- EXACT validator for `residueCompose`: the output modes must reproduce the
    convolution's Taylor jet at t=0. `Σᵢ cmodeMoment(mᵢ, k)` (degree-aware, so a
    τ·e double pole counts as `A·k·μ^{k−1}`) must equal the convolution's k-th
    derivative `y⁽ᵏ⁾(0) = Σ_atoms a·Σ_{j<k} (Σ_q r_q ν_qʲ)·λ^{k−1−j}` (and `y(0)=0`),
    for k=0..K. Max RELATIVE error (normalized by the cancellation-free magnitude).
    Pure complex ±×÷. A wrong sign/denominator/missing-ringing/degeneracy breaks
    a moment. -/
def residueMomentError (voice reverb : Array (Cplx × Cplx)) (K : Nat) : Float :=
  let modes := residueCompose voice reverb
  let hMoment (j : Nat) : Cplx :=
    reverb.foldl (fun s nr => s.add (nr.2.mul (nr.1.powNat j))) ⟨0.0, 0.0⟩
  let momMode (k : Nat) : Cplx :=
    modes.foldl (fun s m => s.add (cmodeMoment m k)) ⟨0.0, 0.0⟩
  let convJet (k : Nat) : Cplx :=
    if k == 0 then ⟨0.0, 0.0⟩ else
    voice.foldl (fun s pa =>
      let jet := (List.range k).foldl (fun t j =>
        t.add ((hMoment j).mul (pa.1.powNat (k - 1 - j)))) ⟨0.0, 0.0⟩
      s.add (pa.2.mul jet)) ⟨0.0, 0.0⟩
  let normScale (k : Nat) : Float :=
    modes.foldl (fun s m => s + (cmodeMoment m k).abs) 0.0
  (List.range (K + 1)).foldl (fun mx k =>
    let e := ((momMode k).sub (convJet k)).abs / (normScale k + 1e-300)
    if e > mx then e else mx) 0.0

/-- LHS `warp(neg) ⋙ fixed-point flanger` — `neg` is the OUTER warp, so tap clock
    = `neg(tapφ(clk))`: `past = neg(clk−δ) = −clk+δ`, `ahead = neg(clk+δ) = −clk−δ`
    — the ±δ taps land SWAPPED vs the RHS. Same `fixedFlangerSum` tree; the
    integer mix is slot-order-invariant. -/
def buildReverseThenFixedFlanger (arena : Arena) : Arena × ProgramIdx :=
  let tapClk (φ : Clock → Clock) : Clock := neg (φ clockLit)
  let dry   := fixedPhase (tapClk (fun c => c))
  let past  := fixedPhase (tapClk (fun c => sub c delta1))
  let ahead := fixedPhase (tapClk (fun c => add c delta1))
  buildExprCarrier "ReverseThenFixedFlanger"
    (fixedOut (fixedFlangerSum dry past ahead)) arena

/-- RHS `fixed-point flanger ⋙ warp(neg)` — `neg` is the INNER warp, so tap clock
    = `tapφ(neg(clk))`: `past = neg(clk)−δ = −clk−δ`, `ahead = neg(clk)+δ = −clk+δ`
    — the ±δ taps keep the source flanger's slot order. Same `fixedFlangerSum`
    tree; only the per-tap clock differs. The integer sum of the SAME three tap
    values is bit-identical to the LHS. -/
def buildFixedFlangerThenReverse (arena : Arena) : Arena × ProgramIdx :=
  let negClk : Clock := neg clockLit
  let tapClk (φ : Clock → Clock) : Clock := φ negClk
  let dry   := fixedPhase (tapClk (fun c => c))
  let past  := fixedPhase (tapClk (fun c => sub c delta1))
  let ahead := fixedPhase (tapClk (fun c => add c delta1))
  buildExprCarrier "FixedFlangerThenReverse"
    (fixedOut (fixedFlangerSum dry past ahead)) arena

/-- The flanger effect with the `FlangeSin` offset `δ = deltaSamples` (the
    offset input), so the slid form reproduces the stdlib `FlangeSin` clocks. -/
def flangeEffect (s : ArrowTerm) : ArrowTerm :=
  flangeEffectWith (fun c => sub c deltaSamples) (fun c => add c deltaSamples) s

/-- THE SLIDE GATE (Test 1). Build `FlangeSin` from the DOWNSTREAM-insert form —
    `osc ⋙ flange`, warps unreduced — then `normalize` (the slide) and emit.
    Byte-identical to stdlib `FlangeSin` ⇒ the compiler turned "flanger dropped
    downstream of the oscillator" into "the oscillator read at warped clocks,"
    reaching the exact hand-written upstream program. The inputs/voice/δ mirror
    `flangeSinSpec` so the only thing under test is the slide. -/
def buildFlangerViaSlide (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let registry ← buildRegistry arena resolved #["FixedSinOsc"]
  let term := normalize (flangeEffect (.gen fixedSinOscVoice "osc" clkIn))
  let (out, b) := emitTerm term {}
  .ok (assemble arena "FlangeSin"
    #[clkInputDecl, pitchInputDecl "freq" 220, offsetInputDecl "depth"]
    #[{ name := "out", type? := some (.scalar .float) }]
    b.decls #[(.port ⟨0⟩, out)] registry)

/-! The slide tests below use closed-form (literal-pitch, no input ports)
    carriers so they render directly as session roots, like the warp-law gates. -/

/-- A literal-δ back/forward warp (`δ = 0.0007 s`, Q32.32). -/
def slideBack : Clock → Clock := fun c => sub c (deltaLit 7 4)
def slideFwd : Clock → Clock := fun c => add c (deltaLit 7 4)
/-- The closed-form base oscillator term (literal pitch, `clk = sampleIndex<<32`). -/
def litOscGen : ArrowTerm := .gen litPitchSinOscVoice "osc" clockLit

/-- DOWNSTREAM `osc ⋙ shaper ⋙ flange` (Test 2): a pointwise `shaper` (square)
    sits BETWEEN the oscillator and the flanger, so the flanger's warps must
    COMMUTE PAST it (R1) to reach the generator's clock. Slid form. -/
def buildSlideShaperDownstream (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  let term := flangeEffectWith slideBack slideFwd (.arrUn (fun s => mul s s) litOscGen)
  let (out, b) := emitTerm (normalize term) {}
  buildVoiceProgram "SlideShaperDown" b.decls out arena resolved

/-- The hand-written UPSTREAM reference for Test 2: the same shaper applied to the
    oscillator read at each of the three warped clocks — what the slide MUST
    produce if the warp commutes past the shaper. No `warp` nodes (already
    upstream). Byte-equal to `buildSlideShaperDownstream` ⇒ R1 fired correctly. -/
def buildSlideShaperUpstream (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  let shaped (φ : Clock → Clock) : ArrowTerm :=
    .arrUn (fun s => mul s s) (.gen litPitchSinOscVoice "osc" (φ clockLit))
  let term : ArrowTerm := .sum #[
    .scale (lit 5 1) (shaped (fun c => c)),
    .scale (lit 25 2) (shaped slideBack),
    .scale (lit 25 2) (shaped slideFwd) ]
  let (out, b) := emitTerm term {}
  buildVoiceProgram "SlideShaperUp" b.decls out arena resolved

/-- A single closed-form flanger via the slide (Test 3 baseline). -/
def buildSlideSingleFlanger (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  let (out, b) := emitTerm (normalize (flangeEffectWith slideBack slideFwd litOscGen)) {}
  buildVoiceProgram "SlideSingleFlange" b.decls out arena resolved

/-- CASCADE `osc ⋙ flange ⋙ flange` (Test 3): two downstream flangers in series.
    The slide pushes the outer flanger's warps through the inner flanger's sum
    (R3) and fuses them with the inner warps (R2), producing the oscillator read
    at the NINE convolved offsets {0, ±δ, ±2δ} automatically — the proper
    multiplicity, derived, not hand-written. (Nine instances, not five: with no
    coincident-offset normalization the algebraically-equal taps stay distinct —
    the multiplicative cost discussed for cascades.) -/
def buildSlideDoubleFlanger (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  let inner := flangeEffectWith slideBack slideFwd litOscGen
  let (out, b) := emitTerm (normalize (flangeEffectWith slideBack slideFwd inner)) {}
  buildVoiceProgram "SlideDoubleFlange" b.decls out arena resolved

/-- The two factors of the product slide-law test: two distinct-pitch oscillators,
    so `x ⊗ y` is a genuine ring-modulation (not `x²`) and reclocking is observable. -/
def prodFactorX : ArrowTerm := .gen (litPitchVoice 220) "a" clockLit
def prodFactorY : ArrowTerm := .gen (litPitchVoice 330) "b" clockLit

/-- PRODUCT slide law (Test 4): `warp φ (x ⊗ y)` DOWNSTREAM — the product formed,
    THEN warped — must byte-equal the hand-written upstream form (φ on each
    factor). Byte-equality ⇒ the slide distributes the warp over `×`, i.e.
    `emitTermC` threads the same clock transform into BOTH factors so each
    generator reclocks. This is the law that makes `prod` (the VCA) lawful under
    the slide, exactly as `slide-past-arr` is the law for a pointwise `arr`. -/
def buildSlideProdDownstream (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  let (out, b) := emitTerm (normalize (.warp slideBack (.prod prodFactorX prodFactorY))) {}
  buildVoiceProgram "SlideProdDown" b.decls out arena resolved

/-- The hand-written UPSTREAM reference for Test 4: the same warp applied to each
    factor before the product. Byte-equal to the downstream form ⇒ the warp
    commuted into both factors of the `prod`. -/
def buildSlideProdUpstream (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  let term : ArrowTerm := .prod (.warp slideBack prodFactorX) (.warp slideBack prodFactorY)
  let (out, b) := emitTerm (normalize term) {}
  buildVoiceProgram "SlideProdUp" b.decls out arena resolved

-- ── The BOOTSTRAP (part 2): FixedSinOsc as a TERM over the clock leaf ──────────

/-- A `FixedSinOsc` built ENTIRELY as a term over the clock leaf:
    `FixedSin(toInt(phasor·2³²))/2³⁰` — the Q2.30 datapath sine at the exactly
    re-landed Q0.32 phase — no `gen`, no `.trop` instance. Warps reach it
    through the `clk` leaf, so it reverses/scrubs like any generator. -/
def fixedSinOscTerm (freqE offsetE : Sig) (c : Clock) : ArrowTerm :=
  ArrowTerm.arrUn
    (fun clkSig =>
      div (toFloatE (fixedSinCycSig
            (toIntE (mul (phasorPhaseSig freqE offsetE clkSig) (lit 4294967296)))))
          (lit 1073741824))
    (ArrowTerm.clk c)

/-- Emit the bootstrapped `FixedSinOsc` (220 Hz, phase 0, `clk = sampleIndex<<32`)
    as an instance-free carrier — the term side of the `bootstrap-sin` gate. -/
def buildBootstrapSinOsc (name : String) (arena : Arena) : Arena × ProgramIdx :=
  let (out, _) := emitTerm (normalize (fixedSinOscTerm (lit 220) (lit 0) clockLit)) {}
  buildExprCarrier name out arena

/-- `expSig` over a clock-driven ramp `x = sampleIndex·(20/2048) − 10` (so x sweeps
    [−10, 10] across a 2048-sample buffer, exercising ~29 distinct `ldexp` octaves).
    The reference side of the `bootstrap-exp` gate: rendered `expSig(x)` vs libm
    `exp(x)`. -/
def buildExpProbe (name : String) (arena : Arena) : Arena × ProgramIdx :=
  let sIdxF := toFloatE (rshift clockLit (lit 32))
  let x := sub (mul sIdxF (lit 9765625 9)) (lit 10)
  buildExprCarrier name (expSig x) arena

/-- `logSig(x)` over a positive ramp `x ∈ [0.02, 200]` (`x = 0.02 + i·0.0977`,
    ~4 decades across 2048 samples, exercising both range-reduction branches and
    ~13 exponent octaves). The reference side of the `bootstrap-log` gate:
    rendered `logSig(x)` vs libm `log(x)`. -/
def buildLogProbe (name : String) (arena : Arena) : Arena × ProgramIdx :=
  let sIdxF := toFloatE (rshift clockLit (lit 32))
  let x := add (lit 2 2) (mul sIdxF (lit 9765625 8))
  buildExprCarrier name (logSig x) arena

/-- `atan2E(sin θ, cos θ)` over a ramp `θ ∈ [−3.1, 3.096]` (all four quadrants across
    2048 samples, inside `(−π, π)` so no wrap) — must recover `θ`. The reference side
    of the `bootstrap-atan2` gate: rendered `atan2E` vs the known angle. -/
def buildAtan2Probe (name : String) (arena : Arena) : Arena × ProgramIdx :=
  let sIdxF := toFloatE (rshift clockLit (lit 32))
  let theta := sub (mul sIdxF (lit 302734375 11)) (lit 31 1)
  buildExprCarrier name (atan2E (sinSig theta) (cosSig theta)) arena

/-- Emit the modal bank through the ARROW path (`arrUn`/`clk`, then `emitTerm`) —
    the term side of the `modal-bank` gate. -/
def buildModalBankArrow (name : String) (modes : Array ModalMode) (anchor : Sig)
    (arena : Arena) : Arena × ProgramIdx :=
  let (out, _) := emitTerm (normalize (modalBankTerm modes anchor clockLit)) {}
  buildExprCarrier name out arena

/-- Emit the SAME bank straight-line (`modalBankSig` on the bare clock, no arrow
    term) — the standard-rep side of the `modal-bank` gate. -/
def buildModalBankDirect (name : String) (modes : Array ModalMode) (anchor : Sig)
    (arena : Arena) : Arena × ProgramIdx :=
  buildExprCarrier name (modalBankSig modes clockLit anchor) arena

/-- The BANKED twin of `buildModalBankDirect` (`modalBankSigTable` on the bare
    clock) — the device-under-test for the `banks-as-data` equivalence gate:
    same modes, byte-identical render, but an O(1)-in-modes plan. -/
def buildModalBankTable (name : String) (modes : Array ModalMode) (anchor : Sig)
    (arena : Arena) : Arena × ProgramIdx :=
  buildExprCarrier name (modalBankSigTable modes clockLit anchor) arena

/-- Emit a bloom-composed Γ-bridge pair bank (`bloomComposedSig`) over the bare
    clock — the `modal-bloom-gamma` gate's device-under-test. -/
def buildBloomComposed (name : String) (pairs : Array BloomPair) (anchor : Sig)
    (arena : Arena) : Arena × ProgramIdx :=
  buildExprCarrier name (bloomComposedSig pairs clockLit anchor) arena

/-- Build the analytic `(Re, Im)` pair (`modalBankSigPairTable`) as TWO carriers
    over the bare clock — the `modal-pair` gate's device-under-test. Each
    component is its own single-`bankSum` program, so `Re` ≡ `buildModalBankTable`
    and `Im` ≡ `buildModalBankTable` of the amp-rotated modes, bit-identically. -/
def buildModalBankPair (nameRe nameIm : String) (modes : Array ModalMode)
    (anchor : Sig) (arena : Arena) : (Arena × ProgramIdx) × (Arena × ProgramIdx) :=
  let (reSig, imSig) := modalBankSigPairTable modes clockLit anchor
  (buildExprCarrier nameRe reSig arena, buildExprCarrier nameIm imSig arena)

/-- Heterodyne FM as a twist at the realization seam: `Re·cosθ − Im·sinθ` over the
    analytic `(Re, Im)` pair, with the static-index modulation phase
    `θ(d) = b·sin(ω_m d)`. ONE complex rotation per sample, independent of bank
    size — the cheap realization of the same FM that `besselFuse` bakes into
    `carrier × (2N+1)` sideband modes. The `modal-heterodyne` gate cross-checks the
    two against each other (D6). ω_m in rad/s, b the index. -/
def buildHeterodyne (name : String) (modes : Array ModalMode) (wm b : Float)
    (anchor : Sig) (arena : Arena) : Arena × ProgramIdx :=
  let (re, im) := modalBankSigPairTable modes clockLit anchor
  let dSec := div (div (toFloatE (relClockQ clockLit anchor)) (lit 4294967296)) .sampleRate
  let theta := mul (litF b) (sinSig (mul (litF wm) dSec))
  buildExprCarrier name (sub (mul re (cosSig theta)) (mul im (sinSig theta))) arena

/-- The integrated-pole reading (LFO→pole): a carrier whose frequency is modulated
    by an LFO, rendered as the EXACT time-varying resonator — the phase advances by
    the INTEGRAL of the modulated frequency, `θ = (ω₀·p)·Re(∫LFO)`, i.e. a
    heterodyne twist (`buildHeterodyne`) with θ taken from the INTEGRATED modulator
    bank (`integrateBank`) rather than a static `b·sin`. This is the exact solution
    of `ẋ = μ(t)x`, NOT the snapshot reading (pole read at τ, applied over the whole
    elapsed d), which is a different, wrong function (`demos/modal_vco.py` D1). -/
def buildIntegratedPoleReading (name : String) (carrier lfo : Array ModalMode)
    (om0Depth : Float) (anchor : Sig) (arena : Arena) : Arena × ProgramIdx :=
  let (re, im) := modalBankSigPairTable carrier clockLit anchor
  let theta := mul (litF om0Depth) (modalBankSig (integrateBank lfo) clockLit anchor)
  buildExprCarrier name (sub (mul re (cosSig theta)) (mul im (sinSig theta))) arena

/-- `voice ⋙ reverb` end to end: run the residue calculus at build time, turn the
    composed complex modes into a `ModalMode` bank, and emit it — the connection is
    an exact symbolic computation, not a hand-tuned coupling table. -/
def buildModalReverb (name : String) (voice reverb : Array (Cplx × Cplx))
    (anchor : Sig) (arena : Arena) : Arena × ProgramIdx :=
  buildModalBankArrow name ((residueCompose voice reverb).map cmodeToModalMode) anchor arena

/-- `voice ⋙ reverb` with the residue done SYMBOLICALLY (`residueComposeE`): the
    poles/coeffs stay `Sig`, so any of them may be a live slot. With literal
    inputs it folds to exactly `buildModalReverb`; the `symbolic-residue` gate
    checks that agreement. -/
def buildModalReverbSym (name : String) (voice reverb : Array ModalMode)
    (anchor : Sig) (arena : Arena) : Arena × ProgramIdx :=
  buildModalBankArrow name (residueComposeE voice reverb) anchor arena

/-- `voice ⋙ reverb` with the COLLECTED residue (`residueComposeEC`, `m+n` modes) —
    the `residue-collected` gate's device-under-test, and the composition
    `lowerModal` uses for a patched reverb. -/
def buildModalReverbSymC (name : String) (voice reverb : Array ModalMode)
    (anchor : Sig) (arena : Arena) : Arena × ProgramIdx :=
  buildModalBankArrow name (residueComposeEC voice reverb) anchor arena

/-- `voice ⋙ reverb` as the FUSED DIVIDED-DIFFERENCE paired-mode bank
    (`residueComposeDD` → `modalBankSigTableDD`) — the stable near-degenerate
    composition (`residue-divdiff` gate's device-under-test). -/
def buildModalReverbDD (name : String) (voice reverb : Array ModalMode)
    (anchor : Sig) (arena : Arena) : Arena × ProgramIdx :=
  buildExprCarrier name (modalBankSigTableDD (residueComposeDD voice reverb) clockLit anchor) arena

/-- `voice ⋙ reverb` with the collected residue's Cauchy inner sums BANKED
    (`residueComposeBanked` → `modalBankSigTable`) — same composition as
    `buildModalReverbSymC`, O(m+n) coeff-fill code (`residue-banked` gate). -/
def buildModalReverbBanked (name : String) (voice reverb : Array ModalMode)
    (anchor : Sig) (arena : Arena) : Arena × ProgramIdx :=
  buildModalBankArrow name (residueComposeBanked voice reverb) anchor arena

/-- Emit the modal bank read through a clock warp φ, via the arrow `.warp` (so φ
    threads through the `.clk` leaf exactly as the master clock does) — for the
    reverse-reverb gate: a reversing φ makes the closed-form tail play backward. -/
def buildModalBankWarped (name : String) (modes : Array ModalMode) (anchor : Sig)
    (φ : Clock → Clock) (arena : Arena) : Arena × ProgramIdx :=
  let (out, _) := emitTerm (normalize (ArrowTerm.warp φ (modalBankTerm modes anchor clockLit))) {}
  buildExprCarrier name out arena

/-- Emit a bank read through a DIRECTION (rotation + per-mode gate + optional
    residue window) — the `modal-direction` gate's device-under-test, mirroring
    what `lowerInput` emits for a reverb carrying a `ModalDir`. -/
def buildModalBankDir (name : String) (modes : Array ModalMode) (anchor : Sig)
    (dir : Sig) (arena : Arena)
    (damp? : Option (Sig × Sig) := none) : Arena × ProgramIdx :=
  let (out, _) := emitTerm (normalize (modalBankTermDir modes anchor clockLit dir damp?)) {}
  buildExprCarrier name out arena

/-- `buildModalBankDir` with the LOWERING chosen explicitly (unrolled
    `modalBankSigDir` vs banked `modalBankSigDirTable`) — the two sides of the
    `banks-as-data-dir` equivalence gate, independent of the strangler flag. -/
def buildModalBankDirWith
    (lower : Array ModalMode → Sig → Sig → Sig → Option Sig → Sig)
    (name : String) (modes : Array ModalMode) (anchor : Sig)
    (dir : Sig) (arena : Arena)
    (damp? : Option (Sig × Sig) := none) : Arena × ProgramIdx :=
  let (out, _) := emitTerm (normalize (modalBankTermDirWith lower modes anchor clockLit dir damp?)) {}
  buildExprCarrier name out arena



/-- Emit the modal bank read through an ABSOLUTE SIGNAL address (`modalAddrWarp`) —
    the `modal-addr` gate's device-under-test, mirroring what `lowerInput` emits
    when a resonator's `addr` inlet is patched. The address is a ramp
    `s(τ) = τ_samples/SR − offsetSec` in seconds; `modalAddrWarp` makes `s` the
    bank's clock, so `dSec = s − anchor`. With `offsetSec = 0` the address IS time
    (identity — agrees with the un-addressed bank, correct scaling); with
    `offsetSec > 0` the causal gate relocates to `τ = offsetSec·SR` (the signal
    drives the strike). -/
def buildModalAddrRamp (name : String) (modes : Array ModalMode) (anchor : Sig)
    (offsetSec : Float) (arena : Arena) : Arena × ProgramIdx :=
  let addrRamp : ArrowTerm := ArrowTerm.arrUn
    (fun clk => sub (div (toFloatE (rshift clk (lit 32))) .sampleRate) (litF offsetSec))
    (ArrowTerm.clk clockLit)
  let bank := modalBankTerm modes anchor clockLit
  let (out, _) := emitTerm (normalize (.swarp modalAddrWarp addrRamp bank)) {}
  buildExprCarrier name out arena

-- The demo patches ─────────────────────────────────────────────

/-- `osc → flange` as a GRAPH (input-ref `FlangeSin` form): a `FixedSinOsc` source
    and a flanger wired downstream of it, offset `δ = deltaSamples`. -/
def flangeGraph : PatchGraph :=
  { nodes := #[
      { id := "osc", node := .source fixedSinOscVoice clkIn },
      { id := "fl",  node := .flange "osc"
          (fun c => sub c deltaSamples) (fun c => add c deltaSamples) } ],
    output := "fl" }

/-- `osc → flange → flange` as a GRAPH (closed form). -/
def doubleFlangeGraph : PatchGraph :=
  { nodes := #[
      { id := "osc", node := .source litPitchSinOscVoice clockLit },
      { id := "f1",  node := .flange "osc" slideBack slideFwd },
      { id := "f2",  node := .flange "f1" slideBack slideFwd } ],
    output := "f2" }

/-- A FAN-OUT patch: `osc` fanned into two flangers (offsets 0.0007 / 0.0011 s),
    summed by a mixer — the diagonal Δ through the lowering. -/
def fanOutGraph : PatchGraph :=
  { nodes := #[
      { id := "osc", node := .source litPitchSinOscVoice clockLit },
      { id := "fa",  node := .flange "osc" slideBack slideFwd },
      { id := "fb",  node := .flange "osc"
          (fun c => sub c (deltaLit 11 4)) (fun c => add c (deltaLit 11 4)) },
      { id := "mix", node := .mix #["fa", "fb"] } ],
    output := "mix" }

/-- THE LOWERING GATE (L1). Lower the GRAPH `osc → flange`, run the slide, emit,
    and wrap as `FlangeSin` — byte-identical to stdlib `FlangeSin`. The user's
    patch graph, lowered, reaches the exact hand-written program: graph → arrow
    → slide → emit, end to end. -/
def buildFlangeFromGraph (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let registry ← buildRegistry arena resolved #["FixedSinOsc"]
  let term ← lowerGraph flangeGraph
  let (out, b) := emitTerm (normalize term) {}
  .ok (assemble arena "FlangeSin"
    #[clkInputDecl, pitchInputDecl "freq" 220, offsetInputDecl "depth"]
    #[{ name := "out", type? := some (.scalar .float) }]
    b.decls #[(.port ⟨0⟩, out)] registry)

/-- Lower the GRAPH `osc → flange → flange` (closed form) — must byte-equal the
    hand-built `buildSlideDoubleFlanger` (L2): the lowering of a chain composes
    effects exactly as the hand-written nested term. -/
def buildDoubleFlangeFromGraph (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let term ← lowerGraph doubleFlangeGraph
  let (out, b) := emitTerm (normalize term) {}
  buildVoiceProgram "GraphDoubleFlange" b.decls out arena resolved

/-- Lower the FAN-OUT patch (closed form) — `osc` fanned into two flangers and
    mixed (L3): the diagonal + the product collapse through the lowering. -/
def buildFanOutFromGraph (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let term ← lowerGraph fanOutGraph
  let (out, b) := emitTerm (normalize term) {}
  buildVoiceProgram "GraphFanOut" b.decls out arena resolved

/-- `osc(mod) → osc(carrier).fm` as a GRAPH — a single FM voice. Lowers to the
    same term as `buildFmCarrier carHz modHz depth` (the bit-exact modulated-clock
    gate's carrier). -/
def fmGraphOf (carHz modHz depth : Int) : PatchGraph :=
  { nodes := #[
      { id := "mod", node := .source (litPitchVoice modHz) clockLit },
      { id := "car", node := .fm "mod" (litPitchVoice carHz) clockLit (lit depth) } ],
    output := "car" }

/-- Two-level PM (DX-style) as a GRAPH: `mod2 → mod.fm → car.fm`. Lowers to the
    same term as `buildPmPmCarrier carHz modHz mod2Hz depth1 depth2` (the
    bit-exact PM-of-PM gate's carrier). -/
def pmPmGraphOf (carHz modHz mod2Hz depth1 depth2 : Int) : PatchGraph :=
  { nodes := #[
      { id := "mod2", node := .source (litPitchVoice mod2Hz) clockLit },
      { id := "mod",  node := .fm "mod2" (litPitchVoice modHz) clockLit (lit depth2) },
      { id := "car",  node := .fm "mod" (litPitchVoice carHz) clockLit (lit depth1) } ],
    output := "car" }

/-- Lower the single-FM graph (M1: ≡ `buildFmCarrier`, the modulated node). -/
def buildFmFromGraph (carHz modHz depth : Int)
    (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let term ← lowerGraph (fmGraphOf carHz modHz depth)
  let (out, b) := emitTerm (normalize term) {}
  buildVoiceProgram "GraphFm" b.decls out arena resolved

/-- Lower the PM-of-PM graph (M2: ≡ `buildPmPmCarrier`, the nested modulated node). -/
def buildPmPmFromGraph (carHz modHz mod2Hz depth1 depth2 : Int)
    (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let term ← lowerGraph (pmPmGraphOf carHz modHz mod2Hz depth1 depth2)
  let (out, b) := emitTerm (normalize term) {}
  buildVoiceProgram "GraphPmPm" b.decls out arena resolved

end Tropical.EmitArrow
