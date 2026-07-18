import Tropical.EmitArrow
import Tropical.Ir.Strata
import Tropical.Ir.Core
import Tropical.Compile
import Tropical.Parse.Raise
import Tropical.Ir.Elaborator
import Tropical.StdlibChain
import Lean.Data.Json

/-!
# `load_patch_graph` — the playground's live arrow entry point (EXPERIMENT)

A downstream-only patch graph (the PureData-style playground GUI) decoded into the
`EmitArrow` `PatchGraph`, lowered through the SAME `lowerGraph → normalize → emitTerm`
the corpus gates exercise, wrapped as a session root, and compiled to a `FlatPlan`
via `compileSession` — the production loadable tail. The slide (`normalize`) pushes
each effect's warps up onto the generators' clocks, so a `flange`/`fm` dropped
downstream of an oscillator genuinely re-clocks that oscillator.

Continuous knobs are LIVE `param:<id>.<knob>` module slots — a knob change drives
`set_param` on the running kernel with no recompile (the modal path emits `paramRef`
via `pref`; the earlier "all knobs baked" design is retired). Only STRUCTURAL edits —
topology, mode-bank size, and baked strike data (gong/string mode tables, anchors) —
re-send the whole graph and hot-swap. Clickless either way, since there is no
per-sample state to carry.

Uncommitted: this file makes `EmitArrow` reachable from the live `frontend`.
-/

namespace Tropical.Playground

open Lean (Json JsonNumber)
open Tropical.Ir
open Tropical.EmitArrow

-- ── JSON field helpers (over `Lean.Json` objects) ───────────────────────────
private def jNum? (obj : Json) (key : String) : Option JsonNumber :=
  match obj.getObjVal? key with
  | .ok j => j.getNum?.toOption
  | .error _ => none

/-- A numeric param as a scalar `Sig` (`Sig.num`), carrying the JSON decimal
    straight through (mantissa · 10^-exponent). -/
private def jExpr (obj : Json) (key : String) (dflt : Sig) : Sig :=
  match jNum? obj key with
  | some n => lit n.mantissa n.exponent
  | none => dflt

/-- A numeric param truncated to `Int` (the `fm` depth, in samples). -/
private def jInt (obj : Json) (key : String) (dflt : Int) : Int :=
  match jNum? obj key with
  | some n => if n.exponent == 0 then n.mantissa else n.mantissa / ((10 : Int) ^ n.exponent)
  | none => dflt

private def jStr (obj : Json) (key : String) (dflt : String) : String :=
  match obj.getObjVal? key with
  | .ok (.str s) => s
  | _ => dflt

/-- A numeric param as a build-time `Float` (the gong's structural strike
    data — mode tables, anchors — is baked, not slotted). -/
private def jFloat (obj : Json) (key : String) (dflt : Float) : Float :=
  ((jNum? obj key).map (·.toFloat)).getD dflt

/-- A gong register's mode table: an array of `[freqHz, sigma, amp, phase]`
    rows → `ModalMode`s (rectangular: `cre = a·cos φ`, `cim = a·sin φ`, so
    the bank's `cre·cos ωd − cim·sin ωd` is `a·cos(ωd + φ)`). Amplitude-bloom
    pairs arrive pre-expanded (two rows, `±a`, two σ). Malformed rows drop. -/
private def jModes (obj : Json) (key : String) : Array ModalMode :=
  match (obj.getObjVal? key).toOption.bind (·.getArr?.toOption) with
  | none => #[]
  | some arr => arr.filterMap fun mj =>
    match mj.getArr?.toOption with
    | some fs =>
      if fs.size < 4 then none else
      let num := fun (i : Nat) => ((fs[i]!.getNum?.toOption).map (·.toFloat)).getD 0.0
      let a := num 2
      let ph := num 3
      some { sigma := litF (num 1), omega := litF (6.283185307179586 * num 0),
             cre := litF (a * Float.cos ph), cim := litF (a * Float.sin ph) }
    | none => none

-- ── Voices (literal pitch, so the knob bakes into the emitted clock) ─────────
/-- The phase-anchor correction — the slide, in the phase domain. A clock `shift`
    (Q32.32) of `shift/2³²` samples maps to a phase shift of
    `(freq − freqInit)·(shift/2³²)/SR` cycles: the phase the oscillator advances over
    the shift, referenced to the compile-time `freqInit` so the effect's delay is
    preserved as a fixed phase. Added to a warped copy's phase port by `emitTermC`,
    it keeps that copy phase-continuous across a live freq change. -/
private def phaseCorr (pitchE freqInit : Sig) : Clock → Sig :=
  fun shift => div (mul (sub pitchE freqInit) (toFloatE shift)) (mul (lit 4294967296) .sampleRate)

/-- The anchor payload: `(phaseSlot, freqInit)`. Present when the voice's freq is a
    live phase-anchored slot; the voice then wires the phase port and installs the
    `phaseAnchor` so every warped copy self-corrects. -/
abbrev Anchor := Sig × Sig

/-- `FixedSinOsc`: freq (port 0), clk (port 1), phase (port 2). -/
private def sineVoiceE (pitchE : Sig) (anchor : Option Anchor := none) : Voice :=
  match anchor with
  | some (phaseE, freqInit) =>
    { programName := "FixedSinOsc",
      wire := fun clkE => #[ ⟨⟨0⟩, pitchE⟩, ⟨⟨1⟩, clkE⟩, ⟨⟨2⟩, phaseE⟩ ],
      phaseAnchor := some (⟨2⟩, phaseCorr pitchE freqInit) }
  | none =>
    { programName := "FixedSinOsc",
      wire := fun clkE => #[ ⟨⟨0⟩, pitchE⟩, ⟨⟨1⟩, clkE⟩ ] }

/-- `MorphOsc`: freq (port 0), morph (port 1), clk (port 2), phase (port 3).
    `morph = 0` is saw, `morph = 1` is sine. -/
private def morphVoiceE (pitchE morphE : Sig) (anchor : Option Anchor := none) : Voice :=
  match anchor with
  | some (phaseE, freqInit) =>
    { programName := "MorphOsc",
      wire := fun clkE => #[ ⟨⟨0⟩, pitchE⟩, ⟨⟨1⟩, morphE⟩, ⟨⟨2⟩, clkE⟩, ⟨⟨3⟩, phaseE⟩ ],
      phaseAnchor := some (⟨3⟩, phaseCorr pitchE freqInit) }
  | none =>
    { programName := "MorphOsc",
      wire := fun clkE => #[ ⟨⟨0⟩, pitchE⟩, ⟨⟨1⟩, morphE⟩, ⟨⟨2⟩, clkE⟩ ] }

/-- The `source` node is always a `MorphOsc` (morph = 0 is saw, morph = 1 is sine),
    so the morph knob is always meaningful. -/
private def voiceOf (pitchE morphE : Sig) (anchor : Option Anchor) : Voice :=
  morphVoiceE pitchE morphE anchor

/-- `PluckedMorphOsc`: freq (0), morph (1), clk (2), event_rate (3), phase (4).
    A `MorphOsc` with the closed-form pluck envelope baked in — dynamic content
    that reverses with the master clock, so any downstream warp (a comb tap) reads
    a delayed PLUCKED copy (an audible echo/pre-echo), not a silent bulk delay. -/
private def pluckedVoiceE (pitchE morphE eventRateE : Sig) (anchor : Option Anchor := none) : Voice :=
  match anchor with
  | some (phaseE, freqInit) =>
    { programName := "PluckedMorphOsc",
      wire := fun clkE => #[ ⟨⟨0⟩, pitchE⟩, ⟨⟨1⟩, morphE⟩, ⟨⟨2⟩, clkE⟩, ⟨⟨3⟩, eventRateE⟩, ⟨⟨4⟩, phaseE⟩ ],
      phaseAnchor := some (⟨4⟩, phaseCorr pitchE freqInit) }
  | none =>
    { programName := "PluckedMorphOsc",
      wire := fun clkE => #[ ⟨⟨0⟩, pitchE⟩, ⟨⟨1⟩, morphE⟩, ⟨⟨2⟩, clkE⟩, ⟨⟨3⟩, eventRateE⟩ ] }

/-- `δ = toInt(seconds · sampleRate · 2³²)` — a Q32.32 sample offset (the stdlib
    flanger's own delay form), but with the `seconds` taken from the knob. Signed:
    a negative `seconds` reads the past, a positive one the future. -/
private def deltaOf (secondsE : Sig) : Sig :=
  toIntE (mul (mul secondsE .sampleRate) (lit 4294967296))

/-- `gᵏ` as a product of `k` live-`g` reads (`g` may be a `paramRef`), so the comb's
    decay is a live knob with no relower. `k = 0 ⇒ 1`. -/
private def gPow (g : Sig) (k : Nat) : Sig :=
  (Array.range k).foldl (fun acc _ => mul acc g) (lit 1)

/-- A struck resonator's modal bank: `npart` harmonics of `f0`, decay
    `σ_k = decay·(1 + 0.4k)`, amplitude `1/k^1.1`. `f0` and `decay` may be live
    `paramRef`s (the pole frequencies/decays sweep with the knobs), because the
    downstream residue calculus is emitted symbolically. -/
private def resonatorBank (f0 decay : Sig) (npart : Nat) : Array ModalMode :=
  (Array.range npart).map fun j =>
    let k := j + 1
    ModalMode.hz (mul (lit (Int.ofNat k)) f0)
                 (mul decay (litF (1.0 + 0.4 * k.toFloat)))
                 (litF (1.0 / Float.pow k.toFloat 1.1))

/-- The default plucked-string bank a bare `string` node strikes: the exact
    Jaffe-Smith pole table of the Karplus-Strong loop `y = x + ρ·½(y[n-N] +
    y[n-N-1])` at fundamental `f0`, closed form — the loop diagonalized, not
    simulated. Mode `k` sits at `f_k = k·SR/(N+½)` (`N = round(SR/f0)`); the
    loop's averaging filter has `|G| = cos(π f_k/SR)`, so one transit multiplies
    partial `k` by `g_k = ρ·|G|` and its decay is `σ_k = −SR·ln(g_k)/(N+½)` —
    highs die faster, which IS the plucked-string sound. Residues use a `1/k`
    displacement rolloff at golden-angle phases (projecting a real burst onto
    the modes is the DATA path's job — this is the audible default). SR is baked
    at 44100 for the default timbre; the `string` kind's real content arrives as
    `modes` data rows. Capped at 48 partials (higher ones decay in ms). -/
def defaultStringModes (f0 rho : Float) : Array ModalMode := Id.run do
  let sr := 44100.0
  let span := (sr / f0).round + 0.5          -- N + ½  (loop transit, samples)
  let pi := 3.141592653589793
  let kmax := min 48 (Float.floor (span / 2.0)).toUInt64.toNat
  let mut modes : Array ModalMode := #[]
  for j in [0:kmax] do
    let k := (j + 1).toFloat
    let fk := k * sr / span
    let g := rho * Float.cos (pi * fk / sr)   -- ρ·|G(ω_k)| per loop transit
    if fk < sr * 0.5 && g > 0.0 then
      let sigma := - sr * Float.log g / span                 -- decay, 1/s
      let amp := 1.0 / k                                      -- displacement rolloff
      let ph := 2.399963 * k                                 -- golden-angle phases
      modes := modes.push
        { sigma := litF sigma, omega := litF (2.0 * pi * fk),
          cre := litF (amp * Float.cos ph), cim := litF (amp * Float.sin ph) }
  return modes

/-- A reverb room as a `ModalMode` bank (pole + residue-as-coeff): `nmode`
    log-spaced modes over `[flo,fhi]` with damping `σ = 6.91/rt60` (live), unit
    residues at golden-ratio phases so the tail isn't a pure comb. Freqs and count
    are structural (baked); only the damping is a live knob. -/
private def reverbRoom (rt60 : Sig) (nmode : Nat) (flo fhi : Float) : Array ModalMode :=
  let sigma := div (lit 691 2) rt60           -- 6.91 / rt60
  let denom : Float := if nmode ≤ 1 then 1.0 else (nmode - 1).toFloat
  (Array.range nmode).map fun j =>
    let fq := flo * Float.pow (fhi / flo) (j.toFloat / denom)
    let ph := 6.283185307179586 * (0.6180339887 * j.toFloat)
    { sigma, omega := mul twoPiE (litF fq),
      cre := litF (Float.cos ph), cim := litF (Float.sin ph) }

/-- A 2-pole resonant filter as its EXACT complex-conjugate pole pair — the
    modal island's filter (the Serge-VCFQ move). "Filtering" a modal source is
    composing its poles with the filter's by the residue calculus, so the
    filter node is a `modalReverb` whose room is one conjugate pair:

      H(s) = ω₀² / (s² + (ω₀/Q)s + ω₀²)      (lowpass, unity DC gain,
                                               peak ≈ Q at fc — resonance ADDS
                                               presence, it never thins the
                                               passband)
      ν = −α ± i·ω_d,  α = ω₀/(2Q),  ω_d = ω₀·√(1 − 1/(4Q²))
      residues R = ∓ i·ω₀²/(2ω_d)

    BOTH conjugates are stored (amps R, R̄), so the composition's `Σ amp/(λ−ν)`
    IS the true rational H(λ) — exact, not a single-sided approximation — and
    the two rendered modes sum to the real impulse response
    `(ω₀²/ω_d)·e^{−αt}·sin(ω_d t)`. At high Q, α→0: the ringing modes barely
    decay and a strike PINGS at fc — the composed pole pair literally is the
    VCFQ's ring; sweep resonance up and the filter crosses into a struck
    resonator, which is the whole aesthetic reason it lives on the modal
    island. Everything is symbolic (`Sig`), so cutoff and resonance are live
    knobs through the composition — no relower.

    `res ∈ [0,1]` maps log to `Q = 0.55·80^res ∈ [0.55, 44]` (Q > ½ keeps ω_d
    real; the top of the knob rings for seconds). -/
private def filterPair (fc res : Sig) : Array ModalMode :=
  let w0 := mul twoPiE fc
  -- Q = 0.55·e^{res·ln 80}   (ln 80 = 4.382026634673881)
  let q := mul (lit 55 2) (expSig (mul res (litF 4.382026634673881)))
  let alpha := div w0 (mul (lit 2) q)
  let wd := mul w0 (.unary .sqrt (sub (lit 1) (div (lit 1) (mul (lit 4) (mul q q)))))
  let rim := div (mul w0 w0) (mul (lit 2) wd)          -- |Im R|
  #[ { sigma := alpha, omega := wd,     cre := lit 0, cim := neg rim },
     { sigma := alpha, omega := neg wd, cre := lit 0, cim := rim } ]

-- ── Node decode (named inlets: `in` is an object {port: [srcId,…]}) ──────────
private def portSources (inObj : Json) (port : String) : Array String :=
  match (inObj.getObjVal? port).toOption.bind (·.getArr?.toOption) with
  | some arr => arr.filterMap (·.getStr?.toOption)
  | none => #[]

/-- Resolve a live param name to a `paramRef` slot read, falling back to `dflt`
    (used only if the param table somehow lacks the entry — the collector always
    allocates one). Every continuous knob is a live slot, so its value is READ from
    the slot at runtime, never baked — turning it drives `set_param`, no relower. -/
private def pref (pidx : String → Option Nat) (name : String) (dflt : Sig) : Sig :=
  match pidx name with
  | some i => .paramRef ⟨i⟩
  | none => dflt

-- ── The port-spec table: ONE description of the vocabulary ──────────────────
-- Every recent vocabulary bug lived in a gap between two hand-maintained
-- descriptions of one thing (buildNode vs collectParams vs nodeSchema — the
-- sflange.rate dead knob was a NAME mismatch between an inlet and the param it
-- supersedes). The table is the single source; the collector, the lowering,
-- and the schema/vocabulary views are derived READERS of it and cannot drift.

/-- How a continuous port's live writes land — the host contract's dispatch
    key. `raw` = plain slot write; `glide` = closed-form smoothstep re-anchor
    (`#v0/#v1/#t0` companion slots); `anchor` = phase-anchored frequency
    (`#phase` companion — gliding a frequency VALUE would reintroduce the
    τ·f' chirp, so pitch keeps phase continuous instead). -/
inductive Discipline where
  | raw | glide | anchor
deriving BEq, Repr

/-- The connection-typing color a port carries or accepts. -/
inductive PortDomain where
  | signal | modal | control
deriving BEq, Repr

/-- Display metadata for a knob — semantics, not decoration: without it every
    frontend reinvents ranges and scales, badly, and drifts (three copies of
    this table have already existed). Served verbatim by `get_vocabulary`. -/
structure KnobMeta where
  min : Float
  max : Float
  log : Bool := false
  unit : String := ""
deriving Repr

/-- One port of a node kind. A port with `accepts ≠ #[]` is an inlet; a port
    with `knob = some (m, e)` carries a continuous param slot whose
    compile-time fallback is `lit m e` (the value `buildNode` bakes only if the
    collector somehow skipped the slot). A port may be BOTH (source `freq`: a
    control inlet that, unwired, is a knob) — wiring it supersedes the slot. -/
structure PortSpec where
  name : String
  accepts : Array PortDomain := #[]
  multi : Bool := false
  knob : Option (Int × Nat) := none
  discipline : Discipline := .raw
  display : Option KnobMeta := none
  /-- The port whose NORMAL this knob parameterizes (an input of the default
      subgraph, not of the node): sflange's `rate` is an input of `mod`'s
      normalled LFO. Wiring the owner replaces the whole default — circuit,
      knob, and slot vanish together, so a dead knob is unrepresentable (the
      sflange.rate bug was this relationship encoded as a name coincidence
      that didn't hold). -/
  ownerPort : Option String := none
deriving Repr

private def sigIn : Array PortDomain := #[.signal, .modal, .control]
private def modalIn : Array PortDomain := #[.modal]
private def ctrlIn : Array PortDomain := #[.control]

/-- THE table. Order matters twice: knob-bearing entries in declaration order
    define the `ParamIdx` scan order (`collectParams`), and the whole layout is
    what `get_vocabulary` will serve. -/
def portSpecs : String → Array PortSpec
  | "knob" => #[
      { name := "value", knob := some (0, 0),
        display := some { min := 0, max := 1000 } }]
  | "source" => #[
      { name := "freq", accepts := ctrlIn, knob := some (220, 0), discipline := .anchor,
        display := some { min := 0.02, max := 2000, log := true, unit := "Hz" } },
      { name := "morph", knob := some (0, 0), discipline := .glide,
        display := some { min := 0, max := 1 } },
      { name := "pm", accepts := sigIn }]
  | "pluck" => #[
      { name := "freq", accepts := ctrlIn, knob := some (110, 0), discipline := .anchor,
        display := some { min := 20, max := 2000, log := true, unit := "Hz" } },
      { name := "morph", knob := some (0, 0), discipline := .glide,
        display := some { min := 0, max := 1 } },
      { name := "event_rate", knob := some (2, 0), discipline := .glide,
        display := some { min := 0.1, max := 20, log := true, unit := "Hz" } }]
  | "comb" => #[
      { name := "in", accepts := sigIn },
      { name := "delay", knob := some (12, 3), discipline := .glide,
        display := some { min := 0.0005, max := 0.05, log := true, unit := "s" } },
      { name := "decay", knob := some (7, 1), discipline := .glide,
        display := some { min := 0, max := 0.95 } }]
  | "flange" => #[
      { name := "in", accepts := sigIn },
      { name := "depth", knob := some (7, 4), discipline := .glide,
        display := some { min := 0.0001, max := 0.01, log := true, unit := "s" } }]
  | "sflange" => #[
      { name := "in", accepts := sigIn },
      { name := "mod", accepts := sigIn },
      { name := "depth", knob := some (2, 3), discipline := .glide,
        display := some { min := 0.0002, max := 0.02, log := true, unit := "s" } },
      -- `rate` parameterizes `mod`'s normal (the built-in LFO): patch `mod`
      -- and the LFO, this knob, and its slot all vanish together.
      { name := "rate", knob := some (3, 1), ownerPort := some "mod",
        display := some { min := 0.02, max := 12, log := true, unit := "Hz" } }]
  | "fm" => #[
      { name := "in", accepts := sigIn },
      { name := "carrier", knob := some (330, 0), discipline := .anchor,
        display := some { min := 20, max := 2000, log := true, unit := "Hz" } },
      { name := "depth", knob := some (8, 0), discipline := .glide,
        display := some { min := 1, max := 400, log := true } }]
  | "delay" => #[
      { name := "in", accepts := sigIn },
      { name := "amount", knob := some (4, 3), discipline := .glide,
        display := some { min := 0.0001, max := 0.02, log := true, unit := "s" } }]
  | "reverse" => #[{ name := "in", accepts := sigIn }]
  | "mix" => #[{ name := "in", accepts := sigIn, multi := true }]
  | "ring" => #[{ name := "in", accepts := sigIn, multi := true }]
  | "resonator" => #[
      { name := "addr", accepts := sigIn },
      { name := "freq", knob := some (220, 0),
        display := some { min := 20, max := 2000, log := true, unit := "Hz" } },
      { name := "decay", knob := some (4, 0),
        display := some { min := 0.5, max := 50, log := true } }]
  | "reverb" => #[
      { name := "in", accepts := modalIn },
      { name := "rt60", knob := some (2, 0),
        display := some { min := 0.2, max := 12, log := true, unit := "sec" } },
      { name := "dir", knob := some (0, 0),
        display := some { min := 0, max := 1 } },
      { name := "sway", knob := some (0, 0),
        display := some { min := 0, max := 0.9 } },
      { name := "rate", knob := some (3, 1),
        display := some { min := 0.05, max := 8, log := true, unit := "Hz" } }]
  | "filter" => #[
      { name := "in", accepts := modalIn },
      { name := "cutoff", knob := some (800, 0), discipline := .glide,
        display := some { min := 20, max := 8000, log := true, unit := "Hz" } },
      { name := "resonance", knob := some (5, 1), discipline := .glide,
        display := some { min := 0, max := 1 } }]
  | "modalmix" => #[{ name := "in", accepts := modalIn, multi := true }]
  -- gong: a source with no inlets and no knobs — its strike data (`t`,
  -- `beta`, `g`, `modes_full`, `modes_half`) is structural, carried in
  -- `params`.
  | "gong" => #[]
  -- string: a plucked/struck string as its diagonalized modal bank. Like gong,
  -- its content (`freq`, `decay`, `t`, `modes`) is structural, carried in
  -- `params`; the optional `addr` inlet drives the pluck's time-address.
  | "string" => #[{ name := "addr", accepts := sigIn }]
  | "out" => #[{ name := "in", accepts := sigIn, multi := true }]
  | _ => #[]

/-- Each kind's outlet color (`none` = no outlet, the dac sink). -/
def outletOf : String → Option PortDomain
  | "knob" => some .control
  | "resonator" | "reverb" | "filter" | "modalmix" | "string" => some .modal
  | "out" => none
  | _ => some .signal

/-- The kinds the table covers, in schema order (`out` last — it has no outlet). -/
def vocabularyKinds : Array String := #[
  "source", "pluck", "comb", "flange", "delay", "reverse", "fm", "sflange",
  "mix", "ring", "gong", "string", "resonator", "reverb", "filter",
  "modalmix", "knob", "out"]

-- Derived views — the ONLY readers of glide/anchor/knob facts from here down.
private def portOf (kind kname : String) : Option PortSpec :=
  (portSpecs kind).find? (·.name == kname)
private def disciplineOf (kind kname : String) : Discipline :=
  ((portOf kind kname).map (·.discipline)).getD .raw
private def isGlided (kind kname : String) : Bool := disciplineOf kind kname == .glide
private def isAnchored (kind kname : String) : Bool := disciplineOf kind kname == .anchor
/-- A knob's compile-time fallback from the table (`lit m e`). -/
private def fallbackOf (kind kname : String) : Sig :=
  match (portOf kind kname).bind (·.knob) with
  | some (m, e) => lit m e
  | none => lit 0

/-- A closed-form smoothstep GLIDE of τ from three slots: `v0 + (v1−v0)·s²(3−2s)`,
    `s = clamp((τ − t0)/dur, 0, 1)`, `dur = 0.02·sampleRate` samples (20 ms at any
    rate, matching the engine's `set_param_glide`). The value eases from `v0` to
    `v1` starting at tick `t0`; the control plane re-anchors the slots on each turn.
    Stateless — the ramp is a pure function of the ambient clock, not an accumulator. -/
private def glideExpr (pidx : String → Option Nat) (base : String) (dflt : Sig) : Sig :=
  let v0 := pref pidx s!"{base}#v0" dflt
  let v1 := pref pidx s!"{base}#v1" dflt
  let t0 := pref pidx s!"{base}#t0" (lit 0)
  let dur := mul (lit 2 2) .sampleRate   -- 0.02·SR = 20 ms
  let s  := clampE (div (sub (toFloatE .sampleIndex) t0) dur) (lit 0) (lit 1)
  let ss := mul (mul s s) (sub (lit 3) (mul (lit 2) s))
  add v0 (mul (sub v1 v0) ss)

-- ── The live master clock (global time-warp) ────────────────────────────────
/-- The two reserved master-clock slots. `velocity` is the live scrub (forward /
    freeze / reverse / varispeed); `tau_base` is the host-held τ-origin the engine
    re-bases on each velocity change (`set_param_velocity`) so the scrub stays
    value-continuous — the stateless `ScrubClock` host-split, promoted to the
    arrow patch's base clock. -/
def masterVelocityParam : String := "master.velocity"
def masterTauBaseParam  : String := "master.tau_base"

/-- Every generator's base clock, so global time is a property of the whole patch
    (not per-oscillator): `M(n) = toInt(tau_base·SR·2³²) + toInt(velocity·2³²)·n`.
    At the defaults `velocity = 1, tau_base = 0` this is exactly `sampleIndex<<32`
    (the old `clockLit`), so the master clock is free until you scrub it. Reading
    it live means one `velocity` knob scrubs/reverses EVERY closed-form voice,
    envelope, and delay tap coherently — nothing downstream holds history. -/
private def masterClock (pidx : String → Option Nat) : Clock :=
  let vel := pref pidx masterVelocityParam (lit 1)
  let tb  := pref pidx masterTauBaseParam (lit 0)
  let velQ := toIntE (mul vel (lit 4294967296))
  let tbQ  := toIntE (mul (mul tb .sampleRate) (lit 4294967296))
  add tbQ (mul velQ .sampleIndex)

/-- Build a node, plus any synthesized helper nodes (a default LFO source for a
    swept flange with no modulator patched). `pidx` maps a live param name
    `<nodeId>.<knob>` to its `ParamIdx`. Every continuous knob reads its slot via
    `paramRef`; only structural selectors (`voice`, warp `mode`) and topology are
    baked, so only they trigger a relower. Every generator reads the live
    `masterClock` as its base, so global time-warp reaches all of them. -/
private def buildNode (pidx : String → Option Nat) (id kind : String)
    (_sel params inObj : Json) : Node × Array PatchNode :=
  let sig := (portSources inObj "in")[0]?.getD "__silence__"
  -- this node's own knob `kname` as a live value: a closed-form glide if glided,
  -- else a raw slot read; falling back to the baked literal if unallocated.
  let p := fun (kname : String) (dflt : Sig) =>
    if isGlided kind kname then glideExpr pidx s!"{id}.{kname}" dflt
    else pref pidx s!"{id}.{kname}" dflt
  -- a knob's request-or-table default: the JSON params value if given, else the
  -- table's fallback — the one place buildNode learns a default from.
  let dv := fun (kname : String) => jExpr params kname (fallbackOf kind kname)
  let clk := masterClock pidx
  match kind with
  | "knob" =>
    match pidx s!"{id}.value" with
    | some i => (.knob i, #[])
    | none => (.mix #[], #[])   -- a knob missing from the table (unreachable): silence
  | "source" =>
    -- a Knob wired into `freq` shadows the baked freq slot: read the WIRED knob's
    -- `<id>.value` slot instead.
    let pitchE := match (portSources inObj "freq")[0]? with
      | some w => pref pidx s!"{w}.value" (dv "freq")
      | none => p "freq" (dv "freq")
    let morphE := p "morph" (dv "morph")
    -- the anchor: (phase slot, compile-time freq). Present only when the source's
    -- own freq is a live slot; the compile-time freq is the reference the warped
    -- copies' phase correction is measured from.
    let anchor := (pidx s!"{id}.freq#phase").map fun i => ((.paramRef ⟨i⟩ : Sig), dv "freq")
    -- dedicated `pm` inlet: an AUDIO node wired here through-zero phase-modulates
    -- this carrier — `depth·mod` (cycles) added to the phase port, NOT the clock, so
    -- the carrier pitch stays put. `freq` is untouched, so the carrier's own freq
    -- knob (and a Knob wired into it) stay live under modulation.
    match (portSources inObj "pm")[0]? with
    | some modId =>
      -- the phase port must be wired for PM to land, so force an anchor (reusing the
      -- live `#phase` slot if present, else a flat base).
      let pmAnchor := anchor.getD ((lit 0 : Sig), dv "freq")
      -- A fixed musical PM index (0.3 cycles peak ≈ 1.9 rad). The GUI has no PM
      -- depth knob yet; a fixed depth makes the patch AUDIBLE on connect.
      (.pm modId (voiceOf pitchE morphE (some pmAnchor)) clk (lit 3 1), #[])
    | none =>
      (.source (voiceOf pitchE morphE anchor) clk, #[])
  | "pluck" =>
    -- a plucked MorphOsc source: pitch (anchored), morph (glided), and event_rate
    -- (glided) drive the baked-in envelope. The dynamic content of the instrument.
    let pitchE := match (portSources inObj "freq")[0]? with
      | some w => pref pidx s!"{w}.value" (dv "freq")
      | none => p "freq" (dv "freq")
    let anchor := (pidx s!"{id}.freq#phase").map fun i => ((.paramRef ⟨i⟩ : Sig), dv "freq")
    (.source (pluckedVoiceE pitchE (p "morph" (dv "morph"))
       (p "event_rate" (dv "event_rate")) anchor) clk, #[])
  | "comb" =>
    -- a one-sided resonant comb: dry (k=0) + a decaying tap series at k·delay.
    -- `delay` is signed (future = pre-echo, the moat), and doubles as the resonant
    -- spacing (small ⇒ pitched comb, large ⇒ discrete echoes). `decay` = gᵏ.
    let d := deltaOf (p "delay" (dv "delay"))   -- 0.012 s, future
    let g := p "decay" (dv "decay")             -- 0.7
    let K := 6
    let tail : Array (Sig × (Clock → Clock)) := (Array.range K).map fun j =>
      let k := j + 1
      (gPow g k, fun c => add c (mul (lit (Int.ofNat k)) d))
    (.comb sig (#[(lit 1, fun c => c)] ++ tail), #[])
  | "flange" =>
    let d := deltaOf (p "depth" (dv "depth"))
    (.flange sig (fun c => sub c d) (fun c => add c d), #[])
  | "delay" =>
    let d := deltaOf (p "amount" (dv "amount"))
    (.warpFx sig (fun c => sub c d), #[])
  | "reverse" => (.warpFx sig (fun c => neg c), #[])
  | "fm" =>
    -- `carrier` is a frequency → phase-anchored (own #phase slot); `depth` glides.
    let carAnchor := (pidx s!"{id}.carrier#phase").map fun i => ((.paramRef ⟨i⟩ : Sig), dv "carrier")
    (.fm sig (sineVoiceE (p "carrier" (dv "carrier")) carAnchor)
      clk (p "depth" (dv "depth")), #[])
  | "sflange" =>
    let depthSec := p "depth" (dv "depth")
    match (portSources inObj "mod")[0]? with
    | some modId => (.sflange sig modId depthSec, #[])
    | none =>
      -- no modulator patched: synthesize a built-in LFO sine at the `rate` knob.
      let lfoId := s!"__lfo_{id}"
      (.sflange sig lfoId depthSec,
       #[ { id := lfoId, node := .source (sineVoiceE (p "rate" (dv "rate"))) clk } ])
  | "mix" => (.mix (portSources inObj "in"), #[])
  | "ring" => (.ring (portSources inObj "in"), #[])
  -- MODAL-island nodes: they carry poles, compose by the residue calculus at
  -- build time, and realize to a Sig at their boundary (a Sig consumer or the
  -- tap) — realized against the live `clk` (master clock) so they scrub too.
  | "resonator" =>
    let f0 := p "freq" (dv "freq")
    let decay := p "decay" (dv "decay")
    -- Mode count is graph-configurable via the optional `"partials"` param
    -- (absent ⇒ 6, so existing graphs are unchanged). Only the bank's SIZE is
    -- structural (baked); f0/decay stay live knobs regardless of count.
    let npart := (jInt params "partials" 6).toNat
    -- optional `addr` inlet: a Sig node whose value BECOMES the bank's absolute
    -- time-address (seconds into the impulse response). Unpatched ⇒ reads the
    -- master clock as before; patched ⇒ the causal gate triggers on the address
    -- signal's crossing and the ring scrubs/pitches with its slope (modalAddrWarp).
    let addr? := (portSources inObj "addr")[0]?
    -- Trip-count-as-data (the room-size knob): the optional STATIC
    -- `"partials_max"` param is the bank's CAPACITY. When present, the mode
    -- list is built at capacity and `partials` becomes a LIVE param slot
    -- (default = its graph value, or 6) whose in-kernel read is the bank's
    -- dynamic trip count, clamped to capacity — turning it changes how many
    -- modes sound with NO recompile (the IR text is knob-invariant). Absent ⇒
    -- exactly the static path above: no new slot, no plan drift.
    match jNum? params "partials_max" with
    | none =>
      (.modalSource (resonatorBank f0 decay npart) (lit 0) clk addr?, #[])
    | some _ =>
      let cap := (jInt params "partials_max" 6).toNat
      let countE := pref pidx s!"{id}.partials" (lit (Int.ofNat npart))
      (.modalSource (resonatorBank f0 decay cap) (lit 0) clk addr? (some countE), #[])
  | "reverb" =>
    let rt60 := p "rt60" (dv "rt60")
    -- reading DIRECTION: θ (radians, live) rotates the composed tail's poles in the
    -- s-plane — 0 = forward decay, π = reverse (pre-verb), interior = a continuous
    -- U(1) morph (σ↔ω at π/2). `window` (live) nulls each mode at its horizon-
    -- crossing (`σ²/(σ²+w²)`), offset off 0 so the kernel never divides 0/0: at the
    -- knob's floor it is near-bare rotation, opened up it is the polite morph.
    -- DIR crossfades the tail's time-direction: 0 = forward ring, 1 = reverse
    -- (pre-verb into the strike), interior = both. Keeps σ/ω fixed, so it stays
    -- audible across the whole range (no pole rotation).
    let dirX := p "dir" (dv "dir")
    -- SWAY: the room's decay breathes — σ ↦ σ·(1 + sway·sin(2π·rate·t)) on the
    -- envelope's clock only (pitch fixed). Continuous CF modulation of RT60 that
    -- stays on-island (no ∫σ dτ, no state); scrubs/reverses with the master clock.
    let sway := p "sway" (dv "sway")
    let swayRate := p "rate" (dv "rate")   -- 0.3 Hz: a slow breath
    let dir : ModalDir := { dir := dirX, damp := some (sway, swayRate) }
    (.modalReverb sig (reverbRoom rt60 32 60.0 6000.0) (some dir), #[])
  | "filter" =>
    -- the filter IS a modalReverb with a computed 2-mode room: the residue
    -- calculus does the "filtering" at build time, knobs stay live through it.
    (.modalReverb sig
      (filterPair (p "cutoff" (dv "cutoff")) (p "resonance" (dv "resonance"))) none, #[])
  | "modalmix" => (.modalMix (portSources inObj "in"), #[])
  | "gong" =>
    -- One STRIKE of the struck nonlinear resonator: two anchored modal banks
    -- (full-glide + stiff half-glide registers) behind per-strike pitch-bloom
    -- warps, composed by `gongStrikeNodes` from existing node kinds. All data
    -- is structural (a score's strikes are baked; the master clock still
    -- scrubs/reverses them live): `t` (strike time, s), `beta` (pitch-bloom
    -- depth, velocity already folded in), `g` (bloom settle rate), and the
    -- two pre-expanded mode tables.
    let t := jFloat params "t" 0.0
    let beta := jFloat params "beta" 0.0
    let gRate := jFloat params "g" 1.8
    let anchor := mul (litF t) .sampleRate
    -- a bare gong (no score data) strikes the built-in default bank at its
    -- anchor — the kind's audible default, like the resonator's 6 partials.
    let full := jModes params "modes_full"
    let half := jModes params "modes_half"
    let (full, half) :=
      if full.isEmpty && half.isEmpty
      then defaultGongModes (jFloat params "freq" 110.0)
      else (full, half)
    -- a data-less drop also gets an audible bloom (β = 0.05 ≈ a solid strike)
    let beta := if beta == 0.0 && (params.getObjVal? "beta").toOption.isNone
      then 0.05 else beta
    gongStrikeNodes id clk anchor (litF beta) (litF gRate) full half
  | "string" =>
    -- A plucked string as its diagonalized modal bank — the Karplus-Strong loop
    -- is an LTI system, so its delay-line recurrence and this closed-form pole
    -- bank are the same object, two realizations. `modes` (data rows
    -- [f, σ, amp, phase]: the exact loop poles + the burst's residues) is
    -- structural, a baked strike like a gong's; a data-less drop strikes the
    -- closed-form default bank at `freq`/`decay`. `t` places the pluck in a
    -- score; the master clock still scrubs/reverses the tail. The pluck holds
    -- NO state — the burst is a seed (a coordinate), not entropy that left — so
    -- unlike a delay-line string this one reverses with zero latency.
    let modes := jModes params "modes"
    let modes := if modes.isEmpty
      then defaultStringModes (jFloat params "freq" 196.0) (jFloat params "decay" 0.996)
      else modes
    let t := jFloat params "t" 0.0
    let anchor := mul (litF t) .sampleRate
    let addr? := (portSources inObj "addr")[0]?
    (.modalSource modes anchor clk addr?, #[])
  | _ => (.mix (portSources inObj "in"), #[])

private structure Raw where
  id : String
  kind : String
  sel : Json
  params : Json
  inObj : Json

private def decodeRaw (nj : Json) : Option Raw :=
  match nj.getObjVal? "id", nj.getObjVal? "kind" with
  | .ok (.str id), .ok (.str kind) =>
    let sel := (nj.getObjVal? "sel").toOption.getD (Json.mkObj [])
    let params := (nj.getObjVal? "params").toOption.getD (Json.mkObj [])
    let inObj := (nj.getObjVal? "in").toOption.getD (Json.mkObj [])
    some { id, kind, sel, params, inObj }
  | _, _ => none

private def rawsOf (j : Json) : Array Raw :=
  match (j.getObjVal? "nodes").toOption.bind (·.getArr?.toOption) with
  | some arr => arr.filterMap decodeRaw
  | none => #[]

/-- The continuous knobs each node kind carries, DERIVED from the port-spec
    table (declaration order = `ParamIdx` scan order). Structural selectors
    (`voice`, warp `mode`) are not ports: changing one alters topology, so it
    relowers. -/
def knobNamesOf (kind : String) : Array String :=
  (portSpecs kind).filterMap fun p => if p.knob.isSome then some p.name else none

-- ── get_vocabulary: the port-spec table, served ──────────────────────────────
private def domStr : PortDomain → String
  | .signal => "signal" | .modal => "modal" | .control => "control"
private def discStr : Discipline → String
  | .raw => "raw" | .glide => "glide" | .anchor => "anchor"

/-- The vocabulary as JSON — the ONE description of the node kinds, GENERATED
    from the port-spec table (the hand-maintained `nodeSchema` this replaces
    was the third copy, and the class of bug this file exists to kill). Per
    kind: outlet color and ports; per port: inlet facts (accepts/multi), knob
    facts (default, write discipline, display metadata), and `owner` when the
    knob parameterizes another port's normal. Clients RENDER this — nothing
    the engine knows may be re-encoded client-side. The connection rule rides
    along: `outlet→inlet` valid iff `outlet.color ∈ inlet.accepts`; a modal
    outlet into a signal inlet REALIZES at the seam; a control outlet is a
    constant stream; signal into a modal inlet is the one hard type error. -/
def vocabularyJson : Json :=
  let portJson := fun (p : PortSpec) => Json.mkObj <|
    [("name", Json.str p.name)]
    ++ (if p.accepts.isEmpty then [] else
        [("accepts", Json.arr (p.accepts.map (Json.str ∘ domStr))),
         ("multi", Json.bool p.multi)])
    ++ (match p.knob with
        | some (m, e) => [("default", Json.num ⟨m, e⟩),
                          ("discipline", Json.str (discStr p.discipline))]
        | none => [])
    ++ (match p.display with
        | some md => [("min", Lean.toJson md.min), ("max", Lean.toJson md.max),
                      ("log", Json.bool md.log), ("unit", Json.str md.unit)]
        | none => [])
    ++ (match p.ownerPort with
        | some o => [("owner", Json.str o)]
        | none => [])
  Json.mkObj [
    ("rule", Json.str
      "outlet→inlet valid iff outlet.color ∈ inlet.accepts; modal→signal realizes, signal→modal is a type error"),
    ("colors", Json.arr #[Json.str "signal", Json.str "modal", Json.str "control"]),
    ("kinds", Json.arr (vocabularyKinds.map fun k =>
      Json.mkObj [
        ("kind", Json.str k),
        ("outlet", match outletOf k with
          | some d => Json.str (domStr d)
          | none => Json.null),
        ("ports", Json.arr ((portSpecs k).map portJson))]))]

/-- The live param table: every node's continuous knobs as `(<id>.<knob>, default)`
    in scan order (node order, then knob order). The position IS the `ParamIdx` the
    node's `paramRef`s carry; `compileSession` allocates each `param:<id>.<knob>`.
    A knob's slot exists exactly while its default is in the compiled graph:
    wiring the port itself (a Knob patched into `freq`) replaces the default, and
    wiring its OWNER port (sflange `mod` over the normalled LFO that `rate`
    parameterizes) removes the whole normal — either way the slot is not
    registered, so a registered-but-unread knob is unrepresentable here. The two
    reserved master-clock slots lead the table: `velocity` (default 1 ⇒ forward
    at unity) and `tau_base` (default 0), so every patch has a live global
    time-warp. -/
private def collectParams (raws : Array Raw) : Array (String × JsonNumber) := Id.run do
  let mut out : Array (String × JsonNumber) :=
    #[(masterVelocityParam, ⟨1, 0⟩), (masterTauBaseParam, ⟨0, 0⟩)]
  for r in raws do
    if r.kind == "out" then continue
    for spec in portSpecs r.kind do
      if spec.knob.isNone then continue
      let kname := spec.name
      let selfWired := !(portSources r.inObj kname).isEmpty
      let ownerWired := match spec.ownerPort with
        | some o => !(portSources r.inObj o).isEmpty
        | none => false
      if !selfWired && !ownerWired then
        let base := s!"{r.id}.{kname}"
        let dflt := (jNum? r.params kname).getD ⟨0, 0⟩
        if isGlided r.kind kname then
          -- three anchor slots for the closed-form ramp; v0=v1=current value and
          -- t0=0 means it starts flat at the knob's value (no ramp until re-anchored).
          out := out.push (s!"{base}#v0", dflt)
          out := out.push (s!"{base}#v1", dflt)
          out := out.push (s!"{base}#t0", ⟨0, 0⟩)
        else
          out := out.push (base, dflt)
          -- a frequency knob (source freq, fm carrier) carries a phase-anchor
          -- offset slot, so a live change bumps `#phase` and keeps the phase
          -- continuous (click-free) instead of jumping by Δf·τ.
          if isAnchored r.kind kname then
            out := out.push (s!"{base}#phase", ⟨0, 0⟩)
    -- Trip-count-as-data: a resonator carrying the optional STATIC
    -- `partials_max` capacity gets a LIVE `partials` slot (the room-size
    -- knob; default = graph `partials`, or 6). `partials_max` absent ⇒ no
    -- slot — exactly the old fully-static behavior.
    if r.kind == "resonator" && (jNum? r.params "partials_max").isSome then
      out := out.push (s!"{r.id}.partials", (jNum? r.params "partials").getD ⟨6, 0⟩)
  return out

/-- Decode the GUI graph, returning the patch plus the knob param table. The
    reserved `out` node is not an arrow node; it becomes a synthetic `mix` of its
    inputs (the dac mix-bus). Effects with no input point at `__silence__` (an
    empty sum), so a half-built patch compiles to silence with no error. -/
def decodeGraph (j : Json) : Except String (PatchGraph × Array (String × JsonNumber)) := do
  let outId := match (j.getObjVal? "out").toOption with
    | some (.str s) => s
    | _ => ""
  let raws := rawsOf j
  let params := collectParams raws
  let pidx : String → Option Nat := fun nm => params.findIdx? (·.1 == nm)
  let mut pnodes : Array PatchNode := #[]
  for r in raws do
    if r.kind == "out" then continue
    let (node, extras) := buildNode pidx r.id r.kind r.sel r.params r.inObj
    pnodes := pnodes.push { id := r.id, node }
    for e in extras do pnodes := pnodes.push e
  let outIns := match raws.find? (·.id == outId) with
    | some r => if r.kind == "out" then portSources r.inObj "in" else #[r.id]
    | none => #[]
  pnodes := pnodes.push { id := "__out__", node := .mix outIns }
  pnodes := pnodes.push { id := "__silence__", node := .mix #[] }
  pure ({ nodes := pnodes, output := "__out__" }, params)

-- ── Host-contract dispatch table (param_disciplines in the manifest) ────────
/-- The per-param write-discipline table this graph's plan carries — the same
    walk and skip rules as `collectParams`, projected to base names. A host
    reads this from the manifest and dispatches param writes itself
    (design/host-param-dispatch.md is the normative math); no client ever
    chooses a write verb. -/
private def paramDisciplinesOf (raws : Array Raw) :
    Array Tropical.Plan.ParamDiscipline := Id.run do
  let mut out : Array Tropical.Plan.ParamDiscipline := #[
    { name := masterVelocityParam, discipline := "velocity",
      companions := #[masterTauBaseParam] },
    { name := masterTauBaseParam, discipline := "raw" }]
  for r in raws do
    if r.kind == "out" then continue
    for spec in portSpecs r.kind do
      if spec.knob.isNone then continue
      let selfWired := !(portSources r.inObj spec.name).isEmpty
      let ownerWired := match spec.ownerPort with
        | some o => !(portSources r.inObj o).isEmpty
        | none => false
      if !selfWired && !ownerWired then
        let base := s!"{r.id}.{spec.name}"
        let d : Tropical.Plan.ParamDiscipline := match spec.discipline with
          | .glide =>
            -- 0.02 s: the engine's glide window
            { name := base, discipline := "glide", glideDurSec := some ⟨2, 2⟩,
              companions := #[s!"{base}#v0", s!"{base}#v1", s!"{base}#t0"] }
          | .anchor =>
            { name := base, discipline := "anchor", companions := #[s!"{base}#phase"] }
          | .raw =>
            { name := base, discipline := "raw" }
        out := out.push d
    -- Trip-count-as-data: the live `partials` slot (present only with the
    -- STATIC `partials_max` capacity) is a plain raw write — same walk and
    -- skip rules as `collectParams`.
    if r.kind == "resonator" && (jNum? r.params "partials_max").isSome then
      out := out.push { name := s!"{r.id}.partials", discipline := "raw" }
  return out

-- ── The realized-state report (the load_patch_graph reply) ──────────────────
/-- FACTS about what compiled — never warnings (legal-but-incomplete states
    compile to silence by contract; the report is how a surface renders truth
    instead of guessing policy). Per user node: `active` (reachable from the
    `out` node, walking inlet edges backwards) or `excluded`. Per inlet:
    `wired` (with sources) or `normalled` (running on its default). Per live
    param: the collected value and write discipline, base names only — the
    glide/anchor companion slots are the discipline's implementation detail,
    not surface. Plus the taps. The silence-with-`{ok:true}` class dies here:
    a patch that gracefully compiled to nothing now SAYS so, as facts. -/
def realizedReport (args : Json) (taps : Array (String × String × String)) : Json := Id.run do
  let raws := rawsOf args
  let outId := match (args.getObjVal? "out").toOption with
    | some (.str s) => s
    | _ => ""
  -- reachability fixed point (≤ |nodes| rounds; GUI graphs are small)
  let mut reach : Array String := #[outId]
  for _ in [0:raws.size] do
    for r in raws do
      if reach.contains r.id then
        for spec in portSpecs r.kind do
          for src in portSources r.inObj spec.name do
            if !reach.contains src then reach := reach.push src
  let nodesJ := raws.map fun r =>
    Json.mkObj [("id", Json.str r.id), ("kind", Json.str r.kind),
      ("status", Json.str (if reach.contains r.id then "active" else "excluded"))]
  let inputsJ := raws.foldl (init := #[]) fun acc r =>
    (portSpecs r.kind).foldl (init := acc) fun acc spec =>
      if spec.accepts.isEmpty then acc else
      let srcs := portSources r.inObj spec.name
      acc.push (Json.mkObj (
        [("node", Json.str r.id), ("port", Json.str spec.name)] ++
        (if srcs.isEmpty then [("state", Json.str "normalled")]
         else [("state", Json.str "wired"),
               ("sources", Json.arr (srcs.map Json.str))])))
  let kindOf : String → Option String := fun id => (raws.find? (·.id == id)).map (·.kind)
  let paramsJ := (collectParams raws).filterMap fun (nm, v) =>
    if nm.endsWith "#v1" || nm.endsWith "#t0" || nm.endsWith "#phase" then none
    else if nm.endsWith "#v0" then
      some (Json.mkObj [("name", Json.str (nm.dropRight 3)),
        ("value", Json.num v), ("discipline", Json.str "glide")])
    else
      let disc :=
        if nm == masterVelocityParam then "velocity"
        else match nm.splitOn "." with
          | [id, kname] => (kindOf id).map (fun k => discStr (disciplineOf k kname)) |>.getD "raw"
          | _ => "raw"
      some (Json.mkObj [("name", Json.str nm),
        ("value", Json.num v), ("discipline", Json.str disc)])
  let tapsJ := taps.map fun (name, inst, out) =>
    Json.mkObj [("name", Json.str name), ("slot", Json.str s!"{inst}.{out}")]
  return Json.mkObj [("ok", Json.bool true), ("nodes", Json.arr nodesJ),
    ("inputs", Json.arr inputsJ), ("params", Json.arr paramsJ),
    ("taps", Json.arr tapsJ)]

-- ── Scope taps ──────────────────────────────────────────────────────────────
/-- A scope tap: `(name, srcInstance, srcOutput)` — the exact `scopeTaps` triple
    the session model uses, so `list_scope_taps` (which builds the slot as
    `<inst>.<out>`) needs no change. For the arrow path the source instance is
    always the synthetic root, and the output is a dedicated `tap:<id>` port. -/
abbrev Tap := String × String × String

/-- The user-facing nodes worth tapping: the raw GUI nodes, minus the reserved
    `out`/`__silence__` and any synthesized helper (`__lfo_*`) — i.e. every node
    that has a visible output jack. `lowerNode` gives each one's closed-form
    output signal (the signal at its jack), which is exactly what a scope shows. -/
private def tapNodeIds (raws : Array Raw) : Array String :=
  raws.filterMap fun r =>
    if r.kind == "out" || "__".isPrefixOf r.id then none else some r.id

-- ── Graph → loadable FlatPlan (mirrors Tropicaltest.compileArrowCarrier) ─────
/-- Compile the GUI graph to a loadable `FlatPlan`, plus the scope taps it
    materializes. The final mix is always tap `out` (`__root__.out`); each user
    node gets a `tap:<id>` output port carrying `emit(normalize(lowerNode id))`
    — the signal at that node's jack — so the collapse keeps the slot alive and
    `render_window` can read it. Taps cost extra kernel compute (each re-emits its
    upstream cone), so they're for the inspection build, not a lean audio path. -/
def compilePlanPure (arena : Arena) (resolved : Array (String × ProgramIdx)) (j : Json) :
    Except String (Tropical.Plan.FlatPlan × Array Tap
      × Array (Array (Option Tropical.Ir.Stage))) := do
  let (g, paramTable) ← decodeGraph j
  let term ← lowerGraph g
  let (out, b0) := emitTerm (normalize term) {}
  -- Per-node taps are opt-in (`"taps": true` in the request): they cost extra
  -- kernel compute, so an audio-only consumer (the playground) omits them and
  -- pays nothing. The final mix (`out`) is always tappable — its slot exists
  -- regardless — so a scope attached to a taps-off patch still sees the output.
  let emitTaps := match j.getObjVal? "taps" with | .ok (.bool b) => b | _ => false
  -- Emit each user node's sub-term into the SAME builder (so instance names stay
  -- unique by `decls.size`), collecting `(id, signal)`. A node that fails to
  -- lower is simply not tapped.
  let tapIds := if emitTaps then tapNodeIds (rawsOf j) else #[]
  let (tapSigs, b) : Array (String × Sig) × Builder :=
    tapIds.foldl (fun (acc : Array (String × Sig) × Builder) id =>
      match lowerInput g id with
      | .ok t => let (s, b') := emitTerm (normalize t) acc.2; (acc.1.push (id, s), b')
      | .error _ => acc) (#[], b0)
  let registry ← buildRegistry arena resolved #["FixedSinOsc", "MorphOsc", "PluckedMorphOsc"]
  -- One `.param` decl per live knob, appended AFTER the instance decls: declaration
  -- order = `ParamIdx` = the index each `paramRef` carries, and keeping params after
  -- instances leaves the emitted voices' `InstanceIdx` untouched. `compileSession`
  -- allocates each a `param:<id>.<knob>` module slot, driven live by `set_param`.
  let paramDecls : Array BodyDecl := paramTable.map (fun (nm, v) => .param nm (some v))
  -- Output port 0 is the audible mix; ports 1.. are the taps (`tap:<id>`).
  let tapOutputs := tapSigs.map fun (id, _) =>
    ({ name := s!"tap:{id}", type? := some (.scalar .float) } : OutputDecl)
  let tapAssigns : Array (Tropical.Ir.OutputTarget × Sig) :=
    tapSigs.mapIdx fun i (_, s) => (.port ⟨i + 1⟩, s)
  -- Assemble through EmitArrow's lowering boundary (interns every `Sig` into the
  -- arena's DAG); the live param decls append after the instance decls.
  let (arena1, idx) := Tropical.EmitArrow.assemble arena "__patch__" #[]
    (#[{ name := "out", type? := some (.scalar .float) }] ++ tapOutputs)
    b.decls (#[(.port ⟨0⟩, out)] ++ tapAssigns) registry
    (extraDecls := paramDecls)
  let (coreArena, core) ← (Tropical.Ir.Strata.runResolved { upto := 5 } arena1 idx).mapError (·.message)
  let input := Tropical.Compile.SessionInput.forRoot core coreArena
    (params := paramTable.map (fun (nm, v) => (nm, Json.num v)))
    (alloc := Tropical.Lowering.allocate (paramTable.map (·.1)) #[])
  let (plan, stageBlocks) ← Tropical.Compile.compileSessionStaged input
  -- The host-contract dispatch table rides the manifest: any runtime host
  -- (C++ today, Swift/Metal or wasm tomorrow) reads per-slot disciplines from
  -- the plan itself and dispatches param writes locally.
  let plan := { plan with paramDisciplines := paramDisciplinesOf (rawsOf j) }
  -- The final mix (`out`) plus one tap per user node, all routed to the synthetic
  -- root's output slots (`__root__.<port>`), ready for `render_window`.
  let root := Tropical.Compile.rootInstancePath
  let taps : Array Tap := #[("out", root, "out")]
    ++ tapSigs.map (fun (id, _) => (id, root, s!"tap:{id}"))
  pure (plan, taps, stageBlocks)

-- ── Stdlib-into-arena (the shared chain; cached below via `getStdlib`) ───────
def elabStdlib : IO (Except String (Arena × Array (String × ProgramIdx))) :=
  Tropical.StdlibChain.elabStdlib

initialize stdlibCache : IO.Ref (Option (Arena × Array (String × ProgramIdx))) ← IO.mkRef none

def getStdlib : IO (Except String (Arena × Array (String × ProgramIdx))) := do
  match ← stdlibCache.get with
  | some v => pure (.ok v)
  | none =>
    match ← elabStdlib with
    | .error e => pure (.error e)
    | .ok v => stdlibCache.set (some v); pure (.ok v)

/-- Every live param as `(name, valueJson)` — the session param mirror the engine
    seeds after a successful load, so `set_param` (which guards on the mirror and
    drives the `param:<name>` slot) reaches EVERY live knob of a `load_patch_graph`
    plan, with no relower. Same `collectParams` the compile uses, so the mirror and
    the allocated slots agree name-for-name; value kept as raw JSON. -/
def knobParams (j : Json) : Array (String × Json) :=
  (collectParams (rawsOf j)).map (fun (nm, v) => (nm, Json.num v))

/-- Decode + lower + compile the GUI graph to a loadable `FlatPlan` + its
    taps + typed stage blocks (the split classification). -/
def compilePlan (j : Json) : IO (Except String (Tropical.Plan.FlatPlan × Array Tap
    × Array (Array (Option Tropical.Ir.Stage)))) := do
  match ← getStdlib with
  | .error e => pure (.error s!"stdlib elaboration: {e}")
  | .ok (arena, resolved) => pure (compilePlanPure arena resolved j)

end Tropical.Playground
