import Tropical.EmitArrow
import Tropical.Ir.Strata
import Tropical.Ir.Core
import Tropical.Compile
import Tropical.Parse.Raise
import Tropical.Ir.Elaborator
import Lean.Data.Json

/-!
# `load_patch_graph` — the playground's live arrow entry point (EXPERIMENT)

A downstream-only patch graph (the PureData-style playground GUI) decoded into the
`EmitArrow` `PatchGraph`, lowered through the SAME `lowerGraph → normalize → emitTerm`
the corpus gates exercise, wrapped as a session root, and compiled to a `FlatPlan`
via `compileSession` — the production loadable tail. The slide (`normalize`) pushes
each effect's warps up onto the generators' clocks, so a `flange`/`fm` dropped
downstream of an oscillator genuinely re-clocks that oscillator.

Knobs are baked literals (EmitArrow emits no `paramRef`), so a knob change re-sends
the whole graph and hot-swaps — clickless, since there is no per-sample state.

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

/-- Which (kind, knob) pairs are driven by a CLOSED-FORM GLIDE (3 slots + a ramp
    expr) instead of a raw slot. PROTOTYPE: just the flanger's `depth`, so a turn
    of it eases as a click-free smoothstep; every other knob stays a raw slot (the
    A/B). -/
private def isGlided (kind kname : String) : Bool :=
  (kind == "source"  && kname == "morph")  ||
  (kind == "pluck"   && (kname == "morph" || kname == "event_rate")) ||
  (kind == "comb"    && (kname == "delay" || kname == "decay")) ||
  (kind == "flange"  && kname == "depth")  ||
  (kind == "sflange" && kname == "depth")  ||
  (kind == "fm"      && kname == "depth")  ||
  (kind == "delay"   && kname == "amount")

/-- Which (kind, knob) pairs are FREQUENCIES — phase-anchored (`#phase` offset slot
    + `set_param_freq`) rather than glided, since gliding a frequency VALUE would
    reintroduce the τ·f' chirp. A phasor-based oscillator's pitch input. -/
private def isAnchored (kind kname : String) : Bool :=
  (kind == "source" && kname == "freq") || (kind == "pluck" && kname == "freq")
    || (kind == "fm" && kname == "carrier")

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
  let clk := masterClock pidx
  match kind with
  | "knob" =>
    match pidx s!"{id}.value" with
    | some i => (.knob i, #[])
    | none => (.mix #[], #[])   -- a knob missing from the table (unreachable): silence
  | "source" =>
    -- a Knob wired into `freq` shadows the baked freq slot: read the WIRED knob's
    -- `<id>.value` slot instead. (Only a knob may wire here — audio-rate into the
    -- pitch PORT is not integration; audio-rate modulation is the FM/sflange path.)
    let pitchE := match (portSources inObj "freq")[0]? with
      | some w => pref pidx s!"{w}.value" (jExpr params "freq" (lit 220))
      | none => p "freq" (jExpr params "freq" (lit 220))
    -- the anchor: (phase slot, compile-time freq). Present only when the source's
    -- own freq is a live slot; the compile-time freq is the reference the warped
    -- copies' phase correction is measured from.
    let anchor := (pidx s!"{id}.freq#phase").map fun i => ((.paramRef ⟨i⟩ : Sig), jExpr params "freq" (lit 220))
    (.source (voiceOf pitchE (p "morph" (jExpr params "morph" (lit 0))) anchor) clk, #[])
  | "pluck" =>
    -- a plucked MorphOsc source: pitch (anchored), morph (glided), and event_rate
    -- (glided) drive the baked-in envelope. The dynamic content of the instrument.
    let pitchE := match (portSources inObj "freq")[0]? with
      | some w => pref pidx s!"{w}.value" (jExpr params "freq" (lit 110))
      | none => p "freq" (jExpr params "freq" (lit 110))
    let anchor := (pidx s!"{id}.freq#phase").map fun i => ((.paramRef ⟨i⟩ : Sig), jExpr params "freq" (lit 110))
    (.source (pluckedVoiceE pitchE (p "morph" (jExpr params "morph" (lit 0)))
       (p "event_rate" (jExpr params "event_rate" (lit 2))) anchor) clk, #[])
  | "comb" =>
    -- a one-sided resonant comb: dry (k=0) + a decaying tap series at k·delay.
    -- `delay` is signed (future = pre-echo, the moat), and doubles as the resonant
    -- spacing (small ⇒ pitched comb, large ⇒ discrete echoes). `decay` = gᵏ.
    let d := deltaOf (p "delay" (jExpr params "delay" (lit 12 3)))   -- 0.012 s, future
    let g := p "decay" (jExpr params "decay" (lit 7 1))              -- 0.7
    let K := 6
    let tail : Array (Sig × (Clock → Clock)) := (Array.range K).map fun j =>
      let k := j + 1
      (gPow g k, fun c => add c (mul (lit (Int.ofNat k)) d))
    (.comb sig (#[(lit 1, fun c => c)] ++ tail), #[])
  | "flange" =>
    let d := deltaOf (p "depth" (jExpr params "depth" (lit 7 4)))
    (.flange sig (fun c => sub c d) (fun c => add c d), #[])
  | "delay" =>
    let d := deltaOf (p "amount" (jExpr params "amount" (lit 4 3)))
    (.warpFx sig (fun c => sub c d), #[])
  | "reverse" => (.warpFx sig (fun c => neg c), #[])
  | "fm" =>
    -- `carrier` is a frequency → phase-anchored (own #phase slot); `depth` glides.
    let carAnchor := (pidx s!"{id}.carrier#phase").map fun i => ((.paramRef ⟨i⟩ : Sig), jExpr params "carrier" (lit 330))
    (.fm sig (sineVoiceE (p "carrier" (jExpr params "carrier" (lit 330))) carAnchor)
      clk (p "depth" (jExpr params "depth" (lit 8))), #[])
  | "sflange" =>
    let depthSec := p "depth" (jExpr params "depth" (lit 2 3))
    match (portSources inObj "mod")[0]? with
    | some modId => (.sflange sig modId depthSec, #[])
    | none =>
      -- no modulator patched: synthesize a built-in LFO sine at the `rate` knob.
      let lfoId := s!"__lfo_{id}"
      (.sflange sig lfoId depthSec,
       #[ { id := lfoId, node := .source (sineVoiceE (p "rate" (jExpr params "rate" (lit 3 1)))) clk } ])
  | "mix" => (.mix (portSources inObj "in"), #[])
  | "ring" => (.ring (portSources inObj "in"), #[])
  -- MODAL-island nodes: they carry poles, compose by the residue calculus at
  -- build time, and realize to a Sig at their boundary (a Sig consumer or the
  -- tap) — realized against the live `clk` (master clock) so they scrub too.
  | "resonator" =>
    let f0 := p "freq" (jExpr params "freq" (lit 220))
    let decay := p "decay" (jExpr params "decay" (lit 4))
    -- optional `addr` inlet: a Sig node whose value BECOMES the bank's absolute
    -- time-address (seconds into the impulse response). Unpatched ⇒ reads the
    -- master clock as before; patched ⇒ the causal gate triggers on the address
    -- signal's crossing and the ring scrubs/pitches with its slope (modalAddrWarp).
    let addr? := (portSources inObj "addr")[0]?
    (.modalSource (resonatorBank f0 decay 6) (lit 0) clk addr?, #[])
  | "reverb" =>
    let rt60 := p "rt60" (jExpr params "rt60" (lit 2))
    -- reading DIRECTION: θ (radians, live) rotates the composed tail's poles in the
    -- s-plane — 0 = forward decay, π = reverse (pre-verb), interior = a continuous
    -- U(1) morph (σ↔ω at π/2). `window` (live) nulls each mode at its horizon-
    -- crossing (`σ²/(σ²+w²)`), offset off 0 so the kernel never divides 0/0: at the
    -- knob's floor it is near-bare rotation, opened up it is the polite morph.
    -- DIR crossfades the tail's time-direction: 0 = forward ring, 1 = reverse
    -- (pre-verb into the strike), interior = both. Keeps σ/ω fixed, so it stays
    -- audible across the whole range (no pole rotation).
    let dirX := p "dir" (jExpr params "dir" (lit 0))
    -- SWAY: the room's decay breathes — σ ↦ σ·(1 + sway·sin(2π·rate·t)) on the
    -- envelope's clock only (pitch fixed). Continuous CF modulation of RT60 that
    -- stays on-island (no ∫σ dτ, no state); scrubs/reverses with the master clock.
    let sway := p "sway" (jExpr params "sway" (lit 0))
    let swayRate := p "rate" (jExpr params "rate" (lit 3 1))   -- 0.3 Hz: a slow breath
    let dir : ModalDir := { dir := dirX, damp := some (sway, swayRate) }
    (.modalReverb sig (reverbRoom rt60 32 60.0 6000.0) (some dir), #[])
  | "modalmix" => (.modalMix (portSources inObj "in"), #[])
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

/-- The continuous knobs each node kind carries — the ones that become live param
    slots (`<nodeId>.<knob>`). Structural selectors (`voice`, warp `mode`) are NOT
    here: changing one alters the graph topology, so it relowers. -/
private def knobNamesOf : String → Array String
  | "source"  => #["freq", "morph"]
  | "pluck"   => #["freq", "morph", "event_rate"]
  | "comb"    => #["delay", "decay"]
  | "flange"  => #["depth"]
  | "sflange" => #["depth", "rate"]
  | "fm"      => #["carrier", "depth"]
  | "delay"     => #["amount"]
  | "reverse"   => #[]
  | "resonator" => #["freq", "decay"]   -- live poles (symbolic residue keeps them live)
  | "reverb"    => #["rt60", "dir", "sway", "rate"]   -- damping + direction + decay sway
  | "knob"      => #["value"]
  | _           => #[]

/-- The live param table: every node's continuous knobs as `(<id>.<knob>, default)`
    in scan order (node order, then knob order). The position IS the `ParamIdx` the
    node's `paramRef`s carry; `compileSession` allocates each `param:<id>.<knob>`. A
    knob shadowed by a wired control inlet of the same name (a Knob patched into
    `freq`) is skipped — the cord's own slot drives it. The two reserved master-clock
    slots lead the table: `velocity` (default 1 ⇒ forward at unity) and `tau_base`
    (default 0), so every patch has a live global time-warp. -/
private def collectParams (raws : Array Raw) : Array (String × JsonNumber) := Id.run do
  let mut out : Array (String × JsonNumber) :=
    #[(masterVelocityParam, ⟨1, 0⟩), (masterTauBaseParam, ⟨0, 0⟩)]
  for r in raws do
    if r.kind == "out" then continue
    for kname in knobNamesOf r.kind do
      if (portSources r.inObj kname).isEmpty then
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
    Except String (Tropical.Plan.FlatPlan × Array Tap) := do
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
  let input : Tropical.Compile.SessionInput := {
    instances := #[(Tropical.Compile.rootInstancePath, core)]
    wiresPost := #[]
    graphOutputs := #[(Tropical.Compile.rootInstancePath, "out")]
    params := paramTable.map (fun (nm, v) => (nm, Json.num v))
    alloc := Tropical.Lowering.allocate (paramTable.map (·.1)) #[]
    root := core
    arena := coreArena
    mode := .fused }
  let plan ← Tropical.Compile.compileSession input
  -- The final mix (`out`) plus one tap per user node, all routed to the synthetic
  -- root's output slots (`__root__.<port>`), ready for `render_window`.
  let root := Tropical.Compile.rootInstancePath
  let taps : Array Tap := #[("out", root, "out")]
    ++ tapSigs.map (fun (id, _) => (id, root, s!"tap:{id}"))
  pure (plan, taps)

-- ── Stdlib-into-arena (cached; mirrors Tropicaltest.arrowElabStdlib) ─────────
def elabStdlib : IO (Except String (Arena × Array (String × ProgramIdx))) := do
  let manifestText ← IO.FS.readFile "stdlib/parsed/manifest.json"
  let names : Except String (Array String) := do
    let jv ← Tropical.Parse.JsonV.parse manifestText |>.mapError (s!"manifest parse error: {·}")
    let some (Tropical.Parse.JsonV.arr items) := jv.getField? "programs"
      | .error "manifest missing 'programs' array"
    items.mapM fun | .str s => .ok s | _ => .error "manifest 'programs' entries must be strings"
  match names with
  | .error e => pure (.error e)
  | .ok names => do
    let mut arena : Arena := {}
    let mut resolved : Array (String × ProgramIdx) := #[]
    for name in names do
      let text ← IO.FS.readFile s!"stdlib/parsed/{name}.json"
      match Tropical.Parse.JsonV.parse text with
      | .error e => return .error s!"{name}.json: JSON parse error: {e}"
      | .ok jv =>
        match Tropical.Parse.decodeProgram jv with
        | .error e => return .error s!"{name}.json: {e}"
        | .ok prog =>
          let r := resolved
          match Tropical.Ir.elaborateInto arena prog (some fun n => (r.find? (·.1 == n)).map (·.2)) with
          | .error e => return .error s!"{name}: {e.message}"
          | .ok (arena', idx) => arena := arena'; resolved := resolved.push (name, idx)
    pure (.ok (arena, resolved))

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

/-- Decode + lower + compile the GUI graph to a loadable `FlatPlan` + its taps. -/
def compilePlan (j : Json) : IO (Except String (Tropical.Plan.FlatPlan × Array Tap)) := do
  match ← getStdlib with
  | .error e => pure (.error s!"stdlib elaboration: {e}")
  | .ok (arena, resolved) => pure (compilePlanPure arena resolved j)

end Tropical.Playground
