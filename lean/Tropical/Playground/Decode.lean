import Tropical.Playground.Vocabulary

/-!
# Playground.Decode

Patch-node construction, master-clock wiring, validation, and graph decoding.
-/

namespace Tropical.Playground

open Lean (Json JsonNumber)
open Tropical.Ir
open Tropical.EmitArrow
open Tropical.Exact (DyadicI)

-- ── The live master clock (global time-warp) ────────────────────────────────
/-- The two reserved master-clock slots. `velocity` is the live scrub (forward /
    freeze / reverse / varispeed); `tau_base` is the host-held τ-origin the engine
    re-bases on each velocity-discipline `set_param` so the scrub stays
    value-continuous — the stateless `ScrubClock` host-split, promoted to the
    arrow patch's base clock. -/
def masterVelocityParam : String := "master.velocity"
def masterTauBaseParam  : String := "master.tau_base"

/-- The reserved master-VOLUME slot. Since the device sink is now a pure summer
    (`defaultSinkGain = 1`, the backend applies no headroom scale of its own),
    amplitude lives in the graph: `decodeGraph` scales the output mix-bus by a
    `knob` reading this slot — a master VCA the frontend owns. Default `3.7`
    lands the reference patch near −20 dBFS RMS (≈ a comfortable listening level
    next to other apps, with headroom before clipping); a live `set_param
    master.gain` moves it. It is a plain τ-constant, so it never disturbs the
    scrub/warp timebase. -/
def masterGainParam : String := "master.gain"

/-- Default master volume: 3.7 (= 37 × 10⁻¹). Calibrated so the reference
    playground patch renders at ≈ −20 dBFS RMS / −2.5 dBFS peak. -/
def masterGainDefault : JsonNumber := ⟨37, 1⟩

/-- Every generator's base clock, so global time is a property of the whole patch
    (not per-oscillator): `M(n) = toInt(tau_base·SR·2³²) + toInt(velocity·2³²)·n`.
    At the defaults `velocity = 1, tau_base = 0` this is exactly `sampleIndex<<32`
    (the old `clockLit`), so the master clock is free until you scrub it. Reading
    it live means one `velocity` knob scrubs/reverses EVERY closed-form voice,
    envelope, and delay tap coherently — nothing downstream holds history. -/
def masterClock (pidx : String → Option Nat) : Clock :=
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
def buildNode (pidx : String → Option Nat) (id kind : String)
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
    (.modalReverb sig (reverbRoom rt60 (displayRangeOf "reverb" "rt60") 32 (60, 0) (6000, 0)) (some dir), #[])
  | "filter" =>
    -- the filter IS a modalReverb with a computed 2-mode room: the residue
    -- calculus does the "filtering" at build time, knobs stay live through it.
    (.modalReverb sig
      (filterPair (p "cutoff" (dv "cutoff")) (p "resonance" (dv "resonance"))) none, #[])
  | "modalmix" => (.modalMix (portSources inObj "in"), #[])
  | "gauge" =>
    -- §5 excitation gauge: re-level the modal input's peak. g=0 identity (unity-DC,
    -- the strike gauge), g=1 unity-peak. A pure Modal ⇝ Modal effect (`normalizePeak`);
    -- the norm is self-measured on the SETTLED poles, so a glided filter input is
    -- Metal-safe and an un-settleable input declines to identity.
    let inId := (portSources inObj "in")[0]?.getD "__silence__"
    (.modalGauge inId (p "g" (dv "g")), #[])
  | "gong" =>
    -- One STRIKE of the struck nonlinear resonator: two anchored modal banks
    -- (full-glide + stiff half-glide registers) behind per-strike pitch-bloom
    -- warps, composed by `gongStrikeNodes` from existing node kinds. All data
    -- is structural (a score's strikes are baked; the master clock still
    -- scrubs/reverses them live): `t` (strike time, s), `beta` (pitch-bloom
    -- depth, velocity already folded in), `g` (bloom settle rate), and the
    -- two pre-expanded mode tables.
    -- the strike anchor and the bloom settle rate as EXACT decimals: the last
    -- two `JsonNumber → Float → litF` round trips on the SERVED `gong` path.
    let gRateD := jExactD params "g" (18, 1)
    let anchor := mul (litOfD (jExactD params "t" (0, 0))) .sampleRate
    -- a bare gong (no score data) strikes the built-in default bank at its
    -- anchor — the kind's audible default, like the resonator's 6 partials.
    let full := jModes params "modes_full"
    let half := jModes params "modes_half"
    let (full, half) :=
      if full.isEmpty && half.isEmpty
      then defaultGongModes (jFloat params "freq" 110.0)
      else (full, half)
    -- β (pitch-bloom depth) is a LIVE slot: the score's baked value initializes it,
    -- and it deepens/relaxes under the knob with no relower (table fallback 0.05 ≈
    -- a solid strike, for a data-less drop).
    gongStrikeNodes id clk anchor (p "beta" (dv "beta")) (litOfD gRateD) full half
  | "bloomgong" =>
    -- A pitch-bloomed gong register that stays MODAL to the boundary — so it can
    -- cross a reverb (or a reverb CHAIN) by the residue calculus at the tap
    -- (`bloomCompose`) instead of realizing at the warp. The reassociated
    -- lowering folds the room chain first and crosses the bloom ONCE. Baked-pole-
    -- bloom contract (besselFuse parity): β, g, scale baked (a change relowers);
    -- amps stay live. Wired to `out` it plays the bare bloom-warped register;
    -- wired to a reverb it crosses — including a LIVE-rt60 reverb since WS-LP
    -- (serOnly pairs lift to s0 CplxE of the live pole; region-crossing pairs
    -- drop gracefully per pair until the Phase 3 region-union emit).
    -- β, g and scale stay `Float` here: they feed `modalSource`'s
    -- `(Float × Float)` bloom pair, whose type this pass does not reach.
    let beta := jFloat params "beta" 0.05
    let gRate := jFloat params "g" 1.8
    let scale := jFloat params "scale" 1.0
    let modes := jModes params "modes"
    let modes := if modes.isEmpty then (defaultGongModes (jFloat params "freq" 110.0)).1 else modes
    let anchor := mul (litOfD (jExactD params "t" (0, 0))) .sampleRate
    (.modalSource modes anchor clk none none (some (beta * scale / gRate, gRate)), #[])
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
      -- the exact decimals the score wrote, straight through: the loop-transit
      -- count `N = round(SR/f0)` is decided from them, so a `Float` round trip
      -- here would put a double rounding upstream of the emitted partial count.
      then defaultStringModes (jDec params "freq" (196, 0)) (jDec params "decay" (996, 3))
      else modes
    let anchor := mul (litOfD (jExactD params "t" (0, 0))) .sampleRate
    let addr? := (portSources inObj "addr")[0]?
    (.modalSource modes anchor clk addr?, #[])
  | _ => (.mix (portSources inObj "in"), #[])

structure Raw where
  id : String
  kind : String
  sel : Json
  params : Json
  inObj : Json

def decodeRaw (nj : Json) : Option Raw :=
  match nj.getObjVal? "id", nj.getObjVal? "kind" with
  | .ok (.str id), .ok (.str kind) =>
    let sel := (nj.getObjVal? "sel").toOption.getD (Json.mkObj [])
    let params := (nj.getObjVal? "params").toOption.getD (Json.mkObj [])
    let inObj := (nj.getObjVal? "in").toOption.getD (Json.mkObj [])
    some { id, kind, sel, params, inObj }
  | _, _ => none

def rawsOf (j : Json) : Array Raw :=
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
def domStr : PortDomain → String
  | .signal => "signal" | .modal => "modal" | .control => "control"
def discStr : Discipline → String
  | .raw => "raw" | .glide => "glide" | .anchor => "anchor"

/-- The connection-typing rule, ENFORCED at decode. The `portSpecs`/`outletOf`
    tables already STATE it (and `vocabularyJson` serves it); this makes a bad edge
    a pre-lowering type error with a clear message instead of a lowering-time
    surprise (or, for modal inlets, the `lowerModal` fallthrough string). For every
    wired inlet, the source's outlet color must be in the inlet's `accepts`:
    `modal→signal` realizes, `control→signal` is a constant stream, but `signal→modal`
    (a Sig has no poles to compose) is a type error — as is feeding an outletless
    sink (`out`) as a source. A derived reader of the single-source table, so it
    cannot drift from the served rule. Silence-on-legal-states is preserved: an
    unwired inlet is not an edge, so it is never flagged. -/
def checkEdgeTypes (raws : Array Raw) : Except String Unit := do
  for r in raws do
    for p in portSpecs r.kind do
      -- A knob-only port (`accepts = #[]`, `knob` set — `rt60`, `decay`, `cutoff`,
      -- …) is a SET value, never a wired inlet. A wire into it slips past the
      -- color loop below (empty accepts) yet makes `collectParams`' `selfWired`
      -- SUPPRESS the slot — a silently dead knob. Reject a non-empty wire here
      -- (an empty `in[knob]` entry is a harmless surface convention, kept legal).
      if p.accepts.isEmpty && p.knob.isSome && !(portSources r.inObj p.name).isEmpty then
        throw s!"connection error: '{r.id}' ({r.kind}) has a wire into '{p.name}', which is a knob (a set value), not an inlet — set it via its param slot (or wire its owner port), do not wire the knob itself"
      unless p.accepts.isEmpty do
        for srcId in portSources r.inObj p.name do
          match raws.find? (·.id == srcId) with
          | none =>
            -- A wire that NAMES no node is a broken document (a typo'd source),
            -- not a legal-incomplete state: surface it here rather than let it die
            -- downstream as `lower: node '…' not found` (or vanish silently if the
            -- edge is unreferenced). An inlet with NO wire is not an edge and is
            -- never reached, so silence-on-legal-states is untouched.
            throw s!"connection error: '{r.id}' ({r.kind}) inlet '{p.name}' is wired from '{srcId}', which is not a node in the patch — a wire must name an existing node"
          | some src =>
            match outletOf src.kind with
            | none =>
              throw s!"connection type error: '{src.id}' ({src.kind}) has no outlet but is wired into '{r.id}' ({r.kind}) inlet '{p.name}'"
            | some col =>
              unless p.accepts.contains col do
                let accepted := String.intercalate "/" (p.accepts.toList.map domStr)
                throw s!"connection type error: '{src.id}' ({src.kind}, {domStr col} outlet) → '{r.id}' ({r.kind}) inlet '{p.name}' which accepts {accepted} — outlet.color ∉ inlet.accepts (modal→signal realizes; signal→modal is a type error)"
  pure ()

/-- The top-level `"out"` id must name an existing node — or be absent/empty,
    which is a legal-incomplete state (nothing routed to the dac yet) that
    compiles to silence. A NON-empty id naming no node is a typo'd output target:
    a broken document, which otherwise renders the WHOLE patch as silence with no
    error (`decodeGraph`'s `outIns = #[]`). Surface it as an error instead. -/
def checkOutTarget (j : Json) (raws : Array Raw) : Except String Unit :=
  match (j.getObjVal? "out").toOption with
  | some (.str outId) =>
    if outId == "" || raws.any (·.id == outId) then pure ()
    else throw s!"output target error: the top-level \"out\" names node '{outId}', which is not in the patch — route the dac from an existing node (or omit \"out\" for a silent patch)"
  | _ => pure ()

/-- Reject a node the served surface does not cover, at the surface boundary —
    BEFORE `checkEdgeTypes`, so the honest message wins over the misleading
    `signal→modal` type error a WITHHELD modal kind (whose `outletOf` falls through
    to `signal`) would otherwise die as downstream. A withheld kind (`withheldKinds`
    — built by `buildNode` but not yet surface-ready) and a genuinely UNKNOWN kind
    (a typo, which `buildNode`'s `_ => .mix` fallthrough would otherwise turn into
    silent silence) each get their own message. Distinct from a legal-incomplete
    state (an unwired inlet → silence, no error): an unknown/withheld kind is a
    broken/unavailable document. A valid patch is untouched — every `vocabularyKinds`
    entry (incl. `out`) passes. -/
def checkServedKinds (raws : Array Raw) : Except String Unit := do
  for r in raws do
    if withheldKinds.contains r.kind then
      throw s!"unserved kind: '{r.id}' has kind '{r.kind}', which the engine builds but WITHHOLDS from the surface vocabulary (its modal factor-site landing has no admission guard yet) — not available as a patch node"
    unless vocabularyKinds.contains r.kind do
      throw s!"unknown kind: '{r.id}' has kind '{r.kind}', which is not a served node kind — see get_vocabulary for the {vocabularyKinds.size} kinds the surface builds"
  pure ()

/-- Gate probe for `exact-playground`, beside `modalClassificationDrift` because
    it is the same pattern: a public reader of PRIVATE facts, so the gate never
    has to force a builder public.

    It returns what the SERVED BUILDERS ACTUALLY EMIT — `resonatorBank` and
    `reverbRoom` are called, not transcribed. An earlier cut of this probe
    recomputed their expressions inline and compared THAT against `libm`, which
    tested the transcription and not the emitter; a builder could have been
    edited out from under it and the differential would have stayed green.

    The libm REFERENCE side deliberately lives one floor out, in the gate. Two
    reasons, and the second is the load-bearing one: a differential's two sides
    should not sit in the module under test, and `exact-corpse` reads the
    GENERATED C of this module, where a reference call to `cos` is
    indistinguishable from a production one. Keeping the reference out is what
    lets the corpse gate cover `Playground.lean` with no exemption.

    `f0 = 1, decay = 1` for the resonator and `rt60 = 1` for the room so the
    live factors fold to identity and each mode's fields fold to exactly the
    baked coefficient under test. -/
def bakedResonatorProbe (npart : Nat) : Array ModalMode :=
  resonatorBank (lit 1) (lit 1) npart

/-- The shipped reverb room's 32 modes over 60…6000 Hz, as emitted. -/
def bakedReverbProbe (nmode : Nat) : Array ModalMode :=
  reverbRoom (lit 1) none nmode (60, 0) (6000, 0)

/-- The `ln 80` literal `filterPair` emits — the one authored transcendental
    constant in the file. This is the SAME `Sig` the builder consumes (one
    definition, shared), not a second spelling of the formula; `filterPair`'s `q`
    wraps it in an `expSig`, which `sigConstF?` does not fold, so the constant
    must be reachable on its own to be observable at all. -/
def bakedFilterLn80 : Sig := lnEightyLit

/-- Fold an emitted field back to the double it carries (`none` if it did not
    come out constant — which for these probes is itself a failure the gate
    reports). -/
def probeFold (s : Sig) : Option Float := sigConstF? s

def modalClassificationDrift : Array String := Id.run do
  let mut drift : Array String := #[]
  for kind in buildNodeKinds do
    if kind == "out" then continue
    let (node, _) := buildNode (fun _ => none) "n" kind
      (Json.mkObj []) (Json.mkObj []) (Json.mkObj [])
    let g : PatchGraph := { nodes := #[{ id := "n", node }], output := "n" }
    if nodeIsModal g "n" != (outletOf kind == some .modal) then
      drift := drift.push kind
  return drift

/-- The stable wire identity for the engine-owned vocabulary envelope. The
    version changes only when the response contract changes incompatibly; the
    fingerprint below changes when its semantic payload changes. -/
def vocabularySchema : String := "tropical_vocabulary"
def vocabularySchemaVersion : Nat := 1

/-- FNV-1a/64 over bytes, with UInt64 arithmetic providing the specified
    modulo-2^64 wrap. This is a compatibility fingerprint, not a security
    primitive: it gives every client a small deterministic identity for the
    exact canonical vocabulary payload it decoded. -/
def fnv1a64 (bytes : ByteArray) : UInt64 := Id.run do
  let mut hash : UInt64 := 14695981039346656037
  for byte in bytes do
    hash := (hash ^^^ byte.toUInt64) * 1099511628211
  return hash

private def hexDigit (n : Nat) : Char :=
  if n < 10 then Char.ofNat ('0'.toNat + n)
  else Char.ofNat ('a'.toNat + n - 10)

/-- A fixed-width lowercase spelling keeps the fingerprint stable across
    platforms and independent of the host's integer formatting. -/
def uint64Hex (value : UInt64) : String := Id.run do
  let mut out := ""
  for i in [0:16] do
    let shift := 4 * (15 - i)
    let nibble := ((value >>> shift.toUInt64) &&& 0xf).toNat
    out := out.push (hexDigit nibble)
  return out

/-- The vocabulary's canonical SEMANTIC payload. Array order is authored table
    order; `Json.compress` serializes object fields through Lean.Json's ordered
    object map, so the byte spelling hashed below is deterministic. Schema and
    fingerprint fields deliberately live outside this payload. -/
def vocabularyPayloadJson : Json :=
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

/-- Algorithm-labelled identity of `vocabularyPayloadJson`. The label is part
    of the value so a future algorithm migration cannot be mistaken for a
    semantic-table change. -/
def vocabularyFingerprint : String :=
  s!"fnv1a64:{uint64Hex (fnv1a64 vocabularyPayloadJson.compress.toUTF8)}"

/-- The vocabulary as JSON — the ONE description of the node kinds, GENERATED
    from the port-spec table (the hand-maintained `nodeSchema` this replaces
    was the third copy, and the class of bug this file exists to kill). Per
    kind: outlet color and ports; per port: inlet facts (accepts/multi), knob
    facts (default, write discipline, display metadata), and `owner` when the
    knob parameterizes another port's normal. Clients RENDER this — nothing
    the engine knows may be re-encoded client-side. The connection rule rides
    along: `outlet→inlet` valid iff `outlet.color ∈ inlet.accepts`; a modal
    outlet into a signal inlet REALIZES at the seam; a control outlet is a
    constant stream; signal into a modal inlet is the one hard type error. The
    original `rule`/`colors`/`kinds` keys remain top-level for compatible
    clients; schema metadata wraps that unchanged semantic payload. -/
def vocabularyJson : Json :=
  match vocabularyPayloadJson with
  | .obj fields => .obj <|
      fields.insert "schema" (Json.str vocabularySchema)
        |>.insert "schema_version" (Lean.toJson vocabularySchemaVersion)
        |>.insert "fingerprint" (Json.str vocabularyFingerprint)
  | payload => Json.mkObj [
      ("schema", Json.str vocabularySchema),
      ("schema_version", Lean.toJson vocabularySchemaVersion),
      ("fingerprint", Json.str vocabularyFingerprint),
      ("payload", payload)]

/-- The live param table: every node's continuous knobs as `(<id>.<knob>, default)`
    in scan order (node order, then knob order). The position IS the `ParamIdx` the
    node's `paramRef`s carry; `compileSession` allocates each `param:<id>.<knob>`.
    A knob's slot exists exactly while its default is in the compiled graph:
    wiring the port itself (a Knob patched into `freq`) replaces the default, and
    wiring its OWNER port (sflange `mod` over the normalled LFO that `rate`
    parameterizes) removes the whole normal — either way the slot is not
    registered, so a registered-but-unread knob is unrepresentable here. The
    reserved master slots lead the table: `velocity` (default 1 ⇒ forward at
    unity) and `tau_base` (default 0) — the global time-warp — plus `gain`
    (default 3.7), the master VCA `decodeGraph` folds onto the output. -/
def collectParams (raws : Array Raw) : Array (String × JsonNumber) := Id.run do
  let mut out : Array (String × JsonNumber) :=
    #[(masterVelocityParam, ⟨1, 0⟩), (masterTauBaseParam, ⟨0, 0⟩),
      (masterGainParam, masterGainDefault)]
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
          -- Metal snapshots ordinary slots as f32, so an absolute t0 would
          -- lose the whole 20 ms ramp past 2^24. Four exact 16-bit limbs carry
          -- the signed i64 source coordinate used by `glideExpr`.
          for i in [0:4] do
            out := out.push (s!"{base}#t0#u{i}", ⟨0, 0⟩)
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
  -- The device sink is a pure summer now, so amplitude is authored here: the
  -- dac mix-bus is scaled by a master VCA — `ring [mixbus, master]`, where
  -- `master` is a knob reading the reserved `master.gain` slot (a τ-constant,
  -- so it rides no clock and never perturbs the scrub timebase). An empty patch
  -- still lowers to silence: `__mixbus__` is an empty sum, `× gain` is still 0.
  let masterIdx := (pidx masterGainParam).getD 0
  pnodes := pnodes.push { id := "__mixbus__", node := .mix outIns }
  pnodes := pnodes.push { id := "__master__", node := .knob masterIdx }
  pnodes := pnodes.push { id := "__out__", node := .ring #["__mixbus__", "__master__"] }
  pnodes := pnodes.push { id := "__silence__", node := .mix #[] }
  pure ({ nodes := pnodes, output := "__out__" }, params)


end Tropical.Playground
