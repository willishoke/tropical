import Tropical.EmitArrow

/-!
# Tropical.Stdlib — the standard library as arrow builders

The stdlib programs authored directly as `Sig`/`assemble` arrow builders,
replacing the literate-`.md` source + committed parse bridge as the boot path.
Each builder reproduces its former `.md` program to the equivalence the two
gates pin: the corpus gate (plan-wire byte-identity — resolved-body semantics)
and the entry-equiv gate (port-surface byte-identity — names/types/defaults).

Two authoring styles, matching the two shapes the post-strata IR takes:

* **Leaf** programs are flat `Sig` trees (no instances) — the closed-form
  scalar datapaths (`sinSig`, `phasorPhaseSig`, …) from `EmitArrow.Numerics`,
  with Lean `let` sharing standing in for the source's `let` binders (a
  DAG-authored program legitimately carries `binderCount = 0`; that field is
  non-load-bearing — see the entry-codec de-risk).

* **Instance-bearing** programs declare `AInst`s and let strata's
  `inlineInstances` flatten them, linking each sub-program by name through
  `buildRegistry` over the chain built so far. Byte-identical to the `.md`
  program whose body declared the same instances in the same order.
-/

namespace Tropical.EmitArrow

open Tropical.Ir

/-- A stdlib builder: append the program to the arena, linking sub-programs by
    name through the chain built so far. The shape the corpus/entry-equiv gates
    and the boot chain both consume. -/
abbrev StdBuilder := Arena → Array (String × ProgramIdx) → Except String (Arena × ProgramIdx)

-- ── Shared port-declaration helpers ──────────────────────────────────────────
-- The elaborator's lowering of the surface bound types, reproduced as defaults:
-- `freq` ⇒ `select(hz > 0, hz, 0)`, `unipolar` ⇒ `clamp(d, 0, 1)`,
-- `signal` ⇒ `clamp(d, -1, 1)`, plain `float`/`int`/`clock` ⇒ a bare default.
-- Bound INPUT types wrap only the default; bound OUTPUT types wrap the assigned
-- body (`unipolar` output ⇒ `clamp(body, 0, 1)`), handled at each `assemble`.

/-- A `freq`-typed input, default `select(hz > 0, hz, 0)` with `hz = m·10⁻ᵉ`. -/
def freqDecl (name : String) (m : Int) (e : Nat := 0) : AInputDecl :=
  { name, type? := some (.scalar .float),
    defaultSig := some (selectE (gt (lit m e) (lit 0)) (lit m e) (lit 0)) }

/-- A `unipolar`-typed input, default `clamp(d, 0, 1)`. -/
def unipolarDecl (name : String) (d : Sig := lit 0) : AInputDecl :=
  { name, type? := some (.scalar .float), defaultSig := some (clampE d (lit 0) (lit 1)) }

/-- A `signal`-typed input, default `clamp(d, -1, 1)`. -/
def signalDecl (name : String) (d : Sig := lit 0) : AInputDecl :=
  { name, type? := some (.scalar .float), defaultSig := some (clampE d (lit (-1)) (lit 1)) }

/-- A plain `float` input with a bare-literal default. -/
def floatDecl (name : String) (d : Sig) : AInputDecl :=
  { name, type? := some (.scalar .float), defaultSig := some d }

/-- A `clock`-typed input (`int` scalar); the default varies by program. -/
def clockInDecl (name : String) (d : Sig) : AInputDecl :=
  { name, type? := some (.scalar .int), defaultSig := some d }

/-- The `clock()` builtin as a wire expression — the Q32.32 root clock. -/
def clockSig : Sig := lshift .sampleIndex (lit 32)

/-- A plain-`float` output named `out`. -/
def floatOut (name : String := "out") : OutputDecl := { name, type? := some (.scalar .float) }

-- ── Leaf programs — flat closed-form `Sig` bodies ────────────────────────────

/-- `Sin` — the minimax float sine: range-reduce `r = x − round(x/π)·π`, sign
    from `n & 1`, Horner in `r²`. Unlike `Numerics.sinSig` this reproduces the
    source `fold`'s exact unrolling — the leading `c₀ + 0·r²` init step survives
    to the plan (it is NOT const-folded away), so byte-identity requires it. -/
def buildSin (arena : Arena) (_resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  let x : Sig := .inputRef ⟨0⟩
  let n := roundE (mul x (lit 3183098861837907 16))
  let r := sub x (mul n (lit 3141592653589793 15))
  let sign := sub (lit 1) (mul (lit 2) (bitAnd n (lit 1)))
  let r2 := mul r r
  -- `fold over=[c₀…c₅] init=0 body=(c + a·r²)`, unrolled highest→lowest power.
  let step (acc coeff : Sig) : Sig := add coeff (mul acc r2)
  let a := step (lit 0) (lit (-2505210838544172) 23)
  let a := step a (lit 27557319223985893 22)
  let a := step a (lit (-1984126984126984) 19)
  let a := step a (lit 8333333333333333 18)
  let a := step a (lit (-16666666666666666) 17)
  let poly := step a (lit 1)
  .ok (assemble arena "Sin"
    #[floatDecl "x" (lit 0)] #[floatOut]
    #[] #[(.port ⟨0⟩, mul sign (mul r poly))] #[])

/-- `Tanh` — the `(27 + c²)/(27 + 9c²)` Padé clip over `c = clamp(x, -3, 3)`. -/
def buildTanh (arena : Arena) (_resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  let x : Sig := .inputRef ⟨0⟩
  let c := clampE x (lit (-3)) (lit 3)
  let c2 := mul c c
  let body := div (mul c (add (lit 27) c2)) (add (lit 27) (mul (lit 9) c2))
  .ok (assemble arena "Tanh"
    #[floatDecl "x" (lit 0)] #[floatOut]
    #[] #[(.port ⟨0⟩, body)] #[])

/-- `ScrubClock` — the master timebase: `τ_base` (seconds → Q32.32) plus a
    `velocity`-scaled sample ramp. Output `clk : clock` (integer, no clamp). -/
def buildScrubClock (arena : Arena) (_resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  let tauBase : Sig := .inputRef ⟨0⟩
  let velocity : Sig := .inputRef ⟨1⟩
  let base := toIntE (mul (mul tauBase .sampleRate) (lit 4294967296))
  let ramp := mul (toIntE (mul velocity (lit 4294967296))) .sampleIndex
  .ok (assemble arena "ScrubClock"
    #[floatDecl "tau_base" (lit 0), floatDecl "velocity" (lit 1)]
    #[{ name := "clk", type? := some (.scalar .int) }]
    #[] #[(.port ⟨0⟩, add base ramp)] #[])

/-- `VCA` — a single multiply, `audio · cv`. -/
def buildVCA (arena : Arena) (_resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  .ok (assemble arena "VCA"
    #[floatDecl "audio" (lit 0), floatDecl "cv" (lit 0)] #[floatOut]
    #[] #[(.port ⟨0⟩, mul (.inputRef ⟨0⟩) (.inputRef ⟨1⟩))] #[])

/-- `FixedPhasor` — the integer phasor over `sampleIndex` (not a clock input):
    `acc = inc·sampleIndex + off`, masked to `[0, 2³²)`, scaled to unipolar. -/
def buildFixedPhasor (arena : Arena) (_resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  let freq : Sig := .inputRef ⟨0⟩
  let offset : Sig := .inputRef ⟨1⟩
  let inc := toIntE (div (mul freq (lit 4294967296)) .sampleRate)
  let off := toIntE (mul offset (lit 4294967296))
  let acc := add (mul inc .sampleIndex) off
  let phase := div (toFloatE (bitAnd acc (lit 4294967295))) (lit 4294967296)
  .ok (assemble arena "FixedPhasor"
    #[freqDecl "freq" 440, unipolarDecl "offset"]
    #[{ name := "phase", type? := some (.scalar .float) }]
    #[] #[(.port ⟨0⟩, clampE phase (lit 0) (lit 1))] #[])

/-- `ClockPhasor` — the phasor over an external Q32.32 `clk` (the split-multiply
    `phasorPhaseSig`), unipolar output. -/
def buildClockPhasor (arena : Arena) (_resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  let clk : Sig := .inputRef ⟨0⟩
  let freq : Sig := .inputRef ⟨1⟩
  let offset : Sig := .inputRef ⟨2⟩
  .ok (assemble arena "ClockPhasor"
    #[clockInDecl "clk" (lit 0), freqDecl "freq" 440, unipolarDecl "offset"]
    #[{ name := "phase", type? := some (.scalar .float) }]
    #[] #[(.port ⟨0⟩, clampE (phasorPhaseSig freq offset clk) (lit 0) (lit 1))] #[])

-- ── Instance-bearing programs — declare `AInst`s, strata inlines ─────────────

/-- `SoftClip` — one `Tanh` over `drive · input`, output clamped to `[-1, 1]`. -/
def buildSoftClip (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let registry ← buildRegistry arena resolved #["Tanh"]
  let input : Sig := .inputRef ⟨0⟩
  let drive : Sig := .inputRef ⟨1⟩
  let decls : Array AInst := #[
    { name := "tanh", programName := "Tanh", inputs := #[⟨⟨0⟩, mul drive input⟩] }]
  .ok (assemble arena "SoftClip"
    #[signalDecl "input", floatDecl "drive" (lit 1)]
    #[{ name := "out", type? := some (.scalar .float) }]
    decls #[(.port ⟨0⟩, clampE (.nestedOut ⟨0⟩ ⟨0⟩) (lit (-1)) (lit 1))] registry)

/-- `ClockPM` — a carrier `FixedSinOsc` phase-modulated by `depth · modu` via its
    clock inlet (through-zero PM). -/
def buildClockPM (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let registry ← buildRegistry arena resolved #["FixedSinOsc"]
  let carrier : Sig := .inputRef ⟨0⟩
  let ratio : Sig := .inputRef ⟨1⟩
  let depth : Sig := .inputRef ⟨2⟩
  let carClk := add clockSig (toIntE (mul (mul depth (.nestedOut ⟨0⟩ ⟨0⟩)) (lit 4294967296)))
  let decls : Array AInst := #[
    { name := "modu", programName := "FixedSinOsc", inputs := #[⟨⟨0⟩, mul carrier ratio⟩] },
    { name := "car", programName := "FixedSinOsc", inputs := #[⟨⟨0⟩, carrier⟩, ⟨⟨1⟩, carClk⟩] }]
  .ok (assemble arena "ClockPM"
    #[freqDecl "carrier" 220, freqDecl "ratio" 1, floatDecl "depth" (lit 40)]
    #[floatOut] decls #[(.port ⟨0⟩, .nestedOut ⟨1⟩ ⟨0⟩)] registry)

/-- `ClockReverseProbe` — one oscillator read forward and at `−clock()`; their
    sum is the reversibility witness. -/
def buildClockReverseProbe (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let registry ← buildRegistry arena resolved #["FixedSinOsc"]
  let freq : Sig := .inputRef ⟨0⟩
  let decls : Array AInst := #[
    { name := "fwd", programName := "FixedSinOsc", inputs := #[⟨⟨0⟩, freq⟩, ⟨⟨1⟩, clockSig⟩] },
    { name := "rev", programName := "FixedSinOsc", inputs := #[⟨⟨0⟩, freq⟩, ⟨⟨1⟩, sub (lit 0) clockSig⟩] }]
  .ok (assemble arena "ClockReverseProbe"
    #[freqDecl "freq" 220] #[floatOut]
    decls #[(.port ⟨0⟩, add (.nestedOut ⟨0⟩ ⟨0⟩) (.nestedOut ⟨1⟩ ⟨0⟩))] registry)

/-- `FMBell` — a 3-oscillator FM bell (breath LFO → index → modulator → carrier
    clock). -/
def buildFMBell (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let registry ← buildRegistry arena resolved #["FixedSinOsc"]
  let carrier : Sig := .inputRef ⟨0⟩
  let ratio : Sig := .inputRef ⟨1⟩
  let index : Sig := .inputRef ⟨2⟩
  let lfo : Sig := .inputRef ⟨3⟩
  let carClk := add clockSig (toIntE (mul (mul (mul index (add (lit 1) (.nestedOut ⟨0⟩ ⟨0⟩)))
                 (.nestedOut ⟨1⟩ ⟨0⟩)) (lit 4294967296)))
  let decls : Array AInst := #[
    { name := "breath", programName := "FixedSinOsc", inputs := #[⟨⟨0⟩, lfo⟩] },
    { name := "modu", programName := "FixedSinOsc", inputs := #[⟨⟨0⟩, mul carrier ratio⟩] },
    { name := "car", programName := "FixedSinOsc", inputs := #[⟨⟨0⟩, carrier⟩, ⟨⟨1⟩, carClk⟩] }]
  .ok (assemble arena "FMBell"
    #[freqDecl "carrier" 220, freqDecl "ratio" 14142135623730951 16,
      floatDecl "index" (lit 90), freqDecl "lfo" 3 1]
    #[floatOut] decls #[(.port ⟨0⟩, .nestedOut ⟨2⟩ ⟨0⟩)] registry)

/-- `ModalVoice` — four inharmonic partials (`ClockPhasor ⋙ Sin`) at the
    Jaffe–Smith ratios, weighted-summed. -/
def buildModalVoice (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let registry ← buildRegistry arena resolved #["ClockPhasor", "Sin"]
  let clk : Sig := .inputRef ⟨0⟩
  let f0 : Sig := .inputRef ⟨1⟩
  let ph (freqE : Sig) : AInst := { name := "", programName := "ClockPhasor", inputs := #[⟨⟨0⟩, clk⟩, ⟨⟨1⟩, freqE⟩] }
  let decls : Array AInst := #[
    { ph f0 with name := "p1" },
    { ph (mul f0 (lit 2414213562373095 15)) with name := "p2" },
    { ph (mul f0 (lit 423606797749979 14)) with name := "p3" },
    { ph (mul f0 (lit 6854101966249685 15)) with name := "p4" },
    { name := "s1", programName := "Sin", inputs := #[⟨⟨0⟩, mul twoPiE (.nestedOut ⟨0⟩ ⟨0⟩)⟩] },
    { name := "s2", programName := "Sin", inputs := #[⟨⟨0⟩, mul twoPiE (.nestedOut ⟨1⟩ ⟨0⟩)⟩] },
    { name := "s3", programName := "Sin", inputs := #[⟨⟨0⟩, mul twoPiE (.nestedOut ⟨2⟩ ⟨0⟩)⟩] },
    { name := "s4", programName := "Sin", inputs := #[⟨⟨0⟩, mul twoPiE (.nestedOut ⟨3⟩ ⟨0⟩)⟩] }]
  let out := add (add (add (mul (lit 4 1) (.nestedOut ⟨4⟩ ⟨0⟩)) (mul (lit 24 2) (.nestedOut ⟨5⟩ ⟨0⟩)))
                 (mul (lit 16 2) (.nestedOut ⟨6⟩ ⟨0⟩))) (mul (lit 1 1) (.nestedOut ⟨7⟩ ⟨0⟩))
  .ok (assemble arena "ModalVoice"
    #[clockInDecl "clk" clockSig, freqDecl "f0" 110]
    #[floatOut] decls #[(.port ⟨0⟩, out)] registry)

/-- `PluckedMorphOsc` — a `MorphOsc` amplitude-shaped by a per-event pluck
    envelope `17.6·f·(1−f)⁶` driven by an `event_rate` phasor. -/
def buildPluckedMorphOsc (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let registry ← buildRegistry arena resolved #["MorphOsc", "ClockPhasor"]
  let freq : Sig := .inputRef ⟨0⟩
  let morph : Sig := .inputRef ⟨1⟩
  let clk : Sig := .inputRef ⟨2⟩
  let eventRate : Sig := .inputRef ⟨3⟩
  let phase : Sig := .inputRef ⟨4⟩
  let decls : Array AInst := #[
    { name := "osc", programName := "MorphOsc",
      inputs := #[⟨⟨0⟩, freq⟩, ⟨⟨1⟩, morph⟩, ⟨⟨2⟩, clk⟩, ⟨⟨3⟩, phase⟩] },
    { name := "ev", programName := "ClockPhasor", inputs := #[⟨⟨0⟩, clk⟩, ⟨⟨1⟩, eventRate⟩] }]
  let f : Sig := .nestedOut ⟨1⟩ ⟨0⟩
  let u := sub (lit 1) f
  let u2 := mul u u
  let env := mul (mul (mul (mul (lit 176 1) f) u2) u2) u2
  .ok (assemble arena "PluckedMorphOsc"
    #[freqDecl "freq" 220, unipolarDecl "morph", clockInDecl "clk" clockSig,
      freqDecl "event_rate" 1, unipolarDecl "phase"]
    #[floatOut] decls #[(.port ⟨0⟩, mul (.nestedOut ⟨0⟩ ⟨0⟩) env)] registry)

/-- `ReverseReverb` — a dry `ModalVoice` plus four `spacing`-delayed taps with
    geometric `decay` weights, scaled by `amount`. -/
def buildReverseReverb (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let registry ← buildRegistry arena resolved #["ModalVoice"]
  let clk : Sig := .inputRef ⟨0⟩
  let f0 : Sig := .inputRef ⟨1⟩
  let spacing : Sig := .inputRef ⟨2⟩
  let decay : Sig := .inputRef ⟨3⟩
  let amount : Sig := .inputRef ⟨4⟩
  -- tap1 carries `spacing` alone; taps 2–4 carry the leading integer coefficient.
  let tapClk (coeff? : Option Int) : Sig :=
    let sp := match coeff? with | none => spacing | some k => mul (lit k) spacing
    add clk (toIntE (mul (mul sp .sampleRate) (lit 4294967296)))
  let mv (nm : String) (clkE : Sig) : AInst :=
    { name := nm, programName := "ModalVoice", inputs := #[⟨⟨0⟩, clkE⟩, ⟨⟨1⟩, f0⟩] }
  let decls : Array AInst := #[
    mv "dry" clk, mv "tap1" (tapClk none), mv "tap2" (tapClk (some 2)),
    mv "tap3" (tapClk (some 3)), mv "tap4" (tapClk (some 4))]
  let d2 := mul decay decay
  let d3 := mul d2 decay
  let d4 := mul d3 decay
  let taps := add (add (add (mul decay (.nestedOut ⟨1⟩ ⟨0⟩)) (mul d2 (.nestedOut ⟨2⟩ ⟨0⟩)))
                  (mul d3 (.nestedOut ⟨3⟩ ⟨0⟩))) (mul d4 (.nestedOut ⟨4⟩ ⟨0⟩))
  .ok (assemble arena "ReverseReverb"
    #[clockInDecl "clk" clockSig, freqDecl "f0" 110, floatDecl "spacing" (lit 45 3),
      floatDecl "decay" (lit 72 2), floatDecl "amount" (lit 7 1)]
    #[floatOut] decls #[(.port ⟨0⟩, add (.nestedOut ⟨0⟩ ⟨0⟩) (mul amount taps))] registry)

/-- The palindromic-scrub clock a `*Probe` drives a comb with: a triangle over
    `sampleIndex` (`nf` up to `half`, back down), in Q32.32. -/
def triScrubClk (half : Sig) : Sig :=
  let nf := toFloatE .sampleIndex
  let tri := selectE (.binary .lt nf half) nf (sub (mul (lit 2) half) nf)
  toIntE (mul tri (lit 4294967296))

/-- `ReversibleProbe` — a `ReversibleComb` on the palindrome clock; the tail
    reverses onto itself. -/
def buildReversibleProbe (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let registry ← buildRegistry arena resolved #["ReversibleComb"]
  let half : Sig := .inputRef ⟨0⟩
  let f0 : Sig := .inputRef ⟨1⟩
  let delta : Sig := .inputRef ⟨2⟩
  let decls : Array AInst := #[
    { name := "comb", programName := "ReversibleComb",
      inputs := #[⟨⟨0⟩, triScrubClk half⟩, ⟨⟨1⟩, f0⟩, ⟨⟨2⟩, delta⟩] }]
  .ok (assemble arena "ReversibleProbe"
    #[floatDecl "half" (lit 2048), freqDecl "f0" 110, floatDecl "delta" (lit 7 4)]
    #[floatOut] decls #[(.port ⟨0⟩, .nestedOut ⟨0⟩ ⟨0⟩)] registry)

/-- `ThroughZeroFlanger` — a `ReversibleComb` whose delay is swept through zero
    by an LFO (`ClockPhasor ⋙ Sin`). -/
def buildThroughZeroFlanger (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let registry ← buildRegistry arena resolved #["ClockPhasor", "Sin", "ReversibleComb"]
  let clk : Sig := .inputRef ⟨0⟩
  let f0 : Sig := .inputRef ⟨1⟩
  let depth : Sig := .inputRef ⟨2⟩
  let rate : Sig := .inputRef ⟨3⟩
  let decls : Array AInst := #[
    { name := "lfoph", programName := "ClockPhasor", inputs := #[⟨⟨0⟩, clk⟩, ⟨⟨1⟩, rate⟩] },
    { name := "lfo", programName := "Sin", inputs := #[⟨⟨0⟩, mul twoPiE (.nestedOut ⟨0⟩ ⟨0⟩)⟩] },
    { name := "comb", programName := "ReversibleComb",
      inputs := #[⟨⟨0⟩, clk⟩, ⟨⟨1⟩, f0⟩, ⟨⟨2⟩, mul depth (.nestedOut ⟨1⟩ ⟨0⟩)⟩] }]
  .ok (assemble arena "ThroughZeroFlanger"
    #[clockInDecl "clk" clockSig, freqDecl "f0" 110, floatDecl "depth" (lit 7 4), freqDecl "rate" 3 1]
    #[floatOut] decls #[(.port ⟨0⟩, .nestedOut ⟨2⟩ ⟨0⟩)] registry)

/-- `ThroughZeroFlangerProbe` — the flanger on the palindrome clock. -/
def buildThroughZeroFlangerProbe (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let registry ← buildRegistry arena resolved #["ThroughZeroFlanger"]
  let half : Sig := .inputRef ⟨0⟩
  let f0 : Sig := .inputRef ⟨1⟩
  let depth : Sig := .inputRef ⟨2⟩
  let lfoRate : Sig := .inputRef ⟨3⟩
  let decls : Array AInst := #[
    { name := "fl", programName := "ThroughZeroFlanger",
      inputs := #[⟨⟨0⟩, triScrubClk half⟩, ⟨⟨1⟩, f0⟩, ⟨⟨2⟩, depth⟩, ⟨⟨3⟩, lfoRate⟩] }]
  .ok (assemble arena "ThroughZeroFlangerProbe"
    #[floatDecl "half" (lit 2048), freqDecl "f0" 110, floatDecl "depth" (lit 7 4), freqDecl "lfoRate" 6]
    #[floatOut] decls #[(.port ⟨0⟩, .nestedOut ⟨0⟩ ⟨0⟩)] registry)

/-- `ClockPhasorProbe` — `FixedPhasor` minus `ClockPhasor` at the root clock;
    their phases must agree, so `out` is ~0. -/
def buildClockPhasorProbe (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let registry ← buildRegistry arena resolved #["FixedPhasor", "ClockPhasor"]
  let freq : Sig := .inputRef ⟨0⟩
  let decls : Array AInst := #[
    { name := "ref", programName := "FixedPhasor", inputs := #[⟨⟨0⟩, freq⟩] },
    { name := "got", programName := "ClockPhasor", inputs := #[⟨⟨0⟩, clockSig⟩, ⟨⟨1⟩, freq⟩] }]
  .ok (assemble arena "ClockPhasorProbe"
    #[freqDecl "freq" 440] #[floatOut]
    decls #[(.port ⟨0⟩, sub (.nestedOut ⟨0⟩ ⟨0⟩) (.nestedOut ⟨1⟩ ⟨0⟩))] registry)

/-- `ClockChord` — three `ModalVoice`s at root / (5/4) / (3/2), the ratios cut as
    exact integer clock warps `(θ·5)>>2`, `(θ·3)>>1`. -/
def buildClockChord (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let registry ← buildRegistry arena resolved #["ModalVoice"]
  let f0 : Sig := .inputRef ⟨0⟩
  let mv (nm : String) (clkE : Sig) : AInst :=
    { name := nm, programName := "ModalVoice", inputs := #[⟨⟨0⟩, clkE⟩, ⟨⟨1⟩, f0⟩] }
  let decls : Array AInst := #[
    mv "root" clockSig,
    mv "third" (rshift (mul clockSig (lit 5)) (lit 2)),
    mv "fifth" (rshift (mul clockSig (lit 3)) (lit 1))]
  let out := add (add (mul (lit 4 1) (.nestedOut ⟨0⟩ ⟨0⟩)) (mul (lit 3 1) (.nestedOut ⟨1⟩ ⟨0⟩)))
                 (mul (lit 3 1) (.nestedOut ⟨2⟩ ⟨0⟩))
  .ok (assemble arena "ClockChord"
    #[freqDecl "f0" 110] #[floatOut]
    decls #[(.port ⟨0⟩, out)] registry)

-- ── The builder tables (dependency order = registration order) ───────────────

/-- Every stdlib program authored in this module, in dependency (registration)
    order — leaves first, then instance-bearing bottom-up. The 5 voices that
    already have production builders (`FixedSin`, `FixedSinOsc`, `MorphOsc`,
    `FlangeSin`, `ReversibleComb`) are woven in when they land here too; this
    list is the gate manifest and the seed of the boot chain. -/
def stdlibNewBuilders : Array (String × StdBuilder) := #[
  ("Sin", buildSin), ("Tanh", buildTanh), ("ScrubClock", buildScrubClock),
  ("VCA", buildVCA), ("FixedPhasor", buildFixedPhasor), ("ClockPhasor", buildClockPhasor),
  ("SoftClip", buildSoftClip), ("ClockPM", buildClockPM),
  ("ClockReverseProbe", buildClockReverseProbe), ("FMBell", buildFMBell),
  ("ModalVoice", buildModalVoice), ("PluckedMorphOsc", buildPluckedMorphOsc),
  ("ReverseReverb", buildReverseReverb), ("ReversibleProbe", buildReversibleProbe),
  ("ThroughZeroFlanger", buildThroughZeroFlanger),
  ("ThroughZeroFlangerProbe", buildThroughZeroFlangerProbe),
  ("ClockPhasorProbe", buildClockPhasorProbe), ("ClockChord", buildClockChord)]

end Tropical.EmitArrow
