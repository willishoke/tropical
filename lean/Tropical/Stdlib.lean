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
  with Lean `let` sharing standing in for source-level binders (the IR has
  none — the host language is the binding layer).

* **Instance-bearing** programs declare `AInst`s and let the lowering's
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

-- ── The voices — promoted from `Testing/ArrowFixtures` ───────────────────────

/-- `FixedSin` — the exact Q2.30 fixed-point sine over a Q0.32 cycles phase. -/
def buildFixedSin (arena : Arena) (_resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  .ok (assemble arena "FixedSin"
    #[{ name := "phase", type? := some (.scalar .int), defaultSig := some (lit 0) }]
    #[{ name := "out", type? := some (.scalar .int) }]
    #[] #[(.port ⟨0⟩, fixedSinCycSig (.inputRef ⟨0⟩))] #[])

/-- `FixedSinOsc` — the foundational voice, built flat (no instances): the integer
    split-multiply phasor composed with the Q2.30 sine, scaled to float once at
    the boundary. -/
def buildFixedSinOsc (arena : Arena) (_resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  let freqIn : Sig := .inputRef ⟨0⟩
  let clk : Sig := .inputRef ⟨1⟩
  let twoPow32 := lit 4294967296
  let mask := lit 4294967295
  let inc := toIntE (div (mul freqIn twoPow32) .sampleRate)
  let thi := rshift clk (lit 32)
  let tlo := bitAnd clk mask
  let off := toIntE (mul (.inputRef ⟨2⟩) twoPow32)
  let acc := add (add (mul inc thi) (rshift (mul inc tlo) (lit 32))) off
  let phase := clampE (div (toFloatE (bitAnd acc mask)) twoPow32) (lit 0) (lit 1)
  let pQ := toIntE (mul phase twoPow32)
  let sine := div (toFloatE (fixedSinCycSig pQ)) (lit 1073741824)
  .ok (assemble arena "FixedSinOsc"
    #[freqDecl "freq" 440, clockInDecl "clk" clockSig, unipolarDecl "phase"]
    #[{ name := "sine", type? := some (.scalar .float) }]
    #[] #[(.port ⟨0⟩, sine)] #[])

/-- `MorphOsc` — `ClockPhasor ⋙ (saw &&& FixedSin) ⋙ crossfade`: the phasor's
    phase drives a naive saw and a fixed-point sine, crossfaded by `morph`. -/
def buildMorphOsc (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let registry ← buildRegistry arena resolved #["ClockPhasor", "FixedSin"]
  let freq : Sig := .inputRef ⟨0⟩
  let morph : Sig := .inputRef ⟨1⟩
  let clk : Sig := .inputRef ⟨2⟩
  let phase : Sig := .inputRef ⟨3⟩
  let phaseOut : Sig := .nestedOut ⟨0⟩ ⟨0⟩
  let decls : Array AInst := #[
    { name := "ph", programName := "ClockPhasor", inputs := #[⟨⟨0⟩, clk⟩, ⟨⟨1⟩, freq⟩, ⟨⟨2⟩, phase⟩] },
    { name := "sin", programName := "FixedSin",
      inputs := #[⟨⟨0⟩, toIntE (mul phaseOut (lit 4294967296))⟩] }]
  let saw := sub (mul (lit 2) phaseOut) (lit 1)
  let sinOut := div (toFloatE (.nestedOut ⟨1⟩ ⟨0⟩)) (lit 1073741824)
  let out := add (mul (sub (lit 1) morph) saw) (mul morph sinOut)
  .ok (assemble arena "MorphOsc"
    #[freqDecl "freq" 220, unipolarDecl "morph", clockInDecl "clk" clockSig, unipolarDecl "phase"]
    #[floatOut] decls #[(.port ⟨0⟩, out)] registry)

/-- The warp-bank of three taps (dry, `−δ`, `+δ`) over one voice, weighted
    0.5/0.25/0.25 — the shared body of `FlangeSin` and `ReversibleComb`. `δ` =
    `toInt(offset · sampleRate · 2³²)`, the offset input in Q32.32 samples. -/
private def warpBank3 (arena : Arena) (name voiceName : String)
    (registry : Array (String × ProgramIdx))
    (clk offsetIn : Sig) (freqDeclHz : Int) (voiceInputs : Sig → Array AInput) :
    Arena × ProgramIdx :=
  let delta := toIntE (mul (mul offsetIn .sampleRate) (lit 4294967296))
  let tap (nm : String) (clkE : Sig) : AInst :=
    { name := nm, programName := voiceName, inputs := voiceInputs clkE }
  let decls : Array AInst := #[tap "dry" clk, tap "past" (sub clk delta), tap "ahead" (add clk delta)]
  let out := add (add (mul (lit 5 1) (.nestedOut ⟨0⟩ ⟨0⟩)) (mul (lit 25 2) (.nestedOut ⟨1⟩ ⟨0⟩)))
                 (mul (lit 25 2) (.nestedOut ⟨2⟩ ⟨0⟩))
  assemble arena name
    #[clockInDecl "clk" clockSig, freqDecl (if voiceName == "ModalVoice" then "f0" else "freq") freqDeclHz,
      floatDecl (if voiceName == "ModalVoice" then "delta" else "depth") (lit 7 4)]
    #[floatOut] decls #[(.port ⟨0⟩, out)] registry

/-- `FlangeSin` — the warp-bank flanger over the `FixedSinOsc` voice
    (freq at port 0, clock at port 1). -/
def buildFlanger (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let registry ← buildRegistry arena resolved #["FixedSinOsc"]
  .ok (warpBank3 arena "FlangeSin" "FixedSinOsc" registry (.inputRef ⟨0⟩) (.inputRef ⟨2⟩) 220
        (fun clkE => #[⟨⟨0⟩, .inputRef ⟨1⟩⟩, ⟨⟨1⟩, clkE⟩]))

/-- `ReversibleComb` — the warp-bank comb over the `ModalVoice` voice (clock at
    port 0, f0 at port 1). The ahead tap reads the future — reverses with zero
    latency. -/
def buildReversibleComb (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let registry ← buildRegistry arena resolved #["ModalVoice"]
  .ok (warpBank3 arena "ReversibleComb" "ModalVoice" registry (.inputRef ⟨0⟩) (.inputRef ⟨2⟩) 110
        (fun clkE => #[⟨⟨0⟩, clkE⟩, ⟨⟨1⟩, .inputRef ⟨1⟩⟩]))

-- ── The builder chain (manifest order = registration order) ──────────────────

/-- Every live stdlib program as an arrow builder, in manifest (registration)
    order — deps before dependents, so `buildRegistry` resolves against the chain
    built so far. THE boot chain and the gate manifest. -/
def stdlibBuilders : Array (String × StdBuilder) := #[
  ("ClockPhasor", buildClockPhasor), ("FixedSin", buildFixedSin),
  ("FixedSinOsc", buildFixedSinOsc), ("MorphOsc", buildMorphOsc),
  ("Sin", buildSin), ("Tanh", buildTanh), ("VCA", buildVCA),
  ("FlangeSin", buildFlanger), ("ModalVoice", buildModalVoice),
  ("PluckedMorphOsc", buildPluckedMorphOsc), ("ReverseReverb", buildReverseReverb),
  ("ReversibleComb", buildReversibleComb), ("ScrubClock", buildScrubClock),
  ("ThroughZeroFlanger", buildThroughZeroFlanger), ("SoftClip", buildSoftClip)]

/-- Build the whole stdlib as one arena by folding `stdlibBuilders` from empty:
    each builder appends its raw program, linking sub-programs by name against the
    chain so far (deps precede dependents). Returns the arena + name→idx table.
    The pure production replacement for the parsed-bridge `StdlibChain.elabStdlib`
    — what the gates, the playground, and the diffcli verbs link against. -/
def buildStdlibChain : Except String (Arena × Array (String × ProgramIdx)) := do
  let mut arena : Arena := {}
  let mut chain : Array (String × ProgramIdx) := #[]
  for (name, build) in stdlibBuilders do
    let (arena', idx) ← build arena chain
    arena := arena'
    chain := chain.push (name, idx)
  pure (arena, chain)

end Tropical.EmitArrow
