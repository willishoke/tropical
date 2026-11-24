import Tropical.Ffi
import Tropical.StagedLoad
import Tropical.Playground
import Tropical.Tropicaltest.Gates
import Tropical.Tropicaltest.Reversibility

/-!
# Anchor continuity: live frequency writes in the rendered audio

`design/host-param-dispatch.md`, `anchor`. An oscillator's phase on ℤ/2³² is

    P(n) = Φ(inc, C(n)) + off,   Φ(inc, C) = ⌊inc · C / 2¹⁶⌋ mod 2³²

with `C(n)` the Q48.16 clock the phasor reads (the playground's master clock,
`trunc(tau_base·SR·2¹⁶) + scaleClockQ(trunc(velocity·2³²), n·2¹⁶)`),
`inc = trunc(f·2³²/SR)` and `off = trunc(φ·2³²)` from the `#phase` companion.
A write `f₀ → f₁` that becomes audible at sample `N` is phase-continuous iff
the new offset satisfies, exactly on ℤ/2³²,

    off₁ = off₀ + Φ(inc₀, C(N)) − Φ(inc₁, C(N)),

i.e. from `N` on the oscillator is the pre-write oscillator's phase at `N`
continued with the new increment. `off₁` is a multiple of 2⁻³² in [0, 1),
which an f64 slot holds exactly, so the prediction is BIT-EXACT: no
tolerance.

Each case drives two runtimes in lockstep through the real compile, load,
`set_param` and audio kernel. `live` takes the writes through the host's
dispatch. `ref` loads the same plan and takes the PREDICTED slot values as
plain slot writes (the prediction is computed here, independently of the C++
host). Every rendered buffer must be bit-identical between the two. The
kernel is a pure function of (τ, slots), so equal audio means equal phase.

Writes land at buffer boundaries: the host applies a write at the next
callback (`now` = the first sample of the next buffer), so there is no
mid-buffer write. The cases use an odd buffer length to show nothing assumes
alignment.
-/

namespace Tropical.Tropicaltest.AnchorContinuity

open Tropical

private def twoPow32 : Int := 4294967296
private def twoPow64 : Int := 18446744073709551616

/-- f64 → integer as the kernel's `toInt` (`llvm.fptosi.sat`): truncate toward
    zero. (Saturation is out of reach of the values used here.) -/
private def truncF (x : Float) : Int :=
  if x >= 0 then Int.ofNat x.toUInt64.toNat else -(Int.ofNat (-x).toUInt64.toNat)

/-- The signed i64 representative of `x` mod 2⁶⁴. -/
private def wrapI64 (x : Int) : Int :=
  let m := x % twoPow64
  if m >= twoPow64 / 2 then m - twoPow64 else m

/-- The playground master clock at whole sample `n`, as the kernel computes it
    (`Playground.Compiler.masterClock`, `EmitArrow.scaleClockQ`). `Int` `/` and
    `%` by a positive constant are floor and non-negative remainder: the
    kernel's `ashr` and mask. -/
private def masterClockAt (velocity tauBase rate : Float) (n : Nat) : Int :=
  let tauQ := truncF ((tauBase * rate) * 65536.0)
  let s := truncF (velocity * 4294967296.0)
  let clk := wrapI64 (Int.ofNat n * 65536)
  let hi := clk / 4294967296
  let lo := clk % 4294967296
  let loHi := lo / 65536
  let loLo := lo % 65536
  let pLoLo := (s * loLo) / 65536
  let pLo := (s * loHi + pLoLo) / 65536
  wrapI64 (tauQ + wrapI64 (s * hi + pLo))

/-- `Φ(inc, C)`: the phasor's phase on ℤ/2³² before the offset. -/
private def phaseQ (inc clk : Int) : Int :=
  (inc * (clk / 65536) + (inc * (clk % 65536)) / 65536) % twoPow32

private def incOf (freq rate : Float) : Int := truncF ((freq * 4294967296.0) / rate)

/-- The `#phase` value that keeps the phase continuous across `f₀ → f₁` at
    sample `now` on the clock `(velocity, tauBase)`. -/
private def anchoredPhase (f0 f1 phi velocity tauBase rate : Float) (now : Nat) : Float :=
  let c := masterClockAt velocity tauBase rate now
  let q := (truncF (phi * 4294967296.0) + phaseQ (incOf f0 rate) c
    - phaseQ (incOf f1 rate) c) % twoPow32
  Float.ofInt q / 4294967296.0

/-- The phase jump a RAW write `f₀ → f₁` makes at `now`, in 2⁻³²-cycle LSBs,
    as the signed representative in [−2³¹, 2³¹). -/
private def rawJumpLsb (f0 f1 velocity tauBase rate : Float) (now : Nat) : Int :=
  let c := masterClockAt velocity tauBase rate now
  let d := (phaseQ (incOf f1 rate) c - phaseQ (incOf f0 rate) c) % twoPow32
  if d >= twoPow32 / 2 then d - twoPow32 else d

inductive Step where
  /-- Render `n` buffers on both runtimes and compare them bit for bit. -/
  | render (n : Nat)
  /-- Reposition both clocks (a seek; before any write it is just a far start). -/
  | seek (sample : Nat)
  /-- A live param write through the host's dispatch. -/
  | write (name : String) (value : Float)
  /-- A structural edit: recompile `graph` and hot-swap it into both runtimes. -/
  | reload (graph : String)

structure Case where
  label : String
  graph : String
  steps : Array Step
  frames : Nat := 61
  /-- `false`: the plan's discipline table is stripped, so every write is raw. -/
  anchored : Bool := true
  /-- An expected failure: the reason, reported as XFAIL while it fails. -/
  xfail : Option String := none

private def compile (arena : Ir.Arena) (resolved : Array (String × Ir.ProgramIdx))
    (graph : String) (anchored : Bool) :
    Except String Playground.CompiledPatch := do
  let j ← Lean.Json.parse graph
  let c ← Playground.compilePlanPure arena resolved j
  if anchored then pure c
  else pure { c with plan := { c.plan with paramDisciplines := #[] } }

private def slot (rt : Ffi.Runtime) (name : String) : IO (Option UInt32) :=
  rt.slotIndex? s!"param:{name}"

private def readSlot (rt : Ffi.Runtime) (name : String) (dflt : Float) : IO Float := do
  match ← slot rt name with
  | some i => rt.getSlot i
  | none => pure dflt

private def writeSlot (rt : Ffi.Runtime) (name : String) (v : Float) : IO Unit := do
  match ← slot rt name with
  | some i => rt.setSlot i v
  | none => throw <| IO.userError s!"no slot param:{name}"

private def load (rt : Ffi.Runtime) (c : Playground.CompiledPatch) : IO Unit :=
  StagedLoad.loadTyped rt c.plan c.stageBlocks

/-- Run one case. `ok` is whether every buffer matched; `detail` says where
    the first mismatch was and how far the companion was off, in LSBs. -/
private def runCase (arena : Ir.Arena) (resolved : Array (String × Ir.ProgramIdx))
    (c : Case) : IO (Except String (Bool × String)) := do
  let compiled ← match compile arena resolved c.graph c.anchored with
    | .error e => return .error s!"compile: {firstLine e}"
    | .ok p => pure p
  let live ← Ffi.Runtime.new c.frames.toUInt32
  let ref ← Ffi.Runtime.new c.frames.toUInt32
  load live compiled
  load ref compiled
  let rate ← live.sampleRate
  let mut plan := compiled.plan
  let mut buffers := 0
  let mut notes : Array String := #[]
  for step in c.steps do
    match step with
    | .seek n =>
      live.setSampleIndex n.toUInt64
      ref.setSampleIndex n.toUInt64
    | .render n =>
      for _ in [0:n] do
        live.process
        ref.process
        let a := decodeF64LE (← live.outputBytes)
        let b := decodeF64LE (← ref.outputBytes)
        let diff := bitDiffCount a b
        if diff != 0 || a.size != b.size then
          let mut maxAbs : Float := 0
          for i in [0:min a.size b.size] do
            maxAbs := max maxAbs (Float.abs (a[i]! - b[i]!))
          -- How far the host's companion is from the prediction, in LSBs.
          let mut lsb := ""
          for d in plan.paramDisciplines do
            if d.discipline == "anchor" then
              let p := truncF ((← readSlot live s!"{d.name}#phase" 0) * 4294967296.0)
              let q := truncF ((← readSlot ref s!"{d.name}#phase" 0) * 4294967296.0)
              let e := (p - q) % twoPow32
              let e := if e >= twoPow32 / 2 then e - twoPow32 else e
              lsb := lsb ++ s!" {d.name}#phase off by {e} LSB"
          return .ok (false,
            s!"buffer {buffers}: {diff}/{a.size} samples differ, max |Δ| {maxAbs};{lsb}")
        buffers := buffers + 1
    | .write name value =>
      let now := (← live.currentSampleIndexU64).toNat
      let anchorD := plan.paramDisciplines.find? fun d =>
        d.name == name && d.discipline == "anchor"
      match anchorD with
      | some _ =>
        -- The prediction, from the reference's own (model) slots.
        let f0 ← readSlot ref name 0
        let phi ← readSlot ref s!"{name}#phase" 0
        let v ← readSlot ref Playground.Metadata.masterVelocityParam 1
        let tb ← readSlot ref Playground.Metadata.masterTauBaseParam 0
        if (← live.setParam name value).isNone then
          return .error s!"set_param: no param '{name}'"
        writeSlot ref name value
        writeSlot ref s!"{name}#phase" (anchoredPhase f0 value phi v tb rate now)
      | none =>
        -- Not anchored (or the table is stripped): the write is the subject
        -- of no prediction here, so mirror what the host wrote — the base and
        -- the companions its discipline lists — into the reference.
        let f0 ← readSlot ref name 0
        let v ← readSlot ref Playground.Metadata.masterVelocityParam 1
        let tb ← readSlot ref Playground.Metadata.masterTauBaseParam 0
        if (← live.setParam name value).isNone then
          return .error s!"set_param: no param '{name}'"
        let touched := match plan.paramDisciplines.find? (·.name == name) with
          | some d => #[name] ++ d.companions
          | none => #[name]
        for t in touched do
          writeSlot ref t (← readSlot live t 0)
        if !c.anchored && (← slot ref s!"{name}#phase").isSome then
          notes := notes.push s!"raw {name} {f0}→{value} at n={now}: phase jump {rawJumpLsb f0 value v tb rate now} LSB"
    | .reload graph =>
      let next ← match compile arena resolved graph c.anchored with
        | .error e => return .error s!"reload compile: {firstLine e}"
        | .ok p => pure p
      -- The prediction across a structural edit: an oscillator the new graph
      -- keeps (same node id, so the same `param:<id>.freq#phase` name) keeps
      -- its phase; its offset is carried.
      let mut carried : Array (String × Float) := #[]
      for d in next.plan.paramDisciplines do
        if d.discipline == "anchor" then
          if let some _ ← slot ref s!"{d.name}#phase" then
            carried := carried.push (s!"{d.name}#phase", ← readSlot ref s!"{d.name}#phase" 0)
      load live next
      load ref next
      for (name, v) in carried do
        writeSlot ref name v
      plan := next.plan
  let detail := s!"{buffers} buffers bit-identical to the prediction"
  return .ok (true, if notes.isEmpty then detail else s!"{detail}; {String.intercalate "; " notes.toList}")

-- ── Graphs ───────────────────────────────────────────────────────────────────

private def sineGraph (freq : Float) : String :=
  "{\"nodes\":[{\"id\":\"osc\",\"kind\":\"sine\",\"params\":{\"freq\":" ++ toString freq ++ "}}," ++
  "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"osc\"]}}],\"out\":\"out\"}"

/-- The same oscillator node `osc` (its freq param carrying the current value,
    as a client's snapshot does), with a second voice added to the mix. -/
private def sinePlusSawGraph (freq : Float) : String :=
  "{\"nodes\":[{\"id\":\"osc\",\"kind\":\"sine\",\"params\":{\"freq\":" ++ toString freq ++ "}}," ++
  "{\"id\":\"osc2\",\"kind\":\"saw\",\"params\":{\"freq\":97}}," ++
  "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"osc\",\"osc2\"]}}],\"out\":\"out\"}"

private def sawGraph : String :=
  "{\"nodes\":[{\"id\":\"osc\",\"kind\":\"saw\",\"params\":{\"freq\":220}}," ++
  "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"osc\"]}}],\"out\":\"out\"}"

private def velocity : String := Playground.Metadata.masterVelocityParam
private def tauBase : String := Playground.Metadata.masterTauBaseParam

def cases : Array Case := #[
  -- (a) plain transport.
  { label := "anchor-continuity/plain"
    graph := sineGraph 440
    steps := #[.render 5, .write "osc.freq" 660, .render 4] },
  -- (a) plain transport, ~9 days in: Δinc·now is past 2⁵³.
  { label := "anchor-continuity/plain-far"
    graph := sineGraph 30
    steps := #[.seek 34359738368, .render 2, .write "osc.freq" 7919, .render 3] },
  -- The polyBLEP saw: the shared `ClockPhasor` phasor.
  { label := "anchor-continuity/saw"
    graph := sawGraph
    steps := #[.render 3, .write "osc.freq" 331.5, .render 3] },
  -- A negative frequency: `trunc` and `floor` differ.
  { label := "anchor-continuity/negative-freq"
    graph := sineGraph 440
    steps := #[.render 3, .write "osc.freq" (-317.3), .render 2,
      .write "osc.freq" (-1001.7), .render 2] },
  -- (b) master velocity ≠ 1 and tau_base ≠ 0, set before any audio.
  { label := "anchor-continuity/velocity"
    graph := sineGraph 440
    steps := #[.write tauBase 0.37, .write velocity 1.5, .render 4,
      .write "osc.freq" 660, .render 3] },
  -- (b) reverse transport.
  { label := "anchor-continuity/reverse"
    graph := sineGraph 440
    steps := #[.write tauBase 12.5, .write velocity (-1), .render 4,
      .write "osc.freq" 523.25, .render 3] },
  -- (b) a velocity change mid-run (the host re-bases tau_base), then a re-pitch.
  { label := "anchor-continuity/velocity-rebased"
    graph := sineGraph 440
    steps := #[.render 3, .write velocity 0.5, .render 3,
      .write "osc.freq" 880, .render 3] },
  -- (c) several writes: consecutive buffers, two in one boundary, under velocity.
  { label := "anchor-continuity/sequence"
    graph := sineGraph 440
    steps := #[.write velocity 0.75, .render 2,
      .write "osc.freq" 523.25, .render 1,
      .write "osc.freq" 659.26, .render 1,
      .write "osc.freq" 392, .write "osc.freq" 1174.66, .render 2,
      .write velocity (-1.25), .render 1,
      .write "osc.freq" 82.4, .render 3] },
  -- (c) the same writes at a power-of-two buffer length.
  { label := "anchor-continuity/sequence-64"
    graph := sineGraph 440
    frames := 64
    steps := #[.write velocity 0.75, .render 2,
      .write "osc.freq" 523.25, .render 1,
      .write "osc.freq" 392, .write "osc.freq" 1174.66, .render 2] },
  -- (d) an anchored write, then a structural edit that keeps `osc`.
  { label := "anchor-continuity/structural-edit"
    graph := sineGraph 440
    steps := #[.render 3, .write "osc.freq" 660, .render 2,
      .reload (sinePlusSawGraph 660), .render 3]
    xfail := some "a hot-swap starts every slot from the fresh plan's defaults, so `#phase` returns to 0; carrying it needs a design decision (see the PR)" },
  -- (e) raw: the same re-pitch with the discipline table stripped. The phase
  -- offset is untouched, so the phase jumps by Φ(inc₁,C) − Φ(inc₀,C).
  { label := "anchor-continuity/raw-jumps"
    graph := sineGraph 440
    anchored := false
    steps := #[.render 5, .write "osc.freq" 660, .render 4] }
]

/-- The raw case must not be vacuous: the anchored and raw renders of the same
    re-pitch differ after the write (so the raw write really jumps). -/
private def runRawDiffers (arena : Ir.Arena) (resolved : Array (String × Ir.ProgramIdx)) :
    IO Bool := do
  let label := "anchor-continuity/raw-differs"
  let render (anchored : Bool) : IO (Except String (Array Float)) := do
    match compile arena resolved (sineGraph 440) anchored with
    | .error e => return .error e
    | .ok c =>
      let rt ← Ffi.Runtime.new 61
      load rt c
      for _ in [0:5] do rt.process
      let _ ← rt.setParam "osc.freq" 660
      rt.process
      return .ok (decodeF64LE (← rt.outputBytes))
  match ← render true, ← render false with
  | .ok a, .ok b =>
    let d := bitDiffCount a b
    if d > a.size / 2 then passGate label s!"{d}/{a.size} samples differ after the write"
    else failGate label s!"only {d}/{a.size} samples differ: the raw write did not jump"
  | .error e, _ | _, .error e => failGate label s!"compile: {firstLine e}"


-- ── The sibling: a modal-control glide on the master clock ──────────────────

/-- A resonator through a phaser: the phaser's `center` is a modal control,
    whose glide the kernel measures on the master clock (`glideExprQAt`). -/
private def phaserGraph : String :=
  "{\"nodes\":[{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":50}}," ++
  "{\"id\":\"ph\",\"kind\":\"phaser\",\"in\":{\"in\":[\"res\"]}}," ++
  "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"ph\"]}}],\"out\":\"out\"}"

/-- Does a glided write ramp? Render the first buffer after `ph.center`
    700 → 1400 written through the host (a 5 ms glide, longer than the
    buffer) and after the same target written as a STEP (`#v0 = #v1 =` the
    target). A glide that ramps differs from the step. -/
private def glideRamps (arena : Ir.Arena) (resolved : Array (String × Ir.ProgramIdx))
    (tauBaseSec : Float) : IO (Except String Nat) := do
  let c ← match compile arena resolved phaserGraph true with
    | .error e => return .error s!"compile: {firstLine e}"
    | .ok c => pure c
  let render (glided : Bool) : IO (Array Float) := do
    let rt ← Ffi.Runtime.new 61
    load rt c
    let _ ← rt.setParam tauBase tauBaseSec
    for _ in [0:4] do rt.process
    if glided then
      let _ ← rt.setParam "ph.center" 1400
    else
      writeSlot rt "ph.center#v0" 1400
      writeSlot rt "ph.center#v1" 1400
    rt.process
    return decodeF64LE (← rt.outputBytes)
  let g ← render true
  let st ← render false
  return .ok (bitDiffCount g st)

private def runGlideClock (arena : Ir.Arena) (resolved : Array (String × Ir.ProgramIdx)) :
    IO Bool := do
  let plain ← match ← glideRamps arena resolved 0 with
    | .error e => return ← failGate "glide-clock/plain" e
    | .ok d => pure d
  let ok ← if plain > 0 then
      passGate "glide-clock/plain" s!"the glide ramps: {plain}/61 samples differ from a step"
    else failGate "glide-clock/plain" "the glided write equals a step"
  match ← glideRamps arena resolved 3 with
  | .error e => return ← failGate "glide-clock/tau-base" e
  | .ok 0 =>
    IO.println s!"  XFAIL glide-clock/tau-base  with tau_base = 3 s the glided write equals a step — expected: the host stamps `#t0` with the sample index while the kernel measures the modal glide on the master clock, so the ramp is already over; fixing it needs a decision on which clock a knob glide runs on (see the PR)"
    return ok
  | .ok d =>
    failGate "glide-clock/tau-base" s!"unexpectedly ramps ({d}/61 differ); remove the expected-failure marker"

def runAnchorContinuity (arena : Ir.Arena) (resolved : Array (String × Ir.ProgramIdx)) :
    IO Bool := do
  let mut ok := true
  for c in cases do
    match ← runCase arena resolved c with
    | .error e => ok := (← failGate c.label e) && ok
    | .ok (passed, detail) =>
      match c.xfail, passed with
      | none, true => ok := (← passGate c.label detail) && ok
      | none, false => ok := (← failGate c.label detail) && ok
      | some why, false =>
        IO.println s!"  XFAIL {c.label}  {detail} — expected: {why}"
      | some _, true =>
        ok := (← failGate c.label s!"unexpectedly passed ({detail}); remove the expected-failure marker") && ok
  ok := (← runRawDiffers arena resolved) && ok
  ok := (← runGlideClock arena resolved) && ok
  return ok

end Tropical.Tropicaltest.AnchorContinuity
