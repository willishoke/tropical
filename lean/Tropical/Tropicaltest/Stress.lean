import Tropical.Tropicaltest.ArrowLaws

/-!
# Tropical.Tropicaltest.Stress

Standard-rep differentials on the hard cases: the convolution oracle (clock-warp FIR ≡ array-shift conv), the fractional nonlinear modulated clock, nested PM, negative-time random access, and the MorphOsc MIMO pipeline.
-/

open Tropical
open Tropical.Plan
open Tropical.Ir (Arena ProgramIdx)

-- ── (h″) The convolution stress test — the bubble, EXECUTED, with a NON-FACADE
-- oracle. An FIR filter is fan-out + clock-warps + scale + sum (a convolution IS
-- the flanger with more taps). We compute the convolution TWO independent ways
-- and demand they agree:
--   tropical: each tap warps the CLOCK by j samples (j·2³² in Q32.32); the
--     oscillator is evaluated at the warped clock, weighted, summed — the bubble
--     doing the work inside the kernel.
--   oracle:   render the BARE oscillator once, then shift the resulting Float
--     array by j, scale by kⱼ, sum — ordinary Lean arithmetic that NEVER touches
--     the warp lowering.
-- Agreement proves "warp the clock by j samples" realizes "delay the output by j
-- samples" IN THE ACTUAL COMPILER, checked by an oracle independent of the
-- lowering (this is what defeats correct-by-facade — eval-walking the same term
-- could not). The filter-effect figure confirms the FIR is non-degenerate.

/-- A j-sample clock delay: subtract `j·2³²` (Q32.32) from the clock. `j = 0` is
    identity (`sub c 0 = c`). -/
private def firShift (j : Nat) : Tropical.EmitArrow.Clock → Tropical.EmitArrow.Clock :=
  fun c => Tropical.EmitArrow.sub c
    (Tropical.EmitArrow.toIntE (Tropical.EmitArrow.lit (Int.ofNat j * 4294967296)))

/-- 3-tap FIR `[0.25, 0.5, 0.25]` at integer-sample delays `[0,1,2]`, as a bank
    of CLOCK warps over the closed-form 12 kHz voice (pitch high enough that the
    lowpass visibly attenuates). -/
private def firTaps : Array Tropical.EmitArrow.Tap := #[
  { name := "k0", warp := fun c => c, weight := Tropical.EmitArrow.lit 25 2 },
  { name := "k1", warp := firShift 1, weight := Tropical.EmitArrow.lit 5 1 },
  { name := "k2", warp := firShift 2, weight := Tropical.EmitArrow.lit 25 2 } ]

/-- The bare voice: a single identity tap, weight 1 — the source samples the
    oracle convolves by hand. -/
private def bareTaps : Array Tropical.EmitArrow.Tap := #[
  { name := "x", warp := fun c => c, weight := Tropical.EmitArrow.lit 1 } ]

/-- Compile a closed-form tap-bank carrier (the 12 kHz voice) to a runnable
    `FlatPlan` via the production session path — same recipe as
    `compileArrowCarrier`. -/
private def compileTapCarrier (arena : Arena) (resolved : Array (String × ProgramIdx))
    (name : String) (taps : Array Tropical.EmitArrow.Tap) :
    Except String Tropical.Plan.FlatPlan :=
  buildAndFinish (Tropical.EmitArrow.buildTapCarrier name
    Tropical.EmitArrow.litPitch12kVoice taps arena resolved)

/-- Render a `FlatPlan` to exactly `n` contiguous mono samples (buffer = n, no
    fade), like `renderSamples` but from an in-hand plan. -/
def renderPlanSamples (plan : Tropical.Plan.FlatPlan) (n : Nat) :
    IO (Except String (Array Float)) := do
  try
    let rt ← Tropical.Ffi.Runtime.new (UInt32.ofNat n)
    Tropical.StagedLoad.load rt plan
    rt.process
    pure (.ok (decodeF64LE (← rt.outputBytes)))
  catch e => pure (.error e.toString)

/-- THE NON-FACADE GATE: tropical's clock-warped FIR ≡ an array-shift convolution
    of the independently-rendered bare oscillator. -/
def runConvolutionOracle (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let n : Nat := 1024
  let kernel : Array Float := #[0.25, 0.5, 0.25]   -- delays 0,1,2
  let maxDelay := kernel.size - 1
  match compileTapCarrier arena resolved "Fir3" firTaps,
        compileTapCarrier arena resolved "Bare" bareTaps with
  | .error e, _ => failGate "convolution-oracle" s!"build fir: {firstLine e}"
  | _, .error e => failGate "convolution-oracle" s!"build bare: {firstLine e}"
  | .ok firPlan, .ok barePlan =>
    match ← renderPlanSamples firPlan n, ← renderPlanSamples barePlan n with
    | .error e, _ | _, .error e =>
      failGate "convolution-oracle" s!"render: {firstLine e}"
    | .ok got, .ok x =>
      let mut maxAbs : Float := 0.0
      let mut filterEffect : Float := 0.0
      let mut energy : Float := 0.0
      for t in [maxDelay:n] do
        let mut acc : Float := 0.0
        for j in [0:kernel.size] do
          acc := acc + kernel[j]! * x[t - j]!
        let g := got[t]!
        let d := (g - acc).abs
        if d > maxAbs then maxAbs := d
        let fe := (g - x[t]!).abs
        if fe > filterEffect then filterEffect := fe
        energy := energy + g * g
      let eps : Float := 1e-9
      if energy <= 1e-6 then
        failGate "convolution-oracle" s!"signal silent (energy={energy})"
      else if maxAbs < eps then
        passGate "convolution-oracle" s!"clock-warp FIR ≡ array-shift conv  (max|Δ|={maxAbs}, filter-effect={filterEffect}, samples={n - maxDelay})"
      else
        failGate "convolution-oracle" s!"max|Δ|={maxAbs} (≥ {eps}); filter-effect={filterEffect}"

-- ── (h‴) The MODULATED-CLOCK stress test — a fractional, NONLINEAR warp, to see
-- whether the bubble is a side-effect of affineness (it should not be). The warp
-- φ(τ) = clk − ⌊depth·mod(τ)·2³²⌋ is sub-sample and nonlinear (mod is a sine);
-- it evaluates the carrier at clock values BETWEEN integer samples, which the
-- array-shift oracle cannot reach. So the oracle is an INDEPENDENT closed-form
-- reference (Lean `Float.sin` on the modulated phase), calibrated against the
-- bare oscillator first: tropical's `Sin` is a polynomial, so this is a
-- TOLERANCE check, not bit-exact — but the tolerance is the bare osc's own
-- poly/quantization floor, so the test isolates the WARP's contribution. A warp
-- that secretly needed affineness would diverge by O(1), far above that floor.

/-- `stdlib/FixedSin.md` transcribed exactly in Lean Int arithmetic: the Q2.30
    datapath sine at a MASKED Q0.32 cycles phase. `Int.fdiv` is floor division
    = the engine's `ashr`; every Horner operand is non-negative by construction
    (all-positive-with-subtractions), so floor = truncate there; the final
    `(r·acc₀) >> 30` is the one signed floor-shift, matched exactly. -/
private def fixedSinQ (p : Int) : Int :=
  let n := Int.fdiv (p + 1073741824) 2147483648
  let r := p - n * 2147483648
  let sign := 1 - 2 * (Int.fmod n 2)
  let z := Int.fdiv (r * r) 1073741824
  let acc6 := 61 - Int.fdiv z 1073741824
  let acc5 := 3864 - Int.fdiv (acc6 * z) 1073741824
  let acc4 := 172272 - Int.fdiv (acc5 * z) 1073741824
  let acc3 := 5026995 - Int.fdiv (acc4 * z) 1073741824
  let acc2 := 85569306 - Int.fdiv (acc3 * z) 1073741824
  let acc1 := 693598668 - Int.fdiv (acc2 * z) 1073741824
  let acc0 := 1686629713 - Int.fdiv (acc1 * z) 1073741824
  sign * Int.fdiv (r * acc0) 1073741824

/-- The voice sine as the engine now computes it: re-land the float phase as
    its exact Q0.32 integer (lossless — P < 2³² ≪ 2⁵³), run `fixedSinQ`, scale
    Q2.30 → float. The standard-rep twin of `FixedSin(toInt(phase·2³²))/2³⁰`. -/
private def voiceSin (phase : Float) : Float :=
  Float.ofInt (fixedSinQ ((phase * 4294967296.0).toUInt64.toNat)) / 1073741824.0

/-- `ClockPhasor.phase` at a Q32.32 clock value, transcribed exactly (integer
    math, offset = 0). inc = ⌊freqHz·2³²/SR⌋. Uses `fdiv`/`fmod` so it matches the
    engine's arithmetic shift (`>>`, floor) and two's-complement mask (`&`) for
    NEGATIVE clocks too — i.e. negative time / backward extrapolation. For clk ≥ 0
    these agree with plain /,%. -/
private def phasorPhase (clk : Int) (freqHz : Int) : Float :=
  let inc : Int := (freqHz * 4294967296) / 44100
  let thi := Int.fdiv clk 4294967296
  let tlo := Int.fmod clk 4294967296
  let acc := inc * thi + Int.fdiv (inc * tlo) 4294967296
  Float.ofInt (Int.fmod acc 4294967296) / 4294967296.0

/-- Float → Int truncation toward zero (matches the engine's `toInt`). -/
private def truncToInt (v : Float) : Int :=
  if v ≥ 0.0 then Int.ofNat v.toUInt64.toNat
  else -(Int.ofNat (-v).toUInt64.toNat)

/-- THE STANDARD-REP GATE: tropical's arrow-emitted, sine-modulated, SUB-SAMPLE
    clock warp vs a straight-line reimplementation using THE SAME Horner `Sin`
    and the SAME integer phasor. The polynomial cancels (it is identical on both
    sides — no true sine in the loop), so the residual is purely the warp/emit
    path and float op-ordering. The warp is genuinely nonlinear and fractional:
    φ(τ) = clk − toInt(depth · Sin(mod phase) · 2³²). -/
def runModulatedClock (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let n : Nat := 1024
  let lo : Nat := 8
  let depth : Float := 3.0
  let twoPi : Float := 6.283185307179586
  let two32 : Float := 4294967296.0
  let sinkGain : Float := 0.05   -- defaultSinkGain (Plan.lean): scales OUTPUTS, not the mid-graph warp
  match buildAndFinish (Tropical.EmitArrow.buildTapCarrier "BareFc"
          (Tropical.EmitArrow.litPitchVoice 2000) bareTaps arena resolved),
        buildAndFinish (Tropical.EmitArrow.buildFmCarrier "FmOsc" 2000 200 3 arena resolved) with
  | .error e, _ => failGate "modulated-clock" s!"build bare: {firstLine e}"
  | _, .error e => failGate "modulated-clock" s!"build fm: {firstLine e}"
  | .ok barePlan, .ok fmPlan =>
    match ← renderPlanSamples barePlan n, ← renderPlanSamples fmPlan n with
    | .error e, _ | _, .error e =>
      failGate "modulated-clock" s!"render: {firstLine e}"
    | .ok bare, .ok got =>
      let mut e0 : Float := 0.0           -- calibration: engine bare vs standard rep
      let mut efm : Float := 0.0          -- engine fm vs standard rep (the warp test)
      let mut warpEffect : Float := 0.0   -- |fm − bare|
      let mut maxBare : Float := 0.0
      let mut calBitDiff : Nat := 0       -- bit-differing samples (engine bare vs std)
      let mut fmBitDiff : Nat := 0        -- bit-differing samples (engine fm vs std)
      for t in [lo:n] do
        let clk : Int := Int.ofNat t * 4294967296
        -- calibration: engine's bare carrier vs the standard-rep carrier
        let refBare := sinkGain * voiceSin (phasorPhase clk 2000)
        if (bare[t]! - refBare).abs > e0 then e0 := (bare[t]! - refBare).abs
        if bare[t]!.toBits != refBare.toBits then calBitDiff := calBitDiff + 1
        if bare[t]!.abs > maxBare then maxBare := bare[t]!.abs
        -- the warp: mid-graph (unit-scale) modulator = Sin at the modulator phase;
        -- offset = toInt(depth·mod·2³²); φ = clk − offset (sub-sample, nonlinear)
        let rawMod := voiceSin (phasorPhase clk 200)
        let phi : Int := clk - truncToInt (depth * rawMod * two32)
        let refFm := sinkGain * voiceSin (phasorPhase phi 2000)
        if (got[t]! - refFm).abs > efm then efm := (got[t]! - refFm).abs
        if got[t]!.toBits != refFm.toBits then fmBitDiff := fmBitDiff + 1
        if (got[t]! - bare[t]!).abs > warpEffect then warpEffect := (got[t]! - bare[t]!).abs
      let samples := n - lo
      IO.println s!"        standard rep = same Horner Sin + same integer phasor (no true sine):"
      IO.println s!"        calibrate  engine bare vs standard rep:  max|Δ|={e0}  ·  bit-differing {calBitDiff}/{samples}"
      IO.println s!"        result     engine fm   vs standard rep:  max|Δ|={efm}  ·  bit-differing {fmBitDiff}/{samples}  ·  warp effect |fm−bare| max={warpEffect}"
      if maxBare < 1e-3 then
        failGate "modulated-clock" s!"carrier silent (maxBare={maxBare})"
      else if e0 > 1e-6 then
        failGate "modulated-clock" s!"calibration off (e0={e0}) — Sin/phasor transcription wrong, test invalid"
      else if warpEffect < 0.2 * maxBare then
        failGate "modulated-clock" s!"modulation negligible (warp {warpEffect} vs amp {maxBare})"
      else if efm < 10.0 * e0 + 1e-9 then
        passGate "modulated-clock" s!"fractional nonlinear warp ≡ standard rep (fm err {efm} ≈ floor {e0}; warp effect {warpEffect})"
      else
        failGate "modulated-clock" s!"fm err {efm} ≫ floor {e0} — warp diverges from the standard rep"

/-- PM-of-PM: the modulator is ITSELF a warped oscillator (mod2 warps mod's
    clock, mod warps the carrier's clock). Bit-exact against a THREE-level nested
    standard rep (same Horner Sin + integer phasor at each level) ⇒ the warp /
    substitution composes through nesting. Also asserts the second level is
    non-trivial: PM(PM) differs from single-level PM. -/
def runPmPm (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let n : Nat := 1024
  let lo : Nat := 8
  let d1 : Float := 3.0
  let d2 : Float := 3.0
  let twoPi : Float := 6.283185307179586
  let two32 : Float := 4294967296.0
  let sinkGain : Float := 0.05
  match buildAndFinish (Tropical.EmitArrow.buildPmPmCarrier "PmPm" 2000 200 700 3 3 arena resolved),
        buildAndFinish (Tropical.EmitArrow.buildFmCarrier "Fm1" 2000 200 3 arena resolved) with
  | .error e, _ => failGate "pm-of-pm" s!"build pmpm: {firstLine e}"
  | _, .error e => failGate "pm-of-pm" s!"build fm1: {firstLine e}"
  | .ok pmpmPlan, .ok fmPlan =>
    match ← renderPlanSamples pmpmPlan n, ← renderPlanSamples fmPlan n with
    | .error e, _ | _, .error e =>
      failGate "pm-of-pm" s!"render: {firstLine e}"
    | .ok got, .ok fm1 =>
      let mut e0 : Float := 0.0           -- engine pm(pm) vs nested standard rep
      let mut bitDiff : Nat := 0
      let mut maxOut : Float := 0.0
      let mut nestEffect : Float := 0.0   -- |pm(pm) − single-level pm| (does level 2 matter)
      for t in [lo:n] do
        let clk : Int := Int.ofNat t * 4294967296
        let mod2 := voiceSin (phasorPhase clk 700)
        let modClk : Int := clk - truncToInt (d2 * mod2 * two32)
        let mod := voiceSin (phasorPhase modClk 200)
        let carClk : Int := clk - truncToInt (d1 * mod * two32)
        let ref := sinkGain * voiceSin (phasorPhase carClk 2000)
        if (got[t]! - ref).abs > e0 then e0 := (got[t]! - ref).abs
        if got[t]!.toBits != ref.toBits then bitDiff := bitDiff + 1
        if got[t]!.abs > maxOut then maxOut := got[t]!.abs
        if (got[t]! - fm1[t]!).abs > nestEffect then nestEffect := (got[t]! - fm1[t]!).abs
      let samples := n - lo
      IO.println s!"        nested standard rep (mod2→mod→carrier, same Horner Sin + integer phasor):"
      IO.println s!"        result   engine pm(pm) vs nested rep: max|Δ|={e0}  ·  bit-differing {bitDiff}/{samples}"
      IO.println s!"        nesting  |pm(pm) − single-level pm| max={nestEffect}  (level-2 must be non-trivial)"
      if maxOut < 1e-3 then
        failGate "pm-of-pm" s!"carrier silent (maxOut={maxOut})"
      else if nestEffect < 1e-3 then
        failGate "pm-of-pm" s!"level-2 negligible (nesting effect {nestEffect}) — not stressing the nest"
      else if bitDiff == 0 then
        passGate "pm-of-pm" s!"nested warp ≡ nested standard rep bit-for-bit ({bitDiff}/{samples}; nesting effect {nestEffect})"
      else
        failGate "pm-of-pm" s!"{bitDiff}/{samples} bit-differing (max|Δ|={e0}) — nested substitution diverges"

/-- The NEGATIVE-TIME boundary — the moat. A 20-sample delay warp pulls the clock
    negative for t < 20, so the carrier is evaluated BEFORE sample 0. Closed-form
    random access gives the exact backward-extrapolated sine; a streaming delay
    line could only emit zeros (no past). Asserts: (1) bit-exact vs a random-access
    standard rep at ALL t including negative time, and (2) the output at negative
    time is non-zero — the engine does NOT zero-pad, which is the random-access
    capability a stream cannot have. -/
def runNegativeClock (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let n : Nat := 1024
  let delta : Nat := 20
  let twoPi : Float := 6.283185307179586
  let sinkGain : Float := 0.05
  let delayTap : Tropical.EmitArrow.Tap :=
    { name := "d"
      warp := fun c => Tropical.EmitArrow.sub c
        (Tropical.EmitArrow.toIntE (Tropical.EmitArrow.lit (Int.ofNat delta * 4294967296)))
      weight := Tropical.EmitArrow.lit 1 }
  match buildAndFinish (Tropical.EmitArrow.buildTapCarrier "DelayFc"
          (Tropical.EmitArrow.litPitchVoice 2000) #[delayTap] arena resolved) with
  | .error e => failGate "negative-clock" s!"build: {firstLine e}"
  | .ok plan =>
    match ← renderPlanSamples plan n with
    | .error e => failGate "negative-clock" s!"render: {firstLine e}"
    | .ok got =>
      let mut bitDiff : Nat := 0
      let mut maxOut : Float := 0.0
      let mut negCount : Nat := 0
      let mut negMag : Float := 0.0      -- |output| at negative-time samples (streaming ⇒ 0)
      let mut negBitDiff : Nat := 0
      for t in [0:n] do
        let clk : Int := Int.ofNat t * 4294967296
        let phi : Int := clk - Int.ofNat delta * 4294967296   -- (t − 20)·2³²
        let ref := sinkGain * voiceSin (phasorPhase phi 2000)
        if got[t]!.toBits != ref.toBits then bitDiff := bitDiff + 1
        if got[t]!.abs > maxOut then maxOut := got[t]!.abs
        if phi < 0 then
          negCount := negCount + 1
          if got[t]!.abs > negMag then negMag := got[t]!.abs
          if got[t]!.toBits != ref.toBits then negBitDiff := negBitDiff + 1
      IO.println s!"        random-access rep: osc(φ), φ=(t−{delta})·2³², INCLUDING negative time:"
      IO.println s!"        result   engine vs random-access rep: bit-differing {bitDiff}/{n}  (neg-time samples {negCount}, differing {negBitDiff})"
      IO.println s!"        moat     |output| at negative time max={negMag}  (a streaming delay line would emit 0 here)"
      if maxOut < 1e-3 then
        failGate "negative-clock" s!"silent (maxOut={maxOut})"
      else if negCount == 0 then
        failGate "negative-clock" s!"delay didn't pull the clock negative ({negCount} neg samples)"
      else if negMag < 1e-3 then
        failGate "negative-clock" s!"engine zero-pads at negative time (negMag={negMag}) — not random-access"
      else if bitDiff == 0 then
        passGate "negative-clock" s!"random-access exact at negative time ({bitDiff}/{n}; {negCount} neg-time samples, |out|≤{negMag} where a stream emits 0)"
      else
        failGate "negative-clock" s!"{bitDiff}/{n} bit-differing — negative-time phasor diverges"

-- ── (h⁶) PRODUCTS / MIMO standard-rep differential (the DATA axis) ────────────
-- `MorphOsc` built from the cartesian combinators (ClockPhasor ⋙ (saw &&& Sin)
-- ⋙ crossfade) vs a straight-line reimplementation reusing the SAME integer
-- phasor and Horner `Sin`. No warp, no sub-sample clock — so this is BIT-EXACT
-- (like the convolution oracle, not the modulated-clock tolerance check). Three
-- morph settings prove the diagonal feeds two GENUINELY DIFFERENT consumers and
-- the crossfade blends them: morph=0 ≡ pure saw, morph=1 ≡ pure sine, morph=0.5
-- ≡ the blend — each bit-exact, and saw ≢ sine (non-degenerate MIMO).
def runMorphOscDifferential (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let n : Nat := 1024
  let lo : Nat := 4
  let freqHz : Int := 2000
  let twoPi : Float := 6.283185307179586
  let sinkGain : Float := 0.05
  -- the standard rep: the SAME crossfade arithmetic the engine emits, on the
  -- SAME integer phasor + Horner Sin (`(1−m)·(2·phase−1) + m·Sin(2π·phase)`).
  let refOut := fun (morphF : Float) (clk : Int) =>
    let phase := phasorPhase clk freqHz
    sinkGain * ((1.0 - morphF) * (2.0 * phase - 1.0) + morphF * voiceSin phase)
  let render := fun (nm : String) (m : Tropical.EmitArrow.Sig) =>
    match buildAndFinish (Tropical.EmitArrow.buildMorphOscLit nm freqHz m arena resolved) with
    | .error e => (pure (.error e) : IO (Except String (Array Float)))
    | .ok plan => renderPlanSamples plan n
  match ← render "MorphSaw" (Tropical.EmitArrow.lit 0),
        ← render "MorphSin" (Tropical.EmitArrow.lit 1),
        ← render "MorphBlend" (Tropical.EmitArrow.lit 5 1) with
  | .error e, _, _ | _, .error e, _ | _, _, .error e =>
    failGate "morphosc-mimo" s!"build/render: {firstLine e}"
  | .ok saw, .ok sinv, .ok blend =>
    let mut sawDiff : Nat := 0
    let mut sinDiff : Nat := 0
    let mut blendDiff : Nat := 0
    let mut maxBlend : Float := 0.0
    let mut sawVsSin : Float := 0.0        -- the diagonal feeds two distinct shapes
    for t in [lo:n] do
      let clk : Int := Int.ofNat t * 4294967296
      if saw[t]!.toBits   != (refOut 0.0 clk).toBits then sawDiff   := sawDiff   + 1
      if sinv[t]!.toBits  != (refOut 1.0 clk).toBits then sinDiff   := sinDiff   + 1
      if blend[t]!.toBits != (refOut 0.5 clk).toBits then blendDiff := blendDiff + 1
      if blend[t]!.abs > maxBlend then maxBlend := blend[t]!.abs
      if (saw[t]! - sinv[t]!).abs > sawVsSin then sawVsSin := (saw[t]! - sinv[t]!).abs
    let samples := n - lo
    IO.println s!"        standard rep = same integer phasor + Horner Sin, same crossfade arithmetic:"
    IO.println s!"        result   engine MorphOsc vs std rep:  bit-differing  saw {sawDiff}/{samples} · sine {sinDiff}/{samples} · blend {blendDiff}/{samples}"
    IO.println s!"        mimo     diagonal feeds distinct consumers: max|saw−sine|={sawVsSin}"
    if maxBlend < 1e-3 then
      failGate "morphosc-mimo" s!"carrier silent (maxBlend={maxBlend})"
    else if sawVsSin < 1e-2 then
      failGate "morphosc-mimo" s!"saw ≈ sine (max|Δ|={sawVsSin}) — diagonal degenerate"
    else if sawDiff == 0 && sinDiff == 0 && blendDiff == 0 then
      passGate "morphosc-mimo" s!"ClockPhasor ⋙ (saw &&& Sin) ⋙ crossfade ≡ standard rep, bit-exact (saw/sine/blend 0/{samples}; max|saw−sine|={sawVsSin})"
    else
      failGate "morphosc-mimo" s!"bit-differing (saw {sawDiff} · sine {sinDiff} · blend {blendDiff}) — MIMO build diverges from the standard rep"
