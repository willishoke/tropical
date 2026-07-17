import Tropical.Tropicaltest.Stress

/-!
# Tropical.Tropicaltest.Slide

The slide (warp-push) gates — downstream insert ≡ upstream warp, by the compiler — plus the bootstrap voices (phasor+sine/exp as terms), fixed-sine accuracy, and the banks-as-data equivalences (looped ≡ unrolled, float and int folds, columnized and ragged-bail).
-/

open Tropical
open Tropical.Plan
open Tropical.Ir (Arena ProgramIdx)

-- ── (h⁷) THE SLIDE (WARP-PUSH): downstream insert → upstream warp, by the compiler ─
-- The reified arrow term + `normalize` push warps up to the generators. Three
-- gates: (1) byte-identity vs stdlib FlangeSin lives in the corpus section
-- (slide(osc ⋙ flange) ≡ hand-written upstream FlangeSin); (2) slide-past-arr —
-- a pointwise shaper between osc and flange, so the warp must COMMUTE PAST it
-- (R1); (3) cascade — osc ⋙ flange ⋙ flange yields the 9-tap convolved
-- multiplicity automatically.

/-- Test 2: the warp must slide PAST a pointwise shaper to reach the generator.
    `slide(osc ⋙ shaper ⋙ flange)` must byte-equal the hand-written upstream form
    (shaper applied to osc at each warped clock). Byte-equal ⇒ R1 fired. -/
def runSlidePastArr (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  match Tropical.EmitArrow.buildSlideShaperDownstream arena resolved,
        Tropical.EmitArrow.buildSlideShaperUpstream arena resolved with
  | .ok (aD, iD), .ok (aU, iU) =>
    match emitResolvedWire aD iD, emitResolvedWire aU iU with
    | .ok bytesD, .ok bytesU =>
      if bytesD == bytesU then
        passGate "slide-past-arr" s!"warp commuted past the shaper: slide(osc ⋙ shaper ⋙ flange) ≡ upstream ({bytesD.length}B)"
      else
        failGate "slide-past-arr" s!"slide(downstream) ≠ upstream (down {bytesD.length}B, up {bytesU.length}B) — R1 (commute past arr) wrong"
    | .error e, _ | _, .error e => failGate "slide-past-arr" s!"emit: {firstLine e}"
  | .error e, _ | _, .error e => failGate "slide-past-arr" s!"build: {firstLine e}"

/-- Test 4: the product slide law. `slide(warp φ (x ⊗ y))` must byte-equal the
    hand-written upstream form (φ on each factor). Byte-equal ⇒ the warp
    distributed over `×` — both factors of the VCA reclock. This is what makes
    `prod` (signal×signal, the amplitude/VCA multiply that `scale` can't express)
    lawful under the slide, so an envelope factored as its own term rides every
    downstream delay tap. -/
def runSlideProd (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  match Tropical.EmitArrow.buildSlideProdDownstream arena resolved,
        Tropical.EmitArrow.buildSlideProdUpstream arena resolved with
  | .ok (aD, iD), .ok (aU, iU) =>
    match emitResolvedWire aD iD, emitResolvedWire aU iU with
    | .ok bytesD, .ok bytesU =>
      if bytesD == bytesU then
        passGate "slide-past-prod" s!"warp distributed over ×: slide(warp(x ⊗ y)) ≡ (warp x) ⊗ (warp y) ({bytesD.length}B)"
      else
        failGate "slide-past-prod" s!"slide(downstream) ≠ upstream (down {bytesD.length}B, up {bytesU.length}B) — warp did NOT distribute over the product"
    | .error e, _ | _, .error e => failGate "slide-past-prod" s!"emit: {firstLine e}"
  | .error e, _ | _, .error e => failGate "slide-past-prod" s!"build: {firstLine e}"

/-- THE BOOTSTRAP gate. A `FixedSinOsc` built as a TERM over `{clk, +, ×, round}`
    (`fixedSinOscTerm` = `Sin(2π·phasor)`, no `gen`, no `.trop` instance) must
    render bit-for-bit identical to the `.trop` `FixedSinOsc` at the same pitch and
    clock. Bit-exact ⇒ the generator IS the term — the arrow layer no longer needs
    `.trop` for its atoms; the phasor and the sine are `{clk, +, ×}` all the way
    down. -/
def runBootstrapSin (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let refPlan := buildAndFinish (Tropical.EmitArrow.buildClockCarrier "boot_ref" Tropical.EmitArrow.clockLit arena resolved)
  let termPlan := buildAndFinish (.ok (Tropical.EmitArrow.buildBootstrapSinOsc "boot_term" arena))
  match refPlan, termPlan with
  | .ok rp, .ok tp =>
    match ← renderPlanSamples rp 2048, ← renderPlanSamples tp 2048 with
    | .ok refS, .ok termS =>
      let n := min refS.size termS.size
      let mut bitDiff := 0
      let mut maxAbs : Float := 0.0
      let mut energy : Float := 0.0
      for i in [0:n] do
        energy := energy + refS[i]! * refS[i]!
        if refS[i]! != termS[i]! then bitDiff := bitDiff + 1
        let d := (refS[i]! - termS[i]!).abs
        if d > maxAbs then maxAbs := d
      IO.println s!"        term = Sin(2π·phasor) over the clock leaf, no gen; ref = .trop FixedSinOsc @220:"
      IO.println s!"        result   term vs .trop:  bit-differing {bitDiff}/{n}  ·  max|Δ|={maxAbs}  ·  energy={energy}"
      if bitDiff == 0 && energy > 1e-6 then
        passGate "bootstrap-sin" s!"phasor+sine as terms ≡ .trop FixedSinOsc, bit-exact ({n} samples, energy={energy})"
      else
        failGate "bootstrap-sin" s!"bit-differing {bitDiff}/{n} (max|Δ|={maxAbs}) — the term diverges from the .trop generator"
    | .error e, _ | _, .error e => failGate "bootstrap-sin" s!"render: {firstLine e}"
  | .error e, _ | _, .error e => failGate "bootstrap-sin" s!"build: {firstLine e}"

/-- THE BOOTSTRAP-EXP gate. `expSig` (the modal envelope primitive, transcribed
    from stdlib/Exp) evaluated by the engine over a ramp `x∈[−10,10]` must match
    libm `exp` to its minimax tolerance. An independent oracle (true exp, not a
    second copy of the same polynomial), so a transcribed-coefficient typo shows
    up as error ≫ 1e-5. This is the envelope's `bootstrap-sin`. -/
def runBootstrapExp (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  match buildAndFinish (.ok (Tropical.EmitArrow.buildExpProbe "exp_probe" arena)) with
  | .ok p =>
    match ← renderPlanSamples p 2048 with
    | .ok s =>
      let n := min s.size 2048
      let sinkGain : Float := Tropical.Plan.defaultSinkGain.toFloat   -- the carrier's output sink
      let mut maxRel : Float := 0.0
      let mut worstX : Float := 0.0
      for i in [0:n] do
        let x := i.toFloat * 0.009765625 - 10.0
        let ref := sinkGain * Float.exp x
        let rel := (s[i]! - ref).abs / ref
        if rel > maxRel then
          maxRel := rel
          worstX := x
      IO.println s!"        expSig(x) vs libm exp, x∈[−10,10] across 2048 samples:"
      IO.println s!"        result   max relative error = {maxRel}  (at x={worstX})"
      if maxRel < 1e-5 then
        passGate "bootstrap-exp" s!"emitted polynomial exp ≡ true exp to {maxRel} (minimax) — transcription correct"
      else
        failGate "bootstrap-exp" s!"max rel err {maxRel} (want <1e-5) at x={worstX}"
    | .error e => failGate "bootstrap-exp" s!"render: {firstLine e}"
  | .error e => failGate "bootstrap-exp" s!"build: {firstLine e}"

/-- THE FIXED-SINE ACCURACY gate. `fixedSinCycSig` (the Q2.30 integer-datapath
    sine) over the integer phasor, rendered by the engine, vs the TRUE sine at
    the exactly-known phase: the phasor model `P(i) = (21426140·i) mod 2³²` is
    replicated in Lean Int arithmetic, so the oracle `sin(2π·P/2³²)` is
    independent of every polynomial under test (a transcribed-coefficient typo
    or a mis-shifted Horner step shows up directly). Budget: coefficient
    rounding + 9 floor-shifts ≈ 1e-8 abs on the sin scale (−160 dB). -/
def runFixedSinAccuracy (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  match buildAndFinish (.ok (Tropical.EmitArrow.buildExprCarrier "fixedsin_acc"
      (Tropical.EmitArrow.fixedOutQ 30
        (Tropical.EmitArrow.fixedSinCycSig
          (Tropical.EmitArrow.fixedPhase Tropical.EmitArrow.clockLit))) arena)) with
  | .ok p =>
    match ← renderPlanSamples p 4096 with
    | .ok s =>
      let n := min s.size 4096
      let sinkGain : Float := Tropical.Plan.defaultSinkGain.toFloat
      let twoPi : Float := 6.283185307179586
      let mut maxAbs : Float := 0.0
      let mut worstI : Nat := 0
      for i in [0:n] do
        let pQ : Int := (21426140 * (Int.ofNat i)) % 4294967296
        let ref := Float.sin (twoPi * (Float.ofInt pQ) / 4294967296.0)
        let d := (s[i]! / sinkGain - ref).abs
        if d > maxAbs then
          maxAbs := d
          worstI := i
      IO.println s!"        fixedSin(Q0.32 phasor @220) vs true sin at the exact integer phase, 4096 samples:"
      IO.println s!"        result   max abs error (sin scale) = {maxAbs * 1e9}e-9  (at sample {worstI})"
      if maxAbs < 2e-8 then
        passGate "fixedsin-accuracy" s!"Q2.30 datapath sine ≡ true sine to {maxAbs * 1e9}e-9 (≈ −160 dB floor)"
      else
        failGate "fixedsin-accuracy" s!"max abs err {maxAbs * 1e9}e-9 (want <2e-8) at sample {worstI}"
    | .error e => failGate "fixedsin-accuracy" s!"render: {firstLine e}"
  | .error e => failGate "fixedsin-accuracy" s!"build: {firstLine e}"

/-- THE FIXED-SINE LONG-τ gate. The fixed oscillator read 2³⁰+12345 samples
    into the future must equal the origin oscillator phase-shifted by the
    EXACTLY-computable Q0.32 offset `(inc·K) mod 2³²` — modular arithmetic on
    the circle, byte-for-byte, at any τ. (The float carrier had no such
    identity: its phase argument grew without bound.) K deliberately has low
    bits set so nothing is accidentally exact. -/
def runFixedSinLongTau (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let K : Int := 1073741824 + 12345
  let Kq : Int := K * 4294967296
  let offset : Int := (21426140 * K) % 4294967296
  let farOsc := Tropical.EmitArrow.fixedOutQ 30
    (Tropical.EmitArrow.fixedSinCycSig
      (Tropical.EmitArrow.fixedPhase
        (Tropical.EmitArrow.add Tropical.EmitArrow.clockLit (Tropical.EmitArrow.litI Kq))))
  let shiftedOsc := Tropical.EmitArrow.fixedOutQ 30
    (Tropical.EmitArrow.fixedSinCycSig
      (Tropical.EmitArrow.bitAnd
        (Tropical.EmitArrow.add
          (Tropical.EmitArrow.fixedPhase Tropical.EmitArrow.clockLit)
          (Tropical.EmitArrow.litI offset))
        (Tropical.EmitArrow.lit 4294967295)))
  match buildAndFinish (.ok (Tropical.EmitArrow.buildExprCarrier "fixedsin_lt_far" farOsc arena)),
        buildAndFinish (.ok (Tropical.EmitArrow.buildExprCarrier "fixedsin_lt_shift" shiftedOsc arena)) with
  | .ok fp, .ok sp =>
    match ← renderPlanSamples fp 2048, ← renderPlanSamples sp 2048 with
    | .ok far, .ok shifted =>
      let n := min far.size shifted.size
      let bitDiff := bitDiffCount far shifted
      let mut energy : Float := 0.0
      for i in [0:n] do energy := energy + far[i]! * far[i]!
      IO.println s!"        fixed osc @ clk+(2³⁰+12345) samples vs origin osc phase-shifted (inc·K mod 2³²):"
      IO.println s!"        result   bit-differing {bitDiff}/{n}  ·  energy={energy}"
      if bitDiff == 0 && energy > 1e-6 then
        passGate "fixedsin-longtau" "modular phase identity byte-exact at τ+2³⁰ samples"
      else
        failGate "fixedsin-longtau" s!"bitDiff={bitDiff} (want 0) energy={energy} (>1e-6)"
    | .error e, _ | _, .error e => failGate "fixedsin-longtau" s!"render: {firstLine e}"
  | .error e, _ | _, .error e => failGate "fixedsin-longtau" s!"build: {firstLine e}"

/-- THE MODAL ISLAND gate. A decaying-resonator bank (`Σ amp·e^{−σd}·cos(ωd)`,
    gated causal at a strike time) built through the ARROW path (`arrUn`/`clk`,
    then `emitTerm`) must render bit-for-bit identical to the same bank built
    straight-line — the standard-rep differential for the pole/modal island's
    emit path. We also assert the two properties that make it a MODAL signal and
    not noise: causality (exactly silent before the strike — a streaming reverb
    could not gate a future-anchored tail) and decay (the tail loses energy).
    Bit-exact ⇒ the arrow layer realises the bank without corruption; silent+
    decaying ⇒ it is a real closed-form resonator bank, random-access by clk. -/
def runModalBank (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let modes : Array Tropical.EmitArrow.ModalMode := #[
    Tropical.EmitArrow.ModalMode.hz (Tropical.EmitArrow.lit 220) (Tropical.EmitArrow.lit 30 1) (Tropical.EmitArrow.lit 6 1),
    Tropical.EmitArrow.ModalMode.hz (Tropical.EmitArrow.lit 330) (Tropical.EmitArrow.lit 40 1) (Tropical.EmitArrow.lit 4 1),
    Tropical.EmitArrow.ModalMode.hz (Tropical.EmitArrow.lit 440) (Tropical.EmitArrow.lit 55 1) (Tropical.EmitArrow.lit 3 1)]
  let anchor := Tropical.EmitArrow.lit 200
  let arrowPlan := buildAndFinish (.ok (Tropical.EmitArrow.buildModalBankArrow "modal_arrow" modes anchor arena))
  let directPlan := buildAndFinish (.ok (Tropical.EmitArrow.buildModalBankDirect "modal_direct" modes anchor arena))
  match arrowPlan, directPlan with
  | .ok ap, .ok dp =>
    match ← renderPlanSamples ap 2048, ← renderPlanSamples dp 2048 with
    | .ok aS, .ok dS =>
      let n := min aS.size dS.size
      let bitDiff := bitDiffCount aS dS
      let mut preMax : Float := 0.0
      for i in [0:200] do
        let a := aS[i]!.abs
        if a > preMax then preMax := a
      let mut eEarly : Float := 0.0
      let mut eLate : Float := 0.0
      for i in [200:600] do eEarly := eEarly + aS[i]! * aS[i]!
      for i in [1648:2048] do eLate := eLate + aS[i]! * aS[i]!
      IO.println s!"        bank = Σ amp·e^(−σd)·cos(2πf·d) @ 220/330/440, struck @ sample 200 (d=clk/2³²/SR−anchor):"
      IO.println s!"        result   arrow vs straight-line:  bit-differing {bitDiff}/{n}  ·  pre-strike |max|={preMax}  ·  E[early]={eEarly}  E[late]={eLate}"
      if bitDiff == 0 && preMax == 0.0 && eEarly > 1e-6 && eLate < eEarly then
        passGate "modal-bank" s!"gated decaying-sinusoid bank: arrow ≡ straight-line bit-exact, causal (silent pre-strike), decaying ({n} samples)"
      else
        failGate "modal-bank" s!"bitDiff={bitDiff} preMax={preMax} (want 0) eEarly={eEarly} (>1e-6) eLate={eLate} (<eEarly)"
    | .error e, _ | _, .error e => failGate "modal-bank" s!"render: {firstLine e}"
  | .error e, _ | _, .error e => failGate "modal-bank" s!"build: {firstLine e}"

/-- Warn-only cost ratchet (gates with rank): the FLATNESS assertions in the
    banks gates are HARD — an asymptotic regression means banking is broken,
    not slow, and that check never false-positives on an honest change. The
    CONSTANTS are honest-change territory — a legitimate emit change may move
    them — so drift prints a WARN line (never fails). Refreeze deliberately by
    updating the constant at the call site. Warnings rot when nobody reads
    them; this one is a single greppable token: `WARN`. -/
def warnBenchConst (gate what : String) (frozen got : Nat) : IO Unit :=
  if got != frozen then
    IO.println s!"  WARN  {gate}  {what}: {got} (frozen {frozen}) — cost constant drifted; refreeze deliberately"
  else pure ()

open Tropical.EmitArrow in
/-- THE BANKS-AS-DATA gate (slice 3b). A decaying-resonator bank lowered through
    the INDEXED REDUCTION (`modalBankSigTable` → `Sig.bankSum` → a `ReduceBegin`
    region) must render BIT-FOR-BIT identical to the same bank unrolled
    (`modalBankSigDirect`) — the i64-modular mode sum is associative, so the loop
    and the fold agree to the bit. This exercises the whole new path end to end:
    `Sig.arr`/`index`/`loopIdx`/`bankSum` through every strata pass, the
    `ENode→CNode` downcast, and the emit-time reduce-region lowering. We also
    assert the PAYOFF: banking shrinks the plan, and the per-mode MARGINAL
    instruction cost drops (the DSP body no longer unrolls — only the coefficient
    fills still scale, and those are destined for the s0 kernel next). -/
def runBanksAsData (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  -- A K-mode decaying bank, all deg-0 (the uniform datapath). Frequencies spread
  -- so the bank is non-trivial; small amps so the i64 sum stays in headroom.
  let mkModes (k : Nat) : Array ModalMode :=
    (Array.range k).map fun i =>
      ModalMode.hz (lit (Int.ofNat (220 + 40 * i))) (lit 30 1) (lit 2 1)
  let anchor := lit 200
  let modes := mkModes 12
  let directPlan := buildAndFinish (.ok (buildModalBankDirect "bank_unrolled" modes anchor arena))
  let tablePlan  := buildAndFinish (.ok (buildModalBankTable  "bank_looped"   modes anchor arena))
  match directPlan, tablePlan with
  | .ok dp, .ok tp =>
    match ← renderPlanSamples dp 2048, ← renderPlanSamples tp 2048 with
    | .ok dS, .ok tS =>
      let n := min dS.size tS.size
      let bitDiff := bitDiffCount dS tS
      let mut preMax : Float := 0.0
      for i in [0:200] do
        let a := tS[i]!.abs
        if a > preMax then preMax := a
      let mut eEarly : Float := 0.0
      let mut eLate : Float := 0.0
      for i in [200:600] do eEarly := eEarly + tS[i]! * tS[i]!
      for i in [1648:2048] do eLate := eLate + tS[i]! * tS[i]!
      -- Compile-scaling: the same bank at two mode counts, both lowerings.
      let dSmall := buildAndFinish (.ok (buildModalBankDirect "d6"  (mkModes 6)  anchor arena))
      let dBig   := buildAndFinish (.ok (buildModalBankDirect "d24" (mkModes 24) anchor arena))
      let tSmall := buildAndFinish (.ok (buildModalBankTable  "t6"  (mkModes 6)  anchor arena))
      let tBig   := buildAndFinish (.ok (buildModalBankTable  "t24" (mkModes 24) anchor arena))
      match dSmall, dBig, tSmall, tBig with
      | .ok ds, .ok db, .ok ts, .ok tb =>
        let dMarginal := planInstrCount db - planInstrCount ds   -- unrolled per-mode marginal (×18)
        let tMarginal := planInstrCount tb - planInstrCount ts   -- banked marginal (fills only)
        let shrinks := decide (planInstrCount tp < planInstrCount dp)
        IO.println s!"        bank = Σ amp·e^(−σd)·cos(2πf·d), 12 modes, struck @ 200 — unrolled vs looped:"
        IO.println s!"        result   bit-differing {bitDiff}/{n}  ·  pre-strike |max|={preMax}  ·  E[early]={eEarly}  E[late]={eLate}"
        IO.println s!"        payoff   plan-instrs 12-mode: unrolled={planInstrCount dp} looped={planInstrCount tp} (shrinks={shrinks})"
        IO.println s!"        payoff   per-mode marginal (6→24 modes): unrolled +{dMarginal}  ·  banked +{tMarginal} (body no longer unrolls)"
        warnBenchConst "banks-as-data" "12-mode looped plan-instrs" 184 (planInstrCount tp)
        warnBenchConst "banks-as-data" "banked per-mode marginal (6→24)" 72 tMarginal
        if bitDiff == 0 && preMax == 0.0 && eEarly > 1e-6 && eLate < eEarly
           && shrinks && tMarginal < dMarginal then
          passGate "banks-as-data" s!"looped ≡ unrolled bit-exact ({n} samples), causal, decaying; plan shrinks, marginal +{tMarginal}<+{dMarginal}"
        else
          failGate "banks-as-data" s!"bitDiff={bitDiff} preMax={preMax} shrinks={shrinks} tMarg={tMarginal} dMarg={dMarginal}"
      | _, _, _, _ =>
        failGate "banks-as-data" "scaling build failed"
    | .error e, _ | _, .error e => failGate "banks-as-data" s!"render: {firstLine e}"
  | .error e, _ | _, .error e => failGate "banks-as-data" s!"build: {firstLine e}"

open Tropical.EmitArrow in
/-- THE BANKS-AS-DATA DIRECTION gate. The direction bank lowered through TWO
    indexed reductions over one set of coefficient columns
    (`modalBankSigDirTable` — the forward and reverse accumulators as two
    `bankFold`s) must render BIT-FOR-BIT identical to the unrolled pair-fold
    (`modalBankSigDir`), across the crossfade (dir = 0.5), the pure reverse
    (dir = 1 — the ANTI-CAUSAL region must actually carry energy, so the mirrored
    phase `modePhaseQFromIncr(incr, −clkRel)` is genuinely exercised), and the
    sway path (`dampScale?` threading the columns). Also asserts the payoff:
    the banked plan shrinks, and both regions loop (per-mode marginal collapses).
    This is the gate that retires the "hand-bank every effect" objection: no
    direction table twin exists — both sides route through the SAME generic
    `bankFold`, and this gate pins that the generic path carries a richer body
    (two accumulators, mirrored phase, sway) without a transcription step. -/
def runBanksAsDataDir (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let mkModes (k : Nat) : Array ModalMode :=
    (Array.range k).map fun i =>
      ModalMode.hz (lit (Int.ofNat (220 + 40 * i))) (lit 30 1) (lit 2 1)
  let anchor := lit 200
  let modes := mkModes 12
  let sway : Option (Sig × Sig) := some (lit 5 1, lit 20 1)
  -- explicit lambdas: Lean eta-expands optParam references by inserting the
  -- default, which would drop the `dampScale?` slot from the function type
  let unrolled : Array ModalMode → Sig → Sig → Sig → Option Sig → Sig :=
    fun ms clk a d s? => modalBankSigDir ms clk a d s?
  let looped : Array ModalMode → Sig → Sig → Sig → Option Sig → Sig :=
    fun ms clk a d s? => modalBankSigDirTable ms clk a d s?
  -- Three configs: crossfade, pure reverse, crossfade+sway — each unrolled vs banked.
  let cfgs : Array (String × Sig × Option (Sig × Sig)) :=
    #[("mid", litF 0.5, none), ("rev", lit 1, none), ("sway", litF 0.5, sway)]
  let mut ok := true
  let mut planPair : Option (Nat × Nat) := none
  for (tag, dir, damp?) in cfgs do
    let uPlan := buildAndFinish (.ok (buildModalBankDirWith unrolled
      s!"dir_{tag}_unrolled" modes anchor dir arena damp?))
    let tPlan := buildAndFinish (.ok (buildModalBankDirWith looped
      s!"dir_{tag}_looped" modes anchor dir arena damp?))
    match uPlan, tPlan with
    | .ok up, .ok tp =>
      match ← renderPlanSamples up 2048, ← renderPlanSamples tp 2048 with
      | .ok uS, .ok tS =>
        let n := min uS.size tS.size
        let bitDiff := bitDiffCount uS tS
        -- the reverse config must carry PRE-STRIKE energy (the anti-causal loop lives)
        let mut preE : Float := 0.0
        for i in [0:200] do preE := preE + tS[i]! * tS[i]!
        let preOk := tag != "rev" || preE > 1e-6
        if tag == "mid" then planPair := some (planInstrCount up, planInstrCount tp)
        if bitDiff != 0 || !preOk then
          IO.println s!"        dir[{tag}]  bitDiff={bitDiff}/{n} preE={preE} — MISMATCH"
          ok := false
        else
          IO.println s!"        dir[{tag}]  bit-identical ({n} samples){if tag == "rev" then s!", pre-strike E={preE}" else ""}"
      | .error e, _ | _, .error e =>
        IO.println s!"  FAIL  banks-as-data-dir  render[{tag}]: {firstLine e}"; ok := false
    | .error e, _ | _, .error e =>
      IO.println s!"  FAIL  banks-as-data-dir  build[{tag}]: {firstLine e}"; ok := false
  -- Payoff: banked direction plan shrinks at 12 modes; per-mode marginal collapses
  -- (BOTH regions loop — the marginal is the column fills alone).
  let uSmall := buildAndFinish (.ok (buildModalBankDirWith unrolled "du6"  (mkModes 6)  anchor (litF 0.5) arena))
  let uBig   := buildAndFinish (.ok (buildModalBankDirWith unrolled "du24" (mkModes 24) anchor (litF 0.5) arena))
  let tSmall := buildAndFinish (.ok (buildModalBankDirWith looped "dt6"  (mkModes 6)  anchor (litF 0.5) arena))
  let tBig   := buildAndFinish (.ok (buildModalBankDirWith looped "dt24" (mkModes 24) anchor (litF 0.5) arena))
  match planPair, uSmall, uBig, tSmall, tBig with
  | some (uc, tc), .ok us, .ok ub, .ok ts, .ok tb =>
    let uMarginal := planInstrCount ub - planInstrCount us
    let tMarginal := planInstrCount tb - planInstrCount ts
    let shrinks := decide (tc < uc)
    IO.println s!"        payoff   plan-instrs 12-mode: unrolled={uc} looped={tc} (shrinks={shrinks})"
    IO.println s!"        payoff   per-mode marginal (6→24 modes): unrolled +{uMarginal}  ·  banked +{tMarginal}"
    warnBenchConst "banks-as-data-dir" "12-mode looped plan-instrs" 317 tc
    warnBenchConst "banks-as-data-dir" "banked per-mode marginal (6→24)" 72 tMarginal
    if ok && shrinks && tMarginal < uMarginal then
      passGate "banks-as-data-dir" s!"looped ≡ unrolled bit-exact (mid/rev/sway), reverse audible; plan shrinks, marginal +{tMarginal}<+{uMarginal}"
    else
      failGate "banks-as-data-dir" s!"ok={ok} shrinks={shrinks} tMarg={tMarginal} uMarg={uMarginal}"
  | _, _, _, _, _ =>
    failGate "banks-as-data-dir" "scaling build failed"

open Tropical.EmitArrow in
/-- THE FLOAT-BANK gate (typed accumulator). A FLOAT fold lowered through
    `Sig.bankSum` must render bit-identical to the same fold unrolled. This is
    the claim that banking needs NO algebraic precondition: the loop visits
    elements in the order the unroll nests its adds, so order preservation —
    not associativity — carries bit-exactness, floats included. (The i64
    restriction in the original `compileBankSum` was scaffolding; the
    accumulator now follows the body's type.) -/
def runBanksFloat (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let k := 16
  -- t = τ seconds (the dSec recipe, anchor 0) — varies per sample so the sum
  -- is a live float datapath, not a constant the optimizer could fold away.
  let t := div (div (toFloatE clockLit) (lit 4294967296)) .sampleRate
  let amps := (Array.range k).map fun i => litF (0.31 + 0.07 * i.toFloat)
  let col := Sig.arr amps
  let unrolled := amps.foldl (fun acc a => add acc (mul a t)) (lit 0)
  let looped := Sig.bankSum k #[col] (mul (Sig.index col (Sig.loopIdx 0)) t) none 0
  let uPlan := buildAndFinish (.ok (buildExprCarrier "fbank_unrolled" unrolled arena))
  let tPlan := buildAndFinish (.ok (buildExprCarrier "fbank_looped" looped arena))
  match uPlan, tPlan with
  | .ok up, .ok tp =>
    match ← renderPlanSamples up 2048, ← renderPlanSamples tp 2048 with
    | .ok uS, .ok tS =>
      let n := min uS.size tS.size
      let bitDiff := bitDiffCount uS tS
      let mut energy : Float := 0.0
      for i in [0:n] do energy := energy + tS[i]! * tS[i]!
      let shrinks := decide (planInstrCount tp < planInstrCount up)
      IO.println s!"        float bank Σₖ ampₖ·t, {k} elements — unrolled vs looped (f64 accumulator):"
      IO.println s!"        result   bit-differing {bitDiff}/{n} · energy={energy} · plan-instrs unrolled={planInstrCount up} looped={planInstrCount tp}"
      warnBenchConst "banks-float" "looped plan-instrs" 12 (planInstrCount tp)
      if bitDiff == 0 && energy > 1e-6 && shrinks then
        passGate "banks-float" "looped ≡ unrolled bit-exact for a FLOAT fold (order preservation, no associativity); plan shrinks"
      else
        failGate "banks-float" s!"bitDiff={bitDiff} energy={energy} shrinks={shrinks}"
    | .error e, _ | _, .error e => failGate "banks-float" s!"render: {firstLine e}"
  | .error e, _ | _, .error e => failGate "banks-float" s!"build: {firstLine e}"

/-- THE TRUNK-FOLD gate (loop-everything). A surface-language SUMMING fold —
    through the FULL front door (raise → elaborate → strata → emit) — lowers to
    an indexed reduction, renders byte-identical to its hand-unrolled add chain,
    and the plan is FLAT in element count (the Pack carries the column; the loop
    body is O(1)). Horner folds (`acc·x + c`) are shape-ineligible and keep
    unrolling, so the transcendental stdlib is untouched — checked implicitly by
    every other gate in this suite. -/
def runBanksFoldTrunk (_arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let jn (m : Nat) (e : Nat) : Lean.Json := Lean.Json.num ⟨Int.ofNat m, e⟩
  let amp (i : Nat) : Lean.Json := jn (31 + 7 * i) 2          -- 0.31 + 0.07·i, exact decimals
  let addJ (a b : Lean.Json) : Lean.Json :=
    Lean.Json.mkObj [("op", Lean.Json.str "add"), ("args", Lean.Json.arr #[a, b])]
  let mulHalf (a : Lean.Json) : Lean.Json :=
    Lean.Json.mkObj [("op", Lean.Json.str "mul"), ("args", Lean.Json.arr #[a, jn 5 1])]
  let binding (n : String) : Lean.Json :=
    Lean.Json.mkObj [("op", Lean.Json.str "binding"), ("name", Lean.Json.str n)]
  let mkPatch (k : Nat) (unrolled : Bool) : Lean.Json :=
    let amps := (Array.range k).map amp
    let expr :=
      if unrolled then
        -- ((0 + c₀·½) + c₁·½) + … — the fold's own unroll order
        amps.foldl (fun acc a => addJ acc (mulHalf a)) (jn 0 0)
      else
        Lean.Json.mkObj [("op", Lean.Json.str "fold"), ("over", Lean.Json.arr amps),
          ("init", jn 0 0), ("acc_var", Lean.Json.str "acc"), ("elem_var", Lean.Json.str "e"),
          ("body", addJ (binding "acc") (mulHalf (binding "e")))]
    let inner := Lean.Json.mkObj [
      ("name", Lean.Json.str "FoldProbe"),
      ("ports", Lean.Json.mkObj [("inputs", Lean.Json.arr #[]),
        ("outputs", Lean.Json.arr #[Lean.Json.str "out"])]),
      ("body", Lean.Json.mkObj [("op", Lean.Json.str "block"),
        ("decls", Lean.Json.arr #[]),
        ("assigns", Lean.Json.arr #[Lean.Json.mkObj [
          ("op", Lean.Json.str "outputAssign"), ("name", Lean.Json.str "out"),
          ("expr", expr)]])])]
    Lean.Json.mkObj [
      ("schema", Lean.Json.str "tropical_program_2"),
      ("name", Lean.Json.str "fold_trunk_probe"),
      ("body", Lean.Json.mkObj [("op", Lean.Json.str "block"),
        ("decls", Lean.Json.arr #[
          Lean.Json.mkObj [("op", Lean.Json.str "programDecl"),
            ("name", Lean.Json.str "FoldProbe"), ("program", inner)],
          Lean.Json.mkObj [("op", Lean.Json.str "instanceDecl"),
            ("name", Lean.Json.str "p"), ("program", Lean.Json.str "FoldProbe"),
            ("inputs", Lean.Json.mkObj [])]]),
        ("assigns", Lean.Json.arr #[])]),
      ("audio_outputs", Lean.Json.arr #[Lean.Json.mkObj [
        ("instance", Lean.Json.str "p"), ("output", Lean.Json.str "out")]])]
  let compileAt (k : Nat) (unrolled : Bool) (tag : String) :
      IO (Except String Tropical.Plan.FlatPlan) := do
    let tmp := s!"/tmp/tropicaltest-fold-{tag}.json"
    IO.FS.writeFile tmp (mkPatch k unrolled).compress
    match ← compilePatch tmp .fused with
    | .error e => pure (.error e)
    | .ok planJson =>
      match Lean.Json.parse planJson with
      | .error e => pure (.error s!"parse: {e}")
      | .ok j => pure ((Tropical.Plan.FlatPlan.ofWire j).mapError (s!"ofWire: {·}"))
  match ← compileAt 16 false "f16", ← compileAt 16 true "u16" with
  | .ok fp, .ok up =>
    match ← renderPlanSamples fp 2048, ← renderPlanSamples up 2048 with
    | .ok fS, .ok uS =>
      let n := min fS.size uS.size
      let bitDiff := bitDiffCount fS uS
      let nonzero := fS.any (· != 0.0)
      match ← compileAt 8 false "f8", ← compileAt 64 false "f64" with
      | .ok f8, .ok f64 =>
        let d := planInstrCount f64 - planInstrCount f8
        let shrinks := decide (planInstrCount fp < planInstrCount up)
        let looping := Tropical.Ir.Strata.ArrayLower.banksLoopEnabled
        IO.println s!"        surface fold Σₖ ampₖ·½ through raise→elab→strata→emit, 16 elements (loop-everything={looping}):"
        IO.println s!"        result   bit-differing {bitDiff}/{n} vs hand-unrolled · nonzero={nonzero}"
        IO.println s!"        payoff   plan-instrs: fold(16)={planInstrCount fp} unrolled(16)={planInstrCount up} · fold(8)={planInstrCount f8} fold(64)={planInstrCount f64} (Δ={d})"
        if looping then
          warnBenchConst "banks-fold-trunk" "fold plan-instrs (any K)" 8 (planInstrCount fp)
          if bitDiff == 0 && nonzero && shrinks && d ≤ 2 then
            passGate "banks-fold-trunk" s!"surface fold banks: byte-equal to unroll, plan FLAT in K (Δ={d} ≤ 2, 8→64)"
          else
            failGate "banks-fold-trunk" s!"bitDiff={bitDiff} nonzero={nonzero} shrinks={shrinks} Δ={d}"
        else
          -- escape hatch: the fold must genuinely revert to unrolling
          if bitDiff == 0 && nonzero && !shrinks && d > 2 then
            passGate "banks-fold-trunk" s!"escape hatch reverts: fold unrolls (Δ={d} grows), byte-equal"
          else
            failGate "banks-fold-trunk" s!"(unroll mode) bitDiff={bitDiff} nonzero={nonzero} shrinks={shrinks} Δ={d}"
      | .error e, _ | _, .error e =>
        failGate "banks-fold-trunk" s!"scaling compile: {firstLine e}"
    | .error e, _ | _, .error e => failGate "banks-fold-trunk" s!"render: {firstLine e}"
  | .error e, _ | _, .error e => failGate "banks-fold-trunk" s!"compile: {firstLine e}"

/-- Wrap a single output expression in the minimal one-instance
    `tropical_program_2` patch the fold gates probe with (`p.out = expr`).
    `typeDefs` (optional) rides the inner program's ports — the tag-fold
    bail case needs a payload-less sum in scope. -/
private def foldProbePatchJson (expr : Lean.Json)
    (typeDefs : Array Lean.Json := #[]) : Lean.Json :=
  let ports :=
    if typeDefs.isEmpty then
      Lean.Json.mkObj [("inputs", Lean.Json.arr #[]),
        ("outputs", Lean.Json.arr #[Lean.Json.str "out"])]
    else
      Lean.Json.mkObj [("inputs", Lean.Json.arr #[]),
        ("outputs", Lean.Json.arr #[Lean.Json.str "out"]),
        ("type_defs", Lean.Json.arr typeDefs)]
  let inner := Lean.Json.mkObj [
    ("name", Lean.Json.str "FoldProbe"),
    ("ports", ports),
    ("body", Lean.Json.mkObj [("op", Lean.Json.str "block"),
      ("decls", Lean.Json.arr #[]),
      ("assigns", Lean.Json.arr #[Lean.Json.mkObj [
        ("op", Lean.Json.str "outputAssign"), ("name", Lean.Json.str "out"),
        ("expr", expr)]])])]
  Lean.Json.mkObj [
    ("schema", Lean.Json.str "tropical_program_2"),
    ("name", Lean.Json.str "fold_probe"),
    ("body", Lean.Json.mkObj [("op", Lean.Json.str "block"),
      ("decls", Lean.Json.arr #[
        Lean.Json.mkObj [("op", Lean.Json.str "programDecl"),
          ("name", Lean.Json.str "FoldProbe"), ("program", inner)],
        Lean.Json.mkObj [("op", Lean.Json.str "instanceDecl"),
          ("name", Lean.Json.str "p"), ("program", Lean.Json.str "FoldProbe"),
          ("inputs", Lean.Json.mkObj [])]]),
      ("assigns", Lean.Json.arr #[])]),
    ("audio_outputs", Lean.Json.arr #[Lean.Json.mkObj [
      ("instance", Lean.Json.str "p"), ("output", Lean.Json.str "out")]])]

/-- Compile a fold-probe expression through the FULL front door
    (raise → elaborate → strata → emit) and parse the resulting plan. -/
def compileFoldProbe (expr : Lean.Json) (tag : String)
    (typeDefs : Array Lean.Json := #[]) : IO (Except String Tropical.Plan.FlatPlan) := do
  let tmp := s!"/tmp/tropicaltest-columnize-{tag}.json"
  IO.FS.writeFile tmp (foldProbePatchJson expr typeDefs).compress
  match ← compilePatch tmp .fused with
  | .error e => pure (.error e)
  | .ok planJson =>
    match Lean.Json.parse planJson with
    | .error e => pure (.error s!"parse: {e}")
    | .ok j => pure ((Tropical.Plan.FlatPlan.ofWire j).mapError (s!"ofWire: {·}"))

-- Shared JSON expression builders for the columnize gates.
def cgJn (m : Nat) (e : Nat) : Lean.Json := Lean.Json.num ⟨Int.ofNat m, e⟩
def cgA (i : Nat) : Lean.Json := cgJn (31 + 7 * i) 2   -- aᵢ = 0.31 + 0.07·i
def cgB (i : Nat) : Lean.Json := cgJn (11 + 5 * i) 2   -- bᵢ = 0.11 + 0.05·i
def cgAdd (a b : Lean.Json) : Lean.Json :=
  Lean.Json.mkObj [("op", Lean.Json.str "add"), ("args", Lean.Json.arr #[a, b])]
private def cgMulHalf (a : Lean.Json) : Lean.Json :=
  Lean.Json.mkObj [("op", Lean.Json.str "mul"), ("args", Lean.Json.arr #[a, cgJn 5 1])]
private def cgIndex (a b : Lean.Json) : Lean.Json :=
  Lean.Json.mkObj [("op", Lean.Json.str "index"), ("args", Lean.Json.arr #[a, b])]
def cgBinding (n : String) : Lean.Json :=
  Lean.Json.mkObj [("op", Lean.Json.str "binding"), ("name", Lean.Json.str n)]
/-- One tuple contribution: `a·½ + b`. -/
private def cgTerm (a b : Lean.Json) : Lean.Json := cgAdd (cgMulHalf a) b
def cgFold (over : Lean.Json) (body : Lean.Json) : Lean.Json :=
  Lean.Json.mkObj [("op", Lean.Json.str "fold"), ("over", over),
    ("init", cgJn 0 0), ("acc_var", Lean.Json.str "acc"), ("elem_var", Lean.Json.str "e"),
    ("body", body)]

/-- THE COLUMNIZE gate (columnize-over-shapes). A surface-language summing fold
    over TUPLE elements — Σ (aᵢ·½ + bᵢ) over [[a₀,b₀],…] through the FULL front
    door — de-structures into per-position coefficient columns (the AoS→SoA iso
    `Array (A×B) ≅ Array A × Array B`, done generically by `tryBankFoldE`) and
    banks as ONE multi-table reduction: byte-equal to the hand-unrolled add
    chain, exactly one `ReduceBegin` region with n=2 column `Pack`s, and the
    plan FLAT in element count (HARD — a growth regression means banking is
    broken). Under `TROPICAL_BANKS_UNROLL` the fold genuinely unrolls
    (0 regions) and still matches bit-exact. -/
def runBanksColumnize (_arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let foldExpr (k : Nat) : Lean.Json :=
    let pairs := (Array.range k).map fun i => Lean.Json.arr #[cgA i, cgB i]
    cgFold (Lean.Json.arr pairs)
      (cgAdd (cgBinding "acc")
        (cgTerm (cgIndex (cgBinding "e") (cgJn 0 0)) (cgIndex (cgBinding "e") (cgJn 1 0))))
  let unrollExpr (k : Nat) : Lean.Json :=
    (Array.range k).foldl (fun acc i => cgAdd acc (cgTerm (cgA i) (cgB i))) (cgJn 0 0)
  match ← compileFoldProbe (foldExpr 8) "f8", ← compileFoldProbe (unrollExpr 8) "u8" with
  | .ok fp, .ok up =>
    match ← renderPlanSamples fp 2048, ← renderPlanSamples up 2048 with
    | .ok fS, .ok uS =>
      let n := min fS.size uS.size
      let bitDiff := bitDiffCount fS uS
      let nonzero := fS.any (· != 0.0)
      match ← compileFoldProbe (foldExpr 64) "f64" with
      | .ok f64 =>
        let d := planInstrCount f64 - planInstrCount fp
        let regions := planTagCount "ReduceBegin" fp
        let packs := planTagCount "Pack" fp
        let looping := Tropical.Ir.Strata.ArrayLower.banksLoopEnabled
        IO.println s!"        surface fold Σₖ (aₖ·½ + bₖ) over [[a₀,b₀],…] (K=8), full front door (loop-everything={looping}):"
        IO.println s!"        result   bit-differing {bitDiff}/{n} vs hand-unrolled · nonzero={nonzero} · regions={regions} · column-Packs={packs}"
        IO.println s!"        payoff   plan-instrs: fold(8)={planInstrCount fp} unrolled(8)={planInstrCount up} fold(64)={planInstrCount f64} (Δ={d})"
        if looping then
          warnBenchConst "banks-columnize" "tuple-fold plan-instrs (any K)" 11 (planInstrCount fp)
          if bitDiff == 0 && nonzero && regions == 1 && packs == 2 && d ≤ 2 then
            passGate "banks-columnize" s!"tuple fold banks as SoA: 1 region × 2 columns, byte-equal to unroll, plan FLAT in K (Δ={d} ≤ 2, 8→64)"
          else
            failGate "banks-columnize" s!"bitDiff={bitDiff} nonzero={nonzero} regions={regions} packs={packs} Δ={d}"
        else
          if bitDiff == 0 && nonzero && regions == 0 && d > 2 then
            passGate "banks-columnize" s!"escape hatch reverts: tuple fold unrolls (0 regions, Δ={d} grows), byte-equal"
          else
            failGate "banks-columnize" s!"(unroll mode) bitDiff={bitDiff} nonzero={nonzero} regions={regions} Δ={d}"
      | .error e =>
        failGate "banks-columnize" s!"scaling compile: {firstLine e}"
    | .error e, _ | _, .error e => failGate "banks-columnize" s!"render: {firstLine e}"
  | .error e, _ | _, .error e => failGate "banks-columnize" s!"compile: {firstLine e}"

/-- THE COLUMNIZE BAIL-OUT gate. The shapes `tryBankFoldE` refuses must still
    compile CORRECTLY via unrolling — never crash, never mis-bank:
    - RAGGED arities (a 2-tuple next to a 3-tuple) → unroll, byte-equal to the
      hand-written chain, 0 regions in BOTH flag states;
    - a NON-LITERAL index into the tuple element (`e[sampleIndex mod 2]`) → the
      symbolic tuple survives lowering, the residual guard bails, unroll — the
      alternating output pins that the dynamic index is genuinely live;
    - a fold over PAYLOAD-LESS TAGS: sum elements cannot reach ArrayLower as
      tags at all — SumLower rewrites them to scalar variant literals first —
      so the fold BANKS AS SCALARS (1 region when looping), today's behavior,
      asserted here so the `.tag` claim stays pinned. -/
def runBanksColumnizeBail (_arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let looping := Tropical.Ir.Strata.ArrayLower.banksLoopEnabled
  let check (name : String) (foldE unrollE : Lean.Json)
      (typeDefs : Array Lean.Json) (wantRegions : Nat) : IO (Option String) := do
    match ← compileFoldProbe foldE s!"bail-{name}-f" typeDefs,
          ← compileFoldProbe unrollE s!"bail-{name}-u" typeDefs with
    | .ok fp, .ok up =>
      match ← renderPlanSamples fp 2048, ← renderPlanSamples up 2048 with
      | .ok fS, .ok uS =>
        let n := min fS.size uS.size
        let bitDiff := bitDiffCount fS uS
        let nonzero := fS.any (· != 0.0)
        let regions := planTagCount "ReduceBegin" fp
        IO.println s!"        bail[{name}]  bit-differing {bitDiff}/{n} · nonzero={nonzero} · regions={regions} (want {wantRegions})"
        if bitDiff == 0 && nonzero && regions == wantRegions then pure none
        else pure (some s!"{name}: bitDiff={bitDiff} nonzero={nonzero} regions={regions} want={wantRegions}")
      | .error e, _ | _, .error e => pure (some s!"{name} render: {firstLine e}")
    | .error e, _ | _, .error e => pure (some s!"{name} compile: {firstLine e}")
  -- (a) ragged arities: [[a₀,b₀],[a₁,b₁,0.99]] — mixed 2-/3-tuples never bank.
  let raggedFold := cgFold
    (Lean.Json.arr #[Lean.Json.arr #[cgA 0, cgB 0],
                     Lean.Json.arr #[cgA 1, cgB 1, cgJn 99 2]])
    (cgAdd (cgBinding "acc")
      (cgTerm (cgIndex (cgBinding "e") (cgJn 0 0)) (cgIndex (cgBinding "e") (cgJn 1 0))))
  let raggedUnroll :=
    cgAdd (cgAdd (cgJn 0 0) (cgTerm (cgA 0) (cgB 0))) (cgTerm (cgA 1) (cgB 1))
  -- (b) non-literal index: e[sampleIndex mod 2] — the projection cannot fold,
  --     the symbolic tuple survives, the residual guard unrolls.
  let sampIdx := Lean.Json.mkObj [("op", Lean.Json.str "sampleIndex")]
  let dynIdx := Lean.Json.mkObj [("op", Lean.Json.str "mod"),
    ("args", Lean.Json.arr #[sampIdx, cgJn 2 0])]
  let dynFold := cgFold
    (Lean.Json.arr #[Lean.Json.arr #[cgA 0, cgB 0], Lean.Json.arr #[cgA 1, cgB 1]])
    (cgAdd (cgBinding "acc") (cgIndex (cgBinding "e") dynIdx))
  let dynUnroll :=
    cgAdd (cgAdd (cgJn 0 0) (cgIndex (Lean.Json.arr #[cgA 0, cgB 0]) dynIdx))
      (cgIndex (Lean.Json.arr #[cgA 1, cgB 1]) dynIdx)
  -- (c) payload-less tags: SumLower rewrites them to variant literals BEFORE
  --     ArrayLower, so the fold banks as scalars — `.tag` never reaches the
  --     shape check as an element.
  let flagDefs : Array Lean.Json := #[Lean.Json.mkObj [
    ("kind", Lean.Json.str "sum"), ("name", Lean.Json.str "Flag"),
    ("variants", Lean.Json.arr #[
      Lean.Json.mkObj [("name", Lean.Json.str "A"), ("payload", Lean.Json.arr #[])],
      Lean.Json.mkObj [("name", Lean.Json.str "B"), ("payload", Lean.Json.arr #[])],
      Lean.Json.mkObj [("name", Lean.Json.str "C"), ("payload", Lean.Json.arr #[])]])]]
  let tagJ (v : String) : Lean.Json :=
    Lean.Json.mkObj [("op", Lean.Json.str "tag"), ("variant", Lean.Json.str v)]
  let tagFold := cgFold
    (Lean.Json.arr #[tagJ "A", tagJ "C", tagJ "B"])
    (cgAdd (cgBinding "acc") (cgMulHalf (cgBinding "e")))
  let tagUnroll :=   -- variant indices 0, 2, 1 in fold order
    cgAdd (cgAdd (cgAdd (cgJn 0 0) (cgMulHalf (cgJn 0 0)))
      (cgMulHalf (cgJn 2 0))) (cgMulHalf (cgJn 1 0))
  let mut fails : Array String := #[]
  if let some f ← check "ragged" raggedFold raggedUnroll #[] 0 then fails := fails.push f
  if let some f ← check "dyn-index" dynFold dynUnroll #[] 0 then fails := fails.push f
  if let some f ← check "tags" tagFold tagUnroll flagDefs (if looping then 1 else 0) then
    fails := fails.push f
  if fails.isEmpty then
    let tagWord := if looping then "banks as SCALARS (1 region)" else "unrolls with the flag off (0 regions)"
    passGate "banks-columnize-bail" s!"ragged + dynamic-index unroll byte-equal (0 regions); tag fold reaches ArrayLower as variant literals post-SumLower and {tagWord}"
  else
    IO.println s!"  FAIL  banks-columnize-bail  {String.intercalate " · " fails.toList}"
    pure false
