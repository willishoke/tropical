import Tropical.Tropicaltest.Slide

/-!
# Tropical.Tropicaltest.Modal

The modal island: nested banks, the decaying-resonator bank (arrow ≡ straight-line), reverse reading, the residue calculus (moments, reverb, degeneracy, symbolic, collected), direction/sway, and the patched/live modal graphs.
-/

open Tropical
open Tropical.Plan
open Tropical.Ir (Arena ProgramIdx)

-- ── Nested banks (WS5) ───────────────────────────────────────────────────────

/-- The linear delimiter sequence of a plan's reduce regions, in emit order
    (`collectBlocks` is the single source of emit-order truth). Two regions
    properly NESTED spell `#[RB, RB, RE, RE]`; sequential ones `#[RB, RE, RB,
    RE]`. -/
private def reduceDelims (p : Tropical.Plan.FlatPlan) : Array String := Id.run do
  let mut out : Array String := #[]
  for f in p.instanceFunctions do
    for block in Tropical.Ir.Stage0.collectBlocks f do
      for i in block do
        if i.tag == "ReduceBegin" || i.tag == "ReduceEnd" then
          out := out.push i.tag
  return out

open Tropical.EmitArrow in
/-- The nested-banks probe: Σᵢ aᵢ·Σⱼ(bⱼ + aᵢ), the Cauchy shape, authored
    DIRECTLY as nested `Sig.bankSum` regions (unique binder ids 0/1 along
    the nesting chain — the JSON fold-of-folds spelling left with the fold
    lowering). The inner body reads the OUTER element: `index(col_a,
    loopIdx 0)` appears BOTH inside the inner region (in the contribution)
    and outside it (the aᵢ· factor) as ONE hash-consed DAG node — exactly
    what makes unique binder ids load-bearing (de Bruijn spellings would
    fork it). -/
private def nestedBankProbe (k1 k2 : Nat) : Sig :=
  let aL (i : Nat) : Sig := lit (Int.ofNat (31 + 7 * i)) 2   -- aᵢ = 0.31 + 0.07·i
  let bL (j : Nat) : Sig := lit (Int.ofNat (11 + 5 * j)) 2   -- bⱼ = 0.11 + 0.05·j
  let colA := Sig.arr ((Array.range k1).map aL)
  let colB := Sig.arr ((Array.range k2).map bL)
  let aElem := Sig.index colA (Sig.loopIdx 0)
  let bElem := Sig.index colB (Sig.loopIdx 1)
  let inner := Sig.bankSum k2 #[colB] (add bElem aElem) none 1
  Sig.bankSum k1 #[colA] (mul aElem inner) none 0

open Tropical.EmitArrow in
/-- The probe's hand-unrolled reference, in the reduce loop's own visit
    order: ((0 + a₀·S₀) + a₁·S₁) + … with Sᵢ = ((0 + (b₀+aᵢ)) + (b₁+aᵢ)) + …. -/
private def nestedBankUnrolled (k1 k2 : Nat) : Sig :=
  let aL (i : Nat) : Sig := lit (Int.ofNat (31 + 7 * i)) 2
  let bL (j : Nat) : Sig := lit (Int.ofNat (11 + 5 * j)) 2
  (Array.range k1).foldl (fun acc i =>
    let s := (Array.range k2).foldl (fun a j => add a (add (bL j) (aL i))) (lit 0)
    add acc (mul (aL i) s)) (lit 0)

open Tropical.EmitArrow in
/-- THE NESTED-BANKS gate (WS5): the nested-`bankSum` probe must emit exactly
    2 reduce regions, properly NESTED in the stream (RB RB RE RE); render
    byte-equal to the hand-unrolled reference over 2048 samples; stay FLAT in
    BOTH trip counts ((4,4) → (16,16), Δ ≤ 2 — HARD); and the TYPED Stage0
    split (the production golden render path — depth-counted `findRegionEnd`,
    outermost-only `tryRegion`) must traverse the nested delimiters and still
    render byte-equal. (`TROPICAL_BANKS_UNROLL` does not apply: the flag
    governs the modal BUILDER's lowering choice; a hand-authored `bankSum`
    is always a region.) -/
def runBanksNested (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let fPlan := buildAndFinish (.ok (buildExprCarrier "nested_f4" (nestedBankProbe 4 4) arena))
  let uPlan := buildAndFinish (.ok (buildExprCarrier "nested_u4" (nestedBankUnrolled 4 4) arena))
  match fPlan, uPlan with
  | .ok fp, .ok up =>
    match ← renderPlanSamples fp 2048, ← renderPlanSamples up 2048 with
    | .ok fS, .ok uS =>
      let n := min fS.size uS.size
      let bitDiff := bitDiffCount fS uS
      let nonzero := fS.any (· != 0.0)
      match buildAndFinish (.ok (buildExprCarrier "nested_f16" (nestedBankProbe 16 16) arena)) with
      | .ok f16 =>
        let d := planInstrCount f16 - planInstrCount fp
        let regions := planTagCount "ReduceBegin" fp
        let delims := reduceDelims fp
        let nested := delims == #["ReduceBegin", "ReduceBegin", "ReduceEnd", "ReduceEnd"]
        IO.println s!"        bank-of-banks Σᵢ aᵢ·Σⱼ(bⱼ+aᵢ) (K=4,4), inner body reads the OUTER element:"
        IO.println s!"        result   bit-differing {bitDiff}/{n} vs hand-unrolled · nonzero={nonzero} · regions={regions} · delims={delims}"
        IO.println s!"        payoff   plan-instrs: nested(4,4)={planInstrCount fp} unrolled(4,4)={planInstrCount up} nested(16,16)={planInstrCount f16} (Δ={d})"
        warnBenchConst "banks-nested" "nested-bank plan-instrs (any K)" 14 (planInstrCount fp)
        -- The TYPED Stage0 split must traverse the nested delimiters and
        -- still render byte-equal.
        let stagedOk ← do
          match buildAndFinishStaged (.ok (buildExprCarrier "nested_f4s" (nestedBankProbe 4 4) arena)),
                buildAndFinishStaged (.ok (buildExprCarrier "nested_u4s" (nestedBankUnrolled 4 4) arena)) with
          | .ok (pf, bf), .ok (pu, bu) =>
            let sf ← renderTypedBytes pf bf
            let su ← renderTypedBytes pu bu
            pure (sf == su)
          | .error e, _ | _, .error e =>
            IO.println s!"        staged   compile failed: {firstLine e}"; pure false
        IO.println s!"        staged   typed-split render byte-equal to unroll: {stagedOk}"
        if bitDiff == 0 && nonzero && regions == 2 && nested && d ≤ 2 && stagedOk then
          passGate "banks-nested" s!"bank-of-banks emits NESTED regions (RB RB RE RE), byte-equal to unroll (plain + typed split), plan FLAT in both counts (Δ={d} ≤ 2, (4,4)→(16,16))"
        else
          failGate "banks-nested" s!"bitDiff={bitDiff} nonzero={nonzero} regions={regions} nested={nested} Δ={d} stagedOk={stagedOk}"
      | .error e =>
        failGate "banks-nested" s!"scaling build: {firstLine e}"
    | .error e, _ | _, .error e => failGate "banks-nested" s!"render: {firstLine e}"
  | .error e, _ | _, .error e => failGate "banks-nested" s!"build: {firstLine e}"

open Tropical.EmitArrow in
/-- THE NESTED-BANKS MSL gate: EmitMsl on the nested plan emits two reduce
    `for` loops, the second opening strictly INSIDE the first (text-level
    depth scan: reduce-for lines push, brace-only lines pop; the probe's body
    is scalar-only, so no other construct emits a bare closing brace before
    the loops close). -/
def runBanksNestedMsl (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  match buildAndFinish (.ok (buildExprCarrier "nested_msl" (nestedBankProbe 4 4) arena)) with
  | .error e => failGate "banks-nested-msl" s!"build: {firstLine e}"
  | .ok fp =>
    match Tropical.Ir.EmitMsl.emitKernel fp with
    | .error e => failGate "banks-nested-msl" s!"EmitMsl: {firstLine e}"
    | .ok msl =>
      -- Depth scan over the kernel text: a reduce-for line opens a loop, a
      -- brace-only line closes one. Record the open depth at each reduce-for.
      let mut depth : Nat := 0
      let mut forDepths : Array Nat := #[]
      for l in msl.splitOn "\n" do
        let t := l.trimAscii
        if t.startsWith "for (long rd" then
          forDepths := forDepths.push depth
          depth := depth + 1
        else if t == "}" && depth > 0 then
          depth := depth - 1
      if forDepths == #[0, 1] then
        passGate "banks-nested-msl" s!"two reduce for-loops, the inner strictly inside the outer (depths {forDepths})"
      else
        failGate "banks-nested-msl" s!"expected reduce-for depths #[0, 1], got {forDepths}"

/-- THE MODAL DEGREE gate. A degree-1 mode `amp·d·e^{−σd}` (a repeated pole — the
    resonance "swell") rendered by the engine must match `sinkGain·d·e^{−σd}` to
    minimax tolerance (an absolute oracle, validating the new `d^deg` factor), and
    must RISE to a peak at d≈1/σ before decaying — the τ·e signature a simple pole
    (monotone decay) cannot produce. -/
def runModalDegree (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let modes : Array Tropical.EmitArrow.ModalMode := #[
    { sigma := Tropical.EmitArrow.lit 25, omega := Tropical.EmitArrow.lit 0,
      cre := Tropical.EmitArrow.lit 1, deg := 1 }]
  let anchor := Tropical.EmitArrow.lit 200
  match buildAndFinish (.ok (Tropical.EmitArrow.buildModalBankArrow "modal_deg" modes anchor arena)) with
  | .ok p =>
    match ← renderPlanSamples p 8192 with
    | .ok s =>
      let sinkGain : Float := Tropical.Plan.defaultSinkGain.toFloat
      let n := min s.size 8192
      let mut preMax : Float := 0.0
      for i in [0:201] do
        let a := s[i]!.abs
        if a > preMax then preMax := a
      let mut maxRel : Float := 0.0
      let mut peakVal : Float := 0.0
      let mut peakI : Nat := 0
      for i in [201:n] do
        let d := (i.toFloat - 200.0) / 44100.0
        let ref := sinkGain * d * Float.exp (-25.0 * d)
        if ref.abs > 1e-5 then
          let rel := (s[i]! - ref).abs / ref.abs
          if rel > maxRel then maxRel := rel
        let a := s[i]!.abs
        if a > peakVal then
          peakVal := a
          peakI := i
      let peakD := (peakI.toFloat - 200.0) / 44100.0
      IO.println s!"        deg-1 τ·e mode (σ=25, f=0) vs sinkGain·d·e^(−25d):"
      IO.println s!"        result   preMax={preMax} · max rel err={maxRel} · peak @ sample {peakI} (d={peakD}s, expect 1/σ=0.04)"
      if preMax == 0.0 && maxRel < 1e-4 && peakI > 1500 && peakI < 2400 then
        passGate "modal-degree" s!"τ·e swell ≡ d·e^(−σd) to {maxRel}; rises to peak at d≈1/σ then decays"
      else
        failGate "modal-degree" s!"preMax={preMax} maxRel={maxRel} peakI={peakI}"
    | .error e => failGate "modal-degree" s!"render: {firstLine e}"
  | .error e => failGate "modal-degree" s!"build: {firstLine e}"

/-- THE LONG-τ gate. Time-translation exactness at astronomical clock offsets:
    the SAME bank struck K samples later, read K samples later, must be BYTE-
    IDENTICAL to the bank at the origin (K = 2³⁰ samples ≈ 6.8 hours at 44.1k).
    Both clocks are FRACTIONAL-sample — the production scrub form
    `M(n) = toInt(velocity·2³²)·n` and a sub-sample offset — because that is
    where the old float path actually rounded: at whole samples
    `toFloat((2³⁰+s)·2³²)` has ~31 significant bits and was accidentally exact,
    but a fractional clock plus the 2³⁰-sample shift needs >53 bits, so
    `toFloat` rounds and the unreduced `ω·dSec` walks off the phase — precision
    decayed with τ exactly when the clock was warped. On the integer relative
    clock (`relClockQ` + `modePhaseW`) `clkRel` is the same i64 on both sides
    at ANY low bits, so every downstream op sees identical bytes. Energy floors
    keep silent agreement from passing. -/
def runLongTauModal (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let modes : Array Tropical.EmitArrow.ModalMode := #[
    Tropical.EmitArrow.ModalMode.hz (Tropical.EmitArrow.lit 220) (Tropical.EmitArrow.lit 30 1) (Tropical.EmitArrow.lit 6 1),
    Tropical.EmitArrow.ModalMode.hz (Tropical.EmitArrow.lit 330) (Tropical.EmitArrow.lit 40 1) (Tropical.EmitArrow.lit 4 1),
    Tropical.EmitArrow.ModalMode.hz (Tropical.EmitArrow.lit 440) (Tropical.EmitArrow.lit 55 1) (Tropical.EmitArrow.lit 3 1)]
  let K : Int := 1073741824                    -- 2³⁰ samples
  let Kq : Int := K * 4294967296               -- the same shift as a Q32.32 clock offset
  let mkPair (tag : String) (clk : Tropical.EmitArrow.Clock) :
      Except String Tropical.Plan.FlatPlan × Except String Tropical.Plan.FlatPlan :=
    (buildAndFinish (.ok (Tropical.EmitArrow.buildExprCarrier s!"modal_lt_{tag}_base"
        (Tropical.EmitArrow.modalBankSig modes clk (Tropical.EmitArrow.lit 200)) arena)),
     buildAndFinish (.ok (Tropical.EmitArrow.buildExprCarrier s!"modal_lt_{tag}_far"
        (Tropical.EmitArrow.modalBankSig modes
          (Tropical.EmitArrow.add clk (Tropical.EmitArrow.litI Kq))
          (Tropical.EmitArrow.lit (K + 200)) ) arena)))
  let check (tag : String) (pair : Except String Tropical.Plan.FlatPlan × Except String Tropical.Plan.FlatPlan) :
      IO (Option (Nat × Float)) := do
    match pair with
    | (.ok bp, .ok fp) =>
      match ← renderPlanSamples bp 1024, ← renderPlanSamples fp 1024 with
      | .ok base, .ok far =>
        let n := min base.size far.size
        let bitDiff := bitDiffCount base far
        let mut energy : Float := 0.0
        for i in [200:n] do energy := energy + base[i]! * base[i]!
        pure (some (bitDiff, energy))
      | .error e, _ | _, .error e =>
        IO.println s!"  FAIL  modal-longtau  render ({tag}): {firstLine e}"; pure none
    | (.error e, _) | (_, .error e) =>
      IO.println s!"  FAIL  modal-longtau  build ({tag}): {firstLine e}"; pure none
  -- the master-clock scrub form: M(n) = toInt(velocity·2³²)·n at velocity 1.001
  -- (toInt(1.001·2³²) = 4299262263) — every sample has populated low bits. The
  -- literals ride `litI`: a bare `lit` would float-promote the clock arithmetic
  -- and round the very bits this gate exists to protect.
  let velClk := Tropical.EmitArrow.mul
    (Tropical.EmitArrow.rshift Tropical.EmitArrow.clockLit (Tropical.EmitArrow.lit 32))
    (Tropical.EmitArrow.litI 4299262263)
  -- a bare sub-sample offset: one 2⁻³² unit off the whole-sample grid.
  let subClk := Tropical.EmitArrow.add Tropical.EmitArrow.clockLit (Tropical.EmitArrow.litI 1)
  match ← check "vel" (mkPair "vel" velClk), ← check "sub" (mkPair "sub" subClk) with
  | some (d1, e1), some (d2, e2) =>
    IO.println s!"        bank @ origin vs struck+read 2³⁰ samples later (≈6.8h), fractional clocks, 1024 samples:"
    IO.println s!"        result   velocity-1.001 clock: bit-differing {d1}/1024 (E={e1})  ·  sub-sample offset: {d2}/1024 (E={e2})"
    if d1 == 0 && d2 == 0 && e1 > 1e-6 && e2 > 1e-6 then
      passGate "modal-longtau" "time-translation byte-exact at τ+2³⁰ samples on fractional (scrub-form) clocks"
    else
      failGate "modal-longtau" s!"bitDiff vel={d1} sub={d2} (want 0) energy vel={e1} sub={e2} (>1e-6)"
  | _, _ => pure false

/-- THE REVERSE-REVERB gate (the moat). The modal bank read through a reversing
    warp φ(c) = 2·C·2³² − c (reflect scene time around sample C=1024) must equal
    the FORWARD bank time-mirrored: rev[i] ≡ fwd[2C−i], bit-for-bit. This is
    zero-latency reverse reverb — a stateless closed form addressed at negative
    velocity, impossible on a streaming delay line. The warp threads through the
    modal `arrUn … (.clk c)` via the same `.warp` a master-clock scrub uses. -/
def runModalReverse (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let modes : Array Tropical.EmitArrow.ModalMode := #[
    Tropical.EmitArrow.ModalMode.hz (Tropical.EmitArrow.lit 220) (Tropical.EmitArrow.lit 30 1) (Tropical.EmitArrow.lit 6 1),
    Tropical.EmitArrow.ModalMode.hz (Tropical.EmitArrow.lit 330) (Tropical.EmitArrow.lit 40 1) (Tropical.EmitArrow.lit 4 1)]
  let anchor := Tropical.EmitArrow.lit 200
  let twoC : Int := 2048 * 4294967296          -- reflect around sample C = 1024
  let revφ : Tropical.EmitArrow.Clock → Tropical.EmitArrow.Clock :=
    fun c => Tropical.EmitArrow.sub (Tropical.EmitArrow.lit twoC) c
  match buildAndFinish (.ok (Tropical.EmitArrow.buildModalBankArrow "modal_fwd" modes anchor arena)),
        buildAndFinish (.ok (Tropical.EmitArrow.buildModalBankWarped "modal_rev" modes anchor revφ arena)) with
  | .ok fp, .ok rp =>
    match ← renderPlanSamples fp 2048, ← renderPlanSamples rp 2048 with
    | .ok fwd, .ok rev =>
      let n := min fwd.size rev.size
      let mut bitDiff := 0
      let mut differsFwd := 0        -- rev ≠ fwd somewhere (warp is non-trivial)
      let mut revEnergy : Float := 0.0
      for i in [1:n] do
        if rev[i]! != fwd[2048 - i]! then bitDiff := bitDiff + 1
        if rev[i]! != fwd[i]! then differsFwd := differsFwd + 1
        revEnergy := revEnergy + rev[i]! * rev[i]!
      IO.println s!"        modal bank forward vs reversed (φ reflects scene time around sample 1024):"
      IO.println s!"        result   rev[i] vs fwd[2048−i]: bit-differing {bitDiff}/{n}  ·  rev≠fwd at {differsFwd} samples  ·  rev energy={revEnergy}"
      if bitDiff == 0 && differsFwd > 0 && revEnergy > 1e-6 then
        passGate "modal-reverse" s!"reversed reading ≡ forward time-mirrored, bit-exact — zero-latency reverse reverb ({n} samples)"
      else
        failGate "modal-reverse" s!"bitDiff={bitDiff} differsFwd={differsFwd} revEnergy={revEnergy}"
    | .error e, _ | _, .error e => failGate "modal-reverse" s!"render: {firstLine e}"
  | .error e, _ | _, .error e => failGate "modal-reverse" s!"build: {firstLine e}"

section ResidueGates
open Tropical.EmitArrow

/-- THE RESIDUE CALCULUS gate (exact, build-time). `voice ⋙ reverb` composed by
    `residueCompose` must reproduce the convolution's Taylor jet at t=0: moment
    `Σ Aᵢμᵢᵏ` equals `y⁽ᵏ⁾(0)` for k=0..6, and the 0th moment `Σ A = 0` (a wrong
    sign, denominator, or a missing ringing term breaks one). `Σ A = 0` also means
    the composed tail starts continuously — the reverb has no onset click for free.
    Pure complex ±×÷; the emit path is checked separately by `modal-reverb`. -/
def runResidueMoments (_arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let tp := 6.283185307179586
  let voice : Array (Cplx × Cplx) := #[
    (⟨-2.0, tp * 220.0⟩, ⟨1.0, 0.0⟩),
    (⟨-2.5, tp * 330.0⟩, ⟨0.6, 0.0⟩)]
  let reverb : Array (Cplx × Cplx) := #[
    (⟨-3.0, tp * 180.0⟩, ⟨0.7, 0.2⟩),
    (⟨-4.0, tp * 260.0⟩, ⟨-0.5, 0.4⟩),
    (⟨-5.0, tp * 350.0⟩, ⟨0.3, -0.6⟩),
    (⟨-6.0, tp * 500.0⟩, ⟨0.4, 0.1⟩)]
  let modes := residueCompose voice reverb
  let err := residueMomentError voice reverb 6
  let sumA := modes.foldl (fun s m => s.add m.amp) (⟨0.0, 0.0⟩ : Cplx)
  let sumAbsA := modes.foldl (fun s m => s + m.amp.abs) 0.0
  let onset := sumA.abs / (sumAbsA + 1e-300)
  IO.println s!"        voice(2 poles) ⋙ reverb(4 poles) → {modes.size} residue modes; jet-match k=0..6:"
  IO.println s!"        result   max relative moment error = {err}  ·  onset ΣA/Σ|A| = {onset}"
  if err < 1e-9 && onset < 1e-9 then
    passGate "residue-moments" s!"composed modes reproduce the convolution jet to k=6 (err={err}); ΣA=0 ⇒ click-free onset"
  else
    failGate "residue-moments" s!"err={err} (want <1e-9) onset={onset} (want <1e-9)"

/-- THE RESIDUE REVERB gate (emit). `buildModalReverb` runs the residue calculus
    and emits the composed bank; it must render a real, causal, DECAYING signal
    that starts CONTINUOUSLY at the strike — the `Σ A = 0` property means the first
    post-strike sample is ≈0 and grows (no onset click), unlike an authored bank
    whose partials all start at full amplitude. -/
def runModalReverb (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let tp := 6.283185307179586
  let voice : Array (Cplx × Cplx) := #[
    (⟨-2.0, tp * 220.0⟩, ⟨1.0, 0.0⟩),
    (⟨-2.5, tp * 330.0⟩, ⟨0.6, 0.0⟩)]
  let reverb : Array (Cplx × Cplx) := #[
    (⟨-3.0, tp * 180.0⟩, ⟨0.7, 0.2⟩),
    (⟨-4.0, tp * 260.0⟩, ⟨-0.5, 0.4⟩),
    (⟨-5.0, tp * 350.0⟩, ⟨0.3, -0.6⟩),
    (⟨-6.0, tp * 500.0⟩, ⟨0.4, 0.1⟩)]
  let anchor := lit 200
  -- render ~370 ms: the composed tail ramps up (click-free onset) over the first
  -- tens of ms, then decays over its RT — so compare energy AFTER the onset peak.
  match buildAndFinish (.ok (buildModalReverb "modal_reverb" voice reverb anchor arena)) with
  | .ok p =>
    match ← renderPlanSamples p 16384 with
    | .ok s =>
      let n := min s.size 16384
      let mut preMax : Float := 0.0
      for i in [0:201] do
        let a := s[i]!.abs
        if a > preMax then preMax := a
      let firstPost := s[201]!.abs
      let mut peak : Float := 0.0
      for i in [201:n] do
        let a := s[i]!.abs
        if a > peak then peak := a
      let mut eMid : Float := 0.0
      let mut eLate : Float := 0.0
      for i in [2048:6144] do eMid := eMid + s[i]! * s[i]!
      for i in [12288:16384] do eLate := eLate + s[i]! * s[i]!
      IO.println s!"        buildModalReverb rendered (voice ⋙ reverb, struck @ sample 200):"
      IO.println s!"        result   pre-strike |max|={preMax} · first-post |s|={firstPost} · peak={peak} · E[mid]={eMid} E[late]={eLate}"
      if preMax == 0.0 && peak > 1e-4 && firstPost < 0.02 * peak && eLate < eMid then
        passGate "modal-reverb" s!"residue-composed bank renders: causal, click-free onset (|first|≪peak), decaying tail ({n} samples)"
      else
        failGate "modal-reverb" s!"preMax={preMax} peak={peak} firstPost={firstPost} eMid={eMid} eLate={eLate}"
    | .error e => failGate "modal-reverb" s!"render: {firstLine e}"
  | .error e => failGate "modal-reverb" s!"build: {firstLine e}"

/-- THE DIRECTION gate. `dir` crossfades the tail's time-direction and must, above
    all, STAY AUDIBLE across its range (the pole-rotation version silently collapsed
    the interior because ω≫σ threw the frequency into the damping). (A) `dir=0`
    reduces bit-for-bit to the forward bank; (B) `dir=1` is that bank TIME-MIRRORED
    (rev[i] ≡ fwd[2C−i]) — genuine reverse reverb, no warp; (C) `dir=0.5` stays finite
    AND carries real energy (a substantial fraction of the forward bank's), i.e. it is
    audible, not a collapsed transient — the property the rotation version lacked. -/
def runModalDirection (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let modes : Array ModalMode := #[
    ModalMode.hz (lit 220) (lit 30 1) (lit 6 1),
    ModalMode.hz (lit 330) (lit 40 1) (lit 4 1)]
  let anchor := lit 1024                        -- mid of 2048 ⇒ 2C = 2048
  let fwdB := buildModalBankArrow "dir_fwd" modes anchor arena
  let idB  := buildModalBankDir "dir_id"  modes anchor (lit 0) arena        -- forward
  let revB := buildModalBankDir "dir_rev" modes anchor (lit 1) arena        -- reverse
  let midB := buildModalBankDir "dir_mid" modes anchor (litF 0.5) arena     -- crossfade
  match buildAndFinish (.ok fwdB), buildAndFinish (.ok idB),
        buildAndFinish (.ok revB), buildAndFinish (.ok midB) with
  | .ok fp, .ok ip, .ok rp, .ok mp =>
    match ← renderPlanSamples fp 2048, ← renderPlanSamples ip 2048,
          ← renderPlanSamples rp 2048, ← renderPlanSamples mp 2048 with
    | .ok fwd, .ok idv, .ok rev, .ok mid =>
      let n := 2048
      let mut idDiff := 0
      let mut fwdE : Float := 0.0
      for i in [0:n] do
        if idv[i]! != fwd[i]! then idDiff := idDiff + 1
        fwdE := fwdE + fwd[i]! * fwd[i]!
      let mut revDiff := 0
      let mut revDiffersFwd := 0
      for i in [1:n] do
        if rev[i]! != fwd[2048 - i]! then revDiff := revDiff + 1
        if rev[i]! != fwd[i]! then revDiffersFwd := revDiffersFwd + 1
      let mut midE : Float := 0.0
      let mut midFinite := true
      for i in [0:n] do
        let a := mid[i]!.abs
        if !a.isFinite then midFinite := false
        midE := midE + a * a
      IO.println s!"        direction crossfade (forward↔reverse, σ/ω fixed):"
      IO.println s!"        (A) dir=0 vs fwd bitDiff={idDiff}  ·  (B) dir=1 mirror bitDiff={revDiff} (differs-fwd @{revDiffersFwd})"
      IO.println s!"        (C) dir=0.5 finite={midFinite} · E={midE} vs forward E={fwdE} (AUDIBLE ⇒ E ≫ 0)"
      let aOk := idDiff == 0
      let bOk := revDiff == 0 && revDiffersFwd > 0
      let cOk := midFinite && fwdE > 1e-6 && midE > 0.1 * fwdE
      if aOk && bOk && cOk then
        passGate "modal-direction" s!"dir=0 forward (bit-exact) · dir=1 reverse (mirror bit-exact) · dir=0.5 AUDIBLE (E={midE}, {midE/fwdE} of fwd)"
      else
        failGate "modal-direction" s!"A={aOk} B={bOk} C={cOk} (idDiff={idDiff} revDiff={revDiff} midE={midE} fwdE={fwdE})"
    | _, _, _, _ => failGate "modal-direction" "render error"
  | _, _, _, _ => failGate "modal-direction" "build error"

/-- THE SWAY gate. Decay sway modulates each mode's damping by `1 + depth·sin(2π·
    rate·t)` on the ENVELOPE clock only. (S1) at depth 0 it is bit-for-bit the
    un-swayed bank (the LFO term folds to ×1); (S2) at depth>0 the tail differs (its
    decay breathes) yet stays causal (silent pre-strike) and bounded. Pitch is
    untouched by construction (the oscillator reads the plain `dSec`); the LFO rides
    the same clock leaf as the bank, so a master scrub reverses it coherently. -/
def runModalSway (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let modes : Array ModalMode := #[
    ModalMode.hz (lit 220) (lit 30 1) (lit 6 1),
    ModalMode.hz (lit 330) (lit 40 1) (lit 4 1)]
  let anchor := lit 200
  let noSway := buildModalBankDir "sway_no" modes anchor (lit 0) arena
  let sway0  := buildModalBankDir "sway_0"  modes anchor (lit 0) arena (some (lit 0, lit 3 1))
  let swayD  := buildModalBankDir "sway_d"  modes anchor (lit 0) arena (some (lit 5 1, lit 20 1))
  match buildAndFinish (.ok noSway), buildAndFinish (.ok sway0), buildAndFinish (.ok swayD) with
  | .ok np, .ok zp, .ok dp =>
    match ← renderPlanSamples np 2048, ← renderPlanSamples zp 2048, ← renderPlanSamples dp 2048 with
    | .ok nos, .ok zos, .ok dos =>
      let n := 2048
      let mut z0Diff := 0
      for i in [0:n] do if zos[i]! != nos[i]! then z0Diff := z0Diff + 1
      let mut modDiff := 0
      let mut preMax : Float := 0.0
      let mut dFinite := true
      let mut dPeak : Float := 0.0
      for i in [0:n] do
        if dos[i]! != nos[i]! then modDiff := modDiff + 1
        let a := dos[i]!.abs
        if !a.isFinite then dFinite := false
        if a > dPeak then dPeak := a
      for i in [0:201] do
        let a := dos[i]!.abs
        if a > preMax then preMax := a
      IO.println s!"        decay sway (σ·(1+depth·sin 2πrt) on the envelope clock only):"
      IO.println s!"        (S1) depth 0 vs no-sway bitDiff={z0Diff}  ·  (S2) depth>0 differs @{modDiff}/{n}, pre-strike |max|={preMax}, peak={dPeak}, finite={dFinite}"
      let s1 := z0Diff == 0
      let s2 := modDiff > 100 && preMax == 0.0 && dFinite && dPeak > 1e-4 && dPeak < 1e3
      if s1 && s2 then
        passGate "modal-sway" "depth 0 ≡ un-swayed (bit-exact) · depth>0 breathes the decay, causal & bounded"
      else
        failGate "modal-sway" s!"S1={s1} (bitDiff {z0Diff}) S2={s2} (modDiff {modDiff} preMax {preMax} peak {dPeak} finite {dFinite})"
    | _, _, _ => failGate "modal-sway" "render error"
  | _, _, _ => failGate "modal-sway" "build error"

/-- THE DEGENERATE RESIDUE gate. A voice pole placed EXACTLY on a reverb pole
    (sympathetic resonance) must compose to a `τ·e^{μd}` DOUBLE POLE, not blow up.
    residueCompose must emit exactly one deg-1 mode, and — crucially — the
    degree-aware moments must STILL reproduce the convolution jet (the double pole
    contributes `A·k·μ^{k−1}`), so the exact-coincidence limit is handled, not
    dodged. -/
def runResidueDegenerate (_arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let tp := 6.283185307179586
  let voice : Array (Cplx × Cplx) := #[
    (⟨-3.0, tp * 260.0⟩, ⟨1.0, 0.0⟩)]        -- λ sits exactly on reverb pole #2
  let reverb : Array (Cplx × Cplx) := #[
    (⟨-3.0, tp * 180.0⟩, ⟨0.7, 0.2⟩),
    (⟨-3.0, tp * 260.0⟩, ⟨-0.5, 0.4⟩),        -- ν = λ (coincident)
    (⟨-5.0, tp * 350.0⟩, ⟨0.3, -0.6⟩)]
  let modes := residueCompose voice reverb
  let nDeg1 := modes.foldl (fun c m => if m.deg == 1 then c + 1 else c) 0
  let err := residueMomentError voice reverb 6
  IO.println s!"        voice pole = reverb pole #2 (sympathetic): {modes.size} modes, {nDeg1} of degree 1:"
  IO.println s!"        result   deg-1 modes = {nDeg1}  ·  degree-aware moment error k=0..6 = {err}"
  if nDeg1 == 1 && err < 1e-9 then
    passGate "residue-degenerate" s!"coincident pole → one τ·e double pole; jet still exact (err={err}) — no blow-up"
  else
    failGate "residue-degenerate" s!"nDeg1={nDeg1} (want 1) err={err} (want <1e-9)"

/-- THE SYMBOLIC RESIDUE gate. The residue calculus emitted as `Expr` couplings
    (`residueComposeE`, so poles/coeffs can be live slots) must, on LITERAL poles,
    fold to the same bank as the validated Float `residueCompose`. Same voice ⋙
    reverb built both ways renders equal (differing only by litF input-vs-output
    rounding). This is what makes modal params live without changing the math. -/
def runResidueSymbolic (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let tp := 6.283185307179586
  let voiceF : Array (Cplx × Cplx) := #[
    (⟨-2.0, tp * 220.0⟩, ⟨1.0, 0.0⟩),
    (⟨-2.5, tp * 330.0⟩, ⟨0.6, 0.0⟩)]
  let reverbF : Array (Cplx × Cplx) := #[
    (⟨-3.0, tp * 180.0⟩, ⟨0.7, 0.2⟩),
    (⟨-4.0, tp * 260.0⟩, ⟨-0.5, 0.4⟩),
    (⟨-5.0, tp * 350.0⟩, ⟨0.3, -0.6⟩),
    (⟨-6.0, tp * 500.0⟩, ⟨0.4, 0.1⟩)]
  let toMode := fun (pa : Cplx × Cplx) =>
    ({ sigma := litF (-pa.1.re), omega := litF pa.1.im,
       cre := litF pa.2.re, cim := litF pa.2.im } : ModalMode)
  let anchor := lit 200
  match buildAndFinish (.ok (buildModalReverb "rv_baked" voiceF reverbF anchor arena)),
        buildAndFinish (.ok (buildModalReverbSym "rv_sym" (voiceF.map toMode) (reverbF.map toMode) anchor arena)) with
  | .ok bp, .ok sp =>
    match ← renderPlanSamples bp 4096, ← renderPlanSamples sp 4096 with
    | .ok bs, .ok ss =>
      let n := min bs.size ss.size
      let mut maxAbs : Float := 0.0
      let mut energy : Float := 0.0
      for i in [0:n] do
        let d := (bs[i]! - ss[i]!).abs
        if d > maxAbs then maxAbs := d
        energy := energy + bs[i]! * bs[i]!
      -- Two builds of the same 10-mode bank whose weights may differ by an ulp
      -- pre-landing (litF round-trip), so each mode may jump one Q4.28 quantum
      -- (design/fixed-carrier.md) × the sink gain, 2× slack.
      let bound := 10.0 * 3.7252903e-9 * Tropical.Plan.defaultSinkGain.toFloat * 2.0
      IO.println s!"        Expr-residue (literal poles) vs Float-baked residue, voice(2)⋙reverb(4):"
      IO.println s!"        result   max|Δ|={maxAbs * 1e9}e-9  ·  quantum bound={bound * 1e9}e-9"
      if maxAbs < bound && energy > 1e-9 then
        passGate "symbolic-residue" s!"Expr couplings fold to the validated Float residue within the Q4.28 landing quantum (max|Δ| {maxAbs * 1e9}e-9) — live-capable, same math"
      else
        failGate "symbolic-residue" s!"max|Δ|={maxAbs * 1e9}e-9 (bound {bound * 1e9}e-9) energy={energy}"
    | .error e, _ | _, .error e => failGate "symbolic-residue" s!"render: {firstLine e}"
  | .error e, _ | _, .error e => failGate "symbolic-residue" s!"build: {firstLine e}"

/-- THE COLLECTED RESIDUE gate. `residueComposeEC` (m+n modes: pole union with
    cross-weighted residues) must render pointwise-equal to the uncollected
    `residueComposeE` (m+m·n modes) — they are the same partial-fraction expansion
    with the per-pair ringing amps summed per reverb pole, so equality is algebraic
    and the tolerance only absorbs the DOCUMENTED datapath quantization: each mode
    lands its envelope×weight once in Q4.28 (design/fixed-carrier.md), so the two
    structures truncate independently and may differ by up to (m+n + m+m·n)
    quanta·sinkGain absolutely — the bound is quantum-tied, not relative. Also
    asserts the collection is structural: m+n modes out, not m+m·n. This is what
    makes `voice ⋙ reverb` affordable as the DEFAULT lowering — a factor m fewer
    transcendentals — which is in turn what lets a reverb keep its source's
    spectrum (and live pitch knob) instead of discarding them. -/
def runResidueCollected (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let tp := 6.283185307179586
  let toMode := fun (pa : Cplx × Cplx) =>
    ({ sigma := litF (-pa.1.re), omega := litF pa.1.im,
       cre := litF pa.2.re, cim := litF pa.2.im } : ModalMode)
  let voice : Array ModalMode := #[
    (⟨-2.0, tp * 220.0⟩, ⟨1.0, 0.0⟩),
    (⟨-2.5, tp * 330.0⟩, ⟨0.6, 0.0⟩),
    (⟨-3.5, tp * 440.0⟩, (⟨0.4, -0.2⟩ : Cplx))].map toMode
  let reverb : Array ModalMode := #[
    (⟨-3.0, tp * 180.0⟩, ⟨0.7, 0.2⟩),
    (⟨-4.0, tp * 260.0⟩, ⟨-0.5, 0.4⟩),
    (⟨-5.0, tp * 350.0⟩, ⟨0.3, -0.6⟩),
    (⟨-6.0, tp * 500.0⟩, ⟨0.4, 0.1⟩)].map toMode
  let nU := (residueComposeE voice reverb).size
  let nC := (residueComposeEC voice reverb).size
  let anchor := lit 200
  match buildAndFinish (.ok (buildModalReverbSym "rv_unc" voice reverb anchor arena)),
        buildAndFinish (.ok (buildModalReverbSymC "rv_col" voice reverb anchor arena)) with
  | .ok up, .ok cp =>
    match ← renderPlanSamples up 4096, ← renderPlanSamples cp 4096 with
    | .ok us, .ok cs =>
      let n := min us.size cs.size
      let mut maxAbs : Float := 0.0
      let mut energy : Float := 0.0
      for i in [0:n] do
        let d := (us[i]! - cs[i]!).abs
        if d > maxAbs then maxAbs := d
        energy := energy + us[i]! * us[i]!
      -- (nU + nC) independent Q4.28 weight landings × the sink gain, with
      -- 2× slack for the poly/final-shift ulps riding along.
      let bound := (nU + nC).toFloat * 3.7252903e-9 * Tropical.Plan.defaultSinkGain.toFloat * 2.0
      IO.println s!"        collected (m+n={nC}) vs uncollected (m+m·n={nU}), voice(3)⋙reverb(4):"
      IO.println s!"        result   max|Δ|={maxAbs * 1e9}e-9  ·  quantum bound={bound * 1e9}e-9"
      if maxAbs < bound && energy > 1e-9 && nC == 7 && nU == 15 then
        passGate "residue-collected" s!"pole-union bank ≡ per-pair bank within the Q4.28 landing quantum (max|Δ| {maxAbs * 1e9}e-9 < {bound * 1e9}e-9); {nU}→{nC} modes — fusion affordable as the default"
      else
        failGate "residue-collected" s!"max|Δ|={maxAbs * 1e9}e-9 (bound {bound * 1e9}e-9) energy={energy} nC={nC} (want 7) nU={nU} (want 15)"
    | .error e, _ | _, .error e => failGate "residue-collected" s!"render: {firstLine e}"
  | .error e, _ | _, .error e => failGate "residue-collected" s!"build: {firstLine e}"

/-- Evaluate a CONSTANT authoring `Sig` (the litF/±×÷/neg subtree the residue
    algebra emits) back to its `Float`, so a gate can read a production
    constructor's REAL emitted coefficients (hardening 0a-4c). Partial: `none` on
    any non-constant/unsupported node. Div mirrors the kernel's zero-guard. Used
    only in tests — never on a compile path. -/
private def evalConstSig : Tropical.EmitArrow.Sig → Option Float
  | .num n            => some n.toFloat
  | .unary .neg a     => (evalConstSig a).map (fun x => -x)
  | .unary .toFloat a => evalConstSig a
  | .binary .add a b  => do pure ((← evalConstSig a) + (← evalConstSig b))
  | .binary .sub a b  => do pure ((← evalConstSig a) - (← evalConstSig b))
  | .binary .mul a b  => do pure ((← evalConstSig a) * (← evalConstSig b))
  | .binary .div a b  => do
      let x ← evalConstSig a; let y ← evalConstSig b
      pure (if y == 0.0 then 0.0 else x / y)
  | _                 => none

/-- Read a `ModalMode`'s emitted (constant) fields back as `(pole μ = −σ+iω,
    amp A = c_re+i·c_im)` — the inverse of `cmodeToModalMode`, for checking a
    residue constructor's real output numerically. -/
private def modeConst (m : Tropical.EmitArrow.ModalMode) : Option (Cplx × Cplx) := do
  let σ ← evalConstSig m.sigma
  let ω ← evalConstSig m.omega
  let cr ← evalConstSig m.cre
  let ci ← evalConstSig m.cim
  pure (⟨-σ, ω⟩, ⟨cr, ci⟩)

/-- THE INTEGRATE gate. `integrateBank` — the antiderivative as a build-time pole
    move (`a ↦ a/μ` + a `μ=0` DC atom fixing `∫|₀=0`) — validated three ways.
    (A) the jet law is checked against `integrateBank`'s REAL emitted output (read
    back with `evalConstSig`): each integrated mode satisfies `μ·A_int=A_src` (~1e-12)
    and `Σ A_int=0` (the DC atom zeroes the onset), with `n → n+1` modes. Since μ is
    read back from the SAME emitted mode as `A_int`, this arm certifies the `cdivE`
    amp arithmetic and is invariant to a coupled pole-shift — the pole PLACEMENT is
    pinned by (B)/(C), which reference the source pole independently. (B) the SYMBOLIC `integrateBank`
    on the same literal bank folds to a Float oracle, rendering equal within the Q4.28
    landing quantum — live-capable, same math. (C) the rendered integral matches the
    cumulative TRAPEZOID of the source render (`demos/modal_vco.py` D3): a physically
    independent numerical integral over the whole tail, truncation-bounded at SR. -/
def runModalIntegrate (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let tp := 6.283185307179586
  -- one undamped (σ=0, an LFO atom) + one damped mode, so the DC atom carries a
  -- nonzero real constant (the onset-zero property is genuinely exercised).
  let srcF : Array (Cplx × Cplx) := #[
    (⟨0.0,  tp * 5.0⟩, ⟨1.0, 0.0⟩),
    (⟨-2.0, tp * 7.0⟩, ⟨0.6, 0.0⟩)]
  let toMode := fun (pa : Cplx × Cplx) =>
    ({ sigma := litF (-pa.1.re), omega := litF pa.1.im,
       cre := litF pa.2.re, cim := litF pa.2.im } : ModalMode)
  -- (A) Float oracle bank: a ↦ a/μ, plus the DC atom −Σ a/μ. A re-derivation used
  -- ONLY as the RENDER reference for arm (B) (`int_ora`, below).
  let integC : Array (Cplx × Cplx) := srcF.map (fun pa => (pa.1, pa.2.div pa.1))
  let sumA := integC.foldl (fun s pa => s.add pa.2) (⟨0.0, 0.0⟩ : Cplx)
  let oracleF : Array (Cplx × Cplx) := integC.push (⟨0.0, 0.0⟩, sumA.neg)
  -- (A′, hardening 0a-4c) the JET LAW checked against `integrateBank`'s REAL
  -- output — the production `Sig` bank read back with `evalConstSig`, NOT a
  -- re-derivation of its `a↦a/μ` formula. (The OLD arms built a local oracle with
  -- the SAME formula and checked `μ·(a/μ)=a` on THAT — a tautology that stayed
  -- green no matter what `integrateBank` did.) Now `μ·A_int` must recover the
  -- INDEPENDENT source amp `A_src`, and `Σ A_int=0` (the DC atom), read from the
  -- actually-emitted modes. Caveat: μ is read from the SAME mode as `A_int`, so this
  -- arm is invariant to a coupled pole-shift — it certifies the `cdivE` amp half;
  -- arm (B)'s `int_ora` (built from the source pole) and arm (C)'s source-render
  -- trapezoid pin the pole PLACEMENT.
  let integOut := integrateBank (srcF.map toMode)
  let integCplx := integOut.filterMap modeConst          -- (pole μ, amp A_int) per mode
  let mut jetErr : Float := 0.0
  for i in [0:srcF.size] do
    match integCplx[i]? with
    | some (μ, aInt) =>
      let recovered := μ.mul aInt                          -- μ · A_int  (must = A_src)
      let e := (recovered.add (srcF[i]!.2).neg).abs
      if e > jetErr then jetErr := e
    | none => jetErr := 1.0e9                              -- an unreadable mode ⇒ fail
  let onsetA := (integCplx.foldl (fun s pa => s.add pa.2) (⟨0.0, 0.0⟩ : Cplx)).abs
  let structOk :=
    integOut.size == srcF.size + 1
      && integCplx.size == integOut.size                   -- every emitted mode read back
      && (match integCplx[integCplx.size - 1]? with
          | some (μ, _) => μ.re == 0.0 && μ.im == 0.0      -- the DC atom sits at pole 0
          | none => false)
  let anchor := lit 0                                            -- strike at sample 0
  match buildAndFinish (.ok (buildModalBankArrow "int_src" (srcF.map toMode) anchor arena)),
        buildAndFinish (.ok (buildModalBankArrow "int_sym" (integrateBank (srcF.map toMode)) anchor arena)),
        buildAndFinish (.ok (buildModalBankArrow "int_ora" (oracleF.map toMode) anchor arena)) with
  | .ok srcP, .ok symP, .ok oraP =>
    match ← renderPlanSamples srcP 4096, ← renderPlanSamples symP 4096,
          ← renderPlanSamples oraP 4096 with
    | .ok sS, .ok iS, .ok oS =>
      let n := min (min sS.size iS.size) oS.size
      -- (B) symbolic ≡ Float-baked within the Q4.28 landing quantum
      let mut symDiff : Float := 0.0
      let mut iEnergy : Float := 0.0
      for k in [0:n] do
        let d := (iS[k]! - oS[k]!).abs
        if d > symDiff then symDiff := d
        iEnergy := iEnergy + iS[k]! * iS[k]!
      let bound := oracleF.size.toFloat * 3.7252903e-9 * Tropical.Plan.defaultSinkGain.toFloat * 4.0
      -- (C) rendered integral ≡ cumulative trapezoid of the source render (D3).
      -- The sink gain scales source and integral alike, and the trapezoid is
      -- linear, so it cancels — no gain correction needed.
      let h := 1.0 / 44100.0
      let mut acc : Float := 0.0
      let mut prev : Float := sS[0]!
      let mut trapErr : Float := 0.0
      let mut trapNrm : Float := 0.0
      for k in [1:n] do
        acc := acc + (sS[k]! + prev) * 0.5 * h
        prev := sS[k]!
        let e := iS[k]! - acc
        trapErr := trapErr + e * e
        trapNrm := trapNrm + iS[k]! * iS[k]!
      let trapRel := Float.sqrt (trapErr / (trapNrm + 1e-300))
      IO.println s!"        ∫ modal bank (a↦a/μ + DC atom), {srcF.size}→{oracleF.size} modes:"
      IO.println s!"        oracle   jet max|μ·a_int − a|={jetErr} · onset |Σa_out|={onsetA} · struct={structOk}"
      IO.println s!"        result   symbolic≡Float max|Δ|={symDiff * 1e9}e-9 (bound {bound * 1e9}e-9) · trapezoid rel-L2={trapRel}"
      if jetErr < 1e-12 && onsetA < 1e-12 && structOk && symDiff < bound
          && iEnergy > 1e-9 && trapRel < 1e-3 then
        passGate "modal-integrate" s!"antiderivative exact by the jet (μ·a_int=a, Σa=0), symbolic folds to Float within Q4.28, render ≡ cumulative trapezoid (rel {trapRel})"
      else
        failGate "modal-integrate" s!"jetErr={jetErr} onset={onsetA} struct={structOk} symDiff={symDiff*1e9}e-9 bound={bound*1e9}e-9 energy={iEnergy} trapRel={trapRel}"
    | .error e, _, _ | _, .error e, _ | _, _, .error e => failGate "modal-integrate" s!"render: {firstLine e}"
  | .error e, _, _ | _, .error e, _ | _, _, .error e => failGate "modal-integrate" s!"build: {firstLine e}"

/-- THE ANALYTIC PAIR gate. `modalBankSigPairTable` emits `(Re, Im)` of one bank
    over one column set — the substrate for heterodyne (`Re·cosθ − Im·sinθ`) and
    the divided-difference paired body. Two bit-identical oracles, no new numerics
    trusted: (Re) the pair's real part ≡ the existing `modalBankSigTable` (same op
    sequence); (Im) the pair's imaginary part ≡ the REAL bank of the amp-rotated
    modes `A ↦ −iA` (`(cre,cim) ↦ (cim,−cre)`), since `Im(A·e^{iφ}) =
    Re(−iA·e^{iφ})`. Both sides are the already-gated table path on relabelled
    coefficients, so equality is structural (bit-for-bit), not tolerance-bounded. -/
def runModalPair (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let tp := 6.283185307179586
  let toMode := fun (pa : Cplx × Cplx) =>
    ({ sigma := litF (-pa.1.re), omega := litF pa.1.im,
       cre := litF pa.2.re, cim := litF pa.2.im } : ModalMode)
  let modes : Array ModalMode := #[
    (⟨-2.0, tp * 220.0⟩, (⟨0.6, 0.2⟩ : Cplx)),
    (⟨-3.0, tp * 337.0⟩, ⟨0.4, -0.3⟩),
    (⟨-4.0, tp * 511.0⟩, ⟨0.3, 0.1⟩)].map toMode
  let rot := modes.map (fun m => { m with cre := m.cim, cim := neg m.cre })
  let anchor := lit 200
  let ((reA, reP), (imA, imP)) := buildModalBankPair "pair_re" "pair_im" modes anchor arena
  match buildAndFinish (.ok (reA, reP)), buildAndFinish (.ok (imA, imP)),
        buildAndFinish (.ok (buildModalBankTable "pair_ref_re" modes anchor arena)),
        buildAndFinish (.ok (buildModalBankTable "pair_ref_im" rot anchor arena)) with
  | .ok rePlan, .ok imPlan, .ok refRe, .ok refIm =>
    match ← renderPlanSamples rePlan 4096, ← renderPlanSamples imPlan 4096,
          ← renderPlanSamples refRe 4096, ← renderPlanSamples refIm 4096 with
    | .ok reS, .ok imS, .ok rReS, .ok rImS =>
      let reDiff := bitDiffCount reS rReS
      let imDiff := bitDiffCount imS rImS
      let reE := reS.foldl (fun s x => s + x * x) 0.0
      let imE := imS.foldl (fun s x => s + x * x) 0.0
      IO.println s!"        analytic (Re,Im) bank, 3 modes: pair vs table oracles:"
      IO.println s!"        result   Re bit-diff {reDiff}/4096 (vs table) · Im bit-diff {imDiff}/4096 (vs A↦−iA table) · E[Re]={reE} E[Im]={imE}"
      if reDiff == 0 && imDiff == 0 && reE > 1e-9 && imE > 1e-9 then
        passGate "modal-pair" s!"analytic pair: Re ≡ table, Im ≡ (A↦−iA) table, both bit-identical — the twist/DD substrate"
      else
        failGate "modal-pair" s!"reDiff={reDiff} imDiff={imDiff} reE={reE} imE={imE}"
    | .error e, _, _, _ | _, .error e, _, _ | _, _, .error e, _ | _, _, _, .error e =>
      failGate "modal-pair" s!"render: {firstLine e}"
  | .error e, _, _, _ | _, .error e, _, _ | _, _, .error e, _ | _, _, _, .error e =>
    failGate "modal-pair" s!"build: {firstLine e}"

/-- THE BESSEL FUSE gate (static-index FM → sideband bank, Jacobi–Anger). Three
    independent checks: (i) `besselJ` satisfies the recurrence
    `Jₙ₋₁(b)+Jₙ₊₁(b) = (2n/b)Jₙ(b)` (~1e-10); (ii) Parseval `Σₙ Jₙ(b)² = 1` — FM
    conserves energy, the signature that the sideband weights ARE the FM
    decomposition (~1e-6); (iii) the FUSED bank converges SUPEREXPONENTIALLY in the
    sideband count — `‖fuse(N) − fuse(N_ref)‖` drops ≫10× per few sidebands as `N`
    passes `⌈b⌉` (the tail `|n|>b` decays faster than any geometric), rendered
    through the real engine path (self-convergence: both sides share the exact
    lowering, so no straight-line phase drift). Correct weights, energy, truncation. -/
def runModalBessel (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let fabs := fun (x : Float) => if x < 0.0 then -x else x
  let b := 3.0
  let wm := 2.0 * 3.141592653589793 * 308.0                 -- ω_m = 2π·308 rad/s
  -- (i) recurrence Jₙ₋₁+Jₙ₊₁ = (2n/b)Jₙ
  let mut recErr : Float := 0.0
  for nn in [1:6] do
    let nf := nn.toFloat
    let e := fabs ((besselJ (nf - 1.0) b + besselJ (nf + 1.0) b) - (2.0 * nf / b) * besselJ nf b)
    if e > recErr then recErr := e
  -- (ii) Parseval Σ Jₙ² = 1
  let mut energy : Float := 0.0
  for i in [0:41] do
    let jn := besselJ (i.toFloat - 20.0) b
    energy := energy + jn * jn
  let parseval := fabs (energy - 1.0)
  -- (iii) superexponential render convergence (self-reference to N=19 ≈ exact)
  let carrier : Array ModalMode := #[ModalMode.hz (lit 220) (lit 20 1) (lit 1)]
  let anchor := lit 200
  let fuse := fun (nm : String) (N : Nat) =>
    buildModalBankArrow nm (besselFuse carrier wm b N) anchor arena
  match buildAndFinish (.ok (fuse "fz5" 5)), buildAndFinish (.ok (fuse "fz8" 8)),
        buildAndFinish (.ok (fuse "fz11" 11)), buildAndFinish (.ok (fuse "fzR" 19)) with
  | .ok p5, .ok p8, .ok p11, .ok pR =>
    match ← renderPlanSamples p5 4096, ← renderPlanSamples p8 4096,
          ← renderPlanSamples p11 4096, ← renderPlanSamples pR 4096 with
    | .ok s5, .ok s8, .ok s11, .ok sR =>
      let relTo := fun (s : Array Float) => Id.run do
        let n := min s.size sR.size
        let mut num : Float := 0.0
        let mut den : Float := 0.0
        for k in [0:n] do
          num := num + (s[k]! - sR[k]!) * (s[k]! - sR[k]!)
          den := den + sR[k]! * sR[k]!
        return Float.sqrt (num / (den + 1e-300))
      let e5 := relTo s5
      let e8 := relTo s8
      let e11 := relTo s11
      let rE := sR.foldl (fun a x => a + x * x) 0.0
      IO.println s!"        static-index FM (b={b}, ω_m=2π·308) fused to a Bessel bank:"
      IO.println s!"        oracle   recurrence err={recErr} · Parseval |ΣJ²−1|={parseval}"
      IO.println s!"        result   ‖fuse(N)−fuse(19)‖/‖·‖: N=5 {e5} → N=8 {e8} → N=11 {e11}"
      if recErr < 1e-10 && parseval < 1e-6 && e5 > 8.0 * e8 && e8 > 8.0 * e11
          && e11 < 1e-3 && rE > 1e-9 then
        passGate "modal-bessel" s!"Jₙ correct (recurrence {recErr}, Parseval {parseval}); FM'd bank truncates superexponentially (N=5→8→11: {e5}→{e8}→{e11})"
      else
        failGate "modal-bessel" s!"recErr={recErr} parseval={parseval} e5={e5} e8={e8} e11={e11} rE={rE}"
    | .error e, _, _, _ | _, .error e, _, _ | _, _, .error e, _ | _, _, _, .error e =>
      failGate "modal-bessel" s!"render: {firstLine e}"
  | .error e, _, _, _ | _, .error e, _, _ | _, _, .error e, _ | _, _, _, .error e =>
    failGate "modal-bessel" s!"build: {firstLine e}"

/-- THE HETERODYNE gate (D6). Heterodyne FM as a twist at the realization seam —
    `Re·cosθ − Im·sinθ` over the analytic pair, `θ = b·sin(ω_m d)` — is ONE
    rotation per sample, independent of bank size. It must render equal to
    `besselFuse`'s EXPLICIT sideband bank (`carrier × (2N+1)` modes at `μ+i·n·ω_m`,
    amp `A·Jₙ(b)`): the same FM two ways, so the cheap twist really is a modal
    object with poles. Tolerance absorbs the float-θ vs baked-sideband datapath
    difference (`sinSig`/`cosSig` poly vs Q4.28 amps); the claim is agreement, and
    the twist's advantage is the trip count — the carrier's modes, not `×(2N+1)`. -/
def runModalHeterodyne (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let b := 3.0
  let wm := 2.0 * 3.141592653589793 * 308.0
  let carrier : Array ModalMode := #[
    ModalMode.hz (lit 220) (lit 20 1) (lit 1),
    ModalMode.hz (lit 330) (lit 30 1) (lit 5 1)]
  let anchor := lit 200
  match buildAndFinish (.ok (buildHeterodyne "het" carrier wm b anchor arena)),
        buildAndFinish (.ok (buildModalBankArrow "hetFuse" (besselFuse carrier wm b 19) anchor arena)) with
  | .ok hetP, .ok fuseP =>
    match ← renderPlanSamples hetP 4096, ← renderPlanSamples fuseP 4096 with
    | .ok hS, .ok fS =>
      let n := min hS.size fS.size
      let mut num : Float := 0.0
      let mut den : Float := 0.0
      for k in [0:n] do
        num := num + (hS[k]! - fS[k]!) * (hS[k]! - fS[k]!)
        den := den + fS[k]! * fS[k]!
      let rel := Float.sqrt (num / (den + 1e-300))
      let hEnergy := hS.foldl (fun a x => a + x * x) 0.0
      IO.println s!"        heterodyne twist vs besselFuse bank (b={b}, 2-mode carrier):"
      IO.println s!"        result   rel-L2 het≡fused-bank={rel} · plan-instrs het={planInstrCount hetP} fuse(19)={planInstrCount fuseP} · E[het]={hEnergy}"
      -- (hardening 0a-4) tightened from 1e-3 to 2e-5 (~10× the observed ~2e-6
      -- float-θ-vs-baked-sideband floor); the old 1e-3 was ~500× above the floor.
      if rel < 2e-5 && hEnergy > 1e-9 then
        passGate "modal-heterodyne" s!"heterodyne twist (Re·cosθ−Im·sinθ) ≡ the fused Bessel bank (rel {rel}) — O(1)-in-sidebands FM, still a modal object (D6)"
      else
        failGate "modal-heterodyne" s!"rel={rel} hEnergy={hEnergy}"
    | .error e, _ | _, .error e => failGate "modal-heterodyne" s!"render: {firstLine e}"
  | .error e, _ | _, .error e => failGate "modal-heterodyne" s!"build: {firstLine e}"

/-- RK4 of the modulated-resonator ODE `ẋ = μ(t)·x`, `μ(t) = −σ + iω₀(1 + p·cos ω_m t)`,
    from `x(0)=1` to `T` in `n` steps — the independent oracle for `modal-vco`. -/
private def rk4Osc (sigma om0 p wm T : Float) (n : Nat) : Cplx := Id.run do
  let h := T / n.toFloat
  let mu := fun (t : Float) => (⟨-sigma, om0 * (1.0 + p * Float.cos (wm * t))⟩ : Cplx)
  let mut x : Cplx := ⟨1.0, 0.0⟩
  let mut t : Float := 0.0
  for _ in [0:n] do
    let k1 := (mu t).mul x
    let k2 := (mu (t + h * 0.5)).mul (x.add (k1.mul ⟨h * 0.5, 0.0⟩))
    let k3 := (mu (t + h * 0.5)).mul (x.add (k2.mul ⟨h * 0.5, 0.0⟩))
    let k4 := (mu (t + h)).mul (x.add (k3.mul ⟨h, 0.0⟩))
    let s := (k1.add (k2.mul ⟨2.0, 0.0⟩)).add ((k3.mul ⟨2.0, 0.0⟩).add k4)
    x := x.add (s.mul ⟨h / 6.0, 0.0⟩)
    t := t + h
  return x

/-- THE LFO→POLE gate (integrated reading, D1/D2). Wiring an LFO to a resonator's
    pole forces a READING of a time-varying pole. The INTEGRATED reading (phase
    advances by `∫` of the modulated frequency, `θ = ω₀p·Re(∫LFO)` — `integrateBank`
    into a heterodyne twist) IS the exact solution of `ẋ = μ(t)x`: it converges to
    an independent RK4 integration at the O(h⁴) RK4 rate (error ratio ≈16 per
    halving). The SNAPSHOT reading (pole read at τ, applied over the whole elapsed
    d) is a different, well-defined function that does NOT solve the ODE — its error
    PLATEAUS (the house discriminator). Plus: the integrated realization builds and
    renders causally in the engine. -/
def runModalVco (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let tp := 6.283185307179586
  let sigma := 0.35
  let f0 := 5.0
  let p := 0.08
  let fm := 0.8
  let om0 := tp * f0
  let wm := tp * fm
  let T := 2.0
  let env := Float.exp (-sigma * T)
  -- integrated reading closed form: phase = ω₀(T + p·sin(ω_m T)/ω_m)
  let phiInt := om0 * (T + p * Float.sin (wm * T) / wm)
  let xInt : Cplx := ⟨env * Float.cos phiInt, env * Float.sin phiInt⟩
  -- snapshot reading: pole read at T, applied over the whole elapsed T
  let omSnap := om0 * (1.0 + p * Float.cos (wm * T))
  let xSnap : Cplx := ⟨env * Float.cos (omSnap * T), env * Float.sin (omSnap * T)⟩
  let relTo := fun (a b : Cplx) => (a.add b.neg).abs / (b.abs + 1e-300)
  let e500 := relTo xInt (rk4Osc sigma om0 p wm T 500)
  let e1000 := relTo xInt (rk4Osc sigma om0 p wm T 1000)
  let e2000 := relTo xInt (rk4Osc sigma om0 p wm T 2000)
  let r1 := e500 / (e1000 + 1e-300)
  let r2 := e1000 / (e2000 + 1e-300)
  let snapErr := relTo xSnap (rk4Osc sigma om0 p wm T 4000)
  -- the integrated realization + a RAW-BANK variant (hardening 0a-1): the same
  -- carrier, but θ from the un-integrated LFO bank instead of `integrateBank lfo`
  -- — a LOCAL fixture copy (production `buildIntegratedPoleReading` untouched). Its
  -- θ = ω₀p·cos(ω_m d), NOT ω₀p·sin(ω_m d)/ω_m, so it must DIVERGE from the oracle:
  -- the proof that the render-vs-oracle check below has teeth.
  let carrier : Array ModalMode := #[ModalMode.hz (litF f0) (litF sigma) (lit 1)]
  let lfo : Array ModalMode := #[ModalMode.hz (litF fm) (lit 0) (lit 1)]
  let anchor := lit 200
  let rawVariant : Arena × ProgramIdx :=
    let (re, im) := modalBankSigPairTable carrier clockLit anchor
    let thetaRaw := mul (litF (om0 * p)) (modalBankSig lfo clockLit anchor)
    buildExprCarrier "vco_raw" (sub (mul re (cosSig thetaRaw)) (mul im (sinSig thetaRaw))) arena
  match buildAndFinish (.ok (buildIntegratedPoleReading "vco" carrier lfo (om0 * p) anchor arena)),
        buildAndFinish (.ok rawVariant) with
  | .ok vp, .ok rawP =>
    match ← renderPlanSamples vp 4096, ← renderPlanSamples rawP 4096 with
    | .ok s, .ok sRaw =>
      let mut preMax : Float := 0.0
      for i in [0:201] do
        if s[i]!.abs > preMax then preMax := s[i]!.abs
      let energy := s.foldl (fun a x => a + x * x) 0.0
      -- (NEW, hardening 0a-1) the RENDERED integrated reading vs the closed-form
      -- oracle `sinkGain·env(d)·cos(φ_int(d))` at MATCHED sample offsets — the
      -- render is samples at SR anchored @200, so d = (i−200)/SR; the oracle is
      -- Re(x_int(d)), env=e^{−σd}, φ_int=ω₀(d + p·sin(ω_m d)/ω_m). This exercises
      -- `buildIntegratedPoleReading`'s ACTUAL render (was only smoke-checked). The
      -- raw variant is measured against the SAME oracle and must blow past `bound`.
      let n := min (min s.size sRaw.size) 4096
      let sinkGain : Float := Tropical.Plan.defaultSinkGain.toFloat
      let oracleAt := fun (i : Nat) =>
        let d := (i.toFloat - 200.0) / 44100.0
        let phi := om0 * (d + p * Float.sin (wm * d) / wm)
        sinkGain * Float.exp (-sigma * d) * Float.cos phi
      let mut renderErr : Float := 0.0
      let mut rawErr : Float := 0.0
      for i in [201:n] do
        let o := oracleAt i
        let e := (s[i]! - o).abs
        if e > renderErr then renderErr := e
        let er := (sRaw[i]! - o).abs
        if er > rawErr then rawErr := er
      let renderBound : Float := 3.0e-5 * Tropical.Plan.defaultSinkGain.toFloat    -- ~10× the observed ~1.46e-7 Q4.28/freq-grid floor (raw) × the sink gain
      IO.println s!"        LFO→pole integrated reading vs RK4 of ẋ=μ(t)x (f0={f0}, p={p}, fm={fm}):"
      IO.println s!"        oracle   integrated vs RK4 rel err: n=500 {e500} 1000 {e1000} 2000 {e2000} (ratios {r1}, {r2}; ~16=h⁴)"
      IO.println s!"        result   snapshot vs ODE={snapErr} (plateaus) · render pre-strike|max|={preMax} E={energy}"
      IO.println s!"        render   ≡ env·cos(φ_int) max|Δ|={renderErr*1e9}e-9 (bound {renderBound*1e9}e-9) · raw-LFO variant max|Δ|={rawErr} (must exceed)"
      if 10.0 < r1 && r1 < 24.0 && 10.0 < r2 && r2 < 24.0 && snapErr > 1e-2
          && preMax == 0.0 && energy > 1e-9
          && renderErr < renderBound && rawErr > renderBound then
        passGate "modal-vco" s!"the integrated reading IS the modulated resonator (RK4 h⁴: ratios {r1}, {r2}); its RENDER ≡ env·cos(φ_int) (max|Δ| {renderErr}); the raw-LFO variant diverges (max|Δ| {rawErr}); snapshot plateaus ({snapErr}) (D1/D2)"
      else
        failGate "modal-vco" s!"r1={r1} r2={r2} snapErr={snapErr} preMax={preMax} energy={energy} renderErr={renderErr} (bound {renderBound}) rawErr={rawErr}"
    | .error e, _ | _, .error e => failGate "modal-vco" s!"render: {firstLine e}"
  | .error e, _ | _, .error e => failGate "modal-vco" s!"build: {firstLine e}"

/-- THE AFFINE RECLOCK gate. `reclockAffine a b` is the pole-space image of the
    affine clock warp `d↦a·d+b`, so the reclocked bank at sample `i` equals the
    ORIGINAL bank at the warped sample `a·i + b·SR`: `reclock[i] ≡ orig[a·i + b·SR]`
    (both anchored at 0) — a direct subsample/shift comparison, no `.warp` path.
    Two arms so each isolates one axis: (A) `a=1, b·SR=10` — poles UNCHANGED, so the
    only change is the amp rotation `A↦A·e^{μb}` (envelope `e^{−σb}` × phase
    `e^{iωb}`); the integer phase is untouched, so this is drift-free and TIGHT.
    (B) `a=2, b=0` — the pole scale `μ↦2μ`; the bank now uses integer phase on `2ω`,
    which quantizes to the SR/2³² frequency grid INDEPENDENTLY of `2×(ω's grid)`, a
    phase drift `≤ N·2⁻³²·2π` rad accumulating over the window (inaudible ~1e-5 Hz),
    so this arm's bound is soft and window-scaled, not bit-exact. -/
def runModalReclock (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let tp := 6.283185307179586
  let fabs := fun (x : Float) => if x < 0.0 then -x else x
  let toMode := fun (pa : Cplx × Cplx) =>
    ({ sigma := litF (-pa.1.re), omega := litF pa.1.im,
       cre := litF pa.2.re, cim := litF pa.2.im } : ModalMode)
  let modes : Array ModalMode := #[
    (⟨-2.0, tp * 220.0⟩, (⟨0.6, 0.2⟩ : Cplx)),
    (⟨-3.0, tp * 337.0⟩, ⟨0.4, -0.3⟩)].map toMode
  let anchor := lit 0
  let dS : Nat := 10
  let recA := reclockAffine (litF 1.0) (litF (dS.toFloat / 44100.0)) modes   -- delay
  let recB := reclockAffine (litF 2.0) (litF 0.0) modes                       -- scale
  match buildAndFinish (.ok (buildModalBankArrow "rc_o" modes anchor arena)),
        buildAndFinish (.ok (buildModalBankArrow "rc_a" recA anchor arena)),
        buildAndFinish (.ok (buildModalBankArrow "rc_b" recB anchor arena)) with
  | .ok op, .ok ap, .ok bp =>
    match ← renderPlanSamples op 8192, ← renderPlanSamples ap 4096,
          ← renderPlanSamples bp 4096 with
    | .ok os, .ok as_, .ok bs =>
      let mut maxA : Float := 0.0                       -- arm A: amp rotation, tight
      for i in [1:4000] do
        if i + dS < os.size then
          let d := fabs (as_[i]! - os[i + dS]!)
          if d > maxA then maxA := d
      let mut maxB : Float := 0.0                       -- arm B: pole scale, soft
      for i in [1:4000] do
        if 2 * i < os.size then
          let d := fabs (bs[i]! - os[2 * i]!)
          if d > maxB then maxB := d
      let boundA := 2.0 * 3.7252903e-9 * Tropical.Plan.defaultSinkGain.toFloat * 100.0                    -- Q + poly ulp
      let boundB := 4000.0 * 2.3283064e-10 * tp * 1.13 * Tropical.Plan.defaultSinkGain.toFloat * 2.0      -- freq-grid drift
      let eB := bs.foldl (fun a x => a + x * x) 0.0
      IO.println s!"        affine reclock: (A) delay a=1,b·SR=10  (B) scale a=2:"
      IO.println s!"        result   armA max|Δ|={maxA * 1e9}e-9 (bound {boundA * 1e9}e-9, tight) · armB max|Δ|={maxB * 1e9}e-9 (bound {boundB * 1e9}e-9, freq-grid)"
      if maxA < boundA && maxB < boundB && eB > 1e-9 then
        passGate "modal-reclock" s!"amp rotation A↦A·e^(μb) exact (armA {maxA*1e9}e-9); pole scale ω↦2ω within the SR/2³² frequency grid (armB {maxB*1e9}e-9)"
      else
        failGate "modal-reclock" s!"maxA={maxA*1e9}e-9 (bound {boundA*1e9}) maxB={maxB*1e9}e-9 (bound {boundB*1e9}) eB={eB}"
    | .error e, _, _ | _, .error e, _ | _, _, .error e => failGate "modal-reclock" s!"render: {firstLine e}"
  | .error e, _, _ | _, .error e, _ | _, _, .error e => failGate "modal-reclock" s!"build: {firstLine e}"

/-- THE DIVIDED-DIFFERENCE gate (WS-B2). `residueComposeDD` → `modalBankSigTableDD`
    composes voice ⋙ reverb as fused paired modes with NO `1/Δ`, stable through pole
    coincidence. Three arms: (1) AWAY from coincidence the DD bank renders the same
    composition as the collected `residueComposeEC` form, agreeing within the SR/2³²
    frequency-grid floor every `modePhaseQ` bank shares (the DD path reconstructs
    each `ω_λ` as `ω_ν + (ω_λ−ω_ν)`, two integer-phase rotators). (2) at EXACT
    coincidence (λ=ν) a single paired mode reproduces the deg-1 `τ·e^{νd}` resonance
    — matching a hand-built deg-1 bank at the SAME frequency within the Q4.28
    landing quantum (a `max|Δ| < bound` TOLERANCE, not `bitDiffCount == 0`
    bit-identity; no branch, no blowup — the `cexpm1` series limit). (3) the coeff `c = a·r` is bounded
    (`|c| < 8`, Q4.28-safe) where the collected ringing amp `|a·r/Δ|` overflows for
    small `Δ` — the fixed-point disqualifier the paired form removes. -/
def runResidueDivDiff (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let tp := 6.283185307179586
  let fabs := fun (x : Float) => if x < 0.0 then -x else x
  let toMode := fun (pa : Cplx × Cplx) =>
    ({ sigma := litF (-pa.1.re), omega := litF pa.1.im,
       cre := litF pa.2.re, cim := litF pa.2.im } : ModalMode)
  let voiceF : Array (Cplx × Cplx) := #[
    (⟨-2.0, tp * 220.0⟩, ⟨1.0, 0.0⟩), (⟨-2.5, tp * 330.0⟩, ⟨0.6, 0.0⟩)]
  let reverbF : Array (Cplx × Cplx) := #[
    (⟨-3.0, tp * 180.0⟩, ⟨0.7, 0.2⟩), (⟨-4.0, tp * 260.0⟩, ⟨-0.5, 0.4⟩),
    (⟨-5.0, tp * 350.0⟩, ⟨0.3, -0.6⟩), (⟨-6.0, tp * 500.0⟩, ⟨0.4, 0.1⟩)]
  let voice := voiceF.map toMode
  let reverb := reverbF.map toMode
  let anchor := lit 200
  let nDD := (residueComposeDD voice reverb).size
  -- arm 3 (algebraic): |c|=|a·r| bounded; collected |a·r/Δ| overflows at Δ=0.03
  let mut cMax : Float := 0.0
  for pa in voiceF do
    for pr in reverbF do
      let m := (Cplx.mul pa.2 pr.2).abs
      if m > cMax then cMax := m
  let overflowsAt003 := cMax / 0.03
  -- arm 2 (coincidence): one paired mode λ=ν ≡ a deg-1 τ·e bank at ω=2π·220
  let cpole : Cplx := ⟨-2.0, tp * 220.0⟩
  let vC := #[(cpole, (⟨1.0, 0.0⟩ : Cplx))].map toMode
  let rC := #[(cpole, (⟨0.5, 0.2⟩ : Cplx))].map toMode
  let ar := Cplx.mul ⟨1.0, 0.0⟩ ⟨0.5, 0.2⟩
  let deg1 : Array ModalMode := #[
    { sigma := litF 2.0, omega := litF (tp * 220.0), cre := litF ar.re, cim := litF ar.im, deg := 1 }]
  match buildAndFinish (.ok (buildModalReverbDD "dd_far" voice reverb anchor arena)),
        buildAndFinish (.ok (buildModalReverbSymC "col_far" voice reverb anchor arena)),
        buildAndFinish (.ok (buildModalReverbDD "dd_coin" vC rC anchor arena)),
        buildAndFinish (.ok (buildModalBankArrow "deg1_ref" deg1 anchor arena)) with
  | .ok ddF, .ok colF, .ok ddC, .ok refC =>
    match ← renderPlanSamples ddF 4096, ← renderPlanSamples colF 4096,
          ← renderPlanSamples ddC 4096, ← renderPlanSamples refC 4096 with
    | .ok dfs, .ok cfs, .ok dcs, .ok rcs =>
      let mut n1 : Float := 0.0
      let mut d1 : Float := 0.0
      for k in [0:min dfs.size cfs.size] do
        n1 := n1 + (dfs[k]! - cfs[k]!) * (dfs[k]! - cfs[k]!)
        d1 := d1 + cfs[k]! * cfs[k]!
      let rel1 := Float.sqrt (n1 / (d1 + 1e-300))
      let mut m2 : Float := 0.0
      let mut e2 : Float := 0.0
      for k in [0:min dcs.size rcs.size] do
        let d := fabs (dcs[k]! - rcs[k]!)
        if d > m2 then m2 := d
        e2 := e2 + dcs[k]! * dcs[k]!
      let bound2 := 3.0 * 3.7252903e-9 * Tropical.Plan.defaultSinkGain.toFloat * 8.0
      -- arm 4 (near-coincidence SERIES sweep, hardening 0a-2): the small-|z| Horner
      -- branch `cexpm1SeriesE` (selected when |z|²<0.01, i.e. |z|<0.1) is
      -- load-bearing only when |λ−ν| is small enough that z=(λ−ν)d stays <0.1
      -- across the window — arms 1-3 never enter it (well-separated ⇒ direct;
      -- exact coincidence ⇒ z=0). Sweep the separation (DIRECTION = e^{i·0.7},
      -- matching demos/divdiff_qdatapath.py's TARGETS) and check each DD render
      -- against a DIRECT-double closed-form oracle Re(c·d·e^{νd}·(e^z−1)/z) — a
      -- reference INDEPENDENT of the series coefficients, so a flipped LOW-ORDER
      -- coeff shows (k≤2 solidly, k=3 marginal). The k≥4 terms sit below the
      -- render's own floor at |z|≤0.088 and pass undetected — they cannot be made
      -- observable without leaving the |z|<0.1 series branch, so the sweep certifies
      -- render accuracy over the branch, not every coefficient. The collected form
      -- can't be the oracle here: its |a·r/Δ| overflows Q4.28 at small Δ (arm 3). At
      -- tgt=1.0 z reaches ~0.088 (<0.1 ⇒ still the series branch), where the
      -- low-order z²/z³ coefficients bite.
      let cexp := fun (w : Cplx) =>
        let m := Float.exp w.re
        (⟨m * Float.cos w.im, m * Float.sin w.im⟩ : Cplx)
      let dirC : Cplx := ⟨Float.cos 0.7, Float.sin 0.7⟩
      let nuC : Cplx := ⟨-2.0, tp * 220.0⟩            -- the shared reverb pole ν
      let rC2 : Cplx := ⟨0.7, 0.2⟩                    -- reverb residue r
      let aC : Cplx := ⟨1.0, 0.0⟩                     -- voice amp a
      let cC := Cplx.mul aC rC2                        -- c = a·r (bounded)
      let targets : Array Float := #[1.0, 3e-1, 1e-1, 3e-2, 1e-2, 1e-3, 1e-4, 1e-6]
      let mut sweepMax : Float := 0.0
      let mut sweepOk := true
      let mut sweepWorstTgt : Float := 0.0
      for ti in [0:targets.size] do
        let tgt := targets[ti]!
        let lamC : Cplx := nuC.add (dirC.mul ⟨tgt, 0.0⟩)   -- λ = ν + tgt·e^{i·0.7}
        let vS := #[(lamC, aC)].map toMode
        let rS := #[(nuC, rC2)].map toMode
        match buildAndFinish (.ok (buildModalReverbDD s!"dd_sw{ti}" vS rS anchor arena)) with
        | .error _ => sweepOk := false
        | .ok ddP =>
          match ← renderPlanSamples ddP 4096 with
          | .error _ => sweepOk := false
          | .ok ds =>
            for i in [201:min ds.size 4096] do
              let d := (i.toFloat - 200.0) / 44100.0
              let z := (Cplx.sub lamC nuC).mul ⟨d, 0.0⟩              -- (λ−ν)·d
              let cxm1 := (Cplx.sub (cexp z) ⟨1.0, 0.0⟩).div z        -- (e^z−1)/z direct
              let enu := cexp (nuC.mul ⟨d, 0.0⟩)                      -- e^{νd}
              let contrib := ((cC.mul ⟨d, 0.0⟩).mul enu).mul cxm1     -- c·d·e^{νd}·cexpm1
              let e := fabs (ds[i]! - Tropical.Plan.defaultSinkGain.toFloat * contrib.re)
              if e > sweepMax then sweepMax := e; sweepWorstTgt := tgt
      let sweepBound : Float := 3.0e-6 * Tropical.Plan.defaultSinkGain.toFloat                -- ~13× the observed Q/freq-grid floor (raw) × the sink gain
      IO.println s!"        divided-difference composition (voice(2)⋙reverb(4), DD {nDD} paired modes):"
      IO.println s!"        arm1     DD≡collected (well-sep) rel-L2={rel1} (freq-grid floor)"
      IO.println s!"        arm2     DD@coincidence≡deg-1 τ·e max|Δ|={m2 * 1e9}e-9 (bound {bound2 * 1e9}e-9, tight)"
      IO.println s!"        arm3     |c|max={cMax} (<8, Q4.28-safe) · collected |a·r/Δ|@Δ=0.03={overflowsAt003} (>8 overflows)"
      IO.println s!"        arm4     series-branch sweep max|Δ|={sweepMax*1e9}e-9 (bound {sweepBound*1e9}e-9) @tgt={sweepWorstTgt} · builds ok={sweepOk}"
      -- (hardening 0a-4) arm 1 tightened from 2e-3 to 2e-5 (~10× the observed
      -- ~2e-6 SR/2³² frequency-grid floor); the old 2e-3 was ~1000× above it.
      if nDD == voice.size * reverb.size && rel1 < 2e-5 && m2 < bound2 && e2 > 1e-9
          && cMax < 8.0 && overflowsAt003 > 8.0 && sweepOk && sweepMax < sweepBound then
        passGate "residue-divdiff" s!"fused paired modes: away-from-coincidence ≡ collected (rel {rel1}); coincidence ≡ τ·e within the Q4.28 quantum (max|Δ| {m2*1e9}e-9, a tolerance — not bit-identity); near-coincidence series sweep ≡ direct-double oracle (max|Δ| {sweepMax}); |c|={cMax}<8 vs collected 1/Δ overflow — stable, no 1/Δ"
      else
        failGate "residue-divdiff" s!"nDD={nDD} rel1={rel1} m2={m2*1e9}e-9 (bound {bound2*1e9}) e2={e2} cMax={cMax} ovf={overflowsAt003} sweepMax={sweepMax} (bound {sweepBound}) sweepOk={sweepOk}"
    | .error e, _, _, _ | _, .error e, _, _ | _, _, .error e, _ | _, _, _, .error e =>
      failGate "residue-divdiff" s!"render: {firstLine e}"
  | .error e, _, _, _ | _, .error e, _, _ | _, _, .error e, _ | _, _, _, .error e =>
    failGate "residue-divdiff" s!"build: {firstLine e}"

/-- THE BANKED CAUCHY FILLS gate (WS-F). `residueComposeBanked` computes the
    collected form's Cauchy inner sums (`Hlam`/`coupling`) as scalar `Sig.bankSum`s
    over the source columns — same value as `residueComposeEC` (bit-identical render,
    per-term `cdivE`, left-assoc), but O(m+n) reduce-region fill code in place of the
    O(m·n) meta-unrolled ops. Two arms: (1) EQUIVALENCE — banked ≡ collected
    bit-for-bit over the render; (2) FLATNESS — with LIVE (`paramRef`) poles (so the
    Cauchy structure survives const-folding), the banked plan carries fewer
    instructions than the unrolled one at a 6⋙6 composition. -/
def runResidueBanked (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let tp := 6.283185307179586
  let toMode := fun (pa : Cplx × Cplx) =>
    ({ sigma := litF (-pa.1.re), omega := litF pa.1.im,
       cre := litF pa.2.re, cim := litF pa.2.im } : ModalMode)
  let voice := #[
    (⟨-2.0, tp * 220.0⟩, (⟨1.0, 0.0⟩ : Cplx)), (⟨-2.5, tp * 330.0⟩, ⟨0.6, 0.0⟩)].map toMode
  let reverb := #[
    (⟨-3.0, tp * 180.0⟩, (⟨0.7, 0.2⟩ : Cplx)), (⟨-4.0, tp * 260.0⟩, ⟨-0.5, 0.4⟩),
    (⟨-5.0, tp * 350.0⟩, ⟨0.3, -0.6⟩), (⟨-6.0, tp * 500.0⟩, ⟨0.4, 0.1⟩)].map toMode
  let anchor := lit 200
  let nB := (residueComposeBanked voice reverb).size
  -- flatness: live paramRef poles keep the Cauchy sums out of the const-folder
  let pr := fun (i : Nat) => (Sig.paramRef ⟨i⟩ : Sig)
  let mkLive := fun (b : Nat) =>
    ({ sigma := pr b, omega := pr (b + 1), cre := pr (b + 2), cim := pr (b + 3) } : ModalMode)
  let voiceL := (Array.range 6).map (fun i => mkLive (4 * i))
  let reverbL := (Array.range 6).map (fun i => mkLive (24 + 4 * i))
  -- FLATNESS (hardening 0a-3): the live-pole (paramRef) builds keep the Cauchy
  -- inner sums out of the const-folder, so the O(m+n) vs O(m·n) plan-instr gap
  -- is observable. A build `.error` here is a REAL failure — route it to
  -- `failGate` exactly like every other arm. (The old code matched `| _ => none`
  -- and then `flatOk := match … | none => true`, so a live-build FAILURE printed
  -- "n/a" inside a PASSING gate — a build failure mapped to a green result.) A
  -- genuine success still checks `b < c`.
  match buildAndFinish (.ok (buildModalReverbSymC "rbcL" voiceL reverbL anchor arena)),
        buildAndFinish (.ok (buildModalReverbBanked "rbbL" voiceL reverbL anchor arena)) with
  | .ok clp, .ok blp =>
    let cN := planInstrCount clp
    let bN := planInstrCount blp
    let flatStr := s!"unrolled {cN} vs banked {bN} plan-instrs (6⋙6)"
    let flatOk := decide (bN < cN)
    match buildAndFinish (.ok (buildModalReverbSymC "rbc" voice reverb anchor arena)),
          buildAndFinish (.ok (buildModalReverbBanked "rbb" voice reverb anchor arena)) with
    | .ok cp, .ok bp =>
      match ← renderPlanSamples cp 4096, ← renderPlanSamples bp 4096 with
      | .ok cs, .ok bs =>
        let bitDiff := bitDiffCount cs bs
        let e := bs.foldl (fun a x => a + x * x) 0.0
        IO.println s!"        banked Cauchy fills (collected form): equivalence + flatness"
        IO.println s!"        result   banked≡collected bit-diff {bitDiff}/4096 · {nB} modes · {flatStr}"
        if bitDiff == 0 && e > 1e-9 && nB == voice.size + reverb.size && flatOk then
          passGate "residue-banked" s!"banked Cauchy fills ≡ collected bit-identical ({nB} modes); {flatStr} — O(m+n) coeff regions"
        else
          failGate "residue-banked" s!"bitDiff={bitDiff} nB={nB} e={e} flatOk={flatOk} flat={flatStr}"
      | .error e, _ | _, .error e => failGate "residue-banked" s!"render: {firstLine e}"
    | .error e, _ | _, .error e => failGate "residue-banked" s!"build: {firstLine e}"
  | .error e, _ | _, .error e => failGate "residue-banked" s!"live flatness build: {firstLine e}"

/-- THE BLOOM Γ-BRIDGE gate (WS-B3). `bloomCompose` → `bloomComposedSig` — the
    residue composition ACROSS a pitch-bloom warp as the two-carrier
    incomplete-gamma atom (series/CF envelopes bridged by the baked Γ★; cockpit
    `demos/modal_bloom_gamma.py`). The pair set includes the stationary-phase
    crossing — a 1023 Hz partial sweeping THROUGH a 1040 Hz reverb pole
    (|κ|≈178, |a|≈59) — the case the Poisson lattice can never render. Checks:
    (o)  build-time lgamma satisfies `exp(lgamma(a+1) − lgamma(a)) = a` on the
         pairs' actual `a` values (branch-insensitive form, ~1e-11);
    (1)  agreement with an independent trapezoid prefix-quadrature of the
         DEFINING convolution (rotator-grid ω, since the datapath deliberately
         quantizes frequency to SR/2³²; h→h/2 self-distance printed as the
         reference's own convergence evidence): rel-L2 < 2e-4 — ~14× the
         observed 1.4e-5 float-envelope-vs-fixed-datapath floor at this 0.74 s
         render (the heterodyne gate's ~2e-6 floor at 8× the τ; drift-type
         error scales with τ). A wrong atom sits at 1e-2–1e0;
    (2)  the κ→0 collapse: `B = 1e-12` renders ≡ `residueComposeEC`'s bank —
         the WS-B2 divided-difference limit (uncollected vs collected datapaths,
         so quantization-floor tolerance, not bit equality);
    (3)  seam continuity: the crossing pair's per-sample branch switch at
         `d_switch` introduces no step (adjacent-sample deltas in a ±16 window
         at the switch bounded by 2× the neighborhood's);
    (4)  causality: exact zero before the anchor. -/
def runModalBloomGamma (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let sr := 44100.0
  let tp := 6.283185307179586
  let g := 1.8
  let B := 0.05 / g                    -- β = 0.05, scale 1: the shipped full register
  let vData : Array (Float × Float × Float) := #[(1023.0, 1.13, 1.0), (353.1, 0.56, 0.6)]
  let rData : Array (Float × Float × Float) := #[(1040.0, 1.0, 0.4), (700.0, 1.5, 0.3)]
  let mk := fun ((f, s, a) : Float × Float × Float) =>
    ({ sigma := litF s, omega := litF (tp * f), cre := litF a } : ModalMode)
  let voice := vData.map mk
  let reverb := rData.map mk
  let anchorN : Nat := 200
  let anchor := lit 200
  let n : Nat := 32768
  let fabs := fun (x : Float) => if x < 0.0 then -x else x
  match bloomCompose voice reverb B g,
        bloomCompose voice reverb 1e-12 g with
  | none, _ | _, none =>
    failGate "modal-bloom-gamma" "bloomCompose: a live pole reached the baked-pole contract"
  | some pairs, some pairs0 =>
    -- (o) the lgamma recurrence on the actual a values (the pairs are BAKED, so
    -- the `Sig` fields fold back to their Floats via `sigConstF?`)
    let cf := fun (s : Tropical.EmitArrow.Sig) => (Tropical.EmitArrow.sigConstF? s).getD 0.0
    let mut lgErr : Float := 0.0
    for p in pairs do
      let aC : CplxB := ⟨(cf p.nuSigma - cf p.muSigma) / g * (-1.0), (cf p.nuOmega - cf p.muOmega) / g⟩
      let ratio := (CplxB.exp ((lgammaB (aC.add ⟨1, 0⟩)).sub (lgammaB aC))).div aC
      lgErr := max lgErr ((ratio.sub ⟨1, 0⟩).abs)
    match buildAndFinish (.ok (buildBloomComposed "bloomg" pairs anchor arena)),
          buildAndFinish (.ok (buildBloomComposed "bloomg0" pairs0 anchor arena)),
          buildAndFinish (.ok (buildModalBankTable "bloomg0r"
            (residueComposeEC voice reverb) anchor arena)) with
    | .ok plan, .ok plan0, .ok plan0r =>
      match ← renderPlanSamples plan n, ← renderPlanSamples plan0 4096,
            ← renderPlanSamples plan0r 4096 with
      | .ok dut, .ok dut0, .ok ref0 =>
        -- independent reference: trapezoid prefix-quadrature of
        -- y_p(d) = e^{νd}·∫₀^d e^{μφ(s) − νs} ds per admitted pair, summed with
        -- the real weights a·r — the defining integral, no gamma anywhere.
        let phi := fun (d : Float) => d + B * (1.0 - Float.exp (-g * d))
        let sinkGain : Float := Tropical.Plan.defaultSinkGain.toFloat   -- defaultSinkGain (Plan.lean): the carrier's output sink
        -- the rotator's documented frequency quantization (`modePhaseQ`:
        -- `incr = ⌊(ω/2π)·2³²/SR⌋`, the SR/2³² grid) — modeled in the reference
        -- so the trapezoid refines toward the RENDERED carriers, not toward
        -- frequencies the datapath deliberately does not carry
        let qOm := fun (om : Float) =>
          Float.floor (om / tp * 4294967296.0 / sr) * (tp * sr / 4294967296.0)
        let refRender := fun (hdiv : Nat) => Id.run do
          let mut y : Array Float := Array.replicate n 0.0
          for (fv, sv, av) in vData do
            let mu : CplxB := ⟨-sv, qOm (tp * fv)⟩
            for (fr, srr, ar) in rData do
              let nuC : CplxB := ⟨-srr, qOm (tp * fr)⟩
              if ((nuC.sub mu).scale (1.0 / g)).abs < 0.5 then continue
              let cAmp := av * ar * sinkGain
              let h := 1.0 / (sr * hdiv.toFloat)
              let fint := fun (s : Float) => CplxB.exp ((mu.scale (phi s)).sub (nuC.scale s))
              let mut J : CplxB := ⟨0, 0⟩
              let mut fPrev := fint 0.0
              for i in [anchorN + 1 : n] do
                let dBase := (i - 1 - anchorN).toFloat / sr
                for k in [0:hdiv] do
                  let fNext := fint (dBase + (k + 1).toFloat * h)
                  J := J.add ((fPrev.add fNext).scale (h * 0.5))
                  fPrev := fNext
                let d := (i - anchorN).toFloat / sr
                let yc := (CplxB.exp (nuC.scale d)).mul J
                y := y.set! i (y[i]! + cAmp * yc.re)
          return y
        let ref1 := refRender 1
        let ref2 := refRender 2
        let relL2 := fun (xa xb : Array Float) (lo hi : Nat) => Id.run do
          let mut nm := 0.0
          let mut dn := 0.0
          for i in [lo:hi] do
            let dd := xa[i]! - xb[i]!
            nm := nm + dd * dd
            dn := dn + xb[i]! * xb[i]!
          return Float.sqrt (nm / (dn + 1e-300))
        let e2 := relL2 dut ref2 (anchorN + 1) n
        let eT := relL2 ref1 ref2 (anchorN + 1) n
        let e0 := relL2 dut0 ref0 (anchorN + 1) 4096
        -- (3) seam continuity around the crossing pair's switch sample
        let crossing := pairs.filter (fun p => cf p.dSwitch > 0.0)
        let seamOk := Id.run do
          if crossing.isEmpty then return false
          let iSw := anchorN + (cf crossing[0]!.dSwitch * sr).toUInt64.toNat
          if iSw + 1000 ≥ n then return false
          let maxStep := fun (lo hi : Nat) => Id.run do
            let mut m := 0.0
            for i in [lo:hi] do
              m := max m (fabs (dut[i]! - dut[i-1]!))
            return m
          let w := maxStep (iSw - 16) (iSw + 16)
          let nb := max (maxStep (iSw - 1000) (iSw - 16)) (maxStep (iSw + 16) (iSw + 1000))
          return w ≤ 2.0 * nb + 1e-9
        -- (4) causality
        let preMax := (Array.range anchorN).foldl (fun m i => max m (fabs dut[i]!)) 0.0
        let energy := dut.foldl (fun a x => a + x * x) 0.0
        let depths := String.intercalate ", " (pairs.toList.map (fun p =>
          s!"{p.invA.size}/{p.cfN.size}"))
        IO.println s!"        bloom⋙reverb Γ-bridge atom ({pairs.size} pairs incl. the in-band crossing; β·|μ|/g up to ~178):"
        IO.println s!"        oracle   lgamma-recurrence {lgErr} · trapezoid h→h/2 self-distance {eT}"
        IO.println s!"        result   ‖dut−ref(h/2)‖ {e2} · κ→0 vs residueComposeEC {e0} · depths ser/cf {depths}"
        if lgErr < 1e-11 && e2 < 2e-4 && e0 < 1e-4
            && seamOk && preMax == 0.0 && energy > 1e-9 && crossing.size == 1 then
          passGate "modal-bloom-gamma" s!"the bloomed voice feeds the reverb in closed form — Γ-bridge atom at the datapath floor (rel {e2}; ref self-distance {eT}), κ→0 = the DD atom ({e0}), seam step-free"
        else
          failGate "modal-bloom-gamma" s!"lgErr={lgErr} e2={e2} eT={eT} e0={e0} seamOk={seamOk} preMax={preMax} energy={energy} crossing={crossing.size}"
      | .error e, _, _ | _, .error e, _ | _, _, .error e =>
        failGate "modal-bloom-gamma" s!"render: {firstLine e}"
    | .error e, _, _ | _, .error e, _ | _, _, .error e =>
      failGate "modal-bloom-gamma" s!"build: {firstLine e}"

end ResidueGates

open Tropical.EmitArrow in
/-- THE MODAL PATCH gate (the session surface). A modal-island `PatchGraph`
    (`resonator → reverb → out`) lowered through `lowerModal` (residue in pole
    space) and realized at its boundary must render a real, causal, decaying
    signal — and, read through a reversing master clock, play the tail backward
    bit-for-bit. This is the whole seam end to end: a patch graph, not a builder. -/
def runModalPatch (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let res : Array ModalMode := #[
    ModalMode.hz (lit 220) (lit 30 1) (lit 6 1),
    ModalMode.hz (lit 440) (lit 45 1) (lit 3 1),
    ModalMode.hz (lit 660) (lit 60 1) (lit 2 1)]
  let room : Array ModalMode := #[
    { sigma := lit 3, omega := mul twoPiE (lit 180), cre := lit 7 1, cim := lit 2 1 },
    { sigma := lit 4, omega := mul twoPiE (lit 300), cre := lit (-5) 1, cim := lit 4 1 },
    { sigma := lit 5, omega := mul twoPiE (lit 520), cre := lit 3 1, cim := lit (-6) 1 }]
  let anchor := lit 200
  let twoC : Int := 2048 * 4294967296
  let mkGraph := fun (clk : Clock) => ({
    nodes := #[
      { id := "res", node := .modalSource res anchor clk none },
      { id := "rev", node := .modalReverb "res" room none }],
    output := "rev" } : PatchGraph)
  let carrier := fun (name : String) (clk : Clock) => (do
    let term ← lowerGraph (mkGraph clk)
    let (out, _) := emitTerm (normalize term) {}
    .ok (buildExprCarrier name out arena) : Except String (Arena × ProgramIdx))
  let revClk : Clock := sub (lit twoC) clockLit
  match buildAndFinish (carrier "mp_fwd" clockLit),
        buildAndFinish (carrier "mp_rev" revClk) with
  | .ok fp, .ok rp =>
    match ← renderPlanSamples fp 2048, ← renderPlanSamples rp 2048 with
    | .ok fwd, .ok rev =>
      let n := min fwd.size rev.size
      let mut preMax : Float := 0.0
      for i in [0:201] do
        let a := fwd[i]!.abs
        if a > preMax then preMax := a
      let mut peak : Float := 0.0
      for i in [201:n] do
        let a := fwd[i]!.abs
        if a > peak then peak := a
      let mut eEarly : Float := 0.0
      let mut eLate : Float := 0.0
      for i in [201:900] do eEarly := eEarly + fwd[i]! * fwd[i]!
      for i in [1349:2048] do eLate := eLate + fwd[i]! * fwd[i]!
      let mut bitDiff := 0
      for i in [1:n] do
        if rev[i]! != fwd[2048 - i]! then bitDiff := bitDiff + 1
      IO.println s!"        patch: resonator(3) → reverb(3) → out, lowered from a PatchGraph:"
      IO.println s!"        result   pre-strike |max|={preMax} · peak={peak} · E[early]={eEarly} E[late]={eLate} · rev≡fwd-mirror bitDiff {bitDiff}/{n}"
      if preMax == 0.0 && peak > 1e-6 && eLate < eEarly && bitDiff == 0 then
        passGate "modal-patch" "resonator→reverb→out compiles from a graph: causal, decaying, reverse-scrubs bit-exact"
      else
        failGate "modal-patch" s!"preMax={preMax} peak={peak} eEarly={eEarly} eLate={eLate} bitDiff={bitDiff}"
    | .error e, _ | _, .error e => failGate "modal-patch" s!"render: {firstLine e}"
  | .error e, _ | _, .error e => failGate "modal-patch" s!"build: {firstLine e}"

open Tropical.EmitArrow in
/-- THE MODAL-FOREST M1 gate. A modal mix remains an authored-order forest and
    therefore agrees byte-for-byte with an explicit signal-side sum of the same
    independently realized branches. Different anchors remain independent and
    enter that sum only at their own causal onset. -/
def runModalForestAnchors (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let aModes : Array ModalMode := #[
    ModalMode.hz (lit 220) (lit 4) (lit 7 1),
    ModalMode.hz (lit 440) (lit 6) (lit 3 1)]
  let bModes : Array ModalMode := #[
    ModalMode.hz (lit 330) (lit 5) (lit 5 1),
    ModalMode.hz (lit 660) (lit 8) (lit 2 1)]
  let firstAnchor := lit 200
  let secondAnchor := lit 700
  let source := fun (id : String) (modes : Array ModalMode) (anchor : Sig) =>
    ({ id, node := Node.modalSource modes anchor clockLit none } : PatchNode)
  let sameGraph : PatchGraph := {
    nodes := #[source "a" aModes firstAnchor, source "b" bModes firstAnchor,
      { id := "mix", node := .modalMix #["a", "b"] }]
    output := "mix" }
  let explicitGraph : PatchGraph := {
    nodes := #[source "a" aModes firstAnchor, source "b" bModes firstAnchor,
      { id := "sum", node := .mix #["a", "b"] }]
    output := "sum" }
  let differentGraph : PatchGraph := {
    nodes := #[source "a" aModes firstAnchor, source "b" bModes secondAnchor,
      { id := "mix", node := .modalMix #["a", "b"] }]
    output := "mix" }
  let firstOnlyGraph : PatchGraph := {
    nodes := #[source "a" aModes firstAnchor]
    output := "a" }
  let carrier := fun (name : String) (graph : PatchGraph) => (do
    let term ← lowerGraph graph
    let (out, _) := emitTerm (normalize term) {}
    .ok (buildExprCarrier name out arena) : Except String (Arena × ProgramIdx))
  match buildAndFinish (carrier "mf_same" sameGraph),
        buildAndFinish (carrier "mf_explicit" explicitGraph),
        buildAndFinish (carrier "mf_different" differentGraph),
        buildAndFinish (carrier "mf_first" firstOnlyGraph) with
  | .ok samePlan, .ok explicitPlan, .ok differentPlan, .ok firstPlan =>
    match ← renderPlanSamples samePlan 2048, ← renderPlanSamples explicitPlan 2048,
          ← renderPlanSamples differentPlan 2048, ← renderPlanSamples firstPlan 2048 with
    | .ok same, .ok explicit, .ok different, .ok firstOnly =>
      let n := min (min same.size explicit.size) (min different.size firstOnly.size)
      let mut sameBitDiff := 0
      let mut beforeSecondBitDiff := 0
      let mut afterSecondBitDiff := 0
      for i in [0:n] do
        if same[i]! != explicit[i]! then sameBitDiff := sameBitDiff + 1
        if i ≤ 700 then
          if different[i]! != firstOnly[i]! then
            beforeSecondBitDiff := beforeSecondBitDiff + 1
        else if different[i]! != firstOnly[i]! then
          afterSecondBitDiff := afterSecondBitDiff + 1
      IO.println "        ModalForest modal-mix, authored-order sum and different-anchor causality:"
      IO.println s!"        result   forest≡explicit-sum bitDiff={sameBitDiff}/{n} · before second anchor={beforeSecondBitDiff}/701 · after second anchor differing={afterSecondBitDiff}/{n - 701}"
      if sameBitDiff == 0 && beforeSecondBitDiff == 0 && afterSecondBitDiff > 0 then
        passGate "modal-forest-anchors" "modalMix preserves authored branch order and equals the explicit terminal sum; a later branch retains its own anchor and joins only after that onset"
      else
        failGate "modal-forest-anchors" s!"same={sameBitDiff} beforeSecond={beforeSecondBitDiff} afterSecond={afterSecondBitDiff}"
    | .error e, _, _, _ | _, .error e, _, _ | _, _, .error e, _ | _, _, _, .error e =>
      failGate "modal-forest-anchors" s!"render: {firstLine e}"
  | .error e, _, _, _ | _, .error e, _, _ | _, _, .error e, _ | _, _, _, .error e =>
    failGate "modal-forest-anchors" s!"build: {firstLine e}"

open Tropical.EmitArrow in
/-- THE MODAL-FOREST M2 gate. A compact, test-only score models sixteen authored
    timed islands without importing the demo scene or its grouped-room cache.
    Every fourth island is bloomed, so the fixture exercises a heterogeneous
    forest: modalMix must retain all sixteen branches in authored order and the
    Modal→Sig seam must realize each at its own anchor. An ordinary signal mix of
    the same sixteen modal sources is the independent seam oracle.

    The oracle is checked under forward, velocity-zero hold, reverse, and seek
    clocks. Hold/reverse/seek are also compared directly with the corresponding
    coordinates of the forward render, pinning the closed-form random-access
    reading rather than merely comparing two compiler plans. Finally, removing
    each island is silent until that island's anchor and audible afterwards. -/
def runModalForestTimedIslands (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let islandCount : Nat := 16
  let frameCount : Nat := 2048
  let anchors := (Array.range islandCount).map fun i => 60 + 100 * i
  let ids := (Array.range islandCount).map fun i => s!"timed-island-{i}"
  let bloomB : Float := 0.0004
  let bloomG : Float := 2.0
  let modesFor := fun (i : Nat) => (#[
    ModalMode.hz (lit (Int.ofNat (180 + 19 * i)))
      (lit (Int.ofNat (6 + i % 5)))
      (lit (Int.ofNat (1 + i % 3)) 1)] : Array ModalMode)
  let sourceFor := fun (clk : Clock) (i : Nat) =>
    let modes := modesFor i
    let anchor := lit (Int.ofNat anchors[i]!)
    let bloom? := if i % 4 == 0 then some (bloomB, bloomG) else none
    ({ id := ids[i]!, node := .modalSource modes anchor clk none none bloom? } : PatchNode)
  let sourcesFor := fun (clk : Clock) =>
    (Array.range islandCount).map (sourceFor clk)
  let modalGraphFor := fun (clk : Clock) (inputs : Array String) => ({
    nodes := (sourcesFor clk).push { id := "timed-modal-mix", node := .modalMix inputs }
    output := "timed-modal-mix" } : PatchGraph)
  let signalGraphFor := fun (clk : Clock) => ({
    nodes := (sourcesFor clk).push { id := "timed-signal-mix", node := .mix ids }
    output := "timed-signal-mix" } : PatchGraph)

  -- Inspect the modal value before realization: size, stable order, anchors,
  -- and the plain/bloomed pattern are all compiler facts, not audio inferences.
  let rankOf := fun (id : String) =>
    if id == "timed-modal-mix" then some 1
    else if ids.contains id then some 0
    else none
  let structureOk := match lowerModal (modalGraphFor clockLit ids)
      rankOf "timed-modal-mix" 1 with
    | .error _ => false
    | .ok forest => forest.size == islandCount &&
        (Array.range islandCount).all fun i => match forest[i]? with
          | none => false
          | some branch =>
            let anchorOk := branch.strikeAnchor == lit (Int.ofNat anchors[i]!)
            let bankOk := branch.stages.isEmpty && match branch.source with
              | .bloomed voice B g =>
                i % 4 == 0 && voice.size == 1 && B == bloomB && g == bloomG
              | .plain voice => i % 4 != 0 && voice.size == 1
            anchorOk && bankOk

  let carrier := fun (name : String) (graph : PatchGraph) => (do
    let term ← lowerGraph graph
    let (out, _) := emitTerm (normalize term) {}
    .ok (buildExprCarrier name out arena) : Except String (Arena × ProgramIdx))
  let renderPair : String → Clock → Nat →
      IO (Except String (Array Float × Array Float)) := fun tag clk n => do
    match buildAndFinish (carrier s!"timed_forest_{tag}" (modalGraphFor clk ids)),
          buildAndFinish (carrier s!"timed_oracle_{tag}" (signalGraphFor clk)) with
    | .error e, _ => pure (.error s!"forest {tag}: {e}")
    | _, .error e => pure (.error s!"oracle {tag}: {e}")
    | .ok forestPlan, .ok oraclePlan =>
      match ← renderPlanSamples forestPlan n, ← renderPlanSamples oraclePlan n with
      | .ok forest, .ok oracle => pure (.ok (forest, oracle))
      | .error e, _ => pure (.error s!"forest render {tag}: {e}")
      | _, .error e => pure (.error s!"oracle render {tag}: {e}")

  let q : Int := 4294967296
  let holdAt : Nat := 1234
  let seekBy : Nat := 257
  let holdClock : Clock := litI (Int.ofNat holdAt * q)
  let reverseClock : Clock := sub (litI (Int.ofNat frameCount * q)) clockLit
  let seekClock : Clock := add clockLit (litI (Int.ofNat seekBy * q))
  match ← renderPair "forward" clockLit frameCount,
        ← renderPair "hold" holdClock frameCount,
        ← renderPair "reverse" reverseClock frameCount,
        ← renderPair "seek" seekClock (frameCount - seekBy) with
  | .error e, _, _, _ | _, .error e, _, _ | _, _, .error e, _ | _, _, _, .error e =>
    failGate "modal-forest-timed-islands" (firstLine e)
  | .ok (forward, forwardOracle), .ok (held, heldOracle),
      .ok (reversed, reversedOracle), .ok (sought, soughtOracle) =>
    let forwardOracleDiff := bitDiffCount forward forwardOracle
    let holdOracleDiff := bitDiffCount held heldOracle
    let reverseOracleDiff := bitDiffCount reversed reversedOracle
    let seekOracleDiff := bitDiffCount sought soughtOracle
    let heldValue := forward[holdAt]!
    let mut holdCoordinateDiff := 0
    for i in [0:held.size] do
      if held[i]! != heldValue then holdCoordinateDiff := holdCoordinateDiff + 1
    let mut reverseCoordinateDiff := 0
    for i in [1:min reversed.size forward.size] do
      if reversed[i]! != forward[frameCount - i]! then
        reverseCoordinateDiff := reverseCoordinateDiff + 1
    let mut seekCoordinateDiff := 0
    for i in [0:min sought.size (forward.size - seekBy)] do
      if sought[i]! != forward[i + seekBy]! then
        seekCoordinateDiff := seekCoordinateDiff + 1

    -- Source-removal is the onset oracle. Before the removed island's anchor it
    -- contributed exact zero; afterwards its one-mode ring must be observable.
    let mut removalsBuilt := true
    let mut preOnsetDiff := 0
    let mut survivingOnsets := 0
    for island in [0:islandCount] do
      let kept := ids.filter (· != ids[island]!)
      match buildAndFinish (carrier s!"timed_without_{island}"
          (modalGraphFor clockLit kept)) with
      | .error _ => removalsBuilt := false
      | .ok removedPlan =>
        match ← renderPlanSamples removedPlan frameCount with
        | .error _ => removalsBuilt := false
        | .ok removed =>
          let anchor := anchors[island]!
          for i in [0:min (anchor + 1) (min forward.size removed.size)] do
            if forward[i]! != removed[i]! then preOnsetDiff := preOnsetDiff + 1
          let mut deltaEnergy : Float := 0.0
          for i in [anchor + 1:min forward.size removed.size] do
            let d := forward[i]! - removed[i]!
            deltaEnergy := deltaEnergy + d * d
          if deltaEnergy > 1e-9 then survivingOnsets := survivingOnsets + 1

    IO.println "        ModalForest 16-island heterogeneous timed score (4 bloom, 12 plain):"
    IO.println s!"        structure order/anchors={structureOk} · explicit-sum bitDiff fwd/hold/rev/seek={forwardOracleDiff}/{holdOracleDiff}/{reverseOracleDiff}/{seekOracleDiff}"
    IO.println s!"        random access hold/rev/seek coordinate bitDiff={holdCoordinateDiff}/{reverseCoordinateDiff}/{seekCoordinateDiff} · removal pre-onset diff={preOnsetDiff} · surviving onsets={survivingOnsets}/{islandCount}"
    if structureOk && removalsBuilt && survivingOnsets == islandCount && preOnsetDiff == 0 &&
        forwardOracleDiff == 0 && holdOracleDiff == 0 && reverseOracleDiff == 0 &&
        seekOracleDiff == 0 && holdCoordinateDiff == 0 && reverseCoordinateDiff == 0 &&
        seekCoordinateDiff == 0 then
      passGate "modal-forest-timed-islands" "16 independently timed plain/bloomed branches retain authored order and onsets; forward, hold, reverse, and seek equal the explicit branch sum and its random-access coordinates byte-for-byte"
    else
      failGate "modal-forest-timed-islands"
        s!"structure={structureOk} removalsBuilt={removalsBuilt} onsets={survivingOnsets} pre={preOnsetDiff} oracle={forwardOracleDiff}/{holdOracleDiff}/{reverseOracleDiff}/{seekOracleDiff} coordinates={holdCoordinateDiff}/{reverseCoordinateDiff}/{seekCoordinateDiff}"

private def modalUniverseLegacyStateTags : Array String := #[
  "Register", "StateReg", "StateLoad", "StateStore", "Update", "NextUpdate",
  "Delay", "DelayInit", "StateInit", "Writeback", "SmoothParam"]

private def modalUniverseFunctionHasNoLegacyState
    (f : Tropical.Plan.InstanceFunction) : Bool :=
  let blocks := f.preambleInstructions ++ f.preInputInstructions ++ f.instructions
  blocks.all (fun i => !modalUniverseLegacyStateTags.contains i.tag) &&
    f.children.attach.all fun child =>
      modalUniverseFunctionHasNoLegacyState child.1
termination_by sizeOf f
decreasing_by
  exact Tropical.Plan.InstanceFunction.sizeOf_lt_of_mem_children child.2

/-- Approximate backend work without invoking LLVM.  Counting operands as well
    as instructions makes a packed coefficient table visible: one `Pack`
    instruction still emits one store per operand in the native kernel. -/
private def modalUniversePlanFootprint (plan : Tropical.Plan.FlatPlan) : Nat :=
  plan.instanceFunctions.foldl (fun total function =>
    (Tropical.Ir.Stage0.collectBlocks function).foldl (fun total block =>
      block.foldl (fun total instruction => total + 1 + instruction.args.size)
        total) total) 0

private def modalUniverseSplitFootprint
    (fixture : Tropical.Playground.CompiledPatch) : Except String Nat := do
  let split ← Tropical.Ir.Stage0.hoistTyped fixture.plan fixture.stageBlocks
  let coefficient := split.coeff?.map modalUniversePlanFootprint |>.getD 0
  return modalUniversePlanFootprint split.audio + coefficient

private def modalUniverseRoutedBegins
    (plan : Tropical.Plan.FlatPlan) : Array Tropical.Plan.NInstr :=
  plan.instanceFunctions.foldl (fun out function =>
    (Tropical.Ir.Stage0.collectBlocks function).foldl (fun out block =>
      out ++ block.filter (·.tag == "RoutedSumBegin")) out) #[]

/-- The U1 temporal-law gate. Production-compiled
    `resonator → room A → room B` fixtures are sampled under live room images,
    a constant absolute branch address, and a downstream reverse warp. Each
    room independently rewrites an already-observed output block; restoring
    the same complete image restores its bytes; seek order and reverse traversal
    are exact at equal effective coordinates, while a true held response clock
    remains stable, block-constant, and nonzero. The graph remains modal through
    both rooms, and every emitted fixture excludes the production vocabulary's
    known legacy state/history instructions. -/
def runModalUniverseHistory (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let oneRoomSrc := "{\"nodes\":[" ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4}}," ++
    "{\"id\":\"room_a\",\"kind\":\"reverb\",\"params\":{\"rt60\":0.6},\"in\":{\"in\":[\"res\"]}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"room_a\"]}}],\"out\":\"out\"}"
  let forwardSrc := "{\"nodes\":[" ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4}}," ++
    "{\"id\":\"room_a\",\"kind\":\"reverb\",\"params\":{\"rt60\":0.6},\"in\":{\"in\":[\"res\"]}}," ++
    "{\"id\":\"room_b\",\"kind\":\"reverb\",\"params\":{\"rt60\":0.9},\"in\":{\"in\":[\"room_a\"]}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"room_b\"]}}],\"out\":\"out\"}"
  -- The product UI path: an oscillator supplies the resonator's absolute
  -- address before the same two-room terminal.  Its extra cross-region clock
  -- captures used to push Metal scratch from 24,272 to 24,864 bytes, just over
  -- the 24,576-byte publication ceiling, even though the unaddressed fixture
  -- passed.
  let addressedSrc := "{\"nodes\":[" ++
    "{\"id\":\"osc\",\"kind\":\"source\",\"params\":{\"freq\":0.63,\"morph\":0}}," ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":56.78,\"decay\":15.48},\"in\":{\"addr\":[\"osc\"]}}," ++
    "{\"id\":\"room_a\",\"kind\":\"reverb\",\"params\":{\"rt60\":0.21},\"in\":{\"in\":[\"res\"]}}," ++
    "{\"id\":\"room_b\",\"kind\":\"reverb\",\"params\":{\"rt60\":2},\"in\":{\"in\":[\"room_a\"]}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"room_b\"]}}],\"out\":\"out\"}"
  let holdSrc := "{\"nodes\":[" ++
    "{\"id\":\"hold\",\"kind\":\"knob\",\"params\":{\"value\":0.25}}," ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4},\"in\":{\"addr\":[\"hold\"]}}," ++
    "{\"id\":\"room_a\",\"kind\":\"reverb\",\"params\":{\"rt60\":0.6},\"in\":{\"in\":[\"res\"]}}," ++
    "{\"id\":\"room_b\",\"kind\":\"reverb\",\"params\":{\"rt60\":0.9},\"in\":{\"in\":[\"room_a\"]}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"room_b\"]}}],\"out\":\"out\"}"
  let reverseSrc := "{\"nodes\":[" ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4}}," ++
    "{\"id\":\"room_a\",\"kind\":\"reverb\",\"params\":{\"rt60\":0.6},\"in\":{\"in\":[\"res\"]}}," ++
    "{\"id\":\"room_b\",\"kind\":\"reverb\",\"params\":{\"rt60\":0.9},\"in\":{\"in\":[\"room_a\"]}}," ++
    "{\"id\":\"reverse\",\"kind\":\"reverse\",\"in\":{\"in\":[\"room_b\"]}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"reverse\"]}}],\"out\":\"out\"}"
  let compile := fun source => do
    let json ← Lean.Json.parse source
    Tropical.Playground.compilePlanPure arena resolved json
  match compile forwardSrc, compile holdSrc, compile reverseSrc with
  | .error e, _, _ | _, .error e, _ | _, _, .error e =>
    failGate "modal-universe-history" s!"compile: {firstLine e}"
  | .ok compiled, .ok holdCompiled, .ok reverseCompiled =>
    let baseline ← match compile oneRoomSrc with
      | .error error =>
          return ← failGate "modal-universe-history"
            s!"one-room footprint compile: {firstLine error}"
      | .ok fixture => pure fixture
    let addressed ← match compile addressedSrc with
      | .error error =>
          return ← failGate "modal-universe-history"
            s!"addressed two-room compile: {firstLine error}"
      | .ok fixture => pure fixture
    let (oneRoomFootprint, twoRoomFootprint) ←
      match modalUniverseSplitFootprint baseline,
          modalUniverseSplitFootprint compiled with
      | .ok one, .ok two => pure (one, two)
      | .error error, _ | _, .error error =>
          return ← failGate "modal-universe-history"
            s!"footprint split: {firstLine error}"
    -- Banking makes the one-room terminal unusually small; the second room
    -- legitimately adds four oriented Cauchy families plus its hot DD rows.
    -- Sevenfold still rejects the measured Cartesian form (8.6×) while the
    -- absolute ceiling prevents both sides from drifting upward together.
    let compact := twoRoomFootprint ≤ 7 * oneRoomFootprint &&
      twoRoomFootprint ≤ 350000
    let split ← match Tropical.Ir.Stage0.hoistTyped compiled.plan compiled.stageBlocks with
      | .error error =>
          return ← failGate "modal-universe-history"
            s!"scratch split: {firstLine error}"
      | .ok split => pure split
    let addressedSplit ← match Tropical.Ir.Stage0.hoistTyped
        addressed.plan addressed.stageBlocks with
      | .error error =>
          return ← failGate "modal-universe-history"
            s!"addressed scratch split: {firstLine error}"
      | .ok split => pure split
    let scratch := split.audio.metalThreadgroupScratchBytes
    let addressedScratch := addressedSplit.audio.metalThreadgroupScratchBytes
    let routed := modalUniverseRoutedBegins split.audio
    let sourceItems := routed.foldl (fun total begin =>
      if (begin.routedOutputCount == 90 || begin.routedOutputCount == 104) &&
          begin.routedRoutes.size == begin.loopCount * 22
      then total + begin.loopCount else total) 0
    let differenceItems := routed.foldl (fun total begin =>
      if begin.routedOutputCount == 112 &&
          (begin.loopCount == 22 || begin.loopCount == 23) &&
          begin.routedRoutes.size == begin.loopCount * 16
      then total + begin.loopCount else total) 0
    let physicalItems := routed.foldl (fun total begin =>
      if begin.routedOutputCount == 112 &&
          (begin.loopCount == 26 || begin.loopCount == 27) &&
          begin.routedRoutes.size == begin.loopCount * 16
      then total + begin.loopCount else total) 0
    let confluenceRows := routed.filter fun begin =>
      begin.loopCount ≤ 192 && begin.routedOutputCount == 2 &&
        begin.routedRoutes.size == 2 * begin.loopCount
    let confluenceItems := confluenceRows.foldl (· + ·.loopCount) 0
    let reciprocalCount := 4 * sourceItems + differenceItems + physicalItems
    let declaredStats := Tropical.EmitArrow.Oriented.factoredTerminalStats 6 14
    let factoredShape := sourceItems == 84 && differenceItems == 91 &&
      physicalItems == 105 && reciprocalCount == 532 &&
      declaredStats.totalReciprocals == 532 && confluenceRows.size ≤ 1
    let scratchOk := scratch ≤ 24576 && addressedScratch ≤ 24576
    IO.println s!"        split compile footprint: one-room={oneRoomFootprint} two-room={twoRoomFootprint} compact={compact}"
    IO.println s!"        routed terminal: source={sourceItems}×4 difference={differenceItems} physical={physicalItems} total reciprocals={reciprocalCount} confluence rows={confluenceItems}"
    IO.println s!"        two-room Metal audio scratch: base={scratch}, addressed={addressedScratch}/24576 bytes (75% M1 Pro cap)"
    if !compact || !factoredShape || !scratchOk then
      return ← failGate "modal-universe-history"
        "two-room modal plan misses the factored schedule, compile envelope, or scratch cap"
    let loadRooms := fun (fixture : Tropical.Playground.CompiledPatch) => do
      let rt ← Tropical.Ffi.Runtime.new 128
      Tropical.StagedLoad.loadTyped rt fixture.plan fixture.stageBlocks
      let glideSlots := fun (name : String) => do
        let v0? ← rt.slotIndex? s!"param:{name}#v0"
        let v1? ← rt.slotIndex? s!"param:{name}#v1"
        pure (v0?.bind fun v0 => v1?.map fun v1 => (v0, v1))
      let a? ← glideSlots "room_a.rt60"
      let b? ← glideSlots "room_b.rt60"
      pure (rt, a?, b?)
    let (rtA, aA?, bA?) ← loadRooms compiled
    let (rtB, aB?, bB?) ← loadRooms compiled
    let (rtHold, aHold?, bHold?) ← loadRooms holdCompiled
    let (rtReverse, aReverse?, bReverse?) ← loadRooms reverseCompiled
    match aA?, bA?, aB?, bB?, aHold?, bHold?, aReverse?, bReverse? with
    | some aA, some bA, some aB, some bB, some aHold, some bHold,
        some aReverse, some bReverse =>
      let setImage := fun (rt : Tropical.Ffi.Runtime)
          (a b : UInt32 × UInt32)
          (x y : Float) => do
        rt.setSlot a.1 x
        rt.setSlot a.2 x
        rt.setSlot b.1 y
        rt.setSlot b.2 y
      let renderAt := fun (rt : Tropical.Ffi.Runtime) (t : UInt64) => do
        rt.setSampleIndex t
        rt.process
        rt.outputBytes

      -- Either room must independently affect the selected counterfactual.
      setImage rtA aA bA 0.6 0.9
      let old ← renderAt rtA 8192
      setImage rtA aA bA 6.0 0.9
      let rewrittenA ← renderAt rtA 8192
      setImage rtA aA bA 0.6 0.9
      let restoredA ← renderAt rtA 8192
      setImage rtA aA bA 0.6 9.0
      let rewrittenB ← renderAt rtA 8192
      setImage rtA aA bA 0.6 0.9
      let restoredB ← renderAt rtA 8192

      -- Visit the same frozen universe in opposite coordinate orders.
      setImage rtA aA bA 1.7 3.4
      setImage rtB aB bB 1.7 3.4
      let a0 ← renderAt rtA 2048
      let a1 ← renderAt rtA 6144
      let a2 ← renderAt rtA 12288
      let b2 ← renderAt rtB 12288
      let b1 ← renderAt rtB 6144
      let b0 ← renderAt rtB 2048

      -- A constant absolute address is a genuine velocity-zero response clock.
      setImage rtHold aHold bHold 1.7 3.4
      setImage rtA aA bA 1.7 3.4
      let held0 ← renderAt rtHold 0
      let held1 ← renderAt rtHold 15000
      let heldSamples := decodeF64LE held0
      let heldConstant := match heldSamples[0]? with
        | none => false
        | some first => heldSamples.all (· == first)
      -- The address warp owns only the response-clock child. Live room-control
      -- terms intentionally remain siblings on the ambient clock, so this is
      -- not the same term as a complete master-clock hold. The hold contract is
      -- the observable response law: invariant across visits, constant within
      -- a block, and non-silent. `grouped-room-reference` separately checks the
      -- held response against its independent scalar oracle.
      let heldNonzero := heldSamples.any (· != 0.0)
      let heldStable := held0 == held1
      IO.println s!"        hold witness: stable={heldStable} constant={heldConstant} nonzero={heldNonzero} held0={heldSamples[0]?}"

      -- The public downstream reverse node visits the same effective response
      -- coordinates as a forward block, in the opposite order. Negative runtime
      -- indices are supplied in the Runtime's documented two's-complement image.
      setImage rtA aA bA 1.7 3.4
      setImage rtReverse aReverse bReverse 1.7 3.4
      let reverseBase : Nat := 12288
      let forwardBlock ← renderAt rtA (UInt64.ofNat (reverseBase - 127))
      let reverseBlock ← renderAt rtReverse (0 - UInt64.ofNat reverseBase)
      let forwardSamples := decodeF64LE forwardBlock
      let reverseSamples := decodeF64LE reverseBlock
      let reverseExact := forwardSamples.size == reverseSamples.size &&
        forwardSamples.any (· != 0.0) &&
        (Array.range reverseSamples.size).all fun i =>
          reverseSamples[i]! == forwardSamples[forwardSamples.size - 1 - i]!

      -- Production structure: two deferred rooms and no early Modal→Sig seam.
      let unit : Array Tropical.EmitArrow.ModalMode := #[
        Tropical.EmitArrow.ModalMode.hz (Tropical.EmitArrow.lit 220)
          (Tropical.EmitArrow.lit 4) (Tropical.EmitArrow.lit 1)]
      let graph : Tropical.EmitArrow.PatchGraph := {
        nodes := #[
          { id := "s", node := .modalSource unit (Tropical.EmitArrow.lit 0)
              Tropical.EmitArrow.clockLit none },
          { id := "a", node := .modalReverb "s" unit none },
          { id := "b", node := .modalReverb "a" unit none }]
        output := "b" }
      let ranks := fun id => if id == "s" then some 0 else if id == "a" then some 1
        else if id == "b" then some 2 else none
      let deferred := match Tropical.EmitArrow.lowerModal graph ranks "b" 2 with
        | .ok #[branch] => match branch.source with
          | .plain _ => branch.stages.size == 2
          | _ => false
        | _ => false
      let modalThroughChain := Tropical.EmitArrow.nodeIsModal graph "a" &&
        Tropical.EmitArrow.nodeIsModal graph "b"
      let eachRoomRewrites := old != rewrittenA && old != rewrittenB
      let restoredExact := old == restoredA && old == restoredB
      let seekOrderExact := a0 == b0 && a1 == b1 && a2 == b2
      let holdExact := heldStable && heldConstant && heldNonzero
      let noLegacyState := #[compiled.plan, holdCompiled.plan,
        reverseCompiled.plan].all fun plan =>
          plan.instanceFunctions.all modalUniverseFunctionHasNoLegacyState
      IO.println "        nested room universe: history rewrite + random-access image equality:"
      IO.println s!"        result   each-room-changed={eachRoomRewrites} restored-bytes={restoredExact} order={seekOrderExact} hold={holdExact} reverse={reverseExact} deferred={deferred} modal={modalThroughChain} no-legacy-state={noLegacyState}"
      if eachRoomRewrites && restoredExact && seekOrderExact && holdExact &&
          reverseExact && deferred && modalThroughChain && noLegacyState then
        passGate "modal-universe-history"
          "each room rewrites the nested counterfactual; restore and seek/reverse coordinate laws are exact; the response clock truly holds; both rooms remain modal; no known legacy state/history instruction is emitted"
      else
        failGate "modal-universe-history"
          s!"eachRoom={eachRoomRewrites} restored={restoredExact} order={seekOrderExact} hold={holdExact} reverse={reverseExact} deferred={deferred} modal={modalThroughChain} noLegacyState={noLegacyState}"
    | _, _, _, _, _, _, _, _ =>
      failGate "modal-universe-history"
        s!"nested room RT60 slots missing: forward-a={aA?.isSome}/{bA?.isSome} forward-b={aB?.isSome}/{bB?.isSome} hold={aHold?.isSome}/{bHold?.isSome} reverse={aReverse?.isSome}/{bReverse?.isSome}"

/-- THE MODAL LIVE gate (the payoff). A JSON patch `resonator(freq) → reverb → out`
    compiled through the real `compilePlanPure` — decode → lowerModal → symbolic
    residue → realize → strata → session compile → a JIT-loadable kernel — and its
    pole frequency/decay and the room rt60 resolve to LIVE module slots
    (`param:<id>.<knob>`), settable via `setSlot` with no relower. That the residue
    calculus is symbolic is exactly what keeps the poles live; `symbolic-residue`
    proves the couplings are the right functions of those slots. (This harness
    can't drive a session plan's DAC to audio — that's the Engine/bun path — so the
    audible sweep is left to those; here we prove it compiles and is live.) -/
def runModalLive (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let src := "{\"nodes\":[" ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4}}," ++
    "{\"id\":\"rev\",\"kind\":\"reverb\",\"params\":{\"rt60\":2},\"in\":{\"in\":[\"res\"]}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"rev\"]}}],\"out\":\"out\"}"
  match Lean.Json.parse src with
  | .error e => failGate "modal-live" s!"json parse: {e}"
  | .ok j =>
  match Tropical.Playground.compilePlanPure arena resolved j with
  | .error e => failGate "modal-live" s!"compile: {firstLine e}"
  | .ok compiled =>
    let plan := compiled.plan
    let stageBlocks := compiled.stageBlocks
    match plan.toWire, Tropical.Ir.EmitLlvm.emitKernel plan with
    | .ok _, .ok _ =>
      -- A slot that EXISTS but is never READ is a dead knob — exactly the
      -- reverb-discards-the-voice regression (the pitch knob accepted writes into
      -- a slot no instruction referenced). So presence is only half the gate: run
      -- two identical runtimes a block, move `res.freq` on ONE, and require the
      -- next blocks to diverge THROUGH the reverb. Identical second blocks =
      -- dead knob = FAIL. Under the stage-0 split this also gates the
      -- coefficient re-run: the amps live in the coefficient kernel, so the
      -- divergence only happens if set_slot re-runs it.
      let rt ← Tropical.Ffi.Runtime.new 2048
      Tropical.StagedLoad.loadTyped rt plan stageBlocks
      let rt2 ← Tropical.Ffi.Runtime.new 2048
      Tropical.StagedLoad.loadTyped rt2 plan stageBlocks
      let fIdx? ← rt.slotIndex? "param:res.freq"
      let dPresent := (← rt.slotIndex? "param:res.decay").isSome
      let rtPresent := (← rt.slotIndex? "param:rev.rt60#v0").isSome
      rt.process
      rt2.process
      let b1a := decodeF64LE (← rt.outputBytes)
      let b1b := decodeF64LE (← rt2.outputBytes)
      if let some fIdx := fIdx? then rt.setSlot fIdx 440.0
      rt.process
      rt2.process
      let b2a := decodeF64LE (← rt.outputBytes)
      let b2b := decodeF64LE (← rt2.outputBytes)
      let mut sameB1 := true
      for i in [0:min b1a.size b1b.size] do
        if b1a[i]! != b1b[i]! then sameB1 := false
      let mut dE : Float := 0.0
      let mut e0 : Float := 0.0
      for i in [0:min b2a.size b2b.size] do
        let d := b2a[i]! - b2b[i]!
        dE := dE + d * d
        e0 := e0 + b2b[i]! * b2b[i]!
      let knobRead := dE > 1e-12 && e0 > 1e-12
      IO.println s!"        JSON resonator(freq,decay) → reverb(rt60) → out compiled via compilePlanPure:"
      IO.println s!"        result   JIT-loadable · slots: freq={fIdx?.isSome} decay={dPresent} rt60={rtPresent} · pre-move blocks identical={sameB1} · post-move ΔE/E={dE / (e0 + 1e-300)}"
      if fIdx?.isSome && dPresent && rtPresent && sameB1 && knobRead then
        passGate "modal-live" "modal params are live slots AND the kernel reads them: moving res.freq moves the signal THROUGH the reverb (setSlot, no relower)"
      else
        failGate "modal-live" s!"freq={fIdx?.isSome} decay={dPresent} rt60={rtPresent} sameB1={sameB1} knobRead={knobRead} (ΔE={dE}) — a present-but-unread slot is a dead knob"
    | .error e, _ => failGate "modal-live" s!"toWire: {firstLine e}"
    | _, .error e => failGate "modal-live" s!"emitKernel: {firstLine e}"

/-- THE PATCH-TYPING gate (WS-G): the connection rule is ENFORCED at decode, not
    just documented. A `signal→modal` edge (a `source` wired into a `reverb`'s modal
    inlet) is REJECTED by `compilePlanPure` with a "connection type error" — the
    decode-time `checkEdgeTypes`, not the downstream `lowerModal` fallthrough; the
    dual, a valid modal edge (`resonator → reverb`), still compiles. The whole
    existing corpus (vocab-driven, dead-slot-lint, modal-*) staying green is the
    no-false-rejection half of the gate. -/
def runPatchTyping (arena : Arena) (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let hasSub := fun (s sub : String) => (s.splitOn sub).length != 1
  let bad := "{\"nodes\":[" ++
    "{\"id\":\"osc\",\"kind\":\"source\",\"params\":{\"freq\":220}}," ++
    "{\"id\":\"rev\",\"kind\":\"reverb\",\"params\":{\"rt60\":2},\"in\":{\"in\":[\"osc\"]}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"rev\"]}}],\"out\":\"out\"}"
  let good := "{\"nodes\":[" ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4}}," ++
    "{\"id\":\"rev\",\"kind\":\"reverb\",\"params\":{\"rt60\":2},\"in\":{\"in\":[\"res\"]}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"rev\"]}}],\"out\":\"out\"}"
  match Lean.Json.parse bad, Lean.Json.parse good with
  | .ok jb, .ok jg =>
    let badMsg := match Tropical.Playground.compilePlanPure arena resolved jb with
      | .error e => firstLine e
      | .ok _ => "(compiled — should have been rejected)"
    let badRejected := hasSub badMsg "connection type error"
    let goodOk := match Tropical.Playground.compilePlanPure arena resolved jg with
      | .ok _ => true | .error _ => false
    IO.println s!"        connection typing enforced at decode:"
    IO.println s!"        result   signal→modal rejected={badRejected} · modal→signal compiles={goodOk}"
    IO.println s!"        message  {badMsg}"
    if badRejected && goodOk then
      passGate "patch-typing" "signal→modal is a decode-time type error; a valid modal edge compiles — the served accepts-rule is enforced, not just documented"
    else
      failGate "patch-typing" s!"badRejected={badRejected} goodOk={goodOk} badMsg={badMsg}"
  | _, _ => failGate "patch-typing" "json parse"

/-- THE GONG STRIKE gate (G7): a `gong` node compiled through the real
    `compilePlanPure` (decode → `gongStrikeNodes` → two anchored modal banks under
    per-register pitch-bloom warps → session compile → JIT-loadable kernel) renders a
    causal, ringing strike. Struck at t=5ms: silent before the strike (the causal
    gate), then a struck-resonator tail that decays over the render. Prints the bloom
    metrics (peak index, early/late energy) for the record. -/
def runGongStrike (arena : Arena) (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let src := "{\"nodes\":[" ++
    "{\"id\":\"g\",\"kind\":\"gong\",\"params\":{\"t\":0.005,\"beta\":0.06,\"g\":1.8,\"freq\":110}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"g\"]}}],\"out\":\"out\"}"
  match Lean.Json.parse src with
  | .error e => failGate "gong-strike" s!"json: {e}"
  | .ok j =>
  match Tropical.Playground.compilePlanPure arena resolved j with
  | .error e => failGate "gong-strike" s!"compile: {firstLine e}"
  | .ok compiled =>
    let plan := compiled.plan
    let stageBlocks := compiled.stageBlocks
    let rt ← Tropical.Ffi.Runtime.new 16384
    Tropical.StagedLoad.loadTyped rt plan stageBlocks
    rt.process
    let s := decodeF64LE (← rt.outputBytes)
    let abs := fun (x : Float) => if x < 0.0 then -x else x
    let mut preMax : Float := 0.0
    for i in [0:200] do
      if abs s[i]! > preMax then preMax := abs s[i]!
    let mut peak : Float := 0.0
    let mut peakI : Nat := 0
    for i in [240:s.size] do
      if abs s[i]! > peak then peak := abs s[i]!; peakI := i
    let mut eEarly : Float := 0.0
    let mut eLate : Float := 0.0
    for i in [240:4240] do if i < s.size then eEarly := eEarly + s[i]! * s[i]!
    for i in [12000:16000] do if i < s.size then eLate := eLate + s[i]! * s[i]!
    IO.println s!"        gong strike (t=5ms, β=0.06) via compilePlanPure → JIT:"
    IO.println s!"        result   pre-strike|max|={preMax} · peak={peak}@{peakI} · E[early]={eEarly} E[late]={eLate}"
    if preMax == 0.0 && peak > 1e-3 && eEarly > 1e-9 && eLate < eEarly then
      passGate "gong-strike" s!"gong compiles + renders: causal (silent pre-strike), rings and decays (peak {peak}@{peakI}, E early {eEarly} > late {eLate}) — the struck resonator end to end"
    else
      failGate "gong-strike" s!"preMax={preMax} peak={peak} eEarly={eEarly} eLate={eLate}"

/-- THE GONG LIVE gate (G4). The gong's pitch-bloom depth `beta` is now a live slot
    (`param:<id>.beta`) — the score's baked value its initial, but adjustable with no
    relower. Presence is only half the gate: a slot no instruction READS is a dead
    knob. So run two identical gong runtimes a block, move `g.beta` on one, and
    require the next block to DIVERGE — the bloom warp genuinely reads the slot. -/
def runGongLive (arena : Arena) (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let src := "{\"nodes\":[" ++
    "{\"id\":\"g\",\"kind\":\"gong\",\"params\":{\"t\":0.0,\"beta\":0.06,\"g\":1.8,\"freq\":110}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"g\"]}}],\"out\":\"out\"}"
  match Lean.Json.parse src with
  | .error e => failGate "gong-live" s!"json: {e}"
  | .ok j =>
  match Tropical.Playground.compilePlanPure arena resolved j with
  | .error e => failGate "gong-live" s!"compile: {firstLine e}"
  | .ok compiled =>
    let plan := compiled.plan
    let stageBlocks := compiled.stageBlocks
    let rt ← Tropical.Ffi.Runtime.new 2048
    Tropical.StagedLoad.loadTyped rt plan stageBlocks
    let rt2 ← Tropical.Ffi.Runtime.new 2048
    Tropical.StagedLoad.loadTyped rt2 plan stageBlocks
    let bIdx? ← rt.slotIndex? "param:g.beta"
    rt.process
    rt2.process                                   -- block 1: the strike + early bloom
    let b1a := decodeF64LE (← rt.outputBytes)
    let b1b := decodeF64LE (← rt2.outputBytes)
    if let some bIdx := bIdx? then rt.setSlot bIdx 0.4
    rt.process
    rt2.process                                   -- block 2: one ring now blooms deeper
    let b2a := decodeF64LE (← rt.outputBytes)
    let b2b := decodeF64LE (← rt2.outputBytes)
    let mut sameB1 := true
    for i in [0:min b1a.size b1b.size] do
      if b1a[i]! != b1b[i]! then sameB1 := false
    let mut dE : Float := 0.0
    let mut e0 : Float := 0.0
    for i in [0:min b2a.size b2b.size] do
      let d := b2a[i]! - b2b[i]!
      dE := dE + d * d
      e0 := e0 + b2b[i]! * b2b[i]!
    let read := dE > 1e-12 && e0 > 1e-12
    IO.println s!"        gong β as a live slot (param:g.beta):"
    IO.println s!"        result   slot present={bIdx?.isSome} · pre-move blocks identical={sameB1} · post-move ΔE/E={dE / (e0 + 1e-300)}"
    if bIdx?.isSome && sameB1 && read then
      passGate "gong-live" "gong β is a live slot AND the bloom warp reads it: moving g.beta diverges the ring (setSlot, no relower)"
    else
      failGate "gong-live" s!"present={bIdx?.isSome} sameB1={sameB1} read={read} (ΔE={dE})"

/-- Count instructions matching `pred` across a plan's instance-function tree. -/
private def countInstrsFn (pred : Tropical.Plan.NInstr → Bool) :
    Tropical.Plan.InstanceFunction → Nat
  | f => (f.instructions.filter pred).size
         + f.children.attach.foldl (fun acc c => acc + countInstrsFn pred c.1) 0
termination_by f => sizeOf f
decreasing_by exact Tropical.Plan.InstanceFunction.sizeOf_lt_of_mem_children c.2

private def countInstrs (pred : Tropical.Plan.NInstr → Bool) (p : Tropical.Plan.FlatPlan) : Nat :=
  p.instanceFunctions.foldl (fun acc f => acc + countInstrsFn pred f) 0

/-- Array-dst fills (`Pack`/`SetElement` — coefficient columns). `sessionArray`
    I/O is excluded (still s1). -/
def planArrayFills (p : Tropical.Plan.FlatPlan) : Nat :=
  countInstrs (fun i => match i.dst with | .array _ => true | _ => false) p

/-- Reduce regions (banked mode loops). -/
def planReduces (p : Tropical.Plan.FlatPlan) : Nat :=
  countInstrs (fun i => i.tag == "ReduceBegin") p

/-- `FloatExponent` ops (the one op whose f32/f64 result differs at 0/subnormal) in a
    plan's instruction tree — the gauge-stage gate reads this per kernel. -/
def planFloatExponents (p : Tropical.Plan.FlatPlan) : Nat :=
  countInstrs (fun i => i.tag == "FloatExponent") p

open Tropical.EmitArrow in
/-- THE EXCITATION-GAUGE gate (§5). `normalizePeak g` rescales a bank's residues by
    the self-measured `1/‖H‖^g` (`‖H‖` = the p=8 norm of the bank's own transfer
    function over its pole frequencies). Because the scale is ONE real shared across
    the bank, the render is EXACTLY `scale · (bare render)` — so
    `peak(normalizePeak g) / peak(bare) = ‖H‖^{−g}` (linearity; option E keeps the
    value identical across the `k` the rescale moves). An INDEPENDENT Float oracle
    recomputes `‖H‖₈` from the raw pole/residue Floats — independent of the emitted
    `logSig`/`expSig`, so a mis-scaled adapter shows as ratio ≠ oracle. Asserts, over
    a 2-resonance bank with the damping σ swept (lower σ ⇒ higher Q ⇒ higher ‖H‖):
    (1) g=0 is a no-op (ratio 1 — strike-invariance / unity-DC); (2) g=1 applies
    1/‖H‖ (ratio = ‖H‖⁻¹), and since ‖H‖ GROWS across the sweep, the normalized
    FREQUENCY peak ‖H‖·scale = 1 is level-invariant (unity-peak); (3) g=½ the √Q
    trim (ratio = ‖H‖^{−½}). -/
def runGaugeAdapter (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let f1 := 300.0; let f2 := 520.0; let a1 := 1.0; let a2 := 0.7
  let tp := 6.283185307179586
  let ws := #[tp * f1, tp * f2]; let amps := #[a1, a2]
  -- the damping sweep (Q ↑ as σ ↓); the small-σ tail drives ‖H‖ to ~33 ⇒ S ~ 1e12,
  -- PAST the pre-fix clamp ceiling (litF 1e300 saturated to ≈1.8e7), so this gate
  -- catches a collapsed-clamp regression: with the old ceiling, high-Q leveling failed.
  let sigmas := #[40.0, 10.0, 3.0, 0.5, 0.1, 0.03]
  -- INDEPENDENT oracle: ‖H‖₈ over the two pole freqs, H(iω) = Σⱼ aⱼ/(σ + i(ω−ωⱼ)).
  let normOf := fun (sig : Float) => Id.run do
    let mut S := 0.0
    for wk in ws do
      let mut hr := 0.0; let mut hi := 0.0
      for j in [0:2] do
        let dr := sig; let di := wk - ws[j]!
        let dn := dr * dr + di * di
        hr := hr + amps[j]! * dr / dn
        hi := hi - amps[j]! * di / dn
      let h2 := hr * hr + hi * hi
      S := S + h2 * h2 * h2 * h2
    return Float.exp (Float.log S / 8.0)              -- S^{1/8}
  let mkModes := fun (sig : Float) => #[
    ModalMode.hz (litF f1) (litF sig) (litF a1),
    ModalMode.hz (litF f2) (litF sig) (litF a2)]
  let peakOf := fun (modes : Array ModalMode) => do
    match buildAndFinish (.ok (buildExprCarrier "gauge_probe"
        (modalBankSig modes clockLit (lit 200)) arena)) with
    | .ok p => match ← renderPlanSamples p 2048 with
      | .ok s => do
          let mut mx := 0.0
          for i in [201:s.size] do mx := max mx s[i]!.abs
          pure (some mx)
      | .error e => IO.println s!"        gauge render: {firstLine e}"; pure none
    | .error e => IO.println s!"        gauge build: {firstLine e}"; pure none
  let mut ok := true
  let mut worst := 0.0
  let mut norms : Array Float := #[]
  for sig in sigmas do
    let nrm := normOf sig
    norms := norms.push nrm
    let some pBare ← peakOf (mkModes sig) | return (← failGate "gauge-adapter" "bare render")
    for g in #[(0.0, "0"), (0.5, "½"), (1.0, "1")] do
      let some pG ← peakOf (normalizePeak (litF g.1) (mkModes sig))
        | return (← failGate "gauge-adapter" s!"g={g.2} render")
      let ratio := pG / pBare
      let oracle := Float.exp (Float.log nrm * (-g.1))   -- ‖H‖^{−g}
      let rel := (ratio - oracle).abs / (max oracle 1e-9)
      if rel > worst then worst := rel
      if rel > 5e-3 then
        IO.println s!"        GAUGE MISMATCH σ={sig} g={g.2}: render ratio {ratio} vs oracle norm^(-g) {oracle} (rel {rel})"
        ok := false
  -- level-invariance: ‖H‖ must GROW monotonically across the sweep (a non-vacuity
  -- guard — the exact invariance is the ratio ≡ ‖H‖^{−g} assertion above, which
  -- proves normalizePeak divides by precisely the self-measured norm); the g=1 case
  -- then re-levels a norm that spans ~0.025 → ~33 (three decades incl. the ceiling).
  let grows := norms.size == sigmas.size &&
    (Array.range (norms.size - 1)).all (fun i => norms[i]! < norms[i+1]!)
  -- the floor path (S→0): an all-zero-residue bank has S = 0, clamped to 1e-30 (not
  -- to `litF 1e-30`'s collapsed 0, which would send logSig(0) ≈ −712 → scale ≈ e⁸⁸);
  -- the render must stay silent (0·anything = 0), never amplified numerical dust.
  let silentBank := #[ModalMode.hz (litF f1) (litF 3.0) (lit 0),
                       ModalMode.hz (litF f2) (litF 3.0) (lit 0)]
  let some pSilent ← peakOf (normalizePeak (litF 1.0) silentBank)
    | return (← failGate "gauge-adapter" "silent render")
  let silentOk := pSilent < 1e-9
  if !silentOk then
    IO.println s!"        GAUGE: silent bank amplified to {pSilent} at g=1 — the S→0 floor is broken"
    ok := false
  IO.println s!"        normalizePeak: render ratio ≡ norm^(-g) over σ∈{sigmas}, g∈0/½/1:"
  IO.println s!"        result   worst rel {worst} · ‖H‖ {norms} (grows across the sweep: {grows})"
  if ok && grows then
    passGate "gauge-adapter" s!"the self-measured excitation gauge: g=0 no-op, g=1 unity-peak (÷‖H‖), g=½ √Q trim — render ≡ norm^(-g) to {worst} rel, ‖H‖ grows {norms}"
  else
    failGate "gauge-adapter" s!"ok={ok} grows={grows} worst={worst}"

open Tropical.EmitArrow in
/-- THE EMITTED-LGAMMA gate (WS-LP foundation). `lgammaE` — the emitted complex
    log-gamma (via `atan2E`/`logSig`), the live twin of build-time `lgammaB` — must
    match `lgammaB` over a `Re z ∈ [−5, 5]` sweep (crossing the reflection boundary
    Re = ½) at both `Im z` signs (`+1`, `−2.5`, exercising the dominant-half select).
    Independent oracle: the build-time Lanczos itself, rendered vs computed. Relative
    error (lgamma grows near the negative reals). This is the primitive under WS-LP's
    live Γ★ bridge; without it the crossing stays baked-pole. -/
def runLgammaEmit (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let sinkGain : Float := Tropical.Plan.defaultSinkGain.toFloat
  let step := 10.0 / 2048.0
  let reRamp := sub (mul (toFloatE (rshift clockLit (lit 32))) (litF step)) (lit 5)
  let mut worst : Float := 0.0
  let mut worstAt : String := ""
  let mut ok := true
  for imVal in #[1.0, -2.5] do
    let lg := lgammaE (reRamp, litF imVal)
    match buildAndFinish (.ok (buildExprCarrier "lg_re" lg.1 arena)),
          buildAndFinish (.ok (buildExprCarrier "lg_im" lg.2 arena)) with
    | .ok pRe, .ok pIm =>
      match ← renderPlanSamples pRe 2048, ← renderPlanSamples pIm 2048 with
      | .ok sRe, .ok sIm =>
        for i in [0:min sRe.size sIm.size] do
          let re := -5.0 + i.toFloat * step
          let ref := lgammaB ⟨re, imVal⟩
          let scale := max (ref.re.abs + ref.im.abs) 1.0
          let e := (max (sRe[i]! / sinkGain - ref.re).abs (sIm[i]! / sinkGain - ref.im).abs) / scale
          if e > worst then worst := e; worstAt := s!"z=({re},{imVal})"
      | .error e, _ | _, .error e => IO.println s!"        lgammaE render: {firstLine e}"; ok := false
    | .error e, _ | _, .error e => IO.println s!"        lgammaE build: {firstLine e}"; ok := false
  IO.println s!"        emitted lgammaE vs build-time lgammaB, Re∈[−5,5] (crosses ½), Im = 1 and −2.5:"
  IO.println s!"        result   worst relative error {worst}  (at {worstAt})"
  if ok && worst < 1e-4 then
    passGate "lgamma-emit" s!"emitted complex lgamma ≡ build-time Lanczos to {worst} rel across the reflection boundary and both Im signs — WS-LP's live Γ★ bridge is sound"
  else
    failGate "lgamma-emit" s!"ok={ok} worst={worst} at {worstAt}"

open Tropical.EmitArrow in
/-- THE OPTION-E k-INVARIANCE gate. Option E's whole correctness rests on the landed
    value being INVARIANT under the per-bank exponent `k` (the `·2^(28−k)` land and the
    `>>(28−k)` shift cancel; only the quantization LSB moves). Every equivalence gate
    only ever exercised `k = 0` (all shipped configs land there), so the k≠0 branch's
    value-preservation was inferred, never witnessed. This pins it directly: a single
    cosine mode at a SWEEP of amplitudes that crosses the `k` boundaries (|A|=32→k1,
    64→k2, 128→k3), rendered through the real datapath — the peak must stay LINEAR in
    the amplitude (peak/amp constant), i.e. NO jump as `k` steps. Consequence (why the
    handoff's earlier "glided-bank floatExponent is a Metal risk" note is retracted): a
    backend that computes a different `k` (f32 maxAbs vs f64 near a power-of-2 boundary)
    lands the SAME value to within one quantization LSB — not a 2× divergence, and
    wasm≡jit (both f64 ⇒ same k) is bit-identical regardless. -/
def runKInvariance (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let amps := #[10.0, 30.0, 40.0, 60.0, 70.0, 130.0]   -- k = 0,0,1,1,2,3 (crosses 32/64/128)
  let peakOf := fun (amp : Float) => do
    let modes := #[ModalMode.hz (litF 300.0) (litF 5.0) (litF amp)]
    match buildAndFinish (.ok (buildExprCarrier "kinv" (modalBankSig modes clockLit (lit 200)) arena)) with
    | .ok p => match ← renderPlanSamples p 2048 with
      | .ok s => do
          let mut mx := 0.0
          for i in [201:s.size] do mx := max mx s[i]!.abs
          pure (some mx)
      | .error e => IO.println s!"        kinv render: {firstLine e}"; pure none
    | .error e => IO.println s!"        kinv build: {firstLine e}"; pure none
  let mut ratios : Array Float := #[]
  for amp in amps do
    let some pk ← peakOf amp | return (← failGate "k-invariance" s!"render amp={amp}")
    ratios := ratios.push (pk / amp)
  -- peak/amp must be one constant across the sweep — no jump at a k boundary
  let mean := ratios.foldl (· + ·) 0.0 / ratios.size.toFloat
  let worst := ratios.foldl (fun w r => max w ((r - mean).abs / mean)) 0.0
  IO.println s!"        single cosine mode, |A| sweep {amps} (k = 0,0,1,1,2,3):"
  IO.println s!"        result   peak/amp {ratios} · worst deviation from constant {worst}"
  if worst < 1e-5 then
    passGate "k-invariance" s!"the landed value is k-invariant: peak/amp constant to {worst} across the k=0→3 boundaries (32/64/128) — the ·2^(28−k)/>>(28−k) cancel, a k-flip moves only the quantization LSB"
  else
    failGate "k-invariance" s!"peak/amp NOT constant (worst {worst}) — a k boundary jumped the value; option E's k-invariance is broken"
