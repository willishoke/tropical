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

/-- THE NESTED-BANKS gate (WS5): a fold-of-folds through the FULL front door
    (raise → elaborate → strata → emit) — Σᵢ aᵢ·Σⱼ(bⱼ + aᵢ), the Cauchy shape:
    the inner body reads the OUTER element, so the outer-index read
    `index(col_a, loopIdx outer)` appears BOTH inside the inner region (in the
    contribution) and outside it (the aᵢ· factor) as ONE hash-consed DAG node —
    exactly what makes unique binder ids load-bearing (de Bruijn spellings
    would fork it). Asserts: exactly 2 reduce regions, properly NESTED in the
    stream (RB RB RE RE); byte-equal to the hand-unrolled reference over 2048
    samples; plan FLAT in BOTH trip counts ((4,4) → (16,16), Δ ≤ 2 — HARD).
    Under TROPICAL_BANKS_UNROLL the whole ladder reverts (0 regions) and still
    matches byte-equal. -/
def runBanksNested (_arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let mulJ (a b : Lean.Json) : Lean.Json :=
    Lean.Json.mkObj [("op", Lean.Json.str "mul"), ("args", Lean.Json.arr #[a, b])]
  -- inner: fold over [b₀..b_{k2-1}] of acc2 + (f + aElem) — aElem is the
  -- OUTER element expression (binding "e" on the fold path; the literal aᵢ
  -- on the unrolled reference path).
  let innerFold (aElem : Lean.Json) (k2 : Nat) : Lean.Json :=
    Lean.Json.mkObj [("op", Lean.Json.str "fold"),
      ("over", Lean.Json.arr ((Array.range k2).map cgB)),
      ("init", cgJn 0 0), ("acc_var", Lean.Json.str "acc2"), ("elem_var", Lean.Json.str "f"),
      ("body", cgAdd (cgBinding "acc2") (cgAdd (cgBinding "f") aElem))]
  let foldExpr (k1 k2 : Nat) : Lean.Json :=
    cgFold (Lean.Json.arr ((Array.range k1).map cgA))
      (cgAdd (cgBinding "acc") (mulJ (cgBinding "e") (innerFold (cgBinding "e") k2)))
  -- The fold's own nesting order, hand-unrolled: ((0 + a₀·S₀) + a₁·S₁) + …
  -- with Sᵢ = ((0 + (b₀+aᵢ)) + (b₁+aᵢ)) + …
  let unrollExpr (k1 k2 : Nat) : Lean.Json :=
    (Array.range k1).foldl (fun acc i =>
      let s := (Array.range k2).foldl (fun a j => cgAdd a (cgAdd (cgB j) (cgA i))) (cgJn 0 0)
      cgAdd acc (mulJ (cgA i) s)) (cgJn 0 0)
  match ← compileFoldProbe (foldExpr 4 4) "nested-f4", ← compileFoldProbe (unrollExpr 4 4) "nested-u4" with
  | .ok fp, .ok up =>
    match ← renderPlanSamples fp 2048, ← renderPlanSamples up 2048 with
    | .ok fS, .ok uS =>
      let n := min fS.size uS.size
      let bitDiff := bitDiffCount fS uS
      let nonzero := fS.any (· != 0.0)
      match ← compileFoldProbe (foldExpr 16 16) "nested-f16" with
      | .ok f16 =>
        let d := planInstrCount f16 - planInstrCount fp
        let regions := planTagCount "ReduceBegin" fp
        let delims := reduceDelims fp
        let nested := delims == #["ReduceBegin", "ReduceBegin", "ReduceEnd", "ReduceEnd"]
        let looping := Tropical.Ir.Strata.ArrayLower.banksLoopEnabled
        IO.println s!"        fold-of-folds Σᵢ aᵢ·Σⱼ(bⱼ+aᵢ) (K=4,4), inner body reads the OUTER element (loop-everything={looping}):"
        IO.println s!"        result   bit-differing {bitDiff}/{n} vs hand-unrolled · nonzero={nonzero} · regions={regions} · delims={delims}"
        IO.println s!"        payoff   plan-instrs: fold(4,4)={planInstrCount fp} unrolled(4,4)={planInstrCount up} fold(16,16)={planInstrCount f16} (Δ={d})"
        if looping then
          warnBenchConst "banks-nested" "nested-fold plan-instrs (any K)" 14 (planInstrCount fp)
          -- The TYPED Stage0 split (the production golden render path) must
          -- traverse the nested delimiters — depth-counted `findRegionEnd`,
          -- outermost-only `tryRegion` — and still render byte-equal.
          let stagedOk ← do
            match ← compilePatchStaged "/tmp/tropicaltest-columnize-nested-f4.json",
                  ← compilePatchStaged "/tmp/tropicaltest-columnize-nested-u4.json" with
            | .ok (pf, bf), .ok (pu, bu) =>
              let sf ← renderTypedBytes pf bf
              let su ← renderTypedBytes pu bu
              pure (sf == su)
            | .error e, _ | _, .error e =>
              IO.println s!"        staged   compile failed: {firstLine e}"; pure false
          IO.println s!"        staged   typed-split render byte-equal to unroll: {stagedOk}"
          if bitDiff == 0 && nonzero && regions == 2 && nested && d ≤ 2 && stagedOk then
            passGate "banks-nested" s!"fold-of-folds banks as NESTED regions (RB RB RE RE), byte-equal to unroll (plain + typed split), plan FLAT in both counts (Δ={d} ≤ 2, (4,4)→(16,16))"
          else
            failGate "banks-nested" s!"bitDiff={bitDiff} nonzero={nonzero} regions={regions} nested={nested} Δ={d} stagedOk={stagedOk}"
        else
          if bitDiff == 0 && nonzero && regions == 0 && d > 2 then
            passGate "banks-nested" s!"escape hatch reverts: the whole ladder unrolls (0 regions, Δ={d} grows), byte-equal"
          else
            failGate "banks-nested" s!"(unroll mode) bitDiff={bitDiff} nonzero={nonzero} regions={regions} Δ={d}"
      | .error e =>
        failGate "banks-nested" s!"scaling compile: {firstLine e}"
    | .error e, _ | _, .error e => failGate "banks-nested" s!"render: {firstLine e}"
  | .error e, _ | _, .error e => failGate "banks-nested" s!"compile: {firstLine e}"

/-- THE NESTED-BANKS MSL gate: EmitMsl on the nested plan emits two reduce
    `for` loops, the second opening strictly INSIDE the first (text-level
    depth scan: reduce-for lines push, brace-only lines pop; the probe's body
    is scalar-only, so no other construct emits a bare closing brace before
    the loops close). Under TROPICAL_BANKS_UNROLL the kernel has no reduce
    loop at all. -/
def runBanksNestedMsl (_arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let mulJ (a b : Lean.Json) : Lean.Json :=
    Lean.Json.mkObj [("op", Lean.Json.str "mul"), ("args", Lean.Json.arr #[a, b])]
  let innerFold (aElem : Lean.Json) (k2 : Nat) : Lean.Json :=
    Lean.Json.mkObj [("op", Lean.Json.str "fold"),
      ("over", Lean.Json.arr ((Array.range k2).map cgB)),
      ("init", cgJn 0 0), ("acc_var", Lean.Json.str "acc2"), ("elem_var", Lean.Json.str "f"),
      ("body", cgAdd (cgBinding "acc2") (cgAdd (cgBinding "f") aElem))]
  let foldExpr : Lean.Json :=
    cgFold (Lean.Json.arr ((Array.range 4).map cgA))
      (cgAdd (cgBinding "acc") (mulJ (cgBinding "e") (innerFold (cgBinding "e") 4)))
  match ← compileFoldProbe foldExpr "nested-msl" with
  | .error e => failGate "banks-nested-msl" s!"compile: {firstLine e}"
  | .ok fp =>
    match Tropical.Ir.EmitMsl.emitKernel fp with
    | .error e => failGate "banks-nested-msl" s!"EmitMsl: {firstLine e}"
    | .ok msl =>
      let looping := Tropical.Ir.Strata.ArrayLower.banksLoopEnabled
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
      if looping then
        if forDepths == #[0, 1] then
          passGate "banks-nested-msl" s!"two reduce for-loops, the inner strictly inside the outer (depths {forDepths})"
        else
          failGate "banks-nested-msl" s!"expected reduce-for depths #[0, 1], got {forDepths}"
      else
        if forDepths.isEmpty then
          passGate "banks-nested-msl" "escape hatch: no reduce loop in the kernel"
        else
          failGate "banks-nested-msl" s!"(unroll mode) expected no reduce loops, got depths {forDepths}"

/-- THE MODAL DEGREE gate. A degree-1 mode `amp·d·e^{−σd}` (a repeated pole — the
    resonance "swell") rendered by the engine must match `sinkGain·d·e^{−σd}` to
    minimax tolerance (an absolute oracle, validating the new `d^deg` factor), and
    must RISE to a peak at d≈1/σ before decaying — the τ·e signature a simple pole
    (monotone decay) cannot produce. -/
def runModalDegree (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let modes : Array Tropical.EmitArrow.ModalMode := #[
    { sigma := Tropical.EmitArrow.lit 25, omega := Tropical.EmitArrow.lit 0,
      cre := Tropical.EmitArrow.lit 1, deg := 1 }]
  let anchor := Tropical.EmitArrow.lit 200
  match buildAndFinish (.ok (Tropical.EmitArrow.buildModalBankArrow "modal_deg" modes anchor arena)) with
  | .ok p =>
    match ← renderPlanSamples p 8192 with
    | .ok s =>
      let sinkGain : Float := 0.05
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
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
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
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
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
def runResidueMoments (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
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
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
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
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
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
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
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
def runResidueDegenerate (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
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
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
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
      -- (design/fixed-carrier.md) × the 0.05 sink gain, 2× slack.
      let bound := 10.0 * 3.7252903e-9 * 0.05 * 2.0
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
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
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
      -- (nU + nC) independent Q4.28 weight landings × the 0.05 sink gain, with
      -- 2× slack for the poly/final-shift ulps riding along.
      let bound := (nU + nC).toFloat * 3.7252903e-9 * 0.05 * 2.0
      IO.println s!"        collected (m+n={nC}) vs uncollected (m+m·n={nU}), voice(3)⋙reverb(4):"
      IO.println s!"        result   max|Δ|={maxAbs * 1e9}e-9  ·  quantum bound={bound * 1e9}e-9"
      if maxAbs < bound && energy > 1e-9 && nC == 7 && nU == 15 then
        passGate "residue-collected" s!"pole-union bank ≡ per-pair bank within the Q4.28 landing quantum (max|Δ| {maxAbs * 1e9}e-9 < {bound * 1e9}e-9); {nU}→{nC} modes — fusion affordable as the default"
      else
        failGate "residue-collected" s!"max|Δ|={maxAbs * 1e9}e-9 (bound {bound * 1e9}e-9) energy={energy} nC={nC} (want 7) nU={nU} (want 15)"
    | .error e, _ | _, .error e => failGate "residue-collected" s!"render: {firstLine e}"
  | .error e, _ | _, .error e => failGate "residue-collected" s!"build: {firstLine e}"

/-- THE INTEGRATE gate. `integrateBank` — the antiderivative as a build-time pole
    move (`a ↦ a/μ` + a `μ=0` DC atom fixing `∫|₀=0`) — validated three ways.
    (A) the FLOAT oracle is EXACT by the jet: each integrated mode satisfies
    `μ·(a/μ)=a` (its derivative recovers the source, 1e-12) and `Σ aₒᵤₜ=0` (the DC
    atom zeroes the onset), with `n → n+1` modes. (B) the SYMBOLIC `integrateBank`
    on the same literal bank folds to that oracle, rendering equal within the Q4.28
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
  -- (A) Float oracle: a ↦ a/μ, plus DC atom −Σ a/μ. Independent of the Sig algebra.
  let integC : Array (Cplx × Cplx) := srcF.map (fun pa => (pa.1, pa.2.div pa.1))
  let sumA := integC.foldl (fun s pa => s.add pa.2) (⟨0.0, 0.0⟩ : Cplx)
  let oracleF : Array (Cplx × Cplx) := integC.push (⟨0.0, 0.0⟩, sumA.neg)
  let mut jetErr : Float := 0.0
  for i in [0:srcF.size] do
    let recovered := (oracleF[i]!.1).mul (oracleF[i]!.2)         -- μ · (a/μ)
    let e := (recovered.add (srcF[i]!.2).neg).abs
    if e > jetErr then jetErr := e
  let onsetA := (oracleF.foldl (fun s pa => s.add pa.2) (⟨0.0, 0.0⟩ : Cplx)).abs
  let lastP := oracleF[oracleF.size - 1]!.1
  let structOk := oracleF.size == srcF.size + 1 && lastP.re == 0.0 && lastP.im == 0.0
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
      let bound := oracleF.size.toFloat * 3.7252903e-9 * 0.05 * 4.0
      -- (C) rendered integral ≡ cumulative trapezoid of the source render (D3).
      -- The 0.05 sink gain scales source and integral alike, and the trapezoid is
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
      if rel < 1e-3 && hEnergy > 1e-9 then
        passGate "modal-heterodyne" s!"heterodyne twist (Re·cosθ−Im·sinθ) ≡ the fused Bessel bank (rel {rel}) — O(1)-in-sidebands FM, still a modal object (D6)"
      else
        failGate "modal-heterodyne" s!"rel={rel} hEnergy={hEnergy}"
    | .error e, _ | _, .error e => failGate "modal-heterodyne" s!"render: {firstLine e}"
  | .error e, _ | _, .error e => failGate "modal-heterodyne" s!"build: {firstLine e}"

end ResidueGates

open Tropical.EmitArrow in
/-- THE MODAL PATCH gate (the session surface). A modal-island `PatchGraph`
    (`resonator → reverb → out`) lowered through `lowerModal` (residue in pole
    space) and realized at its boundary must render a real, causal, decaying
    signal — and, read through a reversing master clock, play the tail backward
    bit-for-bit. This is the whole seam end to end: a patch graph, not a builder. -/
def runModalPatch (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
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
  | .ok (plan, _, stageBlocks) =>
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
      let rtPresent := (← rt.slotIndex? "param:rev.rt60").isSome
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

/-- Count instructions matching `pred` across a plan's instance-function tree. -/
private partial def countInstrsFn (pred : Tropical.Plan.NInstr → Bool) :
    Tropical.Plan.InstanceFunction → Nat
  | f => (f.instructions.filter pred).size
         + f.children.foldl (fun acc c => acc + countInstrsFn pred c) 0

private def countInstrs (pred : Tropical.Plan.NInstr → Bool) (p : Tropical.Plan.FlatPlan) : Nat :=
  p.instanceFunctions.foldl (fun acc f => acc + countInstrsFn pred f) 0

/-- Array-dst fills (`Pack`/`SetElement` — coefficient columns). `sessionArray`
    I/O is excluded (still s1). -/
def planArrayFills (p : Tropical.Plan.FlatPlan) : Nat :=
  countInstrs (fun i => match i.dst with | .array _ => true | _ => false) p

/-- Reduce regions (banked mode loops). -/
def planReduces (p : Tropical.Plan.FlatPlan) : Nat :=
  countInstrs (fun i => i.tag == "ReduceBegin") p
