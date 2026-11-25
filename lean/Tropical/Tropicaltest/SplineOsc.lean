import Tropical.Tropicaltest.SplineSpectrum
import Tropical.EmitArrow.Stdlib

/-!
# Spline-oscillator gates: one spline, alias-bounded at every pitch

`lean/Tropical/EmitArrow/SplineOsc.lean` is the module these gates back;
`design/spline-osc.md` is the note. K = 8, both bases, the registered budget
(`splineOscBudgetDb`, 60 dB).

* **spline-osc-band** — the closed-form class power `r_0 + 2Σ r_k cos(2πkm/K)`
  against a brute-force `Σ_n φ̂(n/K)²` of the closed-form transforms; the
  derived band; refusals (no band for the capacity, 0 dB, past the resolved
  depth, a moving fundamental).
* **spline-osc-oracle** — the JIT render against an independent f64 reference
  of the stated mix `(1 − w)·cubic + w·bank`, at fundamentals below, inside
  and above the band, negative and zero, with a phase offset, at τ near 0,
  2⁴⁰ and 2⁴⁷ (the bank's sine rail and the cubic read one phase there).
* **spline-osc-budget** — the measured error against the ideal band-limited
  curve (every harmonic strictly below Nyquist), relative to the in-band
  power, at a sweep of fundamentals from 20 Hz to 20 kHz, plus each
  worst-case control set (all power in one harmonic class) across the band.
* **spline-osc-continuity** — the fundamental stepped across the band, each
  step a new call whose phase offset is re-anchored at the step (the
  `anchor` discipline's arithmetic): every step is within the derived bound.
* **spline-osc-trip** — the plan's family region carries a live trip count
  over the derived capacity, and the kernel's trip value is 0 below the band
  and the harmonic count below Nyquist above it.
* **spline-osc-live** — a `SplineOscCatmullRom8` patch with `param:` controls:
  a `set_param` reaches the next buffer, no recompile.
-/

namespace Tropical.Tropicaltest.SplineOsc

open Tropical
open Tropical.Ir
open Tropical.EmitArrow
open Tropical.Tropicaltest.Spectral (compileMulti renderAt range maxAbsDiff peakOf sci fmt)
open Tropical.Tropicaltest.Spline (controlsOf controlFloat)
open Tropical.Tropicaltest.SplineSpectrum (piF closedTransform basisLabel curveAt harmonicOf
  freqOfInc smoothF railEps tailBound dbOf)

def rate : Nat := 48000

def bases : Array SplineBasis := #[catmullRomBasis, uniformBSplineBasis]

def bandOf (basis : SplineBasis) : Except String SplineOscBand :=
  splineOscBand basis 8 splineOscBudgetDb

/-- The crossfade coordinate and weight, f64. -/
def weightOf (band : SplineOscBand) (f0 : Float) : Float × Float :=
  let nyq := rate.toFloat * 0.5
  let cap := band.capacity.toFloat
  let edge := band.edge.toFloat
  let t := (f0.abs * cap / nyq - 1.0) * edge / (cap - edge)
  (t, smoothF t)

/-- The harmonics strictly below Nyquist. -/
def belowNyquist (f0 : Float) : Nat :=
  if f0 == 0.0 then 0 else
  let x := rate.toFloat * 0.5 / f0.abs
  (x.ceil - 1.0).toUInt64.toNat

/-- The bank's trip count: the harmonics below Nyquist while `w > 0`. -/
def tripOf (band : SplineOscBand) (f0 : Float) : Nat :=
  if (weightOf band f0).1 > 0.0 then min band.capacity (belowNyquist f0) else 0

/-- A configuration: basis, controls, a fundamental on the rail (signed
    increment) and a phase offset (an integer of the Q0.32 rail). -/
structure Cfg where
  basis : SplineBasis
  cs : Array Float
  inc : Int
  off : Nat := 0
deriving Inhabited

/-- τ = 2⁴⁰ and 2⁴⁷ frames. -/
def tau40 : Nat := 1099511627776
def tau47 : Nat := 140737488355328

def Cfg.f0 (c : Cfg) : Float :=
  let f := freqOfInc c.inc.natAbs rate
  if c.inc < 0 then -f else f

def Cfg.harmonics (c : Cfg) (count : Nat) : Array (Float × Float) :=
  (Array.range count).map fun i => harmonicOf c.basis c.cs (i + 1)

def phaseAt (c : Cfg) (n : Nat) : Nat :=
  ((c.inc * (n : Int) + (c.off : Int)) % 4294967296).toNat

/-- `F_0 + Σ_{p ≤ trip} 2·Re(F_p e^{iθ_p})` at the phase `ph`. -/
def bankRef (cs : Array Float) (hs : Array (Float × Float)) (trip ph : Nat) : Float := Id.run do
  let mut v := cs.foldl (· + ·) 0.0 / cs.size.toFloat
  for p in [1:trip + 1] do
    let theta := 2.0 * piF * ((p * ph) % 4294967296).toFloat / 4294967296.0
    let (fr, fi) := hs[p - 1]!
    v := v + 2.0 * (fr * Float.cos theta - fi * Float.sin theta)
  return v

/-- The stated mix at sample `n`. -/
def oscRef (band : SplineOscBand) (c : Cfg) (hs : Array (Float × Float)) (n : Nat) : Float :=
  let ph := phaseAt c n
  let (_, w) := weightOf band c.f0
  let cubic := curveAt c.basis c.cs (ph.toFloat / 4294967296.0)
  let bank := bankRef c.cs hs (tripOf band c.f0) ph
  (1.0 - w) * cubic + w * bank

def oscSig (band : SplineOscBand) (c : Cfg) : BuildM SplineOscParts := do
  let f0 ← litExact c.f0
  let phase ← litExact (c.off.toFloat / 4294967296.0)
  let ctl ← c.cs.mapM litExact
  splineOscSig c.basis band ctl f0 (← tau) (some phase)

def incOf (f : Float) : Int :=
  let i := (f.abs * 4294967296.0 / rate.toFloat).floor.toUInt64.toNat
  if f < 0.0 then -(i : Int) else i

def lcgControls (seed : Nat) : Array Float := (controlsOf seed 8).map controlFloat

/-- `Σ_p 2|F_p|` over `p ≤ trip`. -/
def massOf (hs : Array (Float × Float)) (trip : Nat) : Float :=
  (hs.extract 0 trip).foldl (fun a (x, y) => a + 2.0 * Float.sqrt (x * x + y * y)) 0.0

-- ── spline-osc-band ──────────────────────────────────────────────────────

def runBand : IO Bool := do
  let label := "spline-osc-band"
  let mut notes := #[]
  let mut worstRel := 0.0
  for b in bases do
    for k in #[2, 5, 8, 16] do
      for m in [0:k / 2 + 1] do
        -- brute force: Σ_{n ≥ 1, n ≡ ±m} φ̂(n/K)², closed-form transform
        let mut brute := if m == 0 then 1.0 else 0.0
        if m != 0 then
          for n in [1:400001] do
            let r := n % k
            if r == m || r == (k - m) % k then
              let v := closedTransform b (n.toFloat / k.toFloat)
              brute := brute + v * v
        let closed := b.classPower k m
        worstRel := max worstRel ((closed - brute).abs / brute)
  notes := notes.push s!"class power vs brute force: worst relative {sci worstRel}"
  let mut ok := decide (worstRel < 1e-9)
  for b in bases do
    match bandOf b with
    | .error e => return ← failGate label s!"{basisLabel b}: {e}"
    | .ok band =>
      notes := notes.push s!"{basisLabel b} K = 8, {splineOscBudgetDb} dB: edge M = {band.edge} (worst alias there {fmt (dbOf band.worstAlias) 1} dB), {band.capacity} partials, band {fmt (24000.0 / band.capacity.toFloat) 1}–{fmt (24000.0 / band.edge.toFloat) 1} Hz at 48 kHz"
      -- the edge is the least M meeting the budget
      let ratios := b.aliasRatios 8 band.edge
      if ratios[band.edge - 1]! > band.budget || (band.edge > 1 && ratios[band.edge - 2]! ≤ band.budget) then
        ok := false
  let refusals : Array (String × Except String SplineOscBand × String) := #[
    ("capacity at the edge", splineOscBand catmullRomBasis 8 60 (some 22), "no band meets"),
    ("0 dB", splineOscBand catmullRomBasis 8 0, "bounds nothing"),
    ("130 dB", splineOscBand catmullRomBasis 8 130, "past what the closed form resolves"),
    ("no controls", splineOscBand catmullRomBasis 0 60, "no control values")]
  for (what, r, needle) in refusals do
    match r with
    | .ok _ =>
      ok := false
      notes := notes.push s!"{what}: not refused"
    | .error e =>
      if (e.splitOn needle).length ≤ 1 then ok := false
      notes := notes.push s!"{what}: refused"
  -- a moving fundamental is refused
  let moving : BuildM SplineOscParts := do
    let band ← match bandOf catmullRomBasis with
      | .ok b => pure b | .error e => throw e
    let f0 ← mul (← sampleIndex) (← lit 1)
    splineOscSig catmullRomBasis band (← (Array.range 8).mapM fun _ => lit 0) (← toFloatE f0) (← tau)
  match assembleCompleteWithResult {} "moving" #[] (do
      let p ← moving
      pure (({ outputs := #[{ name := "out", type? := some (.scalar .float) }],
               assigns := #[(.port ⟨0⟩, p.out)] } : CompleteProgramBody), ())) with
  | .ok _ =>
      ok := false
      notes := notes.push "a moving fundamental: not refused"
  | .error e =>
      if (e.splitOn "must be uniform").length ≤ 1 then
        ok := false
        notes := notes.push s!"a moving fundamental: {e}"
  IO.println s!"        {String.intercalate " · " notes.toList}"
  if ok then
    passGate label "class power ≡ brute force; the derived edge is the least M meeting the budget; no-band, 0 dB, too-deep and moving-fundamental requests refused"
  else
    failGate label (String.intercalate "; " notes.toList)

-- ── spline-osc-oracle ────────────────────────────────────────────────────

def runOracle (arena : Arena) : IO Bool := do
  let _ := arena
  let label := "spline-osc-oracle"
  let mut configs : Array (SplineOscBand × Cfg × String) := #[]
  for b in bases do
    let band ← match bandOf b with
      | .ok x => pure x | .error e => return ← failGate label e
    let nyq := rate.toFloat * 0.5
    let fLo := nyq / band.capacity.toFloat
    let fHi := nyq / band.edge.toFloat
    let cs := lcgControls (b.scale * 31 + 5)
    let points : Array (Float × String) := #[
      (0.0, "0 Hz"), (0.37 * fLo, "below"), (fLo + 0.3 * (fHi - fLo), "in band"),
      (-(fLo + 0.7 * (fHi - fLo)), "in band, negative"), (2.3 * fHi, "above"),
      (8.1 * fHi, "far above")]
    for (f, what) in points do
      configs := configs.push (band, { basis := b, cs, inc := incOf f, off := 1342177280 }, what)
  let outs : BuildM (Array Sig) := configs.mapM fun (band, c, _) => do
    pure (← oscSig band c).out
  let plan ← match compileMulti {} "spline_osc_oracle" outs rate with
    | .ok plan => pure plan
    | .error e => return ← failGate label s!"compile: {firstLine e}"
  let ns := Tropical.Tropicaltest.SplineSpectrum.scattered 512 ++
    (Tropical.Tropicaltest.SplineSpectrum.scattered 512).map (· + tau40) ++
    (Tropical.Tropicaltest.SplineSpectrum.scattered 512).map (· + tau47 + 12345)
  let got ← renderAt plan configs.size ns
  let mut worst := 0.0
  let mut ok := true
  let mut notes := #[]
  for ci in [0:configs.size] do
    let (band, c, what) := configs[ci]!
    let hs := c.harmonics band.capacity
    let want := ns.map (oscRef band c hs)
    let err := maxAbsDiff got[ci]! want
    let trip := tripOf band c.f0
    let tol := railEps * massOf hs trip + 1e-12
    worst := max worst (err / tol)
    if err.isNaN || err > tol || got[ci]!.size != ns.size then
      ok := false
      notes := notes.push s!"{basisLabel c.basis} {what} (f0 = {fmt c.f0 2}): |Δ| {sci err} > {sci tol}"
  IO.println s!"        worst |Δ| / (rail bound + 1e-12) = {fmt worst 3} over {configs.size} configurations × {ns.size} samples (τ near 0, 2^40, 2^47)"
  if ok then
    passGate label "the render ≡ the f64 mix (1 − w)·cubic + w·bank below, inside and above the band, at negative and zero f0, with a phase offset, out to τ ≈ 2^47"
  else
    failGate label (String.intercalate "; " notes.toList)

-- ── spline-osc-budget ────────────────────────────────────────────────────

/-- Error power against the ideal band-limited curve over in-band AC power. -/
def measure (got : Array Float) (c : Cfg) (ns : Array Nat) : Float := Id.run do
  let m := belowNyquist c.f0
  let hs := c.harmonics m
  let dc := c.cs.foldl (· + ·) 0.0 / c.cs.size.toFloat
  let mut e := 0.0
  let mut p := 0.0
  for i in [0:ns.size] do
    let ideal := bankRef c.cs hs m (phaseAt c ns[i]!)
    let d := got[i]! - ideal
    e := e + d * d
    p := p + (ideal - dc) * (ideal - dc)
  return e / p

/-- The budget gate's configurations: a log sweep of LCG controls, and each
    worst-case control set (one harmonic class) across the band. -/
def budgetConfigs (b : SplineBasis) (band : SplineOscBand) : Array (SplineOscBand × Cfg × String) := Id.run do
  let mut configs := #[]
  let cs := lcgControls (b.scale * 7 + 3)
  for i in [0:36] do
    let f := 20.0 * Float.pow 1000.0 (i.toFloat / 35.0)
    configs := configs.push (band, ({ basis := b, cs, inc := incOf f } : Cfg), "sweep")
  let nyq := rate.toFloat * 0.5
  let fLo := nyq / band.capacity.toFloat
  let fHi := nyq / band.edge.toFloat
  for m in [1:5] do
    let csm := (Array.range 8).map fun j => Float.cos (2.0 * piF * (j * m).toFloat / 8.0)
    for i in [0:10] do
      let f := fLo * 0.97 + (fHi * 1.003 - fLo * 0.97) * i.toFloat / 9.0 + 0.0137
      configs := configs.push (band, ({ basis := b, cs := csm, inc := incOf f } : Cfg), s!"class {m}")
  return configs

/-- Measure every configuration: (all within budget, worst dB, where, sweep rows). -/
def budgetReport (configs : Array (SplineOscBand × Cfg × String)) (got : Array (Array Float))
    (ns : Array Nat) : Bool × Float × String × Array String := Id.run do
  let mut ok := true
  let mut worstDb := -1000.0
  let mut worstAt := ""
  let mut sweepRows : Array String := #[]
  for ci in [0:configs.size] do
    let (band, c, what) := configs[ci]!
    let ratio := measure got[ci]! c ns
    let db := dbOf ratio
    if what == "sweep" && ci % 4 == 0 then
      sweepRows := sweepRows.push s!"{fmt c.f0 0}:{fmt db 0}"
    if db > worstDb then
      worstDb := db
      worstAt := s!"{basisLabel c.basis} {what} f0 = {fmt c.f0 1} Hz (w = {fmt (weightOf band c.f0).2 3})"
    if ratio.isNaN || ratio > band.budget then
      ok := false
  return (ok, worstDb, worstAt, sweepRows)

def budgetLen : Nat := 8192

def runBudget : IO Bool := do
  let label := "spline-osc-budget"
  let ns := range (tau40 + 777) budgetLen
  let mut configs : Array (SplineOscBand × Cfg × String) := #[]
  for b in bases do
    match bandOf b with
    | .ok band => configs := configs ++ budgetConfigs b band
    | .error e => return ← failGate label e
  -- at most 64 output channels per plan
  let mut got : Array (Array Float) := #[]
  let mut start := 0
  while start < configs.size do
    let chunk := configs.extract start (start + 32)
    let outs : BuildM (Array Sig) := chunk.mapM fun (band, c, _) => do
      pure (← oscSig band c).out
    match compileMulti {} "spline_osc_budget" outs rate with
    | .error e => return ← failGate label s!"compile: {firstLine e}"
    | .ok plan => got := got ++ (← renderAt plan chunk.size ns)
    start := start + 32
  do
    let (ok, worstDb, worstAt, sweepRows) := budgetReport configs got ns
    IO.println s!"        worst error {fmt worstDb 1} dB re in-band power at {worstAt}; budget {splineOscBudgetDb} dB below; {configs.size} configurations × {budgetLen} samples from τ = 2^40"
    IO.println s!"        sweep (f0 Hz: dB): {String.intercalate " " sweepRows.toList}"
    if ok then
      passGate label s!"alias + loss against the ideal band-limited curve within {splineOscBudgetDb} dB below the in-band power at every fundamental, 20 Hz … 20 kHz, and each worst-case class across the band, both bases (worst {fmt worstDb 1} dB)"
    else
      failGate label s!"worst {fmt worstDb 1} dB at {worstAt}"

-- ── spline-osc-continuity ────────────────────────────────────────────────

/-- `Σ_{n>M} 2|F_n|`: the summed harmonics to `n = N`, then the closed-form
    tail bound. -/
def tailMass (c : Cfg) (hs : Array (Float × Float)) (m : Nat) : Float :=
  let l1 := c.cs.foldl (fun a x => a + x.abs) 0.0
  let n := hs.size
  (massOf hs n - massOf hs m) + tailBound c.basis l1 8 n

def runContinuity : IO Bool := do
  let label := "spline-osc-continuity"
  let steps := 48
  let spacing := 997
  let t0 := tau40 + 12345
  let mut ok := true
  let mut notes := #[]
  for b in bases do
    let band ← match bandOf b with
      | .ok x => pure x | .error e => return ← failGate label e
    let nyq := rate.toFloat * 0.5
    let fLo := nyq / band.capacity.toFloat
    let fHi := nyq / band.edge.toFloat
    let cs := lcgControls (b.scale * 13 + 1)
    -- steps from 0.9·f_lo to 1.1·f_hi, each re-anchored at its start
    let mut cfgs : Array Cfg := #[]
    let mut off : Int := 0
    for i in [0:steps + 1] do
      let f := 0.9 * fLo + (1.1 * fHi - 0.9 * fLo) * i.toFloat / steps.toFloat
      let inc := incOf f
      let at_ := t0 + i * spacing
      if let some prev := cfgs.back? then
        -- off₁ = off₀ + Φ(inc₀, C) − Φ(inc₁, C)  (mod 2³²)
        off := (off + prev.inc * at_ - inc * at_) % 4294967296
      cfgs := cfgs.push { basis := b, cs, inc, off := off.toNat }
    let junctions := (Array.range steps).map fun i => t0 + (i + 1) * spacing
    let outs : BuildM (Array Sig) := cfgs.mapM fun c => do pure (← oscSig band c).out
    let plan ← match compileMulti {} "spline_osc_continuity" outs rate with
      | .ok plan => pure plan
      | .error e => return ← failGate label s!"compile: {firstLine e}"
    let got ← renderAt plan cfgs.size junctions
    let hs := cfgs[0]!.harmonics 20000
    let mut worstStep := 0.0
    let mut worstRatio := 0.0
    let mut hardSwitch := 0.0
    for i in [0:steps] do
      let a := cfgs[i]!
      let c := cfgs[i + 1]!
      let step := (got[i + 1]![i]! - got[i]![i]!).abs
      let (_, wa) := weightOf band a.f0
      let (_, wc) := weightOf band c.f0
      let ma := belowNyquist a.f0
      let mc := belowNyquist c.f0
      let popped := massOf hs (max ma mc) - massOf hs (min ma mc)
      let bound := (wc - wa).abs * tailMass a hs ma + wc * popped +
        railEps * massOf hs band.capacity + 1e-12
      worstStep := max worstStep step
      worstRatio := max worstRatio (step / bound)
      hardSwitch := max hardSwitch (tailMass a hs ma)
      if step.isNaN || step > bound then
        ok := false
    let peak := Tropical.Tropicaltest.SplineSpectrum.weightMass b cs 64
    notes := notes.push s!"{basisLabel b}: worst step {sci worstStep} (worst step / bound {fmt worstRatio 3}; a hard switch could step up to {sci hardSwitch}; Σ2|F_n| {fmt peak 3})"
  IO.println s!"        {String.intercalate " · " notes.toList}"
  if ok then
    passGate label s!"{steps} anchored retunes across the band, each step ≤ |Δw|·Σ_{"{"}n>M{"}"}2|F_n| + w·(harmonics crossing Nyquist) + rail, both bases, from τ = 2^40"
  else
    failGate label (String.intercalate "; " notes.toList)

-- ── spline-osc-trip ──────────────────────────────────────────────────────

/-- The reduce regions of a plan: `(loopCount, has a live count)`. -/
def reduceRegions (p : Tropical.Plan.FlatPlan) : Array (Nat × Bool) := Id.run do
  let mut out := #[]
  for f in p.instanceFunctions do
    for block in Tropical.Ir.Stage0.collectBlocks f do
      for i in block do
        if i.tag == "ReduceBegin" then
          out := out.push (i.loopCount, i.args.size == 2)
  return out

def runTrip : IO Bool := do
  let label := "spline-osc-trip"
  let mut ok := true
  let mut notes := #[]
  for b in bases do
    let band ← match bandOf b with
      | .ok x => pure x | .error e => return ← failGate label e
    let nyq := rate.toFloat * 0.5
    let fLo := nyq / band.capacity.toFloat
    let fHi := nyq / band.edge.toFloat
    let cs := lcgControls 9
    let fs := #[0.0, 0.5 * fLo, 0.999 * fLo, 0.5 * (fLo + fHi), 1.7 * fHi, -1.7 * fHi]
    let cfgs := fs.map fun f => ({ basis := b, cs, inc := incOf f } : Cfg)
    let outs : BuildM (Array Sig) := do
      let mut outs := #[]
      for c in cfgs do
        let parts ← oscSig band c
        outs := outs ++ #[parts.out, parts.trip, parts.weight]
      pure outs
    let plan ← match compileMulti {} "spline_osc_trip" outs rate with
      | .ok plan => pure plan
      | .error e => return ← failGate label s!"compile: {firstLine e}"
    let regions := reduceRegions plan
    unless regions.size == cfgs.size && regions.all (· == (band.capacity, true)) do
      ok := false
      notes := notes.push s!"{basisLabel b}: regions {regions} (want {cfgs.size} × ({band.capacity}, live))"
    let got ← renderAt plan (3 * cfgs.size) #[0, tau40]
    let mut row := #[]
    for ci in [0:cfgs.size] do
      let c := cfgs[ci]!
      let trip := got[3 * ci + 1]![0]!
      let w := got[3 * ci + 2]![0]!
      let want := tripOf band c.f0
      row := row.push s!"{fmt c.f0 0} Hz: trip {fmt trip 0}, w {fmt w 3}"
      if trip != want.toFloat || got[3 * ci + 1]![1]! != trip then
        ok := false
      if (want == 0 && w != 0.0) then ok := false
    notes := notes.push s!"{basisLabel b} ({band.capacity}-partial region, live count): {String.intercalate ", " row.toList}"
  IO.println s!"        {String.intercalate " · " notes.toList}"
  if ok then
    passGate label "the family region trips a live count over the derived capacity; the count is 0 (and w = 0) below the band and the harmonics below Nyquist above it"
  else
    failGate label (String.intercalate "; " notes.toList)

-- ── spline-osc-live ──────────────────────────────────────────────────────

private def frames : Nat := 64

private def liveF0 : Float := 800.0

private def livePatch (values : Array Float) : String :=
  let params := (Array.range 8).toList.map fun j =>
    "{\"op\":\"paramDecl\",\"name\":\"k" ++ toString j ++ "\",\"value\":" ++
      toString values[j]! ++ "}"
  let inputs := (Array.range 8).toList.map fun j =>
    "\"c" ++ toString j ++ "\":{\"op\":\"param\",\"name\":\"k" ++ toString j ++ "\"}"
  "{\"schema\":\"tropical_program_2\",\"name\":\"spline_osc_live\",\"body\":{\"op\":\"block\",\"decls\":[" ++
  ",".intercalate params ++ "," ++
  "{\"op\":\"instanceDecl\",\"name\":\"osc\",\"program\":\"SplineOscCatmullRom8\",\"inputs\":{" ++
  "\"f0\":800," ++ ",".intercalate inputs ++ "}}]," ++
  "\"assigns\":[{\"op\":\"outputAssign\",\"name\":\"dac.out\",\"expr\":" ++
  "{\"op\":\"ref\",\"instance\":\"osc\",\"output\":\"out\"}}]}}"

def runLive : IO Bool := do
  let label := "spline-osc-live"
  let before := lcgControls 11
  let written := -0.796875
  let after := before.set! 5 written
  let registry ← match Tropical.Session.bootRegistry with
    | .error e => return ← failGate label s!"registry: {e.toJson.compress}"
    | .ok st => pure st
  let rt ← match (Tropical.Session.compilePatch (livePatch before)).runOn registry with
    | (.error f, _) => return ← failGate label s!"compile: {f.toJson.compress}"
    | (.ok (plan, blocks), _) =>
      let rt ← Tropical.Ffi.Runtime.new frames.toUInt32
      Tropical.StagedLoad.loadTyped rt plan blocks
      pure rt
  let rateRt := (← rt.sampleRate).toUInt64.toNat
  let band ← match bandOf catmullRomBasis with
    | .ok x => pure x | .error e => return ← failGate label e
  -- the kernel's increment, `toInt(800·2³²/rate)`; the weight reads f0 = 800
  let inc : Int := (liveF0 * 4294967296.0 / rateRt.toFloat).floor.toUInt64.toNat
  let reference := fun (cs : Array Float) (start : Nat) =>
    let c : Cfg := { basis := catmullRomBasis, cs, inc }
    let nyq := rateRt.toFloat * 0.5
    let cap := band.capacity.toFloat
    let edge := band.edge.toFloat
    let t := (liveF0 * cap / nyq - 1.0) * edge / (cap - edge)
    let w := smoothF t
    let trip := if t > 0.0 then min band.capacity ((nyq / liveF0).ceil - 1.0).toUInt64.toNat else 0
    let hs := c.harmonics band.capacity
    (range start frames).map fun n =>
      let ph := phaseAt c n
      (1.0 - w) * curveAt catmullRomBasis cs (ph.toFloat / 4294967296.0) + w * bankRef cs hs trip ph
  rt.process
  let buffer0 := decodeF64LE (← rt.outputBytes)
  if (← rt.setParam "k5" written).isNone then
    return ← failGate label "set_param: the loaded manifest has no param 'k5'"
  rt.process
  let buffer1 := decodeF64LE (← rt.outputBytes)
  let tol := railEps * massOf ((({ basis := catmullRomBasis, cs := after, inc } : Cfg)).harmonics band.capacity) band.capacity + 1e-12
  let err0 := maxAbsDiff buffer0 (reference before 0)
  let err1 := maxAbsDiff buffer1 (reference after frames)
  let moved := maxAbsDiff buffer1 (reference before frames)
  IO.println s!"        rate {rateRt}, f0 = 800 Hz (w = {fmt (smoothF ((liveF0 * band.capacity.toFloat / (rateRt.toFloat * 0.5) - 1.0) * band.edge.toFloat / (band.capacity - band.edge).toFloat)) 3}): buffer 0 vs reference(before) {sci err0} · buffer 1 vs reference(after) {sci err1} · vs reference(before) {sci moved}"
  if buffer0.size == frames && buffer1.size == frames && err0 ≤ tol && err1 ≤ tol && moved > 1e-3 then
    passGate label "a set_param on c5 reaches the next buffer of the loaded SplineOscCatmullRom8 kernel inside the band, matching the reference mix at the new value (no recompile)"
  else
    failGate label s!"before {sci err0}, after {sci err1}, moved {sci moved}"

-- ── cost (printed, not gated) ────────────────────────────────────────────

private def costPatch (f0 : Float) : String :=
  let inputs := (Array.range 8).toList.map fun j =>
    "\"c" ++ toString j ++ "\":{\"op\":\"param\",\"name\":\"k" ++ toString j ++ "\"}"
  let params := (Array.range 8).toList.map fun j =>
    "{\"op\":\"paramDecl\",\"name\":\"k" ++ toString j ++ "\",\"value\":" ++
      toString (lcgControls 3)[j]! ++ "}"
  "{\"schema\":\"tropical_program_2\",\"name\":\"spline_osc_cost\",\"body\":{\"op\":\"block\",\"decls\":[" ++
  ",".intercalate params ++ "," ++
  "{\"op\":\"instanceDecl\",\"name\":\"osc\",\"program\":\"SplineOscCatmullRom8\",\"inputs\":{" ++
  "\"f0\":" ++ toString f0 ++ "," ++ ",".intercalate inputs ++ "}}]," ++
  "\"assigns\":[{\"op\":\"outputAssign\",\"name\":\"dac.out\",\"expr\":" ++
  "{\"op\":\"ref\",\"instance\":\"osc\",\"output\":\"out\"}}]}}"

/-- Wall-clock ns per sample of `SplineOscCatmullRom8` at a few fundamentals
    (informational). -/
def printCost : IO Unit := do
  let registry ← match Tropical.Session.bootRegistry with
    | .error _ => return
    | .ok st => pure st
  let mut row := #[]
  for (f0, what) in #[(100.0, "below"), (800.0, "in band"), (1500.0, "above"), (5000.0, "above")] do
    match (Tropical.Session.compilePatch (costPatch f0)).runOn registry with
    | (.error _, _) => row := row.push s!"{what}: compile failed"
    | (.ok (plan, blocks), _) =>
      let block := 512
      let rt ← Tropical.Ffi.Runtime.new block.toUInt32
      Tropical.StagedLoad.loadTyped rt plan blocks
      for _ in [0:50] do rt.process
      let reps := 400
      let t0 ← IO.monoNanosNow
      for _ in [0:reps] do rt.process
      let t1 ← IO.monoNanosNow
      let ns := (t1 - t0).toFloat / (reps * block).toFloat
      row := row.push s!"{fmt f0 0} Hz ({what}) {fmt ns 1} ns/sample"
  IO.println s!"        cost, SplineOscCatmullRom8 (JIT, wall clock): {String.intercalate " · " row.toList}"

def runSplineOsc (arena : Arena) : IO (Nat × Nat) := do
  let results := #[← runBand, ← runOracle arena, ← runBudget, ← runContinuity, ← runTrip,
    ← runLive]
  printCost
  pure (results.size, results.filter (!·) |>.size)

end Tropical.Tropicaltest.SplineOsc
