import Tropical.Tropicaltest.SplineSpectrum
import Tropical.EmitArrow.Stdlib

/-!
# Spline-oscillator gates: one spline, alias-bounded at every pitch

`lean/Tropical/EmitArrow/SplineOsc.lean` is the module these gates back;
`design/spline-osc.md` is the note. K = 8, both bases, the registered budget
(`splineOscBudgetDb`, 60 dB).

* **spline-osc-band** — the jump-formula `|φ̂|` against the closed-form
  transform; the derived bands (K = 4, 8, 16, 64, both bases) against an
  independent coherent bound (closed-form transform, explicit class test, its
  own tail); refusals (capacity below the least, 0 dB, past the resolved
  depth, one control, a moving fundamental).
* **spline-osc-oracle** — the JIT render against an independent f64 reference
  of the stated mix `(1 − w)·cubic + w·bank`, at fundamentals below, inside
  and above the band, negative and zero, with a phase offset, at τ near 0,
  2⁴⁰ and 2⁴⁷ (the bank's sine rail and the cubic read one phase there).
* **spline-osc-budget** — the measured error against the ideal band-limited
  curve (every harmonic strictly below Nyquist), relative to the in-band
  power: a 20 Hz – 20 kHz sweep, and the exact coherent worst cases at
  rational fundamentals `rate·p/N` (`p ≤ 3`, 48 and 44.1 kHz) with the
  eigenvector control set. A band half an octave too high (edge 16, capacity
  32) must fail.
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
def weightOf (band : SplineOscBand) (f0 : Float) (r : Nat := rate) : Float × Float :=
  let nyq := r.toFloat * 0.5
  let cap := band.capacity.toFloat
  let edge := band.edge.toFloat
  let t := (f0.abs * cap / nyq - 1.0) * edge / (cap - edge)
  (t, smoothF t)

/-- The harmonics strictly below Nyquist. -/
def belowNyquist (f0 : Float) (r : Nat := rate) : Nat :=
  if f0 == 0.0 then 0 else
  let x := r.toFloat * 0.5 / f0.abs
  (x.ceil - 1.0).toUInt64.toNat

/-- The bank's trip count: the harmonics below Nyquist while `w > 0`. -/
def tripOf (band : SplineOscBand) (f0 : Float) (r : Nat := rate) : Nat :=
  if (weightOf band f0 r).1 > 0.0 then min band.capacity (belowNyquist f0 r) else 0

/-- A configuration: basis, controls, a fundamental on the rail (signed
    increment) and a phase offset (an integer of the Q0.32 rail). -/
structure Cfg where
  basis : SplineBasis
  cs : Array Float
  inc : Int
  off : Nat := 0
  rate : Nat := 48000
  /-- The `f0` literal handed to the kernel, when it is not `inc`'s exact
      value (a rational `rate·p/N`; the kernel truncates its increment). -/
  f0lit : Option Float := none
deriving Inhabited

/-- τ = 2⁴⁰ and 2⁴⁷ frames. -/
def tau40 : Nat := 1099511627776
def tau47 : Nat := 140737488355328

def Cfg.f0 (c : Cfg) : Float :=
  match c.f0lit with
  | some f => f
  | none =>
    let f := freqOfInc c.inc.natAbs c.rate
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
  let (_, w) := weightOf band c.f0 c.rate
  let cubic := curveAt c.basis c.cs (ph.toFloat / 4294967296.0)
  let bank := bankRef c.cs hs (tripOf band c.f0 c.rate) ph
  (1.0 - w) * cubic + w * bank

def oscSig (band : SplineOscBand) (c : Cfg) : BuildM SplineOscParts := do
  let f0 ← litExact c.f0
  let phase ← litExact (c.off.toFloat / 4294967296.0)
  let ctl ← c.cs.mapM litExact
  splineOscSig c.basis band ctl f0 (← tau) (some phase)

def incOf (f : Float) (r : Nat := rate) : Int :=
  let i := (f.abs * 4294967296.0 / r.toFloat).floor.toUInt64.toNat
  if f < 0.0 then -(i : Int) else i

def lcgControls (seed : Nat) : Array Float := (controlsOf seed 8).map controlFloat

/-- `Σ_p 2|F_p|` over `p ≤ trip`. -/
def massOf (hs : Array (Float × Float)) (trip : Nat) : Float :=
  (hs.extract 0 trip).foldl (fun a (x, y) => a + 2.0 * Float.sqrt (x * x + y * y)) 0.0

-- ── spline-osc-band ──────────────────────────────────────────────────────

/-- An independent coherent bound `B(M)`, `M = 1 … mMax`: the closed-form
    transform (`sinc⁴`, `sinc³(3 sinc − 2 cos)`), classes assigned by testing
    `n mod K ∈ {m, K − m}`, sums to `n = 200000`, and the closed-form decay
    bound past it (`(πξ)⁻⁴`, or `2(πξ)⁻³ + 3(πξ)⁻⁴`). -/
def independentBounds (b : SplineBasis) (k mMax : Nat) : Array Float := Id.run do
  let big := 200000
  let classes := (Array.range (k / 2)).map (· + 1)
  let inClass := fun (n m : Nat) => n % k == m || n % k == k - m
  let kf := k.toFloat
  let nf := big.toFloat
  let k3 := kf * kf * kf
  let p3 := piF * piF * piF
  let n2 := nf * nf
  let tail := if b.scale == 6 then k3 * kf / (3.0 * p3 * piF * n2 * nf)
    else 2.0 * k3 / (2.0 * p3 * n2) + 3.0 * k3 * kf / (3.0 * p3 * piF * n2 * nf)
  let vals := (Array.range (big + 1)).map fun n =>
    if n == 0 || n % k == 0 then 0.0 else (closedTransform b (n.toFloat / kf)).abs
  let totals := classes.map fun m => Id.run do
    let mut t := 0.0
    for n in [1:big + 1] do
      if inClass n m then t := t + vals[n]!
    return t
  let mut out := #[]
  for mm in [1:mMax + 1] do
    let mut sum := 0.0
    for ci in [0:classes.size] do
      let m := classes[ci]!
      let mut aIn := 0.0
      let mut bIn := 0.0
      for n in [1:mm + 1] do
        if inClass n m then
          aIn := aIn + vals[n]!
          bIn := bIn + vals[n]! * vals[n]!
      let a := totals[ci]! - aIn + tail
      sum := sum + (if bIn > 0.0 then a * a / bIn else (1.0 / 0.0))
    out := out.push (2.0 * sum)
  return out

/-- The band an independent bound gives: the least capacity meeting the
    budget at the bottom edge, then the least edge meeting it on every
    interval, with the crossfade weight computed here. -/
def independentBand (bs : Array Float) (db : Nat) : Option (Nat × Nat) := do
  let mStar ← (bs.findIdx? (withinDb db)).map (· + 1)
  let c := mStar + 1
  if bs.size < c then none
  let wSq := fun (e m : Nat) =>
    let t := ((c - m - 1) * e).toFloat / ((m + 1) * (c - e)).toFloat
    let w := smoothF t
    (1.0 - w) * (1.0 - w)
  let e := (List.range (c - 1)).map (· + 1) |>.find? fun e =>
    (List.range (c - e)).all fun i => withinDb db (wSq e (e + i) * bs[e + i - 1]!)
  pure (e.getD (c - 1), c)

def runBand : IO Bool := do
  let label := "spline-osc-band"
  let mut notes := #[]
  let mut ok := true
  -- the jump formula against the closed-form transform
  let mut worstJump := 0.0
  for b in bases do
    for k in #[3, 8] do
      for n in [4 * k + 1:4 * k + 200] do
        if n % k != 0 then
          let jump := b.jumpTransformAbsAt k n
          let ref := (closedTransform b (n.toFloat / k.toFloat)).abs
          -- |Re| + |Im| bounds |φ̂| above; the imaginary part is rounding
          worstJump := max worstJump ((jump - ref).abs / ref)
  notes := notes.push s!"jump-formula |φ̂| vs closed form past 4K: worst relative {sci worstJump}"
  if worstJump > 1e-6 then ok := false
  -- derived bands against the independent bound
  for b in bases do
    for k in #[4, 8, 16, 64] do
      match splineOscBand b k splineOscBudgetDb with
      | .error e =>
        ok := false
        notes := notes.push s!"{basisLabel b} K = {k}: {e}"
      | .ok band =>
        let bs := independentBounds b k (band.capacity + 8)
        let ind := independentBand bs splineOscBudgetDb
        -- the derived band meets the independent bound everywhere, and its
        -- capacity is within 1% (+1) of the independent least
        let wSq := fun (e c m : Nat) =>
          let t := ((c - m - 1) * e).toFloat / ((m + 1) * (c - e)).toFloat
          let w := smoothF t
          (1.0 - w) * (1.0 - w)
        let meets := (List.range (band.capacity - band.edge)).all fun i =>
          let m := band.edge + i
          withinDb splineOscBudgetDb (wSq band.edge band.capacity m * bs[m - 1]!)
        let close := match ind with
          | some (_, c) => c ≤ band.capacity && band.capacity ≤ c + c / 100 + 1
          | none => false
        if !(meets && close) then ok := false
        notes := notes.push s!"{basisLabel b} K = {k}: edge {band.edge}, capacity {band.capacity} (independent {ind}; meets it everywhere: {meets}), proved worst {fmt (dbOf band.worstBound) 1} dB"
  let refusals : Array (String × Except String SplineOscBand × String) := #[
    ("capacity below the least", splineOscBand catmullRomBasis 8 60 (some 40), "no band meets"),
    ("0 dB", splineOscBand catmullRomBasis 8 0, "bounds nothing"),
    ("130 dB", splineOscBand catmullRomBasis 8 130, "past what the bound resolves"),
    ("one control", splineOscBand catmullRomBasis 1 60, "constant curve")]
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
    passGate label "the derived bands (K = 4, 8, 16, 64, both bases) meet an independent coherent bound (closed-form transform, explicit class test) at every interval, with capacity within 1% of its least; no-band, 0 dB, too-deep, one-control and moving-fundamental requests refused"
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
  let m := belowNyquist c.f0 c.rate
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
  return configs

def budgetLen : Nat := 8192

/-- Control sets spanning the AC subspace, one per real DFT component
    (`cos`, `sin` of `2πjm/K`, and `(−1)^j`): orthogonal in in-band power. -/
def acBasis (k : Nat) : Array (Array Float) := Id.run do
  let mut out := #[]
  for m in [1:(k + 1) / 2] do
    out := out.push ((Array.range k).map fun j => Float.cos (2.0 * piF * (j * m).toFloat / k.toFloat))
    out := out.push ((Array.range k).map fun j => Float.sin (2.0 * piF * (j * m).toFloat / k.toFloat))
  if k % 2 == 0 then
    out := out.push ((Array.range k).map fun j => if j % 2 == 0 then 1.0 else -1.0)
  return out

/-- The cubic's exact worst case over control sets at the rational
    fundamental `f0 = rate·p/N`, where the cubic is periodic in `N` samples
    and every alias lands on a harmonic: the largest eigenvalue of
    `D^{-1/2}·E·D^{-1/2}` (`E` the error Gram matrix over one period, `D` the
    in-band powers, diagonal in `acBasis`), and its control set. -/
def coherentWorst (b : SplineBasis) (k p nn : Nat) (d : Float) : Float × Array Float := Id.run do
  let basisV := acBasis k
  let m := (nn + 2 * p - 1) / (2 * p) - 1  -- harmonics with 2np < N
  let errs := basisV.map fun v => Id.run do
    let hs := (Array.range m).map fun i => harmonicOf b v (i + 1)
    let mean := v.foldl (· + ·) 0.0 / k.toFloat
    let mut e := #[]
    let mut ideal := #[]
    for t in [0:nn] do
      let x := (((t * p) % nn).toFloat + d) / nn.toFloat
      let mut y := mean
      for i in [0:m] do
        let th := 2.0 * piF * (i + 1).toFloat * x
        let (fr, fi) := hs[i]!
        y := y + 2.0 * (fr * Float.cos th - fi * Float.sin th)
      e := e.push (curveAt b v x - y)
      ideal := ideal.push (y - mean)
    return (e, ideal)
  let d := basisV.size
  let pw := errs.map fun (_, ideal) => ideal.foldl (fun a y => a + y * y) 0.0 / nn.toFloat
  let g := (Array.range d).map fun u => (Array.range d).map fun v =>
    let (eu, _) := errs[u]!
    let (ev, _) := errs[v]!
    let dot := (Array.range nn).foldl (fun a t => a + eu[t]! * ev[t]!) 0.0 / nn.toFloat
    dot / Float.sqrt (pw[u]! * pw[v]!)
  -- power iteration
  let mut y := Array.replicate d 1.0
  let mut lam := 0.0
  for _ in [0:300] do
    let z := (Array.range d).map fun u => (Array.range d).foldl (fun a v => a + g[u]![v]! * y[v]!) 0.0
    let norm := Float.sqrt (z.foldl (fun a x => a + x * x) 0.0)
    if norm == 0.0 then break
    lam := norm / Float.sqrt (y.foldl (fun a x => a + x * x) 0.0)
    y := z.map (· / norm)
  let cs := (Array.range k).map fun j =>
    (Array.range d).foldl (fun a v => a + y[v]! * basisV[v]![j]! / Float.sqrt pw[v]!) 0.0
  let peak := cs.foldl (fun a x => max a x.abs) 0.0
  return (lam, cs.map (· / peak))

/-- The rational fundamentals `rate·p/N` (`p ≤ 3`) from `0.8·Nyquist/capacity`
    to `Nyquist/edge`, each with its exact worst control set and the mix's
    model ratio `(1 − w)²·λ`; the `top` largest. -/
def coherentCandidates (b : SplineBasis) (band : SplineOscBand) (r top : Nat) :
    Array (Float × Cfg × String) := Id.run do
  let nyq := r.toFloat * 0.5
  let fLo := 0.97 * nyq / band.capacity.toFloat
  let fHi := nyq / band.edge.toFloat
  let mut all := #[]
  for p in [1:4] do
    let nMin := (p.toFloat * r.toFloat / fHi).ceil.toUInt64.toNat
    let nMax := (p.toFloat * r.toFloat / fLo).floor.toUInt64.toNat
    for nn in [nMin:nMax + 1] do
      if Nat.gcd p nn == 1 then
        let f0 := r.toFloat * p.toFloat / nn.toFloat
        let w := (weightOf band f0 r).2
        -- the sub-sample phase of the sampling grid: d/N, d ∈ [0, 1)
        for di in [0:4] do
          let d := di.toFloat / 4.0
          let (lam, cs) := coherentWorst b 8 p nn d
          let model := (1.0 - w) * (1.0 - w) * lam
          let inc := incOf f0 r
          -- the phase at the window's first sample is d/N cycles
          let target : Int := (d / nn.toFloat * 4294967296.0).floor.toUInt64.toNat
          let off := ((target - inc * ((tau40 + 777 : Nat) : Int)) % 4294967296).toNat
          all := all.push (model, ({ basis := b, cs, inc, off, rate := r, f0lit := some f0 } : Cfg),
            s!"{basisLabel b} {r} Hz · {p}/{nn}, d = {d} (f0 = {fmt f0 1} Hz, w = {fmt w 3}, model {fmt (dbOf model) 2} dB)")
  let sorted := all.qsort (fun x y => x.1 > y.1)
  return sorted.extract 0 top

/-- Render configurations (grouped by rate, 32 per plan) and measure each. -/
def renderMeasure (cfgs : Array (SplineOscBand × Cfg)) (ns : Array Nat) :
    IO (Except String (Array Float)) := do
  let mut out := Array.replicate cfgs.size 0.0
  for r in #[48000, 44100] do
    let idx := (Array.range cfgs.size).filter fun i => cfgs[i]!.2.rate == r
    let mut start := 0
    while start < idx.size do
      let chunk := idx.extract start (start + 32)
      let outs : BuildM (Array Sig) := chunk.mapM fun i => do
        let (band, c) := cfgs[i]!
        pure (← oscSig band c).out
      match compileMulti {} "spline_osc_budget" outs r with
      | .error e => return .error (firstLine e)
      | .ok plan =>
        let got ← renderAt plan chunk.size ns
        for j in [0:chunk.size] do
          out := out.set! chunk[j]! (measure got[j]! cfgs[chunk[j]!]!.2 ns)
      start := start + 32
  return .ok out

def runBudget : IO Bool := do
  let label := "spline-osc-budget"
  let ns := range (tau40 + 777) budgetLen
  let mut configs : Array (SplineOscBand × Cfg × String) := #[]
  for b in bases do
    match bandOf b with
    | .error e => return ← failGate label e
    | .ok band =>
      configs := configs ++ budgetConfigs b band
      for r in #[48000, 44100] do
        configs := configs ++ (coherentCandidates b band r 6).map fun (_, c, what) => (band, c, s!"coherent {what}")
  -- the anti-gate: a band half an octave too high must fail
  let high : SplineOscBand := { edge := 16, capacity := 32, budgetDb := splineOscBudgetDb, worstBound := 0.0 }
  let highCfgs := (coherentCandidates catmullRomBasis high 48000 4).map fun (_, c, what) => (high, c, what)
  let measured ← match ← renderMeasure (configs.map fun (b, c, _) => (b, c)) ns with
    | .error e => return ← failGate label s!"compile: {e}"
    | .ok m => pure m
  let highMeasured ← match ← renderMeasure (highCfgs.map fun (b, c, _) => (b, c)) ns with
    | .error e => return ← failGate label s!"compile: {e}"
    | .ok m => pure m
  let mut ok := true
  let mut worstDb := -1000.0
  let mut worstAt := ""
  let mut rows := #[]
  for i in [0:configs.size] do
    let (band, c, what) := configs[i]!
    let ratio := measured[i]!
    let db := dbOf ratio
    if what == "sweep" && i % 4 == 0 then rows := rows.push s!"{fmt c.f0 0}:{fmt db 0}"
    if db > worstDb then
      worstDb := db
      worstAt := if what == "sweep" then s!"{basisLabel c.basis} sweep f0 = {fmt c.f0 1} Hz" else what
    if ratio.isNaN || !withinDb band.budgetDb ratio then ok := false
  let highWorst := highMeasured.foldl max 0.0
  let highAt := match highCfgs[(highMeasured.toList.idxOf highWorst)]? with
    | some (_, _, what) => what | none => ""
  IO.println s!"        true worst {fmt worstDb 2} dB re in-band power at {worstAt}; budget {splineOscBudgetDb} dB below; {configs.size} configurations × {budgetLen} samples from τ = 2^40"
  IO.println s!"        sweep, generic controls (f0 Hz: dB): {String.intercalate " " rows.toList}"
  IO.println s!"        anti-gate: Catmull-Rom with edge 16 / capacity 32 measures {fmt (dbOf highWorst) 2} dB at {highAt}"
  if withinDb splineOscBudgetDb highWorst then
    ok := false
  if ok then
    passGate label s!"against the ideal band-limited curve, within {splineOscBudgetDb} dB below the in-band power over a 20 Hz … 20 kHz sweep and the exact coherent worst cases (rate·p/N, p ≤ 3, 48 and 44.1 kHz, eigenvector controls), both bases (true worst {fmt worstDb 2} dB); a band half an octave too high fails ({fmt (dbOf highWorst) 2} dB)"
  else
    failGate label s!"worst {fmt worstDb 2} dB at {worstAt}; anti-gate {fmt (dbOf highWorst) 2} dB"

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

private def liveF0 : Float := 450.0

private def livePatch (values : Array Float) : String :=
  let params := (Array.range 8).toList.map fun j =>
    "{\"op\":\"paramDecl\",\"name\":\"k" ++ toString j ++ "\",\"value\":" ++
      toString values[j]! ++ "}"
  let inputs := (Array.range 8).toList.map fun j =>
    "\"c" ++ toString j ++ "\":{\"op\":\"param\",\"name\":\"k" ++ toString j ++ "\"}"
  "{\"schema\":\"tropical_program_2\",\"name\":\"spline_osc_live\",\"body\":{\"op\":\"block\",\"decls\":[" ++
  ",".intercalate params ++ "," ++
  "{\"op\":\"instanceDecl\",\"name\":\"osc\",\"program\":\"SplineOscCatmullRom8\",\"inputs\":{" ++
  "\"f0\":450," ++ ",".intercalate inputs ++ "}}]," ++
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
  -- the kernel's increment, `toInt(450·2³²/rate)`; the weight reads f0 = 450
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
  IO.println s!"        rate {rateRt}, f0 = 450 Hz (w = {fmt (smoothF ((liveF0 * band.capacity.toFloat / (rateRt.toFloat * 0.5) - 1.0) * band.edge.toFloat / (band.capacity - band.edge).toFloat)) 3}): buffer 0 vs reference(before) {sci err0} · buffer 1 vs reference(after) {sci err1} · vs reference(before) {sci moved}"
  if buffer0.size == frames && buffer1.size == frames && err0 ≤ tol && err1 ≤ tol && moved > 1e-3 then
    passGate label "a set_param on c5 reaches the next buffer of the loaded SplineOscCatmullRom8 kernel inside the band, matching the reference mix at the new value (no recompile)"
  else
    failGate label s!"before {sci err0}, after {sci err1}, moved {sci moved}"

-- ── cost (printed, not gated) ────────────────────────────────────────────

private def costPatch (program : String) (k : Nat) (f0 : Float) : String :=
  let inputs := (Array.range k).toList.map fun j =>
    "\"c" ++ toString j ++ "\":{\"op\":\"param\",\"name\":\"k" ++ toString j ++ "\"}"
  let params := (Array.range k).toList.map fun j =>
    "{\"op\":\"paramDecl\",\"name\":\"k" ++ toString j ++ "\",\"value\":" ++
      toString (controlFloat (controlsOf 3 k)[j]!) ++ "}"
  "{\"schema\":\"tropical_program_2\",\"name\":\"spline_osc_cost\",\"body\":{\"op\":\"block\",\"decls\":[" ++
  ",".intercalate params ++ "," ++
  "{\"op\":\"instanceDecl\",\"name\":\"osc\",\"program\":\"" ++ program ++ "\",\"inputs\":{" ++
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
  for (program, k, f0, what) in #[("SplineOscCatmullRom8", 8, 100.0, "below"),
      ("SplineOscCatmullRom8", 8, 450.0, "in band"), ("SplineOscCatmullRom8", 8, 412.0, "band bottom + 1%"),
      ("SplineOscCatmullRom8", 8, 1500.0, "above"), ("SplineOscCatmullRom8", 8, 5000.0, "above"),
      ("SplineOscCatmullRom16", 16, 180.0, "K = 16, band bottom"),
      ("SplineOscBSpline16", 16, 384.0, "K = 16 B-spline, band bottom")] do
    match (Tropical.Session.compilePatch (costPatch program k f0)).runOn registry with
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
      row := row.push s!"{program} {fmt f0 0} Hz ({what}) {fmt ns 1} ns/sample"
  IO.println s!"        cost (JIT, wall clock): {String.intercalate " · " row.toList}"

def runSplineOsc (arena : Arena) : IO (Nat × Nat) := do
  let results := #[← runBand, ← runOracle arena, ← runBudget, ← runContinuity, ← runTrip,
    ← runLive]
  printCost
  pure (results.size, results.filter (!·) |>.size)

end Tropical.Tropicaltest.SplineOsc
