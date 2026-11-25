import Tropical.Session.Compile
import Tropical.EmitArrow.SplineSpectrum
import Tropical.Testing.SplineSpectrumFixtures
import Tropical.Tropicaltest.Spline

/-!
# Spline-spectrum gates: a periodic spline as its harmonics below Nyquist

`lean/Tropical/EmitArrow/SplineSpectrum.lean` is the module these gates back;
`design/spline-spectrum.md` is the note.

* **spline-spectrum-closed-form** — (a) `kernelTransform`, computed from the
  basis matrix, against the closed forms `sinc⁴ξ` (B-spline) and
  `sinc³ξ·(3 sinc ξ − 2 cos πξ)` (Catmull-Rom) on a grid of `ξ`; (b) the
  harmonic `F_n = C[n mod K]·φ̂(n/K)/K` against a brute-force Fourier integral
  of the time-domain spline (Gauss–Legendre, 16 nodes on each of 16
  sub-intervals per segment), both bases, K = 3, 4, 7, 8, 16, n = 0 … 4K.
* **spline-spectrum-oracle** — the JIT render of the band-limited spline
  against an independent f64 reference that sums the same harmonics (closed
  form `φ̂`, the same integer phase rail, libm `cos`/`sin`, the family's
  Nyquist edge and trip count), both bases, K = 8, at three fundamentals.
* **spline-spectrum-envelope** — the band-limited spline under the complex
  SVF lowpass and highpass envelopes (`svfEnvelope`): each partial is
  `Re(2F_n·G(n·f0)·e^{iθ_n})` and the 0 Hz term is `F_0·Re G(0)` (1 for the
  lowpass, 0 for the highpass), against the same f64 reference.
* **spline-spectrum-vs-cubic** — at a 20 Hz fundamental with 1024 partials
  (every partial below the Nyquist edge), the band-limited render against the
  time-domain `splineSig` render on the same phase, within the tail bound
  `Σ_{n>N} 2|F_n|` plus the sine rail's error.
* **spline-spectrum-alias** — at three high fundamentals, the aliased energy
  of the time-domain cubic (its render minus the ideal hard-cut harmonic sum)
  against the closed-form `Σ_{n>M} 2|F_n|²`, and the band-limited render's
  residual against its reference.
* **spline-spectrum-live** — a `SplineSpectrumCatmullRom8` patch with
  `param:` controls: a `set_param` changes the next buffer to the reference at
  the new value, with no recompile.
-/

namespace Tropical.Tropicaltest.SplineSpectrum

open Tropical
open Tropical.Ir
open Tropical.EmitArrow
open Tropical.Tropicaltest.Spectral (compileMulti renderAt range maxAbsDiff peakOf sci fmt)
open Tropical.Tropicaltest.Spline (refWeights controlsOf controlFloat controlSig)

def piF : Float := 3.141592653589793

def sincF (x : Float) : Float :=
  if x == 0.0 then 1.0 else Float.sin (piF * x) / (piF * x)

/-- The closed-form kernel transforms. -/
def closedTransform (basis : SplineBasis) (xi : Float) : Float :=
  let s := sincF xi
  if basis.scale == 6 then s * s * s * s
  else s * s * s * (3.0 * s - 2.0 * Float.cos (piF * xi))

def basisLabel (basis : SplineBasis) : String :=
  if basis.scale == 2 then "catmull-rom" else "b-spline"

/-- The time-domain periodic spline at phase `x` (cycles), in f64. -/
def curveAt (basis : SplineBasis) (cs : Array Float) (x : Float) : Float := Id.run do
  let k := cs.size
  let s := x * k.toFloat
  let i := s.floor
  let t := s - i
  let w := refWeights basis t
  let iN : Int := if i < 0.0 then -((-i).toUInt64.toNat : Int) else (i.toUInt64.toNat : Int)
  let mut v := 0.0
  for j in [0:4] do
    v := v + w[j]! * cs[((iN - 1 + j) % (k : Int)).toNat]!
  return v

/-- The DFT `C[m]` of real values, libm. -/
def dftOf (cs : Array Float) (m : Nat) : Float × Float := Id.run do
  let k := cs.size
  let mut re := 0.0
  let mut im := 0.0
  for j in [0:k] do
    let a := 2.0 * piF * ((j * m) % k).toFloat / k.toFloat
    re := re + cs[j]! * Float.cos a
    im := im - cs[j]! * Float.sin a
  return (re, im)

/-- `F_n` from the closed-form `φ̂` (the reference's, not the build's). -/
def harmonicOf (basis : SplineBasis) (cs : Array Float) (n : Nat) : Float × Float :=
  let k := cs.size
  let (re, im) := dftOf cs (n % k)
  let s := closedTransform basis (n.toFloat / k.toFloat) / k.toFloat
  (re * s, im * s)

/-- Gauss–Legendre nodes and weights on `[−1, 1]` (Newton on `P_n`). -/
def gaussLegendre (n : Nat) : Array (Float × Float) := Id.run do
  let mut out := #[]
  for i in [1:n + 1] do
    let mut x := Float.cos (piF * (i.toFloat - 0.25) / (n.toFloat + 0.5))
    let mut dp := 0.0
    for _ in [0:100] do
      let mut p0 := 1.0
      let mut p1 := x
      for k in [2:n + 1] do
        let p2 := ((2 * k - 1).toFloat * x * p1 - (k - 1).toFloat * p0) / k.toFloat
        p0 := p1
        p1 := p2
      dp := n.toFloat * (x * p1 - p0) / (x * x - 1.0)
      x := x - p1 / dp
    out := out.push (x, 2.0 / ((1.0 - x * x) * dp * dp))
  return out

/-- `∫₀¹ f(x) e^{−2πinx} dx` by Gauss–Legendre on 16 sub-intervals of each
    of the K segments (the integrand is a cubic times an exponential on each). -/
def bruteHarmonic (basis : SplineBasis) (cs : Array Float) (n : Nat)
    (nodes : Array (Float × Float)) : Float × Float := Id.run do
  let k := cs.size
  let sub := 16
  let h := 1.0 / (k * sub).toFloat
  let mut re := 0.0
  let mut im := 0.0
  for cell in [0:k * sub] do
    let a := cell.toFloat * h
    for (node, weight) in nodes do
      let x := a + (node + 1.0) * 0.5 * h
      let v := curveAt basis cs x * weight * 0.5 * h
      let angle := 2.0 * piF * n.toFloat * x
      re := re + v * Float.cos angle
      im := im - v * Float.sin angle
  return (re, im)

-- ── spline-spectrum-closed-form ──────────────────────────────────────────

def runClosedForm : IO Bool := do
  let label := "spline-spectrum-closed-form"
  let bases := #[catmullRomBasis, uniformBSplineBasis]
  -- (a) φ̂ from the matrix vs the closed forms
  let mut worstT := 0.0
  for basis in bases do
    unless basis.kernelIsEven do
      return ← failGate label s!"{basisLabel basis}: kernel not recognised as even"
    for i in [0:4001] do
      let xi := i.toFloat / 100.0
      let d := (basis.kernelTransform i 100 - closedTransform basis xi).abs
      if d.isNaN then return ← failGate label s!"{basisLabel basis}: NaN at ξ = {xi}"
      worstT := max worstT d
  -- (b) F_n vs the brute-force integral
  let nodes := gaussLegendre 16
  let mut worstF := 0.0
  let mut scale := 0.0
  let mut count := 0
  for basis in bases do
    for k in #[3, 4, 7, 8, 16] do
      let cs := (controlsOf (k * 31 + basis.scale) k).map controlFloat
      let (cre, cim) := (Array.range k).foldl (fun (r, i) m =>
        let (a, b) := dftOf cs m
        (r.push a, i.push b)) (#[], #[])
      for n in [0:4 * k + 1] do
        -- the build's φ̂ (from the matrix), not the closed form
        let s := basis.kernelTransform n k / k.toFloat
        let got := (cre[n % k]! * s, cim[n % k]! * s)
        let want := bruteHarmonic basis cs n nodes
        let d := max (got.1 - want.1).abs (got.2 - want.2).abs
        worstF := max worstF d
        scale := max scale (max want.1.abs want.2.abs)
        count := count + 1
  IO.println s!"        φ̂ from the matrix vs closed form, ξ ∈ [0, 40]: worst |Δ| {sci worstT} · F_n vs brute-force integral ({count} harmonics, |F_n| up to {sci scale}): worst |Δ| {sci worstF}"
  -- φ̂ sums four O(1) pieces, so a few ulp of 1; F_n is the build's φ̂ times
  -- an exact DFT of m/64 values.
  if worstT ≤ 4e-15 && worstF ≤ 1e-15 then
    passGate label s!"φ̂ ≡ sinc⁴ / sinc³(3 sinc − 2 cos πξ) to {sci worstT}; F_n = C[n mod K]·φ̂(n/K)/K ≡ the Fourier integral to {sci worstF} (both bases, K = 3, 4, 7, 8, 16, n ≤ 4K)"
  else
    failGate label s!"φ̂ {sci worstT} (tolerance 4e-15), F_n {sci worstF} (tolerance 1e-15)"

-- ── The f64 reference of the band-limited render ─────────────────────────

/-- A fundamental on the rail: `inc` and the exactly representable `f0` with
    `⌊f0·2³²/rate⌋ = inc`. -/
def freqOfInc (inc rate : Nat) : Float := inc.toFloat * rate.toFloat / 4294967296.0

def smoothF (v : Float) : Float :=
  let t := max 0.0 (min 1.0 v)
  t * t * (3.0 - 2.0 * t)

/-- The family's sum at sample `n`: `F_0 + Σ_{p ≤ trip} edge_p·Re(2F_p e^{iθ_p})`,
    with the Nyquist edge (`edge = true`) or a hard cut at Nyquist. -/
def referenceAt (basis : SplineBasis) (cs : Array Float) (inc rate count : Nat)
    (edge : Bool) (n : Nat) : Float := Id.run do
  let f0 := freqOfInc inc rate
  let nyq := rate.toFloat * 0.5
  let trip := min count (nyq / max f0 (nyq / (count + 1).toFloat)).floor.toUInt64.toNat
  let k := cs.size
  let mut v := cs.foldl (· + ·) 0.0 / k.toFloat
  let base := (inc * n) % 4294967296
  for p in [1:trip + 1] do
    let ph := (p * base) % 4294967296
    let theta := 2.0 * piF * ph.toFloat / 4294967296.0
    let (fr, fi) := harmonicOf basis cs p
    let e := if edge then smoothF ((nyq - p.toFloat * f0) * 10.0 / nyq) else
      (if p.toFloat * f0 < nyq then 1.0 else 0.0)
    v := v + e * 2.0 * (fr * Float.cos theta - fi * Float.sin theta)
  return v

/-- `Σ_p 2|F_p|` over the evaluated partials: the scale of the sine rail's
    error. -/
def weightMass (basis : SplineBasis) (cs : Array Float) (trip : Nat) : Float :=
  (Array.range trip).foldl (fun acc i =>
    let (a, b) := harmonicOf basis cs (i + 1)
    acc + 2.0 * Float.sqrt (a * a + b * b)) 0.0

/-- The sine rail's per-partial error bound (cos and sin each within
    `2·10⁻⁹`, the Q30 output's quantum plus its fit). -/
def railEps : Float := 4e-9

def controls8 (seed : Nat) : Array Int := controlsOf seed 8

def familyTerm (basis : SplineBasis) (cs : Array Int) (inc rate count : Nat) :
    BuildM Sig := do
  let f0 ← litExact (freqOfInc inc rate)
  let ctl ← cs.mapM controlSig
  emitTerm (normalize (splineSpectrum basis count (← tau) (.konst f0) (ctl.map .konst)))

def scattered (len : Nat) : Array Nat := (Array.range len).map (· * 7919)

-- ── spline-spectrum-oracle ───────────────────────────────────────────────

def runOracle (arena : Arena) : IO Bool := do
  let label := "spline-spectrum-oracle"
  let rate := 48000
  let count := 256
  let incs : Array Nat := #[9842332, 89478485, 447392427]  -- ≈ 110 Hz, 1 kHz, 5 kHz
  let bases := #[catmullRomBasis, uniformBSplineBasis]
  let configs := bases.flatMap fun b => incs.map fun inc => (b, inc)
  let outs : BuildM (Array Sig) := configs.mapM fun (b, inc) =>
    familyTerm b (controls8 (inc % 1000)) inc rate count
  let plan ← match compileMulti arena "spline_spectrum_oracle" outs rate with
    | .ok plan => pure plan
    | .error e => return ← failGate label s!"compile: {firstLine e}"
  let ns := scattered 2048
  let got ← renderAt plan configs.size ns
  let mut worstRatio := 0.0
  let mut worstAbs := 0.0
  for ci in [0:configs.size] do
    let (b, inc) := configs[ci]!
    let cs := (controls8 (inc % 1000)).map controlFloat
    let want := ns.map (referenceAt b cs inc rate count true)
    let err := maxAbsDiff got[ci]! want
    let f0 := freqOfInc inc rate
    let trip := min count (rate.toFloat * 0.5 / f0).floor.toUInt64.toNat
    let bound := railEps * weightMass b cs trip + 1e-13
    if err.isNaN || got[ci]!.size != ns.size || peakOf want == 0.0 then
      return ← failGate label s!"{basisLabel b} f0 = {fmt f0}: error {err}, {got[ci]!.size} samples"
    worstRatio := max worstRatio (err / bound)
    worstAbs := max worstAbs err
  IO.println s!"        worst |Δ| {sci worstAbs}; worst |Δ| / (rail bound {sci railEps}·Σ2|F_n| + 1e-13) = {fmt worstRatio 3}"
  if worstRatio ≤ 1.0 then
    passGate label s!"{configs.size} configurations (both bases, f0 ≈ 110 Hz, 1 kHz, 5 kHz) × {ns.size} samples ≡ the f64 harmonic sum within the sine rail's bound"
  else
    failGate label s!"error exceeds the rail bound by {fmt worstRatio 3}×"

-- ── spline-spectrum-envelope ─────────────────────────────────────────────

/-- The SVF lowpass (`high = false`) or highpass gain at `f`: `x = f/fc`,
    `D = 1 − x² + i·x/Q`, `G = 1/D` or `−x²/D`. -/
def svfGain (high : Bool) (fc q f : Float) : Float × Float :=
  let x := f / fc
  let re := 1.0 - x * x
  let im := x / q
  let den := re * re + im * im
  let (nr, ni) := if high then (-(x * x), 0.0) else (1.0, 0.0)
  -- N / D = N·conj(D) / |D|²
  ((nr * re + ni * im) / den, (ni * re - nr * im) / den)

def runEnvelope (arena : Arena) : IO Bool := do
  let label := "spline-spectrum-envelope"
  let rate := 48000
  let count := 256
  let inc := 89478485  -- ≈ 1 kHz
  let fc := 2500.0
  let q := 2.0
  let configs : Array (SplineBasis × Bool) :=
    #[(catmullRomBasis, false), (catmullRomBasis, true), (uniformBSplineBasis, false)]
  let outs : BuildM (Array Sig) := configs.mapM fun (b, high) => do
    let f0 ← litExact (freqOfInc inc rate)
    let ctl ← (controls8 (b.scale + 17)).mapM controlSig
    let term := svfEnvelope (if high then .highpass else .lowpass)
      (.konst (← litExact fc)) (.konst (← litExact q))
      (splineSpectrum b count (← tau) (.konst f0) (ctl.map .konst))
    emitTerm (normalize term)
  let plan ← match compileMulti arena "spline_spectrum_envelope" outs rate with
    | .ok plan => pure plan
    | .error e => return ← failGate label s!"compile: {firstLine e}"
  let ns := scattered 2048
  let got ← renderAt plan configs.size ns
  let f0 := freqOfInc inc rate
  let nyq := rate.toFloat * 0.5
  let trip := min count (nyq / f0).floor.toUInt64.toNat
  let mut worstRatio := 0.0
  let mut notes := #[]
  for ci in [0:configs.size] do
    let (b, high) := configs[ci]!
    let cs := (controls8 (b.scale + 17)).map controlFloat
    let dcGain := (svfGain high fc q 0.0).1
    let mut mass := 0.0
    for p in [1:trip + 1] do
      let (fr, fi) := harmonicOf b cs p
      let (gr, gi) := svfGain high fc q (p.toFloat * f0)
      mass := mass + 2.0 * Float.sqrt ((fr * fr + fi * fi) * (gr * gr + gi * gi))
    let want := ns.map fun n => Id.run do
      let mut v := dcGain * cs.foldl (· + ·) 0.0 / 8.0
      let base := (inc * n) % 4294967296
      for p in [1:trip + 1] do
        let theta := 2.0 * piF * ((p * base) % 4294967296).toFloat / 4294967296.0
        let (fr, fi) := harmonicOf b cs p
        let (gr, gi) := svfGain high fc q (p.toFloat * f0)
        let zr := fr * gr - fi * gi
        let zi := fr * gi + fi * gr
        let e := smoothF ((nyq - p.toFloat * f0) * 10.0 / nyq)
        v := v + e * 2.0 * (zr * Float.cos theta - zi * Float.sin theta)
      return v
    let err := maxAbsDiff got[ci]! want
    let bound := railEps * mass + 1e-13
    if err.isNaN || got[ci]!.size != ns.size || peakOf want == 0.0 then
      return ← failGate label s!"{basisLabel b}: error {err}"
    worstRatio := max worstRatio (err / bound)
    notes := notes.push s!"{basisLabel b} {if high then "highpass" else "lowpass"} |Δ| {sci err}"
  IO.println s!"        {String.intercalate " · " notes.toList}; worst |Δ| / rail bound = {fmt worstRatio 3}"
  if worstRatio ≤ 1.0 then
    passGate label "under the SVF lowpass and highpass envelopes, the band-limited spline ≡ Σ Re(2F_n·G(n·f0)·e^{iθ_n}) + F_0·Re G(0) within the sine rail's bound"
  else
    failGate label s!"error exceeds the rail bound by {fmt worstRatio 3}×"

-- ── spline-spectrum-vs-cubic ─────────────────────────────────────────────

/-- `Σ_{n>N} 2|F_n| ≤ 2‖c‖₁·[K²/(π³N²) + K³/(π⁴N³)]` (Catmull-Rom,
    `|φ̂| ≤ 2(π|ξ|)⁻³ + 3(π|ξ|)⁻⁴`) or `2‖c‖₁·K³/(3π⁴N³)` (B-spline,
    `|φ̂| ≤ (π|ξ|)⁻⁴`). -/
def tailBound (basis : SplineBasis) (l1 : Float) (k n : Nat) : Float :=
  let kf := k.toFloat
  let nf := n.toFloat
  let p3 := piF * piF * piF
  let p4 := p3 * piF
  if basis.scale == 6 then 2.0 * l1 * kf * kf * kf / (3.0 * p4 * nf * nf * nf)
  else 2.0 * l1 * (kf * kf / (p3 * nf * nf) + kf * kf * kf / (p4 * nf * nf * nf))

def runVsCubic (arena : Arena) : IO Bool := do
  let label := "spline-spectrum-vs-cubic"
  let rate := 48000
  let count := 1024
  let inc := 1789569  -- ≈ 20 Hz: partial 1024 at ≈ 20.48 kHz, below the 21.6 kHz edge
  let bases := #[catmullRomBasis, uniformBSplineBasis]
  let outs : BuildM (Array Sig) := do
    let mut outs := #[]
    for b in bases do
      let cs := controls8 (b.scale * 101)
      outs := outs.push (← familyTerm b cs inc rate count)
      let phase ← mul (← sampleIndex) (← lit inc)
      outs := outs.push (← splineSig b .wrap (← cs.mapM controlSig) phase)
    pure outs
  let plan ← match compileMulti arena "spline_spectrum_vs_cubic" outs rate with
    | .ok plan => pure plan
    | .error e => return ← failGate label s!"compile: {firstLine e}"
  let ns := scattered 2048
  let got ← renderAt plan 4 ns
  let mut ok := true
  let mut notes := #[]
  for bi in [0:bases.size] do
    let b := bases[bi]!
    let cs := (controls8 (b.scale * 101)).map controlFloat
    let l1 := cs.foldl (fun a x => a + x.abs) 0.0
    let err := maxAbsDiff got[2 * bi]! got[2 * bi + 1]!
    let tail := tailBound b l1 8 count
    let bound := tail + railEps * weightMass b cs count
    notes := notes.push s!"{basisLabel b}: |Δ| {sci err} ≤ {sci bound} (tail {sci tail})"
    if err.isNaN || err > bound || peakOf got[2 * bi + 1]! == 0.0 then ok := false
  IO.println s!"        {String.intercalate " · " notes.toList}"
  if ok then
    passGate label "band-limited ≡ time-domain cubic at 20 Hz (1024 partials) within the closed-form tail bound plus the sine rail's error, both bases"
  else
    failGate label (String.intercalate "; " notes.toList)

-- ── spline-spectrum-alias ────────────────────────────────────────────────

/-- The closed-form aliased-energy fraction of the time-domain cubic at a
    fundamental with `M` harmonics below Nyquist:
    `Σ_{n>M} 2|F_n|² / Σ_{1≤n≤M} 2|F_n|²`. -/
def aliasFraction (basis : SplineBasis) (cs : Array Float) (m : Nat) : Float := Id.run do
  let mut inBand := 0.0
  let mut above := 0.0
  for n in [1:200001] do
    let (a, b) := harmonicOf basis cs n
    let p := 2.0 * (a * a + b * b)
    if n ≤ m then inBand := inBand + p else above := above + p
  return above / inBand

def dbOf (x : Float) : Float := 10.0 * Float.log10 (x + 1e-300)

def runAlias (arena : Arena) : IO Bool := do
  let label := "spline-spectrum-alias"
  let rate := 48000
  let count := 256
  -- rate/f0 ≈ 48.618, 16.18, 7.618: aliases fall ≥ 0.23·f0 from any harmonic
  let incs : Array Nat := #[88341294, 265446559, 563788357]
  let bases := #[catmullRomBasis, uniformBSplineBasis]
  let configs := bases.flatMap fun b => incs.map fun inc => (b, inc)
  let outs : BuildM (Array Sig) := do
    let mut outs := #[]
    for (b, inc) in configs do
      let cs := controls8 (b.scale * 7 + 3)
      outs := outs.push (← familyTerm b cs inc rate count)
      outs := outs.push (← splineSig b .wrap (← cs.mapM controlSig) (← mul (← sampleIndex) (← lit inc)))
    pure outs
  let plan ← match compileMulti arena "spline_spectrum_alias" outs rate with
    | .ok plan => pure plan
    | .error e => return ← failGate label s!"compile: {firstLine e}"
  let len := 65536
  let ns := range 0 len
  let got ← renderAt plan (2 * configs.size) ns
  let mut ok := true
  for ci in [0:configs.size] do
    let (b, inc) := configs[ci]!
    let cs := (controls8 (b.scale * 7 + 3)).map controlFloat
    let f0 := freqOfInc inc rate
    let m := (rate.toFloat * 0.5 / f0).floor.toUInt64.toNat
    let ideal := ns.map (referenceAt b cs inc rate count false)
    let edged := ns.map (referenceAt b cs inc rate count true)
    let dc := cs.foldl (· + ·) 0.0 / 8.0
    let mut pSig := 0.0
    let mut eTd := 0.0
    let mut eBl := 0.0
    let mut eEdge := 0.0
    let mut eStep := 0.0
    let mut stepPeak := 0.0
    for i in [0:len] do
      let s := ideal[i]! - dc
      pSig := pSig + s * s
      let a := got[2 * ci + 1]![i]! - ideal[i]!
      eTd := eTd + a * a
      let c := got[2 * ci]![i]! - edged[i]!
      eBl := eBl + c * c
      let e := edged[i]! - ideal[i]!
      eEdge := eEdge + e * e
      let st := got[2 * ci + 1]![i]! - got[2 * ci]![i]!
      eStep := eStep + st * st
      stepPeak := max stepPeak st.abs
    let predicted := aliasFraction b cs m
    let tdDb := dbOf (eTd / pSig)
    let predDb := dbOf predicted
    let blDb := dbOf (eBl / pSig)
    let edgeNote := if eEdge == 0.0 then "no partial in the edge band" else s!"{fmt (dbOf (eEdge / pSig)) 1} dB"
    let rms := Float.sqrt (pSig / len.toFloat)
    IO.println s!"        {basisLabel b} f0 = {fmt f0 1} Hz (M = {m}): time-domain alias {fmt tdDb 1} dB (closed form {fmt predDb 1} dB) · band-limited residual {fmt blDb 1} dB · soft Nyquist edge (in band, not alias) {edgeNote} · switch step cubic − band-limited: {fmt (dbOf (eStep / pSig)) 1} dB, peak {sci stepPeak} (signal rms {sci rms})"
    if tdDb.isNaN || (tdDb - predDb).abs > 0.5 || blDb > -140.0 then ok := false
  if ok then
    passGate label "time-domain alias energy = Σ_{n>M} 2|F_n|² (±0.5 dB) at three fundamentals, both bases; band-limited residual below −140 dB"
  else
    failGate label "alias energy off the closed form, or the band-limited residual above −140 dB"

-- ── spline-spectrum-live ─────────────────────────────────────────────────

private def frames : Nat := 64

private def liveF0 : Float := 250.0

private def livePatch (values : Array Int) : String :=
  let params := (Array.range 8).toList.map fun j =>
    "{\"op\":\"paramDecl\",\"name\":\"k" ++ toString j ++ "\",\"value\":" ++
      toString (controlFloat values[j]!) ++ "}"
  let inputs := (Array.range 8).toList.map fun j =>
    "\"c" ++ toString j ++ "\":{\"op\":\"param\",\"name\":\"k" ++ toString j ++ "\"}"
  "{\"schema\":\"tropical_program_2\",\"name\":\"spline_spectrum_live\",\"body\":{\"op\":\"block\",\"decls\":[" ++
  ",".intercalate params ++ "," ++
  "{\"op\":\"instanceDecl\",\"name\":\"sp\",\"program\":\"SplineSpectrumCatmullRom8\",\"inputs\":{" ++
  "\"f0\":250," ++ ",".intercalate inputs ++ "}}]," ++
  "\"assigns\":[{\"op\":\"outputAssign\",\"name\":\"dac.out\",\"expr\":" ++
  "{\"op\":\"ref\",\"instance\":\"sp\",\"output\":\"out\"}}]}}"

/-- The stdlib registry plus the band-limited spline fixtures, as
    `diffcli compile --fixtures` sees them. -/
def fixtureRegistry : Except String Tropical.SessionSt := do
  let st ← Tropical.Session.bootRegistry.mapError (·.toJson.compress)
  let act : Tropical.Session.SessionM Unit := do
    for (name, build) in Tropical.Testing.SplineSpectrumFixtures.splineSpectrumFixtures do
      let s ← get
      match build s.arena with
      | .error e => Tropical.internalError e
      | .ok (arena', idx) => discard <| Tropical.Session.registerResolved name arena' idx
  match act.runOn st with
  | (.ok (), st') => .ok st'
  | (.error f, _) => .error f.toJson.compress

def runLive : IO Bool := do
  let label := "spline-spectrum-live"
  let before := controlsOf 11 8
  let written : Int := -51
  let after := before.set! 5 written
  let registry ← match fixtureRegistry with
    | .error e => return ← failGate label s!"registry: {e}"
    | .ok st => pure st
  let rt ← match (Tropical.Session.compilePatch (livePatch before)).runOn registry with
    | (.error f, _) => return ← failGate label s!"compile: {f.toJson.compress}"
    | (.ok (plan, blocks), _) =>
      let rt ← Tropical.Ffi.Runtime.new frames.toUInt32
      Tropical.StagedLoad.loadTyped rt plan blocks
      pure rt
  let rate := (← rt.sampleRate).toUInt64.toNat
  -- the kernel's own increment: `toInt(f0·2³²/rate)`
  let inc := (liveF0 * 4294967296.0 / rate.toFloat).floor.toUInt64.toNat
  let f0 := liveF0
  let reference := fun (cs : Array Int) (start : Nat) =>
    (range start frames).map fun n => Id.run do
      -- `referenceAt` with the literal f0 (not inc·rate/2³²) for the edge
      let csF := cs.map controlFloat
      let nyq := rate.toFloat * 0.5
      let trip := min Testing.SplineSpectrumFixtures.fixtureCount (nyq / f0).floor.toUInt64.toNat
      let mut v := csF.foldl (· + ·) 0.0 / 8.0
      let base := (inc * n) % 4294967296
      for p in [1:trip + 1] do
        let theta := 2.0 * piF * ((p * base) % 4294967296).toFloat / 4294967296.0
        let (fr, fi) := harmonicOf catmullRomBasis csF p
        let e := smoothF ((nyq - p.toFloat * f0) * 10.0 / nyq)
        v := v + e * 2.0 * (fr * Float.cos theta - fi * Float.sin theta)
      return v
  rt.process
  let buffer0 := decodeF64LE (← rt.outputBytes)
  if (← rt.setParam "k5" (controlFloat written)).isNone then
    return ← failGate label "set_param: the loaded manifest has no param 'k5'"
  rt.process
  let buffer1 := decodeF64LE (← rt.outputBytes)
  let tol := railEps * weightMass catmullRomBasis (after.map controlFloat) 256 + 1e-13
  let err0 := maxAbsDiff buffer0 (reference before 0)
  let err1 := maxAbsDiff buffer1 (reference after frames)
  let moved := maxAbsDiff buffer1 (reference before frames)
  IO.println s!"        rate {rate}: buffer 0 vs reference(before) {sci err0} · buffer 1 vs reference(after) {sci err1} · buffer 1 vs reference(before) {sci moved} (tolerance {sci tol})"
  if buffer0.size == frames && buffer1.size == frames && err0 ≤ tol && err1 ≤ tol
      && moved > 1e-3 then
    passGate label "a set_param on c5 reaches the next buffer of the loaded band-limited kernel, which matches the reference at the new value (no recompile)"
  else
    failGate label s!"before {sci err0}, after {sci err1}, moved {sci moved}"

def runSplineSpectrum (arena : Arena) : IO (Nat × Nat) := do
  let results := #[← runClosedForm, ← runOracle arena, ← runEnvelope arena, ← runVsCubic arena,
    ← runAlias arena, ← runLive]
  pure (results.size, results.filter (!·) |>.size)

end Tropical.Tropicaltest.SplineSpectrum
