import Tropical.Tropicaltest.Spectral

/-!
# Spectral gates: families on a ratio column

`design/spectral-family.md` ("Ratio columns") is the note these gates back.
A family's partial `k` runs at `r_k·f0`; the column is harmonic (the default),
rational `n/d` (one shared ring position, exact) or free (one increment per
partial).

* **ratio-integer-column** — an integer column `(k, 1)` is the harmonic family
  bit for bit: three pairs (a phased sine saw; a cosine saw under an animated
  lowpass with a live count and a Nyquist cull; a fundamental that puts partial
  10 exactly on Nyquist), at integer, fractional and far-out (≥ 2⁴⁰ frames)
  clocks.
* **ratio-rational-exact** — a 13-partial chord with `Λ = 60` against a
  big-integer model (`⌊m_k·(inc·τ + off·2¹⁶) / 2³²⌋ mod 2³²`, no wrap, the
  fixed-point sine ported to `Int`), out to τ ≈ 2⁴⁴ frames with fractional τ;
  and exact recurrence: at a fundamental whose ring increment is `2³⁶`, the
  output at `τ + 4096` (the common period, 60 fundamental cycles) and at
  `τ + 2⁴⁰` equals the output at `τ` bit for bit, while `τ + 2048` does not.
* **ratio-free** — the inharmonic bell column (authored out of order) against
  an independent f64 reference in authored order: libm `sin` of the integer
  phase `⌊inc_k·τ⌋ mod 2³²`, `inc_k = trunc(r_k·f0·2³²/rate)`. Bound: the
  fixed-point sine's error. Near τ = 0 also against the ideal real frequency,
  bound `2π·n·2⁻³²` per partial added.
* **ratio-envelope** — the bell under a complex SVF lowpass: per-partial
  reference with `G` at the partial's true frequency `r_k·f0`. The same
  reference with `G` at `k·f0` (the sorted index) is printed and must miss.
* **ratio-cull** — across 40 fundamentals: the trip count is exactly
  `#{k : r_k·|f0| ≤ Nyquist}` (so no partial above Nyquist is evaluated), every
  family render matches the in-band reference with its soft edges, and a
  family wholly above Nyquist renders exact zeros.
* **ratio-cull-sparse** — the same on columns whose largest ratio is far above
  the partial count, f0 = 0 included.
* **ratio-refusals** — `Λ > 2¹⁶`, a free column without the Nyquist edge, a
  free ratio ≥ 2³¹, a weight law, and a mismatched empty column are refused.
* **ratio-realized-pitch** — the realized fundamental, read back from the
  kernel's phase far out, is within the documented bound of the ideal.
-/

namespace Tropical.Tropicaltest.SpectralRatios

open Tropical
open Tropical.Ir
open Tropical.EmitArrow
open Tropical.Testing.SpectralFixtures
open Tropical.Tropicaltest.Spectral

-- ── Harness ──────────────────────────────────────────────────────────────

/-- Render channels `0..channels-1` at raw Q48.16 clocks. Channel-major. -/
def renderTaus (plan : Tropical.Plan.FlatPlan) (channels : Nat) (taus : Array UInt64) :
    IO (Array (Array Float)) := do
  let rt ← Tropical.Ffi.Runtime.new (1024 : Nat).toUInt32
  Tropical.StagedLoad.load rt plan
  let chans := (Array.range channels).map UInt32.ofNat
  let mut out : Array (Array Float) := Array.replicate channels #[]
  let mut i := 0
  while i < taus.size do
    let block := taus.extract i (i + 1024)
    let values := decodeF64LE (← rt.renderTau block chans)
    for c in [0:channels] do
      out := out.modify c (· ++ values.extract (c * block.size) ((c + 1) * block.size))
    i := i + 1024
  pure out

/-- `len` integer clocks from sample `start`. -/
def wholeTaus (start len : Nat) : Array UInt64 :=
  (Array.range len).map fun i => ((start + i) * clockOne).toUInt64

/-- `len` clocks from sample `start` with a fraction that walks the whole
    16-bit field (`i·40503 mod 2¹⁶`, 40503 odd). -/
def fracTaus (start len : Nat) : Array UInt64 :=
  (Array.range len).map fun i => ((start + i) * clockOne + (i * 40503) % clockOne).toUInt64

def twoPi : Float := 6.283185307179586

/-- The fixed-point sine of `fixedSinCycSig` on `Int`: every intermediate is
    far below 2⁶³, so the i64 kernel and this model agree exactly
    (arithmetic shift = floor division). `p` is a Q32 phase in `[0, 2³²)`;
    the result is Q30. -/
def sinCycModel (p : Int) : Int :=
  let sh (a : Int) (s : Nat) : Int := Int.fdiv a (2 ^ s)
  let n := sh (p + 2 ^ 30) 31
  let r := p - n * 2 ^ 31
  let sign := 1 - 2 * Int.emod n 2
  let z := sh (r * r) 30
  let a6 := 61 - sh z 30
  let a5 := 3864 - sh (a6 * z) 30
  let a4 := 172272 - sh (a5 * z) 30
  let a3 := 5026995 - sh (a4 * z) 30
  let a2 := 85569306 - sh (a3 * z) 30
  let a1 := 693598668 - sh (a2 * z) 30
  let a0 := 1686629713 - sh (a1 * z) 30
  sign * sh (r * a0) 30

def intToFloat (x : Int) : Float :=
  if x ≥ 0 then x.toNat.toFloat else -((-x).toNat.toFloat)

/-- `trunc x` for a non-negative double, as a natural. -/
def truncNat (x : Float) : Nat := x.floor.toUInt64.toNat

def nyquistOf (rate : Float) : Float := rate * 0.5

/-- The family's soft edges at partial frequency `f ≥ 0` (standard widths):
    `smooth((N − f)·10/N) · smooth(f/20)`, `smooth(v) = t²(3 − 2t)`,
    `t = clamp(v, 0, 1)`. -/
def edgeOf (rate f : Float) : Float :=
  let smooth (v : Float) : Float :=
    let t := if v < 0.0 then 0.0 else if v > 1.0 then 1.0 else v
    t * t * (3.0 - 2.0 * t)
  let n := nyquistOf rate
  smooth ((n - f) * (10.0 / n)) * smooth (f * 0.05)

def sumAbs (xs : Array Float) : Float := xs.foldl (fun a x => a + x.abs) 0.0

-- ── ratio-integer-column ─────────────────────────────────────────────────

def harmonicColumn (count : Nat) : PartialRatios :=
  .rational ((Array.range count).map fun i => (i + 1, 1))

def runIntegerColumn (arena : Arena) : IO Bool := do
  let label := "ratio-integer-column"
  let built := compileMulti arena "ratio_integer_column" (do
    let clock ← clockLit
    let mut outs : Array Sig := #[]
    -- (A) a phased sine saw
    let a ← sawFamily (← litExact 110.0) (some (← litExact 0.3)) clock 24
    outs := outs.push (← emitTerm (.family a))
    outs := outs.push (← emitTerm (.family { a with ratios := harmonicColumn 24 }))
    -- (B) a cosine saw, culled at Nyquist (⌊22050/1234.5⌋ = 17) under a live
    -- count of 20 and an animated lowpass
    let b ← sawFamily (← litExact 1234.5) none clock 24 (wave := .cosine)
    let b := { b with liveCount := some (← lit 20) }
    let cutoff : ArrowTerm := .arrUn (fun clk => do
      expCutoffLfoSig (← lit 2000) (← lit 2) (← lit 7) clk) (.clk clock)
    let q3 ← lit 3
    let env := fun (fam : PartialFamily) =>
      normalize (svfEnvelope .lowpass cutoff (.konst q3) (.family fam))
    outs := outs.push (← emitTerm (env b))
    outs := outs.push (← emitTerm (env { b with ratios := harmonicColumn 24 }))
    -- (C) partial 10 exactly on Nyquist (2205·10 = 22050)
    let c ← sawFamily (← litExact 2205.0) none clock 24
    outs := outs.push (← emitTerm (.family c))
    outs := outs.push (← emitTerm (.family { c with ratios := harmonicColumn 24 }))
    -- (D) the saw of (A) on an UNREDUCED integer column (2k, 2): reduced
    -- before the lcm, so Λ = 1 and it is harmonic bit for bit
    outs := outs.push (← emitTerm (.family
      { a with ratios := .rational ((Array.range 24).map fun i => (2 * (i + 1), 2)) }))
    pure outs) 44100
  let plan ← match built with
    | .ok p => pure p
    | .error e => return ← failGate label s!"build: {firstLine e}"
  let windows : Array (String × Array UInt64) := #[
    ("τ0 = 0", wholeTaus 0 4096),
    ("τ0 = 1000003", wholeTaus 1000003 4096),
    ("fractional τ from 777", fracTaus 777 4096),
    ("fractional τ from 2^40 + 7", fracTaus (2 ^ 40 + 7) 4096)]
  let mut ok := true
  for (what, taus) in windows do
    let out ← renderTaus plan 7 taus
    let diffs := ((Array.range 3).map fun i => bitDiffCount out[2 * i]! out[2 * i + 1]!)
      |>.push (bitDiffCount out[0]! out[6]!)
    let audible := (Array.range 3).all fun i => energyOf out[2 * i]! > 0.0
    ok := ok && diffs.all (· == 0) && audible
    IO.println s!"        {what}: bit-diffs (saw, culled lowpass, partial on Nyquist, unreduced (2k, 2) saw) = {diffs}{if audible then "" else " (SILENT)"}"
  if ok then
    passGate label "integer column (k, 1) ≡ the harmonic family bit for bit: 3 pairs × 4 windows (integer, fractional, 2^40 frames) × 4096 samples"
  else
    failGate label "the integer column departs from the harmonic family"

-- ── ratio-rational-exact ─────────────────────────────────────────────────

def chordLcm : Nat := chordRatios.foldl (fun acc (_, d) => Nat.lcm acc d) 1

/-- The ring increment of a rational family, as the kernel computes it:
    `trunc(f0·2⁴⁸/(Λ·rate))` (E = 16). -/
def chordInc (f0 rate : Float) : Nat :=
  truncNat (f0 * 281474976710656.0 / (chordLcm.toFloat * rate))

/-- The phase offset on that ring: `trunc(phase·2⁴⁸/Λ)`. -/
def chordOff (phase : Float) : Nat :=
  truncNat (phase * 281474976710656.0 / chordLcm.toFloat)

/-- The big-integer model of the chord at raw clock `tau`: partial `k`'s Q32
    phase `⌊m_k·(inc·τ + off·2¹⁶) / 2³²⌋ mod 2³²`, unreduced before the floor;
    the sum in column order (ascending, as authored) of `(1/k)·sin`. -/
def chordModel (inc off : Nat) (tau : Nat) : Float := Id.run do
  let mut acc := 0.0
  for i in [0:chordRatios.size] do
    let (n, d) := chordRatios[i]!
    let m := n * (chordLcm / d)
    let phase : Nat := (m * (inc * tau + off * clockOne) / 2 ^ 32) % 2 ^ 32
    let s := sinCycModel (phase : Int)
    acc := acc + (1.0 / (i + 1).toFloat) * (intToFloat s / 1073741824.0)
  return acc

def runRationalExact (arena : Arena) : IO Bool := do
  let label := "ratio-rational-exact"
  let rate := 48000.0
  -- f0 = Λ·rate/4096 puts the ring increment at exactly 2³⁶: X recurs every
  -- 2⁴⁸/2³⁶ = 4096 samples, which is Λ = 60 fundamental cycles.
  let fNice := chordLcm.toFloat * rate / 4096.0
  let fPlain := 440.0
  let phasePlain := 0.37
  let built := compileMulti arena "ratio_rational" (do
    let clock ← clockLit
    let nice ← chordFamily (← litExact fNice) none clock
    let plain ← chordFamily (← litExact fPlain) (some (← litExact phasePlain)) clock
    pure #[← emitTerm (.family nice), ← emitTerm (.family plain)]) 48000
  let plan ← match built with
    | .ok p => pure p
    | .error e => return ← failGate label s!"build: {firstLine e}"
  let incNice := chordInc fNice rate
  let incPlain := chordInc fPlain rate
  let offPlain := chordOff phasePlain
  IO.println s!"        Λ = {chordLcm}; f0 = {fNice} Hz → inc = {incNice} (2^36 = {2 ^ 36}); f0 = 440 Hz → inc = {incPlain}"
  let mut ok := incNice == 2 ^ 36
  let peak := sumAbs ((Array.range chordRatios.size).map fun i => 1.0 / (i + 1).toFloat)
  -- (a) the model, far out
  let windows : Array (String × Array UInt64) := #[
    ("τ0 = 0", wholeTaus 0 2048),
    ("fractional τ from 2^40 + 12345", fracTaus (2 ^ 40 + 12345) 2048),
    ("fractional τ from 2^44 + 3", fracTaus (2 ^ 44 + 3) 2048)]
  let mut worst := 0.0
  let mut bitExact := 0
  let mut total := 0
  for (what, taus) in windows do
    let out ← renderTaus plan 2 taus
    let refNice := taus.map fun t => chordModel incNice 0 t.toNat
    let refPlain := taus.map fun t => chordModel incPlain offPlain t.toNat
    let dN := maxAbsDiff out[0]! refNice
    let dP := maxAbsDiff out[1]! refPlain
    worst := max worst (max dN dP)
    let exactHere := (taus.size - bitDiffCount out[0]! refNice) + (taus.size - bitDiffCount out[1]! refPlain)
    bitExact := bitExact + exactHere
    total := total + 2 * taus.size
    IO.println s!"        {what}: max |kernel − model| = {sci dN} (f0 = {fNice}), {sci dP} (f0 = 440, phase 0.37)"
  -- one Q32 phase step moves a partial's sine by ≥ ~1.5e-9·w most of the time;
  -- 1e-12·Σ|w| leaves only the last-place rounding of the f64 sum.
  let tol := 1e-12 * peak
  ok := ok && worst ≤ tol
  IO.println s!"        model agreement: {bitExact}/{total} samples bit-identical; worst {sci worst} ≤ {sci tol}"
  -- (b) exact recurrence after the common period, also 2^40 frames on
  let base := fracTaus 977 4096
  let shifted := fun (d : Nat) => base.map (· + (d * clockOne).toUInt64)
  let y0 ← renderTaus plan 1 base
  let yP ← renderTaus plan 1 (shifted 4096)
  let yFar ← renderTaus plan 1 (shifted (2 ^ 40))
  let yHalf ← renderTaus plan 1 (shifted 2048)
  let dP := bitDiffCount y0[0]! yP[0]!
  let dFar := bitDiffCount y0[0]! yFar[0]!
  let dHalf := bitDiffCount y0[0]! yHalf[0]!
  IO.println s!"        recurrence: τ + 4096 → {dP} bit-diffs, τ + 2^40 → {dFar}; τ + 2048 → {dHalf} of 4096 differ (the partials drift within the period)"
  ok := ok && dP == 0 && dFar == 0 && dHalf > 2000 && energyOf y0[0]! > 0.0
  if ok then
    passGate label s!"Λ = 60 chord ≡ big-integer model (worst {sci worst}, τ to 2^44, fractional); recurs bit for bit every 4096 samples (60 cycles) and 2^40 frames on"
  else
    failGate label "the rational family departs from the model or does not recur"

-- ── ratio-free ───────────────────────────────────────────────────────────

/-- The bell's per-partial increments, as the kernel computes them:
    `trunc(r_k·(f0·2³²/rate))`, authored order. -/
def bellIncs (f0 rate : Float) (ratios : Array Float := bellRatios) : Array Nat :=
  ratios.map fun r => truncNat (r * (f0 * 4294967296.0 / rate))

/-- Independent f64 reference, authored order: `Σ w_k·sin(2π·θ_k)` with the
    integer phase `θ_k = ⌊inc_k·τ / 2¹⁶⌋ mod 2³²` and libm `sin`. `edges` and
    `gain` (the complex gain at the partial's frequency) are optional. -/
def bellReference (f0 rate : Float) (tau : Nat)
    (gain : Float → Float × Float := fun _ => (1.0, 0.0)) (edges : Bool := false)
    (ratios : Array Float := bellRatios) (weights : Array Float := bellWeights) :
    Float := Id.run do
  let incs := bellIncs f0 rate ratios
  let mut acc := 0.0
  for i in [0:ratios.size] do
    let r := ratios[i]!
    let f := r * f0
    if edges && f > nyquistOf rate then continue
    let theta := twoPi * ((incs[i]! * tau / clockOne) % 2 ^ 32).toFloat / 4294967296.0
    let (gre, gim) := gain f
    let e := if edges then edgeOf rate f else 1.0
    acc := acc + weights[i]! * e * (gre * Float.sin theta + gim * Float.cos theta)
  return acc

/-- The fixed-point sine's largest error against libm `sin`, MEASURED (not
    proved) over a dense sweep of 2¹⁸ Q32 phases spread across the whole
    circle (a stride coprime to 2³², plus the quarter points). -/
def sinMeasuredError : Float := Id.run do
  let mut worst := 0.0
  let probes : Array Nat := ((Array.range 262144).map fun i => (i * 16381 * 997) % 2 ^ 32)
    ++ #[0, 2 ^ 30, 2 ^ 31, 3 * 2 ^ 30, 2 ^ 32 - 1]
  for p in probes do
    let fixed := intToFloat (sinCycModel (p : Int)) / 1073741824.0
    let exact := Float.sin (twoPi * p.toFloat / 4294967296.0)
    worst := max worst (fixed - exact).abs
  return worst

/-- The per-unit-weight bound the free, envelope and cull gates use: twice the
    measured sine error (margin for phases the sweep missed), plus 1e-15 for
    the f64 sums. -/
def sinBound : Float := 2.0 * sinMeasuredError + 1e-15

def runFree (arena : Arena) : IO Bool := do
  let label := "ratio-free"
  let rate := 48000.0
  let f0 := 440.0
  let built := compileMulti arena "ratio_free" (do
    let fam ← bellFamily (← litExact f0) (← clockLit)
    pure #[← emitTerm (.family fam)]) 48000
  let plan ← match built with
    | .ok p => pure p
    | .error e => return ← failGate label s!"build: {firstLine e}"
  let tol := sinBound * sumAbs bellWeights
  let mut ok := true
  let mut worst := 0.0
  for (what, taus) in #[("τ0 = 0", wholeTaus 0 4096),
      ("fractional τ from 2^40 + 5", fracTaus (2 ^ 40 + 5) 4096)] do
    let out ← renderTaus plan 1 taus
    let ref := taus.map fun t => bellReference f0 rate t.toNat
    let d := maxAbsDiff out[0]! ref
    worst := max worst d
    ok := ok && energyOf out[0]! > 0.0
    IO.println s!"        {what}: max |kernel − f64 reference on the integer phase| = {sci d}"
  ok := ok && worst ≤ tol
  -- against the ideal real frequency near τ = 0: the increment's truncation
  -- costs < 2⁻³² cycles per sample per partial
  let n := 4096
  let out ← renderTaus plan 1 (wholeTaus 0 n)
  let ideal := (Array.range n).map fun s => Id.run do
    let mut acc := 0.0
    for i in [0:bellRatios.size] do
      let cycles := bellRatios[i]! * f0 * s.toFloat / rate
      acc := acc + bellWeights[i]! * Float.sin (twoPi * (cycles - cycles.floor))
    return acc
  let dIdeal := maxAbsDiff out[0]! ideal
  let idealTol := tol + sumAbs bellWeights * twoPi * n.toFloat / 4294967296.0
  IO.println s!"        vs the ideal real frequencies over {n} samples: {sci dIdeal} ≤ {sci idealTol}"
  ok := ok && dIdeal ≤ idealTol
  if ok then
    passGate label s!"9-partial bell (unsorted column) ≡ f64 reference on the integer phase within {sci worst} ≤ {sci tol} at τ0 = 0 and 2^40; ≡ ideal frequencies within the 2^-32-per-sample increment bound"
  else
    failGate label "the free family departs from the f64 reference"

-- ── ratio-envelope ───────────────────────────────────────────────────────

def lowpassGain (fc q : Float) (f : Float) : Float × Float :=
  let x := f / fc
  let re := 1.0 - x * x
  let im := x / q
  let den := re * re + im * im
  (re / den, -im / den)

def runEnvelope (arena : Arena) : IO Bool := do
  let label := "ratio-envelope"
  let rate := 48000.0
  let f0 := 440.0
  let fc := 700.0
  let q := 4.0
  let built := compileMulti arena "ratio_envelope" (do
    let fam ← bellFamily (← litExact f0) (← clockLit)
    let term := svfEnvelope .lowpass (.konst (← litExact fc)) (.konst (← litExact q)) (.family fam)
    pure #[← emitTerm (normalize term)]) 48000
  let plan ← match built with
    | .ok p => pure p
    | .error e => return ← failGate label s!"build: {firstLine e}"
  let taus := fracTaus (2 ^ 40 + 99) 4096
  let out ← renderTaus plan 1 taus
  let ref := taus.map fun t => bellReference f0 rate t.toNat (lowpassGain fc q)
  -- the wrong reference: G at k·f0, k the partial's place in the sorted column
  let sortedRatios := bellRatios.qsort (· < ·)
  let indexGain := fun (f : Float) => Id.run do
    let mut g := (1.0, 0.0)
    for i in [0:sortedRatios.size] do
      if sortedRatios[i]! * f0 == f then g := lowpassGain fc q ((i + 1).toFloat * f0)
    return g
  let wrong := taus.map fun t => bellReference f0 rate t.toNat indexGain
  let d := maxAbsDiff out[0]! ref
  let dWrong := maxAbsDiff out[0]! wrong
  -- the lowpass peak: |G|max = Q/√(1 − 1/(4Q²)) (≈ 4.03 at Q 4); the gain
  -- rotates the (sin, cos) error pair, so it can grow by √2
  let gMax := q / Float.sqrt (1.0 - 1.0 / (4.0 * q * q))
  let tol := sinBound * gMax * Float.sqrt 2.0 * sumAbs bellWeights
  IO.println s!"        G at r_k·f0: max |kernel − reference| = {sci d} ≤ {sci tol}; G at k·f0 (the index) would miss by {sci dWrong}"
  if d ≤ tol && dWrong > 1e-3 && energyOf out[0]! > 0.0 then
    passGate label s!"lowpass (fc 700 Hz, Q 4) on the bell: each partial's gain at its true frequency, {sci d} from the per-partial reference"
  else
    failGate label "the envelope is not evaluated at the partials' frequencies"

-- ── ratio-cull ───────────────────────────────────────────────────────────

def cullFundamentals : Array Float :=
  -- log-spaced 50 Hz → 50 kHz (the top few put every partial above Nyquist),
  -- plus fundamentals that put a partial exactly on Nyquist (22050/r, r = 2, 3)
  ((Array.range 38).map fun i => 50.0 * Float.pow 1000.0 (i.toFloat / 37.0))
    ++ #[22050.0 / 2.0, 22050.0 / 3.0]

def runCull (arena : Arena) : IO Bool := do
  let label := "ratio-cull"
  let rate := 44100.0
  let nyq := nyquistOf rate
  let sorted := bellRatios.qsort (· < ·)
  let trips := compileMulti arena "ratio_cull_trips" (do
    let table ← arr (← sorted.mapM litExact)
    let nyquist ← mul (← sampleRate) (← lit 5 1)
    cullFundamentals.mapM fun f => do
      toFloatE (← ratioTripCount table sorted.size sorted.back! (← absE (← litExact f)) nyquist)) 44100
  let renders := compileMulti arena "ratio_cull_renders" (do
    let clock ← clockLit
    cullFundamentals.mapM fun f => do
      emitTerm (.family (← bellFamily (← litExact f) clock))) 44100
  let (tripPlan, renderPlan) ← match trips, renders with
    | .ok a, .ok b => pure (a, b)
    | .error e, _ | _, .error e => return ← failGate label s!"build: {firstLine e}"
  let n := cullFundamentals.size
  let taus := fracTaus (2 ^ 40 + 1) 1024
  let out := (← renderTaus tripPlan n taus) ++ (← renderTaus renderPlan n taus)
  let tol := sinBound * sumAbs bellWeights
  let mut ok := true
  let mut tripsShown : Array String := #[]
  let mut worst := 0.0
  let mut silentAbove := 0
  for i in [0:n] do
    let f := cullFundamentals[i]!
    -- The trip count divides once (Nyquist/|f0|) where this multiplies; the two
    -- can disagree at the last ulp when r·|f0| is within ~4e-12 of Nyquist.
    -- Such a partial has edge weight 0, so the output is unaffected; the
    -- expectation accepts either count in that band.
    let want := (sorted.filter fun r => r * f ≤ nyq).size
    let wantLo := (sorted.filter fun r => r * f ≤ nyq * (1.0 - 1e-12)).size
    let wantHi := (sorted.filter fun r => r * f ≤ nyq * (1.0 + 1e-12)).size
    let got := out[i]![0]!
    ok := ok && wantLo.toFloat ≤ got && got ≤ wantHi.toFloat
    if i % 6 == 0 || got != want.toFloat then
      tripsShown := tripsShown.push s!"{fmt f 1} Hz→{got}/{want}"
    let ref := taus.map fun t => bellReference f rate t.toNat (edges := true)
    let d := maxAbsDiff out[n + i]! ref
    worst := max worst d
    if want == 0 then
      let zeros := out[n + i]!.all (· == 0.0)
      ok := ok && zeros
      if zeros then silentAbove := silentAbove + 1
  ok := ok && worst ≤ tol
  IO.println s!"        trip (got/want): {tripsShown}"
  IO.println s!"        renders vs in-band reference with soft edges: worst {sci worst} ≤ {sci tol}; {silentAbove} fundamentals wholly above Nyquist render exact zeros"
  if ok then
    passGate label s!"{n} fundamentals: trip = #(r_k·f0 ≤ Nyquist) exactly over an unsorted column; renders ≡ in-band reference ({sci worst})"
  else
    failGate label "the cull admits a partial above Nyquist or drops one below"

/-- Sparse columns, where the largest ratio is far above the partial count
    (the case a floor of `Nyquist/(K+1)` on `|f0|` would cull wrongly): the
    trip count against `#{k : r_k·|f0| ≤ Nyquist}` (all of them at f0 = 0),
    and the family renders against the in-band reference, at 48 kHz. -/
def sparseCases : Array (Array Float × Float) := #[
  (#[1.0, 2.76, 5.4], 440.0), (#[1.0, 20.0], 100.0), (#[1.0, 20.0], 1000.0),
  (#[1.0, 20.0], 1200.0), (#[3.0, 40.0, 0.5], 37.0), (#[1.0, 2.76, 5.4], 0.0),
  (#[1.0, 1000.0], 23.9), (#[1.0, 1000.0], 24.0), (#[1.0, 1000.0], 24.1)]

def runSparseCull (arena : Arena) : IO Bool := do
  let label := "ratio-cull-sparse"
  let rate := 48000.0
  let nyq := nyquistOf rate
  let built := compileMulti arena "ratio_cull_sparse" (do
    let clock ← clockLit
    let nyquist ← mul (← sampleRate) (← lit 5 1)
    let mut outs : Array Sig := #[]
    for (ratios, f) in sparseCases do
      let sorted := ratios.qsort (· < ·)
      let table ← arr (← sorted.mapM litExact)
      outs := outs.push (← toFloatE (← ratioTripCount table sorted.size sorted.back!
        (← absE (← litExact f)) nyquist))
    for (ratios, f) in sparseCases do
      let ws ← ratios.mapM fun _ => litExact 1.0
      -- the DC edge is off so the f0 = 0 case evaluates its partials (each
      -- contributes sin 0 = 0); the Nyquist edge stays on
      let freq ← litExact f
      let fam : PartialFamily :=
        { name := "sparse", freq, clk := clock, count := ratios.size, weights := .table ws,
          ratios := .free ratios, edges := { dc := .off } }
      outs := outs.push (← emitTerm (.family fam))
    pure outs) 48000
  let plan ← match built with
    | .ok p => pure p
    | .error e => return ← failGate label s!"build: {firstLine e}"
  let n := sparseCases.size
  let taus := fracTaus (2 ^ 40 + 3) 1024
  let out ← renderTaus plan (2 * n) taus
  let mut ok := true
  let mut rows : Array String := #[]
  let mut worst := 0.0
  for i in [0:n] do
    let (ratios, f) := sparseCases[i]!
    -- (exact agreement is not relied on within ~ulp of Nyquist; see ratio-cull)
    let want := (ratios.filter fun r => r * f ≤ nyq).size
    let wantLo := (ratios.filter fun r => r * f ≤ nyq * (1.0 - 1e-12)).size
    let wantHi := (ratios.filter fun r => r * f ≤ nyq * (1.0 + 1e-12)).size
    let got := out[i]![0]!
    ok := ok && wantLo.toFloat ≤ got && got ≤ wantHi.toFloat
    rows := rows.push s!"{ratios}@{fmt f 1}Hz→{got}/{want}"
    -- the reference with the DC edge off: Nyquist edge only
    let ref := taus.map fun t => Id.run do
      let incs := bellIncs f rate ratios
      let mut acc := 0.0
      for j in [0:ratios.size] do
        let fr := ratios[j]! * f
        if fr > nyq then continue
        let v := (nyq - fr) * (10.0 / nyq)
        let tt := if v < 0.0 then 0.0 else if v > 1.0 then 1.0 else v
        let theta := twoPi * ((incs[j]! * t.toNat / clockOne) % 2 ^ 32).toFloat / 4294967296.0
        acc := acc + tt * tt * (3.0 - 2.0 * tt) * Float.sin theta
      return acc
    worst := max worst (maxAbsDiff out[n + i]! ref)
  let tol := sinBound * 3.0
  ok := ok && worst ≤ tol
  IO.println s!"        trip (got/want): {rows}"
  IO.println s!"        renders vs in-band reference: worst {sci worst} ≤ {sci tol}"
  if ok then
    passGate label s!"{n} sparse columns (r_max ≫ K, f0 = 0 included): trip = #(r_k·f0 ≤ Nyquist) exactly; renders ≡ reference"
  else
    failGate label "a sparse column's cull drops a partial below Nyquist or keeps one above"

-- ── ratio-refusals ───────────────────────────────────────────────────────

/-- What is refused at assemble time, each with its message: a common period
    above 2¹⁶ (the review's near-unity fractions; denominators 1 and 65537),
    complex weights, a weight law, a 0 Hz term on a sine family (the family's
    own check, ahead of the ratio dispatch), and an empty family whose column
    does not match its count (validated before the empty family returns its
    0 Hz term). Accepted: Λ = 2¹⁶ exactly, an unreduced `2/131072` (reduced
    first, so Λ = 2¹⁶), and a free column with the Nyquist edge off. -/
def runRefusals (arena : Arena) : IO Bool := do
  let label := "ratio-refusals"
  let attempt : String → (Sig → Clock → BuildM PartialFamily) →
      Except String Tropical.Plan.FlatPlan := fun name mk =>
    compileMulti arena name (do
      let fam ← mk (← litExact 440.0) (← clockLit)
      pure #[← emitTerm (.family fam)]) 48000
  let ones : Nat → BuildM PartialWeights := fun n => do
    pure (PartialWeights.table (← (Array.range n).mapM fun _ => litExact 1.0))
  let cases : Array (String × (Sig → Clock → BuildM PartialFamily) × Option String) := #[
    ("near-unity fractions (Λ ≈ 9.4e11)", fun f c => do
      let rs := #[(1, 1), (998, 997), (992, 991), (984, 983), (978, 977)]
      pure { freq := f, clk := c, count := 5, weights := (← ones 5), ratios := .rational rs },
      some "exceeds 65536"),
    ("denominators 1 and 65537", fun f c => do
      pure { freq := f, clk := c, count := 2, weights := (← ones 2), ratios := .rational #[(1, 1), (1, 65537)] },
      some "exceeds 65536"),
    ("Λ = 2^16 exactly", fun f c => do
      pure { freq := f, clk := c, count := 2, weights := (← ones 2), ratios := .rational #[(1, 1), (3, 65536)] },
      none),
    ("unreduced (2, 131072) = 1/65536", fun f c => do
      pure { freq := f, clk := c, count := 2, weights := (← ones 2), ratios := .rational #[(1, 1), (2, 131072)] },
      none),
    ("free column, Nyquist edge off", fun f c => do
      pure { freq := f, clk := c, count := 2, weights := (← ones 2), ratios := .free #[1.0, 2.5],
             edges := { nyquist := .off } },
      none),
    ("complex weights", fun f c => do
      pure { freq := f, clk := c, count := 2, ratios := .free #[1.0, 2.5],
             weights := .complex #[] (fun _ _ _ _ => do pure (← lit 1, ← lit 0)) },
      some "not complex weights"),
    ("a 0 Hz term on a sine family", fun f c => do
      pure { freq := f, clk := c, count := 2, weights := (← ones 2), ratios := .free #[1.0, 2.5],
             dc := some (← lit 1), edges := { dc := .off } },
      some "a 0 Hz term needs the cosine projection"),
    ("a weight law", fun f c => do
      pure { freq := f, clk := c, count := 2, weights := .law (fun k => pure k), ratios := .free #[1.0, 2.5] },
      some "not a weight law"),
    ("empty family, one ratio", fun f c => do
      pure { freq := f, clk := c, count := 0, weights := .table #[], ratios := .rational #[(1, 2)] },
      some "1 ratios for 0 partials")]
  let mut ok := true
  for (what, mk, needle?) in cases do
    match attempt s!"refusal_{what.length}" mk, needle? with
    | .ok _, none => IO.println s!"        {what}: accepted"
    | .error e, some needle =>
      let hit := (e.splitOn needle).length > 1
      ok := ok && hit
      IO.println s!"        {what}: refused{if hit then "" else s!" WITH THE WRONG MESSAGE: {firstLine e}"}"
    | .ok _, some _ => ok := false; IO.println s!"        {what}: ACCEPTED (expected a refusal)"
    | .error e, none => ok := false; IO.println s!"        {what}: REFUSED: {firstLine e}"
  if ok then passGate label "Λ > 2^16, complex weights, a weight law, a 0 Hz term on a sine family and a mismatched empty column are refused by name; Λ = 2^16, an unreduced 2/131072 and a free column with the Nyquist edge off are accepted"
  else failGate label "a refusal or an acceptance is wrong"

-- ── ratio-realized-pitch ─────────────────────────────────────────────────

/-- The realized fundamental against the ideal frequency, measured from the
    kernel: a family whose only weighted partial has ratio 1 is rendered as
    sine and cosine, its phase read back with `atan2` at sample `n`, and
    compared with the exact `frac(440·n/48000)`. The documented bounds:
    rational `Λ·rate/2⁴⁸` Hz (phase `Λ·n/2⁴⁸` cycles), free `rate/2³²` Hz
    (`n/2³²` cycles), each with 4% slack for the double that computes the
    increment, plus 1e-8 cycles for the sine and the floors. The rational
    regime's bound is checked at `Λ = 60` (n = 2⁴⁰) and at the cap
    `Λ = 2¹⁶` (n = 2³⁰). -/
def runRealizedPitch (arena : Arena) : IO Bool := do
  let label := "ratio-realized-pitch"
  let cases : Array (String × PartialRatios × Nat × Float) := #[
    ("rational, Λ = 60", .rational #[(1, 1), (1, 60)], 2 ^ 40, 60.0 / 281474976710656.0),
    ("rational, Λ = 2^16", .rational #[(1, 1), (1, 65536)], 2 ^ 30, 65536.0 / 281474976710656.0),
    ("free", .free #[1.0, 0.01], 2 ^ 30, 1.0 / 4294967296.0)]
  let built := compileMulti arena "ratio_pitch" (do
    let clock ← clockLit
    let mut outs : Array Sig := #[]
    for (_, ratios, _, _) in cases do
      for wave in #[PartialWave.sine, .cosine] do
        let ws := #[← litExact 1.0, ← litExact 0.0]
        let freq ← litExact 440.0
        let fam : PartialFamily :=
          { name := "pitch", freq, clk := clock, count := 2, weights := .table ws, ratios, wave }
        outs := outs.push (← emitTerm (.family fam))
    pure outs) 48000
  let plan ← match built with
    | .ok p => pure p
    | .error e => return ← failGate label s!"build: {firstLine e}"
  let mut ok := true
  for i in [0:cases.size] do
    let (what, _, n, perSample) := cases[i]!
    let taus := wholeTaus n 4
    let out ← renderTaus plan (2 * cases.size) taus
    let mut worst := 0.0
    for j in [0:taus.size] do
      let s := out[2 * i]![j]!
      let c := out[2 * i + 1]![j]!
      let meas := Float.atan2 s c / twoPi
      let ideal := ((440 * (n + j)) % 48000).toFloat / 48000.0
      let d0 := meas - ideal
      let d := (d0 - d0.round).abs
      worst := max worst d
    let bound := perSample * (n + 4).toFloat * 1.04 + 1e-8
    ok := ok && worst ≤ bound
    IO.println s!"        {what}: phase vs ideal 440 Hz at n = {n}: {sci worst} cycles ≤ {sci bound} (frequency error ≤ {sci (perSample * 48000.0)} Hz)"
  if ok then passGate label "the realized fundamental is within the documented bound: rational Λ·rate/2^48 Hz (Λ = 60 and 2^16), free rate/2^32 Hz"
  else failGate label "a realized fundamental is outside its documented bound"

-- ── ratio-dc ───────────────────────────────────────────────────────────

/-- The 0 Hz term on a ratio family (as on a harmonic one): a cosine chord
    with `dc = 0.25` under a lowpass minus the same chord without it is
    `0.25·Re G(0) = 0.25`; an empty rational family with `dc` plays exactly
    `0.25·Re G(0)`. -/
def runDc (arena : Arena) : IO Bool := do
  let label := "ratio-dc"
  let built := compileMulti arena "ratio_dc" (do
    let clock ← clockLit
    let a0 ← litExact 0.25
    let base ← chordFamily (← litExact 440.0) none clock
    let base := { base with wave := .cosine, edges := { dc := .off } }
    let fc ← litExact 900.0
    let q ← litExact 2.0
    let lp := fun (fam : PartialFamily) =>
      normalize (svfEnvelope .lowpass (.konst fc) (.konst q) (.family fam))
    let withDc ← emitTerm (lp { base with dc := some a0 })
    let without ← emitTerm (lp base)
    let empty : PartialFamily :=
      { base with count := 0, weights := .table #[], ratios := .rational #[], dc := some a0 }
    let emptyOut ← emitTerm (lp empty)
    pure #[withDc, without, emptyOut]) 48000
  let plan ← match built with
    | .ok p => pure p
    | .error e => return ← failGate label s!"build: {firstLine e}"
  let out ← renderTaus plan 3 (fracTaus (2 ^ 40 + 17) 2048)
  let diff := (Array.range 2048).foldl (fun m i =>
    max m ((out[0]![i]! - out[1]![i]! - 0.25).abs)) 0.0
  let emptyOk := out[2]!.all (· == 0.25)
  IO.println s!"        (chord + 0.25 at 0 Hz) − chord − 0.25: max {sci diff}; empty family with the term: {if emptyOk then "exactly 0.25" else "WRONG"}"
  if diff ≤ 1e-14 && emptyOk && energyOf out[1]! > 0.0 then
    passGate label "a 0 Hz term on a rational family adds a0·Re G(0) outside the region; an empty one plays it alone"
  else
    failGate label "the 0 Hz term is lost or wrong on a ratio family"

def runRatios (arena : Arena) : IO (Nat × Nat) := do
  let results := #[
    ← runIntegerColumn arena,
    ← runRationalExact arena,
    ← runFree arena,
    ← runEnvelope arena,
    ← runCull arena,
    ← runSparseCull arena,
    ← runRefusals arena,
    ← runRealizedPitch arena,
    ← runDc arena]
  pure (results.size, results.filter (!·) |>.size)

end Tropical.Tropicaltest.SpectralRatios
