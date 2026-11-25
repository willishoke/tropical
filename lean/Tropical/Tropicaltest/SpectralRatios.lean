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
    let out ← renderTaus plan 6 taus
    let diffs := (Array.range 3).map fun i => bitDiffCount out[2 * i]! out[2 * i + 1]!
    let audible := (Array.range 3).all fun i => energyOf out[2 * i]! > 0.0
    ok := ok && diffs.all (· == 0) && audible
    IO.println s!"        {what}: bit-diffs (saw, culled lowpass, partial on Nyquist) = {diffs}{if audible then "" else " (SILENT)"}"
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
def bellIncs (f0 rate : Float) : Array Nat :=
  bellRatios.map fun r => truncNat (r * (f0 * 4294967296.0 / rate))

/-- Independent f64 reference, authored order: `Σ w_k·sin(2π·θ_k)` with the
    integer phase `θ_k = ⌊inc_k·τ / 2¹⁶⌋ mod 2³²` and libm `sin`. `edges` and
    `gain` (the complex gain at the partial's frequency) are optional. -/
def bellReference (f0 rate : Float) (tau : Nat)
    (gain : Float → Float × Float := fun _ => (1.0, 0.0)) (edges : Bool := false) :
    Float := Id.run do
  let incs := bellIncs f0 rate
  let mut acc := 0.0
  for i in [0:bellRatios.size] do
    let r := bellRatios[i]!
    let f := r * f0
    if edges && f > nyquistOf rate then continue
    let theta := twoPi * ((incs[i]! * tau / clockOne) % 2 ^ 32).toFloat / 4294967296.0
    let (gre, gim) := gain f
    let e := if edges then edgeOf rate f else 1.0
    acc := acc + bellWeights[i]! * e * (gre * Float.sin theta + gim * Float.cos theta)
  return acc

/-- The fixed-point sine's error bound used by the free and envelope gates,
    per unit weight (it is the Q30 polynomial's, measured well below this). -/
def sinBound : Float := 1e-8

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
  let gMax := 4.0  -- |G| ≤ Q·(1 + small) for the lowpass at Q 4 near fc
  let tol := sinBound * gMax * sumAbs bellWeights
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
      toFloatE (← ratioTripCount table sorted.size (← absE (← litExact f)) nyquist)) 44100
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
    let want := (sorted.filter fun r => r ≤ nyq / f).size
    let got := out[i]![0]!
    ok := ok && got == want.toFloat
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

def runRatios (arena : Arena) : IO (Nat × Nat) := do
  let results := #[
    ← runIntegerColumn arena,
    ← runRationalExact arena,
    ← runFree arena,
    ← runEnvelope arena,
    ← runCull arena]
  pure (results.size, results.filter (!·) |>.size)

end Tropical.Tropicaltest.SpectralRatios
