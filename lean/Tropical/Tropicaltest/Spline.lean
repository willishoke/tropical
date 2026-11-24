import Tropical.Session.Compile
import Tropical.EmitArrow.Spline
import Tropical.Tropicaltest.Spectral

/-!
# Spline gates: a uniform cubic spline read by a phase

`lean/Tropical/EmitArrow/Spline.lean` is the module these gates back.

* **spline-oracle** — the JIT against an independent f64 reference, for both
  bases, both boundaries and K = 4, 7, 16, over 4096 hashed phases and 4096
  grid phases each (the grid hits the knots, and in clamp mode phase 0, phase
  1 and both clamped sides). The reference finds the segment by float `floor`
  on `u·n` and weighs the window by the cardinal basis functions; in clamp
  mode it evaluates phase 1 as segment `K − 2` at `t = 1`, where the kernel
  reads segment `K − 1` at `t = 0`.
* **spline-seam** — wrap mode. (1) Rotating the control values by one moves
  the curve by one knot, bit for bit: the seam of the rotated spline renders
  exactly as an interior knot of the original (K = 8, both bases). (2) Value
  and first derivative are continuous across phase 1 → 0, by one-sided
  finite differences from each side (K = 7 and 8, both bases).
* **spline-live** — a `PeriodicCatmullRom8` patch whose control values are
  `param:` slots. A `set_param` on the loaded runtime changes the next buffer
  of the audio kernel to the reference at the new value, with no recompile.
-/

namespace Tropical.Tropicaltest.Spline

open Tropical
open Tropical.Ir
open Tropical.EmitArrow
open Tropical.Tropicaltest.Spectral (compileMulti renderAt range maxAbsDiff peakOf sci)

-- ── The reference ────────────────────────────────────────────────────────

/-- The cardinal weights of the four window values at `t`. -/
def refWeights (basis : SplineBasis) (t : Float) : Array Float :=
  let t2 := t * t
  let t3 := t2 * t
  if basis.scale == 2 then
    #[(-t3 + 2 * t2 - t) / 2, (3 * t3 - 5 * t2 + 2) / 2,
      (-3 * t3 + 4 * t2 + t) / 2, (t3 - t2) / 2]
  else
    let u := 1 - t
    #[u * u * u / 6, (3 * t3 - 6 * t2 + 4) / 6,
      (-3 * t3 + 3 * t2 + 3 * t + 1) / 6, t3 / 6]

/-- `d/dt` of `refWeights`. -/
def refWeightsDt (basis : SplineBasis) (t : Float) : Array Float :=
  let t2 := t * t
  if basis.scale == 2 then
    #[(-3 * t2 + 4 * t - 1) / 2, (9 * t2 - 10 * t) / 2,
      (-9 * t2 + 8 * t + 1) / 2, (3 * t2 - 2 * t) / 2]
  else
    let u := 1 - t
    #[-u * u / 2, (9 * t2 - 12 * t) / 6, (-9 * t2 + 6 * t + 3) / 6, t2 / 2]

/-- Control `m` of the (wrapped or end-extrapolated) sequence. -/
def refControl (boundary : SplineBoundary) (cs : Array Float) (m : Int) : Float :=
  let k := cs.size
  match boundary with
  | .wrap => cs[(m % (k : Int)).toNat]!
  | .clamp =>
    if m < 0 then 2.0 * cs[0]! - cs[1]!
    else if m ≥ k then 2.0 * cs[k - 1]! - cs[k - 2]!
    else cs[m.toNat]!

/-- The segment and local coordinate of an integer phase, by float floor. -/
def refSegment (boundary : SplineBoundary) (k : Nat) (phase : Int) : Int × Float :=
  let cycle : Float := 4294967296.0
  match boundary with
  | .wrap =>
    let p := phase % 4294967296
    let x := (Float.ofInt p / cycle) * k.toFloat
    let i := x.floor
    (Int.ofNat i.toUInt64.toNat, x - i)
  | .clamp =>
    let u := max 0.0 (min 1.0 (Float.ofInt phase / cycle))
    let n := (k - 1).toFloat
    let x := u * n
    let i := min x.floor (n - 1)
    (Int.ofNat i.toUInt64.toNat, x - i)

def refEval (basis : SplineBasis) (boundary : SplineBoundary) (cs : Array Float)
    (phase : Int) : Float := Id.run do
  let (i, t) := refSegment boundary cs.size phase
  let w := refWeights basis t
  let mut v := 0.0
  for j in [0:4] do
    v := v + w[j]! * refControl boundary cs (i - 1 + j)
  return v

/-- `dv/dt` on segment `i` at `t`. -/
def refEvalDt (basis : SplineBasis) (boundary : SplineBoundary) (cs : Array Float)
    (i : Int) (t : Float) : Float := Id.run do
  let w := refWeightsDt basis t
  let mut v := 0.0
  for j in [0:4] do
    v := v + w[j]! * refControl boundary cs (i - 1 + j)
  return v

-- ── Fixtures ─────────────────────────────────────────────────────────────

/-- `count` control values `m/64`, `m ∈ [−64, 64]` from a fixed LCG: exact in
    binary and in the decimal literal. -/
def controlsOf (seed count : Nat) : Array Int := Id.run do
  let mut x := seed
  let mut out := #[]
  for _ in [0:count] do
    x := (x * 1103515245 + 12345) % 2147483648
    out := out.push (Int.ofNat (x / 65536 % 129) - 64)
  return out

def controlFloat (m : Int) : Float := Float.ofInt m / 64.0

def controlSig (m : Int) : BuildM Sig := lit (m * 15625) 6

def hashMul : Nat := 2654435761

/-- The phase laws over the sample index `n`, as kernel signals and as
    integers. `hashed` scatters phases over the cycle; `grid` steps by 2²⁴
    (256 phases per cycle). In clamp mode both laws span `[−2³⁰, 2³² + 2³⁰)`
    and `[−2³¹, 2³² + 2³¹)`, so both clamped sides are exercised. -/
inductive PhaseLaw | hashed | grid
deriving BEq, Inhabited

def phaseInt (law : PhaseLaw) (boundary : SplineBoundary) (n : Nat) : Int :=
  match law, boundary with
  | .hashed, .wrap => Int.ofNat (n * hashMul)
  | .hashed, .clamp => Int.ofNat ((n * hashMul) % 4294967296 * 3 / 2) - 1073741824
  | .grid, .wrap => Int.ofNat (n * 16777216)
  | .grid, .clamp => Int.ofNat ((n * 16777216) % 8589934592) - 2147483648

def phaseSig (law : PhaseLaw) (boundary : SplineBoundary) : BuildM Sig := do
  let n ← sampleIndex
  match law, boundary with
  | .hashed, .wrap => mul n (← lit hashMul)
  | .hashed, .clamp =>
    let p ← bitAnd (← mul n (← lit hashMul)) (← lit 4294967295)
    sub (← rshift (← mul p (← lit 3)) (← lit 1)) (← lit 1073741824)
  | .grid, .wrap => mul n (← lit 16777216)
  | .grid, .clamp =>
    sub (← bitAnd (← mul n (← lit 16777216)) (← lit 8589934591)) (← lit 2147483648)

structure Config where
  basis : SplineBasis
  boundary : SplineBoundary
  k : Nat
  law : PhaseLaw
deriving Inhabited

def Config.label (c : Config) : String :=
  let b := if c.basis.scale == 2 then "catmull-rom" else "b-spline"
  let m := if c.boundary == .wrap then "wrap" else "clamp"
  let l := if c.law == .hashed then "hashed" else "grid"
  s!"{b}/{m}/K={c.k}/{l}"

def configs : Array Config := Id.run do
  let mut out := #[]
  for basis in #[catmullRomBasis, uniformBSplineBasis] do
    for boundary in #[SplineBoundary.wrap, .clamp] do
      for k in #[4, 7, 16] do
        for law in #[PhaseLaw.hashed, .grid] do
          out := out.push { basis, boundary, k, law }
  return out

def seedOf (c : Config) : Nat := c.k * 7919 + (if c.boundary == .wrap then 0 else 31)

-- ── spline-oracle ────────────────────────────────────────────────────────

def samples : Nat := 4096

def runOracle (arena : Arena) : IO Bool := do
  let label := "spline-oracle"
  let outs : BuildM (Array Sig) := configs.mapM fun c => do
    let cs ← (controlsOf (seedOf c) c.k).mapM controlSig
    splineSig c.basis c.boundary cs (← phaseSig c.law c.boundary)
  let plan ← match compileMulti arena "spline_oracle" outs 48000 with
    | .ok plan => pure plan
    | .error e => return ← failGate label s!"compile: {firstLine e}"
  let ns := range 0 samples
  let got ← renderAt plan configs.size ns
  let mut worst := 0.0
  let mut worstLabel := ""
  for ci in [0:configs.size] do
    let c := configs[ci]!
    let cs := (controlsOf (seedOf c) c.k).map controlFloat
    let want := ns.map fun n => refEval c.basis c.boundary cs (phaseInt c.law c.boundary n)
    let err := maxAbsDiff got[ci]! want
    if err > worst || worstLabel == "" then
      worst := err
      worstLabel := c.label
    if got[ci]!.size != samples || peakOf want == 0.0 then
      return ← failGate label s!"{c.label}: rendered {got[ci]!.size} samples, reference peak {peakOf want}"
  if worst ≤ 1e-12 then
    passGate label s!"{configs.size} configurations × {samples} phases ≡ f64 reference, worst |Δ| {sci worst} ({worstLabel})"
  else
    failGate label s!"worst |Δ| {sci worst} at {worstLabel} (tolerance 1e-12)"

-- ── spline-seam ──────────────────────────────────────────────────────────

/-- Samples `n ∈ [0, 2·half]` read phase `(n − half)·step`, straddling 0. -/
def seamPhase (half step n : Nat) : Int := (Int.ofNat n - Int.ofNat half) * Int.ofNat step

def seamPhaseSig (half step : Nat) : BuildM Sig := do
  mul (← sub (← sampleIndex) (← lit half)) (← lit step)

def runSeam (arena : Arena) : IO Bool := do
  let label := "spline-seam"
  let half := 64
  let width := 2 * half + 1
  let bases := #[catmullRomBasis, uniformBSplineBasis]
  -- (1) Rotation: spline(rot c, q) ≡ spline(c, q + 2³²/8), bit for bit.
  let rotOuts : BuildM (Array Sig) := do
    let mut outs := #[]
    for basis in bases do
      let base := controlsOf 101 8
      let rotated := (Array.range 8).map fun j => base[(j + 1) % 8]!
      let q ← seamPhaseSig half 4096
      let shifted ← add q (← lit 536870912)
      outs := outs.push (← splineSig basis .wrap (← rotated.mapM controlSig) q)
      outs := outs.push (← splineSig basis .wrap (← base.mapM controlSig) shifted)
    pure outs
  let rotPlan ← match compileMulti arena "spline_rotation" rotOuts 48000 with
    | .ok plan => pure plan
    | .error e => return ← failGate label s!"compile rotation: {firstLine e}"
  let rot ← renderAt rotPlan 4 (range 0 width)
  let rotDiff := bitDiffCount rot[0]! rot[1]! + bitDiffCount rot[2]! rot[3]!
  -- (2) One-sided differences across the seam, step h = 2¹⁸ phase units.
  let h := 262144
  let fdConfigs : Array (SplineBasis × Nat) :=
    #[(catmullRomBasis, 7), (catmullRomBasis, 8), (uniformBSplineBasis, 7),
      (uniformBSplineBasis, 8)]
  let fdOuts : BuildM (Array Sig) := fdConfigs.mapM fun (basis, k) => do
    splineSig basis .wrap (← (controlsOf (k * 13) k).mapM controlSig) (← seamPhaseSig 3 h)
  let fdPlan ← match compileMulti arena "spline_seam_fd" fdOuts 48000 with
    | .ok plan => pure plan
    | .error e => return ← failGate label s!"compile seam: {firstLine e}"
  let fd ← renderAt fdPlan fdConfigs.size (range 0 7)
  let mut worstValue := 0.0
  let mut worstSlope := 0.0
  let mut slopeScale := 0.0
  for ci in [0:fdConfigs.size] do
    let (basis, k) := fdConfigs[ci]!
    let v : Array Float := fd[ci]!
    -- v[3] is phase 0; v[0..2] are −3h..−h (the end of the cycle), v[4..6] +h..+3h.
    let dt : Float := k.toFloat * h.toFloat / 4294967296.0
    let leftValue : Float := 3.0 * v[2]! - 3.0 * v[1]! + v[0]!
    let rightValue : Float := 3.0 * v[4]! - 3.0 * v[5]! + v[6]!
    let leftSlope : Float := (3.0 * v[3]! - 4.0 * v[2]! + v[1]!) / (2.0 * dt)
    let rightSlope : Float := (-3.0 * v[3]! + 4.0 * v[4]! - v[5]!) / (2.0 * dt)
    let cs := (controlsOf (k * 13) k).map controlFloat
    let exact := refEvalDt basis .wrap cs 0 0.0
    worstValue := max worstValue (max (leftValue - v[3]!).abs (rightValue - v[3]!).abs)
    worstSlope := max worstSlope (leftSlope - rightSlope).abs
    slopeScale := max slopeScale exact.abs
  IO.println s!"        rotation: {rotDiff} of {4 * width / 2} samples differ · extrapolated value gap {sci worstValue} · slope gap {sci worstSlope} (|dv/dt| up to {sci slopeScale})"
  if rotDiff == 0 && worstValue ≤ 1e-6 && worstSlope ≤ 1e-3 then
    passGate label "the seam renders as an interior knot bit for bit (rotation, K = 8); value and dv/dt continuous across 1 → 0 (K = 7, 8; both bases)"
  else
    failGate label s!"rotation differs at {rotDiff} samples, value gap {sci worstValue}, slope gap {sci worstSlope}"

-- ── spline-live ──────────────────────────────────────────────────────────

private def frames : Nat := 64

private def liveInc : Nat := 89478485

private def livePatch (values : Array Int) : String :=
  let params := (Array.range 8).toList.map fun j =>
    "{\"op\":\"paramDecl\",\"name\":\"k" ++ toString j ++ "\",\"value\":" ++
      toString (controlFloat values[j]!) ++ "}"
  let inputs := (Array.range 8).toList.map fun j =>
    "\"c" ++ toString j ++ "\":{\"op\":\"param\",\"name\":\"k" ++ toString j ++ "\"}"
  "{\"schema\":\"tropical_program_2\",\"name\":\"spline_live\",\"body\":{\"op\":\"block\",\"decls\":[" ++
  ",".intercalate params ++ "," ++
  "{\"op\":\"instanceDecl\",\"name\":\"sp\",\"program\":\"PeriodicCatmullRom8\",\"inputs\":{" ++
  "\"phase\":{\"op\":\"mul\",\"args\":[{\"op\":\"sampleIndex\"}," ++ toString liveInc ++ "]}," ++
  ",".intercalate inputs ++ "}}]," ++
  "\"assigns\":[{\"op\":\"outputAssign\",\"name\":\"dac.out\",\"expr\":" ++
  "{\"op\":\"ref\",\"instance\":\"sp\",\"output\":\"out\"}}]}}"

def runLive : IO Bool := do
  let label := "spline-live"
  let before := controlsOf 7 8
  let written : Int := 57
  let after := before.set! 3 written
  let registry ← match Tropical.Session.bootRegistry with
    | .error f => return ← failGate label s!"stdlib boot failed: {f.toJson.compress}"
    | .ok st => pure st
  let rt ← match (Tropical.Session.compilePatch (livePatch before)).runOn registry with
    | (.error f, _) => return ← failGate label s!"compile: {f.toJson.compress}"
    | (.ok (plan, blocks), _) =>
      let rt ← Tropical.Ffi.Runtime.new frames.toUInt32
      Tropical.StagedLoad.loadTyped rt plan blocks
      pure rt
  let reference := fun (cs : Array Int) (start : Nat) =>
    (range start frames).map fun n =>
      refEval catmullRomBasis .wrap (cs.map controlFloat) (Int.ofNat (n * liveInc))
  rt.process
  let buffer0 := decodeF64LE (← rt.outputBytes)
  if (← rt.setParam "k3" (controlFloat written)).isNone then
    return ← failGate label "set_param: the loaded manifest has no param 'k3'"
  rt.process
  let buffer1 := decodeF64LE (← rt.outputBytes)
  let err0 := maxAbsDiff buffer0 (reference before 0)
  let err1 := maxAbsDiff buffer1 (reference after frames)
  let moved := maxAbsDiff buffer1 (reference before frames)
  IO.println s!"        buffer 0 vs reference(before) {sci err0} · buffer 1 vs reference(after) {sci err1} · buffer 1 vs reference(before) {sci moved}"
  if buffer0.size == frames && buffer1.size == frames && err0 ≤ 1e-12 && err1 ≤ 1e-12
      && moved > 1e-3 then
    passGate label "a set_param on c3 reaches the next buffer of the loaded kernel, which matches the reference at the new value (no recompile)"
  else
    failGate label s!"before {sci err0}, after {sci err1}, moved {sci moved}"

def runSpline (arena : Arena) : IO (Nat × Nat) := do
  let results := #[← runOracle arena, ← runSeam arena, ← runLive]
  pure (results.size, results.filter (!·) |>.size)

end Tropical.Tropicaltest.Spline
