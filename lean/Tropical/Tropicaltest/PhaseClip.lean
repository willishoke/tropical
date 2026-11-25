import Tropical.Session.Compile
import Tropical.EmitArrow.PhaseClip
import Tropical.Tropicaltest.Spline

/-!
# PhaseClip gates: the pulse width as a phase warp, and the envelope it makes

`lean/Tropical/EmitArrow/PhaseClip.lean` is the module these gates back.

* **phaseclip-oracle** — the JIT's clipped phase and gate against an exact
  integer reference (`⌊p · 2³² / W⌋` in `Int`, `W` from IEEE `round(w · 2³²)`
  clamped to `[1, 2³²]`), bit for bit, for 58 widths: decimal and dyadic
  widths across `(0, 1]`, both ends (2⁻²⁰, 1), and moving widths outside the
  interval (−½, 0, 10⁻¹², 3·10⁻¹⁰, 1.5, 2), which clamp. Every width is
  rendered at every phase of the set: 4096 hashed phases, the cycle's ends,
  and `W − 1`, `W`, `W + 1` of every width. The phase arrives as
  `n − 2³³` for half the widths and `n + 3·2³²` for the other half, so the
  kernel's reduction to the cycle is exercised on both sides. The width is
  computed in the kernel per sample (not folded at compile time).
* **phaseclip-static** — `widthWord` on known decimals (1, ½, 0.3, ¼,
  2⁻³³, 2·10⁻¹⁰, 0.999999999, 7·10⁻⁶), and the kernel of
  `PulseWidth.static` for those widths, bit for bit against the exact
  reference at `W − 1`, `W`, `W + 1` and hashed phases.
* **phaseclip-envelope** — a periodic spline read through the clip. Catmull-Rom
  starts at `c0` and holds at `c0` for the rest of the cycle, bit for bit;
  the B-spline holds at its own phase-0 value. The read reaches phase 1
  exactly when the gate falls (at `p = W`), one step after the last sweep
  sample, for every width. With `c1 = c7` (an extremum at phase 0) the left
  slope at the hold point is 0, matching the held side; without it the gate
  measures the slope jump the reference predicts, so the slope check is not
  vacuous. A Catmull-Rom at K = 5 and a B-spline at K = 3 hold at their
  phase-0 value too (phase 2³² at K not a power of two). Values match the
  f64 reference everywhere.
* **phaseclip-live** — a patch `phase → PhaseClip → PeriodicCatmullRom8` with the
  width a `param:` slot. A `set_param` on the width changes the next buffer
  of the loaded kernel to the reference at the new width, with no recompile.
* **phaseclip-refusal** — a static width outside `[2⁻³³, 1]` is refused: as a
  bare literal at session compile (`0`, `1.5`, `-0.25`, `1e-10`), and as
  `PulseWidth.static` at assemble time (`0`, `1.5`, `10⁻¹⁰`, `10⁻¹²`, whose
  words would round to 0); `1`, `0.25` and `2e-10` are accepted.
-/

namespace Tropical.Tropicaltest.PhaseClip

open Tropical
open Tropical.Ir
open Tropical.EmitArrow
open Tropical.Tropicaltest.Spectral (compileMulti renderAt range maxAbsDiff peakOf sci)
open Tropical.Tropicaltest.Spline (refEval refEvalDt controlFloat controlSig hashMul)

def cycle : Int := 4294967296

-- ── The reference ────────────────────────────────────────────────────────

/-- `W = clamp(round(w · 2³²), 1, 2³²)`, half away from zero (Lean's
    `Float.round`, like `llvm.round`). -/
def refWord (w : Float) : Int :=
  let x := (w * 4294967296.0).round
  let i : Int := if x < 0.0 then -((-x).toUInt64.toNat : Int) else (x.toUInt64.toNat : Int)
  max 1 (min cycle i)

/-- Whether `w · 2³²` is a tie for rounding: the fixture avoids ties, where
    the reference's `round` convention would be the only thing tested. -/
def isTie (w : Float) : Bool :=
  let x := w * 4294967296.0
  x - x.floor == 0.5

/-- The clipped phase and the gate, exactly. -/
def refClip (raw W : Int) : Int × Bool :=
  let p := raw % cycle
  if p < W then (p * cycle / W, true) else (cycle, false)

-- ── Fixtures ─────────────────────────────────────────────────────────────

/-- A width `m / d`, computed in the kernel per sample: `(n >> 62)` is 0 for
    every rendered sample but is not a constant, so the division, the
    rounding and the clamp run in the kernel. -/
def widthSig (m d : Int) : BuildM Sig := do
  let z ← rshift (← sampleIndex) (← lit 62)
  div (← toFloatE (← add z (← lit m))) (← lit d)

def widthFloat (m d : Int) : Float := Float.ofInt m / Float.ofInt d

/-- The phase input `n + offset` with `offset` a multiple of 2³²: the same
    cycle position as `n`. `lowOffset` makes it negative, `highOffset` puts it
    at or above 2³². -/
def lowOffset : Int := -8589934592
def highOffset : Int := 12884901888

def rawPhaseSig (offset : Int := lowOffset) : BuildM Sig := do
  add (← sampleIndex) (← lit offset)

def rawPhase (n : Nat) (offset : Int := lowOffset) : Int := Int.ofNat n + offset

/-- Even-numbered widths read a negative raw phase, odd ones a raw phase at or
    above 2³². -/
def offsetOf (c : Nat) : Int := if c % 2 == 0 then lowOffset else highOffset

def oracleWidths : Array (Int × Int) := Id.run do
  let mut out : Array (Int × Int) := #[(1, 1), (1, 2), (1, 3), (2, 3), (3, 10), (1, 10),
    (999999999, 1000000000), (1, 1048576), (3, 1048576), (1048575, 1048576),
    (3, 8), (5, 7), (11, 13), (1, 1000), (1, 7919), (7, 1000000),
    -- moving widths outside (0, 1]: they clamp
    (-1, 2), (0, 1), (1, 1000000000000), (3, 10000000000), (3, 2), (2, 1)]
  let mut x := 12345
  for _ in [0:20] do
    x := (x * 1103515245 + 12345) % 2147483648
    out := out.push (Int.ofNat (x / 7 % 1000) + 1, 1000)
  for _ in [0:16] do
    x := (x * 1103515245 + 12345) % 2147483648
    out := out.push (Int.ofNat (x % 1048576) + 1, 1048576)
  return out

def oraclePhases (words : Array Int) : Array Nat := Id.run do
  let mut out : Array Nat := #[0, 1, 2, 4294967295, 4294967296, 4294967297]
  for W in words do
    for d in #[-1, 0, 1] do
      let p := W + d
      if 0 ≤ p ∧ p ≤ cycle + 1 then out := out.push p.toNat
  for n in [0:4096] do
    out := out.push (n * hashMul % 4294967296)
  return out

-- ── phaseclip-oracle ────────────────────────────────────────────────────────

def runOracle (arena : Arena) : IO Bool := do
  let label := "phaseclip-oracle"
  let widths := oracleWidths
  if let some (m, d) := widths.find? fun (m, d) => isTie (widthFloat m d) then
    return ← failGate label s!"fixture: width {m}/{d} is a rounding tie"
  -- One channel per width (a plan has at most 64): the clipped phase plus
  -- the gate times 2³⁴, exact in f64.
  let outs : BuildM (Array Sig) := (Array.range widths.size).mapM fun c => do
    let (m, d) := widths[c]!
    let clip ← phaseClipSig (← rawPhaseSig (offsetOf c)) (.live (← widthSig m d))
    add (← toFloatE clip.phase) (← mul clip.gate (← lit 17179869184))
  let plan ← match compileMulti arena "phaseclip_oracle" outs 48000 with
    | .ok plan => pure plan
    | .error e => return ← failGate label s!"compile: {firstLine e}"
  let words := widths.map fun (m, d) => refWord (widthFloat m d)
  let ns := oraclePhases words
  let got ← renderAt plan widths.size ns
  let mut bad := 0
  let mut firstBad := ""
  for c in [0:widths.size] do
    let W := words[c]!
    let gotCh := got[c]!
    if gotCh.size != ns.size then
      return ← failGate label s!"rendered {gotCh.size} samples, want {ns.size}"
    for i in [0:ns.size] do
      let (q, high) := refClip (rawPhase ns[i]! (offsetOf c)) W
      let want := Float.ofInt q + (if high then 17179869184.0 else 0.0)
      if gotCh[i]!.toBits != want.toBits then
        if bad == 0 then
          let (m, d) := widths[c]!
          firstBad := s!"width {m}/{d} (W = {W}) at p = {ns[i]!}: got {gotCh[i]!}, want phase {q} + gate {high}"
        bad := bad + 1
  if bad == 0 then
    passGate label s!"{widths.size} widths × {ns.size} phases: clipped phase and gate ≡ exact reference, bit for bit (edges W−1, W, W+1; moving widths outside (0, 1] clamp)"
  else
    failGate label s!"{bad} mismatches; first: {firstBad}"

-- ── phaseclip-envelope ──────────────────────────────────────────────────────

/-- Catmull-Rom controls with `c1 = c7` (an extremum at phase 0), the same
    with `c7` moved (no extremum), and B-spline controls. Values are `m/64`. -/
def extremumControls : Array Int := #[0, 20, 48, 64, 40, 30, 10, 20]
def slopedControls : Array Int := #[0, 20, 48, 64, 40, 30, 10, 5]
def bsplineControls : Array Int := #[-6, 20, 64, 52, 40, 26, 12, 6]

def envelopeWidths : Array (Int × Int) :=
  #[(1, 1), (1, 2), (3, 10), (7, 100), (3, 1048576), (999, 1000)]

/-- The step `h` (in input phase units) for one-sided differences at `W`: the
    read phase moves about 2¹⁸ per step. -/
def fdStep (W : Int) : Int := max 1 (W / 16384)

def runEnvelope (arena : Arena) : IO Bool := do
  let label := "phaseclip-envelope"
  let widths := envelopeWidths
  let words := widths.map fun (m, d) => refWord (widthFloat m d)
  let sets : Array (SplineBasis × Array Int) :=
    #[(catmullRomBasis, extremumControls), (catmullRomBasis, slopedControls),
      (uniformBSplineBasis, bsplineControls), (catmullRomBasis, #[10, 40, 64, 30, 5]),
      (uniformBSplineBasis, #[-6, 50, 12])]
  -- Channels: for each width, the clipped phase, then one envelope per set.
  let perWidth := 1 + sets.size
  let outs : BuildM (Array Sig) := do
    let mut out := #[]
    for (m, d) in widths do
      let clip ← phaseClipSig (← rawPhaseSig) (.live (← widthSig m d))
      out := out.push (← toFloatE clip.phase)
      for (basis, cs) in sets do
        out := out.push (← splineSig basis (← cs.mapM controlSig) clip.phase)
    pure out
  let plan ← match compileMulti arena "phaseclip_envelope" outs 48000 with
    | .ok plan => pure plan
    | .error e => return ← failGate label s!"compile: {firstLine e}"
  let mut ns : Array Nat := (Array.range 4097).map (· * 1048576)
  for W in words do
    let h := fdStep W
    for d in #[-2 * h, -h, -1, 0, 1] do
      if 0 ≤ W + d ∧ W + d ≤ cycle then ns := ns.push (W + d).toNat
  let got ← renderAt plan (widths.size * perWidth) ns
  let mut worstRef := 0.0
  let mut holdBad := 0
  let mut endBad := 0
  let mut worstExtremumSlope := 0.0
  let mut worstSlopedGap := 0.0
  let mut slopedScale := 0.0
  for wi in [0:widths.size] do
    let W := words[wi]!
    let base := wi * perWidth
    let phaseCh := got[base]!
    if phaseCh.size != ns.size then
      return ← failGate label s!"rendered {phaseCh.size} samples, want {ns.size}"
    -- The read completes within the gate: the last sample inside the gate is
    -- still sweeping, within one step of the end; the first one outside it
    -- is at the end.
    let stepUp : Float := (Float.ofInt cycle / Float.ofInt W).ceil + 1.0
    for i in [0:ns.size] do
      let p := ns[i]! % 4294967296
      if (p : Int) == W - 1 then
        let q := phaseCh[i]!
        unless q < 4294967296.0 && q ≥ 4294967296.0 - stepUp do endBad := endBad + 1
      if (p : Int) == W then
        unless phaseCh[i]! == 4294967296.0 do endBad := endBad + 1
    for si in [0:sets.size] do
      let (basis, cs) := sets[si]!
      let csF := cs.map controlFloat
      let v := got[base + 1 + si]!
      let start := refEval basis csF 0
      for i in [0:ns.size] do
        let p : Int := ns[i]! % 4294967296
        let (q, _) := refClip (rawPhase ns[i]!) W
        worstRef := max worstRef (v[i]! - refEval basis csF q).abs
        -- Start and hold: phase 0 and every phase outside the gate read the
        -- same value, bit for bit; for Catmull-Rom it is c0.
        let held := if basis.scale == 2 then csF[0]! else v[0]!
        if (p == 0 || p ≥ W) && v[i]!.toBits != held.toBits then holdBad := holdBad + 1
        if basis.scale == 2 && p == 0 && v[i]!.toBits != start.toBits then holdBad := holdBad + 1
      -- One-sided slope at the hold point, in value per cycle of the read.
      if si < 2 then
        let h := fdStep W
        let valueAt := fun (target : Int) =>
          match (Array.range ns.size).find? fun (i : Nat) => Int.ofNat ns[i]! == target with
          | some i => v[i]!
          | none => 0.0 / 0.0
        let du : Float := Float.ofInt (h * cycle / W) / 4294967296.0
        let left := (3.0 * valueAt W - 4.0 * valueAt (W - h) + valueAt (W - 2 * h)) / (2.0 * du)
        let expected := refEvalDt basis csF 0 0.0 * cs.size.toFloat
        if si == 0 then
          worstExtremumSlope := max worstExtremumSlope left.abs
        else
          worstSlopedGap := max worstSlopedGap (left - expected).abs
          slopedScale := max slopedScale expected.abs
  IO.println s!"        vs reference {sci worstRef} · hold/start mismatches {holdBad} · end-of-gate misses {endBad} · left slope at hold: extremum {sci worstExtremumSlope}, sloped gap {sci worstSlopedGap} (|ds/du| {sci slopedScale})"
  if worstRef ≤ 1e-12 && holdBad == 0 && endBad == 0 && worstExtremumSlope ≤ 1e-3
      && slopedScale > 0.1 && worstSlopedGap ≤ 1e-3 * slopedScale then
    passGate label s!"{widths.size} widths: starts and holds at c0 bit for bit, completes exactly at the gate's fall, slope continuous at the hold when phase 0 is an extremum (and the jump measured when it is not); ≡ f64 reference"
  else
    failGate label s!"reference {sci worstRef}, hold {holdBad}, end {endBad}, extremum slope {sci worstExtremumSlope}, sloped gap {sci worstSlopedGap}"

-- ── phaseclip-live ──────────────────────────────────────────────────────────

private def frames : Nat := 64

/-- About 2³²/40: the input clock runs 1.6 cycles per buffer. -/
private def liveInc : Nat := 107374182

private def livePatch (values : Array Int) (width : Float) : String :=
  let params := (Array.range 8).toList.map fun j =>
    "{\"op\":\"paramDecl\",\"name\":\"k" ++ toString j ++ "\",\"value\":" ++
      toString (controlFloat values[j]!) ++ "}"
  let inputs := (Array.range 8).toList.map fun j =>
    "\"c" ++ toString j ++ "\":{\"op\":\"param\",\"name\":\"k" ++ toString j ++ "\"}"
  "{\"schema\":\"tropical_program_2\",\"name\":\"phaseclip_live\",\"body\":{\"op\":\"block\",\"decls\":[" ++
  ",".intercalate params ++ "," ++
  "{\"op\":\"paramDecl\",\"name\":\"w\",\"value\":" ++ toString width ++ "}," ++
  "{\"op\":\"instanceDecl\",\"name\":\"clip\",\"program\":\"PhaseClip\",\"inputs\":{" ++
  "\"phase\":{\"op\":\"mul\",\"args\":[{\"op\":\"sampleIndex\"}," ++ toString liveInc ++ "]}," ++
  "\"width\":{\"op\":\"param\",\"name\":\"w\"}}}," ++
  "{\"op\":\"instanceDecl\",\"name\":\"sp\",\"program\":\"PeriodicCatmullRom8\",\"inputs\":{" ++
  "\"phase\":{\"op\":\"ref\",\"instance\":\"clip\",\"output\":\"phase\"}," ++
  ",".intercalate inputs ++ "}}]," ++
  "\"assigns\":[{\"op\":\"outputAssign\",\"name\":\"dac.out\",\"expr\":" ++
  "{\"op\":\"ref\",\"instance\":\"sp\",\"output\":\"out\"}}]}}"

def runLive : IO Bool := do
  let label := "phaseclip-live"
  let cs := extremumControls
  let before := 0.3
  let after := 0.6
  let registry ← match Tropical.Session.bootRegistry with
    | .error f => return ← failGate label s!"stdlib boot failed: {f.toJson.compress}"
    | .ok st => pure st
  let rt ← match (Tropical.Session.compilePatch (livePatch cs before)).runOn registry with
    | (.error f, _) => return ← failGate label s!"compile: {f.toJson.compress}"
    | (.ok (plan, blocks), _) =>
      let rt ← Tropical.Ffi.Runtime.new frames.toUInt32
      Tropical.StagedLoad.loadTyped rt plan blocks
      pure rt
  let reference := fun (w : Float) (start : Nat) =>
    (range start frames).map fun n =>
      refEval catmullRomBasis (cs.map controlFloat)
        (refClip (Int.ofNat (n * liveInc)) (refWord w)).1
  rt.process
  let buffer0 := decodeF64LE (← rt.outputBytes)
  if (← rt.setParam "w" after).isNone then
    return ← failGate label "set_param: the loaded manifest has no param 'w'"
  rt.process
  let buffer1 := decodeF64LE (← rt.outputBytes)
  let err0 := maxAbsDiff buffer0 (reference before 0)
  let err1 := maxAbsDiff buffer1 (reference after frames)
  let moved := maxAbsDiff buffer1 (reference before frames)
  IO.println s!"        buffer 0 vs reference(w = {before}) {sci err0} · buffer 1 vs reference(w = {after}) {sci err1} · buffer 1 vs reference(w = {before}) {sci moved}"
  if buffer0.size == frames && buffer1.size == frames && err0 ≤ 1e-12 && err1 ≤ 1e-12
      && moved > 1e-3 then
    passGate label "a set_param on the width reaches the next buffer of the loaded kernel, which matches the reference at the new width (no recompile)"
  else
    failGate label s!"before {sci err0}, after {sci err1}, moved {sci moved}"

-- ── phaseclip-refusal ───────────────────────────────────────────────────────

private def literalPatch (width : String) : String :=
  "{\"schema\":\"tropical_program_2\",\"name\":\"phaseclip_literal\",\"body\":{\"op\":\"block\",\"decls\":[" ++
  "{\"op\":\"instanceDecl\",\"name\":\"clip\",\"program\":\"PhaseClip\",\"inputs\":{" ++
  "\"phase\":{\"op\":\"mul\",\"args\":[{\"op\":\"sampleIndex\"},1000]},\"width\":" ++ width ++ "}}]," ++
  "\"assigns\":[{\"op\":\"outputAssign\",\"name\":\"dac.out\",\"expr\":" ++
  "{\"op\":\"ref\",\"instance\":\"clip\",\"output\":\"gate\"}}]}}"

def runRefusal (arena : Arena) : IO Bool := do
  let label := "phaseclip-refusal"
  let registry ← match Tropical.Session.bootRegistry with
    | .error f => return ← failGate label s!"stdlib boot failed: {f.toJson.compress}"
    | .ok st => pure st
  let mut problems : Array String := #[]
  for (width, refused) in #[("0", true), ("1.5", true), ("-0.25", true), ("1e-10", true),
      ("1", false), ("0.25", false), ("2e-10", false)] do
    match (Tropical.Session.compilePatch (literalPatch width)).runOn registry with
    | (.error f, _) =>
      let msg := f.toJson.compress
      if !refused then problems := problems.push s!"literal {width} refused: {msg}"
      else if (msg.splitOn "interval").length ≤ 1 then
        problems := problems.push s!"literal {width} refused for another reason: {msg}"
    | (.ok _, _) => if refused then problems := problems.push s!"literal {width} accepted"
  let staticBuild := fun (w : Lean.JsonNumber) =>
    assemble arena "phaseclip_static" #[{ name := "out", type? := some (.scalar .float) }] #[] do
      let clip ← phaseClipSig (← sampleIndex) (.static w)
      pure { inputs := #[], assigns := #[(.port ⟨0⟩, clip.gate)] }
  for (w, refused) in #[((⟨0, 0⟩ : Lean.JsonNumber), true), (⟨15, 1⟩, true), (⟨1, 10⟩, true),
      (⟨1, 12⟩, true), (⟨1, 0⟩, false), (⟨25, 2⟩, false), (⟨116415321826934814453125, 33⟩, false)] do
    match staticBuild w, refused with
    | .ok _, true => problems := problems.push s!"static width {w} accepted at assemble"
    | .error e, false => problems := problems.push s!"static width {w} refused: {e}"
    | _, _ => pure ()
  if problems.isEmpty then
    passGate label "a static width outside [2^-33, 1] is refused (bare literal at session compile: 0, 1.5, -0.25, 1e-10; PulseWidth.static at assemble: 0, 1.5, 1e-10, 1e-12); 1, 0.25, 2e-10 and 2^-33 accepted"
  else
    failGate label (String.intercalate "; " problems.toList)

-- ── phaseclip-static ─────────────────────────────────────────────────────

/-- Decimals and the word `round(w · 2³²)` each must give, worked by hand. -/
def staticWords : Array (Lean.JsonNumber × Int) := #[
  (⟨1, 0⟩, 4294967296), (⟨5, 1⟩, 2147483648), (⟨3, 1⟩, 1288490189),
  (⟨25, 2⟩, 1073741824), (⟨116415321826934814453125, 33⟩, 1), (⟨2, 10⟩, 1),
  (⟨999999999, 9⟩, 4294967292), (⟨7, 6⟩, 30065)]

def runStatic (arena : Arena) : IO Bool := do
  let label := "phaseclip-static"
  let wrong := staticWords.filter fun (w, word) => widthWord w != word
  unless wrong.isEmpty do
    return ← failGate label s!"widthWord: {wrong.map fun (w, word) => s!"{w} gave {widthWord w}, want {word}"}"
  let outs : BuildM (Array Sig) := (Array.range staticWords.size).mapM fun c => do
    let clip ← phaseClipSig (← rawPhaseSig (offsetOf c)) (.static staticWords[c]!.1)
    add (← toFloatE clip.phase) (← mul clip.gate (← lit 17179869184))
  let plan ← match compileMulti arena "phaseclip_static" outs 48000 with
    | .ok plan => pure plan
    | .error e => return ← failGate label s!"compile: {firstLine e}"
  let ns := oraclePhases (staticWords.map (·.2))
  let got ← renderAt plan staticWords.size ns
  let mut bad := 0
  for c in [0:staticWords.size] do
    let W := staticWords[c]!.2
    for i in [0:ns.size] do
      let (q, high) := refClip (rawPhase ns[i]! (offsetOf c)) W
      let want := Float.ofInt q + (if high then 17179869184.0 else 0.0)
      if got[c]!.size != ns.size || got[c]![i]!.toBits != want.toBits then bad := bad + 1
  if bad == 0 then
    passGate label s!"widthWord ≡ the hand-worked word for {staticWords.size} decimals (2^-33 → 1); PulseWidth.static renders ≡ exact reference over {ns.size} phases each"
  else
    failGate label s!"{bad} mismatches"

def runPhaseClip (arena : Arena) : IO (Nat × Nat) := do
  let results := #[← runOracle arena, ← runStatic arena, ← runEnvelope arena, ← runLive,
    ← runRefusal arena]
  pure (results.size, results.filter (!·) |>.size)

end Tropical.Tropicaltest.PhaseClip
