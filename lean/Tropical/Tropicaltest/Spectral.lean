import Tropical.Ffi
import Tropical.StagedLoad
import Tropical.Compile
import Tropical.Ir.Strata
import Tropical.Playground
import Tropical.EmitArrow
import Tropical.Testing.SpectralFixtures
import Tropical.Tropicaltest.Gates
import Tropical.Tropicaltest.Reversibility

/-!
# Spectral gates: the partial family (S2) and the complex envelope (S3)

`design/spectral-family.md` is the note these gates back.

* **family-plan-size** — a family's plan is O(1) in its partial count: the
  instruction stream is identical for K = 24 … 8000; only the coefficient
  table's operands grow (K of them).
* **family-old-saw** — the 24-partial saw of `90242aa`
  (`web/site/synced-scope/gen_graph.ts`: 24 `source` sines per voice, `1/n`
  ring × knob weights, host-written partial frequencies) re-expressed as one
  family term per voice is BIT-IDENTICAL to that commit's render. The four
  voices' dry-saw taps were rendered by `90242aa`'s own `diffcli render-graph`
  over two windows (τ = 0 and τ = 1000003, 16384 samples each); their
  SHA-256s are pinned here. The same graph (committed as
  `tests/fixtures/spectral/synced-scope-90242aa.json`) through today's compiler
  is checked too, bitwise.
* **envelope-oracle** — the complex SVF-lowpass envelope reproduces the
  filter-FM listening set's `sweep_q0.7`, `sweep_q4` and `sweep_q12` spectral
  renders (40 s, 110 Hz saw, 181 zero-phase cosine partials, c(τ) =
  800·2^(2 sin φm(τ)), fm 0.25 → 440 Hz exponential) at windows across the
  sweep, the fast end included, against the closed form ported below
  (`oracleCutoff`/`oracleSample`, from the listening set's `scripts/dsp.ts`).
  Two comparisons: the OPERATOR, fed the kernel's own cutoff samples (so the
  only difference is the family + envelope arithmetic), and END TO END, with
  the cutoff law evaluated by libm. Phase convention: the script's partial
  phase is `2π·k·f·n/rate`; tropical's is the exact integer rail
  `2π·((k·inc·n) mod 2³²)/2³²`, `inc = ⌊f·2³²/rate⌋`. The reference uses the
  rail (the documented conversion); the rail-vs-exact difference is printed.
* **envelope-residue** — steady state: for constant-frequency partials and a
  fixed cutoff, the envelope set to the modal resonant Filter's response
  (`filterPair`, Q read off the filter's own pole) gives each partial exactly
  the forced-mode amplitude `residueComposeEC` computes.
* **envelope-algebra** — the envelope distributes over sum and commutes with
  scale (normal form and emitted IDs), and the refusals: an envelope over an
  opaque generator, over a warp, and a family with a moving fundamental.
-/

namespace Tropical.Tropicaltest.Spectral

open Tropical
open Tropical.Ir
open Tropical.EmitArrow
open Tropical.Testing.SpectralFixtures

-- ── Harness ──────────────────────────────────────────────────────────────

/-- Compile a multi-output carrier (output `i` is channel `i`) at `rate`. -/
def compileMulti (arena : Arena) (name : String) (outs : BuildM (Array Sig))
    (rate : Nat) : Except String Tropical.Plan.FlatPlan := do
  let (a, idx, n) ← assembleCompleteWithResult arena name #[] (do
    let sigs ← outs
    let portName := fun (i : Nat) => if i == 0 then "out" else s!"ch{i}"
    let outputs : Array OutputDecl := sigs.mapIdx fun i _ =>
      { name := portName i, type? := some (.scalar .float) }
    let assigns : Array (OutputTarget × Sig) := sigs.mapIdx fun i s => (.port ⟨i⟩, s)
    pure (({ outputs, assigns } : CompleteProgramBody), sigs.size))
  let (coreArena, core) ← (Tropical.Ir.Strata.runResolved {} a idx).mapError (·.message)
  let extra := (Array.range (n - 1)).map fun i => s!"ch{i + 1}"
  let plan ← Tropical.Compile.compileSession (.forRoot core coreArena (extraOutputs := extra))
  pure { plan with sampleRate := (rate : Lean.JsonNumber) }

private def chunk : Nat := 1024

/-- Load a plan and evaluate channels `0..channels-1` at the given sample
    indices (integer τ). Channel-major: `result[c][i]`. -/
def renderAt (plan : Tropical.Plan.FlatPlan) (channels : Nat) (samples : Array Nat) :
    IO (Array (Array Float)) := do
  let rt ← Tropical.Ffi.Runtime.new chunk.toUInt32
  Tropical.StagedLoad.load rt plan
  let chans := (Array.range channels).map UInt32.ofNat
  let mut out : Array (Array Float) := Array.replicate channels #[]
  let mut i := 0
  while i < samples.size do
    let block := samples.extract i (i + chunk)
    let taus := block.map fun n => (n * clockOne).toUInt64
    let values := decodeF64LE (← rt.renderTau taus chans)
    for c in [0:channels] do
      out := out.modify c (· ++ values.extract (c * block.size) ((c + 1) * block.size))
    i := i + chunk
  pure out

def range (start len : Nat) : Array Nat := (Array.range len).map (start + ·)

/-- Little-endian f64 bytes of channel-major sample arrays. -/
def bytesOf (chans : Array (Array Float)) : ByteArray := Id.run do
  let mut b := ByteArray.empty
  for ch in chans do
    for x in ch do
      let u := x.toBits
      for j in [0:8] do
        b := b.push ((u >>> (8 * j).toUInt64).toUInt8)
  return b

def nmseDb (dut ref : Array Float) : Float := Id.run do
  let mut e := 0.0
  let mut p := 0.0
  for i in [0:min dut.size ref.size] do
    e := e + (dut[i]! - ref[i]!) * (dut[i]! - ref[i]!)
    p := p + ref[i]! * ref[i]!
  return 10.0 * Float.log10 (e / p + 1e-300)

def maxAbsDiff (a b : Array Float) : Float := Id.run do
  let mut m := 0.0
  for i in [0:min a.size b.size] do
    m := max m (a[i]! - b[i]!).abs
  return m

def peakOf (a : Array Float) : Float := a.foldl (fun m x => max m x.abs) 0.0

/-- `x` rounded to `digits` decimals, trailing zeros trimmed. -/
def fmt (x : Float) (digits : Nat := 2) : String :=
  let s := 10.0 ^ digits.toFloat
  let chars := (toString ((x * s).round / s)).toList
  if chars.contains '.' then
    let trimmed := (chars.reverse.dropWhile (· == '0')).dropWhile (· == '.')
    String.ofList trimmed.reverse
  else String.ofList chars

/-- `x` in scientific notation, 3 significant digits. -/
def sci (x : Float) : String :=
  if x == 0.0 then "0" else
  let e := (Float.log10 x.abs).floor
  let ei : Int := if e ≥ 0.0 then e.toUInt64.toNat else -(((-e).toUInt64.toNat : Nat) : Int)
  s!"{fmt (x / Float.pow 10.0 e) 2}e{ei}"

/-- Instruction count and total operand count over a plan's instance tree. -/
partial def planCensus (plan : Tropical.Plan.FlatPlan) : Nat × Nat × Nat :=
  let rec go (f : Tropical.Plan.InstanceFunction) : Nat × Nat × Nat :=
    let instrs := f.preambleInstructions ++ f.instructions ++ f.preInputInstructions
    let packs := instrs.foldl (fun acc i => if i.tag == "Pack" then acc + i.args.size else acc) 0
    f.children.foldl (fun (a, b, c) child =>
        let (x, y, z) := go child
        (a + x, b + y, c + z))
      (instrs.size, instrs.foldl (fun acc i => acc + i.args.size) 0, packs)
  plan.instanceFunctions.foldl (fun (a, b, c) f =>
      let (x, y, z) := go f
      (a + x, b + y, c + z)) (0, 0, 0)

-- ── family-plan-size ─────────────────────────────────────────────────────

def runPlanSize (arena : Arena) : IO Bool := do
  let label := "family-plan-size"
  let counts : Array Nat := #[24, 181, 1000, 8000]
  let mut rows : Array (Nat × Nat × Nat × Nat) := #[]
  for k in counts do
    let built := compileMulti arena s!"family_size_{k}" (do
      let fam ← sawFamily (← lit 110) none (← clockLit) k
      pure #[← emitTerm (.family fam)]) 44100
    match built with
    | .error e => return ← failGate label s!"K={k}: {firstLine e}"
    | .ok plan =>
      let (instrs, operands, packed) := planCensus plan
      rows := rows.push (k, instrs, operands, packed)
  for (k, i, o, p) in rows do
    IO.println s!"        K={k}: {i} instructions, {o} operands ({p} in Pack, the coefficient table among them)"
  let i0 := rows[0]!.2.1
  let sameInstrs := rows.all (·.2.1 == i0)
  -- operands grow by exactly the table: K per partial count step
  let (k0, _, o0, _) := rows[0]!
  let tableOnly := rows.all fun (k, _, o, _) => o - o0 == k - k0
  if sameInstrs && tableOnly then
    passGate label s!"{i0} instructions at every K in {counts}; operands grow only by the K-entry weight column"
  else
    failGate label s!"instruction counts {rows.map (·.2.1)} (want one value) / operand growth not K"

-- ── family-old-saw ───────────────────────────────────────────────────────

def oldGraphPath : String := "tests/fixtures/spectral/synced-scope-90242aa.json"

/-- SHA-256 of `90242aa`'s `saw1..saw4` taps (channels 7–10 of
    `diffcli render-graph web/patches/synced-scope.json --all-channels
    --frames 16 --buffer 1024 [--start 1000003]`), channel-major f64 LE,
    16384 samples per channel. Rendered at 44100 Hz by that commit's build. -/
def oldSawHashes : Array (Nat × String) := #[
  (0, "6ed0da1eda66f741fa39bba1d72bb36990067620818c398c98f48179f6ce1968"),
  (1000003, "485f7bb553d991be6d884446ffc8e8df3c055c4a82b0f57addfaedcb0e9508cd")]

def runOldSaw (arena : Arena) (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let label := "family-old-saw"
  let text ← IO.FS.readFile oldGraphPath
  let graph ← match Lean.Json.parse text with
    | .ok j => pure j
    | .error e => return ← failGate label s!"fixture parse: {e}"
  let nodes := (graph.getObjValD "nodes").getArr?.toOption.getD #[]
  let freqOf := fun (id : String) => Id.run do
    for n in nodes do
      if (n.getObjValD "id").getStr?.toOption == some id then
        if let .ok (.num v) := (n.getObjValD "params").getObjVal? "freq" then
          return some v.toFloat
    return none
  let some freqs := (#["v1", "v2", "v3", "v4"].mapM freqOf)
    | return ← failGate label "fixture: missing v1..v4 freq"
  let familyPlan := compileMulti arena "old_saw_families" (do
    let clock ← clockLit
    freqs.mapM fun f => do
      let fam ← sawFamily (← litExact f) none clock 24
      emitTerm (.family fam)) 44100
  let plan ← match familyPlan with
    | .ok p => pure p
    | .error e => return ← failGate label s!"family build: {firstLine e}"
  -- today's compile of the old graph, taps saw1..saw4 (channels 1..4)
  let tapped := graph.setObjVal! "taps" (Lean.Json.arr #["saw1", "saw2", "saw3", "saw4"])
  let oldPlan ← match Tropical.Playground.compilePlanPure arena resolved tapped 44100 with
    | .ok c => pure c.plan
    | .error e => return ← failGate label s!"old graph compile: {firstLine e}"
  let mut ok := true
  let mut detail : Array String := #[]
  for (start, golden) in oldSawHashes do
    let fam ← renderAt plan 4 (range start 16384)
    let hash ← sha256Hex (bytesOf fam)
    let oldNow ← renderAt oldPlan 5 (range start 16384)
    let oldChans := oldNow.extract 1 5
    let diffs := (Array.range 4).foldl (fun acc c => acc + bitDiffCount fam[c]! oldChans[c]!) 0
    let goldenOk := hash == golden
    ok := ok && goldenOk && diffs == 0 && energyOf fam[0]! > 0.0
    detail := detail.push s!"τ0={start}: 90242aa hash {if goldenOk then "matches" else s!"DIFFERS ({hash})"}, today's compile of the old graph {diffs} bit-diffs"
  for d in detail do IO.println s!"        {d}"
  if ok then
    passGate label "4 voices × 24 partials as 4 family terms ≡ 90242aa's 96 sine nodes, bit-exact (2 windows × 4 × 16384 samples)"
  else
    failGate label "family render differs from the old saw"

-- ── envelope-oracle ──────────────────────────────────────────────────────

def oracleRate : Float := 48000.0
def oracleF0 : Float := 110.0
def oracleK : Nat := 181
def sweepFa : Float := 0.25
def sweepFb : Float := 440.0
def sweepT : Float := 40.0
def twoPi : Float := 6.283185307179586

/-- `sweepCutoff(0.25, 440, 40, 48000)` of `scripts/dsp.ts`, verbatim:
    `φ = k·(exp(lr·τ/T) − 1)`, `k = 2π·fa·T/lr`, `c = 800·2^(2 sin φ)`. -/
def oracleCutoff (n : Nat) : Float :=
  let lr := Float.log (sweepFb / sweepFa)
  let k := twoPi * sweepFa * sweepT / lr
  let tau := n.toFloat / oracleRate
  let phi := k * (Float.exp (lr * tau / sweepT) - 1.0)
  800.0 * Float.pow 2.0 (2.0 * Float.sin phi)

/-- The phasor increment of a 110 Hz partial family at 48 kHz:
    `⌊110·2³²/48000⌋`. -/
def oracleInc : Nat := (oracleF0 * 4294967296.0 / oracleRate).floor.toUInt64.toNat

/-- `renderSpectralDirect` of `scripts/dsp.ts` (complex mode) at sample `n`
    with cutoff `c`: `Σ_k (1/k)·|H|·cos(θ_k + arg H)`, `H = 1/(1 − x² + j·x/Q)`,
    `x = k·f/c`. `rail` selects tropical's integer phase for `θ_k`; otherwise
    the script's exact `2π·k·f·n/rate` (reduced exactly: f and rate are
    integers here). -/
def oracleSample (q c : Float) (n : Nat) (rail : Bool) : Float := Id.run do
  let mut acc := 0.0
  for kk in [1:oracleK + 1] do
    let k := kk.toFloat
    let x := k * oracleF0 / c
    let re := 1.0 - x * x
    let im := x / q
    let mag := 1.0 / Float.sqrt (re * re + im * im)
    let arg := -(Float.atan2 im re)
    let theta :=
      if rail then twoPi * ((kk * oracleInc * n) % 4294967296).toFloat / 4294967296.0
      else twoPi * ((kk * 110 * n) % 48000).toFloat / 48000.0
    acc := acc + (1.0 / k) * mag * Float.cos (theta + arg)
  return acc

/-- Window centres: where the sweep's fm takes these values (Hz), plus the
    last full window before 40 s. -/
def sweepFms : Array Float := #[0.5, 1, 3, 10, 30, 100, 200, 400]
def windowLen : Nat := 512

def sweepCentre (fm : Float) : Nat :=
  (sweepT * Float.log (fm / sweepFa) / Float.log (sweepFb / sweepFa) * oracleRate).toUInt64.toNat

def runOracle (arena : Arena) : IO Bool := do
  let label := "envelope-oracle"
  let ends := (40 * 48000) - windowLen
  let starts := (sweepFms.map fun fm => sweepCentre fm - windowLen / 2).push ends
  let mut ok := true
  for q in #[((7 : Nat), (1 : Nat)), (4, 0), (12, 0)] do
    let qF := q.1.toFloat / (10.0 ^ q.2.toFloat)
    let built := compileMulti arena s!"sweep_q{q.1}e{q.2}" (do
      let clock ← clockLit
      let cutoff : ArrowTerm := .arrUn (fun clk => do
        chirpCutoffSig (← lit 800) (← lit 2) sweepFa sweepFb sweepT clk)
        (.clk clock)
      let fam ← sawFamily (← lit 110) none clock oracleK (wave := .cosine)
      let y ← emitTerm (normalize
        (svfEnvelope .lowpass cutoff (.konst (← lit (q.1 : Int) q.2)) (.family fam)))
      let c ← chirpCutoffSig (← lit 800) (← lit 2) sweepFa sweepFb sweepT clock
      pure #[y, c]) 48000
    let plan ← match built with
      | .ok p => pure p
      | .error e => return ← failGate label s!"Q={qF}: build {firstLine e}"
    let mut worstOp := -1000.0
    let mut worstEnd := -1000.0
    let mut worstCut := 0.0
    let mut railVsExact := -1000.0
    for start in starts do
      let samples := range start windowLen
      let out ← renderAt plan 2 samples
      let y := out[0]!
      let cKernel := out[1]!
      let refOp := samples.mapIdx fun i n => oracleSample qF cKernel[i]! n true
      let refEnd := samples.map fun n => oracleSample qF (oracleCutoff n) n true
      let refExact := samples.map fun n => oracleSample qF (oracleCutoff n) n false
      let cOracle := samples.map oracleCutoff
      let cutRel := (Array.range samples.size).foldl (fun m i =>
        max m ((cKernel[i]! - cOracle[i]!).abs / cOracle[i]!)) 0.0
      let op := nmseDb y refOp
      let e2e := nmseDb y refEnd
      let rx := nmseDb refEnd refExact
      worstOp := max worstOp op
      worstEnd := max worstEnd e2e
      worstCut := max worstCut cutRel
      railVsExact := max railVsExact rx
      let fmHere := sweepFa * Float.pow (sweepFb / sweepFa) ((start + windowLen / 2).toFloat / oracleRate / sweepT)
      IO.println s!"        Q={fmt qF 1} τ={fmt ((start + windowLen / 2).toFloat / oracleRate)} s (fm {fmt fmHere} Hz): operator NMSE {fmt op 1} dB, end-to-end {fmt e2e 1} dB (max |Δc|/c {sci cutRel}), peak {fmt (peakOf y) 3}; rail vs exact phase {fmt rx 1} dB"
    let qOk := worstOp < -160.0 && worstEnd < -120.0
    ok := ok && qOk
    IO.println s!"        Q={fmt qF 1}: worst operator NMSE {fmt worstOp 1} dB, worst end-to-end {fmt worstEnd 1} dB, worst cutoff error {sci worstCut}, rail-vs-exact phase up to {fmt railVsExact 1} dB"
  if ok then
    passGate label "sweep_q0.7/q4/q12 reproduced at 9 windows (fm 0.5 → 440 Hz): operator < −160 dB NMSE, end to end < −120 dB"
  else
    failGate label "a window exceeds the tolerance (operator −160 dB, end to end −120 dB)"

-- ── envelope-residue ─────────────────────────────────────────────────────

def runResidue (arena : Arena) : IO Bool := do
  let label := "envelope-residue"
  let partials : Nat := 12
  let mut ok := true
  for (fc, res) in #[((300 : Nat), (0 : Nat)), (300, 4), (1200, 7), (700, 7)] do
    let built := compileMulti arena s!"residue_{fc}_{res}" (do
      let f0 ← lit 110
      let twoPi ← twoPiE
      let voice ← (Array.range partials).mapM fun (i : Nat) => do
        let k ← lit ((i + 1 : Nat) : Int)
        let omega ← mul twoPi (← mul k f0)
        let amp ← div (← lit 1) k
        pure ({ sigma := ← lit 0, omega, cre := ← lit 0, cim := ← neg amp } : ModalMode)
      let fcS ← lit fc
      let filter ← Tropical.Playground.Compiler.filterPair fcS (← lit res 1)
      let forced := (← residueComposeEC voice filter).extract 0 partials
      -- the filter's own Q, read off its pole: σ = ω0/(2Q)
      let q ← div (← mul twoPi fcS) (← mul (← lit 2) filter[0]!.sigma)
      let law := svfEnvelopeLaw .lowpass
      let prepared ← law.prepare #[fcS, q] f0
      let mut outs : Array Sig := #[]
      for i in [0:partials] do
        let k ← lit ((i + 1 : Nat) : Int)
        let (gre, gim) ← law.gain prepared k (← mul k f0)
        let a ← div (← lit 1) k
        -- the family's partial `Im(a·G·e^{iθ})` is the mode `Re(A·e^{iθ})`
        -- with `A = −i·a·G = (a·Gim, −a·Gre)`
        outs := outs ++ #[forced[i]!.cre, forced[i]!.cim, ← mul a gim, ← neg (← mul a gre)]
      pure outs) 44100
    let plan ← match built with
      | .ok p => pure p
      | .error e => return ← failGate label s!"build: {firstLine e}"
    let out ← renderAt plan (4 * partials) #[0]
    let mut worst := 0.0
    let mut peak := 0.0
    for i in [0:partials] do
      let fr := out[4 * i]![0]!
      let fi := out[4 * i + 1]![0]!
      let er := out[4 * i + 2]![0]!
      let ei := out[4 * i + 3]![0]!
      let mag := Float.sqrt (fr * fr + fi * fi)
      peak := max peak mag
      worst := max worst (Float.sqrt ((fr - er) * (fr - er) + (fi - ei) * (fi - ei)) / (mag + 1e-300))
    IO.println s!"        fc={fc} Hz, res={fmt (res.toFloat / 10) 1}: {partials} forced-mode amps vs envelope amps, worst relative error {sci worst} (largest amp {fmt peak 4})"
    ok := ok && worst < 1e-12
  if ok then
    passGate label "LP envelope at the modal Filter's (fc, Q) ≡ residueComposeEC forced-mode amps a·H(iω_k) (< 1e-12 relative, 4 settings × 12 partials)"
  else
    failGate label "forced-mode amps and envelope amps disagree"

-- ── envelope-algebra ─────────────────────────────────────────────────────

private def isEnvelopedFamily : ArrowTerm → Bool
  | .envelope _ _ (.family _) => true
  | _ => false

def runAlgebra (arena : Arena) : IO Bool := do
  let label := "envelope-algebra"
  let pieces : BuildM (ArrowTerm × ArrowTerm × ArrowTerm × Sig) := do
    let clock ← clockLit
    let a ← sawFamily (← lit 110) none clock 16
    let b ← sawFamily (← lit 165) none clock 16 (wave := .cosine)
    let cutoff : ArrowTerm := .konst (← lit 900)
    let q : ArrowTerm := .konst (← lit 3)
    let w ← lit 5 1
    let env := fun (t : ArrowTerm) => svfEnvelope .bandpass cutoff q t
    pure (env (.sum #[.scale w (.family a), .family b]),
      .sum #[.scale w (env (.family a)), env (.family b)], .family a, w)
  let check : BuildM (Bool × Bool × Sig × Sig) := do
    let (outer, distributed, _, _) ← pieces
    let normalForm := match normalize outer with
      | .sum #[.scale _ x, y] => isEnvelopedFamily x && isEnvelopedFamily y
      | _ => false
    let s1 ← emitTerm (normalize outer)
    let s2 ← emitTerm distributed
    let s3 ← emitTerm outer
    pure (normalForm, s1 == s3, s1, s2)
  let (normalForm, rawEq, s1, s2) ← match (check.run { exprs := arena.exprs }) with
    | .ok (r, _) => pure r
    | .error e => return ← failGate label s!"build: {firstLine e}"
  let refused := fun (t : BuildM ArrowTerm) (needle : String) =>
    match ((do emitTerm (← t)).run { exprs := arena.exprs } : Except String _) with
    | .error e => (e.splitOn needle).length > 1
    | .ok _ => false
  let overGen := refused (do
      let clock ← clockLit
      let osc : Voice := { programName := "FixedSinOsc", wire := fun c => do pure #[⟨⟨0⟩, ← lit 220⟩, ⟨⟨1⟩, c⟩] }
      pure (svfEnvelope .lowpass (.konst (← lit 900)) (.konst (← lit 1)) (.gen osc "osc" clock)))
    "spectral envelope over the generator"
  let overWarp := refused (do
      let clock ← clockLit
      let fam ← sawFamily (← lit 110) none clock 8
      pure (svfEnvelope .lowpass (.konst (← lit 900)) (.konst (← lit 1))
        (.warp (fun c => do sub c (← lit 4096)) (.family fam))))
    "spectral envelope over a warped term"
  let moving := refused (do
      let clock ← clockLit
      let freq ← add (← lit 110) (← toFloatE (← sampleIndex))
      pure (.family (← sawFamily freq none clock 8)))
    "the fundamental must be uniform"
  IO.println s!"        normal form (envelope at the families): {normalForm}; emit(normalize t) = emit t: {rawEq}; envelope over sum ≡ sum of envelopes (same IDs): {s1 == s2}"
  IO.println s!"        refusals: over a generator {overGen}, over a warp {overWarp}, moving fundamental {moving}"
  if normalForm && rawEq && s1 == s2 && overGen && overWarp && moving then
    passGate label "G·(w·a + b) normalizes to w·(G·a) + G·b and emits the identical IDs; opaque/warped/moving cases refused"
  else
    failGate label "envelope algebra or refusals wrong"

def runSpectral (arena : Arena) (resolved : Array (String × ProgramIdx)) : IO (Nat × Nat) := do
  let results := #[
    ← runPlanSize arena,
    ← runOldSaw arena resolved,
    ← runOracle arena,
    ← runResidue arena,
    ← runAlgebra arena]
  pure (results.size, results.filter (!·) |>.size)

end Tropical.Tropicaltest.Spectral
