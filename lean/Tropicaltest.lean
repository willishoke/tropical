import Tropical.Ffi
import Tropical.Engine
import Tropical.StagedLoad
import Tropical.Playground
import Tropical.Plan
import Tropical.Ir.EmitLlvm
import Tropical.Ir.EmitMsl
import Tropical.PlanDecode
import Tropical.Parse.Surface.Markdown
import Tropical.Parse.Raise
import Tropical.Ir.Elaborator
import Tropical.Ir.Strata
import Tropical.Ir.Core
import Tropical.Ir.CompileResolved
import Tropical.Compile
import Tropical.EmitArrow
import Tropical.Stdlib
import Tropical.Testing.ArrowFixtures
import Tropical.Testing.EngineMirror
import Tropical.Testing.PlanWire
import Lean.Data.Json
import Tropical.Tropicaltest.Patcher

/-!
# tropicaltest — the post-TS golden + native-equiv runner (Phase 8)

Replaces `scripts/validate_stdlib.ts` and the native equiv suite. Anchored by
the **frozen audio goldens** (the correctness floor) plus the **native
mode-equiv** cross-check (fused vs microkernel — both ship), all driven off
`Tropical.Ffi.Runtime` (the same `libtropical` the JIT serves). No TS, no koffi.

Goldens reproduce because the Lean engine emits the same plan (`diff-plan`)
and the Lean FFI renders the same dylib byte-for-byte (`diff-render`).
-/


open Tropical
open Tropical.Plan
open Tropical.Ir (Arena ProgramIdx)

set_option maxRecDepth 1024 in
def main (args : List String) : IO UInt32 := do
  let writeMode := args.contains "--write"
  let mut failed := 0
  let mut total := 0

  -- ── (a) Patch audio goldens (tests/golden/*.hash) ──────────────────────────
  IO.println "patch goldens:"
  for name in ← sortedNames "tests/golden" ".hash" do
    let patchPath := s!"patches/{name}.json"
    if ← System.FilePath.pathExists patchPath then
      total := total + 1
      if !(← runGolden writeMode name patchPath s!"tests/golden/{name}.hash") then failed := failed + 1

  -- ── (b) Migration audio goldens (tests/golden/migration/*.json) ────────────
  IO.println "migration goldens:"
  for fixture in ← sortedNames "tests/golden/migration" ".json" do
    let goldenText ← IO.FS.readFile s!"tests/golden/migration/{fixture}.json"
    let expected? : Option String := do
      let g ← (Lean.Json.parse goldenText).toOption
      let h ← (g.getObjVal? "hash").toOption
      h.getStr?.toOption
    let fixText ← IO.FS.readFile s!"tests/fixtures/flat_plan/{fixture}.json"
    let input? : Option Lean.Json := do
      let f ← (Lean.Json.parse fixText).toOption
      (f.getObjVal? "input").toOption
    match expected?, input? with
    | some expected, some input =>
      total := total + 1
      let tmpPatch := "/tmp/tropicaltest-fixture.json"
      IO.FS.writeFile tmpPatch input.compress
      if !(← checkGoldenHash fixture tmpPatch expected) then failed := failed + 1
    | _, _ => IO.println s!"  SKIP  {fixture}  (missing hash/input)"

  -- ── (c) Synthetic op-coverage: EmitLlvm over the rare ops, frozen hash ─────
  -- The patch corpus exercises 24 of 29 ops; this funnels the rest
  -- (GreaterEq, NotEqual, Or, BitOr, BitNot, FloorDiv, Sqrt, Floor, Ceil,
  -- Abs, ToInt, ToBool, Not) through one sink. The expected hash was frozen
  -- from the C++ plan compiler before it was retired (Phase 2), so this
  -- catches any EmitLlvm regression on those ops now that the differential
  -- oracle is gone.
  IO.println "op coverage (EmitLlvm, golden hash):"
  total := total + 1
  match ← renderIrBytes opCoveragePlan with
  | .error e => IO.println s!"  FAIL  op-coverage  {firstLine e}"; failed := failed + 1
  | .ok bytes =>
    let got ← sha256Hex bytes
    let expected := "9d47595cec2e690076b395ca072c03fc20cb8ba838a7b8ac60c16a91da0ea1b8"
    if got == expected then IO.println "  PASS  op-coverage"
    else
      IO.println s!"  FAIL  op-coverage  expected {expected.take 16} got {got.take 16}"
      failed := failed + 1

  -- ── (c⁗) Reduce region: loop ≡ unrolled, frozen hash (banks slice 3a) ──────
  IO.println "reduce coverage (ReduceBegin/End ≡ unrolled, EmitLlvm):"
  total := total + 1
  if !(← runReduceCoverage) then failed := failed + 1

  -- ── (c⁗′) Region-aware Stage0: an all-s0 region hoists as a unit (WS3a) ────
  IO.println "banks region hoist (all-s0 reduce region → coefficient kernel):"
  total := total + 1
  if !(← runBanksRegionHoist) then failed := failed + 1

  -- ── (c″) Stage differential: intern-time attribute ⊑ the flow pass ─────────
  IO.println "stage differential (typed StageSig vs Stage0 flow classification):"
  total := total + 1
  if !(← runStageDifferential) then failed := failed + 1

  -- ── (c‴) Split equivalence: typed split ≡ flow split, rendered bytes ───────
  IO.println "split equivalence (typed hoist ≡ flow hoist, byte-for-byte):"
  total := total + 1
  if !(← runSplitEquiv) then failed := failed + 1

  -- ── (d) let-binding serialization order (ordered-array round-trip) ─────────
  IO.println "let serialization order:"
  total := total + 1
  if !(← runLetRoundtrip) then failed := failed + 1

  -- ── (f) CF goldens (tests/golden/cf/*.hash) — the closed-form corpus ───────
  -- The corpus that must stay green through every phase of the CF-only
  -- migration (it is rendered via the same path as the legacy goldens but only
  -- ever holds register-free, closed-form-in-τ patches). Same shape as (a):
  -- scan the dir, compile patches/<name>.json fused, freeze the render hash.
  IO.println "cf goldens:"
  for name in ← sortedNames "tests/golden/cf" ".hash" do
    let patchPath := s!"patches/{name}.json"
    if ← System.FilePath.pathExists patchPath then
      total := total + 1
      if !(← runGolden writeMode name patchPath s!"tests/golden/cf/{name}.hash") then failed := failed + 1

  -- ── (f′) MSL emitter goldens: the Metal backend's codegen, text-frozen ─────
  IO.println "msl emitter (EmitMsl text-frozen + the f64 fold):"
  for (name, patchPath) in [
      ("pure-sine-440", "web/patches/pure-sine-440.json"),
      ("tz-flanger", "web/patches/tz-flanger.json"),
      ("reverse_reverb", "patches/reverse_reverb.json")] do
    total := total + 1
    if !(← runMslGolden writeMode name patchPath) then failed := failed + 1
  total := total + 1
  if !(← runMslFold) then failed := failed + 1

  -- ── (g) CF-only enforcement: cfOnly strata mode rejects per-sample state ───
  IO.println "cf-only enforcement (reg/next unrepresentable):"
  total := total + 1
  if !(← runCfOnly "CfProbe" cfOnlyRejectSrc (expectReject := true)) then failed := failed + 1
  total := total + 1
  if !(← runCfOnly "Sin" (← IO.FS.readFile "stdlib/Sin.md") (expectReject := false)) then
    failed := failed + 1

  -- ── (h) EmitArrow arrow laws (slice 3): warp algebra ≡ in rendered audio ────
  IO.println "arrow laws (warp algebra ≡ byte-identical audio):"
  match ← arrowElabStdlib with
  | .error e =>
    IO.println s!"  FAIL  arrow-laws  elaborate stdlib: {firstLine e}"
    total := total + 13; failed := failed + 13
  | .ok (arena, resolved) =>
    -- ── (h′) EmitArrow corpus gate: EmitArrow's emit ≡ strata's emit byte-wise ─
    -- The cutover instrument (phase C1). FlangeSin/ReversibleComb generalize the
    -- slices-1/2 byte-gate (formerly an external `diff` of two diffcli verbs);
    -- FixedSinOsc builds the foundational voice DIRECTLY (phasor + Sin poly, no
    -- sourced instance) and is byte-identical to `emit-stdlib FixedSinOsc`.
    IO.println "emitarrow corpus gate (EmitArrow emit ≡ strata emit, byte-identical):"
    total := total + 1
    if !(← runEmitCorpusGate "FlangeSin" "FlangeSin" arena resolved
          Tropical.EmitArrow.buildFlanger) then
      failed := failed + 1
    total := total + 1
    if !(← runEmitCorpusGate "ReversibleComb" "ReversibleComb" arena resolved
          Tropical.EmitArrow.buildReversibleComb) then
      failed := failed + 1
    total := total + 1
    if !(← runEmitCorpusGate "FixedSinOsc" "FixedSinOsc" arena resolved
          Tropical.EmitArrow.buildFixedSinOsc) then
      failed := failed + 1
    -- The fixed-point DATAPATH sine (scope A): the Sig builder and the literate
    -- .md describe one algorithm, byte-identical through the same emit recipe.
    total := total + 1
    if !(← runEmitCorpusGate "FixedSin" "FixedSin" arena resolved
          Tropical.EmitArrow.buildFixedSin) then
      failed := failed + 1
    -- products/MIMO: MorphOsc — a real multi-port body (ClockPhasor ⋙ (saw &&&
    -- Sin) ⋙ crossfade) built from the cartesian combinators, byte-identical.
    total := total + 1
    if !(← runEmitCorpusGate "MorphOsc" "MorphOsc" arena resolved
          Tropical.EmitArrow.buildMorphOsc) then
      failed := failed + 1
    -- THE SLIDE (WARP-PUSH), Test 1: build FlangeSin from the DOWNSTREAM-insert
    -- form (osc ⋙ flange, warps unreduced), run the slide, emit — byte-identical
    -- to stdlib FlangeSin. The compiler turns "flanger dropped downstream" into
    -- "oscillator read at warped clocks." First compiler-driven downstream→upstream.
    total := total + 1
    if !(← runEmitCorpusGate "FlangeSinSlide" "FlangeSin" arena resolved
          Tropical.EmitArrow.buildFlangerViaSlide) then
      failed := failed + 1
    -- THE PATCHER LOWERING, L1: lower the GRAPH `osc → flange` (instances + a
    -- downstream wire), slide, emit — byte-identical to stdlib FlangeSin. The
    -- user's patch graph, lowered end to end, reaches the exact hand-written
    -- program. This is the MVP front end hitting the frozen artifact.
    total := total + 1
    if !(← runEmitCorpusGate "FlangeFromGraph" "FlangeSin" arena resolved
          Tropical.EmitArrow.buildFlangeFromGraph) then
      failed := failed + 1
    -- ── (h″) Entry-codec equivalence (Stage 1 de-risk): the builder's stored
    -- catalog entry (resolved codec + port metadata) ≡ the bridge's, byte-wise.
    -- This is what `registerResolved` will store and session compiles link.
    IO.println "entry-codec equivalence (builder concreteEntry ≡ bridge, byte-identical):"
    total := total + 1
    if !(← runEntryEquivGate "FixedSinOsc" "FixedSinOsc" arena resolved
          Tropical.EmitArrow.buildFixedSinOsc) then
      failed := failed + 1
    total := total + 1
    if !(← runEntryEquivGate "FixedSin" "FixedSin" arena resolved
          Tropical.EmitArrow.buildFixedSin) then
      failed := failed + 1
    total := total + 1
    if !(← runEntryEquivGate "MorphOsc" "MorphOsc" arena resolved
          Tropical.EmitArrow.buildMorphOsc) then
      failed := failed + 1
    total := total + 1
    if !(← runEntryEquivGate "FlangeSin" "FlangeSin" arena resolved
          Tropical.EmitArrow.buildFlanger) then
      failed := failed + 1
    total := total + 1
    if !(← runEntryEquivGate "ReversibleComb" "ReversibleComb" arena resolved
          Tropical.EmitArrow.buildReversibleComb) then
      failed := failed + 1
    -- ── (h‴) Stdlib builder gates: each arrow builder ≡ its bridge program,
    -- plan-wire (resolved-body semantics) AND port surface. These builders are
    -- what replaces the .md/parsed-bridge stdlib at boot.
    IO.println "stdlib builder gates (arrow builder ≡ bridge program, plan + ports):"
    for (nm, bld) in Tropical.EmitArrow.stdlibNewBuilders do
      total := total + 1
      if !(← runStdlibGate nm arena resolved bld) then failed := failed + 1
    IO.println "arrow laws (warp algebra ≡ byte-identical audio):"
    -- ── PER-LAW CARRIER TABLE (load-bearing — see design/fixed-carrier.md) ──
    -- Which value carrier each law rides is a CHOICE, not an accident: laws
    -- whose two sides evaluate the same ops on bit-equal clocks are exact on
    -- EITHER carrier (1-5 ride float to also witness the float path); law 6
    -- deliberately rides FLOAT as the documented reassociation exhibit (the
    -- ±δ tap swap moves bytes in float — denotational-only equality); law 7
    -- is the SAME law on the FIXED carrier where integer-add associativity
    -- makes it byte-exact; laws 8-10 mirror 1/4/5 on the fixed source.
    --   1 inverse ················ float   (clock-exact on either)
    --   2 additive ··············· float   (clock-exact on either)
    --   3 diagonal/fan-out ······· float   (clock-exact on either)
    --   4 reverse-involution ····· float   (clock-exact on either)
    --   5 reverse-swaps-delay ···· float   (clock-exact on either)
    --   6 reverse⋄flanger ········ FLOAT   (the reassociation exhibit: NOT byte)
    --   7 reverse⋄flanger ········ FIXED   (byte-exact — the scope-A property)
    --   8-10 single-source laws ·· FIXED   (byte-exact)
    -- The modal island (modal-bank/-reverse/-direction/-longtau gates above)
    -- rides the fixed datapath end to end as of scope-A B3.
    -- Law 1 — inverse/cancellation:  warp(back δ) ⋙ warp(fwd δ) = id
    total := total + 1
    if !(← runArrowLaw "inverse"
          Tropical.EmitArrow.invLawLhsClock Tropical.EmitArrow.invLawRhsClock
          arena resolved) then
      failed := failed + 1
    -- Law 2 — additive delay/functoriality:
    --   warp(back δ₁) ⋙ warp(back δ₂) = warp(back (δ₁+δ₂))
    total := total + 1
    if !(← runArrowLaw "additive"
          Tropical.EmitArrow.addLawLhsClock Tropical.EmitArrow.addLawRhsClock
          arena resolved) then
      failed := failed + 1
    -- Law 3 — the cartesian diagonal / fan-out: one source ⋙ (flanger δ₁ &&&
    --   flanger δ₂) ≡ two independent (source+flanger) pairs (+ the COST story).
    total := total + 1
    if !(← runDiagonalLaw arena resolved) then
      failed := failed + 1
    -- ── slice 5 — REVERSE (the moat) as warp(neg) ───────────────────────────
    -- Law 4 — involution:  warp(neg) ⋙ warp(neg) = id  (byte-identical: −(−x)=x)
    total := total + 1
    if !(← runArrowLaw "reverse-involution"
          Tropical.EmitArrow.revInvolutionLhsClock Tropical.EmitArrow.revInvolutionRhsClock
          arena resolved) then
      failed := failed + 1
    -- Law 5 — reverse-swaps-delay:
    --   warp(neg) ⋙ warp(back δ) = warp(fwd δ) ⋙ warp(neg)  (byte-identical: −clk+δ)
    total := total + 1
    if !(← runArrowLaw "reverse-swaps-delay"
          Tropical.EmitArrow.revSwapLhsClock Tropical.EmitArrow.revSwapRhsClock
          arena resolved) then
      failed := failed + 1
    -- Law 6 — reverse commutes with the symmetric flanger:
    --   warp(neg) ⋙ flanger ≡ flanger ⋙ warp(neg)  (denotational ≡; ±δ tap-sum
    --   reassociates in float — the finding; byte-exact only with fixed-point values)
    total := total + 1
    if !(← runReverseFlangerCommute arena resolved) then
      failed := failed + 1
    -- ── slice 6 — a FIXED-POINT VALUE carrier: slice-5's law 6 byte-exact ────
    -- THE GATE — Law 7: reverse commutes with the FIXED-POINT flanger,
    --   warp(neg) ⋙ flanger ≡ flanger ⋙ warp(neg) over the integer source +
    --   integer shift-add mix. Slice 5's float version was 1271/4096 differing;
    --   integer add is associative so this is now BYTE-IDENTICAL (0/N).
    total := total + 1
    if !(← runReverseFlangerCommuteFixedpoint arena) then
      failed := failed + 1
    -- Laws 8-10: the single-source warp laws over the FIXED-POINT source —
    --   involution / reverse-swaps-delay / additive — all byte-exact (the clock
    --   side is exact int64), mirroring the float `runArrowLaw` cases.
    total := total + 1
    if !(← runFixedSourceLaw "reverse-involution-fixedpoint"
          Tropical.EmitArrow.revInvolutionLhsClock Tropical.EmitArrow.revInvolutionRhsClock
          arena) then
      failed := failed + 1
    total := total + 1
    if !(← runFixedSourceLaw "reverse-swaps-delay-fixedpoint"
          Tropical.EmitArrow.revSwapLhsClock Tropical.EmitArrow.revSwapRhsClock
          arena) then
      failed := failed + 1
    total := total + 1
    if !(← runFixedSourceLaw "additive-fixedpoint"
          Tropical.EmitArrow.addLawLhsClock Tropical.EmitArrow.addLawRhsClock
          arena) then
      failed := failed + 1
    -- ── (h″) convolution stress test: the bubble executed; oracle independent of
    --   the lowering (array-shift of the bare osc), so it can catch a wrong-but-
    --   self-consistent lowering — correct-by-facade goes red here, not green.
    IO.println "convolution stress test (clock-warp FIR ≡ independent array-shift conv):"
    total := total + 1
    if !(← runConvolutionOracle arena resolved) then
      failed := failed + 1
    -- ── (h‴) modulated clock: a fractional NONLINEAR warp vs an independent
    --   closed-form reference, calibrated against the bare osc — tests that the
    --   bubble is not a side-effect of affineness.
    IO.println "modulated-clock stress test (fractional nonlinear warp ≡ closed form):"
    total := total + 1
    if !(← runModulatedClock arena resolved) then
      failed := failed + 1
    -- ── (h⁗) PM-of-PM: the modulator is itself a warped oscillator; bit-exact
    --   against a 3-level nested standard rep ⇒ the substitution composes.
    IO.println "pm-of-pm stress test (nested warp ≡ nested standard rep):"
    total := total + 1
    if !(← runPmPm arena resolved) then
      failed := failed + 1
    -- ── (h⁵) negative-time boundary (the moat): a delay pulls the clock < 0;
    --   random access gives the exact backward sine where a stream would zero-pad.
    IO.println "negative-time boundary (random access ≠ streaming zero-pad):"
    total := total + 1
    if !(← runNegativeClock arena resolved) then
      failed := failed + 1
    -- ── (h⁶) products / MIMO: a real multi-port body from the cartesian
    --   combinators vs a straight-line standard rep — the DATA axis (the warp
    --   gates above are all the CLOCK axis).
    IO.println "products/MIMO standard-rep differential (multi-port body ≡ closed form):"
    total := total + 1
    if !(← runMorphOscDifferential arena resolved) then
      failed := failed + 1
    -- ── (h⁷) THE SLIDE (WARP-PUSH): the compiler pushes a downstream effect's
    --   warps up to the generators. R1 (commute past arr) + the cascade
    --   multiplicity; Test 1 (byte-identity vs FlangeSin) is in the corpus block.
    IO.println "warp-push slide (downstream insert → upstream warp, by the compiler):"
    total := total + 1
    if !(← runSlidePastArr arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runSlideCascade arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runSlideProd arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runBootstrapSin arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runBootstrapExp arena resolved) then
      failed := failed + 1
    IO.println "fixed-point datapath sine (scope A — the sample values in i64):"
    total := total + 1
    if !(← runFixedSinAccuracy arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runFixedSinLongTau arena resolved) then
      failed := failed + 1
    IO.println "modal island (decaying-resonator bank as a term over the clock):"
    total := total + 1
    if !(← runModalBank arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runBanksAsData arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runBanksAsDataDir arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runBanksFloat arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runBanksFoldTrunk arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runBanksColumnize arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runBanksNested arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runBanksNestedMsl arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runBanksColumnizeBail arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runModalDegree arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runModalReverse arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runLongTauModal arena resolved) then
      failed := failed + 1
    IO.println "residue calculus (voice ⋙ reverb composed at build time):"
    total := total + 1
    if !(← runResidueMoments arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runModalReverb arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runModalDirection arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runModalSway arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runResidueDegenerate arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runResidueCollected arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runResidueSymbolic arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runModalPatch arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runModalLive arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runBanksStaging arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runMslColumnGuard arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runBanksBench arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runBanksCount arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runBanksCountCache arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runModalFilter arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runModalAddr arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runRealizedReport) then
      failed := failed + 1
    total := total + 1
    if !(← runManifestDisciplines arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runVocabDriven arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runDeadSlotLint arena resolved) then
      failed := failed + 1
    -- ── (h⁸) THE PATCHER LOWERING: downstream-only patch graph → arrow term →
    --   slide → emit. L1 (byte-identity vs FlangeSin from a graph) is in the
    --   corpus block; here L2 (chain graph ≡ hand-term) and L3 (fan-out + mix).
    IO.println "patcher lowering (downstream-only patch graph → arrow term):"
    total := total + 1
    if !(← runLoweringChain arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runLoweringFanOut arena resolved) then
      failed := failed + 1
    -- modulated-effect node: signal-into-clock (FM/PM), the osc → flange →
    -- osc.fm edge; ≡ the bit-exact-proven FM / PM-of-PM carriers.
    total := total + 1
    if !(← runModulatedNode arena resolved) then
      failed := failed + 1

  IO.println ""
  IO.println s!"{total - failed}/{total} passed"
  return if failed == 0 then 0 else 1
