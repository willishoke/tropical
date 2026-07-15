import Tropical.Tropicaltest.Vocabulary

/-!
# Tropical.Tropicaltest.Patcher

The patcher lowering (downstream-only patch graph → arrow term → slide → emit): chain and fan-out lowering, the modulated-effect node, the direct session-root equivalence, and the typed-vs-flow stage/split differentials.
-/

open Tropical
open Tropical.Plan
open Tropical.Ir (Arena ProgramIdx)

-- ── (h⁸) THE PATCHER LOWERING: a downstream-only patch graph → arrow term ──────
-- The MVP front end. A wire is the effect applied to the upstream term (⋙), a
-- fan-out is the shared upstream term (Δ), a mixer is the sum. L1 (byte-identity
-- vs FlangeSin from a GRAPH) is in the corpus section; here: L2 (a chain graph ≡
-- the hand-built term) and L3 (a fan-out graph renders, with the diagonal).

/-- L2: lowering the chain graph `osc → flange → flange` must byte-equal the
    hand-written nested term (`buildSlideDoubleFlanger`). Graph-lowering ≡
    hand-term ⇒ the front end composes effects exactly as `⋙`. -/
def runLoweringChain (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  match Tropical.EmitArrow.buildDoubleFlangeFromGraph arena resolved,
        Tropical.EmitArrow.buildSlideDoubleFlanger arena resolved with
  | .ok (aG, iG), .ok (aH, iH) =>
    match emitResolvedWire aG iG, emitResolvedWire aH iH with
    | .ok bytesG, .ok bytesH =>
      if bytesG == bytesH then
        passGate "lowering-chain" s!"lower(osc→flange→flange) ≡ hand-built nested term ({bytesG.length}B)"
      else
        failGate "lowering-chain" s!"graph {bytesG.length}B ≠ hand-term {bytesH.length}B"
    | .error e, _ | _, .error e => failGate "lowering-chain" s!"emit: {firstLine e}"
  | .error e, _ | _, .error e => failGate "lowering-chain" s!"build: {firstLine e}"

/-- L3: a fan-out patch — `osc` fanned into two flangers, mixed (the diagonal +
    the product collapse through the lowering). Asserts six generator instances
    (3 per flanger; the source re-derived per tap) and a real, non-silent mix. -/
def runLoweringFanOut (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  match Tropical.EmitArrow.buildFanOutFromGraph arena resolved with
  | .ok (aF, iF) =>
    let ninst := ((aF.program? iF).map (·.decls.size)).getD 0
    match buildAndFinish (.ok (aF, iF)) with
    | .ok plan =>
      match ← renderPlanSamples plan 512 with
      | .ok got =>
        let mut energy : Float := 0.0
        for t in [8:512] do energy := energy + got[t]! * got[t]!
        IO.println s!"        fan-out osc → (flange δ₁ &&& flange δ₂) → mix: {ninst} generator instances (the diagonal re-sources the osc per tap)"
        if ninst == 6 && energy > 1e-6 then
          passGate "lowering-fanout" s!"diagonal + mix through the lowering ({ninst} instances, energy={energy})"
        else
          failGate "lowering-fanout" s!"ninst={ninst} (want 6) energy={energy}"
      | .error e => failGate "lowering-fanout" s!"render: {firstLine e}"
    | .error e => failGate "lowering-fanout" s!"finish: {firstLine e}"
  | .error e => failGate "lowering-fanout" s!"build: {firstLine e}"

/-- Modulated-effect node: a `.fm` node routes one node's signal into a carrier's
    clock (FM/PM). Gated byte-identical against the hand-built carriers the
    bit-exact modulated-clock / PM-of-PM differentials already render: M1 a single
    FM node ≡ `buildFmCarrier`; M2 nested `.fm` nodes ≡ `buildPmPmCarrier`. So the
    `osc → flange → osc.fm` edge lowers to the proven modulated warp. -/
def runModulatedNode (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let cmp := fun (label : String)
      (g h : Except String (Arena × ProgramIdx)) =>
    match g, h with
    | .ok (aG, iG), .ok (aH, iH) =>
      match emitResolvedWire aG iG, emitResolvedWire aH iH with
      | .ok bg, .ok bh =>
        if bg == bh then (true, s!"  PASS  modulated-node/{label}  graph fm node ≡ hand-built carrier ({bg.length}B)")
        else (false, s!"  FAIL  modulated-node/{label}  graph {bg.length}B ≠ carrier {bh.length}B")
      | .error e, _ | _, .error e => (false, s!"  FAIL  modulated-node/{label}  emit: {firstLine e}")
    | .error e, _ | _, .error e => (false, s!"  FAIL  modulated-node/{label}  build: {firstLine e}")
  let (ok1, msg1) := cmp "fm"
    (Tropical.EmitArrow.buildFmFromGraph 2000 200 3 arena resolved)
    (Tropical.EmitArrow.buildFmCarrier "FmRef" 2000 200 3 arena resolved)
  IO.println msg1
  let (ok2, msg2) := cmp "pm-of-pm"
    (Tropical.EmitArrow.buildPmPmFromGraph 2000 200 50 3 2 arena resolved)
    (Tropical.EmitArrow.buildPmPmCarrier "PmRef" 2000 200 50 3 2 arena resolved)
  IO.println msg2
  pure (ok1 && ok2)

-- ── (h¹⁰) Stage differential: intern-time attribute vs the flow pass ──────────
-- Phase 1 of the typed stage-0 refactor. The typed side (StageSig at
-- `intern`, resolved along the partition recursion) must never classify an
-- instruction LATER than the plan-level flow pass (`Stage0.classify`) — the
-- flow pass is the trusted reference, and typed ⊑ flow means the attribute
-- is at least as precise everywhere and wrong nowhere the flow pass can
-- see. Strictly-earlier divergences are expected in exactly one category —
-- fold-valued wire crossings the flow pass can't see through (its slot
-- availability rule stops at the surviving writer) — and are reported, not
-- failed; Phase 2's byte-identical-audio differential is the semantic gate
-- on hoisting them.
/-- The v1 array/param/input conservatism overlay: the flow pass pins
    these to s1 at the *instruction* level, so the typed side must be
    compared under the same placement rule. -/
private def overlayS1 (i : Tropical.Plan.NInstr) : Bool :=
  i.tag == "ReduceBegin" || i.tag == "ReduceEnd"
  || (match i.dst with
    | .array _ => true | .sessionArray _ => true | _ => false)
  || i.args.any fun a => match a with
    | .arrayReg _ | .sessionArrayReg _ | .param _ _ | .input _ _ | .loopIdx => true
    | _ => false

def runStageDifferential : IO Bool := do
  let entries ← (System.FilePath.mk "patches").readDir
  let names := (entries.filterMap fun e =>
    if e.fileName.endsWith ".json" then some e.fileName else none).qsort (· < ·)
  let mut ok := true
  let mut compared := 0
  let mut skipped := 0
  let mut unmapped := 0
  let mut divergent := 0
  for fn in names do
    match ← compilePatchStaged s!"patches/{fn}" with
    | .error _ => skipped := skipped + 1
    | .ok (plan, typedBlocks) =>
      let mut flowBlocks : Array (Array Tropical.Plan.NInstr) := #[]
      for f in plan.instanceFunctions do
        flowBlocks := flowBlocks ++ Tropical.Ir.Stage0.collectBlocks f
      if typedBlocks.size != flowBlocks.size then
        IO.println s!"  FAIL  stage-diff/{fn}  block count: typed {typedBlocks.size}, flow {flowBlocks.size}"
        ok := false
        continue
      let (flowStages, _) := Tropical.Ir.Stage0.classify plan
      let linear := flowBlocks.flatten
      let typedLinear := typedBlocks.flatten
      if typedLinear.size != linear.size || flowStages.size != linear.size then
        IO.println s!"  FAIL  stage-diff/{fn}  length: typed {typedLinear.size}, instrs {linear.size}, flow {flowStages.size}"
        ok := false
        continue
      for idx in [0:linear.size] do
        match typedLinear[idx]! with
        | none => unmapped := unmapped + 1
        | some t0 =>
          let t := if overlayS1 linear[idx]! then Tropical.Ir.Stage.s1 else t0
          let f := flowStages[idx]!
          if !(t.le f) then
            IO.println s!"  FAIL  stage-diff/{fn}  instr {idx}: typed {repr t} > flow {repr f} ({linear[idx]!.tag})"
            ok := false
          else if t != f then
            divergent := divergent + 1
          compared := compared + 1
  if ok then
    IO.println (s!"  PASS  stage-diff  typed ⊑ flow over {compared} instructions "
      ++ s!"({divergent} strictly earlier, {unmapped} unmapped"
      ++ s!"{if skipped > 0 then s!"; {skipped} non-session skipped" else ""})")
  pure ok

-- ── (h¹¹) Split equivalence: typed split ≡ flow split, in rendered bytes ──────
-- Phase 2's semantic gate. The typed split hoists strictly more (the
-- fold-crossing category), and every extra hoist must move NO output bit:
-- render each corpus patch through both splits and require byte equality.
def runSplitEquiv : IO Bool := do
  let entries ← (System.FilePath.mk "patches").readDir
  let names := (entries.filterMap fun e =>
    if e.fileName.endsWith ".json" then some e.fileName else none).qsort (· < ·)
  let mut ok := true
  let mut matched := 0
  let mut skipped := 0
  for fn in names do
    match ← compilePatchStaged s!"patches/{fn}" with
    | .error _ => skipped := skipped + 1
    | .ok (plan, blocks) =>
      let typed ← renderTypedBytes plan blocks
      match ← renderIrBytes plan with
      | .error e =>
        IO.println s!"  FAIL  split-equiv/{fn}  flow render: {firstLine e}"; ok := false
      | .ok flow =>
        if typed == flow then matched := matched + 1
        else
          IO.println s!"  FAIL  split-equiv/{fn}  typed and flow renders differ"
          ok := false
  if ok then
    IO.println (s!"  PASS  split-equiv  typed split ≡ flow split byte-for-byte "
      ++ s!"({matched} patches{if skipped > 0 then s!"; {skipped} non-session skipped" else ""})")
  pure ok

-- The gate ledger below is one long do-block; its elaboration depth tracks the
-- gate count, and the default 512 is now too small.
