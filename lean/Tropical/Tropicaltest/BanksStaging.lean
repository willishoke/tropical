import Tropical.Tropicaltest.SeamSweep

/-!
# Tropical.Tropicaltest.BanksStaging

Stage-0 banking gates: region hoist (an all-s0 reduce region moves as a unit), staged column kernels, the MSL column-binding guard, the banks benchmark, and trip-count-as-data (live count, knob-invariant cache, modal filter/address).
-/

open Tropical
open Tropical.Plan
open Tropical.Ir (Arena ProgramIdx)

-- ── Region-aware Stage0 (WS3a): an all-s0 reduce region hoists AS A UNIT ─────
section RegionHoist
open Tropical.Plan

/-- A reduce region whose WHOLE computation is coefficient-shaped: the table
    Packs from live param slots (never written in-plan → s0-external), the body
    indexes it at `loopIdx` and weights by (k+1) — no τ anywhere inside the
    region. An s1 consumer mixes the accumulator with τ so the render is
    per-sample. `dyn` adds the trip-count-as-data operand (`ReduceBegin`
    args[1] ← the `param:n` slot, default 2 < capacity 4, so the dynamic
    render differs from the static one — a real clamp path, not a no-op).
    Temps: acc=1, body 2..5, tail 6..8. Built synthetically (FlatPlan + hand
    stage blocks through `hoistTyped`, the reduce-coverage pattern): no
    surface graph reaches an s0 region yet — the modal banks all read τ. -/
private def regionS0PlanOf (dyn : Bool) : FlatPlan :=
  let countOp? : Option NOperand := if dyn then some (.slot 3 .float) else none
  let body : Array NInstr := #[
    instrPack 0 #[.slot 1 .float, .slot 2 .float, .slot 1 .float, .slot 2 .float],
    instrReduceBegin 1 (cF 0) 4 .float countOp?,
    instrIndex 2 #[.arrayReg 0, .loopIdx] .float,               -- v = table[k]
    instrScalar "Add" 3 #[.loopIdx, cI 1] .int,                 -- k+1
    instrScalar "ToFloat" 4 #[rgI 3] .float,
    instrScalar "Mul" 5 #[rgF 2, rgF 4] .float,                 -- v·(k+1)
    instrScalar "Add" 1 #[rgF 1, rgF 5] .float,
    instrReduceEnd 1 .float,
    instrScalar "ToFloat" 6 #[Tropical.Plan.opTick] .float,     -- the s1 consumer
    instrScalar "Mod" 7 #[rgF 6, cF 64] .float,
    instrScalar "Mul" 8 #[rgF 1, rgF 7] .float,
    instrWriteSlot 0 (rgF 8)]
  let inst := InstanceFunction.mk "root" "root" #[] body #[] 0 0 9 #[]
  { sampleRate := jn 44100, compilationMode := .fused,
    arraySlotNames := #["table"], registerCount := 9, arraySlotCount := 1,
    arraySlotSizes := #[4], instanceFunctions := #[inst],
    sinks := #[{ inputs := #[0], gain := jn 1, target := 0 }],
    sources := defaultSources, slotCount := 4,
    slotNames := #["out", "param:a", "param:b", "param:n"],
    slotDefaults := #[Lean.Json.num (jn 0), Lean.Json.num (jn 5 1),
      Lean.Json.num (jn 25 2), Lean.Json.num (jn 2)] }

/-- Hand stage blocks in the partitioner's shape (`collectBlocks`: preamble,
    body): everything in and around the region is s0 (exactly what the
    intern-time attribute derives with `loopIdx` stage-neutral), the τ tail s1. -/
private def regionS0Stages : Array (Array (Option Tropical.Ir.Stage)) :=
  let s0 : Option Tropical.Ir.Stage := some .s0
  let s1 : Option Tropical.Ir.Stage := some .s1
  #[#[], #[s0, s0, s0, s0, s0, s0, s0, s0, s1, s1, s1, s1]]

/-- THE REGION-HOIST gate (region-aware Stage0, WS3a). A reduce region whose
    body is entirely s0 (param-slot-derived, no clock/tick dependence) plus an
    s1 consumer of its result: the typed split moves the WHOLE region
    (delimiters + body, with its table Pack) into the coefficient stream — the
    audio kernel contains ZERO regions and reads the sum via a `coef:` slot —
    and the render is BYTE-EXACT against the flow split (which never hoists
    regions: effectively the unsplit reference). Checked for the static region
    AND the dynamic-count region (args[1] ← a param slot). This also verifies
    the runtime claim end to end: the coefficient kernel containing a
    `ReduceBegin` region is emitted through the same `EmitLlvm`, JIT-compiled,
    and executed by `run_coeff` before buffer 0. -/
def runBanksRegionHoist : IO Bool := do
  let check := fun (label : String) (dyn : Bool) => do
    let plan := regionS0PlanOf dyn
    match Tropical.Ir.Stage0.hoistTyped plan regionS0Stages with
    | .error e => failGate "banks-region-hoist" s!"{label} split: {firstLine e}"
    | .ok split =>
      let audioReduces := planReduces split.audio
      let coeffReduces := match split.coeff? with | some c => planReduces c | none => 0
      let coeffFills := match split.coeff? with | some c => planArrayFills c | none => 0
      let hasCoefSlot := split.audio.slotNames.any (· == "coef:0")
      let cols := split.audio.coeffArraySlots
      let typed ← renderTypedBytes plan regionS0Stages
      match ← renderIrBytes plan with
      | .error e => failGate "banks-region-hoist" s!"{label} flow render: {firstLine e}"
      | .ok flow =>
        let n := min typed.size flow.size
        let mut bitDiff := 0
        for i in [0:n] do
          if typed[i]! != flow[i]! then bitDiff := bitDiff + 1
        let mut energy : Float := 0.0
        for s in decodeF64LE typed do energy := energy + s * s
        IO.println (s!"        {label}: regions audio={audioReduces} coeff={coeffReduces} · "
          ++ s!"coeff fills={coeffFills} · coef:0 slot={hasCoefSlot} · columns={cols} · "
          ++ s!"typed≡flow bitDiff={bitDiff}/{n} · E={energy}")
        pure (audioReduces == 0 && coeffReduces == 1 && coeffFills == 1
          && hasCoefSlot && cols == #[0] && typed.size == flow.size
          && bitDiff == 0 && energy > 1e-6)
  let okS ← check "static count" false
  let okD ← check "dynamic count (slot, 2 < capacity 4)" true
  if okS && okD then
    IO.println ("  PASS  banks-region-hoist  all-s0 region moves AS A UNIT "
      ++ "(delimiters + body + table Pack) to the coeff kernel; audio is region-free, "
      ++ "reads the sum via coef:0; typed ≡ flow byte-exact, static AND dynamic count")
    pure true
  else
    failGate "banks-region-hoist" s!"static={okS} dynamic={okD}"

end RegionHoist

/-- THE PER-ARRAY STAGING gate (banks-as-data blocker 3). `modal-live` proves the
    banked lowering still renders correctly under live knobs; this proves the
    PAYOFF structurally — with the banked lowering on, a live-param bank's
    coefficient columns (`Pack` fills) move OUT of the audio kernel and INTO the
    s0 coefficient kernel, and the audio kernel is left array-fill-free (its
    in-loop `Index` reads the shared, coeff-filled storage). Adapts to the flag:
    flag on ⇒ columns hoist; flag off ⇒ unrolled, no columns (no spurious hoist). -/
def runBanksStaging (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  -- A bare resonator → out: a UNIFORM (deg-0) forward bank with live freq/decay,
  -- so with the flag on it banks (a reverb would compose to a possibly-ragged
  -- bank via residueComposeEC's deg-1 coincident poles — not the payoff we gate).
  let src := "{\"nodes\":[" ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"res\"]}}],\"out\":\"out\"}"
  match Lean.Json.parse src with
  | .error e => failGate "banks-staging" s!"json: {e}"
  | .ok j =>
  match Tropical.Playground.compilePlanPure arena resolved j with
  | .error e => failGate "banks-staging" s!"compile: {firstLine e}"
  | .ok compiled =>
    let plan := compiled.plan
    let stageBlocks := compiled.stageBlocks
    match Tropical.Ir.Stage0.hoistTyped plan stageBlocks with
    | .error e => failGate "banks-staging" s!"split: {firstLine e}"
    | .ok split =>
      let flagOn := Tropical.EmitArrow.banksTableEnabled
      let coeffFills := match split.coeff? with | some c => planArrayFills c | none => 0
      let audioFills := planArrayFills split.audio
      let reduces := planReduces split.audio
      -- banks-region-stay: the resonator's region is CLOCK-DEPENDENT (the
      -- body reads τ), so the whole-region move (banks-region-hoist's
      -- positive) must NOT fire here — the region stays in the audio
      -- kernel, only its live columns hoist.
      let coeffReduces := match split.coeff? with | some c => planReduces c | none => 0
      -- Correctness: render the SAME banked plan via the TYPED split (live columns
      -- hoisted to the coeff kernel, audio reads shared array_ptrs) and via the
      -- FLOW split (arrays stay s1, everything in audio). Byte-identical ⇒
      -- per-array staging preserves the render (the coeff kernel fills the shared
      -- coefficient storage the audio kernel reads — run_coeff before buffer 0).
      let typedBytes ← renderTypedBytes plan stageBlocks
      match ← renderIrBytes plan with
      | .error e => failGate "banks-staging" s!"flow render: {firstLine e}"
      | .ok flowBytes =>
        let n := min typedBytes.size flowBytes.size
        let mut bitDiff := 0
        for i in [0:n] do
          if typedBytes[i]! != flowBytes[i]! then bitDiff := bitDiff + 1
        let mut energy : Float := 0.0
        for s in decodeF64LE typedBytes do energy := energy + s * s
        IO.println s!"        resonator→out (live freq/decay), banks-table={flagOn}:"
        IO.println s!"        result   reduce regions audio={reduces} coeff={coeffReduces} · array fills coeff={coeffFills} audio={audioFills} · typed≡flow bitDiff={bitDiff}/{n} · E={energy}"
        let renderOk := bitDiff == 0 && energy > 1e-6
        if flagOn then
          -- The bank loops and its LIVE columns (incr←freq, sigma←decay) hoist to
          -- the s0 kernel. CONST columns (cre=1/k^1.1, cim=0) stay in audio as fold
          -- Packs (one instruction each, LICM'd out of the sample loop). A Pack is
          -- O(1) instructions regardless of K, so the audio kernel is flat in mode
          -- count; per-array staging moves the LIVE coefficient mass off the audio
          -- thread. Byte-identity to the flow split proves the shared-array crossing.
          -- banks-region-stay: exactly ONE region, in the AUDIO kernel — the
          -- τ-reading body keeps the loop per-sample; the coeff kernel is
          -- region-free (the whole-region move must not fire on a clock-
          -- dependent bank).
          if reduces == 1 && coeffReduces == 0 && coeffFills > 0 && renderOk then
            passGate "banks-staging" s!"bank looped ({reduces} region, in audio; 0 in coeff — clock-dependent region stays); {coeffFills} live column(s) → s0 kernel via shared array_ptrs; {audioFills} const baked; typed split ≡ flow byte-exact"
          else
            failGate "banks-staging" s!"flag on: reduces={reduces} coeffReduces={coeffReduces} coeff={coeffFills} renderOk={renderOk}"
        else
          if reduces == 0 && coeffReduces == 0 && coeffFills == 0 && renderOk then
            passGate "banks-staging" "flag off: unrolled bank, no loop/columns, typed ≡ flow byte-exact"
          else
            failGate "banks-staging" s!"flag off: reduces={reduces} coeffReduces={coeffReduces} coeff={coeffFills} renderOk={renderOk}"

/-- The Metal column-crossing gate (WS4; supersedes the WS0 refusal
    tripwire). Hoisted coefficient columns (`coeff_array_slots`) cross
    to the GPU as ONE packed `constant float* coeff_columns
    [[buffer(3)]]` device buffer, filled host-side by the stage-0
    coefficient kernel and uploaded from the generation `process()`
    captures. The gate is structural, on the emitted TEXT: the typed
    split's audio plan must EMIT (no refusal) with the buffer(3)
    declaration, each hoisted slot read via `coeff_columns[<offset> +
    …]` at its compile-time packed offset, and NO thread-private
    `float arr<s>[` local declared for it — while the columns-free
    UNSPLIT plan keeps the exact 3-binding header (the msl-golden ABI,
    byte-frozen). Under `TROPICAL_BANKS_UNROLL` nothing hoists: both
    emissions clean, both on the plain header (no false positive). -/
def runMslColumnGuard (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let src := "{\"nodes\":[" ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"res\"]}}],\"out\":\"out\"}"
  match Lean.Json.parse src with
  | .error e => failGate "msl-column-guard" s!"json: {e}"
  | .ok j =>
  match Tropical.Playground.compilePlanPure arena resolved j with
  | .error e => failGate "msl-column-guard" s!"compile: {firstLine e}"
  | .ok compiled =>
    let plan := compiled.plan
    let stageBlocks := compiled.stageBlocks
    match Tropical.Ir.Stage0.hoistTyped plan stageBlocks with
    | .error e => failGate "msl-column-guard" s!"split: {firstLine e}"
    | .ok split =>
      let banked := Tropical.EmitArrow.banksTableEnabled
      let cols := split.audio.coeffArraySlots
      let has : String → String → Bool := fun hay needle =>
        (hay.splitOn needle).length > 1
      let plainHeader := "constant TropicalKernelConsts& k             [[buffer(2)]],\n    uint s [[thread_position_in_grid]])"
      let columnsHeader := "constant float*                coeff_columns [[buffer(3)]],\n    uint s [[thread_position_in_grid]])"
      match Tropical.Ir.EmitMsl.emitKernel split.audio,
            Tropical.Ir.EmitMsl.emitKernel plan with
      | .ok splitMsl, .ok unsplitMsl =>
        -- The unsplit plan advertises no columns: exact 3-binding header,
        -- no buffer(3) anywhere (the text-frozen ABI must not move).
        let unsplitClean := has unsplitMsl plainHeader && !(has unsplitMsl "buffer(3)")
        if banked then
          -- Recompute the packed offsets the emitter promises (plan order,
          -- capacity-summed) and check each hoisted slot: read at ITS
          -- offset, no thread-private local.
          let sizes := split.audio.arraySlotSizes
          let binding := has splitMsl columnsHeader
          let mut off := 0
          let mut reads := true
          let mut noLocals := true
          for s in cols do
            if !(has splitMsl s!"coeff_columns[{off} + ") then reads := false
            if has splitMsl s!"float arr{s}[" then noLocals := false
            off := off + max (sizes[s]?.getD 1) 1
          IO.println (s!"        banked={banked} · hoisted columns={cols.size} ({off} floats packed) · "
            ++ s!"buffer(3)={binding} · offset reads={reads} · locals suppressed={noLocals} · unsplit 3-binding={unsplitClean}")
          if cols.size > 0 && binding && reads && noLocals && unsplitClean then
            passGate "msl-column-guard" s!"{cols.size} hoisted column(s) EMIT in column-binding mode: buffer(3) declared, reads at packed offsets, no arrN locals; columns-free plan keeps the frozen 3-binding header"
          else
            failGate "msl-column-guard" s!"banked: cols={cols.size} binding={binding} reads={reads} noLocals={noLocals} unsplitClean={unsplitClean}"
        else
          let splitClean := has splitMsl plainHeader && !(has splitMsl "buffer(3)")
          IO.println s!"        banked={banked} · hoisted columns={cols.size} · split 3-binding={splitClean} · unsplit 3-binding={unsplitClean}"
          if cols.isEmpty && splitClean && unsplitClean then
            passGate "msl-column-guard" "unrolled: no columns hoisted, both emissions on the plain 3-binding header (no false positive)"
          else
            failGate "msl-column-guard" s!"unrolled: cols={cols.size} splitClean={splitClean} unsplitClean={unsplitClean}"
      | .error e, _ =>
        failGate "msl-column-guard" s!"split plan refused (the WS0 stopgap is retired — columns must emit): {firstLine e}"
      | _, .error e =>
        failGate "msl-column-guard" s!"unsplit plan refused: {firstLine e}"

/-- THE COMPILE-FLATNESS BENCHMARK (banks-as-data payoff). Where `banks-staging`
    proves the payoff STRUCTURALLY at one mode count (columns hoist, audio is
    fill-free), this MEASURES it across scale: compile the SAME room at K=6 and
    K=512 modes and show the AUDIO kernel's plan instruction count is flat in K
    (within a small constant) while only the coefficient kernel grows.

    Why the audio kernel is flat: the resonator's freq/decay are LIVE (session
    slots → stage-0), so their coefficient columns (incr←freq, sigma←decay) hoist
    to the s0 coefficient kernel — those fills scale with K there, at knob rate,
    O0. The amps (cre=1/k^1.1, cim=0) are compile-TIME constants, so their columns
    bake into the audio kernel as fold `Pack`s — but a `Pack` is ONE instruction
    regardless of K, so the audio kernel stays flat. The banked audio body is then
    a fixed O(1) reduce region reading the shared, coeff-filled storage.

    NOTE the nuance: this flatness rides on freq/decay being live. A fully-STATIC
    bank would NOT be flat — with no live columns to hoist, its column arithmetic
    stays as fold in the audio kernel and grows with K. True static flatness needs
    blocker 4's fill-as-reduce. Flag off ⇒ the bank is unrolled and the audio
    kernel grows ~linearly in K; we only REPORT that (the documented contrast). -/
def runBanksBench (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let flagOn := Tropical.EmitArrow.banksTableEnabled
  -- A bare resonator → out with live freq/decay, at a graph-configurable mode
  -- count K (the `"partials"` param threaded through Playground.buildNode).
  let mkSrc := fun (k : Nat) => "{\"nodes\":[" ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4,\"partials\":"
      ++ toString k ++ "}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"res\"]}}],\"out\":\"out\"}"
  -- Compile at K, split via the typed stage-0 hoist, and report
  -- (audio-kernel instrs, coeff-kernel instrs).
  let compileAt : Nat → Except String (Nat × Nat) := fun k => do
    let j ← Lean.Json.parse (mkSrc k)
    let compiled ← Tropical.Playground.compilePlanPure arena resolved j
    let split ← Tropical.Ir.Stage0.hoistTyped compiled.plan compiled.stageBlocks
    let audioN := planInstrCount split.audio
    let coeffN := match split.coeff? with | some c => planInstrCount c | none => 0
    pure (audioN, coeffN)
  match compileAt 6, compileAt 512 with
  | .error e, _ => failGate "banks-bench" s!"K=6 compile: {firstLine e}"
  | _, .error e => failGate "banks-bench" s!"K=512 compile: {firstLine e}"
  | .ok (a6, c6), .ok (a512, c512) =>
    let dAudio := if a512 ≥ a6 then a512 - a6 else a6 - a512
    IO.println s!"        bare resonator→out (live freq/decay), banks-table={flagOn}:"
    IO.println s!"        audio-kernel instrs: K=6 → {a6}   K=512 → {a512}   Δ={dAudio}"
    IO.println s!"        coeff-kernel instrs: K=6 → {c6}   K=512 → {c512}"
    if flagOn then
      -- Banked: the audio kernel is a fixed reduce body + O(1) const Packs, flat
      -- in K; the LIVE columns' fills live in the coeff kernel and scale there.
      let flat := dAudio ≤ 8
      let coeffGrows := decide (c512 > c6)
      if flat then
        passGate "banks-bench" s!"flag on: audio kernel FLAT in K (Δ={dAudio} ≤ 8, K=6→512); coeff kernel scales with K ({c6}→{c512}, grows={coeffGrows}) at knob rate"
      else
        failGate "banks-bench" s!"flag on: audio kernel NOT flat (Δ={dAudio} > 8) — a K-dependent audio instruction leaked past the coeff hoist"
    else
      -- Unrolled: no loop, no columns; the whole bank's arithmetic is in the
      -- audio kernel and grows with K. Not a failure — the documented contrast.
      passGate "banks-bench" s!"flag off: unrolled bank, audio kernel GROWS with K ({a6}→{a512}, Δ={dAudio}) — the contrast the banked path removes"

/-- THE TRIP-COUNT gate (trip-count-as-data v1: the room-size knob). A resonator
    with the optional STATIC `partials_max` capacity carries a LIVE `partials`
    slot whose in-kernel read is the bank's effective trip count, clamped to
    capacity — mode count stops being topology. (a) at the default knob
    (= capacity) it renders BIT-EXACT to the fully-static graph at the same
    count; (b) knob at 4 ≡ static partials=4 (the clamped loop visits the same
    mode prefix in unroll order — same ops, same bits; the dynamic plan's
    capacity-sized columns beyond index 4 are never read); (c) knob above
    capacity clamps (≡ the capacity render); (d) knob at 0 sums no modes —
    silence from the bank, the patch's only source. A dynamic-count bank always
    BANKS (a runtime count cannot unroll), so this holds in both flag states of
    TROPICAL_BANKS_UNROLL. The static graph must NOT grow the slot (opt-in:
    `partials_max` absent ⇒ no `param:res.partials`). -/
def runBanksCount (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let staticSrc := fun (k : Nat) => "{\"nodes\":[" ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4,\"partials\":"
      ++ toString k ++ "}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"res\"]}}],\"out\":\"out\"}"
  let dynSrc := "{\"nodes\":[" ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4,\"partials\":16,\"partials_max\":16}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"res\"]}}],\"out\":\"out\"}"
  let compile := fun (src : String) => do
    let j ← Lean.Json.parse src
    Tropical.Playground.compilePlanPure arena resolved j
  -- Load a compiled graph, optionally preset a slot, render one 2048-sample
  -- block (the modal-live pattern: a live slot driven through a render).
  let render := fun (plan : Tropical.Plan.FlatPlan)
      (blocks : Array (Array (Option Tropical.Ir.Stage)))
      (preset? : Option Float) => do
    let rt ← Tropical.Ffi.Runtime.new 2048
    Tropical.StagedLoad.loadTyped rt plan blocks
    let mut slotOk := true
    if let some v := preset? then
      match ← rt.slotIndex? "param:res.partials" with
      | some idx => rt.setSlot idx v
      | none => slotOk := false
    rt.process
    pure (slotOk, decodeF64LE (← rt.outputBytes))
  match compile (staticSrc 16), compile (staticSrc 4), compile dynSrc with
  | .error e, _, _ => failGate "banks-count" s!"static-16 compile: {firstLine e}"
  | _, .error e, _ => failGate "banks-count" s!"static-4 compile: {firstLine e}"
  | _, _, .error e => failGate "banks-count" s!"dynamic compile: {firstLine e}"
  | .ok c16, .ok c4, .ok cd =>
    -- opt-in: the static graph must NOT have grown a partials slot.
    let rtS ← Tropical.Ffi.Runtime.new 2048
    Tropical.StagedLoad.loadTyped rtS c16.plan c16.stageBlocks
    let staticHasSlot := (← rtS.slotIndex? "param:res.partials").isSome
    let (_, s16) ← render c16.plan c16.stageBlocks none
    let (_, s4)  ← render c4.plan c4.stageBlocks none
    let (_, dDef)   ← render cd.plan cd.stageBlocks none          -- knob at its default (16)
    let (ok4, d4)   ← render cd.plan cd.stageBlocks (some 4.0)    -- knob at 4
    let (okC, dC)   ← render cd.plan cd.stageBlocks (some 100.0)  -- above capacity → clamps to 16
    let (okZ, dZ)   ← render cd.plan cd.stageBlocks (some 0.0)    -- zero modes → silence
    let slotLive := ok4 && okC && okZ
    let e16 := energyOf s16
    let dA := bitDiffCount dDef s16
    let dB := bitDiffCount d4 s4
    let dCn := bitDiffCount dC s16
    let eZ := energyOf dZ
    IO.println s!"        resonator partials_max=16 (LIVE partials slot) vs fully-static graphs:"
    IO.println s!"        result   default(16)≡static16 bitDiff={dA}/{s16.size} · knob4≡static4 bitDiff={dB}/{s4.size} · knob100≡static16 bitDiff={dCn}/{s16.size}"
    IO.println s!"        result   E[static16]={e16} · E[knob0]={eZ} · slot live={slotLive} · static graph has slot={staticHasSlot} (want false)"
    if dA == 0 && dB == 0 && dCn == 0 && eZ ≤ 1e-24 && e16 > 1e-6 && slotLive && !staticHasSlot then
      passGate "banks-count" "live trip count ≡ static at 16/4, clamps at 100, silent at 0 — mode count is data, not topology"
    else
      failGate "banks-count" s!"dA={dA} dB={dB} dC={dCn} eZ={eZ} e16={e16} slotLive={slotLive} staticHasSlot={staticHasSlot}"

/-- THE CACHE-INVARIANCE gate (the trip-count payoff). The kernel cache is keyed
    by md5(ir_text) (`OrcJitEngine`), so a knob that changed the IR text would
    force a full recompile. Two compiles of the SAME graph differing only in the
    `partials` DEFAULT (4 vs 12, same `partials_max`) must emit IDENTICAL LLVM
    IR — the count is a slot read; its default lives in plan metadata, never in
    the kernel text. A `partials_max` change must CHANGE the text: capacity IS
    topology (column sizes, the loop's static bound). Asserted on both the
    unsplit kernel and the typed-split audio kernel (the artifact the staged
    load actually caches). -/
def runBanksCountCache (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let src := fun (dflt cap : Nat) => "{\"nodes\":[" ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4,\"partials\":"
      ++ toString dflt ++ ",\"partials_max\":" ++ toString cap ++ "}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"res\"]}}],\"out\":\"out\"}"
  let irOf : Nat → Nat → Except String (String × String) := fun dflt cap => do
    let j ← (Lean.Json.parse (src dflt cap)).mapError (s!"json: {·}")
    let compiled ← Tropical.Playground.compilePlanPure arena resolved j
    let split ← Tropical.Ir.Stage0.hoistTyped compiled.plan compiled.stageBlocks
    pure (← Tropical.Ir.EmitLlvm.emitKernel compiled.plan,
      ← Tropical.Ir.EmitLlvm.emitKernel split.audio)
  match irOf 4 16, irOf 12 16, irOf 4 24 with
  | .error e, _, _ | _, .error e, _ | _, _, .error e =>
    failGate "banks-count-cache" s!"compile/emit: {firstLine e}"
  | .ok (u4, a4), .ok (u12, a12), .ok (u24, a24) =>
    let knobInvariant := u4 == u12 && a4 == a12
    let capMoves := u4 != u24 && a4 != a24
    IO.println s!"        same graph, partials default 4 vs 12 (cap 16) vs cap 24:"
    IO.println s!"        result   knob-invariant IR: unsplit={u4 == u12} audio={a4 == a12} ({u4.length}B) · capacity moves IR: unsplit={u4 != u24} audio={a4 != a24}"
    if knobInvariant && capMoves then
      passGate "banks-count-cache" "IR text is knob-invariant (md5 cache hit across counts); partials_max changes it (capacity is topology)"
    else
      failGate "banks-count-cache" s!"knobInvariant={knobInvariant} capMoves={capMoves}"

/-- Build the resonator → reverb → out patch graph as Json (the dir-landing path:
    a `reverb` node attaches a `ModalDir` unconditionally, so it routes through
    `modalBankSigDirTable`). `rt60` is a `.raw` slot, so the value sets its default. -/
private def reverbPatchJson (srcF srcDecay : Int) (rtM : Int) (rtE : Nat) : Lean.Json :=
  let node := fun (id kind : String) (params : List (String × Lean.Json))
                  (ins : List (String × Lean.Json)) =>
    Lean.Json.mkObj <|
      [("id", Lean.Json.str id), ("kind", Lean.Json.str kind),
       ("params", Lean.Json.mkObj params)] ++
      (if ins.isEmpty then [] else [("in", Lean.Json.mkObj ins)])
  Lean.Json.mkObj [
    ("nodes", Lean.Json.arr #[
      node "res" "resonator" [("freq", Lean.Json.num (jn srcF)), ("decay", Lean.Json.num (jn srcDecay))] [],
      node "rev" "reverb" [("rt60", Lean.Json.num (jn rtM rtE))]
        [("in", Lean.Json.arr #[Lean.Json.str "res"])],
      node "out" "out" [] [("in", Lean.Json.arr #[Lean.Json.str "rev"])]]),
    ("out", Lean.Json.str "out")]

/-- Build the resonator → filter → out patch graph as Json. -/
private def filterPatchJson (fc : Int) (resM : Int) (resE : Nat) (srcF srcDecay : Int) : Lean.Json :=
  let node := fun (id kind : String) (params : List (String × Lean.Json))
                  (ins : List (String × Lean.Json)) =>
    Lean.Json.mkObj <|
      [("id", Lean.Json.str id), ("kind", Lean.Json.str kind),
       ("params", Lean.Json.mkObj params)] ++
      (if ins.isEmpty then [] else [("in", Lean.Json.mkObj ins)])
  Lean.Json.mkObj [
    ("nodes", Lean.Json.arr #[
      node "res" "resonator" [("freq", Lean.Json.num (jn srcF)), ("decay", Lean.Json.num (jn srcDecay))] [],
      node "flt" "filter" [("cutoff", Lean.Json.num (jn fc)), ("resonance", Lean.Json.num (jn resM resE))]
        [("in", Lean.Json.arr #[Lean.Json.str "res"])],
      node "out" "out" [] [("in", Lean.Json.arr #[Lean.Json.str "flt"])]]),
    ("out", Lean.Json.str "out")]

private def renderFilterPatch (arena : Arena) (resolved : Array (String × ProgramIdx))
    (j : Lean.Json) (n : Nat) : IO (Except String (Array Float)) := do
  match Tropical.Playground.compilePlanPure arena resolved j with
  | .error e => pure (.error s!"compile: {firstLine e}")
  | .ok compiled =>
    match ← renderPlanSamples compiled.plan n with
    | .error e => pure (.error s!"render: {firstLine e}")
    | .ok samples => pure (.ok samples)

private def tailEnergy (xs : Array Float) (lo : Nat) : Float := Id.run do
  let mut acc : Float := 0.0
  for i in [lo:xs.size] do
    acc := acc + xs[i]! * xs[i]!
  pure acc

/-- THE MODAL FILTER gate (the VCFQ). A `filter` node is a `modalReverb`
    whose room is one EXACT conjugate pole pair (`filterPair`), so three
    behaviors must hold, all through the ordinary graph surface:
    (A) LOWPASS: the same struck resonator through cutoff=4000 vs cutoff=60
        loses most of its energy (the composition's forced modes carry
        `a·H(λ)`, and |H| is small far above fc).
    (B) THE PING: with a fast-dying excitation and resonance at the top of
        the knob (Q ≈ 44), the tail is the FILTER ringing at ω_d ≈ 2π·fc —
        zero-crossing rate within 3% of fc. This is the Serge character the
        node exists for: high resonance IS a struck resonator.
    (C) LIVE: cutoff is a glided live knob — writing its glide endpoints
        (#v0/#v1) diverges the output vs an untouched twin, THROUGH the
        composition (no relower, no dead knob). -/
def runModalFilter (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  -- (A) lowpass attenuation
  match ← renderFilterPatch arena resolved (filterPatchJson 4000 3 1 220 4) 4096,
        ← renderFilterPatch arena resolved (filterPatchJson 60 3 1 220 4) 4096 with
  | .error e, _ | _, .error e => failGate "modal-filter" s!"(A) {e}"
  | .ok openS, .ok closedS =>
    let eOpen := tailEnergy openS 200
    let eClosed := tailEnergy closedS 200
    -- (B) the ping: fast-dying strike at 1800 Hz, filter fc=500 res=1 (Q≈44);
    -- by half the window the source is gone and the tail is the filter's ring.
    match ← renderFilterPatch arena resolved (filterPatchJson 500 1 0 1800 60) 8192 with
    | .error e => failGate "modal-filter" s!"(B) {e}"
    | .ok ping =>
      let mut crossings := 0
      for i in [4097:8192] do
        if (ping[i-1]! < 0.0 && ping[i]! >= 0.0) || (ping[i-1]! >= 0.0 && ping[i]! < 0.0) then
          crossings := crossings + 1
      let tailSec := (8192.0 - 4097.0) / 44100.0
      let ringHz := crossings.toFloat / 2.0 / tailSec
      let eTail := tailEnergy ping 4097
      -- (C) live cutoff on one of two twin runtimes
      match Tropical.Playground.compilePlanPure arena resolved (filterPatchJson 800 5 1 220 4) with
      | .error e => failGate "modal-filter" s!"(C) compile: {firstLine e}"
      | .ok compiled =>
      let plan := compiled.plan
      let stageBlocks := compiled.stageBlocks
      match plan.toWire, Tropical.Ir.EmitLlvm.emitKernel plan with
      | .ok _, .ok _ =>
        let rt ← Tropical.Ffi.Runtime.new 2048
        Tropical.StagedLoad.loadTyped rt plan stageBlocks
        let rt2 ← Tropical.Ffi.Runtime.new 2048
        Tropical.StagedLoad.loadTyped rt2 plan stageBlocks
        let v0? ← rt.slotIndex? "param:flt.cutoff#v0"
        let v1? ← rt.slotIndex? "param:flt.cutoff#v1"
        rt.process; rt2.process
        if let some v0 := v0? then rt.setSlot v0 60.0
        if let some v1 := v1? then rt.setSlot v1 60.0
        rt.process; rt2.process
        let a := decodeF64LE (← rt.outputBytes)
        let b := decodeF64LE (← rt2.outputBytes)
        let mut dE : Float := 0.0
        let mut e0 : Float := 0.0
        for i in [0:min a.size b.size] do
          let d := a[i]! - b[i]!
          dE := dE + d * d
          e0 := e0 + b[i]! * b[i]!
        let slotsPresent := v0?.isSome && v1?.isSome
        let knobsLive := slotsPresent && dE > 1e-9 * e0 && e0 > 1e-9
        IO.println s!"        filter = residue-composed conjugate pole pair (H exact); resonator → filter → out:"
        IO.println s!"        (A) E[cutoff 4kHz]={eOpen} vs E[60Hz]={eClosed} (ratio {eOpen/(eClosed+1e-300)})"
        IO.println s!"        (B) Q≈44 ping: tail rings at {ringHz} Hz (fc=500, want ±3%), E[tail]={eTail}"
        IO.println s!"        (C) cutoff glide slots present={slotsPresent} · ΔE/E after move={dE/(e0+1e-300)}"
        if eOpen > 20.0 * eClosed && eClosed > 0.0 &&
           ringHz > 485.0 && ringHz < 515.0 && eTail > 1e-8 && knobsLive then
          passGate "modal-filter" s!"lowpass attenuates ({eOpen/(eClosed+1e-300)}x), Q≈44 pings at {ringHz} Hz, cutoff live through the composition"
        else
          failGate "modal-filter" s!"eOpen={eOpen} eClosed={eClosed} ringHz={ringHz} (want 485-515) eTail={eTail} live={knobsLive}"
      | .error e, _ | _, .error e =>
        failGate "modal-filter" s!"(C) emit: {firstLine e}"

/-- Oracle-free spike count: samples deviating from the mean of their neighbours by
    more than 25% of peak. The rail probe's discriminator (design/prod-rail-probe):
    a modal signal is a sum of decaying sinusoids ≤ 4.8 kHz on a 44.1 kHz grid, so
    it is smooth by construction, while the i64 wrap's signature is ISOLATED
    single-sample glitches. Reads 0 for every benign config INCLUDING loud ones
    (peak 91), and jumps to hundreds exactly at the rail. Returns (peak, spikes). -/
private def spikeStats (s : Array Float) : Float × Nat := Id.run do
  let mut peak : Float := 0.0
  for x in s do
    if x.isFinite && x.abs > peak then peak := x.abs
  let mut spikes : Nat := 0
  if s.size ≥ 3 then
    for i in [1:s.size-1] do
      let d := (s[i]! - 0.5 * (s[i-1]! + s[i+1]!)).abs
      if d > 0.25 * peak && peak > 1e-9 then spikes := spikes + 1
  pure (peak, spikes)

/-- THE MODAL RAIL WITNESS (option E, the production-path red witness). The exact
    gesture the rail incident named: `resonator(800,4) ⋙ filter(cutoff 800, res) ⋙
    out`, compiled through `compilePlanPure` on the PRODUCTION path (master clock,
    anchor 0, master gain 3.7), the SAME path the shipped GUI drives. The filter's
    amplitudes are LIVE param slots (`resonance` is a `pref` slot), so this
    exercises option E's DYNAMIC (kernel-time) per-bank exponent — the case the
    static path cannot reach, and the one the rail actually lives on.

    Two arms on ONE knob, both MEASURED (design/prod-rail-probe.local.lean.txt):
    - GREEN CONTROL res 0.91 — peak ≈ 91, spikes 0. Loud-but-correct: it proves
      the discriminator distinguishes *loud* from *broken* and is not trivially
      satisfied by silence (a peak floor guards that).
    - RED ARM res 0.95 — pre-fix peak 234 with 211 spikes (the i64 wrap: `|A| ≈ Q =
      0.55·80^0.95 ≈ 35` collected past the rail of 32 wraps to a ±64 full-scale
      burst); post-fix `k = ⌊log₂ 35⌋−4 = 1` lands at Q3.27 and the ring renders
      SMOOTH — peak ≈ 130 (the true driven-resonator output, legitimately louder
      than the control) and spikes 0.

    Discriminator is spikes, NOT peak: the correct high-Q output is LOUDER than the
    control, so an amplitude bound would false-fail it. MUTATION-VERIFIED: revert
    any plain-site landing to `lit 268435456`/`lit 28` and the res-0.95 arm returns
    to hundreds of spikes → RED. -/
def runModalRail (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  match ← renderFilterPatch arena resolved (filterPatchJson 800 91 2 800 4) 4096,
        ← renderFilterPatch arena resolved (filterPatchJson 800 95 2 800 4) 4096 with
  | .error e, _ | _, .error e => failGate "modal-rail" s!"render: {e}"
  | .ok green, .ok red =>
    let (pkG, spG) := spikeStats green
    let (pkR, spR) := spikeStats red
    let finiteG := green.all (·.isFinite)
    let finiteR := red.all (·.isFinite)
    IO.println s!"        resonator(800,4) ⋙ filter(800, res) ⋙ out, production path (live amps ⇒ dynamic k):"
    IO.println s!"        res 0.91 (green control): peak={pkG} spikes={spG} finite={finiteG}"
    IO.println s!"        res 0.95 (red arm)      : peak={pkR} spikes={spR} finite={finiteR}"
    if spG == 0 && spR == 0 && finiteG && finiteR && pkG > 50.0 then
      passGate "modal-rail" s!"the top of the resonance knob renders SMOOTH through the fix (res 0.91 & 0.95 both spike-free; control peak {pkG} proves the discriminator sees loud≠broken) — the i64 landing rail is gone on the production path"
    else
      failGate "modal-rail" s!"spikes green={spG} red={spR} (want 0/0), finite {finiteG}/{finiteR}, control peak {pkG} (want >50) — the datapath wraps or the render is trivial"

/-- THE MODAL DIR-RAIL WITNESS (option E, the reverb/dir-path red witness). The
    `modalBankSigDirTable` landing had ZERO rail coverage, yet every `reverb` node
    in the vocabulary routes through it (`reverb` attaches a `ModalDir`
    unconditionally, unlike `filter`), so the same i64 wrap lived there unwitnessed.
    At the default dir knob (0) the dir table reduces to its forward accumulator,
    whose per-mode Q4.28 landing is the SAME overflow site as the plain table.

    The gesture: `resonator(60,4) ⋙ reverb(rt60) ⋙ out`. The reverb room mode 0
    sits at EXACTLY 60 Hz and resonator partial 1 at 1·60 Hz, so the poles coincide
    in frequency and |Δ| collapses to |Δσ|; sweeping rt60 through the punctured
    neighbourhood of the coincidence (σ_room = 6.91/rt60 crossing σ_res = decay·1.4
    = 5.6 at rt60 ≈ 1.234) drives the collected |A| past the rail.

    MEASURED on the production dir path (the derivation lab, fix forced off):
    the red set is ERRATIC — the wrap needs differing per-mode wrap counts, so it
    is necessary-not-sufficient and fires only at some rt60 in the neighbourhood
    (rt60 1.230/1.236/1.238 → peak ≈237 with 15/3/21 spikes; the exact-coincidence
    1.232/1.234 stay benign — the point itself yields a deg-1 mode on the unrolled
    path, and benign wrap-parity pockets sit between). So the red arm is a
    MEASURED rt60 (1.238, the strongest), never a derived threshold.

    - GREEN CONTROL rt60 = 2.0 — peak ≈0.30, spikes 0 (far from coincidence).
    - RED ARM rt60 = 1.238 — pre-fix peak 237 / 21 spikes; post-fix peak ≈0.28 / 0
      spikes. Near coincidence the CORRECT reverb output is quiet (the ±c/Δ
      residues cancel in the sum), so unlike the filter witness a peak bound is a
      safe secondary catch here (the wrap's 237 is the anomaly, not the music).

    MUTATION-VERIFIED: revert the dir landing to `lit 268435456`/`lit 28` and the
    rt60-1.238 arm returns to 21 spikes / peak 237 → RED. -/
def runModalRailDir (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  match ← renderFilterPatch arena resolved (reverbPatchJson 60 4 2 0) 4096,
        ← renderFilterPatch arena resolved (reverbPatchJson 60 4 1238 3) 4096 with
  | .error e, _ | _, .error e => failGate "modal-rail-dir" s!"render: {e}"
  | .ok green, .ok red =>
    let (pkG, spG) := spikeStats green
    let (pkR, spR) := spikeStats red
    let finiteG := green.all (·.isFinite)
    let finiteR := red.all (·.isFinite)
    IO.println s!"        resonator(60,4) ⋙ reverb(rt60) ⋙ out, production dir path (60 Hz pole coincidence):"
    IO.println s!"        rt60 2.0 (green control)  : peak={pkG} spikes={spG} finite={finiteG}"
    IO.println s!"        rt60 1.238 (red arm)      : peak={pkR} spikes={spR} finite={finiteR}"
    if spG == 0 && spR == 0 && finiteG && finiteR && pkG > 0.05 && pkR < 10.0 then
      passGate "modal-rail-dir" s!"the reverb dir landing survives the 60 Hz pole coincidence (rt60 1.238 spike-free, peak {pkR} — the pre-fix ±237 wrap is gone; control rt60 2.0 renders) — the i64 rail is fixed on the dir path too"
    else
      failGate "modal-rail-dir" s!"spikes green={spG} red={spR} (want 0/0), finite {finiteG}/{finiteR}, control peak {pkG} (>0.05), red peak {pkR} (want <10 — the wrap is ≈237)"

open Tropical.EmitArrow in
/-- THE OPTION-E STRUCTURAL-IDENTITY gate — the anchor that makes "when max|A| < 32,
    k = 0 and NOTHING moves" TESTABLE (no frozen plan/IR hash exists on the
    EmitArrow modal-island path, so nothing else would catch a k=0 drift). Two
    ALL-LITERAL-amp banks, so `bankLandExp` takes the STATIC path and `k` is a
    compile-time `Nat` decided here — no dynamic machinery, no live slots:

    - SMALL amps (L1 max 0.2 < 32) ⇒ `k = 0` ⇒ the landing is `·2²⁸` / `>>28`
      VERBATIM: the emitted IR carries the pre-fix double `0x41B0000000000000`
      (= 2²⁸) at the weight multiply — byte-identical structure, so the JIT reuses
      the pre-fix kernel-cache object (the cache keys on md5 of the IR text).
    - LARGE amps (L1 max 100) ⇒ `k = ⌊log₂ 100⌋ − 4 = 2` ⇒ the landing rescales to
      `·2²⁶` (`0x4190000000000000`) and the 2²⁸ constant is ABSENT: option E fires
      statically, exactly as the dynamic path does at knob rate.

    Pins both directions at once. MUTATION-VERIFIED: force `bankLandExp` to
    `.static 0` and the large bank keeps the 2²⁸ landing (option E stops firing) →
    the `k=2`/`2²⁶` assertions go RED. -/
def runModalRailIdentity (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let mkModes := fun (amp : Sig) => (Array.range 6).map fun i =>
    ModalMode.hz (lit (Int.ofNat (220 + 40 * i))) (lit 30 1) amp
  let small := mkModes (lit 2 1)      -- amp 0.2 ⇒ maxAbs 0.2 < 32 ⇒ k = 0
  let large := mkModes (lit 100)      -- amp 100  ⇒ maxAbs 100    ⇒ k = 2
  -- a deg-1 bank with small σ: |A|=10 < 32 (amp-only would give k=0), but env₂ =
  -- d·e^{−σd} peaks at (1/(σe)) ≈ 3.68, so the LANDED sup ≈ 36.8 > 32 ⇒ the
  -- envelope-aware bound bumps k to 1. Proves the deg>0 lift is not amp-only.
  let degBank := (Array.range 3).map fun i =>
    ({ sigma := litF 0.1, omega := mul twoPiE (lit (Int.ofNat (300 + 40 * i))),
       cre := litF 10.0, cim := lit 0, deg := 1 } : ModalMode)
  let hasSub := fun (s sub : String) => (s.splitOn sub).length != 1
  let land2p28 := "0x41b0000000000000"   -- 2²⁸, the verbatim Q4.28 landing (lowercase hex)
  let land2p26 := "0x4190000000000000"   -- 2²⁶, the k=2 landing
  let kSmall := match bankLandExp small with | .static k => some k | .dynamic _ => none
  let kLarge := match bankLandExp large with | .static k => some k | .dynamic _ => none
  let kDeg   := match bankLandExp degBank with | .static k => some k | .dynamic _ => none
  match buildAndFinish (.ok (buildModalBankTable "id_small" small (lit 200) arena)),
        buildAndFinish (.ok (buildModalBankTable "id_large" large (lit 200) arena)) with
  | .ok pSmall, .ok pLarge =>
    match Tropical.Ir.EmitLlvm.emitKernel pSmall, Tropical.Ir.EmitLlvm.emitKernel pLarge with
    | .ok irSmall, .ok irLarge =>
      let smallVerbatim := hasSub irSmall land2p28    -- k=0 keeps 2²⁸
      let largeRescaled := hasSub irLarge land2p26 && !(hasSub irLarge land2p28)  -- k=2 ⇒ 2²⁶, no 2²⁸
      let degLifts := kDeg == some 1    -- env₂-aware bound bumps a small-|A| deg-1 bank
      IO.println s!"        static banks: small amp 0.2 ⇒ k={kSmall} (want 0), large amp 100 ⇒ k={kLarge} (want 2), deg-1 σ=0.1 amp=10 ⇒ k={kDeg} (want 1, env-lifted):"
      IO.println s!"        small IR keeps 2²⁸ landing={smallVerbatim} · large IR rescales to 2²⁶ & drops 2²⁸={largeRescaled} · deg-1 env₂ lift={degLifts}"
      if kSmall == some 0 && kLarge == some 2 && degLifts && smallVerbatim && largeRescaled then
        passGate "modal-rail-identity" "k=0 emits the Q4.28 landing VERBATIM (byte-identical, reused kernel-cache); a loud static bank rescales to 2²⁶ (option E fires); a small-|A| deg-1 bank still lifts k via its env₂ peak (deg>0 is not amp-only) — pinned every way"
      else
        failGate "modal-rail-identity" s!"kSmall={kSmall} (want 0) kLarge={kLarge} (want 2) kDeg={kDeg} (want 1) smallVerbatim={smallVerbatim} largeRescaled={largeRescaled}"
    | .error e, _ | _, .error e => failGate "modal-rail-identity" s!"emit: {firstLine e}"
  | .error e, _ | _, .error e => failGate "modal-rail-identity" s!"build: {firstLine e}"

open Tropical.EmitArrow in
/-- THE MODAL ADDRESS gate. A resonator's `addr` inlet: a patched CF signal BECOMES
    the bank's absolute time-coordinate (`modalAddrWarp`), so the causal gate — the
    strike — tracks the SIGNAL, not the master clock. Three things, end to end:
    (1) SCALING — address = time (offset 0) reproduces the un-addressed bank (only a
        float round-trip apart), proving `s`-seconds maps to exactly the right clock;
    (2) TRIGGER — address = time − 0.01 s stays silent until sample 0.01·SR, then
        rings, proving the signal RELOCATES the strike (scrub-to-trigger, not a
        re-strike); (3) DECODE — a JSON `osc → res.addr, res → out` compiles through
        the real `compilePlanPure` to codegen (the graph surface wires the address). -/
def runModalAddr (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let modes : Array ModalMode := #[
    ModalMode.hz (lit 220) (lit 30 1) (lit 6 1),
    ModalMode.hz (lit 440) (lit 45 1) (lit 3 1)]
  let n := 2048
  let onsetSec : Float := 0.01
  let onset := (onsetSec * 44100.0).toUInt64.toNat
  -- (3) decode: a real patched-signal address (an LFO osc) through the JSON surface.
  let src := "{\"nodes\":[" ++
    "{\"id\":\"lfo\",\"kind\":\"source\",\"params\":{\"freq\":40,\"morph\":0}}," ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4},\"in\":{\"addr\":[\"lfo\"]}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"res\"]}}],\"out\":\"out\"}"
  let decodeOk : Bool := match Lean.Json.parse src with
    | .error _ => false
    | .ok j => match Tropical.Playground.compilePlanPure arena resolved j with
      | .error _ => false
      | .ok compiled => (Tropical.Ir.EmitLlvm.emitKernel compiled.plan).toOption.isSome
  match buildAndFinish (.ok (buildModalBankArrow "ma_ref" modes (lit 0) arena)),
        buildAndFinish (.ok (buildModalAddrRamp "ma_id" modes (lit 0) 0.0 arena)),
        buildAndFinish (.ok (buildModalAddrRamp "ma_off" modes (lit 0) onsetSec arena)) with
  | .ok refp, .ok idp, .ok offp =>
    match ← renderPlanSamples refp n, ← renderPlanSamples idp n, ← renderPlanSamples offp n with
    | .ok ref, .ok ida, .ok off =>
      let mut maxErr : Float := 0.0
      for i in [0:n] do
        let e := (ida[i]! - ref[i]!).abs
        if e > maxErr then maxErr := e
      let mut preMax : Float := 0.0
      for i in [0:onset] do
        let a := off[i]!.abs
        if a > preMax then preMax := a
      let mut postPeak : Float := 0.0
      for i in [onset+50:n] do
        let a := off[i]!.abs
        if a > postPeak then postPeak := a
      IO.println s!"        addressed resonator (a patched CF signal AS the time coordinate):"
      IO.println s!"        result   identity-addr max|Δ| vs un-addressed={maxErr} · offset-addr pre-onset|max|={preMax} post-onset peak={postPeak} · graph decode ok={decodeOk}"
      if maxErr < 1e-4 && preMax == 0.0 && postPeak > 1e-6 && decodeOk then
        passGate "modal-addr" "a patched signal drives the bank's time: address=time ≡ un-addressed; offset relocates the strike; graph decode compiles"
      else
        failGate "modal-addr" s!"maxErr={maxErr} preMax={preMax} postPeak={postPeak} decodeOk={decodeOk}"
    | .error e, _, _ | _, .error e, _ | _, _, .error e => failGate "modal-addr" s!"render: {firstLine e}"
  | _, _, _ => failGate "modal-addr" "build"

/-- THE GAUGE-STAGE gate (§5).  Gauge measures the complete current modal
    universe at its authored position.  With glided poles that norm is therefore
    sample-stage: inserting gauge must add `FloatExponent` work to the audio
    kernel instead of silently settling the controls or declining to identity.
    This gate makes that semantic choice explicit; the trust ledger separately
    keeps the bilateral live-gauge cost/backend envelope open. -/
def runGaugeStage (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let mk := fun (withGauge : Bool) =>
    "{\"nodes\":[" ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4}}," ++
    "{\"id\":\"flt\",\"kind\":\"filter\",\"params\":{\"cutoff\":800,\"resonance\":0.5},\"in\":{\"in\":[\"res\"]}}," ++
    (if withGauge then "{\"id\":\"gau\",\"kind\":\"gauge\",\"params\":{\"g\":1},\"in\":{\"in\":[\"flt\"]}}," else "") ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"" ++ (if withGauge then "gau" else "flt") ++ "\"]}}],\"out\":\"out\"}"
  let feOf : Bool → IO (Option (Nat × Nat)) := fun withGauge => do
    match Lean.Json.parse (mk withGauge) with
    | .error e => IO.println s!"        gauge-stage json: {e}"; pure none
    | .ok j => match Tropical.Playground.compilePlanPure arena resolved j with
      | .error e => IO.println s!"        gauge-stage compile: {firstLine e}"; pure none
      | .ok compiled => match Tropical.Ir.Stage0.hoistTyped compiled.plan compiled.stageBlocks with
        | .error e => IO.println s!"        gauge-stage split: {firstLine e}"; pure none
        | .ok split =>
          let a := planFloatExponents split.audio
          let c := match split.coeff? with | some k => planFloatExponents k | none => 0
          pure (some (a, c))
  let some (aBare, cBare) ← feOf false | return (← failGate "gauge-stage" "bare compile")
  let some (aGauge, cGauge) ← feOf true | return (← failGate "gauge-stage" "gauge compile")
  IO.println s!"        resonator ⋙ filter(glided) ⋙ [gauge] ⋙ out — FloatExponent (audio, coeff):"
  IO.println s!"        result   without gauge ({aBare}, {cBare}) · with gauge ({aGauge}, {cGauge})"
  if aGauge > aBare then
    passGate "gauge-stage" s!"current-universe gauge remains live: its norm adds {aGauge - aBare} FloatExponent operations to the audio kernel (coeff {cBare}→{cGauge}); backend qualification remains an open trust obligation"
  else
    failGate "gauge-stage" s!"audio {aBare}→{aGauge} (must grow for a live current-universe norm), coeff {cBare}→{cGauge}"
