import Std.Data.HashMap
import Tropical.Ir.Core
import Tropical.Plan

/-!
# Stage0 — plan-level binding-time split (`FlatPlan → audio + coefficient`)

Binding time is a SYNTACTIC property of the instruction stream, not an
ontology: an instruction whose operands never mention τ (`source(tick)`)
computes the same value at every sample, so it belongs in a one-sample
**coefficient kernel** run on the control thread at param-write time, not
in the per-sample audio kernel. This is NOT a control rate — no signal is
demoted and no second signal kind exists in the language. Patch a
τ-dependent expression into the same port tomorrow and it stays in the
audio kernel with zero language change.

The split is a forward dataflow pass over the plan's instructions in
**emit order** (the exact order `EmitLlvm.emitKernelBlock` walks:
preamble → per-child {pre_input, child} → body, then the next
instance function), so the flow-sensitive temp/slot tracking matches the
emitter's SSA temp model instruction for instruction:

Three stages, ordered `fold < s0 < s1`; an instruction's stage is the
join of its operands':

- **fold** — const/rate-only. These chains STAY in the audio plan:
  EmitMsl constant-folds them in f64 at emit (exact literals in the MSL
  source) and LLVM folds them on the JIT, so hoisting them would only
  DEMOTE their GPU precision to an f32 slot crossing (measured: the
  fixed-gain reverb gates dropped from >125 to ~103 dB SNR when consts
  were hoisted wholesale). A fold-stage def consumed by a hoisted
  instruction is *duplicated* into the coefficient stream — it is a pure
  constant computation, so both kernels computing it is legal and
  f64-deterministic.
- **s0** — τ-independent but slot-derived (live params): the hoist
  target. Leaves: a `slot` never written in the plan (params, defaults —
  constant between control writes; never `fold`, because `set_slot` can
  move it). A `reg` with no reaching def (the emitter's zero fallback)
  is `fold`.
- **s1** — per-sample: `source(tick)`, `input`, `param` (per-program FFI
  path — no re-run hook), any array operand (arrays are conservatively
  per-sample in v1), a `slot` whose reaching write is stage-1 or not yet
  seen (previous-sample value).

An instruction hoists iff its stage is exactly `s0` and its dst is
scalar (`temp`) or a module slot it is the ONLY in-plan writer of
(hoisting one of several writers would reorder the write sequence).
The TYPED placement (`placementFromStages`) additionally moves a whole
`ReduceBegin`/`ReduceEnd` region when its entire body is s0 — see the
two-layer note there; this flow pass never hoists regions.

Stage-0 instructions move verbatim — same ops, same relative order, no
reassociation — into the coefficient stream. A stage-0 temp read by a
surviving stage-1 instruction is a **boundary** value: the coefficient
kernel writes it to a fresh `coef:<n>` module slot and the consumer's
`reg` operand is rewritten to a `slot` read carrying the *producer's*
result type (the emitter's `tempVals` returned the producer-typed value,
ignoring the operand's declared type — the slot read must reproduce
that). Boundary values cross as f64 slots, exactly the coercion the plan
already uses at instance output ports (`emitWriteSlots`): bit-exact for
floats (raw double store/load, NaN included) and bools, exact for ints
below 2^53 — the same house semantics as every existing port crossing.

The coefficient plan is an ordinary `FlatPlan` (one synthetic instance
function, no sinks, same sources/registerCount) emitted through the SAME
`EmitLlvm.emitKernel` — same div-by-zero guards, same op lowering — and
run by the engine with `buffer_length = 1`. Never evaluate stage-0 in
Lean float math and hope it matches. Both kernels keep temps as SSA
(never store `%temps`), so they can share the runtime's zero-initialized
scratch without racing.

If nothing hoists, the plan is returned unchanged (`coeff? = none`) —
the pass is the identity on τ-only programs.
-/

namespace Tropical.Ir.Stage0

open Tropical.Plan
open Std (HashMap)

/-- The split result: the audio plan (stage-0 instructions removed, its
    slot table extended with the `coef:<n>` slots) and, when anything
    hoisted, the coefficient plan sharing that slot table. -/
structure Split where
  audio : FlatPlan
  coeff? : Option FlatPlan
deriving Inhabited

-- ─────────────────────────────────────────────────────────────
-- Block skeleton — the single source of emit-order truth
-- ─────────────────────────────────────────────────────────────

/-- Instruction blocks of one instance function in emit order (mirrors
    `EmitLlvm.emitKernelBlock`): preamble, then per child its pre_input
    block followed recursively by the child's own blocks, then the body.
    Public: the stage differential linearizes plans the same way. -/
def collectBlocks (f : InstanceFunction) : Array (Array NInstr) := Id.run do
  let mut out := #[f.preambleInstructions]
  for _h : child in f.children do
    out := out.push child.preInputInstructions
    out := out ++ collectBlocks child
  return out.push f.instructions
termination_by sizeOf f
decreasing_by exact Tropical.Plan.InstanceFunction.sizeOf_lt_of_mem_children _h

/-- Every instruction block in a plan, in emitter order.  Keeping this walk
    public gives typed staging and its proofs one canonical block skeleton. -/
def collectPlanBlocks (plan : FlatPlan) : Array (Array NInstr) := Id.run do
  let mut blocks : Array (Array NInstr) := #[]
  for fn in plan.instanceFunctions do
    blocks := blocks ++ collectBlocks fn
  return blocks

/-- Typed classifications align block-for-block with emitter order.  A flat
    total alone is insufficient: adjacent block-length mistakes can cancel. -/
def typedStagesAligned (blocks : Array (Array NInstr))
    (stageBlocks : Array (Array (Option Stage))) : Bool :=
  blocks.map Array.size == stageBlocks.map Array.size

/-- No instruction is selected for coefficient-time execution.  Missing
    classifications and explicit `s1` are both conservative audio placement. -/
def noTypedSelection (stageBlocks : Array (Array (Option Stage))) : Bool :=
  stageBlocks.all fun block => block.all fun stage =>
    match stage with
    | some .fold | some .s0 => false
    | some .s1 | none => true

/-- Reassemble an instance function from rewritten blocks, consuming them
    in the same order `collectBlocks` produced. Returns the rebuilt
    function and the next unconsumed block index. -/
private def rebuildFn (f : InstanceFunction) (blocks : Array (Array NInstr))
    (start : Nat) : InstanceFunction × Nat := Id.run do
  let preamble := blocks[start]!
  let mut i := start + 1
  let mut children : Array InstanceFunction := #[]
  for _h : child in f.children do
    let preInput := blocks[i]!
    let (child', i') := rebuildFn child blocks (i + 1)
    children := children.push (child'.withPreInput preInput)
    i := i'
  let body := blocks[i]!
  return (.mk f.name f.instanceName preamble body f.preInputInstructions
    f.registerOffset f.arraySlotOffset f.registerCount children, i + 1)
termination_by sizeOf f
decreasing_by exact Tropical.Plan.InstanceFunction.sizeOf_lt_of_mem_children _h

-- ─────────────────────────────────────────────────────────────
-- Analysis — one forward pass in emit order
-- ─────────────────────────────────────────────────────────────

-- `Stage` (fold < s0 < s1) and `Stage.join` are the shared binding-time
-- type from `Tropical.Ir.ExprArena` — the same lattice the intern-time
-- attribute uses, so this pass's flow-derived classification and the
-- typed signature resolution are directly comparable.

private structure Analysis where
  /-- Per linear instruction index: the value's binding-time stage. -/
  stages : Array Stage
  /-- Per linear instruction index: moves to the coefficient stream. -/
  hoisted : Array Bool
  /-- Fold-stage def sites the coefficient stream needs a duplicate of
      (transitive const support of hoisted instructions). -/
  needFold : HashMap Nat Unit
  /-- Hoisted temp-def sites read by a surviving instruction, ascending
      by def index: (defIdx, temp, producer result type). Position in
      this array is the coefficient slot ordinal. -/
  boundary : Array (Nat × Nat × ScalarType)
  /-- Surviving-instruction arg rewrites: instr index → (argPos, defIdx). -/
  rewrites : HashMap Nat (Array (Nat × Nat))

/-- The stage of one operand given the flow state at this point.
    `tempDef`/`stages` mirror the emitter's `tempVals`: the reaching def
    in linear order (a read before any def is the zero fallback —
    constant, `fold`, even if the temp is defined later). -/
private def operandStage (sources : Array SourceKind)
    (slotWritten : HashMap Nat Nat) (stages : Array Stage)
    (tempDef : HashMap Nat Nat) (slotStage : HashMap Nat Stage) : NOperand → Stage
  | .const _ _ => .fold
  | .reg t _ =>
    match tempDef.get? t with
    | some d => stages[d]!
    | none => .fold
  | .source i _ =>
    match sources[i]? with
    | some .tick => .s1
    | some .rate => .fold
    | some .tilePhase => .s1
    | some .tileTick => .s1
    | none => .fold           -- out-of-range source resolves to 0.0
  | .slot i _ =>
    match slotStage.get? i with
    -- The reaching write decides AVAILABILITY, not value constancy: a
    -- hoisted write precedes this read in the coefficient stream (s0 —
    -- never fold, set_slot can move any slot); a surviving write of ANY
    -- stage runs per-sample in the audio kernel, so a hoisted reader
    -- would see stale defaults at coefficient-run time (s1).
    | some s => s
    -- No write seen yet this pass: params/defaults are constant between
    -- control writes (stage-0); a slot written later in the stream holds
    -- the previous sample's value here — conservatively stage-1.
    | none => if slotWritten.contains i then .s1 else .s0
  | .input _ _ => .s1
  | .param _ _ => .s1         -- per-program FFI path: no re-run hook
  | .arrayReg _ => .s1        -- arrays are conservatively per-sample (v1)
  | .sessionArrayReg _ => .s1
  | .loopIdx _ => .s1         -- per-iteration inside a reduce region (any id)

private def analyze (plan : FlatPlan) (blocks : Array (Array NInstr)) : Analysis := Id.run do
  -- Prepass: in-plan write counts per module slot. A stage-0 slot write
  -- hoists only when it is the slot's sole writer.
  let mut slotWritten : HashMap Nat Nat := {}
  for block in blocks do
    for instr in block do
      if let .moduleSlot m := instr.dst then
        slotWritten := slotWritten.insert m ((slotWritten.getD m 0) + 1)

  let mut hoisted : Array Bool := #[]
  let mut stages : Array Stage := #[]
  -- Temp def metadata per linear index (temp, resultType); none for
  -- non-temp dsts.
  let mut defMeta : Array (Option (Nat × ScalarType)) := #[]
  -- Per linear index: the reaching-def indices of the instruction's reg
  -- args (for the fold-support closure below).
  let mut regDefs : Array (Array Nat) := #[]
  let mut boundarySet : HashMap Nat Unit := {}
  let mut rewrites : HashMap Nat (Array (Nat × Nat)) := {}
  let mut tempDef : HashMap Nat Nat := {}
  let mut slotStage : HashMap Nat Stage := {}
  -- Routed reductions are placement-atomic: neither delimiter nor any mapped
  -- body instruction may peel into the coefficient stream.  Keep the depth
  -- across block boundaries because `blocks` is the emitter's linear order.
  let mut routedDepth : Nat := 0
  let mut idx := 0
  for block in blocks do
    for instr in block do
      let mut stage : Stage := .fold
      let mut rds : Array Nat := #[]
      for arg in instr.args do
        stage := stage.join
          (operandStage plan.sources slotWritten stages tempDef slotStage arg)
        if let .reg t _ := arg then
          if let some d := tempDef.get? t then
            rds := rds.push d
      let inRouted := routedDepth > 0 || instr.tag == "RoutedSumBegin"
      let (hoist, vStage) :=
        if inRouted then (false, Stage.s1)
        -- Reduce delimiters are per-sample loop structure — never moved.
        -- (The FLOW classifier is the fallback splitter for plans parsed
        -- from JSON, where the arena is gone; it stays conservative and
        -- never hoists regions. Whole-region moves are a TYPED-placement
        -- decision — see `tryRegion` in `placementFromStages`.)
        else if instr.tag == "ReduceBegin" || instr.tag == "ReduceEnd" then (false, Stage.s1)
        else match instr.dst with
        | .temp _ => (stage == .s0, stage)
        -- Strictly s0, like temps: a fold-valued slot write must STAY —
        -- EmitMsl's f64 emit-time folding propagates through in-kernel
        -- slot write→read, so hoisting it would demote an exact f64
        -- literal on the GPU to an f32 host-slot crossing (measured:
        -- pure-sine dropped to the f32 floor when fold writes hoisted).
        -- Its readers stay behind too, via the availability rule below.
        | .moduleSlot m => (stage == .s0 && slotWritten.getD m 0 == 1, stage)
        | .array _ => (false, .s1)
        | .sessionArray _ => (false, .s1)
      if !hoist then
        -- Reads of hoisted defs must come through coefficient slots.
        let mut rw : Array (Nat × Nat) := #[]
        for pos in [0:instr.args.size] do
          if let .reg t _ := instr.args[pos]! then
            if let some d := tempDef.get? t then
              if hoisted[d]! then
                rw := rw.push (pos, d)
                boundarySet := boundarySet.insert d ()
        if !rw.isEmpty then
          rewrites := rewrites.insert idx rw
      hoisted := hoisted.push hoist
      stages := stages.push vStage
      regDefs := regDefs.push rds
      match instr.dst with
      | .temp t =>
        defMeta := defMeta.push (some (t, instr.resultType))
        tempDef := tempDef.insert t idx
      | .moduleSlot m =>
        defMeta := defMeta.push none
        -- Availability, not value stage: hoisted writes are visible to
        -- later coefficient-stream reads (s0); surviving writes are not
        -- until the audio kernel runs (s1).
        slotStage := slotStage.insert m (if hoist then .s0 else .s1)
      | _ => defMeta := defMeta.push none
      if instr.tag == "RoutedSumBegin" then
        routedDepth := routedDepth + 1
      else if instr.tag == "RoutedSumEnd" then
        routedDepth := routedDepth - 1
      idx := idx + 1

  -- Fold-support closure: fold-stage defs read by hoisted instructions
  -- (transitively through other fold defs) get DUPLICATED into the
  -- coefficient stream — pure constant computations, legal in both
  -- kernels, and keeping the originals in the audio plan preserves the
  -- emitters' f64 constant folding (the GPU-precision reason `fold`
  -- exists at all).
  let mut needFold : HashMap Nat Unit := {}
  let mut work : Array Nat := #[]
  for d in [0:hoisted.size] do
    if hoisted[d]! then
      for r in regDefs[d]! do
        if stages[r]! == .fold then
          work := work.push r
  while !work.isEmpty do
    let d := work.back!
    work := work.pop
    if !(needFold.contains d) then
      needFold := needFold.insert d ()
      for r in regDefs[d]! do
        if stages[r]! == .fold then
          work := work.push r

  -- Coefficient slot ordinals: ascending def index, deterministic.
  let mut boundary : Array (Nat × Nat × ScalarType) := #[]
  for d in [0:defMeta.size] do
    if boundarySet.contains d then
      let some (t, ty) := defMeta[d]!
        | panic! "Stage0: boundary def is not a temp def"
      boundary := boundary.push (d, t, ty)
  return { stages, hoisted, needFold, boundary, rewrites }

/-- The flow-derived classification, exposed for the stage differential:
    per linear instruction (the `collectBlocks` emit-order walk), its
    value stage and whether this pass hoists it. -/
def classify (plan : FlatPlan) : Array Stage × Array Bool := Id.run do
  let allBlocks := collectPlanBlocks plan
  let a := analyze plan allBlocks
  return (a.stages, a.hoisted)

-- ─────────────────────────────────────────────────────────────
-- Split
-- ─────────────────────────────────────────────────────────────

/-- Shared split assembly: given a placement (`Analysis` — from the flow
    `analyze` or the typed `placementFromStages`), rebuild the audio plan
    and the coefficient plan. Identity when nothing hoists. -/
private def rebuildCore (plan : FlatPlan) (allBlocks : Array (Array NInstr))
    (a : Analysis) : Split := Id.run do
  if !(a.hoisted.any id) then
    return { audio := plan, coeff? := none }

  -- Coefficient slot table extension.
  let slotBase := plan.slotCount
  -- def idx → (absolute coefficient slot, producer result type)
  let mut boundaryInfo : HashMap Nat (Nat × ScalarType) := {}
  let mut boundaryWrite : HashMap Nat NInstr := {}
  let mut slotNames := plan.slotNames
  let mut slotDefaults := plan.slotDefaults
  for i in [0:a.boundary.size] do
    let (d, t, ty) := a.boundary[i]!
    boundaryInfo := boundaryInfo.insert d (slotBase + i, ty)
    boundaryWrite := boundaryWrite.insert d (instrWriteSlot (slotBase + i) (.reg t ty) ty)
    slotNames := slotNames.push s!"coef:{i}"
    slotDefaults := slotDefaults.push (Lean.toJson (0 : Nat))

  -- Rebuild: surviving instructions (args rewritten) stay in their
  -- blocks; hoisted ones move to the coefficient stream in emit order,
  -- each boundary def followed immediately by its slot write; fold-stage
  -- defs the hoisted instructions read are DUPLICATED into the stream
  -- (kept in the audio plan too — see the fold-support closure).
  let mut coeffStream : Array NInstr := #[]
  let mut newBlocks : Array (Array NInstr) := #[]
  let mut idx := 0
  for block in allBlocks do
    let mut block' : Array NInstr := #[]
    for instr in block do
      if a.hoisted[idx]! then
        coeffStream := coeffStream.push instr
        if let some w := boundaryWrite.get? idx then
          coeffStream := coeffStream.push w
      else
        if a.needFold.contains idx then
          coeffStream := coeffStream.push instr
        let instr' := match a.rewrites.get? idx with
          | none => instr
          | some rw => Id.run do
            let mut args := instr.args
            for (pos, d) in rw do
              let (slot, ty) := boundaryInfo.get! d
              args := args.set! pos (.slot slot ty)
            return { instr with args }
        block' := block'.push instr'
      idx := idx + 1
    newBlocks := newBlocks.push block'

  -- Reassemble the instance-function tree from the rewritten blocks.
  let mut fns : Array InstanceFunction := #[]
  let mut cursor := 0
  for f in plan.instanceFunctions do
    let (f', cursor') := rebuildFn f newBlocks cursor
    fns := fns.push f'
    cursor := cursor'

  -- The array slots the coeff stream FILLS (banks-as-data coefficient columns).
  -- The audio plan advertises them so the runtime double-buffers exactly these:
  -- the coeff kernel writes a back generation, one atomic flip publishes it, and
  -- the audio kernel reads a whole consistent generation (no cross-column tear).
  let filledArraySlots : Array Nat := (coeffStream.filterMap fun i =>
    match i.dst with | .array s => some s | _ => none).foldl
      (fun acc s => if acc.contains s then acc else acc.push s) #[]
  let coeffArraySlots := filledArraySlots.foldl
    (fun acc s => if acc.contains s then acc else acc.push s)
    plan.coeffArraySlots
  let audio : FlatPlan := { plan with
    instanceFunctions := fns
    slotCount := slotBase + a.boundary.size
    slotNames := slotNames
    slotDefaults := slotDefaults
    coeffArraySlots := coeffArraySlots }
  -- If any coefficient-column fill hoisted, the coeff kernel writes shared
  -- `array_ptrs` storage (banks-as-data), so it needs the array metadata (same
  -- slot indices as the audio plan — the storage is allocated once from the
  -- audio plan and both kernels index it). Scalar-only splits keep zero arrays.
  let coeffUsesArrays := !coeffArraySlots.isEmpty || coeffStream.any fun i =>
    i.args.any fun a => match a with | .arrayReg _ => true | _ => false
  let coeff : FlatPlan := { audio with
    -- Staging kernels publish slots/arrays, never device channels. Keeping
    -- their scratch output mono preserves the single-cell host invocation
    -- even when the audio residual has independent stereo sinks.
    outputChannelCount := 1
    coeffArraySlots := #[]
    compilationMode := .fused
    arraySlotNames := if coeffUsesArrays then plan.arraySlotNames else #[]
    arraySlotCount := if coeffUsesArrays then plan.arraySlotCount else 0
    arraySlotSizes := if coeffUsesArrays then plan.arraySlotSizes else #[]
    instanceFunctions := #[.mk "coefficient" "coefficient" #[] coeffStream #[]
      0 0 plan.registerCount #[]]
    sinks := #[]
    paramDisciplines := #[] }
  return { audio, coeff? := some coeff }

/-- Normalize the split's external interfaces at the assembly boundary.  The
    core already constructs these fields this way; spelling the invariant at
    the boundary prevents later residualization changes from accidentally
    collapsing stereo/multi-sink audio or exposing coefficient sinks. -/
private def rebuild (plan : FlatPlan) (allBlocks : Array (Array NInstr))
    (a : Analysis) : Split :=
  let result := rebuildCore plan allBlocks a
  { result with
    audio := { result.audio with
      sinks := plan.sinks
      outputChannelCount := plan.outputChannelCount }
    coeff? := result.coeff?.map fun coefficient =>
      { coefficient with sinks := #[], outputChannelCount := 1 } }

/-- Split a plan into its audio and coefficient stages via the FLOW
    classification (the plan-level reference pass — the only splitter
    available where the arena is gone, i.e. plans parsed from JSON). -/
def hoist (plan : FlatPlan) : Split := Id.run do
  let allBlocks := collectPlanBlocks plan
  return rebuild plan allBlocks (analyze plan allBlocks)

-- ─────────────────────────────────────────────────────────────
-- Typed placement — the split driven by the intern-time attribute
-- ─────────────────────────────────────────────────────────────

/-- The stage-independent per-sample pins shared by BOTH placement layers
    (individual moves and whole-region moves): SESSION I/O arrays
    (`sessionArray*` — genuinely per-sample device/wire buffers) and the
    per-program FFI leaves (`param` handles, raw `input` reads — no
    control-time evaluator). -/
private def overlayPinnedS1 (i : NInstr) : Bool :=
  (match i.dst with | .sessionArray _ => true | _ => false)
  || i.args.any fun a => match a with
    | .sessionArrayReg _ | .param _ _ | .input _ _ => true
    | _ => false

/-- The instruction-level conservatism overlay for INDIVIDUAL moves. Kept
    per-sample regardless of the value stage: the pins above, plus the reduce
    delimiters and every `loopIdx` reader — one instruction of a loop can never
    move alone. A `loopIdx`-reading instruction varies per iteration even when
    its VALUE stage is s0 (the intern-time attribute treats `loopIdx` as
    stage-neutral — see `enodeSig`), so it pins s1 here, and everything
    downstream of it stays behind via the availability walk. Loop code leaves
    the audio kernel only through the whole-region decision (`tryRegion` in
    `placementFromStages`). NOT pinned: plain `array`/`arrayReg` — the
    coefficient columns (banks-as-data). Their stage defers to the value
    attribute (join of the fills), so an array whose fills are all s0 hoists
    into the coefficient kernel and the audio kernel's in-loop `Index` reads
    the shared, coefficient-filled storage (`run_coeff` and `process` share
    `state.array_ptrs`). -/
private def overlayS1 (i : NInstr) : Bool :=
  i.tag == "ReduceBegin" || i.tag == "ReduceEnd"
  || overlayPinnedS1 i
  || i.args.any fun a => match a with | .loopIdx _ => true | _ => false

/-- The matching `ReduceEnd` of the `ReduceBegin` at `b` in the linear
    stream, DEPTH-COUNTING (regions nest): a nested `ReduceBegin` opens a
    subregion whose own `ReduceEnd` must close before ours matches. A
    missing `ReduceEnd` yields `none` and the region stays put. -/
private def findRegionEnd (flat : Array NInstr) (b : Nat) : Option Nat := Id.run do
  let mut depth : Nat := 0
  for i in [b+1:flat.size] do
    if flat[i]!.tag == "ReduceBegin" then depth := depth + 1
    else if flat[i]!.tag == "ReduceEnd" then
      if depth == 0 then return some i
      depth := depth - 1
  return none

/-- Placement from the TYPED stages (per linear instruction, the
    partitioner's emit-order blocks). Unlike the flow pass — whose
    availability facts are baked into its classification — the typed
    stages are value facts only, so placement enforces availability
    explicitly: an instruction hoists only if every temp def and every
    sole-writer slot it reads is itself hoisted, fold-duplicable, or
    external (never written in-plan). Fold-valued support — temp chains
    AND sole-writer slot writes (the wire crossings the flow pass could
    not see through) — is duplicated into the coefficient stream; a
    dependency that is neither available nor duplicable simply keeps its
    reader in the audio kernel (the cascade is the forward walk itself,
    since defs precede uses).

    Placement is TWO-LAYERED. Layer 1 is the per-instruction walk above,
    under `overlayS1` (delimiters and `loopIdx` readers pinned — no
    instruction of a loop moves alone; loop-invariant s0 body
    instructions still hoist individually, shrinking the region:
    staging-as-LICM). Layer 2 is the whole-region move (`tryRegion`): a
    delimiter-matched `ReduceBegin`/`ReduceEnd` unit whose entire body is
    coefficient-shaped hoists AS A UNIT, in original relative order, its
    result crossing back through the ordinary scalar boundary (the
    accumulator's reaching def is the `ReduceEnd`, so the existing
    `coef:<n>` rewrite machinery applies unchanged). -/
private def placementFromStages (blocks : Array (Array NInstr))
    (linStages : Array (Option Stage)) : Except String Analysis := do
  -- Prepass: per-slot and per-array-slot in-plan writers. An array slot whose
  -- writers are all hoisted is a coefficient column filled in the coeff kernel
  -- (banks-as-data); the shared `array_ptrs` storage is the boundary, so no
  -- `coef:` slot crosses (unlike a scalar temp).
  let mut slotWriters : HashMap Nat (Array Nat) := {}
  let mut arrayWriters : HashMap Nat (Array Nat) := {}
  let mut flat : Array NInstr := #[]
  for block in blocks do
    for instr in block do
      if let .moduleSlot m := instr.dst then
        slotWriters := slotWriters.insert m ((slotWriters.getD m #[]).push flat.size)
      if let .array s := instr.dst then
        arrayWriters := arrayWriters.insert s ((arrayWriters.getD s #[]).push flat.size)
      flat := flat.push instr
  if flat.size != linStages.size then
    throw s!"Stage0.placementFromStages: {flat.size} instructions but {linStages.size} stages"

  -- Static routed reductions are indivisible placement regions.  Mark their
  -- full delimiter spans once, then make the ordinary placement lattice see
  -- every member as s1.  This prevents both delimiter/body separation and
  -- accidental whole-region movement through an enclosing ordinary reduce.
  let routedMask : Array Bool := Id.run do
    let mut mask := Array.replicate flat.size false
    let mut depth : Nat := 0
    for i in [0:flat.size] do
      if flat[i]!.tag == "RoutedSumBegin" then depth := depth + 1
      if depth > 0 then mask := mask.set! i true
      if flat[i]!.tag == "RoutedSumEnd" then depth := depth - 1
    return mask

  let stageAt (i : Nat) : Stage :=
    if routedMask[i]! then .s1
    else match linStages[i]? with
      | some (some s) => if overlayS1 flat[i]! then .s1 else s
      | _ => .s1
  -- Region-neutral value stage: for the WHOLE-REGION decision the
  -- delimiters and `loopIdx` are defined by the region itself, so only
  -- the session-I/O / FFI pins apply.
  let regionStageAt (i : Nat) : Stage :=
    match linStages[i]? with
    | some (some s) => if overlayPinnedS1 flat[i]! then .s1 else s
    | _ => .s1

  -- ── Layer 2: the whole-region move ──
  -- At `b` = a `ReduceBegin` with matching `ReduceEnd` at `e`, decide
  -- whether the ENTIRE delimiter-matched unit moves to the coefficient
  -- stream. Conditions, checked as an aggregate:
  --   1. every instruction's VALUE stage is ≤ s0 under the region-neutral
  --      overlay (`loopIdx` counts as stage-neutral for this check only —
  --      it is defined by the region itself; a τ-reading body is s1 by
  --      attribute and keeps the region in the audio kernel);
  --   2. every dst is a plain temp (the accumulator and body SSA temps —
  --      internal to the unit; no slot/array writes move this way, v1);
  --   3. availability holds for the aggregate: every temp/slot the region
  --      reads from OUTSIDE itself is hoisted, fold-duplicable, or
  --      external (the individual pass's discipline), and every array it
  --      reads has ALL its fills already hoisted — the shared-`array_ptrs`
  --      crossing (a fill kept in the audio kernel, e.g. a fold Pack under
  --      the EmitMsl f64 rule, keeps the region there too: the coefficient
  --      kernel must never read a column only the audio kernel fills);
  --   4. the dynamic-count operand (`ReduceBegin` args[1], trip-count-as-
  --      data), when present, is an ordinary operand, so rule 3 covers it
  --      (a never-written param slot is external, hence s0-available).
  -- The only value that escapes a region is the accumulator (body temps
  -- are region-internal by the emit contract — `compileBankSum` snapshots
  -- the CSE memo, and post-region reads fall back to zero scratch), so
  -- the boundary rewrite fires exactly on the `ReduceEnd` def.
  -- Returns the fold-duplication seeds, or `none` when the region stays.
  let tryRegion := fun (b e : Nat) (hoisted : Array Bool)
      (tempDef : HashMap Nat Nat) => Id.run do
    let mut seeds : Array Nat := #[]
    let mut regTemps : HashMap Nat Unit := {}
    for i in [b:e+1] do
      let instr := flat[i]!
      if regionStageAt i == .s1 then return none
      match instr.dst with
      | .temp _ => pure ()
      | _ => return none
      for arg in instr.args do
        match arg with
        | .reg t _ =>
          if regTemps.contains t then pure ()      -- region-internal def
          else match tempDef.get? t with
            | none => pure ()                      -- zero fallback: constant
            | some d =>
              if hoisted[d]! then pure ()
              else if stageAt d == .fold then seeds := seeds.push d
              else return none
        | .slot s _ =>
          match slotWriters.getD s #[] with
          | #[] => pure ()                         -- external (param/default)
          | #[w] =>
            if w < b && hoisted[w]! then pure ()
            else if w < b && stageAt w == .fold then seeds := seeds.push w
            else return none
          | _ => return none
        | .arrayReg s =>
          let ws := arrayWriters.getD s #[]
          if ws.isEmpty || !(ws.all fun w => w < b && hoisted[w]!) then
            return none
        | .loopIdx _ => pure ()                    -- defined by the unit (any id: ours or a nested subregion's)
        | .const _ _ | .source _ _ => pure ()      -- value stage (rule 1) covers these
        | .input _ _ | .param _ _ | .sessionArrayReg _ => return none
      if let .temp t := instr.dst then regTemps := regTemps.insert t ()
    return some seeds

  let mut hoisted : Array Bool := #[]
  let mut defMeta : Array (Option (Nat × ScalarType)) := #[]
  let mut boundarySet : HashMap Nat Unit := {}
  let mut rewrites : HashMap Nat (Array (Nat × Nat)) := {}
  let mut tempDef : HashMap Nat Nat := {}
  let mut dupSeeds : Array Nat := #[]
  -- The matching `ReduceEnd` index while inside a whole-region move.
  let mut regionEnd : Option Nat := none
  -- Only OUTERMOST regions are whole-move candidates (nested v1 policy): a
  -- nested subregion moves with its enclosing unit (it lies inside [b..e])
  -- and is never considered separately. When an outermost region STAYS, its
  -- matching end is recorded here so the begins nested inside it are skipped.
  let mut noRegionUntil : Nat := 0
  for idx in [0:flat.size] do
    let instr := flat[idx]!
    -- Whole-region decision at each OUTERMOST `ReduceBegin`. Region
    -- membership is derived from depth-counted delimiter matching in the
    -- linear stream (`findRegionEnd`).
    if regionEnd.isNone && idx ≥ noRegionUntil && instr.tag == "ReduceBegin" then
      if let some e := findRegionEnd flat idx then
        match tryRegion idx e hoisted tempDef with
        | some regionSeeds =>
          regionEnd := some e
          dupSeeds := dupSeeds ++ regionSeeds
        | none =>
          noRegionUntil := e + 1
    let inRegion := regionEnd.isSome
    let stage := stageAt idx
    -- Availability of every read, given the placement so far (skipped
    -- inside a moving region — `tryRegion` checked the aggregate).
    let mut avail := true
    let mut seeds : Array Nat := #[]
    if stage == .s0 && !inRegion then
      for arg in instr.args do
        match arg with
        | .reg t _ =>
          match tempDef.get? t with
          | none => pure ()                       -- zero fallback: constant
          | some d =>
            if hoisted[d]! then pure ()
            else if stageAt d == .fold then seeds := seeds.push d
            else avail := false
        | .slot i _ =>
          match slotWriters.getD i #[] with
          | #[] => pure ()                        -- external (param/default)
          | #[w] =>
            if w < idx && hoisted[w]! then pure ()
            else if w < idx && stageAt w == .fold then seeds := seeds.push w
            else avail := false
          | _ => avail := false                   -- multi-writer: stay
        | _ => pure ()
    let hoist := inRegion || (stage == .s0 && avail &&
      (match instr.dst with
        | .temp _ => true
        | .moduleSlot m => slotWriters.getD m #[] == #[idx]
        -- A coefficient column: hoistable when THIS instruction is its sole
        -- writer (a `Pack`, or the last of a fully-s0 `SetElement` group — the
        -- group hoists together via the forward availability walk). The audio
        -- kernel's `Index` reads the coeff-filled shared array; no `coef:` slot.
        | .array s => arrayWriters.getD s #[] == #[idx]
        | _ => false))
    if hoist then
      dupSeeds := dupSeeds ++ seeds
    else
      -- Reads of hoisted defs must come through coefficient slots.
      let mut rw : Array (Nat × Nat) := #[]
      for pos in [0:instr.args.size] do
        if let .reg t _ := instr.args[pos]! then
          if let some d := tempDef.get? t then
            if hoisted[d]! then
              rw := rw.push (pos, d)
              boundarySet := boundarySet.insert d ()
      if !rw.isEmpty then
        rewrites := rewrites.insert idx rw
    hoisted := hoisted.push hoist
    match instr.dst with
    | .temp t =>
      defMeta := defMeta.push (some (t, instr.resultType))
      tempDef := tempDef.insert t idx
    | _ => defMeta := defMeta.push none
    if regionEnd == some idx then regionEnd := none

  -- Duplication closure over fold support: temp defs and sole-writer
  -- slot writes referenced (transitively) by hoisted instructions.
  let mut needFold : HashMap Nat Unit := {}
  let mut work := dupSeeds
  -- Reaching defs must be recomputed per reference point for exactness;
  -- for fold chains a simpler global map suffices because fold defs are
  -- never shadowed by later defs of the same temp in-corpus. Guard it:
  -- walk each duplicated instr's args through the same availability
  -- rules and fail loudly on anything non-duplicable.
  let lastDef : HashMap Nat Nat := Id.run do
    let mut m : HashMap Nat Nat := {}
    for i in [0:flat.size] do
      if let .temp t := flat[i]!.dst then
        if stageAt i == .fold then m := m.insert t i
      pure ()
    return m
  while !work.isEmpty do
    let d := work.back!
    work := work.pop
    if !(needFold.contains d) then
      if stageAt d != .fold then
        throw s!"Stage0.placementFromStages: duplicated instr {d} is not fold-stage (placement bug)"
      needFold := needFold.insert d ()
      for arg in flat[d]!.args do
        match arg with
        | .reg t _ =>
          match lastDef.get? t with
          | some dd => work := work.push dd
          | none => pure ()                       -- zero fallback
        | .slot i _ =>
          match slotWriters.getD i #[] with
          | #[] => pure ()
          | #[w] =>
            if !(hoisted[w]!) then
              if stageAt w == .fold then work := work.push w
              else throw s!"Stage0.placementFromStages: fold instr {d} reads slot {i} with non-fold surviving writer (placement bug)"
          | _ => throw s!"Stage0.placementFromStages: fold instr {d} reads multi-writer slot {i} (placement bug)"
        | _ => pure ()

  -- Coefficient slot ordinals: ascending def index, deterministic.
  let mut boundary : Array (Nat × Nat × ScalarType) := #[]
  for d in [0:defMeta.size] do
    if boundarySet.contains d then
      match defMeta[d]! with
      | some (t, ty) => boundary := boundary.push (d, t, ty)
      | none => throw "Stage0.placementFromStages: boundary def is not a temp def"
  let stages := (Array.range flat.size).map stageAt
  return { stages, hoisted, needFold, boundary, rewrites }

/-- Split a plan via the TYPED per-instruction stages (the partitioner's
    emit-order blocks from `compileSessionStaged`). -/
def hoistTyped (plan : FlatPlan)
    (stageBlocks : Array (Array (Option Stage))) : Except String Split := do
  let allBlocks := collectPlanBlocks plan
  if !typedStagesAligned allBlocks stageBlocks then
    throw "Stage0.hoistTyped: typed stage blocks do not align with emitter blocks"
  if noTypedSelection stageBlocks then
    return { audio := plan, coeff? := none }
  let a ← placementFromStages allBlocks (stageBlocks.flatten)
  return rebuild plan allBlocks a

/-- Every typed split preserves the complete externally observable audio
    interface.  In particular, independent stereo (or wider) sink routing is
    never collapsed by staging; only the private coefficient kernel is mono. -/
theorem hoistTyped_preserves_audio_interface (plan : FlatPlan)
    (stageBlocks : Array (Array (Option Stage))) (result : Split)
    (h : hoistTyped plan stageBlocks = .ok result) :
    result.audio.sinks = plan.sinks ∧
    result.audio.outputChannelCount = plan.outputChannelCount ∧
    ∀ coefficient, result.coeff? = some coefficient →
      coefficient.sinks = #[] ∧ coefficient.outputChannelCount = 1 := by
  unfold hoistTyped at h
  by_cases hMisaligned : typedStagesAligned (collectPlanBlocks plan)
      stageBlocks = false
  · simp [hMisaligned, bind, Except.bind] at h
  · by_cases hNone : noTypedSelection stageBlocks = true
    · simp [hMisaligned, hNone] at h
      cases h
      exact ⟨rfl, rfl, by simp⟩
    · cases hp : placementFromStages (collectPlanBlocks plan)
          stageBlocks.flatten with
      | error message =>
        simp [hMisaligned, hNone, hp, bind, Except.bind] at h
        change (Except.error message : Except String Split) = .ok result at h
        contradiction
      | ok analysis =>
        simp [hMisaligned, hNone, hp, bind, Except.bind] at h
        cases h
        simp [rebuild]

end Tropical.Ir.Stage0
