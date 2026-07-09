import Std.Data.HashMap
import Tropical.Ir.CoreArena
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
partial def collectBlocks (f : InstanceFunction) : Array (Array NInstr) := Id.run do
  let mut out := #[f.preambleInstructions]
  for child in f.children do
    out := out.push child.preInputInstructions
    out := out ++ collectBlocks child
  return out.push f.instructions

/-- Reassemble an instance function from rewritten blocks, consuming them
    in the same order `collectBlocks` produced. Returns the rebuilt
    function and the next unconsumed block index. -/
private partial def rebuildFn (f : InstanceFunction) (blocks : Array (Array NInstr))
    (start : Nat) : InstanceFunction × Nat := Id.run do
  let preamble := blocks[start]!
  let mut i := start + 1
  let mut children : Array InstanceFunction := #[]
  for child in f.children do
    let preInput := blocks[i]!
    let (child', i') := rebuildFn child blocks (i + 1)
    children := children.push (child'.withPreInput preInput)
    i := i'
  let body := blocks[i]!
  return (.mk f.name f.instanceName preamble body f.preInputInstructions
    f.registerOffset f.arraySlotOffset f.registerCount children, i + 1)

-- ─────────────────────────────────────────────────────────────
-- Analysis — one forward pass in emit order
-- ─────────────────────────────────────────────────────────────

-- `Stage` (fold < s0 < s1) and `Stage.join` are the shared binding-time
-- type from `Tropical.Ir.CoreArena` — the same lattice the intern-time
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
  | .loopIdx => .s1           -- per-iteration inside a reduce region

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
      let (hoist, vStage) :=
        -- Reduce delimiters are per-sample loop structure — never moved.
        if instr.tag == "ReduceBegin" || instr.tag == "ReduceEnd" then (false, Stage.s1)
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
  let mut allBlocks : Array (Array NInstr) := #[]
  for f in plan.instanceFunctions do
    allBlocks := allBlocks ++ collectBlocks f
  let a := analyze plan allBlocks
  return (a.stages, a.hoisted)

-- ─────────────────────────────────────────────────────────────
-- Split
-- ─────────────────────────────────────────────────────────────

/-- Shared split assembly: given a placement (`Analysis` — from the flow
    `analyze` or the typed `placementFromStages`), rebuild the audio plan
    and the coefficient plan. Identity when nothing hoists. -/
private def rebuild (plan : FlatPlan) (allBlocks : Array (Array NInstr))
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

  let audio : FlatPlan := { plan with
    instanceFunctions := fns
    slotCount := slotBase + a.boundary.size
    slotNames := slotNames
    slotDefaults := slotDefaults }
  let coeff : FlatPlan := { audio with
    compilationMode := .fused
    arraySlotNames := #[]
    arraySlotCount := 0
    arraySlotSizes := #[]
    instanceFunctions := #[.mk "coefficient" "coefficient" #[] coeffStream #[]
      0 0 plan.registerCount #[]]
    sinks := #[]
    paramDisciplines := #[] }
  return { audio, coeff? := some coeff }

/-- Split a plan into its audio and coefficient stages via the FLOW
    classification (the plan-level reference pass — the only splitter
    available where the arena is gone, i.e. plans parsed from JSON). -/
def hoist (plan : FlatPlan) : Split := Id.run do
  let mut allBlocks : Array (Array NInstr) := #[]
  for f in plan.instanceFunctions do
    allBlocks := allBlocks ++ collectBlocks f
  return rebuild plan allBlocks (analyze plan allBlocks)

-- ─────────────────────────────────────────────────────────────
-- Typed placement — the split driven by the intern-time attribute
-- ─────────────────────────────────────────────────────────────

/-- The v1 conservatism overlay at instruction level: arrays stay
    per-sample, and the per-program FFI leaves (`param` handles, raw
    `input` reads) have no control-time evaluator. -/
private def overlayS1 (i : NInstr) : Bool :=
  i.tag == "ReduceBegin" || i.tag == "ReduceEnd"
  || (match i.dst with
    | .array _ => true | .sessionArray _ => true | _ => false)
  || i.args.any fun a => match a with
    | .arrayReg _ | .sessionArrayReg _ | .param _ _ | .input _ _ | .loopIdx => true
    | _ => false

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
    since defs precede uses). -/
private def placementFromStages (blocks : Array (Array NInstr))
    (linStages : Array (Option Stage)) : Except String Analysis := do
  -- Prepass: per-slot in-plan writers.
  let mut slotWriters : HashMap Nat (Array Nat) := {}
  let mut flat : Array NInstr := #[]
  for block in blocks do
    for instr in block do
      if let .moduleSlot m := instr.dst then
        slotWriters := slotWriters.insert m ((slotWriters.getD m #[]).push flat.size)
      flat := flat.push instr
  if flat.size != linStages.size then
    throw s!"Stage0.placementFromStages: {flat.size} instructions but {linStages.size} stages"

  let stageAt (i : Nat) : Stage :=
    match linStages[i]? with
    | some (some s) => if overlayS1 flat[i]! then .s1 else s
    | _ => .s1

  let mut hoisted : Array Bool := #[]
  let mut defMeta : Array (Option (Nat × ScalarType)) := #[]
  let mut boundarySet : HashMap Nat Unit := {}
  let mut rewrites : HashMap Nat (Array (Nat × Nat)) := {}
  let mut tempDef : HashMap Nat Nat := {}
  let mut dupSeeds : Array Nat := #[]
  for idx in [0:flat.size] do
    let instr := flat[idx]!
    let stage := stageAt idx
    -- Availability of every read, given the placement so far.
    let mut avail := true
    let mut seeds : Array Nat := #[]
    if stage == .s0 then
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
    let hoist := stage == .s0 && avail &&
      (match instr.dst with
        | .temp _ => true
        | .moduleSlot m => slotWriters.getD m #[] == #[idx]
        | _ => false)
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
  let mut allBlocks : Array (Array NInstr) := #[]
  for f in plan.instanceFunctions do
    allBlocks := allBlocks ++ collectBlocks f
  let a ← placementFromStages allBlocks (stageBlocks.flatten)
  return rebuild plan allBlocks a

end Tropical.Ir.Stage0
